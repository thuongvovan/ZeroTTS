/**
 * Resolving and loading a ZeroTTS model in the browser.
 *
 * Files come straight from the Hugging Face CDN (or any static host with the
 * same layout). Nothing is uploaded anywhere — the model runs on the viewer's
 * machine.
 */

import * as ort from 'onnxruntime-web';

import { fetchWithCache, ProgressFn, totalBytes } from './cache';
import { CodecMeta, MossCodecDecoder } from './codec';
import {
  DEFAULT_ENGINE, engineDefinition, engineFromLegacy,
} from './engine';
import type { EngineFallback, EngineId } from './engine';
import {
  Backend, DEFAULT_BACKEND, DEFAULT_GGML_DEVICE, DEFAULT_GGUF, GgmlDevice,
  modelUrls, repoBaseUrl,
} from './repo';
import { BpeTokenizer } from './tokenizer';
import { ZeroTTSBrowser } from './synthesizer';
import type { ZeroTTSGgml } from './ggmlBackend';
import { VoiceIndex, ZeroTTSConfig } from './types';

// Re-exported so the runtime side keeps one import site; the definitions live in
// repo.ts because the page needs them without the runtime.
export type { Backend, GgmlDevice } from './repo';
export type { EngineFallback, EngineId, EngineLoadOptions } from './engine';
export {
  DEFAULT_BACKEND, DEFAULT_ENGINE, DEFAULT_GGML_DEVICE, DEFAULT_GGUF, DEFAULT_REPO,
  ENGINES, GGUF_BUILDS, GGUF_REPO, ONNX_REPO, defaultRepo,
  defaultRepoForEngine, downloadInfo, engineDefinition, loadVoice, modelFiles,
  modelUrls, repoBaseUrl, voicePreviewUrl,
} from './repo';

export interface LoadOptions {
  /** Stable engine selector. Prefer this over backend + ggmlDevice. */
  engine?: EngineId;
  repo?: string;
  revision?: string;
  onProgress?: ProgressFn;
  threads?: number;
  /** Which runtime generates frames. Defaults to ggml — see repo.ts. */
  backend?: Backend;
  /** Which GGUF to fetch, for the ggml backend. */
  gguf?: string;
  /** CPU is the stable default. WebGPU is experimental and falls back to CPU
   * when the browser or adapter cannot initialize the ggml backend. */
  ggmlDevice?: GgmlDevice;
  /** CPU fallback is the application default for WebGPU. Use `none` when an
   * exact engine is required. */
  fallback?: EngineFallback;
}

export interface LoadedModel {
  tts: ZeroTTSBrowser;
  voices: VoiceIndex;
  base: string;
  /** Requested and actual engine are separate so fallback is never hidden. */
  requestedEngine: EngineId;
  engine: EngineId;
  backend: Backend;
  /** Actual ggml device after compatibility fallback; absent for ONNX. */
  ggmlDevice?: GgmlDevice;
  /** Why a requested WebGPU load continued on CPU instead. */
  fallbackReason?: string;
  /** Whether the selected GGML artifact uses pthreads. */
  wasmThreads?: boolean;
  /** Actual inference thread count after origin/runtime constraints. */
  threads: number;
}

export async function loadModel(options: LoadOptions = {}): Promise<LoadedModel> {
  const requestedEngine = options.engine
    ?? engineFromLegacy(options.backend ?? DEFAULT_BACKEND,
      options.ggmlDevice ?? DEFAULT_GGML_DEVICE);
  const selected = engineDefinition(requestedEngine ?? DEFAULT_ENGINE);
  const backend = selected.backend;
  const requestedGgmlDevice = selected.ggmlDevice ?? DEFAULT_GGML_DEVICE;
  const fallback = options.fallback ?? 'cpu';
  const gguf = options.gguf ?? DEFAULT_GGUF;
  const base = repoBaseUrl(options.repo ?? selected.defaultRepo, options.revision);

  // Multi-threaded WASM needs SharedArrayBuffer, which needs the page to be
  // cross-origin isolated (the COOP/COEP headers in vite.config.ts). Asking for
  // threads without it is not an error, just a silent fallback — say so instead,
  // because the difference is several times the generation time.
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  if (!isolated) {
    console.warn('not cross-origin isolated — using the single-thread WASM artifact; '
      + 'use localhost or HTTPS with COOP/COEP for threaded CPU inference');
  }
  ort.env.wasm.numThreads =
    options.threads ?? (isolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1);
  ort.env.wasm.simd = true;

  // ONNX stays on WASM. The WebGPU option belongs to the separate ggml
  // generator; moving these codec/legacy-generator sessions to ONNX WebGPU was
  // slower in measurements and would make sampling provider-dependent.
  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  };

  const overall = { loaded: 0, total: await totalBytes(modelUrls(base, backend, gguf)) };
  const get = (path: string) =>
    fetchWithCache(`${base}/${path}`, options.onProgress, overall);
  const getJson = async (path: string) => {
    // `no-cache` = revalidate, not "don't cache": these are small, and one of
    // them is the voice manifest. Served from the HTTP cache without asking, a
    // voice removed on the Hub would keep showing up in the picker.
    const buf = await fetch(`${base}/${path}`, { cache: 'no-cache' }).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    return JSON.parse(new TextDecoder().decode(buf));
  };

  const [config, tokenizerJson, codecMeta, voices] = await Promise.all([
    getJson('config.json') as Promise<ZeroTTSConfig>,
    getJson('tokenizer.json'),
    getJson('onnx/codec/codec_browser_onnx_meta.json') as Promise<CodecMeta>,
    getJson('voices/index.json').catch(() => ({ voices: [] })) as Promise<VoiceIndex>,
  ]);

  // The codec decoder graphs share one external .data file; ORT must be told
  // about it explicitly — it is not fetched implicitly.
  const sharedData = await get('onnx/codec/moss_audio_tokenizer_decode_shared.data');
  const codecOptions: ort.InferenceSession.SessionOptions = {
    ...sessionOptions,
    externalData: [{
      path: 'moss_audio_tokenizer_decode_shared.data',
      data: sharedData,
    }],
  };

  // The ggml backend fetches one GGUF where the ONNX backend fetches three
  // graphs; the codec and the small side files are the same either way.
  const [decodeFull, decodeStep, nullVoiceBuf] = await Promise.all([
    get('onnx/codec/moss_audio_tokenizer_decode_full.onnx'),
    get('onnx/codec/moss_audio_tokenizer_decode_step.onnx'),
    get('null_voice_emb.npy'),
  ]);
  const codec = await MossCodecDecoder.create(
    codecMeta, { decodeFull, decodeStep }, codecOptions);

  const tokenizer = await BpeTokenizer.create(tokenizerJson);

  let sessions: {
    textEncoder: ort.InferenceSession;
    prefixStep: ort.InferenceSession;
    localFrameDecode: ort.InferenceSession;
  } | null = null;
  let frameSource: ZeroTTSGgml | null = null;
  let ggmlDevice: GgmlDevice | undefined;
  let fallbackReason: string | undefined;

  if (backend === 'ggml') {
    const { ZeroTTSGgml, assertGgmlWebGpuSupport } = await import('./ggmlBackend');
    const ggufBuffer = await get(gguf);
    ggmlDevice = requestedGgmlDevice;
    if (requestedGgmlDevice === 'webgpu') {
      try {
        await assertGgmlWebGpuSupport();
        frameSource = await ZeroTTSGgml.create(
          ggufBuffer, tokenizer, options.threads, 'webgpu');
      } catch (error) {
        fallbackReason = (error as Error)?.message ?? String(error);
        if (fallback === 'none') throw error;
        console.warn(`ggml WebGPU unavailable — falling back to CPU: ${fallbackReason}`);
        ggmlDevice = 'cpu';
      }
    }
    if (!frameSource) {
      try {
        frameSource = await ZeroTTSGgml.create(
          ggufBuffer, tokenizer, options.threads, 'cpu');
      } catch (error) {
        if (!fallbackReason) throw error;
        const cpuReason = (error as Error)?.message ?? String(error);
        throw new Error(`WebGPU failed (${fallbackReason}); CPU fallback failed (${cpuReason})`);
      }
    }
  } else {
    const [prefixBuf, localBuf, textBuf] = await Promise.all([
      get('onnx/prefix_step.onnx'),
      get('onnx/local_frame_decode.onnx'),
      get('onnx/text_encoder.onnx'),
    ]);
    const [prefixStep, localFrameDecode, textEncoder] = await Promise.all([
      ort.InferenceSession.create(prefixBuf, sessionOptions),
      ort.InferenceSession.create(localBuf, sessionOptions),
      ort.InferenceSession.create(textBuf, sessionOptions),
    ]);
    sessions = { textEncoder, prefixStep, localFrameDecode };
  }

  // The codec's canonical silence frame, used to pad between segments. Absent
  // in model directories built before it was shipped; the synthesizer then
  // skips padding rather than guessing (zeros are a real code, not silence).
  let silenceFrame: BigInt64Array | null = null;
  try {
    silenceFrame = parseNpyInt64(await get('silence_frame.npy'));
  } catch {
    console.warn('silence_frame.npy missing — segments will not be padded');
  }

  const tts = new ZeroTTSBrowser(
    sessions, codec, tokenizer, config, parseNpyFloat32(nullVoiceBuf),
    silenceFrame, frameSource);

  await tts.warmup();
  const engine = backend === 'onnx'
    ? 'onnx-wasm' : (ggmlDevice === 'webgpu' ? 'ggml-webgpu' : 'ggml-cpu');
  return {
    tts, voices, base, requestedEngine, engine, backend, ggmlDevice,
    fallbackReason, wasmThreads: frameSource?.wasmThreads,
    threads: frameSource?.threadCount ?? Number(ort.env.wasm.numThreads ?? 1),
  };
}

/**
 * Minimal .npy reader for the one array we load that way (null_voice_emb.npy).
 * Only handles the little-endian float32, C-order case that file is written in;
 * anything else is a repo problem worth failing loudly on.
 */
function npyPayload(buffer: ArrayBuffer, descr: RegExp, what: string): [number, string] {
  const bytes = new Uint8Array(buffer);
  if (String.fromCharCode(...bytes.subarray(1, 6)) !== 'NUMPY') {
    throw new Error(`${what}: not a .npy file`);
  }
  const major = bytes[6];
  const view = new DataView(buffer);
  const headerLen = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const headerStart = major >= 2 ? 12 : 10;
  const header = new TextDecoder().decode(bytes.subarray(headerStart, headerStart + headerLen));

  if (!descr.test(header)) throw new Error(`${what}: unexpected dtype — ${header}`);
  if (/'fortran_order':\s*True/.test(header)) {
    throw new Error(`${what}: Fortran order is not supported`);
  }
  return [headerStart + headerLen, header];
}

function parseNpyFloat32(buffer: ArrayBuffer): Float32Array {
  const [offset] = npyPayload(buffer, /'descr':\s*'[<|]f4'/, 'null_voice_emb.npy');
  return new Float32Array(buffer, offset);
}

function parseNpyInt64(buffer: ArrayBuffer): BigInt64Array {
  const [offset] = npyPayload(buffer, /'descr':\s*'[<|]i8'/, 'silence_frame.npy');
  // A BigInt64Array view requires an 8-byte-aligned offset. NumPy pads its
  // header to a 64-byte boundary so this normally holds, but a hand-written or
  // re-saved file need not — copy rather than throw a RangeError nobody can act
  // on.
  if (offset % 8 === 0) return new BigInt64Array(buffer, offset);
  return new BigInt64Array(buffer.slice(offset));
}

/** One isolated benchmark process. The page creates a fresh worker per backend
 * so a retired WASM heap cannot distort the next measurement. */
import * as ort from 'onnxruntime-web/wasm';

import { BenchConfig, BenchResult, BenchWorkerRequest, BenchWorkerResponse, backendLabel } from './benchTypes';
import { fetchWithCache } from './cache';
import type { MossCodecDecoder } from './codec';
import { ZeroTTSGgml } from './ggmlBackend';
import { DEFAULT_RUNTIME_ASSETS } from './runtimeAssets';
import { ZeroTTSBrowser } from './synthesizer';
import { BpeTokenizer } from './tokenizer';
import { ZeroTTSConfig } from './types';

type Generator = ZeroTTSGgml | ZeroTTSBrowser;
const post = (message: BenchWorkerResponse) => self.postMessage(message);
const log = (message: string) => post({ type: 'log', message });
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

const getJson = async <T>(base: string, path: string): Promise<T> => {
  const response = await fetch(`${base}/${path}`, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json() as Promise<T>;
};
const getBin = (base: string, path: string) => fetchWithCache(`${base}/${path}`);

function parseNpyFloat32(buffer: ArrayBuffer): Float32Array {
  const bytes = new Uint8Array(buffer);
  if (String.fromCharCode(...bytes.subarray(1, 6)) !== 'NUMPY') {
    throw new Error('null_voice_emb.npy không hợp lệ');
  }
  const view = new DataView(buffer);
  const major = bytes[6];
  const headerLength = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  return new Float32Array(buffer, (major >= 2 ? 12 : 10) + headerLength);
}

async function loadSource(config: BenchConfig): Promise<Generator> {
  const tokenizerJson = await getJson<unknown>(config.modelBase, 'tokenizer.json');
  const tokenizer = await BpeTokenizer.create(tokenizerJson);

  if (config.backend !== 'onnx-wasm') {
    const gguf = await getBin(config.ggufBase, config.gguf);
    return ZeroTTSGgml.create(gguf, tokenizer, DEFAULT_RUNTIME_ASSETS, config.threads);
  }

  ort.env.wasm.numThreads = self.crossOriginIsolated ? config.threads : 1;
  ort.env.wasm.simd = true;
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'], graphOptimizationLevel: 'all',
  };
  log('Đang tạo ba ONNX session…');
  const [textEncoder, prefixStep, localFrameDecode, configJson, nullVoice] = await Promise.all([
    getBin(config.modelBase, 'onnx/text_encoder.onnx')
      .then((buffer) => ort.InferenceSession.create(buffer, options)),
    getBin(config.modelBase, 'onnx/prefix_step.onnx')
      .then((buffer) => ort.InferenceSession.create(buffer, options)),
    getBin(config.modelBase, 'onnx/local_frame_decode.onnx')
      .then((buffer) => ort.InferenceSession.create(buffer, options)),
    getJson<ZeroTTSConfig>(config.modelBase, 'config.json'),
    getBin(config.modelBase, 'null_voice_emb.npy'),
  ]);
  // This benchmark calls generateFrames() only. A codec would add two sessions
  // and ~45 MB without taking part in any measured operation.
  const unusedCodec = null as unknown as MossCodecDecoder;
  return new ZeroTTSBrowser(
    { textEncoder, prefixStep, localFrameDecode }, unusedCodec, tokenizer, configJson,
    parseNpyFloat32(nullVoice));
}

async function generateCodes(
  generator: Generator, config: BenchConfig, voice: Float32Array,
): Promise<Int32Array> {
  const flattened: number[] = [];
  for await (const frame of generator.generateFrames(
    config.text, voice,
    { minFrames: config.frames, maxFrames: config.frames }, config.seed)) {
    for (const code of frame) flattened.push(Number(code));
  }
  return Int32Array.from(flattened);
}

function hashCodes(codes: Int32Array): string {
  let hash = 0x811c9dc5;
  for (const code of codes) {
    for (let shift = 0; shift < 32; shift += 8) {
      hash ^= (code >>> shift) & 0xff;
      hash = Math.imul(hash, 0x01000193);
    }
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

function differingCodes(a: Int32Array, b: Int32Array): number {
  let different = Math.abs(a.length - b.length);
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) different++;
  return different;
}

async function benchmark(config: BenchConfig): Promise<BenchResult> {
  let label = backendLabel(config);
  log(`${label}: tải backend/model…`);
  const loadStart = performance.now();
  const generator = await loadSource(config);
  const effectiveThreads = generator instanceof ZeroTTSGgml
    ? generator.threadCount
    : (self.crossOriginIsolated ? config.threads : 1);
  if (effectiveThreads !== config.threads) {
    label = `${backendLabel(config, effectiveThreads)} · HTTP fallback`;
  }
  const voice = new Float32Array(await getBin(
    config.modelBase, `voices/${encodeURIComponent(config.voice)}/voice.bin`));
  const loadMs = performance.now() - loadStart;

  log(`${label}: warm-up ${config.frames} frame…`);
  const warmupStart = performance.now();
  await generateCodes(generator, config, voice);
  const warmupMs = performance.now() - warmupStart;

  const measurements = [];
  const outputs: Int32Array[] = [];
  for (let index = 0; index < config.runs; index++) {
    if (generator instanceof ZeroTTSGgml) generator.resetTimings();
    const started = performance.now();
    const codes = await generateCodes(generator, config, voice);
    const wallMs = performance.now() - started;
    const audioSeconds = config.frames / 12.5;
    const run = {
      wallMs,
      realtime: audioSeconds / (wallMs / 1000),
      hash: hashCodes(codes),
      firstFrame: Array.from(codes.slice(0, 16)),
      stages: generator instanceof ZeroTTSGgml ? generator.timings() : undefined,
    };
    outputs.push(codes);
    measurements.push(run);
    log(`${label}: lượt ${index + 1}/${config.runs} · ${wallMs.toFixed(1)} ms · `
      + `${run.realtime.toFixed(2)}x realtime · hash ${run.hash}`);
  }

  const driftFromFirst = outputs.map((codes) => differingCodes(outputs[0], codes));
  return {
    backend: config.backend,
    label,
    config: { ...config },
    status: 'ok',
    loadMs,
    warmupMs,
    effectiveThreads,
    measurements,
    medianMs: median(measurements.map((run) => run.wallMs)),
    medianRealtime: median(measurements.map((run) => run.realtime)),
    driftFromFirst,
    reproducible: driftFromFirst.every((count) => count === 0),
  };
}

self.onmessage = async (event: MessageEvent<BenchWorkerRequest>) => {
  if (event.data.type !== 'run') return;
  const config = event.data.config;
  try {
    post({ type: 'result', result: await benchmark(config) });
  } catch (error) {
    post({
      type: 'result',
      result: {
        backend: config.backend,
        label: backendLabel(config),
        config,
        status: 'error',
        error: (error as Error)?.message ?? String(error),
      },
    });
  }
};

/** Stable, public engine identifiers and metadata.
 *
 * UI code and applications select one id; the loader is responsible for the
 * backend/device details. Adding a runtime therefore does not change the
 * streaming or generation API.
 */
export type Backend = 'ggml' | 'onnx';
export type GgmlDevice = 'cpu' | 'webgpu';
export type EngineId = 'ggml-cpu' | 'ggml-webgpu' | 'onnx-wasm';
export type EngineFallback = 'none' | 'cpu';

export interface EngineCapabilities {
  /** Classifier-free guidance (`cfgScale > 1`). */
  cfg: boolean;
  /** Can load quantized GGUF weights. */
  quantizedWeights: boolean;
  /** Uses the browser WebGPU API. */
  webgpu: boolean;
  /** Same seed is expected to reproduce the same codes on the same platform. */
  reproducible: boolean;
}

export interface EngineDefinition {
  id: EngineId;
  label: string;
  backend: Backend;
  ggmlDevice?: GgmlDevice;
  defaultRepo: string;
  experimental: boolean;
  capabilities: EngineCapabilities;
}

export const ENGINES: Readonly<Record<EngineId, EngineDefinition>> = {
  'ggml-cpu': {
    id: 'ggml-cpu',
    label: 'GGML · CPU / WebAssembly',
    backend: 'ggml',
    ggmlDevice: 'cpu',
    defaultRepo: 'zeroweight-ai/ZeroTTS-GGUF',
    experimental: false,
    capabilities: {
      cfg: false, quantizedWeights: true, webgpu: false, reproducible: true,
    },
  },
  'ggml-webgpu': {
    id: 'ggml-webgpu',
    label: 'GGML · GPU / WebGPU',
    backend: 'ggml',
    ggmlDevice: 'webgpu',
    defaultRepo: 'zeroweight-ai/ZeroTTS-GGUF',
    experimental: true,
    capabilities: {
      cfg: false, quantizedWeights: true, webgpu: true, reproducible: false,
    },
  },
  'onnx-wasm': {
    id: 'onnx-wasm',
    label: 'ONNX Runtime · CPU / WebAssembly',
    backend: 'onnx',
    defaultRepo: 'zeroweight-ai/ZeroTTS',
    experimental: false,
    capabilities: {
      cfg: true, quantizedWeights: false, webgpu: false, reproducible: true,
    },
  },
};

export const DEFAULT_ENGINE: EngineId = 'ggml-cpu';

export interface EngineSupport {
  supported: boolean;
  reason?: string;
}

export function engineDefinition(id: EngineId = DEFAULT_ENGINE): EngineDefinition {
  const definition = ENGINES[id];
  if (!definition) throw new Error(`Unknown ZeroTTS web engine: ${String(id)}`);
  return definition;
}

/** Convert the original backend/device pair to the stable engine id. */
export function engineFromLegacy(
  backend: Backend = 'ggml', device: GgmlDevice = 'cpu',
): EngineId {
  if (backend === 'onnx') return 'onnx-wasm';
  return device === 'webgpu' ? 'ggml-webgpu' : 'ggml-cpu';
}

/** Lightweight compatibility probe. This deliberately checks capability, not
 * performance: the benchmark is the only reliable way to choose the fastest
 * engine on a particular browser/GPU combination. */
export async function probeEngine(id: EngineId): Promise<EngineSupport> {
  engineDefinition(id); // reject unknown ids consistently
  if (typeof WebAssembly === 'undefined') {
    return { supported: false, reason: 'WebAssembly is unavailable' };
  }
  if (id !== 'ggml-webgpu') return { supported: true };
  if (typeof navigator === 'undefined') {
    return { supported: false, reason: 'WebGPU probing requires a browser' };
  }
  if (typeof crossOriginIsolated === 'undefined' || !crossOriginIsolated) {
    return {
      supported: false,
      reason: 'This WebGPU runtime requires cross-origin isolation (HTTPS/localhost + COOP/COEP)',
    };
  }

  const gpu = (navigator as Navigator & {
    gpu?: {
      requestAdapter(options?: { powerPreference?: 'high-performance' }): Promise<{
        features: { has(name: string): boolean };
      } | null>;
    };
  }).gpu;
  if (!gpu) return { supported: false, reason: 'WebGPU is unavailable' };
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { supported: false, reason: 'No WebGPU adapter was found' };
    if (!adapter.features.has('shader-f16')) {
      return { supported: false, reason: 'The adapter does not expose shader-f16' };
    }
    return { supported: true };
  } catch (error) {
    return { supported: false, reason: (error as Error)?.message ?? String(error) };
  }
}

/** Pick the first compatible engine in caller-defined order. The order should
 * come from benchmarks or a user preference, not from a hard-coded assumption
 * that a GPU is faster. */
export async function selectSupportedEngine(
  preferred: readonly EngineId[],
): Promise<EngineId> {
  if (!preferred.length) throw new Error('At least one engine preference is required');
  const failures: string[] = [];
  for (const id of preferred) {
    const support = await probeEngine(id);
    if (support.supported) return id;
    failures.push(`${id}: ${support.reason ?? 'unsupported'}`);
  }
  throw new Error(`No preferred ZeroTTS engine is supported (${failures.join('; ')})`);
}

/** Options safe to send across a Worker boundary. */
export interface EngineLoadOptions {
  engine?: EngineId;
  repo?: string;
  revision?: string;
  gguf?: string;
  threads?: number;
  /** WebGPU defaults to CPU fallback. Set `none` when the selected engine must
   * be used exactly (for example during a benchmark). */
  fallback?: EngineFallback;
}

/** Stable, public engine identifiers and metadata.
 *
 * UI code and applications select one id; the loader is responsible for the
 * runtime details. Adding a backend therefore does not change the
 * streaming or generation API.
 */
export type Backend = 'ggml' | 'onnx';
export type EngineId = 'ggml-cpu' | 'onnx-wasm';

export interface EngineCapabilities {
  /** Classifier-free guidance (`cfgScale > 1`). */
  cfg: boolean;
  /** Can load quantized GGUF weights. */
  quantizedWeights: boolean;
  /** Same seed is expected to reproduce the same codes on the same platform. */
  reproducible: boolean;
}

export interface EngineDefinition {
  id: EngineId;
  label: string;
  backend: Backend;
  defaultRepo: string;
  experimental: boolean;
  capabilities: EngineCapabilities;
}

export const ENGINES: Readonly<Record<EngineId, EngineDefinition>> = {
  'ggml-cpu': {
    id: 'ggml-cpu',
    label: 'GGML · CPU / WebAssembly',
    backend: 'ggml',
    defaultRepo: 'zeroweight-ai/ZeroTTS-GGUF',
    experimental: false,
    capabilities: {
      cfg: false, quantizedWeights: true, reproducible: true,
    },
  },
  'onnx-wasm': {
    id: 'onnx-wasm',
    label: 'ONNX Runtime · CPU / WebAssembly',
    backend: 'onnx',
    defaultRepo: 'zeroweight-ai/ZeroTTS',
    experimental: false,
    capabilities: {
      cfg: true, quantizedWeights: false, reproducible: true,
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

/** Convert the original backend selector to the stable engine id. */
export function engineFromLegacy(
  backend: Backend = 'ggml',
): EngineId {
  return backend === 'onnx' ? 'onnx-wasm' : 'ggml-cpu';
}

/** Lightweight compatibility probe. Performance should still be measured on
 * each target browser and machine. */
export async function probeEngine(id: EngineId): Promise<EngineSupport> {
  engineDefinition(id); // reject unknown ids consistently
  if (typeof WebAssembly === 'undefined') {
    return { supported: false, reason: 'WebAssembly is unavailable' };
  }
  return { supported: true };
}

/** Pick the first compatible engine in caller-defined order. The order should
 * come from benchmarks or a user preference. */
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
}

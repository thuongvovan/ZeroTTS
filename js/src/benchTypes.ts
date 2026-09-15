import type { GgmlTimings } from './ggmlBackend';
import type { EngineId } from './engine';

export type BenchBackend = EngineId;

export interface BenchConfig {
  backend: BenchBackend;
  gguf: string;
  threads: number;
  frames: number;
  runs: number;
  seed: number;
  voice: string;
  text: string;
  modelBase: string;
  ggufBase: string;
}

export interface RunResult {
  wallMs: number;
  realtime: number;
  hash: string;
  firstFrame: number[];
  stages?: GgmlTimings;
}

export interface BenchResult {
  backend: BenchBackend;
  label: string;
  config: BenchConfig;
  status: 'ok' | 'error';
  loadMs?: number;
  warmupMs?: number;
  effectiveThreads?: number;
  measurements?: RunResult[];
  medianMs?: number;
  medianRealtime?: number;
  driftFromFirst?: number[];
  reproducible?: boolean;
  error?: string;
}

export type BenchWorkerRequest = { type: 'run'; config: BenchConfig };
export type BenchWorkerResponse =
  | { type: 'log'; message: string }
  | { type: 'result'; result: BenchResult };

export function backendLabel(config: BenchConfig, threads = config.threads): string {
  if (config.backend === 'onnx-wasm') return `ONNX/WASM · ${threads}t`;
  const device = config.backend === 'ggml-webgpu' ? 'WebGPU' : `CPU ${threads}t`;
  return `GGML ${config.gguf.replace(/^zerotts-|\.gguf$/g, '')} · ${device}`;
}

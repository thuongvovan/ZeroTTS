/**
 * Main-thread handle on the synthesis worker (see worker.ts for why there is
 * one). Turns the message port into promises and an async iterator, so the UI
 * code reads the same as it did when the model ran inline.
 *
 * The worker module is NOT imported statically anywhere on the page: the
 * `new Worker(new URL(...))` form is what lets Vite build it as its own bundle,
 * which is what keeps onnxruntime-web off the UI thread's dependency graph.
 */

import type { DownloadProgress } from './cache';
import {
  DEFAULT_ENGINE, engineDefinition, engineFromLegacy,
} from './engine';
import type { EngineId, EngineLoadOptions } from './engine';
import type { Backend, GgmlDevice } from './repo';
import {
  GenerateParams, LoadedInfo, WorkerRequest, WorkerResponse,
} from './workerProtocol';

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onProgress?: (p: DownloadProgress) => void;
  onChunk?: (chunk: Float32Array) => void;
}

export class TtsWorker {
  private worker: Worker;
  private pending = new Map<number, Pending>();
  private nextId = 1;
  private hasLoadAttempted = false;
  private lastLoadOptions: EngineLoadOptions = {};
  private activeEngine: EngineId | null = null;
  private disposed = false;

  constructor() {
    this.worker = this.createWorker();
  }

  private createWorker(): Worker {
    const worker = new Worker(new URL('./worker.ts', import.meta.url), {
      type: 'module',
      name: 'zerotts',
    });
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      const entry = this.pending.get(message.id);
      if (!entry) return;
      switch (message.type) {
        case 'progress': entry.onProgress?.(message.progress); break;
        case 'chunk': entry.onChunk?.(message.chunk); break;
        case 'result':
          this.pending.delete(message.id);
          entry.resolve(message.value);
          break;
        case 'error':
          this.pending.delete(message.id);
          entry.reject(new Error(message.message));
          break;
      }
    };
    // A worker that dies (OOM on a 900 MB model is the realistic cause) would
    // otherwise leave every caller awaiting forever.
    worker.onerror = (event) => this.failAll(event.message || 'worker crashed');
    worker.onmessageerror = () => this.failAll('worker sent an unreadable message');
    return worker;
  }

  private restartWorker(): void {
    this.failAll('engine switched; previous operation was cancelled');
    this.worker.terminate();
    this.worker = this.createWorker();
    this.activeEngine = null;
  }

  get currentEngine(): EngineId | null {
    return this.activeEngine;
  }

  private failAll(message: string): void {
    for (const [, entry] of this.pending) entry.reject(new Error(message));
    this.pending.clear();
  }

  private request<T>(
    message: WorkerRequest, hooks: Omit<Pending, 'resolve' | 'reject'> = {},
  ): Promise<T> {
    if (this.disposed) return Promise.reject(new Error('ZeroTTS worker was disposed'));
    return new Promise<T>((resolve, reject) => {
      this.pending.set(message.id, { resolve: resolve as Pending['resolve'], reject, ...hooks });
      this.worker.postMessage(message);
    });
  }

  downloadInfo(options?: EngineLoadOptions): Promise<{ bytes: number; cached: boolean }>;
  /** @deprecated Use downloadInfo({ engine, repo, gguf }). */
  downloadInfo(repo: string, backend?: Backend, gguf?: string): Promise<{
    bytes: number; cached: boolean;
  }>;
  downloadInfo(
    optionsOrRepo: EngineLoadOptions | string = {}, backend?: Backend, gguf?: string,
  ): Promise<{ bytes: number; cached: boolean }> {
    const options: EngineLoadOptions = typeof optionsOrRepo === 'string'
      ? { repo: optionsOrRepo, engine: engineFromLegacy(backend), gguf }
      : optionsOrRepo;
    return this.request({ type: 'downloadInfo', id: this.nextId++, options });
  }

  clearCache(): Promise<void> {
    return this.request({ type: 'clearCache', id: this.nextId++ });
  }

  load(options?: EngineLoadOptions, onProgress?: (p: DownloadProgress) => void): Promise<LoadedInfo>;
  /** @deprecated Use load({ engine, repo, gguf }, onProgress). */
  load(
    repo: string, onProgress?: (p: DownloadProgress) => void,
    backend?: Backend, gguf?: string, ggmlDevice?: GgmlDevice,
  ): Promise<LoadedInfo>;
  load(
    optionsOrRepo: EngineLoadOptions | string = {}, onProgress?: (p: DownloadProgress) => void,
    backend?: Backend, gguf?: string, ggmlDevice?: GgmlDevice,
  ): Promise<LoadedInfo> {
    const options: EngineLoadOptions = typeof optionsOrRepo === 'string'
      ? {
          repo: optionsOrRepo,
          engine: engineFromLegacy(backend, ggmlDevice),
          gguf,
        }
      : optionsOrRepo;
    // Terminating the previous Worker is the only portable way to release its
    // WASM heap, pthread pool and WebGPU context before another engine loads.
    if (this.hasLoadAttempted) this.restartWorker();
    this.hasLoadAttempted = true;
    this.lastLoadOptions = { ...options };
    return this.request<LoadedInfo>(
      { type: 'load', id: this.nextId++, options }, { onProgress }).then((loaded) => {
        this.activeEngine = loaded.engine;
        return loaded;
      });
  }

  /** Switch engines without changing the generate/streaming API. Any active
   * request is cancelled and the old runtime memory is released with its
   * Worker before the new one starts loading. */
  switchEngine(
    selection: EngineId | EngineLoadOptions,
    onProgress?: (p: DownloadProgress) => void,
  ): Promise<LoadedInfo> {
    const options: EngineLoadOptions = typeof selection === 'string'
      ? { ...this.lastLoadOptions, engine: selection }
      : { ...this.lastLoadOptions, ...selection };
    const previous = engineDefinition(this.lastLoadOptions.engine ?? DEFAULT_ENGINE);
    const next = engineDefinition(options.engine ?? DEFAULT_ENGINE);
    // A default repo follows the engine; an explicitly self-hosted repo remains
    // untouched because it may intentionally contain assets for every engine.
    if (options.repo === previous.defaultRepo && previous.defaultRepo !== next.defaultRepo) {
      options.repo = next.defaultRepo;
    }
    return this.load(options, onProgress);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.failAll('ZeroTTS worker was disposed');
    this.worker.terminate();
    this.hasLoadAttempted = false;
    this.lastLoadOptions = {};
    this.activeEngine = null;
  }

  /**
   * Stream a take. Chunks arrive as they are decoded; `cancel()` on the returned
   * handle stops the worker's loop at the next frame boundary.
   */
  generate(params: GenerateParams): { chunks: AsyncIterable<Float32Array>; cancel: () => void } {
    const id = this.nextId++;

    // A queue rather than a callback: the consumer (`for await`) may be slower
    // than the worker, and chunks must not be dropped when it is.
    const queue: Float32Array[] = [];
    // A plain object, not locals: these are written from callbacks, and the
    // narrowing TypeScript applies to a `let` read inside the generator would be
    // wrong about both of them.
    const state = { finished: false, failure: null as Error | null };
    let notify: (() => void) | null = null;

    const wake = () => { notify?.(); notify = null; };

    this.request(
      { type: 'generate', id, params },
      { onChunk: (chunk) => { queue.push(chunk); wake(); } },
    ).then(
      () => { state.finished = true; wake(); },
      (error: Error) => { state.failure = error; state.finished = true; wake(); },
    );

    const chunks: AsyncIterable<Float32Array> = {
      async *[Symbol.asyncIterator]() {
        for (;;) {
          while (queue.length) yield queue.shift() as Float32Array;
          if (state.failure) throw state.failure;
          if (state.finished) return;
          await new Promise<void>((resolve) => { notify = resolve; });
        }
      },
    };

    return {
      chunks,
      cancel: () => this.worker.postMessage(
        { type: 'cancel', id: this.nextId++, target: id } satisfies WorkerRequest),
    };
  }
}

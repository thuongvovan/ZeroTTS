import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';

/** Runtime files that live beside the installed package entry. Keeping these
 * URL expressions on the main-thread side lets a consuming bundler discover,
 * copy and fingerprint every file needed by the otherwise pre-built Worker. */
export interface RuntimeAssets {
  ggmlThreadedModule: string;
  ggmlThreadedWasm: string;
  ggmlSingleModule: string;
  ggmlSingleWasm: string;
  ortWasm: string;
}

export const DEFAULT_RUNTIME_ASSETS: RuntimeAssets = {
  ggmlThreadedModule: new URL(
    '../runtime/ggml/zerotts-wasm.js', import.meta.url,
  ).href,
  ggmlThreadedWasm: new URL(
    '../runtime/ggml/zerotts-wasm.wasm', import.meta.url,
  ).href,
  ggmlSingleModule: new URL(
    '../runtime/ggml-single/zerotts-wasm.js', import.meta.url,
  ).href,
  ggmlSingleWasm: new URL(
    '../runtime/ggml-single/zerotts-wasm.wasm', import.meta.url,
  ).href,
  ortWasm: ortWasmUrl,
};

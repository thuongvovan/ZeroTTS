import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  root,
  // Package assets must resolve next to the installed ESM entry rather than at
  // the consuming application's origin root.
  base: './',
  publicDir: false,
  resolve: {
    // Transformers.js imports the broad ORT entry even when only its tokenizer
    // is used. Point that exact import at the CPU/WASM build so the package does
    // not ship an unused JSEP/WebGPU WASM binary.
    alias: [{ find: /^onnxruntime-web$/, replacement: 'onnxruntime-web/wasm' }],
  },
  build: {
    target: 'es2022',
    outDir: fileURLToPath(new URL('../dist', import.meta.url)),
    emptyOutDir: true,
    assetsInlineLimit: 0,
    rollupOptions: {
      // Use an ESM application entry rather than Vite's library mode. Library
      // mode always inlines imported assets, which would turn tens of MB of
      // WASM into base64 JavaScript and defeat browser caching.
      input: fileURLToPath(new URL('./src/index.ts', import.meta.url)),
      preserveEntrySignatures: 'strict',
      output: {
        entryFileNames: 'index.js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
  worker: {
    format: 'es',
    rollupOptions: {
      output: {
        // A consuming bundler treats the built Worker as an opaque asset. Keep
        // its lazy GGML backend in the Worker so no sibling JS chunk is lost.
        inlineDynamicImports: true,
      },
    },
  },
});

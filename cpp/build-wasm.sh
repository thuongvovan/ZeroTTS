#!/usr/bin/env bash
# Build threaded and single-thread CPU browser runtimes, then drop them where
# Vite serves them.
#
# Needs the Emscripten SDK on PATH (source ~/emsdk/emsdk_env.sh). Threaded CPU
# artifacts land in js/public/ggml/ and the HTTP-safe single-thread fallback in
# js/public/ggml-single/. Vite copies these directories verbatim. Each JS file
# locates its sibling WASM, so each pair must stay together.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v emcmake >/dev/null; then
    echo "emcmake not found — source your emsdk_env.sh first" >&2
    exit 1
fi

emcmake cmake -B build-wasm -DCMAKE_BUILD_TYPE=Release -DZEROTTS_WASM_THREADS=ON .
cmake --build build-wasm -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

emcmake cmake -B build-wasm-single -DCMAKE_BUILD_TYPE=Release -DZEROTTS_WASM_THREADS=OFF .
cmake --build build-wasm-single -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

mkdir -p ../js/public/ggml ../js/public/ggml-single
cp build-wasm/zerotts-wasm.js build-wasm/zerotts-wasm.wasm ../js/public/ggml/
cp build-wasm-single/zerotts-wasm.js build-wasm-single/zerotts-wasm.wasm ../js/public/ggml-single/

# bench-ggml.html fetches GGUFs from /ggml/models/. A relative symlink keeps the
# (large, gitignored) weights in cpp/models rather than copying them into the
# served tree. Vite's dev server follows it; both ends of the link are ignored.
ln -sfn ../../../cpp/models ../js/public/ggml/models

ls -la ../js/public/ggml/ ../js/public/ggml-single/

#!/usr/bin/env bash
# Build threaded and single-thread CPU browser runtimes, then stage the package
# assets consumed by Vite and downstream application bundlers.
#
# Needs the Emscripten SDK on PATH (source ~/emsdk/emsdk_env.sh). Threaded CPU
# artifacts land in js/runtime/ggml/ and the HTTP-safe single-thread fallback in
# js/runtime/ggml-single/.
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

mkdir -p ../js/runtime/ggml ../js/runtime/ggml-single

# Emscripten's pthread launcher hardcodes the un-hashed output filename. The
# runtime may be renamed by an application bundler, so make it spawn its current
# module URL instead. Keep this checked: a changed Emscripten template must fail
# the build rather than produce a package whose threaded mode breaks at runtime.
threaded_js=build-wasm/zerotts-wasm.js
if ! grep -q 'new URL("zerotts-wasm.js",import.meta.url)' "$threaded_js"; then
    echo "unexpected Emscripten pthread launcher in $threaded_js" >&2
    exit 1
fi
sed 's/new URL("zerotts-wasm.js",import.meta.url)/new URL(import.meta.url)/g' \
    "$threaded_js" > ../js/runtime/ggml/zerotts-wasm.js
cp build-wasm/zerotts-wasm.wasm ../js/runtime/ggml/
cp build-wasm-single/zerotts-wasm.js build-wasm-single/zerotts-wasm.wasm \
    ../js/runtime/ggml-single/

# bench-ggml.html fetches GGUFs from /ggml/models/. A relative symlink keeps the
# (large, gitignored) weights in cpp/models rather than copying them into the
# served tree. Vite's dev server follows it; both ends of the link are ignored.
mkdir -p ../js/public/ggml
ln -sfn ../../../cpp/models ../js/public/ggml/models

ls -la ../js/runtime/ggml/ ../js/runtime/ggml-single/

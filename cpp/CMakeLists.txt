cmake_minimum_required(VERSION 3.16)
project(zerotts-ggml CXX C)

option(ZEROTTS_WASM_THREADS "Build the browser runtime with pthreads" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The browser and native builds use ggml's CPU backend.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if (EMSCRIPTEN)
    # WASM SIMD is the entire reason this is worth doing in a browser; without
    # it ggml falls back to scalar kernels and loses to onnxruntime-web.
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    add_compile_options(-msimd128 -O3)
    if (ZEROTTS_WASM_THREADS)
        add_compile_options(-pthread)
    endif()
endif()

add_subdirectory(vendor-ggml)

add_library(zerotts STATIC src/zerotts.cpp)
target_include_directories(zerotts PUBLIC include)
target_link_libraries(zerotts PUBLIC ggml)

if (EMSCRIPTEN)
    add_executable(zerotts-wasm src/wasm.cpp)
    target_link_libraries(zerotts-wasm PRIVATE zerotts)
    target_link_options(zerotts-wasm PRIVATE
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sEXPORT_NAME=createZeroTTS
        -sALLOW_MEMORY_GROWTH=1
        -sMAXIMUM_MEMORY=4GB
        -sSTACK_SIZE=1048576
        -sEXPORTED_RUNTIME_METHODS=['HEAPU8','HEAPF32','HEAP32','ccall','cwrap']
        -sENVIRONMENT=web,worker,node
        -O3
    )
    if (ZEROTTS_WASM_THREADS)
        target_link_options(zerotts-wasm PRIVATE
            -pthread
            -sPTHREAD_POOL_SIZE=8
        )
    endif()
else()
    add_executable(zerotts-bench test/bench.cpp)
    target_link_libraries(zerotts-bench PRIVATE zerotts)

    add_executable(zerotts-quality test/quant_quality.cpp)
    target_link_libraries(zerotts-quality PRIVATE zerotts)
endif()

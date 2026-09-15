import assert from 'node:assert/strict';

import {
  DEFAULT_ENGINE, ENGINES, engineDefinition, engineFromLegacy,
  selectSupportedEngine,
} from '../src/engine.ts';

assert.equal(DEFAULT_ENGINE, 'ggml-cpu');
assert.deepEqual(Object.keys(ENGINES), ['ggml-cpu', 'ggml-webgpu', 'onnx-wasm']);
assert.equal(engineFromLegacy('ggml', 'cpu'), 'ggml-cpu');
assert.equal(engineFromLegacy('ggml', 'webgpu'), 'ggml-webgpu');
assert.equal(engineFromLegacy('onnx', 'webgpu'), 'onnx-wasm');
assert.equal(engineDefinition('ggml-webgpu').capabilities.webgpu, true);
assert.equal(engineDefinition('onnx-wasm').capabilities.cfg, true);
assert.equal(engineDefinition('ggml-cpu').defaultRepo, engineDefinition('ggml-webgpu').defaultRepo);
assert.equal(await selectSupportedEngine(['ggml-cpu', 'onnx-wasm']), 'ggml-cpu');

console.log('engine registry: ok');

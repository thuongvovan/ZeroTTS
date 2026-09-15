import assert from 'node:assert/strict';

import {
  DEFAULT_ENGINE, ENGINES, engineDefinition, engineFromLegacy,
  selectSupportedEngine,
} from '../src/engine.ts';

assert.equal(DEFAULT_ENGINE, 'ggml-cpu');
assert.deepEqual(Object.keys(ENGINES), ['ggml-cpu', 'onnx-wasm']);
assert.equal(engineFromLegacy('ggml'), 'ggml-cpu');
assert.equal(engineFromLegacy('onnx'), 'onnx-wasm');
assert.equal(engineDefinition('onnx-wasm').capabilities.cfg, true);
assert.equal(await selectSupportedEngine(['ggml-cpu', 'onnx-wasm']), 'ggml-cpu');

console.log('engine registry: ok');

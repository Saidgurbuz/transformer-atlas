import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { tensorExample, logicalValues, view, tensor, cat, squeeze, storageIndex } from '../src/lessons/tensors.js';
const expected = JSON.parse(readFileSync(new URL('./fixtures/lessons/tensor_ops.json', import.meta.url)));
const actual = tensorExample();
for (const [key, wanted] of Object.entries(expected)) {
  const t = actual[key];
  assert.deepEqual(t.shape, wanted.shape, `${key}: shape`);
  assert.deepEqual(Array.from(logicalValues(t)), wanted.data, `${key}: logical values`);
  assert.deepEqual(t.strides, wanted.strides, `${key}: strides`);
  assert.equal(t.data === actual.x.data, wanted.sharesX, `${key}: storage alias`);
}
assert.throws(() => view(actual.heads, [2, 6, 3]));
assert.throws(() => cat([actual.x, tensor([1], [1])], 1));
assert.equal(squeeze(actual.x, 1), actual.x);
assert.throws(() => storageIndex(actual.x, [0, 0, 9]));
console.log('12 tensor operations match PyTorch values, shapes, strides and aliases; invalid operations rejected.');

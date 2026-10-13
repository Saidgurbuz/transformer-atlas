import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createEncoderDecoder, encoderDecoderForward } from '../src/lessons/encoder-decoder.js';
const expected = JSON.parse(readFileSync(new URL('./fixtures/lessons/encdec.json', import.meta.url)));
const m = createEncoderDecoder();
let checks = 0;
function close(got, want) {
  if (want.shape) assert.deepEqual(got.shape, want.shape);
  const a = got.data ?? got, b = want.data ?? want;
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) { const v = b[i] === null ? -Infinity : b[i]; assert.ok(a[i] === v || Math.abs(a[i] - v) < 1e-9); }
  checks++;
}
for (const c of expected) {
  const got = encoderDecoderForward(m, c.source, c.target);
  assert.deepEqual(Object.keys(got.trace).sort(), Object.keys(c.trace).sort());
  for (const [key, want] of Object.entries(c.trace)) close(got.trace[key], want);
}
const a = encoderDecoderForward(m, [2, 3, 4, 5], [0, 1, 2]);
close(a.logits, encoderDecoderForward(m, [2, 3, 4], [0, 1, 2]).logits);
close(a.logits.data.slice(0, 12), encoderDecoderForward(m, [2, 3, 4, 5], [0, 1, 3]).logits.data.slice(0, 12));
assert.throws(() => encoderDecoderForward(m, [5, 5], [0]));
assert.throws(() => encoderDecoderForward(m, [2], [5]));
console.log(`${checks} PyTorch encoder/decoder comparisons passed, plus source-padding invariance and decoder causality.`);

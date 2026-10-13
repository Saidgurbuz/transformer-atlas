import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { streamAttentionRow } from '../src/lessons/flash.js';
const cases = JSON.parse(readFileSync(new URL('./fixtures/lessons/flash.json', import.meta.url)));
let checks = 0;
function close(a, b) {
  assert.equal(a.length, b.length);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-10)); checks++;
}
for (const c of cases) for (const [size, prefixes] of Object.entries(c.prefixes)) {
  const out = streamAttentionRow(c.scores.map(v => v === null ? -Infinity : v), c.values, c.width, Number(size));
  close(out.output, c.output);
  out.tiles.forEach((tile, i) => close(tile.output, prefixes[i]));
}
assert.throws(() => streamAttentionRow([-Infinity], [0], 1));
assert.throws(() => streamAttentionRow([NaN], [0], 1));
assert.throws(() => streamAttentionRow([Infinity], [0], 1));
assert.throws(() => streamAttentionRow([0], [0], 1, 0));
console.log(`${checks} dense PyTorch output/prefix comparisons passed, including extreme scores and masked leading/trailing tiles.`);

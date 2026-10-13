// Branch-aware source correspondence and exact cell coordinates, including GQA.
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { MODEL_PY } from '../src/generated/model_py.js';
import { codeLines, linesForKey, lineForKey, keysForLine, sourceForKey } from '../src/core/codemap.js';
import { normalizeCell, cellValue, selectionForCell } from '../src/core/cells.js';
import { nodeForLine, validFor } from '../src/atlas/nodes.js';
import { DEFAULT_CONFIG, validateConfig } from '../src/engine/model.js';

assert.equal(MODEL_PY, readFileSync(new URL('../reference/model.py', import.meta.url), 'utf8'), 'embedded source must stay fresh');
let checked = 0;
for (const name of readdirSync(new URL('./fixtures/', import.meta.url)).filter((n) => n.endsWith('.json'))) {
  const fx = JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
  const state = { config: fx.config, trace: fx.trace, ids: fx.ids, sel: { layer: 0, head: fx.config.n_head - 1, token: 5 } };
  for (const key of fx.order) {
    const line = lineForKey(key, fx.config), layer = Number(/^L(\d+)/.exec(key)?.[1] ?? 0);
    assert.ok(line, `${name}: ${key} needs a source line`);
    assert.ok(keysForLine(line, fx.config, layer, fx.trace).includes(key));
    const target = nodeForLine(line, fx.config, { layer, head: state.sel.head });
    assert.ok(target && validFor(target, fx.config), `${key} must lead to an available node`);
    for (const other of linesForKey(key)) if (other !== line) {
      assert.equal(keysForLine(other, fx.config, layer, fx.trace).includes(key), false, 'inactive branch must not claim this tensor');
      assert.equal(nodeForLine(other, fx.config, { layer }), null);
    }
    if (fx.trace[key].shape.length >= 2) {
      const t = fx.trace[key], c = normalizeCell(state, key, { band: 99, r: 99, x: 99 });
      assert.equal(c.x, t.shape.at(-1) - 1);
      assert.equal(c.r, t.shape.at(-2) - 1);
      assert.equal(cellValue(state, key, c), t.data.at(-1));
    }
    checked++;
  }
}
const cfg = { ...DEFAULT_CONFIG, n_head: 4, n_kv_head: 2 };
const state = { config: cfg, ids: [0, 1, 2], sel: { head: 3, token: 1 }, trace: { 'L1.attn.k_heads': { shape: [2, 3, 4], data: Float64Array.from({ length: 24 }, (_, i) => i) } } };
assert.deepEqual(normalizeCell(state, 'L1.attn.k_heads'), { band: 1, r: 1, x: 0 });
assert.deepEqual(selectionForCell(state, 'L1.attn.k_heads', { band: 1, r: 2, x: 3 }), { layer: 1, head: 3, token: 2 });
assert.equal(selectionForCell(state, 'L1.attn.k_heads', { band: 0 }).head, 0);
assert.deepEqual(normalizeCell(state, 'L1.attn.k_heads', { r: -5, x: NaN, band: Infinity }), { band: 0, r: 0, x: 0 });
assert.equal(sourceForKey('L0.ln1', { ...cfg, norm: 'rmsnorm' }).helper, 'RMSNorm');
assert.equal(sourceForKey('L0.attn.k_rot', cfg).helper, 'apply_rope');
for (const patch of [{ norm_eps: 0 }, { norm_eps: NaN }, { rope_base: Infinity }, { prefix_len: -1 }, { prefix_len: 0.5 }, { residual: 1 }]) assert.ok(validateConfig({ ...DEFAULT_CONFIG, ...patch }).length);
assert.deepEqual(validateConfig(DEFAULT_CONFIG), []);
assert.ok(codeLines.some((l) => /def decode_step/.test(l.raw)));
console.log(`${checked} trace/source round trips; branch selection, GQA coordinates, freshness and configuration checks passed.`);

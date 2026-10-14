// Input edits resume next-token prediction; exploratory selections survive model refreshes.
import assert from 'node:assert/strict';
import { store } from '../src/core/store.js';
import { DEFAULT_CONFIG, cloneModel } from '../src/engine/index.js';

store.state.config = { ...DEFAULT_CONFIG, n_layer: 1, n_head: 2, n_kv_head: 2, d_model: 8, d_ff: 16, block_size: 12 };
const observed = [];
const off = store.on('trace', st => observed.push(st.sel.token));
store.init();
assert.equal(observed.at(-1), store.state.ids.length - 1, 'initial trace already follows the final token');

store.select({ token: 0 });
store.setText('the cat');
assert.equal(store.state.sel.token, 6, 'typing resumes prediction at the end');
assert.equal(observed.at(-1), 6, 'trace subscribers see the new prediction position');

store.select({ token: 2 });
store.setText('the dog');
assert.equal(store.state.sel.token, 6, 'equal-length replacements also follow the end');
store.setText('the dog sat');
assert.equal(store.state.sel.token, 10, 'appending input advances prediction');

store.select({ token: 1 });
store.recompute();
assert.equal(store.state.sel.token, 1, 'rerunning the current input preserves an explicit selection');
store.setConfig({ attn_scale: false });
assert.equal(store.state.sel.token, 1, 'changing model settings preserves the explored token');
store.reinitialize({ seed: 42 });
assert.equal(store.state.sel.token, 1, 'refreshing weights preserves the explored token');
store.setModel(cloneModel(store.state.model));
assert.equal(store.state.sel.token, 1, 'swapping same-shape weights preserves the explored token');

store.select({ token: 10 });
store.setConfig({ block_size: 4 });
assert.equal(store.state.sel.token, 3, 'context reduction clamps the selection');
store.setText('the cat sat on the mat', { quiet: true });
assert.equal(store.state.sel.token, 3, 'prediction follows the final retained token when input is truncated');
assert.equal(store.state.ids.length, 4);

store.focusCell('L0.attn.probs', { band: 1, r: 3, x: 3 });
store.setText('the');
assert.equal(store.state.sel.token, 2);
assert.equal(store.state.cellFocus.key, 'L0.attn.probs', 'source navigation remains available after edits');
assert.equal(store.state.cellFocus.cell.r, 2, 'stale source coordinates are clamped');
assert.equal(store.state.cellFocus.cell.x, 2);

store.setText('a');
assert.equal(store.state.sel.token, 0, 'a single token is a valid prediction position');
store.setText('');
assert.equal(store.state.sel.token, 0, 'empty input retains the existing safe fallback');
assert.equal(store.state.ids.length, 1);
off();
console.log('Input edits, trace events, exploratory selections, context limits and source focus validated.');

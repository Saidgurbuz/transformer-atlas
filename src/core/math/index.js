// Math catalogue: one entry per trace-key pattern (and 'concept:*' entries).
// Each entry: {title, tex, compact, symbols, derivation, live?}. live(ctx) recomputes
// one cell from upstream tensors and returns {tex, result, expected, note}; tests/math.mjs
// checks that every equation compiles and every live result equals the traced value.
import { parseKey } from '../describe.js';
import { ENTRIES as attention } from './attention.js';
import { ENTRIES as embed } from './embed.js';
import { ENTRIES as positions } from './positions.js';
import { ENTRIES as qkv } from './qkv.js';
import { ENTRIES as norm } from './norm.js';
import { ENTRIES as mlp } from './mlp.js';
import { ENTRIES as head } from './head.js';
import { ENTRIES as concepts } from './concepts.js';

export const MATH = { ...embed, ...positions, ...qkv, ...attention, ...norm, ...mlp, ...head, ...concepts };

export function mathFor(key) { return MATH[parseKey(key).pattern] ?? MATH[key] ?? null; }

/**
 * Build the context a live() function receives for one cell of a traced tensor.
 * cell: {r, band, x} as produced by the 3D sheet picker or a heatmap.
 */
export function liveContext(state, key, cell) {
  const { layer } = parseKey(key);
  const t = state.trace[key];
  const three = t.shape.length === 3;
  return {
    key, trace: state.trace, cfg: state.config, model: state.model, ids: state.ids,
    label: (i) => state.tokenizer.label(state.ids[i]),
    vocab: (v) => state.tokenizer.label(v),
    layer: layer ?? 0,
    head: three ? cell.band : state.sel.head,
    row: cell.r,
    col: cell.x,
  };
}

export function runLive(state, key, cell) {
  const e = mathFor(key);
  if (!e?.live || !state.trace[key]) return null;
  try { return e.live(liveContext(state, key, cell)); } catch (err) { return { tex: '', error: String(err) }; }
}

// One coordinate convention for DOM cells, canvas slices, maths and source code.
import { parseKey } from './describe.js';

const index = (n, size) => Math.max(0, Math.min(size - 1, Number.isFinite(Number(n)) ? Math.trunc(Number(n)) : 0));
export const isKVHeadKey = (key) => /\.attn\.(k_heads|v_heads|k_rot)$/.test(key);

export function normalizeCell(state, key, cell = null) {
  const t = state.trace?.[key];
  if (!t || t.shape.length < 2) return null;
  const three = t.shape.length === 3;
  const head = isKVHeadKey(key) ? Math.floor(state.sel.head / (state.config.n_head / state.config.n_kv_head)) : state.sel.head;
  return {
    band: three ? index(cell?.band ?? head, t.shape[0]) : 0,
    r: index(cell?.r ?? state.sel.token, t.shape[three ? 1 : 0]),
    x: index(cell?.x ?? 0, t.shape[three ? 2 : 1]),
  };
}

export function cellValue(state, key, cell) {
  const t = state.trace?.[key];
  if (!t) return undefined;
  if (!t.shape.length) return t.data[0];
  const c = normalizeCell(state, key, cell);
  if (!c) return undefined;
  const cols = t.shape.at(-1), rows = t.shape.at(-2);
  return t.data[(c.band * rows + c.r) * cols + c.x];
}

export function cellLabel(state, key, cell) {
  const c = normalizeCell(state, key, cell), t = state.trace?.[key];
  return c ? `${key}[${t.shape.length === 3 ? `${c.band}, ` : ''}${c.r}, ${c.x}]` : key;
}

export function selectionForCell(state, key, cell) {
  const c = normalizeCell(state, key, cell), { layer } = parseKey(key), patch = {};
  if (layer != null) patch.layer = layer;
  if (!c) return patch;
  if (state.trace[key].shape.at(-2) === state.ids.length) patch.token = c.r;
  if (state.trace[key].shape.length === 3) {
    // A KV slice can serve several query heads. Keep the current one when possible.
    const group = state.config.n_head / state.config.n_kv_head;
    patch.head = isKVHeadKey(key)
      ? (Math.floor(state.sel.head / group) === c.band ? state.sel.head : c.band * group)
      : c.band;
  }
  return patch;
}

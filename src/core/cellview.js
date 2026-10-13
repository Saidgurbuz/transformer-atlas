import { h, fmt } from '../ui/index.js';
import { normalizeCell, cellLabel, cellValue, isKVHeadKey } from './cells.js';

/** Explicit coordinates make the code/cell bridge usable with a keyboard or touch. */
export function cellControls(state, key, cell, onChange) {
  const c = normalizeCell(state, key, cell), t = state.trace?.[key];
  if (!t) return null;
  const el = h('div', { class: 'cell-focus', dataset: { focusKey: key, ...(c ? { focusCell: `${c.r},${c.x},${c.band}` } : {}) } },
    h('div', { class: 'mono small cell-address' }, cellLabel(state, key, c)),
    h('div', { class: 'num cell-value' }, `= ${fmt(cellValue(state, key, c), 6)}`));
  if (!c) return el;
  const axes = t.shape.length === 3 ? [['band', isKVHeadKey(key) ? 'KV head' : 'head', t.shape[0]], ['r', 'row', t.shape[1]], ['x', 'column', t.shape[2]]]
    : [['r', 'row', t.shape[0]], ['x', 'column', t.shape[1]]];
  const coordinates = axes.map(([axis, label, size]) => h('label', { class: 'small' }, label,
    h('input', { type: 'number', min: 0, max: size - 1, step: 1, value: c[axis], 'aria-label': `${key} ${label}`, dataset: { axis },
      onChange: (e) => { if (e.target.value !== '') onChange?.(normalizeCell(state, key, { ...c, [axis]: Number(e.target.value) })); } })));
  el.append(h('div', { class: 'cell-coordinates' }, coordinates));
  el.append(h('div', { class: 'small muted' }, 'Zero-based indices. Values come from the current forward pass.'));
  return el;
}

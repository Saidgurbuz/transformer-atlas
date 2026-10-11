// Renders math catalogue entries: the full card (equation, symbols with live
// shapes, derivations, numbers plugged in) and the compact hover version.
import { h, tex, texHTML, num, fmt } from '../ui/index.js';
import { mathFor, runLive } from './math/index.js';
import { dimValues } from './describe.js';

function dimsText(dims, cfg, T) {
  if (!dims?.length) return '';
  const vals = dimValues(cfg, T);
  return dims.map((s) => `${s}=${s.split('·').reduce((p, x) => p * (vals[x] ?? 1), 1)}`).join(' × ');
}

/** Full card. `cell` ({r, band, x}) enables the live instantiation. */
export function mathCard(state, key, cell = null, { compact = false } = {}) {
  const e = mathFor(key);
  if (!e) return null;
  const T = state.ids.length;
  const live = cell ? runLive(state, key, cell) : null;
  const parts = [
    h('div', { class: 'math-title' }, e.title),
    tex(e.tex, { display: true }),
  ];
  if (e.symbols?.length) {
    parts.push(h('div', { class: 'math-symbols' }, e.symbols.map((s) => h('div', { class: 'math-sym' },
      tex(s.tex), h('span', null, s.text), h('span', { class: 'mono small muted' }, dimsText(s.dims, state.config, T))))));
  }
  if (live && live.tex) {
    const ok = live.expected === undefined || (live.result === live.expected) || Math.abs(live.result - live.expected) <= 1e-6 * Math.max(1, Math.abs(live.expected));
    parts.push(h('div', { class: 'math-live' },
      h('div', { class: 'eyebrow' }, 'With this cell’s numbers'),
      tex(live.tex, { display: true }),
      h('div', { class: ['small', ok ? 'math-ok' : 'math-bad'] },
        ok ? `✓ matches the traced value ${fmt(live.expected, 6)}` : `✗ traced value is ${fmt(live.expected, 6)}`,
        live.note ? ` · ${live.note}` : '')));
  }
  if (!compact && e.derivation?.length) {
    for (const d of e.derivation) {
      parts.push(h('details', { class: 'math-deriv' }, h('summary', null, d.title),
        d.steps.map((s) => tex(s, { display: true })), d.note && h('p', { class: 'small muted' }, d.note)));
    }
  }
  return h('div', { class: 'math-card' }, parts);
}

/** HTML for tooltips: compact formula and, if available, the live line. */
export function mathTipHTML(state, key, cell) {
  const e = mathFor(key);
  if (!e) return '';
  let html = `<div class="tip-math">${texHTML(e.compact || e.tex)}</div>`;
  const live = cell ? runLive(state, key, cell) : null;
  if (live?.tex) html += `<div class="tip-math">${texHTML(live.tex)}</div>`;
  return html;
}

export { mathFor, num };

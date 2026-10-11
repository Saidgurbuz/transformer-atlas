// HTML formula helpers (no TeX engine). Compose strings, then wrap with formula().
//   formula(`softmax(${frac('QK<sup>⊤</sup>', sqrt('d<sub>k</sub>'))})V`)
import { h } from './dom.js';

export const frac = (num, den) => `<span class="frac"><span>${num}</span><span>${den}</span></span>`;
export const sqrt = (x) => `√<span style="border-top:1px solid currentColor;padding-left:1px">${x}</span>`;
export const sub = (b, s) => `${b}<sub>${s}</sub>`;
export const sup = (b, s) => `${b}<sup>${s}</sup>`;
export const v = (name) => `<i>${name}</i>`;
// coloured symbol, e.g. col('q', 'Q')
export const col = (kind, x) => `<span style="color:var(--c-${kind});font-weight:600">${x}</span>`;

export function formula(html, { block = true } = {}) {
  return h(block ? 'div' : 'span', { class: block ? 'formula math' : 'math', html });
}

// Typeset maths with KaTeX (vendored). Component symbols are coloured with the
// same tokens as the rest of the app: \Q, \K, \V, \Wq ... (see MACROS).
//   tex('A = \\softmax(\\frac{\\Q\\K^\\top}{\\sqrt{d_h}})', { display: true })
import katex from '../vendor/katex/katex.mjs';
import { h } from './dom.js';

const kind = (k, body) => `\\htmlClass{k-${k}}{${body}}`;

export const MACROS = {
  // the visible-space glyph ␣ of subword labels, drawn with rules (KaTeX's fonts have no ␣);
  // for use inside \text{...} only, where it switches to math for the rules
  '\\vsp': '$\\kern0.03em\\rule[-0.14em]{0.06em}{0.32em}\\rule[-0.14em]{0.4em}{0.06em}\\rule[-0.14em]{0.06em}{0.32em}\\kern0.03em$',
  '\\R': '\\mathbb{R}',
  '\\softmax': '\\operatorname{softmax}',
  '\\LN': '\\operatorname{LN}',
  '\\RMSNorm': '\\operatorname{RMSNorm}',
  '\\GELU': '\\operatorname{GELU}',
  '\\SiLU': '\\operatorname{SiLU}',
  '\\ReLU': '\\operatorname{ReLU}',
  '\\concat': '\\operatorname{concat}',
  '\\CE': '\\operatorname{CE}',
  '\\Attn': '\\operatorname{Attn}',
  '\\MLP': '\\operatorname{MLP}',
  '\\Q': kind('q', 'Q'),
  '\\K': kind('k', 'K'),
  '\\V': kind('v', 'V'),
  '\\Wq': kind('q', 'W_Q'),
  '\\Wk': kind('k', 'W_K'),
  '\\Wv': kind('v', 'W_V'),
  '\\Wo': kind('o', 'W_O'),
  '\\WE': kind('embed', 'W_E'),
  '\\WU': kind('head', 'W_U'),
  '\\Wup': kind('mlp', 'W_{\\text{up}}'),
  '\\Wdown': kind('mlp', 'W_{\\text{down}}'),
  '\\Wgate': kind('mlp', 'W_{\\text{gate}}'),
  '\\Aw': kind('attn', 'A'),
  '\\Mk': kind('mask', 'M'),
  '\\Pe': kind('pos', 'P'),
  '\\xres': kind('resid', 'x'),
  '\\kc': '\\htmlClass{k-#1}{#2}',
};

export function texHTML(latex, { display = false } = {}) {
  return katex.renderToString(latex, {
    displayMode: display,
    throwOnError: false,
    strict: 'ignore',
    output: 'html',
    macros: { ...MACROS },
    trust: (c) => c.command === '\\htmlClass',
  });
}

/** Rendered element; display maths gets its own scrollable block. */
export function tex(latex, { display = false, className = '' } = {}) {
  return h(display ? 'div' : 'span', { class: ['tex', display && 'tex-display', className], html: texHTML(latex, { display }) });
}

/** Format a number for LaTeX: 4 significant-ish digits, proper minus, -inf. */
export function num(v, digits = 3) {
  if (v === -Infinity) return '-\\infty';
  if (v === Infinity) return '\\infty';
  if (!Number.isFinite(v)) return '\\text{NaN}';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) {
    const [m, e] = v.toExponential(2).split('e');
    return `${m}\\times10^{${Number(e)}}`;
  }
  return v.toFixed(digits);
}

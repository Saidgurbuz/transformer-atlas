// Helpers for math catalogue entries (pure functions, no DOM; usable in Node tests).
import { num } from '../../ui/tex.js';
export { num };

export const at2 = (t, r, c) => t.data[r * t.shape[1] + c];
export const at3 = (t, h, r, c) => t.data[(h * t.shape[1] + r) * t.shape[2] + c];
export const row2 = (t, r) => Array.from(t.data.subarray(r * t.shape[1], (r + 1) * t.shape[1]));
export const row3 = (t, h, r) => Array.from(t.data.subarray((h * t.shape[1] + r) * t.shape[2], (h * t.shape[1] + r + 1) * t.shape[2]));
export const dh = (cfg) => cfg.d_model / cfg.n_head;
export const kvHead = (cfg, h) => Math.floor(h / (cfg.n_head / cfg.n_kv_head));

/** Tensor that enters attention in layer i (ln1 for pre-LN, the raw stream for post-LN). */
export function attnInput(trace, cfg, i) {
  if (cfg.norm_placement === 'pre') return trace[`L${i}.ln1`];
  return i === 0 ? trace['emb.out'] : trace[`L${i - 1}.resid_out`];
}
/** Tensor that enters the MLP in layer i. */
export function mlpInput(trace, cfg, i) {
  return cfg.norm_placement === 'pre' ? trace[`L${i}.ln2`] : trace[`L${i}.resid_mid`];
}
/** Stream before layer i (input to the residual add). */
export function streamBefore(trace, i) { return i === 0 ? trace['emb.out'] : trace[`L${i - 1}.resid_out`]; }

/** Row c of an nn.Linear weight [out, in] as an array. */
export function wrow(model, name, r) {
  const W = model.params[name];
  return Array.from(W.data.subarray(r * W.shape[1], (r + 1) * W.shape[1]));
}

/**
 * LaTeX for a sum of products with real numbers, eliding the middle:
 *   dotTerms([a...], [b...]) -> "(0.12)(-0.40) + (0.03)(1.10) + \cdots + (…)(…)"
 */
export function dotTerms(a, b, { show = 3 } = {}) {
  const term = (i) => `(${num(a[i], 2)})(${num(b[i], 2)})`;
  if (a.length <= show + 1) return a.map((_, i) => term(i)).join(' + ');
  return `${Array.from({ length: show }, (_, i) => term(i)).join(' + ')} + \\cdots + ${term(a.length - 1)}`;
}

/** LaTeX list of values with elision: "e^{1.2} + e^{0.3} + \cdots" */
export function listTerms(vals, fmt, { show = 3, sep = ' + ' } = {}) {
  if (vals.length <= show + 1) return vals.map(fmt).join(sep);
  return `${vals.slice(0, show).map(fmt).join(sep)}${sep}\\cdots${sep}${fmt(vals[vals.length - 1])}`;
}

export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

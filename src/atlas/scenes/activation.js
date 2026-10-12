// Activation: the non-linearity between the two MLP projections, on this model's real numbers.
//   GELU (tanh form, as in model.py) and ReLU: the curve maps each pre-activation h to its output a = act(h).
//   SwiGLU: the gate g goes through SiLU to make the curve, and the up value h scales it: a = SiLU(g) · h.
// Four steps: the pre-activations sit on the axis; they rise onto the curve; the distribution of every value
// in this layer, with the share that comes out near zero; and other non-linearities on the same inputs
// (illustrations). Every point comes from ctx.state.trace (L{l}.mlp.up, .gate, .act). The scene recomputes
// each output with the engine's own functions, checks it against the traced value, and recounts the near-zero
// outputs from the pre-activations. Each step dwells MOTION.step on first view; SwiGLU step 1 has three beats,
// spaced (BEAT_AT_SWIGLU) so the last one settles inside that dwell.
// Colour is the diverging ramp, linear in the value, as in the MLP and neuron scenes. Near-zero outputs
// (|a| < 0.05) are a solid neutral, not hatched: hatch means masked everywhere else in the app.
// Reference scene for contributors: see docs/ATLAS_GUIDE.md and docs/STYLE_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, chip, appear, countTo, useStyle, MOTION, ZONES } from '../kit.js';
import { h, s, fmt, texHTML, num, reducedMotion, clamp, diverging, onThemeChange } from '../../ui/index.js';
import { geluScalar, siluScalar } from '../../engine/index.js';

const STEPS = ['Pre-activations', 'Through the curve', 'Distribution', 'Compare'];
const NEAR = 0.05;                                  // |output| below this counts as near zero (stated on screen)
const PX = 160, PY = 262, PW = 960, PH = 300;       // the plot frame (design units)
const LANE = PH + 42;                               // SwiGLU: the row of up values, under the plot (plot-relative)
const HY = PY + PH + 60, HH = 124, HEAD = 36;       // the histogram band under the plot (absolute); its bars start below a header row
const NB = 56;                                      // histogram bins across the x window
const SX = ZONES.side.x, SY = 262;                  // the side column
const BEAT_AT = [0, 700];                           // GELU and ReLU step 1: ms from the start of the step to each beat
const BEAT_AT_SWIGLU = [0, 380, 800];               // SwiGLU step 1: the axis, the gate onto the curve, then the product
const FIRST_DWELL = 900;                            // ms on step 0 before the first auto-advance (as softmax and head)
const NAME = { gelu: 'GELU · tanh', relu: 'ReLU', swiglu: 'SwiGLU' };
const ALTS = { gelu: ['relu', 'silu'], relu: ['gelu', 'silu'], swiglu: ['gelu', 'relu'] };   // compared with the model's own
const ALT_NAME = { gelu: 'GELU', relu: 'ReLU', silu: 'SiLU' };
const ALT_DASH = { gelu: '7 6', relu: '2 5', silu: '12 4 2 4' };   // each illustration's line style; its pill shows the same
const GELU_C = Math.sqrt(2 / Math.PI);
const fn = {
  gelu: (x) => geluScalar(x),
  relu: (x) => (x > 0 ? x : 0),
  silu: (x) => siluScalar(x),
};

let uid = 0;

useStyle('activation-scene', `
.ac-plot, .ac-hist { --ac-nz: color-mix(in srgb, var(--fg-3) 45%, var(--surface)); }
.ac-plot { position: absolute; left: 0; top: 0; }
.ac-svg { position: absolute; left: 0; top: 0; overflow: visible; }
.ac-svg .ac-grid { stroke: var(--line); stroke-width: 1; }
.ac-svg .ac-zero { stroke: var(--line-strong); stroke-width: 1.5; }
.ac-svg .ac-tick { font: 500 15px/1 var(--font-mono); fill: var(--fg-3); }
.ac-svg .ac-axis { font: italic 600 18px/1 var(--font-mono); fill: var(--fg-2); }
.ac-svg .ac-curvelab { font: 600 15px/1 var(--font-mono); fill: var(--c-mlp); }
.ac-svg .ac-curve { fill: none; stroke: var(--c-mlp); stroke-width: 3.5; stroke-linejoin: round; stroke-linecap: round; opacity: 0; transition: opacity .4s; }
.ac-svg .ac-alt { fill: none; stroke: var(--fg-2); stroke-width: 2.25; stroke-linecap: round; opacity: 0; transition: opacity .5s; }
.ac-svg .ac-alt.on { opacity: .9; }
.ac-svg .ac-band { opacity: 0; transition: opacity .6s; }
.ac-svg .ac-band.on { opacity: 1; }
.ac-svg .ac-band line { stroke: var(--fg-3); stroke-width: 1.25; stroke-dasharray: 4 4; }
.ac-dots { position: absolute; left: 0; top: 0; width: ${PW}px; height: ${PH}px; pointer-events: none; }
.ac-dot { position: absolute; left: 0; top: 0; width: 10px; height: 10px; margin: -5px 0 0 -5px; border-radius: 50%; pointer-events: auto; cursor: pointer;
  box-shadow: 0 0 0 1.5px var(--surface), 0 0 0 2.5px color-mix(in srgb, var(--fg-3) 26%, transparent);
  transition: left .6s cubic-bezier(.65,0,.35,1), top .6s cubic-bezier(.65,0,.35,1), background-color .4s, box-shadow .3s, width .3s, height .3s, margin .3s, border-radius .3s, transform .3s; }
.ac-dot.edge { width: 8px; height: 8px; margin: -4px 0 0 -4px; border-radius: 2px; transform: rotate(45deg); }
.ac-dot.sel { box-shadow: 0 0 0 1.5px var(--surface), 0 0 0 4px var(--accent); }
.ac-dot:focus-visible { outline: 2px solid var(--accent); outline-offset: 4px; }
.ac-tog-row { position: absolute; left: 12px; top: 12px; display: flex; gap: 8px; opacity: 0; pointer-events: none; transition: opacity .4s; }
.ac-tog-row.on { opacity: 1; pointer-events: auto; }
.ac-tog { display: inline-flex; align-items: center; gap: 8px; font: 600 15px/1 var(--font-mono); padding: 7px 12px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--fg-3); cursor: pointer; transition: border-color .2s, color .2s; }
.ac-tog[aria-pressed="true"] { color: var(--fg); border-color: var(--fg-3); }
.ac-tog:hover { border-color: var(--line-strong); color: var(--fg); }
.ac-sw { flex: none; }
.ac-sw path { stroke: var(--fg-2); stroke-width: 2.25; stroke-linecap: round; fill: none; }
.ac-tog[aria-pressed="false"] .ac-sw path { stroke: var(--fg-3); }
.ac-hist { position: absolute; left: 0; top: 0; }
.ac-hhead { position: absolute; left: 0; top: 0; height: 24px; display: flex; align-items: center; justify-content: space-between; gap: 24px; font: 500 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.ac-hleft, .ac-hright { display: flex; align-items: center; gap: 10px; min-width: 0; }
.ac-hpick { font: 600 15px/1 var(--font-mono); padding: 4px 9px; border-radius: 999px; border: 1px solid transparent; background: none; color: var(--fg-3); cursor: pointer; }
.ac-hpick[aria-pressed="true"] { color: var(--fg); border-color: var(--line-strong); background: var(--surface); }
.ac-hpick:hover { color: var(--fg); }
.ac-swnz { display: inline-block; width: 12px; height: 12px; border-radius: 3px; background: var(--ac-nz); box-shadow: inset 0 0 0 1px var(--line-strong); }
.ac-hplot { position: absolute; left: 0; top: 0; width: ${PW}px; height: ${HH}px; transition: opacity .4s; }
.ac-hist.dim .ac-hplot { opacity: .35; }
.ac-hsvg { position: absolute; left: 0; top: 0; overflow: visible; opacity: .45; transition: opacity .5s; }
.ac-hist.on .ac-hsvg { opacity: 1; }
.ac-hsvg .ac-hbase { stroke: var(--line-strong); stroke-width: 1; }
.ac-hsvg .ac-grid { stroke: var(--line); stroke-width: 1; }
.ac-hsvg .ac-tick { font: 500 15px/1 var(--font-mono); fill: var(--fg-3); }
.ac-bar { position: absolute; bottom: 0; transform: scaleY(0); transform-origin: 50% 100%; transition: transform .6s cubic-bezier(.65,0,.35,1), height .6s cubic-bezier(.65,0,.35,1); }
.ac-hist.on .ac-bar { transform: scaleY(1); }
.ac-bar > i { position: absolute; left: 0; right: 0; display: block; border-radius: 2px; transition: height .6s cubic-bezier(.65,0,.35,1), bottom .6s cubic-bezier(.65,0,.35,1), background-color .4s; }
.ac-bar > i.ac-nz { bottom: 0; background: var(--ac-nz); }
.ac-side { position: absolute; width: ${ZONES.side.w}px; display: grid; gap: 12px; align-content: start; }
.ac-sidelab { text-align: left; margin: 0; color: var(--c-mlp); }
.ac-eqmain { font-size: 22px; color: var(--fg); overflow-x: auto; }
.ac-eqlive { font-size: 16px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); box-shadow: 0 6px 16px rgba(0, 0, 0, .06); overflow-x: auto; display: grid; gap: 6px; }
.ac-eqlive:empty { display: none; }
.ac-rows { display: grid; gap: 6px; color: var(--fg); }
.ac-blk { display: grid; gap: 2px; min-width: 0; }
.ac-tag { font: 500 15px/1.2 var(--font-mono); color: var(--fg-3); }
.ac-tag.own { color: var(--c-mlp); }
.ac-row { min-width: 0; overflow-x: auto; white-space: nowrap; }
.ac-big { font: 650 30px/1 var(--font-display); color: var(--fg); }
.ac-note { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); }
.ac-checkrow { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.ac-checkrow[hidden] { display: none; }
.ac-checkrow small { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); }
.ac-stats { display: grid; font: 500 15px/1.3 var(--font-mono); color: var(--fg-2); }
.ac-stat { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 12px; padding: 6px 2px; border-bottom: 1px solid var(--line); }
.ac-stat > span { min-width: 0; }
.ac-stat b { color: var(--fg); font-weight: 600; text-align: right; white-space: nowrap; }
.ac-stat.off { opacity: .45; }
.ac-still, .ac-still * { transition: none !important; }
@media (prefers-reduced-motion: reduce) { .ac-dot, .ac-bar, .ac-bar > i, .ac-hplot, .ac-hsvg, .ac-tog-row, .ac-svg .ac-curve, .ac-svg .ac-alt, .ac-svg .ac-band { transition: none !important; } }
`);

// ------------------------------------------------------------------ pure helpers

const quantile995 = (sorted) => sorted[Math.min(sorted.length - 1, Math.floor(0.995 * (sorted.length - 1)))];
// a window edge: half units up to 2, then whole units
const niceUp = (x) => (x <= 2 ? Math.ceil(Math.max(0, x) * 2) / 2 : Math.ceil(x));
const tickStep = (v) => (v > 5 ? 2 : v > 2 ? 1 : 0.5);
const ticks = (a, b, step) => {
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
};
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
// colour: the diverging ramp, linear in the value, as in the MLP and neuron scenes (no floor on the saturation)
const tone = (v, scale) => diverging(clamp(v / scale, -1, 1));
// Plain text uses one minus glyph everywhere (fmt() and num() do not: num() returns LaTeX for tiny values).
const minus = (v) => String(v).replace(/-/g, '−');
// A plain-text number as HTML: one minus glyph, and a tiny or huge value as a power of ten.
function numText(v, d = 2) {
  if (!Number.isFinite(v)) return fmt(v, d);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 10 ** -d && a < 1e5) return minus(v.toFixed(d));
  const [m, e] = v.toExponential(d).split('e');
  return `${minus(m)} × 10<sup>${minus(Number(e))}</sup>`;
}
const numEl = (v, d) => h('span', { html: numText(v, d) });
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const C = (v) => v.toLocaleString('en-US');
const aligned = (lines) => `\\begin{aligned}${lines.map((l) => `&${l}`).join(' \\\\ ')}\\end{aligned}`;
const tx = (tex) => h('div', { class: 'ac-tex', html: texHTML(tex, { display: true }) });
const note = (...parts) => h('div', { class: 'ac-note' }, ...parts);
const statRow = (label, value, off = false) => h('div', { class: ['ac-stat', off && 'off'] }, h('span', { html: label }), h('b', { html: value }));
// the x ranges where |f(x)| < NEAR: the curve is near zero there (GELU and ReLU only; the gate band is not an output band)
function bandRanges(c) {
  const f = fn[c.kind], out = [], steps = 800;
  let start = null;
  for (let k = 0; k <= steps; k++) {
    const x = -c.W + (2 * c.W * k) / steps, inside = Math.abs(f(x)) < NEAR;
    if (inside && start === null) start = x;
    if (!inside && start !== null) { out.push([start, x]); start = null; }
  }
  if (start !== null) out.push([start, c.W]);
  return out;
}
const xPx = (c, x) => ((x + c.W) / (2 * c.W)) * PW;
const yPx = (c, y) => ((c.Yc - y) / (c.Yc - c.yLo)) * PH;
const curvePts = (c, f, n = 240) => {
  const pts = [];
  for (let k = 0; k <= n; k++) { const x = -c.W + (2 * c.W * k) / n; pts.push([xPx(c, x), yPx(c, f(x))]); }
  return pts;
};
const pathD = (pts) => pts.map((p, k) => `${k ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('');
const pathLen = (pts) => pts.reduce((acc, p, k) => (k ? acc + Math.hypot(p[0] - pts[k - 1][0], p[1] - pts[k - 1][1]) : 0), 0);

// the scene's own recomputation of one unit's activation, written out in short lines for the live equation
function workedTex(c, r) {
  const u = c.U[r], x = c.X[r], v = c.calc(r);
  if (c.kind === 'relu') return { tex: aligned([`h = ${num(u)}`, `a = \\max(0,\\,${num(u)}) = ${num(v, 4)}`]) };
  if (c.kind === 'gelu') {
    const z = GELU_C * (u + 0.044715 * u * u * u);
    return { tex: aligned([`h = ${num(u)}`, `z = ${num(GELU_C, 3)}\\,(h + 0.044715\\,h^{3}) = ${num(z, 2)}`, `a = \\tfrac12\\,h\\,(1+\\tanh z) = ${num(v, 4)}`]) };
  }
  const sg = fn.silu(x);
  return { tex: aligned([`g = ${num(x)},\\quad h = ${num(u)}`, `\\SiLU(g) = \\frac{g}{1+e^{-g}} = ${num(sg, 4)}`, `a = ${num(sg, 4)}\\cdot h = ${num(v, 4)}`]) };
}
// step 4: one row per curve. The model's own function comes first; every illustration is tagged as one.
const FN_TEX = {
  gelu: (x) => `\\GELU(${x}) = \\tfrac12\\,${x}\\,(1+\\tanh z)`,
  relu: (x) => `\\ReLU(${x}) = \\max(0,\\,${x})`,
  silu: (x) => `\\SiLU(${x}) = ${x}\\,\\sigma(${x})`,
};
const zTex = (x) => `z = \\sqrt{2/\\pi}\\,(${x}+0.044715\\,${x}^{3})`;
function compareRows(c) {
  const x = c.sym ? 'g' : 'h';
  const names = c.sym ? ['silu', 'gelu', 'relu'] : [c.kind, ...ALTS[c.kind]];
  const blk = (tag, tex, cls) => h('div', { class: 'ac-blk' }, h('div', { class: ['ac-tag', cls] }, tag), h('div', { class: 'ac-row', html: texHTML(tex) }));
  const blocks = c.sym ? [blk('per unit', 'a_{t,f} = \\text{gate}(g_{t,f})\\,h_{t,f}')] : [];
  names.forEach((n, k) => {
    blocks.push(blk(k === 0 ? 'this model' : 'illustration', FN_TEX[n](x), k === 0 && 'own'));
    if (n === 'gelu') blocks.push(blk('where', zTex(x)));
  });
  return h('div', { class: 'ac-rows' }, ...blocks);
}
// rank of each unit by its x value, so the dots rise in a wave from left to right
function rankOf(c) {
  const order = Array.from({ length: c.F }, (_, f) => f).sort((a, b) => c.X[c.row + a] - c.X[c.row + b]);
  const rank = new Array(c.F);
  order.forEach((f, r) => { rank[f] = r; });
  return rank;
}

// The trace for this layer, read once per forward pass or selection. Null when the layer has no MLP trace.
function analyse(ctx, L) {
  const S = ctx.state, cfg = S.config, kind = cfg.activation, sym = kind === 'swiglu';
  const up = S.trace[`L${L}.mlp.up`], out = S.trace[`L${L}.mlp.act`];
  const gate = sym ? S.trace[`L${L}.mlp.gate`] : null;
  if (!up || !out || (sym && !gate)) return null;
  const T = up.shape[0], F = up.shape[1], N = T * F;
  const U = up.data, A = out.data, X = gate ? gate.data : U;      // X: the value on the x-axis (the gate for SwiGLU)
  // the x window: the 99.5th percentile of |x|, rounded up, at least one unit and at most eight
  const W = clamp(niceUp(quantile995(Float64Array.from(X, Math.abs).sort())), 1, 8);
  const calc = (i) => (sym ? fn.silu(X[i]) * U[i] : fn[kind](U[i]));     // the scene's own output for value i
  // the histogram of one pre-activation vector over the window [-W, W], with the near-zero outputs counted per bin
  const histOf = (vals) => {
    const bins = Array.from({ length: NB }, () => ({ n: 0, z: 0 })), bw = (2 * W) / NB;
    let inside = 0, outside = 0;
    for (let i = 0; i < N; i++) {
      const v = vals[i];
      if (v < -W || v > W) { outside++; continue; }
      inside++;
      const b = Math.min(NB - 1, Math.floor((v + W) / bw));
      bins[b].n++;
      if (Math.abs(A[i]) < NEAR) bins[b].z++;
    }
    return { bins, inside, outside };
  };
  const hist = { x: histOf(X) };
  if (sym) hist.u = histOf(U);
  let zeros = 0, recount = 0, maxAll = 0, xLo = Infinity, xHi = -Infinity, aLo = Infinity, aHi = -Infinity;
  for (let i = 0; i < N; i++) {
    const v = calc(i);
    if (Math.abs(A[i]) < NEAR) zeros++;
    if (Math.abs(v) < NEAR) recount++;                  // near-zero outputs, recomputed from the pre-activations
    maxAll = Math.max(maxAll, Math.abs(v - A[i]));
    xLo = Math.min(xLo, X[i]); xHi = Math.max(xHi, X[i]);
    aLo = Math.min(aLo, A[i]); aHi = Math.max(aHi, A[i]);
  }
  // the y window. SwiGLU outputs are not bounded by its curve, so its window holds the layer's real extent; GELU and
  // ReLU keep the 99.5th percentile of |a|, since their curve bounds the outputs. Never below W, never above 8.
  const Yc = clamp(niceUp(sym ? Math.max(Math.abs(aLo), Math.abs(aHi)) : quantile995(Float64Array.from(A, Math.abs).sort())), W, 8);
  const yLo = sym ? -Yc : -0.5;
  let outA = 0;
  for (let i = 0; i < N; i++) if (A[i] < yLo || A[i] > Yc) outA++;
  const q = clamp(S.sel.token ?? 0, 0, T - 1), row = q * F;
  let nearTok = 0, negTok = 0, argmin = 0, tXLo = Infinity, tXHi = -Infinity, tAHi = -Infinity, tOutX = 0, tOutU = 0;
  for (let f = 0; f < F; f++) {
    const i = row + f;
    if (Math.abs(A[i]) < NEAR) nearTok++;
    if (A[i] < 0) negTok++;
    if (X[i] < X[row + argmin]) argmin = f;
    tXLo = Math.min(tXLo, X[i]); tXHi = Math.max(tXHi, X[i]); tAHi = Math.max(tAHi, A[i]);
    if (X[i] < -W || X[i] > W) tOutX++;
    if (sym && (U[i] < -W || U[i] > W)) tOutU++;
  }
  const pre = cfg.norm_placement !== 'post', rms = cfg.norm === 'rmsnorm', bias = !!cfg.bias;
  return {
    kind, sym, T, F, N, q, row, W, Yc, yLo, bw: (2 * W) / NB, U, A, X, hist,
    // the MLP's input, named as mlp.js names it: x̂ = LN2(x^mid) for pre-LN (defined on its own line), x^mid for post-LN
    xs: pre ? '\\hat{x}_t' : 'x^{\\text{mid}}_t',
    xDef: pre ? (rms ? '\\hat{x}_t = \\RMSNorm_2(x^{\\text{mid}}_t)' : '\\hat{x}_t = \\LN_2(x^{\\text{mid}}_t)') : null,
    bGate: bias ? ' + b_{\\text{gate}}' : '', bUp: bias ? ' + b_{\\text{up}}' : '',   // the bias terms exist only with cfg.bias
    zeros, recount, maxAll, xLo, xHi, aLo, aHi, outA,
    nearTok, negTok, argmin, tXLo, tXHi, tAHi, tOutX, tOutU, calc, u: 0,
  };
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const L = Number(ctx.params.l);
    const clipId = `ac-clip-${++uid}`;

    // -------------------------------------------------------------- shared kit parts
    const header = sceneHeader(st, { chips: [[`Layer ${L}`, 'resid']], title: 'Activation' });
    const actChip = chip('', 'mlp');
    header.el.querySelector('.k-eyebrow')?.append(actChip);
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'mlp' });
    const cap = sceneCaption(st);

    // -------------------------------------------------------------- DOM: the plot, the histogram, the side column
    const plot = h('div', { class: 'ac-plot', style: { width: `${PW}px`, height: `${PH}px` } });
    const svg = s('svg', { class: 'ac-svg', width: PW, height: PH, viewBox: `0 0 ${PW} ${PH}` });
    const dotLayer = h('div', { class: 'ac-dots', 'data-own-keys': '' });   // the dots take arrow keys themselves
    const togRow = h('div', { class: 'ac-tog-row' });
    plot.append(svg, dotLayer, togRow);
    st.add(plot, { x: PX, y: PY });

    // the histogram: a header row (what is counted, the near-zero key) over the bars, which sit in histPlot
    const histBox = h('div', { class: 'ac-hist', style: { width: `${PW}px`, height: `${HH}px` } });
    const hsvg = s('svg', { class: 'ac-hsvg', width: PW, height: HH, viewBox: `0 0 ${PW} ${HH}` });
    const histPlot = h('div', { class: 'ac-hplot' }, hsvg);
    const hleft = h('div', { class: 'ac-hleft' });
    const hright = h('div', { class: 'ac-hright' }, h('span', { class: 'ac-swnz', 'aria-hidden': 'true' }), h('span', null, '|a| < 0.05, near zero'));
    const hhead = h('div', { class: 'ac-hhead', style: { width: `${PW}px` } }, hleft, hright);
    histBox.append(histPlot, hhead);
    st.add(histBox, { x: PX, y: HY });

    const sideLab = h('div', { class: 'k-label ac-sidelab' });
    const eqMain = h('div', { class: 'ac-eqmain' });
    const eqLive = h('div', { class: 'ac-eqlive' });
    const checkRow = h('div', { class: 'ac-checkrow', hidden: true });
    const stats = h('div', { class: 'ac-stats' });
    const side = h('div', { class: 'ac-side' }, sideLab, eqMain, eqLive, checkRow, stats);
    st.add(side, { x: SX, y: SY });

    // -------------------------------------------------------------- state
    let step = 0, alive = true, cur = null, unit = null, mode = 'axis', countKey = '';
    let hvar = 'x';                                          // the histogram counts 'x' (the x-axis value) or 'u' (SwiGLU up value h)
    let curveEl = null, curveLen = 0, curveShown = false, bandEl = null, altEls = [];
    let dots = [], laneDots = [], bars = [];
    const toggles = {};
    let gen = 0, timers = new Set(), autoT = null, driving = false, capText = null;

    const later = (ms, fn2) => {
      const g = gen;
      const t = setTimeout(() => { timers.delete(t); if (alive && g === gen) fn2(); }, ms);
      timers.add(t);
    };
    const clearBeats = () => { gen++; timers.forEach(clearTimeout); timers.clear(); };
    const cancelAuto = () => { clearTimeout(autoT); autoT = null; };
    const still = (animate, fn2) => {
      if (animate) { fn2(); return; }
      st.root.classList.add('ac-still');
      fn2();
      void st.root.offsetWidth;
      st.root.classList.remove('ac-still');
    };
    const beatsOf = (i) => (i === 1 ? (cur?.sym ? 3 : 2) : 1);
    const beatAt = (c) => (c?.sym ? BEAT_AT_SWIGLU : BEAT_AT);
    // A newer caption cancels a fade still running, so only the latest text is ever written (lm-head does the same).
    const capSet = (html) => {
      if (html === capText) return;
      capText = html;
      if (reducedMotion()) { cap.el.innerHTML = html; return; }
      cap.el.getAnimations().forEach((a) => a.cancel());
      cap.el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 140 }).finished.then(() => {
        if (html !== capText) return;
        cap.el.innerHTML = html;
        cap.el.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 260, easing: MOTION.easeOut });
      }).catch(() => {});
    };

    // -------------------------------------------------------------- the static plot (rebuilt when the trace changes)
    function buildPlot(c) {
      const fx = (x) => xPx(c, x), fy = (y) => yPx(c, y);
      const xt = ticks(-c.W, c.W, tickStep(c.W));
      const yt = c.sym ? ticks(-c.Yc, c.Yc, tickStep(c.Yc)) : ticks(0, c.Yc, tickStep(c.Yc));
      const pts = curvePts(c, c.sym ? fn.silu : fn[c.kind]);
      const inner = [];
      xt.forEach((v) => inner.push(s('line', { class: 'ac-grid', x1: fx(v), x2: fx(v), y1: 0, y2: PH })));
      yt.forEach((v) => inner.push(s('line', { class: 'ac-grid', x1: 0, x2: PW, y1: fy(v), y2: fy(v) })));
      inner.push(s('line', { class: 'ac-zero', x1: 0, x2: PW, y1: fy(0), y2: fy(0) }));
      bandEl = null;
      if (!c.sym) {
        // the near-zero band: dashed edges at ±NEAR over the inputs whose output is near zero
        bandEl = s('g', { class: 'ac-band' });
        for (const [a, b] of bandRanges(c)) {
          const x1 = fx(a), x2 = Math.max(x1 + 2, fx(b));
          bandEl.append(s('line', { x1, x2, y1: fy(NEAR), y2: fy(NEAR) }), s('line', { x1, x2, y1: fy(-NEAR), y2: fy(-NEAR) }));
        }
        inner.push(bandEl);
      }
      altEls = ALTS[c.kind].map((name) => {
        const el = s('path', { class: 'ac-alt', d: pathD(curvePts(c, fn[name])), 'data-alt': name });
        el.style.strokeDasharray = ALT_DASH[name];
        inner.push(el);
        return { name, el };
      });
      curveEl = s('path', { class: 'ac-curve', d: pathD(pts), 'data-trace': `L${L}.mlp.act` });
      curveEl.style.opacity = curveShown ? '1' : '0';
      curveLen = pathLen(pts);
      inner.push(curveEl);
      const defs = s('defs', null, s('clipPath', { id: clipId }, s('rect', { x: 0, y: 0, width: PW, height: PH })));
      const labels = [
        ...yt.map((v) => s('text', { class: 'ac-tick', x: -10, y: fy(v) + 5, 'text-anchor': 'end' }, minus(v))),
        ...xt.map((v) => s('text', { class: 'ac-tick', x: fx(v), y: PH + 22, 'text-anchor': 'middle' }, minus(v))),
        s('text', { class: 'ac-axis', x: -12, y: -8, 'text-anchor': 'end' }, 'a'),
        s('text', { class: 'ac-axis', x: PW + 12, y: fy(0) + 6, 'text-anchor': 'start' }, c.sym ? 'g' : 'h'),
        s('text', { class: 'ac-curvelab', x: PW - 10, y: fy(0) - 10, 'text-anchor': 'end' }, c.sym ? 'SiLU(g) · this model' : `${NAME[c.kind]} · this model`),
      ];
      if (c.sym) {
        // the up row: its own baseline and name, on the same x scale as the plot
        labels.push(
          s('line', { class: 'ac-grid', x1: 0, x2: PW, y1: LANE, y2: LANE }),
          s('text', { class: 'ac-axis', x: -12, y: LANE + 6, 'text-anchor': 'end' }, 'up h'),
          s('text', { class: 'ac-tick', x: PW - 6, y: LANE - 8, 'text-anchor': 'end' }, 'same scale as g'));
      }
      svg.replaceChildren(defs, s('g', { 'clip-path': `url(#${clipId})` }, ...inner), ...labels);
    }

    // -------------------------------------------------------------- toggles for the illustrations (step 4)
    function buildToggles(c) {
      const names = ALTS[c.kind];
      names.forEach((n) => { if (toggles[n] === undefined) toggles[n] = true; });
      togRow.replaceChildren(...names.map((n) => {
        // the pill shows the line style of its curve
        const sw = s('svg', { class: 'ac-sw', width: 28, height: 8, viewBox: '0 0 28 8', 'aria-hidden': 'true' }, s('path', { d: 'M1 4 H27', style: { strokeDasharray: ALT_DASH[n] } }));
        const b = h('button', { type: 'button', class: 'ac-tog', dataset: { alt: n }, 'aria-pressed': String(toggles[n] !== false) }, sw, `${ALT_NAME[n]} · illustration`);
        b.addEventListener('click', () => {
          toggles[n] = toggles[n] === false;
          b.setAttribute('aria-pressed', String(toggles[n]));
          syncAlts(step);
          texts(step);
        });
        return b;
      }));
    }
    function syncAlts(i) {
      altEls.forEach(({ name, el }) => el.classList.toggle('on', i === 3 && toggles[name] !== false));
    }

    // -------------------------------------------------------------- dots: one per unit of this token (and one per unit in the up row for SwiGLU)
    const unitTitle = (c, f) => {
      const i = c.row + f;
      const inputs = c.sym ? `g ${fmt(c.X[i], 2)} · h ${fmt(c.U[i], 2)}` : `h ${fmt(c.U[i], 2)}`;
      return `unit ${f} · ${inputs} · a ${fmt(c.A[i], 2)}`;
    };
    function buildDots(c) {
      dots.forEach((d) => d.remove());
      laneDots.forEach((d) => d.remove());
      dots = [];
      laneDots = [];
      for (let f = 0; f < c.F; f++) {
        const d = h('div', { class: 'ac-dot', role: 'button', tabindex: '-1' });
        d.addEventListener('click', () => selectUnit(f));
        d.addEventListener('keydown', (e) => onDotKey(e, f));
        dotLayer.append(d);
        dots.push(d);
        if (c.sym) {
          const ld = h('div', { class: 'ac-dot', 'aria-hidden': 'true' });
          ld.addEventListener('click', () => selectUnit(f));
          dotLayer.append(ld);
          laneDots.push(ld);
        }
      }
    }
    function selectUnit(f, { focus = false } = {}) {
      if (!cur) return;
      unit = f;
      cur.u = f;
      paintDots(cur, mode, false);
      texts(step);
      if (focus) dots[f]?.focus({ preventScroll: true });
    }
    // arrow keys step along the units (the dots are a roving tab stop: only the picked unit is in the tab order)
    function onDotKey(e, f) {
      if (!cur) return;
      const last = cur.F - 1;
      const to = { ArrowRight: f + 1, ArrowUp: f + 1, ArrowLeft: f - 1, ArrowDown: f - 1, Home: 0, End: last }[e.key];
      if (to === undefined) return;
      e.preventDefault();
      selectUnit(clamp(to, 0, last), { focus: true });
    }
    // mode: 'axis' (on the x-axis at the pre-activation), 'gate' (on the SiLU curve, SwiGLU), 'out' (on the curve, at the output)
    function paintDots(c, m, animate) {
      mode = m;
      const rank = rankOf(c);
      const trace = m === 'axis' ? (c.sym ? 'gate' : 'up') : m === 'gate' ? 'gate' : 'act';
      for (let f = 0; f < c.F; f++) {
        const i = c.row + f, d = dots[f], x = c.X[i], u = c.U[i], a = c.A[i];
        const y = m === 'axis' ? 0 : m === 'gate' ? fn.silu(x) : a;
        const xc = clamp(x, -c.W, c.W), yc = clamp(y, c.yLo, c.Yc);
        const nz = m !== 'axis' && Math.abs(y) < NEAR;
        const delay = animate ? `${(rank[f] / c.F) * 260}ms` : '0ms';
        d.style.left = `${xPx(c, xc)}px`;
        d.style.top = `${yPx(c, yc)}px`;
        d.style.transitionDelay = delay;
        d.style.background = nz ? 'var(--ac-nz)' : m === 'axis' ? tone(x, c.W) : tone(y, c.Yc);
        d.classList.toggle('nz', nz);
        d.classList.toggle('edge', x < -c.W || x > c.W || y < c.yLo || y > c.Yc);   // clipped to the plot edge: an off-scale unit
        d.classList.toggle('sel', f === c.u);
        d.tabIndex = f === c.u ? 0 : -1;
        d.title = unitTitle(c, f);
        d.setAttribute('aria-label', d.title);
        d.dataset.trace = `L${L}.mlp.${trace}`;
        d.dataset.cell = `${c.q},${f},0`;
        if (c.sym) {
          const ld = laneDots[f];
          ld.style.left = `${xPx(c, clamp(u, -c.W, c.W))}px`;
          ld.style.top = `${LANE}px`;
          ld.style.transitionDelay = delay;
          ld.style.background = tone(u, c.W);
          ld.classList.toggle('edge', u < -c.W || u > c.W);
          ld.classList.toggle('sel', f === c.u);
          ld.dataset.trace = `L${L}.mlp.up`;
          ld.dataset.cell = `${c.q},${f},0`;
        }
      }
    }

    // -------------------------------------------------------------- the curve draws in once, then stays
    function setCurve(show, animate) {
      if (!curveEl) return;
      if (show && !curveShown && animate && curveLen) {
        const dash = curveLen * 1.05;
        curveEl.style.transition = 'none';
        curveEl.style.strokeDasharray = `${dash} ${curveLen * 2}`;
        curveEl.style.strokeDashoffset = `${dash}`;
        curveEl.style.opacity = '1';
        void curveEl.getBoundingClientRect();
        curveEl.style.transition = 'stroke-dashoffset 700ms cubic-bezier(.65,0,.35,1), opacity .4s';
        curveEl.style.strokeDashoffset = '0';
      } else {
        curveEl.style.transition = '';
        curveEl.style.strokeDasharray = 'none';
        curveEl.style.strokeDashoffset = '0';
        curveEl.style.opacity = show ? '1' : '0';
      }
      curveShown = show;
    }

    // -------------------------------------------------------------- the histogram of every value in this layer
    // The bars are built once per forward pass and repainted in place when the SwiGLU switch changes.
    function buildHist(c) {
      bars.forEach(({ bar }) => bar.remove());
      const cw = PW / NB;
      bars = Array.from({ length: NB }, (_, j) => {
        const nz = h('i', { class: 'ac-nz' }), rest = h('i', { class: 'ac-rest' });
        const bar = h('div', { class: 'ac-bar', style: { left: `${j * cw + 1.5}px`, width: `${cw - 3}px` } }, nz, rest);
        histPlot.append(bar);
        return { bar, nz, rest };
      });
      buildSwitch(c);
      paintHist(c);
    }
    function paintHist(c) {
      const key = c.sym ? hvar : 'x', hg = c.hist[key], tk = c.sym && key === 'x' ? 'gate' : 'up';
      const top = Math.max(...hg.bins.map((b) => b.n)) / c.N || 1e-9;   // share of all values in the tallest bin
      const k = (HH - HEAD - 6) / top;                                  // px per unit share; the tallest bin stops below the header row
      bars.forEach(({ bar, nz, rest }, j) => {
        const b = hg.bins[j];
        const tot = (b.n / c.N) * k, nzp = (b.z / c.N) * k;
        bar.style.height = `${tot}px`;
        bar.dataset.trace = `L${L}.mlp.${tk}`;
        nz.style.height = `${nzp}px`;
        rest.style.bottom = `${nzp}px`;
        rest.style.height = `${Math.max(0, tot - nzp)}px`;
        rest.style.background = tone(-c.W + (j + 0.5) * c.bw, c.W);
      });
      const fx = (x) => xPx(c, x);
      const xt = ticks(-c.W, c.W, tickStep(c.W));
      hsvg.replaceChildren(
        s('line', { class: 'ac-hbase', x1: 0, x2: PW, y1: HH, y2: HH }),
        ...xt.map((v) => s('line', { class: 'ac-grid', x1: fx(v), x2: fx(v), y1: HEAD, y2: HH })),
        ...xt.map((v) => s('text', { class: 'ac-tick', x: fx(v), y: HH + 19, 'text-anchor': 'middle' }, minus(v))),
        s('text', { class: 'ac-tick', x: -10, y: HEAD + 11, 'text-anchor': 'end' }, pct(top)),
      );
    }
    // the header row: what is counted, and for SwiGLU the switch between the gate g and the up value h
    function buildSwitch(c) {
      const label = h('span', null, c.sym ? '% of all values per bin · histogram of' : '% of all values per bin · pre-activation h');
      const picks = c.sym ? [['x', 'g', 'gate g: the histogram counts the gate values'], ['u', 'h', 'up h: the histogram counts the up values']] : [];
      hleft.replaceChildren(label, ...picks.flatMap(([key, text, title], k) => {
        const b = h('button', { type: 'button', class: 'ac-hpick', dataset: { hkey: key }, title }, text);
        b.addEventListener('click', () => {
          if (!cur || hvar === key) return;
          hvar = key;
          paintHist(cur);
          syncSwitch();
          texts(step);
        });
        return k ? [h('span', { class: 'ac-hsep' }, '|'), b] : [b];
      }));
      syncSwitch();
    }
    function syncSwitch() {
      hleft.querySelectorAll('[data-hkey]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.hkey === hvar)));
    }

    // -------------------------------------------------------------- the visual state of one step and beat
    function apply(i, b, animate) {
      const c = cur;
      if (!c) return;
      const m = i === 0 ? 'axis' : i === 1 ? (b === 0 ? 'axis' : c.sym && b === 1 ? 'gate' : 'out') : 'out';
      still(animate, () => {
        paintDots(c, m, animate);
        setCurve(i >= 1, animate);
        bandEl?.classList.toggle('on', i >= 2);
        histBox.classList.toggle('on', i >= 2);
        histBox.classList.toggle('dim', i === 3);      // the compare step: the histogram steps back, the comparison leads
        bars.forEach(({ bar }, j) => { bar.style.transitionDelay = i >= 2 ? `${j * 7}ms` : '0ms'; });
        togRow.classList.toggle('on', i === 3);
        syncAlts(i);
      });
      texts(i);
    }

    function goStep(i) {
      if (!cur) { step = clamp(i, 0, STEPS.length - 1); return; }
      clearBeats();
      const from = step;
      step = clamp(i, 0, STEPS.length - 1);
      const anim = !reducedMotion();
      if (step > from && anim) {
        apply(step, 0, true);
        for (let b = 1; b < beatsOf(step); b++) later(beatAt(cur)[b], () => apply(step, b, true));
      } else {
        apply(step, beatsOf(step) - 1, anim);
      }
    }
    const onStep = (i) => { if (!driving) cancelAuto(); goStep(i); };

    // -------------------------------------------------------------- text: header, equations, checks, stats, caption
    function texts(i) {
      const c = cur;
      if (!c) return;
      const q = c.q, f = c.u, r = c.row + f, lab = ctx.label(q);
      const h0 = c.U[r], x0 = c.X[r], a0 = c.A[r], calc = c.calc(r);
      header.setSub(c.sym
        ? `Token “<b>${esc(lab)}</b>” at position ${q}: ${c.F} gate values set the curve, and ${c.F} up values scale it.`
        : `Token “<b>${esc(lab)}</b>” at position ${q}: its ${c.F} pre-activations go through ${c.kind === 'relu' ? 'ReLU, which zeroes every negative value' : 'GELU, in the tanh form model.py uses'}.`);
      actChip.textContent = NAME[c.kind];
      sideLab.textContent = STEPS[i];
      eqLive.replaceChildren();
      checkRow.replaceChildren();
      checkRow.hidden = true;
      stats.replaceChildren();
      const xs = c.sym ? 'g' : 'h';
      const hkey = c.sym ? hvar : 'x', hg = c.hist[hkey], vn = hkey === 'u' ? 'h' : xs;   // the variable the histogram counts
      const every = c.T > 1;                                   // with one token, "every token" and "this token" are the same rows

      if (i === 0) {
        const lines = c.sym
          ? [`g_{t,f} = ${c.xs}\\,\\Wgate^{\\top}${c.bGate}`, `h_{t,f} = ${c.xs}\\,\\Wup^{\\top}${c.bUp}`]
          : [`h_{t,f} = ${c.xs}\\,\\Wup^{\\top}${c.bUp}`];
        eqMain.replaceChildren(tx(aligned(c.xDef ? [c.xDef, ...lines] : lines)));
        eqLive.append(
          tx(c.sym ? aligned([`g_{${q},${f}} = ${num(x0, 2)}`, `h_{${q},${f}} = ${num(h0, 2)}`]) : `h_{${q},${f}} = ${num(h0, 2)}`),
          note(`unit ${f} · token “${lab}” at position ${q}`),
          note('click a dot to pick a unit'));
        stats.append(
          statRow('units per token', String(c.F)),
          statRow(`this token, ${xs}`, `${numText(c.tXLo, 2)} … ${numText(c.tXHi, 2)}`),
          ...(every ? [statRow(`every token, ${xs}`, `${numText(c.xLo, 2)} … ${numText(c.xHi, 2)}`)] : []),
          ...(c.sym
            ? [statRow(`off-scale, this token, |g| > ${c.W}`, `${c.tOutX} of ${c.F}`), statRow(`off-scale, this token, |h| > ${c.W}`, `${c.tOutU} of ${c.F}`)]
            : [statRow(`off-scale, this token, |h| > ${c.W}`, `${c.tOutX} of ${c.F}`)]));
      } else if (i === 1) {
        eqMain.replaceChildren(tx(c.sym
          ? 'a_{t,f} = \\SiLU(g_{t,f})\\,h_{t,f}'
          : `a_{t,f} = ${c.kind === 'relu' ? '\\ReLU' : '\\GELU'}(h_{t,f})`));
        const w = workedTex(c, r);
        eqLive.append(tx(w.tex), note(`unit ${f} · traced a = `, numEl(a0, 3)), note('click a dot to pick a unit'));
        checkRow.hidden = false;
        checkRow.append(
          checkBadge(calc, a0, { label: 'equals the traced activation' }),
          h('small', null, `all ${C(c.N)} values in this layer: max |Δ| `, numEl(c.maxAll, 1)));
        stats.append(
          statRow('negative, this token', `${c.negTok} of ${c.F}`),
          statRow('largest, this token', numText(c.tAHi, 2)),
          statRow('every output', `${numText(c.aLo, 2)} … ${numText(c.aHi, 2)}`),
          statRow(`off-scale, every token, a outside ±${c.Yc}`, `${C(c.outA)} of ${C(c.N)}`));
      } else if (i === 2) {
        eqMain.replaceChildren(tx(aligned(['\\text{share near zero}', '= \\tfrac{\\#\\{\\,|a_{t,f}|<0.05\\,\\}}{T\\,F}'])));
        const big = h('span');
        const key = `${c.N}|${c.zeros}`;
        if (key !== countKey) { countTo(big, (c.zeros / c.N) * 100, { digits: 1 }); countKey = key; }
        else big.textContent = (c.zeros / c.N * 100).toFixed(1);
        eqLive.append(
          tx(`= \\dfrac{${c.zeros}}{${c.T}\\cdot${c.F}} = \\dfrac{${c.zeros}}{${c.N}}`),
          h('div', { class: 'ac-big' }, big, '%'),
          note('of every output in this layer'));
        checkRow.hidden = false;
        checkRow.append(
          checkBadge(c.recount, c.zeros, { label: 'matches the traced outputs' }),
          h('small', null, `all ${C(c.N)} outputs, recomputed from their pre-activations`));
        stats.append(
          statRow('near zero · every token', `${C(c.zeros)} of ${C(c.N)}`),
          ...(every ? [statRow('near zero · this token', `${c.nearTok} of ${c.F}`)] : []),
          statRow(`off-scale, every token, |${vn}| > ${c.W}`, `${C(hg.outside)} of ${C(c.N)}`));
      } else {
        eqMain.replaceChildren(compareRows(c));
        eqLive.append(c.sym
          ? note(`unit ${f} · g = `, numEl(x0, 2), ', h = ', numEl(h0, 2))
          : note(`unit ${f} · h = `, numEl(h0, 2)));
        if (c.sym) {
          stats.append(statRow('SiLU(g)·h · this model', numText(a0, 3)));
          for (const n of ALTS.swiglu) stats.append(statRow(`${ALT_NAME[n]}(g)·h · illustration`, numText(fn[n](x0) * h0, 3), toggles[n] === false));
        } else {
          stats.append(statRow(`${NAME[c.kind]} · this model`, numText(a0, 3)));
          for (const n of ALTS[c.kind]) stats.append(statRow(`${ALT_NAME[n]} · illustration`, numText(fn[n](h0), 3), toggles[n] === false));
        }
      }

      const words = [
        c.sym
          ? 'Two projections read the same input: the gate g sets the curve, and the up value h scales it. Gate values sit on the axis; up values sit in the row below.'
          : `Each unit gives one pre-activation h per token, a linear function of the input before any bend. Here are the ${c.F} values for this token on the axis.`,
        c.sym
          ? 'Each gate value passes through SiLU, and the up value multiplies that output: a = SiLU(g) · h. A large h scales it up, and a negative h flips its sign.'
          : c.kind === 'relu'
            ? 'ReLU sets every negative pre-activation to exactly zero and passes positive ones unchanged. Orange is positive, blue negative, grey zero.'
            : 'GELU maps each pre-activation along the curve: negative inputs move toward zero and large positive ones pass almost unchanged. Orange is positive, blue negative, grey near zero.',
        c.sym
          ? `Across all ${C(c.N)} values in this layer, <b>${pct(c.zeros / c.N)}</b> of the outputs are within 0.05 of zero (solid grey). For SwiGLU that depends on g and h together, so the switch bins the same values by g or by h.`
          : `Across all ${C(c.N)} values in this layer, <b>${pct(c.zeros / c.N)}</b> come out within 0.05 of zero (solid grey, at the foot of each bar).`,
        c.sym
          ? 'Dashed curves put other gate functions in place of SiLU, drawn for h = 1 (illustration), each in its pill’s line style. SiLU keeps a small negative lobe; ReLU cuts it to zero.'
          : c.kind === 'relu'
            ? 'Dashed curves apply other non-linearities to the same inputs (illustration), each in its pill’s line style. GELU and SiLU keep a small negative lobe where this model’s ReLU is exactly zero.'
            : 'Dashed curves apply other non-linearities to the same inputs (illustration), each in its pill’s line style. ReLU zeroes every negative input exactly; this model’s GELU keeps a small negative lobe, the blue dip near zero.',
      ];
      capSet(words[i]);
    }

    // -------------------------------------------------------------- build from the trace (mount, new forward pass, token)
    function rebuild(kind) {
      if (!alive) return;
      const c = analyse(ctx, L);
      if (!c) {
        cur = null;
        plot.hidden = true; histBox.hidden = true; side.hidden = true;
        header.setSub(`Layer ${L} has no MLP trace in the current model.`);
        capSet('This layer has no MLP trace in the current model, so there is no activation to show.');
        return;
      }
      plot.hidden = false; histBox.hidden = false; side.hidden = false;
      c.u = unit != null && unit < c.F ? unit : c.argmin;
      const same = !!cur && cur.kind === c.kind && cur.F === c.F && cur.W === c.W && cur.Yc === c.Yc && dots.length === c.F && (c.sym === laneDots.length > 0);
      const anim = kind !== 'first' && !reducedMotion();
      if (kind === 'select' && same) {
        cur = c;                                             // the histogram is per forward pass, so only the token changed
        paintDots(c, mode, anim);
        texts(step);
        return;
      }
      cur = c;
      if (!same || dots.length !== c.F || (c.sym !== laneDots.length > 0)) buildDots(c);
      buildPlot(c);
      buildToggles(c);
      buildHist(c);
      apply(step, beatsOf(step) - 1, anim);
    }

    // -------------------------------------------------------------- wiring, first paint, auto-advance
    const offs = [
      ctx.on('trace', () => rebuild('trace')),
      ctx.on('select', () => { cancelAuto(); rebuild('select'); }),
    ];
    const offTheme = onThemeChange(() => {
      if (!cur) return;
      paintDots(cur, mode, false);
      paintHist(cur);
    });
    ctx.setSteps(STEPS, onStep);
    el.addEventListener('pointerdown', cancelAuto);
    el.addEventListener('atlas:interrupt', cancelAuto);
    rebuild('first');
    if (cur) {
      appear([header.el, picker.el, plot, histBox, side, cap.el], { stagger: MOTION.stagger });
      if (!reducedMotion()) {
        // auto-advance the first time the scene is shown, one MOTION.step per step
        const run = (k) => {
          if (!alive || k >= STEPS.length) return;
          driving = true; ctx.setStep(k); driving = false;
          autoT = setTimeout(() => run(k + 1), MOTION.step);
        };
        autoT = setTimeout(() => run(1), FIRST_DWELL);
      }
    }

    return {
      destroy() {
        alive = false;
        cancelAuto();
        clearBeats();
        offs.forEach((f) => f());
        offTheme();
        st.destroy();
      },
    };
  },
};

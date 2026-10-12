// Layer normalisation on one token's vector, in steps on the real numbers.
//   LayerNorm: input → μ → x − μ → σ → x̂ = (x − μ)/σ → h = γ·x̂ + β, then every token.
//   RMSNorm:   input → rms → x̂ = x/rms → h = γ·x̂, then every token (no mean, no shift).
// Every number comes from the trace (the stream entering the norm, and the norm's output L{l}.ln{n})
// and from model.params (γ, β). The scene recomputes the output and checks it against the trace.
// Two vertical scales per token. Until the centring, the scale fits this vector's own range. Since μ is small next
// to σ, the Centre step adds a magnified inset (marked illustration) for the move −μ. The second beat of the centring
// (the second beat of the RMS step for RMSNorm) moves to one shared scale for every later step, so the divide that
// follows visibly stretches the bars on a fixed axis. A short tick marks where each bar was before the step.
// Post-LN models have no norm in front of a sub-layer: the scene says so and points at the residual adds.
// Reference scene for contributors: see docs/ATLAS_GUIDE.md and docs/STYLE_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, card, chip, appear, countTo, pulse, useStyle, MOTION, ZONES } from '../kit.js';
import { h, fmt, texHTML, heatmap, diverging, onThemeChange, reducedMotion } from '../../ui/index.js';

const STEPS_LN = ['input', 'Mean', 'Centre', 'Scale', 'γ, β', 'All tokens'];
const STEPS_RMS = ['input', 'RMS', 'Scale', 'γ', 'All tokens'];
const BX = ZONES.hero.x, BY = ZONES.hero.y, BW = 920, BH = 330;   // the bars of one token (design units)
const GY = BY + BH + 44, PY = GY + 56, SH = 40;                    // the γ strip and the β strip, SH tall each
const SX = ZONES.side.x, SY = ZONES.hero.y, SW = ZONES.side.w;     // equation column
const CAP_W = 960;                                                 // keeps 110 units clear of the side column
const CX = ZONES.hero.x + ZONES.hero.w / 2;                        // the hero's centre line (x = 800): the post-LN page is centred on it
const BEAT = 900;                                                  // ms between the two beats of a step
const LX = 130, LW = 420;                                          // the magnified inset: where its number lines start and how long they are
const MOVE = `${MOTION.move}ms ${MOTION.ease}`;
const ANCHOR = { tl: '0 0', c: '-50% -50%', tc: '-50% 0', bc: '-50% -100%', lc: '0 -50%', rc: '-100% -50%', tr: '-100% 0', bl: '0 -100%', br: '-100% -100%' };

useStyle('norm-scene', `
.nm-main { position: absolute; left: 0; top: 0; width: 0; height: 0; }
.nm-main.nm-still, .nm-main.nm-still * { transition: none !important; }
.nm-chart { position: absolute; width: ${BW}px; height: ${BH}px; border-left: 1px solid var(--line-strong); transition: opacity ${MOVE}; }
.nm-zero { position: absolute; left: 0; right: 0; height: 0; border-top: 1px dashed var(--line-strong); transition: top ${MOVE}; }
.nm-bar { position: absolute; border-radius: 3px 3px 1px 1px; box-shadow: 0 2px 6px rgba(0, 0, 0, .12); transition: top ${MOVE}, height ${MOVE}, background-color .4s; }
.nm-bar.hl { box-shadow: 0 0 0 2px color-mix(in srgb, var(--fg) 55%, transparent), 0 2px 6px rgba(0, 0, 0, .12); }
.nm-was { position: absolute; height: 2px; border-radius: 1px; background: var(--fg); opacity: 0; pointer-events: none; }
.nm-was.on { opacity: .6; }
.nm-val { display: none; position: absolute; left: 50%; translate: -50% 0; font: 600 13px/1 var(--font-mono); color: var(--fg); white-space: nowrap; pointer-events: none; transition: top ${MOVE}; }
.nm-bar.hl .nm-val { display: block; }
.nm-mu { position: absolute; left: 0; right: 0; height: 0; border-top: 2px solid var(--fg); opacity: 0; transition: top ${MOVE}, opacity .4s; }
.nm-band { position: absolute; left: 0; right: 0; height: 0; opacity: 0; background: color-mix(in srgb, var(--c-norm) 16%, transparent); border-block: 1px dashed color-mix(in srgb, var(--c-norm) 85%, transparent); transition: top ${MOVE}, height ${MOVE}, opacity .4s; }
.nm-tag { position: absolute; left: ${BW + 10}px; translate: 0 -50%; font: 600 14px/1 var(--font-mono); color: var(--fg-2); padding: 5px 9px; border-radius: 7px; background: var(--surface); border: 1px solid var(--line); white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, .08); opacity: 0; transition: top ${MOVE}, opacity .4s; }
.nm-tag.on { opacity: 1; }
.nm-tag.band { color: var(--c-norm); }
.nm-dims { position: absolute; width: ${BW}px; height: 14px; transition: opacity ${MOVE}; }
.nm-dims span { position: absolute; top: 0; translate: -50% 0; font: 500 12px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.nm-dims .nm-dimname { translate: 0 0; font-style: italic; }
.nm-strip { position: absolute; width: ${BW}px; height: ${SH}px; transition: opacity ${MOVE}; }
.nm-w { position: absolute; border-radius: 2px; transition: top ${MOVE}, height ${MOVE}; }
.nm-slab { position: absolute; margin: 0; font: 600 17px/1 var(--font-mono); color: var(--c-norm); transition: opacity ${MOVE}; }
.nm-note-b { position: absolute; left: 50%; top: 50%; translate: -50% -50%; font: 500 14px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.nm-note-b[hidden] { display: none; }
.nm-hm { position: absolute; left: 0; top: 0; width: ${BW}px; display: flex; gap: 28px; align-items: flex-start; opacity: 0; pointer-events: none; transition: opacity ${MOVE}; }
.nm-hmcol { flex: none; display: grid; gap: 8px; justify-items: start; }
.nm-hmlab { font: 600 15px/1.2 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.nm-hmlab code { font: 500 14px/1 var(--font-mono); color: var(--c-norm); }
.nm-hmnote { position: absolute; font: 500 14px/1.4 var(--font-mono); color: var(--fg-3); white-space: nowrap; opacity: 0; transition: opacity ${MOVE}; }
.nm-hmarrow { align-self: center; display: grid; justify-items: center; gap: 6px; width: 80px; color: var(--fg-3); }
.nm-hmarrow span { font: 500 34px/1 var(--font-mono); }
.nm-hmarrow small { font: 500 13px/1 var(--font-mono); }
.nm-side { position: absolute; width: ${SW}px; display: grid; gap: 10px; align-content: start; }
.nm-eq-main { font-size: 22px; color: var(--fg); overflow-x: auto; }
.nm-eq-live { font-size: 18px; padding: 10px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); box-shadow: 0 6px 16px rgba(0, 0, 0, .06); overflow-x: auto; }
.nm-checkrow { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.nm-checkrow[hidden] { display: none; }
.nm-checkrow small { font: 500 13px/1 var(--font-mono); color: var(--fg-3); }
.nm-stats { display: grid; font: 500 15px/1.3 var(--font-mono); color: var(--fg-2); }
.nm-stat { display: flex; justify-content: space-between; gap: 12px; padding: 4px 2px; border-bottom: 1px solid var(--line); }
.nm-stat b { color: var(--fg); font-weight: 600; }
.nm-mini { display: grid; gap: 6px; justify-items: start; }
.nm-mini[hidden] { display: none; }
.nm-minilab { font: 600 14px/1 var(--font-mono); color: var(--fg-3); }
.nm-note { position: absolute; width: 900px; display: grid; justify-items: center; gap: 18px; text-align: center; }
.nm-note[hidden] { display: none; }
.nm-note-text { font-size: 17px; line-height: 1.4; color: var(--fg-2); }
.nm-card-eq { font-size: 17px; color: var(--fg); margin-top: 6px; }
.nm-card[hidden] { display: none; }
.nm-axis { position: absolute; font: 500 15px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; transition: opacity .4s; }
.nm-shift { position: absolute; width: ${BW}px; height: 130px; pointer-events: none; transition: opacity ${MOVE}; }
.nm-sh-head { position: absolute; left: 0; top: 0; font: 500 15px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.nm-sh-row { position: absolute; left: 0; width: ${BW}px; height: 22px; }
.nm-sh-row[hidden] { display: none; }
.nm-sh-lab { position: absolute; left: 0; top: 50%; translate: 0 -50%; font: 600 15px/1 var(--font-mono); color: var(--fg); white-space: nowrap; }
.nm-sh-line { position: absolute; top: 50%; height: 0; border-top: 1px solid var(--line-strong); }
.nm-sh-arrow { position: absolute; top: 50%; height: 3px; translate: 0 -50%; border-radius: 2px; background: var(--c-norm); transition: left ${MOVE}, width ${MOVE}, opacity .4s; }
.nm-sh-dot { position: absolute; top: 50%; width: 12px; height: 12px; box-sizing: border-box; border-radius: 50%; translate: -50% -50%; transition: left ${MOVE}, opacity .4s; }
.nm-sh-dot.was { border: 2px solid var(--fg-2); background: var(--surface); }
.nm-sh-dot.now { border: 2px solid var(--c-norm); background: var(--c-norm); }
.nm-sh-val { position: absolute; top: 50%; translate: 0 -50%; font: 500 15px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.nm-sh-foot { position: absolute; left: 0; top: 112px; font: 500 14px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
`);

// ------------------------------------------------------------------ pure helpers
const absMax = (a) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m || 1; };
const peak = (...arrs) => { let m = 0; for (const a of arrs) for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };
const span = (a) => [Math.min(...a), Math.max(...a)];
const barW = (d) => Math.max(1.5, Math.min(26, (BW / d) * 0.62));
// y-scale for a short strip: maps values into [0, H] with the zero line at y0
const mapOf = (arr, H) => {
  let lo = 0, hi = 0;
  for (const v of arr) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = (hi - lo || 1) * 0.15;
  lo -= pad; hi += pad;
  const K = H / (hi - lo);
  return { K, Y: (v) => (hi - v) * K, y0: hi * K };
};
const aligned = (lines) => `\\begin{aligned}${lines.map((l) => `&${l}`).join(' \\\\ ')}\\end{aligned}`;
const stack = (lines) => `\\begin{aligned}${lines.join(' \\\\ ')}\\end{aligned}`;
const tk = (k) => `\\mathtt{${k.replace(/_/g, '\\_')}}`;
// Live numbers. Every printed result is computed from the printed operands, so each line can be redone by
// hand and gives the same digits (the bars and the check still use the exact values).
const r5 = (v) => { const r = Math.round(v * 1e5) / 1e5; return r === 0 ? 0 : r; };
const r4 = (v) => { const r = Math.round(v * 1e4) / 1e4; return r === 0 ? 0 : r; };
const r3 = (v) => { const r = Math.round(v * 1e3) / 1e3; return r === 0 ? 0 : r; };
const s5 = (v) => r5(v).toFixed(5);
const t5 = (v) => v.toFixed(5);
const t4 = (v) => r4(v).toFixed(4);
const t3 = (v) => r3(v).toFixed(3);
const tp4 = (v) => (r4(v) < 0 ? `(${t4(v)})` : t4(v));
const tp3 = (v) => (r3(v) < 0 ? `(${t3(v)})` : t3(v));
// ε written short: 10^{-5} rather than 1.00×10^{-5}, so the live line fits the column
const epsTeX = (e) => { const [m, x] = e.toExponential(2).split('e'); return m === '1.00' ? `10^{${Number(x)}}` : `${m}\\times10^{${Number(x)}}`; };
const pctTxt = (x) => (x < 0.01 ? 'under 1%' : `${Math.round(x * 100)}%`);
const sig3 = (v) => Number(v).toPrecision(3);                       // three significant figures for small spreads
const rangeSig = (a) => `${sig3(a[0])} … ${sig3(a[1])}`;
const rangeFix = (a, n) => `${fmt(a[0], n)} … ${fmt(a[1], n)}`;
const ratio = (a) => `${(a[1] / a[0]).toFixed(1)}×`;
// Dividing by s (σ or rms) multiplies every value by k = 1/s: the bars shrink when s > 1 and stretch when s < 1.
const divideTxt = (s, unit) => {
  const k = 1 / s, sym = `${unit} = ${fmt(s, 2)}`;
  if (Math.abs(k - 1) < 0.005) return `Dividing by ${sym} leaves every bar where it is, so the band stays at ±1.`;
  return k < 1
    ? `Dividing by ${sym} shrinks every bar to ${k.toFixed(2)}× its size, so the band shrinks from ±${unit} to ±1.`
    : `Dividing by ${sym} stretches every bar by ${k.toFixed(2)}×, so the band grows from ±${unit} to ±1.`;
};

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const L = Number(ctx.params.l), N = Number(ctx.params.n);
    let labels = [], labelsKey = '', step = 0, lastFrame = [0, 0], cur = null, alive = true, driving = false;
    let gen = 0, beatTimers = new Set(), autoT = null, revealed = new Set(), checkShown = false;
    let bars = [], ticks = [], gBars = [], bBars = [], gMap = null, bMap = null;
    let heroHM = [], heroKey = '', miniHM = null, miniKey = '';

    // the stream entering the norm, and the norm's own output (both traced)
    const inKey = () => (L === 0 && N === 1 ? 'emb.out' : N === 1 ? `L${L - 1}.resid_out` : `L${L}.resid_mid`);
    const outKey = () => `L${L}.ln${N}`;
    const available = () => { const s = ctx.state; return !!(s.trace && s.model && s.trace[inKey()] && s.trace[outKey()]); };
    const preChips = () => [[`Layer ${L}`, 'resid'], [N === 1 ? 'before attention' : 'before MLP', 'norm']];

    // ------------------------------------------------------------ shared kit parts
    const header = sceneHeader(st, { chips: preChips(), title: 'Normalisation' });
    const setChips = (list) => header.el.querySelector('.k-eyebrow')?.replaceChildren(...list.map(([t, k]) => chip(t, k)));
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'norm' });
    const cap = sceneCaption(st, { w: CAP_W });

    // ------------------------------------------------------------ DOM: everything here is hidden for post-LN
    const main = h('div', { class: 'nm-main' });
    st.add(main, { x: 0, y: 0 });
    const place = (node, x, y, anchor = 'tl') => {
      node.style.position = 'absolute';
      node.style.left = `${x}px`;
      node.style.top = `${y}px`;
      node.style.translate = ANCHOR[anchor];
      main.append(node);
      return node;
    };

    // hero: the bars of one token
    const zero = h('div', { class: 'nm-zero' });
    const mu = h('div', { class: 'nm-mu' });
    const muTag = h('div', { class: 'nm-tag' });
    const band = h('div', { class: 'nm-band' });
    const bandTag = h('div', { class: 'nm-tag band' });
    const chart = place(h('div', { class: 'nm-chart' }, zero, mu, band, muTag, bandTag), BX, BY);
    const dims = place(h('div', { class: 'nm-dims' }), BX, BY + BH + 8);

    // γ and β: one bar per dimension, under the chart, in the diverging ramp (they are signed weights)
    const gZero = h('div', { class: 'nm-zero' }), bZero = h('div', { class: 'nm-zero' });
    const bNote = h('div', { class: 'nm-note-b' }, 'no shift β in this model (bias off)');
    const gStrip = place(h('div', { class: 'nm-strip' }, gZero), BX, GY);
    const bStrip = place(h('div', { class: 'nm-strip' }, bZero, bNote), BX, PY);
    const gLab = place(h('div', { class: 'nm-slab' }, 'γ'), BX - 14, GY + SH / 2, 'rc');
    const bLab = place(h('div', { class: 'nm-slab' }, 'β'), BX - 14, PY + SH / 2, 'rc');

    // which scale the bars are on (it changes on the second beat of the centring, or of the RMS step), and the magnified
    // shift for the Centre step: up to three dimensions, each zoomed around its own values, the value before (ring) and after (dot)
    const axisLab = place(h('div', { class: 'nm-axis' }), BX, BY - 16);
    const shiftHead = h('div', { class: 'nm-sh-head' });
    const shiftRows = Array.from({ length: 3 }, (_, r) => {
      const R = {
        lab: h('div', { class: 'nm-sh-lab' }),
        arrow: h('div', { class: 'nm-sh-arrow' }),
        was: h('div', { class: 'nm-sh-dot was' }),
        now: h('div', { class: 'nm-sh-dot now' }),
        val: h('div', { class: 'nm-sh-val', style: { left: `${LX + LW + 28}px` } }),
        pos: null, txt: null,
      };
      R.row = h('div', { class: 'nm-sh-row', style: { top: `${30 + r * 26}px` } }, R.lab,
        h('div', { class: 'nm-sh-line', style: { left: `${LX}px`, width: `${LW}px` } }), R.arrow, R.was, R.now, R.val);
      return R;
    });
    const shiftFoot = h('div', { class: 'nm-sh-foot' });
    const shift = place(h('div', { class: 'nm-shift' }, shiftHead, ...shiftRows.map((R) => R.row), shiftFoot), BX, GY);

    // final step: all tokens before and after, two heatmaps in the hero zone, and a key under them
    const hero = place(h('div', { class: 'nm-hm' }), BX, BY);
    const heroNote = place(h('div', { class: 'nm-hmnote' }, 'rows: the tokens · columns: the dimensions · blue below 0, orange above'), BX, BY);

    // side column: the step's equation, the live numbers, the check, the ledger and the stream
    const sideLab = h('div', { class: 'k-label', style: { textAlign: 'left' } });   // the kit's step title, as in softmax.js
    const eqMain = h('div', { class: 'nm-eq-main' });
    const eqLive = h('div', { class: 'nm-eq-live' });
    const check = h('div', { class: 'nm-checkrow', hidden: true });
    const stats = h('div', { class: 'nm-stats' });
    const miniBox = h('div', { class: 'nm-mini' }, h('div', { class: 'nm-minilab' }, 'stream · every token'));
    const side = place(h('div', { class: 'nm-side' }, sideLab, eqMain, eqLive, check, stats, miniBox), SX, SY);

    // post-LN: no norm before the sub-layer; the norm acts on each residual add
    const note = h('div', { class: 'nm-note', hidden: true }, h('div', { class: 'nm-note-text' }, 'Post-LN: the norm runs on each residual sum, so each residual add below is followed by its own norm.'));
    st.add(note, { x: CX, y: 340, anchor: 'tc' });
    const eqCard = (tex) => h('div', { class: 'nm-card-eq', html: texHTML(tex) });
    const noteCards = [
      card({ title: 'Residual add · attention', sub: 'the attention output joins the stream, then the sum is normalised', kind: 'resid', node: `residual.${L}.1`, w: 440, body: eqCard(`x^{\\text{mid}}_{${L}} = \\text{ln1}\\big(x_{${L}} + \\Attn(x_{${L}})\\big)`) }),
      card({ title: 'Residual add · MLP', sub: 'the MLP output joins the stream, then the sum is normalised', kind: 'resid', node: `residual.${L}.2`, w: 440, body: eqCard(`x^{\\text{out}}_{${L}} = \\text{ln2}\\big(x^{\\text{mid}}_{${L}} + \\MLP(x^{\\text{mid}}_{${L}})\\big)`) }),
    ];
    noteCards.forEach((c, k) => { c.classList.add('nm-card'); c.hidden = true; st.add(c, { x: CX + (k ? 240 : -240), y: 430, anchor: 'tc' }); });

    // ------------------------------------------------------------ the maths of the current token
    function compute() {
      const s = ctx.state, cfg = s.config, P = s.model.params;
      const d = cfg.d_model, T = s.ids.length, eps = cfg.norm_eps, isRms = cfg.norm === 'rmsnorm';
      const ik = inKey(), ok = outKey(), X = s.trace[ik].data, Y = s.trace[ok].data;
      const pn = `blocks.${L}.ln${N}`;
      const g = Array.from(P[`${pn}.weight`].data);
      // the engine creates the shift only for a LayerNorm with bias on (the same test as normOp)
      const beta = !isRms && P[`${pn}.bias`] ? Array.from(P[`${pn}.bias`].data) : null;
      const q = Math.max(0, Math.min(T - 1, s.sel.token ?? T - 1));
      // mean, variance and rms of the row of src that starts at offset o (same order of sums as the engine)
      const stat = (src, o) => {
        let s1 = 0, s2 = 0;
        for (let j = 0; j < d; j++) { const v = src[o + j]; s1 += v; s2 += v * v; }
        const m = s1 / d;
        let v2 = 0;
        for (let j = 0; j < d; j++) { const e = src[o + j] - m; v2 += e * e; }
        v2 /= d;
        return { mu: m, v2, sig: Math.sqrt(v2 + eps), ms: s2 / d, rms: Math.sqrt(s2 / d + eps) };
      };
      const o = q * d, s0 = stat(X, o);
      const row = Array.from(X.subarray(o, o + d));
      const centred = row.map((v) => v - s0.mu);
      const xh = isRms ? row.map((v) => v / s0.rms) : centred.map((v) => v / s0.sig);
      const gx = xh.map((v, j) => g[j] * v);
      const hh = gx.map((v, j) => v + (beta ? beta[j] : 0));
      let maxDiff = 0, jOut = 0;
      for (let j = 0; j < d; j++) {
        maxDiff = Math.max(maxDiff, Math.abs(hh[j] - Y[o + j]));
        if (Math.abs(row[j]) > Math.abs(row[jOut])) jOut = j;
      }
      // every token: the scale before the norm, the normalised rows (before γ and β), the output check,
      // and |μ| ÷ σ, the size of the shift relative to the spread
      const XH = new Float64Array(T * d), before = [], after = [], rel = [];
      let maxAll = 0;
      for (let t = 0; t < T; t++) {
        const k = t * d, sk = stat(X, k);
        before.push(isRms ? sk.rms : sk.sig);
        rel.push(isRms ? 0 : Math.abs(sk.mu) / sk.sig);
        for (let j = 0; j < d; j++) XH[k + j] = isRms ? X[k + j] / sk.rms : (X[k + j] - sk.mu) / sk.sig;
        const sh = stat(XH, k);
        after.push(isRms ? sh.ms : sh.v2);
        for (let j = 0; j < d; j++) maxAll = Math.max(maxAll, Math.abs(g[j] * XH[k + j] + (beta ? beta[j] : 0) - Y[k + j]));
      }
      let jm = 0;
      for (let t = 1; t < T; t++) if (rel[t] > rel[jm]) jm = t;
      // the printed denominators of the live lines (σ or rms from the printed σ² or mean x², to 5 decimals)
      const sigP = r5(Math.sqrt(r5(s0.v2) + eps)), rmsP = r5(Math.sqrt(r5(s0.ms) + eps));
      // two vertical scales. The stream scale fits this vector's own range (with μ and the centred values), so the bars
      // fill the chart before the centring. The shared scale comes in on the second beat of the centring and holds for
      // every later frame (the centred values, the normalised values and the bands), so the divide visibly stretches them.
      const bandV = isRms ? s0.rms : s0.sig, mu0 = isRms ? 0 : s0.mu;
      const scaleOf = (arrs, lo, hi) => {
        for (const arr of arrs) for (const v of arr) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        const pad = (hi - lo || 1) * 0.08;
        lo -= pad; hi += pad;
        const K = BH / (hi - lo);
        return { K, Y: (v) => (hi - v) * K, y0: hi * K };
      };
      const axS = scaleOf(isRms ? [row] : [row, centred], Math.min(0, mu0, -bandV), Math.max(0, mu0, bandV));
      const axN = scaleOf(isRms ? [row, xh, gx, hh] : [row, centred, xh, gx, hh], Math.min(-1, -bandV), Math.max(1, bandV));
      // the magnified shift: a window of half-width hw around the midpoint of each value and its centred value, with hw = 2|μ|
      // (at least 0.002), so the move −μ is a quarter of the line for a typical μ. mag is the zoom of that window on the stream scale.
      const hw = Math.max(2 * Math.abs(mu0), 2e-3), mag = LW / (2 * hw) / axS.K;
      const maxS = Math.max(peak(row), isRms ? 0 : peak(centred), Math.abs(mu0)), maxN = Math.max(peak(xh), peak(gx), peak(hh), 1);
      return {
        isRms, hasBias: !!beta, ik, ok, d, T, q, eps, X, Y, XH, g, beta, row, centred, xh, gx, hh, s0,
        sumC: centred.reduce((a, b) => a + b, 0), maxDiff, maxAll, jOut, maxAbs: Math.abs(row[jOut]),
        before: span(before), after: span(after),
        rel: rel[q], relMax: rel[jm], relAt: jm, sigP, rmsP,
        axS, axN, maxS, maxN, mag, hw,
        // colour scales: one for the stream before normalising, one for the normalised values (same value, same colour within a space)
        Rx: Math.max(absMax(row), absMax(centred)), Rn: Math.max(absMax(xh), absMax(gx), absMax(hh)),
        gR: absMax(g), bR: beta ? absMax(beta) : 1,
      };
    }

    // what the bars show in each step and beat (band = the ±value drawn, tag = its label). A beat that only moves the axis
    // (the second beat of the centring, and of the RMS step) shows the same bars as the beat before it.
    function frame(i, b) {
      const c = cur;
      if (c.isRms) {
        if (i === 0) return { vals: c.row };
        if (i === 1) return { vals: c.row, band: c.s0.rms, tag: '±rms' };
        if (i === 2) return { vals: c.xh, band: 1, tag: '±1' };
        if (i === 3) return { vals: c.hh, gamma: true };
        return { vals: c.hh, gamma: true, heat: true };
      }
      if (i === 0) return { vals: c.row };
      if (i === 1) return { vals: c.row, mu: c.s0.mu };
      if (i === 2) return { vals: c.centred, mu: 0, muTag: 'x − μ' };
      if (i === 3) return b === 0 ? { vals: c.centred, band: c.s0.sig, tag: '±σ' } : { vals: c.xh, band: 1, tag: '±1' };
      if (i === 4) return b === 0 ? { vals: c.gx, gamma: true } : { vals: c.hh, gamma: true, beta: c.hasBias };
      return { vals: c.hh, gamma: true, beta: c.hasBias, heat: true };
    }

    // is this frame in the normalised space (after the divide) rather than the stream before it?
    const inNorm = (i, b) => (cur.isRms ? i >= 2 : i >= 4 || (i === 3 && b === 1));
    // is this frame on the shared scale? From the second beat of the centring (or of the RMS step) on: the axis changes
    // there, and the divide that follows happens on the same axis
    const onShared = (i, b) => (cur.isRms ? i > 1 || (i === 1 && b === 1) : i > 2 || (i === 2 && b === 1));

    // the bars a frame moves from (drawn as ticks); null at the very first frame
    function priorVals(i, b) {
      if (b > 0) return frame(i, b - 1).vals;
      if (i === 0) return null;
      return frame(i - 1, beatsOf(i - 1) - 1).vals;
    }

    // reduced motion and first paint: write the end state without transitions
    function still(animate, fn) {
      if (animate) { fn(); return; }
      main.classList.add('nm-still');
      fn();
      void main.offsetWidth;
      main.classList.remove('nm-still');
    }

    function applyFrame(i, b, animate) {
      const c = cur, F = frame(i, b), vals = F.vals, d = c.d, R = inNorm(i, b) ? c.Rn : c.Rx;
      const shared = onShared(i, b), ax = shared ? c.axN : c.axS;   // the scale the bars are drawn on
      const shiftOn = !c.isRms && i === 2;                          // the magnified shift: LayerNorm, the Centre step only
      const anim = animate && !reducedMotion();            // reduced motion: end states, no transitions
      lastFrame = [i, b];
      const prev = priorVals(i, b);
      const moved = !!prev && prev.some((v, j) => Math.abs(v - vals[j]) > 1e-12);
      const heatOn = !!F.heat;
      const finalOut = c.isRms ? i === 3 : i === 4 && (b === 1 || !c.hasBias);
      const showCheck = c.isRms ? i >= 3 : (i === 4 && (b === 1 || !c.hasBias)) || i === 5;
      still(anim, () => {
        zero.style.top = `${ax.y0}px`;
        if (F.mu != null) {
          mu.style.top = `${ax.Y(F.mu)}px`;
          mu.style.opacity = '1';
          muTag.textContent = F.muTag ?? `μ = ${fmt(F.mu, 4)}`;
          muTag.style.top = `${ax.Y(F.mu)}px`;
          muTag.classList.add('on');
        } else { mu.style.opacity = '0'; muTag.classList.remove('on'); }
        if (F.band != null) {
          band.style.top = `${ax.Y(F.band)}px`;
          band.style.height = `${ax.Y(-F.band) - ax.Y(F.band)}px`;
          band.style.opacity = '1';
          bandTag.textContent = F.tag;
          bandTag.style.top = `${ax.Y(F.band)}px`;
          bandTag.classList.add('on');
        } else { band.style.opacity = '0'; bandTag.classList.remove('on'); }
        for (let j = 0; j < d; j++) {
          const v = vals[j], bar = bars[j], hp = Math.abs(v) * ax.K;
          bar.style.top = `${v >= 0 ? ax.Y(v) : ax.y0}px`;
          bar.style.height = `${Math.max(1.5, hp)}px`;
          bar.style.background = diverging(v / R);
          if (j === c.jOut) {
            const lab = bar.firstChild;
            lab.textContent = fmt(v, 2);
            lab.style.top = v >= 0 ? '-22px' : `${hp + 6}px`;
          }
          // where this bar was before the step (only when the step moves it)
          ticks[j].classList.toggle('on', moved);
          if (moved) ticks[j].style.top = `${ax.Y(prev[j]) - 1}px`;
        }
        // γ and β: the strips light up and their bars grow to the values when the step uses them
        for (let j = 0; j < d; j++) {
          const gv = c.g[j], gb = gBars[j];
          gb.style.top = `${gv >= 0 ? gMap.Y(gv) : gMap.y0}px`;
          gb.style.height = `${F.gamma ? Math.max(1.5, Math.abs(gv) * gMap.K) : 0}px`;
          gb.style.background = diverging(gv / c.gR);
          const bb = bBars[j];
          if (bMap) {
            const bv = c.beta[j];
            bb.style.top = `${bv >= 0 ? bMap.Y(bv) : bMap.y0}px`;
            bb.style.height = `${F.beta ? Math.max(1.5, Math.abs(bv) * bMap.K) : 0}px`;
            bb.style.background = diverging(bv / c.bR);
          } else bb.style.height = '0px';          // no shift β (bias off): no bars, so the note stays clear
        }
        // the final step swaps the bars for the two heatmaps, so nothing sits underneath them
        chart.style.opacity = heatOn ? '0' : '1';
        chart.style.pointerEvents = heatOn ? 'none' : '';
        dims.style.opacity = heatOn ? '0' : '1';
        // γ and β fade in with their step. With bias off, the β strip carries its note from the γ step on.
        const betaOn = F.beta || (!c.isRms && !c.hasBias && i >= 4);
        gStrip.style.opacity = heatOn || shiftOn || !F.gamma ? '0' : '1';
        bStrip.style.opacity = heatOn || shiftOn || !betaOn ? '0' : '1';
        // the magnified shift takes the γ and β place on the Centre step; the axis label names the scale in use
        shift.style.opacity = shiftOn ? '1' : '0';
        axisLab.style.opacity = heatOn ? '0' : '1';
        // the axis names its units: data units until the divide has happened, units of σ (or rms) after it
        axisLab.textContent = !shared
          ? `scale ±${fmt(c.maxS, 2)} · fitted to this vector’s own range`
          : inNorm(i, b)
            ? `scale ±${fmt(c.maxN, 1)} · shared from here, in units of ${c.isRms ? 'rms' : 'σ'}`
            : `scale ±${fmt(c.maxN, 1)} · shared from here, data units`;
        if (shiftOn) {
          const zoom = c.mag >= 1.5 ? `zoomed ×${Math.round(c.mag)}` : 'at this scale';
          shiftHead.textContent = `illustration · each value moves by −μ = ${fmt(-c.s0.mu, 4)} · ${zoom}`;
        }
        shiftRows.forEach((sr) => {
          if (!sr.txt) return;
          const [pa, pb] = sr.pos;
          sr.was.style.left = `${pa}px`;
          sr.now.style.left = `${i >= 2 ? pb : pa}px`;   // the dot starts on its ring and moves to the centred value
          sr.arrow.style.left = `${Math.min(pa, pb)}px`;
          sr.arrow.style.width = `${Math.abs(pb - pa)}px`;
          sr.now.style.opacity = shiftOn ? '1' : '0';
          sr.arrow.style.opacity = shiftOn ? '1' : '0';
          sr.val.textContent = sr.txt;
        });
        gLab.style.opacity = gStrip.style.opacity;
        bLab.style.opacity = bStrip.style.opacity;
        hero.style.opacity = heatOn ? '1' : '0';
        hero.style.pointerEvents = heatOn ? 'auto' : 'none';
        heroNote.style.opacity = heatOn ? '1' : '0';
        miniBox.hidden = heatOn || c.T < 4;                // one or two tokens: the stream strip is a sliver, so it goes
        if (i === 0) chart.dataset.trace = c.ik;
        else if (finalOut) chart.dataset.trace = c.ok;
        else delete chart.dataset.trace;
      });
      check.hidden = !showCheck;
      if (showCheck && !checkShown && anim) pulse(check, { scale: 1.03 });
      checkShown = showCheck;
    }

    // ------------------------------------------------------------ text: equations, ledger, check, caption
    // Two equations per step: the general one (main) and the same step for token q and dimension j (live).
    function maths(c, i, b) {
      const { q, d, jOut: j, row: x } = c, eps = epsTeX(c.eps), k = c.ik, dim = `${q},${j}`;
      // the printed numerator and denominator of x̂ (x or x′; rms or σ), and the printed results from them
      const num = c.isRms ? r4(x[j]) : r4(r4(x[j]) - r4(c.s0.mu));
      const den = c.isRms ? c.rmsP : c.sigP;
      const xhP = r3(num / den);
      const gxP = r3(r3(c.g[j]) * xhP);
      const bP = c.hasBias ? r3(c.beta[j]) : 0;
      if (i === 0) return { main: `\\xres_{t} \\in \\mathbb{R}^{${d}}`, live: `${tk(k)}_{${dim}} = ${tp4(x[j])}` };
      if (c.isRms) {
        if (i === 1) return {
          main: `\\operatorname{rms}_{t} = \\sqrt{\\frac{1}{${d}}\\sum_{c} \\xres_{t,c}^{2} + \\varepsilon}`,
          live: stack([`\\frac{1}{${d}}\\sum_{c} x_{${q},c}^{2} &= ${s5(c.s0.ms)}`, `\\operatorname{rms}_{${q}} &= \\sqrt{${s5(c.s0.ms)} + ${eps}}`, `&= ${s5(c.rmsP)}`]),
        };
        if (i === 2) return { main: `\\hat x_{t,c} = \\frac{\\xres_{t,c}}{\\operatorname{rms}_{t}}`, live: `\\hat x_{${dim}} = \\frac{${tp4(x[j])}}{${s5(c.rmsP)}} = ${t3(xhP)}` };
        if (i === 3) return {
          main: `h_{t,c} = \\gamma_{c}\\,\\hat x_{t,c}`,
          live: stack([`h_{${dim}} &= \\gamma_{${j}}\\hat x_{${dim}}`, `&= ${t3(c.g[j])} \\cdot ${tp3(xhP)}`, `&= ${t3(gxP)}`]),
        };
        return {
          main: aligned([`\\frac{1}{${d}}\\sum_{c} \\hat x_{t,c}^{2} = \\frac{\\operatorname{ms}_{t}}{\\operatorname{ms}_{t}+\\varepsilon} \\approx 1`]),
          live: stack([`\\min_{t} &= ${t5(c.after[0])}`, `\\max_{t} &= ${t5(c.after[1])}`]),
        };
      }
      if (i === 1) return {
        main: `\\mu_{t} = \\frac{1}{${d}}\\sum_{c=1}^{${d}} \\xres_{t,c}`,
        live: stack([`\\mu_{${q}} &= \\frac{1}{${d}}\\sum_{c} x_{${q},c}`, `&= ${t4(c.s0.mu)}`]),
      };
      if (i === 2) return {
        main: `x'_{t,c} = \\xres_{t,c} - \\mu_{t}`,
        live: stack([`x'_{${dim}} &= x_{${dim}} - \\mu_{${q}}`, `&= ${tp4(x[j])} - ${tp4(c.s0.mu)}`, `&= ${t4(num)}`]),
      };
      if (i === 3) return b === 0
        ? { main: `\\sigma_{t} = \\sqrt{\\frac{1}{${d}}\\sum_{c} {x'_{t,c}}^{2} + \\varepsilon}`,
            live: stack([`\\sigma^{2}_{${q}} &= ${s5(c.s0.v2)}`, `\\sigma_{${q}} &= \\sqrt{${s5(c.s0.v2)} + ${eps}}`, `&= ${s5(c.sigP)}`]) }
        : { main: `\\hat x_{t,c} = \\frac{x'_{t,c}}{\\sigma_{t}}`,
            live: `\\hat x_{${dim}} = \\frac{${tp4(num)}}{${s5(c.sigP)}} = ${t3(xhP)}` };
      if (i === 4) return b === 0
        ? { main: `h_{t,c} = \\gamma_{c}\\,\\hat x_{t,c}`,
            live: stack([`h_{${dim}} &= \\gamma_{${j}}\\hat x_{${dim}}`, `&= ${t3(c.g[j])} \\cdot ${tp3(xhP)}`, `&= ${t3(gxP)}`]) }
        : { main: `h_{t,c} = \\gamma_{c}\\,\\hat x_{t,c} + \\beta_{c}`,
            live: stack([`h_{${dim}} &= \\gamma_{${j}}\\hat x_{${dim}} + \\beta_{${j}}`, `&= ${t3(gxP)} + ${tp3(bP)}`, `&= ${t3(r3(gxP + bP))}`]) };
      return {
        main: aligned([`\\operatorname{Var}_{c}\\,\\hat x_{t,c} = \\frac{\\sigma^{2}_{t}}{\\sigma^{2}_{t}+\\varepsilon} \\approx 1`]),
        live: stack([`\\min_{t} &= ${t5(c.after[0])}`, `\\max_{t} &= ${t5(c.after[1])}`]),
      };
    }

    // The ledger: the live numbers this step has computed so far (count-up the first time a step shows them).
    function statRows(i, animate) {
      const c = cur, rows = [], many = c.T > 1;      // one token shows one value, not a range
      // the digits match the live lines: μ to 4 places, σ and σ² to 5 (the equation's rounding), the rest to 3
      const line = (label, value, key, digits = 3) => {
        const v = h('b');
        if (typeof value === 'number') {
          if (animate && !revealed.has(key)) { revealed.add(key); countTo(v, value, { digits }); }
          else v.textContent = fmt(value, digits);
        } else v.innerHTML = value;
        rows.push(h('div', { class: 'nm-stat' }, h('span', { html: label }), v));
      };
      if (c.isRms) {
        if (i === 0) { line('values per token', String(c.d), 'd'); line('largest |x|', `${fmt(c.maxAbs, 2)} · dim ${c.jOut}`, 'max'); }
        if (i >= 1 && i <= 3) { line('mean x²', r5(c.s0.ms), 'ms', 5); line('rms = √(mean x² + ε)', c.rmsP, 'rms', 5); }
        if (i === 4) {
          if (many) line('rms per token', rangeSig(c.before), 'rb'); else line('rms', sig3(c.before[0]), 'rb');
          if (many) line('largest ÷ smallest', ratio(c.before), 'ratio');
          line('mean x̂² after', many ? rangeFix(c.after, 5) : fmt(c.after[0], 5), 'ra');
        }
      } else {
        if (i === 0) { line('values per token', String(c.d), 'd'); line('largest |x|', `${fmt(c.maxAbs, 2)} · dim ${c.jOut}`, 'max'); }
        if (i === 1) { line('μ', r4(c.s0.mu), 'mu', 4); line('largest |μ| ÷ σ', `${fmt(c.relMax, 3)} · position ${c.relAt}`, 'relmax'); }
        if (i === 2) { line('μ', r4(c.s0.mu), 'mu', 4); line('Σ x′', Math.abs(c.sumC) < 1e-9 ? '0' : fmt(c.sumC, 3), 'sum'); }
        if (i === 3 || i === 4) { line('σ²', r5(c.s0.v2), 'v2', 5); line('σ = √(σ² + ε)', c.sigP, 'sig', 5); line('|μ| ÷ σ', c.rel, 'rel'); }
        if (i === 5) {
          if (many) line('σ per token', rangeSig(c.before), 'sb'); else line('σ', sig3(c.before[0]), 'sb');
          if (many) line('largest ÷ smallest', ratio(c.before), 'ratio');
          line('Var(x̂) after', many ? rangeFix(c.after, 5) : fmt(c.after[0], 5), 'va');
        }
      }
      return rows;
    }

    // where the input comes from: depends on the layer, the norm (1 or 2) and the residual switch
    function origin(c) {
      const added = ['learned', 'sinusoidal'].includes(ctx.state.config.pos_encoding);   // RoPE and ALiBi act inside attention
      const res = ctx.state.config.residual;
      if (L === 0 && N === 1) return `The token’s embedding${added ? ' (token plus position)' : ''}, the input to layer 0. Its scale comes from the embedding table.`;
      if (N === 1) return res
        ? `The residual stream after block ${L - 1}, one vector per token. Its scale is whatever the earlier layers left behind.`
        : `Residuals are off, so the input is the MLP output of block ${L - 1} alone, one vector per token.`;
      return res
        ? `The residual stream after attention in block ${L}, one vector per token. Its scale is whatever the earlier layers left behind.`
        : `Residuals are off, so the input is the attention output of block ${L} alone, one vector per token.`;
    }

    function caption(c, i, b) {
      if (c.isRms) {
        if (i === 0) return origin(c);
        if (i === 1) return b === 0
          ? 'rms is the root mean square of the values. Nothing is subtracted, so the band is centred on zero.'
          : 'The axis changes here. From now on every step shares one scale, wide enough for the normalised values, so the bars look smaller on it. The next step divides them.';
        if (i === 2) return `${divideTxt(c.rmsP, 'rms')} Every token now has root mean square 1 (up to ε).`;
        if (i === 3) return 'The learned gain γ rescales each dimension. RMSNorm has no shift β. The check on the right confirms the output.';
        return 'Left: every token’s stream before the norm. Right: the same rows after normalising, before γ. Every row now has root mean square 1 (up to ε), so all rows share one scale.';
      }
      if (i === 0) return origin(c);
      if (i === 1) return `μ is the average of the ${c.d} values in this vector (the line). It is ${pctTxt(c.rel)} of σ, so on this axis it ${c.rel < 0.1 ? 'sits next to zero' : 'is visibly off zero'}.`;
      if (i === 2) {
        if (b === 1) return 'The axis changes here. From now on every step shares one scale, wide enough for the normalised values, so the centred bars look smaller on it. The next step divides them.';
        const seen = c.rel < 0.1 ? 'On this axis that move is too small to see' : 'On this axis that move is visible';
        const how = c.mag >= 1.5 ? `the inset below zooms in ×${Math.round(c.mag)}` : 'the inset below shows it at this scale';
        return `Subtracting μ moves every value by the same amount, −μ = ${fmt(-c.s0.mu, 4)}, so the centred values sum to exactly 0. ${seen}; ${how}.`;
      }
      if (i === 3) return b === 0
        ? 'σ is the spread of the centred values: the band marks ±σ on the shared scale.'
        : `${divideTxt(c.sigP, 'σ')} Every token now has spread 1 (up to ε), whatever its input scale was.`;
      if (i === 4) {
        if (b === 0) return c.hasBias
          ? 'The learned gain γ sets the size of each dimension.'
          : 'The learned gain γ sets the size of each dimension. This model has no shift β (bias off), so this is the output, and the check confirms it.';
        return 'The learned shift β moves each dimension by its own offset. The check on the right confirms the output.';
      }
      return 'Left: every token’s stream before the norm, all rows on one colour scale. Right: the same rows after normalising, before γ and β. Every row now has the same spread (up to ε).';
    }

    function setTexts(i, b, animate) {
      const c = cur, q = c.q, d = c.d;
      const where = N === 1 ? `entering layer ${L}’s attention` : `after attention, entering layer ${L}’s MLP`;
      header.setSub(`Token “<b>${ctx.label(q)}</b>” at position ${q}: one vector of ${d} values, ${where}.`);
      sideLab.textContent = labels[i];
      const m = maths(c, i, b);
      eqMain.innerHTML = texHTML(m.main, { display: true });
      eqLive.innerHTML = texHTML(m.live, { display: true });
      stats.replaceChildren(...statRows(i, animate));
      // the check: the output recomputed from x, γ and β against the traced output
      const atToken = c.isRms ? i === 3 : i === 4 && (b === 1 || !c.hasBias);
      const atAll = i === labels.length - 1;
      if (atToken || atAll) {
        const got = atAll ? c.maxAll : c.maxDiff;
        check.replaceChildren(
          checkBadge(got, 0, { label: atAll ? `equals ${c.ok} for ${c.T === 1 ? 'its only token' : `all ${c.T} tokens`}` : `equals ${c.ok}[${q}]`, tol: 1e-9 }),
          h('small', null, `max |Δ| = ${fmt(got, 2)}`));
      }
      cap.set(caption(c, i, b));
    }

    // ------------------------------------------------------------ DOM that depends on the trace
    function buildBars(c) {
      if (bars.length !== c.d) {
        const colW = BW / c.d, bw = barW(c.d);
        bars = Array.from({ length: c.d }, (_, j) => h('div', { class: 'nm-bar', style: { left: `${j * colW + (colW - bw) / 2}px`, width: `${bw}px` } }, h('span', { class: 'nm-val' })));
        ticks = Array.from({ length: c.d }, (_, j) => h('div', { class: 'nm-was', style: { left: `${j * colW + (colW - bw) / 2 - 2}px`, width: `${bw + 4}px` } }));
        chart.replaceChildren(zero, mu, band, muTag, bandTag, ...bars, ...ticks);
      }
      bars.forEach((bar, j) => { bar.dataset.cell = `${c.q},${j},0`; bar.classList.toggle('hl', j === c.jOut); });
    }

    function buildStrips(c) {
      const d = c.d, colW = BW / d, bw = barW(d);
      if (gBars.length !== d) {
        const mk = () => Array.from({ length: d }, (_, j) => h('div', { class: 'nm-w', style: { left: `${j * colW + (colW - bw) / 2}px`, width: `${bw}px` } }));
        gBars = mk();
        bBars = mk();
        gStrip.replaceChildren(gZero, ...gBars);
        bStrip.replaceChildren(bZero, bNote, ...bBars);
      }
      gMap = mapOf(c.g, SH);
      bMap = c.beta ? mapOf(c.beta, SH) : null;
      gZero.style.top = `${gMap.y0}px`;
      bZero.style.top = `${bMap ? bMap.y0 : SH / 2}px`;
      bStrip.hidden = c.isRms;
      bLab.hidden = c.isRms;
      bNote.hidden = c.isRms || c.hasBias;
    }

    // the magnified shift (LayerNorm): up to three dimensions (the largest |x|, the first and the last), each zoomed
    // around its own values, with the window from compute() (hw), so the move −μ is drawn at the same length in every row.
    function buildShift(c) {
      const picks = c.isRms ? [] : [...new Set([c.jOut, 0, c.d - 1])].slice(0, 3);
      const why = (j) => (j === c.jOut ? ' (largest |x|)' : j === 0 ? ' (first)' : ' (last)');
      shiftFoot.textContent = picks.length ? `rows: ${picks.map((j) => `dim ${j}${why(j)}`).join(', ')}` : '';
      shiftRows.forEach((R, r) => {
        const j = picks[r];
        R.row.hidden = j === undefined;
        R.txt = null;
        if (j === undefined) return;
        const x = c.row[j], xp = c.centred[j], mid = (x + xp) / 2;
        const at = (v) => LX + ((v - (mid - c.hw)) / (2 * c.hw)) * LW;
        R.lab.textContent = `dim ${j}`;
        R.pos = [at(x), at(xp)];                         // the value before and after the centring, along the line
        R.txt = `${fmt(x, 4)} → ${fmt(xp, 4)}`;
      });
    }

    function buildDims(c) {
      if (dims.dataset.d === String(c.d)) return;
      dims.dataset.d = String(c.d);
      const colW = BW / c.d, stride = Math.max(1, Math.ceil(26 / colW)), kids = [];
      for (let j = 0; j < c.d; j += stride) kids.push(h('span', { style: { left: `${j * colW + colW / 2}px` } }, String(j)));
      kids.push(h('span', { class: 'nm-dimname', style: { left: `${BW + 12}px` } }, 'dim'));
      dims.replaceChildren(...kids);
    }

    // final step: every token before the norm, and every normalised row after it (before γ and β).
    // Only the selected token's row is labelled, so the matrix stays quiet.
    function buildHero(c) {
      const key = `${c.T}|${c.d}|${c.ik}|${c.ok}`;
      const rl = Array.from({ length: c.T }, (_, t) => (t === c.q ? ctx.label(t) : ''));
      // cells are about square, a little taller (at most twice as tall) when there are few tokens
      const s = Math.max(3, Math.min(8, Math.floor(440 / c.d)));
      const cellH = Math.max(s, Math.min(s * 2, Math.floor(400 / c.T)));
      const rX = absMax(c.X), rN = absMax(c.XH);
      if (heroKey === key && heroHM.length === 2) {
        heroHM[0].update({ data: c.X, range: rX, rowLabels: rl, highlight: { rows: [c.q] } });
        heroHM[1].update({ data: c.XH, range: rN, rowLabels: rl, highlight: { rows: [c.q] } });
      } else {
        heroHM.forEach((x) => x.destroy());
        const mk = (data, range, traceKey, label) => heatmap({
          data, rows: c.T, cols: c.d, mode: 'diverging', range, cell: s, cellH, rowLabels: rl,
          highlight: { rows: [c.q] }, showValues: false, traceKey, ariaLabel: label,
        });
        heroHM = [mk(c.X, rX, c.ik, 'every token before the norm'), mk(c.XH, rN, null, 'every token after normalising, before gamma and beta')];
        heroKey = key;
        const col = (lab, el) => h('div', { class: 'nm-hmcol' }, h('div', { class: 'nm-hmlab', html: lab }), el);
        hero.replaceChildren(
          col(`before · <code>${c.ik}</code>`, heroHM[0].el),
          h('div', { class: 'nm-hmarrow' }, h('span', null, '→'), h('small', null, 'normalise')),
          col('after · x̂, before γ and β', heroHM[1].el));
      }
      // the key sits 16 units under the taller matrix (offsetHeight is in design units: the stage scale does not change layout)
      heroNote.style.top = `${BY + hero.offsetHeight + 16}px`;
    }

    // the stream before the norm, every token, with the current token's row highlighted (side context)
    function buildMini(c) {
      const key = `${c.T}|${c.d}|${c.ik}`;
      // the strip is about 100 units tall at most, so long texts get 1-unit rows (the side column has to fit above the caption)
      const cw = Math.max(2, Math.min(4, Math.floor(240 / c.d))), ch = Math.max(1, Math.min(4, Math.floor(96 / c.T)));
      const rl = Array.from({ length: c.T }, (_, t) => (t === c.q ? ctx.label(t) : ''));
      if (miniHM && miniKey === key) {
        miniHM.update({ data: c.X, range: absMax(c.X), rowLabels: rl, highlight: { rows: [c.q] } });
        return;
      }
      if (miniHM) { miniHM.destroy(); miniHM.el.remove(); }
      miniHM = heatmap({
        data: c.X, rows: c.T, cols: c.d, mode: 'diverging', range: absMax(c.X), cell: cw, cellH: ch, rowLabels: rl,
        highlight: { rows: [c.q] }, showValues: false, traceKey: c.ik, ariaLabel: 'the stream of every token before the norm',
      });
      miniKey = key;
      miniBox.append(miniHM.el);
    }

    // ------------------------------------------------------------ steps and beats
    // two beats: the second beat of the centring (or of the RMS step) moves the axis; LayerNorm's Scale step has a band
    // beat, then the divide; the γ, β step has a beat for β when there is one
    const beatsOf = (i) => {
      if (!cur) return 1;
      if (cur.isRms) return i === 1 ? 2 : 1;
      return i === 2 || i === 3 || (i === 4 && cur.hasBias) ? 2 : 1;
    };

    function later(ms, fn) {
      const g = gen;
      const t = setTimeout(() => { beatTimers.delete(t); if (alive && g === gen) fn(); }, ms);
      beatTimers.add(t);
    }
    function clearBeats() { gen++; beatTimers.forEach(clearTimeout); beatTimers.clear(); }

    function goStep(i) {
      clearBeats();
      step = Math.max(0, Math.min(labels.length - 1, i));
      const anim = !reducedMotion();
      revealed = new Set();
      applyFrame(step, 0, anim);
      setTexts(step, 0, anim);
      if (beatsOf(step) > 1) {
        if (anim) later(BEAT, () => { applyFrame(step, 1, true); setTexts(step, 1, true); });
        else { applyFrame(step, 1, false); setTexts(step, 1, false); }
      }
    }

    function setStep(i) {
      if (!driving) cancelAuto();
      if (cur) goStep(i);
    }

    // auto-advance the first time the scene is shown, one step every MOTION.step ms
    function startAuto() {
      cancelAuto();
      const go = (k) => {
        if (!alive || k >= labels.length) return;
        driving = true; ctx.setStep(k); driving = false;
        autoT = setTimeout(() => go(k + 1), MOTION.step);
      };
      autoT = setTimeout(() => go(1), 1200);
    }
    function cancelAuto() { clearTimeout(autoT); autoT = null; }

    // ------------------------------------------------------------ build (mount, new forward pass, new token)
    function build(animate) {
      if (!alive) return;
      clearBeats();
      const ok = available();
      main.hidden = !ok;
      note.hidden = ok;
      noteCards.forEach((n) => { n.hidden = ok; });
      picker.el.style.display = ok ? '' : 'none';
      if (!ok) {
        cur = null;
        cancelAuto();
        if (labelsKey) { labelsKey = ''; labels = []; ctx.setSteps([], () => {}); }
        header.title.textContent = 'Normalisation';
        setChips([[`Layer ${L}`, 'resid'], ['post-LN', 'norm']]);
        header.setSub('This model is post-LN, so no norm sits in front of this sub-layer.');
        cap.set('In post-LN the norm runs on each residual sum. The residual adds below show where.');
        return;
      }
      cur = compute();
      const c = cur;
      header.title.textContent = c.isRms ? 'RMSNorm' : 'LayerNorm';
      setChips(preChips());
      const want = c.isRms ? STEPS_RMS : STEPS_LN;
      const relabel = want.join('|') !== labelsKey;
      if (relabel) { labels = want; labelsKey = want.join('|'); ctx.setSteps(labels, setStep); }
      step = Math.min(step, labels.length - 1);
      buildBars(c);
      buildStrips(c);
      buildShift(c);
      buildDims(c);
      buildHero(c);
      buildMini(c);
      const fb = beatsOf(step) - 1;
      applyFrame(step, fb, animate);
      setTexts(step, fb, false);
      if (relabel) { driving = true; ctx.setStep(step); driving = false; }
    }
    // ------------------------------------------------------------ wiring
    const offs = [
      ctx.on('trace', () => build(false)),
      ctx.on('select', () => { cancelAuto(); build(!reducedMotion()); }),
    ];
    const offTheme = onThemeChange(() => { if (cur) applyFrame(lastFrame[0], lastFrame[1], false); });
    const onDown = () => cancelAuto();
    el.addEventListener('pointerdown', onDown);

    build(false);
    if (available()) {
      appear([header.el, picker.el, chart, side, cap.el], { stagger: MOTION.stagger });
      if (!reducedMotion()) startAuto();
    }

    return {
      destroy() {
        alive = false;
        cancelAuto();
        clearBeats();
        el.removeEventListener('pointerdown', onDown);
        offs.forEach((f) => f());
        offTheme();
        heroHM.forEach((x) => x.destroy());
        miniHM?.destroy();
        st.destroy();
      },
    };
  },
};

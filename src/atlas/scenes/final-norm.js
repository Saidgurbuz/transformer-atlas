// Final norm: the last block's output for one token, normalised once more before the LM head.
// Three steps on the real numbers:
//   1. the last stream, and its norm after every layer (the stream grows with depth);
//   2. the normalisation, condensed into beats: subtract the mean, divide by the spread, then
//      apply the learned gain γ and shift β (RMSNorm: divide by the rms, then γ);
//   3. the head's fixed scale: before γ, every token's normalised vector has norm √d up to ε.
// Only pre-LN models have a final norm. For post-LN configs the atlas leaves this node
// (nodes.js validFor), and the scene paints nothing while it does.
// Every number comes from the trace (L{last}.resid_out, emb.out, lnf) or from model.params
// (ln_f.weight, ln_f.bias). The scene recomputes lnf from the stream and checks it against the
// traced lnf. Reference scene for contributors: see docs/ATLAS_GUIDE.md and docs/STYLE_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, appear, pulse, useStyle, MOTION, ZONES } from '../kit.js';
import { h, fmt, texHTML, num, diverging, onThemeChange, reducedMotion } from '../../ui/index.js';
import { validFor } from '../nodes.js';

const STEPS = ['Last stream', 'Normalise', 'Ready for the head'];
// One beat per operation of step 2 (four for LayerNorm, three for RMSNorm), 900 ms each. A beat is
// one operation of the maths; MOTION.step (1700 ms) is the time for a whole step, too slow for a beat.
const BEAT = 900;
const MOVE = `${MOTION.move}ms ${MOTION.ease}`;
const BX = ZONES.hero.x, BY = 300, BW = 940, BH = 290;       // bar area in design units
const LAB_Y = 262;                                           // label above the bars
const GS = 30, G_Y = BY + BH + 46, B_Y = G_Y + GS + 14;      // gain and shift strips
const SX = ZONES.side.x, SY = 262, SW = ZONES.side.w;        // side column
const CV_Y = 506, CH = 140;                                  // context chart: label, legend, plot CH high
const ANCHOR = { tl: '0,0', c: '-50%,-50%', tc: '-50%,0', bc: '-50%,-100%', lc: '0,-50%', rc: '-100%,-50%', tr: '-100%,0', bl: '0,-100%', br: '-100%,-100%' };

useStyle('final-norm', `
.fn-main { position: absolute; left: 0; top: 0; width: 0; height: 0; }
.fn-main[hidden] { display: none; }
.fn-main.fn-still, .fn-main.fn-still * { transition: none !important; }
.fn-herolab { text-align: left; margin: 0; }
.fn-hero { position: absolute; width: ${BW}px; height: ${BH}px; border-left: 1px solid var(--line-strong); }
.fn-zero { position: absolute; left: 0; right: 0; height: 0; border-top: 1px dashed var(--line-strong); transition: top ${MOVE}; }
.fn-bar { position: absolute; border-radius: 3px 3px 1px 1px; box-shadow: 0 2px 6px rgba(0, 0, 0, .12); transition: top ${MOVE}, height ${MOVE}, background-color .4s; }
.fn-bar.hl { outline: 2px solid var(--fg); outline-offset: 2px; }
.fn-val { display: none; position: absolute; left: 50%; translate: -50% 0; font: 600 13px/1 var(--font-mono); color: var(--fg); white-space: nowrap; pointer-events: none; transition: top ${MOVE}; }
.fn-bar.hl .fn-val { display: block; }
.fn-mu { position: absolute; left: 0; right: 0; height: 0; border-top: 2px solid var(--fg); opacity: 0; transition: top ${MOVE}, opacity .4s; }
.fn-band { position: absolute; left: 0; right: 0; height: 0; opacity: 0; background: color-mix(in srgb, var(--c-norm) 16%, transparent); border-block: 1px dashed color-mix(in srgb, var(--c-norm) 85%, transparent); transition: top ${MOVE}, height ${MOVE}, opacity .4s; }
.fn-tag { position: absolute; left: ${BW + 10}px; translate: 0 -50%; font: 600 15px/1 var(--font-mono); color: var(--c-norm); padding: 5px 9px; border-radius: 7px; background: var(--surface); border: 1px solid var(--line); white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, .08); opacity: 0; transition: top ${MOVE}, opacity .4s; }
.fn-tag.on { opacity: 1; }
.fn-dims { position: absolute; width: ${BW}px; height: 16px; }
.fn-dims span { position: absolute; top: 0; translate: -50% 0; font: 500 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.fn-dims .fn-dimname { translate: 0 0; font-style: italic; }
.fn-strip { position: absolute; width: ${BW}px; height: ${GS}px; transition: opacity ${MOVE}; }
.fn-w { position: absolute; border-radius: 2px; box-shadow: 0 1px 3px rgba(0, 0, 0, .10); transition: top ${MOVE}, height ${MOVE}, background-color .4s; }
.fn-slab { margin: 0; }
.fn-side { position: absolute; width: ${SW}px; display: grid; gap: 10px; align-content: start; }
.fn-sidelab { text-align: left; margin: 0; }
.fn-side .katex-display { margin: 0; }
.fn-eq-main { font-size: 22px; color: var(--fg); overflow-x: auto; }
.fn-eq-live { font-size: 17px; padding: 10px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); box-shadow: 0 6px 16px rgba(0, 0, 0, .06); overflow-x: auto; }
.fn-checkrow { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.fn-checkrow[hidden] { display: none; }
.fn-checkrow small { font: 500 13px/1 var(--font-mono); color: var(--fg-3); }
.fn-cblock { position: absolute; width: ${SW}px; }
.fn-clab { font: 600 17px/1.2 var(--font-mono); color: var(--c-resid); white-space: nowrap; margin-bottom: 8px; }
.fn-cleg { display: flex; align-items: center; gap: 8px; height: 16px; margin-bottom: 8px; font: 500 15px/1 var(--font-mono); color: var(--c-norm); white-space: nowrap; opacity: 0; transition: opacity .4s; }
.fn-swatch { display: inline-block; width: 18px; border-top: 1.5px dashed var(--c-norm); }
.fn-chart { position: relative; width: ${SW}px; height: ${CH + 22}px; }
.fn-cbase { position: absolute; left: 0; right: 0; top: ${CH}px; border-top: 1px solid var(--line-strong); }
.fn-cbar { position: absolute; border-radius: 4px 4px 2px 2px; background: var(--c-resid); transition: top ${MOVE}, height ${MOVE}; }
.fn-cbar.mid { opacity: .55; }
.fn-cbar.last { outline: 2px solid var(--fg); outline-offset: 2px; }
.fn-cval { position: absolute; translate: -50% 0; z-index: 1; padding: 0 3px; border-radius: 4px; background: var(--surface); font: 600 13px/1.25 var(--font-mono); color: var(--fg-2); white-space: nowrap; transition: top ${MOVE}; }
.fn-cname { position: absolute; top: ${CH + 6}px; translate: -50% 0; font: 500 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.fn-cline { position: absolute; left: 0; right: 0; height: 0; border-top: 1.5px dashed var(--c-norm); opacity: 0; transition: top ${MOVE}, opacity .4s; }
.fn-note { position: absolute; width: 760px; display: grid; justify-items: center; gap: 16px; text-align: center; }
.fn-note[hidden] { display: none; }
.fn-note-eq { font-size: 28px; color: var(--fg); }
`);

// ------------------------------------------------------------ helpers (pure functions of the numbers)
const absMax = (a) => { let m = 0; for (const v of a) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v)); return m || 1; };
const norm2 = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));
const argAbsMax = (a) => { let j = 0; for (let i = 1; i < a.length; i++) if (Math.abs(a[i]) > Math.abs(a[j])) j = i; return j; };
const mapOf = (arr, H) => {
  let lo = 0, hi = 0;
  for (const v of arr) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = (hi - lo || 1) * 0.15;
  lo -= pad; hi += pad;
  const K = H / (hi - lo);
  return { K, Y: (v) => (hi - v) * K, y0: hi * K };
};
const fp = (v) => (v < 0 ? `(${num(v, 3)})` : num(v, 3));
const aligned = (lines) => `\\begin{aligned}${lines.map((l) => `&${l}`).join(' \\\\ ')}\\end{aligned}`;
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const sig2 = (v) => (Math.abs(v) < 1e-3 ? v.toExponential(1) : v.toPrecision(2));   // two significant figures

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    let alive = true, step = 0, frame = 0, cur = null, V = null, gen = 0, auto = null, driving = false;
    let stepsSet = false, checkShown = false, chartN = 0, gMap = null, bMap = null, gR = 1, bR = 1;
    let bars = [], gBars = [], bBars = [], cBars = [], cVals = [];
    const beatTimers = new Set();

    // ------------------------------------------------------------ shared kit parts
    const header = sceneHeader(st, { chips: [['after the last block', 'resid'], ['before the LM head', 'head']], title: 'Final norm' });
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'norm' });
    const cap = sceneCaption(st);

    // ------------------------------------------------------------ DOM: one absolute layer in design units
    const main = h('div', { class: 'fn-main' });
    st.add(main, { x: 0, y: 0 });
    const place = (node, x, y, anchor = 'tl') => {
      node.style.position = 'absolute';
      node.style.left = `${x}px`;
      node.style.top = `${y}px`;
      node.style.translate = ANCHOR[anchor].replace(',', ' ');
      main.append(node);
      return node;
    };

    // hero: the bars of one token (its stream, then the normalised vector)
    const heroLab = place(h('div', { class: 'k-label fn-herolab', style: { '--kind': 'var(--c-resid)' } }), BX, LAB_Y);
    const zero = h('div', { class: 'fn-zero' });
    const mu = h('div', { class: 'fn-mu' });
    const muTag = h('div', { class: 'fn-tag' });
    const band = h('div', { class: 'fn-band' });
    const bandTag = h('div', { class: 'fn-tag' });
    const heroEl = place(h('div', { class: 'fn-hero' }, zero, mu, band, muTag, bandTag), BX, BY);
    const dims = place(h('div', { class: 'fn-dims' }), BX, BY + BH + 8);

    // gain (γ) and shift (β) strips under the bars; their labels sit at the right end, like "dim"
    const gZero = h('div', { class: 'fn-zero' });
    const gStrip = place(h('div', { class: 'fn-strip' }, gZero), BX, G_Y);
    place(h('div', { class: 'k-label fn-slab', style: { '--kind': 'var(--c-norm)' } }, 'γ'), BX + BW + 12, G_Y + GS / 2, 'lc');
    const bZero = h('div', { class: 'fn-zero' });
    const bStrip = place(h('div', { class: 'fn-strip' }, bZero), BX, B_Y);
    const bLab = place(h('div', { class: 'k-label fn-slab', style: { '--kind': 'var(--c-norm)' } }, 'β'), BX + BW + 12, B_Y + GS / 2, 'lc');

    // side column: the step's name, the equation, the live numbers, the check
    const sideLab = h('div', { class: 'k-label fn-sidelab', style: { '--kind': 'var(--c-norm)' } });
    const eqMain = h('div', { class: 'fn-eq-main' });
    const eqLive = h('div', { class: 'fn-eq-live' });
    const check = h('div', { class: 'fn-checkrow', hidden: true });
    const sideEl = place(h('div', { class: 'fn-side' }, sideLab, eqMain, eqLive, check), SX, SY);

    // context chart: the stream norm after the embedding and after every layer. The √d reference
    // has its own legend row above the plot, so no label sits on a bar or on the dashed line.
    const cLab = h('div', { class: 'fn-clab' }, '‖x‖ after each layer');
    const cLegTxt = h('span');
    const cLegend = h('div', { class: 'fn-cleg' }, h('i', { class: 'fn-swatch' }), cLegTxt);
    const cWrap = h('div', { class: 'fn-chart' });
    const cLine = h('div', { class: 'fn-cline' });
    const chartEl = place(h('div', { class: 'fn-cblock' }, cLab, cLegend, cWrap), SX, CV_Y);

    // post-LN: no final norm to show (only reached if the node is valid but the trace is missing)
    const note = h('div', { class: 'fn-note', hidden: true },
      h('div', { class: 'fn-note-eq', html: texHTML('\\xres^{\\text{out}}_{L} = \\LN_{2}\\big(\\xres^{\\text{mid}} + \\MLP(\\xres^{\\text{mid}})\\big)', { display: true }) }),
      h('div', { class: 'k-caption' }, 'Post-LN blocks end in a norm, so the stream leaving the last block is already normalised. There is no final norm.'));
    st.add(note, { x: BX + BW / 2, y: 330, anchor: 'tc' });

    // ------------------------------------------------------------ maths of the current token
    function available() {
      const s = ctx.state, last = s.config.n_layer - 1;
      return !!(s.trace && s.model && s.trace.lnf && s.trace[`L${last}.resid_out`] && s.model.params['ln_f.weight']);
    }

    function compute() {
      const s = ctx.state, cfg = s.config, P = s.model.params;
      const last = cfg.n_layer - 1, T = s.ids.length;
      const q = Math.max(0, Math.min(T - 1, s.sel.token ?? T - 1));
      const xKey = `L${last}.resid_out`, X = s.trace[xKey], Y = s.trace.lnf;
      const d = X.shape[1], o = q * d, eps = cfg.norm_eps, isRms = cfg.norm === 'rmsnorm';
      const x = Array.from(X.data.subarray(o, o + d));
      const lnfRow = Array.from(Y.data.subarray(o, o + d));
      const g = Array.from(P['ln_f.weight'].data);
      const beta = !isRms && cfg.bias && P['ln_f.bias'] ? Array.from(P['ln_f.bias'].data) : null;
      const sum = x.reduce((a, v) => a + v, 0), mu0 = sum / d;
      const xc = x.map((v) => v - mu0), sumC = xc.reduce((a, v) => a + v, 0);
      const v2 = xc.reduce((a, v) => a + v * v, 0) / d;
      const ss = x.reduce((a, v) => a + v * v, 0), ms = ss / d;
      const sig = Math.sqrt(v2 + eps), rms = Math.sqrt(ms + eps);
      const xh = isRms ? x.map((v) => v / rms) : xc.map((v) => v / sig);
      const hh = xh.map((v, k) => g[k] * v + (beta ? beta[k] : 0));
      // the largest deviation from the traced lnf; the check compares that one dimension
      let maxDiff = 0, kd = 0;
      for (let k = 0; k < d; k++) { const e = Math.abs(hh[k] - lnfRow[k]); if (e > maxDiff) { maxDiff = e; kd = k; } }
      // ‖x̂‖² = d (1 − ε / s²) exactly, with s² = σ² or rms² (both include ε): ‖x̂‖ = √d only up to ε
      const ratio = 1 - eps / (isRms ? rms * rms : sig * sig);
      // the stream after the embedding and after every block, for this token
      const norms = ['emb.out', ...Array.from({ length: cfg.n_layer }, (_, i) => `L${i}.resid_out`)].map((key) => {
        const t = s.trace[key], dd = t.shape[1], oo = q * dd;
        let acc = 0;
        for (let k = 0; k < dd; k++) acc += t.data[oo + k] * t.data[oo + k];
        return Math.sqrt(acc);
      });
      return {
        q, d, eps, isRms, xKey, x, xc, xh, hh, lnfRow, g, beta, sum, sumC, mu: mu0, v2, ss, ms, sig, rms, ratio, maxDiff, kd,
        j: argAbsMax(x), norms, nX: norms[norms.length - 1], nEmb: norms[0], nXh: norm2(xh),
      };
    }

    // The views of one token: the stream, the beats of the normalisation, and the head's input.
    // Each view carries its geometry (vals, mu, band, gain, check, trace) and its text.
    // A view without a trace key (x′, x̂) has no hover card; its bars carry a native title instead.
    function views(c) {
      const q = c.q, d = c.d, j = c.j, tok = esc(ctx.label(q)), who = `token “${tok}” at position ${q}`;
      const cfg = ctx.state.config, hasB = !!c.beta;
      const r = c.nEmb > 1e-12 ? c.nX / c.nEmb : 1, emb = fmt(c.nEmb, 2);
      // with residuals on the stream carries the embedding forward, so its growth is the story;
      // with residuals off the stream is only the last MLP output, and no trend is claimed
      const trend = !cfg.residual ? ''
        : r > 1.05 ? `, up ${r.toFixed(1)}× from the embedding’s ${emb}`
          : r < 0.95 ? `, down from the embedding’s ${emb}` : `, close to the embedding’s ${emb}`;
      const src = cfg.residual ? 'the embedding plus every layer’s additions' : 'only the last MLP output, since residuals are off';
      const stream = {
        hero: `${who} · ${c.xKey}`, side: 'Last stream',
        main: `\\lVert \\xres_{t} \\rVert_2 = \\sqrt{\\sum_{c=1}^{${d}} \\xres_{t,c}^{2}}`,
        live: `\\lVert \\xres_{${q}} \\rVert_2 = \\sqrt{${num(c.ss, 2)}} = ${num(c.nX, 2)}`,
        cap: `This is the output of the last block for “<b>${tok}</b>”: ${src}. Its norm is <b>${fmt(c.nX, 2)}</b>${trend}.`,
        vals: c.x, trace: c.xKey,
      };

      const F = [];
      if (c.isRms) {
        F.push({
          hero: `${who} · ${c.xKey}`, side: 'Normalise · root mean square',
          main: `\\operatorname{rms}_{t} = \\sqrt{\\frac{1}{${d}}\\sum_{c} \\xres_{t,c}^{2} + \\varepsilon}`,
          live: aligned([`\\frac{1}{${d}}\\sum_{c} \\xres_{${q},c}^{2} = ${num(c.ms, 3)}`, `\\operatorname{rms}_{${q}} = \\sqrt{${num(c.ms, 3)} + \\varepsilon} = ${num(c.rms, 3)}`]),
          cap: `RMSNorm divides by the root mean square of the ${d} values. It takes no mean, so the band is centred on zero.`,
          vals: c.x, band: c.rms, bandTag: '±rms', trace: c.xKey,
        });
        F.push({
          hero: `${who} · x̂ = x ÷ rms`, side: 'Normalise · divide by rms',
          main: `\\hat x_{t,c} = \\frac{\\xres_{t,c}}{\\operatorname{rms}_{t}}`,
          live: `\\hat x_{${q},${j}} = ${fp(c.x[j])} / ${num(c.rms, 3)} = ${num(c.xh[j], 3)}`,
          cap: 'Dividing by the rms gives the values a root mean square of 1, up to ε. Whatever scale the stream had is removed.',
          vals: c.xh, band: 1, bandTag: '±1', tip: 'x̂',
        });
      } else {
        F.push({
          hero: `${who} · ${c.xKey}`, side: 'Normalise · mean',
          main: `\\mu_{t} = \\frac{1}{${d}}\\sum_{c=1}^{${d}} \\xres_{t,c}`,
          live: `\\mu_{${q}} = \\frac{${num(c.sum, 2)}}{${d}} = ${num(c.mu, 3)}`,
          cap: `The norm starts with the mean μ of this token’s ${d} values. The next beats subtract it, divide by the spread, then apply γ${hasB ? ' and β' : ''}.`,
          vals: c.x, mu: c.mu, trace: c.xKey,
        });
        F.push({
          hero: `${who} · x′ = x − μ`, side: 'Normalise · centre',
          main: `x'_{t,c} = \\xres_{t,c} - \\mu_{t}`,
          live: aligned([`x'_{${q},${j}} = ${fp(c.x[j])} - ${fp(c.mu)} = ${num(c.xc[j], 3)}`, `\\sum_{c} x'_{${q},c} = ${Math.abs(c.sumC) < 1e-9 ? '0' : num(c.sumC, 3)}`]),
          cap: 'Subtracting μ balances the values around zero. The band marks ±σ, the spread that the next beat divides by.',
          vals: c.xc, band: c.sig, bandTag: '±σ', tip: 'x′',
        });
        F.push({
          hero: `${who} · x̂ = x′ ÷ σ`, side: 'Normalise · scale',
          main: `\\hat x_{t,c} = \\frac{x'_{t,c}}{\\sigma_{t}}`,
          live: aligned([`\\sigma^{2}_{${q}} = ${num(c.v2, 3)}`, `\\sigma_{${q}} = \\sqrt{${num(c.v2, 3)} + \\varepsilon} = ${num(c.sig, 3)}`, `\\hat x_{${q},${j}} = ${fp(c.xc[j])} / ${num(c.sig, 3)} = ${num(c.xh[j], 3)}`]),
          cap: 'Dividing by σ gives the values a spread of 1, up to ε. Whatever scale the stream had is removed.',
          vals: c.xh, band: 1, bandTag: '±1', tip: 'x̂',
        });
      }
      // the last beat: gain and shift, checked against the traced lnf
      const gainLive = hasB
        ? aligned([`h_{${q},${j}} = \\gamma_{${j}}\\,\\hat x_{${q},${j}} + \\beta_{${j}}`, `= ${fp(c.g[j])} \\cdot ${fp(c.xh[j])} + ${fp(c.beta[j])} = ${num(c.hh[j], 3)}`])
        : aligned([`h_{${q},${j}} = \\gamma_{${j}}\\,\\hat x_{${q},${j}}`, `= ${fp(c.g[j])} \\cdot ${fp(c.xh[j])} = ${num(c.hh[j], 3)}`]);
      F.push({
        hero: `${who} · lnf`,
        side: hasB ? 'Normalise · gain and shift' : 'Normalise · gain',
        main: hasB ? `h_{t,c} = \\gamma_{c}\\,\\hat x_{t,c} + \\beta_{c}` : `h_{t,c} = \\gamma_{c}\\,\\hat x_{t,c}`,
        live: gainLive,
        cap: `The learned gain γ ${hasB ? 'and shift β set each dimension’s size and offset' : 'sets each dimension’s size'}. ${hasB ? '' : (c.isRms ? 'RMSNorm has no shift. ' : 'This model has no shift β (bias off). ')}The check confirms the result equals the traced lnf.`,
        vals: c.hh, gain: true, check: true, trace: 'lnf',
      });

      // the head reads lnf (the bars, traced); the √d fact is about x̂ before γ, so the text says so
      const s2 = c.isRms ? 'rms²' : 'σ²';
      const head = {
        hero: `${who} · lnf, the head’s input`, side: 'Ready for the head · norm before γ',
        main: `\\lVert \\hat x_{t} \\rVert_2 \\approx \\sqrt{${d}}`,
        live: `\\lVert \\hat x_{${q}} \\rVert_2 = \\sqrt{\\textstyle\\sum_{c} \\hat x_{${q},c}^{2}} = ${num(c.nXh, 3)}`,
        cap: `The head reads lnf, the bars. Before γ, this token’s normalised vector has norm √${d} up to ε: 1 − ε/${s2} = ${fmt(c.ratio, 3)}, so ‖x̂‖ = ${fmt(c.nXh, 3)}. The dashed line is that fixed scale.`,
        vals: c.hh, gain: true, check: true, trace: 'lnf', line: true,
      };
      return { stream, frames: F, head };
    }

    const viewAt = (i, f) => (i === 0 ? V.stream : i === 2 ? V.head : V.frames[Math.min(f, V.frames.length - 1)]);

    // ------------------------------------------------------------ DOM that depends on the data
    function buildBars() {
      const d = cur.d, colW = BW / d, bw = Math.max(1.5, Math.min(26, colW * 0.62));
      bars = Array.from({ length: d }, (_, k) => h('div', {
        class: 'fn-bar', style: { left: `${k * colW + (colW - bw) / 2}px`, width: `${bw}px` },
      }, h('span', { class: 'fn-val' })));
      heroEl.replaceChildren(zero, mu, band, muTag, bandTag, ...bars);
      // every 4th or 8th dimension, so the labels stay ≥ 36 units apart
      const stride = 2 ** Math.ceil(Math.log2(Math.max(1, Math.ceil(36 / colW))));
      const kids = [];
      for (let k = 0; k < d; k += stride) kids.push(h('span', { style: { left: `${k * colW + colW / 2}px` } }, String(k)));
      kids.push(h('span', { class: 'fn-dimname', style: { left: `${BW + 12}px` } }, 'dim'));
      dims.replaceChildren(...kids);
    }

    function buildStrips() {
      const d = cur.d, colW = BW / d, bw = Math.max(1.5, Math.min(26, colW * 0.62));
      const mk = () => Array.from({ length: d }, (_, k) => h('div', { class: 'fn-w', style: { left: `${k * colW + (colW - bw) / 2}px`, width: `${bw}px`, height: '0px' } }));
      gBars = mk();
      bBars = mk();
      gStrip.replaceChildren(gZero, ...gBars);
      bStrip.replaceChildren(bZero, ...bBars);
    }

    function syncStrips() {
      const c = cur;
      gMap = mapOf(c.g, GS);
      gR = absMax(c.g);
      gZero.style.top = `${gMap.y0}px`;
      bMap = c.beta ? mapOf(c.beta, GS) : null;
      bR = c.beta ? absMax(c.beta) : 1;
      if (bMap) bZero.style.top = `${bMap.y0}px`;
      bStrip.hidden = !bMap;
      bLab.hidden = !bMap;
    }

    // the context chart: one bar per stream (embedding, then every layer), the last one is the focus
    function buildChart() {
      const n = cur.norms.length, slot = SW / n, bw = Math.min(34, slot * 0.56);
      cBars = []; cVals = [];
      const kids = [h('div', { class: 'fn-cbase' })];
      cur.norms.forEach((_, k) => {
        const cx = k * slot + slot / 2;
        // the last stream is the focus; the layers in between are dimmed, the embedding keeps its colour
        const bar = h('div', { class: ['fn-cbar', k === n - 1 && 'last', k > 0 && k < n - 1 && 'mid'], style: { left: `${cx - bw / 2}px`, width: `${bw}px`, background: k === 0 ? 'var(--c-embed)' : null } });
        const val = h('div', { class: 'fn-cval', style: { left: `${cx}px` } });
        const name = h('div', { class: 'fn-cname', style: { left: `${cx}px` } }, k === 0 ? 'emb' : `L${k - 1}`);
        kids.push(bar, val, name);
        cBars.push(bar); cVals.push(val);
      });
      kids.push(cLine);
      cWrap.replaceChildren(...kids);
      chartN = n;
    }

    function paintChartValues(animate) {
      const c = cur, d = c.d, sq = Math.sqrt(d), plotH = CH - 22;
      const top = Math.max(...c.norms, 1e-9);
      // the reference √d sets the scale, unless it dwarfs the streams (random weights): then the
      // streams get the full height and the reference is drawn at the top edge, marked as off scale
      const clip = sq > 3 * top;
      const maxV = (clip ? top : Math.max(top, sq)) * 1.08;
      const yL = clip ? 0 : CH - (sq / maxV) * plotH;
      // two significant figures while every norm is below 1, so 0.14 and 0.20 do not both read 0.1 and 0.2
      const lab = top < 1 ? sig2 : (v) => v.toFixed(1);
      still(animate, () => {
        c.norms.forEach((nv, k) => {
          const hpx = Math.max(2, (nv / maxV) * plotH);
          cBars[k].style.top = `${CH - hpx}px`;
          cBars[k].style.height = `${hpx}px`;
          cVals[k].style.top = `${CH - hpx - 20}px`;
          cVals[k].textContent = lab(nv);
        });
        cLine.style.top = `${yL}px`;
        cLegTxt.textContent = `${clip ? '↑ ' : ''}√${d} ≈ ${sq.toFixed(2)} after the norm`;
      });
    }

    // ------------------------------------------------------------ painting one view
    // reduced motion and first paint: write the end state without transitions
    function still(animate, fn) {
      if (animate && !reducedMotion()) { fn(); return; }
      main.classList.add('fn-still');
      fn();
      void main.offsetWidth;
      main.classList.remove('fn-still');
    }

    function paintHero(v, animate) {
      const vals = v.vals, d = cur.d, j = cur.j, q = cur.q;
      let lo = 0, hi = 0;
      for (let k = 0; k < d; k++) { lo = Math.min(lo, vals[k]); hi = Math.max(hi, vals[k]); }
      if (v.mu != null) { lo = Math.min(lo, v.mu); hi = Math.max(hi, v.mu); }
      if (v.band != null) { lo = Math.min(lo, -v.band); hi = Math.max(hi, v.band); }
      const pad = (hi - lo || 1) * 0.08;
      lo -= pad; hi += pad;
      const K = BH / (hi - lo), Y = (x) => (hi - x) * K, y0 = Y(0), R = absMax(vals);
      still(animate, () => {
        zero.style.top = `${y0}px`;
        if (v.mu != null) {
          mu.style.top = `${Y(v.mu)}px`; mu.style.opacity = '1';
          muTag.textContent = `μ = ${fmt(v.mu, 3)}`; muTag.style.top = `${Y(v.mu)}px`; muTag.classList.add('on');
        } else { mu.style.opacity = '0'; muTag.classList.remove('on'); }
        if (v.band != null) {
          band.style.top = `${Y(v.band)}px`; band.style.height = `${Y(-v.band) - Y(v.band)}px`; band.style.opacity = '1';
          bandTag.textContent = v.bandTag; bandTag.style.top = `${Y(v.band)}px`; bandTag.classList.add('on');
        } else { band.style.opacity = '0'; bandTag.classList.remove('on'); }
        for (let k = 0; k < d; k++) {
          const x = vals[k], b = bars[k], hp = Math.abs(x) * K;
          b.style.top = `${x >= 0 ? Y(x) : y0}px`;
          b.style.height = `${Math.max(1.5, hp)}px`;
          b.style.background = diverging(x / R);
          // the cell names the token on screen: the hover card and the maths read it, so it is set on every paint
          b.dataset.cell = `${q},${k},0`;
          b.title = v.trace ? '' : `${v.tip}[${k}] = ${fmt(x, 3)}`;
          b.classList.toggle('hl', k === j);
          if (k === j) { // the one number followed through every beat
            const lab = b.firstChild;
            lab.textContent = fmt(x, 2);
            lab.style.top = x >= 0 ? '-22px' : `${hp + 6}px`;
          }
        }
      });
    }

    // γ and β grow in when the gain beat applies them, and sit faint before that
    function paintStrips(v, animate) {
      const c = cur, on = !!v.gain;
      still(animate, () => {
        for (let k = 0; k < c.d; k++) {
          const gv = c.g[k], gb = gBars[k];
          gb.style.top = `${gv >= 0 ? gMap.Y(gv) : gMap.y0}px`;
          gb.style.height = on ? `${Math.max(1.5, Math.abs(gv) * gMap.K)}px` : '0px';
          gb.style.background = diverging(gv / gR);
          gb.title = `γ[${k}] = ${fmt(gv, 3)}`;
          if (bMap) {
            const bv = c.beta[k], bb = bBars[k];
            bb.style.top = `${bv >= 0 ? bMap.Y(bv) : bMap.y0}px`;
            bb.style.height = on ? `${Math.max(1.5, Math.abs(bv) * bMap.K)}px` : '0px';
            bb.style.background = diverging(bv / bR);
            bb.title = `β[${k}] = ${fmt(bv, 3)}`;
          }
        }
        gStrip.style.opacity = on ? '1' : '0.35';
        bStrip.style.opacity = on ? '1' : '0.35';
      });
    }

    function paintText(v, animate) {
      heroLab.innerHTML = v.hero;
      // the stream is coloured as the residual stream, the normalised values as the norm
      heroLab.style.setProperty('--kind', v.trace === cur.xKey ? 'var(--c-resid)' : 'var(--c-norm)');
      sideLab.textContent = v.side;
      // text style keeps sums and fractions compact enough for the side column
      eqMain.innerHTML = texHTML(`\\textstyle ${v.main}`, { display: true });
      eqLive.innerHTML = texHTML(`\\textstyle ${v.live}`, { display: true });
      const want = !!v.check;
      // the badge compares the dimension with the largest deviation, so its tooltip names that pair
      if (want) check.replaceChildren(
        checkBadge(cur.hh[cur.kd], cur.lnfRow[cur.kd], { label: 'equals the traced lnf', tol: 1e-9 }),
        h('small', null, `max |Δ| = ${fmt(cur.maxDiff, 2)}`));
      check.hidden = !want;
      if (want && !checkShown && animate) pulse(check, { scale: 1.03 });
      checkShown = want;
      if (v.trace) heroEl.dataset.trace = v.trace; else delete heroEl.dataset.trace;
      cap.set(v.cap);
    }

    function paintChartFlags(i, animate) {
      const on = i === 2;
      still(animate, () => { cLine.style.opacity = on ? '1' : '0'; cLegend.style.opacity = on ? '1' : '0'; });
    }

    function show(i, f, animate) {
      if (!cur) return;
      const v = viewAt(i, f);
      paintHero(v, animate);
      paintStrips(v, animate);
      paintChartFlags(i, animate);
      paintText(v, animate);
    }

    // ------------------------------------------------------------ steps and beats
    function clearBeats() { gen++; beatTimers.forEach(clearTimeout); beatTimers.clear(); }
    function later(ms, fn) {
      const g = gen;
      const t = setTimeout(() => { beatTimers.delete(t); if (alive && g === gen) fn(); }, ms);
      beatTimers.add(t);
    }

    function goStep(i, animate = true) {
      clearBeats();
      step = Math.max(0, Math.min(STEPS.length - 1, i));
      if (!cur) return;
      const anim = animate && !reducedMotion();
      if (step !== 1) { frame = 0; show(step, 0, anim); return; }
      const n = V.frames.length;
      // reduced motion: the end state of the step, with its equation and check
      if (!anim) { frame = n - 1; show(1, frame, false); return; }
      frame = 0;
      show(1, 0, true);
      // the beats of the normalisation, one BEAT apart
      for (let k = 1; k < n; k++) later(k * BEAT, () => { frame = k; show(1, k, true); });
    }

    function refresh(animate) {
      if (!cur) return;
      if (step === 1) show(1, Math.min(frame, V.frames.length - 1), animate);
      else show(step, 0, animate);
    }

    function setStep(i) {
      if (!driving) cancelAuto();
      goStep(i);
    }

    // auto-advance the first time the scene is shown: the beats of step 2 get their own time
    function cancelAuto() { clearTimeout(auto); auto = null; }
    function startAuto() {
      cancelAuto();
      const run = (k) => {
        if (!alive || k >= STEPS.length || !cur) return;
        driving = true; ctx.setStep(k); driving = false;
        auto = setTimeout(() => run(k + 1), k === 1 ? (V.frames.length + 0.5) * BEAT : MOTION.step);
      };
      auto = setTimeout(() => run(1), 1200);
    }

    // ------------------------------------------------------------ build (mount, new forward pass, new token)
    function build(animate) {
      if (!alive) return;
      // the atlas leaves this node when the config invalidates it (post-LN: no final norm). It does
      // so from the config event, while this scene still hears the forward pass, so paint nothing here.
      if (!validFor(ctx.node.id, ctx.state.config)) {
        cur = null; V = null;
        cancelAuto(); clearBeats();
        main.hidden = true; note.hidden = true; picker.el.style.display = 'none';
        cap.set('');                                  // the caption described bars that are now hidden
        if (stepsSet) { stepsSet = false; ctx.setSteps([], () => {}); }
        return;
      }
      const ok = available();
      main.hidden = !ok;
      note.hidden = ok;
      picker.el.style.display = ok ? '' : 'none';   // .k-picker-wrap sets display, so [hidden] would not hide it
      if (!ok) {
        cur = null; V = null;
        cancelAuto(); clearBeats();
        if (stepsSet) { stepsSet = false; ctx.setSteps([], () => {}); }
        header.title.textContent = ctx.node.title;
        header.setSub('Post-LN model: the norm sits inside each block, so there is no final norm before the head.');
        cap.set('Post-LN blocks normalise the sum of the residual and the sub-layer output. The last block’s output is already normalised, so there is nothing left to do.');
        return;
      }
      cur = compute();
      V = views(cur);
      if (!stepsSet) { stepsSet = true; ctx.setSteps(STEPS, setStep); }
      if (bars.length !== cur.d) buildBars();
      if (gBars.length !== cur.d) buildStrips();
      if (chartN !== cur.norms.length) buildChart();
      syncStrips();
      header.setSub(`Token “<b>${esc(ctx.label(cur.q))}</b>” at position ${cur.q}: the output of the last block, normalised once more before the LM head.`);
      refresh(animate);
      paintChartValues(animate);
    }

    // ------------------------------------------------------------ wiring
    const offs = [
      ctx.on('trace', () => build(false)),
      ctx.on('select', () => { cancelAuto(); build(true); }),
    ];
    const offTheme = onThemeChange(() => { if (cur) refresh(false); });
    el.addEventListener('pointerdown', cancelAuto);

    build(false);
    if (available() && validFor(ctx.node.id, ctx.state.config)) {
      appear([header.el, picker.el, heroEl, sideEl, chartEl, cap.el], { stagger: MOTION.stagger });
      if (!reducedMotion()) startAuto();
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

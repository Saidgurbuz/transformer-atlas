// Block: one Transformer block on the real forward pass. The residual stream is a river; attention and the
// MLP are detours that read it, compute, and add their output back. The wiring follows cfg.norm_placement:
//   pre:  x_mid = x + Attn(Norm(x)),   x_out = x_mid + MLP(Norm(x_mid))   (a norm sits on each leg)
//   post: x_mid = Norm(x + Attn(x)),   x_out = Norm(x_mid + MLP(x_mid))   (a norm sits on the river after each add)
// With cfg.residual off the river is broken and each sub-layer replaces the stream (no add, no plus sign).
// Every number comes from ctx.state.trace. Each sum is recomputed from the traced vectors and checked against
// the traced sum (checkBadge). The norm is named from cfg.norm (LN or RMSNorm): one word in the chip, the captions and the pill.
import { stage2d, card, op, vec, chip, appear, pulse, sceneHeader, sceneCaption, tokenPicker, checkBadge, useStyle, ZONES, MOTION } from '../kit.js';
import { h, s, texHTML, num, reducedMotion, clamp } from '../../ui/index.js';

const STEPS = ['Stream in', 'Attention reads & writes', 'MLP reads & writes', 'Stream out'];
const TITLE = 'One block';
const TEXT_OUT = 140, TEXT_IN = 260;   // the equation, its check and the caption fade out, then in (as lm-head.js)
const SUB_ON = 'The residual stream is a river. Attention and the MLP each read it, compute, and add their output back.';
const SUB_OFF = 'The residual stream is broken. Attention and the MLP each read it, compute, and replace the stream.';

// Design units (1600 x 900). The river spans the hero band; the detours rise above it, the readouts sit below.
const X0 = 110, X1 = 1490;           // river ends (the hero band)
const RY = 460;                      // river centre line
const YC = 312;                      // detour lane: the sub-layer cards sit on it
const F1 = 280, J1 = 700;            // attention: fork and join on the river
const F2 = 880, J2 = 1300;           // MLP: fork and join
const JR = 28;                       // join radius: the operator is 56 units across
const RING = JR + 8;                 // join radius including its halo
const RAD = 24;                      // detour corner radius
const CARD_W = 280;
const LEG_Y = (YC + RY) / 2;         // pre-LN norms sit on the legs
const READ_Y = RY - 36;              // arrowhead where the stream is read (fork)
const CHIP_Y = 388;                  // write chips beside the return legs
const GAP_Y = RY - 40;               // "no skip" labels above a broken river
const ST_Y = 512;                    // readout stations under the river
const ST = { in: 226, mid: (J1 + F2) / 2, out: 1390 };   // in: its strip starts on the text edge (x 110)
// post-LN norms sit on the river between a join and the next fork (or the river end); the last one keeps air from the arrow
const LN = { 1: (J1 + RING + F2 - 6) / 2, 2: (J2 + RING + X1 - 14) / 2 - 6 };
const EQ_Y = 642;                    // equation band: 40 units under the station readouts (they end near 602)
const CHECK_X = 1010;
const ACT = { gelu: 'GELU', relu: 'ReLU', swiglu: 'SwiGLU' };
const MASK = { causal: 'causal mask', none: 'no mask', sliding: 'sliding window mask', prefix: 'prefix mask' };
const headWord = (n) => (n === 1 ? 'head' : 'heads');
const normWord = (cfg) => (cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LN');   // the same two names as residual.js
const X = '\\xres';                  // the residual stream symbol (coloured macro)

// The live line rounds every number to three decimals. A sum shows '=' only when its printed operands add up to
// the printed result; otherwise it shows '≈', so the reader is never shown arithmetic that does not add up.
const r3 = (v) => Number(v.toFixed(3));
const n3 = (v) => (r3(v) === 0 ? '0.000' : num(v, 3));    // a value that prints as zero never prints as -0.000
const sumLine = (lhs, x0, y0, z0) => {
  const adds = r3(r3(x0) + r3(y0)) === r3(z0);
  return `${lhs} = ${n3(x0)} ${r3(y0) < 0 ? '-' : '+'} ${n3(Math.abs(y0))} ${adds ? '=' : '\\approx'} ${n3(z0)}`;
};

useStyle('block-scene', `
.bl-wrap { position: absolute; left: 0; top: 0; width: 1600px; height: 900px; pointer-events: none; }
.bl-wrap > * { pointer-events: auto; }
.bl-wrap > .bl-svg { pointer-events: none; }
.bl-svg { position: absolute; left: 0; top: 0; width: 1600px; height: 900px; overflow: visible; }
.bl-river-base { fill: none; stroke: var(--c-resid); stroke-opacity: .2; stroke-width: 40; }
.bl-river-line { fill: none; stroke: var(--c-resid); stroke-opacity: .45; stroke-width: 1.5; }
.bl-river-beads { fill: none; stroke: var(--c-resid); stroke-width: 9; stroke-linecap: round; stroke-dasharray: 1 22; opacity: .6; }
.bl-wrap[data-step="1"] .bl-river-beads, .bl-wrap[data-step="2"] .bl-river-beads { animation: bl-flow 1.6s linear infinite; }   /* the stream moves only while a sub-layer reads it */
.bl-arrow-river { fill: var(--c-resid); opacity: .6; }
.bl-gap { fill: none; stroke: var(--fg-3); stroke-width: 2; stroke-dasharray: 4 6; opacity: .8; }
.bl-det-ghost { fill: none; stroke-width: 2; stroke-linecap: round; stroke-dasharray: 2 8; opacity: .4; }
.bl-det-solid { fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: var(--len, 900) var(--len, 900); stroke-dashoffset: var(--len, 900); opacity: 0; transition: stroke-dashoffset 600ms cubic-bezier(.2,.7,.2,1), opacity .5s; }
.bl-beads { fill: none; stroke-width: 6; stroke-linecap: round; stroke-dasharray: 1 17; opacity: 0; transition: opacity .6s; animation: bl-flow-b 1.2s linear infinite; }
.bl-det-tip { opacity: 0; transition: opacity .5s; }
.bl-fork { fill: var(--bg); stroke-width: 2.5; }
.bl-det-ghost.bl-attn, .bl-det-solid.bl-attn, .bl-beads.bl-attn { stroke: var(--c-attn); }
.bl-det-ghost.bl-mlp, .bl-det-solid.bl-mlp, .bl-beads.bl-mlp { stroke: var(--c-mlp); }
.bl-det-tip.bl-attn { fill: var(--c-attn); }
.bl-det-tip.bl-mlp { fill: var(--c-mlp); }
.bl-fork.bl-attn { stroke: var(--c-attn); }
.bl-fork.bl-mlp { stroke: var(--c-mlp); }
/* the steps: 1 attention reads and writes, 2 the MLP reads and writes, 3 the stream out (both detours step back) */
.bl-wrap[data-step="1"] .bl-det-solid.bl-attn { opacity: 1; stroke-dashoffset: 0; }
.bl-wrap[data-step="2"] .bl-det-solid.bl-attn { opacity: .6; stroke-dashoffset: 0; }
.bl-wrap[data-step="2"] .bl-det-solid.bl-mlp { opacity: 1; stroke-dashoffset: 0; }
.bl-wrap[data-step="3"] .bl-det-solid { opacity: .35; stroke-dashoffset: 0; }
.bl-wrap[data-step="1"] .bl-det-tip.bl-attn, .bl-wrap[data-step="2"] .bl-det-tip.bl-attn,
.bl-wrap[data-step="2"] .bl-det-tip.bl-mlp { opacity: 1; }
.bl-wrap[data-step="3"] .bl-det-tip { opacity: .35; }
.bl-wrap[data-step="1"] .bl-beads.bl-attn, .bl-wrap[data-step="2"] .bl-beads.bl-mlp { opacity: .95; }
@keyframes bl-flow { to { stroke-dashoffset: -23px; } }
@keyframes bl-flow-b { to { stroke-dashoffset: -18px; } }
.bl-wrap[data-res="on"] .bl-res-off, .bl-wrap[data-res="off"] .bl-res-on { display: none; }
.bl-wrap[data-res="off"] .bl-join { border-style: dashed; }

.bl-card { background: var(--surface); transition: opacity .6s cubic-bezier(.2,.7,.2,1), box-shadow .45s cubic-bezier(.2,.7,.2,1), transform .25s var(--ease); }
.bl-wrap[data-step="0"] .bl-card { opacity: .55; }
.bl-wrap[data-step="1"] .bl-card-mlp { opacity: .55; }
.bl-wrap[data-step="2"] .bl-card-attn { opacity: .7; }
.bl-wrap[data-step="3"] .bl-card { opacity: .55; }
.bl-wrap[data-step="1"] .bl-card-attn, .bl-wrap[data-step="2"] .bl-card-mlp { box-shadow: 0 0 0 2px color-mix(in srgb, var(--kind) 70%, transparent), 0 20px 44px rgba(0, 0, 0, .2); }

.bl-pill { position: absolute; width: 112px; height: 28px; border-radius: 999px; background: var(--surface); border: 1.5px solid var(--c-norm); color: var(--c-norm); font: 600 15px/25px var(--font-mono); text-align: center; white-space: nowrap; box-shadow: var(--shadow-1); transition: box-shadow .4s, color .4s; }
.bl-pill[data-node] { cursor: zoom-in; }
.bl-pill[data-node]:hover { box-shadow: 0 0 0 4px color-mix(in srgb, var(--c-norm) 35%, transparent), var(--shadow-1); }
.bl-pill.hot { color: var(--fg); box-shadow: 0 0 0 4px color-mix(in srgb, var(--c-norm) 45%, transparent), var(--shadow-1); }

.bl-write { position: absolute; display: inline-flex; align-items: center; gap: 8px; height: 30px; padding: 0 12px; border-radius: 999px; font: 600 15px/1 var(--font-mono); white-space: nowrap; color: var(--kind); background: color-mix(in srgb, var(--kind) 14%, var(--surface)); border: 1px solid color-mix(in srgb, var(--kind) 45%, transparent); opacity: 0; pointer-events: none; transition: opacity .6s cubic-bezier(.2,.7,.2,1); }
.bl-write.bl-attn { --kind: var(--c-attn); }
.bl-write.bl-mlp { --kind: var(--c-mlp); }
.bl-write.on { opacity: 1; pointer-events: auto; }
.bl-wrap[data-step="3"] .bl-write.on { opacity: .5; }
.bl-w-l { color: var(--fg-3); font-weight: 500; }
.bl-w-v { color: var(--fg); font-weight: 650; }
.bl-w-sym .katex { font-size: 1.12em; }

.bl-st { position: absolute; width: 248px; text-align: center; }
.bl-st-label { height: 24px; font: 500 15px/24px var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.bl-st-label .katex { color: var(--c-resid); font-size: 1.12em; }
.bl-st-note { color: var(--fg-3); font-size: 15px; margin-left: 8px; }
.bl-st-box { position: relative; height: 36px; display: flex; align-items: center; justify-content: center; }
.bl-st-box .k-vec { transition: opacity .6s cubic-bezier(.2,.7,.2,1); }
.bl-st.pending .bl-st-box .k-vec { opacity: 0; }
.bl-st-ghost { position: absolute; left: 6px; right: 6px; top: 2px; bottom: 2px; border-radius: 10px; border: 1.5px dashed var(--line-strong); opacity: 0; transition: opacity .6s; }
.bl-st.pending .bl-st-ghost { opacity: 1; }
.bl-st-norm { height: 24px; margin-top: 6px; font: 500 15px/24px var(--font-mono); color: var(--fg-3); white-space: nowrap; transition: opacity .6s; }
.bl-st-norm b { color: var(--fg); font-weight: 650; }
.bl-st.pending .bl-st-norm { opacity: 0; }
.bl-st.hot .bl-st-box .k-cells { box-shadow: 0 0 0 3px color-mix(in srgb, var(--c-resid) 40%, transparent), 0 6px 16px rgba(0, 0, 0, .12); }
.bl-wrap[data-step="3"] .bl-st-out .bl-st-box .k-cells { box-shadow: 0 0 0 3px var(--accent), 0 6px 16px rgba(0, 0, 0, .12); }   /* the one accent: the stream out */

.bl-readout { fill: none; stroke: var(--c-resid); stroke-width: 1.5; stroke-linecap: round; opacity: .6; }
.bl-readout-tip { fill: var(--c-resid); opacity: .6; }
.bl-readout, .bl-readout-tip { transition: opacity .6s cubic-bezier(.2,.7,.2,1); }
.bl-readout.pending, .bl-readout-tip.pending { opacity: .2; }   /* a readout waits for its station */
.bl-river-label { font: 600 15px/1 var(--font-mono); color: var(--c-resid); white-space: nowrap; }
.bl-gap-label { font: 600 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }

.bl-eq { width: 860px; display: grid; gap: 10px; }
.bl-eq-label { font: 650 15px/1.2 var(--font-mono); color: var(--c-resid); }
.bl-eq-main { font-size: 24px; color: var(--fg); }
.bl-eq-main .katex-display { margin: 0; }
.bl-eq-main .katex-display > .katex { text-align: left; }
.bl-eq-live { font-size: 18px; color: var(--fg-2); }
.bl-checks { width: 480px; display: grid; gap: 14px; font: 500 15px/1.4 var(--font-mono); color: var(--fg-3); }
.bl-check-row { display: grid; gap: 4px; }
.bl-check-line { display: flex; align-items: center; gap: 12px; flex-wrap: nowrap; color: var(--fg); }
.bl-check-line .k-check { flex: none; }
.bl-dim { color: var(--fg-3); font-size: 15px; }

@media (prefers-reduced-motion: reduce) {
  .bl-river-beads, .bl-beads, .bl-wrap[data-step="1"] .bl-river-beads, .bl-wrap[data-step="2"] .bl-river-beads { animation: none; }
  .bl-det-solid, .bl-card, .bl-st-box .k-vec, .bl-st-ghost, .bl-st-norm, .bl-write, .bl-readout, .bl-readout-tip { transition: none; }
}
`);

const norm = (v) => Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
const maxDiff = (x, y) => { let e = 0; for (let i = 0; i < x.length; i++) e = Math.max(e, Math.abs(x[i] - y[i])); return e; };
const pathLen = (p) => { try { return Math.ceil(p.getTotalLength()); } catch { return 1000; } };
const detourD = (f, j) => `M${f},${RY} V${YC + RAD} Q${f},${YC} ${f + RAD},${YC} H${j - RAD} Q${j},${YC} ${j},${YC + RAD} V${RY - JR}`;
const riverPieces = (x0, x1) => ['bl-river-base', 'bl-river-line', 'bl-river-beads'].map((cls) => s('path', { d: `M${x0},${RY} H${x1}`, class: cls }));
const errText = (e) => (e === 0 ? '0' : e.toExponential(1));
// Norms to two decimals; a tiny norm reads "<0.01" rather than exponent notation.
const nfmt = (v) => (v > 0 && v < 0.005 ? '<0.01' : v.toFixed(2));
const nHTML = (v) => nfmt(v).replace('<', '&lt;');     // for text that is set through innerHTML

function setNum(el, v) { el._gen = (el._gen || 0) + 1; el.textContent = nfmt(v); }

function tweenNum(el, to, ms = MOTION.move) {
  el._gen = (el._gen || 0) + 1;
  const gen = el._gen;
  if (reducedMotion()) { el.textContent = nfmt(to); return; }
  const t0 = performance.now();
  const frame = (now) => {
    if (el._gen !== gen || !el.isConnected) return;
    const p = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - p, 3);
    el.textContent = nfmt(to * e);
    if (p < 1) requestAnimationFrame(frame);
  };
  el.textContent = nfmt(0);
  requestAnimationFrame(frame);
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const L = ctx.params.l;
    let step = 0, P = null, D = null, structKey = '', introDone = false, curQ = 0, auto = null;
    const vecs = [];

    // The river and everything on it. Rebuilt when the wiring or the sequence length changes.
    const wrap = h('div', { class: 'bl-wrap' });
    st.add(wrap, { x: 0, y: 0 });

    // ---- shared parts (kit): header, token picker, caption
    const header = sceneHeader(st, { chips: [[`Block ${L}`, 'resid']], title: TITLE, sub: SUB_ON });
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'resid' });
    const cap = sceneCaption(st);

    // ---- equation and checks: one step at a time, under the stations
    const eqLabel = h('div', { class: 'bl-eq-label' });
    const eqMain = h('div', { class: 'bl-eq-main' });
    const eqLive = h('div', { class: 'bl-eq-live' });
    const eqPanel = h('div', { class: 'bl-eq' }, eqLabel, eqMain, eqLive);
    st.add(eqPanel, { x: ZONES.caption.x, y: EQ_Y });
    const checksPanel = h('div', { class: 'bl-checks' });
    st.add(checksPanel, { x: CHECK_X, y: EQ_Y + 4 });

    const put = (node, x, y, anchor = 'tl') => {
      node.style.position = 'absolute';
      node.style.left = `${x}px`;
      node.style.top = `${y}px`;
      node.style.translate = { tl: '0 0', c: '-50% -50%', rc: '-100% -50%', tr: '-100% 0' }[anchor];
      wrap.append(node);
      return node;
    };

    const structureKey = () => {
      const c = ctx.state.config;
      return JSON.stringify([c.norm_placement, !!c.residual, c.n_layer, c.n_head, c.n_kv_head, c.d_model, c.d_ff, c.activation, c.norm, c.mask, ctx.state.ids.length]);
    };

    // Everything the scene shows for the selected token, read from the trace.
    function compute() {
      const S = ctx.state, c = S.config, T = S.ids.length;
      if (!S.trace || !T) return null;
      const q = clamp(S.sel.token ?? 0, 0, T - 1);
      const get = (key) => {
        const t = S.trace[key];
        if (!t || t.shape.length !== 2) return null;
        const dd = t.shape[1];
        return Array.from(t.data.subarray(q * dd, (q + 1) * dd));
      };
      const xin = get(L === 0 ? 'emb.out' : `L${L - 1}.resid_out`);
      const a = get(`L${L}.attn.out`), mid = get(`L${L}.resid_mid`), m = get(`L${L}.mlp.out`), out = get(`L${L}.resid_out`);
      if (!xin || !a || !mid || !m || !out) return null;
      const pre = c.norm_placement === 'pre', res = !!c.residual;
      // the traced sums before each norm (post-LN); in pre-LN the stream itself is the sum
      const add1 = pre ? mid : get(`L${L}.add1`), add2 = pre ? out : get(`L${L}.add2`);
      if (!add1 || !add2) return null;
      // the adds, recomputed from the traced inputs
      const sumA = res ? xin.map((v, i) => v + a[i]) : a;
      const sumM = res ? mid.map((v, i) => v + m[i]) : m;
      const n = { in: norm(xin), mid: norm(mid), out: norm(out), a: norm(a), m: norm(m) };
      return { q, pre, res, xin, a, mid, m, out, add1, add2, sumA, sumM, n, d: xin.length };
    }

    // A station strip stamps data-cell (token and element index) when it is built, so a station is rebuilt
    // when the token changes. This leaves the kit's elision rule to the kit.
    const stationVec = (traceKey, values, range) => vec(values, {
      orientation: 'h', cell: 12, gap: 1, kind: 'resid', range, trace: traceKey, cellOf: (i) => `${curQ},${i},0`,
    });

    // Draw the whole river for the current config and token. Called on mount and when the wiring changes.
    function build() {
      const S = ctx.state, c = S.config;
      const pre = c.norm_placement === 'pre', res = !!c.residual;
      const T = S.ids.length, q = clamp(S.sel.token ?? 0, 0, Math.max(0, T - 1)), d = c.d_model;
      const nw = normWord(c);
      const attnSub = c.n_kv_head < c.n_head
        ? `${c.n_head} query ${headWord(c.n_head)} · ${c.n_kv_head} KV ${headWord(c.n_kv_head)}`
        : `${c.n_head} ${headWord(c.n_head)} · ${MASK[c.mask] ?? `${c.mask} mask`}`;
      const mlpSub = `${ACT[c.activation] ?? c.activation} · ${c.d_ff} hidden units`;
      const inKey = L === 0 ? 'emb.out' : `L${L - 1}.resid_out`;
      curQ = q;

      // the header follows the config: chips and the context line
      header.el.querySelector('.k-eyebrow')?.replaceChildren(
        chip(`Block ${L}`, 'resid'), chip(`${pre ? 'Pre' : 'Post'}-${nw}`, 'norm'), chip(res ? 'Residual on' : 'Residual off', 'resid'));
      header.setSub(res ? SUB_ON : SUB_OFF);

      vecs.splice(0).forEach((v) => v.destroy?.());
      wrap.replaceChildren();
      wrap.dataset.res = res ? 'on' : 'off';
      wrap.dataset.step = String(step);
      const intro = [];

      // ---- the river: one unbroken band, or pieces with gaps when the residual is off
      const svg = s('svg', { class: 'bl-svg', viewBox: '0 0 1600 900', width: 1600, height: 900 });
      const gOn = s('g', { class: 'bl-res-on' }, ...riverPieces(X0, X1));
      const gOff = s('g', { class: 'bl-res-off' },
        ...riverPieces(X0, F1), ...riverPieces(J1, F2), ...riverPieces(J2, X1),
        s('path', { d: `M${F1},${RY} H${J1}`, class: 'bl-gap' }), s('path', { d: `M${F2},${RY} H${J2}`, class: 'bl-gap' }));
      svg.append(gOn, gOff, s('path', { d: `M${X1 - 14},${RY - 9} L${X1},${RY} L${X1 - 14},${RY + 9} Z`, class: 'bl-arrow-river' }));

      // ---- the two detours: a ghost guide, a solid path that draws in, flowing beads, arrowheads
      const detour = (f, j, kind) => {
        const dd = detourD(f, j);
        const solid = s('path', { d: dd, class: `bl-det-solid bl-${kind}` });
        svg.append(
          s('path', { d: dd, class: `bl-det-ghost bl-${kind}` }), solid,
          s('path', { d: dd, class: `bl-beads bl-${kind}` }),
          s('path', { d: `M${j - 6},${RY - JR - 18} L${j + 6},${RY - JR - 18} L${j},${RY - JR - 8} Z`, class: `bl-det-tip bl-${kind}` }),
          s('path', { d: `M${f - 6},${READ_Y + 10} L${f + 6},${READ_Y + 10} L${f},${READ_Y} Z`, class: `bl-det-tip bl-${kind}` }),
          s('circle', { cx: f, cy: RY, r: 6, class: `bl-fork bl-${kind}` }));
        return solid;
      };
      const solidA = detour(F1, J1, 'attn'), solidM = detour(F2, J2, 'mlp');
      wrap.append(svg);

      // ---- sub-layer cards on the detour lanes: zoom targets
      const cAttn = put(card({ title: 'Attention', sub: attnSub, kind: 'attn', node: `attention.${L}`, w: CARD_W }), (F1 + J1) / 2, YC, 'c');
      cAttn.classList.add('bl-card', 'bl-card-attn');
      const cMlp = put(card({ title: 'MLP', sub: mlpSub, kind: 'mlp', node: `mlp.${L}`, w: CARD_W }), (F2 + J2) / 2, YC, 'c');
      cMlp.classList.add('bl-card', 'bl-card-mlp');
      intro.push(cAttn, cMlp);

      // ---- norms. Pre-LN: on each leg, in front of its sub-layer; the pill is the norm.
      // Post-LN: on the river after each add, with no norm in front of a sub-layer; the pill zooms to the
      // residual add it normalises (the residual scene shows the sum and its norm) and traces that sum.
      const pill = (node, trace) => {
        const p = h('div', { class: 'bl-pill' }, nw);
        p.dataset.node = node;
        if (trace) p.dataset.trace = trace;
        return p;
      };
      const pN1 = pre ? put(pill(`norm.${L}.1`, `L${L}.ln1`), F1, LEG_Y, 'c') : put(pill(`residual.${L}.1`, `L${L}.add1`), LN[1], RY, 'c');
      const pN2 = pre ? put(pill(`norm.${L}.2`, `L${L}.ln2`), F2, LEG_Y, 'c') : put(pill(`residual.${L}.2`, `L${L}.add2`), LN[2], RY, 'c');
      intro.push(pN1, pN2);

      // ---- the residual adds (zoom targets). Residual off: the join is a replacement, drawn dashed.
      const join = (x, node) => {
        const j = op(res ? '\\oplus' : '=', { kind: 'resid', size: JR * 2, tex: true });
        j.classList.add('bl-join');
        j.dataset.node = node;
        return put(j, x, RY, 'c');
      };
      const join1 = join(J1, `residual.${L}.1`), join2 = join(J2, `residual.${L}.2`);
      intro.push(join1, join2);

      // ---- what each sub-layer writes back, with its norm. Residual off: the write replaces, so no plus sign.
      const wSign = res ? '+' : '=';
      const wChip = (kind, sym, trace) => h('div', { class: `bl-write bl-${kind}`, dataset: { trace } },
        h('span', { class: 'bl-w-sym', html: texHTML(`${wSign}\\,${sym}`) }),
        h('span', { class: 'bl-w-l', html: texHTML(`\\|${sym}\\|_2`) }),
        h('b', { class: 'bl-w-v' }, '–'));
      const wA = put(wChip('attn', 'a', `L${L}.attn.out`), J1 - 14, CHIP_Y, 'rc');
      const wM = put(wChip('mlp', 'm', `L${L}.mlp.out`), J2 - 14, CHIP_Y, 'rc');

      // ---- the stream read out under the river: block input, after attention, after MLP
      const station = (cls, x, symTex, note, traceKey) => {
        const v = stationVec(traceKey, new Array(d).fill(0));
        vecs.push(v);
        const nv = h('b', { class: 'bl-st-nv' }, '–');
        const node = put(h('div', { class: `bl-st ${cls}` },
          h('div', { class: 'bl-st-label', html: `${texHTML(symTex)}<span class="bl-st-note">${note}</span>` }),
          h('div', { class: 'bl-st-box' }, v, h('div', { class: 'bl-st-ghost' })),
          h('div', { class: 'bl-st-norm' }, '‖x‖₂ = ', nv)), x - 124, ST_Y);
        return { node, vec: v, nv, trace: traceKey, q };
      };
      const sIn = station('bl-st-in', ST.in, '\\xres^{\\text{in}}', 'block input', inKey);
      const sMid = station('bl-st-mid', ST.mid, '\\xres^{\\text{mid}}', 'after attention', `L${L}.resid_mid`);
      const sOut = station('bl-st-out', ST.out, '\\xres^{\\text{out}}', 'after MLP', `L${L}.resid_out`);
      intro.push(sIn.node, sMid.node, sOut.node);
      // readouts: a small arrow from the river down into each station. The block input is read at once; the other two wait for their station.
      const readArrow = (x) => [
        s('path', { d: `M${x},${RY + 24} V${ST_Y - 16}`, class: 'bl-readout' }),
        s('path', { d: `M${x - 5},${ST_Y - 16} L${x + 5},${ST_Y - 16} L${x},${ST_Y - 6} Z`, class: 'bl-readout-tip' }),
      ];
      const reads = { mid: readArrow(ST.mid), out: readArrow(ST.out) };
      svg.append(...readArrow(ST.in), ...reads.mid, ...reads.out);

      // ---- labels on the river
      put(h('div', { class: 'bl-river-label' }, res ? 'residual stream' : 'residual stream (off)'), X0, RY - 44, 'tl');
      put(h('div', { class: 'bl-gap-label bl-res-off' }, 'no skip'), (F1 + J1) / 2, GAP_Y, 'c');
      put(h('div', { class: 'bl-gap-label bl-res-off' }, 'no skip'), (F2 + J2) / 2, GAP_Y, 'c');

      P = { cAttn, cMlp, pN1, pN2, join1, join2, wA, wM, sIn, sMid, sOut, reads };
      // path lengths drive the draw-in (the paths are in the DOM now)
      [solidA, solidM].forEach((p) => p.style.setProperty('--len', String(pathLen(p))));

      D = compute();
      paintValues();
      applyStep(step, { animate: false });
      if (!introDone) {
        introDone = true;
        appear(intro, { stagger: MOTION.stagger });
      }
    }

    // Values: vector strips, norms, write chips. Recomputed for the selected token.
    function paintValues() {
      if (!P) return;
      D = compute();
      if (!D) return;
      curQ = D.q;
      const range = Math.max(1e-9, ...[D.xin, D.mid, D.out].map((v) => Math.max(...v.map((x) => Math.abs(x)))));
      const feed = (stn, values) => {
        if (stn.q === D.q) { stn.vec.update(values, { range }); return; }
        // a new token: the strip is rebuilt so that its data-cell stamps carry the new token index
        const next = stationVec(stn.trace, values, range);
        stn.vec.replaceWith(next);
        stn.vec.destroy?.();
        const k = vecs.indexOf(stn.vec);
        if (k >= 0) vecs[k] = next; else vecs.push(next);
        stn.vec = next;
        stn.q = D.q;
      };
      feed(P.sIn, D.xin);
      feed(P.sMid, D.mid);
      feed(P.sOut, D.out);
      setNum(P.sIn.nv, D.n.in);
      setNum(P.sMid.nv, D.n.mid);
      setNum(P.sOut.nv, D.n.out);
      P.wA.querySelector('.bl-w-v').textContent = nfmt(D.n.a);
      P.wM.querySelector('.bl-w-v').textContent = nfmt(D.n.m);
    }

    // The equation, its live line, its check and the caption change in one step: they fade out together, are
    // replaced together and fade back in. The swap is timed on the clock rather than on the end of the fade, so the
    // text is never held up by a frame. A newer change cancels an older one that is still fading.
    let textSig = '', textGen = 0, textTimer = null;
    function swapText(sig, write, animate) {
      if (sig === textSig) return;
      textSig = sig;
      const gen = ++textGen;
      const group = [eqPanel, checksPanel, cap.el];
      clearTimeout(textTimer);
      group.forEach((el) => el.getAnimations().forEach((a) => a.cancel()));
      if (!animate || reducedMotion()) { write(); return; }
      group.forEach((el) => el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: TEXT_OUT, fill: 'forwards' }));
      textTimer = setTimeout(() => {
        if (gen !== textGen) return;
        const held = group.flatMap((el) => el.getAnimations());   // the fade-out, held at 0 until the new text is in
        write();
        group.forEach((el) => el.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: TEXT_IN, easing: MOTION.easeOut }));
        held.forEach((a) => a.cancel());   // the new fade has taken over, so no fill can keep the text hidden
      }, TEXT_OUT);
    }

    // Text: the equation for the step, a live line for dimension 0, the caption and the check for this step.
    function paintText({ animate = true } = {}) {
      if (!P || !D) return;
      const { q, pre, res, n, xin, a, mid, m, out, add1, add2, sumA, sumM, d } = D;
      const c = ctx.state.config;
      const last = L === c.n_layer - 1;
      const act = ACT[c.activation] ?? c.activation;
      const NORM = c.norm === 'rmsnorm' ? '\\RMSNorm' : '\\LN';
      const nw = normWord(c);
      let eq = '', live = '', words = '';

      switch (step) {
        case 0:
          eq = `${X}_{t} \\in \\R^{${d}}`;
          live = `${X}_{${q},0} = ${n3(xin[0])}`;
          words = res
            ? `Each token’s vector is one row of the residual stream. It enters block ${L} as x, and each sub-layer adds to it rather than replacing it.`
            : 'Residual off: nothing carries x past a sub-layer. Each sub-layer replaces the stream with its own output.';
          break;
        case 1:
          if (pre) {
            eq = res ? `${X}^{\\text{mid}} = ${X} + \\Attn(${NORM}(${X}))` : `${X}^{\\text{mid}} = \\Attn(${NORM}(${X}))`;
            live = res ? sumLine(`${X}^{\\text{mid}}_{${q},0}`, xin[0], a[0], mid[0]) : `${X}^{\\text{mid}}_{${q},0} = ${n3(a[0])}`;
            words = res
              ? `Pre-${nw}: the norm comes first. Attention reads the normalised stream and returns its output a, with ‖a‖₂&nbsp;=&nbsp;<b>${nHTML(n.a)}</b>, which is added to the river.`
              : 'Residual off: attention reads the normalised stream, and its output a replaces the stream.';
          } else {
            eq = res ? `${X}^{\\text{mid}} = ${NORM}(${X} + \\Attn(${X}))` : `${X}^{\\text{mid}} = ${NORM}(\\Attn(${X}))`;
            live = res ? sumLine(`(${X} + \\Attn(${X}))_{${q},0}`, xin[0], a[0], add1[0]) : `a_{${q},0} = ${n3(a[0])}`;
            words = res
              ? `Post-${nw}: attention reads the raw stream. The sum goes through the norm, so the norm sits on the river after the add.`
              : 'Residual off: attention’s output replaces the stream, and the norm then rescales it.';
          }
          break;
        case 2:
          if (pre) {
            eq = res ? `${X}^{\\text{out}} = ${X}^{\\text{mid}} + \\MLP(${NORM}(${X}^{\\text{mid}}))` : `${X}^{\\text{out}} = \\MLP(${NORM}(${X}^{\\text{mid}}))`;
            live = res ? sumLine(`${X}^{\\text{out}}_{${q},0}`, mid[0], m[0], out[0]) : `${X}^{\\text{out}}_{${q},0} = ${n3(m[0])}`;
            words = res
              ? `The MLP reads the normalised stream, one token at a time: it expands to ${c.d_ff} hidden units, applies ${act}, and projects back to d. It writes m with ‖m‖₂&nbsp;=&nbsp;<b>${nHTML(n.m)}</b>.`
              : 'Residual off: the MLP reads the normalised stream, and its output replaces the stream.';
          } else {
            eq = res ? `${X}^{\\text{out}} = ${NORM}(${X}^{\\text{mid}} + \\MLP(${X}^{\\text{mid}}))` : `${X}^{\\text{out}} = ${NORM}(\\MLP(${X}^{\\text{mid}}))`;
            live = res ? sumLine(`(${X}^{\\text{mid}} + \\MLP(${X}^{\\text{mid}}))_{${q},0}`, mid[0], m[0], add2[0]) : `m_{${q},0} = ${n3(m[0])}`;
            words = res
              ? 'The MLP reads the stream directly. Its output is added, and the sum is normalised again.'
              : 'Residual off: the MLP reads the stream directly, and its output replaces it before the norm.';
          }
          break;
        default: {
          eq = last
            ? (pre ? `\\text{logits} = \\WU\\,${NORM}_f(${X}^{\\text{out}})` : `\\text{logits} = \\WU\\,${X}^{\\text{out}}`)
            : `${X}^{(${L + 1})} = ${X}^{\\text{out}}`;
          live = `\\|${X}_{${q}}\\|_2 = ${num(n.in, 3)} \\to ${num(n.out, 3)}`;
          if (last) {
            words = pre ? 'After the last block, the stream goes through the final norm, then the LM head.' : 'After the last block, the stream goes straight to the LM head.';
          } else {
            const ret = !pre
              ? (res ? 'The block returns the normalised sum' : 'The block returns the normalised MLP output')
              : (res ? 'The block returns the input vector with both writes added' : 'The block returns the MLP output alone');
            words = `${ret}, and block ${L + 1} reads it next.`;
          }
          // the norm change is named only when it is visible at two decimals
          if (nfmt(n.in) !== nfmt(n.out)) words += ` ‖x‖₂ went from <b>${nHTML(n.in)}</b> to <b>${nHTML(n.out)}</b>.`;
        }
      }

      // The check for this step: each sum recomputed from the traced inputs and compared entry by entry with the traced
      // sum. Only the check for the current step is shown. Its label names the sum with symbols the equation shows.
      const attnName = pre ? (res ? 'x_mid = x + a' : 'x_mid = a') : (res ? 'before the norm = x + a' : 'before the norm = a');
      const mlpName = pre ? (res ? 'x_out = x_mid + m' : 'x_out = m') : (res ? 'before the norm = x_mid + m' : 'before the norm = m');
      const check = step === 1 ? { name: attnName, sumV: sumA, tracedV: add1 } : step >= 2 ? { name: mlpName, sumV: sumM, tracedV: add2 } : null;
      const err = check ? maxDiff(check.sumV, check.tracedV) : 0;
      const rows = check ? [h('div', { class: 'bl-check-row' },
        h('div', { class: 'bl-check-line' }, checkBadge(err, 0, { label: `${check.name}: equals the traced value`, tol: 1e-5 })),
        h('div', { class: 'bl-dim' }, `max |Δ| over all ${d} entries: ${errText(err)}`))] : [];

      const label = `${STEPS[step]} · token “${ctx.label(q)}” at position ${q}`;
      const sig = [step, label, eq, live, words, check ? `${check.name}|${err}` : ''].join('\n');
      swapText(sig, () => {
        eqLabel.textContent = label;
        eqMain.innerHTML = texHTML(eq, { display: true });
        eqLive.innerHTML = texHTML(live);
        cap.el.innerHTML = words;
        checksPanel.replaceChildren(...rows);
      }, animate);
    }

    // The step decides which parts are lit; the wiring itself is drawn by CSS from data-step.
    function applyStep(i, { animate = true } = {}) {
      step = clamp(i, 0, STEPS.length - 1);
      if (!P) return;
      wrap.dataset.step = String(step);
      P.sMid.node.classList.toggle('pending', step < 1);
      P.sOut.node.classList.toggle('pending', step < 2);
      // the readout that this step reads or writes is highlighted
      P.sIn.node.classList.toggle('hot', step === 0);
      P.sMid.node.classList.toggle('hot', step === 1);
      P.sOut.node.classList.toggle('hot', step >= 2);
      P.wA.classList.toggle('on', step >= 1);
      P.wM.classList.toggle('on', step >= 2);
      P.pN1.classList.toggle('hot', step === 1);
      P.pN2.classList.toggle('hot', step === 2);
      P.reads.mid.forEach((el) => el.classList.toggle('pending', step < 1));
      P.reads.out.forEach((el) => el.classList.toggle('pending', step < 2));
      if (animate && D) {
        if (step === 0) tweenNum(P.sIn.nv, D.n.in);
        if (step === 1) { pulse(P.join1); tweenNum(P.sMid.nv, D.n.mid); }
        if (step === 2) { pulse(P.join2); tweenNum(P.sOut.nv, D.n.out); }
      }
      paintText({ animate });
    }

    function rebuild() {
      structKey = structureKey();
      build();
    }

    rebuild();
    ctx.setSteps(STEPS, (i) => { clearTimeout(auto); applyStep(i); });
    appear([header.el, picker.el, eqPanel, checksPanel], { stagger: MOTION.stagger });

    // explain itself the first time it is shown; any press takes over
    if (!reducedMotion()) {
      const advance = () => {
        if (step < STEPS.length - 1) { ctx.setStep(step + 1); auto = setTimeout(advance, MOTION.step); }
      };
      auto = setTimeout(advance, 1200);
      el.addEventListener('pointerdown', () => clearTimeout(auto), { once: true });
      el.addEventListener('atlas:interrupt', () => clearTimeout(auto), { once: true });   // the story bar or the arrow keys took over
    }

    const offs = [
      // a trace event follows every config change, so the wiring is rebuilt there
      ctx.on('trace', () => { if (!P || structureKey() !== structKey) rebuild(); else { paintValues(); paintText(); } }),
      ctx.on('select', () => { if (P) { paintValues(); paintText(); } }),
    ];

    return {
      pause() { clearTimeout(auto); },
      destroy() {
        clearTimeout(auto);
        clearTimeout(textTimer);
        offs.forEach((f) => f());
        vecs.splice(0).forEach((v) => v.destroy?.());
        st.destroy();
      },
    };
  },
};

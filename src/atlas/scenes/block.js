// Block: one Transformer block on the real forward pass. The residual stream is a river; attention and the
// MLP are detours that read it, compute, and add their output back. The wiring follows cfg.norm_placement:
//   pre:  x_mid = x + Attn(Norm(x)),   x_out = x_mid + MLP(Norm(x_mid))   (a norm sits on each leg)
//   post: x_mid = Norm(x + Attn(x)),   x_out = Norm(x_mid + MLP(x_mid))   (a norm sits on the river after each add)
// With cfg.residual off the river is broken and each sub-layer replaces the stream (no add, no plus sign).
// Every number comes from ctx.state.trace. Each sum is recomputed from the traced vectors and checked against
// the traced sum (checkBadge). The norm is named from cfg.norm (LayerNorm or RMSNorm).
import { stage2d, card, op, vec, chip, appear, pulse, sceneHeader, sceneCaption, tokenPicker, checkBadge, useStyle, ZONES, MOTION } from '../kit.js';
import { h, s, texHTML, num, reducedMotion, clamp } from '../../ui/index.js';

const STEPS = ['Stream in', 'Attention reads & writes', 'MLP reads & writes', 'Stream out'];
const TITLE = 'What one block does';
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
const ST = { in: 210, mid: (J1 + F2) / 2, out: 1390 };
// post-LN norms sit on the river between a join and the next fork (or the river end)
const LN = { 1: (J1 + RING + F2 - 6) / 2, 2: (J2 + RING + X1 - 14) / 2 };
const EQ_Y = 628;                    // equation band under the stations
const CHECK_X = 1010;
const ACT = { gelu: 'GELU', relu: 'ReLU', swiglu: 'SwiGLU' };
const MASK = { causal: 'causal', none: 'bidirectional', sliding: 'sliding window', prefix: 'prefix' };
const X = '\\xres';                  // the residual stream symbol (coloured macro)

useStyle('block-scene', `
.bl-wrap { position: absolute; left: 0; top: 0; width: 1600px; height: 900px; pointer-events: none; }
.bl-wrap > * { pointer-events: auto; }
.bl-wrap > .bl-svg { pointer-events: none; }
.bl-svg { position: absolute; left: 0; top: 0; width: 1600px; height: 900px; overflow: visible; }
.bl-river-base { fill: none; stroke: var(--c-resid); stroke-opacity: .2; stroke-width: 40; }
.bl-river-line { fill: none; stroke: var(--c-resid); stroke-opacity: .45; stroke-width: 1.5; }
.bl-river-beads { fill: none; stroke: var(--c-resid); stroke-width: 9; stroke-linecap: round; stroke-dasharray: 1 22; opacity: .6; animation: bl-flow 1.6s linear infinite; }
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
.bl-wrap[data-step="1"] .bl-det-solid.bl-attn { opacity: 1; stroke-dashoffset: 0; }
.bl-wrap[data-step="2"] .bl-det-solid.bl-attn { opacity: .6; stroke-dashoffset: 0; }
.bl-wrap[data-step="3"] .bl-det-solid.bl-attn { opacity: .6; stroke-dashoffset: 0; }
.bl-wrap[data-step="2"] .bl-det-solid.bl-mlp { opacity: 1; stroke-dashoffset: 0; }
.bl-wrap[data-step="3"] .bl-det-solid.bl-mlp { opacity: .6; stroke-dashoffset: 0; }
.bl-wrap[data-step="1"] .bl-det-tip.bl-attn, .bl-wrap[data-step="2"] .bl-det-tip.bl-attn, .bl-wrap[data-step="3"] .bl-det-tip.bl-attn,
.bl-wrap[data-step="2"] .bl-det-tip.bl-mlp, .bl-wrap[data-step="3"] .bl-det-tip.bl-mlp { opacity: 1; }
.bl-wrap[data-step="1"] .bl-beads.bl-attn, .bl-wrap[data-step="2"] .bl-beads.bl-mlp { opacity: .95; }
@keyframes bl-flow { to { stroke-dashoffset: -23px; } }
@keyframes bl-flow-b { to { stroke-dashoffset: -18px; } }
.bl-wrap[data-res="on"] .bl-res-off, .bl-wrap[data-res="off"] .bl-res-on { display: none; }
.bl-wrap[data-res="off"] .bl-join { border-style: dashed; }

.bl-card { background: var(--surface); transition: opacity .6s cubic-bezier(.2,.7,.2,1), box-shadow .45s cubic-bezier(.2,.7,.2,1), transform .25s var(--ease); }
.bl-wrap[data-step="0"] .bl-card { opacity: .55; }
.bl-wrap[data-step="1"] .bl-card-mlp { opacity: .55; }
.bl-wrap[data-step="2"] .bl-card-attn { opacity: .7; }
.bl-wrap[data-step="1"] .bl-card-attn, .bl-wrap[data-step="2"] .bl-card-mlp { box-shadow: 0 0 0 2px color-mix(in srgb, var(--kind) 70%, transparent), 0 20px 44px rgba(0, 0, 0, .2); }

.bl-pill { position: absolute; width: 112px; height: 28px; border-radius: 999px; background: var(--surface); border: 1.5px solid var(--c-norm); color: var(--c-norm); font: 600 15px/25px var(--font-mono); text-align: center; white-space: nowrap; box-shadow: var(--shadow-1); transition: box-shadow .4s, color .4s; }
.bl-pill[data-node] { cursor: zoom-in; }
.bl-pill[data-node]:hover { box-shadow: 0 0 0 4px color-mix(in srgb, var(--c-norm) 35%, transparent), var(--shadow-1); }
.bl-pill.hot { color: var(--fg); box-shadow: 0 0 0 4px color-mix(in srgb, var(--c-norm) 45%, transparent), var(--shadow-1); }

.bl-write { position: absolute; display: inline-flex; align-items: center; gap: 8px; height: 30px; padding: 0 12px; border-radius: 999px; font: 600 15px/1 var(--font-mono); white-space: nowrap; color: var(--kind); background: color-mix(in srgb, var(--kind) 14%, var(--surface)); border: 1px solid color-mix(in srgb, var(--kind) 45%, transparent); opacity: 0; pointer-events: none; transition: opacity .6s cubic-bezier(.2,.7,.2,1); }
.bl-write.bl-attn { --kind: var(--c-attn); }
.bl-write.bl-mlp { --kind: var(--c-mlp); }
.bl-write.on { opacity: 1; pointer-events: auto; }
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

.bl-readout { fill: none; stroke: var(--c-resid); stroke-width: 1.5; stroke-linecap: round; opacity: .6; }
.bl-readout-tip { fill: var(--c-resid); opacity: .6; }
.bl-river-label { font: 600 15px/1 var(--font-mono); color: var(--c-resid); white-space: nowrap; }
.bl-gap-label { font: 600 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }

.bl-switch { display: inline-flex; align-items: center; gap: 2px; padding: 4px; border-radius: 999px; background: color-mix(in srgb, var(--surface) 80%, transparent); border: 1px solid var(--line); box-shadow: var(--shadow-1); }
.bl-nav { border: 0; background: transparent; color: var(--fg-2); font: 600 15px/1 var(--font-mono); padding: 9px 12px; border-radius: 999px; cursor: pointer; white-space: nowrap; }
.bl-nav:hover { background: var(--surface-2); color: var(--fg); }
.bl-nav:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.bl-sep { color: var(--fg-3); font-size: 15px; padding: 0 2px; }

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
  .bl-river-beads, .bl-beads { animation: none; }
  .bl-det-solid, .bl-card, .bl-st-box .k-vec, .bl-st-ghost, .bl-st-norm, .bl-write { transition: none; }
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

function tweenNum(el, to, ms = 800) {
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
      const nL = c.n_layer;
      const normName = c.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';
      const attnSub = c.n_kv_head < c.n_head ? `${c.n_head} query · ${c.n_kv_head} KV heads` : `${c.n_head} heads · ${MASK[c.mask] ?? c.mask} mask`;
      const mlpSub = `${ACT[c.activation] ?? c.activation} · ${c.d_ff} hidden units`;
      const inKey = L === 0 ? 'emb.out' : `L${L - 1}.resid_out`;
      curQ = q;

      // the header follows the config: chips and the context line
      header.el.querySelector('.k-eyebrow')?.replaceChildren(
        chip(`Block ${L}`, 'resid'), chip(pre ? 'Pre-LN' : 'Post-LN', 'norm'), chip(res ? 'Residual on' : 'Residual off', 'resid'));
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
        const p = h('div', { class: 'bl-pill' }, normName);
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
      // readouts: a small arrow from the river down into each station
      [ST.in, ST.mid, ST.out].forEach((x) => svg.append(
        s('path', { d: `M${x},${RY + 24} V${ST_Y - 16}`, class: 'bl-readout' }),
        s('path', { d: `M${x - 5},${ST_Y - 16} L${x + 5},${ST_Y - 16} L${x},${ST_Y - 6} Z`, class: 'bl-readout-tip' })));

      // ---- labels on the river
      put(h('div', { class: 'bl-river-label' }, 'residual stream'), X0, RY - 44, 'tl');
      put(h('div', { class: 'bl-gap-label bl-res-off' }, 'no skip'), (F1 + J1) / 2, GAP_Y, 'c');
      put(h('div', { class: 'bl-gap-label bl-res-off' }, 'no skip'), (F2 + J2) / 2, GAP_Y, 'c');

      // ---- block switcher, top right: ‹ Block l−1 · Block l+1 ›. At either end only the neighbour is shown.
      const navs = [];
      if (L > 0) navs.push(h('button', { type: 'button', class: 'bl-nav', 'aria-label': `Go to block ${L - 1}`, onClick: () => ctx.go(`block.${L - 1}`) }, `‹ Block ${L - 1}`));
      if (L > 0 && L < nL - 1) navs.push(h('span', { class: 'bl-sep' }, '·'));
      if (L < nL - 1) navs.push(h('button', { type: 'button', class: 'bl-nav', 'aria-label': `Go to block ${L + 1}`, onClick: () => ctx.go(`block.${L + 1}`) }, `Block ${L + 1} ›`));
      if (navs.length) intro.push(put(h('div', { class: 'bl-switch' }, ...navs), X1, 28, 'tr'));

      P = { cAttn, cMlp, pN1, pN2, join1, join2, wA, wM, sIn, sMid, sOut };
      // path lengths drive the draw-in (the paths are in the DOM now)
      [solidA, solidM].forEach((p) => p.style.setProperty('--len', String(pathLen(p))));

      D = compute();
      paintValues();
      applyStep(step, { animate: false });
      if (!introDone) {
        introDone = true;
        appear(intro, { stagger: 70 });
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

    // Text: the equation for the step, a live line for dimension 0, the caption and the check for this step.
    function paintText({ animate = false } = {}) {
      if (!P || !D) return;
      const { q, pre, res, n, xin, a, mid, m, out, add1, add2, sumA, sumM, d } = D;
      const c = ctx.state.config;
      const last = L === c.n_layer - 1;
      const act = ACT[c.activation] ?? c.activation;
      const NORM = c.norm === 'rmsnorm' ? '\\RMSNorm' : '\\LN';
      const sgn = (v) => (v < 0 ? '-' : '+');
      const line = (lhs, x0, y0, z0) => `${lhs} = ${num(x0, 3)} ${sgn(y0)} ${num(Math.abs(y0), 3)} = ${num(z0, 3)}`;
      let eq = '', live = '', words = '';

      switch (step) {
        case 0:
          eq = `${X}_{t} \\in \\R^{${d}}`;
          live = `${X}_{${q},0} = ${num(xin[0], 3)}`;
          words = res
            ? `Each token’s vector is one row of the residual stream. It enters block ${L} as <b>x</b>, and each sub-layer adds to it rather than replacing it.`
            : `Residual off: nothing carries <b>x</b> past a sub-layer. Each sub-layer replaces the stream with its own output.`;
          break;
        case 1:
          if (pre) {
            eq = res ? `${X}^{\\text{mid}} = ${X} + \\Attn(${NORM}(${X}))` : `${X}^{\\text{mid}} = \\Attn(${NORM}(${X}))`;
            live = res ? line(`${X}^{\\text{mid}}_{${q},0}`, xin[0], a[0], mid[0]) : `${X}^{\\text{mid}}_{${q},0} = ${num(a[0], 3)}`;
            words = res
              ? `Pre-LN: the norm comes first. Attention reads the normalised stream and returns its output <b>a</b>, with ‖a‖₂ = <b>${nHTML(n.a)}</b>, which is added to the river.`
              : 'Residual off: attention reads the normalised stream, and its output <b>a</b> replaces the stream.';
          } else {
            eq = res ? `${X}^{\\text{mid}} = ${NORM}(${X} + \\Attn(${X}))` : `${X}^{\\text{mid}} = ${NORM}(\\Attn(${X}))`;
            live = res ? line(`s^{(1)}_{${q},0}`, xin[0], a[0], add1[0]) : `a_{${q},0} = ${num(a[0], 3)}`;
            words = res
              ? 'Post-LN: attention reads the raw stream. The sum goes through the norm, so the norm sits on the river after the add.'
              : 'Residual off: attention’s output replaces the stream, and the norm then rescales it.';
          }
          break;
        case 2:
          if (pre) {
            eq = res ? `${X}^{\\text{out}} = ${X}^{\\text{mid}} + \\MLP(${NORM}(${X}^{\\text{mid}}))` : `${X}^{\\text{out}} = \\MLP(${NORM}(${X}^{\\text{mid}}))`;
            live = res ? line(`${X}^{\\text{out}}_{${q},0}`, mid[0], m[0], out[0]) : `${X}^{\\text{out}}_{${q},0} = ${num(m[0], 3)}`;
            words = res
              ? `The MLP does the same, one token at a time: normalise, expand to ${c.d_ff} hidden units, apply ${act}, and project back to d. It writes <b>m</b> with ‖m‖₂ = <b>${nHTML(n.m)}</b>.`
              : 'Residual off: the MLP reads the normalised stream, and its output replaces the stream.';
          } else {
            eq = res ? `${X}^{\\text{out}} = ${NORM}(${X}^{\\text{mid}} + \\MLP(${X}^{\\text{mid}}))` : `${X}^{\\text{out}} = ${NORM}(\\MLP(${X}^{\\text{mid}}))`;
            live = res ? line(`s^{(2)}_{${q},0}`, mid[0], m[0], add2[0]) : `m_{${q},0} = ${num(m[0], 3)}`;
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

      // the check for this step: each sum recomputed from the traced inputs and compared with the traced sum.
      // Only the check for the current step is shown; its formula is named in the badge text.
      const attnName = pre ? (res ? 'x_mid = x + a' : 'x_mid = a') : (res ? 's_1 = x + a' : 's_1 = a');
      const mlpName = pre ? (res ? 'x_out = x_mid + m' : 'x_out = m') : (res ? 's_2 = x_mid + m' : 's_2 = m');
      const checkRow = (name, sumV, tracedV) => h('div', { class: 'bl-check-row' },
        h('div', { class: 'bl-check-line' }, checkBadge(norm(sumV), norm(tracedV), { label: `${name}: equals the traced value` })),
        h('div', { class: 'bl-dim' }, `max |Δ| over all ${d} entries: ${errText(maxDiff(sumV, tracedV))}`));
      const rows = [];
      if (step === 1) rows.push(checkRow(attnName, sumA, add1));
      if (step >= 2) rows.push(checkRow(mlpName, sumM, add2));

      eqLabel.textContent = `${STEPS[step]} · token “${ctx.label(q)}” at position ${q}`;
      eqMain.innerHTML = texHTML(eq, { display: true });
      eqLive.innerHTML = texHTML(live);
      cap.set(words);
      checksPanel.replaceChildren(...rows);
      if (animate && !reducedMotion()) eqPanel.animate([{ opacity: 0.55 }, { opacity: 1 }], { duration: 380, easing: 'ease-out' });
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
        offs.forEach((f) => f());
        vecs.splice(0).forEach((v) => v.destroy?.());
        st.destroy();
      },
    };
  },
};

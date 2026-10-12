// The MLP of one layer, on one token, in four steps on the real numbers:
//   input → expand to F (W_up; SwiGLU adds W_gate) → activate (GELU, ReLU or SwiGLU) → project back (W_down).
// The input is L{l}.ln2 (pre-LN: the stream normalised by the config's norm) or L{l}.resid_mid (post-LN: already normalised by ln1).
// Each step recomputes its numbers from the weights in state.model.params and checks them against the traced
// L{l}.mlp.up, .mlp.gate, .mlp.act and .mlp.out. Zoom targets: the neuron grid is activation.<l>, the
// down-projection fan (and its label) is neurons.<l>. Reference scene for contributors: see docs/ATLAS_GUIDE.md and docs/STYLE_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, op, texLabel, pulse, appear, useStyle, ZONES, MOTION } from '../kit.js';
import { h, s, fmt, texHTML, diverging, onThemeChange, reducedMotion, clamp } from '../../ui/index.js';
import { geluScalar, siluScalar } from '../../engine/index.js';

const STEPS = ['Input', 'Expand to F', 'Activate', 'Project back'];
const CELL = 14, GAP = 3;          // cell size of a vector strip (the same as the kit's vec())
const SHOWN = 17;                  // a long strip shows the first 12 cells, a gap, and the last 4
const GX = 625, GY = 510;          // centre of the neuron grid, design units
const VX = 122, OX = 1128;         // centres of the input and output strips (26 units wide)
const OP = 48;                     // the × operation nodes: the size of head.js and lm-head.js
const PAD = 10;                    // padding inside the neuron grid
const MAX_GRID = 300;              // the grid is at most this many units on a side: this holds up to F = 4 900 (70 × 70)
const SIDE_Y = 262;                // top of the side column (ZONES.side, equation column)
const GELU_RANGE = '(−0.171, 0)';  // the tanh GELU never goes below −0.17004
// the kit's zoom glyph (a magnifier with a plus), drawn so it never depends on a font having the character
const ZOOM_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5 21 21M10.5 7.5v6M7.5 10.5h6"/></svg>';

useStyle('mlp-scene', `
.ml-side { position: absolute; width: 340px; display: grid; gap: 12px; align-content: start; }
.ml-kick { font: 600 17px/1.2 var(--font-mono); color: var(--c-mlp); }
.ml-main { font-size: 24px; color: var(--fg); }
.ml-main .katex-display, .ml-live .katex-display { margin: 0; text-align: left; }
.ml-live { display: grid; gap: 6px; font-size: 18px; line-height: 1.35; padding: 10px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); box-shadow: 0 6px 16px rgba(0, 0, 0, .06); overflow-x: auto; }
.ml-checks { display: flex; flex-wrap: wrap; gap: 8px; }
.ml-checks:empty { display: none; }
.ml-params { display: flex; align-items: center; gap: 12px; white-space: nowrap; font: 500 16px/1.2 var(--font-mono); color: var(--fg-2); }
.ml-params b { color: var(--fg); font-weight: 600; }
.ml-tag { font: 600 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.ml-tag.top { color: var(--c-mlp); }
.ml-illus { font-weight: 500; color: var(--fg-3); }
.ml-grid { position: absolute; border-radius: 12px; border: 1px solid color-mix(in srgb, var(--c-mlp) 45%, var(--line)); background: color-mix(in srgb, var(--surface) 62%, transparent); box-shadow: 0 14px 34px rgba(0, 0, 0, .12), 0 2px 0 color-mix(in srgb, var(--c-mlp) 22%, transparent); transition: transform .25s var(--ease), box-shadow .25s var(--ease); }
.ml-grid:hover { transform: translateY(-4px) scale(1.015); box-shadow: 0 0 0 2px color-mix(in srgb, var(--c-mlp) 70%, transparent), 0 22px 46px rgba(0, 0, 0, .2); }
.ml-grid > .k-zoom-badge { top: -26px; right: 0; color: var(--c-mlp); opacity: .7; }
.ml-grid:hover > .k-zoom-badge { opacity: 1; }
.ml-dot { position: absolute; border-radius: 50%; background: var(--surface-3); box-shadow: inset 0 0 0 1px var(--line); transition: background-color 420ms ease, box-shadow 420ms ease; }
.ml-dot.hl { outline: 2px solid var(--accent); outline-offset: 2px; }
.ml-fanhit { position: absolute; border-radius: 12px; cursor: zoom-in; }
.ml-fanhit:hover { background: color-mix(in srgb, var(--c-mlp) 7%, transparent); }
.ml-fanlab { font: 600 15px/1 var(--font-mono); color: var(--c-mlp); white-space: nowrap; opacity: .85; transition: opacity .2s ease; }
.ml-fanlab:hover { opacity: 1; }
.ml-glyph { display: inline-block; margin-left: 6px; vertical-align: -3px; }
.ml-glyph svg { display: block; width: 14px; height: 14px; }
.ml-line { fill: none; stroke: var(--c-mlp); stroke-width: 0.8; stroke-linecap: round; transition: stroke-dashoffset 540ms cubic-bezier(.2,.7,.2,1), stroke-opacity 420ms ease; }
.ml-arrow { transition: opacity 420ms ease; }
.ml-arrow-line { fill: none; stroke: var(--kind); stroke-width: 2.5; stroke-linecap: round; }
.ml-arrow-head { fill: var(--kind); }
.ml-op { transition: opacity 420ms ease; }
.ml-focus { position: absolute; right: calc(100% + 10px); transform: translateY(-50%); text-align: right; font: 600 15px/1.25 var(--font-mono); color: var(--accent); white-space: nowrap; pointer-events: none; opacity: 0; transition: opacity 420ms ease; }
.ml-focus.on { opacity: 1; }
.ml-focus span { display: block; font-weight: 500; color: var(--fg-2); }
.k-cell.ml-empty { background: var(--surface-3); }
.ml-still, .ml-still * { transition: none !important; }
`);

const esc = (x) => String(x).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);
const tf = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');     // 32 768
/** A number at d decimals. A value that rounds to zero reads 0.00, never -0.00. */
const fixed = (v, d) => (Math.abs(v) < 0.5 * 10 ** -d ? (0).toFixed(d) : v.toFixed(d));
const n2 = (v) => fixed(v, 2);
const n3 = (v) => fixed(v, 3);                                      // an activation's argument, to three decimals
const r3 = (v) => Number(v.toFixed(3));                             // the same argument as the number the line prints
const minus = (str) => String(str).replace(/-/g, '−');              // the typographic minus of the plain-text labels
/** The check badge of checkBadge, for values recomputed here: a relative tolerance (1e-6, as in the kit) and the largest
 *  difference in exponent form. The kit prints three decimals, so a real failure below 5e-4 would read 0.000. */
function check(pairs, label) {
  const diff = Math.max(...pairs.map(([got, want]) => Math.abs(got - want)));
  const scale = Math.max(1, ...pairs.map(([, want]) => Math.abs(want)));
  const ok = diff <= 1e-6 * scale;
  return h('span', { class: ['k-check', ok ? 'ok' : 'bad'], title: pairs.map(([got, want]) => `computed ${got} · traced ${want}`).join('\n') },
    ok ? `✓ ${label}` : `✗ differs from the traced value by ${diff.toExponential(1)}`);
}

/** A vector as a column of cells, with the kit's look: the first 12 cells, a gap, and the last 4 when it is long. */
function strip(n, kind) {
  const idx = n <= SHOWN ? [...Array(n).keys()] : [...Array(12).keys(), -1, n - 4, n - 3, n - 2, n - 1];
  const cells = idx.map((i) => h('div', { class: i < 0 ? 'k-cell ell' : 'k-cell' }, i < 0 ? '⋮' : null));
  const el = h('div', { class: ['k-vec', 'v', `kind-${kind}`], style: { '--cell': `${CELL}px`, '--gap': `${GAP}px`, '--kind': `var(--c-${kind})` } },
    h('div', { class: 'k-cells' }, cells));
  const shown = Math.min(n, SHOWN);
  return { el, idx, cells, height: shown * CELL + (shown - 1) * GAP + 12 };
}

/** Colour the cells of a strip with the values of token t (filled) or leave them empty. */
function paintStrip(S, vals, t, { range = 1, filled = true, hl = -1, delay = () => 0 } = {}) {
  S.idx.forEach((i, p) => {
    if (i < 0) return;
    const c = S.cells[p];
    c.dataset.cell = `${t},${i},0`;
    c.classList.toggle('ml-empty', !filled);
    c.classList.toggle('hl', i === hl);
    c.style.transitionDelay = `${filled ? delay(i) : 0}ms`;
    c.style.background = filled ? diverging(vals[i] / range) : '';
    c.title = filled ? `${i}: ${fmt(vals[i], 4)}` : `${i}`;
  });
}

/** A straight arrow with a head at x2, in the colour of the component kind. */
function arrowSvg(x1, x2, y, kind) {
  return s('g', { class: 'ml-arrow', style: { '--kind': `var(--c-${kind})` } },
    s('path', { class: 'ml-arrow-line', d: `M${x1} ${y} L${x2 - 10} ${y}` }),
    s('path', { class: 'ml-arrow-head', d: `M${x2} ${y} L${x2 - 11} ${y - 6} L${x2 - 11} ${y + 6} Z` }));
}

/** Everything this step needs for the selected token, read from the trace and the weights. */
function dataFor(ctx, L) {
  const S = ctx.state, c = S.config, T = S.ids.length, tr = S.trace;
  if (!T || !tr || !c) return null;
  const t = clamp(S.sel.token ?? 0, 0, T - 1);
  const d = c.d_model, F = c.d_ff, kind = c.activation;
  const sw = kind === 'swiglu', relu = kind === 'relu', gelu = !sw && !relu, bias = !!c.bias;
  const pre = c.norm_placement !== 'post', resid = !!c.residual, rms = c.norm === 'rmsnorm';
  const inKey = tr[`L${L}.ln2`] ? `L${L}.ln2` : `L${L}.resid_mid`;
  const rowOf = (key) => {
    const x = tr[key];
    if (!x) return null;
    const X = x.shape[x.shape.length - 1];
    return Array.from(x.data.subarray(t * X, t * X + X));
  };
  const xin = rowOf(inKey), up = rowOf(`L${L}.mlp.up`), act = rowOf(`L${L}.mlp.act`), out = rowOf(`L${L}.mlp.out`);
  const gate = sw ? rowOf(`L${L}.mlp.gate`) : null;
  const P = S.model?.params ?? {};
  const W = (name) => P[name]?.data ?? null;
  const Wu = W(`blocks.${L}.mlp.up_proj.weight`), Wd = W(`blocks.${L}.mlp.down_proj.weight`);
  const Wg = sw ? W(`blocks.${L}.mlp.gate_proj.weight`) : null;
  const bu = bias ? W(`blocks.${L}.mlp.up_proj.bias`) : null;
  const bg = bias && sw ? W(`blocks.${L}.mlp.gate_proj.bias`) : null;
  const bd = bias ? W(`blocks.${L}.mlp.down_proj.bias`) : null;
  if (!xin || !up || !act || !out || (sw && !gate) || !Wu || !Wd || (sw && !Wg) || xin.length !== d || up.length !== F) return null;

  const dot = (Wm, r, n, x) => { let acc = 0; for (let k = 0; k < n; k++) acc += x[k] * Wm[r * n + k]; return acc; };
  const maxAbs = (v) => v.reduce((m, x) => Math.max(m, Math.abs(x)), 0) || 1;
  const absAct = act.map((v) => Math.abs(v));
  const order = [...Array(F).keys()].sort((a, b) => absAct[b] - absAct[a] || a - b);   // strongest first
  const rank = new Array(F);
  order.forEach((f, k) => { rank[f] = k; });
  const fStar = order[0];
  const cStar = out.reduce((best, v, k) => (Math.abs(v) > Math.abs(out[best]) ? k : best), 0);
  const hRe = dot(Wu, fStar, d, xin) + (bu ? bu[fStar] : 0);      // h (u for SwiGLU), from the weights
  const gRe = sw ? dot(Wg, fStar, d, xin) + (bg ? bg[fStar] : 0) : null;
  const aRe = relu ? Math.max(0, hRe) : sw ? siluScalar(gRe) * hRe : geluScalar(hRe);
  const mRe = dot(Wd, cStar, F, act) + (bd ? bd[cStar] : 0);
  const nNeg = sw ? act.filter((v) => v < 0).length : up.filter((v) => v < 0).length;
  const total = Wu.length + Wd.length + (Wg ? Wg.length : 0);
  const expected = (sw ? 3 : 2) * d * F;
  const biasTotal = bias ? (sw ? 2 * F : F) + d : 0;
  return {
    t, d, F, sw, relu, gelu, bias, pre, rms, resid, inKey, xin, up, act, out, gate,
    rUp: maxAbs(up), rAct: maxAbs(act), rOut: maxAbs(out), rIn: maxAbs(xin),
    order, rank, fStar, cStar, hRe, gRe, aRe, mRe, nNeg, total, expected, biasTotal,
    sig: [L, d, F, sw, relu, bias, pre, rms, inKey].join('|'),
  };
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const L = Number(ctx.params.l);
    let step = 0, built = null, parts = [], els = null, destroyed = false;
    let auto = null, pulseT = null, advancing = false;

    // ------------------------------------------------ header, picker, caption, side column (persistent)
    const header = sceneHeader(st, { chips: [[`Layer ${L}`, 'resid'], ['MLP', 'mlp']], title: 'MLP' });
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'mlp' });
    const cap = sceneCaption(st, {});
    const side = h('div', { class: 'ml-side' });
    st.add(side, { x: ZONES.side.x, y: SIDE_Y });
    const fans = s('g', { class: 'ml-fans' });            // every fan line and arrow lives in this group
    st.svg.append(fans);

    /** Build the static parts for one configuration: strips, neuron grid, fans, operations, labels. */
    function build(D) {
      parts.forEach((p) => p.remove());
      parts = [];
      fans.replaceChildren();
      const place = (node, x, y, anchor = 'c') => { parts.push(st.add(node, { x, y, anchor })); return node; };
      const { d, F } = D;
      const cols = Math.ceil(Math.sqrt(F)), rows = Math.ceil(F / cols);
      const pitch = clamp(Math.floor(MAX_GRID / Math.max(cols, rows)), 4, 22), dia = Math.max(3, Math.round(pitch * 0.72));
      const bw = cols * pitch + 2 * PAD, bh = rows * pitch + 2 * PAD;
      const gl = GX - bw / 2, gt = GY - bh / 2, gr = gl + bw, gb = gt + bh;
      const OP1 = gl - 150, OP2 = gr + 150;
      const inTip = OP1 - OP / 2 - 3, outStart = OP2 + OP / 2 + 3;   // the arrows stop 3 units short of their operation
      const at = (f) => ({ x: gl + PAD + (f % cols) * pitch + pitch / 2, y: gt + PAD + Math.floor(f / cols) * pitch + pitch / 2 });

      // input vector (left) and output vector (right), each with its symbol and its size
      const inS = strip(d, 'norm');
      inS.el.dataset.trace = D.inKey;
      inS.el.title = `input to the MLP: trace ${D.inKey}`;
      place(inS.el, VX, GY, 'c');
      place(texLabel(D.pre ? '\\hat{x}' : 'x^{\\text{mid}}', { size: 22, kind: 'norm' }), VX, GY - inS.height / 2 - 12, 'bc');
      place(h('div', { class: 'ml-tag' }, `d = ${d}`), VX, GY + inS.height / 2 + 12, 'tc');
      const outS = strip(d, 'mlp');
      outS.el.dataset.trace = `L${L}.mlp.out`;
      // the label of the largest output (the focus of step 4), used when the strip elides that cell: see paint()
      const focus = h('div', { class: 'ml-focus', 'aria-hidden': 'true' });
      outS.el.append(focus);
      place(outS.el, OX, GY, 'c');
      place(texLabel('m', { size: 22, kind: 'mlp' }), OX, GY - outS.height / 2 - 12, 'bc');
      place(h('div', { class: 'ml-tag' }, `d = ${d}`), OX, GY + outS.height / 2 + 12, 'tc');

      // the neuron grid: one dot per hidden unit. It is the zoom target activation.<l>.
      const dots = [];
      const grid = h('div', { class: 'ml-grid zoomable', style: { width: `${bw}px`, height: `${bh}px` }, dataset: { node: `activation.${L}` } },
        h('span', { class: 'k-zoom-badge', 'aria-hidden': 'true', html: ZOOM_ICON }));
      for (let f = 0; f < F; f++) {
        const p = at(f);
        const dt = h('div', { class: 'ml-dot', style: { left: `${p.x - gl - dia / 2}px`, top: `${p.y - gt - dia / 2}px`, width: `${dia}px`, height: `${dia}px` } });
        grid.append(dt);
        dots.push(dt);
      }
      place(grid, gl, gt, 'tl');
      const tagTop = place(h('div', { class: 'ml-tag top' }), GX, gt - 14, 'bc');
      place(h('div', { class: 'ml-tag' }, `F = ${F} hidden units`), GX, gb + 14, 'tc');
      place(h('div', { class: 'ml-tag ml-illus' }, 'illustration · one line per unit, not per weight'), GX, gb + 38, 'tc');
      // the layer's size, shown once under the hero: 2·d·F weights (3·d·F with SwiGLU). paint() fills it.
      const params = place(h('div', { class: 'ml-params' }), GX, gb + 62, 'tc');

      // the down-projection fan is the zoom target neurons.<l>: its hit area, and a label under it that reads at rest
      const fanW = OP2 - gr - 22;
      place(h('div', { class: 'ml-fanhit', dataset: { node: `neurons.${L}` }, style: { width: `${fanW}px`, height: `${bh - 16}px` } }), gr + 8, gt + 8, 'tl');
      place(h('div', { class: 'ml-fanlab', dataset: { node: `neurons.${L}` } }, 'neurons', h('span', { class: 'ml-glyph', 'aria-hidden': 'true', html: ZOOM_ICON })), gr + 8 + fanW / 2, gb + 14, 'tc');

      // operations and arrows: input → × W_up → fan → grid → fan → × W_down → output
      const opUp = place(op('\\times', { kind: 'mlp', size: OP }), OP1, GY, 'c');
      const opDown = place(op('\\times', { kind: 'mlp', size: OP }), OP2, GY, 'c');
      opUp.classList.add('ml-op');
      opDown.classList.add('ml-op');
      const arrow1 = arrowSvg(VX + 14, inTip, GY, 'norm');
      const arrow2 = arrowSvg(outStart, OX - 15, GY, 'mlp');
      fans.append(arrow1, arrow2);
      const inLabel = D.sw ? '\\Wup,\\ \\Wgate' : `\\Wup${D.bias ? ' + b_{\\text{up}}' : ''}`;
      place(texLabel(inLabel, { size: 18, kind: 'mlp' }), (VX + 14 + inTip) / 2, GY - 14, 'bc');
      place(texLabel(`\\Wdown${D.bias ? ' + b_{\\text{down}}' : ''}`, { size: 18, kind: 'mlp' }), (outStart + OX - 15) / 2, GY - 14, 'bc');

      // fans: one thin line per unit. Up runs from the operation to the unit, down from the unit to the operation.
      // Each line is drawn with a dash pattern as long as itself, so it can grow from its start.
      const up = [], down = [], upLen = [], downLen = [];
      for (let f = 0; f < F; f++) {
        const p = at(f);
        const lu = Math.hypot(p.x - OP1, p.y - GY), ld = Math.hypot(OP2 - p.x, GY - p.y);
        const pu = s('path', { class: 'ml-line', d: `M${OP1} ${GY} L${p.x} ${p.y}`, style: { strokeDasharray: `${lu} ${lu}`, strokeDashoffset: `${lu}px` } });
        const pd = s('path', { class: 'ml-line', d: `M${p.x} ${p.y} L${OP2} ${GY}`, style: { strokeDasharray: `${ld} ${ld}`, strokeDashoffset: `${ld}px` } });
        fans.append(pu, pd);
        up.push(pu);
        down.push(pd);
        upLen.push(lu);
        downLen.push(ld);
      }
      els = { cols, F, d, inS, outS, dots, up, down, upLen, downLen, tagTop, opUp, opDown, arrow2, focus, params };
    }

    /** Paint step i for the selected token. Transitions carry the change; animate:false jumps. */
    function paint(D, i, { animate = true } = {}) {
      const { t, F, d, sw, relu, gelu, pre, rms, resid, bias, up, act, out, xin, rUp, rAct, rOut, rIn, rank, fStar, cStar, hRe, gRe, aRe, mRe, nNeg, total, expected, biasTotal } = D;
      const { cols } = els;
      const tok = esc(ctx.label(t) === '␣' ? 'space' : ctx.label(t));    // the space token reads as “space”, as elsewhere
      const n = n2;
      st.root.classList.toggle('ml-still', !animate || reducedMotion());

      // up-projection lines draw in, column by column, from the operation to every unit
      els.up.forEach((p, f) => {
        p.style.transitionDelay = `${Math.round(((f % cols) * 700) / cols)}ms, 0ms`;
        p.style.strokeDashoffset = i >= 1 ? '0px' : `${els.upLen[f]}px`;
        p.style.strokeOpacity = i >= 2 ? '0.12' : '0.2';
      });
      // down-projection lines converge on the output operation, strongest contributions first
      els.down.forEach((p, f) => {
        p.style.transitionDelay = `${Math.round((rank[f] * 600) / F)}ms, 0ms`;
        p.style.strokeDashoffset = i >= 3 ? '0px' : `${els.downLen[f]}px`;
        p.style.strokeOpacity = i >= 3 ? (0.08 + (0.42 * Math.abs(act[f])) / rAct).toFixed(3) : '0.12';
      });
      // units: pre-activation colours after the up-projection, then the activations, lit strongest first.
      // The strongest unit carries the accent ring in steps 1 and 2 only: step 3 is about the output.
      els.dots.forEach((dt, f) => {
        const cur = i === 0 ? null : i === 1 ? up[f] : act[f];
        dt.dataset.cell = `${t},${f},0`;
        dt.dataset.trace = i <= 1 ? `L${L}.mlp.up` : `L${L}.mlp.act`;
        dt.classList.toggle('hl', i >= 1 && i < 3 && f === fStar);
        if (cur === null) {
          dt.style.background = '';
          dt.style.boxShadow = '';
          dt.style.transitionDelay = '0ms';
          return;
        }
        const col = diverging(cur / (i === 1 ? rUp : rAct));
        const delay = i === 1 ? Math.round(((f % cols) * 700) / cols) : i === 2 ? Math.round(rank[f] * Math.min(4, 900 / F)) : 0;
        const strong = i >= 2 && Math.abs(act[f]) >= 0.5 * rAct;
        dt.style.background = col;
        dt.style.transitionDelay = `${delay}ms`;
        dt.style.boxShadow = strong ? `0 0 0 2px color-mix(in srgb, ${col} 35%, transparent), 0 0 12px 2px color-mix(in srgb, ${col} 65%, transparent)` : '';
      });

      // the input vector, then the output vector filling in after the down-projection
      paintStrip(els.inS, xin, t, { range: rIn });
      els.inS.el.style.opacity = i === 0 ? '1' : '0.6';
      if (i >= 3) paintStrip(els.outS, out, t, { range: rOut, hl: cStar, delay: (k) => 560 + Math.round((k * 700) / d) });
      else paintStrip(els.outS, out, t, { range: rOut, filled: false });
      // the largest output is the focus of step 4. If the strip elides it, the gap cell takes the ring, the hover and a label.
      const gp = els.outS.idx.indexOf(-1);
      const gapEl = gp >= 0 ? els.outS.cells[gp] : null;
      const focusHidden = i >= 3 && gapEl !== null && !els.outS.idx.includes(cStar);
      if (gapEl) {
        gapEl.classList.toggle('hl', focusHidden);
        if (focusHidden) { gapEl.dataset.cell = `${t},${cStar},0`; gapEl.title = `${cStar}: ${fmt(out[cStar], 4)}`; }
        else delete gapEl.dataset.cell;
      }
      if (focusHidden) {
        els.focus.style.top = `${6 + gp * (CELL + GAP) + CELL / 2}px`;   // strip border and padding, then the gap cell's centre
        els.focus.replaceChildren(`cell ${cStar}`, h('span', null, minus(n2(out[cStar]))));
      }
      els.focus.classList.toggle('on', focusHidden);
      els.opUp.style.opacity = i >= 1 ? '1' : '0.45';
      els.opDown.style.opacity = i >= 3 ? '1' : '0.45';
      els.arrow2.style.opacity = i >= 3 ? '1' : '0.3';
      els.tagTop.textContent = ['hidden units', sw ? 'u and g · projections' : 'h · up-projection', gelu ? 'a = GELU(h)' : relu ? 'a = ReLU(h)' : 'a = SiLU(g) ⊙ u', 'a · read by W_down'][i];

      // one equation for this step, its live instance on the strongest unit or output, and the checks that tie them to the trace
      const f = fStar, c = cStar;
      const xs = pre ? '\\hat{x}' : 'x^{\\text{mid}}';
      const normOp = rms ? '\\RMSNorm' : '\\LN', normWord = rms ? 'RMSNorm' : 'LayerNorm';
      const actOp = gelu ? '\\GELU' : '\\ReLU';
      const bUp = bias ? ' + b_{\\text{up}}' : '', bDown = bias ? ' + b_{\\text{down}}' : '';
      const mains = [
        pre ? `\\hat{x} = ${normOp}(x^{\\text{mid}})` : `x^{\\text{mid}} = ${normOp}_1(${resid ? '\\xres + \\Attn(\\xres)' : '\\Attn(\\xres)'})`,
        sw ? `\\binom{u}{g} = \\binom{\\Wup}{\\Wgate}\\,${xs}${bias ? ' + \\binom{b_{\\text{up}}}{b_{\\text{gate}}}' : ''}` : `h = \\Wup\\,${xs}${bUp}`,
        gelu ? 'a = \\GELU(h)' : relu ? 'a = \\ReLU(h)' : 'a = \\SiLU(g)\\odot u',
        `m = \\Wdown\\,a${bDown}`,
      ][i];
      const live = [];
      let checks = [];
      if (i === 0) {
        live.push(`${xs} = (${xin.slice(0, 3).map(n).join(',\\ ')}${d > 3 ? ',\\ \\dots' : ''})`, `\\in \\R^{${d}}`);
      } else if (i === 1) {
        live.push(sw
          ? `\\binom{u_{${f}}}{g_{${f}}} = \\binom{${n3(hRe)}}{${n3(gRe)}}`
          : `h_{${f}} = ${xs}\\cdot W^{\\text{up}}_{${f}}${bias ? ` + b_{${f}}` : ''} = ${n3(hRe)}`);
        checks = sw
          ? [check([[hRe, up[f]], [gRe, D.gate[f]]], `equals L${L}.mlp.up and L${L}.mlp.gate`)]
          : [check([[hRe, up[f]]], `equals L${L}.mlp.up`)];
      } else if (i === 2) {
        // the printed argument (three decimals) gives the printed result: "=" when it does, "≈" when rounding shifts a digit
        const shown = sw ? siluScalar(r3(gRe)) * r3(hRe) : relu ? Math.max(0, r3(hRe)) : geluScalar(r3(hRe));
        const eq = n2(shown) === n2(aRe) ? '=' : '\\approx';
        live.push(sw
          ? `\\begin{aligned} a_{${f}} &= \\SiLU(g_{${f}})\\,u_{${f}} \\\\ &= \\SiLU(${n3(gRe)})\\,(${n3(hRe)}) \\\\ &${eq} ${n2(aRe)} \\end{aligned}`
          : `\\begin{aligned} a_{${f}} &= ${actOp}(h_{${f}}) \\\\ &= ${actOp}(${n3(hRe)}) ${eq} ${n2(aRe)} \\end{aligned}`);
        checks = [check([[aRe, act[f]]], `equals L${L}.mlp.act`)];
      } else {
        live.push(`m_{${c}} = a\\cdot W^{\\text{down}}_{${c}}${bias ? ` + b_{${c}}` : ''} = ${n2(mRe)}`);
        checks = [check([[mRe, out[c]]], `equals L${L}.mlp.out`)];
      }

      // parameters: 2·d·F weights (3·d·F with SwiGLU), counted from the model's own tensors. They sit under the hero, not
      // in the side column, which keeps to this step: its equation and its check. The per-matrix sizes and the bias count go in the tooltip.
      const sym = sw ? '3' : '2';
      const perMatrix = [`W_up ${F} × ${d} = ${tf(F * d)}`, sw ? `W_gate ${F} × ${d} = ${tf(F * d)}` : null, `W_down ${d} × ${F} = ${tf(d * F)}`].filter(Boolean);
      els.params.title = [...perMatrix, bias ? `Biases add ${tf(biasTotal)} more, outside ${sym}·d·F.` : null].filter(Boolean).join('\n');
      els.params.replaceChildren(
        h('span', null, `${sym}·d·F = ${sym} × ${d} × ${F} = `, h('b', null, `${tf(expected)} weights`)),
        checkBadge(total, expected, { label: 'matches the model’s tensors' }));
      // the side column: the equation for this step and its check
      side.replaceChildren(
        h('div', { class: 'ml-kick' }, STEPS[i]),
        h('div', { class: 'ml-main', html: texHTML(mains, { display: true }) }),
        h('div', { class: 'ml-live' }, ...live.map((tex) => h('div', { html: texHTML(tex, { display: true }) }))),
        h('div', { class: 'ml-checks' }, ...checks));

      // the residual wording: the stream after the block is m itself without the residual; with post-LN it is normalised
      const resWord = !resid && pre ? 'replaces the stream, because the residual is off'
        : !resid ? 'replaces the stream, and ln2 normalises it'
          : pre ? 'is added to the residual stream' : 'is added to the stream and normalised again';
      const where = pre ? `normalised by ln2 (${normWord})` : `already normalised by ln1 (${normWord})`;
      const words = [
        `The MLP reads one token at a time. Its input is the d = <b>${d}</b> vector of “<b>${tok}</b>” at position <b>${t}</b>, ${where}.`,
        sw ? `W_up and W_gate read the same input, so one fan stands for both. The grid colours each unit by its value <b>u</b>; the gate <b>g</b> of unit <b>${f}</b> is in the equation.`
          : `W_up expands the input to F = <b>${F}</b> hidden units: one dot product with the whole vector${bias ? ', plus a bias' : ''}. Each line is one unit, not one weight.`,
        gelu ? `GELU is the only non-linear step in the MLP. Units are lit from the strongest |a| down; the largest is unit <b>${f}</b>, and the <b>${nNeg}</b> negative values now sit in ${GELU_RANGE}.`
          : relu ? `ReLU is the only non-linear step: the <b>${nNeg}</b> negative pre-activations become exactly 0. Units are lit from the strongest |a| down.`
            : `SwiGLU multiplies two units: SiLU of the gate decides how much of each up-value passes. Units are lit from the strongest |a| down; the largest is unit <b>${f}</b>.`,
        `W_down mixes the ${F} activations back into ${d} numbers; the result, <b>m</b>, ${resWord}. The ringed cell, <b>${c}</b>, is the largest in magnitude.`,
      ];
      header.setSub(`Token “<b>${tok}</b>” at position <b>${t}</b> goes from d = ${d} to F = ${F} hidden units and back. The same network runs on every position, separately.`);
      cap.set(words[i]);
    }

    /** Make the DOM match the current trace (rebuilding when the shapes change), then paint the current step. */
    function refresh({ animate = true } = {}) {
      if (destroyed) return;
      const D = dataFor(ctx, L);
      if (!D) { cap.set('This layer has no MLP trace for the selected token.'); return; }
      if (!els || built !== D.sig) {
        build(D);
        built = D.sig;
        requestAnimationFrame(() => { if (!destroyed) refresh({ animate: true }); });   // let the new parts lay out, then animate
        return;
      }
      paint(D, step, { animate });
    }

    /** The first-view explanation ends on any pointer press in the scene, or on any step the reader picks. */
    function stopAuto() { clearTimeout(auto); auto = null; }

    function setStep(i) {
      if (!advancing) stopAuto();     // a story-bar dot, an arrow or a key: the reader is choosing the step now
      step = i;
      refresh({ animate: true });
      if (!els) return;
      clearTimeout(pulseT);
      if (i === 1) pulse(els.opUp);
      if (i === 3) { pulse(els.opDown); pulseT = setTimeout(() => pulse(els.outS.el), 1100); }
    }

    /** One step per MOTION.step, from the first view, until the reader takes over. */
    function autoNext() {
      auto = null;
      if (destroyed || step >= STEPS.length - 1) return;
      advancing = true;
      try { ctx.setStep(step + 1); } finally { advancing = false; }
      auto = setTimeout(autoNext, MOTION.step);
    }

    ctx.setSteps(STEPS, setStep);
    const offs = [ctx.on('trace', () => refresh({ animate: true })), ctx.on('select', () => refresh({ animate: true }))];
    const offTheme = onThemeChange(() => refresh({ animate: false }));
    refresh({ animate: false });
    appear([header.el, picker.el, side], { stagger: MOTION.stagger });

    // explain itself the first time the scene is shown (not under reduced motion)
    if (!reducedMotion()) {
      auto = setTimeout(autoNext, 900);
      el.addEventListener('pointerdown', stopAuto, { once: true });
      el.addEventListener('atlas:interrupt', stopAuto, { once: true });
    }

    return {
      destroy() {
        destroyed = true;
        stopAuto();
        clearTimeout(pulseT);
        offs.forEach((f) => f());
        offTheme();
        st.destroy();
      },
    };
  },
};

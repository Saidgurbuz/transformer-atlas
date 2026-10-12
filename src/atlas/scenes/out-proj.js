// Concatenation and output projection for one layer. The heads' outputs are placed side by side
// (attn.concat), then W_O mixes them into attn.out. For the selected token, head h contributes its output
// times its own column block W_O^(h) (d x d_h); the contributions and the bias add up to attn.out[t].
// Composition (docs/STYLE_GUIDE.md section 7): steps 0 to 2 are a pipeline; step 3 is focus plus context. The
// token strips are the focus, the side column is the whole attn.out with the query row highlighted, and
// attn.concat and W_O stay on screen, dimmed, as the matrices the contributions were computed with.
// Recorded deviation (STYLE_GUIDE section 2): the step equation sits under the pipeline or the strips, not in the
// side column. Its numeric line can hold up to seven signed terms, which does not fit the 340-unit side column on
// one line at a readable size. The sum reads best directly under the strips it adds up.
// Not tilted: the pipeline is laid out flat, and a tilt would move the edges that the operators sit on. W_O carries a
// title with its shape instead of a maths lookup, since the trace has no key for a weight.
// Colour: head outputs are L{l}.attn.ctx, kind v (green) as src/core/describe.js catalogues them, each head with its
// own tint for identity. attn.concat keeps kind attn; W_O and attn.out are kind o. The accent marks the query row
// only; the feature being summed is outlined in neutral.
// Matrices: DOM cells while a cell is 4 units or more and the matrix has at most 2,500 cells; a canvas heatmap
// otherwise, and always for W_O (the kit's tool for large tensors). Below 7 units there is no gap between cells, so
// DOM and canvas matrices look alike.
// Motion cost: a token selection moves no box, so it costs no layout. Only boxes that jump are flushed, once.
// Reference-style scene: see docs/ATLAS_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, mat, vec, op, chip, checkBadge, appear, useStyle, MOTION, ZONES } from '../kit.js';
import { h, texHTML, heatmap, clamp } from '../../ui/index.js';

const STEPS = ['Head outputs', 'Concatenate', 'Mix with W_O', 'Who contributed'];
const HERO_L = ZONES.hero.x, HERO_W = ZONES.hero.w;      // the hero: x 110 to 1490 (steps 0 to 2 use all of it)
const SIDE_X = ZONES.side.x;                              // side column: attn.out in step 3, the check in steps 1 and 2
const CY = 490;                                           // centre line of the hero matrices; W_O's tag clears the token picker
const HDR = 40;                                           // above each matrix: tag, coloured band, air
const PAD = 9;                                            // from a matrix element's left edge to its first cell (padding 8 + border 1)
const OPD = 52, OPG = 60, OPG_MIN = 36;                   // operation node size; its gap to a matrix (OPG_MIN only when the pipeline is tight)
const GAP_HEADS = 150, GAP_MIN = 8;                       // gap between head blocks in step 0 (up to GAP_HEADS, so the heads fill the hero)
const LABEL_W = 120;                                      // the strips' row labels (attn.out[21] fits)
const ROW_L = ZONES.hero.x, ROW_TOP = 300;                // token strips (step 3)
const ROW_W = 18 + LABEL_W + 10 + 282 + 22 + 290;         // a strip row: sign, label, strip (17 slots), bars
const SIDE_TOP = 282;                                     // side column top, level with the context's tags
const CTX_X = 905, CTX_Y = 282, CTX_GAP = 30;             // step 3 context: dimmed attn.concat over W_O, between the strips and the side column
const CTX_BOTTOM = 770, CTX_RIGHT = 1140;                 // the context stays above the caption and clear of the side column
const EQ_Y = 700, EQ_W = 1000;                            // the step equation under the pipeline or the strips (two lines)
const BAR_MAX = 170;
const VIS_HEAD = 12, VIS_TAIL = 4;                        // the elision the kit's vec() uses
const SHOW_HEADS = 6;                                     // step 3: up to six head rows; with more, the weakest are summed into one row
const CANVAS_CELLS = 2500;                                // a matrix above this many cells is a canvas heatmap (ATLAS_GUIDE: mat() up to about 2,500)
const SIDE_MAX = 80;                                      // d above this: the side view shows the first 12 and last 4 features
const SIDE_MIN_H = 24;                                    // the side view is at least this tall (a one-token attn.out is a single row)
const M = MOTION;

// Cells are 5 units with no gap below 7 units, as head.js does; a gap is 1 unit from 7 units up.
const gapOf = (p) => (p >= 7 ? 1 : 0);
const boxOf = (p, n) => n * p - gapOf(p) + 18;            // matrix width or height for n cells (padding and border)
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
// One tint per head of the head-output colour (kind v), from full green to a lighter tint, mixed toward the
// text grey so both themes keep contrast.
const headColor = (hh, nH) => `color-mix(in srgb, var(--c-v) ${Math.round(100 - (hh * 55) / Math.max(1, nH - 1))}%, var(--fg-3))`;
// the visible slots of a vector of length n, elided the way vec() elides (-1 is the ellipsis)
const stripIdx = (n) => (n <= VIS_HEAD + VIS_TAIL + 1 ? [...Array(n).keys()] : [...Array(VIS_HEAD).keys(), -1, ...Array.from({ length: VIS_TAIL }, (_, i) => n - VIS_TAIL + i)]);
// the side view's columns: every feature up to SIDE_MAX, else the first 12, an ellipsis (-1) and the last 4
const sideIdxOf = (d) => (d > SIDE_MAX ? [...Array(VIS_HEAD).keys(), -1, ...Array.from({ length: VIS_TAIL }, (_, i) => d - VIS_TAIL + i)] : [...Array(d).keys()]);
const sideVals = (out, T, d, idx) => {
  const n = idx.length, v = new Float64Array(T * n);
  for (let t = 0; t < T; t++) for (let c = 0; c < n; c++) v[t * n + c] = idx[c] < 0 ? 0 : out[t * d + idx[c]];
  return v;
};
// The decimals of one printed line: three significant figures on its largest value, and at least three decimals.
const decimalsFor = (big) => clamp(2 - Math.floor(Math.log10(Math.max(1e-9, big))), 3, 9);

// The printed sum of step 3. The terms are rounded by largest remainders, so the printed terms add up exactly to the
// printed total, which is the traced value rounded the same way. Each printed term is within one unit of its value.
function sumLine(terms, total) {
  const big = Math.max(1e-9, Math.abs(total), ...terms.map((x) => Math.abs(x)));
  const k = decimalsFor(big);
  const u = 10 ** k;
  const xs = terms.map((x) => x * u);
  const f = xs.map((x) => Math.floor(x));
  const N = Math.round(total * u);
  const order = xs.map((_, i) => i).sort((a, b) => (xs[b] - f[b]) - (xs[a] - f[a]));
  let extra = N - f.reduce((acc, v) => acc + v, 0);
  for (let m = 0; extra > 0 && m < order.length; m++, extra--) f[order[m]] += 1;
  for (let m = order.length - 1; extra < 0 && m >= 0; m--, extra++) f[order[m]] -= 1;
  const word = (n, first) => {
    const s = (Math.abs(n) / u).toFixed(k);
    if (first) return n < 0 ? `-${s}` : s;
    return n < 0 ? `- ${s}` : `+ ${s}`;
  };
  return `${f.map((n, i) => word(n, i === 0)).join(' ')} \\approx ${N < 0 ? '-' : ''}${(Math.abs(N) / u).toFixed(k)}`;
}

useStyle('out-proj', `
.oc-wrap { position: absolute; left: 0; top: 0; width: 0; height: 0; }
.oc-grp, .oc-op { position: absolute; transition: left ${M.move}ms ${M.ease}, top ${M.move}ms ${M.ease}, opacity ${M.appear}ms ${M.easeOut}, visibility ${M.appear}ms; }
.oc-row, .oc-sep { position: absolute; transition: top ${M.move}ms ${M.ease}, opacity ${M.appear}ms ${M.easeOut}, visibility ${M.appear}ms; }
.oc-row { display: flex; align-items: center; height: 30px; }
.oc-hidden { opacity: 0; visibility: hidden; }
.oc-grp.oc-dim { opacity: .3; pointer-events: none; }
.oc-grp > .k-mat { display: flex; width: max-content; align-items: flex-start; }
.oc-grp > .oc-hdr { transition: opacity .3s ease; }
.oc-grp.oc-merged > .oc-hdr { opacity: 0; transition: opacity .3s ease .6s; }
.oc-grp.oc-merged > .k-mat > .k-mat-body { box-shadow: none; background: transparent; border-color: transparent; transition: box-shadow .3s ease .6s, background-color .3s ease .6s, border-color .3s ease .6s; }
.oc-hdr { display: flex; flex-direction: column; gap: 6px; height: 40px; padding-bottom: 10px; box-sizing: border-box; }
.oc-tag { font: 600 15px/18px var(--font-mono); height: 18px; color: var(--kind, var(--fg-2)); white-space: nowrap; }
.oc-band { position: relative; height: 6px; border-radius: 3px; }
.oc-seg { position: absolute; top: 0; bottom: 0; border-radius: 3px; background: var(--hc); }
.oc-div { position: absolute; top: 8px; bottom: 8px; width: 2px; border-radius: 1px; background: var(--line-strong); }
.oc-note { margin-top: 8px; font: 500 14px/1.3 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.oc-op { display: grid; }
.oc-sign { width: 18px; font: 600 18px/1 var(--font-mono); color: var(--fg-3); }
.oc-rl { width: ${LABEL_W}px; font: 600 15px/1 var(--font-mono); color: var(--kind, var(--fg-2)); white-space: nowrap; }
.oc-strip { margin-left: 10px; }
.oc-bars { display: flex; align-items: center; gap: 10px; width: 290px; height: 14px; margin-left: 22px; }
.oc-bar { height: 12px; width: 0; border-radius: 6px; transition: width ${M.move}ms ${M.ease}; }
.oc-norm { font: 600 14px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.oc-sep { height: 1px; background: var(--line-strong); }
.oc-j { outline: 2px solid color-mix(in srgb, var(--fg-2) 60%, transparent); outline-offset: 1px; }
.oc-side .k-cell { height: var(--oc-ch, var(--cell)); }
.oc-side .k-cell.ell { overflow: visible; font-size: 11px; line-height: 1; }
.oc-eq { width: ${EQ_W}px; font-size: 22px; display: grid; gap: 6px; }
.oc-eq .katex-display { margin: 0; text-align: left; }
.oc-eq-line, .oc-eq-live { white-space: nowrap; overflow-x: auto; overflow-y: hidden; }
.oc-eq-live { font-size: 19px; }
.oc-big .heatmap-title { display: none; }
.oc-big .heatmap { display: block; max-width: none; }
.oc-big .heatmap canvas { max-width: none; }
@media (prefers-reduced-motion: reduce) { .oc-grp, .oc-op, .oc-row, .oc-sep, .oc-bar { transition: none !important; } }
`);

// A matrix with its header: a tag line and a band of colour segments (one per head where the columns are heads).
function group(label, kind, bw, body, segs) {
  const band = h('div', { class: 'oc-band', style: { width: `${bw}px` } },
    ...segs.map((s) => h('span', { class: 'oc-seg', style: { left: `${s.left}px`, width: `${s.width}px`, '--hc': s.color } })));
  const tag = h('div', { class: 'oc-tag', style: { '--kind': `var(--c-${kind})` } }, label);
  return h('div', { class: 'oc-grp', style: { width: `${bw}px` } }, h('div', { class: 'oc-hdr' }, tag, band), body);
}

// A matrix at pitch p: DOM cells (the kit's mat) while the cells are 4 units or more and the matrix has at most
// CANVAS_CELLS cells; otherwise one canvas heatmap. W_O (force) is always a heatmap. Both give the element, its
// .k-mat-body (for overlays), update(values, range), highlightRow(r) and destroy().
function matrix(data, rows, cols, { p, range, kind, trace, cellOf, force = false }) {
  if (!(force || p < 4 || rows * cols > CANVAS_CELLS)) {
    const g = gapOf(p);
    const m = mat(data, rows, cols, { cell: p - g, gap: g, mode: 'diverging', range, kind, trace, cellOf });
    return {
      el: m, body: m.querySelector('.k-mat-body'),
      update: (vals, r) => m.update(vals, { range: r }),
      highlightRow: (r) => m.highlightRow(r),
      destroy: () => m.destroy?.(),
    };
  }
  const hm = heatmap({
    data, rows, cols, mode: 'diverging', range, cell: p, cellH: p, minCell: 1, maxCell: 64, showValues: false,
    tooltip: () => '', traceKey: trace || null, ariaLabel: trace || 'matrix',
    // the hovered cell is named on the canvas, so the atlas hover card and the right-click maths read its numbers
    onHover: (r, c) => { if (cellOf) hm.canvas.dataset.cell = cellOf(r, c); },
    onLeave: () => { delete hm.canvas.dataset.cell; },
  });
  const body = h('div', { class: 'k-mat-body' }, hm.el);
  const el = h('div', { class: ['k-mat', 'oc-big', kind && `kind-${kind}`], style: { '--kind': kind ? `var(--c-${kind})` : null } }, body);
  return {
    el, body,
    update: (vals, r) => hm.update({ data: vals, range: r ?? range }),
    highlightRow: (r) => hm.update({ highlight: { rows: [r] } }),
    destroy: () => hm.destroy(),
  };
}

const rowLabel = (text, color) => h('span', { class: 'oc-rl', style: { '--kind': color } }, text);

// One token strip with its label, sign and trailing elements (bars or the check).
function rowEl({ sign, label, strip, bars, top }) {
  strip.classList.add('oc-strip');
  return h('div', { class: 'oc-row', style: { left: `${ROW_L}px`, top: `${top}px` } },
    h('span', { class: 'oc-sign' }, sign), label, strip, h('span', { class: 'oc-bars' }, ...bars));
}

// The pitch of step 3's context: attn.concat over W_O at 2 units per cell when they fit above the caption and beside
// the strips, else 1 unit per cell.
function ctxPitch(T, d) {
  for (const pt of [2, 1]) {
    const bottom = CTX_Y + HDR + boxOf(pt, T) + CTX_GAP + HDR + boxOf(pt, d);
    if (bottom <= CTX_BOTTOM && CTX_X + boxOf(pt, d) <= CTX_RIGHT) return pt;
  }
  return 1;
}

// Pitch of the hero matrices and the sizes the layout needs. The pipeline must fit the hero: the operator gaps
// shrink first, then the pitch.
function geometry(T, d, nH, dh) {
  const fits = (p, opg) => 3 * boxOf(p, d) + 2 * OPD + 4 * opg <= HERO_W && nH * boxOf(p, dh) + (nH - 1) * GAP_MIN <= HERO_W;
  let p = clamp(Math.floor(330 / Math.max(T, d)), 1, 8), opg = OPG;
  while (!fits(p, opg) && (opg > OPG_MIN || p > 1)) {
    if (opg > OPG_MIN) opg = OPG_MIN;
    else { p--; opg = OPG; }
  }
  const hw = boxOf(p, dh);
  const gh = nH > 1 ? clamp((HERO_W - nH * hw) / (nH - 1), GAP_MIN, GAP_HEADS) : 0;
  const sideIdx = sideIdxOf(d), nS = sideIdx.length;
  const sp = Math.min(clamp(Math.floor(330 / T), 4, 8), Math.floor(322 / nS));
  return { p, g: gapOf(p), opg, hw, gh, sideIdx, sp, sh: Math.max(sp, Math.ceil(SIDE_MIN_H / T)), pt: ctxPitch(T, d) };
}

// Writes that do not change anything are skipped, so a refresh with nothing new touches no style.
function setFlag(el, name, on) { if (el.classList.contains(name) !== on) el.classList.toggle(name, on); }
function setDelay(el, delay) { if (el.style.transitionDelay !== delay) el.style.transitionDelay = delay; }

// Position a box and set its state. A move without animation is a jump: it gets no transition, and the jumps are
// flushed once at the end of layout(). A box that does not move is not written at all.
function place(el, x, y, { on = false, dim = false, merged = false, delay = '', animate = true, jumps }) {
  const L = `${x}px`, T = `${y}px`;
  if (el.style.left !== L || el.style.top !== T) {
    if (!animate) { el.style.transition = 'none'; jumps.push(el); }
    el.style.left = L;
    el.style.top = T;
  }
  setFlag(el, 'oc-hidden', !on);
  setFlag(el, 'oc-dim', on && dim);
  setFlag(el, 'oc-merged', on && merged);
  setDelay(el, on && delay ? delay : '');
}

function show(el, on, delay = '') {
  setFlag(el, 'oc-hidden', !on);
  setDelay(el, on ? delay : '');
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const L = Number(ctx.params.l);
    let step = 0, D = null, G = null, sig = '', ready = false, lastQ = -1;
    let mats = { heads: [], cat: null, w: null, out: null, side: null, ctxCat: null, ctxW: null };
    let groups = { heads: [], cat: null, w: null, out: null, side: null, ctxCat: null, ctxW: null };
    let strips = { heads: [], rest: null, bias: null, out: null, vecs: [] };
    let made = [], ops = [], rowEls = [], sepEl = null, outSlot = null;
    let auto = null, autoStep = false;

    const wrap = h('div', { class: 'oc-wrap' });            // all matrices, ops and strips live in this box
    st.add(wrap, { x: 0, y: 0 });

    // ------------------------------------------------------------ shared parts and text
    const header = sceneHeader(st, { chips: [[`Layer ${L}`, 'resid']], title: ctx.node.title });
    const picker = tokenPicker(st, ctx, { label: 'token' });
    const eq = h('div', { class: 'oc-eq' });
    st.add(eq, { x: ROW_L, y: EQ_Y });
    const check = h('div');
    st.add(check, { x: SIDE_X, y: EQ_Y + 6 });
    const cap = sceneCaption(st, {});

    // ------------------------------------------------------------ data for the selected token
    function load() {
      const S = ctx.state, T = S.ids.length, cfg = S.config;
      if (L >= cfg.n_layer) return null;
      const tr = S.trace || {}, params = S.model?.params || {};
      const nH = cfg.n_head, nKV = cfg.n_kv_head ?? nH, d = cfg.d_model, dh = d / nH, K = nH * dh;
      const ctxT = tr[`L${L}.attn.ctx`], catT = tr[`L${L}.attn.concat`], outT = tr[`L${L}.attn.out`];
      const Wt = params[`blocks.${L}.attn.o_proj.weight`], bT = params[`blocks.${L}.attn.o_proj.bias`];
      if (!T || !ctxT || !catT || !outT || !Wt) return null;
      const q = clamp(S.sel.token ?? 0, 0, T - 1);
      const cx = ctxT.data, cat = catT.data, out = outT.data, W = Wt.data, bias = bT ? bT.data : null;
      // contribution of head h to every output feature o: sum over its dh columns of ctx[h,q,c] * W_O[o, h*dh + c]
      const contrib = [], norms = [];
      for (let hh = 0; hh < nH; hh++) {
        const c = new Float64Array(d);
        for (let o = 0; o < d; o++) {
          let acc = 0;
          for (let k = 0; k < dh; k++) acc += cx[(hh * T + q) * dh + k] * W[o * K + hh * dh + k];
          c[o] = acc;
        }
        contrib.push(c);
        norms.push(Math.hypot(...c));
      }
      // the recomputed sum against the traced attention output (row q), and the full product against attn.out
      const sum = new Float64Array(d);
      let errQ = 0;
      for (let o = 0; o < d; o++) {
        let s = bias ? bias[o] : 0;
        for (const c of contrib) s += c[o];
        sum[o] = s;
        errQ = Math.max(errQ, Math.abs(s - out[q * d + o]));
      }
      let errAll = 0;
      for (let t = 0; t < T; t++) for (let o = 0; o < d; o++) {
        let s = bias ? bias[o] : 0;
        for (let k = 0; k < K; k++) s += cat[t * K + k] * W[o * K + k];
        errAll = Math.max(errAll, Math.abs(s - out[t * d + o]));
      }
      let errCat = 0;
      for (let t = 0; t < T; t++) for (let hh = 0; hh < nH; hh++) for (let c = 0; c < dh; c++)
        errCat = Math.max(errCat, Math.abs(cat[t * K + hh * dh + c] - cx[(hh * T + t) * dh + c]));
      const maxAbs = (arr) => { let m = 0; for (let i = 0; i < arr.length; i++) m = Math.max(m, Math.abs(arr[i])); return m || 1; };
      // step 3 rows: every head when there are few; with more, the strongest heads by norm and one summed row for the rest
      const all = [...Array(nH).keys()];
      let shown = all, rest = [];
      if (nH > SHOW_HEADS) {
        shown = [...all].sort((a, b) => norms[b] - norms[a] || a - b).slice(0, SHOW_HEADS - 1).sort((a, b) => a - b);
        rest = all.filter((hh) => !shown.includes(hh));
      }
      const restVec = rest.length ? Float64Array.from({ length: d }, (_, o) => rest.reduce((acc, hh) => acc + contrib[hh][o], 0)) : null;
      const restNorm = restVec ? Math.hypot(...restVec) : 0;
      const rCtx = maxAbs(cx), rW = maxAbs(W);
      const rStrip = Math.max(maxAbs(out), ...contrib.map((c) => maxAbs(c)), bias ? maxAbs(bias) : 0, restVec ? maxAbs(restVec) : 0);
      // the feature used in the equation: the largest visible value of the output row
      const stripI = stripIdx(d);
      let j = stripI.find((i) => i >= 0);
      for (const i of stripI) if (i >= 0 && Math.abs(out[q * d + i]) > Math.abs(out[q * d + j])) j = i;
      return { T, d, nH, nKV, dh, K, q, tok: ctx.label(q), cx, cat, out, W, bias, contrib, norms, errQ, errAll, errCat, rCtx, rW, rStrip, j, stripI, shown, rest, restVec, restNorm };
    }

    // ------------------------------------------------------------ DOM (rebuilt when the shape changes)
    function destroyParts() {
      [...mats.heads, mats.cat, mats.w, mats.out, mats.side, mats.ctxCat, mats.ctxW].forEach((m) => m?.destroy?.());
      strips.vecs.forEach((v) => v.destroy?.());
      mats = { heads: [], cat: null, w: null, out: null, side: null, ctxCat: null, ctxW: null };
      strips = { heads: [], rest: null, bias: null, out: null, vecs: [] };
    }

    function removeDom() {
      made.forEach((n) => n.remove());
      destroyParts();
      made = []; ops = []; rowEls = []; sepEl = null; outSlot = null;
      groups = { heads: [], cat: null, w: null, out: null, side: null, ctxCat: null, ctxW: null };
    }

    function buildDom() {
      removeDom();
      const { T, d, nH, dh, K, bias, q, shown, rest } = D, { p, g, hw, gh, sideIdx, sp, sh, pt } = G;
      const add = (n) => { wrap.append(n); made.push(n); return n; };
      // the head segments of a band at pitch `pitch`: one per head, in the matrix's own columns
      const segsAt = (pitch) => Array.from({ length: nH }, (_, hh) => ({ left: PAD + hh * dh * pitch, width: dh * pitch - gapOf(pitch), color: headColor(hh, nH) }));
      const headTag = hw + gh >= 120;                          // the full tag fits between neighbouring heads only when there is room

      // the H head outputs (kind v, as the catalogue has L{i}.attn.ctx), each T x d_h, spread across the hero
      for (let hh = 0; hh < nH; hh++) {
        const m = matrix(D.cx.subarray(hh * T * dh, (hh + 1) * T * dh), T, dh,
          { p, range: D.rCtx, kind: 'v', trace: `L${L}.attn.ctx`, cellOf: (r, c) => `${r},${c},${hh}` });
        mats.heads.push(m);
        const tag = headTag ? `h${hh} · ${T} × ${dh}` : `h${hh}`;
        groups.heads.push(add(group(tag, 'v', hw, m.el, [{ left: PAD, width: dh * p - g, color: headColor(hh, nH) }])));
      }
      // attn.concat (T x K): the head blocks side by side
      mats.cat = matrix(D.cat, T, K, { p, range: D.rCtx, kind: 'attn', trace: `L${L}.attn.concat`, cellOf: (r, c) => `${r},${c},0` });
      groups.cat = add(group(`attn.concat · ${T} × ${K}`, 'attn', boxOf(p, K), mats.cat.el, segsAt(p)));
      // W_O (d x K) with one column block per head; always a canvas heatmap, so it has no maths key of its own
      mats.w = matrix(D.W, d, K, { p, range: D.rW, kind: 'o', force: true });
      groups.w = add(group(`W_O · ${d} × ${K}`, 'o', boxOf(p, K), mats.w.el, segsAt(p)));
      groups.w.title = `W_O · ${d} × ${K}: the output projection, one column block per head`;
      for (let hh = 1; hh < nH; hh++) mats.w.body.append(h('div', { class: 'oc-div', style: { left: `${8 + hh * dh * p - 1}px` } }));
      // attn.out (T x d), the pipeline's last matrix
      mats.out = matrix(D.out, T, d, { p, range: D.rStrip, kind: 'o', trace: `L${L}.attn.out`, cellOf: (r, c) => `${r},${c},0` });
      groups.out = add(group(`attn.out · ${T} × ${d}`, 'o', boxOf(p, d), mats.out.el, [{ left: PAD, width: d * p - g, color: 'var(--c-o)' }]));
      // the side view (step 3): attn.out with its visible features; for d above 80 the ellipsis column marks the gap
      const nS = sideIdx.length, elided = sideIdx.includes(-1);
      mats.side = matrix(sideVals(D.out, T, d, sideIdx), T, nS,
        { p: sp, range: D.rStrip, kind: 'o', trace: `L${L}.attn.out`, cellOf: (r, c) => (sideIdx[c] < 0 ? null : `${r},${sideIdx[c]},0`) });
      const sideEl = mats.side.el;
      sideEl.classList.add('oc-side');
      sideEl.style.setProperty('--oc-ch', `${sh}px`);            // a one-token side view is at least SIDE_MIN_H tall
      if (elided) {
        for (let r = 0; r < T; r++) sideIdx.forEach((f, c) => {
          if (f >= 0) return;
          const cell = sideEl.cellEl?.(r, c);
          if (cell) { cell.classList.add('ell'); cell.textContent = '⋮'; delete cell.dataset.cell; }
        });
      }
      groups.side = add(group(`attn.out · ${T} × ${d}`, 'o', boxOf(sp, nS), sideEl, [{ left: PAD, width: nS * sp - gapOf(sp), color: 'var(--c-o)' }]));
      if (elided) groups.side.append(h('div', { class: 'oc-note' }, `${nS - 1} of ${d} features shown`));
      // step 3's context, dimmed: attn.concat and W_O again at the context pitch
      mats.ctxCat = matrix(D.cat, T, K, { p: pt, range: D.rCtx, kind: 'attn', trace: `L${L}.attn.concat`, cellOf: (r, c) => `${r},${c},0` });
      groups.ctxCat = add(group(`attn.concat · ${T} × ${K}`, 'attn', boxOf(pt, K), mats.ctxCat.el, segsAt(pt)));
      mats.ctxW = matrix(D.W, d, K, { p: pt, range: D.rW, kind: 'o', force: true });
      groups.ctxW = add(group(`W_O · ${d} × ${K}`, 'o', boxOf(pt, K), mats.ctxW.el, segsAt(pt)));
      groups.ctxW.title = `W_O · ${d} × ${K}: the output projection, one column block per head`;
      for (let hh = 1; hh < nH; hh++) mats.ctxW.body.append(h('div', { class: 'oc-div', style: { left: `${8 + hh * dh * pt - 1}px` } }));

      ops = [op('\\times', { kind: 'o', size: OPD }), op('=', { kind: 'o', size: OPD, tex: false })].map((o) => {
        o.classList.add('oc-op');
        return add(o);
      });

      // token strips (step 3): a slot per shown head, the summed rest, the bias, and the traced out row
      const restOn = rest.length > 0;
      const nRows = shown.length + (restOn ? 1 : 0) + (bias ? 1 : 0) + 1;
      const pitch = clamp(Math.floor(330 / nRows), 30, 46);
      const rowAt = (k) => ROW_TOP + k * pitch + (k === nRows - 1 ? 14 : 0);
      const stripOpts = { orientation: 'h', cell: 14, gap: 2, mode: 'diverging', range: D.rStrip, head: VIS_HEAD, tail: VIS_TAIL };
      // a slot: its values, label, and the bar and norm of its contribution (painted by paintStrips)
      const slot = (sign, k, labelColor) => {
        const strip = vec(new Float64Array(d), stripOpts);
        strips.vecs.push(strip);
        const label = rowLabel('', labelColor), bar = h('span', { class: 'oc-bar' }), norm = h('span', { class: 'oc-norm' });
        rowEls.push({ el: add(rowEl({ sign, label, strip, bars: [bar, norm], top: rowAt(k) })), k });
        return { strip, label, bar, norm };
      };
      let k = 0;
      shown.forEach((_, s) => { strips.heads.push(slot(s === 0 ? '' : '+', k, 'var(--fg-2)')); k++; });
      if (restOn) { strips.rest = slot('+', k, 'var(--fg-3)'); k++; }
      if (bias) {
        strips.bias = vec(bias, stripOpts);
        strips.vecs.push(strips.bias);
        rowEls.push({ el: add(rowEl({ sign: '+', label: rowLabel('b_O', 'var(--c-o)'), strip: strips.bias, bars: [], top: rowAt(k) })), k });
        k++;
      }
      strips.out = vec(new Float64Array(d), { ...stripOpts, trace: `L${L}.attn.out`, cellOf: (i) => `${q},${i},0` });
      strips.vecs.push(strips.out);
      outSlot = h('span', { class: 'oc-slot' });
      rowEls.push({ el: add(rowEl({ sign: '=', label: rowLabel(`attn.out[${q}]`, 'var(--c-o)'), strip: strips.out, bars: [outSlot], top: rowAt(k) })), k });
      sepEl = add(h('div', { class: 'oc-sep', style: { left: `${ROW_L}px`, top: `${rowAt(k) - 7}px`, width: `${ROW_W}px` } }));
    }

    // ------------------------------------------------------------ values (cheap, on every trace event)
    function updateMats() {
      const { T, d, dh, cx, cat, W, out, rCtx, rW, rStrip } = D;
      mats.heads.forEach((m, hh) => m.update(cx.subarray(hh * T * dh, (hh + 1) * T * dh), rCtx));
      mats.cat.update(cat, rCtx);
      mats.w.update(W, rW);
      mats.out.update(out, rStrip);
      mats.side.update(sideVals(out, T, d, G.sideIdx), rStrip);
      mats.ctxCat.update(cat, rCtx);
      mats.ctxW.update(W, rW);
    }

    // the query row is the one accent focus: the row band of every matrix that holds it
    function highlightRows() {
      const q = D.q;
      mats.heads.forEach((m) => m.highlightRow(q));
      mats.cat.highlightRow(q);
      mats.out.highlightRow(q);
      mats.side.highlightRow(q);
    }

    function paintStrips() {
      const { d, q, out, contrib, norms, bias, rStrip, errQ, shown, rest, restVec, restNorm, nH, stripI } = D;
      const maxN = Math.max(1e-12, ...shown.map((hh) => norms[hh]));     // the bars scale to the heads only
      const kN = decimalsFor(Math.max(maxN, restNorm));                  // one precision for every norm on the screen
      const posJ = stripI.indexOf(D.j);                                  // the feature in the equation, as a DOM cell index
      const markJ = (strip) => strip.querySelectorAll('.k-cell').forEach((c, n) => c.classList.toggle('oc-j', n === posJ));
      const bar = (el, n, color) => { el.style.background = color; el.style.width = `${(n / maxN) * BAR_MAX}px`; };
      strips.heads.forEach((s, k) => {
        const hh = shown[k], c = headColor(hh, nH);
        s.strip.update(contrib[hh], { range: rStrip });
        markJ(s.strip);
        s.label.textContent = `h${hh}`;
        s.label.style.setProperty('--kind', c);
        bar(s.bar, norms[hh], c);
        s.norm.textContent = `‖c‖ ${norms[hh].toFixed(kN)}`;
      });
      if (strips.rest) {
        const s = strips.rest;
        s.strip.update(restVec, { range: rStrip });
        markJ(s.strip);
        s.label.textContent = `${rest.length} more`;
        // a sum of heads is not one head: no bar, and its norm is printed as the norm of the sum
        s.bar.style.width = '0px';
        s.norm.textContent = `‖Σc‖ ${restNorm.toFixed(kN)}`;
      }
      if (strips.bias) { strips.bias.update(bias, { range: rStrip }); markJ(strips.bias); }
      strips.out.update(out.subarray(q * d, (q + 1) * d), { range: rStrip });
      markJ(strips.out);
      // the out strip names the selected token's row, so its hover card and right-click maths read that row
      strips.out.querySelectorAll('.k-cell').forEach((c, n) => { if (stripI[n] >= 0) c.dataset.cell = `${q},${stripI[n]},0`; });
      const badge = checkBadge(errQ, 0, { label: `sum equals attn.out[${q}]` });
      badge.title = `largest difference from attn.out[${q}]: ${errQ.toExponential(1)}`;
      outSlot.replaceChildren(badge);
    }

    // ------------------------------------------------------------ layout for a step
    function layout(i, animate) {
      const { d, dh, nH, T } = D, { p, opg, hw, gh, pt } = G;
      const jumps = [];                                         // boxes that move without animation (flushed once, below)
      const top = (rows) => CY - boxOf(p, rows) / 2 - HDR;     // group top of a pipeline matrix with `rows` rows (its centre is CY)
      const bw = boxOf(p, d), total = 3 * bw + 2 * OPD + 4 * opg;
      const xC = HERO_L + (HERO_W - bw) / 2;                    // attn.concat: centred in steps 0 and 1
      const x0 = HERO_L + (HERO_W - total) / 2;                 // the pipeline: concat, W_O, out
      const xW = x0 + bw + 2 * opg + OPD, xO = xW + bw + 2 * opg + OPD;
      const hx0 = HERO_L + (HERO_W - (nH * hw + (nH - 1) * gh)) / 2;
      const APPEAR = '0s, 0s, .3s, .3s';                        // move at once, fade in after the move (left, top, opacity, visibility)
      const MERGE = '0s, 0s, .6s, .6s';                         // slide into the concat first, then dim
      const P = (el, x, y, o = {}) => place(el, x, y, { ...o, animate, jumps });
      // head blocks: spread (step 0); slide into their concat columns and dim (step 1); hidden after that
      groups.heads.forEach((el, hh) => {
        const colX = xC + hh * dh * p;
        if (i === 0) P(el, hx0 + hh * (hw + gh), top(T), { on: true });
        else if (i === 1) P(el, colX, top(T), { on: true, dim: true, merged: true, delay: MERGE });
        else P(el, colX, top(T), { on: false });
      });
      P(groups.cat, i === 2 ? x0 : xC, top(T), { on: i === 1 || i === 2 });
      P(groups.w, xW, top(d), { on: i === 2, delay: i === 2 ? APPEAR : '' });
      P(groups.out, xO, top(T), { on: i === 2, delay: i === 2 ? APPEAR : '' });
      const opX = [x0 + bw + opg + OPD / 2, xW + bw + opg + OPD / 2];
      ops.forEach((o, k) => P(o, opX[k] - OPD / 2, CY - OPD / 2, { on: i === 2, delay: i === 2 ? APPEAR : '' }));
      // step 3: the side view is the focus's context; attn.concat and W_O stay on screen, dimmed
      P(groups.side, SIDE_X, SIDE_TOP, { on: i === 3, delay: i === 3 ? APPEAR : '' });
      P(groups.ctxCat, CTX_X, CTX_Y, { on: i === 3, dim: true, delay: i === 3 ? APPEAR : '' });
      P(groups.ctxW, CTX_X, CTX_Y + HDR + boxOf(pt, T) + CTX_GAP, { on: i === 3, dim: true, delay: i === 3 ? APPEAR : '' });
      rowEls.forEach(({ el, k }) => show(el, i === 3, i === 3 ? `${k * M.stagger}ms` : ''));
      show(sepEl, i === 3, i === 3 ? '.2s' : '');
      if (jumps.length) {
        // one style flush so each jump is committed without a transition, then the transitions come back
        jumps.forEach((el) => getComputedStyle(el).left);
        jumps.forEach((el) => { el.style.transition = ''; });
      }
    }

    // ------------------------------------------------------------ text: equation, check, caption
    function paintText() {
      const { nH, nKV, d, q, tok, j, out, contrib, bias, shown, restVec, errCat, errAll, rest } = D;
      header.setSub(`Token “<b>${esc(tok)}</b>” at position ${q}. The heads are concatenated into attn.concat, then W_O mixes them into attn.out.`);
      const bo = bias ? ' + b_O' : '';
      const gqa = nKV === nH ? 'h' : `\\lfloor h / ${nH / nKV} \\rfloor`;     // the KV head of query head h
      // the terms of step 3, in the order of the rows: the shown heads, the summed rest, the bias
      const terms = [...shown.map((hh) => contrib[hh][j]), ...(restVec ? [restVec[j]] : []), ...(bias ? [bias[j]] : [])];
      // two lines per step; the numeric line of step 3 is the live one (19 px, one line)
      const eqs = [
        [`\\text{head}_h = \\Aw_h\\,\\V_{g(h)} \\in \\R^{T\\times d_h}`, `h = 0,\\dots,${nH - 1}, \\qquad g(h) = ${gqa}`],
        [`\\text{attn.concat} = \\concat_{h}\\big(\\text{head}_h\\big) \\in \\R^{T\\times d}`,
          `\\text{attn.concat}_{t,\\;h\\,d_h + c} = \\text{head}_{h,t,c}`],
        [`\\text{attn.out} = \\text{attn.concat}\\,\\Wo^{\\top}${bo} \\in \\R^{T\\times d}`,
          `\\text{attn.out}_{t,o} = \\textstyle\\sum_{h}\\textstyle\\sum_{c=1}^{d_h} \\text{head}_{h,t,c}\\,W_{O,\\,o,\\;h\\,d_h + c}${bias ? ' + b_{O,o}' : ''}`],
        [`\\text{attn.out}_{q} = \\textstyle\\sum_{h}\\,\\text{head}_{h,q}\\,\\big(\\Wo^{(h)}\\big)^{\\top}${bo} \\in \\R^{d}`,
          `\\text{attn.out}_{${q},${j}} = ${sumLine(terms, out[q * d + j])}`],
      ];
      eq.replaceChildren(...eqs[step].map((line, n) => h('div', {
        class: step === 3 && n === 1 ? 'oc-eq-live' : 'oc-eq-line',
        html: texHTML(line, { display: false }),
      })));

      if (step === 1) {
        check.replaceChildren(checkBadge(errCat, 0, { label: 'the blocks are attn.concat' }));
      } else if (step === 2) {
        const b = checkBadge(errAll, 0, { label: 'equals attn.out' });
        b.title = `largest difference over every token and feature: ${errAll.toExponential(1)}`;
        check.replaceChildren(b);
      } else {
        check.replaceChildren();
      }

      const CAP = [
        'Each head produces a T × d<sub>h</sub> output: its attention weights times its value vectors. Its colour stays with it through the scene.',
        'Concatenation places the heads side by side in head order: head h fills columns h·d<sub>h</sub> to (h+1)·d<sub>h</sub> − 1 of attn.concat. Nothing is summed yet.',
        'W<sub>O</sub> is d × d. Its column block W<sub>O</sub><sup>(h)</sup> reads only head h, so the projection mixes the heads into the same d features at every position.',
        `For “<b>${esc(tok)}</b>”, each head contributes its output times its own column block of W<sub>O</sub>${rest.length ? ' (the weakest heads share one row)' : ''}. The contributions${bias ? ' and the bias' : ''} add up to the traced attn.out row.`,
      ];
      cap.set(CAP[step]);
    }

    // ------------------------------------------------------------ refresh from the store
    function refresh({ values = false, animate = false } = {}) {
      const next = load();
      if (!next) {
        // this layer has no trace for the config: nothing of the previous layer's data stays on screen
        removeDom();
        D = null; G = null; sig = '';
        header.setSub('');
        header.el.querySelector('.k-eyebrow')?.replaceChildren(chip(`Layer ${L}`, 'resid'));
        eq.replaceChildren(); check.replaceChildren();
        cap.set(L >= ctx.cfg.n_layer ? `Layer ${L} is not in this model (n_layer = ${ctx.cfg.n_layer}).` : 'This layer has no trace for the current configuration.');
        return;
      }
      D = next;
      G = geometry(D.T, D.d, D.nH, D.dh);
      const s = `${D.T}|${D.d}|${D.nH}|${D.bias ? 1 : 0}|${G.p}|${G.sp}|${G.pt}`;
      const rebuilt = s !== sig;
      if (rebuilt) {
        sig = s;
        header.el.querySelector('.k-eyebrow')?.replaceChildren(chip(`Layer ${L}`, 'resid'), chip(`${D.nH} head${D.nH === 1 ? '' : 's'}`, 'attn'));
        buildDom();
      } else if (values) {
        updateMats();
      }
      layout(step, animate && !rebuilt);
      // the row band only moves when the token, the values or the shapes changed
      if (rebuilt || values || D.q !== lastQ) highlightRows();
      lastQ = D.q;
      paintStrips();
      paintText();
    }

    function applyStep(i, { animate = true } = {}) {
      step = i;
      if (!ready || !D) return;
      layout(step, animate);
      paintText();
    }

    // ------------------------------------------------------------ steps and the first-view explanation
    // Auto-advance runs the first time the scene is shown. Any step the reader chooses (a dot, an arrow, a key) and
    // any press anywhere on the page stops it, so it never moves the step under the reader.
    function stopAuto() { clearTimeout(auto); auto = null; }
    function advance() {
      auto = null;
      if (step >= STEPS.length - 1) return;
      autoStep = true;
      try { ctx.setStep(step + 1); } finally { autoStep = false; }
      if (step < STEPS.length - 1) auto = setTimeout(advance, MOTION.step);
    }

    ctx.setSteps(STEPS, (i) => { if (!autoStep) stopAuto(); applyStep(i); });
    refresh({ values: true, animate: false });
    ready = true;
    appear([header.el, picker.el, wrap, eq], { stagger: MOTION.stagger });
    auto = setTimeout(advance, 900);
    document.addEventListener('pointerdown', stopAuto, { capture: true, once: true });

    const offs = [ctx.on('trace', () => refresh({ values: true })), ctx.on('select', () => refresh({ values: false }))];
    return {
      pause: stopAuto,
      resume() { if (D) { highlightRows(); paintText(); } },   // shown again: the row band is measured while visible
      destroy() {
        stopAuto();
        document.removeEventListener('pointerdown', stopAuto, { capture: true });
        offs.forEach((f) => f());
        removeDom();
        st.destroy();
      },
    };
  },
};

// Token lookup: row t of W_E is the embedding of token t. The one-hot vector of its id, multiplied by
// W_E, selects that row, so a lookup and a matrix product are one operation. Steps on the real weights:
//   one-hot    the vector e, its 1 in orange; the table W_E dimmed as context
//   multiply   one entry of x as a sum over u of e_u · W_E[u, j]: every term but one is 0
//   lookup     row t of W_E is copied into x, entry by entry, and checked against the traced embedding
//   embedding  a PCA map of every row of W_E, the tokens of this text highlighted
// One vertical axis (the hero's centre line) carries the operations; the one-hot row and the copied row
// span the same width. A group fades out before the next one fades in. Reference scene for contributors:
// see docs/ATLAS_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, chip, op, checkBadge, appear, useStyle, MOTION, ZONES } from '../kit.js';
import { h, s, fmt, num, texHTML, clamp, showTip, hideTip, reducedMotion, diverging, heatmap, onThemeChange } from '../../ui/index.js';
import { mulberry32 } from '../../engine/random.js';

const STEPS = ['One-hot', 'Multiply', 'Same as a lookup', 'Embedding space'];
const HERO_W = 1030, HERO_H = 480;          // hero: x 110 … 1140, y 270 … 750 (ZONES.hero without the side column)
const STRIP_W = 960;                        // the one-hot row, the column, the products and the copied row span this width
const PITCH_E_MAX = 30, PITCH_X_MAX = 15;   // the widest cell pitch of an entry of e, and of an entry of x
const STRIP_H = 26;                         // the height of a strip of cells
const LBL_H = 20, GAP = 72;                 // a label line; the air between groups, where the operation nodes sit
const OP_D = 48;                            // operation nodes (design units)
const TBL_CELL = 3;                         // the table W_E is a small reference: at most 3-unit cells
const MAX_DRAW_V = 96;                      // larger vocabularies skip the strips: their cells would be too small
const MAX_MAP_V = 1500;                     // larger vocabularies skip the map
const MAP_PAD = 64;                         // room around the map for its labels
const PCA_SEED = 20261011;
const COPY_LEAD = 120, COPY_STEP = 14, COPY_FADE = 260;   // the copied row fills left to right (ms)
const SWEEP_MS = 40;                        // ms per entry of the sum in step 1
const OUT_MS = 300, IN_MS = 420;            // a group fades out, then the next one fades in
const GROUPS_OF_STEP = [['gE', 'gT'], ['gE', 'gD'], ['gE', 'gT', 'gB', 'gX'], ['gM']];

useStyle('token-embedding', `
.te-hero { position: relative; width: ${HERO_W}px; height: ${HERO_H}px; }
.te-group { position: absolute; left: 0; top: 0; width: ${HERO_W}px; height: ${HERO_H}px;
  transition: opacity ${IN_MS}ms var(--ease), transform .5s var(--ease), visibility 0s linear 0s; }
.te-group.off { opacity: 0; transform: translateY(10px); visibility: hidden; pointer-events: none;
  transition: opacity ${OUT_MS}ms ease, transform .4s var(--ease), visibility 0s linear ${OUT_MS}ms; }
.te-cap { position: absolute; font: 500 15px/20px var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.te-cap.embed { color: var(--c-embed); }
.te-idx { position: absolute; font: 500 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; pointer-events: none; }
.te-idx.lit { color: var(--accent); font-weight: 700; }
.te-strip { position: absolute; display: flex; height: ${STRIP_H}px; transition: opacity .4s var(--ease); }
.te-cell { flex: none; height: 100%; border-radius: 3px; box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--fg) 9%, transparent);
  transition: opacity .3s ease, background-color .4s ease; }
.te-cell.dim { opacity: .16; }
.te-cell.hl { outline: 2px solid var(--accent); outline-offset: 2px; }
.te-table { position: absolute; opacity: .5; }
.te-table .heatmap { display: block; line-height: 0; }
.te-table .heatmap-title { display: none; }
.te-band { position: absolute; box-sizing: border-box; border: 2px solid var(--c-embed); border-radius: 5px;
  background: color-mix(in srgb, var(--c-embed) 12%, transparent); pointer-events: none; }
.te-callout { position: absolute; font: 600 15px/20px var(--font-mono); color: var(--accent); white-space: nowrap; }
.te-sum { position: absolute; font: 600 24px/28px var(--font-mono); color: var(--fg); white-space: nowrap; }
.te-sum sub { font-size: .7em; line-height: 0; }
.te-map { position: absolute; left: 0; top: 0; width: ${HERO_W}px; height: ${HERO_H}px; }
.te-svg { position: absolute; left: 0; top: 0; overflow: visible; }
.te-axis { stroke: var(--line-strong); stroke-width: 1; stroke-dasharray: 4 5; }
.te-axis-lab { font: 500 15px var(--font-mono); fill: var(--fg-3); }
.te-pt .te-dot { fill: var(--fg-3); opacity: .45; transform-box: fill-box; transform-origin: center; transition: transform .45s var(--ease), opacity .45s, fill .45s; }
.te-pt .te-lab { font: 500 15px var(--font-mono); fill: var(--fg-3); opacity: 0; transition: fill .45s, opacity .45s; paint-order: stroke; stroke: var(--bg); stroke-width: 3px; stroke-linejoin: round; }
.te-pt .te-lead { stroke: var(--fg-3); stroke-width: 1; opacity: 0; transition: opacity .3s; }
.te-pt.lead .te-lead { opacity: .9; }
.te-pt .te-sel-ring { fill: none; stroke: var(--accent); stroke-width: 2; opacity: 0; transition: opacity .45s; }
.te-pt:hover .te-dot { opacity: 1; }
.te-pt:hover .te-lab { opacity: 1; fill: var(--fg); }
.te-pt.in { cursor: pointer; }
.te-pt.in .te-dot { fill: var(--c-embed); opacity: 1; transform: scale(1.45); }
.te-pt.in .te-lab { font: 700 15px var(--font-mono); fill: var(--fg); opacity: 1; }
.te-pt.sel .te-dot { fill: var(--accent); transform: scale(1.9); }
.te-pt.sel .te-lab { fill: var(--accent); }
.te-pt.sel .te-sel-ring { opacity: 1; }
.te-side { width: 340px; display: grid; gap: 14px; align-content: start; }
.te-side .katex-display, .te-side .katex-display > .katex { margin: 0; text-align: left; }
.te-label { text-align: left; color: var(--c-embed); }
.te-eq { text-align: left; color: var(--fg); }
.te-live { font-size: 18px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid color-mix(in srgb, var(--c-embed) 30%, var(--line)); overflow-x: auto; }
.te-check { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; opacity: 0; transform: translateY(6px); transition: opacity .5s var(--ease), transform .5s var(--ease); }
.te-check.on { opacity: 1; transform: none; }
.te-note { font-size: 16px; line-height: 1.5; }
.te-random { color: var(--fg-3); }
.te-legend { display: grid; gap: 8px; }
.te-legend-row { display: flex; align-items: center; gap: 10px; font: 500 16px/1.3 var(--font-body); color: var(--fg-2); }
.te-sw { width: 12px; height: 12px; border-radius: 50%; flex: none; background: var(--fg-3); opacity: .5; }
.te-sw.in { background: var(--c-embed); opacity: 1; }
.te-sw.sel { background: var(--accent); opacity: 1; box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 25%, transparent); }
@media (prefers-reduced-motion: reduce) {
  .te-group, .te-group.off, .te-strip, .te-cell, .te-check, .te-pt .te-dot, .te-pt .te-lab, .te-pt .te-lead, .te-pt .te-sel-ring { transition: none !important; }
}
`);

const ANCHOR = { tl: '0 0', tc: '-50% 0', tr: '-100% 0', lc: '0 -50%', c: '-50% -50%' };
const px = (v) => `${v}px`;
/** Place a node at (x, y) of the hero, the anchor being a point of the node (tl, tc, tr, lc or c). */
function place(node, x, y, anchor = 'tl') {
  node.style.position = 'absolute';
  node.style.left = px(x);
  node.style.top = px(y);
  node.style.translate = ANCHOR[anchor];
  return node;
}
const pct = (v) => `${Math.round(v * 100)}%`;
const legendRow = (sw, text) => h('div', { class: 'te-legend-row' }, sw, text);
const fadeIn = (node) => {
  if (reducedMotion() || !node.animate) return;
  node.animate([{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'none' }], { duration: MOTION.appear, easing: MOTION.easeOut });
};
const maxAbs = (a) => { let m = 0; for (const v of a) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v)); return m || 1; };
const copyMs = (D) => COPY_LEAD + (D.d - 1) * COPY_STEP + COPY_FADE;
const paintStrip = (sp) => sp.cells.forEach((c, u) => { c.style.background = diverging(sp.vals[u] / sp.range); });

/**
 * Cell pitches, strip widths, the table's cell and the vertical axis for a vocabulary of V entries and width d.
 * Every strip is at most STRIP_W wide; the axis is the centre of the wider strip.
 */
function geometry(V, d) {
  const pe = Math.max(3, Math.min(PITCH_E_MAX, Math.floor(STRIP_W / V))), pd = Math.max(3, Math.min(PITCH_X_MAX, Math.floor(STRIP_W / d)));
  const ge = pe >= 8 ? 2 : 1, gd = pd >= 8 ? 2 : 1;
  const eW = V * pe - ge, dW = d * pd - gd;
  const cw = clamp(Math.floor(120 / V), 2, TBL_CELL);
  return { pe, pd, ge, gd, ce: pe - ge, cd: pd - gd, eW, dW, cw, tW: d * cw, tH: V * cw, ax: Math.max(eW, dW) / 2 };
}

/** Vertical positions of the parts, in hero units. The one-hot row is at the top in every step. */
function rowsOf(g) {
  const e = LBL_H + 8, eIdx = e + STRIP_H + 6;                   // the one-hot row and its index labels
  const tLbl = eIdx + 15 + GAP, t = tLbl + LBL_H + 4;            // the table (steps 0 and 2); the column of W_E (step 1) takes its place
  const xLbl = t + g.tH + GAP, x = xLbl + LBL_H + 4;             // the copied row (step 2)
  const pLbl = t + STRIP_H + GAP, p = pLbl + LBL_H + 4;          // the products (step 1)
  const sumY = p + STRIP_H + GAP;                                // the sum (step 1)
  return {
    e, eIdx, tLbl, t, xLbl, x, xIdx: x + STRIP_H + 6, pLbl, p, sumY,
    opTimes: (eIdx + 15 + tLbl) / 2, opEq: (t + g.tH + xLbl) / 2,
    opDot: (eIdx + 15 + tLbl) / 2, opSum: (p + STRIP_H + sumY) / 2,
  };
}

/** A strip of cells, one per entry, coloured on the diverging ramp of `range`. */
function strip(parent, vals, { left, top, pitch, cell, gap, range, trace = null, cellOf = null, title = (u, v) => `${u}: ${fmt(v, 3)}` }) {
  const cells = Array.from(vals, (v, u) => {
    const c = h('span', { class: 'te-cell', title: title(u, v), style: { width: px(cell) } });
    if (cellOf) c.dataset.cell = cellOf(u);
    return c;
  });
  const el = h('div', { class: 'te-strip', style: { left: px(left), top: px(top), gap: px(gap), width: px(vals.length * pitch - gap) } }, ...cells);
  if (trace) el.dataset.trace = trace;
  parent.append(el);
  const sp = { el, cells, vals, range };
  paintStrip(sp);
  return sp;
}

/** Top two principal components of the rows of X (V × d): centre the rows, then power iteration on XᵀX. */
function pca2(X, V, d) {
  const mean = new Float64Array(d);
  for (let i = 0; i < V; i++) for (let c = 0; c < d; c++) mean[c] += X[i * d + c] / V;
  const Xc = new Float64Array(V * d);
  let total = 0;
  for (let i = 0; i < V; i++) for (let c = 0; c < d; c++) {
    const v = X[i * d + c] - mean[c];
    Xc[i * d + c] = v;
    total += v * v;
  }
  const times = (v) => {                       // Xc · v: one value per row
    const u = new Float64Array(V);
    for (let i = 0; i < V; i++) {
      let a = 0;
      for (let c = 0; c < d; c++) a += Xc[i * d + c] * v[c];
      u[i] = a;
    }
    return u;
  };
  const timesT = (u) => {                      // Xcᵀ · u: one value per dimension
    const w = new Float64Array(d);
    for (let i = 0; i < V; i++) for (let c = 0; c < d; c++) w[c] += Xc[i * d + c] * u[i];
    return w;
  };
  const basis = [];
  const orthoUnit = (v) => {                   // remove the components found so far, then normalise
    for (const p of basis) {
      let dot = 0;
      for (let c = 0; c < d; c++) dot += v[c] * p[c];
      for (let c = 0; c < d; c++) v[c] -= dot * p[c];
    }
    let n = 0;
    for (let c = 0; c < d; c++) n += v[c] * v[c];
    n = Math.sqrt(n);
    if (!(n > 1e-12)) return null;
    for (let c = 0; c < d; c++) v[c] /= n;
    return v;
  };
  const rnd = mulberry32(PCA_SEED);
  const eig = [];
  for (let k = 0; k < Math.min(2, d); k++) {
    let v = orthoUnit(Float64Array.from({ length: d }, () => rnd() - 0.5));
    for (let it = 0; v && it < 500; it++) {
      const w = orthoUnit(timesT(times(v)));
      if (!w) break;
      let move = 0;
      for (let c = 0; c < d; c++) move = Math.max(move, Math.abs(w[c] - v[c]));
      v = w;
      if (move < 1e-12) break;
    }
    if (!v) break;
    basis.push(v);
    let lam = 0;
    for (const u of times(v)) lam += u * u;
    eig.push(lam);
  }
  const xy = new Float64Array(2 * V);
  basis.forEach((v, k) => {
    const p = times(v);
    for (let i = 0; i < V; i++) xy[2 * i + k] = p[i];
  });
  const ratio = [0, 1].map((k) => (total > 0 && eig[k] ? eig[k] / total : 0));
  return { xy, ratio };
}

/** The k other rows of X that point most nearly the same way as row id (cosine over all d dimensions). */
function nearest(X, V, d, id, k = 3) {
  const norms = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    let n = 0;
    for (let c = 0; c < d; c++) n += X[i * d + c] ** 2;
    norms[i] = Math.sqrt(n);
  }
  const out = [];
  if (!(norms[id] > 0)) return out;
  for (let j = 0; j < V; j++) {
    if (j === id || !(norms[j] > 0)) continue;
    let dot = 0;
    for (let c = 0; c < d; c++) dot += X[id * d + c] * X[j * d + c];
    out.push({ j, cos: dot / (norms[id] * norms[j]) });
  }
  return out.sort((a, b) => b.cos - a.cos).slice(0, k);
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const offs = [];
    let step = 0, cur = null, destroyed = false, ticking = false;
    let parts = null;                        // the DOM of the lookup and the sum, built for the current data
    let strips = [];                         // the strips of parts, repainted when the theme changes
    let map = null, mapWE = null;            // step 3: the PCA map, and the W_E it was computed from
    let nearOf = null;                       // the tokens closest to the selected one (its tooltip on the map)
    let checkEl = null;                      // the check in the side column (steps 1 and 2)
    let auto = null, autoOn = !reducedMotion();
    let timers = [];

    // ------------------------------------------------------------ header, token picker, caption (shared parts)
    const header = sceneHeader(st, { chips: [], title: 'Token lookup' });
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'embed' });
    const cap = sceneCaption(st);

    // ------------------------------------------------------------ hero: one group per part, faded in and out by step
    const hero = h('div', { class: 'te-hero' });
    st.add(hero, { x: ZONES.hero.x, y: ZONES.hero.y });
    const gT = h('div', { class: 'te-group' });     // steps 0 and 2: the table, and the times node
    const gD = h('div', { class: 'te-group' });     // step 1: the column of W_E, the products and their sum
    const gB = h('div', { class: 'te-group' });     // step 2: the band on row t of the table
    const gX = h('div', { class: 'te-group' });     // step 2: the copied row and the equals node
    const gE = h('div', { class: 'te-group' });     // steps 0 to 2: the one-hot row
    const gM = h('div', { class: 'te-group' });     // step 3: the PCA map
    hero.append(gT, gD, gB, gX, gE, gM);
    const GROUP = { gT, gD, gB, gX, gE, gM };
    setVisible(0, { immediate: true });      // before any content is styled, so nothing fades at mount
    const mapHost = h('div', { class: 'te-map' });
    const mapNote = h('div', { class: 'k-caption', style: { width: px(HERO_W), display: 'none' } });
    gM.append(mapHost, mapNote);

    // ------------------------------------------------------------ right column: this step's equation, live numbers, check
    const side = h('div', { class: 'te-side' });
    st.add(side, { x: ZONES.side.x, y: ZONES.hero.y });

    // ------------------------------------------------------------ helpers
    const at = (i) => ctx.label(i);                          // the token at position i
    const lab = (v) => ctx.state.tokenizer.label(v);         // vocabulary entry v
    const later = (ms, fn) => timers.push(setTimeout(fn, ms));
    const stopTimers = () => { timers.forEach(clearTimeout); timers = []; };
    const stopAuto = () => { autoOn = false; clearTimeout(auto); auto = null; };
    const pointTip = (p) => {
      const name = lab(p.v) === '␣' ? 'space' : lab(p.v);
      const near = nearOf && p.v === nearOf.id && nearOf.list.length
        ? ` · closest by cosine: ${nearOf.list.map((n) => `“${lab(n.j)}” ${fmt(n.cos, 2)}`).join(', ')}` : '';
      return `<b>${name}</b> · id ${p.v}${p.positions.length ? ` · in this text at position ${p.positions.join(', ')}` : ' · not in this text'}${near}`;
    };

    /** Everything the scene shows, read from the store. */
    function data() {
      const cs = ctx.state;
      const T = cs.ids.length;
      const WE = cs.model.params['wte.weight'];
      const [V, d] = WE.shape;
      const t = clamp(Number.isFinite(cs.sel.token) ? cs.sel.token : T - 1, 0, T - 1);
      const id = cs.ids[t];
      const row = cs.trace['emb.tok'].data;                  // [T, d]: row t is the traced embedding of token t
      const x = Float64Array.from({ length: d }, (_, c) => row[t * d + c]);
      const onehot = new Float64Array(V);
      if (id < V) onehot[id] = 1;
      let jStar = 0;                                         // the output entry of largest magnitude: the one step 1 sums
      for (let c = 1; c < d; c++) if (Math.abs(x[c]) > Math.abs(x[jStar])) jStar = c;
      // e · W_E computed as the sum it is (one live term per entry), compared with the traced row
      let maxDelta = 0;
      for (let c = 0; c < d; c++) {
        let a = 0;
        for (let k = 0; k < V; k++) a += onehot[k] * WE.data[k * d + c];
        maxDelta = Math.max(maxDelta, Math.abs(a - x[c]));
      }
      // step 1: x_j as the running sum over u of e_u · W_E[u, j]
      const col = Float64Array.from({ length: V }, (_, u) => WE.data[u * d + jStar]);
      const prod = Float64Array.from({ length: V }, (_, u) => onehot[u] * col[u]);
      const partial = new Float64Array(V);
      let run = 0;
      for (let u = 0; u < V; u++) { run += prod[u]; partial[u] = run; }
      return { cfg: cs.config, ids: cs.ids, T, WE, V, d, t, id, x, onehot, jStar, maxDelta, col, prod, partial, sumU: run, random: cs.weights === 'random' };
    }

    // ------------------------------------------------------------ the lookup, the column and the sum (steps 0 to 2)
    function destroyHero() {
      parts?.heat.destroy();
      parts = null;
      strips = [];
      for (const g of [gT, gD, gB, gX, gE]) g.replaceChildren();
    }

    /** Builds the one-hot row, the table, the band, the copied row, and the column and products of step 1, for this data. */
    function buildHero(D) {
      destroyHero();
      const { V, d, WE, id, t, x, onehot, jStar: j, col, prod } = D;
      if (V > MAX_DRAW_V) {
        gE.append(place(h('div', { class: 'k-caption', style: { width: px(STRIP_W) } },
          `W_E has ${V} rows, too many to draw entry by entry. The lookup is still exact: the one-hot row picks row ${id}, the traced embedding of “${at(t)}” at position ${t}.`), 0, 0));
        return;
      }
      const g = geometry(V, d), R = rowsOf(g), rW = maxAbs(WE.data);
      const eL = g.ax - g.eW / 2, dL = g.ax - g.dW / 2, tL = g.ax - g.tW / 2;
      const add = (parent, vals, o) => { const sp = strip(parent, vals, o); strips.push(sp); return sp; };

      // the one-hot row: its 1 is the thing the lookup is about; index labels every 8, and the lit entry
      const e = add(gE, onehot, { left: eL, top: R.e, pitch: g.pe, cell: g.ce, gap: g.ge, range: 1 });
      gE.append(place(h('div', { class: 'te-cap' }, `one-hot e, ${V} entries`), eL, 0));
      for (let u = 0; u < V; u += 8) if (Math.abs(u - id) >= 3) gE.append(place(h('div', { class: 'te-idx' }, String(u)), eL + u * g.pe + g.ce / 2, R.eIdx, 'tc'));
      gE.append(place(h('div', { class: 'te-idx lit' }, `${id} · “${at(t)}”`), eL + id * g.pe + g.ce / 2, R.eIdx,
        id < 4 ? 'tl' : id > V - 5 ? 'tr' : 'tc'));

      // the table W_E: a small dimmed reference (steps 0 and 2), with the times node between it and e
      const heat = heatmap({ data: WE.data, rows: V, cols: d, mode: 'diverging', range: rW, cell: g.cw, minCell: 1, maxCell: 8, showValues: false, ariaLabel: 'W_E, the embedding table' });
      gT.append(place(h('div', { class: 'te-table' }, heat.el), tL, R.t));
      gT.append(place(h('div', { class: 'te-cap embed' }, `W_E (${V} × ${d}), the table`), tL, R.tLbl));
      gT.append(place(op('\\times', { kind: 'embed', size: OP_D }), g.ax, R.opTimes, 'c'));

      // the band on row t of the table (step 2): the row the lookup takes
      const rowY = R.t + id * g.cw + g.cw / 2;
      gB.append(h('div', { class: 'te-band', style: { left: px(tL - 8), top: px(rowY - 8), width: px(g.tW + 16), height: '16px' } }),
        place(h('div', { class: 'te-cap embed' }, `row ${id} · “${at(t)}”`), tL + g.tW + 14, rowY, 'lc'));

      // the copied row x (step 2), tied to the table by the equals node; its entries carry the hover card of emb.tok
      const xs = add(gX, x, { left: dL, top: R.x, pitch: g.pd, cell: g.cd, gap: g.gd, range: rW, trace: 'emb.tok', cellOf: (c) => `${t},${c},0` });
      gX.append(place(h('div', { class: 'te-cap embed' }, `x = row ${id} of W_E, copied`), dL, R.xLbl));
      for (let c = 0; c < d; c += 8) if (Math.abs(c - j) >= 3) gX.append(place(h('div', { class: 'te-idx' }, String(c)), dL + c * g.pd + g.cd / 2, R.xIdx, 'tc'));
      gX.append(place(h('div', { class: 'te-idx lit' }, String(j)), dL + j * g.pd + g.cd / 2, R.xIdx, 'tc'));
      gX.append(place(op('=', { kind: 'embed', size: OP_D }), g.ax, R.opEq, 'c'));

      // step 1: the column of W_E in entry j takes the table's place; the products e_u · W_E[u, j] and their sum
      const colS = add(gD, col, { left: eL, top: R.t, pitch: g.pe, cell: g.ce, gap: g.ge, range: rW, title: (u, v) => `W_E[${u}, ${j}] = ${fmt(v, 3)}` });
      gD.append(place(h('div', { class: 'te-cap embed' }, `W_E[:, ${j}], column ${j} of the table`), eL, R.tLbl));
      gD.append(place(op('\\odot', { kind: 'embed', size: OP_D }), g.ax, R.opDot, 'c'));
      const prodS = add(gD, prod, { left: eL, top: R.p, pitch: g.pe, cell: g.ce, gap: g.ge, range: rW, title: (u, v) => `e_${u} · W_E[${u}, ${j}] = ${fmt(v, 3)}` });
      prodS.cells.forEach((c) => c.classList.add('dim'));
      gD.append(place(h('div', { class: 'te-cap' }, `e_u · W_E[u, ${j}], one product per entry`), eL, R.pLbl));
      gD.append(place(h('div', { class: 'te-callout' }, `u = ${id}: 1 × ${fmt(col[id], 3)}`), eL + id * g.pe + g.ce / 2, R.pLbl, 'tc'));
      gD.append(place(op('\\Sigma', { kind: 'embed', size: OP_D }), g.ax, R.opSum, 'c'));
      const readout = place(h('div', { class: 'te-sum' }), g.ax, R.sumY, 'tc');
      gD.append(readout);

      parts = { heat, e, x: xs, col: colS, prod: prodS, readout };
    }

    /** Step 1: the sum up to entry u. Later products are dim, the cursor marks entry u, the readout holds the running sum. */
    function showSum(u) {
      const D = cur;
      if (!parts || !D) return;
      const { V, id, jStar: j, partial, sumU } = D;
      parts.prod.cells.forEach((c, k) => c.classList.toggle('dim', k > u));
      const live = u >= V - 1 ? id : u;      // the cursor rests on the live term once the sum is complete
      for (const sp of [parts.e, parts.col, parts.prod]) {
        sp.cells.forEach((c) => c.classList.remove('hl'));
        if (live >= 0) sp.cells[live].classList.add('hl');
      }
      parts.readout.innerHTML = u >= V - 1
        ? `x<sub>${j}</sub> = ${fmt(sumU, 3)}`
        : `Σ<sub>u ≤ ${u}</sub> = ${fmt(u < 0 ? 0 : partial[u], 3)}`;
      if (u === id && u < V - 1 && !reducedMotion()) {
        parts.prod.cells[id].animate([{ transform: 'scale(1)' }, { transform: 'scale(1.22)' }, { transform: 'scale(1)' }], { duration: 600, easing: 'ease-out' });
      }
      if (u >= V - 1) checkEl?.classList.add('on');
    }

    /** Step 2: row t of W_E is copied into x, one entry after the other. */
    function copyRow(delay) {
      if (!parts || reducedMotion()) return;
      parts.x.cells.forEach((c, j) => c.animate(
        [{ opacity: 0, transform: 'translateY(-8px)' }, { opacity: 1, transform: 'none' }],
        { duration: COPY_FADE, delay: delay + COPY_LEAD + j * COPY_STEP, easing: MOTION.easeOut, fill: 'backwards' }));
    }

    /** The static state of the lookup for the current step: the one-hot row dims while the copy is read, its 1 is outlined. */
    function paintLook() {
      if (!parts || !cur) return;
      parts.e.el.style.opacity = step === 2 ? '.6' : '1';
      if (step !== 1) parts.e.cells.forEach((c, u) => c.classList.toggle('hl', u === cur.id));
    }

    // ------------------------------------------------------------ step 3: the embedding space, PCA of every row of W_E
    function buildMap(D) {
      const { xy, ratio } = pca2(D.WE.data, D.V, D.d);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let v = 0; v < D.V; v++) {
        x0 = Math.min(x0, xy[2 * v]); x1 = Math.max(x1, xy[2 * v]);
        y0 = Math.min(y0, xy[2 * v + 1]); y1 = Math.max(y1, xy[2 * v + 1]);
      }
      const k = Math.min((HERO_W - 2 * MAP_PAD) / (x1 - x0 || 1), (HERO_H - 2 * MAP_PAD) / (y1 - y0 || 1));
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const toX = (x) => HERO_W / 2 + (x - cx) * k, toY = (y) => HERO_H / 2 + (y - cy) * k;   // one scale for both axes
      const ox = toX(0), oy = toY(0);                                                       // the PCA origin: the mean row
      const svg = s('svg', { class: 'te-svg', viewBox: `0 0 ${HERO_W} ${HERO_H}`, width: HERO_W, height: HERO_H },
        s('line', { class: 'te-axis', x1: 0, y1: oy, x2: HERO_W, y2: oy }),
        s('line', { class: 'te-axis', x1: ox, y1: 0, x2: ox, y2: HERO_H }),
        s('text', { class: 'te-axis-lab', x: HERO_W - 4, y: oy - 8, 'text-anchor': 'end' }, `PC1 · ${pct(ratio[0])} of variance`),
        s('text', { class: 'te-axis-lab', x: ox + 8, y: 18 }, `PC2 · ${pct(ratio[1])} of variance`));
      const pts = [];
      for (let v = 0; v < D.V; v++) {
        const x = toX(xy[2 * v]), y = toY(xy[2 * v + 1]);
        const lead = s('line', { class: 'te-lead', x1: 0, y1: 0, x2: 0, y2: 0 });
        const lbl = s('text', { class: 'te-lab', x: 10, y: 5 }, lab(v));
        const g = s('g', { class: 'te-pt' },
          s('circle', { class: 'te-sel-ring', r: 11, cx: 0, cy: 0 }),
          lead,
          s('circle', { class: 'te-dot', r: 5.5, cx: 0, cy: 0 }),
          lbl);
        const p = { v, x, y, g, lbl, lead, positions: [], delay: 0, dist: Math.hypot(x - ox, y - oy) };
        g.addEventListener('pointerenter', (e) => showTip(e.clientX, e.clientY, pointTip(p)));
        g.addEventListener('pointerleave', hideTip);
        g.addEventListener('click', () => { if (p.positions.length) ctx.store.select({ token: p.positions[0] }); });
        pts.push(p);
        svg.append(g);
      }
      const far = Math.max(1, ...pts.map((p) => p.dist));
      pts.forEach((p) => { p.delay = 60 + Math.round((620 * p.dist) / far); });
      mapHost.replaceChildren(svg);
      map = { svg, pts, ox, oy, ratio };
    }

    /** Rebuild the map only when W_E itself changes. Returns true when it was rebuilt. */
    function syncMap(D) {
      const big = D.V > MAX_MAP_V;
      mapHost.style.display = big ? 'none' : '';
      mapNote.style.display = big ? '' : 'none';
      if (big) {
        mapNote.textContent = `The map shows up to ${MAX_MAP_V} vocabulary rows; this model has ${D.V}.`;
        map = null; mapWE = null;
        return false;
      }
      if (map && mapWE === D.WE) return false;
      buildMap(D);
      mapWE = D.WE;
      return true;
    }

    /** Points fly out from the centre, nearest first. */
    function bloom() {
      if (!map) return;
      const still = reducedMotion();
      for (const p of map.pts) {
        p.g.style.transition = 'none';
        p.g.style.transform = `translate(${map.ox}px, ${map.oy}px)`;
        p.g.style.opacity = still ? '1' : '0';
      }
      void map.svg.getBoundingClientRect();     // commit the start state before the transition
      for (const p of map.pts) {
        p.g.style.transition = still ? 'none' : `transform 1s cubic-bezier(.2,.7,.2,1) ${p.delay}ms, opacity .6s ease ${p.delay}ms`;
        p.g.style.transform = `translate(${p.x}px, ${p.y}px)`;
        p.g.style.opacity = '1';
      }
    }

    /** Highlights on the map: the tokens of this text, and the selected token. */
    function paintMap(D) {
      const where = new Map();
      D.ids.forEach((id, i) => where.set(id, [...(where.get(id) || []), i]));
      for (const p of map.pts) {
        p.positions = where.get(p.v) || [];
        p.g.classList.toggle('in', p.positions.length > 0);
        p.g.classList.toggle('sel', p.v === D.id);
      }
      const sel = map.pts.find((p) => p.v === D.id);
      if (sel) map.svg.append(sel.g);          // the selected token is drawn last, above its neighbours
      nearOf = { id: D.id, list: D.V >= 2 ? nearest(D.WE.data, D.V, D.d, D.id, 3) : [] };
      const inText = map.pts.filter((p) => p.positions.length > 0 && p !== sel).sort((a, b) => a.positions[0] - b.positions[0]);
      placeLabels([...(sel ? [sel] : []), ...inText]);
    }

    /**
     * Labels for the tokens of this text (and the selected one): each takes the first free spot around its
     * point, and a spot away from the point gets a leader line. The other labels show on hover only.
     */
    function placeLabels(list) {
      const LH = 16, CW = 9.4;                  // 15 px mono: line box and advance, in hero units
      for (const p of map.pts) {
        p.lbl.setAttribute('x', 10); p.lbl.setAttribute('y', 5);
        p.lead.setAttribute('x2', 0); p.lead.setAttribute('y2', 0);
        p.g.classList.remove('lead');
      }
      const boxes = [];
      for (const p of list) {
        const w = CW * [...p.lbl.textContent].length + 6;
        const cands = [                          // [left, top, leader?] of the label box, nearest first
          [p.x + 10, p.y - 8, 0], [p.x - 10 - w, p.y - 8, 0],
          [p.x - w / 2, p.y - 12 - LH, 0], [p.x - w / 2, p.y + 12, 0],
          [p.x + 8, p.y - 12 - LH, 0], [p.x + 8, p.y + 12, 0],
          [p.x - 8 - w, p.y - 12 - LH, 0], [p.x - 8 - w, p.y + 12, 0],
          [p.x + 30, p.y - 8, 1], [p.x - 30 - w, p.y - 8, 1], [p.x - w / 2, p.y - 34, 1], [p.x - w / 2, p.y + 30, 1],
        ];
        const free = ([bx, by]) => bx >= 0 && by >= 0 && bx + w <= HERO_W && by + LH <= HERO_H
          && boxes.every(([qx, qy, qw]) => bx < qx + qw + 3 && bx + w + 3 > qx && by < qy + LH + 3 && by + LH + 3 > qy);
        const [bx, by, leader] = cands.find(free) || cands[0];
        boxes.push([bx, by, w]);
        p.lbl.setAttribute('x', bx - p.x);
        p.lbl.setAttribute('y', by + 12 - p.y);
        p.lead.setAttribute('x2', bx + w / 2 - p.x);
        p.lead.setAttribute('y2', by + LH / 2 - p.y);
        p.g.classList.toggle('lead', !!leader);
      }
    }

    // ------------------------------------------------------------ the right column, the caption, the header
    /** The equation of the current step with its numbers, and the check that ties it to the trace. */
    function renderSide(D) {
      const { id, t, V, d, WE, x, jStar: j, cfg, ids, maxDelta, sumU, random } = D;
      const eq = (tex, size = 24) => h('div', { class: 'te-eq', style: { fontSize: `${size}px` }, html: texHTML(tex, { display: true }) });
      const live = (tex) => h('div', { class: 'te-live', html: texHTML(tex, { display: true }) });
      const kids = [h('div', { class: 'k-label te-label' }, STEPS[step])];
      checkEl = null;
      if (step === 0) {
        kids.push(
          eq(String.raw`\mathbf{e}_{${id}} \in \{0,1\}^{${V}}`),
          live(String.raw`\begin{aligned} (\mathbf{e}_{${id}})_u &= \mathbf{1}[u = ${id}] \\ &= (0,\dots,0,\,\underbrace{1}_{u=${id}},\,0,\dots,0) \end{aligned}`));
      } else if (step === 1) {
        kids.push(
          eq(String.raw`x_{${j}} = \sum_{u} e_u\,\WE_{u,${j}}`),
          live(String.raw`\begin{aligned} &= \cdots + 1\cdot\WE_{${id},${j}} + \cdots \\ &= ${num(WE.data[id * d + j], 3)} \end{aligned}`));
        checkEl = h('div', { class: 'te-check' },
          checkBadge(sumU, x[j], { label: 'equals the traced value' }),
          h('span', { class: 'k-caption', style: { fontSize: '15px' } }, `all ${V} terms · ${V - 1} of them 0`));
        kids.push(checkEl);
      } else if (step === 2) {
        kids.push(
          eq(String.raw`\mathbf{x} = \WE_{${id},:}`),
          live(String.raw`x_{${j}} = \WE_{${id},${j}} = ${num(WE.data[id * d + j], 3)}`));
        checkEl = h('div', { class: 'te-check' },
          checkBadge(maxDelta, 0, { label: 'equals the traced embedding' }),
          h('span', { class: 'k-caption', style: { fontSize: '15px' } }, `all ${d} entries · max |Δ| = ${fmt(maxDelta, 3)}`));
        kids.push(checkEl);
      } else {
        const distinct = new Set(ids).size;
        kids.push(
          eq(String.raw`\begin{aligned} \WE &\in \R^{${V}\times${d}} \\ &\xrightarrow{\text{PCA}} \R^{${V}\times 2} \end{aligned}`, 22),
          h('div', { class: 'te-legend' },
            legendRow(h('span', { class: 'te-sw' }), 'vocabulary entry'),
            legendRow(h('span', { class: 'te-sw in' }), `token in this text (${distinct} distinct)`),
            legendRow(h('span', { class: 'te-sw sel' }), `selected: “${at(t)}” at position ${t}`)),
          h('div', { class: 'k-caption te-note', html: cfg.tie_weights
            ? 'Weight tying is on: the output layer reuses this matrix, so the same points also score the next token.'
            : 'Weight tying is off: this map shows W<sub>E</sub> only.' }));
      }
      if (random) kids.push(h('div', { class: 'k-caption te-note te-random' }, 'Random weights: the model is untrained, so the numbers are real but nothing has been learned.'));
      side.replaceChildren(...kids);
    }

    function captionFor(D) {
      const { id, t, V, d, jStar: j, maxDelta, sumU } = D;
      const drawn = V <= MAX_DRAW_V;
      if (step === 0) return `The one-hot vector of token <b>“${at(t)}”</b> has a 1 at index ${id} and 0 everywhere else, one entry per vocabulary item.${drawn ? ' W_E, the table, is shown dimmed.' : ''}`;
      if (step === 1) return `Each entry of x is a sum over the ${V} rows of W_E, and every term but one is multiplied by 0. So <b>x<sub>${j}</sub> = W_E[${id}, ${j}] = ${fmt(sumU, 3)}</b>.`;
      if (step === 2) {
        const copy = drawn ? `Row ${id} of W_E is copied into x, entry by entry.` : `The product is row ${id} of W_E, the vector of this token.`;
        return `${copy} It matches the traced embedding of <b>“${at(t)}”</b> at position ${t} (max |Δ| = ${fmt(maxDelta, 3)}), so a lookup is a one-hot product.`;
      }
      return `Each row of W_E is a point in ${d} dimensions; PCA keeps the two directions of greatest spread, so nearby points tend to have similar vectors. The map is a projection, so distances are approximate.`;
    }

    function render(D) {
      cur = D;
      const { id, t, V, d } = D;
      header.el.querySelector('.k-eyebrow')?.replaceChildren(chip(`position ${t}`, 'embed'), chip(`id ${id}`, 'embed'), chip(`W_E ${V}×${d}`, 'embed'));
      header.setSub(`Token <b>“${at(t)}”</b> at position ${t} has id ${id}. Its one-hot vector selects row ${id} of W_E: the token part of its embedding, before the position vector is added.`);
      renderSide(D);
      cap.set(captionFor(D));
      if (map) paintMap(D);
      paintLook();
    }

    /** Which groups show in step i. A group that goes out fades first; one that comes in waits delayIn ms for it. */
    function setVisible(i, { delayIn = 0, immediate = false } = {}) {
      const want = new Set(GROUPS_OF_STEP[i]);
      for (const [name, g] of Object.entries(GROUP)) {
        const on = want.has(name), was = !g.classList.contains('off');
        g.classList.toggle('off', !on);
        g.style.transitionDelay = on && !was && !immediate && delayIn ? `${delayIn}ms` : '';
      }
    }

    // ------------------------------------------------------------ steps and motion
    /** How long a step stays on screen when the scene plays itself: long enough for its motion. */
    function holdMs(D) {
      if (step === 0) return MOTION.step;
      if (step === 1) return Math.max(MOTION.step, OUT_MS + D.V * SWEEP_MS + 900);
      if (step === 2) return Math.max(MOTION.step, OUT_MS + copyMs(D) + 500);
      return 0;
    }

    /** The motion of the current step, from its start (after `delay` ms, while the previous group fades out). */
    function playStep({ delay = 0, entering = false, instant = false } = {}) {
      stopTimers();
      const D = cur;
      if (!D) return;
      const still = instant || reducedMotion();
      if (step === 1) {
        if (still) { showSum(D.V - 1); checkEl?.classList.add('on'); return; }
        showSum(-1);
        for (let u = 0; u < D.V; u++) later(delay + u * SWEEP_MS, () => showSum(u));
        later(delay + D.V * SWEEP_MS, () => checkEl?.classList.add('on'));
      } else if (step === 2) {
        if (!still) copyRow(delay);
        later(still ? 0 : delay + copyMs(D) - 80, () => checkEl?.classList.add('on'));
      } else if (step === 3 && entering) {
        if (still) bloom(); else later(delay, bloom);
      }
    }

    function scheduleNext() {
      clearTimeout(auto);
      if (!autoOn || destroyed) return;
      if (step >= STEPS.length - 1) { autoOn = false; return; }
      auto = setTimeout(() => {
        ticking = true;
        try { ctx.setStep(step + 1); } finally { ticking = false; }
        scheduleNext();
      }, holdMs(cur));
    }

    function enterStep(i) {
      if (destroyed) return;
      if (!ticking) stopAuto();                // a step chosen by hand ends the auto-play
      const prev = step;
      step = i;
      const D = data();
      const outgoing = GROUPS_OF_STEP[prev].some((name) => !GROUPS_OF_STEP[i].includes(name));
      const delay = outgoing ? OUT_MS : 0;
      setVisible(i, { delayIn: delay });
      render(D);
      playStep({ delay, entering: true });
      fadeIn(side);
    }

    /** Store events: a new token, a new forward pass, a new W_E. The current step plays again. */
    function refresh({ instant = false } = {}) {
      if (destroyed) return;
      const D = data();
      buildHero(D);
      const rebuilt = syncMap(D);
      setVisible(step, { immediate: true });
      render(D);
      playStep({ entering: rebuilt, instant });
    }

    // ------------------------------------------------------------ start
    ctx.setSteps(STEPS, enterStep);
    refresh();
    appear([header.el, picker.el, gE, side], { stagger: MOTION.stagger });
    if (autoOn) scheduleNext();
    el.addEventListener('pointerdown', stopAuto);
    offs.push(ctx.on('trace', () => refresh()), ctx.on('select', () => refresh()));
    const offTheme = onThemeChange(() => { if (!destroyed) strips.forEach(paintStrip); });

    return {
      pause() { stopAuto(); stopTimers(); },
      resume() { if (!destroyed) refresh({ instant: true }); },
      destroy() {
        destroyed = true;
        stopAuto();
        stopTimers();
        offs.forEach((off) => off());
        offTheme();
        destroyHero();
        mapHost.replaceChildren();
        hideTip();
        st.destroy();
      },
    };
  },
};

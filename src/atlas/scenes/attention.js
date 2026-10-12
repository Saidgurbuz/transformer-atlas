// Attention for one layer, on the real trace. Four steps: a query is asked; its arcs show who it reads
// (swept once); every head does the same with its own pattern; the heads are combined by W_O.
// Hero: the token row, with the selected head's attention arcs (arcDiagram, real weights) standing on a
// baseline above it. The selected query's arcs draw in with travelling dots, and its strongest arcs carry
// exact percentages from the trace (its self tick too). The query line and the head picker sit under the
// tiles, and the head picker ends where the token row ends. The arcs stand over the tokens: a deliberate
// deviation from the controls zone that softmax.js uses. Below: a pipeline of zoom targets for this layer
// (Q K V, one thumbnail per head, Shared K/V under grouped-query attention, and Concat + W_O). A thumbnail
// is the head's whole pattern, max-pooled to at most 24 × 24 cells. Reference scene: softmax.js.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, headPicker, card, vec, texLabel, checkBadge, appear, pulse, fly, useStyle, MOTION } from '../kit.js';
import { h, s, fmt, reducedMotion } from '../../ui/index.js';
import { sequential, cssVar, onThemeChange } from '../../ui/colors.js';
import { arcDiagram } from '../../ui/arcs.js';
import { ropeAngles } from '../../engine/index.js';

const STEPS = ['Query', 'Who it reads', 'Every head', 'Combine heads'];
const PAD = 16;          // side padding inside the arc diagram (arcs.js)
const HERO_W = 1380;     // hero width in design units: x 110 … 1490
const STAGE_W = 196;     // Q K V, Shared K/V and Concat cards
const ARROW_W = 28;      // arrow glyph width
const PIPE_GAP = 12;     // the pipeline row's gap between its children (.at-pipe)
const HEAD_GAP = 8;      // between two head thumbnails
const THUMB_MAX = 72;    // largest head thumbnail; 24 for one to four tokens, so a short pattern is not a large solid square
const THUMB_BINS = 24;   // a thumbnail has at most 24 × 24 cells: longer rows and columns are max-pooled
const THUMB_PX = 3;      // thumbnail canvas: device pixels per design unit
// Rows, top to bottom. The arcs stand on a baseline (ARC_BASE): each arc lands on its token, and the token
// tiles (42 units tall) sit just under it. The query line and the head picker sit under the tiles, then the
// pipeline, the equation row and the caption.
const ARC_BASE = 430, TILE_Y = ARC_BASE + 34;
const CTRL_Y = 522, CTRL_MID = CTRL_Y + 16;     // the query line and the head picker (32 units tall)
const PIPE_Y = 580, PIPE_H = 124;               // the pipeline row; its cards are centred on PIPE_H
const EQ_Y = 722, CAP_Y = 790;
const CLIP_BELOW = 30;   // the arc area is cut 30 units under the baseline, so it never covers the tiles
const ARC_MIN = 0.02;    // weight from which an arc is drawn (the arcs.js threshold on the arcs steps)
const TICK_TOP = 21;     // arcs.js draws the self tick from the baseline up to 21 units above it
const LABELS_N = 2;      // exact weights written on the selected query's strongest arcs
const SWEEP_MS = 4500;   // the one-off sweep over all queries, whatever the length of the row
const PAUSE_SWEEP = 1100, PAUSE_CYCLE = 900;   // rests before the next auto step

useStyle('attention', `
.at-label { display: flex; align-items: baseline; gap: 12px; font: 600 15px/1 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--fg-3); white-space: nowrap; }
.at-live { font: 500 16px/1 var(--font-mono); letter-spacing: 0; text-transform: none; color: var(--fg-2); }
.at-live b { font-weight: 650; color: var(--fg); }
.at-pipe { display: flex; align-items: center; justify-content: space-between; gap: ${PIPE_GAP}px; width: ${HERO_W}px; height: ${PIPE_H}px; }
.at-heads { position: relative; display: flex; flex-direction: column; gap: ${HEAD_GAP}px; }
.at-hrow { display: flex; align-items: center; gap: ${HEAD_GAP}px; }
.at-glab { position: absolute; left: 0; bottom: 100%; margin-bottom: 8px; font: 500 13px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.at-arrow { flex: none; width: ${ARROW_W}px; height: 12px; color: var(--fg-3); overflow: visible; }
.k-card.at-stage { width: ${STAGE_W}px; height: 60px; padding: 10px 12px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; }
.k-card.at-stage .k-card-title { font: 650 17px/1.15 var(--font-display); }
.k-card.at-stage .k-card-sub { font: 500 14px/1.35 var(--font-mono); margin-top: 4px; color: var(--fg-2); }
.k-card.at-head { padding: 7px 8px 8px; box-sizing: border-box; }
.k-card.at-head .k-card-title { font: 600 15px/1 var(--font-mono); color: var(--kind, var(--fg)); margin-bottom: 6px; }
.k-card.at-head .k-zoom-badge { top: 3px; right: 4px; font-size: 11px; }
.at-thumb { position: relative; border-radius: 3px; overflow: hidden; }
.at-trow { position: absolute; left: -2px; right: -2px; box-sizing: border-box; border: 1.5px solid var(--accent); border-radius: 3px; pointer-events: none; }
.at-pipe .k-card { box-shadow: 0 6px 16px rgba(0, 0, 0, .10); transition: opacity .5s cubic-bezier(.65,0,.35,1), transform .25s, box-shadow .25s, border-color .25s; }
.at-pipe .k-card.at-dim { opacity: .42; }
.at-pipe .k-card.at-on { outline: 2px solid var(--kind, var(--accent)); outline-offset: 3px; }
.at-eq { display: flex; align-items: center; gap: 22px; min-height: 36px; white-space: nowrap; }
.at-note { font: 500 15px/1.2 var(--font-mono); color: var(--fg-3); }
.at-arcs { overflow: hidden; transition: opacity .5s cubic-bezier(.65,0,.35,1); }
.at-arcs .arcs-svg .lab, .at-arcs .arcs-svg .idx { display: none; }
.at-arcs .arcs-svg:focus-visible { outline-offset: -3px; }
.at-arcs .arcs-svg .at-val { font: 650 15px/1 var(--font-mono); fill: var(--fg); stroke: var(--bg); stroke-width: 4px; stroke-linejoin: round; paint-order: stroke; text-anchor: middle; }
.at-vals { opacity: 0; transition: opacity .24s ease; }
.at-vals.on { opacity: 1; transition: opacity .32s ease .36s; }
.at-tokens .k-pick small { font: 500 12px/1 var(--font-mono); }
.at-tokens.at-dense .k-pick small { display: none; }
@media (prefers-reduced-motion: reduce) { .at-arcs, .at-vals, .at-vals.on { transition: none; } }
`);

const E = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
// Percentages are formatted from the exact probability, never from a rounded copy: one decimal below 9.5 %.
const pct = (w) => {
  const p = 100 * w;
  if (p < 0.05) return '<0.1%';
  return `${p >= 9.5 ? Math.round(p) : p.toFixed(1)}%`;
};
const diffText = (x) => (x === 0 ? '0' : x < 1e-12 ? '< 1e-12' : x.toExponential(1));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
// An arc's path is "M x0 yb Q cx cy x1 yb" (arcs.js). Its apex is the midpoint of that quadratic curve.
const ARC_D = /^M(-?[\d.]+) (-?[\d.]+)Q(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+)$/;

// Thumbnails: the T × T pattern max-pooled to R × R cells (R = min(T, THUMB_BINS)). A cell whose entries are all
// masked is NaN and is drawn hatched.
function pool(P, T, R) {
  const out = new Float64Array(R * R), e = (b) => Math.floor((b * T) / R);
  for (let br = 0; br < R; br++) for (let bc = 0; bc < R; bc++) {
    let m = -Infinity;
    for (let i = e(br); i < e(br + 1); i++) for (let j = e(bc); j < e(bc + 1); j++) m = Math.max(m, P[i * T + j]);
    out[br * R + bc] = m === -Infinity ? NaN : m;
  }
  return out;
}
// The thumbnail row that holds each query row.
function binRows(T, R) {
  const out = new Int32Array(T), e = (b) => Math.floor((b * T) / R);
  for (let b = 0; b < R; b++) for (let i = e(b); i < e(b + 1); i++) out[i] = b;
  return out;
}
// The head thumbnails share the pipeline row with the stage cards. One row when it gives the larger thumbnails
// (each takes the width that is left, up to THUMB_MAX; 24 for one to four tokens). Two rows for many heads, where
// the thumbnails are then about 40 units, without titles (the head picker names the heads). Never below 16 units.
function headLayout(T, H, gqa) {
  const stages = gqa ? 3 : 2, arrows = gqa ? 3 : 2, items = stages + arrows + 1;   // stage cards, arrows, the head group
  const avail = HERO_W - (stages * STAGE_W + arrows * ARROW_W + (items - 1) * PIPE_GAP);
  const cap = T <= 4 ? 24 : THUMB_MAX;
  const one = { rows: 1, perRow: H, titles: false, S: 0 };
  one.S = clamp(Math.min(cap, Math.floor((avail - (H - 1) * HEAD_GAP) / H) - 16, PIPE_H - 38), 16, cap);
  one.titles = one.S >= 24;
  const perRow = Math.ceil(H / 2);
  const two = { rows: 2, perRow, titles: false, S: 0 };
  two.S = clamp(Math.min(cap, Math.floor((avail - (perRow - 1) * HEAD_GAP) / perRow) - 16, Math.floor((PIPE_H - HEAD_GAP) / 2) - 17), 16, cap);
  return two.S > one.S ? two : one;
}
// Paints one thumbnail from its pooled cells, with the component ramp and hatched masked cells.
function drawThumb(x) {
  const g = x.canvas.getContext('2d'), { S, R, pooled } = x, cw = S / R;
  const hatch = cssVar('--c-mask'), surface = cssVar('--surface-2');
  g.setTransform(THUMB_PX, 0, 0, THUMB_PX, 0, 0);
  g.clearRect(0, 0, S, S);
  for (let br = 0; br < R; br++) for (let bc = 0; bc < R; bc++) {
    const v = pooled[br * R + bc], x0 = bc * cw, y0 = br * cw;
    if (Number.isNaN(v)) {
      g.fillStyle = surface;
      g.fillRect(x0, y0, cw + 0.05, cw + 0.05);
      g.save();
      g.beginPath(); g.rect(x0, y0, cw, cw); g.clip();
      g.strokeStyle = hatch; g.lineWidth = 0.6; g.beginPath();
      for (let t = -cw; t < cw; t += 2) { g.moveTo(x0 + t, y0 + cw); g.lineTo(x0 + t + cw, y0); }
      g.stroke(); g.restore();
    } else {
      g.fillStyle = sequential(v, 'attn');
      g.fillRect(x0, y0, cw + 0.05, cw + 0.05);
    }
  }
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const l = ctx.params.l;
    let M = buildModel();               // derived view of the current trace (see buildModel)
    if (!M) {
      st.add(h('div', { class: 'at-note' }, `No attention trace for layer ${l} in this model.`), { x: 110, y: 300 });
      return { destroy() { st.destroy(); } };
    }

    let destroyed = false, internal = 0, autoCall = false;
    let step = 0, autoOn = !reducedMotion(), autoTimer = null;
    let sweeping = false, sweepOrig = null, mirrored = null, playedOnce = false, sweepTimer = null;
    let cycleTimer = null, cycleOrig = null, shownHead = -1;
    let heads = [], qkvEl = null, sharedEl = null, concatEl = null, pipeSig = '', rowBin = null;
    let eqVec = null, heroX = null, valueKey = '', valueTimer = null, capHtml = null;

    // ------------------------------------------------------------------ data
    // Everything the scene shows, read once per forward pass. Patterns hide the masked cells as −∞
    // (hatched), so the causal triangle is visibly empty.
    function buildModel() {
      const S = ctx.state, cfg = ctx.cfg, tr = S.trace, P = S.model?.params ?? {};
      const probs = tr[`L${l}.attn.probs`];
      const T = S.ids.length;
      if (!probs || !T) return null;
      const H = cfg.n_head, Hkv = cfg.n_kv_head, d = cfg.d_model, dh = d / H;
      const mk = tr.mask?.data ?? null;
      const maskData = mk && mk.length >= T * T ? mk : null;
      const ok = (i, j) => !maskData || maskData[i * T + j] === 1;
      const patterns = Array.from({ length: H }, (_, k) => {
        const v = new Float64Array(T * T);
        for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) v[i * T + j] = ok(i, j) ? probs.data[(k * T + i) * T + j] : -Infinity;
        return v;
      });
      const rope = cfg.pos_encoding === 'rope', alibi = cfg.pos_encoding === 'alibi', scale = !!cfg.attn_scale;
      // The score the trace holds before the mask, as model.py forms it: q·k, divided by sqrt(d_h) when
      // attn_scale is on; with ALiBi the bias is added after that (the 'biased' key).
      const score = alibi ? tr[`L${l}.attn.biased`] : scale ? tr[`L${l}.attn.scaled`] : tr[`L${l}.attn.scores`];
      return {
        sig: `${T}|${H}|${Hkv}|${l}`, T, H, Hkv, g: H / Hkv, d, dh, rope, alibi, scale, mask: cfg.mask,
        norm: cfg.norm, preLN: cfg.norm_placement === 'pre', hasBias: !!P[`blocks.${l}.attn.q_proj.bias`],
        probs: probs.data, maskData, ok, patterns,
        qHeads: tr[`L${l}.attn.${rope ? 'q_rot' : 'q_heads'}`]?.data ?? null,
        kHeads: tr[`L${l}.attn.${rope ? 'k_rot' : 'k_heads'}`]?.data ?? null,
        score: score?.data ?? null,
        bias: alibi ? tr.alibi?.data ?? null : null,
        rot: rope ? ropeAngles(T, dh, cfg.rope_base || 10000) : null,
        ln1: tr[`L${l}.ln1`]?.data ?? null,                                              // pre-LN: the normed input
        blockIn: (l === 0 ? tr['emb.out'] : tr[`L${l - 1}.resid_out`])?.data ?? null,   // post-LN: the block input
        Wq: P[`blocks.${l}.attn.q_proj.weight`]?.data ?? null, bq: P[`blocks.${l}.attn.q_proj.bias`]?.data ?? null,
        Wo: P[`blocks.${l}.attn.o_proj.weight`]?.data ?? null, bo: P[`blocks.${l}.attn.o_proj.bias`]?.data ?? null,
        concat: tr[`L${l}.attn.concat`]?.data ?? null, out: tr[`L${l}.attn.out`]?.data ?? null,
      };
    }
    const headData = (k) => M.probs.subarray(k * M.T * M.T, (k + 1) * M.T * M.T);
    const curTok = () => Math.max(0, Math.min(ctx.state.sel.token ?? 0, M.T - 1));
    const curHead = () => Math.max(0, Math.min(ctx.state.sel.head ?? 0, M.H - 1));
    const lab = (i) => ctx.label(i);
    const qNow = () => arcs.query ?? curTok();   // the sweep moves the query; otherwise it is the selection

    function strongest(k, q) {
      let j = 0, w = -1;
      for (let c = 0; c < M.T; c++) {
        if (!M.ok(q, c)) continue;
        const v = M.probs[(k * M.T + q) * M.T + c];
        if (v > w) { w = v; j = c; }
      }
      return { j, w: Math.max(0, w) };
    }
    // The strongest visible key other than the query itself, or null when the query sees nothing else.
    function strongestOther(k, q) {
      let j = -1, w = -1;
      for (let c = 0; c < M.T; c++) {
        if (c === q || !M.ok(q, c)) continue;
        const v = M.probs[(k * M.T + q) * M.T + c];
        if (v > w) { w = v; j = c; }
      }
      return j < 0 ? null : { j, w };
    }

    // The traced values, recomputed from the weights (checks shown on screen). Each returns the largest
    // difference from the trace, or null when this config does not trace what the check needs.
    function checkQuery(hd, q) {
      // model.py: q = W_Q x + b_Q, then RoPE on the interleaved pairs (2m, 2m+1) when rope is on.
      const { T, d, dh } = M;
      const src = M.preLN ? M.ln1 : M.blockIn;     // pre-LN: the normed input; post-LN: the block input
      if (!src || !M.Wq || !M.qHeads) return null;
      const x = src.subarray(q * d, (q + 1) * d);
      const v = new Float64Array(dh);
      for (let c = 0; c < dh; c++) {
        const r = hd * dh + c;
        let acc = M.bq ? M.bq[r] : 0;
        for (let k = 0; k < d; k++) acc += M.Wq[r * d + k] * x[k];
        v[c] = acc;
      }
      if (M.rot) {
        const half = dh / 2;
        for (let m = 0; m < half; m++) {
          const cs = M.rot.cos[q * half + m], sn = M.rot.sin[q * half + m];
          const x1 = v[2 * m], x2 = v[2 * m + 1];
          v[2 * m] = x1 * cs - x2 * sn;
          v[2 * m + 1] = x1 * sn + x2 * cs;
        }
      }
      let worst = 0;
      for (let c = 0; c < dh; c++) worst = Math.max(worst, Math.abs(v[c] - M.qHeads[(hd * T + q) * dh + c]));
      return worst;
    }
    function checkScore(hd, q, j) {   // q·k / sqrt(d_h) (if attn_scale) + ALiBi bias, for the strongest key j
      const { T, dh, g } = M;
      const kv = Math.floor(hd / g);
      const qo = (hd * T + q) * dh, ko = (kv * T + j) * dh;
      let acc = 0;
      for (let c = 0; c < dh; c++) acc += M.qHeads[qo + c] * M.kHeads[ko + c];
      const computed = acc / (M.scale ? Math.sqrt(dh) : 1) + (M.bias ? M.bias[(hd * T + q) * T + j] : 0);
      return { computed, traced: M.score[(hd * T + q) * T + j] };
    }
    function checkOut(q) {            // out[q, :] = concat[q] · W_Oᵀ + b_O, every component of the output
      const HD = M.H * M.dh, d = M.d;
      let worst = 0;
      for (let c = 0; c < d; c++) {
        let acc = M.bo ? M.bo[c] : 0;
        for (let k = 0; k < HD; k++) acc += M.Wo[c * HD + k] * M.concat[q * HD + k];
        worst = Math.max(worst, Math.abs(acc - M.out[q * d + c]));
      }
      return worst;
    }

    // ------------------------------------------------------------------ shared parts and the hero
    const header = sceneHeader(st, {
      chips: [[`Layer ${l}`, 'resid'], ['', 'attn']], title: 'Attention',
      sub: 'Each token gathers information from the tokens it may see, weighted by query–key match.',
    });
    const chipHeads = header.el.querySelectorAll('.k-chip')[1];

    // The token tiles sit under the arcs, and each arc lands on its tile (layoutHero aligns them). The query
    // line and the head picker sit under the tiles that they name.
    const picker = tokenPicker(st, ctx, { x: 110, y: TILE_Y, label: 'query', kind: 'attn' });
    picker.el.classList.add('at-tokens');
    picker.el.querySelector('.k-picker-label')?.remove();     // the query line above names the selection
    const qLive = h('span', { class: 'at-live' });
    const qRow = h('div', { class: 'at-label' }, h('span', null, 'query'), qLive);
    st.add(qRow, { x: 110, y: CTRL_MID, anchor: 'lc' });
    const hp = headPicker(st, ctx, { x: 110, y: CTRL_MID });
    hp.el.style.translate = '-100% -50%';                    // its right edge is set in layoutHero

    const pipe = h('div', { class: 'at-pipe' });
    st.add(pipe, { x: 110, y: PIPE_Y });

    // Value labels are not arcs.js labels: arcs.js rounds each weight to two decimals, so the scene writes
    // its own from the trace (see valueItems). showLabels: false keeps arcs.js from writing any.
    const arcs = arcDiagram({
      labels: [], weights: { data: [], rows: 0, cols: 0 }, layout: 'arc', kind: 'attn', threshold: 2,
      maxWidth: HERO_W, showLabels: false, onSelect: onArcSelect,
    });
    const arcSvg = arcs.el.querySelector('svg.arcs-svg');
    arcSvg?.setAttribute('aria-label', 'Attention arcs for the selected query. Arrow keys move the query.');
    const vals = s('g', { class: 'at-vals' });
    const valTexts = Array.from({ length: LABELS_N }, () => s('text', { class: 'at-val' }));
    vals.append(...valTexts);
    arcSvg?.append(vals);
    const arcWrap = h('div', { class: 'at-arcs' }, arcs.el);
    st.add(arcWrap, { x: 110, y: ARC_BASE - 80 });
    // The arc area is the tensor: hovering it gives the maths and the live weight from the query to the key
    // under the pointer (cell = query, key, head), and the right-click menu offers the code and the values.
    arcWrap.dataset.trace = `L${l}.attn.probs`;
    arcWrap.addEventListener('pointermove', (e) => {
      const j = keyAt(e.clientX);
      if (j == null) delete arcWrap.dataset.cell;
      else arcWrap.dataset.cell = `${qNow()},${j},${curHead()}`;
    });
    arcWrap.addEventListener('pointerleave', () => { delete arcWrap.dataset.cell; });
    const eqRow = h('div', { class: 'at-eq' });
    st.add(eqRow, { x: 110, y: EQ_Y });
    const cap = sceneCaption(st, { y: CAP_Y });

    // The caption: a newer text replaces the old one at once and fades in; a fade still running is cancelled
    // (as lm-head.js does), so the caption never lags the step or the equation.
    function setCaption(html) {
      if (html === capHtml) return;
      capHtml = html;
      cap.el.getAnimations().forEach((a) => a.cancel());
      cap.el.innerHTML = html;
      if (reducedMotion()) return;
      cap.el.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 260, easing: MOTION.easeOut });
    }

    // ------------------------------------------------------------------ building and laying out
    function applyTrace() {
      M = buildModel();
      if (!M) return;
      if (pipeSig !== M.sig) buildPipeline();
      else heads.forEach((x, k) => setThumb(x, M.patterns[k]));
      chipHeads.textContent = `${M.H} head${M.H > 1 ? 's' : ''}${M.Hkv < M.H ? ` · ${M.Hkv} K/V` : ''}`;
      shownHead = curHead();
      arcs.update({
        labels: Array.from({ length: M.T }, (_, i) => lab(i)),
        weights: { data: headData(shownHead), rows: M.T, cols: M.T },
        mask: M.maskData, query: curTok(), threshold: step === 0 ? 2 : ARC_MIN,
      });
    }

    // Pools one head's pattern into its thumbnail and draws it.
    function setThumb(x, vals) {
      x.pooled = pool(vals, M.T, x.R);
      drawThumb(x);
    }

    // One card per head (its thumbnail), then the layer's other zoom targets, joined by arrows. Rebuilt only
    // when the shape changes.
    function buildPipeline() {
      heads.forEach((x) => x.off());
      const gqa = M.Hkv < M.H;
      const lay = headLayout(M.T, M.H, gqa), S = lay.S, R = Math.min(M.T, THUMB_BINS);
      rowBin = binRows(M.T, R);
      heads = M.patterns.map((vals, k) => {
        const canvas = h('canvas', { width: S * THUMB_PX, height: S * THUMB_PX, role: 'img', 'aria-label': `Attention pattern of head ${k}, a thumbnail` });
        canvas.style.width = `${S}px`;
        canvas.style.height = `${S}px`;
        canvas.dataset.trace = `L${l}.attn.probs`;
        const row = h('i', { class: 'at-trow' });
        const box = h('div', { class: 'at-thumb', style: { width: `${S}px`, height: `${S}px` } }, canvas, row);
        const c = card({ title: lay.titles ? `h${k}` : null, kind: 'attn', node: `head.${l}.${k}`, w: S + 16, body: box });
        c.classList.add('at-head');
        const x = { card: c, canvas, row, S, R, pooled: null, off: null };
        x.off = onThemeChange(() => drawThumb(x));
        return x;
      });
      const qw = M.H * M.dh, kw = M.Hkv * M.dh;
      qkvEl = card({ title: 'Q K V', sub: `Q ${qw} · K ${kw} · V ${kw}`, kind: 'q', node: `qkv.${l}`, w: STAGE_W });
      concatEl = card({ title: 'Concat + W_O', sub: `${qw} → ${M.d}`, kind: 'o', node: `out-proj.${l}`, w: STAGE_W });
      sharedEl = gqa ? card({ title: 'Shared K/V', sub: `${M.Hkv} K/V · ${M.g} Q each`, kind: 'k', node: `gqa.${l}`, w: STAGE_W }) : null;
      for (const c of [qkvEl, concatEl, sharedEl]) c?.classList.add('at-stage');
      const arrow = () => s('svg', { class: 'at-arrow', viewBox: '0 0 28 12', 'aria-hidden': 'true' },
        s('path', { d: 'M1 6 H26 M21 1 L27 6 L21 11', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.6, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
      const label = M.T > THUMB_BINS ? 'each head · thumbnail, max-pooled' : 'each head · thumbnail';
      const rowEls = Array.from({ length: lay.rows }, (_, r) => h('div', { class: 'at-hrow' },
        ...heads.slice(r * lay.perRow, (r + 1) * lay.perRow).map((x) => x.card)));
      const group = h('div', { class: 'at-heads' }, h('div', { class: 'at-glab' }, label), ...rowEls);
      pipe.replaceChildren(qkvEl, arrow(), ...(sharedEl ? [sharedEl, arrow()] : []), group, arrow(), concatEl);
      pipeSig = M.sig;
      heads.forEach((x, k) => setThumb(x, M.patterns[k]));
    }

    // The arc diagram is as wide as the token row, and token 0's arc node sits over tile 0. The head picker
    // ends where the last tile ends.
    function layoutHero() {
      if (picker.el.querySelectorAll('.k-pick').length !== M.T) picker.render();
      const tiles = picker.el.querySelectorAll('.k-pick');
      if (!tiles.length) return;
      const c0 = st.center(tiles[0]);
      const pitch = tiles.length > 1 ? st.center(tiles[1]).x - c0.x : c0.w + 4;
      const W = Math.min(HERO_W, Math.round(M.T * pitch + 2 * PAD));
      arcWrap.style.width = `${W}px`;
      heroX = c0.x - PAD - pitch / 2;
      arcs.update({ maxWidth: W });
      // The head picker ends where the token row ends, but never over the query line (a very short token row).
      const last = st.center(tiles[tiles.length - 1]), q = st.center(qRow);
      st.move(hp.el, Math.max(last.x + last.w / 2, q.x + q.w / 2 + 48 + st.center(hp.el).w), CTRL_MID);
      picker.el.classList.toggle('at-dense', M.T > 26);
      placeArcs();
    }

    // The arcs stand on a fixed baseline whatever their height (the height follows the reach of the arcs),
    // so the tiles, the pipeline and the equation never move when the arcs grow or shrink.
    function placeArcs() {
      const node = arcSvg?.querySelector('circle.node');
      if (!node || heroX == null) return;
      const yb = Number(node.getAttribute('cy'));
      arcWrap.style.height = `${yb + CLIP_BELOW}px`;
      st.move(arcWrap, heroX, ARC_BASE - yb);
    }

    // The key under the pointer: the token whose node is nearest the pointer, in design units.
    function keyAt(clientX) {
      const nodes = arcSvg?.querySelectorAll('circle.node');
      if (!nodes || !M || nodes.length !== M.T) return null;
      const rect = arcSvg.getBoundingClientRect();
      if (!rect.width) return null;
      const x = (clientX - rect.left) * (Number(arcSvg.getAttribute('width')) / rect.width);
      let best = null, bd = Infinity;
      nodes.forEach((n, j) => { const d = Math.abs(Number(n.getAttribute('cx')) - x); if (d < bd) { bd = d; best = j; } });
      return best;
    }

    // ------------------------------------------------------------------ value labels
    // The weights of the selected query's strongest arcs, as exact percentages from the trace, at the apex of
    // each arc. The self tick is labelled too, above the tick. Shown on the arcs step once the sweep has ended.
    // Not during the head cycle: the labels would change every half second.
    function arcApex(x0, x1) {
      for (const p of arcSvg.querySelectorAll('path.arc')) {
        const m = ARC_D.exec(p.getAttribute('d') || '');
        if (m && Math.abs(Number(m[1]) - x0) < 0.02 && Math.abs(Number(m[5]) - x1) < 0.02) {
          // The midpoint of a quadratic with both ends on the baseline: x is halfway, y is halfway to the control point.
          return { x: (Number(m[1]) + Number(m[5])) / 2, y: (Number(m[2]) + Number(m[4])) / 2 };
        }
      }
      return null;
    }
    function valueItems() {
      if (!M || sweeping || step !== 1 || !arcSvg) return null;
      const nodes = arcSvg.querySelectorAll('circle.node');
      if (nodes.length !== M.T) return null;
      const q = qNow(), hd = curHead(), T = M.T;
      const xs = Array.from(nodes, (n) => Number(n.getAttribute('cx')));
      const yb = Number(nodes[0].getAttribute('cy'));
      const top = [];
      for (let j = 0; j < T; j++) {
        if (!M.ok(q, j)) continue;
        const w = M.probs[(hd * T + q) * T + j];
        if (w >= ARC_MIN) top.push({ j, w });
      }
      top.sort((a, b) => b.w - a.w);
      const items = [];
      for (const { j, w } of top.slice(0, LABELS_N)) {
        if (j === q) { items.push({ x: xs[q], y: yb - TICK_TOP - 8, t: pct(w) }); continue; }   // the self tick: its label sits over the tick
        const apex = arcApex(xs[j], xs[q]);
        if (!apex) continue;
        items.push({ x: apex.x, y: apex.y - 9, t: pct(w) });
      }
      // Two labels that would touch: the second moves up.
      items.sort((a, b) => a.x - b.x);
      for (let k = 1; k < items.length; k++) {
        if (Math.abs(items[k].x - items[k - 1].x) < 56 && Math.abs(items[k].y - items[k - 1].y) < 16) items[k].y = items[k - 1].y - 18;
      }
      return items;
    }
    function paintValues() {
      const items = valueItems();
      const key = items ? items.map((i) => `${i.x.toFixed(1)},${i.y.toFixed(1)},${i.t}`).join('|') : '';
      if (key === valueKey) return;
      valueKey = key;
      clearTimeout(valueTimer);
      vals.classList.remove('on');                  // the old labels fade out first, then the new ones come in
      if (!items) return;
      valueTimer = setTimeout(() => {
        valueTimer = null;
        valTexts.forEach((t, k) => {
          const it = items[k];
          t.style.display = it ? '' : 'none';
          if (!it) return;
          t.setAttribute('x', it.x.toFixed(2));
          t.setAttribute('y', it.y.toFixed(2));
          t.textContent = it.t;
        });
        vals.classList.add('on');
      }, reducedMotion() ? 0 : 260);
    }

    // ------------------------------------------------------------------ readouts for the selection
    function refresh() {
      if (destroyed) return;
      if (!M) { eqRow.replaceChildren(h('div', { class: 'at-note' }, 'No attention trace for this layer.')); return; }
      const q = qNow(), hd = curHead();
      qLive.innerHTML = `<b>“${E(lab(q))}”</b> · position ${q}`;
      heads.forEach((x, k) => {
        const best = strongest(k, q);
        x.card.title = `Strongest key for query “${lab(q)}” in head ${k}: “${lab(best.j)}”, ${pct(best.w)}`;
        x.row.style.top = `${rowBin[q] * (x.S / x.R)}px`;
        x.row.style.height = `${x.S / x.R}px`;
        x.card.classList.toggle('at-dim', step === 0);
        x.card.classList.toggle('at-on', (step === 1 || step === 2) && k === hd);
      });
      qkvEl.classList.toggle('at-on', step === 0);
      qkvEl.classList.toggle('at-dim', step === 3);
      sharedEl?.classList.toggle('at-dim', step === 3);
      concatEl.classList.toggle('at-on', step === 3);
      arcWrap.style.opacity = step === 3 ? '0.5' : '1';
      renderEq(q, hd);
      setCaption(caption(q, hd));
      placeArcs();
      paintValues();
    }

    // One equation per step, with the live numbers and a check against the trace.
    function renderEq(q, hd) {
      eqVec?.destroy?.(); eqVec = null;
      const { T, dh } = M;
      const parts = [];
      if (step === 0) {
        const normOp = M.norm === 'rmsnorm' ? '\\RMSNorm' : '\\LN';
        const inp = M.preLN ? `${normOp}(x_{${q}})` : `x_{${q}}`;
        let rhs = `\\Wq^{(${hd})}\\,${inp}${M.hasBias ? ' + b_Q' : ''}`;
        if (M.rope) rhs = `\\mathrm{RoPE}\\big(${rhs}\\big)`;
        parts.push(texLabel(`${M.rope ? '\\tilde q' : 'q'}_{${q}}^{(${hd})} = ${rhs}`, { size: 24 }));
        if (M.qHeads) {
          eqVec = vec(M.qHeads.subarray((hd * T + q) * dh, (hd * T + q + 1) * dh), {
            orientation: 'h', cell: 14, gap: 2, kind: 'q', mode: 'diverging', trace: `L${l}.attn.${M.rope ? 'q_rot' : 'q_heads'}`,
            cellOf: (c) => `${q},${c},${hd}`,
          });
          parts.push(eqVec);
        }
        const worst = checkQuery(hd, q);
        parts.push(worst === null
          ? h('div', { class: 'at-note' }, 'no traced input for a check in this config')
          : checkBadge(worst, 0, { label: M.rope ? 'matches the traced rotated query' : 'matches the traced query', tol: 1e-9 }));
      }
      if (step === 1) {
        let sum = 0, vis = 0;
        for (let c = 0; c < T; c++) if (M.ok(q, c)) { sum += M.probs[(hd * T + q) * T + c]; vis++; }
        parts.push(texLabel(`\\sum_{j}\\,\\Aw_{${q}j} = 1`, { size: 24 }),
          checkBadge(sum, 1, { label: 'the weights sum to 1', tol: 1e-9 }),
          h('div', { class: 'at-note' }, `${vis} ${vis === 1 ? 'key' : 'keys'} visible to query ${q}`));
      }
      if (step === 2) {
        const kv = Math.floor(hd / M.g);
        // Under RoPE the scores come from the rotated query and key (the traced q_rot and k_rot).
        const qk = M.rope ? `\\tilde{\\Q}^{(${hd})}\\,\\tilde{\\K}^{(${kv})\\top}` : `\\Q^{(${hd})}\\K^{(${kv})\\top}`;
        const scaled = M.scale ? `\\frac{${qk}}{\\sqrt{${dh}}}` : qk;
        const bias = M.alibi ? ` + \\mathrm{ALiBi}^{(${hd})}` : '';
        const mask = M.mask === 'none' ? '' : ' + \\Mk';          // no mask term when the config has no mask
        parts.push(texLabel(`\\Aw^{(${hd})} = \\softmax\\!\\Big(${scaled}${bias}${mask}\\Big)`, { size: 22 }));
        if (M.qHeads && M.kHeads && M.score) {
          const { j } = strongest(hd, q);
          const { computed, traced } = checkScore(hd, q, j);
          parts.push(h('div', { class: 'at-note' }, `score (${q}, ${j}) = ${fmt(computed, 2)}`));
          if (M.bias) {   // the bias at the strongest key other than the query itself: at distance 0 it is always 0
            const o = strongestOther(hd, q);
            if (o) parts.push(h('div', { class: 'at-note' }, `ALiBi bias (${q}, ${o.j}) = ${fmt(M.bias[(hd * T + q) * T + o.j], 2)}`));
          }
          parts.push(checkBadge(computed, traced, { label: 'equals the traced score', tol: 1e-9 }));
        }
      }
      if (step === 3) {
        parts.push(texLabel(`\\mathrm{out}_{${q}} = \\concat_{${q}}\\,\\Wo^{\\top}${M.hasBias ? ' + b_O' : ''}`, { size: 24 }));
        if (M.concat && M.out && M.Wo) {
          const worst = checkOut(q);
          parts.push(checkBadge(worst, 0, { label: `all ${M.d} values equal the traced output`, tol: 1e-9 }),
            h('div', { class: 'at-note' }, `out[${q}, 0] = ${fmt(M.out[q * M.d], 3)}`),
            h('div', { class: 'at-note' }, `largest difference ${diffText(worst)}`));
        }
      }
      eqRow.replaceChildren(...parts);
    }

    function caption(q, hd) {
      const { H, dh, Hkv, g, T } = M;
      if (step === 0) return `The query of <b>“${E(lab(q))}”</b> at position ${q}, head ${hd}, is W_Q applied to this position’s own residual stream. No other position enters this formula.`;
      if (step === 1 && sweeping) return 'Each query in turn: arcs run from the keys it reads to it. Thicker arcs carry more weight, and every query’s weights sum to 1.';
      if (step === 1) {
        const { j, w } = strongest(hd, q);
        if (w < ARC_MIN) return `No weight reaches 2%, so no arc is drawn. The strongest key, <b>“${E(lab(j))}”</b>, gets ${pct(w)}.`;
        if (j === q) {
          let others = 0, arcsOn = 0;
          for (let c = 0; c < T; c++) {
            if (c === q || !M.ok(q, c)) continue;
            others++;
            if (M.probs[(hd * T + q) * T + c] >= ARC_MIN) arcsOn++;
          }
          if (!others) return `The query can see only itself, so its own weight is <b>${pct(w)}</b>, shown at the tick.`;
          return `The query reads itself most: its own weight is <b>${pct(w)}</b>, shown at the tick.${arcsOn ? ' The arcs show the other keys it reads.' : ''}`;
        }
        return `Arcs run from each key to query <b>“${E(lab(q))}”</b>, and thicker arcs carry more weight. The strongest, <b>“${E(lab(j))}”</b>, gets ${pct(w)}.`;
      }
      if (step === 2) {
        // The sentences that apply to this config: the heads, the scale, ALiBi, and the cycle through the heads.
        const root = Number.isInteger(Math.sqrt(dh)) ? Math.sqrt(dh) : Math.sqrt(dh).toFixed(2);
        const parts = [];
        if (H === 1) parts.push('This layer has one head, so its pattern is the one shown.');
        else if (Hkv >= H) parts.push('Each head has its own W_Q and W_K, so each finds a different pattern.');
        // Under grouped-query or multi-query attention the K/V heads are shared: only W_Q is per head.
        else parts.push(`Each head has its own W_Q, and ${Hkv === 1 ? `all ${H} query heads share the one K/V head` : `${g} query heads share each K/V head`}.`);
        parts.push(M.scale ? `Scores are divided by √d_h = ${root}.` : 'Scaling is off in this config, so the scores are not divided.');
        if (M.alibi) parts.push('ALiBi adds −m|q − j| to each allowed score.');
        if (H > 1) parts.push(reducedMotion() ? 'Each head’s pattern is on its card, and the arcs show the selected head.' : 'The arcs cycle through the heads.');
        return parts.join(' ');
      }
      return `The ${H} head outputs (${H} × ${dh} = ${H * dh} values) are concatenated and mixed by W_O into ${M.d} values per token. The check above recomputes every value of the output of <b>“${E(lab(q))}”</b> from them.`;
    }

    // ------------------------------------------------------------------ steps and motion
    function applyStep(i) {
      if (destroyed) return;
      const prev = step;
      step = clamp(i, 0, STEPS.length - 1);
      clearTimeout(autoTimer); autoTimer = null;
      if (prev === 2 && step !== 2) stopCycle(true);
      if (prev === 1 && step !== 1) stopSweep(false);
      if (M) arcs.update({ threshold: step === 0 ? 2 : ARC_MIN });   // step 0: the question only, no arcs yet
      if (step === 1) enterArcs();
      if (step === 2) startCycle();
      if (step === 3) enterCombine();
      refresh();
      armAuto();
    }

    // The first time the arcs step is shown, the arcs sweep once over every query, and the token picker
    // follows the sweep. When it ends (or the reader interrupts it) the reader's own token is restored.
    // Later visits show the selected query's arcs directly.
    function enterArcs() {
      if (!M) return;
      if (!playedOnce) { playedOnce = true; if (reducedMotion()) syncQuery(); else startSweep(); }
      else syncQuery();
    }
    // The sweep visits every query in turn, one step every `per` ms, so the whole row takes SWEEP_MS. The steps
    // are timed here rather than by arcs.play(): arcs.js caps its play speed at 4x, which would make a 64-token
    // row take 9.6 s. Each step moves the arcs (setQuery) and the store's token (mirrored).
    function startSweep() {
      sweeping = true;
      mirrored = null;
      sweepOrig = curTok();
      const per = clamp(SWEEP_MS / M.T, 60, 600);
      const t0 = performance.now();
      let at = -1;
      arcs.update({ speed: clamp(600 / per, 1, 4) });
      const tick = () => {
        sweepTimer = null;
        if (destroyed || !sweeping) return;
        // The clock picks the query, so slow work skips a token rather than stretching the sweep (as arcs.js does).
        const i = Math.floor((performance.now() - t0) / per);
        if (i >= M.T) { stopSweep(true); return; }
        if (i !== at) {
          at = i;
          mirrored = i;
          selInternal({ token: i });
          syncQuery();
          refresh();
        }
        sweepTimer = setTimeout(tick, Math.max(0, t0 + (i + 1) * per - performance.now()));
      };
      tick();
    }
    // Ends the sweep. Finished: the reader's own token comes back and the auto-play goes on. Stopped while the
    // sweep still held the token (a pointer press, a key, or leaving the step): that token comes back too.
    // Stopped after the reader chose another token: their choice stands.
    function stopSweep(finished) {
      clearTimeout(sweepTimer); sweepTimer = null;
      if (!sweeping) return;
      sweeping = false;
      const restore = finished || (mirrored !== null && ctx.state.sel.token === mirrored);
      mirrored = null;
      if (restore) selInternal({ token: sweepOrig });
      arcs.update({ speed: 1 });
      if (finished && autoOn && step === 1) scheduleAuto(PAUSE_SWEEP, () => goStep(2));
      syncQuery();
      refresh();
    }

    // The every-head step: the head shown in the arcs cycles through all heads and ends on the selected
    // one, so the arcs morph from head to head. Under reduced motion there is no cycle: the end state (the
    // selected head, with every head's pattern on its card) is shown at once.
    function startCycle() {
      stopCycle(false);
      if (!M) return;
      const n = M.H, cur = curHead();
      if (n < 2 || reducedMotion()) { pulse(heads[cur]?.card); afterCycle(); return; }
      cycleOrig = cur;
      const dwell = Math.max(250, Math.min(500, Math.round(2000 / n)));   // about 2 s for the whole cycle
      let k = 0;
      const tick = () => {
        cycleTimer = null;
        if (destroyed || step !== 2) return;
        if (!M || M.H !== n) { cycleOrig = null; return; }   // the model changed under the cycle: stop it
        k++;
        const next = (cycleOrig + k) % n;
        selInternal({ head: next });
        pulse(heads[next]?.card);
        if (k < n) cycleTimer = setTimeout(tick, dwell);
        else { cycleOrig = null; afterCycle(); }
      };
      cycleTimer = setTimeout(tick, 400);
    }
    function stopCycle(restore) {
      clearTimeout(cycleTimer); cycleTimer = null;
      if (cycleOrig == null) return;
      const back = cycleOrig;
      cycleOrig = null;
      if (restore && curHead() !== back) selInternal({ head: back });
    }
    function afterCycle() { if (autoOn && step === 2) scheduleAuto(PAUSE_CYCLE, () => goStep(3)); }

    // The heads-are-combined step: each head's tag flies from its card into the concatenation, in turn,
    // and the concatenation pulses when all have arrived.
    function enterCombine() {
      if (!concatEl || !heads.length) return;
      if (reducedMotion()) { pulse(concatEl); return; }
      const flights = heads.map((x, k) => new Promise((resolve) => setTimeout(resolve, 150 + k * 110)).then(() => {
        if (destroyed || step !== 3 || !x.card.isConnected) return;
        const from = st.center(x.card), to = st.center(concatEl);
        const tag = h('span', { class: 'k-chip', style: { '--kind': 'var(--c-attn)' } }, `h${k}`);
        const dest = h('span', { class: 'k-chip', style: { '--kind': 'var(--c-attn)', opacity: '0' } }, `h${k}`);
        st.add(tag, { x: from.x, y: from.y, anchor: 'c' });
        st.add(dest, { x: to.x, y: to.y, anchor: 'c' });   // same size as the tag, so the chip does not scale
        const done = fly(st, tag, dest, { duration: 780, arc: 50 });   // clones the tag now, before it is hidden
        tag.style.visibility = 'hidden';
        return done.then(() => { tag.remove(); dest.remove(); });
      }));
      Promise.all(flights).then(() => { if (!destroyed && step === 3) pulse(concatEl); });
    }

    // The first view walks through the steps by itself. Any pointer or key use stops that.
    function armAuto() {
      if (!autoOn || destroyed) return;
      if (step === 0) scheduleAuto(MOTION.step, () => goStep(1));
      else if (step === 1 && !sweeping) scheduleAuto(MOTION.step, () => goStep(2));
    }
    function scheduleAuto(ms, fn) {
      clearTimeout(autoTimer);
      autoTimer = setTimeout(() => { autoTimer = null; if (!destroyed && autoOn) fn(); }, ms);
    }
    function goStep(i) { autoCall = true; try { ctx.setStep(i); } finally { autoCall = false; } }

    // Any reader action ends the auto-play. A running sweep stops and, if it still holds the token, hands the
    // reader's token back. A running head cycle stops and hands back the head (restore). A selection made by
    // the reader already stands, so the handler that receives it passes restore = false.
    function userTookOver(restore = true) {
      autoOn = false;
      clearTimeout(autoTimer); autoTimer = null;
      stopSweep(false);
      stopCycle(restore);
    }

    // ------------------------------------------------------------------ selection
    function selInternal(patch) { internal++; try { ctx.store.select(patch); } finally { internal--; } }
    function syncQuery() { if (!M) return; const q = curTok(); if (arcs.query !== q) arcs.setQuery(q); }

    // Arc clicks, keys and sweep steps. A sweep step moves the store's token too (mirrored), so the
    // picker, the label and the readouts all follow the arcs. A click or key is the reader's choice.
    // The query always follows the selection: clearing it (a second click, Escape) re-syncs it.
    function onArcSelect(q) {
      if (destroyed) return;
      if (q == null) { syncQuery(); return; }
      mirrored = null;
      selInternal({ token: q });
    }
    function onStepSelect(i) {
      if (destroyed) return;
      if (!autoCall) userTookOver();   // any click on the story bar is the reader's
      if (i !== step) applyStep(i);
    }
    function onSelect() {
      if (destroyed) return;
      if (!internal) userTookOver(false);   // the reader chose: auto-play and cycle stop; a running sweep yields (stopSweep)
      if (!M) return;
      syncQuery();
      if (curHead() !== shownHead) {
        shownHead = curHead();
        arcs.update({ weights: { data: headData(shownHead), rows: M.T, cols: M.T } });
      }
      refresh();
    }
    function onTrace() {
      if (destroyed) return;
      applyTrace();
      hp.render();
      if (M) layoutHero();
      refresh();
    }

    // ------------------------------------------------------------------ mount
    applyTrace();
    layoutHero();
    ctx.setSteps(STEPS, onStepSelect);
    appear([header.el, picker.el, qRow, hp.el, pipe, arcWrap, eqRow], { stagger: MOTION.stagger });
    applyStep(0);
    const offs = [ctx.on('trace', onTrace), ctx.on('select', onSelect)];
    el.addEventListener('pointerdown', () => userTookOver());
    el.addEventListener('keydown', (e) => {
      userTookOver();
      // The arcs take the arrow keys for the query, so the atlas must not also step the story with them.
      if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && arcWrap.contains(e.target)) e.stopPropagation();
      // The atlas leaves keys from a focused [tabindex] element alone, so with the arcs focused Escape zooms out
      // here. The arcs have already cleared their query by then (their own Escape handler runs first).
      if (e.key === 'Escape' && arcWrap.contains(e.target) && ctx.node.parent) { e.preventDefault(); ctx.go(ctx.node.parent); }
    });

    return {
      pause() {
        userTookOver();
        arcs.update({ dots: false });
      },
      resume() { arcs.update({ dots: true }); },
      destroy() {
        destroyed = true;
        clearTimeout(autoTimer); clearTimeout(cycleTimer); clearTimeout(valueTimer); clearTimeout(sweepTimer);
        arcs.destroy();
        heads.forEach((x) => x.off());
        eqVec?.destroy?.();
        offs.forEach((f) => f());
        st.destroy();
      },
    };
  },
};

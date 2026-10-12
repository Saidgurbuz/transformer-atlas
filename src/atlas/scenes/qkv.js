// Q, K and V: where queries, keys and values come from. Each one is the attention input X times one
// head's rows of its weight matrix, plus its bias. The hero is one token's row of that product on the
// real weights, as a matrix-vector product: the row x_t lies over W (d_h × d, the stored layout), and
// each entry of the output column is the dot product of one row of W with x_t. The largest entry runs
// its dot product term by term, the rest of the row fills in, and a check compares the whole row with
// the trace. The side column shows the full projection with the token's row and the head's columns
// marked. The last step splits Q into its head bands. Steps: input X, queries, keys, values, split.
// Reference scene for contributors: see docs/ATLAS_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, headPicker, checkBadge, texLabel, appear, ZONES, MOTION, useStyle } from '../kit.js';
import { h, fmt, num, texHTML, clamp, diverging, onThemeChange, reducedMotion } from '../../ui/index.js';
import { runLive } from '../../core/math/index.js';
import { attnInput } from '../../core/math/util.js';

const STEPS = ['Input X', 'Queries', 'Keys', 'Values', 'Split into heads'];
const PROJ_OF_STEP = ['q', 'q', 'k', 'v', 'q'];          // the split is of Q, as the spec asks
const SYM = { q: 'Q', k: 'K', v: 'V' };
const SYM_TEX = { q: String.raw`\Q`, k: String.raw`\K`, v: String.raw`\V` };
const NAME = { q: 'Queries', k: 'Keys', v: 'Values' };
const INPUT_SUB = 'The same X feeds Q, K and V: one row per token, each mixed by its own weights.';
const HERO = { x: 110, y: 260, w: 1030, h: 490 };        // the hero zone (design units)
const SIDE = { x: 1180, y: 260 };                        // the side column, 340 wide, top-aligned with the hero
const CTRL_Y = 196;                                       // controls row: token tiles on the left, head chips on the right
const TOKEN_W = 1010;                                     // the token row's width, so the head chips fit beside it
const HEADS = { x: 1154, w: 336 };                        // the head chips' box, right edge at 1490
const DENSE_PICKER_T = 48;                                // above this many tokens the tile row wraps to two rows
const PAD = 8, GAP = 1, GAP_Q = 52;                       // box padding, hairline between cells, output column to W (its running sum sits here)
const CELL_MIN = 4, CELL_MAX = 20;                        // design units
const HERO_V = 388;                                       // height left for the cells: HERO.h less the tags and padding
const ROWS_MAX = 77;                                      // the most W rows drawn at the minimum cell; beyond this rows are elided
const CTX_ROWS = 20, CTX_COLS = 64;                       // side context: entries per axis, ellipses included
const CTX_CELL = 4, CTX_GAP = 1, CTX_ELL_ROW = 8, CTX_ELL_COL = 6;
const SPLIT_GAP = [28, 120];                              // the gap between bands in the split: at least 28, so the 10-unit frames keep clear
const SWEEP_MS = 900;                                     // a dot product runs in about this long
const TAG = 22;                                           // height of a tag above or below a box

useStyle('qkv', `
.qkv-picker { width: ${TOKEN_W}px; }
.qkv-picker .k-picker { flex-wrap: wrap; row-gap: 4px; max-width: 960px; }
.qkv-picker.dense .k-pick { height: 28px; }
.qkv-heads { width: ${HEADS.w}px; justify-content: flex-end; flex-wrap: wrap; gap: 4px; }
.qkv-heads .k-picker-label { display: none; }
.qkv-heads .k-pick-head { width: 36px; height: 26px; font-size: 12px; }
.qkv-hero { position: relative; }
.qkv-prod, .qkv-bands { position: absolute; inset: 0; transition: opacity .45s ease, visibility 0s linear 0s; }
.qkv-prod.off, .qkv-bands.off { opacity: 0; visibility: hidden; pointer-events: none; transition: opacity .45s ease, visibility 0s linear .45s; }
.qkv-box { position: absolute; box-sizing: border-box; padding: ${PAD}px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 72%, transparent);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--kind) 40%, var(--line)), 0 14px 34px rgba(0, 0, 0, .16); transition: opacity .45s ease, box-shadow .3s ease; }
.qkv-box.dim { opacity: .3; }
.qkv-box.focus { box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--kind) 40%, var(--line)), 0 0 0 3px color-mix(in srgb, var(--accent) 80%, transparent), 0 14px 34px rgba(0, 0, 0, .16); }
.qkv-grid { display: grid; }
.qkv-ell { display: grid; place-items: center; font: 600 14px/1 var(--font-mono); color: var(--fg-3); }
.qkv-tag { position: absolute; display: flex; align-items: center; gap: 8px; height: ${TAG}px; white-space: nowrap; }
.qkv-tag .k-tex { color: var(--kind, var(--fg)); }
.qkv-shape { font: 500 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.qkv-scan { position: absolute; box-sizing: border-box; border: 2px solid var(--accent); border-radius: 4px; pointer-events: none; opacity: 0;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent); transition: left .14s linear, top .3s ease, opacity .3s ease; }
.qkv-rowband { position: absolute; box-sizing: border-box; border: 2px solid var(--accent); border-radius: 5px; pointer-events: none; opacity: 0; transition: top .3s ease, opacity .3s ease; }
.qkv-sum { position: absolute; font: 600 12px/16px var(--font-mono); color: var(--fg); white-space: nowrap; pointer-events: none; opacity: 0; transition: top .3s ease, opacity .3s ease; }
.qkv-key { position: absolute; right: 0; top: 0; display: flex; align-items: center; gap: 6px; font: 600 14px/1 var(--font-mono); color: var(--fg-3); transition: opacity .35s ease; }
.qkv-key.off { opacity: 0; visibility: hidden; }
.qkv-key i { display: block; width: 110px; height: 6px; border-radius: 3px; }
.qkv-bands.instant .qkv-band { transition: none; }
.qkv-joined { position: absolute; border-radius: 12px; border: 1px solid color-mix(in srgb, var(--c-q) 45%, var(--line)); background: color-mix(in srgb, var(--surface) 72%, transparent);
  box-shadow: 0 14px 34px rgba(0, 0, 0, .16); transition: opacity .45s ease; }
.qkv-bands.split .qkv-joined { opacity: 0; }
.qkv-band { position: absolute; top: 0; transition: left ${MOTION.move}ms ${MOTION.ease}; }
.qkv-frame { position: absolute; inset: -10px; border-radius: 10px; border: 1px solid color-mix(in srgb, var(--c-q) 50%, var(--line)); background: color-mix(in srgb, var(--surface) 72%, transparent);
  box-shadow: 0 14px 34px rgba(0, 0, 0, .16); opacity: 0; transition: opacity .45s ease, border-color .3s ease, box-shadow .3s ease; }
.qkv-bands.split .qkv-frame { opacity: 1; }
.qkv-band-bar { position: absolute; left: 0; right: 0; top: -17px; height: 4px; border-radius: 2px; background: var(--c-q); transition: opacity .4s ease, background-color .3s ease; }
.qkv-bands.split .qkv-band-bar { opacity: 0; }
.qkv-band-lab { position: absolute; left: -12px; right: -12px; top: -46px; text-align: center; font: 600 15px/1 var(--font-mono); color: var(--c-q); white-space: nowrap; transition: color .3s ease; }
.qkv-band-row { position: absolute; left: -3px; right: -3px; border-radius: 3px; border: 2px solid color-mix(in srgb, var(--c-q) 80%, transparent); pointer-events: none; }
.qkv-band.sel .qkv-frame { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent), 0 14px 34px rgba(0, 0, 0, .16); }
.qkv-band.sel .qkv-band-bar { background: var(--accent); }
.qkv-band.sel .qkv-band-lab { color: var(--accent); }
.qkv-side { width: 340px; display: grid; gap: 8px; align-content: start; }
.qkv-lab { text-align: left; margin-bottom: 0; color: var(--kind, var(--fg-2)); }
.qkv-sshape { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); }
.qkv-eq { font-size: 24px; overflow-x: auto; }
.qkv-eq .katex-display, .qkv-livetex .katex-display { margin: 0; }
.qkv-live { font-size: 18px; padding: 10px 12px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.qkv-livetex { margin: 4px 0; }
.qkv-note { font: 500 13px/1.35 var(--font-mono); color: var(--fg-3); }
.qkv-math { margin-top: 4px; }
.qkv-check { min-height: 28px; }
.qkv-ctx { display: grid; gap: 6px; justify-items: start; }
.qkv-ctx-lab { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); }
.qkv-ctxbox { position: relative; padding: ${PAD}px; border-radius: 10px; background: color-mix(in srgb, var(--surface) 72%, transparent);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--kind) 40%, var(--line)), 0 10px 24px rgba(0, 0, 0, .14); }
.qkv-cgrid { display: flex; flex-direction: column; gap: ${CTX_GAP}px; }
.qkv-cr { display: flex; gap: ${CTX_GAP}px; border-radius: 2px; }
.qkv-cr.hit { box-shadow: 0 0 0 2px var(--accent); }
.qkv-ce { flex: none; width: ${CTX_ELL_COL}px; height: ${CTX_CELL}px; }
.qkv-cell-ell { height: ${CTX_ELL_ROW}px; font: 600 10px/${CTX_ELL_ROW}px var(--font-mono); color: var(--fg-3); text-align: center; }
.qkv-cmask { position: absolute; background: color-mix(in srgb, var(--surface) 62%, transparent); pointer-events: none; }
.qkv-cband { position: absolute; border: 2px solid var(--kind); border-radius: 3px; pointer-events: none; }
@media (prefers-reduced-motion: reduce) {
  .qkv-prod, .qkv-bands, .qkv-box, .qkv-scan, .qkv-rowband, .qkv-key, .qkv-band, .qkv-frame, .qkv-joined, .qkv-band-bar, .qkv-band-lab { transition: none !important; }
}
`);

// ------------------------------------------------------------------ helpers

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));   // HTML strings only
const maxAbs = (a) => { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m && Number.isFinite(v)) m = v; } return m || 1; };
const article = (n) => (/^(8\d*|11|18)$/.test(String(n)) ? 'an' : 'a');   // "an 8-wide slice", "a 16-wide slice"
/** The attention input's trace key: LN1(x) for pre-LN, the block input (the stream) for post-LN. */
const attnInKey = (l, preLN) => (preLN ? `L${l}.ln1` : l === 0 ? 'emb.out' : `L${l - 1}.resid_out`);

/** Indices to draw along an axis of n entries: all of them, or the span [lo, hi] in full, as much of the
 *  rest as fits in max entries, and -1 where entries are cut (an ellipsis). */
function keepSpan(n, lo, hi, max) {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const span = hi - lo + 1;
  const e1 = lo > 0 ? 1 : 0, e2 = hi < n - 1 ? 1 : 0;
  const budget = Math.max(0, max - span - e1 - e2);
  let right = Math.min(n - 1 - hi, Math.ceil(budget / 2));
  const left = Math.min(lo, budget - right);
  right = Math.min(n - 1 - hi, budget - left);
  const out = Array.from({ length: left }, (_, i) => i);
  if (lo - left > 0) out.push(-1);
  for (let i = lo; i <= hi; i++) out.push(i);
  if (n - 1 - hi - right > 0) out.push(-1);
  for (let i = n - right; i < n; i++) out.push(i);
  return out;
}

/** Cell size and placement of the hero product: the input row over W (d_h rows of d cells), the output
 *  column (d_h cells) on the left of W, all centred in the hero. The cell is as large as the hero allows. */
function heroGeometry(d, R) {
  const byWidth = (HERO.w - 4 * PAD - GAP_Q - (d - 1) * GAP) / (d + 1);     // the output column and d columns of W
  const byHeight = (HERO_V - (R - 1) * GAP) / (R + 1);                     // the input row and R rows of W
  const s = clamp(Math.floor(Math.min(CELL_MAX, byWidth, byHeight) * 10) / 10, CELL_MIN, CELL_MAX);
  const boxW = d * s + (d - 1) * GAP + 2 * PAD;                            // the W box and the input box
  const colW = s + 2 * PAD;                                                // the output column box
  const xH = s + 2 * PAD, boxH = R * s + (R - 1) * GAP + 2 * PAD;
  const totalW = colW + GAP_Q + boxW, totalH = 2 * TAG + 26 + xH + boxH;   // a tag, 8, the input box, 10, W, 8, a tag
  const x0 = (HERO.w - totalW) / 2, y0 = (HERO.h - totalH) / 2;
  const wX = x0 + colW + GAP_Q, xY = y0 + TAG + 8, wY = xY + xH + 10;
  return { s, pitch: s + GAP, boxW, boxH, colW, xH, x0, y0, qX: x0, wX, xY, wY, tagY: wY + boxH + 8 };
}

/** Running sums of entry c's products over the inputs: ps[k] = sum over j ≤ k of X[t, j] · W[c, j]. */
function partialSums(D, c) {
  const { d, t, Xd, WS } = D, ps = new Float64Array(d);
  let acc = 0;
  for (let k = 0; k < d; k++) { acc += Xd[t * d + k] * WS[c * d + k]; ps[k] = acc; }
  return ps;
}
/** The recomputed entry c of the token's row: its full dot product plus bias. */
const entryValue = (D, c) => partialSums(D, c)[D.d - 1] + (D.bS ? D.bS[c] : 0);
/** The largest difference between the recomputed row and the traced row of this head. */
function rowError(D) {
  let err = 0;
  for (let c = 0; c < D.dh; c++) err = Math.max(err, Math.abs(entryValue(D, c) - D.C[D.t * D.dh + c]));
  return err;
}

/** Everything the scene shows for one projection, read from the store. */
function read(ctx, l, p) {
  const s = ctx.state, cfg = s.config, T = s.ids.length;
  const H = cfg.n_head, Hkv = cfg.n_kv_head, d = cfg.d_model, dh = d / H, G = H / Hkv;
  const hd = clamp(s.sel.head ?? 0, 0, H - 1);            // the selected query head
  const kvh = Math.floor(hd / G);                          // its KV head: floor(h · Hkv / H)
  const hi = p === 'q' ? hd : kvh;                         // which head of this projection it reads
  const nH = p === 'q' ? H : Hkv;
  const width = nH * dh;
  const name = `blocks.${l}.attn.${p}_proj`;
  const W = s.model.params[`${name}.weight`];              // nn.Linear layout [out, in]
  const b = cfg.bias ? s.model.params[`${name}.bias`] ?? null : null;   // no bias parameters when cfg.bias is off
  const full = s.trace[`L${l}.attn.${p}`];                 // [T, nH·dh], before RoPE
  const Xd = attnInput(s.trace, cfg, l).data;              // [T, d]: ln1 (pre-LN) or the stream (post-LN)
  const t = clamp(s.sel.token ?? T - 1, 0, T - 1);
  const C = new Float64Array(T * dh);                      // the head's slice of the traced projection
  for (let r = 0; r < T; r++) for (let c = 0; c < dh; c++) C[r * dh + c] = full.data[r * width + hi * dh + c];
  let cMax = 0;                                            // the largest entry of the token's row in this head
  for (let c = 1; c < dh; c++) if (Math.abs(C[t * dh + c]) > Math.abs(C[t * dh + cMax])) cMax = c;
  return {
    T, H, Hkv, d, dh, G, hd, kvh, hi, p, t, cMax, width, nH, full, Xd, C,
    WS: Float64Array.from(W.data.subarray(hi * dh * d, (hi + 1) * dh * d)),
    bS: b ? Float64Array.from(b.data.subarray(hi * dh, (hi + 1) * dh)) : null,
    preLN: cfg.norm_placement === 'pre',
    rope: cfg.pos_encoding === 'rope',
  };
}

function eqTeX(D, step) {
  const P = SYM[D.p], i = D.hi, bias = D.bS ? ` + b_{${P},${i}}` : '';
  if (step === 0) return D.preLN
    ? String.raw`X = \LN_1(x) \in \R^{${D.T}\times ${D.d}}`
    : String.raw`X = x \in \R^{${D.T}\times ${D.d}}`;
  if (step === 4) return String.raw`(Q_{${D.hd}})_{t,c} = Q_{t,\;${D.hd}\,d_h + c}`;
  const sym = D.p === 'q' ? String.raw`\Q` : D.p === 'k' ? String.raw`\K` : String.raw`\V`;
  return String.raw`${sym}_{${i}} = X\,W_{${P},${i}}^{\top}${bias}`;
}

function shapeText(D, step) {
  if (step === 0) return `X  ${D.T} × ${D.d}  ·  one row per token`;
  if (step === 4) return `${D.H} blocks of ${D.T} × ${D.dh}, from Q  ${D.T} × ${D.width}`;
  return `${D.T}×${D.d} · ${D.d}×${D.dh} = ${D.T}×${D.dh}${D.bS ? '  + bias' : ''}`;
}

function subText(D) {
  const a = D.hi * D.dh, b = a + D.dh - 1;
  const rope = D.rope ? ', before RoPE rotates them' : '';
  if (D.p === 'q') return `Head <b>${D.hi}</b> reads rows ${a}–${b} of W_Q. Its queries are ${article(D.dh)} ${D.dh}-wide slice of X · W_Qᵀ${rope}.`;
  const g0 = D.kvh * D.G, g1 = g0 + D.G - 1;
  const who = D.G === 1 ? `Query head ${D.hd} has its own KV head` : `Query head ${D.hd} reads KV head <b>${D.kvh}</b>, shared by query heads ${g0} to ${g1}`;
  return `${who}. Its ${NAME[D.p].toLowerCase()} are ${article(D.dh)} ${D.dh}-wide slice of X · W_${SYM[D.p]}ᵀ${rope}.`;
}

function splitSub(D) {
  return `Splitting Q gives one <b>${D.T} × ${D.dh}</b> matrix per head. K and V split the same way, into ${D.Hkv} KV head${D.Hkv > 1 ? 's' : ''}.`;
}

function keyNote(D) {
  if (D.G === 1) return 'Every query head has its own key head.';
  if (D.Hkv === 1) return `All ${D.H} query heads share one key head (multi-query attention).`;
  const g0 = D.kvh * D.G, g1 = g0 + D.G - 1;
  return `Query heads ${g0} to ${g1} share key head ${D.kvh}, so K is ${D.Hkv * D.dh} columns wide, not ${D.H * D.dh}.`;
}

// Captions are HTML: the token name is escaped by the caller, and only there.
function capText(D, step, tok) {
  const { t, dh, hi } = D;
  switch (step) {
    case 0: return `Row <b>${t}</b> is the input of ${tok} at position ${t}. Q, K and V are each computed from this row alone.`;
    case 1: return `Each entry of the query is one dot product: a row of W<sub>Q,${hi}</sub> with this row of X. The largest entry is computed term by term, and the check compares the whole row with the trace.`;
    case 2: return `Keys use W<sub>K</sub> in the same way: row ${t} is the key of ${tok} at position ${t}. ${keyNote(D)}`;
    case 3: return `Values use W<sub>V</sub> and are read through the same KV head as the keys. Row ${t} is the value of ${tok}, the vector attention copies from it.`;
    default: return `Q is ${D.H} blocks of <b>${dh}</b> columns side by side. Row ${t} is the query of ${tok} in every head.`;
  }
}

// ------------------------------------------------------------------ scene

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const l = ctx.params.l ?? 0;
    let step = 0, auto = null, advancing = false, paused = false, holdMs = MOTION.step;
    let timers = [], sweepT = [], hoverT = 0, sweepRaf = 0;
    let prod = null, bands = null, themed = [], lastHead = ctx.state.sel.head;
    const offs = [];

    const later = (ms, fn) => { timers.push(setTimeout(fn, ms)); };
    const stopSweep = () => {
      sweepT.forEach(clearTimeout); sweepT = [];
      if (sweepRaf) cancelAnimationFrame(sweepRaf);
      sweepRaf = 0;
      clearTimeout(hoverT); hoverT = 0;
    };
    const clearTimers = () => { timers.forEach(clearTimeout); timers = []; stopSweep(); };
    const stopAuto = () => { clearTimeout(auto); auto = null; };
    /** One cell's colour from a value in [−1, 1]; remembered so a theme change can repaint it. */
    const paint = (cell, x) => { cell.style.background = diverging(x); themed.push([cell, x]); };
    const cellEl = (size, x) => {
      const c = h('div', { class: 'k-cell', style: { width: `${size}px`, height: `${size}px` } });
      if (x != null) paint(c, x);
      return c;
    };

    // ------------------------------------------------------------ header, controls
    const header = sceneHeader(st, { chips: [[`Layer ${l}`, 'resid'], [`Head ${ctx.state.sel.head}`, 'attn']], title: 'Q, K and V projections' });
    const picker = tokenPicker(st, ctx, { x: ZONES.controls.x, y: CTRL_Y, label: 'token', maxWidth: TOKEN_W });
    picker.el.classList.add('qkv-picker');
    const syncDense = () => picker.el.classList.toggle('dense', ctx.state.ids.length > DENSE_PICKER_T);
    syncDense();
    const heads = headPicker(st, ctx, { x: HEADS.x, y: CTRL_Y });
    heads.el.classList.add('qkv-heads');

    // ------------------------------------------------------------ hero: the token's product, or the split bands
    const prodEl = h('div', { class: 'qkv-prod' });
    const bandsEl = h('div', { class: 'qkv-bands off' });
    const keyBar = h('i');
    const keyEl = h('div', { class: 'qkv-key' }, h('span', null, '−'), keyBar, h('span', null, '+'));
    const heroEl = h('div', { class: 'qkv-hero', style: { width: `${HERO.w}px`, height: `${HERO.h}px` } }, prodEl, bandsEl, keyEl);
    st.add(heroEl, { x: HERO.x, y: HERO.y });
    const keyPaint = () => { keyBar.style.background = `linear-gradient(90deg, ${diverging(-1)}, ${diverging(0)}, ${diverging(1)})`; };
    keyPaint();

    // ------------------------------------------------------------ side column: label, shape, equation, live entry, check, context
    const sLab = h('div', { class: 'k-label qkv-lab' });
    const sShape = h('div', { class: 'qkv-sshape' });
    const sEq = h('div', { class: 'qkv-eq' });
    const sNote = h('div', { class: 'qkv-note' });
    const sTex = h('div', { class: 'qkv-livetex' });
    const sMath = h('div', { class: 'qkv-note qkv-math' });
    const sCheck = h('div', { class: 'qkv-check' });
    const sCtx = h('div', { class: 'qkv-ctx' });
    const sideEl = h('div', { class: 'qkv-side' }, sLab, sShape, sEq, h('div', { class: 'qkv-live' }, sNote, sTex, sMath), sCheck, sCtx);
    st.add(sideEl, { x: SIDE.x, y: SIDE.y });
    const cap = sceneCaption(st, { x: ZONES.caption.x, y: ZONES.caption.y, w: ZONES.caption.w });

    // ------------------------------------------------------------ the token's product (steps 0 to 3)
    /** Tags name the three boxes, with the symbol and the true shape. */
    function setTags(D) {
      const { d, dh, t, hi, p, preLN } = D;
      prod.tagX.replaceChildren(texLabel(`X_{${t},:}`, { size: 16, kind: preLN ? 'norm' : 'resid' }), h('span', { class: 'qkv-shape' }, `1 × ${d}`));
      prod.tagQ.replaceChildren(texLabel(`(${SYM_TEX[p]}_{${hi}})_{${t},:}`, { size: 16, kind: p }), h('span', { class: 'qkv-shape' }, `1 × ${dh}`));
      prod.tagW.replaceChildren(texLabel(`W_{${SYM[p]},${hi}}`, { size: 16, kind: p }), h('span', { class: 'qkv-shape' }, `${dh} × ${d}`));
    }

    /** The hero's boxes, cells and overlays for one projection. Rebuilt when the head, the step or the
     *  drawn rows change; a token change only repaints the input row and the output column. */
    function buildProduct(D, shown, key) {
      prodEl.replaceChildren();
      const { d, dh, t, hi, p, cMax, preLN } = D;
      const G = heroGeometry(d, shown.length);
      const s = G.s, pitch = G.pitch, perCol = clamp(SWEEP_MS / d, 7, 24);
      const rangeX = maxAbs(D.Xd), rangeW = maxAbs(D.WS), rangeC = maxAbs(D.full.data);
      const grid = (cols) => h('div', { class: 'qkv-grid', style: { gridTemplateColumns: `repeat(${cols}, ${s}px)`, gap: `${GAP}px` } });
      const ellRow = () => h('div', { class: 'qkv-ell', style: { gridColumn: '1 / -1', height: `${s}px` } }, '⋮');
      const xGrid = grid(d), wGrid = grid(d), qGrid = grid(1);
      for (let k = 0; k < d; k++) {
        const e = cellEl(s, D.Xd[t * d + k] / rangeX);
        e.dataset.cell = `${t},${k},0`;
        xGrid.append(e);
      }
      const qEls = new Array(dh).fill(null), qMap = new Map();
      shown.forEach((r) => {
        if (r < 0) { wGrid.append(ellRow()); qGrid.append(ellRow()); return; }
        for (let k = 0; k < d; k++) wGrid.append(cellEl(s, D.WS[r * d + k] / rangeW));
        const q = cellEl(s, null);
        q.dataset.cell = `${t},${hi * dh + r},0`;
        qGrid.append(q);
        qEls[r] = q;
        qMap.set(q, r);
      });
      const box = (cls, x, y, w, hh, kind, kid) => h('div', { class: `qkv-box ${cls}`, style: { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${hh}px`, '--kind': `var(--c-${kind})` } }, kid);
      const xBox = box('qkv-xbox', G.wX, G.xY, G.boxW, G.xH, preLN ? 'norm' : 'resid', xGrid);
      xBox.dataset.trace = attnInKey(l, preLN);
      const wBox = box('qkv-wbox', G.wX, G.wY, G.boxW, G.boxH, p, wGrid);
      const qBox = box('qkv-qbox', G.qX, G.wY, G.colW, G.boxH, p, qGrid);
      qBox.dataset.trace = `L${l}.attn.${p}`;
      const tagX = h('div', { class: 'qkv-tag', style: { left: `${G.wX}px`, top: `${G.y0}px` } });
      const tagQ = h('div', { class: 'qkv-tag', style: { left: `${G.qX}px`, top: `${G.tagY}px` } });
      const tagW = h('div', { class: 'qkv-tag', style: { right: `${HERO.w - G.wX - G.boxW}px`, top: `${G.tagY}px` } });
      const rowBand = h('div', { class: 'qkv-rowband', style: { left: `${G.wX + PAD - 3}px`, width: `${G.boxW - 2 * PAD + 6}px`, height: `${s + 6}px` } });
      const scanBox = () => h('div', { class: 'qkv-scan', style: { width: `${s + 4}px`, height: `${s + 4}px`, transitionDuration: `${2 * perCol}ms, 300ms, 300ms` } });
      const scanX = scanBox(), scanW = scanBox();
      scanX.style.top = `${G.xY + PAD - 2}px`;
      const sumEl = h('div', { class: 'qkv-sum', style: { left: `${G.qX + G.colW + 6}px` } });
      prodEl.append(qBox, wBox, xBox, tagX, tagQ, tagW, rowBand, scanW, scanX, sumEl);
      prod = { key, D, G, pitch, shown, qEls, qMap, rangeC, filled: new Set(), focusC: cMax, xBox, wBox, qBox, tagX, tagQ, tagW, rowBand, scanX, scanW, sumEl };
      setTags(D);
      qBox.addEventListener('pointerover', onQHover);
    }

    /** A token change: the input row and the output column take the new row's values. */
    function paintRow(D) {
      const { d, dh, t, hi } = D;
      const rangeX = maxAbs(D.Xd);
      prod.D = D;
      prod.xBox.querySelectorAll('.k-cell').forEach((cell, k) => { cell.dataset.cell = `${t},${k},0`; paint(cell, D.Xd[t * d + k] / rangeX); });
      prod.qMap.forEach((r, cell) => {
        cell.dataset.cell = `${t},${hi * dh + r},0`;
        if (prod.filled.has(r)) paint(cell, D.C[t * dh + r] / prod.rangeC);
      });
      setTags(D);
    }

    /** Puts the highlight on row c of W and on its output cell, and the running sum beside them, without moving the scan. */
    function placeRowBand(c) {
      const { G, shown, pitch } = prod, s = G.s, di = shown.indexOf(c), y = G.wY + PAD + di * pitch;
      prod.rowBand.style.top = `${y - 3}px`;
      prod.rowBand.style.opacity = '1';
      prod.scanW.style.top = `${y - 2}px`;
      prod.sumEl.style.top = `${y + s / 2 - 8}px`;
      prod.sumEl.style.opacity = '1';
      prod.qEls.forEach((cell) => cell?.classList.remove('hl'));
      prod.qEls[c]?.classList.add('hl');
    }
    /** The scan's column x: the same column of the input row and of W's row. */
    function moveScan(k) {
      const x = prod.G.wX + PAD + k * prod.pitch - 2;
      prod.scanX.style.left = `${x}px`;
      prod.scanW.style.left = `${x}px`;
    }
    /** The live line for entry c: its traced value (once computed) and the sum it stands for. */
    function setEntryLive(c, value) {
      const { D } = prod, { t, hi, dh, d, p } = D, col = hi * dh + c;
      const live = runLive(ctx.state, `L${l}.attn.${p}`, { r: t, x: col, band: 0 });
      const where = `token “${ctx.label(t)}” at position ${t}`;
      sNote.textContent = live?.note ? `${live.note} · ${where}` : where;
      const lhs = `${SYM_TEX[p]}_{${t},${col}}`;
      if (value == null) {
        sTex.innerHTML = texHTML(`${lhs} = \\dots`, { display: true });
        sMath.textContent = '';
        return;
      }
      sTex.innerHTML = texHTML(`${lhs} = ${num(live?.result ?? value, 3)}`, { display: true });
      sMath.textContent = `Σ over the ${d} inputs: X[${t}, ·] · W_${SYM[p]},${hi}[${c}, ·]${D.bS ? ` + b_${SYM[p]},${hi}[${c}]` : ''}`;
    }
    /** The row check: the recomputed row against the trace, one line under the live entry. */
    function showRowCheck(D) {
      sCheck.replaceChildren(checkBadge(rowError(D), 0, {
        tol: 2e-4, label: `equals the traced ${SYM[D.p]} row, ${D.p === 'q' ? 'head' : 'KV head'} ${D.hi}`,
      }));
    }
    /** An output entry is drawn: its colour from the trace, with a small pop when it is the focus. */
    function fillEntry(r, pop) {
      if (prod.filled.has(r)) return;
      prod.filled.add(r);
      const cell = prod.qEls[r];
      if (!cell) return;
      paint(cell, prod.D.C[prod.D.t * prod.D.dh + r] / prod.rangeC);
      if (pop && !reducedMotion()) cell.animate([{ transform: 'scale(1.3)' }, { transform: 'scale(1)' }], { duration: 360, easing: MOTION.easeOut });
    }
    /** The dot product of entry c, term by term over the inputs (about SWEEP_MS), then the entry is drawn.
     *  The sweep follows the clock, one frame at a time, so a slow frame cannot stretch it. */
    function startEntry(c, { cascade = false, delay = 0 } = {}) {
      stopSweep();
      const { D } = prod;
      prod.focusC = c;
      placeRowBand(c);
      prod.scanX.style.opacity = '1';
      prod.scanW.style.opacity = '1';
      moveScan(0);
      const ps = partialSums(D, c), final = ps[D.d - 1] + (D.bS ? D.bS[c] : 0);
      const perCol = clamp(SWEEP_MS / D.d, 7, 24), total = D.d * perCol;
      setEntryLive(c, null);
      prod.sumEl.textContent = fmt(ps[0], 2);
      let t0 = null, lastK = -1;
      const frame = (now) => {
        sweepRaf = 0;
        if (t0 === null) t0 = now + delay;
        const e = now - t0;
        if (e >= 0) {
          const k = Math.min(D.d - 1, Math.floor(e / perCol));
          if (k !== lastK) {
            lastK = k;
            moveScan(k);
            prod.sumEl.textContent = fmt(ps[k], 2);
            sMath.textContent = `Σ over k ≤ ${k} of ${D.d} products`;
          }
        }
        if (e >= total + 140) { finishEntry(c, final, cascade); return; }
        sweepRaf = requestAnimationFrame(frame);
      };
      sweepRaf = requestAnimationFrame(frame);
    }
    function finishEntry(c, value, cascade) {
      prod.scanX.style.opacity = '0';
      prod.scanW.style.opacity = '0';
      prod.sumEl.textContent = fmt(value, 2);
      fillEntry(c, true);
      setEntryLive(c, value);
      showRowCheck(prod.D);
      if (cascade) {
        let n = 0;
        for (let r = 0; r < prod.D.dh; r++) if (!prod.filled.has(r)) sweepT.push(setTimeout(() => fillEntry(r, false), 16 * ++n));
      }
    }
    /** Reduced motion: the end state at once, with the same numbers and check. */
    function settle(D) {
      prod.scanX.style.opacity = '0';
      prod.scanW.style.opacity = '0';
      placeRowBand(D.cMax);
      prod.sumEl.textContent = fmt(entryValue(D, D.cMax), 2);
      for (let r = 0; r < D.dh; r++) fillEntry(r, false);
      setEntryLive(D.cMax, entryValue(D, D.cMax));
      showRowCheck(D);
    }
    /** Step 0: the input row is the focus; W and the empty output column wait for the next step. */
    function showInput(D) {
      prod.rowBand.style.opacity = '0';
      prod.scanX.style.opacity = '0';
      prod.scanW.style.opacity = '0';
      prod.sumEl.style.opacity = '0';
      sCheck.replaceChildren();
      const { d, t, Xd } = D;
      sNote.textContent = `token “${ctx.label(t)}” at position ${t}`;
      sTex.innerHTML = texHTML(`X_{${t},:} = (${num(Xd[t * d], 2)},\\ ${num(Xd[t * d + 1], 2)},\\ ${num(Xd[t * d + 2], 2)},\\ \\dots)`, { display: true });
      sMath.textContent = `Row ${t} of X: ${d} entries, one per input feature.`;
    }
    /** Hovering an output entry runs its dot product (once the row is drawn). */
    function onQHover(e) {
      if (!prod || step === 0 || prod.filled.size < prod.D.dh) return;
      const cell = e.target.closest?.('.k-cell');
      const c = cell ? prod.qMap.get(cell) : undefined;
      if (c === undefined || c === prod.focusC) return;
      clearTimeout(hoverT);
      hoverT = setTimeout(() => {
        hoverT = 0;
        if (reducedMotion()) {
          prod.focusC = c;
          placeRowBand(c);
          prod.sumEl.textContent = fmt(entryValue(prod.D, c), 2);
          setEntryLive(c, entryValue(prod.D, c));
        } else startEntry(c, { cascade: false });
      }, 120);
    }

    /** A projection step: the matrix of this head is shown, and the focus entry's dot product runs. */
    function showProduct(D, replay) {
      prodEl.classList.remove('off');
      bandsEl.classList.add('off');
      const shown = keepSpan(D.dh, D.cMax, D.cMax, ROWS_MAX);
      const key = shown.join(',');
      const rebuild = replay || !prod || prod.key !== key || prod.D.d !== D.d;
      if (rebuild) buildProduct(D, shown, key); else paintRow(D);
      prod.xBox.classList.toggle('focus', step === 0);
      prod.wBox.classList.toggle('dim', step === 0);
      prod.qBox.classList.toggle('dim', step === 0);
      if (step === 0) { showInput(D); return; }
      const incomplete = prod.filled.size < D.dh;
      if (reducedMotion()) { settle(D); return; }
      if (replay) prod.wBox.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 420, easing: MOTION.easeOut });
      startEntry(D.cMax, { cascade: replay || incomplete, delay: replay ? 260 : 0 });
      const perCol = clamp(SWEEP_MS / D.d, 7, 24);
      holdMs = Math.max(MOTION.step, 260 + D.d * perCol + 140 + 16 * D.dh + 500);
    }

    // ------------------------------------------------------------ the bands (step 4)
    function buildBands(D) {
      bandsEl.replaceChildren();
      const { T, H, dh, width, t } = D;
      const QH = ctx.state.trace[`L${l}.attn.q_heads`];    // [H, T, dh]
      const range = maxAbs(D.full.data);
      const cellFit = ((HERO.w * 0.9 - (H - 1) * SPLIT_GAP[0]) / H - (dh - 1) * GAP) / dh;   // the bands fit 90% of the hero
      const cell = clamp(Math.min(360 / T, cellFit, 14), 3, 14);
      const bw = dh * cell + (dh - 1) * GAP, bh = T * cell + (T - 1) * GAP;
      const joinW = H * bw;
      const splitGap = H > 1 ? clamp((HERO.w * 0.9 - joinW) / (H - 1), SPLIT_GAP[0], SPLIT_GAP[1]) : 0;
      const splitW = joinW + (H - 1) * splitGap;
      const top = Math.max(54, (HERO.h - bh) / 2 + 8);     // clears the band labels from the token picker's second row
      const x0J = (HERO.w - joinW) / 2, x0S = (HERO.w - splitW) / 2;
      bandsEl.append(h('div', { class: 'qkv-joined', style: { left: `${x0J - 10}px`, top: `${top - 10}px`, width: `${joinW + 20}px`, height: `${bh + 20}px` } }));
      const pitch = cell + GAP, items = [];
      let err = 0;
      for (let hh = 0; hh < H; hh++) {
        const grid = h('div', { class: 'qkv-grid', style: { gridTemplateColumns: `repeat(${dh}, ${cell}px)`, gap: `${GAP}px` } });
        for (let r = 0; r < T; r++) for (let c = 0; c < dh; c++) {
          const v = QH.data[(hh * T + r) * dh + c];
          const ce = cellEl(cell, v / range);
          ce.dataset.cell = `${r},${c},${hh}`;   // r, column within the head, band (head)
          grid.append(ce);
          err = Math.max(err, Math.abs(v - D.full.data[r * width + hh * dh + c]));
        }
        const el = h('div', {
          class: 'qkv-band', dataset: { trace: `L${l}.attn.q_heads`, node: `head.${l}.${hh}` },
          style: { width: `${bw}px`, height: `${bh}px`, top: `${top}px`, left: '0px' },
        },
        h('div', { class: 'qkv-frame' }), h('div', { class: 'qkv-band-bar' }),
        h('div', { class: 'qkv-band-lab' }, H >= 9 ? `${hh}` : `head ${hh}`), grid,
        h('div', { class: 'qkv-band-row', style: { top: `${t * pitch - 1}px`, height: `${cell + 2}px` } }));
        bandsEl.append(el);
        items.push({ el, joinX: x0J + hh * bw, splitX: x0S + hh * (bw + splitGap) });
      }
      bands = { items, err };
      items.forEach((it, i) => it.el.classList.toggle('sel', i === D.hd));
    }

    function placeBands(isSplit, instant = false) {
      if (!bands) return;
      bandsEl.classList.toggle('split', isSplit);
      if (instant) bandsEl.classList.add('instant');
      bands.items.forEach((it) => { it.el.style.left = `${isSplit ? it.splitX : it.joinX}px`; });
      if (instant) { void bandsEl.offsetWidth; bandsEl.classList.remove('instant'); }
    }

    function showSplitCheck() {
      if (!bands) return;
      sCheck.replaceChildren(checkBadge(bands.err, 0, { label: `each head is a band of L${l}.attn.q` }));
    }

    function setSplitLive(D) {
      const { t, hd, dh, cMax: c } = D;
      const live = runLive(ctx.state, `L${l}.attn.q_heads`, { r: t, x: c, band: hd });
      const where = `token “${ctx.label(t)}” at position ${t}`;
      sNote.textContent = live?.note ? `${live.note} · ${where}` : where;
      sTex.innerHTML = texHTML(`(${SYM_TEX.q}_{${hd}})_{${t},${c}} = ${num(live?.result ?? D.C[t * dh + c], 3)}`, { display: true });
      sMath.textContent = `the same entry as Q[${t}, ${hd * dh + c}] in the full matrix`;
    }

    function showBands(D, replay) {
      prodEl.classList.add('off');
      bandsEl.classList.remove('off');
      const fresh = replay || !bands;
      buildBands(D);                                       // rebuilt every time: the row marker follows the token
      if (fresh && !reducedMotion()) {
        placeBands(false, true);                           // the full Q first, as one block
        later(450, () => placeBands(true));                // then the bands slide apart
        later(450 + MOTION.move + 50, showSplitCheck);
      } else {
        placeBands(true, true);
        showSplitCheck();
      }
      setSplitLive(D);
      holdMs = MOTION.step;
    }

    // ------------------------------------------------------------ the side context: the full projection, one token's row marked
    /** The side context: the input X (step 0) or the whole projection, with the token's row marked and the
     *  head's columns outlined. Rows and columns beyond the budget are elided; every drawn cell carries data-cell. */
    function ctxParts(D, isX) {
      const { T, d, t, hi, dh, width } = D;
      const data = isX ? D.Xd : D.full.data, cols = isX ? d : width;
      const range = maxAbs(data);
      const rows = keepSpan(T, t, t, CTX_ROWS);
      const colsIdx = isX
        ? keepSpan(cols, 0, 0, CTX_COLS)
        : keepSpan(cols, hi * dh, Math.min(hi * dh + dh - 1, hi * dh + CTX_COLS - 3), CTX_COLS);
      const grid = h('div', { class: 'qkv-cgrid' });
      rows.forEach((r) => {
        if (r < 0) { grid.append(h('div', { class: 'qkv-cell-ell' }, '⋮')); return; }
        const row = h('div', { class: ['qkv-cr', r === t && 'hit'] });
        colsIdx.forEach((c) => {
          if (c < 0) { row.append(h('span', { class: 'qkv-ce' })); return; }
          const cell = h('div', { class: 'k-cell', dataset: { cell: `${r},${c},0` }, style: { width: `${CTX_CELL}px`, height: `${CTX_CELL}px`, borderRadius: '1px' } });
          paint(cell, data[r * cols + c] / range);
          row.append(cell);
        });
        grid.append(row);
      });
      const colX = [];
      let x = 0;
      colsIdx.forEach((c) => { colX.push(x); x += (c < 0 ? CTX_ELL_COL : CTX_CELL) + CTX_GAP; });
      const gridW = x - CTX_GAP;
      const gridH = rows.reduce((a, r) => a + (r < 0 ? CTX_ELL_ROW : CTX_CELL), 0) + (rows.length - 1) * CTX_GAP;
      const kind = isX ? (D.preLN ? 'norm' : 'resid') : D.p;
      const box = h('div', {
        class: 'qkv-ctxbox',
        dataset: { trace: isX ? attnInKey(l, D.preLN) : `L${l}.attn.${D.p}` },
        style: { '--kind': `var(--c-${kind})` },
      }, grid);
      if (!isX) {
        const inHead = (c) => c >= hi * dh && c <= hi * dh + dh - 1;
        const first = colsIdx.findIndex(inHead);
        let last = -1;
        colsIdx.forEach((c, i) => { if (inHead(c)) last = i; });
        if (first >= 0) {
          const x0 = colX[first], x1 = colX[last] + CTX_CELL;
          const put = (cls, left, width, extra = {}) => box.append(h('div', { class: cls, style: { left: `${PAD + left}px`, top: `${PAD}px`, width: `${Math.max(0, width)}px`, height: `${gridH}px`, ...extra } }));
          put('qkv-cmask', 0, x0);
          put('qkv-cmask', x1, gridW - x1);
          box.append(h('div', { class: 'qkv-cband', style: { left: `${PAD + x0 - 2}px`, top: `${PAD - 2}px`, width: `${x1 - x0 + 4}px`, height: `${gridH + 4}px` } }));
        }
      }
      const label = isX
        ? `X  ${T} × ${d}  ·  row ${t} outlined`
        : `${SYM[D.p]}  ·  ${D.p === 'q' ? 'head' : 'KV head'} ${hi} = columns ${hi * dh}–${(hi + 1) * dh - 1} of ${width}`;
      return [h('div', { class: 'qkv-ctx-lab' }, label), box];
    }

    // ------------------------------------------------------------ steps
    function render({ replay = true } = {}) {
      clearTimers();
      themed = [];
      syncDense();
      const D = read(ctx, l, PROJ_OF_STEP[step]);
      const tok = `“${esc(ctx.label(D.t))}”`;
      holdMs = MOTION.step;
      header.setChips([[`Layer ${l}`, 'resid'], [`Head ${D.hd}`, 'attn']]);
      header.setSub(step === 0 ? INPUT_SUB : step === 4 ? splitSub(D) : subText(D));
      sLab.textContent = STEPS[step];
      sLab.style.setProperty('--kind', `var(--c-${step === 0 ? (D.preLN ? 'norm' : 'resid') : D.p})`);
      sShape.textContent = shapeText(D, step);
      sEq.innerHTML = texHTML(eqTeX(D, step), { display: true });
      sCtx.replaceChildren(...ctxParts(D, step === 0));
      sCheck.replaceChildren();
      keyEl.classList.toggle('off', step === 4);
      if (step === 4) showBands(D, replay); else showProduct(D, replay);
      cap.set(capText(D, step, tok));
    }

    function enterStep(i) {
      if (!advancing) stopAuto();                          // a step chosen by hand stops the auto-walk
      step = i;
      render({ replay: true });
    }

    // ------------------------------------------------------------ start
    ctx.setSteps(STEPS, enterStep);
    render({ replay: true });
    appear([header.el, picker.el, heads.el, heroEl, sideEl], { stagger: MOTION.stagger });

    // Auto-advance the first time the scene is shown. Each step holds for its own animation plus a moment to read it.
    const advance = () => {
      auto = null;
      if (step >= STEPS.length - 1) return;
      advancing = true;
      try { ctx.setStep(step + 1); } finally { advancing = false; }
      if (step < STEPS.length - 1) auto = setTimeout(advance, holdMs);
    };
    if (!reducedMotion()) auto = setTimeout(advance, MOTION.step + 800);
    el.addEventListener('atlas:interrupt', stopAuto);
    el.addEventListener('pointerdown', stopAuto, { once: true });

    offs.push(ctx.on('trace', () => { heads.render(); render({ replay: true }); }));
    offs.push(ctx.on('select', () => {
      const head = ctx.state.sel.head, changedHead = head !== lastHead;   // a new head replays the product; a new token re-reads it
      lastHead = head;
      render({ replay: changedHead });
    }));
    const themeOff = onThemeChange(() => {
      keyPaint();
      themed.forEach(([cell, x]) => { if (cell.isConnected) cell.style.background = diverging(x); });
    });

    return {
      pause() { paused = true; stopAuto(); clearTimers(); },
      resume() { if (!paused) return; paused = false; render({ replay: false }); },
      destroy() {
        stopAuto();
        clearTimers();
        offs.forEach((off) => off());
        themeOff();
        st.destroy();
      },
    };
  },
};

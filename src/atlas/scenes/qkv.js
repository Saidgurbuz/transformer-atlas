// Q, K and V: where queries, keys and values come from. Each one is the attention input X times
// one head's rows of its weight matrix, plus bias. The hero is that product on the real weights,
// computed cell by cell and checked against the traced projection: the matrix fills row by row,
// then the selected token's largest entry runs its dot product (the entry the live line shows).
// Every cell of the product and of the side context carries the atlas hover card and menu.
// The last step splits Q into its head bands. Steps: input X, queries, keys, values, split into heads.
// Reference scene for contributors: see docs/ATLAS_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, headPicker, checkBadge, mat, chip, appear, ZONES, MOTION, useStyle } from '../kit.js';
import { h, num, texHTML, clamp, segmented, diverging, onThemeChange, reducedMotion, hideTip } from '../../ui/index.js';
import { matmulAnim } from '../../ui/matmul.js';
import { runLive } from '../../core/math/index.js';
import { attnInput } from '../../core/math/util.js';

const STEPS = ['Input X', 'Queries', 'Keys', 'Values', 'Split into heads'];
const STEP_CHIP = [['Input X', 'norm'], ['Queries · Q', 'q'], ['Keys · K', 'k'], ['Values · V', 'v'], ['Split · Q', 'q']];
const PROJ_OF_STEP = ['q', 'q', 'k', 'v', 'q'];           // the split is of Q, as the spec asks
const STEP_OF_PROJ = { q: 1, k: 2, v: 3 };
const SYM = { q: 'Q', k: 'K', v: 'V' };
const SYM_TEX = { q: String.raw`\Q`, k: String.raw`\K`, v: String.raw`\V` };
const NAME = { q: 'Queries', k: 'Keys', v: 'Values' };
const HERO = { x: 110, y: 262, w: 1030, ht: 490 };       // hero zone (design units)
const SIDE = { x: 1180, y: 262 };                        // side column, 340 wide
const MM_MAX_H = HERO.ht;                                // the product canvas may use the whole hero: no title or equation bar
const MM_OVERHEAD = 26 + 16 + 30;                        // ui/matmul.js: name row above B, gap B to C, name row under C
const MM_GAP_AC = 26;                                    // ui/matmul.js: gap between A and the C block (B without row labels)
const CELL_MODE_MAX = 36, CELL_PLAY_MS = 4200, ROW_PLAY_MS = [3000, 6000];   // mirror ui/matmul.js playback
const DENSE_PICKER_T = 48;                               // above this many tokens the token picker wraps to two rows

useStyle('qkv', `
.qkv-seg { display: flex; }
.qkv-heads { flex-wrap: wrap; width: 340px; row-gap: 6px; }
.qkv-heads .k-pick-head { width: 30px; }
.qkv-picker { width: ${HERO.w}px; }
.qkv-picker .k-picker { flex-wrap: wrap; row-gap: 4px; max-width: 1000px; }
.qkv-picker.dense .k-pick { height: 30px; }
.qkv-hero { position: relative; }
.qkv-legend { position: absolute; right: 0; top: 0; transition: opacity .35s ease; }
.qkv-legend.off { opacity: 0; visibility: hidden; }
.qkv-slot { position: absolute; left: 0; top: 0; width: 100%; height: 100%; transition: opacity .35s ease, visibility 0s linear 0s; }
.qkv-slot.off { opacity: 0; visibility: hidden; pointer-events: none; transition: opacity .35s ease, visibility 0s linear .35s; }
.qkv-mm { display: flex; align-items: center; justify-content: center; }
.qkv-mm .mm { width: 560px; max-width: 100%; }
.qkv-mm .mm-stage { position: relative; display: flex; justify-content: center; }
.qkv-mm .mm-head, .qkv-mm .mm-eq, .qkv-mm .mm-foot { display: none; }
.qkv-hit { position: absolute; display: grid; touch-action: pan-y; }
.qkv-hit > i { display: block; }
.qkv-band { position: absolute; top: 0; transition: left ${MOTION.move}ms ${MOTION.ease}; }
.qkv-bands.instant .qkv-band { transition: none; }
.qkv-grid { display: grid; gap: 1px; }
.qkv-frame { position: absolute; inset: -10px; border-radius: 10px; border: 1px solid color-mix(in srgb, var(--tint) 50%, var(--line)); background: color-mix(in srgb, var(--surface) 72%, transparent); box-shadow: 0 14px 34px rgba(0,0,0,.16); opacity: 0; transition: opacity .45s ease, border-color .3s ease, box-shadow .3s ease; }
.qkv-bands.split .qkv-frame { opacity: 1; }
.qkv-joined { position: absolute; border-radius: 12px; border: 1px solid color-mix(in srgb, var(--c-q) 45%, var(--line)); background: color-mix(in srgb, var(--surface) 72%, transparent); box-shadow: 0 14px 34px rgba(0,0,0,.16); transition: opacity .45s ease; }
.qkv-bands.split .qkv-joined { opacity: 0; }
.qkv-band-bar { position: absolute; left: 0; right: 0; top: -17px; height: 4px; border-radius: 2px; background: var(--tint); transition: opacity .4s ease; }
.qkv-bands.split .qkv-band-bar { opacity: 0; }
.qkv-band-lab { position: absolute; left: -12px; right: -12px; top: -46px; text-align: center; font: 600 15px/1 var(--font-mono); color: var(--tint); white-space: nowrap; transition: color .3s ease; }
.qkv-band.sel .qkv-frame { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent), 0 14px 34px rgba(0,0,0,.16); }
.qkv-band.sel .qkv-band-bar { background: var(--accent); }
.qkv-band.sel .qkv-band-lab { color: var(--accent); }
.qkv-side { width: 340px; display: grid; gap: 10px; align-content: start; }
.qkv-lab { text-align: left; color: var(--kind, var(--fg-2)); }
.qkv-eq { font-size: 25px; overflow-x: auto; }
.qkv-lab-row { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 2px 10px; }
.qkv-lab-row .k-label { margin-bottom: 0; }
.qkv-shape { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); text-align: right; }
.qkv-live { font-size: 17px; padding: 10px 12px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.qkv-note { font: 500 13px/1.35 var(--font-mono); color: var(--fg-3); margin-bottom: 4px; }
.qkv-ctx { display: grid; gap: 6px; justify-items: start; }
.qkv-ctx-lab { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); }
.qkv-hm-mask { position: absolute; background: color-mix(in srgb, var(--bg) 62%, transparent); pointer-events: none; }
.qkv-hm-band { position: absolute; border: 2px solid var(--kind); border-radius: 4px; pointer-events: none; }
.qkv-check { min-height: 28px; }
@media (prefers-reduced-motion: reduce) { .qkv-slot, .qkv-band, .qkv-frame, .qkv-joined, .qkv-band-bar, .qkv-band-lab { transition: none !important; } }
`);

// ------------------------------------------------------------------ helpers

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));   // HTML strings only
const maxAbs = (a) => { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m && Number.isFinite(v)) m = v; } return m || 1; };
/** Matmul playback length at speed 1 for a T × d_h product (the same rule as ui/matmul.js). */
const playMs = (T, dh) => (T * dh <= CELL_MODE_MAX ? CELL_PLAY_MS : clamp(120 * T, ROW_PLAY_MS[0], ROW_PLAY_MS[1]));
/** One entry's dot product over K inputs (ui/matmul.js: K steps of 50 to 170 ms, then a 520 ms tail). */
const focusMs = (K) => K * clamp(1000 / K, 50, 170) + 520;
/** The attention input's trace key: LN1(x) for pre-LN, the block input (the stream) for post-LN. */
const attnInKey = (l, preLN) => (preLN ? `L${l}.ln1` : l === 0 ? 'emb.out' : `L${l - 1}.resid_out`);
const article = (n) => (/^(8\d*|11|18)$/.test(String(n)) ? 'an' : 'a');   // "an 8-wide slice", "a 16-wide slice"

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

// Captions are HTML: the token name is escaped here, and only here.
function capText(D, step, tok) {
  const { t, dh, hi } = D;
  switch (step) {
    case 0: return `Q, K and V are three linear maps of one matrix <b>X</b>, one row per token. Row ${t} is the input of ${tok} at position ${t}.`;
    case 1: return `Each query is a row of X times the <b>${dh}</b> rows of W_Q that belong to head ${hi}. Row ${t} is the query of ${tok} at position ${t}.`;
    case 2: return `Keys use W_K in the same way: row ${t} is the key of ${tok} at position ${t}. ${keyNote(D)}`;
    case 3: return `Values use W_V and are read through the same KV head as the keys. Row ${t} is the value of ${tok}, the vector attention copies from it.`;
    default: return `Q is ${D.H} blocks of <b>${dh}</b> columns side by side. Row ${t} is the query of ${tok} in every head.`;
  }
}

/** DOM cells laid over one block of the product canvas. Each cell carries data-cell for the atlas
 *  hover card and menu. Pointer events are forwarded to the canvas, so its own hover, pin and arrow
 *  keys keep working; the canvas tooltip is hidden in favour of the atlas card. */
function hitLayer(parent, canvas, { x, y, rows, cols, s, trace, cellOf, cursor }) {
  const grid = h('div', {
    class: 'qkv-hit', dataset: { trace },
    style: {
      left: `${canvas.offsetLeft + x}px`, top: `${canvas.offsetTop + y}px`, width: `${cols * s}px`, height: `${rows * s}px`,
      gridTemplateColumns: `repeat(${cols}, ${s}px)`, gridTemplateRows: `repeat(${rows}, ${s}px)`, cursor,
    },
  });
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) grid.append(h('i', { dataset: { cell: cellOf(r, c) } }));
  const forward = (type, ev) => canvas.dispatchEvent(new MouseEvent(type, { clientX: ev.clientX, clientY: ev.clientY }));
  grid.addEventListener('pointermove', (ev) => { forward('pointermove', ev); hideTip(); });
  grid.addEventListener('pointerleave', (ev) => forward('pointerleave', ev));
  grid.addEventListener('click', (ev) => forward('click', ev));
  parent.append(grid);
  return grid;
}

/** Origin and cell pitch of the product canvas. The origin is read back from the component; the
 *  pitch follows its height rule, which binds for every shape this scene shows, and is checked
 *  against the drawn cells so the hit cells line up with them. */
function matGeom(m, rows, cols, k) {
  const c0 = m.cellRect(0, 0);
  const sF = (MM_MAX_H - MM_OVERHEAD) / (k + rows);
  const sRow = rows > 1 ? (m.cellRect(rows - 1, 0).y - c0.y) / (rows - 1) : sF;
  const s = Math.abs(sRow - sF) <= 0.02 ? sF : sRow;
  return { xC: c0.x, yC: c0.y, xA: c0.x - MM_GAP_AC - k * s, s };
}

// ------------------------------------------------------------------ scene

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const l = ctx.params.l ?? 0;
    let step = 0, auto = null, advancing = false, timers = [], pollRaf = 0;
    let mm = null, bands = null, ctxMat = null, mmTouched = false;
    let lastHead = ctx.state.sel.head;
    const offs = [];

    const later = (ms, fn) => { timers.push(setTimeout(fn, ms)); };
    const clearTimers = () => { timers.forEach(clearTimeout); timers = []; cancelAnimationFrame(pollRaf); pollRaf = 0; };

    // ------------------------------------------------------------ header, controls
    const header = sceneHeader(st, { title: 'Q, K and V projections' });
    const eyebrow = header.el.querySelector('.k-eyebrow');
    const picker = tokenPicker(st, ctx, { label: 'token', maxWidth: HERO.w });
    picker.el.classList.add('qkv-picker');
    const syncDense = () => picker.el.classList.toggle('dense', ctx.state.ids.length > DENSE_PICKER_T);
    syncDense();

    const seg = segmented({
      options: [
        { value: 'q', label: 'Q', title: 'Queries: X · W_Qᵀ' },
        { value: 'k', label: 'K', title: 'Keys: X · W_Kᵀ' },
        { value: 'v', label: 'V', title: 'Values: X · W_Vᵀ' },
      ],
      value: 'q', ariaLabel: 'projection', onChange: (p) => ctx.setStep(STEP_OF_PROJ[p]),
    });
    st.add(h('div', { class: 'qkv-seg' }, seg.el), { x: SIDE.x, y: SIDE.y });
    const heads = headPicker(st, ctx, { x: SIDE.x, y: SIDE.y + 46 });
    heads.el.classList.add('qkv-heads');

    // ------------------------------------------------------------ hero: the product, or the bands
    const mmSlot = h('div', { class: 'qkv-slot qkv-mm' });
    const bandSlot = h('div', { class: 'qkv-slot qkv-bands off' });
    const legendSlot = h('div', { class: 'qkv-legend' });
    st.add(h('div', { class: 'qkv-hero', style: { width: `${HERO.w}px`, height: `${HERO.ht}px` } }, mmSlot, bandSlot, legendSlot), { x: HERO.x, y: HERO.y });
    mmSlot.addEventListener('pointerdown', () => { mmTouched = true; });

    // ------------------------------------------------------------ side column
    const sLab = h('div', { class: 'k-label qkv-lab' });
    const sEq = h('div', { class: 'qkv-eq' });
    const sShape = h('div', { class: 'qkv-shape' });
    const sLive = h('div', { class: 'qkv-live' });
    const sCheck = h('div', { class: 'qkv-check' });
    const sCtx = h('div', { class: 'qkv-ctx' });
    st.add(h('div', { class: 'qkv-side' }, h('div', { class: 'qkv-lab-row' }, sLab, sShape), sEq, sLive, sCheck, sCtx), { x: SIDE.x, y: SIDE.y + 90 });
    const cap = sceneCaption(st, { x: ZONES.caption.x, y: ZONES.caption.y, w: ZONES.caption.w });

    // ------------------------------------------------------------ the product (steps 0 to 3)
    function buildProduct(D, replay) {
      mm?.destroy();
      const P = SYM[D.p], i = D.hi, who = D.p === 'q' ? 'head' : 'KV head';
      const rowLabels = Array.from({ length: D.T }, (_, r) => ctx.label(r));
      mm = matmulAnim({
        A: { data: D.Xd, rows: D.T, cols: D.d, name: 'X', kind: D.preLN ? 'norm' : 'resid', shape: 'T×d', rowLabels },
        B: { data: D.WS, rows: D.dh, cols: D.d, name: `W_${P},${i}`, kind: D.p, shape: 'd_h×d', bias: D.bS ?? undefined, biasName: `b_${P},${i}` },
        transposeB: true,
        C: { data: D.C, name: `${P}_${i}`, kind: D.p, shape: 'T×d_h' },
        title: `${P}, ${who} ${i}:  X · W_${P},${i}ᵀ${D.bS ? ` + b_${P},${i}` : ''}`,
        maxWidth: HERO.w, maxHeight: MM_MAX_H, maxCell: 56, digits: 2,
      });
      mmSlot.replaceChildren(mm.el);
      const scale = mm.el.querySelector('.mm-scale');        // the colour key moves to the hero corner
      legendSlot.replaceChildren(...(scale ? [scale] : []));
      addHitLayers(D);
      if (replay && !reducedMotion()) mmSlot.animate([{ opacity: 0.25 }, { opacity: 1 }], { duration: 360, easing: MOTION.easeOut });
    }

    /** Cells over the input block (X, the attention input) and the product block (C): the hover card
     *  and menu read the traced tensors, and the pointer drives the matmul's own hover and pin. */
    function addHitLayers(D) {
      const canvas = mm.el.querySelector('canvas'), stageEl = mm.el.querySelector('.mm-stage');
      const g = matGeom(mm, D.T, D.dh, D.d);
      hitLayer(stageEl, canvas, {
        x: g.xA, y: g.yC, rows: D.T, cols: D.d, s: g.s, cursor: 'default',
        trace: attnInKey(l, D.preLN), cellOf: (r, kk) => `${r},${kk},0`,
      });
      hitLayer(stageEl, canvas, {
        x: g.xC, y: g.yC, rows: D.T, cols: D.dh, s: g.s, cursor: 'pointer',
        trace: `L${l}.attn.${D.p}`, cellOf: (r, c) => `${r},${D.hi * D.dh + c},0`,
      });
    }

    function showCheck(D) {
      if (!mm) return;
      sCheck.replaceChildren(checkBadge(mm.maxError() ?? 0, 0, { label: `equals L${l}.attn.${D.p}, ${D.p === 'q' ? 'head' : 'KV head'} ${D.hi}` }));
    }

    /** The matrix fills row by row (the matmul's own play). When it has finished, the check appears and
     *  the selected token's largest entry runs its dot product. Timed by the component, not the clock. */
    function settleProduct(D) {
      cancelAnimationFrame(pollRaf);
      const tick = () => {
        if (mm.playing) { pollRaf = requestAnimationFrame(tick); return; }
        pollRaf = 0;
        showCheck(D);
        if (!mmTouched) mm.focus(D.t, D.cMax);
      };
      pollRaf = requestAnimationFrame(tick);
    }

    function showProduct(D, replay) {
      mmSlot.classList.remove('off');
      bandSlot.classList.add('off');
      legendSlot.classList.remove('off');
      if (bands) placeBands(false);                        // the bands slide back while the product returns
      if (replay || !mm) {
        buildProduct(D, replay);
        if (step > 0) { mmTouched = false; mm.play(); settleProduct(D); }
        return;
      }
      if (step === 0) return;                              // a new token only re-reads the input row
      mmTouched = false;
      settleProduct(D);                                    // its entry runs once the matrix is filled
    }

    // ------------------------------------------------------------ the bands (step 4)
    function placeBands(isSplit, instant = false) {
      if (!bands) return;
      bandSlot.classList.toggle('split', isSplit);
      if (instant) bandSlot.classList.add('instant');
      bands.items.forEach((it) => { it.el.style.left = `${isSplit ? it.splitX : it.joinX}px`; });
      if (instant) { void bandSlot.offsetWidth; bandSlot.classList.remove('instant'); }
    }

    function updateSel(D) {
      if (!bands) return;
      bands.items.forEach((it, i) => it.el.classList.toggle('sel', i === D.hd));
    }

    function buildBands(D) {
      bandSlot.replaceChildren();
      const { T, H, dh, width } = D;
      const QH = ctx.state.trace[`L${l}.attn.q_heads`];    // [H, T, dh]
      const range = maxAbs(D.full.data);
      const cell = clamp(Math.min(360 / T, (HERO.w * 0.7) / (H * dh), 14), 3, 14);
      const gap = 1;
      const bw = dh * cell + (dh - 1) * gap, bh = T * cell + (T - 1) * gap;
      const joinW = H * bw;
      const splitGap = H > 1 ? clamp((HERO.w * 0.92 - H * bw) / (H - 1), 16, 120) : 0;
      const splitW = H * bw + (H - 1) * splitGap;
      const top = (HERO.ht - bh) / 2 + 8;
      const x0J = (HERO.w - joinW) / 2, x0S = (HERO.w - splitW) / 2;
      bandSlot.append(h('div', { class: 'qkv-joined', style: { left: `${x0J - 10}px`, top: `${top - 10}px`, width: `${joinW + 20}px`, height: `${bh + 20}px` } }));
      const items = [], allCells = [];
      let err = 0;
      for (let hh = 0; hh < H; hh++) {
        const tint = `color-mix(in srgb, var(--c-q) ${Math.round(100 - (hh * 50) / Math.max(1, H - 1))}%, var(--fg-3))`;
        const grid = h('div', { class: 'qkv-grid', style: { '--cell': `${cell}px`, gridTemplateColumns: `repeat(${dh}, ${cell}px)` } });
        for (let t = 0; t < T; t++) for (let c = 0; c < dh; c++) {
          const v = QH.data[(hh * T + t) * dh + c];
          const ce = h('div', { class: 'k-cell', dataset: { cell: `${t},${c},${hh}` } });   // r, column within the head, band (head)
          ce.style.background = diverging(v / range);
          grid.append(ce);
          allCells.push([ce, v]);
          err = Math.max(err, Math.abs(v - D.full.data[t * width + hh * dh + c]));
        }
        const el = h('div', {
          class: 'qkv-band', dataset: { trace: `L${l}.attn.q_heads`, node: `head.${l}.${hh}` },
          style: { width: `${bw}px`, height: `${bh}px`, top: `${top}px`, left: '0px', '--tint': tint },
        },
        h('div', { class: 'qkv-frame' }), h('div', { class: 'qkv-band-bar' }),
        h('div', { class: 'qkv-band-lab' }, `${D.p === 'q' ? 'head' : 'KV head'} ${hh}`), grid);
        bandSlot.append(el);
        items.push({ el, joinX: x0J + hh * bw, splitX: x0S + hh * (bw + splitGap) });
      }
      bands = { items, cells: allCells, range, err };
      updateSel(D);
    }

    function showSplitCheck() {
      if (!bands) return;
      sCheck.replaceChildren(checkBadge(bands.err, 0, { label: `each head is a band of L${l}.attn.q` }));
    }

    function showBands(D, replay) {
      mmSlot.classList.add('off');
      bandSlot.classList.remove('off');
      legendSlot.classList.add('off');                     // the bands' labels sit where the colour key would be
      if (replay || !bands) {
        buildBands(D);
        placeBands(false, true);                           // the full Q first, as one block
        later(450, () => placeBands(true));                // then the bands slide apart
        later(450 + MOTION.move + 50, showSplitCheck);
      } else {
        updateSel(D);
      }
    }

    // ------------------------------------------------------------ text: equation, live entry, context, caption
    function liveFor(D) {
      const s = ctx.state, t = D.t, dh = D.dh, c = D.cMax;
      const where = `token “${ctx.label(t)}” at position ${t}`;
      if (step === 0) {
        const row = D.Xd.subarray(t * D.d, (t + 1) * D.d);
        return {
          note: where,
          tex: String.raw`X_{${t},:} = (${num(row[0], 2)},\ ${num(row[1], 2)},\ ${num(row[2], 2)},\ \dots)`,
        };
      }
      if (step === 4) {
        const r = runLive(s, `L${l}.attn.q_heads`, { r: t, x: c, band: D.hd });
        return r ? { note: `${r.note} · ${where}`, tex: String.raw`(Q_{${D.hd}})_{${t},${c}} = ${num(r.result, 3)}`, math: `the same entry as Q[${t}, ${D.hd * dh + c}] in the full matrix` } : null;
      }
      const col = D.hi * dh + c;                           // the entry's column in the full projection
      const r = runLive(s, `L${l}.attn.${D.p}`, { r: t, x: col, band: 0 });
      return r ? {
        note: `${r.note} · ${where}`,
        tex: `${SYM_TEX[D.p]}_{${t},${col}} = ${num(r.result, 3)}`,
        math: `Σ over the ${D.d} inputs: X[${t}, ·] · W_${SYM[D.p]},${D.hi}[${c}, ·]${D.bS ? ` + b_${SYM[D.p]},${D.hi}[${c}]` : ''}`,
      } : null;
    }

    /** The side context: the whole X (step 0) or the whole projection, with the selected token's row
     *  outlined and, for Q, K and V, the selected head's columns outlined. Every cell carries data-cell. */
    function ctxFor(D) {
      ctxMat?.destroy();
      ctxMat = null;
      const isX = step === 0;
      const data = isX ? D.Xd : D.full.data;
      const rows = D.T, cols = isX ? D.d : D.width;
      const cell = clamp(Math.floor(Math.min(300 / cols, 120 / rows)), 2, 4);
      ctxMat = mat(data, rows, cols, {
        cell, gap: 0, mode: 'diverging', range: maxAbs(data),
        kind: isX ? (D.preLN ? 'norm' : 'resid') : D.p,
        trace: isX ? attnInKey(l, D.preLN) : `L${l}.attn.${D.p}`, cellOf: (r, c) => `${r},${c},0`,
      });
      const lab = isX
        ? `X · ${rows}×${cols}, the row of “${ctx.label(D.t)}” outlined`
        : `${SYM[D.p]} · ${D.p === 'q' ? 'head' : 'KV head'} ${D.hi} = columns ${D.hi * D.dh}–${(D.hi + 1) * D.dh - 1} of ${D.width}`;
      return [h('div', { class: 'qkv-ctx-lab' }, lab), ctxMat];
    }

    /** Outlines the selected head's columns in the side context (after the context is in the page). */
    function placeHead(D) {
      const body = ctxMat.querySelector('.k-mat-body');
      if (!body || step === 0) return;
      const first = ctxMat.cellEl(0, 0), last = ctxMat.cellEl(D.T - 1, D.width - 1);
      const a = ctxMat.cellEl(0, D.hi * D.dh), b = ctxMat.cellEl(0, D.hi * D.dh + D.dh - 1);
      const top = first.offsetTop, bottom = last.offsetTop + last.offsetHeight;
      const left = first.offsetLeft, right = last.offsetLeft + last.offsetWidth;
      const x0 = a.offsetLeft, x1 = b.offsetLeft + b.offsetWidth;
      const box = (cls, x, w, extra = {}) => h('div', { class: cls, style: { left: `${x}px`, top: `${top}px`, width: `${w}px`, height: `${bottom - top}px`, ...extra } });
      body.append(
        box('qkv-hm-mask', left, x0 - left),
        box('qkv-hm-mask', x1, right - x1),
        box('qkv-hm-band', x0, x1 - x0, { '--kind': `var(--c-${D.p})` }));
    }

    function showText(D) {
      const tok = `“${esc(ctx.label(D.t))}”`;
      sEq.innerHTML = texHTML(eqTeX(D, step), { display: true });
      sShape.textContent = shapeText(D, step);
      const live = liveFor(D);
      sLive.replaceChildren(...(live ? [
        h('div', { class: 'qkv-note' }, live.note),
        h('div', { html: texHTML(live.tex, { display: true }) }),
        ...(live.math ? [h('div', { class: 'qkv-note' }, live.math)] : []),
      ] : []));
      sCtx.replaceChildren(...ctxFor(D));
      ctxMat.highlightRow(D.t);
      placeHead(D);
      cap.set(capText(D, step, tok));
    }

    // ------------------------------------------------------------ steps
    function render({ replay = true } = {}) {
      clearTimers();
      syncDense();
      const D = read(ctx, l, PROJ_OF_STEP[step]);
      const [chipText, chipKind] = STEP_CHIP[step];
      eyebrow.replaceChildren(chip(`Layer ${l}`, 'resid'), chip(`Head ${D.hd}`, 'attn'), chip(chipText, chipKind));
      header.setSub(step === 4 ? splitSub(D) : subText(D));
      seg.set(D.p);
      sLab.textContent = STEPS[step];
      sLab.style.setProperty('--kind', `var(--c-${chipKind})`);
      if (replay) sCheck.replaceChildren();
      if (step === 4) showBands(D, replay); else showProduct(D, replay);
      showText(D);
    }

    function enterStep(i) {
      if (!advancing) clearTimeout(auto);                  // a step chosen by hand stops the auto-play
      step = i;
      render({ replay: true });
    }

    // ------------------------------------------------------------ start
    ctx.setSteps(STEPS, enterStep);
    render({ replay: true });
    appear([header.el, picker.el, mmSlot.parentElement, sLab.parentElement], { stagger: MOTION.stagger });

    // auto-advance the first time the scene is shown. A product step holds for its whole animation:
    // the fill, then one entry's dot product, then a second to read it.
    const holdFor = (i) => {
      if (i === 0) return 2400;
      if (i === 4) return 3200;
      const s = ctx.state, d = s.config.d_model, dh = d / s.config.n_head;
      return playMs(s.ids.length, dh) + focusMs(d) + 1000;
    };
    if (!reducedMotion()) {
      const advance = () => {
        advancing = true;
        ctx.setStep(step + 1);
        advancing = false;
        if (step < STEPS.length - 1) auto = setTimeout(advance, holdFor(step));
      };
      auto = setTimeout(advance, holdFor(0));
    }
    el.addEventListener('pointerdown', () => clearTimeout(auto), { once: true });

    offs.push(ctx.on('trace', () => { heads.render(); render({ replay: true }); }));
    offs.push(ctx.on('select', () => {
      const head = ctx.state.sel.head;
      const changedHead = head !== lastHead;               // a new head replays the product; a new token only re-reads it
      lastHead = head;
      render({ replay: changedHead });
    }));
    const themeOff = onThemeChange(() => {
      if (bands) bands.cells.forEach(([ce, v]) => { ce.style.background = diverging(v / bands.range); });
    });

    return {
      destroy() {
        clearTimeout(auto);
        clearTimers();
        offs.forEach((off) => off());
        themeOff();
        ctxMat?.destroy();
        mm?.destroy();
        st.destroy();
      },
    };
  },
};

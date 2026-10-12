// Grouped-query attention: several query heads read the same key and value head.
// Step 0 feeds the attention input of this token (LN(x) under pre-LN, x under post-LN) to every query head, and each
// head computes its own q from its slice of W_Q. Step 1 wires each query head to KV head floor(h * n_kv_head / n_head).
// The cached K and V of each KV head (k_rot under RoPE, else k_heads; and v_heads) show the selected row, and each
// head gets that row of the k_rep and v_rep copies it reads (the reference code builds them with repeat_interleave).
// Step 2 compares KV-cache bytes per token for MHA, GQA and MQA on this model and on two published configurations.
// Every number is traced, read from the config, or computed by the engine; the published configurations are the only
// constants.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, headPicker, checkBadge, card, mat, vec, chip, fly, appear, useStyle, ZONES, MOTION } from '../kit.js';
import { h, texHTML, reducedMotion, clamp } from '../../ui/index.js';

const STEPS = ['Query heads', 'Shared K and V', 'Memory saved'];
const HERO = { x: 110, w: 1030, qTop: 268, kvTop: 540 };   // hero column; the side column starts at x 1180
const SIDE = { x: 1180, y: 268 };
const QGAP = 10, KVGAP = 16;                               // gaps between neighbouring query tiles and KV tiles
const RISE = MOTION.move;                                  // the query tiles rise over this long (a CSS transform)
const FLY = 760, FLY_GAP = 40;                             // a copy flight lasts FLY ms; heads start FLY_GAP apart
const BASE_Y = 700, BAR_MAX = 310;                         // memory bars: baseline and the full (MHA) height
const PUBLISHED = [                                        // layers, query heads, KV heads, head dim
  { name: 'Mistral-7B', L: 32, H: 32, Hkv: 8, dh: 128 },
  { name: 'LLaMA-2-70B', L: 80, H: 64, Hkv: 8, dh: 128 },
];
const CONTEXT = 4096;                                      // tokens, for the caption's absolute numbers
const HERO_MID = ZONES.hero.y + ZONES.hero.h / 2;          // the vertical middle of the hero zone (510)
// Geometry, in design units (border-box). A tile has a 1 px border; its sides and top are 10 px and its bottom 12 px.
// Its name row is 22 px, its rows are 8 px apart, and a tag is 15 px of text and a 4 px margin. A vector has a 1 px
// border and a 3 px frame around its cells; a block of cells (a mat) has a 5 px frame.
const HEAD_COLS = 12, TAIL_COLS = 4;                       // long vectors: the first 12 and the last 4 columns
const TILE_IN = 22, TILE_V = 24;                           // a tile's content width is its width less 22, its height less 24
const NAME_ROW = 22, ROW_GAP = 8, TAG = 19, VEC_FRAME = 8;
const KV_PAD = 84, KV_STACK_PAD = 120;                     // a KV tile's height besides its blocks: K and V side by side, or stacked
const KV_ROOM = 238, STACK_ROWS = 12;                      // room for a KV tile above the caption; rows shown when K stacks above V
const XCELL = 18, XLABEL = 28, XGAP = 60;                  // step 0: the input vector's cells, its label, and the gap down to the heads

useStyle('gqa-scene', `
.gq-stage .gq-tile, .gq-stage .gq-kvtile { position: absolute; box-sizing: border-box; padding: 10px 10px 12px; border-radius: 14px;
  transition: opacity .5s cubic-bezier(.2,.7,.2,1), transform .6s cubic-bezier(.65,0,.35,1), height .6s cubic-bezier(.65,0,.35,1), visibility .5s, box-shadow .25s; }
.gq-stage .gq-kvtile { border-radius: 16px; }
.gq-stage .gq-tile.sel { box-shadow: 0 0 0 2px var(--accent), 0 16px 34px rgba(0, 0, 0, .16); }
.gq-stage .gq-tile.gq-tiny, .gq-stage .gq-kvtile.gq-tiny { overflow: hidden; }   /* beyond the settings panel's range: clip, never spill */
/* the zoom badge sits beside the name row, which keeps clear of it */
.gq-stage .gq-tile .k-zoom-badge { top: 8px; right: 8px; }
.gq-stage .gq-tile .k-zoom-badge svg { width: 14px; height: 14px; }
.gq-stage .gq-narrow .k-zoom-badge { top: 6px; right: 3px; }
.gq-stage .gq-narrow .k-zoom-badge svg { width: 12px; height: 12px; }
.gq-stage[data-step="0"] .gq-kvtile, .gq-stage[data-step="2"] .gq-kvtile { opacity: 0; visibility: hidden; transform: translateY(14px); }
.gq-stage[data-step="2"] .gq-tile { opacity: 0; visibility: hidden; transform: translateY(-14px); }
.gq-stage[data-step="0"] .gq-tile { transform: translateY(var(--gq-off0, 120px)); }
.gq-stage[data-step="0"] .gq-tile:hover { transform: translateY(calc(var(--gq-off0, 120px) - 4px)) scale(1.015); }
.gq-thead { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; height: 22px; padding-right: 16px; }
.gq-thead b { font: 600 15px/1 var(--font-mono); color: var(--kind); white-space: nowrap; }
.gq-stage .gq-compact .gq-thead { gap: 6px; }
.gq-stage .gq-narrow .gq-thead { padding-right: 0; }
.gq-stage .gq-narrow .gq-thead b { font-size: 14px; }
.gq-sub { font: 500 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; transition: opacity .5s; }
.gq-stage[data-step="0"] .gq-sub { opacity: 0; }
.gq-tbody { display: grid; gap: 8px; }
.gq-tag { font: 600 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; height: 15px; margin: 0 0 4px 2px; }
.gq-tag.q { color: var(--c-q); }
.gq-tag.k { color: var(--c-k); }
.gq-tag.v { color: var(--c-v); }
.gq-qrow { display: grid; justify-items: start; }
.gq-col { display: grid; justify-items: start; }
.gq-copies { display: grid; gap: 6px; justify-items: start; transition: opacity .5s cubic-bezier(.2,.7,.2,1), visibility .5s; }
.gq-copies.side { display: flex; align-items: flex-start; gap: 3px; }
.gq-copies .gq-tag { transition: opacity .5s cubic-bezier(.2,.7,.2,1); }
.gq-stage[data-step="0"] .gq-copies { opacity: 0; visibility: hidden; }
.gq-stage .gq-copy { opacity: 0; transition: opacity .35s cubic-bezier(.2,.7,.2,1); }
.gq-stage .gq-copy.landed { opacity: 1; }
.gq-stage .k-vec .k-cells { padding: 3px; }                  /* query strips, copies and the input vector */
.gq-stage .gq-kvtile .k-mat-body { padding: 5px; }
.gq-ell { width: 12px; text-align: center; font: 600 14px/1 var(--font-mono); color: var(--fg-3); }
.gq-block { display: flex; align-items: center; }
.gq-pair { display: flex; gap: 10px; align-items: flex-start; justify-content: center; }
.gq-pair.stack { flex-direction: column; align-items: center; gap: 6px; }
.gq-pair .gq-tag { transition: opacity .5s cubic-bezier(.2,.7,.2,1); }
.gq-xrow, .gq-xarrow { transition: opacity .5s cubic-bezier(.2,.7,.2,1), visibility .5s; }
.gq-stage:not([data-step="0"]) .gq-xrow, .gq-stage:not([data-step="0"]) .gq-xarrow { opacity: 0; visibility: hidden; }
.gq-karrow { transition: opacity .5s cubic-bezier(.2,.7,.2,1), visibility .5s; }
.gq-stage:not([data-step="1"]) .gq-karrow { opacity: 0; visibility: hidden; }
.gq-side { position: absolute; width: 340px; display: grid; gap: 14px; }
.gq-eq { font-size: 24px; color: var(--fg); }
.gq-live { font-size: 19px; line-height: 1.4; padding: 10px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.gq-note { font-size: 15px; line-height: 1.5; color: var(--fg-3); }
.gq-heads { flex-wrap: wrap; width: 340px; row-gap: 6px; }
.gq-heads .k-picker-label { display: none; }
.gq-heads .k-pick-head { width: 36px; }
.gq-stage .k-picker-wrap { transition: opacity .4s, visibility .4s; }
.gq-stage[data-step="2"] .k-picker-wrap { opacity: 0; visibility: hidden; }
.gq-mem { transition: opacity .5s cubic-bezier(.2,.7,.2,1), visibility .5s; }
.gq-stage:not([data-step="2"]) .gq-mem { opacity: 0; visibility: hidden; }
.gq-legend { display: flex; gap: 16px; align-items: center; font: 500 15px/1.2 var(--font-mono); color: var(--fg-3); white-space: nowrap; transition: opacity .5s, visibility .5s; }
.gq-stage:not([data-step="2"]) .gq-legend { opacity: 0; visibility: hidden; }
.gq-sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.gq-sw.k { background: var(--c-k); }
.gq-sw.v { background: var(--c-v); }
.gq-gname { display: grid; justify-items: center; gap: 4px; text-align: center; white-space: nowrap; }
.gq-gname b { font: 650 20px/1.1 var(--font-display); color: var(--fg); }
.gq-gname span { font: 500 15px/1.2 var(--font-mono); color: var(--fg-3); }
.gq-bar { position: absolute; width: 76px; display: flex; flex-direction: column; overflow: hidden; border-radius: 7px 7px 3px 3px;
  box-shadow: 0 10px 22px rgba(0, 0, 0, .16); transition: top .6s cubic-bezier(.65,0,.35,1), height .6s cubic-bezier(.65,0,.35,1), opacity .5s, visibility .5s; }
.gq-bar > i { display: block; flex: 1 1 0; }
.gq-bar > .gq-bv { background: linear-gradient(180deg, color-mix(in srgb, var(--c-v) 88%, white), var(--c-v)); }
.gq-bar > .gq-bk { background: linear-gradient(180deg, color-mix(in srgb, var(--c-k) 88%, white), var(--c-k)); }
.gq-val { position: absolute; font: 600 14px/1 var(--font-mono); color: var(--fg); white-space: nowrap;
  transition: top .6s cubic-bezier(.65,0,.35,1), opacity .5s, visibility .5s; }
.gq-tick { display: grid; justify-items: center; gap: 4px; white-space: nowrap; transition: opacity .5s, visibility .5s; }
.gq-tick b { font: 600 15px/1 var(--font-mono); color: var(--fg-2); }
.gq-tick span { font: 500 15px/1 var(--font-mono); color: var(--fg-3); }
.gq-ratio { font: 600 14px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; transition: opacity .5s, visibility .5s; }
.gq-base { width: 1030px; height: 1px; background: var(--line-strong); transition: opacity .5s, visibility .5s; }
/* The layer paints the atlas backdrop (as .atlas in styles/atlas.css) so that it hides the arcs of the parent layer,
   which are hidden but keep their own visibility. */
.atlas-layer.gq-backdrop {
  background:
    radial-gradient(ellipse at 50% 40%, color-mix(in srgb, var(--accent) 7%, transparent), transparent 60%),
    radial-gradient(circle, color-mix(in srgb, var(--fg) 7%, transparent) 1px, transparent 1.4px) 0 0 / 28px 28px,
    var(--bg);
}
@media (prefers-reduced-motion: reduce) { .gq-stage * { transition: none !important; } }
`);

const bytesPerToken = (L, Hkv, dh) => 2 * L * Hkv * dh * 2;      // K and V (2) x layers x KV heads x d_h x fp16 (2 bytes)
const sig = (v) => { const s = v >= 10 ? v.toFixed(0) : v.toFixed(2); return s.includes('.') ? s.replace(/\.?0+$/, '') : s; };
function bytesLabel(n) {
  const K = 1024;
  if (n >= K ** 3) return `${sig(n / K ** 3)} GiB`;
  if (n >= K ** 2) return `${sig(n / K ** 2)} MiB`;
  if (n >= K) return `${sig(n / K)} KiB`;
  return `${n} B`;
}
// Largest cell size in (min, max] for which ok(c) holds; min when none does.
function cellFor(max, min, ok) { for (let c = max; c > min; c--) if (ok(c)) return c; return min; }
// Apply a style change without the transition (instant updates, e.g. after a new forward pass).
function snap(root, sel, fn) {
  const els = root.querySelectorAll(sel);
  els.forEach((e) => { e.style.transition = 'none'; });
  fn();
  void root.offsetHeight;
  els.forEach((e) => { e.style.transition = ''; });
}
// First view: the query tiles fade in where they stand. Their step offset is a CSS transform, and an
// animation of transform would hand it back with a jump when it ends, so only opacity is animated.
function fadeIn(els) {
  if (reducedMotion()) return;
  els.forEach((e, i) => e.animate([{ opacity: 0 }, { opacity: 1 }], { duration: MOTION.appear, delay: i * MOTION.stagger, easing: MOTION.easeOut, fill: 'backwards' }));
}

// ------------------------------------------------------------ block layout (pure)
const span = (lo, hi) => Array.from({ length: hi - lo }, (_, i) => lo + i);
const longVec = (dh) => dh > HEAD_COLS + TAIL_COLS + 1;          // vec() elides a vector of more than 17 cells
// Columns drawn for one d_h-wide block: all of them, or the first 12 and the last 4 around a ⋮.
const colGroups = (dh, elide) => (elide ? [span(0, HEAD_COLS), span(dh - TAIL_COLS, dh)] : [span(0, dh)]);
// Outer width of one block at cell size c: its cells (1 px apart) in each group, a 12 px ⋮ between groups, and the frame.
const blockW = (groups, c) => groups.reduce((a, g) => a + g.length * (c + 1) - 1, 0) + (groups.length - 1) * 12 + 12;
// How the K block and the V block of one KV tile sit in `inner` units (10 apart, side by side). Side by side: every
// column at 4 px or more if that fits, else (long vectors) the elided columns, else every column at 3 px. With
// `stackRows`, K above V instead, at the largest cell that keeps the tile inside KV_ROOM. Null when nothing fits.
function planBlocks(dh, inner, cmax, { stackRows = 0 } = {}) {
  const full = colGroups(dh, false), cut = colGroups(dh, true), elide = longVec(dh);
  const side = (groups, c) => 2 * blockW(groups, c) + 10 <= inner;
  for (let c = cmax; c >= 4; c--) if (side(full, c)) return { c, groups: full, stack: false };
  if (elide) for (let c = cmax; c >= 3; c--) if (side(cut, c)) return { c, groups: cut, stack: false };
  for (let c = cmax; c >= 3; c--) if (side(full, c)) return { c, groups: full, stack: false };
  if (stackRows) {
    const cStack = Math.min(cmax, Math.floor((KV_ROOM - KV_STACK_PAD) / (2 * stackRows)) - 1);
    for (let c = cStack; c >= 3; c--) {
      if (blockW(full, c) <= inner) return { c, groups: full, stack: true };
      if (elide && blockW(cut, c) <= inner) return { c, groups: cut, stack: true };
    }
  }
  return null;
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    st.root.classList.add('gq-stage');
    el.classList.add('gq-backdrop');
    // The first step must be in place before anything is laid out: otherwise the first style pass sees the
    // tiles without their step offset, and the offset would then transition in on mount (the tiles sinking).
    st.root.dataset.step = '0';
    const L = ctx.params.l;
    let step = 0, paused = false, dirty = false, arrowsDrawn = false, copiesDone = false, inAuto = false, autoTimer = null, flightGen = 0;
    let made = [], comps = [], copies = [], kvK = [], kvV = [], mem = [], sides = [], texts = [];
    let queryTiles = [], heights = [0, 0];                 // query tiles of this build; their height in step 0 and in steps 1 and 2
    const kvArrows = [], xArrows = [];                     // one arrow per head, reused by every build (see arrowSlot)
    const timers = [];
    const stopAuto = () => clearTimeout(autoTimer);
    // A flight belongs to the copy it is aimed at: when that copy goes, so does the flight (its ghost clone too).
    const clearGhosts = () => st.layers.front.querySelectorAll('.k-ghost').forEach((g) => g.remove());
    const cancelFlights = () => { flightGen++; timers.splice(0).forEach(clearTimeout); clearGhosts(); };

    // ------------------------------------------------------------ fixed parts (shared kit)
    const header = sceneHeader(st, { chips: [], title: ctx.node.title });
    const picker = tokenPicker(st, ctx, { label: 'token' });
    const heads = headPicker(st, ctx, { x: SIDE.x, y: SIDE.y });
    heads.el.classList.add('gq-heads');
    const cap = sceneCaption(st);
    // The kit's caption fades can write an older text after a newer one. Here a newer text cancels a fade still
    // running, so only the latest text is ever written (the same guard as lm-head.js).
    let capText = null;
    const capSet = (html) => {
      if (html === capText) return;
      capText = html;
      if (reducedMotion()) { cap.el.innerHTML = html; return; }
      cap.el.getAnimations().forEach((a) => a.cancel());
      cap.el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 140 }).finished.then(() => {
        if (html !== capText) return;
        cap.el.innerHTML = html;
        cap.el.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 260, easing: MOTION.easeOut });
      }).catch(() => {});
    };
    const legend = h('div', { class: 'gq-legend' },
      h('span', null, h('i', { class: 'gq-sw k' }), 'K'),
      h('span', null, h('i', { class: 'gq-sw v' }), 'V'),
      h('span', null, 'bytes per token, fp16 · bars: share of each model’s MHA cache'));
    st.add(legend, { x: ZONES.controls.x, y: ZONES.controls.y });
    const sideEl = h('div', { class: 'gq-side' });
    st.add(sideEl, { x: SIDE.x, y: 344 });
    ctx.setSteps(STEPS, setStep);

    // Arrows come from a pool: the kit keeps a cleanup entry for every arrow it creates, so a rebuild moves the
    // paths of the arrows it already has (one per head at most) instead of making new ones.
    function arrowSlot(pool, i, kind) {
      if (!pool[i]) { pool[i] = st.arrow({ x: 0, y: 0 }, { x: 1, y: 1 }, { kind, width: 2 }); pool[i].el.classList.add('gq-arrow'); }
      return pool[i];
    }
    function moveArrow(a, from, to, { kind, width }) {
      a.path.setAttribute('d', `M${from.x},${from.y} L${to.x},${to.y}`);
      a.path.setAttribute('stroke-width', String(width));
      a.el.style.setProperty('--kind', `var(--c-${kind})`);
      a.el.style.display = '';
    }
    function hideArrows(pool, from = 0) { for (let i = from; i < pool.length; i++) pool[i].el.style.display = 'none'; }

    function setStep(i) {
      if (!inAuto) stopAuto();                 // a reader's choice ends the auto-walk
      step = clamp(i, 0, STEPS.length - 1);
      paint();
    }

    // One K or V block of a KV tile: its column groups (every column, or the first 12 and last 4 around a ⋮) as mats.
    // `band` is the KV head that owns the block.
    function block(src, band, { s0, R, T, dh, plan, kind, range, trace }) {
      const ms = [];
      const wrap = h('div', { class: ['gq-block', plan.groups.length > 1 && 'cut'] });
      plan.groups.forEach((cols, gi) => {
        if (gi) wrap.append(h('div', { class: 'gq-ell', 'aria-hidden': 'true' }, '⋮'));
        const vals = new Float64Array(R * cols.length);
        for (let r = 0; r < R; r++) for (let j = 0; j < cols.length; j++) vals[r * cols.length + j] = src[(band * T + s0 + r) * dh + cols[j]];
        const m = mat(vals, R, cols.length, {
          cell: plan.c, gap: plan.c >= 3 ? 1 : 0, kind, mode: 'diverging', range, trace, cellOf: (r, j) => `${s0 + r},${cols[j]},${band}`,
        });
        wrap.append(m);
        comps.push(m);
        ms.push(m);
      });
      wrap.highlightRow = (r) => ms.forEach((m) => m.highlightRow(r));
      return wrap;
    }

    // ------------------------------------------------------------ the diagram for the current forward pass
    function build() {
      made.forEach((n) => n.remove());
      comps.forEach((c) => c.destroy?.());     // every mat and strip registers a theme listener: release the old ones
      comps = [];
      cancelFlights();                         // flights aimed at the old copies must not fire or stay on screen
      made = []; copies = []; kvK = []; kvV = []; mem = []; queryTiles = [];
      arrowsDrawn = false; copiesDone = false;
      const place = (node, opts) => { made.push(st.add(node, opts)); return node; };
      const later = [];                        // highlights need the laid-out DOM

      const s = ctx.state, cfg = s.config;
      const H = cfg.n_head, Hkv = cfg.n_kv_head, dh = ctx.engine.dHead(cfg), T = s.ids.length;
      const t = clamp(s.sel.token ?? T - 1, 0, T - 1), hs = clamp(s.sel.head ?? 0, 0, H - 1);
      const tr = s.trace, p = `L${L}.attn.`;
      const preLN = cfg.norm_placement === 'pre', rms = cfg.norm === 'rmsnorm';
      // with RoPE the scores use the rotated q and k, and the reference code repeats k_rot (not k_heads) into k_rep
      const rope = cfg.pos_encoding === 'rope', qKey = rope ? 'q_rot' : 'q_heads', kKey = rope ? 'k_rot' : 'k_heads';
      const qh = tr[`${p}${qKey}`], kh = tr[`${p}${kKey}`], vh = tr[`${p}v_heads`], kr = tr[`${p}k_rep`], vr = tr[`${p}v_rep`];
      const inName = preLN ? (rms ? 'RMSNorm(x)' : 'LN(x)') : 'x';   // the attention input, named for the reader
      const grp = (x) => Math.floor((x * Hkv) / H);
      const readers = (g) => Array.from({ length: H }, (_, x) => x).filter((x) => grp(x) === g);
      header.el.querySelector('.k-eyebrow').replaceChildren(chip(`Layer ${L}`, 'resid'), chip(`${H} query · ${Hkv} KV`, 'k'));
      heads.render();
      // the head chips wrap eight to a row (32 units high, 6 apart): the side column starts below them
      const rows = Math.ceil(H / 8);
      st.move(sideEl, SIDE.x, Math.max(344, SIDE.y + rows * 32 + (rows - 1) * 6 + 14));

      if (!kr || Hkv >= H) {                   // not a grouped config: the atlas hides this node, but never crash
        header.setSub('This configuration has one KV head per query head, so no heads share keys and values.');
        place(h('div', { class: 'gq-note' }, 'Choose a model with n_kv_head < n_head. The llama preset has 4 query heads and 2 KV heads.'), { x: HERO.x, y: HERO.qTop });
        hideArrows(kvArrows); hideArrows(xArrows);
        sides = [[], [], []]; texts = ['', '', ''];
        return;
      }

      const peak = (a) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m || 1; };
      const rQ = peak(qh.data), rK = peak(kh.data), rV = peak(vh.data);
      const wq = Math.min(260, Math.floor((HERO.w - (H - 1) * QGAP) / H));
      const wk = Math.min(Hkv === 1 ? 640 : 520, Math.floor((HERO.w - (Hkv - 1) * KVGAP) / Hkv));
      // tiles share one pitch and sit centred under the hero: neighbours are exactly QGAP (KVGAP) apart
      const qX = (x) => HERO.x + HERO.w / 2 + (x - (H - 1) / 2) * (wq + QGAP);
      const kX = (g) => HERO.x + HERO.w / 2 + (g - (Hkv - 1) / 2) * (wk + KVGAP);
      const narrowQ = wq < 90, compactQ = wq < 150, narrowKV = wk < 160;
      const inQ = wq - TILE_IN, inKV = wk - TILE_IN;     // content widths of a query tile and a KV tile

      // The KV blocks decide the rows shown: 16 tokens around the selected one (12 when K stacks above V).
      let R = Math.min(T, 16), kvPlan = planBlocks(dh, inKV, clamp(Math.floor(112 / R), 4, 9));
      if (!kvPlan) { R = Math.min(T, STACK_ROWS); kvPlan = planBlocks(dh, inKV, clamp(Math.floor(112 / R), 4, 9), { stackRows: R }); }
      const s0 = clamp(t - Math.floor(R / 2), 0, T - R), row = t - s0;

      // Query strips: the selected token's vector of each head, and the K and V copy of that vector that each head reads.
      // A strip that cannot run across its tile runs down it, and the copies run the same way.
      const stripCells = longVec(dh) ? HEAD_COLS + TAIL_COLS + 1 : dh;
      const rowW = (c) => stripCells * (c + 1) + VEC_FRAME - 1;         // cells one px apart, in their frame
      const qVert = rowW(3) > inQ;
      const cq = qVert ? cellFor(6, 3, (c) => c + VEC_FRAME <= inQ) : cellFor(10, 3, (c) => rowW(c) <= inQ);
      const showStrip = inQ >= 15;             // below one cell and its frame a tile holds no strip (heads above 16)
      const stripBox = !showStrip ? 0 : qVert ? stripCells * (cq + 1) - 1 + VEC_FRAME : cq + VEC_FRAME;
      // tile heights: step 0 shows the query tiles without the copies (the rise to step 1 is the offset)
      const qH0 = TILE_V + NAME_ROW + ROW_GAP + TAG + stripBox;
      const copyBox = qVert ? TAG + stripBox : 2 * (TAG + stripBox) + 6;   // K and V copies: side by side, or one above the other
      const qH1 = qH0 + ROW_GAP + copyBox;
      heights = [qH0, qH1];
      // step 0: the input vector of this token, above the query tiles that read it
      const compH = XLABEL + XCELL + VEC_FRAME + XGAP + qH0;
      const compTop = HERO_MID - compH / 2;
      const tileTop0 = compTop + XLABEL + XCELL + VEC_FRAME + XGAP;
      st.root.style.setProperty('--gq-off0', `${Math.round(tileTop0 - HERO.qTop)}px`);
      const kvH = (k) => (!kvPlan ? KV_PAD : k ? KV_STACK_PAD + 2 * R * (kvPlan.c + 1) : KV_PAD + R * (kvPlan.c + 1));
      const col = (kind, tag, blk) => h('div', { class: 'gq-col' }, h('div', { class: `gq-tag ${kind}` }, tag), blk);
      const wideQ = !compactQ;
      const kTag = wideQ ? 'k_rep' : 'k', vTag = wideQ ? 'v_rep' : 'v';

      // query heads: the selected token's q, and the K and V copies that head reads
      for (let x = 0; x < H; x++) {
        const g = grp(x), off = (x * T + t) * dh;
        const opts = (kind, range, trace) => ({
          orientation: qVert ? 'v' : 'h', cell: cq, gap: 1, kind, mode: 'diverging', range, trace,
          cellOf: (c) => `${t},${c},${x}`, head: longVec(dh) ? HEAD_COLS : 1000, tail: longVec(dh) ? TAIL_COLS : 0,
        });
        const strip = showStrip ? vec(qh.data.subarray(off, off + dh), opts('q', rQ, `${p}${qKey}`)) : null;
        const cK = vec(kr.data.subarray(off, off + dh), opts('k', rK, `${p}k_rep`));
        const cV = vec(vr.data.subarray(off, off + dh), opts('v', rV, `${p}v_rep`));
        cK.classList.add('gq-copy'); cV.classList.add('gq-copy');
        if (strip) comps.push(strip);
        comps.push(cK, cV);
        const body = h('div', { class: 'gq-tbody' },
          h('div', { class: 'gq-thead' }, h('b', null, `h${x}`), narrowQ ? null : h('span', { class: 'gq-sub' }, compactQ ? `→ KV ${g}` : `reads KV ${g}`)),
          h('div', { class: 'gq-qrow' }, h('div', { class: 'gq-tag q' }, wideQ ? `${qKey}[${x}, ${t}]` : 'q'), strip),
          h('div', { class: qVert ? 'gq-copies side' : 'gq-copies stack' }, col('k', kTag, cK), col('v', vTag, cV)));
        copies.push({ g, x, cK, cV });
        const tile = card({ kind: 'q', node: `head.${L}.${x}`, body });   // the head itself is the zoom target
        tile.classList.add('gq-tile');
        if (narrowQ) tile.classList.add('gq-narrow');
        if (compactQ) tile.classList.add('gq-compact');
        if (x === hs) tile.classList.add('sel');
        tile.style.width = `${wq}px`; tile.style.height = `${step === 0 ? qH0 : qH1}px`;
        queryTiles.push(tile);
        place(tile, { x: qX(x), y: HERO.qTop, anchor: 'tc' });
      }

      // KV heads: the cache. The window of rows around the selected token, its selected row highlighted.
      for (let g = 0; g < Hkv; g++) {
        const rd = readers(g);
        const body = h('div', { class: 'gq-tbody' },
          h('div', { class: 'gq-thead' }, h('b', null, `KV ${g}`), h('span', { class: 'gq-sub' }, narrowKV ? `${rd.length}×` : `${rd.length} reader${rd.length > 1 ? 's' : ''}`)));
        if (kvPlan) {
          const bK = block(kh.data, g, { s0, R, T, dh, plan: kvPlan, kind: 'k', range: rK, trace: `${p}${kKey}` });
          const bV = block(vh.data, g, { s0, R, T, dh, plan: kvPlan, kind: 'v', range: rV, trace: `${p}v_heads` });
          body.append(h('div', { class: ['gq-pair', kvPlan.stack && 'stack'] },
            col('k', narrowKV ? 'K' : kKey, bK), col('v', narrowKV ? 'V' : 'v_heads', bV)));
          kvK[g] = bK; kvV[g] = bV;
          later.push(() => { bK.highlightRow(row); bV.highlightRow(row); });
        } else body.append(h('div', { class: 'gq-note' }, 'Too narrow to draw the cache at this width.'));
        const tile = card({ kind: 'k', body });
        tile.classList.add('gq-kvtile');
        if (wk < 60) tile.classList.add('gq-tiny');
        tile.style.width = `${wk}px`; tile.style.height = `${kvH(kvPlan?.stack)}px`;
        place(tile, { x: kX(g), y: HERO.kvTop, anchor: 'tc' });
      }

      // head x reads KV head floor(x * Hkv / H): one arrow per query head, from its KV tile up to the head
      for (let x = 0; x < H; x++) {
        const a = arrowSlot(kvArrows, x, 'k');
        moveArrow(a, { x: kX(grp(x)), y: HERO.kvTop - 4 }, { x: qX(x), y: HERO.qTop + qH1 + 4 }, { kind: 'k', width: x === hs ? 3.5 : 2 });
        a.el.classList.add('gq-karrow');
      }
      hideArrows(kvArrows, H);

      // step 0: the attention input of this token, read by every query head (each arrow is one head's W_Q)
      const xKey = preLN ? `L${L}.ln1` : (L === 0 ? 'emb.out' : `L${L - 1}.resid_out`);
      const xT = tr[xKey];
      const xKind = preLN ? 'norm' : 'resid';
      if (xT) {
        const d = xT.shape[1];
        const xVec = vec(Array.from(xT.data.subarray(t * d, (t + 1) * d)), {
          orientation: 'h', cell: XCELL, gap: 2, kind: xKind, mode: 'diverging', range: peak(xT.data),
          label: `${inName} at position ${t}`, trace: xKey, cellOf: (i) => `${t},${i},0`,
        });
        xVec.classList.add('gq-xrow');
        comps.push(xVec);
        const xCx = HERO.x + HERO.w / 2;
        place(xVec, { x: xCx, y: compTop, anchor: 'tc' });
        const from = { x: xCx, y: compTop + XLABEL + XCELL + VEC_FRAME + 4 };
        for (let x = 0; x < H; x++) {
          const a = arrowSlot(xArrows, x, xKind);
          moveArrow(a, from, { x: qX(x), y: tileTop0 - 4 }, { kind: xKind, width: 2 });
          a.el.classList.add('gq-xarrow');
        }
        hideArrows(xArrows, H);
      } else hideArrows(xArrows);

      // memory: bytes per token for MHA, GQA and MQA. Each model's bars are shares of its own MHA cache.
      const groups = [
        { name: 'This model', L: cfg.n_layer, H, Hkv, dh },
        ...PUBLISHED,
      ];
      // the engine's own cacheBytes for each configuration (published ones are only described by layers, heads, d_h)
      const cacheOf = (gp, hk, len = 1) => ctx.engine.cacheBytes({ n_layer: gp.L, n_head: gp.H, n_kv_head: hk, d_model: gp.dh * gp.H }, len, 2);
      const GW = HERO.w / groups.length;
      groups.forEach((gp, i) => {
        const gx = HERO.x + (i + 0.5) * GW;
        const full = cacheOf(gp, gp.H);
        place(h('div', { class: 'gq-gname gq-mem' },
          h('b', null, gp.name),
          h('span', null, `${gp.L} layers · ${gp.H} / ${gp.Hkv} heads · d_h ${gp.dh}`)), { x: gx, y: 268, anchor: 'tc' });
        [['MHA', gp.H], ['GQA', gp.Hkv], ['MQA', 1]].forEach(([name, hk], j) => {
          const bytes = cacheOf(gp, hk);
          const hT = (bytes / full) * BAR_MAX, bx = gx + (j - 1) * 104;
          const bar = place(h('div', { class: 'gq-bar gq-mem' }, h('i', { class: 'gq-bv' }), h('i', { class: 'gq-bk' })), { x: bx - 38, y: BASE_Y });
          const val = place(h('div', { class: 'gq-val gq-mem' }, bytesLabel(bytes)), { x: bx, y: BASE_Y - hT - 26, anchor: 'tc' });
          const sameAsMQA = name === 'GQA' && gp.Hkv === 1;
          place(h('div', { class: 'gq-tick gq-mem' }, h('b', null, name), h('span', null, sameAsMQA ? '= MQA' : `${hk} KV`)), { x: bx, y: BASE_Y + 12, anchor: 'tc' });
          mem.push({ bar, val, hT });
        });
        const ratio = gp.Hkv === 1 ? `GQA = MQA ÷${gp.H}` : `GQA ÷${gp.H / gp.Hkv} · MQA ÷${gp.H}`;
        place(h('div', { class: 'gq-ratio gq-mem' }, ratio), { x: gx, y: BASE_Y + 60, anchor: 'tc' });
      });
      place(h('div', { class: 'gq-base gq-mem' }), { x: HERO.x, y: BASE_Y });

      // side column and captions, one set per step
      const gh = grp(hs), rdh = readers(gh);
      let md = 0;                              // largest |difference| between the copies and the KV heads they came from
      for (let tt = 0; tt < T; tt++) for (let c = 0; c < dh; c++) {
        const a = (hs * T + tt) * dh + c, b = (gh * T + tt) * dh + c;
        md = Math.max(md, Math.abs(kr.data[a] - kh.data[b]), Math.abs(vr.data[a] - vh.data[b]));
      }
      const label = (text, kind) => h('div', { class: 'k-label', style: { '--kind': `var(--c-${kind})`, textAlign: 'left' } }, text);
      const eq = (tex) => h('div', { class: 'gq-eq', html: texHTML(tex, { display: true }) });
      const live = (tex) => h('div', { class: 'gq-live', html: texHTML(tex) });
      const note = (text) => h('div', { class: 'gq-note' }, text);
      const bytesHere = bytesPerToken(cfg.n_layer, Hkv, dh);
      const xTex = preLN ? (rms ? String.raw`\RMSNorm(x)` : String.raw`\LN(x)`) : 'x';
      // what the layout did at this width, said in the note of the step that shows it
      const layoutNote = [
        longVec(dh) ? `Columns ${HEAD_COLS} to ${dh - TAIL_COLS - 1} (counted from 0) of ${dh} are elided (⋮).` : '',
        kvPlan?.stack ? 'K and V are stacked: the two blocks do not fit side by side at this width.' : '',
      ].filter(Boolean).join(' ');
      sides = [
        [label(STEPS[0], 'q'),
          eq(rope ? String.raw`q_{\mathrm{rot},h} = \mathrm{RoPE}\big(${xTex}\,W_Q^{(h)}\big)` : String.raw`q_h = ${xTex}\,W_Q^{(h)}`),
          live(String.raw`H\,d_h = ${H}\cdot ${dh} = ${H * dh} = d`),
          note(`Token “${ctx.label(t)}” at position ${t}. ${inName} is its attention input.${rope ? ' RoPE then rotates each pair of dimensions by its position.' : ''}`),
          checkBadge(qh.shape[0] * qh.shape[2], cfg.d_model, { label: 'traced H · d_h = d_model' })],
        [label(STEPS[1], 'k'),
          eq(String.raw`g(h) = \left\lfloor \tfrac{h\,H_{kv}}{H} \right\rfloor`),
          live(String.raw`g(${hs}) = \left\lfloor \tfrac{${hs}\cdot ${Hkv}}{${H}} \right\rfloor = ${gh}`),
          note(`Query “${ctx.label(t)}” at position ${t}. Each head gets the K and V of its KV head at this position. Rows ${s0} to ${s0 + R - 1} of ${T} are shown in the cache. ${layoutNote}`.trim()),
          checkBadge(md, 0, { label: `k_rep[${hs}], v_rep[${hs}] = KV ${gh}`, tol: 1e-9 })],
        [label(STEPS[2], 'k'),
          eq(String.raw`B = 2 \cdot n_{\text{layer}} \cdot H_{kv} \cdot d_h \cdot 2`),
          live(String.raw`2 \cdot ${cfg.n_layer} \cdot ${Hkv} \cdot ${dh} \cdot 2 = ${bytesHere}\ \text{B}`),
          checkBadge(bytesHere, ctx.engine.cacheBytes(cfg, 1, 2), { label: 'equals cacheBytes(cfg, 1)' })],
      ];
      const L70 = PUBLISHED[1];
      texts = [
        `The same ${inName} of “<b>${ctx.label(t)}</b>” enters every query head. Each head multiplies it by its own slice of W_Q, so each one asks its own question; the ${H} slices together span d = <b>${cfg.d_model}</b>.`,
        Hkv === 1
          ? `All ${H} query heads read the same key and value head. The reference code copies it into every head (k_rep, v_rep), but the cache keeps one K and one V per layer and token.`
          : `Head <b>${hs}</b> reads KV head <b>${gh}</b>, shared by <b>${rdh.length}</b> heads. The reference code copies that K and V into each reader (k_rep, v_rep), but the cache keeps only the ${Hkv} K and V heads.`,
        `Keeping K and V for ${Hkv} KV head${Hkv > 1 ? 's' : ''} instead of ${H} divides the cache by <b>${H / Hkv}</b>. LLaMA-2-70B at ${CONTEXT.toLocaleString('en-US')} tokens: ${bytesLabel(cacheOf(L70, L70.H, CONTEXT))} with MHA, ${bytesLabel(cacheOf(L70, L70.Hkv, CONTEXT))} with GQA; the pickers are hidden because the cache depends on neither the token nor the head.`,
      ];
      header.setSub(Hkv === 1
        ? `All ${H} query heads read one key and value head, so the cache holds one K and one V per layer.`
        : `${H} query heads share ${Hkv} key and value heads: each K and V head serves ${H / Hkv} queries.`);
      later.forEach((f) => f());
    }

    // ------------------------------------------------------------ one step at a time
    function paint({ instant = false } = {}) {
      st.root.dataset.step = String(step);
      const setHeights = () => queryTiles.forEach((tile) => { tile.style.height = `${step === 0 ? heights[0] : heights[1]}px`; });
      if (instant) snap(st.root, '.gq-tile', setHeights); else setHeights();
      if (step === 1) {
        if (!arrowsDrawn) {
          arrowsDrawn = true;
          if (!instant && !reducedMotion()) kvArrows.forEach((a, i) => { if (a.el.style.display !== 'none') a.draw(560, 180 + i * 50); });
        }
        if (!copiesDone) {
          copiesDone = true;
          const gen = ++flightGen;               // a flight that lands after the reader left step 1 does nothing
          copies.forEach((c) => {
            const land = () => { if (gen === flightGen) { c.cK.classList.add('landed'); c.cV.classList.add('landed'); } };
            if (instant || reducedMotion() || !kvK[c.g]) { land(); return; }
            // the query tiles are still rising for RISE ms, so the copy slots are measured only once they have settled;
            // the cached K and V of each group then fly into each head that reads them
            timers.push(setTimeout(() => {
              Promise.all([fly(st, kvK[c.g], c.cK, { duration: FLY, arc: 40 }), fly(st, kvV[c.g], c.cV, { duration: FLY, arc: 40 })]).then(land);
            }, RISE + FLY_GAP * c.x));
          });
        }
      } else {
        copiesDone = false;
        cancelFlights();
        copies.forEach((c) => { c.cK.classList.remove('landed'); c.cV.classList.remove('landed'); });
      }
      const on = step === 2;
      const setBars = () => mem.forEach(({ bar, val, hT }, k) => {
        const hNow = on ? hT : 0;
        bar.style.top = `${BASE_Y - hNow}px`;
        bar.style.height = `${hNow}px`;
        val.style.top = `${BASE_Y - hNow - 26}px`;
        const delay = on && !instant ? `${k * 70}ms` : '0ms';
        bar.style.transitionDelay = delay;
        val.style.transitionDelay = delay;
      });
      if (instant) snap(st.root, '.gq-bar, .gq-val', setBars); else setBars();
      sideEl.replaceChildren(...(sides[step] || []));
      if (!instant && !reducedMotion()) sideEl.animate([{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'none' }], { duration: 380, easing: 'cubic-bezier(.2,.7,.2,1)' });
      capSet(texts[step] || '');
    }

    function rebuild() {
      if (paused) { dirty = true; return; }
      build();
      paint({ instant: true });
      dirty = false;
    }

    const offs = [ctx.on('trace', rebuild), ctx.on('select', rebuild)];
    rebuild();
    fadeIn(made.filter((n) => n.classList?.contains('gq-tile')));
    appear([header.el, picker.el, heads.el, sideEl], { stagger: MOTION.stagger });

    // walk the steps once on first view; a pointer press stops it
    el.addEventListener('pointerdown', stopAuto, { once: true });
    autoTimer = setTimeout(function next() {
      if (step < STEPS.length - 1) {
        inAuto = true; ctx.setStep(step + 1); inAuto = false;
        autoTimer = setTimeout(next, MOTION.step);
      }
    }, 900);

    return {
      pause() { paused = true; stopAuto(); cancelFlights(); copiesDone = false; },
      resume() {
        paused = false;
        if (dirty) { rebuild(); return; }
        if (step === 1 && !copiesDone) paint({ instant: true });   // back from a zoom: the copies are simply in place
      },
      destroy() { stopAuto(); timers.forEach(clearTimeout); offs.forEach((f) => f()); comps.forEach((c) => c.destroy?.()); st.destroy(); },
    };
  },
};

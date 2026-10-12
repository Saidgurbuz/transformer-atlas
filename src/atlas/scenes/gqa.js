// Grouped-query attention: several query heads read the same key and value head.
// Step 1 wires each query head to KV head floor(h * n_kv_head / n_head). The cached K and V of each
// KV head (k_rot under RoPE, else k_heads; and v_heads) sit next to the k_rep and v_rep copies that
// each head reads (the reference code builds them with repeat_interleave). Step 2 compares KV-cache
// bytes per token for MHA, GQA and MQA on this model and on two published configurations. Every
// number is traced, read from the config, or computed by the engine; the published configurations
// are the only constants.
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
// Block layout. A long vector (more than 17 columns, as in the vec kit) is drawn as its first 12 and last 4
// columns around a ⋮, as the style guide asks. The KV row has KV_ROOM units above the caption zone.
const HEAD_COLS = 12, TAIL_COLS = 4;
const KV_ROOM = 238, KV_PAD = 80, KV_STACK_PAD = 117;      // room; tile overhead side by side; and when K stacks above V
const STACK_ROWS = 12;                                     // rows shown when K and V stack (so the tile fits the room)

useStyle('gqa-scene', `
.gq-stage .gq-tile, .gq-stage .gq-kvtile { position: absolute; box-sizing: border-box; padding: 10px 12px 12px; border-radius: 14px;
  transition: opacity .5s cubic-bezier(.2,.7,.2,1), transform .6s cubic-bezier(.65,0,.35,1), height .6s cubic-bezier(.65,0,.35,1), visibility .5s, box-shadow .25s; }
.gq-stage .gq-kvtile { border-radius: 16px; }
.gq-stage .gq-tile.sel { box-shadow: 0 0 0 2px var(--accent), 0 16px 34px rgba(0, 0, 0, .16); }
.gq-stage .gq-tile.gq-tiny, .gq-stage .gq-kvtile.gq-tiny { overflow: hidden; }   /* beyond the settings panel's range: clip, never spill */
.gq-stage .gq-tile .k-zoom-badge { top: 9px; right: 10px; }
.gq-stage[data-step="0"] .gq-kvtile, .gq-stage[data-step="2"] .gq-kvtile { opacity: 0; visibility: hidden; transform: translateY(14px); }
.gq-stage[data-step="2"] .gq-tile { opacity: 0; visibility: hidden; transform: translateY(-14px); }
.gq-stage[data-step="0"] .gq-tile { transform: translateY(var(--gq-off0, 120px)); }
.gq-stage[data-step="0"] .gq-tile:hover { transform: translateY(calc(var(--gq-off0, 120px) - 4px)) scale(1.015); }
.gq-thead { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; height: 22px; padding-right: 16px; }
.gq-thead b { font: 650 19px/1 var(--font-display); color: var(--kind); white-space: nowrap; }
.gq-stage .gq-narrow .gq-thead b { font-size: 14px; }
.gq-stage .gq-compact .gq-thead { padding-right: 0; gap: 6px; }
.gq-sub { font: 500 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; transition: opacity .5s; }
.gq-stage[data-step="0"] .gq-sub { opacity: 0; }
.gq-tbody { display: grid; gap: 8px; }
.gq-tag { font: 600 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; height: 15px; margin: 0 0 4px 2px; }
.gq-tag.q { color: var(--c-q); }
.gq-tag.k { color: var(--c-k); }
.gq-tag.v { color: var(--c-v); }
.gq-qrow { display: grid; justify-items: start; }
.gq-pair { display: flex; gap: 10px; align-items: flex-start; justify-content: center; border-radius: 10px; outline: 1px dashed transparent; outline-offset: 6px;
  transition: outline-color .5s cubic-bezier(.2,.7,.2,1), visibility .5s; }
.gq-pair.stack { flex-direction: column; align-items: center; gap: 6px; }
.gq-pair .gq-tag { transition: opacity .5s cubic-bezier(.2,.7,.2,1); }
.gq-stage[data-step="1"] .gq-pair { outline-color: var(--line-strong); }
.gq-stage[data-step="0"] .gq-pair { visibility: hidden; }
.gq-stage[data-step="0"] .gq-pair .gq-tag { opacity: 0; }
.gq-col { display: grid; justify-items: start; }
.gq-block { display: flex; align-items: center; }
.gq-ell { width: 12px; text-align: center; font: 600 14px/1 var(--font-mono); color: var(--fg-3); }
.gq-stage .gq-tile .k-mat-body, .gq-stage .gq-kvtile .k-mat-body { padding: 5px; }
.gq-stage .k-vec .k-cells { padding: 5px; }
.gq-stage .gq-copy { opacity: 0; transition: opacity .35s cubic-bezier(.2,.7,.2,1); }
.gq-stage .gq-copy.landed { opacity: 1; }
.gq-stage .gq-arrow { transition: opacity .5s cubic-bezier(.2,.7,.2,1); }
.gq-stage .k-pick.sel { --kind: var(--accent) !important; }
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
.gq-ratio { font: 600 14px/1 var(--font-mono); color: var(--accent); white-space: nowrap; transition: opacity .5s, visibility .5s; }
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

// ------------------------------------------------------------ block layout (pure, see build())
const span = (lo, hi) => Array.from({ length: hi - lo }, (_, i) => lo + i);
const longVec = (dh) => dh > HEAD_COLS + TAIL_COLS + 1;          // vec() elides a vector of more than 17 cells
// Columns drawn for one d_h-wide block: all of them, or the first 12 and the last 4 around a ⋮.
const colGroups = (dh, elide) => (elide ? [span(0, HEAD_COLS), span(dh - TAIL_COLS, dh)] : [span(0, dh)]);
// Width of one block drawn at cell size c: each group's cells (with their 1 px gaps) in its own frame, and a 12 px ⋮ between groups.
const blockW = (groups, c) => groups.reduce((a, g) => a + g.length * (c + 1) + 12, 0) + (groups.length - 1) * 12;
// How the K block and the V block of one tile sit in `inner` units (10 apart, side by side). Side by side: every
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
    let made = [], comps = [], arrows = [], arrowSig = '', copies = [], kvK = [], kvV = [], mem = [], sides = [], texts = [];
    let queryTiles = [], heights = [0, 0];                 // query tiles of this build; their height in step 0 and in steps 1 and 2
    const timers = [];
    const stopAuto = () => clearTimeout(autoTimer);
    // A flight belongs to the copy it is aimed at: when that copy goes, so does the flight (its ghost clone too).
    const clearGhosts = () => st.layers.front.querySelectorAll('.k-ghost').forEach((g) => g.remove());
    const cancelFlights = () => { flightGen++; timers.splice(0).forEach(clearTimeout); clearGhosts(); };

    // One K or V block of a tile: its column groups (every column, or the first 12 and last 4 around a ⋮) as mats.
    // `band` is the head that owns the block: the query head for a copy, the KV head for the cache.
    function block(src, band, { s0, R, T, dh, plan, kind, range, trace, cls }) {
      const ms = [];
      const wrap = h('div', { class: ['gq-block', cls, plan.groups.length > 1 && 'cut'] });
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

    // ------------------------------------------------------------ fixed parts (shared kit)
    const header = sceneHeader(st, { chips: [], title: ctx.node.title });
    const picker = tokenPicker(st, ctx, { label: 'token' });
    const heads = headPicker(st, ctx, { x: SIDE.x, y: SIDE.y });
    heads.el.classList.add('gq-heads');
    const cap = sceneCaption(st);
    const legend = h('div', { class: 'gq-legend' },
      h('span', null, h('i', { class: 'gq-sw k' }), 'K'),
      h('span', null, h('i', { class: 'gq-sw v' }), 'V'),
      h('span', null, 'bytes per token, fp16'));
    st.add(legend, { x: ZONES.controls.x, y: ZONES.controls.y });
    const sideEl = h('div', { class: 'gq-side' });
    st.add(sideEl, { x: SIDE.x, y: 344 });
    ctx.setSteps(STEPS, setStep);

    function setStep(i) {
      if (!inAuto) stopAuto();                 // a reader's choice ends the auto-walk
      step = clamp(i, 0, STEPS.length - 1);
      paint();
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
      // with RoPE the scores use the rotated q and k, and the reference code repeats k_rot (not k_heads) into k_rep
      const rope = cfg.pos_encoding === 'rope', qKey = rope ? 'q_rot' : 'q_heads', kKey = rope ? 'k_rot' : 'k_heads';
      const qh = tr[`${p}${qKey}`], kh = tr[`${p}${kKey}`], vh = tr[`${p}v_heads`], kr = tr[`${p}k_rep`], vr = tr[`${p}v_rep`];
      const grp = (x) => Math.floor((x * Hkv) / H);
      const readers = (g) => Array.from({ length: H }, (_, x) => x).filter((x) => grp(x) === g);
      header.el.querySelector('.k-eyebrow').replaceChildren(chip(`Layer ${L}`, 'resid'), chip(`${H} query · ${Hkv} KV`, 'k'));
      heads.render();
      // the head chips wrap eight to a row (32 units each): the side column starts below them
      st.move(sideEl, SIDE.x, Math.max(344, SIDE.y + Math.ceil(H / 8) * 32 + 12));

      if (!kr || Hkv >= H) {                   // not a grouped config: the atlas hides this node, but never crash
        header.setSub('This configuration has one KV head per query head, so no heads share keys and values.');
        place(h('div', { class: 'gq-note' }, 'Choose a model with n_kv_head < n_head. The llama preset has 4 query heads and 2 KV heads.'), { x: HERO.x, y: HERO.qTop });
        arrows.forEach((a) => a.destroy()); arrows = []; arrowSig = '';
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

      // The KV blocks decide the rows shown: 16 tokens around the selected one (12 when K stacks above V). The copies
      // show the same rows, and are drawn only when they fit side by side at 3 px a cell or more.
      let R = Math.min(T, 16), kvPlan = planBlocks(dh, wk - 24, clamp(Math.floor(112 / R), 4, 9));
      if (!kvPlan) { R = Math.min(T, STACK_ROWS); kvPlan = planBlocks(dh, wk - 24, clamp(Math.floor(112 / R), 4, 9), { stackRows: R }); }
      const s0 = clamp(t - Math.floor(R / 2), 0, T - R), row = t - s0;
      const copyPlan = kvPlan && planBlocks(dh, wq - 24, clamp(Math.floor(64 / R), 2, 5));
      const showCopies = !!copyPlan;

      // query strips: the selected token's vector of each head. A strip that cannot run across its tile runs down it.
      const stripCells = longVec(dh) ? HEAD_COLS + TAIL_COLS + 1 : dh;
      const rowW = (c) => stripCells * (c + 1) + 12;
      const qVert = rowW(3) > wq - 24;
      const cq = qVert ? cellFor(6, 3, (c) => c + 12 <= wq - 24) : cellFor(10, 3, (c) => rowW(c) <= wq - 24);
      const showStrip = wq - 24 >= 15;         // below one cell and its frame a tile holds no strip (heads above 16)
      const stripH = !showStrip ? 0 : qVert ? stripCells * (cq + 1) - 1 : cq;

      // tile heights: step 0 shows the query tiles without their copy slots (the rise to step 1 is the offset)
      const qH0 = 80 + stripH;
      const qH = showCopies ? 116 + stripH + R * (copyPlan.c + 1) : 80 + stripH;
      heights = [qH0, qH];
      st.root.style.setProperty('--gq-off0', `${Math.round(HERO_MID - qH0 / 2 - HERO.qTop)}px`);
      const kvH = !kvPlan ? KV_PAD : kvPlan.stack ? KV_STACK_PAD + 2 * R * (kvPlan.c + 1) : KV_PAD + R * (kvPlan.c + 1);
      const col = (kind, tag, blk) => h('div', { class: 'gq-col' }, h('div', { class: `gq-tag ${kind}` }, tag), blk);

      // query heads: the selected token's q, and the K and V copies each head reads
      for (let x = 0; x < H; x++) {
        const g = grp(x);
        const strip = showStrip ? vec(qh.data.subarray((x * T + t) * dh, (x * T + t + 1) * dh), {
          orientation: qVert ? 'v' : 'h', cell: cq, gap: 1, kind: 'q', mode: 'diverging', range: rQ,
          trace: `${p}${qKey}`, cellOf: (c) => `${t},${c},${x}`, head: longVec(dh) ? HEAD_COLS : 1000, tail: longVec(dh) ? TAIL_COLS : 0,
        }) : null;
        if (strip) comps.push(strip);
        const body = h('div', { class: 'gq-tbody' },
          h('div', { class: 'gq-thead' }, h('b', null, `h${x}`), narrowQ ? null : h('span', { class: 'gq-sub' }, compactQ ? `→ KV ${g}` : `reads KV ${g}`)),
          h('div', { class: 'gq-qrow' }, h('div', { class: 'gq-tag q' }, compactQ ? 'q' : `${qKey}[${x}, ${t}]`), strip));
        if (showCopies) {
          const cK = block(kr.data, x, { s0, R, T, dh, plan: copyPlan, kind: 'k', range: rK, trace: `${p}k_rep`, cls: 'gq-copy' });
          const cV = block(vr.data, x, { s0, R, T, dh, plan: copyPlan, kind: 'v', range: rV, trace: `${p}v_rep`, cls: 'gq-copy' });
          body.append(h('div', { class: 'gq-pair' }, col('k', compactQ ? 'k' : 'k_rep', cK), col('v', compactQ ? 'v' : 'v_rep', cV)));
          copies.push({ g, x, cK, cV });
          later.push(() => { cK.highlightRow(row); cV.highlightRow(row); });
        }
        const tile = card({ kind: 'q', node: `head.${L}.${x}`, body });   // the head itself is the zoom target
        tile.classList.add('gq-tile');
        if (narrowQ) tile.classList.add('gq-narrow');
        if (compactQ) tile.classList.add('gq-compact');
        if (wq < 40) tile.classList.add('gq-tiny');
        if (x === hs) tile.classList.add('sel');
        tile.style.width = `${wq}px`; tile.style.height = `${step === 0 ? qH0 : qH}px`;
        queryTiles.push(tile);
        place(tile, { x: qX(x), y: HERO.qTop, anchor: 'tc' });
      }

      // KV heads: the cache. Full sequence window, the selected token's row highlighted.
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
        tile.style.width = `${wk}px`; tile.style.height = `${kvH}px`;
        place(tile, { x: kX(g), y: HERO.kvTop, anchor: 'tc' });
      }

      // head x reads KV head floor(x * Hkv / H): one arrow per query head, kept while the geometry holds
      const aSig = [H, Hkv, qH, wq, wk].join('|');
      if (aSig !== arrowSig) {
        arrows.forEach((a) => a.destroy());
        arrowSig = aSig;
        arrows = Array.from({ length: H }, (_, x) => {
          const a = st.arrow({ x: kX(grp(x)), y: HERO.kvTop - 4 }, { x: qX(x), y: HERO.qTop + qH + 4 }, { kind: 'k', width: 2 });
          a.el.classList.add('gq-arrow');
          return a;
        });
      }
      arrows.forEach((a, x) => a.el.style.setProperty('--kind', x === hs ? 'var(--accent)' : 'var(--c-k)'));

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
          h('span', null, `${gp.L} layers · ${gp.H} / ${gp.Hkv} heads · d_h ${gp.dh}`),
          h('span', null, 'bars: share of its MHA cache')), { x: gx, y: 268, anchor: 'tc' });
        [['MHA', gp.H], ['GQA', gp.Hkv], ['MQA', 1]].forEach(([name, hk], j) => {
          const bytes = cacheOf(gp, hk);
          const hT = (bytes / full) * BAR_MAX, bx = gx + (j - 1) * 104;
          const bar = place(h('div', { class: 'gq-bar gq-mem' }, h('i', { class: 'gq-bv' }), h('i', { class: 'gq-bk' })), { x: bx - 38, y: BASE_Y });
          const val = place(h('div', { class: 'gq-val gq-mem' }, bytesLabel(bytes)), { x: bx, y: BASE_Y - hT - 26, anchor: 'tc' });
          place(h('div', { class: 'gq-tick gq-mem' }, h('b', null, name), h('span', null, `${hk} KV`)), { x: bx, y: BASE_Y + 12, anchor: 'tc' });
          mem.push({ bar, val, hT });
        });
        place(h('div', { class: 'gq-ratio gq-mem' }, `GQA ÷${gp.H / gp.Hkv} · MQA ÷${gp.H}`), { x: gx, y: BASE_Y + 60, anchor: 'tc' });
      });
      place(h('div', { class: 'gq-base gq-mem' }), { x: HERO.x, y: BASE_Y });

      // side column and captions, one set per step
      const gh = grp(hs), rdh = readers(gh);
      let md = 0;                              // largest |difference| between the copies and the KV heads they came from
      for (let tt = 0; tt < T; tt++) for (let c = 0; c < dh; c++) {
        const a = (hs * T + tt) * dh + c, b = (gh * T + tt) * dh + c;
        md = Math.max(md, Math.abs(kr.data[a] - kh.data[b]), Math.abs(vr.data[a] - vh.data[b]));
      }
      const label = (text, kind) => h('div', { class: 'k-label', style: { '--kind': kind === 'accent' ? 'var(--accent)' : `var(--c-${kind})`, textAlign: 'left' } }, text);
      const eq = (tex) => h('div', { class: 'gq-eq', html: texHTML(tex, { display: true }) });
      const live = (tex) => h('div', { class: 'gq-live', html: texHTML(tex) });
      const note = (text) => h('div', { class: 'gq-note' }, text);
      const bytesHere = bytesPerToken(cfg.n_layer, Hkv, dh);
      // what the layout did at this width, said in the note of the step that shows it
      const elided = (kvPlan?.groups.length > 1) || (copyPlan?.groups.length > 1);
      const layoutNote = [
        elided ? `Columns ${HEAD_COLS} to ${dh - TAIL_COLS - 1} of ${dh} are elided (⋮), as for long vectors.` : '',
        kvPlan?.stack ? 'K and V are stacked: the two blocks do not fit side by side at this width.' : '',
        showCopies ? '' : 'The copies are not drawn at this width.',
      ].filter(Boolean).join(' ');
      sides = [
        [label(STEPS[0], 'q'),
          eq(rope ? String.raw`q_{\mathrm{rot},h} = \mathrm{RoPE}\big(x\,W_Q^{(h)}\big)` : String.raw`q_h = x\,W_Q^{(h)}`),
          live(String.raw`H\,d_h = ${H}\cdot ${dh} = ${H * dh} = d`),
          note(`Token “${ctx.label(t)}” at position ${t}. x is its attention input.${rope ? ' RoPE then rotates each pair of dimensions by its position.' : ''}`),
          checkBadge(H * dh, cfg.d_model, { label: 'H · d_h = d_model' })],
        [label(STEPS[1], 'k'),
          eq(String.raw`g(h) = \left\lfloor \tfrac{h\,H_{kv}}{H} \right\rfloor`),
          live(String.raw`g(${hs}) = \left\lfloor \tfrac{${hs}\cdot ${Hkv}}{${H}} \right\rfloor = ${gh}`),
          note(`Query “${ctx.label(t)}” at position ${t}. Rows ${s0} to ${s0 + R - 1} of ${T} are shown. ${layoutNote}`.trim()),
          checkBadge(md, 0, { label: `k_rep[${hs}], v_rep[${hs}] = KV ${gh}`, tol: 1e-9 })],
        [label(STEPS[2], 'accent'),
          eq(String.raw`B = 2 \cdot n_{\text{layer}} \cdot H_{kv} \cdot d_h \cdot 2`),
          live(String.raw`2 \cdot ${cfg.n_layer} \cdot ${Hkv} \cdot ${dh} \cdot 2 = ${bytesHere}\ \text{B}`),
          checkBadge(bytesHere, ctx.engine.cacheBytes(cfg, 1, 2), { label: 'equals cacheBytes(cfg, 1)' })],
      ];
      const L70 = PUBLISHED[1];
      texts = [
        `Each query head has its own slice of W_Q, so each one asks its own question of the sequence. The ${H} slices together span the full width d = <b>${cfg.d_model}</b>.`,
        Hkv === 1
          ? `All ${H} query heads read the same key and value head. The reference code copies it into every head (k_rep, v_rep), but the cache keeps one K and one V per layer and token.`
          : `Head <b>${hs}</b> reads KV head <b>${gh}</b>, shared by <b>${rdh.length}</b> heads. The reference code copies that K and V into each reader (k_rep, v_rep), but the cache keeps only the ${Hkv} K and V heads.`,
        `Keeping K and V for ${Hkv} KV head${Hkv > 1 ? 's' : ''}, not one per query head (${H}), divides the cache by <b>${H / Hkv}</b>. For LLaMA-2-70B at ${CONTEXT.toLocaleString('en-US')} tokens, fp16 K and V take ${bytesLabel(cacheOf(L70, L70.H, CONTEXT))} with MHA and ${bytesLabel(cacheOf(L70, L70.Hkv, CONTEXT))} with GQA.`,
      ];
      header.setSub(Hkv === 1
        ? `All ${H} query heads read one key and value head, so the cache holds one K and one V per layer.`
        : `${H} query heads share ${Hkv} key and value heads: each K and V head serves ${H / Hkv} queries.`);
      later.forEach((f) => f());
    }

    // ------------------------------------------------------------ one step at a time
    function paint({ instant = false } = {}) {
      st.root.dataset.step = String(step);
      const setHeights = () => queryTiles.forEach((t) => { t.style.height = `${step === 0 ? heights[0] : heights[1]}px`; });
      if (instant) snap(st.root, '.gq-tile', setHeights); else setHeights();
      arrows.forEach((a) => a.setOpacity(step === 1 ? 1 : 0));
      if (step === 1) {
        if (!arrowsDrawn) {
          arrowsDrawn = true;
          if (!instant && !reducedMotion()) arrows.forEach((a, i) => a.draw(560, 180 + i * 50));
        }
        if (!copiesDone) {
          copiesDone = true;
          const gen = ++flightGen;               // a flight that lands after the reader left step 1 does nothing
          copies.forEach((c) => {
            const land = () => { if (gen === flightGen) { c.cK.classList.add('landed'); c.cV.classList.add('landed'); } };
            if (instant || reducedMotion()) { land(); return; }
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
      cap.set(texts[step] || '');
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

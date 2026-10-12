// Mask: which keys may each query see? Four steps on the real model:
//   the T x T mask builds itself cell by cell (a diagonal wipe) · the masked scores of this head
//   and the probabilities they produce · the four masks as a live gallery (switching the model
//   keeps its weights) · the leak test: change the last token and see which earlier logits move.
// The head is the node's own (mask.L.H) and never changes here. A head picker would change the
// app's head under a node that stays on its own head, so the breadcrumb, the chip and the story
// order would disagree. Reference scene for the atlas: see docs/ATLAS_GUIDE.md and softmax.js.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, mat, op, checkBadge, appear, useStyle, rangeOf, MOTION, ZONES } from '../kit.js';
import { h, num, texHTML, heatmap, reducedMotion, sequential, diverging, onThemeChange } from '../../ui/index.js';
import { infer, cloneModel, maskAllows } from '../../engine/index.js';

const STEPS = ['The rule', 'Masked scores', 'Other masks', 'Why causal'];
const BOX = 400;                       // side of a T × T matrix (design units)
const TOP = 272;                       // top of the hero zone
const LAB = 28;                        // a matrix label: 20 units of line and 8 of gap
const PAD = 9;                         // a matrix body: 1 unit of border and 8 of padding (kit .k-mat-body)
const X0 = ZONES.hero.x;               // left edge of the hero (110)
const EQ = { x: 1180, y: 280 };        // equation column (ZONES.side)
const STRIP = { x: 560, w: 560 };      // the query strip of step 1: left edge and width budget
const MINI = 150;                      // side of a gallery mask
const BARS = { w: 1030, h: 170 };      // leak bars: hero width and bar height
const LIT = 1, OFF = -Infinity;        // mask values: may see / masked (hatched by the kit)
const MASKS = [
  { id: 'causal', name: 'causal', sub: 'j ≤ i' },
  { id: 'none', name: 'none', sub: 'every key' },
  { id: 'sliding', name: 'sliding', sub: 'last w keys' },
  { id: 'prefix', name: 'prefix', sub: 'prefix visible' },
];
const CANDIDATES = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h'];   // replacement tokens, in order
const DENSE_T = 40;                    // above this many tokens the query picker uses narrow tiles
const KEY_GAP = 12;                    // minimum spacing (units) between key labels of the query row
const VALUE_GAP = 44;                  // minimum spacing (units) between bar value labels

useStyle('mask-scene', `
.mk-fade { transition: opacity .45s ease, left .6s cubic-bezier(.65,0,.35,1), top .6s cubic-bezier(.65,0,.35,1); }
.mk-off { opacity: 0 !important; pointer-events: none; }
.mk-band { position: absolute; border-radius: 4px; border: 2px solid var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent); pointer-events: none; transition: opacity .45s ease, left .6s cubic-bezier(.65,0,.35,1), top .6s cubic-bezier(.65,0,.35,1); }
.mk-lab { height: 20px; margin: 0 0 8px; display: flex; align-items: flex-end; justify-content: center; }
.mk-eq { width: 340px; display: grid; gap: 12px; }
.mk-eq [hidden] { display: none !important; }
.mk-eq-main { overflow-x: auto; }
.mk-eq-live { font-size: 17px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.mk-legend { display: flex; gap: 22px; align-items: center; font: 500 15px/1.2 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.mk-sw { display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 8px; vertical-align: -2px; }
.mk-sw.on { background: var(--c-attn); }
.mk-sw.off { background: repeating-linear-gradient(135deg, var(--c-mask) 0 2px, transparent 2px 5px); box-shadow: inset 0 0 0 1px var(--c-mask); }
.mk-keys { display: flex; font: 500 15px/1 var(--font-mono); color: var(--fg-3); }
.mk-keys span { display: block; flex: none; text-align: center; white-space: nowrap; }
.mk-mini { box-sizing: border-box; width: 176px; padding: 10px 10px 12px; display: grid; justify-items: center; gap: 6px; border-radius: 12px; border: 1px solid var(--line); background: color-mix(in srgb, var(--surface) 86%, transparent); box-shadow: 0 10px 24px rgba(0, 0, 0, .12); color: var(--fg); font: inherit; cursor: pointer; transition: opacity .45s ease, border-color .25s, box-shadow .25s, transform .25s; }
.mk-mini:hover { transform: translateY(-2px); border-color: var(--line-strong); }
.mk-mini.sel { border-color: var(--accent); box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 45%, transparent), 0 10px 24px rgba(0, 0, 0, .14); }
.mk-mini:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.mk-mini canvas { display: block; }
.mk-mini-name { font: 600 16px/1.2 var(--font-mono); }
.mk-mini-sub { font: 500 15px/1.2 var(--font-mono); color: var(--fg-3); }
.mk-row { border-bottom: 1px solid var(--line-strong); }
.mk-bar { position: absolute; bottom: 0; border-radius: 4px 4px 2px 2px; background: linear-gradient(180deg, color-mix(in srgb, var(--c-attn) 92%, white), var(--c-attn)); box-shadow: 0 6px 14px rgba(0, 0, 0, .16); transition: height .6s cubic-bezier(.65,0,.35,1); }
.mk-bar-val { position: absolute; left: 50%; translate: -50% 0; bottom: 100%; margin-bottom: 5px; font: 600 13px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.mk-changed { position: absolute; top: 0; bottom: 0; border: 1px dashed var(--line-strong); border-radius: 6px; background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--c-mask) 22%, transparent) 0 2px, transparent 2px 6px); }
.mk-changed span { position: absolute; left: 50%; translate: -50% 0; top: 10px; writing-mode: vertical-rl; font: 600 13px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.mk-axis { border-top: 1px solid transparent; }
.mk-tok { position: absolute; top: 4px; text-align: center; font: 600 15px/1 var(--font-mono); color: var(--fg-2); overflow: hidden; }
.mk-tok.hot { color: var(--fg); font-weight: 700; }
.k-op.mk-op { font-size: 15px; }
.mk-dense .k-picker { gap: 1px; }
.mk-dense .k-pick { width: 14px !important; }
.mk-slide-line { display: flex; align-items: center; gap: 10px; }
.mk-range { flex: 1; min-width: 0; accent-color: var(--c-attn); }
.mk-select { font: 600 16px/1 var(--font-mono); color: var(--fg); background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
`);

const clampInt = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.floor(v)));
const gapFor = (T) => (T > 24 ? 0 : 1);
const cellFor = (T, span, gap) => clampInt((span - (T - 1) * gap) / T, 2, 22);
const hide = (el, off) => { el.classList.toggle('mk-off', off); el.inert = off; };
const showGroup = (list, on) => list.forEach((el) => hide(el, !on));
const setHTML = (el, html) => { if (el._html !== html) { el.innerHTML = html; el._html = html; } };

// KaTeX output depends only on its source, so each equation is typeset once and then reused.
const texCache = new Map();
const tex = (src, display = false) => {
  const k = `${display ? 'D' : 'I'}${src}`;
  let out = texCache.get(k);
  if (out === undefined) {
    if (texCache.size > 300) texCache.clear();
    out = texHTML(src, { display });
    texCache.set(k, out);
  }
  return out;
};

// A label above a matrix body, as wide as the body so that it centres over it.
const labEl = (src, span, kind = 'attn') => h('div', { class: 'k-label mk-lab', style: { width: `${span + 2 * PAD}px`, '--kind': `var(--c-${kind})` }, html: tex(src) });
// The row band of a matrix (the kit's .k-row-band look), placed by arithmetic rather than measured.
const bandEl = (span, cell) => h('div', { class: 'mk-band', 'aria-hidden': 'true', style: { width: `${span + 6}px`, height: `${cell + 4}px` } });

// Colour a grid in place, the way mat() colours it (sequential or diverging ramp, hatched for −∞), but
// write only the cells whose value changed. mat.update repaints every cell on each call, and the wipe
// changes only a few cells per frame. Every colour is worked out before the first write.
function gridPainter(m, cols, { mode, kind, range = null }) {
  const n = cols * cols;
  let shown = new Array(n).fill(0), shownRange = 1;           // mat() paints zeros with range 1
  function paint(vals, force = false) {
    const rg = range ?? rangeOf(vals, mode);
    const full = force || rg !== shownRange;
    const memo = new Map(), idx = [], col = [];
    for (let k = 0; k < n; k++) {
      const v = vals[k];
      if (!full && v === shown[k]) continue;
      let c = memo.get(v);
      if (c === undefined) {
        c = v === OFF ? null : mode === 'sequential' ? sequential(Math.max(0, v) / rg, kind) : diverging(v / rg);
        memo.set(v, c);
      }
      idx.push(k);
      col.push(c);
    }
    idx.forEach((k, i) => {
      const el = m.cellEl(Math.floor(k / cols), k % cols), c = col[i];
      el.style.background = c ?? 'transparent';
      el.classList.toggle('masked', c === null);
      shown[k] = vals[k];
    });
    shownRange = rg;
  }
  const off = onThemeChange(() => paint(shown.slice(), true));  // the colours follow the theme
  return { paint, off };
}

const ruleTex = (cfg) => {
  if (cfg.mask === 'none') return '\\Mk_{ij} = 1';
  if (cfg.mask === 'sliding') return `\\Mk_{ij} = [\\,j \\le i\\,]\\;[\\,j > i - ${cfg.window}\\,]`;
  if (cfg.mask === 'prefix') return `\\Mk_{ij} = [\\,j \\le i\\,] \\lor [\\,j < ${cfg.prefix_len}\\,]`;
  return '\\Mk_{ij} = [\\,j \\le i\\,]';
};

const maskName = (cfg) => ({
  causal: 'causal', none: 'none (bidirectional)', sliding: `sliding window, w = ${cfg.window}`, prefix: `prefix-LM, p = ${cfg.prefix_len}`,
})[cfg.mask] || cfg.mask;

const RULE_WORDS = {
  causal: () => 'Query <b>i</b> may see key <b>j</b> only when j ≤ i: the lower triangle, diagonal included. The wipe runs diagonal by diagonal, so every token sees itself first, then its predecessor.',
  none: () => 'No mask: every query may see every key, so no cell is hatched. A decoder cannot work this way, because it would read the token it is predicting.',
  sliding: (c) => `Sliding window: query <b>i</b> sees only the last <b>${c.window}</b> keys, j in (i − w, i]. The lit band is w diagonals wide.`,
  prefix: (c) => `Prefix-LM: the first <b>${c.prefix_len}</b> tokens see each other, and every query sees them. Everything after the prefix stays causal.`,
};

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const l = ctx.params.l, hd = ctx.params.h;    // the node's own layer and head
    let step = 0, T = 0, key = '', q = 0, rowQ = -1;
    let G = {};                                   // element groups of the current structure
    let geo = null;                               // grid geometry of the current structure
    let mMask = null, maskPaint = null, mBand = null, maskLab = null;
    let mScores = null, scPaint = null, mProbs = null, prPaint = null, opEl = null, bandS = null, bandP = null, scLab = null, prLab = null;
    let rowMat = null, rowLabel = null, rowKeys = null, legendEl = null;
    let titleA = null, titleB = null;
    let minis = [];
    let barsA = [], barsB = [], axisLabels = [];
    let lastMask = null, buildRaf = null, buildOrder = null, morphTimer = null;
    let newTok = null, leakRes = null, optKey = '', sliderKey = 'window';
    let auto = null, autoDriving = false;
    const kids = [];                              // structure elements, disposed on rebuild
    const stopAuto = () => { clearTimeout(auto); auto = null; };

    // ------------------------------------------------ shared parts (persist across rebuilds)
    const header = sceneHeader(st, { chips: [[`Layer ${l}`, 'resid'], [`Head ${hd}`, 'attn']], title: 'Mask' });
    const picker = tokenPicker(st, ctx, { label: 'query', maxWidth: 1000 });
    const cap = sceneCaption(st, {});

    const charSel = h('select', { class: 'mk-select', 'aria-label': 'replacement for the last token', onChange: () => { newTok = Number(charSel.value); render(); } });
    const charRow = st.add(h('div', { class: 'k-picker-wrap mk-fade' }, h('span', { class: 'k-picker-label' }, 'replace last token'), charSel), { x: ZONES.controls.x, y: ZONES.controls.y });

    const eqHead = h('div', { class: 'k-label', style: { textAlign: 'left' } });
    const eqMain = h('div', { class: 'mk-eq-main' });
    const eqLive = h('div', { class: 'mk-eq-live' });
    const eqCheck = h('div');
    const sliderLab = h('span', { class: 'k-picker-label' });
    const slider = h('input', { type: 'range', class: 'mk-range', 'aria-label': 'mask size', onInput: () => ctx.store.setConfig(sliderKey === 'window' ? { window: Number(slider.value) } : { prefix_len: Number(slider.value) }) });
    const sliderVal = h('b', { class: 'k-label' });
    const sliderRow = h('div', { class: 'mk-slide-line' }, sliderLab, slider, sliderVal);
    const eq = st.add(h('div', { class: 'mk-eq mk-fade' }, eqHead, eqMain, eqLive, eqCheck, sliderRow), { x: EQ.x, y: EQ.y });

    // ------------------------------------------------ structure: rebuilt when the tokens change
    function add(node, x, y, anchor = 'tl', off = null) {
      st.add(node, { x, y, anchor });
      node.classList.add('mk-fade');
      kids.push({ el: node, off });
      return node;
    }
    // place a band on query row `row` of any grid (the grids share their top and pitch)
    const placeBand = (band, row) => { band.style.top = `${geo.gy + row * geo.pitch - 2}px`; };

    // mini gallery: the rule for any mask, from the engine's own maskAllows (mirrors build_mask)
    function miniVals(id) {
      const c = { ...ctx.state.config, mask: id }, out = new Array(T * T);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) out[i * T + j] = maskAllows(c, i, j) ? LIT : OFF;
      return out;
    }
    // a card changes only with its own parameter: sliding with the window, prefix with its length
    function miniKey(id) {
      const c = ctx.state.config;
      return id === 'sliding' ? `w${c.window}` : id === 'prefix' ? `p${c.prefix_len}` : '';
    }

    function build() {
      stopBuild();
      cancelMorph();
      for (const k of kids.splice(0)) { try { k.off?.(); } catch { /* ignore */ } k.el.remove(); }
      const s = ctx.state;
      T = s.ids.length;
      key = s.ids.join(',');
      rowQ = -1;
      lastMask = null;
      mScores = null; mProbs = null; opEl = null; scPaint = null; prPaint = null;
      bandS = null; bandP = null; scLab = null; prLab = null;

      // Geometry in design units. Every position below is arithmetic on these numbers, so no step
      // reads the layout (a read after the cells change forces a full layout: ~200 ms on 40 × 40).
      const gap = gapFor(T), cell = cellFor(T, BOX, gap), span = T * cell + (T - 1) * gap, pitch = cell + gap;
      const gy = TOP + LAB + PAD;                 // top of every grid
      geo = { cell, gap, span, pitch, gy };
      const zeros = new Array(T * T).fill(0);
      const grid = (opts) => mat(zeros, T, T, { cell, gap, kind: 'attn', ...opts });

      // the mask: built by the wipe, then the traced mask; the label and the band belong to it
      mMask = grid({ mode: 'sequential', range: 1, trace: 'mask', cellOf: (r, c) => `${r},${c}` });
      maskPaint = gridPainter(mMask, T, { mode: 'sequential', kind: 'attn', range: 1 });
      add(mMask, X0, TOP + LAB, 'tl', () => { maskPaint.off(); mMask.destroy(); });
      maskLab = add(labEl('\\Mk', span), X0, TOP);
      mBand = add(bandEl(span, cell), X0 + PAD - 3, gy - 2);

      // the query strip (step 1 of the story) follows the query row; the legend sits below the
      // lowest possible strip, whose keys reach 15 units under it
      const rowCell = clampInt((STRIP.w - (T - 1)) / T, 3, 18), stripH = rowCell + 2 * PAD;
      const stripAt = (row) => {
        const top = gy + row * pitch + cell / 2 - rowCell / 2 - PAD;   // top of the strip's body
        return { label: top - 44, strip: top, keys: top + stripH + 6 };
      };
      geo.stripAt = stripAt;
      legendEl = add(h('div', { class: 'mk-legend' },
        h('span', null, h('i', { class: 'mk-sw on' }), 'may see'),
        h('span', null, h('i', { class: 'mk-sw off' }), 'masked: −∞ before the softmax, 0 after')), X0, stripAt(T - 1).keys + 15 + 12);

      const every = Math.max(1, Math.ceil(KEY_GAP / (rowCell + 1)));
      rowLabel = add(h('div', { class: 'k-label', style: { textAlign: 'left' } }), STRIP.x, stripAt(0).label);
      rowMat = mat(zeros.slice(0, T), 1, T, { cell: rowCell, gap: 1, kind: 'attn', mode: 'sequential', range: 1, trace: 'mask', cellOf: (r, c) => `${q},${c}` });
      add(rowMat, STRIP.x, stripAt(0).strip, 'tl', rowMat.destroy);
      rowKeys = add(h('div', { class: 'mk-keys', style: { marginLeft: `${PAD}px` } },
        Array.from({ length: T }, (_, j) => h('span', { style: { width: `${rowCell + 1}px` } }, j % every === 0 ? ctx.label(j) : ''))), STRIP.x, stripAt(0).keys);

      // gallery: the four masks as live switches (the minis are the rule itself, not a picture of it)
      minis = MASKS.map((m, k) => {
        const hm = heatmap({ data: miniVals(m.id), rows: T, cols: T, mode: 'sequential', kind: 'attn', range: [0, 1], cell: clampInt(MINI / T, 3, 34), showValues: false, tooltip: () => '', ariaLabel: `${m.name} mask` });
        const btn = h('button', {
          type: 'button', class: 'mk-mini', 'aria-label': `${m.name} mask`,
          onClick: () => { const c = ctx.state.config; ctx.store.setConfig({ mask: m.id, window: c.window, prefix_len: c.prefix_len }); },
        }, h('div', { class: 'mk-mini-name' }, m.name), hm.el, h('div', { class: 'mk-mini-sub' }, m.sub));
        add(btn, 760 + (k % 2) * 196, TOP + Math.floor(k / 2) * 236, 'tl', hm.destroy);
        return { m, hm, btn, key: miniKey(m.id) };
      });

      // leak bars: causal row above, the no-mask row below (same weights, mask changed)
      const slot = BARS.w / T, rowStyle = { width: `${BARS.w}px`, height: `${BARS.h}px` };
      const rowEls = [h('div', { class: 'mk-row', style: rowStyle }), h('div', { class: 'mk-row', style: rowStyle })];
      const bars = [[], []];
      for (let i = 0; i < T; i++) for (let r = 0; r < 2; r++) {
        if (i === T - 1) {
          rowEls[r].append(h('div', { class: 'mk-changed', style: { left: `${i * slot + 3}px`, width: `${slot - 6}px` } }, h('span', null, 'changed')));
          bars[r].push(null);
          continue;
        }
        const b = h('div', { class: 'mk-bar', style: { left: `${i * slot + slot * 0.2}px`, width: `${slot * 0.6}px`, height: '0px' } }, h('span', { class: 'mk-bar-val' }));
        rowEls[r].append(b);
        bars[r].push(b);
      }
      const axis = h('div', { class: 'mk-axis', style: { width: `${BARS.w}px`, height: '24px' } });
      axisLabels = Array.from({ length: T }, (_, i) => {
        const t = h('div', { class: 'mk-tok', style: { left: `${i * slot}px`, width: `${slot}px` } }, ctx.label(i));
        axis.append(t);
        return t;
      });
      const yA = TOP + 34, yB = yA + BARS.h + 56;
      titleA = add(h('div', { class: 'k-label', style: { textAlign: 'left' } }), X0, TOP);
      titleB = add(h('div', { class: 'k-label', style: { textAlign: 'left' } }), X0, yB - 30);
      add(rowEls[0], X0, yA);
      add(rowEls[1], X0, yB);
      add(axis, X0, yB + BARS.h + 6);
      barsA = bars[0];
      barsB = bars[1];

      G = {
        mask: [mMask, maskLab, mBand],
        scores: [],
        row: [rowLabel, rowMat, rowKeys],
        legend: [legendEl],
        minis: minis.map((x) => x.btn),
        bars: [titleA, titleB, rowEls[0], rowEls[1], axis],
      };
    }

    function ensure() {
      if (T > 0 && ctx.state.ids.join(',') === key) return false;
      build();
      return true;
    }

    // The masked scores and the probabilities of this head. Only step 1 shows them, so they are built
    // on first use in that step (and again after a rebuild), keeping the other steps light.
    function ensureGrids() {
      if (mScores || !geo) return;
      const { cell, gap, span, gy } = geo;
      const zeros = new Array(T * T).fill(0);
      const grid = (opts) => mat(zeros, T, T, { cell, gap, kind: 'attn', ...opts });
      const PX = 640;                             // the probabilities sit to the right of the scores
      mScores = grid({ mode: 'diverging', trace: `L${l}.attn.masked`, cellOf: (r, c) => `${r},${c},${hd}` });
      scPaint = gridPainter(mScores, T, { mode: 'diverging' });
      add(mScores, X0, TOP + LAB, 'tl', () => { scPaint.off(); mScores.destroy(); });
      scLab = add(labEl(`\\hat S^{(${l},${hd})}`, span), X0, TOP);
      mProbs = grid({ mode: 'sequential', range: 1, trace: `L${l}.attn.probs`, cellOf: (r, c) => `${r},${c},${hd}` });
      prPaint = gridPainter(mProbs, T, { mode: 'sequential', kind: 'attn', range: 1 });
      add(mProbs, PX, TOP + LAB, 'tl', () => { prPaint.off(); mProbs.destroy(); });
      prLab = add(labEl(`\\Aw^{(${l},${hd})}`, span), PX, TOP);
      bandS = add(bandEl(span, cell), X0 + PAD - 3, gy - 2);
      bandP = add(bandEl(span, cell), PX + PAD - 3, gy - 2);
      // centred between the two grids (x 527 … 640)
      opEl = add(op('\\softmax', { kind: 'attn', size: 64 }), 575, gy + span / 2, 'c');
      opEl.classList.add('mk-op');
      G.scores = [mScores, scLab, mProbs, prLab, opEl, bandS, bandP];
    }

    // ------------------------------------------------ the wipe (step 1) and the morph (gallery)
    // Cells are revealed by lag |i − j| (the main diagonal first); within a lag, lit cells first.
    function wipeOrder() {
      const traced = ctx.state.trace.mask.data, idx = Array.from({ length: T * T }, (_, k) => k);
      idx.sort((a, b) => {
        const ia = Math.floor(a / T), ja = a % T, ib = Math.floor(b / T), jb = b % T;
        const la = Math.abs(ia - ja), lb = Math.abs(ib - jb);
        if (la !== lb) return la - lb;
        const aa = traced[a] === LIT ? 0 : 1, ab = traced[b] === LIT ? 0 : 1;
        if (aa !== ab) return aa - ab;
        return ia - ib;
      });
      return idx;
    }

    function maskAt(p) {
      const traced = ctx.state.trace.mask.data, N = T * T, out = new Array(N).fill(0), n = Math.floor(p * N);
      for (let k = 0; k < n; k++) { const idx = buildOrder[k]; out[idx] = traced[idx] === LIT ? LIT : OFF; }
      return out;
    }

    function startBuild() {
      stopBuild();
      cancelMorph();
      lastMask = null;
      if (reducedMotion() || !mMask) return;
      buildOrder = wipeOrder();
      const dur = clampInt(T * 40, 900, 2000), t0 = performance.now();
      const frame = (now) => {
        const p = Math.min(1, (now - t0) / dur);
        const vals = maskAt(p);
        maskPaint.paint(vals);
        if (p < 1) { buildRaf = requestAnimationFrame(frame); return; }
        buildRaf = null;
        lastMask = vals;          // the finished wipe is the starting point for the next morph
      };
      buildRaf = requestAnimationFrame(frame);
    }

    function stopBuild() {
      if (buildRaf) cancelAnimationFrame(buildRaf);
      buildRaf = null;
      buildOrder = null;
    }

    function cancelMorph() {
      clearTimeout(morphTimer);
      morphTimer = null;
    }

    // Set the live mask. A morph fades lit cells out first, then hatches them (hatching cannot fade).
    function setMask(vals, morph) {
      const pending = morphTimer !== null;     // an earlier morph's final frame is still due
      cancelMorph();
      if (!morph || reducedMotion() || !lastMask || lastMask.length !== vals.length) {
        maskPaint.paint(vals);
        lastMask = vals;
        return;
      }
      let changed = false;
      const interim = vals.slice();
      for (let k = 0; k < vals.length; k++) {
        if (lastMask[k] === vals[k]) continue;
        changed = true;
        if (lastMask[k] === LIT && vals[k] === OFF) interim[k] = 0;
      }
      lastMask = vals;
      if (!changed) { if (pending) maskPaint.paint(vals); return; }   // never leave the interim frame on screen
      maskPaint.paint(interim);
      morphTimer = setTimeout(() => { morphTimer = null; maskPaint.paint(vals); }, 420);
    }

    // ------------------------------------------------ the leak test (step 4)
    function pickTok() {
      const s = ctx.state, last = s.ids[s.ids.length - 1], V = s.tokenizer.size;
      if (newTok == null || newTok === last || newTok >= V) {
        newTok = (last + 1) % V;
        for (const c of CANDIDATES) { const id = s.tokenizer.vocab.indexOf(c); if (id >= 0 && id !== last) { newTok = id; break; } }
      }
      return newTok;
    }

    // Earlier logits under the causal clone, the no-mask clone and the live model (same weights),
    // before and after the last token changes. The passes run one per task (a pass is ~30 ms on 40
    // tokens), so the step and its caption never wait for all of them. The live model is one of the
    // two clones when its mask is causal or none, so its logits are reused. The result is cached per
    // forward pass and token; the pass that finishes last redraws the step.
    let leakJob = null, leakGen = 0, alive = true;
    function leakPlan(tok) {
      const s = ctx.state, ids = s.ids, n = ids.length, model = s.model;
      const ids2 = ids.slice(0, n - 1);
      ids2.push(tok);
      const clone = (mask) => { const m = cloneModel(model); m.config = { ...m.config, mask }; return m; };
      const run = (m, x) => infer(m, x, { trace: false }).logits.data;
      const job = { src: s.logits, tok, n, V: s.logits.shape[1], liveMask: model.config.mask, gen: ++leakGen, P: {} };
      const P = job.P;
      const passes = [
        () => { P.causal = clone('causal'); P.A0 = run(P.causal, ids); },
        () => { P.A1 = run(P.causal, ids2); },
        () => { P.bidi = clone('none'); P.B0 = run(P.bidi, ids); },
        () => { P.B1 = run(P.bidi, ids2); },
      ];
      if (job.liveMask !== 'causal' && job.liveMask !== 'none') {
        passes.push(() => { P.live = cloneModel(model); P.L0 = run(P.live, ids); }, () => { P.L1 = run(P.live, ids2); });
      }
      let i = 0;
      const next = () => {
        if (!alive || job.gen !== leakGen) return;      // superseded by a newer forward pass or token
        passes[i++]();
        if (i < passes.length) { setTimeout(next, 0); return; }
        leakJob = null;
        leakRes = leakFinish(job);
        if (step === 3) render();
      };
      leakJob = job;
      setTimeout(next, 0);
    }

    // The leak numbers from the finished passes: the largest |Δ logit| per position, for each model.
    function leakFinish(job) {
      const P = job.P, n = job.n, V = job.V;
      let L0 = P.L0, L1 = P.L1;
      if (job.liveMask === 'causal') { L0 = P.A0; L1 = P.A1; }
      else if (job.liveMask === 'none') { L0 = P.B0; L1 = P.B1; }
      const dC = new Array(n - 1).fill(0), dB = new Array(n - 1).fill(0), dL = new Array(n - 1).fill(0);
      for (let i = 0; i < n - 1; i++) {
        for (let v = 0; v < V; v++) {
          const k = i * V + v;
          dC[i] = Math.max(dC[i], Math.abs(P.A1[k] - P.A0[k]));
          dB[i] = Math.max(dB[i], Math.abs(P.B1[k] - P.B0[k]));
          dL[i] = Math.max(dL[i], Math.abs(L1[k] - L0[k]));
        }
      }
      // the clone of the live weights with the live config must reproduce the traced logits
      let rep = 0;
      for (let k = 0; k < L0.length; k++) rep = Math.max(rep, Math.abs(L0[k] - job.src.data[k]));
      return { src: job.src, tok: job.tok, dC, dB, dL, rep, mxC: Math.max(0, ...dC), mxB: Math.max(0, ...dB), mxL: Math.max(0, ...dL) };
    }

    // The leak result for the current forward pass and token, or null while its passes are running.
    function leakRun() {
      const s = ctx.state;
      if (s.ids.length < 2) return null;
      const tok = pickTok();
      if (leakRes && leakRes.src === s.logits && leakRes.tok === tok) return leakRes;
      if (!(leakJob && leakJob.src === s.logits && leakJob.tok === tok)) leakPlan(tok);
      return null;
    }

    // ------------------------------------------------ render the current step
    function render({ morph = false } = {}) {
      if (!T) return;
      const s = ctx.state, cfg = s.config, N = T * T;
      if (step === 1) ensureGrids();
      q = Math.min(s.sel.token, T - 1);
      const traced = s.trace.mask.data;
      let nVis = 0;
      for (let j = 0; j < T; j++) if (traced[q * T + j] === LIT) nVis++;

      showGroup(G.mask, step === 0 || step === 2);
      showGroup(G.scores, step === 1);
      showGroup(G.row, step === 0);
      showGroup(G.legend, step !== 3);
      showGroup(G.minis, step === 2);
      showGroup(G.bars, step === 3);
      hide(picker.el, step === 3);
      hide(charRow, step !== 3);
      picker.el.classList.toggle('mk-dense', T > DENSE_T);

      // the mask: the wipe owns it while it runs, otherwise it is the traced mask; the band marks the query row
      if (step === 0 || step === 2) {
        if (!buildRaf) setMask(Array.from(traced, (v) => (v === LIT ? LIT : OFF)), morph);
        placeBand(mBand, q);
      }
      if (step === 0) {
        if (rowQ !== q) { for (let c = 0; c < T; c++) rowMat.cellEl(0, c).dataset.cell = `${q},${c}`; rowQ = q; }
        rowMat.update(Array.from(traced.subarray(q * T, q * T + T), (v) => (v === LIT ? LIT : OFF)));
        rowLabel.textContent = `query “${ctx.label(q)}” at position ${q}`;
        // the strip slides under the highlighted row (CSS transition on top of these positions)
        const g = geo.stripAt(q);
        rowLabel.style.top = `${g.label}px`;
        rowMat.style.top = `${g.strip}px`;
        rowKeys.style.top = `${g.keys}px`;
      }
      if (step === 1) {
        const off = hd * N, sc = s.trace[`L${l}.attn.masked`].data, pr = s.trace[`L${l}.attn.probs`].data;
        scPaint.paint(Array.from(sc.subarray(off, off + N)));
        // masked cells are exactly 0; the kit hatches −∞, so the display marks them the same way
        prPaint.paint(Array.from(pr.subarray(off, off + N), (p, k) => (traced[k] === LIT ? p : OFF)));
        placeBand(bandS, q);
        placeBand(bandP, q);
      }
      if (step === 2) {
        for (const x of minis) {
          const k = miniKey(x.m.id);
          if (x.key !== k) { x.hm.update({ data: miniVals(x.m.id) }); x.key = k; }
          x.btn.classList.toggle('sel', cfg.mask === x.m.id);
        }
      }

      const sliderOn = step === 2 && (cfg.mask === 'sliding' || cfg.mask === 'prefix');
      sliderRow.hidden = !sliderOn;
      if (sliderOn) {
        sliderKey = cfg.mask === 'sliding' ? 'window' : 'prefix_len';
        slider.min = sliderKey === 'window' ? 1 : 0;
        slider.max = T;
        if (document.activeElement !== slider) slider.value = String(cfg[sliderKey]);
        sliderLab.textContent = sliderKey === 'window' ? 'window w' : 'prefix p';
        sliderVal.textContent = slider.value;
      }

      // equation column, check, caption and context line for this step
      eqHead.textContent = STEPS[step];
      let main = '', liveEq = '', check = null, words = '', sub = '';
      if (step === 0) {
        let bad = 0;
        for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) if ((traced[i * T + j] === LIT) !== maskAllows(cfg, i, j)) bad++;
        main = tex(ruleTex(cfg), true);
        liveEq = tex(`\\text{query } ${q}\\text{:}\\quad \\sum_{j} \\Mk_{${q}j} = ${nVis} \\text{ of } ${T}`, true);
        check = checkBadge(bad, 0, { label: 'equals the traced mask' });
        words = (RULE_WORDS[cfg.mask] || RULE_WORDS.causal)(cfg);
        sub = `Live mask: <b>${maskName(cfg)}</b>. Query “<b>${ctx.label(q)}</b>” at position ${q} may see ${nVis} of its ${T} keys.`;
      } else if (step === 1) {
        const off = hd * N, sc = s.trace[`L${l}.attn.masked`].data, pr = s.trace[`L${l}.attn.probs`].data;
        let bad = 0, jm = -1, mx = -Infinity;
        for (let k = 0; k < N; k++) {
          const masked = traced[k] !== LIT;
          if (masked !== (sc[off + k] === OFF)) bad++;
          if (masked && pr[off + k] !== 0) bad++;
        }
        for (let j = 0; j < T; j++) if (traced[q * T + j] !== LIT) { jm = j; break; }
        for (let j = 0; j < T; j++) if (traced[q * T + j] === LIT) mx = Math.max(mx, sc[off + q * T + j]);
        let Z = 0;
        for (let j = 0; j < T; j++) if (traced[q * T + j] === LIT) Z += Math.exp(sc[off + q * T + j] - mx);
        const ek = Math.exp(sc[off + q * T + q] - mx), pq = pr[off + q * T + q];
        main = tex('\\hat S_{ij} = \\begin{cases} \\tilde S_{ij} & \\Mk_{ij} = 1 \\\\ -\\infty & \\Mk_{ij} = 0 \\end{cases}', true);
        const first = jm >= 0
          ? `\\hat S_{${q},${jm}} = -\\infty \\;\\Rightarrow\\; \\Aw_{${q},${jm}} = 0`
          : `\\text{every key is visible to query } ${q}`;
        liveEq = tex(`\\begin{aligned} ${first} \\\\ \\Aw_{${q},${q}} = \\frac{${num(ek, 3)}}{${num(Z, 3)}} = ${num(pq, 3)} \\end{aligned}`, true);
        check = checkBadge(bad, 0, { label: 'masked: −∞ score, zero weight' });
        words = 'Disallowed scores become <b>−∞</b> before the softmax. e<sup>−∞</sup> = 0, so those weights are exactly zero, and each row still sums to 1 over the keys it may see.';
        sub = `Live mask: <b>${maskName(cfg)}</b>. Query “<b>${ctx.label(q)}</b>” at position ${q} keeps ${nVis} of its ${T} scores; the rest are −∞.`;
      } else if (step === 2) {
        main = tex(ruleTex(cfg), true);
        words = 'Four masks, one trained model. Picking a mask changes only the mask: the weights stay the same, and the pattern and the probabilities follow it.';
        sub = `Live mask: <b>${maskName(cfg)}</b>. Click a card to switch the model.`;
      } else {
        const lk = leakRun();
        sub = T > 1
          ? `Live mask: <b>${maskName(cfg)}</b>. The last token “<b>${ctx.label(T - 1)}</b>” at position ${T - 1} is replaced by “<b>${s.tokenizer.label(pickTok())}</b>”.`
          : `Live mask: <b>${maskName(cfg)}</b>. The leak test needs at least two tokens.`;
        if (!lk && T < 2) {
          words = 'Use at least two tokens. The leak test needs an earlier position to look at.';
        } else if (!lk) {
          // the passes are running; the equation column keeps its content until they finish
          header.setSub(sub);
          cap.set('Running the leak test on the same weights.');
          return;
        } else {
          const slot = BARS.w / T, every = Math.max(1, Math.ceil(VALUE_GAP / slot));
          const scale = Math.max(...lk.dB, 1e-12), H = BARS.h - 24;     // headroom for the value labels
          for (let i = 0; i < T - 1; i++) {
            const show = i % every === 0;             // thin the labels so neighbours never touch
            barsA[i].style.height = `${(lk.dC[i] / scale) * H}px`;
            barsB[i].style.height = `${(lk.dB[i] / scale) * H}px`;
            barsA[i].firstChild.textContent = show ? lk.dC[i].toFixed(2) : '';
            barsB[i].firstChild.textContent = show ? lk.dB[i].toFixed(2) : '';
          }
          // the rows are named by the mask they compute: the live model is the row with its own mask
          titleA.textContent = cfg.mask === 'causal' ? 'the model (mask: causal)' : 'causal mask · illustration, same weights';
          titleB.textContent = cfg.mask === 'none' ? 'the model (mask: none)' : 'no mask · illustration, same weights';
          const vocab = s.tokenizer.vocab, last = s.ids[T - 1], optNow = `${last}|${vocab.length}`;
          if (optKey !== optNow) {
            charSel.replaceChildren(...vocab.map((_, id) => h('option', { value: String(id), disabled: id === last }, s.tokenizer.label(id))));
            optKey = optNow;
          }
          charSel.value = String(lk.tok);
          axisLabels[T - 1].textContent = s.tokenizer.label(lk.tok);
          axisLabels[T - 1].classList.add('hot');
          main = tex('\\Delta_i = \\max_{v} \\big|\\, \\ell_{iv}(x\') - \\ell_{iv}(x) \\,\\big|', true);
          // the row titles name the model; the equation keeps to the numbers so the lines fit the side column
          const eqRow = (label, v) => `\\text{${label}: } & \\max_{i<${T - 1}} \\Delta_i = ${num(v, 2)}`;
          const lines = [eqRow('causal', lk.mxC), eqRow('no mask', lk.mxB)];
          if (cfg.mask !== 'causal' && cfg.mask !== 'none') lines.push(eqRow(cfg.mask, lk.mxL));
          liveEq = tex(`\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`, true);
          check = checkBadge(lk.rep, 0, { label: 'reproduces the traced logits', tol: 1e-9 });
          words = `Changing the last token moves the earlier logits by <b>${lk.mxC.toFixed(2)}</b> with the causal mask and by <b>${lk.mxB.toFixed(2)}</b> with no mask.`
            + (cfg.mask === 'sliding' || cfg.mask === 'prefix' ? ` The live mask (${maskName(cfg)}) moves them by <b>${lk.mxL.toFixed(2)}</b>.` : '');
        }
      }
      eqMain.style.fontSize = '22px';
      setHTML(eqMain, main);
      eqMain.hidden = !main;
      setHTML(eqLive, liveEq);
      eqLive.hidden = !liveEq;
      eqCheck.replaceChildren(...(check ? [check] : []));
      eqCheck.hidden = !check;
      header.setSub(sub);
      cap.set(words);
    }

    function setStep(i) {
      if (!autoDriving) stopAuto();            // a step the user picks (story dot, arrow or key) ends the autoplay
      step = i;
      if (i === 0) startBuild(); else stopBuild();
      render();
    }

    // ------------------------------------------------ mount
    // The node's head becomes the app's head selection, so the story, maths and code drawers agree.
    if (ctx.state.sel.head !== hd) ctx.store.select({ head: hd });
    build();
    if (step === 0) startBuild();
    render();

    const offTrace = ctx.on('trace', () => {
      const rebuilt = ensure();
      if (rebuilt) { if (step === 0) startBuild(); } else if (buildRaf) stopBuild();
      render({ morph: !rebuilt });
    });
    const offSel = ctx.on('select', () => render());

    ctx.setSteps(STEPS, setStep);
    appear([header.el, picker.el, eq], { stagger: MOTION.stagger });

    // auto-advance the first time the scene is shown, so it explains itself
    const autoNext = () => {
      auto = null;
      if (step >= STEPS.length - 1) return;
      autoDriving = true;
      try { ctx.setStep(step + 1); } finally { autoDriving = false; }
      auto = setTimeout(autoNext, MOTION.step);
    };
    auto = setTimeout(autoNext, MOTION.step);
    el.addEventListener('pointerdown', stopAuto, { once: true });

    return {
      destroy() {
        alive = false;
        leakJob = null;
        stopAuto();
        cancelMorph();
        stopBuild();
        offTrace();
        offSel();
        for (const k of kids.splice(0)) { try { k.off?.(); } catch { /* ignore */ } }
        st.destroy();
      },
    };
  },
};

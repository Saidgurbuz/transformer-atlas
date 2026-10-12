// Generation and the KV cache, on the real model.
// The prompt is read once into a KV cache. Each step takes the last position's scores, chooses a
// token (greedy, or a seeded draw at the chosen temperature), feeds only that token through the model
// with engine.decodeStep, and appends one K row and one V row per layer. Every cached step is checked
// against a full engine.infer over the same prefix. The checks run one at a time between frames, so no
// step waits for one. Masks the engine cannot cache (none, prefix) are recomputed with engine.infer, and
// the scene says why. Generated text stays in this scene unless the user asks to use it as the input.
// The store's trace is the prompt's forward pass, so trace links (hover, right-click) cover prompt
// positions only; generated positions show their maths without numbers.
import { stage2d, sceneHeader, sceneCaption, checkBadge, chip, fly, pulse, appear, wait, useStyle, mat, MOTION, ZONES } from '../kit.js';
import { h, s, fmt, fmtBytes, texHTML, num, clamp, sequential, reducedMotion, slider, button } from '../../ui/index.js';

const STEPS = ['Prompt', 'One step', 'Keep going', 'The cache'];
const SEED = 7;           // sampling seed: the same input and settings always give the same text
const MAX_NEW = 16;       // tokens per run
const VIS_TILES = 64;     // sequence tiles shown; a longer context elides its start
const CELL = 4;           // px per cache cell (square)
const MAX_ROWS = 28;      // cache rows per stack; the newest rows sit at the bottom
const KV_BUDGET = 350;    // px for all K and V stacks together, so the side column ends above the caption
const SEQ = { x: ZONES.hero.x, y: 302, w: 1030 };
const MODEL = { x: ZONES.hero.x, y: 440, w: 290, h: 160 };
const CANDS = { x: 470, y: 440 };
const BUS_Y = 404;
const EQ = { x: ZONES.hero.x, y: 640 };
const SIDE = { x: ZONES.side.x, y: ZONES.hero.y, w: ZONES.side.w };
const PACE = 180, FLY_MS = 620;   // pause between generated tokens; flight of the new token
const TOL = 1e-9;         // the parity tolerance of the engine tests

let uidSeq = 0;

const sci = (v) => (v === 0 ? '0' : v.toExponential(1));
const pct = (p) => { const v = p * 100; return v >= 99.95 ? '>99.9%' : `${v.toFixed(v < 10 || v >= 99 ? 1 : 0)}%`; };
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const maxAbs = (a, b, off) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[off + i])); return m; };
const htmlEsc = (t) => t.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const TEX_ESC = { '\\': '\\textbackslash ', '{': '\\{', '}': '\\}', '$': '\\$', '&': '\\&', '#': '\\#', '^': '\\textasciicircum ', '_': '\\_', '%': '\\%', '~': '\\textasciitilde ' };
const texEsc = (t) => t.replace(/[\\{}$&#^_%~]/g, (c) => TEX_ESC[c]);
const yieldNow = () => new Promise((r) => setTimeout(r, 0));

// Why a mask cannot use the cache. Prefix tokens see each other in both directions, so the prompt
// cannot be built one token at a time; the other masks change earlier rows when a token arrives.
function noCacheWhy(cfg) {
  return cfg.mask === 'prefix'
    ? 'Prefix tokens attend to each other in both directions, so the prompt cannot be cached one token at a time.'
    : 'Every token sees every other token, so from layer 1 on, a new token changes the keys and values of the earlier positions, and caching them would go stale.';
}
const noCacheNote = (cfg) => `No KV cache for mask “${cfg.mask}”. ${noCacheWhy(cfg)} Every step runs the whole sequence through the model.`;

// k and v of one position in layer l, written from the layer input x in the model's own order:
// norm (pre-LN only), projection, bias (when the model has one), RoPE on k only.
function kvLaw(cfg, t) {
  const hx = cfg.norm_placement === 'pre' ? `${cfg.norm === 'rmsnorm' ? '\\RMSNorm' : '\\LN'}\\big(x^{(\\ell)}_{${t}}\\big)` : `x^{(\\ell)}_{${t}}`;
  const k = `${hx}\\,\\Wk^{\\top}${cfg.bias ? '+\\,b_K' : ''}`;
  return {
    k: cfg.pos_encoding === 'rope' ? `\\operatorname{RoPE}_{${t}}\\big(${k}\\big)` : k,
    v: `${hx}\\,\\Wv^{\\top}${cfg.bias ? '+\\,b_V' : ''}`,
  };
}

// The attention of token t over the keys it may see, for this config. It follows reference/model.py: scores,
// then 1/√d_h when attn_scale, then the ALiBi bias, then the mask (causal j ≤ t, sliding t − w < j ≤ t).
function attnLaw(cfg) {
  const range = cfg.mask === 'sliding' ? `t-${cfg.window}<j\\le t` : '\\le t';
  const Kr = `\\K_{${range}}`, Vr = `\\V_{${range}}`;
  const rope = cfg.pos_encoding === 'rope';
  let s = `${rope ? '\\operatorname{RoPE}_{t}\\big(q_{t}\\big)' : 'q_{t}'}\\,${rope ? `\\operatorname{RoPE}\\big(${Kr}\\big)` : Kr}^{\\top}`;
  if (cfg.attn_scale) s = `\\tfrac{${s}}{\\sqrt{d_h}}`;
  if (cfg.pos_encoding === 'alibi') s += '-\\,m_h\\,(t-j)';
  return `\\Aw_{t}=\\softmax\\!\\Big(${s}\\Big),\\qquad o_{t}=\\Aw_{t}\\,${Vr}`;
}

// Why a cached step equals a full pass, in words for this mask.
function cacheCaption(cfg) {
  const reads = cfg.mask === 'sliding'
    ? `Token t reads the last ${cfg.window} rows. The older rows stay cached and the mask hides them.`
    : 'Token t reads rows 0 to t.';
  return `The cache is exact: row j of K and V depends only on tokens 0 to j, so a later token never changes a stored row. ${reads}`;
}

// The context line under the title. A no-cache mask says why every step runs the whole sequence.
function subFor(cfg, kvOn) {
  const first = 'The model predicts one token, appends it, and predicts again.';
  if (kvOn) return `${first} The KV cache makes each step cost one position, not the whole sequence.`;
  const why = cfg.mask === 'prefix' ? 'prefix tokens attend to each other both ways' : 'a new token would change the earlier keys and values';
  return `${first} Mask “${cfg.mask}” has no KV cache: ${why}, so every step runs the whole sequence.`;
}

useStyle('generation', `
.gn-scene .gn-label { font: 600 15px/1.2 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.gn-controls { display: flex; align-items: center; gap: 10px; }
.gn-controls .btn { height: 36px; padding: 0 14px; }
.gn-controls .field { min-width: 240px; }
.gn-seq { display: flex; align-items: flex-start; }
.gn-tile { position: relative; flex: 0 0 auto; box-sizing: border-box; border-radius: 10px; display: grid; place-items: center; background: var(--surface); border: 1px solid var(--line); box-shadow: 0 3px 8px rgba(0, 0, 0, .08); font: 600 22px/1 var(--font-mono); color: var(--fg); transition: background-color .45s ease, border-color .45s ease, box-shadow .45s ease; }
.gn-tile.gn-gen { background: color-mix(in srgb, var(--c-embed) 20%, var(--surface)); border-color: color-mix(in srgb, var(--c-embed) 60%, var(--line)); }
.gn-tile.gn-empty { background: transparent; border: 1px dashed var(--line-strong); box-shadow: none; color: transparent; }
.gn-tile.gn-new { border: 2px solid var(--accent); box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 22%, transparent), 0 8px 16px rgba(0, 0, 0, .14); }
.gn-tile.gn-ell { background: none; border: 0; box-shadow: none; color: var(--fg-3); font-size: 20px; }
.gn-idx { position: absolute; left: -8px; right: -8px; text-align: center; font: 600 12px/1 var(--font-mono); color: var(--fg-3); opacity: 0; transition: opacity .3s; pointer-events: none; }
.gn-idx.gn-on { opacity: 1; }
.gn-model { box-sizing: border-box; width: 290px; height: 160px; padding: 18px 20px; border-radius: 16px; display: grid; align-content: center; gap: 6px; background: color-mix(in srgb, var(--c-resid) 9%, var(--surface)); border: 1px solid color-mix(in srgb, var(--c-resid) 50%, var(--line)); box-shadow: 0 16px 34px rgba(0, 0, 0, .14), 0 2px 0 color-mix(in srgb, var(--c-resid) 26%, transparent); }
.gn-model b { font: 650 27px/1.1 var(--font-display); color: var(--fg); letter-spacing: -.01em; }
.gn-model small { font: 500 14px/1.45 var(--font-mono); color: var(--fg-2); }
.gn-cands { width: 670px; display: grid; gap: 5px; }
.gn-cands[hidden] { display: none; }
.gn-cand { display: grid; grid-template-columns: 30px 1fr 56px 84px; align-items: center; gap: 12px; height: 26px; box-sizing: border-box; padding: 0 8px; border-radius: 8px; border: 1px solid transparent; transition: background-color .4s ease, border-color .4s ease; }
.gn-cand.gn-chosen { background: var(--surface-2); border-color: var(--fg-3); }
.gn-cand-ch { display: grid; place-items: center; width: 28px; height: 22px; border-radius: 7px; background: var(--surface); border: 1px solid var(--line); font: 600 15px/1 var(--font-mono); color: var(--fg); }
.gn-cand-track { height: 10px; border-radius: 5px; background: var(--surface-2); overflow: hidden; }
.gn-cand-track i { display: block; width: 0; height: 100%; border-radius: 5px; transition: width .6s cubic-bezier(.65, 0, .35, 1), background-color .4s ease; }
.gn-cand-p { font: 600 14px/1 var(--font-mono); color: var(--fg); text-align: right; }
.gn-cand-tag { font: 600 13px/1 var(--font-mono); color: var(--fg-2); text-transform: uppercase; letter-spacing: .06em; white-space: nowrap; }
.gn-bus.gn-run { stroke-dasharray: 7 7; animation: gn-flow .7s linear infinite; }
@keyframes gn-flow { to { stroke-dashoffset: -14; } }
.gn-pop { display: grid; place-items: center; box-sizing: border-box; width: 40px; height: 40px; border-radius: 10px; font: 650 22px/1 var(--font-mono); color: var(--fg); background: color-mix(in srgb, var(--c-embed) 30%, var(--surface)); border: 2px solid var(--c-embed); box-shadow: 0 12px 24px rgba(0, 0, 0, .2); }
.gn-eq { display: grid; gap: 4px; font-size: 24px; color: var(--fg); }
.gn-eq .katex-display { margin: 0; text-align: left; }
.gn-eq .katex-display > .katex { text-align: left; }
.gn-eq .gn-live { font-size: 20px; color: var(--fg-2); }
.gn-checks { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; min-height: 28px; }
.gn-side { display: grid; gap: 12px; align-content: start; }
.gn-side-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
.gn-side-title { font: 600 17px/1.2 var(--font-mono); color: var(--fg); }
.gn-side-note { font: 500 14px/1.2 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.gn-kv { display: grid; grid-template-columns: auto auto; gap: 14px 20px; justify-content: start; }
.gn-block { display: grid; gap: 6px; justify-items: start; }
.gn-block-l { font: 600 15px/1.2 var(--font-mono); color: var(--fg-2); }
.gn-pair { display: flex; gap: 4px; }
.gn-col { display: grid; gap: 4px; justify-items: start; }
.gn-col.k { --c: var(--c-k); }
.gn-col.v { --c: var(--c-v); }
.gn-cap { font: 600 15px/1.2 var(--font-mono); color: var(--c); }
.gn-stack-wrap { position: relative; }
.gn-scene .gn-stack-wrap > .k-mat { display: flex; width: max-content; }
.gn-scene .k-mat-body { padding: 3px; border-radius: 6px; background: var(--surface-2); box-shadow: 0 6px 14px rgba(0, 0, 0, .08); }
.gn-scene .k-grid { gap: 0; }
.gn-scene .k-cell { border-radius: 1px; box-shadow: none; }
.gn-scene .k-cell.gn-future { background: color-mix(in srgb, var(--fg-3) 14%, transparent) !important; }
.gn-scene .k-cell.ell { background: none !important; }
.gn-scene .k-row-band { border-radius: 2px; }
.gn-scene .gn-stack-k .k-row-band { border-color: var(--c-k); box-shadow: 0 0 8px var(--c-k); }
.gn-scene .gn-stack-v .k-row-band { border-color: var(--c-v); box-shadow: 0 0 8px var(--c-v); }
.gn-ruler { position: absolute; left: -30px; top: 4px; width: 24px; display: grid; grid-auto-rows: var(--rh); text-align: right; font: 500 12px/var(--rh) var(--font-mono); color: var(--fg-3); pointer-events: none; }
.gn-ruler span { display: block; height: var(--rh); white-space: nowrap; }
.gn-ruler .gn-now { color: var(--fg-2); font-weight: 650; }
.gn-counts { display: grid; gap: 6px; }
.gn-count { display: grid; grid-template-columns: 104px 1fr 56px; align-items: center; gap: 10px; font: 500 15px/1 var(--font-mono); color: var(--fg-2); }
.gn-count > b { font: 650 16px/1 var(--font-mono); color: var(--fg); text-align: right; }
.gn-track { height: 8px; border-radius: 4px; background: var(--surface-2); overflow: hidden; }
.gn-track i { display: block; width: 0; height: 100%; border-radius: 4px; background: var(--fg-2); transition: width .6s cubic-bezier(.65, 0, .35, 1); }
.gn-count.without .gn-track i { background: var(--fg-3); }
.gn-bytes { font: 500 14px/1.4 var(--font-mono); color: var(--fg-3); }
.gn-muted { font-size: 15px; line-height: 1.4; color: var(--fg-3); }
.gn-note-card { padding: 14px 16px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); font-size: 16px; line-height: 1.5; color: var(--fg-2); }
@media (prefers-reduced-motion: reduce) { .gn-scene * { transition: none !important; animation: none !important; } }
`);

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const root = st.root;
    root.classList.add('gn-scene');
    root.dataset.step = '0';
    const E = ctx.engine;
    const uid = `gn${++uidSeq}`;

    // ------------------------------------------------------------ state
    let base = null;          // readBase(): the model and the prompt for the current input
    let seq = [];             // prompt + generated ids
    let g = 0;                // generated tokens so far
    let cache = null;         // engine KV cache (causal and sliding masks only)
    let next = null;          // the upcoming choice: {id, probs, logits}
    let rng = null;           // seeded uniform source, used only when sampling
    let chk = null;           // checks against a full pass: queued, done, largest differences so far
    const checkQueue = [];    // pending checks: {ep, ids, row}
    let pumping = false;
    let epoch = 0;            // bumps on every reset, so a stale check or animation does not paint
    let temp = 0;             // 0 = greedy
    let step = 0, running = false, runId = 0, buildId = 0, dead = false;
    let autoOn = true, autoCalling = false;
    let work = Promise.resolve();
    const waiters = [];       // settle() callers, woken when a prompt feed ends or is superseded
    let tiles = [], layers = [], vis = null, rowsShown = 0, firstRow = 0, cnt = {};

    // ------------------------------------------------------------ static parts
    const header = sceneHeader(st, {
      chips: [], title: 'Generation & KV cache',
      sub: 'The model predicts one token, appends it, and predicts again. The KV cache makes each step cost one position, not the whole sequence.',
    });
    const eyebrow = header.el.querySelector('.k-eyebrow');

    const playBtn = button('Play', () => onPlay(), { variant: 'primary', title: 'Generate up to 16 tokens' });
    const resetBtn = button('Reset', () => ctx.setStep(0), { title: 'Back to the original input' });
    const useBtn = button('Use as input', () => useAsInput(), { title: 'Make the generated text the new input' });
    const tempCtl = slider({
      id: `${uid}-temp`, label: 'temperature', min: 0, max: 1.5, step: 0.05, value: 0,
      format: (v) => (v === 0 ? 'greedy' : `T = ${v.toFixed(2)}`), onInput: () => cancelAuto(),
    });
    tempCtl.input.addEventListener('change', () => applyTemp());
    const controls = h('div', { class: 'gn-controls' }, playBtn, resetBtn, tempCtl.el, useBtn);
    st.add(controls, ZONES.controls);

    const seqLabel = h('div', { class: 'gn-label' });
    st.add(seqLabel, { x: ZONES.hero.x, y: ZONES.hero.y });
    const seqRow = h('div', { class: 'gn-seq' });
    st.add(seqRow, { x: SEQ.x, y: SEQ.y });

    const markId = `${uid}-head`;
    const bus = s('path', { class: 'gn-bus', fill: 'none', stroke: 'var(--c-embed)', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'marker-end': `url(#${markId})` });
    st.defs.append(s('marker', { id: markId, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' },
      s('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'var(--c-embed)' })));
    st.svg.append(bus);

    const modelEl = h('div', { class: 'gn-model' });
    st.add(modelEl, { x: MODEL.x, y: MODEL.y });

    const candLabel = h('div', { class: 'gn-label', style: { marginBottom: '8px' } });
    const candBox = h('div', { class: 'gn-cands' });
    const cands = Array.from({ length: 5 }, () => {
      const ch = h('span', { class: 'gn-cand-ch' });
      const fill = h('i');
      const pctEl = h('span', { class: 'gn-cand-p' });
      const tag = h('span', { class: 'gn-cand-tag' });
      const row = h('div', { class: 'gn-cand' }, ch, h('span', { class: 'gn-cand-track' }, fill), pctEl, tag);
      candBox.append(row);
      return { row, ch, fill, pctEl, tag };
    });
    const candWrap = h('div', null, candLabel, candBox);
    st.add(candWrap, { x: CANDS.x, y: CANDS.y });

    const eqMain = h('div'), eqLive = h('div', { class: 'gn-live' }), eqChecks = h('div', { class: 'gn-checks' });
    const eqEl = h('div', { class: 'gn-eq' }, eqMain, eqLive, eqChecks);
    st.add(eqEl, { x: EQ.x, y: EQ.y });
    const cap = sceneCaption(st, ZONES.caption);

    const sideEl = h('div', { class: 'gn-side', style: { width: `${SIDE.w}px` } });
    st.add(sideEl, { x: SIDE.x, y: SIDE.y });

    // ------------------------------------------------------------ model -> geometry
    function readBase() {
      const s0 = ctx.state, cfg = s0.model.config, P = s0.ids.length;
      const maxNew = Math.max(0, Math.min(MAX_NEW, cfg.block_size - P));
      return {
        model: s0.model, cfg, tk: s0.tokenizer, ids: s0.ids.slice(), P, maxNew, S: P + maxNew,
        V: cfg.vocab_size, dh: E.dHead(cfg), kvOn: E.kvCacheSupported(cfg), rope: cfg.pos_encoding === 'rope',
      };
    }

    function glyph(id) { const c = base.tk.token(id); return c === ' ' ? '␣' : c === '\n' ? '⏎' : c; }

    function layoutVis() {
      const S = base.S, hidden = S > VIS_TILES ? S - (VIS_TILES - 1) : 0;
      const n = hidden ? VIS_TILES : S;
      const gap = n > 40 ? 2 : 4;
      const w = clamp(Math.floor((SEQ.w - (n - 1) * gap) / n), 10, 34);
      const th = clamp(w + 20, 30, 46);          // tiles stay near square when they are narrow
      return { hidden, n, gap, w, th, labelEvery: w >= 20 ? 1 : w >= 14 ? 4 : 8 };
    }
    const tileX = (i) => SEQ.x + i * (vis.w + vis.gap) + vis.w / 2;
    const tileIndex = (pos) => (pos < vis.hidden ? -1 : vis.hidden ? pos - vis.hidden + 1 : pos);
    const withoutRows = (L) => { let n = base.P; for (let k = base.P; k < L; k++) n += k + 1; return n; };

    // ------------------------------------------------------------ DOM that depends on the input
    function buildDom() {
      vis = layoutVis();
      seqRow.style.gap = `${vis.gap}px`;
      tiles = Array.from({ length: vis.n }, (_, i) => {
        const pos = vis.hidden ? (i === 0 ? -1 : vis.hidden + i - 1) : i;
        const ch = h('span', { class: 'gn-ch', style: { fontSize: `${Math.min(22, vis.w - 2)}px` } });
        const idx = pos >= 0 ? h('span', { class: 'gn-idx', style: { top: `${vis.th + 6}px` } }, String(pos)) : null;
        const el = h('div', { class: 'gn-tile', style: { width: `${vis.w}px`, height: `${vis.th}px` }, title: pos >= 0 ? `position ${pos}` : `positions 0 to ${vis.hidden - 1} are not shown` }, ch, idx);
        if (pos < 0) { el.classList.add('gn-ell'); ch.textContent = '⋯'; }
        return { el, ch, idx, pos };
      });
      seqRow.replaceChildren(...tiles.map((t) => t.el));
      const { cfg } = base;
      modelEl.replaceChildren(
        h('b', null, 'Transformer'),
        h('small', null, `${cfg.n_layer} ${cfg.n_layer === 1 ? 'layer' : 'layers'} · ${cfg.n_head} ${cfg.n_head === 1 ? 'head' : 'heads'} · d ${cfg.d_model}`),
        h('small', null, base.kvOn ? 'reads only the new token' : 'reads the whole sequence'),
        h('small', null, `${base.V} logits per position`));
      buildSide();
    }

    function buildSide() {
      const { cfg, kvOn, S } = base;
      layers = [];
      cnt = { with: h('b'), without: h('b'), barWith: h('i'), barWithout: h('i'), bytes: h('div', { class: 'gn-bytes' }) };
      let body;
      if (kvOn) {
        const geo = stackGeom();
        rowsShown = Math.min(S, geo.rows);
        firstRow = S - rowsShown;
        body = buildStacks(geo);
      } else {
        rowsShown = 0;
        firstRow = S;
        body = h('div', { class: 'gn-note-card' }, noCacheNote(cfg));
      }
      sideEl.replaceChildren(
        h('div', { class: 'gn-side-head' },
          h('span', { class: 'gn-side-title' }, kvOn ? 'KV cache' : 'no KV cache'),
          h('span', { class: 'gn-side-note' }, kvOn ? `head 0 · rows ${firstRow}–${S - 1} of ${S}` : `mask “${cfg.mask}”`)),
        body,
        h('div', { class: 'gn-counts' },
          h('div', { class: 'gn-count' }, h('span', null, 'with cache'), h('span', { class: 'gn-track' }, cnt.barWith), cnt.with),
          h('div', { class: 'gn-count without' }, h('span', null, 'without'), h('span', { class: 'gn-track' }, cnt.barWithout), cnt.without)),
        cnt.bytes);
    }

    // Shape of one K or V stack: kv head 0, one row per position, the first 12 and last 4 dims for wide heads
    // (the kit's convention). Rows are cut to what the side column can hold; the newest rows are kept.
    function stackGeom() {
      const { cfg, dh } = base;
      const dims = dh <= 17 ? Array.from({ length: dh }, (_, d) => d)
        : [...Array.from({ length: 12 }, (_, d) => d), -1, ...Array.from({ length: 4 }, (_, k) => dh - 4 + k)];
      const cell = clamp(Math.floor(64 / dims.length), 2, CELL);
      const blockRows = Math.ceil(cfg.n_layer / 2);
      const perStack = (KV_BUDGET - 12 * (blockRows - 1)) / blockRows - 46;   // minus the layer and K/V labels
      const rows = clamp(Math.floor((perStack - 8) / cell), 6, MAX_ROWS);     // minus the stack's frame
      return { dims, cell, rows };
    }

    function buildStacks({ dims, cell }) {
      const { cfg, P, rope } = base;
      const cols = dims.length, rows = rowsShown;
      const grid = h('div', { class: 'gn-kv' });
      for (let l = 0; l < cfg.n_layer; l++) {
        const blk = {};
        const pair = h('div', { class: 'gn-pair' });
        for (const kind of ['k', 'v']) {
          const trace = `L${l}.attn.${kind === 'k' ? (rope ? 'k_rot' : 'k_heads') : 'v_heads'}`;
          const m = mat(new Float64Array(rows * cols), rows, cols, {
            cell, gap: 0, mode: 'diverging', range: 1, kind, trace,
            cellOf: (r, c) => `${firstRow + r},${dims[c]},0`,
          });
          m.classList.add(kind === 'k' ? 'gn-stack-k' : 'gn-stack-v');
          // Every cell starts as a placeholder; paintStacks fills the rows the cache holds. Elided columns and generated
          // positions have no trace value (the store holds the prompt only), so they get no hover card.
          for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
              const cellEl = m.cellEl(r, c);
              cellEl.classList.add('gn-future');
              if (dims[c] < 0) cellEl.classList.add('ell');
              if (dims[c] < 0 || firstRow + r >= P) cellEl.removeAttribute('data-cell');
            }
          }
          const wrap = h('div', { class: 'gn-stack-wrap' }, m);
          let ruler = null;
          if (kind === 'k' && l % 2 === 0) {          // the first column of each block row carries the position ruler
            ruler = Array.from({ length: rows }, () => h('span'));
            wrap.append(h('div', { class: 'gn-ruler', style: { '--rh': `${cell}px` } }, ruler));
          }
          blk[kind] = { m, dims, ruler, cell };
          pair.append(h('div', { class: ['gn-col', kind] }, h('span', { class: 'gn-cap' }, kind === 'k' ? 'K' : 'V'), wrap));
        }
        layers.push(blk);
        grid.append(h('div', { class: 'gn-block' }, h('span', { class: 'gn-block-l' }, `layer ${l}`), pair));
      }
      return grid;
    }

    // ------------------------------------------------------------ choosing and running the model
    function choose(row) {
      const probs = E.samplingDistribution(row, { temperature: temp > 0 ? temp : 1 });
      const id = temp > 0 ? E.sampleFrom(probs, () => rng()) : E.argmax(row);
      return { id, probs, logits: row };
    }

    // max over layers, kv heads, rows and dims of |cached K/V - the full pass's K/V| for the first L positions
    function kvDiff(trace, L) {
      const { cfg, dh, rope } = base;
      let m = 0;
      for (let l = 0; l < cfg.n_layer; l++) {
        const K = trace[`L${l}.attn.${rope ? 'k_rot' : 'k_heads'}`];
        const Vh = trace[`L${l}.attn.v_heads`];
        for (let kh = 0; kh < cfg.n_kv_head; kh++) {
          for (let t = 0; t < L; t++) {
            const kr = cache.layers[l].k[kh][t], vr = cache.layers[l].v[kh][t];
            for (let d = 0; d < dh; d++) {
              const ix = (kh * L + t) * dh + d;
              m = Math.max(m, Math.abs(kr[d] - K.data[ix]), Math.abs(vr[d] - Vh.data[ix]));
            }
          }
        }
      }
      return m;
    }

    // A cached step is checked against a full pass over the same prefix. Rows already in the cache never
    // change, so the check can run later. Checks run one per macrotask: no step waits for an engine.infer.
    function queueCheck(L, row) {
      chk.queued++;
      checkQueue.push({ ep: epoch, ids: seq.slice(0, L), row });
      if (!pumping) pumpChecks();
    }
    function pumpChecks() {
      pumping = true;
      const runOne = () => {
        const job = checkQueue.shift();
        if (dead || !job) { pumping = false; return; }
        if (job.ep !== epoch) { setTimeout(runOne, 0); return; }        // the input changed since: skip it
        const L = job.ids.length, V = base.V;
        const ref = E.infer(base.model, job.ids);
        chk.done++;
        chk.maxLogits = Math.max(chk.maxLogits, maxAbs(job.row, ref.logits.data, (L - 1) * V));
        chk.maxKV = Math.max(chk.maxKV, kvDiff(ref.trace, L));
        paintChecks();
        setTimeout(runOne, 0);
      };
      setTimeout(runOne, 0);
    }

    // A fresh state for the current input: the prompt is the sequence, nothing generated yet. There is no choice
    // until the prompt has been read, so the state is not consistent until then (see consistent()).
    function beginState() {
      epoch++;
      checkQueue.length = 0;
      seq = base.ids.slice();
      g = 0;
      next = null;
      rng = E.mulberry32(SEED);
      chk = { queued: 0, done: 0, maxLogits: 0, maxKV: 0 };
    }

    // Consistent: the cache holds every position of the sequence, and `next` is the choice after it. Only then may
    // a step decode or a frame paint. A prompt feed leaves the state inconsistent until it ends.
    const consistent = () => !!base && !!next && (!base.kvOn || (!!cache && cache.length === seq.length));
    // Resolves once the state is consistent (or the scene is gone). Steps and Play wait for a prompt feed this way.
    function settle() {
      if (dead || consistent()) return Promise.resolve();
      return new Promise((r) => waiters.push(r)).then(settle);
    }
    function wakeWaiters() { waiters.splice(0).forEach((r) => r()); }

    // One generated token joins the sequence. With a cache this is one decodeStep (cheap); without one,
    // the whole sequence is run again, which is what the scene is showing.
    function advance() {
      if (!consistent()) return null;               // never decode against a half-built cache
      const pos = seq.length, id = next.id;
      seq.push(id);
      g++;
      const L = seq.length, V = base.V;
      let row;
      if (base.kvOn) {
        row = E.decodeStep(base.model, cache, id).logits;     // only this position is computed
        queueCheck(L, row);
      } else {
        const ref = E.infer(base.model, seq);                  // no cache: the whole sequence again
        row = ref.logits.data.slice((L - 1) * V, L * V);
      }
      next = choose(row);
      return { pos, id };
    }

    // Jump to n generated tokens with no animation (story steps, temperature and input changes).
    // The prompt is fed through the cache a few tokens per frame, and generated tokens yield every few steps
    // (without a cache every token is a full pass), so a jump never holds the page for long.
    // A newer rebuild ends an unfinished feed (buildId). A run or a stop ends only the generation (runId), so a feed
    // always finishes and leaves a consistent state for the step that waits for it.
    async function rebuild(n) {
      const b = ++buildId;               // a newer rebuild supersedes this one
      const id = ++runId;                // a newer run, stop or jump ends this one's generation
      running = false;
      beginState();
      if (base.kvOn) {
        cache = E.createCache(base.model);
        let out = null;
        for (let i = 0; i < seq.length; i++) {
          out = E.decodeStep(base.model, cache, seq[i]);
          if (i % 6 === 5) { await yieldNow(); if (b !== buildId || dead) { wakeWaiters(); return; } }
        }
        next = choose(out.logits);
        queueCheck(seq.length, out.logits);
      } else {
        cache = null;
        const ref = E.infer(base.model, seq);
        next = choose(ref.logits.data.slice((seq.length - 1) * base.V, seq.length * base.V));
      }
      wakeWaiters();                     // consistent from here on: a waiting step or Play may advance
      paintAll();
      const target = Math.min(n, base.maxNew);
      let k = 0;
      while (id === runId && g < target && !dead) {
        advance();
        k++;
        if (!base.kvOn || k % 4 === 0) { paintAll(); await yieldNow(); }
      }
      if (id === runId && !dead) paintAll();
    }

    function stop() { runId++; running = false; updateControls(); }

    // Animated run up to `target` tokens. Resolves when the run ends (reached, or stopped).
    function run(target) {
      const id = ++runId;
      running = true;
      updateControls();
      return (async () => {
        await settle();                                  // a prompt still being read into the cache finishes first
        const end = () => Math.min(target, base.maxNew);
        while (id === runId && g < end() && !dead) {
          await animateStep(id);
          if (id !== runId) break;
          if (g < end()) await wait(PACE);
        }
        if (id === runId) { running = false; updateControls(); paintBus(); }
      })();
    }

    // One visible step: the chosen token pops out of the model and flies into its slot while the cache
    // step runs; then the cache row, the distribution and the bus update together.
    async function animateStep(id) {
      if (!base || !consistent() || g >= base.maxNew) return false;
      const e = epoch, pos = seq.length, tokId = next.id;
      const tile = tiles.find((t) => t.pos === pos);
      let flight = null;
      if (tile && !reducedMotion() && !dead && id === runId) {
        pulse(modelEl);
        const pop = h('div', { class: 'gn-pop' }, glyph(tokId));
        st.add(pop, { x: MODEL.x + MODEL.w / 2, y: MODEL.y, anchor: 'c', layer: 'front' });
        flight = fly(st, pop, tile.el, { duration: FLY_MS, arc: 90 });   // fly clones the token, so the original can leave now
        pop.remove();
      }
      advance();
      if (flight) await flight;
      if (e === epoch && !dead) paintAll({ animate: true, fresh: pos });
      return true;
    }

    // ------------------------------------------------------------ controls
    function onPlay() {
      cancelAuto();
      if (running) { stop(); return; }
      if (step < 2) { ctx.setStep(2); return; }        // "Keep going" is what Play does
      if (g < base.maxNew) work = run(base.maxNew);
    }

    function applyTemp() {
      if (!base) return;
      temp = tempCtl.get();
      cancelAuto();
      work = rebuild(g);                                 // the same number of tokens, chosen at the new setting
    }

    function useAsInput() {
      if (!base || g === 0) return;
      cancelAuto();
      stop();
      ctx.store.setText(base.tk.decode(seq), { quiet: true });   // the trace event rebuilds this view
    }

    // Generation stops at the 16-token limit or when the context window is full. Play then says so and stays off.
    const isFull = () => seq.length >= base.cfg.block_size;
    const stopReason = () => (g < base.maxNew ? '' : isFull() ? 'context window full' : `${MAX_NEW}-token limit reached`);
    function updateControls() {
      if (!base) return;
      const done = g >= base.maxNew;
      playBtn.textContent = running ? 'Pause' : done ? (isFull() ? 'Context full' : 'Limit reached') : (g > 0 ? 'Continue' : 'Play');
      playBtn.disabled = !running && done;
      playBtn.title = done && !running ? 'Generation has stopped. Reset starts again from the input.' : `Generate up to ${MAX_NEW} tokens`;
      useBtn.disabled = g === 0 || running;
    }

    function cancelAuto() { autoOn = false; }

    // Story steps: each one sets the state it describes. Going back rewinds; going forward animates.
    function setStep(i) {
      if (dead || !base) return;
      if (!autoCalling) cancelAuto();
      step = clamp(i, 0, STEPS.length - 1);
      root.dataset.step = String(step);
      stop();
      const max = base.maxNew;
      const settled = () => settle().then(() => { if (!dead) paintAll(); });   // a frame, once the state is consistent
      if (step === 0) work = rebuild(0);
      else if (step === 1) {
        const t = Math.min(1, max);
        if (g > t) work = rebuild(t);
        else if (g < t) work = run(t);
        else work = settled();
      } else if (step === 2) {
        work = g < max ? run(max) : settled();
      } else work = g < max ? rebuild(max) : settled();
    }

    // ------------------------------------------------------------ painting
    // A count-up that a newer value cancels: an old tween must never overwrite a settled number.
    const tweenTok = new WeakMap();
    function setNum(elx, to, animate) {
      const from = Number(elx.dataset.v || 0), tok = (tweenTok.get(elx) || 0) + 1;
      tweenTok.set(elx, tok);
      elx.dataset.v = String(to);
      if (!animate || reducedMotion() || from === to) { elx.textContent = fmt(to, 0); return; }
      const t0 = performance.now();
      const frame = (now) => {
        if (dead || tweenTok.get(elx) !== tok) return;
        const p = Math.min(1, (now - t0) / 600), e = 1 - Math.pow(1 - p, 3);
        elx.textContent = fmt(from + (to - from) * e, 0);
        if (p < 1) requestAnimationFrame(frame);
      };
      requestAnimationFrame(frame);
    }

    function paintChrome() {
      const { cfg, P, kvOn } = base;
      eyebrow.replaceChildren(
        chip(`${cfg.n_layer} ${cfg.n_layer === 1 ? 'layer' : 'layers'}`, 'resid'),
        chip(temp > 0 ? `T = ${temp.toFixed(2)}` : 'greedy', 'head'),
        chip(kvOn ? 'KV cache' : 'no cache', kvOn ? 'k' : 'mask'));
      seqLabel.textContent = `tokens · prompt ${P}${g ? ` · generated ${g}` : ''} · context ${seq.length} of ${cfg.block_size}${stopReason() ? ' · stopped' : ''}`;
    }

    function paintSeq(fresh) {
      const L = seq.length;
      for (const t of tiles) {
        if (t.pos < 0) continue;
        const has = t.pos < L;
        t.el.classList.toggle('gn-gen', has && t.pos >= base.P);
        t.el.classList.toggle('gn-empty', !has);
        t.el.classList.toggle('gn-new', has && g > 0 && t.pos === L - 1);
        t.ch.textContent = has ? glyph(seq[t.pos]) : '';
        t.idx.classList.toggle('gn-on', t.pos % vis.labelEvery === 0 || t.pos === L - 1);
        if (t.pos === fresh && !reducedMotion()) {
          t.el.animate([{ transform: 'scale(.4)', opacity: 0 }, { transform: 'scale(1)', opacity: 1 }], { duration: 460, easing: MOTION.easeOut });
        }
      }
    }

    function paintBus() {
      const i = tileIndex(seq.length - 1);
      if (i < 0) { bus.setAttribute('d', ''); return; }
      const y0 = SEQ.y + vis.th + 24, xBox = MODEL.x + MODEL.w / 2;
      if (base.kvOn) {
        bus.setAttribute('d', `M${tileX(i)},${y0} V${BUS_Y} H${xBox} V${MODEL.y - 4}`);
      } else {
        // no cache: every visible token is read again, so a bracket spans the sequence and feeds the box
        const first = tileX(vis.hidden ? 1 : 0), last = tileX(i), mid = (first + last) / 2;
        bus.setAttribute('d', `M${first},${y0} V${BUS_Y - 12} H${last} V${y0} M${mid},${BUS_Y - 12} V${BUS_Y} H${xBox} V${MODEL.y - 4}`);
      }
      bus.classList.toggle('gn-run', running);
    }

    // Candidates are the top five next tokens. The store's trace holds the prompt's own distribution at T = 1,
    // so the list is traced only for the first choice at greedy settings, which is exactly that distribution.
    // Once generation has stopped the next token is never appended, so the list gives way to the reason.
    function paintCands() {
      const why = stopReason();
      if (why) {
        candLabel.textContent = `${why}: the next token is not appended`;
        candBox.hidden = true;
        delete candBox.dataset.trace;
        candBox.querySelectorAll('[data-cell]').forEach((r) => r.removeAttribute('data-cell'));
        return;
      }
      candBox.hidden = false;
      const { probs } = next;
      let order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, cands.length);
      if (!order.includes(next.id)) order[order.length - 1] = next.id;   // always show the chosen token
      order = order.sort((a, b) => probs[b] - probs[a]);
      const traced = g === 0 && temp === 0;
      if (traced) candBox.dataset.trace = 'probs'; else delete candBox.dataset.trace;
      cands.forEach((c, i) => {
        const id = order[i];
        c.row.style.visibility = id === undefined ? 'hidden' : '';
        if (id === undefined) { delete c.row.dataset.cell; return; }
        const p = probs[id], chosen = id === next.id;
        c.ch.textContent = glyph(id);
        c.fill.style.width = `${(p * 100).toFixed(2)}%`;
        c.fill.style.background = sequential(p, 'head');
        c.pctEl.textContent = pct(p);
        c.row.classList.toggle('gn-chosen', chosen);
        c.tag.textContent = chosen ? (temp > 0 ? 'sampled' : 'greedy') : '';
        if (traced) c.row.dataset.cell = `${base.P - 1},${id},0`; else delete c.row.dataset.cell;
      });
      candLabel.textContent = temp > 0 ? `next token · T = ${temp.toFixed(2)}` : 'next token · greedy takes the highest score';
    }

    // Each stack: rows are positions, columns are the dims of kv head 0. Rows the cache does not hold yet are placeholders.
    function paintStacks(fresh) {
      if (!base.kvOn || !cache || !layers.length) return;
      const L = seq.length, cols = layers[0].k.dims.length;
      const showNew = g > 0 && L - 1 >= firstRow;
      layers.forEach((blk, l) => {
        for (const kind of ['k', 'v']) {
          const { m, dims, ruler, cell } = blk[kind];
          const data = cache.layers[l][kind][0];        // kv head 0: one row per position
          const have = data.length;                     // rows in the cache: a short cache paints placeholders
          let range = 0;
          for (let t = 0; t < have; t++) for (const x of data[t]) range = Math.max(range, Math.abs(x));
          const vals = new Float64Array(rowsShown * cols);
          for (let i = 0; i < rowsShown; i++) {
            const t = firstRow + i, filled = t < have;
            for (let c = 0; c < cols; c++) {
              if (filled && dims[c] >= 0) vals[i * cols + c] = data[t][dims[c]];
              m.cellEl(i, c).classList.toggle('gn-future', !filled);
            }
          }
          m.update(vals, { range: range || 1 });
          // The band's place follows from the cell size and the frame (3 px padding, 1 px border), so it is set
          // directly: reading layout here would force a new layout for every stack on every step.
          const band = m.querySelector('.k-row-band');
          if (showNew && L - 1 < have && band) {
            const r = L - 1 - firstRow;
            band.style.top = `${2 + r * cell}px`;
            band.style.left = '1px';
            band.style.width = `${cols * cell + 6}px`;
            band.style.height = `${cell + 4}px`;
            band.hidden = false;
            if (fresh >= firstRow && fresh === L - 1 && !reducedMotion()) band.animate([{ opacity: 0.2 }, { opacity: 1 }], { duration: 520, easing: MOTION.easeOut });
          } else if (band) band.hidden = true;
          if (ruler) {
            for (let i = 0; i < rowsShown; i++) {
              const t = firstRow + i, show = t < L && (t % 8 === 0 || t === firstRow || t === L - 1);
              ruler[i].textContent = show ? String(t) : '';
              ruler[i].classList.toggle('gn-now', t === L - 1);
            }
          }
        }
      });
    }

    function paintCounts(animate) {
      const L = seq.length, without = withoutRows(L);
      setNum(cnt.without, without, animate);
      cnt.barWithout.style.width = '100%';
      if (base.kvOn) {
        setNum(cnt.with, L, animate);
        cnt.barWith.style.width = `${(L / Math.max(1, without)) * 100}%`;
        cnt.bytes.textContent = `KV cache: ${fmtBytes(E.cacheBytes(base.cfg, L, 2))} at fp16`;
      } else {
        cnt.with.textContent = 'off';
        cnt.barWith.style.width = '0%';
        cnt.bytes.textContent = 'Nothing is stored between steps.';
      }
    }

    // The checks sit under the equations, where the reader reads the maths.
    function paintChecks() {
      if (!base || !chk) return;
      if (!base.kvOn) {
        eqChecks.replaceChildren(h('span', { class: 'gn-muted' }, 'Nothing to check: every step is a full pass.'));
        return;
      }
      const pending = chk.done < chk.queued;
      const note = pending
        ? `checking against engine.infer: ${chk.done} of ${chk.queued}`
        : `${g ? `the prompt and ${plural(g, 'step')} match` : 'the prompt matches'} engine.infer on the same prefix`;
      eqChecks.replaceChildren(
        checkBadge(chk.maxLogits, 0, { label: `logits max |Δ| = ${sci(chk.maxLogits)}`, tol: TOL }),
        checkBadge(chk.maxKV, 0, { label: `K, V max |Δ| = ${sci(chk.maxKV)}`, tol: TOL }),
        h('span', { class: 'gn-muted' }, note));
    }

    function setEq(main, live) {
      eqMain.innerHTML = texHTML(main, { display: true });
      eqLive.innerHTML = live ? texHTML(live, { display: true }) : '';
    }

    function paintText() {
      const { P, cfg, kvOn, maxNew } = base;
      const L = seq.length, c = next, greedy = temp === 0, without = withoutRows(L);
      const ch = htmlEsc(glyph(c.id)), tch = texEsc(glyph(c.id)), T = temp.toFixed(2);
      if (maxNew === 0) {
        cap.set(`The prompt already fills the ${cfg.block_size}-token window, so there is no room to generate. Shorten the input to see the loop run.`);
        setEq(`\\text{context window full: } ${P} = ${cfg.block_size}`, '');
        return;
      }
      let capHTML = '', main = '', live = '';
      if (step === 0) {
        capHTML = kvOn
          ? `The prompt goes through the model once and fills the cache: one key and one value row per token, in every layer. The last position scores the next token, <b>“${ch}”</b>.`
          : `The prompt goes through the model once. The last position scores the next token, <b>“${ch}”</b>. No cache is kept, so later steps run the whole sequence again.`;
        main = greedy ? `\\mathit{id}_{${P}}=\\arg\\max_{v}\\, z_{${P - 1},v}` : `\\mathit{id}_{${P}}\\sim\\softmax(z_{${P - 1}}/T)`;
        live = `z_{${P - 1},\\,\\text{“${tch}”}}=${num(c.logits[c.id], 2)},\\qquad p(\\text{“${tch}”})=${num(c.probs[c.id], 3)}${greedy ? '' : `\\text{ at } T=${T}`}`;
      } else if (step === 1) {
        const lastCh = htmlEsc(glyph(seq[L - 1])), lastTex = texEsc(glyph(seq[L - 1]));
        const head = greedy
          ? `The highest score wins, so the model writes <b>“${lastCh}”</b>.`
          : `At T = ${T} the model draws <b>“${lastCh}”</b> from its distribution.`;
        capHTML = kvOn
          ? `${head} Only that token is fed in: it adds one row to every K and V stack, and the older rows stay as they are.`
          : `${head} With no cache, all ${L} positions ran again to produce it.`;
        if (kvOn) {
          const law = kvLaw(cfg, L - 1);
          main = `k^{(\\ell)}_{${L - 1}}=${law.k}`;
          live = `v^{(\\ell)}_{${L - 1}}=${law.v}`;
        } else {
          main = `\\mathit{id}_{${L - 1}}=\\text{“${lastTex}”},\\qquad z_{${L - 1}}=\\text{Transformer}(\\mathit{id}_{0},\\dots,\\mathit{id}_{${L - 1}})`;
          live = `\\text{positions run: } ${L}`;
        }
      } else if (step === 2) {
        const why = stopReason(), tail = why ? ` Generation stops: ${why}.` : '';
        capHTML = kvOn
          ? `After ${plural(g, 'token')} the cache has processed each of the ${L} positions once: <b>${L}</b> rows. Without a cache, every step reruns the sequence: <b>${without}</b> positions in all.${tail}`
          : `Every step reruns the whole sequence: <b>${without}</b> positions in all, for ${plural(g, 'token')}. This mask has no cache (see the last step).${tail}`;
        main = kvOn
          ? `\\text{rows}_{\\text{cache}}=${L},\\qquad \\text{rows}_{\\text{no cache}}=${P}+\\textstyle\\sum_{s=${P}}^{${L - 1}}(s+1)=${without}`
          : `\\text{positions run}=${P}+\\textstyle\\sum_{s=${P}}^{${L - 1}}(s+1)=${without}`;
        live = kvOn
          ? `\\text{without the cache: } ${num(without / L, 1)}\\times\\text{ the work with it}`
          : `\\text{the last step alone ran } ${L} \\text{ positions}`;
      } else {
        if (kvOn) {
          capHTML = cacheCaption(cfg);
          main = `x^{(\\ell)}_{j}=f_{\\ell}\\big(x^{(\\ell-1)}_{0},\\dots,x^{(\\ell-1)}_{j}\\big)`;
          live = attnLaw(cfg);
        } else {
          capHTML = `No cache for mask “${cfg.mask}”. ${noCacheWhy(cfg)}`;
          main = cfg.mask === 'prefix'
            ? `x^{(\\ell)}_{j}=f_{\\ell}\\big(x^{(\\ell-1)}_{0},\\dots,x^{(\\ell-1)}_{\\max(j,\\,${Math.min(cfg.prefix_len, L) - 1})}\\big)`
            : `x^{(\\ell)}_{j}=f_{\\ell}\\big(x^{(\\ell-1)}_{0},\\dots,x^{(\\ell-1)}_{${L - 1}}\\big)`;
          live = cfg.mask === 'prefix'
            ? `\\text{rows } j<${cfg.prefix_len} \\text{ see the whole prefix}`
            : '\\text{from layer } 1 \\text{ on, } x_j \\text{ depends on the later tokens}';
        }
      }
      cap.set(capHTML);
      setEq(main, live);
    }

    function paintAll({ animate = false, fresh = -1 } = {}) {
      if (dead || !base || !consistent()) return;      // mid-feed: the finished state paints when the feed ends
      paintChrome();
      paintSeq(animate ? fresh : -1);
      paintBus();
      paintCands();
      paintStacks(animate ? fresh : -1);
      paintCounts(animate);
      paintText();
      paintChecks();
      updateControls();
    }

    // ------------------------------------------------------------ input changes (the store's trace event)
    function setup() {
      base = readBase();
      buildDom();
      header.setSub(subFor(base.cfg, base.kvOn));
      step = 0;
      root.dataset.step = '0';
    }

    setup();
    work = rebuild(0);
    ctx.setSteps(STEPS, setStep);
    appear([header.el, controls, seqRow, modelEl, candWrap, sideEl, eqEl], { stagger: MOTION.stagger });
    el.addEventListener('pointerdown', cancelAuto, { once: true });

    // explain itself once on first view: one token, then the run, then the cache; the reader takes over on any input
    async function autoChain() {
      await wait(MOTION.step);
      for (const i of [1, 2, 3]) {
        if (!autoOn || dead) return;
        autoCalling = true;
        ctx.setStep(i);
        autoCalling = false;
        await work;
        if (!autoOn || dead) return;
        await wait(MOTION.step);
      }
    }
    autoChain();

    const off = ctx.on('trace', () => {
      if (dead) return;
      const s0 = ctx.state;
      const same = base && s0.model === base.model && s0.ids.length === base.P && s0.ids.every((v, i) => v === base.ids[i]);
      if (same) return;
      cancelAuto();
      stop();
      setup();
      ctx.setStep(0);
    });

    return {
      destroy() { dead = true; autoOn = false; runId++; running = false; checkQueue.length = 0; wakeWaiters(); off(); st.destroy(); },
      pause() { cancelAuto(); stop(); },
      resume() {},
    };
  },
};

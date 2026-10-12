// Generation and the KV cache, on the real model.
// The prompt is read once into a KV cache. Each step takes the last position's scores, chooses a token
// (greedy, or a seeded draw at the chosen temperature), feeds only that token through the model with
// engine.decodeStep, and appends one K row and one V row per layer. Every cached step is checked against a
// full forward pass (engine.infer) over the same prefix. The checks run one at a time between frames, so no
// step waits for one. Masks the engine cannot cache (none, prefix) are recomputed, and the scene says why.
// Generated text stays in this scene unless the user asks to use it as the input.
// The store's trace is the prompt's forward pass, so trace links (hover, right-click) cover prompt
// positions only; generated positions show their maths without numbers.
import { stage2d, sceneHeader, sceneCaption, chip, fly, pulse, appear, wait, useStyle, mat, MOTION, ZONES } from '../kit.js';
import { h, s, fmt, texHTML, num, clamp, sequential, reducedMotion, slider, button } from '../../ui/index.js';

const STEPS = ['Prompt', 'One step', 'Keep going', 'The cache'];
const STEP_KIND = ['embed', 'embed', 'embed', 'k'];   // each step's label takes the colour of what the step is about
const SEED = 7;           // sampling seed: the same input and settings always give the same text
const MAX_NEW = 16;       // tokens per run
const TAPE = { h: 40, font: 16, pad: 6, minW: 26, ell: 28, tick: 12 };   // subword tape: tile height, label px, padding, min width, ellipsis width, empty slot width
const CELL = 4;           // px per cache cell (square)
const MAX_ROWS = 40;      // cache rows per stack; the newest rows sit at the bottom (all of them for a short text)
const KV_BUDGET = 350;    // px for all K and V stacks together, so the side column ends above the caption
const SEQ = { x: ZONES.hero.x, y: 310, w: 1030 };
const MODEL = { x: ZONES.hero.x, y: 440, w: 290, h: 160 };
const CANDS = { x: 470, y: 436 };
const CAND_ROWS = 5, CAND_H = 24, CAND_GAP = 4;
const BUS_Y = 404;
const EQ = { x: ZONES.hero.x, y: 626, w: 1030 };
const SIDE = { x: ZONES.side.x, y: ZONES.hero.y, w: ZONES.side.w };
const PACE = 180, FLY_MS = 620;   // pause between generated tokens; flight of the new token
const TOL = 1e-9;         // the parity tolerance of the engine tests
const RULER_EVERY = 8;    // a position label every 8 rows, none within NEAR_ROWS of the newest row
const NEAR_ROWS = 3;

let uidSeq = 0;

const sci = (v) => (v === 0 ? '0' : v.toExponential(1));
const pct = (p) => { const v = p * 100; return v >= 99.95 ? '>99.9%' : `${v.toFixed(v < 10 || v >= 99 ? 1 : 0)}%`; };
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const kib = (b) => (b < 1024 ? `${b} B` : `${(b / 1024).toFixed(b < 10240 ? 1 : 0)} KiB`);
const maxAbs = (a, b, off) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[off + i])); return m; };
const htmlEsc = (t) => t.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const TEX_ESC = { '\\': '\\textbackslash ', '{': '\\{', '}': '\\}', '$': '\\$', '&': '\\&', '#': '\\#', '^': '\\textasciicircum ', '_': '\\_', '%': '\\%', '~': '\\textasciitilde ' };
const texEsc = (t) => t.replace(/[\\{}$&#^_%~]/g, (c) => TEX_ESC[c]);
// A display label as TeX text. The visible space ␣ is drawn by the \vsp macro (KaTeX's fonts have no ␣); a newline
// and ⟨eot⟩ become text KaTeX typesets, as \n and angle brackets.
const texLab = (label) => texEsc(label).replace(/␣/g, '\\vsp{}').replace(/↵/g, '\\textbackslash{}n').replace(/⟨/g, '\\textless{}').replace(/⟩/g, '\\textgreater{}');
// Width of a subword tile: its label at the tape's font (0.64 em per glyph covers monospace faces), plus padding and border.
const labelWidth = (label) => Math.max(TAPE.minW, Math.ceil(Array.from(label).length * TAPE.font * 0.64) + 2 * TAPE.pad + 2);
const vecTex = (row) => `${Array.from(row.slice(0, 4), (x) => num(x, 2)).join(',\\;')},\\;\\dots`;   // first four entries
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
.gn-seqhead { display: flex; align-items: center; justify-content: space-between; gap: 16px; height: 32px; }
.gn-seqctl { display: flex; align-items: center; gap: 8px; }
.gn-seqctl .btn { height: 32px; padding: 0 12px; font-size: 14px; }
.gn-seq { display: flex; align-items: flex-start; }
.gn-tile { position: relative; flex: 0 0 auto; box-sizing: border-box; border-radius: 10px; display: grid; place-items: center; background: var(--surface); border: 1px solid var(--line); box-shadow: 0 3px 8px rgba(0, 0, 0, .08); font: 600 22px/1 var(--font-mono); color: var(--fg); white-space: nowrap; transition: background-color .45s ease, border-color .45s ease, box-shadow .45s ease; }
.gn-tile.gn-gen { background: color-mix(in srgb, var(--c-embed) 20%, var(--surface)); border-color: color-mix(in srgb, var(--c-embed) 60%, var(--line)); }
.gn-tile.gn-empty { background: transparent; border: 1px dashed var(--line-strong); box-shadow: none; color: transparent; }
.gn-tile.gn-new { border: 2px solid var(--accent); box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 22%, transparent), 0 8px 16px rgba(0, 0, 0, .14); }
.gn-tile.gn-ell { background: none; border: 0; box-shadow: none; color: var(--fg-3); font-size: 20px; }
.gn-idx { position: absolute; left: -8px; right: -8px; text-align: center; font: 600 13px/1 var(--font-mono); color: var(--fg-3); opacity: 0; transition: opacity .3s; pointer-events: none; }
.gn-idx.gn-on { opacity: 1; }
.gn-bus.gn-run { stroke-dasharray: 7 7; animation: gn-flow .7s linear infinite; }
@keyframes gn-flow { to { stroke-dashoffset: -14; } }
.gn-model { box-sizing: border-box; width: 290px; height: 160px; padding: 18px 20px; border-radius: 16px; display: grid; align-content: center; gap: 6px; background: color-mix(in srgb, var(--c-resid) 9%, var(--surface)); border: 1px solid color-mix(in srgb, var(--c-resid) 50%, var(--line)); box-shadow: 0 16px 34px rgba(0, 0, 0, .14), 0 2px 0 color-mix(in srgb, var(--c-resid) 26%, transparent); }
.gn-model b { font: 650 27px/1.1 var(--font-display); color: var(--fg); letter-spacing: -.01em; }
.gn-model small { font: 500 14px/1.45 var(--font-mono); color: var(--fg-2); }
.gn-pop { display: grid; place-items: center; box-sizing: border-box; min-width: 40px; width: max-content; height: 40px; padding: 0 6px; white-space: nowrap; border-radius: 10px; font: 650 22px/1 var(--font-mono); color: var(--fg); background: color-mix(in srgb, var(--c-embed) 30%, var(--surface)); border: 2px solid var(--c-embed); box-shadow: 0 12px 24px rgba(0, 0, 0, .2); }
.gn-candwrap { width: 670px; }
.gn-candhead { display: flex; align-items: center; justify-content: space-between; gap: 16px; height: 30px; margin-bottom: 6px; }
.gn-temp { display: grid; grid-template-columns: 170px 120px; align-items: center; gap: 10px; }
.gn-temp > label { display: flex; justify-content: space-between; gap: 8px; font: 600 14px/1 var(--font-mono); color: var(--fg-2); }
.gn-temp .value { font: 500 14px/1 var(--font-mono); color: var(--fg); }
.gn-cands { display: grid; gap: ${CAND_GAP}px; }
.gn-cands[hidden] { display: none; }
.gn-cand { display: grid; grid-template-columns: var(--chip, 30px) 1fr 56px 84px; align-items: center; gap: 12px; height: ${CAND_H}px; box-sizing: border-box; padding: 0 8px; border-radius: 8px; border: 1px solid transparent; transition: background-color .4s ease, border-color .4s ease; }
.gn-cand.gn-chosen { background: var(--surface-2); border-color: var(--fg-3); }
.gn-cand-ch { display: grid; place-items: center; justify-self: start; width: calc(var(--chip, 30px) - 2px); height: 20px; border-radius: 7px; background: var(--surface); border: 1px solid var(--line); font: 600 15px/1 var(--font-mono); color: var(--fg); white-space: nowrap; }
.gn-cand-track { height: 10px; border-radius: 5px; background: var(--surface-2); overflow: hidden; }
.gn-cand-track i { display: block; width: 0; height: 100%; border-radius: 5px; transition: width .6s cubic-bezier(.65, 0, .35, 1), background-color .4s ease; }
.gn-cand-p { font: 600 14px/1 var(--font-mono); color: var(--fg); text-align: right; }
.gn-cand-tag { font: 600 13px/1 var(--font-mono); color: var(--fg-2); text-transform: uppercase; letter-spacing: .06em; white-space: nowrap; }
.gn-stop { box-sizing: border-box; width: 670px; height: ${CAND_ROWS * CAND_H + (CAND_ROWS - 1) * CAND_GAP}px; display: grid; align-content: center; padding: 14px 18px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); font-size: 16px; line-height: 1.5; color: var(--fg-2); }
.gn-stop[hidden] { display: none; }
.gn-eq { display: grid; gap: 6px; width: ${EQ.w}px; font-size: 24px; color: var(--fg); }
.gn-eqhead { display: flex; align-items: center; justify-content: space-between; gap: 16px; min-height: 26px; }
.gn-steplab { text-align: left; margin-bottom: 0; }
.gn-eq-main { overflow-x: auto; }
.gn-eq .katex-display { margin: 0; text-align: left; }
.gn-eq .katex-display > .katex { text-align: left; }
.gn-eq-live { font-size: 20px; padding: 10px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.gn-eq-live[hidden] { display: none; }
.gn-checks { display: flex; align-items: center; justify-content: flex-end; min-width: 0; }
.gn-scene .k-check.wait { color: var(--fg-3); background: var(--surface-2); }
.gn-muted { font-size: 15px; line-height: 1.4; color: var(--fg-3); }
.gn-side { display: grid; gap: 10px; align-content: start; }
.gn-side-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
.gn-side-title { font: 600 17px/1.2 var(--font-mono); color: var(--fg); }
.gn-side-note { font: 500 14px/1.2 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.gn-side-sub { display: grid; gap: 2px; font: 500 13px/1.35 var(--font-mono); color: var(--fg-3); margin-top: -4px; }
.gn-kv { display: grid; grid-template-columns: auto auto; gap: 12px 20px; justify-content: start; }
.gn-block { display: grid; gap: 6px; justify-items: start; }
.gn-block[hidden] { display: none; }
.gn-block-l { font: 600 15px/1.2 var(--font-mono); color: var(--fg-2); }
.gn-more { font: 500 14px/1.2 var(--font-mono); color: var(--fg-3); }
.gn-more[hidden] { display: none; }
.gn-pair { display: flex; gap: 4px; }
.gn-col { display: grid; gap: 4px; justify-items: start; }
.gn-col.k { --c: var(--c-k); }
.gn-col.v { --c: var(--c-v); }
.gn-cap { font: 600 15px/1.2 var(--font-mono); color: var(--c); }
.gn-stack-wrap { position: relative; }
.gn-scene .gn-stack-wrap > .k-mat { display: flex; width: max-content; }
.gn-scene .k-mat-body { padding: 3px; border-radius: 6px; background: var(--surface-2); box-shadow: 0 6px 14px rgba(0, 0, 0, .08); }
.gn-scene .k-grid { gap: 0; }
.gn-scene .k-cell { border-radius: 1px; box-shadow: none; transition: background-color .5s ease, opacity .45s ease; }
.gn-scene .k-cell.gn-future { background: color-mix(in srgb, var(--fg-3) 14%, transparent) !important; }
.gn-scene .k-cell.gn-unread { opacity: .2; }
.gn-scene .k-cell.ell { background: none !important; }
.gn-scene .k-row-band { border-radius: 2px; }
.gn-scene .gn-stack-k .k-row-band { border-color: var(--c-k); background: color-mix(in srgb, var(--c-k) 16%, transparent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--c-k) 26%, transparent), 0 0 8px var(--c-k); }
.gn-scene .gn-stack-v .k-row-band { border-color: var(--c-v); background: color-mix(in srgb, var(--c-v) 16%, transparent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--c-v) 26%, transparent), 0 0 8px var(--c-v); }
.gn-read { position: absolute; left: 3px; border-radius: 3px; border: 1.5px solid var(--c); background: color-mix(in srgb, var(--c) 12%, transparent); pointer-events: none; transition: top .45s cubic-bezier(.65, 0, .35, 1), height .45s cubic-bezier(.65, 0, .35, 1); }
.gn-read[hidden] { display: none; }
.gn-ruler { position: absolute; left: -30px; top: 3px; width: 24px; display: grid; grid-auto-rows: var(--rh); text-align: right; font: 500 13px/var(--rh) var(--font-mono); color: var(--fg-3); pointer-events: none; }
.gn-ruler span { display: block; height: var(--rh); white-space: nowrap; }
.gn-ruler .gn-now { color: var(--fg); font-weight: 650; }
.gn-counts { display: grid; gap: 6px; }
.gn-count { display: grid; grid-template-columns: 96px 1fr 150px; align-items: center; gap: 10px; font: 500 15px/1 var(--font-mono); color: var(--fg-2); }
.gn-count .gn-val { text-align: right; white-space: nowrap; }
.gn-count .gn-val b { font: 650 16px/1 var(--font-mono); color: var(--fg); }
.gn-count .gn-val small { font: 500 13px/1 var(--font-mono); color: var(--fg-3); }
.gn-track { height: 8px; border-radius: 4px; background: var(--surface-2); overflow: hidden; }
.gn-track i { display: block; width: 0; height: 100%; border-radius: 4px; background: var(--fg-2); transition: width .6s cubic-bezier(.65, 0, .35, 1); }
.gn-count.without .gn-track i { background: var(--fg-3); }
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
    let g = 0;                // generated tokens so far (a step in flight counts as soon as it starts)
    let cache = null;         // engine KV cache (causal and sliding masks only)
    let next = null;          // the upcoming choice: {id, probs, logits}
    let rng = null;           // seeded uniform source, used only when sampling
    let chk = null;           // checks against a full pass: jobs done, prefix checked, largest differences, new rows
    const checkQueue = [];    // pending checks: {ep, ids, row}
    let pumping = false;
    let epoch = 0;            // bumps on every reset, so a stale check or animation does not paint
    let temp = 0;             // 0 = greedy
    let step = 0, running = false, runId = 0, buildId = 0, dead = false;
    let autoOn = true, autoCalling = false;
    let work = Promise.resolve();
    const waiters = [];       // settle() callers, woken when a prompt feed ends or is superseded
    let tiles = [], ell = null, layers = [], vis = null, rowsShown = 0, stackRange = 1, cnt = {}, sideRefs = {};
    let shownL = 0, shownG = 0;   // what paintAll last put on screen: cache length and generated count
    let capShown = null;

    // ------------------------------------------------------------ static parts
    const header = sceneHeader(st, {
      chips: [], title: 'Generation & KV cache',
      sub: 'The model predicts one token, appends it, and predicts again. The KV cache makes each step cost one position, not the whole sequence.',
    });
    const eyebrow = header.el.querySelector('.k-eyebrow');

    // Play, Reset and Use as input act on the sequence, so they sit on its header row.
    const playBtn = button('Play', () => onPlay(), { variant: 'primary', title: `Generate up to ${MAX_NEW} tokens` });
    const resetBtn = button('Reset', () => ctx.setStep(0), { title: 'Back to the original input' });
    const useBtn = button('Use as input', () => useAsInput(), { variant: 'ghost', title: 'Make the generated text the new input' });
    // Temperature changes the candidate list, so it sits on that list's header.
    const tempCtl = slider({
      id: `${uid}-temp`, label: 'temperature', min: 0, max: 1.5, step: 0.05, value: 0,
      format: (v) => (v === 0 ? 'greedy' : `T = ${v.toFixed(2)}`), onInput: () => cancelAuto(),
    });
    tempCtl.input.addEventListener('change', () => applyTemp());
    tempCtl.el.classList.add('gn-temp');

    const seqLabel = h('div', { class: 'gn-label' });
    const seqHead = h('div', { class: 'gn-seqhead', style: { width: `${SEQ.w}px` } }, seqLabel, h('div', { class: 'gn-seqctl' }, playBtn, resetBtn, useBtn));
    st.add(seqHead, { x: SEQ.x, y: ZONES.hero.y });
    const seqRow = h('div', { class: 'gn-seq' });
    st.add(seqRow, { x: SEQ.x, y: SEQ.y });

    const markId = `${uid}-head`;
    const bus = s('path', { class: 'gn-bus', fill: 'none', stroke: 'var(--c-embed)', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'marker-end': `url(#${markId})` });
    st.defs.append(s('marker', { id: markId, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' },
      s('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'var(--c-embed)' })));
    st.svg.append(bus);

    const modelEl = h('div', { class: 'gn-model' });
    st.add(modelEl, { x: MODEL.x, y: MODEL.y });

    const candLabel = h('div', { class: 'gn-label' });
    const candHead = h('div', { class: 'gn-candhead' }, candLabel, tempCtl.el);
    const candBox = h('div', { class: 'gn-cands' });
    const cands = Array.from({ length: CAND_ROWS }, () => {
      const ch = h('span', { class: 'gn-cand-ch' });
      const fill = h('i');
      const pctEl = h('span', { class: 'gn-cand-p' });
      const tag = h('span', { class: 'gn-cand-tag' });
      const row = h('div', { class: 'gn-cand' }, ch, h('span', { class: 'gn-cand-track' }, fill), pctEl, tag);
      candBox.append(row);
      return { row, ch, fill, pctEl, tag };
    });
    const stopPanel = h('div', { class: 'gn-stop', hidden: true });
    const candWrap = h('div', { class: 'gn-candwrap' }, candHead, candBox, stopPanel);
    st.add(candWrap, { x: CANDS.x, y: CANDS.y });

    // The step's label heads its equation, with the check on the same row; the card holds the live numbers.
    const stepLab = h('div', { class: 'k-label gn-steplab' });
    const eqChecks = h('div', { class: 'gn-checks' });
    const eqMain = h('div', { class: 'gn-eq-main' });
    const eqLive = h('div', { class: 'gn-eq-live' });
    const eqEl = h('div', { class: 'gn-eq' }, h('div', { class: 'gn-eqhead' }, stepLab, eqChecks), eqMain, eqLive);
    st.add(eqEl, { x: EQ.x, y: EQ.y });
    const cap = sceneCaption(st, ZONES.caption);
    const capEl = cap.el;

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

    // A display label: ␣ marks a leading space and ↵ a newline, as the tokenizer labels them (the token picker shows the same).
    function glyph(id) { return base.tk.label(id); }

    // The tape's geometry for the current sequence. Character tokens keep the fixed grid (one glyph per cell). Subword
    // tokens are as wide as their label, so a word stays whole; positions not generated yet are narrow empty slots.
    // When the row is too long the oldest positions are elided, and an ellipsis tile names them. The geometry is
    // computed from the labels, not measured, so it is right even while the scene is hidden.
    function layoutVis() {
      const bpe = base.tk.kind === 'bpe';
      if (bpe) return { bpe, gap: 4, th: TAPE.h, w: 0, labelEvery: 1 };
      const n = base.S, gap = n > 40 ? 2 : 4;
      const w = clamp(Math.floor((SEQ.w - (n - 1) * gap) / n), 10, 34);
      const th = clamp(w + 20, 30, 46);          // tiles stay near square when they are narrow
      return { bpe, gap, w, th, labelEvery: w >= 20 ? 1 : w >= 14 ? 4 : 8 };
    }
    // The shown positions are a..e. The tokens and the next empty slot must fit: the oldest tokens are elided until they
    // do, so an empty slot never pushes a generated token out of the tape. More empty slots are shown while they fit.
    function tapeLayout(ids = seq) {          // ids: the tokens shown (the sequence, or the sequence plus the token about to join)
      const L = ids.length, S = base.S, { bpe, gap } = vis;
      const ws = Array.from({ length: S }, (_, p) => (!bpe ? vis.w : p < L ? labelWidth(glyph(ids[p])) : TAPE.tick));
      const ellW = bpe ? TAPE.ell : vis.w;
      const width = (a, e) => ws.slice(a, e + 1).reduce((s, w) => s + w, 0) + (e - a) * gap + (a > 0 ? ellW + gap : 0);
      const must = Math.min(S, L + 1) - 1;        // the newest token, and the next slot when there is one
      let a = 0;                                  // the first position shown; the newest token always is
      while (a < L - 1 && width(a, must) > SEQ.w) a++;
      let e = must;
      while (e < S - 1 && width(a, e + 1) <= SEQ.w) e++;
      const cx = new Array(S);                    // centre x of every shown position
      let x = SEQ.x + (a > 0 ? ellW + gap : 0);
      for (let p = a; p <= e; p++) { cx[p] = x + ws[p] / 2; x += ws[p] + gap; }
      return { a, e, ws, cx, ellW };
    }
    const withoutRows = (L) => { let n = base.P; for (let k = base.P; k < L; k++) n += k + 1; return n; };

    // ------------------------------------------------------------ DOM that depends on the input
    function buildDom() {
      vis = layoutVis();
      seqRow.style.gap = `${vis.gap}px`;
      tiles = Array.from({ length: base.S }, (_, pos) => {
        const ch = h('span', { class: 'gn-ch', style: { fontSize: `${vis.bpe ? TAPE.font : Math.min(22, vis.w - 2)}px` } });
        const idx = h('span', { class: 'gn-idx', style: { top: `${vis.th + 6}px` } }, String(pos));
        const el = h('div', { class: 'gn-tile', style: { height: `${vis.th}px` }, title: `position ${pos}` }, ch, idx);
        return { el, ch, idx, pos };
      });
      const ellIdx = h('span', { class: 'gn-idx', style: { top: `${vis.th + 6}px`, left: '0', right: 'auto', textAlign: 'left' } });   // "0–17" starts at the tape's edge
      ell = { el: h('div', { class: 'gn-tile gn-ell', style: { height: `${vis.th}px` } }, h('span', { class: 'gn-ch' }, '⋯'), ellIdx), idx: ellIdx };
      seqRow.replaceChildren(ell.el, ...tiles.map((t) => t.el));
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
      cnt = { with: h('b'), without: h('b'), barWith: h('i'), barWithout: h('i'), bytes: h('small') };
      sideRefs = { note: h('span', { class: 'gn-side-note' }), sub: h('div', { class: 'gn-side-sub' }), more: h('div', { class: 'gn-more' }) };
      let body;
      if (kvOn) {
        const geo = stackGeom();
        rowsShown = Math.min(geo.rows, S);     // never more rows than the input can ever hold
        body = buildStacks(geo);
      } else {
        rowsShown = 0;
        sideRefs.note.textContent = `mask “${cfg.mask}”`;
        body = h('div', { class: 'gn-note-card' }, noCacheNote(cfg));
      }
      const counts = h('div', { class: 'gn-counts' },
        h('div', { class: 'gn-count' }, h('span', null, 'with cache'), h('span', { class: 'gn-track' }, cnt.barWith), h('span', { class: 'gn-val' }, cnt.with, cnt.bytes)),
        h('div', { class: 'gn-count without' }, h('span', null, 'without'), h('span', { class: 'gn-track' }, cnt.barWithout), h('span', { class: 'gn-val' }, cnt.without)));
      sideEl.replaceChildren(
        h('div', { class: 'gn-side-head' }, h('span', { class: 'gn-side-title' }, kvOn ? 'KV cache' : 'no KV cache'), sideRefs.note),
        ...(kvOn ? [sideRefs.sub] : []),
        body, sideRefs.more, counts);
      setLayers();
    }

    // Steps 0 and 1 show layer 0 only; the note says the rest share its shape. Steps 2 and 3 show every layer.
    function setLayers() {
      const all = step >= 2 || !base.kvOn || base.cfg.n_layer < 2;
      layers.forEach((blk, l) => { blk.el.hidden = !all && l > 0; });
      const more = base.cfg.n_layer - 1;
      sideRefs.more.hidden = all;
      sideRefs.more.textContent = `+${more} more ${more === 1 ? 'layer' : 'layers'} · same shape`;
    }

    // Shape of one K or V stack: kv head 0, one row per position, the first 12 and last 4 dims for wide heads
    // (the kit's convention). Rows are cut to what the side column can hold.
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
      const { cfg, rope } = base;
      const cols = dims.length, rows = rowsShown;
      const grid = h('div', { class: 'gn-kv' });
      for (let l = 0; l < cfg.n_layer; l++) {
        const blk = {};
        const pair = h('div', { class: 'gn-pair' });
        for (const kind of ['k', 'v']) {
          const trace = `L${l}.attn.${kind === 'k' ? (rope ? 'k_rot' : 'k_heads') : 'v_heads'}`;
          const m = mat(new Float64Array(rows * cols), rows, cols, { cell, gap: 0, mode: 'diverging', range: 1, kind, trace });
          m.classList.add(kind === 'k' ? 'gn-stack-k' : 'gn-stack-v');
          // Every cell starts as a placeholder; paintStacks fills the rows the cache holds.
          for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
              const cellEl = m.cellEl(r, c);
              cellEl.classList.add('gn-future');
              if (dims[c] < 0) cellEl.classList.add('ell');
            }
          }
          const read = h('div', { class: 'gn-read', hidden: true });     // the rows the newest token reads (step 3)
          m.querySelector('.k-mat-body').append(read);
          const wrap = h('div', { class: 'gn-stack-wrap' }, m);
          let ruler = null;
          if (kind === 'k' && l % 2 === 0) {          // the first column of each block row carries the position ruler
            ruler = Array.from({ length: rows }, () => h('span'));
            wrap.append(h('div', { class: 'gn-ruler', style: { '--rh': `${cell}px` } }, ruler));
          }
          blk[kind] = { m, dims, ruler, cell, read, band: m.querySelector('.k-row-band') };
          pair.append(h('div', { class: ['gn-col', kind] }, h('span', { class: 'gn-cap' }, kind === 'k' ? 'K' : 'V'), wrap));
        }
        blk.el = h('div', { class: 'gn-block' }, h('span', { class: 'gn-block-l' }, `layer ${l}`), pair);
        layers.push(blk);
        grid.append(blk.el);
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

    // The newest row of layer 0 (kv head 0, first entries) against a full pass of the same prefix.
    function newRowCheck(trace, L) {
      const { dh, rope } = base, t = L - 1, n = Math.min(4, dh);
      const K = trace[`L0.attn.${rope ? 'k_rot' : 'k_heads'}`], V = trace['L0.attn.v_heads'];
      const ck = cache.layers[0].k[0][t], cv = cache.layers[0].v[0][t];
      const k = [], v = [];
      let diff = 0;
      for (let d = 0; d < n; d++) {
        k.push(K.data[t * dh + d]);
        v.push(V.data[t * dh + d]);
        diff = Math.max(diff, Math.abs(ck[d] - k[d]), Math.abs(cv[d] - v[d]));
      }
      return { k, v, diff };
    }

    // A cached step is checked against a full pass over the same prefix. Rows already in the cache never
    // change, so the check can run later. Checks run one per macrotask: no step waits for a full pass.
    function queueCheck(L, row) {
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
        chk.doneL = Math.max(chk.doneL, L);
        chk.maxLogits = Math.max(chk.maxLogits, maxAbs(job.row, ref.logits.data, (L - 1) * V));
        chk.maxKV = Math.max(chk.maxKV, kvDiff(ref.trace, L));
        if (L > base.P) chk.newRow.set(L, newRowCheck(ref.trace, L));
        paintChecks();                                                   // reads only what is on screen (shownL)
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
      chk = { done: 0, doneL: 0, maxLogits: 0, maxKV: 0, newRow: new Map() };
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
    // step runs; then the cache row, the distribution and the bus update together. The caption and equation
    // of the step were already on screen (paintStep), so the flight moves only the token, the row and the bus.
    async function animateStep(id) {
      if (!base || !consistent() || g >= base.maxNew) return false;
      const e = epoch, pos = seq.length, tokId = next.id;
      let flight = null, dest = null;
      if (!reducedMotion() && !dead && id === runId) {
        // the flight lands where the token will sit once the tape is repainted (the window may move as it joins)
        const lay = tapeLayout(seq.concat([tokId]));
        dest = h('div', { style: { width: `${lay.ws[pos]}px`, height: `${vis.th}px` } });
        st.add(dest, { x: lay.cx[pos], y: SEQ.y + vis.th / 2, anchor: 'c' });
        pulse(modelEl);
        const pop = h('div', { class: 'gn-pop', style: vis.bpe ? { fontSize: `${TAPE.font}px` } : null }, glyph(tokId));
        st.add(pop, { x: MODEL.x + MODEL.w / 2, y: MODEL.y, anchor: 'c', layer: 'front' });
        flight = fly(st, pop, dest, { duration: FLY_MS, arc: 90 });   // fly clones the token, so the original can leave now
        pop.remove();
      }
      advance();
      if (flight) await flight;
      dest?.remove();
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
      paintStep();                                       // the label, caption and equation change now, not after the flight
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
      // A frame that never comes (a background tab, a slow browser) must not leave a stale number: settle it.
      setTimeout(() => { if (tweenTok.get(elx) === tok) { tweenTok.set(elx, tok + 1); elx.textContent = fmt(to, 0); } }, 700);
    }

    // The caption's words change at once; only the opacity animates, so a slow frame never leaves old text behind.
    function setCap(html) {
      if (html === capShown) return;
      capShown = html;
      capEl.innerHTML = html;
      capEl.getAnimations().forEach((a) => a.cancel());
      if (!reducedMotion()) capEl.animate([{ opacity: 0.45 }, { opacity: 1 }], { duration: 260, easing: MOTION.easeOut });
    }
    function setEq(main, live) {
      eqMain.innerHTML = texHTML(main, { display: true });
      eqLive.innerHTML = live ? texHTML(live, { display: true }) : '';
      eqLive.hidden = !live;
    }

    // The caption and equations of the current step. Before the state moves (live = false) they name the step and
    // its symbols without this state's numbers, so the text changes with the step. Once the state has settled
    // (live = true) they carry its numbers.
    function stepText(live) {
      const { P, cfg, kvOn, maxNew } = base;
      const greedy = temp === 0, T = temp.toFixed(2);
      if (maxNew === 0) {
        return {
          cap: `The prompt already fills the ${cfg.block_size}-token window, so there is no room to generate. Shorten the input to see the loop run.`,
          main: `\\text{context window full: } ${P} = ${cfg.block_size}`, live: '',
        };
      }
      if (step === 0) {
        const c = live ? next : null;
        const scores = `Position <b>${P - 1}</b> (<b>“${htmlEsc(glyph(base.ids[P - 1]))}”</b>) scores the next token${c ? `, <b>“${htmlEsc(glyph(c.id))}”</b>` : ''}.`;
        const cap = kvOn
          ? `The prompt goes through the model once and fills the cache: one key and one value row per token, in every layer. ${scores}`
          : `The prompt goes through the model once. ${scores} No cache is kept, so later steps run the whole sequence again.`;
        const main = greedy ? `\\mathit{id}_{${P}}=\\arg\\max_{v}\\, z_{${P - 1},v}` : `\\mathit{id}_{${P}}\\sim\\softmax(z_{${P - 1}}/T)`;
        let lv = '';
        if (c) {
          const tch = texLab(glyph(c.id));
          lv = `z_{${P - 1},\\,\\text{“${tch}”}}=${num(c.logits[c.id], 2)},\\qquad p(\\text{“${tch}”})=${num(c.probs[c.id], 3)}${greedy ? '' : `\\text{ at } T=${T}`}`;
        }
        return { cap, main, live: lv };
      }
      if (step === 1) {
        const tok = live && g >= 1 ? seq[P] : null;         // the token this step appends, once it has been appended
        const who = tok != null ? `<b>“${htmlEsc(glyph(tok))}”</b>` : 'the next token';
        const head = greedy ? `The highest score wins, so the model writes ${who}.` : `At T = ${T} the model draws ${who} from its distribution.`;
        if (!kvOn) {
          return {
            cap: `${head} With no cache, all ${P + 1} positions run again to produce it.`,
            main: `\\mathit{id}_{${P}}${tok != null ? `=\\text{“${texLab(glyph(tok))}”}` : ''},\\qquad z_{${P}}=\\text{Transformer}(\\mathit{id}_{0},\\dots,\\mathit{id}_{${P}})`,
            live: `\\text{positions run: } ${P + 1}`,
          };
        }
        const lv = tok != null
          ? `k^{(0)}_{${P}}=\\big(${vecTex(cache.layers[0].k[0][P])}\\big),\\qquad v^{(0)}_{${P}}=\\big(${vecTex(cache.layers[0].v[0][P])}\\big)`
          : '';
        return {
          cap: `${head} Only that token is fed in: it adds one row to every K and V stack, and the older rows stay as they are.`,
          main: `k^{(\\ell)}_{${P}}=${kvLaw(cfg, P).k}`,
          live: lv,
        };
      }
      if (step === 2) {
        if (!live) {
          return {
            cap: kvOn
              ? 'Each step feeds one new token through the model. Its K and V rows join the cache, and the rows already stored are not recomputed.'
              : 'Every step reruns the whole sequence, so each step costs more than the one before.',
            main: kvOn
              ? `\\text{rows}_{\\text{cache}}=${P}+n,\\qquad \\text{rows}_{\\text{no cache}}=${P}+\\textstyle\\sum_{s=${P}}^{${P}+n-1}(s+1)`
              : `\\text{positions run}=${P}+\\textstyle\\sum_{s=${P}}^{${P}+n-1}(s+1)`,
            live: '',
          };
        }
        const L = seq.length, without = withoutRows(L), why = stopReason();
        const tail = why ? ` Generation stops: ${why}.` : '';
        return {
          cap: kvOn
            ? `After ${plural(g, 'token')} the cache has processed each of the ${L} positions once: <b>${L}</b> rows. Without a cache, every step reruns the sequence: <b>${without}</b> positions in all.${tail}`
            : `Every step reruns the whole sequence: <b>${without}</b> positions in all, for ${plural(g, 'token')}. This mask has no cache (see the last step).${tail}`,
          main: kvOn
            ? `\\text{rows}_{\\text{cache}}=${L},\\qquad \\text{rows}_{\\text{no cache}}=${P}+\\textstyle\\sum_{s=${P}}^{${L - 1}}(s+1)=${without}`
            : `\\text{positions run}=${P}+\\textstyle\\sum_{s=${P}}^{${L - 1}}(s+1)=${without}`,
          live: kvOn
            ? `\\text{without the cache: } ${num(without / L, 1)}\\times\\text{ the work with it}`
            : `\\text{the last step alone ran } ${L} \\text{ positions}`,
        };
      }
      // step 3: the cache step; its text depends on the config only
      if (kvOn) {
        return {
          cap: cacheCaption(cfg),
          main: `x^{(\\ell)}_{j}=f_{\\ell}\\big(x^{(\\ell-1)}_{0},\\dots,x^{(\\ell-1)}_{j}\\big)`,
          live: attnLaw(cfg),
        };
      }
      const L = live ? seq.length : P + maxNew;
      return {
        cap: `No cache for mask “${cfg.mask}”. ${noCacheWhy(cfg)}`,
        main: cfg.mask === 'prefix'
          ? `x^{(\\ell)}_{j}=f_{\\ell}\\big(x^{(\\ell-1)}_{0},\\dots,x^{(\\ell-1)}_{\\max(j,\\,${Math.min(cfg.prefix_len, L) - 1})}\\big)`
          : `x^{(\\ell)}_{j}=f_{\\ell}\\big(x^{(\\ell-1)}_{0},\\dots,x^{(\\ell-1)}_{${L - 1}}\\big)`,
        live: cfg.mask === 'prefix' ? `\\text{rows } j<${cfg.prefix_len} \\text{ see the whole prefix}` : '\\text{from layer } 1 \\text{ on, } x_j \\text{ depends on the later tokens}',
      };
    }

    // The step's label, layers and text, before any state moves. The numbers follow in paintAll.
    function paintStep() {
      if (!base) return;
      stepLab.textContent = STEPS[step];
      stepLab.style.setProperty('--kind', `var(--c-${STEP_KIND[step]})`);
      setLayers();
      if (consistent() && shownL === seq.length) paintStacks(-1);     // not while a token is in flight
      const t = stepText(false);
      setCap(t.cap);
      setEq(t.main, t.live);
    }

    function paintText() {
      const t = stepText(true);
      setCap(t.cap);
      setEq(t.main, t.live);
    }

    function paintChrome() {
      const { cfg, P, kvOn } = base;
      eyebrow.replaceChildren(
        chip(`${cfg.n_layer} ${cfg.n_layer === 1 ? 'layer' : 'layers'}`, 'resid'),
        chip(temp > 0 ? `T = ${temp.toFixed(2)}` : 'greedy'),        // a setting, so a neutral chip
        chip(kvOn ? 'KV cache' : 'no cache'),                          // a method, so a neutral chip
        ...(base.tk.kind === 'bpe' ? [chip('␣ = leading space')] : []));   // the subword labels need this key
      seqLabel.textContent = `tokens · prompt ${P}${g ? ` · generated ${g}` : ''} · context ${seq.length} of ${cfg.block_size}${stopReason() ? ' · stopped' : ''}`;
    }

    function paintSeq(fresh) {
      const L = seq.length, lay = tapeLayout();
      // subword tape: every token is numbered; of the empty slots, the next one and every fourth after it (narrow slots would crowd)
      const labelOn = (pos) => pos === L - 1 || (vis.bpe ? pos < L || (pos - L) % 4 === 0
        : pos % vis.labelEvery === 0 && Math.abs(pos - (L - 1)) > vis.labelEvery / 2);
      tiles.forEach((t, p) => {
        const has = p < L;
        t.el.style.display = p >= lay.a && p <= lay.e ? '' : 'none';
        t.el.style.width = `${lay.ws[p]}px`;
        t.el.classList.toggle('gn-gen', has && p >= base.P);
        t.el.classList.toggle('gn-empty', !has);
        t.el.classList.toggle('gn-new', has && g > 0 && p === L - 1);
        t.ch.textContent = has ? glyph(seq[p]) : '';
        t.idx.classList.toggle('gn-on', labelOn(p));
        if (p === fresh && !reducedMotion()) {
          t.el.animate([{ transform: 'scale(.4)', opacity: 0 }, { transform: 'scale(1)', opacity: 1 }], { duration: 460, easing: MOTION.easeOut });
        }
      });
      ell.el.style.display = lay.a > 0 ? '' : 'none';
      ell.el.style.width = `${lay.ellW}px`;
      ell.el.title = `positions 0 to ${lay.a - 1} are not shown`;
      ell.idx.textContent = lay.a > 1 ? `0–${lay.a - 1}` : '0';
      ell.idx.classList.add('gn-on');
    }

    function paintBus() {
      if (!seq.length) { bus.setAttribute('d', ''); return; }
      const lay = tapeLayout(), y0 = SEQ.y + vis.th + 24, xBox = MODEL.x + MODEL.w / 2, xNew = lay.cx[seq.length - 1];
      if (base.kvOn) {
        bus.setAttribute('d', `M${xNew},${y0} V${BUS_Y} H${xBox} V${MODEL.y - 4}`);
      } else {
        // no cache: every visible token is read again, so a bracket spans the sequence and feeds the box
        const first = lay.cx[lay.a], mid = (first + xNew) / 2;
        bus.setAttribute('d', `M${first},${y0} V${BUS_Y - 12} H${xNew} V${y0} M${mid},${BUS_Y - 12} V${BUS_Y} H${xBox} V${MODEL.y - 4}`);
      }
      bus.classList.toggle('gn-run', running);
    }

    // Candidates are the top five next tokens. The store's trace holds the prompt's own distribution at T = 1,
    // so the list is traced only for the first choice at greedy settings, which is exactly that distribution.
    // Once generation has stopped the next token is never appended, so the list gives way to the reason.
    function paintCands() {
      const why = stopReason();
      candBox.hidden = !!why;
      stopPanel.hidden = !why;
      if (why) {
        stopPanel.textContent = `Generation has stopped: ${why}. The next token is not appended.`;
        candLabel.textContent = 'next token';
        delete candBox.dataset.trace;
        candBox.querySelectorAll('[data-cell]').forEach((r) => r.removeAttribute('data-cell'));
        return;
      }
      const { probs } = next;
      let order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, cands.length);
      if (!order.includes(next.id)) order[order.length - 1] = next.id;   // always show the chosen token
      order = order.sort((a, b) => probs[b] - probs[a]);
      // the chip column is as wide as the longest label shown (15 px glyphs), so the probability bars line up
      candBox.style.setProperty('--chip', `${Math.max(30, Math.ceil(Math.max(...order.map((id) => Array.from(glyph(id)).length)) * 15 * 0.64) + 14)}px`);
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
      candLabel.textContent = temp > 0 ? `next token · drawn at T = ${temp.toFixed(2)}` : 'next token · the highest score wins';
    }

    // Each stack: rows are positions, columns are the dims of kv head 0. The window ends at the newest row and holds
    // rowsShown rows (every row of a short input); rows the cache does not hold yet are placeholders. One colour scale for all.
    function paintStacks(fresh) {
      if (!base.kvOn || !cache || !layers.length) return;
      const { cfg, P } = base;
      const L = seq.length, qpos = L - 1;
      const top = Math.max(0, L - rowsShown);
      // step 3: the rows the newest token reads (all of them, or the last `window` for a sliding mask)
      const readLo = step === 3 ? (cfg.mask === 'sliding' ? Math.max(0, qpos - cfg.window + 1) : 0) : -1;
      layers.forEach((blk, l) => {
        if (blk.el.hidden) return;
        for (const kind of ['k', 'v']) {
          const { m, dims, ruler, cell, band, read } = blk[kind];
          const cols = dims.length, data = cache.layers[l][kind][0];   // kv head 0: one row per position
          const have = data.length;
          const vals = new Float64Array(rowsShown * cols);
          for (let i = 0; i < rowsShown; i++) {
            const t = top + i, filled = t < have, unread = readLo >= 0 && filled && (t < readLo || t > qpos);
            for (let c = 0; c < cols; c++) {
              const cellEl = m.cellEl(i, c);
              if (filled && dims[c] >= 0) vals[i * cols + c] = data[t][dims[c]];
              cellEl.classList.toggle('gn-future', !filled);
              cellEl.classList.toggle('gn-unread', unread);
              if (filled && dims[c] >= 0 && t < P) cellEl.dataset.cell = `${t},${dims[c]},0`;   // prompt rows have trace values
              else cellEl.removeAttribute('data-cell');
            }
          }
          m.update(vals, { range: stackRange });
          if (read) {
            const r0 = Math.max(readLo, top) - top, r1 = Math.min(qpos, top + rowsShown - 1) - top;
            read.hidden = readLo < 0 || r1 < r0;
            if (!read.hidden) {
              read.style.top = `${3 + r0 * cell}px`;
              read.style.height = `${(r1 - r0 + 1) * cell}px`;
              read.style.width = `${cols * cell}px`;
            }
          }
          // The band's place follows from the cell size and the frame (3 px padding, 1 px border), so it is set
          // directly: reading layout here would force a new layout for every stack on every step.
          if (band) {
            const r = qpos - top, on = g > 0 && r >= 0 && r < rowsShown;
            band.hidden = !on;
            if (on) {
              band.style.top = `${2 + r * cell}px`;
              band.style.left = '1px';
              band.style.width = `${cols * cell + 6}px`;
              band.style.height = `${cell + 4}px`;
              if (fresh === qpos && !reducedMotion()) band.animate([{ opacity: 0.2 }, { opacity: 1 }], { duration: 520, easing: MOTION.easeOut });
            }
          }
          if (ruler) {
            // a label every RULER_EVERY rows; none within NEAR_ROWS of the newest row, which carries its own label
            for (let i = 0; i < rowsShown; i++) {
              const t = top + i, has = t < L;
              const show = has && (t === qpos || (t % RULER_EVERY === 0 && Math.abs(t - qpos) > NEAR_ROWS));
              ruler[i].textContent = show ? String(t) : '';
              ruler[i].classList.toggle('gn-now', has && t === qpos);
            }
          }
        }
      });
      const heads = cfg.n_head, kvHeads = cfg.n_kv_head;
      sideRefs.note.textContent = `${L} rows cached`;
      const lines = [
        `kv head 0 of ${kvHeads} · rows ${top}–${qpos} shown`,
        `row ${qpos} is the token “${glyph(seq[qpos])}”`,
        `one colour scale for every stack: ±${fmt(stackRange, 2)}`,
      ];
      if (readLo >= 0) lines.push(`token ${qpos} reads rows ${readLo}–${qpos}`);
      if (kvHeads < heads) lines.push(`each kv head serves ${heads / kvHeads} query heads`);
      sideRefs.sub.replaceChildren(...lines.map((t) => h('div', null, t)));
    }

    function paintCounts(animate) {
      const L = seq.length, without = withoutRows(L);
      setNum(cnt.without, without, animate);
      cnt.barWithout.style.width = '100%';
      if (base.kvOn) {
        setNum(cnt.with, L, animate);
        cnt.barWith.style.width = `${(L / Math.max(1, without)) * 100}%`;
        cnt.bytes.textContent = ` · ${kib(E.cacheBytes(base.cfg, L, 2))}`;    // fp16
      } else {
        cnt.with.textContent = 'off';
        cnt.barWith.style.width = '0%';
        cnt.bytes.textContent = ' · none stored';
      }
    }

    // One line under the equations. Step 1 checks the new row; the other steps check the whole cache. Both are built
    // from what is on screen (shownL, shownG), never from the state that is still moving.
    function checkLine(kind, text) {
      eqChecks.replaceChildren(h('span', { class: ['k-check', kind] }, text));
    }
    function paintChecks() {
      if (!base || !chk) return;
      if (!base.kvOn) { eqChecks.replaceChildren(h('span', { class: 'gn-muted' }, 'Nothing to check: every step is a full pass.')); return; }
      const L = shownL;
      if (!L) { checkLine('wait', 'checking against a full forward pass over the same prefix…'); return; }
      if (step === 1 && L > base.P) {
        const nr = chk.newRow.get(L);
        if (!nr) checkLine('wait', 'checking the new row against a full forward pass…');
        else if (nr.diff <= TOL) checkLine('ok', '✓ the new row equals a full forward pass');
        else checkLine('bad', `✗ the new row differs from a full forward pass by ${sci(nr.diff)}`);
        return;
      }
      if (chk.doneL < L) { checkLine('wait', 'checking against a full forward pass over the same prefix…'); return; }
      const worst = Math.max(chk.maxLogits, chk.maxKV);
      if (worst <= TOL) checkLine('ok', `✓ ${shownG ? `the prompt and ${plural(shownG, 'step')} match` : 'the prompt matches'} a full forward pass · max |Δ| ${sci(worst)}`);
      else checkLine('bad', `✗ differs from a full forward pass by ${sci(worst)}`);
    }

    function paintAll({ animate = false, fresh = -1 } = {}) {
      if (dead || !base || !consistent()) return;      // mid-feed: the finished state paints when the feed ends
      shownL = seq.length;
      shownG = g;
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
    // One colour scale for every stack of this input: the largest prompt value, so a value keeps its colour
    // across steps and layers. Generated rows above it saturate.
    function promptRange(trace) {
      const { cfg, dh, rope, P } = base;
      let m = 0;
      for (let l = 0; l < cfg.n_layer; l++) {
        for (const key of [`L${l}.attn.${rope ? 'k_rot' : 'k_heads'}`, `L${l}.attn.v_heads`]) {
          const t = trace[key];
          if (!t) continue;
          const n = Math.min(t.data.length, P * dh);   // kv head 0 holds the first P rows
          for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(t.data[i]));
        }
      }
      return m || 1;
    }

    function setup() {
      step = 0;
      root.dataset.step = '0';
      next = null;                                       // not consistent until the prompt is read (rebuild)
      base = readBase();
      buildDom();
      stackRange = promptRange(ctx.state.trace);
      header.setSub(subFor(base.cfg, base.kvOn));
      paintStep();
    }

    setup();
    work = rebuild(0);
    ctx.setSteps(STEPS, setStep);
    appear([header.el, seqHead, seqRow, modelEl, candWrap, sideEl, eqEl], { stagger: MOTION.stagger });
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

// Demo page for the motion kit (src/ui/flow.js). Hidden route: #_demo_flow.
//   1. Embedding lookup: rows of W_E fly into the sequence, positional rows merge in, norms tick.
//   2. Residual stream: one vertical strip per depth stage; change the layer or token and it moves.
//   3. One attention head as seven sequence() steps, with a playbar and live captions.
// Every number is read from ctx.state.trace or ctx.state.model, or computed from them. Nothing is invented.
import {
  h, s, viewHeader, segmented, tokenStrip, button, callout, fmt, shapeBadge, formula, clamp,
} from '../ui/index.js';
import {
  vectorStrip, flyClone, ticker, drawPath, sequence, sequencePlaybar, stagger, easing, arcPathD, reducedMotion,
} from '../ui/flow.js';

const MAX_ROWS = 32;          // the embedding demo shows at most this many positions
const BASE_STEP_MS = 900;     // duration of one attention step at 1x
const CSS = `
.fxd-block { display: grid; gap: 16px; min-width: 0; }
.fxd-block .row { gap: 10px; }
.fxd-status { font: 500 var(--fs-xs)/1.5 var(--font-mono); color: var(--fg-2); min-height: 1.5em; contain: layout; }
.fxd-warn { margin: 0; color: var(--warn); font-size: var(--fs-sm); }
.fxd-embed { position: relative; display: flex; flex-wrap: wrap; gap: 28px 34px; align-items: flex-start; min-width: 0; }
.fxd-mat { display: grid; gap: 10px; align-content: start; min-width: 0; }
.fxd-mat-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.fxd-key { font: 600 var(--fs-sm)/1 var(--font-mono); color: var(--fg); }
.fxd-subkey { font: 500 var(--fs-xs)/1 var(--font-mono); color: var(--fg-3); }
.fxd-rows, .fxd-xrows, .fxd-prows { display: grid; gap: 1px; }
.fxd-row, .fxd-xrow, .fxd-prow { display: flex; align-items: center; gap: 7px; min-width: 0; }
.fxd-lab { flex: none; width: 30px; text-align: right; font: 500 11px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.fxd-pos { flex: none; width: 22px; text-align: right; font: 500 11px/1 var(--font-mono); color: var(--fg-3); }
.fxd-row.fxd-used .fxd-lab { color: var(--fg); font-weight: 700; }
.fxd-xrow.fxd-cur .fxd-lab { color: var(--accent); font-weight: 700; }
.fxd-xrow.fxd-cur .fxd-pos { color: var(--fg); font-weight: 700; }
.fxd-slot { position: relative; flex: none; border-radius: 3px; background: var(--surface-2); outline: 1px dashed var(--line-strong); outline-offset: -1px; }
.fxd-slot.fxd-filled { outline-color: transparent; }
.fxd-slot .fx-strip { visibility: hidden; }
.fxd-slot.fxd-filled .fx-strip { visibility: visible; }
.fxd-xrow.fxd-cur .fxd-slot { outline: 2px solid var(--accent); outline-offset: 2px; }
.fxd-norm { font: 500 11px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; font-variant-numeric: tabular-nums; min-width: 5.5em; contain: layout; }
.fxd-trails { position: absolute; left: 0; top: 0; width: 0; height: 0; overflow: visible; pointer-events: none; z-index: 2; }
.fxd-trail { position: absolute; overflow: visible; transition: opacity .5s ease; }
.fxd-trail.fxd-trail-out { opacity: 0; }
.fxd-trail path { fill: none; stroke-width: 1.6; stroke-linecap: round; opacity: .85; }
.fxd-trail-tok { stroke: var(--c-embed); }
.fxd-trail-pos { stroke: var(--c-pos); }
.fxd-cols { display: flex; gap: 6px; align-items: flex-start; padding: 6px 2px 8px; min-width: min-content; }
.fxd-col { display: grid; justify-items: center; gap: 8px; padding: 8px 6px 10px; border-radius: var(--r-md); border: 1px solid transparent; transition: background .25s, border-color .25s; }
.fxd-col.fxd-on { background: var(--accent-soft); border-color: color-mix(in srgb, var(--accent) 40%, transparent); }
.fxd-col.fxd-layer-start { margin-left: 16px; }
.fxd-col-head { display: grid; justify-items: center; gap: 3px; border: 0; background: none; padding: 2px 4px; cursor: pointer; color: var(--fg); font: 600 11px/1.1 var(--font-mono); }
.fxd-col-head:hover { color: var(--accent); }
.fxd-col-sub { font: 500 10px/1 var(--font-mono); color: var(--fg-3); }
.fxd-readout { margin: 0; color: var(--fg-2); font-size: var(--fs-sm); }
.fxd-att { display: grid; gap: 16px; grid-template-columns: minmax(0, 1fr) minmax(0, 1.55fr); align-items: start; min-width: 0; }
@media (max-width: 820px) { .fxd-att { grid-template-columns: minmax(0, 1fr); } }
.fxd-cap { display: grid; gap: 10px; align-content: start; padding: 16px; border: 1px solid var(--line); border-radius: var(--r-lg); background: var(--surface); min-width: 0; }
.fxd-cap-title { font-size: var(--fs-lg); }
.fxd-cap-text { margin: 0; color: var(--fg-2); font-size: var(--fs-sm); }
.fxd-stage-att { position: relative; display: grid; align-content: start; min-height: 200px; padding: 16px; border: 1px solid var(--line); border-radius: var(--r-lg); background: var(--surface); min-width: 0; overflow-x: auto; }
.fxd-panel { display: grid; gap: 16px; align-content: start; min-width: max-content; }
.fxd-panel[hidden] { display: none; }
.fxd-lane { display: grid; gap: 6px; min-width: 0; }
.fxd-lane-h { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.fxd-lane-label { font: 600 var(--fs-xs)/1 var(--font-mono); color: var(--fg-2); }
.fxd-lane-note { font: 500 var(--fs-xs)/1.2 var(--font-mono); color: var(--fg-3); }
.fxd-before { opacity: .55; }
.fxd-before .fxd-lane-label { color: var(--fg-3); }
`;

// ----------------------------------------------------------------- helpers

const zeros = (n) => new Float64Array(n);
const norm = (arr) => { let acc = 0; for (let i = 0; i < arr.length; i++) acc += arr[i] * arr[i]; return Math.sqrt(acc); };
const maxAbs = (arr) => {
  let m = 0;
  for (let i = 0; i < arr.length; i++) { const a = Math.abs(arr[i]); if (Number.isFinite(a) && a > m) m = a; }
  return m || 1;
};
const scaled = (arr, p) => Float64Array.from(arr, (v) => v * p);
const normText = (v) => `‖x‖ ${fmt(v, 2)}`;
// Row r of a [T, C] tensor, and row (head, r) of a [H, T, C] tensor. Both are copies.
const row2 = (t, r) => { if (!t) return null; const C = t.shape[1]; return t.data.slice(r * C, (r + 1) * C); };
const row3 = (t, hd, r) => { if (!t) return null; const [, R, C] = t.shape; const o = (hd * R + r) * C; return t.data.slice(o, o + C); };
const injectCss = () => {
  if (document.getElementById('fxd-style')) return;
  document.head.append(h('style', { id: 'fxd-style' }, CSS));
};

// True when the input has no character from the vocabulary. store.setText then shows one placeholder token.
const isPlaceholder = (st) => st.tokenizer.encode(st.text ?? '').ids.length === 0;

// The embedding lede depends on how the position enters the model.
function embeddingLede(cfg) {
  switch (cfg.pos_encoding) {
    case 'learned': return 'Each token id copies one row of W_E. The positional row P[t] (the learned table wpe) is added to it, and the sum is the first residual vector, x₀ = W_E[id] + P[t].';
    case 'sinusoidal': return 'Each token id copies one row of W_E. The fixed sinusoidal row P[t] is added to it, and the sum is the first residual vector, x₀ = W_E[id] + P[t].';
    case 'rope': return 'Each token id copies one row of W_E, and that row is x₀. RoPE adds no vector here: it rotates q and k inside attention.';
    case 'alibi': return 'Each token id copies one row of W_E, and that row is x₀. ALiBi adds no vector here: it penalises distant keys inside attention.';
    default: return 'Each token id copies one row of W_E, and that row is x₀. This configuration adds no positional information (pos_encoding = none).';
  }
}

// A section with a lede the page can rewrite (the kit's section() does not return its lede).
function sectionWithLede(title, lede, child) {
  const p = h('p');
  p.innerHTML = lede;
  const el = h('section', { class: 'section' }, h('div', { class: 'section-head' }, h('h2', null, title), p), child);
  return { el, lede: p };
}

// The mask rule, written from src/engine maskAllows(): causal keeps j <= t, sliding keeps
// t - window < j <= t, and prefix also keeps every key j < prefix_len.
function maskEquation(cfg) {
  switch (cfg.mask) {
    case 'none': return 'mask = none (every key visible)';
    case 'sliding': return `s<sub>tj</sub> = −∞ unless t − ${cfg.window} &lt; j ≤ t`;
    case 'prefix': return `s<sub>tj</sub> = −∞ where j &gt; t and j ≥ ${cfg.prefix_len}`;
    default: return 's<sub>tj</sub> = −∞ where j &gt; t';
  }
}

function maskNote(cfg) {
  switch (cfg.mask) {
    case 'none': return 'no mask';
    case 'sliding': return `keys outside the last ${cfg.window}: −∞`;
    case 'prefix': return 'future keys after the prefix: −∞';
    default: return 'future keys: −∞';
  }
}

// ----------------------------------------------------------- 1. embeddings

function embeddingBlock(ctx, width) {
  const st = ctx.state, cfg = st.config, d = cfg.d_model, V = cfg.vocab_size;
  const Tall = st.ids.length;
  const T = Math.min(Tall, MAX_ROWS);
  const ids = st.ids.slice(0, T);
  const placeholder = isPlaceholder(st);
  const shownNote = Tall > T ? `first ${T} of ${Tall}` : null;
  const positions = Tall > T ? `the first ${T} of ${Tall} positions` : `all ${T} positions`;
  const labels = ids.map((id) => st.tokenizer.label(id));
  const WE = st.model.params['wte.weight'];
  const out = st.trace['emb.out'];
  const pos = st.trace['emb.pos'] ?? null;
  const posKey = { learned: 'learned rows wpe', sinusoidal: 'sin/cos table' }[cfg.pos_encoding] ?? '';

  const wide = width >= 860;
  // Cell width so that every row fits: a sequence row also carries its position, token and norm labels.
  const rowOverhead = 150;
  const cw = wide
    ? clamp(Math.floor((width - rowOverhead - 134) / (3 * d)), 3, 6)
    : clamp(Math.floor((width - rowOverhead) / d), 3, 7);
  const wRowH = clamp(Math.floor(330 / Math.max(1, V)), 4, 11);
  const xRowH = clamp(Math.floor(330 / Math.max(1, T)), 9, 14);
  // Flights start this far apart: a long input gets a shorter gap, so a run stays near 3 s.
  const gap = clamp(Math.round(2400 / Math.max(1, T)), 45, 110);
  const rE = maxAbs(WE.data), rX = maxAbs(out.data), rP = pos ? maxAbs(pos.data) : 1;
  const used = new Set(ids);

  const strips = [];
  const make = (o) => { const sp = vectorStrip(o); strips.push(sp); return sp; };
  let ac = null, busy = false, finished = false, speedMul = 1;
  const fades = new Set();          // timers that remove faded trails
  const weRow = (id) => WE.data.subarray(id * d, (id + 1) * d);
  const dur = (ms) => ms * speedMul;

  // W_E: one strip per vocabulary row. These canvases are what flies.
  const weStrips = [];
  const weRowEls = [];
  for (let r = 0; r < V; r++) {
    const sp = make({ values: weRow(r), orientation: 'h', cell: cw, thickness: wRowH, pad: 0, mode: 'diverging', range: rE, ariaLabel: `W_E row ${r}` });
    weStrips.push(sp);
    weRowEls.push(h('div', { class: ['fxd-row', used.has(r) ? 'fxd-used' : ''], style: { height: `${wRowH}px` } },
      h('span', { class: 'fxd-lab', style: { lineHeight: `${wRowH}px`, fontSize: `${Math.min(11, wRowH + 1)}px` } }, st.tokenizer.label(r)), sp.el));
  }

  // Sequence x0: one slot per position, filled as token rows land and positional rows merge in.
  const slots = [];
  for (let t = 0; t < T; t++) {
    const sp = make({ values: zeros(d), orientation: 'h', cell: cw, thickness: xRowH, pad: 0, mode: 'diverging', range: rE, ariaLabel: `x0 position ${t}` });
    const el = h('div', { class: 'fxd-slot', style: { width: `${d * cw}px`, height: `${xRowH}px` } }, sp.el);
    const normEl = h('span', { class: 'fxd-norm' }, '‖x‖ —');
    const lh = { lineHeight: `${xRowH}px` };
    const row = h('div', { class: 'fxd-xrow', style: { height: `${xRowH}px` } },
      h('span', { class: 'fxd-pos', style: lh }, String(t)), h('span', { class: 'fxd-lab', style: lh }, labels[t]), el, normEl);
    slots.push({ el, strip: sp, norm: normEl, row });
  }

  // Positional rows P[t] (emb.pos). Absent when pos_encoding adds nothing to the embedding.
  const posStrips = [];
  let pBlock;
  if (pos) {
    const posEls = [];
    for (let t = 0; t < T; t++) {
      const sp = make({ values: pos.data.subarray(t * d, (t + 1) * d), orientation: 'h', cell: cw, thickness: xRowH, pad: 0, mode: 'diverging', range: rP, ariaLabel: `positional row ${t}` });
      posStrips.push(sp);
      posEls.push(h('div', { class: 'fxd-prow', style: { height: `${xRowH}px` } }, h('span', { class: 'fxd-pos', style: { lineHeight: `${xRowH}px` } }, String(t)), sp.el));
    }
    pBlock = h('div', { class: 'fxd-mat' }, matHead('P', posKey, [['T', Tall], ['d', d]], shownNote), h('div', { class: 'fxd-prows' }, posEls));
  } else {
    pBlock = callout('', h('strong', null, 'No positional rows. '),
      `pos_encoding = ${cfg.pos_encoding} adds nothing to the embedding, so x₀ is the token embedding alone.`);
    pBlock.style.maxWidth = '300px';
  }

  const weBlock = h('div', { class: 'fxd-mat' }, matHead('W_E', 'wte.weight', [['V', V], ['d', d]]), h('div', { class: 'fxd-rows' }, weRowEls));
  const xBlock = h('div', { class: 'fxd-mat' }, matHead('x₀', pos ? 'W_E[ids] + P' : 'W_E[ids]', [['T', Tall], ['d', d]], shownNote), h('div', { class: 'fxd-xrows' }, slots.map((sl) => sl.row)));
  const stage = h('div', { class: 'fxd-embed' }, weBlock, xBlock, pBlock);
  // Holds the arcs. It has no size and is absolutely positioned, so adding an arc never changes
  // the flow layout of the matrices.
  const trailLayer = h('div', { class: 'fxd-trails', 'aria-hidden': 'true' });
  stage.append(trailLayer);

  const status = h('div', { class: 'fxd-status', 'aria-live': 'polite' });
  const setStatus = (text) => { status.textContent = text; };
  const idleText = pos
    ? 'Press Play. Each row of W_E is copied into its position, then the positional row P[t] is added.'
    : 'Press Play. Each row of W_E is copied into its position.';
  const finalText = () => (pos
    ? `x₀[t] = W_E[id] + P[t] for ${positions}. Colour scale ±${fmt(rX, 2)}.`
    : `x₀[t] = W_E[id] for ${positions} (no positional rows in this configuration).`);
  const emptyNote = placeholder
    ? h('p', { class: 'fxd-warn' }, 'No input text: this shows one placeholder token (id 0), not real text. Enter some text to see its embeddings.')
    : null;

  const playBtn = button('Play', () => startRun(), { variant: 'primary' });
  const skipBtn = button('Skip to end', () => skip(), { variant: 'ghost' });
  const resetBtn = button('Reset', () => reset(), { variant: 'ghost' });
  const speedSeg = segmented({
    options: [{ value: 2, label: '0.5×', title: 'Half speed' }, { value: 1, label: '1×' }, { value: 0.5, label: '2×', title: 'Double speed' }],
    value: 1, ariaLabel: 'Animation speed', onChange: (v) => { speedMul = v; },
  });
  const controls = h('div', { class: 'row' }, playBtn, skipBtn, resetBtn, h('span', { class: 'eyebrow' }, 'speed'), speedSeg.el);
  const el = h('div', { class: 'fxd-block' }, controls, emptyNote, status, stage);

  function syncButtons() {
    playBtn.disabled = busy;
    playBtn.textContent = busy ? 'Running…' : finished ? 'Replay' : 'Play';
    skipBtn.disabled = finished;
  }
  function clearTrails() {
    trailLayer.replaceChildren();
    fades.forEach((t) => clearTimeout(t));
    fades.clear();
  }
  function stopRun() {
    if (ac) ac.abort();
    ac = null;
    busy = false;
    clearTrails();
  }
  function resetSlots() {
    slots.forEach((sl) => {
      sl.el.classList.remove('fxd-filled');
      sl.strip.update(zeros(d), { animate: false, range: rE });
      sl.norm.textContent = '‖x‖ —';
    });
  }
  function reset() {
    stopRun();
    resetSlots();
    finished = false;
    setStatus(idleText);
    syncButtons();
  }
  function skip() {
    stopRun();
    slots.forEach((sl, i) => {
      const sum = out.data.subarray(i * d, (i + 1) * d);
      sl.strip.update(sum, { animate: false, range: rX });
      sl.el.classList.add('fxd-filled');
      sl.norm.textContent = normText(norm(sum));
    });
    finished = true;
    setStatus(finalText());
    syncButtons();
  }

  // A dashed arc from one element to another: it draws, then fades out. The path is in stage
  // coordinates, the same space as the matrices, so it moves with them when the page scrolls.
  function drawTrail(fromEl, toEl, cls, sig) {
    if (reducedMotion()) return;
    const sr = stage.getBoundingClientRect();
    const rel = (node) => { const r = node.getBoundingClientRect(); return { left: r.left - sr.left, top: r.top - sr.top, width: r.width, height: r.height }; };
    const path = s('path', { d: arcPathD(rel(fromEl), rel(toEl)), class: cls });
    // Each arc is its own small layer, sized to the arc's box, so a frame repaints only that box.
    const svg = s('svg', { class: 'fxd-trail', 'aria-hidden': 'true' }, path);
    trailLayer.append(svg);
    const box = path.getBBox(), m = 6;
    svg.setAttribute('viewBox', `${box.x - m} ${box.y - m} ${box.width + 2 * m} ${box.height + 2 * m}`);
    Object.assign(svg.style, { left: `${box.x - m}px`, top: `${box.y - m}px`, width: `${box.width + 2 * m}px`, height: `${box.height + 2 * m}px` });
    drawPath(path, { duration: dur(460), signal: sig }).then((ok) => {
      if (!ok) { svg.remove(); return; }
      svg.classList.add('fxd-trail-out');
      const t = setTimeout(() => { fades.delete(t); svg.remove(); }, 520);
      fades.add(t);
    });
  }

  function flyToken(i, sig) {
    const id = ids[i], src = weStrips[id], sl = slots[i];
    const row = weRow(id);
    drawTrail(src.canvas, sl.el, 'fxd-trail-tok', sig);
    setStatus(`lookup · W_E[${id}] “${labels[i]}” → x₀[${i}]`);
    return flyClone(src.canvas, sl.el, {
      duration: dur(640), easing: easing.smooth, signal: sig,
      onDone: () => {
        sl.strip.update(row, { animate: false, range: rE });
        sl.el.classList.add('fxd-filled');
        sl.strip.pulse();
        ticker(sl.norm, 0, norm(row), { format: normText, duration: dur(420), signal: sig });
      },
    });
  }

  function flyPos(i, sig) {
    const sl = slots[i], src = posStrips[i];
    const sum = out.data.subarray(i * d, (i + 1) * d);
    const nTok = norm(weRow(ids[i])), nSum = norm(sum);
    drawTrail(src.canvas, sl.el, 'fxd-trail-pos', sig);
    setStatus(`add · P[${i}] → x₀[${i}] (“${labels[i]}”)`);
    let merged = false;
    // The positional row dissolves into the slot as it lands, so the sum appears in place.
    const merge = (animate) => {
      if (merged) return;
      merged = true;
      sl.strip.update(sum, { animate, range: rX, duration: dur(420) });
    };
    return flyClone(src.canvas, sl.el, {
      duration: dur(640), easing: easing.smooth, signal: sig, fadeOut: 0.4,
      onProgress: (e) => { if (e >= 0.6) merge(true); },
      onDone: () => {
        merge(false);
        sl.strip.pulse();
        ticker(sl.norm, nTok, nSum, { format: normText, duration: dur(480), signal: sig });
      },
    });
  }

  function complete() {
    busy = false;
    ac = null;
    finished = true;
    setStatus(finalText());
    syncButtons();
  }

  // A flight failed (the error is in the console): show the final values, not a half-filled stage.
  function recover() {
    skip();
    setStatus('The animation stopped early (the error is in the console). The final values are shown.');
  }

  async function run() {
    if (busy) return;
    stopRun();
    resetSlots();
    finished = false;
    const ctrl = new AbortController();
    ac = ctrl;
    const sig = ctrl.signal;
    busy = true;
    syncButtons();
    const idx = slots.map((_, i) => i);
    const tokensOk = await stagger(idx, (i) => flyToken(i, sig), dur(gap), { signal: sig });
    if (sig.aborted) return;
    if (!tokensOk) { recover(); return; }
    if (!pos) { complete(); return; }
    setStatus('Positional rows P[t] fly in. Each one merges into its token row, which becomes x₀[t].');
    const posOk = await stagger(idx, (i) => flyPos(i, sig), dur(gap), { signal: sig });
    if (sig.aborted) return;
    if (!posOk) { recover(); return; }
    complete();
  }
  const startRun = () => { run().catch((err) => console.error('[demo] embedding lookup failed', err)); };

  // Marks the row of the selected position.
  function markCurrent() {
    const cur = ctx.state.sel.token;
    slots.forEach((sl, i) => sl.row.classList.toggle('fxd-cur', i === cur));
  }
  markCurrent();

  setStatus(idleText);
  syncButtons();

  return {
    el,
    select: markCurrent,
    destroy() {
      stopRun();
      strips.splice(0).forEach((sp) => sp.destroy());
    },
  };
}

function matHead(title, key, dims, note = null) {
  return h('div', { class: 'fxd-mat-head' },
    h('span', { class: 'fxd-key' }, title),
    key && h('span', { class: 'fxd-subkey' }, key),
    shapeBadge(dims),
    note && h('span', { class: 'fxd-subkey' }, note));
}

// ----------------------------------------------------- 2. residual stream

function residualBlock(ctx) {
  const st = ctx.state, cfg = st.config, d = cfg.d_model, L = cfg.n_layer;
  const life = new AbortController();
  const stages = [{ key: 'emb.out', name: 'x₀', sub: 'emb', layer: -1 }];
  for (let i = 0; i < L; i++) {
    stages.push({ key: `L${i}.resid_mid`, name: `L${i}`, sub: 'mid', layer: i });
    stages.push({ key: `L${i}.resid_out`, name: `L${i}`, sub: 'out', layer: i });
  }
  const cell = clamp(Math.floor(230 / d), 2, 6);
  const strips = [];
  const prevNorm = stages.map(() => null);
  let lastLayer = null;

  const cols = stages.map((stg, i) => {
    const sp = vectorStrip({ values: zeros(d), orientation: 'v', cell, thickness: 16, pad: 6, mode: 'diverging', range: 1, ariaLabel: `${stg.key}, ${d} dimensions` });
    strips.push(sp);
    const normEl = h('span', { class: 'fxd-norm' }, '—');
    const head = h('button', { type: 'button', class: 'fxd-col-head', title: `Open ${stg.key} in the inspector`, onClick: () => ctx.inspect(stg.key) },
      h('span', null, stg.name), h('span', { class: 'fxd-col-sub' }, stg.sub));
    const wrap = h('div', { class: ['fxd-col', i > 0 && (i - 1) % 2 === 0 ? 'fxd-layer-start' : ''] }, head, sp.el, normEl);
    return { wrap, strip: sp, norm: normEl, stg };
  });

  const layerSeg = segmented({
    options: Array.from({ length: L }, (_, i) => ({ value: i, label: `L${i}` })),
    value: st.sel.layer, ariaLabel: 'Layer', onChange: (v) => ctx.store.select({ layer: v }),
  });
  const labels = st.ids.map((id) => st.tokenizer.label(id));
  const tokSel = tokenStrip({ labels, selected: st.sel.token, onSelect: (i) => ctx.store.select({ token: i }) });
  const readout = h('p', { class: 'fxd-readout', 'aria-live': 'polite' });
  const colsEl = h('div', { class: 'fxd-cols' }, cols.map((c) => c.wrap));
  const el = h('div', { class: 'fxd-block' },
    h('div', { class: 'row' }, h('span', { class: 'eyebrow' }, 'layer'), layerSeg.el),
    tokSel.el,
    h('div', { class: 'scroll-x' }, colsEl),
    readout);

  // Every stage of the selected token, on one colour scale (the largest |value| of the token).
  function update(animate) {
    const sel = ctx.state.sel, t = sel.token, tr = ctx.state.trace;
    const rows = stages.map((stg) => row2(tr[stg.key], t));
    let R = 0;
    for (const r of rows) if (r) for (let j = 0; j < r.length; j++) R = Math.max(R, Math.abs(r[j]));
    R = R || 1;
    cols.forEach((c, i) => {
      const r = rows[i];
      if (!r) return;
      c.strip.update(r, { animate, range: R, duration: 320 });
      const nv = norm(r);
      ticker(c.norm, animate && prevNorm[i] != null ? prevNorm[i] : nv, nv, { format: normText, duration: 360, signal: life.signal });
      prevNorm[i] = nv;
      c.wrap.classList.toggle('fxd-on', c.stg.layer === sel.layer);
    });
    if (animate && lastLayer !== null && lastLayer !== sel.layer) {
      cols.forEach((c) => { if (c.stg.layer === sel.layer) c.strip.pulse(); });
    }
    lastLayer = sel.layer;
    layerSeg.set(sel.layer);
    tokSel.set(t);
    const lab = labels[t] ?? '';
    const placeholderNote = isPlaceholder(ctx.state) ? 'No input text, so this is the placeholder token (id 0). ' : '';
    readout.textContent = `${placeholderNote}Token “${lab}” at position ${t}. Mid is after the attention sub-layer and out is after the MLP. Colour scale ±${fmt(R, 2)}, shared by every stage of this token. Click a stage name to inspect it.`;
  }

  update(false);

  return {
    el,
    select() { update(true); },
    destroy() {
      life.abort();
      strips.forEach((sp) => sp.destroy());
    },
  };
}

// ------------------------------------------------- 3. one head, seven steps

function attentionBlock(ctx, width) {
  const st = ctx.state, cfg = st.config, H = cfg.n_head, d = cfg.d_model;
  const dh = ctx.engine.dHead(cfg);
  const T = st.ids.length;
  const rope = cfg.pos_encoding === 'rope';
  const alibi = cfg.pos_encoding === 'alibi';
  const post = cfg.norm_placement === 'post';
  const scaleOn = cfg.attn_scale;
  const resid = cfg.residual;
  const normName = cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LN';
  const inpText = post ? 'x' : `${normName}(x)`;                  // what the query reads
  const inpHtml = post ? 'x<sub>t</sub>' : `${normName}(x<sub>t</sub>)`;
  const biasHtml = cfg.bias ? ' + b<sub>Q</sub>' : '';
  const labels = st.ids.map((id) => st.tokenizer.label(id));
  const strips = [];
  const make = (o) => { const sp = vectorStrip(o); strips.push(sp); return sp; };

  const wide = width >= 820;
  const stageW = wide ? Math.floor(width * 0.6) : width;
  const cellFor = (n, max) => clamp(Math.floor((stageW - 120) / Math.max(1, n)), 6, max);
  const dhCell = cellFor(dh, 22), tCell = cellFor(T, 22), dCell = cellFor(d, 14);

  // The data behind the seven steps: rows of the trace for the selected layer, head and token.
  function compute() {
    const sel = ctx.state.sel, L = sel.layer, hd = sel.head, t = sel.token, tr = ctx.state.trace;
    const T3 = (name) => tr[`L${L}.${name}`] ?? null;
    const q = row3(T3(rope ? 'attn.q_rot' : 'attn.q_heads'), hd, t) ?? zeros(dh);
    const sc = row3(T3('attn.scores'), hd, t) ?? zeros(T);
    const scR = row3(T3('attn.scaled'), hd, t);
    const biR = row3(T3('attn.biased'), hd, t);
    const pre = biR ?? scR ?? sc;
    const masked = row3(T3('attn.masked'), hd, t);
    const probs = row3(T3('attn.probs'), hd, t) ?? zeros(T);
    const ctxv = row3(T3('attn.ctx'), hd, t) ?? zeros(dh);
    const out = row2(T3('attn.out'), t) ?? zeros(d);
    const x = (L === 0 ? row2(tr['emb.out'], t) : row2(tr[`L${L - 1}.resid_out`], t)) ?? zeros(d);
    const inp = post ? x : (row2(T3('ln1'), t) ?? x);      // pre-norm: the query reads LN(x)
    const sum = post
      ? (row2(T3('add1'), t) ?? zeros(d))
      : (resid ? x.map((v, j) => v + out[j]) : Float64Array.from(out));
    let maxS = -Infinity, argS = 0;
    for (let j = 0; j < sc.length; j++) if (sc[j] > maxS) { maxS = sc[j]; argS = j; }
    let maxP = -Infinity, argP = 0, sumP = 0;
    for (let j = 0; j < probs.length; j++) {
      sumP += probs[j];
      if (probs[j] > maxP) { maxP = probs[j]; argP = j; }
    }
    let hidden = 0;
    if (masked) for (let j = 0; j < masked.length; j++) if (masked[j] === -Infinity) hidden++;
    return {
      q, sc, pre, masked, probs, ctx: ctxv, out, x, inp, sum,
      info: { hd, t, L, maxS, argS, maxP, argP, sumP, hidden, sqrt: Math.sqrt(dh) },
    };
  }
  let D = compute();

  // Panel builders. Each panel owns its strips; paint(p) draws the step at progress p.
  const panelEls = [];
  const lane = (label, note, sp) => h('div', { class: 'fxd-lane' },
    h('div', { class: 'fxd-lane-h' }, h('span', { class: 'fxd-lane-label' }, label), note ?? null), sp.el);
  const beforeLane = (label, sp) => h('div', { class: 'fxd-lane fxd-before' },
    h('div', { class: 'fxd-lane-h' }, h('span', { class: 'fxd-lane-label' }, label), h('span', { class: 'fxd-lane-note' }, 'before')), sp.el);
  const laneNote = (text) => h('span', { class: 'fxd-lane-note' }, text);

  // Panel 0: the query. Pre-norm attention reads the normalised input LN(x).
  const gInp = make({ values: zeros(d), orientation: 'h', cell: dCell, pad: 8, mode: 'diverging', ariaLabel: 'input the query reads' });
  const sQ = make({ values: zeros(dh), orientation: 'h', cell: dhCell, pad: 8, mode: 'diverging', ariaLabel: 'query vector' });
  const qLabel = rope ? `q′ = R(t θ) W_Q ${inpText}` : `q = W_Q ${inpText}`;
  const p0 = h('div', { class: 'fxd-panel' }, beforeLane(inpText, gInp), lane(qLabel, laneNote(`length d_h = ${dh}`), sQ));

  // Panel 1: raw scores.
  const sS = make({ values: zeros(T), orientation: 'h', cell: tCell, pad: 8, mode: 'diverging', labels, ariaLabel: 'raw scores' });
  const p1 = h('div', { class: 'fxd-panel' }, lane(rope ? 's = q′ · k′' : 's = q · kⱼ', laneNote('one score per key j'), sS));

  // Panel 2: scale, and the ALiBi bias when it is on.
  const scaleLabel = `${scaleOn ? 's / √d_h' : 's'}${alibi ? ' − ALiBi' : ''}`;
  const scaleNoteText = scaleOn ? `√${dh} = ${Math.sqrt(dh).toFixed(2)}` : 'attn_scale off';
  const gSc = make({ values: zeros(T), orientation: 'h', cell: tCell, pad: 8, mode: 'diverging', ariaLabel: 'raw scores before scaling' });
  const sA = make({ values: zeros(T), orientation: 'h', cell: tCell, pad: 8, mode: 'diverging', labels, ariaLabel: 'scaled scores' });
  const p2 = h('div', { class: 'fxd-panel' }, beforeLane('s', gSc), lane(scaleLabel, laneNote(scaleNoteText), sA));

  // Panel 3: mask.
  const gPre = make({ values: zeros(T), orientation: 'h', cell: tCell, pad: 8, mode: 'diverging', ariaLabel: 'scores before the mask' });
  const sM = make({ values: zeros(T), orientation: 'h', cell: tCell, pad: 8, mode: 'diverging', labels, ariaLabel: 'masked scores' });
  const p3 = h('div', { class: 'fxd-panel' }, beforeLane(scaleLabel, gPre), lane('mask', laneNote(maskNote(cfg)), sM));

  // Panel 4: softmax over each row.
  const gMask = make({ values: zeros(T), orientation: 'h', cell: tCell, pad: 8, mode: 'diverging', ariaLabel: 'masked scores before softmax' });
  const sP = make({ values: zeros(T), orientation: 'h', cell: tCell, pad: 8, mode: 'sequential', kind: 'attn', range: 1, labels, ariaLabel: 'attention weights' });
  const sumNote = laneNote('Σ = 0.000');
  const p4 = h('div', { class: 'fxd-panel' }, beforeLane('masked s', gMask), lane('softmax(s)', sumNote, sP));

  // Panel 5: mix the value vectors.
  const gProbs = make({ values: zeros(T), orientation: 'h', cell: tCell, pad: 8, mode: 'sequential', kind: 'attn', range: 1, ariaLabel: 'attention weights before the mix' });
  const sC = make({ values: zeros(dh), orientation: 'h', cell: dhCell, pad: 8, mode: 'diverging', ariaLabel: 'mixed values' });
  const p5 = h('div', { class: 'fxd-panel' }, beforeLane('a', gProbs), lane('ctx = Σⱼ aⱼ vⱼ', laneNote(`length d_h = ${dh}`), sC));

  // Panel 6: project with W_O and add to the residual stream.
  const gCtx = make({ values: zeros(dh), orientation: 'h', cell: dhCell, pad: 8, mode: 'diverging', ariaLabel: 'head output before W_O' });
  const sX = make({ values: zeros(d), orientation: 'h', cell: dCell, pad: 8, mode: 'diverging', ariaLabel: 'residual input' });
  const sO = make({ values: zeros(d), orientation: 'h', cell: dCell, pad: 8, mode: 'diverging', ariaLabel: 'attention output' });
  const sSum = make({ values: zeros(d), orientation: 'h', cell: dCell, pad: 8, mode: 'diverging', ariaLabel: 'residual after attention' });
  const sumLabel = resid ? (post ? 'x + W_O·ctx' : 'x_mid = x + W_O·ctx') : (post ? 'W_O·ctx' : 'x_mid = W_O·ctx');
  const sumNoteText = post ? 'before LN' : resid ? 'residual add' : 'residual off';
  const p6 = h('div', { class: 'fxd-panel' },
    lane('x', laneNote('residual stream, raw'), sX),
    beforeLane('ctx', gCtx),
    lane('W_O · ctx', laneNote(H === 1 ? `the one head, d = ${d}` : `all ${H} heads, d = ${d}`), sO),
    lane(sumLabel, laneNote(sumNoteText), sSum));
  panelEls.push(p0, p1, p2, p3, p4, p5, p6);

  // Paint functions. Each one reads D, the data for the current selection.
  const P = [
    (p) => {
      gInp.update(D.inp, { animate: false, range: maxAbs(D.inp) });
      sQ.update(scaled(D.q, p), { animate: false, range: maxAbs(D.q) });
    },
    (p) => sS.update(scaled(D.sc, p), { animate: false, range: maxAbs(D.sc) }),
    (p) => {
      gSc.update(D.sc, { animate: false, range: maxAbs(D.sc) });
      sA.update(Float64Array.from(D.sc, (a, j) => a + (D.pre[j] - a) * p), { animate: false, range: maxAbs(D.pre) });
    },
    (p) => {
      const R = maxAbs(D.pre), v = new Float64Array(D.pre.length);
      for (let j = 0; j < v.length; j++) {
        const hidden = D.masked ? D.masked[j] === -Infinity : false;
        if (!hidden) v[j] = D.pre[j];
        else v[j] = p >= 1 ? -Infinity : D.pre[j] + (-R - D.pre[j]) * p;   // slides to the bottom, then hatches
      }
      gPre.update(D.pre, { animate: false, range: R });
      sM.update(v, { animate: false, range: R });
    },
    (p) => {
      gMask.update(D.masked ?? D.pre, { animate: false, range: maxAbs(D.pre) });
      sP.update(scaled(D.probs, p), { animate: false, range: 1 });
      sumNote.textContent = `Σ = ${fmt(D.info.sumP * p, 3)}`;
    },
    (p) => {
      gProbs.update(D.probs, { animate: false, range: 1 });
      sC.update(scaled(D.ctx, p), { animate: false, range: maxAbs(D.ctx) });
    },
    (p) => {
      const base = resid ? D.x : zeros(d);
      const R = Math.max(maxAbs(D.x), maxAbs(D.out), maxAbs(D.sum));
      gCtx.update(D.ctx, { animate: false, range: maxAbs(D.ctx) });
      sX.update(D.x, { animate: false, range: R });
      sO.update(scaled(D.out, p), { animate: false, range: R });
      sSum.update(Float64Array.from(D.sum, (v, j) => base[j] + (v - base[j]) * p), { animate: false, range: R });
    },
  ];

  // Captions. Every number is computed from D.
  const lab = (j) => `“${labels[j] ?? ''}”`;
  function copy(k) {
    const I = D.info, t = I.t, hd = I.hd;
    const kvNote = cfg.n_kv_head < H
      ? ` It reads KV head ${Math.floor(hd / (H / cfg.n_kv_head))}, shared by ${H / cfg.n_kv_head} query heads.` : '';
    switch (k) {
      case 0: return {
        title: 'Query', dims: [['d_h', dh]],
        eq: rope ? `q′<sub>t</sub> = R(t·θ) (W<sub>Q</sub> ${inpHtml}${biasHtml})` : `q<sub>t</sub> = W<sub>Q</sub> ${inpHtml}${biasHtml}`,
        text: `${isPlaceholder(ctx.state) ? 'No input text, so this is the placeholder token (id 0). ' : ''}Head ${hd}, position ${t} (${lab(t)}): one query of length d_h = ${dh}${rope ? ', rotated by RoPE' : ''}. ${post
          ? 'This is post-norm, so the query reads the raw input x.'
          : `This is pre-norm, so the query reads ${normName}(x), the normalised input, not the raw residual stream.`}`,
      };
      case 1: return {
        title: 'Scores', dims: [['T', T]],
        eq: rope ? 's<sub>tj</sub> = q′<sub>t</sub> · k′<sub>j</sub>' : 's<sub>tj</sub> = q<sub>t</sub> · k<sub>j</sub>',
        text: `The dot product with every key, future positions included. Largest raw score ${fmt(I.maxS, 2)} at position ${I.argS}${I.argS > t ? ', a future position' : ''}.`,
      };
      case 2: {
        const parts = [scaleOn
          ? `Divide by √${dh} = ${I.sqrt.toFixed(2)} so the logits stay near unit variance.`
          : 'Scaling is off (attn_scale = false), so the raw dot products pass through.'];
        let eq;
        if (scaleOn && alibi) eq = 's<sub>tj</sub> / √d<sub>h</sub> − m<sub>h</sub>·|t − j|';
        else if (scaleOn) eq = 's<sub>tj</sub> / √d<sub>h</sub>';
        else if (alibi) eq = 's<sub>tj</sub> − m<sub>h</sub>·|t − j|';
        else eq = 's<sub>tj</sub>';
        if (alibi) {
          const m = ctx.engine.alibiSlopes(H)[hd];
          parts.push(`Then subtract the ALiBi distance penalty m_h·|t − j|, with slope m_h = ${fmt(m, 4)} for head ${hd}.`);
        }
        const title = scaleOn ? (alibi ? 'Scale and bias' : 'Scale') : (alibi ? 'Bias' : 'Scale (off)');
        return { title, dims: [['T', T]], eq, text: parts.join(' ') };
      }
      case 3: return {
        title: 'Mask', dims: [['T', T]],
        eq: maskEquation(cfg),
        text: cfg.mask === 'none'
          ? 'mask = none, so every key stays visible.'
          : `mask = ${cfg.mask}${cfg.mask === 'sliding' ? ` (window ${cfg.window})` : cfg.mask === 'prefix' ? ` (prefix_len ${cfg.prefix_len})` : ''}: ${I.hidden} key${I.hidden === 1 ? '' : 's'} hidden for this query. They are set to −∞ and drawn hatched.`,
      };
      case 4: return {
        title: 'Softmax', dims: [['T', T]],
        eq: 'a<sub>tj</sub> = exp(s<sub>tj</sub>) / Σ<sub>k</sub> exp(s<sub>tk</sub>)',
        text: `Each row is normalised. The weights sum to ${fmt(I.sumP, 3)}. Largest ${fmt(I.maxP, 3)} on ${lab(I.argP)} (position ${I.argP}).`,
      };
      case 5: return {
        title: 'Mix values', dims: [['d_h', dh]],
        eq: 'ctx<sub>t</sub> = Σ<sub>j</sub> a<sub>tj</sub> v<sub>j</sub>',
        text: `A weighted average of the value vectors, length d_h = ${dh}.${kvNote} This head writes one slice of the concatenated output.`,
      };
      default: {
        let eq;
        if (post) eq = resid ? `x<sub>mid</sub> = ${normName}(x + W<sub>O</sub> · concat<sub>h</sub>(ctx<sub>h</sub>))` : `x<sub>mid</sub> = ${normName}(W<sub>O</sub> · concat<sub>h</sub>(ctx<sub>h</sub>))`;
        else eq = resid ? 'x<sub>mid</sub> = x + W<sub>O</sub> · concat<sub>h</sub>(ctx<sub>h</sub>)' : 'x<sub>mid</sub> = W<sub>O</sub> · concat<sub>h</sub>(ctx<sub>h</sub>)';
        const heads = H === 1 ? 'The output of the one head' : `The output of all ${H} heads`;
        const tail = post
          ? (resid ? `With post-norm, ${normName} then runs on the sum.` : `Residual off: the output replaces the stream, then ${normName} runs.`)
          : (resid ? 'It is added to the residual stream.' : 'Residual off: the output replaces the stream.');
        return { title: 'Project and add', dims: [['d', d]], eq, text: `${heads} goes through W_O into d = ${d}. ${tail}` };
      }
    }
  }

  // Caption card (left) and stage (right).
  const capNo = h('span', { class: 'eyebrow' });
  const capTitle = h('h3', { class: 'fxd-cap-title' });
  const capEq = h('div');
  const capText = h('p', { class: 'fxd-cap-text' });
  const capDims = h('div', { class: 'row' });
  const capCard = h('div', { class: 'fxd-cap' }, capNo, capTitle, capEq, capText, capDims);
  const stageEl = h('div', { class: 'fxd-stage-att' }, panelEls);

  function showStep(k) {
    panelEls.forEach((pel, j) => { pel.hidden = j !== k; });
    const c = copy(k);
    capNo.textContent = `Step ${k + 1} of 7`;
    capTitle.textContent = c.title;
    capEq.replaceChildren(formula(c.eq));
    capText.textContent = c.text;
    capDims.replaceChildren(shapeBadge(c.dims));
  }

  // The sequence: each step's run(progress) calls its paint function.
  const stepDefs = P.map((paint, k) => ({ label: copy(k).title, duration: BASE_STEP_MS, run: (p) => paint(p) }));
  const ctl = sequence(stepDefs, { onStep: (k) => showStep(k) });
  const bar = sequencePlaybar(ctl, { labels: stepDefs.map((step) => step.label) });

  // Selection controls. Their values are kept in sync when the selection changes elsewhere.
  const sel0 = st.sel;
  const layerSeg = segmented({
    options: Array.from({ length: cfg.n_layer }, (_, i) => ({ value: i, label: `L${i}` })),
    value: sel0.layer, ariaLabel: 'Layer', onChange: (v) => ctx.store.select({ layer: v }),
  });
  const headSeg = segmented({
    options: Array.from({ length: H }, (_, i) => ({ value: i, label: `h${i}` })),
    value: sel0.head, ariaLabel: 'Head', onChange: (v) => ctx.store.select({ head: v }),
  });
  const tokSel = tokenStrip({ labels, selected: sel0.token, onSelect: (i) => ctx.store.select({ token: i }) });
  const speedSeg = segmented({
    options: [{ value: 2, label: '0.5×', title: 'Half speed' }, { value: 1, label: '1×' }, { value: 0.5, label: '2×', title: 'Double speed' }],
    value: 1, ariaLabel: 'Animation speed', onChange: (v) => { ctl.steps.forEach((step) => { step.duration = BASE_STEP_MS * v; }); },
  });
  const el = h('div', { class: 'fxd-block' },
    h('div', { class: 'row' }, h('span', { class: 'eyebrow' }, 'layer'), layerSeg.el, h('span', { class: 'eyebrow' }, 'head'), headSeg.el),
    h('div', { class: 'row' }, h('span', { class: 'eyebrow' }, 'token'), tokSel.el),
    h('div', { class: 'row' }, bar.el, h('span', { class: 'eyebrow' }, 'speed'), speedSeg.el),
    h('div', { class: 'fxd-att' }, capCard, stageEl));

  showStep(ctl.index);

  return {
    el,
    // Selection changed: recompute the data for the same step, keeping the playback position.
    select() {
      const sel = ctx.state.sel;
      layerSeg.set(sel.layer);
      headSeg.set(sel.head);
      tokSel.set(sel.token);
      D = compute();
      ctl.refresh();
      showStep(ctl.index);
    },
    destroy() {
      ctl.destroy();
      bar.destroy();
      strips.forEach((sp) => sp.destroy());
    },
  };
}

// ------------------------------------------------------------- the page

export default {
  id: '_demo_flow',
  title: 'Demo: flow & motion kit',
  mount(root, ctx) {
    injectCss();
    const mounts = { emb: h('div'), res: h('div'), att: h('div') };
    const blocks = { emb: null, res: null, att: null };
    const embSec = sectionWithLede('Embedding lookup', embeddingLede(ctx.state.config), mounts.emb);

    root.append(
      viewHeader({
        eyebrow: 'Demo · motion kit',
        title: 'Flow and motion',
        lede: 'Three demos of <code>src/ui/flow.js</code> on the live forward pass: an embedding lookup, the residual stream across depth, and one attention head as seven steps.',
      }),
      embSec.el,
      sectionWithLede('Residual stream across depth', 'The residual stream at every depth for the selected token. Change the layer or the token and the strips move.', mounts.res).el,
      sectionWithLede('One attention head, seven steps', 'One query token, one head. Each step draws a row of the trace. The captions are computed from the same numbers.', mounts.att).el,
      callout('note', h('strong', null, 'Kit: '),
        'vectorStrip, flyClone, ticker, drawPath, sequence, sequencePlaybar, stagger, arcPathD, easing, reducedMotion. Every animation returns a cancellable Promise, honours reduced motion and redraws on theme change.'),
    );

    function rebuildAll() {
      Object.values(blocks).forEach((b) => b?.destroy());
      embSec.lede.innerHTML = embeddingLede(ctx.state.config);
      blocks.emb = embeddingBlock(ctx, mounts.emb.clientWidth || 900);
      mounts.emb.replaceChildren(blocks.emb.el);
      blocks.res = residualBlock(ctx);
      mounts.res.replaceChildren(blocks.res.el);
      blocks.att = attentionBlock(ctx, mounts.att.clientWidth || 900);
      mounts.att.replaceChildren(blocks.att.el);
    }

    ctx.on('trace', rebuildAll);
    ctx.on('select', () => { blocks.emb?.select(); blocks.res?.select(); blocks.att?.select(); });
    rebuildAll();

    let lastW = mounts.emb.clientWidth, resizeTimer = null;
    const onResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        const w = mounts.emb.clientWidth;
        if (Math.abs(w - lastW) > 40) { lastW = w; rebuildAll(); }
      }, 250);
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      clearTimeout(resizeTimer);
      Object.values(blocks).forEach((b) => b?.destroy());
    };
  },
};

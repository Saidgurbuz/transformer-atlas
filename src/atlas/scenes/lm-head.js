// LM head: how a vector becomes a guess for the next token. For the selected position t, its final
// vector h_t is dotted with every row of W_U, giving one logit per vocabulary entry. Softmax turns the
// logits into probabilities. The next input token is highlighted with its probability and its loss,
// and the mean loss and perplexity are shown over the sequence. Steps: final vector, dot with every
// token, logits, probabilities. The probability bars are the zoom target into sampling.
// Structure follows src/atlas/scenes/softmax.js; shared parts come from kit.js.
//
// Pipeline (design units): h_t strip -> x -> W_U plate -> logit bars -> softmax -> probability bars.
// The h_t strip and the operations are untilted; only the W_U plate carries the kit tilt, hinged at its
// top edge so the columns stay under the strip.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, chip, checkBadge, op, appear, pulse, useStyle, MOTION } from '../kit.js';
import { h, fmt, num, texHTML, clamp, diverging, sequential, onThemeChange, reducedMotion } from '../../ui/index.js';

const STEPS = ['Final vector', 'Dot with every token', 'Logits', 'Probabilities'];

// Geometry. Columns are 5 units wide with a 1-unit gap (pitch 6). Rows have pitch 12 up to 32 rows;
// beyond that the rows share ROWS_SPAN units so the plate stays on the canvas.
const CELL_W = 5, COL_GAP = 1, ELL_W = 12;
const ROW_GAP = 2, PITCH_MAX = 12, ROWS_SPAN = 384;
const PAD = 10, LABEL_W = 30, LABEL_GAP = 8, GUTTER = LABEL_W + LABEL_GAP;
const MAX_COLS = 64, MAX_ROWS = 40;              // longer axes: first entries, an ellipsis, the last 4
const TRACK_W = 290, VAL_W = 64, VAL_GAP = 10;
const TRACK_X = GUTTER;                          // where the bar track starts inside the bar panel
const TILT = 12;                                 // degrees: W_U is a tilted plane, hinged at its top edge
const LABEL_CLEAR = 14;                          // a probability is written only if its row is this far from the last one

// Placement on the 10-unit grid (design units)
const BODY_X = 110, CELLS_X = BODY_X + 1 + PAD + GUTTER;     // first column of W_U and h_t: x 159
const H_Y = 260, H_H = 20, H_BOT = H_Y + H_H;               // the h_t strip: 260 .. 280
const OP_D = 48, OP_CY = 324, OP_TOP = OP_CY - OP_D / 2, OP_BOT = OP_CY + OP_D / 2;   // the times node: 300 .. 348
const PLATE_Y = 368;                                         // the W_U plate top (772 bottom at 32 rows)
const PROB_X = 600, PROB_Y = 260, PROB_W = 420, PROB_H = 512;// bars: logits, then probabilities
const SM_X = 1094, SM_Y = 570, SM_D = 64;                    // softmax node, right of the bars
const SIDE_X = 1180, SIDE_Y = 260, CAP_Y = 800;   // below the plate's shadow; two lines end near 857
const SWEEP_TRANS = 240, MORPH_TRANS = 600, SWEEP_MS = 46, SWEEP_BUDGET = 1100;
const ZOOM_MS = 800;                             // the zoom into this scene takes about 760 ms

useStyle('lm-head', `
.lh-body { position: relative; padding: ${PAD}px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 72%, transparent);
  border: 1px solid color-mix(in srgb, var(--c-head) 40%, transparent);
  box-shadow: 0 14px 34px rgba(0, 0, 0, .16), 0 2px 0 color-mix(in srgb, var(--c-head) 22%, transparent);
  transform: perspective(1100px) rotateX(${TILT}deg); transform-origin: 50% 0%; transition: opacity .4s ease; }
.lh-body.dim { opacity: .35; }
.lh-row { display: flex; align-items: center; height: var(--lh-ch, 10px); }
.lh-row + .lh-row { margin-top: ${ROW_GAP}px; }
.lh-row.ell .lh-lab { color: var(--fg-3); }
.lh-lab { flex: none; width: ${LABEL_W}px; margin-right: ${LABEL_GAP}px; text-align: right; font: 600 var(--lh-fs, 15px)/var(--lh-ch, 10px) var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.lh-cells { display: flex; gap: ${COL_GAP}px; }
.lh-c { flex: none; width: ${CELL_W}px; height: var(--lh-ch, 10px); border-radius: 2px; transition: background-color .5s ease; }
.lh-c.ell { width: ${ELL_W}px; background: none !important; font: 600 11px/var(--lh-ch, 10px) var(--font-mono); color: var(--fg-3); text-align: center; }
.lh-ellrow { display: block; font: 600 14px/var(--lh-ch, 10px) var(--font-mono); color: var(--fg-3); text-align: center; }
.lh-hstrip { display: flex; gap: ${COL_GAP}px; height: ${H_H}px; border-radius: 4px; transition: opacity .4s ease; }
.lh-hstrip .lh-c { height: ${H_H}px; }
.lh-hstrip.focus { outline: 2px solid var(--accent); outline-offset: 6px; }
.lh-hlab { font-size: 18px; color: var(--c-resid); text-align: right; white-space: nowrap; }
.lh-hvals { font: 500 12px/14px var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.lh-op { transition: opacity .4s ease; }
.lh-op.lh-sm { font-size: 12px; }
.lh-arrow { transition: opacity .4s ease; }
.lh-wlab { text-align: right; font: 600 15px/1.25 var(--font-mono); color: var(--c-head); white-space: nowrap; transition: opacity .4s ease; }
.lh-wlab small { display: block; font: 500 12px/1.3 var(--font-mono); color: var(--fg-3); }
.lh-probs { position: absolute; width: ${PROB_W}px; height: ${PROB_H}px; border-radius: 14px; border: 1px solid transparent;
  transition: transform .25s ease, background-color .25s ease, border-color .25s ease, box-shadow .25s ease; }
.lh-probs[data-node]:hover { transform: translateY(-3px); background: color-mix(in srgb, var(--surface) 80%, transparent);
  border-color: color-mix(in srgb, var(--c-head) 45%, transparent); box-shadow: 0 16px 36px rgba(0, 0, 0, .14); }
.lh-probs-head { position: absolute; left: 12px; top: 10px; height: 20px; font: 600 15px/20px var(--font-mono); color: var(--c-head); white-space: nowrap; }
.lh-hint { position: absolute; right: 12px; top: 10px; height: 20px; font: 500 14px/20px var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.lh-bars { position: absolute; left: 12px; width: ${TRACK_X + TRACK_W + VAL_GAP + VAL_W}px; transition: opacity .4s ease; }
.lh-bars.off { opacity: .35; }
.lh-brow { position: absolute; left: 0; right: 0; display: flex; align-items: center; height: var(--lh-ch, 10px); }
.lh-track { position: relative; flex: none; width: ${TRACK_W}px; height: var(--lh-ch, 10px); border-radius: 3px; transition: background-color .4s ease; }
.lh-bars.probs .lh-track { background: color-mix(in srgb, var(--fg) 4%, transparent); }
.lh-bars.probs .lh-brow.dim .lh-track { background: color-mix(in srgb, var(--fg) 2%, transparent); }
.lh-bars.probs .lh-bar { box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--c-head) 50%, transparent); }
.lh-bar { position: absolute; top: 1px; left: 0; width: 0; height: calc(var(--lh-ch, 10px) - 2px); border-radius: 3px;
  transition: left var(--lh-dur, ${MORPH_TRANS}ms) cubic-bezier(.65,0,.35,1), width var(--lh-dur, ${MORPH_TRANS}ms) cubic-bezier(.65,0,.35,1), background-color .5s ease, opacity .4s ease; }
.lh-val { flex: none; width: ${VAL_W}px; margin-left: ${VAL_GAP}px; text-align: right; font: 500 var(--lh-vfs, 13px)/var(--lh-ch, 10px) var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.lh-zero { position: absolute; top: 0; bottom: 0; left: 0; width: 0; border-left: 1px dashed var(--line-strong);
  transition: left var(--lh-dur, ${MORPH_TRANS}ms) cubic-bezier(.65,0,.35,1); }
.lh-brow.top .lh-val { color: var(--fg); font-weight: 650; }
.lh-brow.dim .lh-bar, .lh-brow.dim .lh-lab { opacity: .4; }
.lh-brow.next .lh-bar { outline: 2px solid var(--accent); outline-offset: 1px; }
.lh-brow.next .lh-lab { color: var(--accent); }
.lh-side { width: 340px; display: grid; gap: 12px; align-content: start; }
.lh-step { font: 600 17px/1.2 var(--font-mono); color: var(--c-head); }
.lh-eq-main { font-size: 22px; min-height: 52px; overflow-x: auto; }
.lh-eq-live { font-size: 18px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.lh-note { margin-top: 6px; font: 500 13px/1.4 var(--font-mono); color: var(--fg-3); }
.lh-check-row { display: flex; flex-wrap: wrap; gap: 8px; min-height: 28px; }
.lh-read { display: grid; gap: 8px; opacity: 0; transition: opacity .4s ease; }
.lh-read-row { display: grid; grid-template-columns: 124px 1fr; gap: 10px; align-items: baseline; }
.lh-read-k { font: 500 13px/1.3 var(--font-mono); color: var(--fg-3); }
.lh-read-v { font: 600 15px/1.3 var(--font-mono); color: var(--fg); }
.lh-read-v small { font: 500 12px/1.3 var(--font-mono); color: var(--fg-3); margin-left: 4px; }
.lh-warn { font: 500 13px/1.45 var(--font-body); color: var(--warn); }
@media (prefers-reduced-motion: reduce) { .lh-bar, .lh-zero, .lh-c, .lh-bars, .lh-read, .lh-probs, .lh-body, .lh-op, .lh-arrow, .lh-wlab, .lh-hstrip { transition: none !important; } }
`);

// ------------------------------------------------------------------ pure helpers
/** Indices to draw for an axis of length n: all of them, or the first ones, -1 (ellipsis), the last 4. */
function elide(n, max) {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const tail = 4, head = max - tail - 1;
  return [...Array.from({ length: head }, (_, i) => i), -1, ...Array.from({ length: tail }, (_, i) => n - tail + i)];
}
/** Row pitch, label sizes and plate width for this many columns and rows. */
function geometry(cols, nRows) {
  const pitch = Math.min(PITCH_MAX, Math.floor(ROWS_SPAN / Math.max(1, nRows)));
  const cellsW = cols.reduce((w, c, i) => w + (c < 0 ? ELL_W : CELL_W) + (i ? COL_GAP : 0), 0);
  return {
    pitch, ch: pitch - ROW_GAP, fs: clamp(pitch + 3, 11, 15), vfs: clamp(pitch + 1, 11, 13),
    cellsW, plateW: cellsW + 2 * PAD + GUTTER + 2,
  };
}
const maxAbs = (xs) => { let m = 0; for (const x of xs) m = Math.max(m, Math.abs(x)); return m || 1; };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const pct = (p) => `${(p * 100).toFixed(1)}%`;

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    let step = 0, lastD = null, built = null, lastMain = '', alive = true;
    let layoutCols = [], layoutRows = [], geo = null;
    let hCells = [], wCells = [], bars = [], band = null, zeroEl = null;
    let flow = {}, smIn = null, smOut = null, drawn = false;
    let timers = [], auto = null, autoStepping = false;
    const stopTimers = () => { timers.forEach(clearTimeout); timers = []; };
    const stopAuto = () => { clearTimeout(auto); auto = null; };
    const lab = (v) => ctx.state.tokenizer.label(v);                       // vocabulary entry v
    const name = (v) => (lab(v) === '␣' ? 'space' : lab(v));
    const tokName = (i) => (ctx.label(i) === '␣' ? 'space' : ctx.label(i)); // the token at position i

    // ------------------------------------------------------------ shared header, picker, caption
    const header = sceneHeader(st, { chips: [], title: 'LM head' });
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'resid' });

    // ------------------------------------------------------------ the final vector h_t (untilted)
    const hStrip = h('div', { class: 'lh-hstrip' });
    const hLab = h('div', { class: 'lh-hlab', html: texHTML('h_t') });
    const hVals = h('div', { class: 'lh-hvals' });
    st.add(hStrip, { x: CELLS_X, y: H_Y });
    st.add(hLab, { x: CELLS_X - 10, y: H_Y + H_H / 2, anchor: 'rc' });
    st.add(hVals, { x: CELLS_X, y: H_BOT + 4 });

    // ------------------------------------------------------------ W_U plate (tilted) and its operation
    const body = h('div', { class: 'lh-body' });
    st.add(body, { x: BODY_X, y: PLATE_Y });
    const wLab = h('div', { class: 'lh-wlab' });
    st.add(wLab, { x: CELLS_X - 40, y: OP_CY, anchor: 'rc' });
    const timesOp = op('\\times', { kind: 'head', size: OP_D });
    timesOp.classList.add('lh-op');
    st.add(timesOp, { x: CELLS_X, y: OP_CY, anchor: 'c' });

    // ------------------------------------------------------------ bars: logits, then probabilities
    const probHead = h('div', { class: 'lh-probs-head' }, 'logits z');
    const hint = h('div', { class: 'lh-hint' }, '→ sampling');
    const barsHost = h('div', { class: 'lh-bars off' });
    const probsWrap = h('div', { class: 'lh-probs', dataset: { node: 'sampling' } }, probHead, hint, barsHost);
    st.add(probsWrap, { x: PROB_X, y: PROB_Y });

    // ------------------------------------------------------------ softmax between logits and probabilities
    const smOp = op('\\softmax', { kind: 'head', size: SM_D });
    smOp.classList.add('lh-op', 'lh-sm');
    st.add(smOp, { x: SM_X, y: SM_Y, anchor: 'c' });
    smIn = st.arrow({ x: PROB_X + PROB_W + 4, y: SM_Y - 14 }, { x: SM_X - 31, y: SM_Y - 14 }, { kind: 'head', width: 2 });
    smOut = st.arrow({ x: SM_X - 31, y: SM_Y + 14 }, { x: PROB_X + PROB_W + 4, y: SM_Y + 14 }, { kind: 'head', width: 2 });
    smIn.el.classList.add('lh-arrow');
    smOut.el.classList.add('lh-arrow');

    // ------------------------------------------------------------ right column: equations, checks, readout
    const stepLab = h('div', { class: 'lh-step' });
    const eqMain = h('div', { class: 'lh-eq-main' });
    const eqLive = h('div', { class: 'lh-eq-live' });
    const eqCheck = h('div', { class: 'lh-check-row' });
    const readout = h('div', { class: 'lh-read' });
    const side = h('div', { class: 'lh-side' }, stepLab, eqMain, eqLive, eqCheck, readout);
    st.add(side, { x: SIDE_X, y: SIDE_Y });
    const cap = sceneCaption(st, { y: CAP_Y, w: 1060 });
    // The kit cross-fades captions. Here a newer caption cancels a fade still running, so only the latest text is ever written.
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

    // ------------------------------------------------------------ data for the selected position
    function snap() {
      const cs = ctx.state, cfg = cs.config, tr = cs.trace || {};
      const lg = tr.logits, pr = tr.probs;
      const preLN = cfg.norm_placement === 'pre';
      const hKey = preLN ? 'lnf' : `L${cfg.n_layer - 1}.resid_out`;   // post-LN: the last block output
      const hT = tr[hKey];
      if (!lg || !pr || !hT) return null;
      const T = cs.ids.length, V = lg.shape[1], d = hT.shape[1];
      const tied = !!cfg.tie_weights;
      // same matrix as the engine: wte.weight when tied, lm_head.weight otherwise
      const W = (tied ? cs.model.params['wte.weight'] : cs.model.params['lm_head.weight']) ?? cs.model.params['wte.weight'];
      if (!W || W.shape[0] !== V || W.shape[1] !== d) return null;
      const t = clamp(cs.sel.token ?? T - 1, 0, T - 1);
      const z = Array.from(lg.data.subarray(t * V, (t + 1) * V));
      const p = Array.from(pr.data.subarray(t * V, (t + 1) * V));
      const hv = Array.from(hT.data.subarray(t * d, (t + 1) * d));
      const hasNext = t + 1 < T, next = hasNext ? cs.ids[t + 1] : -1;
      // mean next-token loss over the positions that have a next input token, from the traced probabilities
      const M = tr.mask;
      let nll = 0, nScored = 0, leak = false;
      for (let u = 0; u + 1 < T; u++) {
        nll -= Math.log(pr.data[u * V + cs.ids[u + 1]]);
        nScored++;
        if (M && M.data[u * T + u + 1] === 1) leak = true;     // a mask that shows the next token
      }
      const zMax = Math.max(...z), zMin = Math.min(...z);
      const zLo = Math.min(0, zMin), zHi = Math.max(0, zMax);
      const rank = [...Array(V).keys()].sort((a, b) => p[b] - p[a] || a - b);
      return {
        T, V, d, t, z, p, hv, W, hKey, preLN, rms: cfg.norm === 'rmsnorm', tied, nLayer: cfg.n_layer, ids: cs.ids,
        hasNext, next, pNext: hasNext ? p[next] : null,
        meanLoss: nScored ? nll / nScored : null, nScored, leak,
        zMax, zMin, zLo, zHi, zSpan: zHi - zLo || 1, zAbs: Math.max(-zLo, zHi) || 1,
        hRange: maxAbs(hv), wRange: maxAbs(W.data),
        rank, top5: new Set(rank.slice(0, 5)), argmaxZ: z.indexOf(zMax),
        Zs: z.reduce((s, v) => s + Math.exp(v - zMax), 0),        // softmax denominator, shifted by the max
        focus: hasNext ? next : rank[0],                          // the row the live equation shows
      };
    }

    // ------------------------------------------------------------ DOM: built when the weights or shape change
    function ensurePlate(D) {
      if (built && built.W === D.W && built.V === D.V && built.d === D.d) return;
      built = { W: D.W, V: D.V, d: D.d };
      layoutCols = elide(D.d, MAX_COLS);
      layoutRows = elide(D.V, MAX_ROWS);
      geo = geometry(layoutCols, layoutRows.length);
      st.canvas.style.setProperty('--lh-ch', `${geo.ch}px`);
      st.canvas.style.setProperty('--lh-fs', `${geo.fs}px`);
      st.canvas.style.setProperty('--lh-vfs', `${geo.vfs}px`);
      const HX = CELLS_X + geo.cellsW / 2;                 // the operation sits over the centre of the columns
      st.move(timesOp, HX, OP_CY);
      st.move(wLab, HX - 32, OP_CY);
      buildStrip();
      buildPlate(D);
      buildBars();
      buildFlow();
    }

    const wTip = (v, c, D) => `W_U[${v}, ${c}] = ${fmt(D.W.data[v * D.d + c], 3)}\n${name(v)} · row ${v} · column ${c}\n${D.tied ? 'wte.weight, tied with W_E · model.py:241' : 'lm_head.weight · model.py:278'}`;

    function buildStrip() {
      hCells = [];
      hStrip.replaceChildren(...layoutCols.map((c) => {
        if (c < 0) return h('span', { class: 'lh-c ell' }, '⋯');
        const el = h('span', { class: 'lh-c' });
        hCells.push({ c, el });
        return el;
      }));
    }

    function buildPlate(D) {
      wCells = [];
      const cellsW = `${geo.cellsW}px`;
      const rows = layoutRows.map((v) => {
        if (v < 0) return h('div', { class: ['lh-row', 'ell'] }, h('span', { class: 'lh-lab' }, '⋮'), h('span', { class: 'lh-ellrow', style: { width: cellsW } }, '⋯'));
        return h('div', { class: 'lh-row' },
          h('span', { class: 'lh-lab' }, lab(v)),
          h('div', { class: 'lh-cells' }, ...layoutCols.map((c) => {
            if (c < 0) return h('span', { class: 'lh-c ell' }, '⋯');
            const el = h('span', { class: 'lh-c', title: wTip(v, c, D) });
            wCells.push({ v, c, el });
            return el;
          })));
      });
      band = h('div', { class: 'k-row-band', hidden: true });
      body.replaceChildren(...rows, band);
    }

    function buildBars() {
      zeroEl = h('div', { class: 'lh-zero' });
      bars = layoutRows.map((v) => {
        const bar = h('div', { class: 'lh-bar' });
        const val = h('span', { class: 'lh-val' });
        const row = h('div', { class: ['lh-brow', v < 0 && 'ell'] },
          h('span', { class: 'lh-lab' }, v < 0 ? '⋮' : lab(v)),
          h('div', { class: 'lh-track' }, bar),
          val);
        return { v, row, bar, val };
      });
      barsHost.replaceChildren(...bars.map((b) => b.row), zeroEl);
    }

    /** The arrows that join the stages. Their ends depend on the plate, so they are rebuilt with it. */
    function buildFlow() {
      Object.values(flow).forEach((a) => a.destroy());
      const HX = CELLS_X + geo.cellsW / 2;
      const mid = PLATE_Y + body.offsetHeight / 2;
      flow = {
        hop: st.arrow({ x: HX, y: H_BOT }, { x: HX, y: OP_TOP }, { kind: 'resid', width: 2 }),
        opw: st.arrow({ x: HX, y: OP_BOT }, { x: HX, y: PLATE_Y }, { kind: 'head', width: 2 }),
        out: st.arrow({ x: BODY_X + geo.plateW + 4, y: mid }, { x: PROB_X - 4, y: mid }, { kind: 'head', width: 2 }),
      };
      Object.values(flow).forEach((a) => a.el.classList.add('lh-arrow'));
    }

    /** Level each bar row with its row of the tilted W_U plate: measured after the tilt, in design units. */
    function alignBars() {
      const rowEls = [...body.querySelectorAll('.lh-row')];
      if (!rowEls.length || rowEls.length !== bars.length || !body.getClientRects().length) return;  // hidden: resume() re-aligns
      const ys = rowEls.map((r) => st.center(r).y);
      barsHost.style.top = `${ys[0] - geo.ch / 2 - PROB_Y}px`;
      barsHost.style.height = `${ys[ys.length - 1] - ys[0] + geo.ch}px`;
      bars.forEach((b, i) => { b.row.style.top = `${ys[i] - ys[0]}px`; });
    }

    // ------------------------------------------------------------ painting
    function paintW(D) {
      for (const { v, c, el } of wCells) el.style.background = diverging(D.W.data[v * D.d + c] / D.wRange);
    }

    function paintH(D, animate) {
      hStrip.dataset.trace = D.hKey;
      for (const { c, el } of hCells) {
        el.style.background = diverging(D.hv[c] / D.hRange);
        el.style.transitionDelay = animate && step === 0 ? `${c * 12}ms` : '0ms';   // cascade on the first step
        el.dataset.cell = `${D.t},${c},0`;
      }
      hVals.textContent = [0, 1, 2, 3].filter((c) => c < D.d).map((c) => fmt(D.hv[c], 2)).join('   ');
    }

    /** Probability rows that write their value: the top five and the next token, largest first, skipping any that would touch a value already written. */
    function valueRows(D) {
      const rowOf = new Map(bars.map((b, i) => [b.v, i]));
      const cands = new Set(D.rank.slice(0, 5));
      if (D.hasNext) cands.add(D.next);
      const want = [...cands].sort((a, b) => D.p[b] - D.p[a] || a - b);
      const picked = new Set(), taken = [];
      for (const v of want) {
        const ri = rowOf.get(v);
        if (ri === undefined || picked.has(ri)) continue;
        if (D.p[v] < 0.0005) continue;
        if (taken.some((r) => Math.abs(r - ri) * geo.pitch < LABEL_CLEAR)) continue;
        picked.add(ri);
        taken.push(ri);
      }
      return picked;
    }

    /** mode: 'zero' (not computed), 'logit' (the first `upTo` rows are computed), 'prob' (softmax). */
    function paintBars(D, mode, upTo = Infinity) {
      const x0 = (TRACK_W * -D.zLo) / D.zSpan;          // where logit 0 sits on the track
      const xOf = (z) => x0 + (z / D.zSpan) * TRACK_W;
      const probs = mode === 'prob';
      zeroEl.style.left = `${TRACK_X + (probs ? 0 : x0)}px`;
      const writes = probs ? valueRows(D) : null;
      let k = 0;
      bars.forEach((b, ri) => {
        if (b.v < 0) { b.bar.style.width = '0px'; b.val.textContent = ''; return; }
        const on = mode !== 'zero' && k < upTo;
        k++;
        const v = b.v;
        b.row.dataset.trace = probs ? 'probs' : 'logits';
        b.row.dataset.cell = `${D.t},${v},0`;
        if (probs) {
          const q = D.p[v];                             // a fixed 0 … 100% scale, as in the attention pattern
          b.bar.style.left = '0px';
          b.bar.style.width = `${q * TRACK_W}px`;
          b.bar.style.background = sequential(q, 'head');
          const top = D.top5.has(v), next = D.hasNext && v === D.next;
          b.val.textContent = writes.has(ri) ? pct(q) : '';
          b.row.className = ['lh-brow', top && 'top', !top && 'dim', next && 'next'].filter(Boolean).join(' ');
        } else {
          const z = D.z[v];
          b.bar.style.left = `${on ? Math.min(x0, xOf(z)) : x0}px`;
          b.bar.style.width = `${on ? Math.abs(xOf(z) - x0) : 0}px`;
          b.bar.style.background = diverging(on ? z / D.zAbs : 0);
          b.val.textContent = on ? fmt(z, 2) : '';
          b.row.className = 'lh-brow';
        }
      });
    }

    function moveBand(i) {
      band.hidden = false;
      band.style.left = `${PAD - 4}px`;
      band.style.width = `${body.clientWidth - 2 * PAD + 8}px`;
      band.style.top = `${PAD + i * geo.pitch - 3}px`;
      band.style.height = `${geo.ch + 6}px`;
    }

    /** Step 1: each row of W_U is dotted with h_t in turn; its bar grows as the band passes. */
    function sweep(D) {
      const order = layoutRows.map((v, i) => ({ v, i })).filter((o) => o.v >= 0);
      const per = Math.min(SWEEP_MS, Math.floor(SWEEP_BUDGET / Math.max(1, order.length)));   // the sweep ends inside the step
      band.style.transition = `top ${SWEEP_TRANS}ms cubic-bezier(.2,.7,.2,1)`;
      barsHost.style.setProperty('--lh-dur', `${SWEEP_TRANS}ms`);
      let k = 0;
      const tick = () => {
        if (k >= order.length) {
          band.hidden = true;
          barsHost.style.setProperty('--lh-dur', `${MORPH_TRANS}ms`);
          paintBars(D, 'logit');
          renderSide(D);
          return;
        }
        moveBand(order[k].i);
        paintBars(D, 'logit', k + 1);
        renderSide(D, order[k].v);
        k++;
        timers.push(setTimeout(tick, per));
      };
      paintBars(D, 'logit', 0);
      tick();
    }

    /** The flow reads left to right and top to bottom: the stage in play is bright, the others wait. */
    function paintFlow() {
      const early = step === 0;
      hStrip.classList.toggle('focus', early);
      timesOp.style.opacity = early ? '0.35' : '1';
      wLab.style.opacity = early ? '0.35' : '1';
      flow.opw?.setOpacity(early ? 0.35 : 1);
      flow.out?.setOpacity(early ? 0.35 : 1);
      const late = step >= 3;
      smOp.style.opacity = late ? '1' : '0.35';
      smIn.setOpacity(late ? 1 : 0.35);
      smOut.setOpacity(late ? 1 : 0.35);
      if (late && !drawn) { drawn = true; smIn.draw(420); smOut.draw(420, 260); }
    }

    function chrome(D) {
      const { V, d, t } = D;
      header.el.querySelector('.k-eyebrow').replaceChildren(
        chip(`V = ${V}`, 'head'), chip(`d = ${d}`, 'resid'),
        ...(D.tied ? [chip('W_U = W_E', 'embed')] : []));
      header.setSub(`Position <b>${t}</b> (<b>“${esc(tokName(t))}”</b>) reads its final vector h<sub>t</sub> and scores all ${V} vocabulary entries.`);
      wLab.innerHTML = `${texHTML('\\WU')} · ${V} × ${d}<small>${D.tied ? 'tied to W_E (wte.weight)' : 'lm_head.weight'}</small>`;
      probHead.textContent = step >= 3 ? 'probabilities p' : 'logits z';
      stepLab.textContent = STEPS[step];
      readout.style.opacity = step >= 3 ? '1' : '0';
    }

    function captionFor(D) {
      const { V, d, t } = D;
      if (step === 0) return D.preLN
        ? `The head reads one vector per position: h<sub>t</sub>, the state of position <b>${t}</b> after the final norm. Earlier tokens reach the prediction only through this vector.`
        : `The head reads one vector per position: h<sub>t</sub>, the output of the last block at position <b>${t}</b>. Earlier tokens reach the prediction only through this vector.`;
      if (step === 1) return `Each of the ${V} rows of W<sub>U</sub> is dotted with h<sub>t</sub>: one multiply-and-add over each of the ${d} entries, giving one score per vocabulary entry.`;
      if (step === 2) return 'These are the logits: raw scores that can be negative. Adding one constant to every logit changes nothing, so only their differences matter.';
      if (D.hasNext) return `Softmax makes every score positive and rescales the row to sum to 1. The next input token <b>“${esc(tokName(t + 1))}”</b> gets <b>${pct(D.pNext)}</b>, so its loss is <b>${fmt(-Math.log(D.pNext), 2)} nats</b>.`;
      return `Softmax makes every score positive and rescales the row to sum to 1. Position <b>${t}</b> is the last one, so no token in this text is scored.`;
    }

    /** Equations, checks and caption for the current step. `v` is the row the sweep is on (step 1). */
    function renderSide(D, v = null) {
      const { t, z, p, hv, d, V, W } = D;
      const f = v ?? D.focus;
      const K = (x) => num(x, 2);
      let main = '', live = '', note = '', checks = [];
      if (step === 0) {
        const hNorm = D.preLN ? (D.rms ? '\\RMSNorm_f\\big(\\xres_t\\big)' : '\\LN_f\\big(\\xres_t\\big)') : `\\xres^{(${D.nLayer - 1})}_t`;
        main = `h_t = ${hNorm} \\in \\R^{${d}}`;
        note = `${d} numbers at position ${t}. Entries 0 to 3 are written under the strip.`;
      } else if (step === 1) {
        let zc = 0;
        for (let c = 0; c < d; c++) zc += hv[c] * W.data[f * d + c];
        main = `z_{t,v} = \\sum_{c=1}^{${d}} h_{t,c}\\,\\WU_{v,c}`;
        live = `\\begin{aligned} z_{${t},${f}} &= (${K(hv[0])})(${K(W.data[f * d])}) + \\cdots \\\\ &= ${K(z[f])} \\end{aligned}`;
        note = `${d} terms · vocabulary entry ${f}, “${esc(name(f))}”`;
        checks = [checkBadge(zc, z[f], { label: 'equals the traced logit' })];
      } else if (step === 2) {
        main = `z_t = h_t\\,\\WU^{\\top} \\in \\R^{${V}}`;
        live = `\\max_v z_{t,v} = ${K(D.zMax)}`;
        note = `largest: entry ${D.argmaxZ}, “${esc(name(D.argmaxZ))}” · smallest: ${K(D.zMin)}`;
      } else {
        const zf = z[f], m = D.zMax;
        const pRe = Math.exp(zf - m) / D.Zs;
        const shifted = Math.abs(zf - m) < 5e-3 ? 0 : zf - m;   // no "-0.00" in the exponent
        const sum = p.reduce((a, b) => a + b, 0);
        main = `p_{t,v} = \\frac{e^{z_{t,v}}}{\\sum_{k} e^{z_{t,k}}}`;
        live = `p_{${t},${f}} = \\frac{e^{${K(shifted)}}}{${num(D.Zs, 3)}} = ${num(pRe, 3)}`;
        note = `entry ${f}, “${esc(name(f))}”, scores shifted by the largest logit ${K(m)}`;
        checks = [checkBadge(pRe, p[f], { label: 'equals the traced probability' }), checkBadge(sum, 1, { label: 'sums to 1' })];
      }
      if (main !== lastMain) { eqMain.innerHTML = texHTML(main, { display: true }); lastMain = main; }
      eqLive.innerHTML = (live ? texHTML(live, { display: true }) : '') + (note ? `<div class="lh-note">${note}</div>` : '');
      eqCheck.replaceChildren(...checks);
      renderReadout(D);
      capSet(captionFor(D));
    }

    /** Next-token probability and loss, the mean loss and perplexity over the sequence. */
    function renderReadout(D) {
      const { t, z, p, hasNext, next, meanLoss, nScored, zMax, Zs } = D;
      const pNext = hasNext ? p[next] : null;
      const lossNext = pNext == null ? null : -Math.log(pNext);
      const ppl = meanLoss == null ? null : Math.exp(meanLoss);
      const line = (k, v) => h('div', { class: 'lh-read-row' }, h('span', { class: 'lh-read-k' }, k), h('span', { class: 'lh-read-v', html: v }));
      const rows = [
        line('next input', hasNext ? `“${esc(tokName(t + 1))}” <small>position ${t + 1}</small>` : `none <small>position ${t} is the last</small>`),
        line('p(next)', pNext == null ? '—' : pct(pNext)),
        line('loss −log p', lossNext == null ? '—' : `${fmt(lossNext, 2)} nats`),
        line('mean loss', meanLoss == null ? '—' : `${fmt(meanLoss, 2)} nats <small>${nScored} positions</small>`),
        line('perplexity', ppl == null ? '—' : `${fmt(ppl, 2)} <small>= e^loss</small>`),
      ];
      const extra = [];
      if (lossNext != null) extra.push(checkBadge(zMax + Math.log(Zs) - z[next], lossNext, { label: 'equals −log p from the logits' }));
      if (D.leak) extra.push(h('div', { class: 'lh-warn' }, 'This mask lets some positions see the token they predict, so these losses do not measure prediction.'));
      readout.replaceChildren(...rows, ...extra);
    }

    function missing() {
      stepLab.textContent = STEPS[step];
      eqMain.textContent = 'No forward pass yet.';
      eqLive.textContent = '';
      eqCheck.replaceChildren();
      lastMain = '';
    }

    function render({ animate = true } = {}) {
      stopTimers();
      const D = snap();
      if (!D) { missing(); return; }
      lastD = D;
      ensurePlate(D);
      chrome(D);
      paintW(D);
      paintH(D, animate);
      body.classList.toggle('dim', step === 0);
      barsHost.classList.toggle('off', step === 0);
      barsHost.classList.toggle('probs', step === 3);
      alignBars();
      paintFlow();
      if (step === 1 && animate && !reducedMotion()) { sweep(D); return; }
      band.hidden = true;
      barsHost.style.setProperty('--lh-dur', `${MORPH_TRANS}ms`);
      paintBars(D, step === 0 ? 'zero' : step === 3 ? 'prob' : 'logit');
      renderSide(D);
      if (step === 3 && animate) pulse(probsWrap);
    }

    /** A step set by the reader (story bar, arrow keys, step dots) ends the auto-walk. */
    function setStep(i) {
      if (!autoStepping) stopAuto();
      step = i;
      render({ animate: true });
    }

    /** The auto-walk: one step per MOTION.step, started after the zoom has finished. */
    function autoNext() {
      auto = null;
      if (step >= STEPS.length - 1) return;
      autoStepping = true;
      try { ctx.setStep(step + 1); } finally { autoStepping = false; }
      auto = setTimeout(autoNext, MOTION.step);
    }

    // ------------------------------------------------------------ mount
    render({ animate: true });
    // The plate only fades: a transform animation would replace its tilt while it runs, and the bars are measured against the tilt.
    if (!reducedMotion()) body.animate([{ opacity: 0 }, { opacity: 1 }], { duration: MOTION.appear, easing: MOTION.easeOut });
    appear([header.el, picker.el, hStrip, hLab, hVals, wLab, timesOp, probsWrap, smOp, side], { stagger: MOTION.stagger })
      .then(() => { if (alive && lastD) alignBars(); });
    ctx.setSteps(STEPS, setStep);
    const offs = [ctx.on('trace', () => render({ animate: true })), ctx.on('select', () => render({ animate: true }))];
    const offTheme = onThemeChange(() => { if (lastD) render({ animate: false }); });

    // auto-advance the first time the scene is shown, so it explains itself
    auto = setTimeout(autoNext, ZOOM_MS + MOTION.step);
    el.addEventListener('pointerdown', stopAuto, { once: true });

    return {
      pause() { stopAuto(); stopTimers(); },
      resume() { if (lastD) render({ animate: false }); },   // re-measures the tilted plate after it was hidden
      destroy() {
        alive = false;
        stopAuto();
        stopTimers();
        offs.forEach((f) => f());
        offTheme();
        st.destroy();
      },
    };
  },
};

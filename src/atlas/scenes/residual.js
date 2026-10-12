// The residual add for one sub-layer, on the real numbers of one token.
//   x (stream before) + a (sub-layer output) = sum; post-LN then normalises the sum.
// Steps: Stream → Sub-layer output → Add → Across depth → Without the adds.
// Every value comes from ctx.state.trace (and model.params for the post-LN norm check).
// "Without the adds" compares two runs of the same weights: as configured, and with every residual add in
// every block switched the other way (engine.cloneModel + config + infer, cached until the trace changes).
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, fly, appear, useStyle, ZONES, MOTION } from '../kit.js';
import { h, fmt, texHTML, num, reducedMotion, clamp, diverging } from '../../ui/index.js';

const SUB = { 1: 'attention', 2: 'MLP' };

// Hero table: x over a (the ⊕ sits between them), a rule, the sum, then the normalised row (post-LN only).
const CELL = 44, CELL_H = 60, GAP = 3;                     // cell width and height, gap between cells (design units)
const G_IN = 30, G_SUM = 40;                               // gap from x to a; gap from a to the sum, and from the sum to the normalised row
const TOPS = [0, CELL_H + G_IN, 2 * CELL_H + G_IN + G_SUM, 3 * CELL_H + G_IN + 2 * G_SUM];   // x, a, sum, normalised
const TX = 330;                                            // table origin; the row labels sit left of it
const LABEL_X = ZONES.hero.x, LABEL_W = 150;               // row labels
const tableTop = (pre) => ZONES.hero.y + (ZONES.hero.h - (pre ? TOPS[2] : TOPS[3]) - CELL_H) / 2;   // centred in the hero
const OP = 48;                                             // the ⊕ node (guide: 48 to 64 units)
const SX = ZONES.side.x, SW = ZONES.side.w;                // right column: the equation
const FLOW = 480, STAGGER = 20, LAND = FLOW + 90;          // add: flight time, delay per column, landing of the last a-flight (ms)
const CHART = { x: ZONES.hero.x, y: 340, w: 1030, h: 340 };   // depth profile, across the hero
// Step 4: one token list, two bar columns (every add on, every add off), one scale.
const BARS = { x: ZONES.hero.x, y: 340, row: 40, n: 10 };
const TOK_W = 56, COL_GAP = 14, TRACK_W = 360, PCT_GAP = 10, PCT_W = 74, PAIR_GAP = 40;
const COL_A = TOK_W + COL_GAP;                             // x of the first bar track, inside a row
const COL_B = COL_A + TRACK_W + PCT_GAP + PCT_W + PAIR_GAP; // x of the second bar track
const ROW_W = COL_B + TRACK_W + PCT_GAP + PCT_W;
const EASE = MOTION.easeOut, MOVE = MOTION.ease;

useStyle('residual-scene', `
.rs-part { transition: opacity ${MOTION.appear}ms ${EASE}, transform ${MOTION.appear}ms ${EASE}; }
.rs-part.rs-off { opacity: 0; pointer-events: none; }
.rs-table { position: absolute; }
.rs-row { position: absolute; left: 0; display: flex; gap: ${GAP}px; transition: opacity ${MOTION.appear}ms ${EASE}; }
.rs-idx { width: ${CELL}px; text-align: center; font: 500 12px/1 var(--font-mono); color: var(--fg-3); }
.rs-cell { width: ${CELL}px; height: ${CELL_H}px; border-radius: 3px; display: grid; place-items: center;
  font: 600 13px/1 var(--font-mono); color: var(--fg); box-shadow: inset 0 -1px 0 rgba(0,0,0,.14);
  transition: background-color 500ms ease, box-shadow 300ms ease; }
.rs-row.pending .rs-cell:not(.ell) { background: transparent; box-shadow: inset 0 0 0 1px var(--line); }
.rs-cell.ell { background: none !important; box-shadow: none; color: var(--fg-3); font-size: 14px; }
.rs-rowlabel { position: absolute; width: ${LABEL_W}px; height: ${CELL_H}px; display: flex; flex-direction: column; justify-content: center;
  transition: opacity ${MOTION.appear}ms ${EASE}; }
.rs-rowlabel.pending { opacity: .4; }
.rs-rowlabel .rs-sym { font-size: 17px; line-height: 1.2; }
.rs-rowlabel .rs-sub { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); margin-top: 4px; }
.rs-op { position: absolute; width: ${OP}px; height: ${OP}px; border-radius: 50%; display: grid; place-items: center;
  background: var(--surface); border: 2px solid var(--c-resid); color: var(--fg); font-size: 20px; box-shadow: var(--shadow-1); }
.rs-rule { position: absolute; height: 0; border-top: 1.5px solid var(--line-strong); }
.rs-eq { width: ${SW}px; display: grid; gap: 10px; }
.rs-eq-main { font-size: 26px; }
.rs-eq-live { font-size: 19px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.rs-stat { font: 500 15px/1.4 var(--font-mono); color: var(--fg-2); }
.rs-chart { position: absolute; }
.rs-chart text { font: 500 15px/1 var(--font-mono); fill: var(--fg-3); }
.rs-chart .rs-hl-text { fill: var(--fg); font-weight: 700; }
.rs-chart .rs-grid { stroke: var(--line); }
.rs-chart .rs-axis { stroke: var(--line-strong); }
.rs-bars { position: absolute; width: ${ROW_W}px; height: ${BARS.n * BARS.row}px; }
.rs-colhead { position: absolute; top: -34px; font: 600 15px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.rs-brow { position: absolute; left: 0; top: 0; width: ${ROW_W}px; height: 32px; display: flex; align-items: center;
  transition: transform ${MOTION.move}ms ${MOVE}; }
.rs-tok { width: ${TOK_W}px; text-align: left; font: 600 17px/1 var(--font-mono); color: var(--fg); white-space: pre; }
.rs-track { position: relative; width: ${TRACK_W}px; height: 22px; margin-left: ${COL_GAP}px; border-radius: 6px; background: var(--surface-2); overflow: hidden; }
.rs-track.b { margin-left: ${PAIR_GAP}px; }
.rs-track.top { outline: 2px solid var(--accent); outline-offset: 2px; }
.rs-fill { position: absolute; left: 0; top: 0; bottom: 0; width: 0; border-radius: 6px; background: var(--c-resid); transition: width ${MOTION.move}ms ${MOVE}; }
.rs-p { width: ${PCT_W}px; margin-left: ${PCT_GAP}px; font: 600 15px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.rs-p.top { color: var(--fg); font-weight: 700; }
@media (prefers-reduced-motion: reduce) { .rs-part, .rs-row, .rs-cell, .rs-rowlabel, .rs-brow, .rs-fill { transition: none !important; } }
`);

const norm2 = (v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
const maxAbsDiff = (a, b) => { let e = 0; for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i])); return e; };
const sumVec = (a, b) => a.map((v, i) => v + b[i]);
const cellText = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2).replace('-', '−'));
const sg = (v) => (v < 0 ? `- ${num(-v, 3)}` : `+ ${num(v, 3)}`);
const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const valueRange = (D) => Math.max(1e-9, ...[D.x, D.a, D.s, D.lnOut || D.s].map((v) => Math.max(...v.map((z) => Math.abs(z)))));
const f3 = (v) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3).replace('-', '−');
// A token as a word: the space token reads “space”; TeX text escapes the characters that LaTeX would read as code.
const wordOf = (label) => (label === '␣' ? 'space' : String(label));
const TEX_ESC = { '\\': '\\textbackslash{}', '{': '\\{', '}': '\\}', '$': '\\$', '&': '\\&', '#': '\\#', '^': '\\textasciicircum{}', '_': '\\_', '%': '\\%', '~': '\\textasciitilde{}' };
const texWord = (label) => wordOf(label).replace(/[\\{}$&#^_%~]/g, (c) => TEX_ESC[c]);
const pctOf = (p) => `${(p * 100).toFixed(1)}%`;
const texPct = (p) => `${(p * 100).toFixed(1)}\\%`;

// Mean and spread of one row as model.py defines them (biased variance, eps = norm_eps; RMSNorm has no mean).
function normStats(v, cfg) {
  const d = v.length, eps = cfg.norm_eps;
  if (cfg.norm === 'rmsnorm') return { mu: 0, sd: Math.sqrt(v.reduce((a, x) => a + x * x, 0) / d + eps) };
  const mu = v.reduce((a, x) => a + x, 0) / d;
  const vr = v.reduce((a, x) => a + (x - mu) * (x - mu), 0) / d;
  return { mu, sd: Math.sqrt(vr + eps) };
}

// The block's norm applied to one row: gain, and bias where the norm has one.
function normRow(v, model, name, stats) {
  const P = model.params, g = P[`${name}.weight`]?.data, b = P[`${name}.bias`]?.data;
  return v.map((x, i) => ((x - stats.mu) / stats.sd) * (g ? g[i] : 1) + (b ? b[i] : 0));
}

// The block's norm, named from the config (model.py make_norm). RMSNorm has no mean and no shift;
// LayerNorm has a shift only when bias is on.
function normNames(c) {
  const rms = c.norm === 'rmsnorm';
  return { rms, shift: !rms && !!c.bias, tex: rms ? '\\RMSNorm' : '\\LN', word: rms ? 'RMSNorm' : 'LN', post: rms ? 'Post-RMSNorm' : 'Post-LN' };
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const l = ctx.params.l, n = ctx.params.n === 2 ? 2 : 1;
    const sub = SUB[n];
    const STEPS = ['Stream', 'Sub-layer output', 'Add', 'Across depth', 'Without the adds'];
    let step = 0, gen = 0, timers = [], auto = null, autoTick = false;
    let altProbs = null, altKey = '';
    let cols = [];                                   // displayed dimensions: all of them, or first 12, a gap (-1) and last 4
    let cellsX = [], cellsA = [], cellsS = [], cellsN = [];
    const barRows = new Map();

    const clr = () => { gen++; timers.forEach(clearTimeout); timers = []; };
    const later = (ms, fn) => { const g = gen; timers.push(setTimeout(() => { if (g === gen) fn(); }, ms)); };
    const stopAuto = () => { clearTimeout(auto); auto = null; };
    // A row that is not on yet is a ghost: an outline where its values will land, and a dimmed label.
    const setRow = (rowEl, labEl, on) => {
      if (!on) rowEl.querySelectorAll('.rs-cell:not(.ell)').forEach((c) => { c.textContent = ''; c.style.background = ''; });
      rowEl.classList.toggle('pending', !on);
      labEl.classList.toggle('pending', !on);
    };
    const blank = (cells) => cells.forEach((c) => { if (!c.classList.contains('ell')) { c.textContent = ''; c.style.background = ''; } });

    // ------------------------------------------------------------ shared parts
    const header = sceneHeader(st, { chips: [[`Block ${l}`, 'resid'], [n === 1 ? 'attention' : 'MLP', n === 1 ? 'attn' : 'mlp']], title: ctx.node.title });
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'resid' });
    const cap = sceneCaption(st, { w: 1060 });

    // ------------------------------------------------------------ hero table: steps 0 to 2 (its parts share one fade)
    const table = h('div', { class: 'rs-table rs-part' });
    st.add(table, { x: TX, y: tableTop(false) });
    const rowX = h('div', { class: 'rs-row', style: { top: `${TOPS[0]}px` } });
    const rowA = h('div', { class: 'rs-row pending', style: { top: `${TOPS[1]}px` } });
    const rowS = h('div', { class: 'rs-row pending', style: { top: `${TOPS[2]}px` } });
    const rowN = h('div', { class: 'rs-row pending', style: { top: `${TOPS[3]}px` } });
    const idxRow = h('div', { class: 'rs-row', style: { top: '-22px' } });
    const rule = h('div', { class: 'rs-rule rs-part rs-off', style: { top: `${(TOPS[1] + CELL_H + TOPS[2]) / 2}px` } });
    const opEl = h('div', {
      class: 'rs-op rs-part rs-off', html: texHTML('\\oplus'),
      style: { left: `${-40 - OP / 2}px`, top: `${(CELL_H / 2 + TOPS[1] + CELL_H / 2) / 2 - OP / 2}px` },
    });
    const labelAt = (top, pending, color) => h('div', { class: pending ? 'rs-rowlabel pending' : 'rs-rowlabel', style: { top: `${top}px`, left: `${LABEL_X - TX}px` } },
      h('div', { class: 'rs-sym', style: { color } }), h('div', { class: 'rs-sub' }));
    const labX = labelAt(TOPS[0], false, 'var(--c-resid)');
    const labA = labelAt(TOPS[1], true, n === 1 ? 'var(--c-attn)' : 'var(--c-mlp)');
    const labS = labelAt(TOPS[2], true, 'var(--c-resid)');
    const labN = labelAt(TOPS[3], true, 'var(--c-norm)');
    labA.querySelector('.rs-sym').innerHTML = texHTML(n === 1 ? '\\Attn' : '\\MLP');
    labA.querySelector('.rs-sub').textContent = `${sub} output`;
    labX.querySelector('.rs-sub').textContent = 'stream in';
    labN.querySelector('.rs-sub').textContent = 'normalised';
    table.append(idxRow, rowX, rowA, rule, rowS, rowN, opEl, labX, labA, labS, labN);

    // ------------------------------------------------------------ depth profile: step 3
    const chart = h('div', { class: 'rs-chart rs-part rs-off' });
    st.add(chart, { x: CHART.x, y: CHART.y });

    // ------------------------------------------------------------ without the adds: two bar columns, step 4
    const bars = h('div', { class: 'rs-bars rs-part rs-off' });
    bars.append(
      h('div', { class: 'rs-colhead', style: { left: `${COL_A}px` } }, 'with every add'),
      h('div', { class: 'rs-colhead', style: { left: `${COL_B}px` } }, 'without every add'));
    st.add(bars, { x: BARS.x, y: BARS.y });

    // ------------------------------------------------------------ side column: the equation for this step
    const eq = h('div', { class: 'rs-eq' });
    st.add(eq, { x: SX, y: 300 });

    // ------------------------------------------------------------ data
    function data() {
      const S = ctx.state, c = S.config, T = S.ids.length, q = clamp(S.sel.token ?? 0, 0, T - 1);
      const tr = S.trace, pre = c.norm_placement === 'pre';
      const xKey = n === 2 ? `L${l}.resid_mid` : (l === 0 ? 'emb.out' : `L${l - 1}.resid_out`);
      const subKey = n === 1 ? `L${l}.attn.out` : `L${l}.mlp.out`;
      const sumKey = pre ? (n === 1 ? `L${l}.resid_mid` : `L${l}.resid_out`) : (n === 1 ? `L${l}.add1` : `L${l}.add2`);
      const outKey = n === 1 ? `L${l}.resid_mid` : `L${l}.resid_out`;
      const row = (k) => { const t = tr[k]; if (!t) return null; const d = t.shape[1]; return Array.from(t.data.subarray(q * d, (q + 1) * d)); };
      const x = row(xKey), a = row(subKey), s = row(sumKey), out = row(outKey);
      if (!x || !a || !s || !out) return null;
      const res = !!c.residual;
      const eSum = maxAbsDiff(res ? sumVec(x, a) : a, s);      // what the add should give, against the traced sum
      let lnOut = null, eLn = 0, stats = null, unitNorm = true;
      if (!pre) {
        stats = normStats(s, c);
        lnOut = normRow(s, S.model, `blocks.${l}.ln${n}`, stats);
        eLn = maxAbsDiff(lnOut, out);
        // gain 1 and bias 0 is the initial norm; once trained, the block points take other norms
        const P = S.model.params, g = P[`blocks.${l}.ln${n}.weight`]?.data, b = P[`blocks.${l}.ln${n}.bias`]?.data;
        unitNorm = (!g || g.every((v) => v === 1)) && (!b || b.every((v) => v === 0));
      }
      // depth profile: the embedding, then resid_mid and resid_out of every block
      const pts = [{ k: 'emb.out', label: 'emb' }];
      for (let i = 0; i < c.n_layer; i++) pts.push({ k: `L${i}.resid_mid`, label: `${i}·mid` }, { k: `L${i}.resid_out`, label: `${i}·out` });
      const prof = pts.map((p) => { const r = row(p.k); return { label: p.label, key: p.k, norm: r ? norm2(r) : NaN }; });
      return {
        q, T, pre, res, c, x, a, s, out, d: x.length, xKey, subKey, sumKey, outKey, eSum, eLn, lnOut, stats, unitNorm,
        nx: norm2(x), na: norm2(a), nout: norm2(out), prof, hiIdx: prof.findIndex((p) => p.key === outKey),
        weightsRandom: S.weights === 'random',
      };
    }

    // ------------------------------------------------------------ table build (values are painted by applyStep)
    function build() {
      const D = data();
      if (!D) return;
      cols = D.d <= 17 ? [...Array(D.d).keys()] : [...Array(12).keys(), -1, ...Array.from({ length: 4 }, (_, k) => D.d - 4 + k)];
      const W = cols.length * CELL + (cols.length - 1) * GAP;
      table.style.width = `${W}px`;
      rule.style.width = `${W}px`;
      rowN.style.display = D.pre ? 'none' : '';    // pre-LN has no normalised row: the table is three rows then
      labN.style.display = D.pre ? 'none' : '';
      st.move(table, TX, tableTop(D.pre));
      const mk = (rowEl, key) => {
        rowEl.replaceChildren();
        rowEl.dataset.trace = key;
        return cols.map((i) => {
          const c = h('div', { class: i < 0 ? 'rs-cell ell' : 'rs-cell' }, i < 0 ? '⋮' : '');
          if (i >= 0) { c.dataset.trace = key; c.dataset.cell = `${D.q},${i},0`; }
          rowEl.append(c);
          return c;
        });
      };
      cellsX = mk(rowX, D.xKey);
      cellsA = mk(rowA, D.subKey);
      cellsS = mk(rowS, D.sumKey);
      cellsN = mk(rowN, D.outKey);
      idxRow.replaceChildren(...cols.map((i) => h('div', { class: 'rs-idx' }, i < 0 ? '' : String(i))));
      // labels follow the wiring: pre-LN has the sum after the add; post-LN has the sum s, then the normalised output
      const outTex = n === 1 ? '\\xres^{\\text{mid}}' : '\\xres^{\\text{out}}';
      labX.querySelector('.rs-sym').innerHTML = texHTML(`\\xres_{${D.q}}`);
      labS.querySelector('.rs-sym').innerHTML = texHTML(D.pre ? outTex : 's');
      labS.querySelector('.rs-sub').textContent = D.pre ? 'after the add' : 'the sum';
      labN.querySelector('.rs-sym').innerHTML = texHTML(outTex);
      // the context line says what this scene is about; the add itself is in the step text
      const who = n === 1 ? 'attention' : 'MLP';
      const where = `token “${esc(wordOf(ctx.label(D.q)))}” at position ${D.q}`;
      header.setSub(D.res
        ? `Block ${l}’s ${who} writes into the residual stream of ${where}.`
        : `With the residual off, block ${l}’s ${who} output replaces the stream of ${where}.`);
    }

    function paintRow(cells, values, range) {
      cols.forEach((i, k) => {
        if (i < 0) return;
        cells[k].style.background = diverging(clamp(values[i] / range, -1, 1));
        cells[k].textContent = cellText(values[i]);
      });
    }

    // the add, animated: x and a fly down column by column and land in the sum row (without x, residual off).
    // Post-LN: the normalised row starts as a copy of the sum and rescales to the norm of s. The whole sequence
    // ends inside the step's dwell (MOTION.step), so the auto-tour never cuts it short.
    function addAnimated(D) {
      const range = valueRange(D);
      setRow(rowN, labN, false);
      blank(cellsS);
      paintRow(cellsX, D.x, range);
      paintRow(cellsA, D.a, range);
      cols.forEach((i, k) => {
        if (i < 0) return;
        const delay = k * STAGGER;
        later(delay, () => {
          if (D.res) fly(st, cellsX[k], cellsS[k], { duration: FLOW, arc: 0 });
          fly(st, cellsA[k], cellsS[k], { duration: FLOW, delay: 90, arc: 0 });
        });
        later(delay + LAND, () => {
          cellsS[k].style.background = diverging(clamp(D.s[i] / range, -1, 1));
          cellsS[k].textContent = cellText(D.s[i]);
        });
      });
      if (D.pre) return;
      later(cols.length * STAGGER + LAND + 100, () => {
        setRow(rowN, labN, true);
        paintRow(cellsN, D.s, range);
        for (let k = 1; k <= 12; k++) later(k * 40, () => paintRow(cellsN, D.s.map((v, j) => v + (D.lnOut[j] - v) * (k / 12)), range));
      });
    }

    // ------------------------------------------------------------ the two runs (step 4)
    // The clone runs every residual add of every block switched the other way (cfg.residual is model-wide).
    function altDistribution(D) {
      const S = ctx.state, want = !D.c.residual;
      const key = `${S.ids.join(',')}|${want}|${S.config.n_layer}|${S.config.norm_placement}|${S.config.norm}|${S.config.pos_encoding}|${S.config.activation}|${S.config.mask}|${S.weights}`;
      if (altKey === key && altProbs) return altProbs;
      const m = ctx.engine.cloneModel(S.model);
      m.config = { ...m.config, residual: want };
      const out = ctx.engine.infer(m, S.ids);
      altProbs = Array.from(out.probs.data);   // full [T, V] matrix, sliced by the selected row below
      altKey = key;
      return altProbs;
    }

    // Every add on, and every add off, for the selected token. Each run's top five go on one token list,
    // ordered by the larger of the two probabilities; both columns share one scale.
    function compare(D) {
      const S = ctx.state, V = S.config.vocab_size, q = D.q, cfgOn = !!D.c.residual;
      const traced = Array.from(S.trace.probs.data.subarray(q * V, (q + 1) * V));
      const clone = altDistribution(D).slice(q * V, (q + 1) * V);
      const withP = cfgOn ? traced : clone, withoutP = cfgOn ? clone : traced;
      const topOf = (p, k) => Array.from(p.keys()).sort((i, j) => p[j] - p[i]).slice(0, k);
      const tW = topOf(withP, 1)[0], tO = topOf(withoutP, 1)[0];
      const ids = [...new Set([...topOf(withP, 5), ...topOf(withoutP, 5)])]
        .sort((i, j) => Math.max(withP[j], withoutP[j]) - Math.max(withP[i], withoutP[i]));
      const maxP = Math.max(1e-9, ...ids.map((id) => Math.max(withP[id], withoutP[id])));
      return { withP, withoutP, tW, tO, ids, maxP };
    }

    function paintBars(D) {
      const S = ctx.state, K = compare(D);
      for (const [id, row] of barRows) if (!K.ids.includes(id)) { row.el.remove(); barRows.delete(id); }
      K.ids.forEach((id, r) => {
        let row = barRows.get(id);
        if (!row) {
          const ta = h('div', { class: 'rs-track' }, h('div', { class: 'rs-fill' }));
          const tb = h('div', { class: 'rs-track b' }, h('div', { class: 'rs-fill' }));
          const pa = h('span', { class: 'rs-p' }), pb = h('span', { class: 'rs-p' });
          const el = h('div', { class: 'rs-brow' }, h('span', { class: 'rs-tok' }, S.tokenizer.label(id)), ta, pa, tb, pb);
          row = { el, ta, tb, pa, pb, fa: ta.firstChild, fb: tb.firstChild };
          barRows.set(id, row);
          bars.append(el);
        }
        row.el.style.transform = `translateY(${r * BARS.row}px)`;
        row.ta.classList.toggle('top', id === K.tW);
        row.pa.classList.toggle('top', id === K.tW);
        row.tb.classList.toggle('top', id === K.tO);
        row.pb.classList.toggle('top', id === K.tO);
        row.fa.style.width = `${(K.withP[id] / K.maxP) * 100}%`;
        row.fb.style.width = `${(K.withoutP[id] / K.maxP) * 100}%`;
        row.pa.textContent = pctOf(K.withP[id]);
        row.pb.textContent = pctOf(K.withoutP[id]);
      });
    }

    // ------------------------------------------------------------ the equations and captions
    function paintText(D) {
      const S = ctx.state;
      const { q, pre, res, x, a, s, c, stats, lnOut, eSum, eLn, d } = D;
      const N = normNames(c);
      const outSym = n === 1 ? '\\xres^{\\text{mid}}' : '\\xres^{\\text{out}}';
      const subTex = n === 1 ? (pre ? `\\Attn(${N.tex}(\\xres))` : '\\Attn(\\xres)') : (pre ? `\\MLP(${N.tex}(\\xres^{\\text{mid}}))` : '\\MLP(\\xres^{\\text{mid}})');
      const posNote = c.pos_encoding === 'rope' || c.pos_encoding === 'alibi' ? ' Position is not in x here: RoPE or ALiBi act inside attention.' : '';
      const embWords = c.pos_encoding === 'learned' || c.pos_encoding === 'sinusoidal' ? 'the embedding (token plus position)' : 'the token embedding';
      const from = n === 1 ? (l === 0 ? embWords : `block ${l - 1}’s output`) : `the attention add of block ${l}`;
      const where = `token “${esc(wordOf(ctx.label(q)))}” at position ${q}`;
      let eqMain = '', eqLive = '', words = null;
      switch (step) {
        case 0:
          eqMain = `\\xres_{${q}} \\in \\R^{${d}}`;
          eqLive = `\\|\\xres_{${q}}\\|_2 = ${num(D.nx, 2)}`;
          words = `The stream entering this sub-layer is x, the vector of ${where}. It comes from ${from}.${posNote}`;
          break;
        case 1:
          eqMain = `a = ${subTex}`;
          eqLive = `\\|a_{${q}}\\|_2 = ${num(D.na, 2)}`;
          words = res
            ? `The ${sub} reads ${pre ? 'the normalised stream' : 'the stream'} and writes a vector a of the same width d. On its own, a is only a delta.`
            : `Residual off: the ${sub} reads ${pre ? 'the normalised stream' : 'the stream'}, and its output replaces the stream.`;
          break;
        case 2:
          if (pre) {
            eqMain = res ? `${outSym} = \\xres + a` : `${outSym} = a`;
            eqLive = res ? `${num(x[0], 3)} ${sg(a[0])} = ${num(s[0], 3)}\\;(i{=}0)` : `${outSym}_{0} = ${num(s[0], 3)}`;
            words = res
              ? 'Each dimension adds: the sum at i is x_i + a_i. The result is the new stream, and the next norm reads it.'
              : 'Residual off: nothing is added. The sub-layer output replaces the stream.';
          } else {
            const first = res ? `${N.post}: the sum s is formed first, then normalised.` : 'Residual off: the sub-layer output is the sum, and the norm then rescales it.';
            const after = N.shift ? 'then the gain and bias apply.' : 'then the gain applies.';
            const how = N.rms ? `It is divided by the root mean square, with no mean removed and no shift, ${after}` : `The mean μ is removed and the result is divided by σ = √(var + ε), ${after}`;
            eqMain = `\\begin{aligned} s &= ${res ? '\\xres + a' : 'a'} \\\\ ${outSym} &= ${N.tex}(s) \\end{aligned}`;
            eqLive = res ? `${num(x[0], 3)} ${sg(a[0])} = ${num(s[0], 3)}\\;(i{=}0)` : `s_{0} = ${num(s[0], 3)}`;
            words = `${first} ${how}${D.weightsRandom ? ' The weights are untrained (random).' : ''}`;
          }
          break;
        case 3:
          eqMain = `\\|${outSym}\\|_2 \\text{ across depth}`;
          eqLive = `\\|${outSym}_{${q}}\\|_2 = ${num(D.nout, 2)}`;
          words = !res
            ? 'Residual off: each block point is only that block’s output, so nothing accumulates. The highlighted point is this block, and the first point is the embedding.'
            : pre
              ? 'The highlighted point is this block’s stream after the add. Each add keeps what earlier blocks wrote, so the norm reflects every write so far.'
              : D.unitNorm
                ? `The highlighted point is this block’s normalised output. With gain 1${N.shift ? ' and bias 0' : ''} here, its norm is √${d} = ${num(Math.sqrt(d), 2)}, up to the norm’s epsilon. The embedding is not normalised.`
                : `The highlighted point is this block’s normalised output. The gain${N.shift ? ' or bias' : ''} is not the identity here, so its norm is the live value, not necessarily √${d}. The embedding is not normalised.`;
          break;
        default: {
          const K = compare(D);
          const wTex = texWord(S.tokenizer.label(K.tW)), oTex = texWord(S.tokenizer.label(K.tO));
          eqMain = `p^{\\text{with}}_{${q}}(w^{\\star}) \\;\\text{vs}\\; p^{\\text{without}}_{${q}}(w^{\\star})`;
          eqLive = `\\begin{aligned} p^{\\text{with}}_{${q}}(\\text{“${wTex}”}) &= ${texPct(K.withP[K.tW])} \\\\ p^{\\text{without}}_{${q}}(\\text{“${oTex}”}) &= ${texPct(K.withoutP[K.tO])} \\end{aligned}`;
          const tok = esc(wordOf(ctx.label(q))), wW = esc(wordOf(S.tokenizer.label(K.tW))), wO = esc(wordOf(S.tokenizer.label(K.tO)));
          words = K.tW === K.tO
            ? `Top prediction after “${tok}”: “${wW}” stays on top, at ${pctOf(K.withP[K.tW])} with every residual add in every block and ${pctOf(K.withoutP[K.tW])} without them.`
            : `Without every residual add in every block, the top prediction after “${tok}” is “${wO}” at ${pctOf(K.withoutP[K.tO])}. With them it is “${wW}” at ${pctOf(K.withP[K.tW])}.`;
          break;
        }
      }
      const liveProps = { class: 'rs-eq-live', html: texHTML(eqLive, { display: true }) };
      if (step === 2) liveProps.title = 'values rounded to three decimals';
      eq.replaceChildren(...[
        h('div', { class: 'k-label', style: { textAlign: 'left' } }, STEPS[step]),
        h('div', { class: 'rs-eq-main', html: texHTML(eqMain, { display: true }) }),
        eqLive ? h('div', liveProps) : null,
      ].filter(Boolean));
      // an equation wider than the side column shrinks to fit it (a sum of two small numbers, for example)
      for (const [sel, fit] of [['.rs-eq-main', '22px'], ['.rs-eq-live', '15px']]) {
        const el = eq.querySelector(sel);
        if (el) { el.style.fontSize = ''; if (el.scrollWidth > el.clientWidth + 1) el.style.fontSize = fit; }
      }
      if (step === 2) {
        const bd = checkBadge(eSum, 0, { label: `${res ? 'sum' : 'replacement'} equals the traced value` });
        bd.title = `max |Δ| ${fmt(eSum, 1)} over ${d} dimensions`;
        eq.append(bd);
      }
      if (step === 2 && !pre && lnOut) {
        const bl = checkBadge(eLn, 0, { label: `${N.word}(s) equals the traced stream` });
        bl.title = `max |Δ| ${fmt(eLn, 1)} over ${d} dimensions`;
        const stat = N.rms ? `√(mean s² + ε) = ${f3(stats.sd)}` : `μ = ${f3(stats.mu)} · √(var + ε) = ${f3(stats.sd)}`;
        eq.append(bl, h('div', { class: 'rs-stat', title: `${N.rms ? 'the root mean square of s' : 'the mean and variance of s'} over ${d} dimensions` }, stat));
      }
      if (words != null) cap.set(words);
    }

    // ------------------------------------------------------------ the depth profile (step 3)
    function paintChart(D) {
      const W = CHART.w, Hh = CHART.h, m = { l: 46, r: 64, t: 56, b: 60 };   // the right margin holds the last point's label
      const prof = D.prof, n2 = prof.length;
      const finite = prof.map((p) => p.norm).filter(Number.isFinite);
      const ymax = Math.max(1e-9, ...finite) * 1.12;
      const X = (i) => m.l + (i / Math.max(1, n2 - 1)) * (W - m.l - m.r);
      const Y = (v) => Hh - m.b - (v / ymax) * (Hh - m.t - m.b);
      const yOf = (p) => Y(Number.isFinite(p.norm) ? p.norm : 0);
      const NS = 'http://www.w3.org/2000/svg';
      const mk = (tag, attrs, txt) => {
        const e = document.createElementNS(NS, tag);
        for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
        if (txt != null) e.textContent = txt;
        return e;
      };
      const svg = mk('svg', { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}` });
      for (let t = 0; t <= 4; t++) {
        const v = (ymax / 1.12) * (t / 4);
        svg.append(mk('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'rs-grid' }));
        svg.append(mk('text', { x: m.l - 10, y: Y(v) + 5, 'text-anchor': 'end' }, Math.abs(v) < 0.1 && v !== 0 ? v.toFixed(2) : fmt(v, 1)));
      }
      svg.append(mk('line', { x1: m.l, x2: W - m.r, y1: Hh - m.b, y2: Hh - m.b, class: 'rs-axis' }));
      const path = prof.map((p, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${yOf(p).toFixed(1)}`).join('');
      svg.append(mk('path', { d: path, fill: 'none', style: 'stroke: var(--c-resid)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }));
      const dense = n2 > 9;                         // many blocks: label every other point so the labels do not collide
      prof.forEach((p, i) => {
        const hi = i === D.hiIdx;
        svg.append(mk('circle', {
          cx: X(i), cy: yOf(p), r: hi ? 7 : 4.5,
          style: hi ? 'fill: var(--fg); stroke: var(--surface)' : 'fill: var(--surface); stroke: var(--c-resid)',
          'stroke-width': hi ? 2.5 : 2,
        }));
        if (hi || !dense || i % 2 === 0) svg.append(mk('text', { x: X(i), y: Hh - m.b + 24, 'text-anchor': 'middle', class: hi ? 'rs-hl-text' : '' }, p.label));
        if (hi) svg.append(mk('text', { x: X(i), y: yOf(p) - 16, 'text-anchor': 'middle', class: 'rs-hl-text' }, `‖·‖₂ ${fmt(p.norm, 2)}`));
      });
      svg.append(mk('text', { x: 0, y: m.t - 24 }, `‖stream‖₂ of token “${wordOf(ctx.label(D.q))}” at position ${D.q}`));
      svg.append(mk('text', { x: W - m.r, y: Hh - 6, 'text-anchor': 'end' }, 'emb, then each block: mid, out'));
      chart.replaceChildren(svg);
      chart.dataset.trace = D.outKey;
    }

    // ------------------------------------------------------------ steps
    function applyStep(i, { animate = true } = {}) {
      clr();
      step = clamp(i, 0, STEPS.length - 1);
      const D = data();
      if (!D) return;
      const show = (el, on) => el.classList.toggle('rs-off', !on);
      show(table, step <= 2);
      show(opEl, step >= 1);
      show(rule, step === 2);
      show(chart, step === 3);
      show(bars, step === 4);
      if (step === 3) paintChart(D);
      if (step === 4) paintBars(D);
      const range = valueRange(D);
      if (step <= 1) {
        setRow(rowX, labX, true);
        setRow(rowA, labA, step === 1);
        setRow(rowS, labS, false);
        setRow(rowN, labN, false);
        paintRow(cellsX, D.x, range);
        if (step === 1) paintRow(cellsA, D.a, range);
      } else if (step === 2) {
        setRow(rowX, labX, true); setRow(rowA, labA, true); setRow(rowS, labS, true);
        if (!reducedMotion() && animate) addAnimated(D);
        else {
          setRow(rowN, labN, !D.pre);
          paintRow(cellsX, D.x, range);
          paintRow(cellsA, D.a, range);
          paintRow(cellsS, D.s, range);
          if (!D.pre) paintRow(cellsN, D.lnOut, range);
        }
      }
      paintText(D);
    }

    // ------------------------------------------------------------ mount
    build();
    if (!data()) cap.set('No trace for this block yet.');
    applyStep(0, { animate: false });
    ctx.setSteps(STEPS, (i) => { if (!autoTick) stopAuto(); applyStep(i); });
    appear([header.el, picker.el, table], { stagger: MOTION.stagger });
    // explain itself the first time it is shown; the first pointer interaction stops it
    const advance = () => {
      if (step >= STEPS.length - 1) return;
      autoTick = true; ctx.setStep(step + 1); autoTick = false;
      auto = setTimeout(advance, MOTION.step);
    };
    auto = setTimeout(advance, 900);
    el.addEventListener('pointerdown', stopAuto);
    el.addEventListener('atlas:interrupt', stopAuto);

    const offs = [
      ctx.on('trace', () => { altKey = ''; altProbs = null; build(); applyStep(step, { animate: false }); }),
      ctx.on('select', () => { build(); applyStep(step, { animate: false }); }),
    ];
    return {
      destroy() { clr(); stopAuto(); offs.forEach((f) => f()); st.destroy(); },
    };
  },
};

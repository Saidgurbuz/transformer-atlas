// The residual add for one sub-layer, on the real numbers of one token.
//   x (stream before) + a (sub-layer output) = sum; post-LN then normalises the sum.
// Steps: Stream → Sub-layer output → Add → Across depth → Without it.
// Every value comes from ctx.state.trace (and model.params for the post-LN norm check).
// "Without it" runs a clone with every residual add switched off (engine.cloneModel + config + infer).
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, fly, appear, useStyle, ZONES, MOTION } from '../kit.js';
import { h, fmt, texHTML, num, reducedMotion, clamp, diverging } from '../../ui/index.js';

const SUB = { 1: 'attention', 2: 'MLP' };
const CELL = 42, CELL_H = 44, GAP = 4;            // hero table: cell width and height, gap (design units)
const TX = 330, TY = 300;                         // hero table origin; its row labels sit left of it
const LABEL_X = ZONES.hero.x, LABEL_W = 150;      // row labels
const ROW_H = 72;
const TOPS = [0, ROW_H, 2 * ROW_H + 14, 3 * ROW_H + 14];   // rows inside the table: x, a, sum, normalised
const OP = 48;                                    // the ⊕ node (guide: 48 to 64 units)
const SX = ZONES.side.x, SW = ZONES.side.w;       // right column: the equation
const FLOW = 640, STAGGER = 34;                   // add animation: flight time and per-column delay (ms)
const CHART = { x: TX, y: TY, w: 640, h: 320 };   // depth chart: small, beside the step
const SWITCH = { x: ZONES.hero.x, y: 270 };       // step 4: the switch sits above the list
const BARS = { x: ZONES.hero.x, y: 330, w: 640, row: 40, n: 8 };
const READ = { x: 790, y: 330, w: 340 };          // step 4: the readout, beside the top bar
const EASE = MOTION.easeOut, MOVE = MOTION.ease;

useStyle('residual-scene', `
.rs-part { transition: opacity ${MOTION.appear}ms ${EASE}, transform ${MOTION.appear}ms ${EASE}; }
.rs-part.rs-off { opacity: 0; pointer-events: none; }
.rs-table { position: absolute; }
.rs-row { position: absolute; left: 0; display: flex; gap: ${GAP}px; transition: opacity ${MOTION.appear}ms ${EASE}; }
.rs-row.pending, .rs-rowlabel.pending { opacity: 0; pointer-events: none; }
.rs-idx { width: ${CELL}px; text-align: center; font: 500 12px/1 var(--font-mono); color: var(--fg-3); }
.rs-cell { width: ${CELL}px; height: ${CELL_H}px; border-radius: 3px; display: grid; place-items: center;
  font: 600 12.5px/1 var(--font-mono); color: var(--fg); box-shadow: inset 0 -1px 0 rgba(0,0,0,.14);
  transition: background-color 500ms ease, opacity 300ms ease; }
.rs-cell.ell { background: none !important; box-shadow: none; color: var(--fg-3); font-size: 14px; }
.rs-cell.hid { opacity: 0; pointer-events: none; }
.rs-rowlabel { position: absolute; width: ${LABEL_W}px; transition: opacity ${MOTION.appear}ms ${EASE}; }
.rs-rowlabel .rs-sym { font-size: 17px; line-height: 1.2; }
.rs-rowlabel .rs-sub { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); margin-top: 4px; }
.rs-op { position: absolute; width: ${OP}px; height: ${OP}px; border-radius: 50%; display: grid; place-items: center;
  background: var(--surface); border: 2px solid var(--c-resid); color: var(--fg); font-size: 20px; box-shadow: var(--shadow-1); }
.rs-rule { position: absolute; height: 0; border-top: 1.5px solid var(--line-strong); }
.rs-eq { width: ${SW}px; display: grid; gap: 12px; }
.rs-eq-main { font-size: 26px; }
.rs-eq-live { font-size: 18px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.rs-stat { font: 500 15px/1.4 var(--font-mono); color: var(--fg-2); }
.rs-chart { position: absolute; }
.rs-chart text { font: 500 15px/1 var(--font-mono); fill: var(--fg-3); }
.rs-chart .rs-hl-text { fill: var(--fg); font-weight: 700; }
.rs-chart .rs-grid { stroke: var(--line); }
.rs-chart .rs-axis { stroke: var(--line-strong); }
.rs-switch { position: absolute; display: inline-flex; align-items: center; gap: 12px; height: 40px; padding: 0 16px 0 6px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--fg-2); font: 600 15px/1 var(--font-mono); cursor: pointer; box-shadow: var(--shadow-1); }
.rs-switch .rs-track { position: relative; width: 44px; height: 26px; border-radius: 999px; background: var(--surface-3); transition: background-color ${MOTION.appear}ms ease; }
.rs-switch .rs-knob { position: absolute; top: 3px; left: 3px; width: 20px; height: 20px; border-radius: 50%; background: var(--surface); box-shadow: var(--shadow-1); transition: transform ${MOTION.move}ms ${MOVE}; }
.rs-switch[aria-checked="true"] .rs-track { background: var(--c-resid); }
.rs-switch[aria-checked="true"] .rs-knob { transform: translateX(18px); }
.rs-switch:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.rs-bars { position: absolute; }
.rs-brow { position: absolute; left: 0; top: 0; width: ${BARS.w}px; height: 36px; display: flex; align-items: center; gap: 14px; transition: transform ${MOTION.move}ms ${MOVE}, opacity ${MOTION.appear}ms ${EASE}; }
.rs-brow .rs-tok { width: 64px; text-align: right; font: 600 17px/1 var(--font-mono); color: var(--fg); white-space: pre; }
.rs-brow .rs-track2 { position: relative; flex: 1; height: 22px; border-radius: 6px; background: var(--surface-2); overflow: hidden; }
.rs-brow .rs-fill { position: absolute; left: 0; top: 0; bottom: 0; width: 0; border-radius: 6px; background: var(--c-resid); transition: width ${MOTION.move}ms ${MOVE}, background-color 400ms ease; }
.rs-brow.top .rs-track2 { outline: 2px solid var(--accent); outline-offset: 2px; }
.rs-brow.top .rs-tok, .rs-brow.top .rs-p { font-weight: 700; }
.rs-brow .rs-p { width: 84px; font: 600 15px/1 var(--font-mono); color: var(--fg-2); text-align: right; }
.rs-readout { position: absolute; font: 500 15px/1.6 var(--font-mono); color: var(--fg-3); }
.rs-readout > div.on { color: var(--fg); }
.rs-readout b { font-weight: 700; }
@media (prefers-reduced-motion: reduce) { .rs-part, .rs-row, .rs-cell, .rs-rowlabel, .rs-brow, .rs-fill, .rs-knob, .rs-track { transition: none !important; } }
`);

const norm2 = (v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
const maxAbsDiff = (a, b) => { let e = 0; for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i])); return e; };
const sumVec = (a, b) => a.map((v, i) => v + b[i]);
const cellText = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2).replace('-', '−'));
const sg = (v) => (v < 0 ? `- ${num(-v, 3)}` : `+ ${num(v, 3)}`);
const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const valueRange = (D) => Math.max(1e-9, ...[D.x, D.a, D.s, D.lnOut || D.s].map((v) => Math.max(...v.map((z) => Math.abs(z)))));
const f3 = (v) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3).replace('-', '−');

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
    const STEPS = ['Stream', 'Sub-layer output', 'Add', 'Across depth', 'Without it'];
    let step = 0, gen = 0, timers = [], auto = null, autoTick = false;
    let altProbs = null, altKey = '', showAlt = false;
    let cols = [];                                  // displayed dimensions: all of them, or first 12, a gap (-1) and last 4
    let cellsX = [], cellsA = [], cellsS = [], cellsN = [];
    const barRows = new Map();

    const clr = () => { gen++; timers.forEach(clearTimeout); timers = []; };
    const later = (ms, fn) => { const g = gen; timers.push(setTimeout(() => { if (g === gen) fn(); }, ms)); };
    const stopAuto = () => { clearTimeout(auto); auto = null; };
    const setRow = (rowEl, labEl, on) => { rowEl.classList.toggle('pending', !on); labEl.classList.toggle('pending', !on); };

    // ------------------------------------------------------------ shared parts
    const header = sceneHeader(st, { chips: [[`Block ${l}`, 'resid'], [n === 1 ? 'attention' : 'MLP', n === 1 ? 'attn' : 'mlp']], title: ctx.node.title });
    const picker = tokenPicker(st, ctx, { label: 'query', kind: 'resid' });
    const cap = sceneCaption(st, { w: 1060 });

    // ------------------------------------------------------------ hero table: steps 0 to 2 (its parts share one fade)
    const table = h('div', { class: 'rs-table rs-part' });
    st.add(table, { x: TX, y: TY });
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

    // ------------------------------------------------------------ depth chart: step 3
    const chart = h('div', { class: 'rs-chart rs-part rs-off' });
    st.add(chart, { x: CHART.x, y: CHART.y });

    // ------------------------------------------------------------ without it: switch, bars, readout: step 4
    const sw = h('button', { type: 'button', class: 'rs-switch rs-part rs-off', role: 'switch', 'aria-checked': 'true' },
      h('span', { class: 'rs-track' }, h('span', { class: 'rs-knob' })), h('span', { class: 'rs-swlab' }, 'all residual adds: on'));
    st.add(sw, { x: SWITCH.x, y: SWITCH.y });
    const bars = h('div', { class: 'rs-bars rs-part rs-off', style: { width: `${BARS.w}px`, height: `${BARS.n * BARS.row}px` } });
    st.add(bars, { x: BARS.x, y: BARS.y });
    const readout = h('div', { class: 'rs-readout rs-part rs-off', style: { width: `${READ.w}px` } });
    st.add(readout, { x: READ.x, y: READ.y });

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
      let lnOut = null, eLn = 0, stats = null;
      if (!pre) {
        stats = normStats(s, c);
        lnOut = normRow(s, S.model, `blocks.${l}.ln${n}`, stats);
        eLn = maxAbsDiff(lnOut, out);
      }
      // depth profile: the embedding, then resid_mid and resid_out of every block
      const pts = [{ k: 'emb.out', label: 'emb' }];
      for (let i = 0; i < c.n_layer; i++) pts.push({ k: `L${i}.resid_mid`, label: `${i}·mid` }, { k: `L${i}.resid_out`, label: `${i}·out` });
      const prof = pts.map((p) => { const r = row(p.k); return { label: p.label, key: p.k, norm: r ? norm2(r) : NaN }; });
      return {
        q, T, pre, res, c, x, a, s, out, d: x.length, xKey, subKey, sumKey, outKey, eSum, eLn, lnOut, stats,
        nx: norm2(x), na: norm2(a), nout: norm2(out), prof, hiIdx: prof.findIndex((p) => p.key === outKey),
        weightsRandom: S.weights === 'random',
      };
    }

    // ------------------------------------------------------------ table build
    function build() {
      const D = data();
      if (!D) return;
      cols = D.d <= 17 ? [...Array(D.d).keys()] : [...Array(12).keys(), -1, ...Array.from({ length: 4 }, (_, k) => D.d - 4 + k)];
      const W = cols.length * CELL + (cols.length - 1) * GAP;
      table.style.width = `${W}px`;
      rule.style.width = `${W}px`;
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
      const who = n === 1 ? 'attention' : 'MLP';
      const N = normNames(D.c);
      const tok = `For query “${esc(ctx.label(D.q))}” at position ${D.q},`;
      const first = D.res ? `Block ${l} adds its ${who} output a to the stream.` : `Residual off: block ${l}’s ${who} output a replaces the stream.`;
      const then = !D.res ? `${tok} a is the new stream.` : D.pre ? `${tok} x + a is the stream after this add.` : `${tok} ${N.word}(x + a) is the stream after this add.`;
      header.setSub(`${first} ${then}`);
      paintValues(D);
    }

    function paintRow(cells, values, range) {
      cols.forEach((i, k) => {
        if (i < 0) return;
        cells[k].style.background = diverging(clamp(values[i] / range, -1, 1));
        cells[k].textContent = cellText(values[i]);
      });
    }

    // paint numbers and colours for the current selection, without animation
    function paintValues(D, { sum = true, norm = true } = {}) {
      const range = valueRange(D);
      paintRow(cellsX, D.x, range);
      paintRow(cellsA, D.a, range);
      if (sum) paintRow(cellsS, D.s, range);
      if (norm && D.lnOut) paintRow(cellsN, D.lnOut, range);
    }

    // the add, animated: x and a fly down column by column and land in the sum row (without x, residual off)
    function addAnimated(D) {
      const range = valueRange(D);
      setRow(rowN, labN, false);
      cellsS.forEach((c, k) => { if (cols[k] >= 0) c.classList.add('hid'); });
      paintValues(D, { sum: false, norm: false });
      cols.forEach((i, k) => {
        if (i < 0) return;
        const delay = k * STAGGER;
        later(delay, () => {
          if (D.res) fly(st, cellsX[k], cellsS[k], { duration: FLOW, arc: 0 });
          fly(st, cellsA[k], cellsS[k], { duration: FLOW, delay: 90, arc: 0 });
        });
        later(delay + FLOW, () => {
          cellsS[k].classList.remove('hid');
          cellsS[k].style.background = diverging(clamp(D.s[i] / range, -1, 1));
          cellsS[k].textContent = cellText(D.s[i]);
        });
      });
      if (D.pre) return;
      // post-LN: the normalised row starts as the sum and rescales to the norm of s
      paintRow(cellsN, D.s, range);
      later(cols.length * STAGGER + FLOW + 220, () => {
        setRow(rowN, labN, true);
        for (let k = 1; k <= 14; k++) later(k * 40, () => paintRow(cellsN, D.s.map((v, j) => v + (D.lnOut[j] - v) * (k / 14)), range));
      });
    }

    // ------------------------------------------------------------ the equations and captions
    function paintText(D) {
      const { q, pre, res, x, a, s, c, stats, lnOut, eSum, eLn, d } = D;
      const N = normNames(c);
      const outSym = n === 1 ? '\\xres^{\\text{mid}}' : '\\xres^{\\text{out}}';
      const subTex = n === 1 ? (pre ? `\\Attn(${N.tex}(\\xres))` : '\\Attn(\\xres)') : (pre ? `\\MLP(${N.tex}(\\xres^{\\text{mid}}))` : '\\MLP(\\xres^{\\text{mid}})');
      const posNote = c.pos_encoding === 'rope' || c.pos_encoding === 'alibi' ? ' Position is not in x here: RoPE or ALiBi act inside attention.' : '';
      const embWords = c.pos_encoding === 'learned' || c.pos_encoding === 'sinusoidal' ? 'the embedding (token plus position)' : 'the token embedding';
      const from = n === 1 ? (l === 0 ? embWords : `block ${l - 1}’s output`) : `the attention add of block ${l}`;
      let eqMain = '', eqLive = '', words = null;
      switch (step) {
        case 0:
          eqMain = `\\xres_{${q}} \\in \\R^{${d}}`;
          eqLive = `\\|\\xres_{${q}}\\|_2 = ${num(D.nx, 2)}`;
          words = `The stream entering this sub-layer is x, the vector of query “${esc(ctx.label(q))}” at position ${q}. It comes from ${from}.${posNote}`;
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
            const how = N.rms ? `It is divided by the root mean square, with no mean removed and no shift, ${after}` : `The mean μ is removed and the result is divided by σ, ${after}`;
            eqMain = `\\begin{aligned} s &= ${res ? '\\xres + a' : 'a'} \\\\ ${outSym} &= ${N.tex}(s) \\end{aligned}`;
            eqLive = res ? `${num(x[0], 3)} ${sg(a[0])} = ${num(s[0], 3)}\\;(i{=}0)` : `s_{0} = ${num(s[0], 3)}`;
            words = `${first} ${how}${D.weightsRandom ? ' The weights are untrained (random).' : ''}`;
          }
          break;
        case 3:
          eqMain = `\\|${outSym}\\|_2 \\text{ across depth}`;
          eqLive = `\\|${outSym}_{${q}}\\|_2 = ${num(D.nout, 2)}`;
          words = !res
            ? 'Residual off: each block point is only its sub-layer output, so nothing accumulates. The highlighted point is this block, and the first point is the embedding.'
            : pre
              ? 'The highlighted point is this block’s stream after the add. Each add keeps what earlier blocks wrote, so the norm reflects every write so far.'
              : `The highlighted point is this block’s normalised output. With gain 1${N.shift ? ' and bias 0' : ''}, a block point has norm √d up to the norm’s epsilon (here √${d} = ${num(Math.sqrt(d), 2)}). The embedding is not normalised.`;
          break;
        default:
          eqMain = `p_{${q}}(\\cdot) = \\softmax(\\text{logits}_{${q}})`;
      }
      const liveProps = { class: 'rs-eq-live', html: texHTML(eqLive, { display: true }) };
      if (step === 2) liveProps.title = 'values rounded to three decimals';
      eq.replaceChildren(...[
        h('div', { class: 'k-label', style: { textAlign: 'left', color: 'var(--c-resid)' } }, STEPS[step]),
        h('div', { class: 'rs-eq-main', html: texHTML(eqMain, { display: true }) }),
        eqLive ? h('div', liveProps) : null,
      ].filter(Boolean));
      if (step === 2) {
        const bd = checkBadge(eSum, 0, { label: `${res ? 'sum' : 'replacement'} equals the traced value` });
        bd.title = `max |Δ| ${fmt(eSum, 1)} over ${d} dimensions`;
        eq.append(bd);
      }
      if (step === 2 && !pre && lnOut) {
        const bl = checkBadge(eLn, 0, { label: `${N.word}(s) equals the traced stream` });
        bl.title = `max |Δ| ${fmt(eLn, 1)} over ${d} dimensions`;
        const stat = N.rms ? `rms(s) = ${f3(stats.sd)} over ${d} dimensions` : `μ = ${f3(stats.mu)} · σ = ${f3(stats.sd)} over ${d} dimensions`;
        eq.append(bl, h('div', { class: 'rs-stat' }, stat));
      }
      if (words != null) cap.set(words);
    }

    // ------------------------------------------------------------ the depth chart (step 3)
    function paintChart(D) {
      const W = CHART.w, Hh = CHART.h, m = { l: 62, r: 20, t: 46, b: 56 };
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
        svg.append(mk('text', { x: m.l - 10, y: Y(v) + 5, 'text-anchor': 'end' }, fmt(v, 1)));
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
      svg.append(mk('text', { x: m.l, y: m.t - 18 }, '‖stream‖₂ for this token'));
      svg.append(mk('text', { x: W - m.r, y: Hh - 6, 'text-anchor': 'end' }, 'emb, then each block: mid, out'));
      chart.replaceChildren(svg);
      chart.dataset.trace = D.outKey;
    }

    // ------------------------------------------------------------ without it (step 4)
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

    function paintBars(D) {
      const S = ctx.state, V = S.config.vocab_size, q = D.q, cfgOn = !!D.c.residual;
      const trueP = Array.from(S.trace.probs.data.subarray(q * V, (q + 1) * V));
      const alt = showAlt ? altDistribution(D).slice(q * V, (q + 1) * V) : null;
      const shown = alt ?? trueP;
      const topOf = (p, k = 5) => Array.from(p.keys()).sort((i, j) => p[j] - p[i]).slice(0, k);
      const top1 = topOf(trueP, 1)[0], top1Alt = alt ? topOf(alt, 1)[0] : -1;
      const cand = [...new Set([...topOf(trueP, 5), ...(alt ? topOf(alt, 5) : [])])];
      // the named tokens (each run’s top) always get a bar; the other slots go to the highest on screen
      const named = [...new Set([top1, top1Alt])].filter((id) => id >= 0);
      const others = cand.filter((id) => !named.includes(id) && Math.max(trueP[id], alt ? alt[id] : 0) >= 0.005)
        .sort((i, j) => shown[j] - shown[i]);
      const union = [...named, ...others].slice(0, BARS.n).sort((i, j) => shown[j] - shown[i]);
      const maxP = Math.max(1e-9, ...union.map((id) => Math.max(trueP[id], alt ? alt[id] : 0)));   // one scale for both runs
      // keyed rows: the same element survives a toggle, so its bar and position tween
      for (const [id, el] of barRows) if (!union.includes(id)) { el.remove(); barRows.delete(id); }
      union.forEach((id, r) => {
        let row = barRows.get(id);
        if (!row) {
          row = h('div', { class: 'rs-brow' }, h('span', { class: 'rs-tok' }, S.tokenizer.label(id)),
            h('div', { class: 'rs-track2' }, h('div', { class: 'rs-fill' })), h('span', { class: 'rs-p' }));
          barRows.set(id, row);
          bars.append(row);
        }
        row.style.transform = `translateY(${r * BARS.row}px)`;
        row.style.opacity = r < 5 || named.includes(id) ? '1' : '0.35';   // the named tokens are never faded
        row.classList.toggle('top', r === 0);
        row.querySelector('.rs-fill').style.width = `${(shown[id] / maxP) * 100}%`;
        row.querySelector('.rs-p').textContent = `${(shown[id] * 100).toFixed(1)}%`;
      });
      // the readout names each run by its residual state; the one on screen is marked
      const pct = (p, id) => `${(p[id] * 100).toFixed(1)}%`;
      const lab = (id) => `“${esc(S.tokenizer.label(id))}”`;
      const line = (name, p, id, on) => `<div class="${on ? 'on' : ''}">${name}: ${lab(id)} <b>${pct(p, id)}</b></div>`;
      let readHTML, headline;
      if (!alt) {
        readHTML = line(cfgOn ? 'with every add' : 'without every add', trueP, top1, true);
        headline = cfgOn
          ? 'As configured, every residual add in every block is on. Switch them off to run the same weights without any of them.'
          : 'As configured, every residual add in every block is off. Switch them on to run the same weights with all of them.';
      } else {
        const withP = cfgOn ? trueP : alt, withoutP = cfgOn ? alt : trueP;
        const tW = cfgOn ? top1 : top1Alt, tO = cfgOn ? top1Alt : top1;
        const withOn = showAlt !== cfgOn;                 // the run with every add is the one on screen
        readHTML = line('with every add', withP, tW, withOn) + line('without every add', withoutP, tO, !withOn);
        headline = tW === tO
          ? `Top prediction after “${esc(ctx.label(q))}”: ${lab(tW)} stays on top, at ${pct(withP, tW)} with every residual add in every block and ${pct(withoutP, tW)} without them.`
          : `Without every residual add in every block, the top prediction after “${esc(ctx.label(q))}” is ${lab(tO)} at ${pct(withoutP, tO)}. With them it is ${lab(tW)} at ${pct(withP, tW)}.`;
      }
      readout.innerHTML = readHTML;
      cap.set(headline);
      const onNow = showAlt ? !cfgOn : cfgOn;
      sw.setAttribute('aria-checked', String(onNow));
      sw.querySelector('.rs-swlab').textContent = `all residual adds: ${onNow ? 'on' : 'off'}`;
      sw.title = `Run the same weights with every residual add in every block switched ${onNow ? 'off' : 'on'}`;
    }

    // ------------------------------------------------------------ steps
    function applyStep(i, { animate = true } = {}) {
      clr();
      step = clamp(i, 0, STEPS.length - 1);
      const D = data();
      if (!D) return;
      cellsS.forEach((c) => c.classList.remove('hid'));
      const show = (el, on) => el.classList.toggle('rs-off', !on);
      show(table, step <= 2);
      show(opEl, step >= 1);
      show(rule, step === 2);
      show(chart, step === 3);
      show(bars, step === 4);
      show(sw, step === 4);
      show(readout, step === 4);
      if (step === 3) paintChart(D);
      if (step === 4) paintBars(D);
      if (step <= 1) {
        setRow(rowX, labX, true); setRow(rowA, labA, step === 1); setRow(rowS, labS, false); setRow(rowN, labN, false);
        paintValues(D);
      } else if (step === 2) {
        setRow(rowX, labX, true); setRow(rowA, labA, true); setRow(rowS, labS, true);
        if (!reducedMotion() && animate) addAnimated(D);
        else { setRow(rowN, labN, !D.pre); paintValues(D); }
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
    sw.addEventListener('click', () => {
      stopAuto();
      showAlt = !showAlt;
      const D = data();
      if (D) paintBars(D);
    });

    const offs = [
      ctx.on('trace', () => { altKey = ''; altProbs = null; build(); applyStep(step, { animate: false }); }),
      ctx.on('select', () => { build(); applyStep(step, { animate: false }); }),
    ];
    return {
      destroy() { clr(); stopAuto(); offs.forEach((f) => f()); st.destroy(); },
    };
  },
};

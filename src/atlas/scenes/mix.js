// Weighted sum of values: what one head outputs for each token.
// For the selected query q, the value vectors v_j of the KV head are columns (one per key
// position) and their attention weights A_qj are bars above them. Each column shrinks to its
// weight, flies into a running sum, and the sum is o_q, checked against the traced
// L{l}.attn.ctx[h, q]. The last step shows every query at once: the head's T x d_h output, checked
// against the whole trace.
// Composition: focus + context (STYLE_GUIDE section 7). The hero is one query's computation; the
// side column holds the equation and a thumbnail of the whole output. The last step makes the
// all-queries output the hero, and the side keeps its equation.
// Not offered: a head picker. The head is part of this node's id (mix.l.h), so another head is
// another node, which needs sibling navigation (see the handoff notes).
// Reference scene: softmax.js. Every number comes from the trace.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, chip, vec, op, texLabel, fly, appear, useStyle, MOTION, ZONES } from '../kit.js';
import { h, fmt, num, texHTML, clamp, reducedMotion, diverging, cssVar, onThemeChange } from '../../ui/index.js';

const STEPS = ['Values', 'Weights', 'Weighted sum', 'Every query'];
// the key row is at most KEYS_MAX wide (slot pitch KEYS_MAX / T), so the running sum stays left of x 1090
const KEYS_MAX = 800;
// the value columns are at most COL_MAX tall, so the key row and the output name clear the caption
const COL_MAX = 224;
const BIG = { y: 270, h: 490 };       // the step-3 matrix is centred in the hero band y 270 … 760
const HERO = { x: 110, w: 1030 };     // hero zone x 110 … 1140; the side column starts at 1180
const SIDE = { x: 1180, y: 270, w: 340 };
const YZ = 470;                       // zero line of the weight bars
const YCOL = 504;                     // top of the value columns and of the running sum
const BAR_MAX = 170;                  // bar height of a weight of 1, design units
const GUT = 56;                       // gutter for the axis tags (A, v_j)
const GAP = 48;                       // gap between the last key slot and the running sum
const FLY_MS = 640;                   // one term flies into the sum (kit fly: 600–800 ms)
const SUM_MS = 1860;                  // the gaps between arrivals share this: the whole sum takes about 2.5 s
const LABEL_CH = 7.6;                 // width of one character of a 12 px mono bar label, with a margin

useStyle('mix', `
.mx-col .k-cells { padding: 2px; border-radius: 5px; transition: opacity .6s cubic-bezier(.65,0,.35,1); }
.mx-col.mx-masked .k-cell { background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--c-mask) 70%, transparent) 0 2px, transparent 2px 6px) !important; }
.mx-col.mx-q .k-cells { outline: 2px solid var(--accent); outline-offset: 3px; }
.mx-fade { transition: opacity .5s cubic-bezier(.2,.7,.2,1); }
.mx-off { opacity: 0 !important; pointer-events: none !important; }
.mx-still, .mx-still * { transition: none !important; }
.mx-bars { position: absolute; }
.mx-zero { position: absolute; left: 0; right: 0; bottom: 0; border-top: 1px dashed var(--line-strong); }
.mx-bar { position: absolute; bottom: 0; border-radius: 4px 4px 2px 2px; background: linear-gradient(180deg, color-mix(in srgb, var(--c-attn) 92%, white), var(--c-attn));
  box-shadow: 0 6px 14px rgba(0,0,0,.18); transition: height .6s cubic-bezier(.65,0,.35,1); }
.mx-bar.masked { background: repeating-linear-gradient(135deg, var(--c-mask) 0 2px, transparent 2px 5px); box-shadow: none; border-radius: 2px; }
.mx-bar-val { position: absolute; left: 50%; bottom: 100%; margin-bottom: 5px; translate: -50% 0; font: 600 12px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.mx-keys { display: flex; }
.mx-key { flex: none; text-align: center; font: 600 15px/1 var(--font-mono); color: var(--fg-2); overflow: hidden; white-space: nowrap; transition: opacity .4s, color .4s; }
.mx-key.dim { opacity: .3; }
.mx-key.cur { color: var(--accent); }
.mx-well { position: absolute; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--line); box-shadow: 0 10px 24px rgba(0,0,0,.12); }
.mx-count { font: 600 13px/1.35 var(--font-mono); color: var(--fg-2); white-space: pre; text-align: center; }
.mx-canvas { display: block; }
.mx-side { display: flex; flex-direction: column; gap: 10px; width: ${SIDE.w}px; }
.mx-side > * { flex: none; }
.mx-stepname { font: 600 17px/1.2 var(--font-mono); color: var(--fg-2); }
.mx-eq-main { font-size: 22px; white-space: nowrap; overflow-x: auto; }
.mx-eq-main .katex-display { margin: .3em 0; }
.mx-eq-live { font-size: 17px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); display: grid; gap: 8px; }
.mx-tbl { display: grid; gap: 3px 8px; font: 500 14px/1.35 var(--font-mono); color: var(--fg); align-items: baseline; }
.mx-tbl > div { white-space: nowrap; }
.mx-tbl .hd { color: var(--fg-3); font-size: 12px; }
.mx-tbl .r { text-align: right; }
.mx-tbl .dim { color: var(--fg-3); }
.mx-tbl .tot { border-top: 1px solid var(--line); padding-top: 4px; }
.mx-thumb { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 8px 12px; }
.mx-thumb-note { flex: 1 1 110px; font: 500 13px/1.45 var(--font-mono); color: var(--fg-3); }
`);

// A matrix drawn on a canvas in the kit's diverging colours. A repaint is one pass over the cells
// and reads no layout, so a new forward pass or query does not stall the page (a DOM cell per entry
// does). Hovering marks the cell with data-cell, so the atlas hover card shows its traced value.
function matrixCanvas({ rows, cols, cell, traceKey, head, label, accent }) {
  const gap = cell > 6 ? 1 : 0, W = cols * cell, H = rows * cell;
  const dpr = window.devicePixelRatio || 1;
  const cv = h('canvas', { class: 'mx-canvas', role: 'img', 'aria-label': label, dataset: { trace: traceKey } });
  // the matrix sits in the kit's matrix frame (.k-mat-body: padding, radius, hairline border, shadow)
  const frame = h('div', { class: 'k-mat', style: { '--kind': 'var(--c-v)' } }, h('div', { class: 'k-mat-body' }, cv));
  cv.style.width = `${W}px`;
  cv.style.height = `${H}px`;
  cv.width = Math.round(W * dpr);
  cv.height = Math.round(H * dpr);
  let data = null, range = 1, hiRow = -1;
  function draw() {
    if (!data) return;
    const g2 = cv.getContext('2d');
    g2.setTransform(dpr, 0, 0, dpr, 0, 0);
    g2.clearRect(0, 0, W, H);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      g2.fillStyle = diverging(data[r * cols + c] / range);
      g2.fillRect(c * cell, r * cell, cell - gap, cell - gap);
    }
    if (hiRow >= 0) { g2.strokeStyle = accent(); g2.lineWidth = 2; g2.strokeRect(1, hiRow * cell + 1, W - 2, cell - 2); }
  }
  let last = '';
  cv.addEventListener('pointermove', (e) => {
    const b = cv.getBoundingClientRect();
    const c = Math.floor(((e.clientX - b.left) / b.width) * cols), r = Math.floor(((e.clientY - b.top) / b.height) * rows);
    const key = r >= 0 && c >= 0 && r < rows && c < cols ? `${r},${c},${head}` : '';
    if (key === last) return;
    last = key;
    if (key) cv.dataset.cell = key; else delete cv.dataset.cell;
  });
  const offTheme = onThemeChange(draw);
  return {
    el: frame,
    update({ data: d = null, range: rg = null, hiRow: hr = null } = {}) {
      const changed = d !== null || (rg !== null && rg !== range) || (hr !== null && hr !== hiRow);
      if (d !== null) data = d;
      if (rg !== null) range = rg;
      if (hr !== null) hiRow = hr;
      if (changed) draw();
    },
    destroy() { offTheme(); },
  };
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const l = ctx.params.l, hd = ctx.params.h;
    // the accent colour is read once per theme, not on every repaint (a style read in a repaint stalls)
    const theme = { accent: cssVar('--accent') };
    const offTheme = onThemeChange(() => { theme.accent = cssVar('--accent'); });   // registered before the matrices
    const accent = () => theme.accent;
    const cfg = () => ctx.cfg;
    const esc = (x) => String(x).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const label = (j) => esc(ctx.label(j));
    const pct = (a) => (a >= 0.1 ? `${Math.round(a * 100)}%` : a >= 0.001 ? `${(a * 100).toFixed(1)}%` : a > 0 ? '<0.1%' : '0%');
    const pct1 = (a) => `${(a * 100).toFixed(1)}%`;
    const table = (cols, rows) => {
      const t = h('div', { class: 'mx-tbl', style: { gridTemplateColumns: cols.map((c) => c.w).join(' ') } });
      cols.forEach((c) => t.append(h('div', { class: ['hd', c.r && 'r'] }, c.t)));
      rows.forEach((row) => row.cells.forEach((cell, i) => t.append(h('div', { class: [cols[i].r && 'r', row.dim && 'dim', row.tot && 'tot'] }, cell))));
      return t;
    };
    const inputTex = () => (cfg().norm_placement === 'post' ? 'x' : cfg().norm === 'rmsnorm' ? '\\RMSNorm(x)' : '\\LN(x)');
    const chipsFor = () => {
      const c = cfg(), grp = c.n_head / c.n_kv_head;
      const out = [[`Layer ${l}`, 'resid'], [`Head ${hd}`, 'attn']];
      if (grp > 1) { const kv = Math.floor(hd / grp); out.push([`KV head ${kv} · heads ${kv * grp}–${kv * grp + grp - 1}`, 'v']); }
      return out;
    };

    // ------------------------------------------------------------ state
    let step = 0, gen = 0, D = null, built = [], timers = [], active = true;
    let autoOn = !reducedMotion(), autoT = null, autoDriving = false;
    let geoKey = null, geo = null;                   // the layout is rebuilt only when T, d_h or the heads change
    let colEls = [], barEls = [], keyEls = [], heroEls = [], bigEls = [];
    let accWrap = null, accVec = null, readout = null, outLabel = null, big = null, thumb = null;
    let header, tp, cap, stepLabel, eqMain, eqLive, checkSlot, side, thumbSlot, thumbNote;

    const put = (node, opts) => { built.push(node); return st.add(node, opts); };
    const geometryKey = (d) => `${d.T}|${d.dh}|${d.H}|${d.Hkv}`;
    const gapMs = (N) => (N > 1 ? clamp(Math.floor(SUM_MS / (N - 1)), 25, 110) : 0);
    const flightMs = (N) => (N > 1 ? (N - 1) * gapMs(N) : 0) + FLY_MS;
    const colStagger = (T) => Math.min(MOTION.stagger, Math.floor(1200 / T));

    // ------------------------------------------------------------ data (trace only)
    function read() {
      const s = ctx.state, tr = s.trace, T = s.ids.length;
      const keys = [`L${l}.attn.probs`, `L${l}.attn.masked`, `L${l}.attn.v_heads`, `L${l}.attn.ctx`];
      if (!tr || !keys.every((k) => tr[k]) || T < 1) return null;
      const P = tr[keys[0]], M = tr[keys[1]], V = tr[keys[2]], C = tr[keys[3]];
      const H = P.shape[0], dh = C.shape[2], Hkv = V.shape[0];
      if (hd >= H) return null;
      const kv = Math.floor(hd / (H / Hkv));                    // the KV head this query head reads
      const q = Math.min(s.sel.token ?? T - 1, T - 1);
      const row = (hd * T + q) * T;
      const A = Array.from(P.data.subarray(row, row + T));
      const vis = Array.from(M.data.subarray(row, row + T), (x) => x !== -Infinity);
      const v = V.data.subarray(kv * T * dh, (kv + 1) * T * dh);            // [T, dh] of the KV head
      const o = Array.from(C.data.subarray((hd * T + q) * dh, (hd * T + q + 1) * dh));
      const visList = [...Array(T).keys()].filter((j) => vis[j]);
      const oc = new Array(dh).fill(0);                         // o_q recomputed: sum_j A_qj v_j
      for (let j = 0; j < T; j++) for (let k = 0; k < dh; k++) oc[k] += A[j] * v[j * dh + k];
      let kStar = 0;                                            // the output entry shown in the table
      for (let k = 1; k < dh; k++) if (Math.abs(o[k]) > Math.abs(o[kStar])) kStar = k;
      const Amax = Math.max(0, ...visList.map((j) => A[j]));
      const top = [...visList].sort((a, b) => A[b] - A[a]).slice(0, 3);
      let R = 0;                                                // one colour range for v and o: same value, same colour
      for (let i = 0; i < v.length; i++) R = Math.max(R, Math.abs(v[i]));
      R = R || 1;
      const maxOne = Math.max(...oc.map((x, k) => Math.abs(x - o[k])));
      // every query: A (T x T) times V (T x dh) against the traced ctx of this head
      const Aall = P.data.subarray(hd * T * T, (hd + 1) * T * T);
      const Call = C.data.subarray(hd * T * dh, (hd + 1) * T * dh);
      let maxAll = 0;
      for (let r = 0; r < T; r++) for (let k = 0; k < dh; k++) {
        let acc = 0;
        for (let j = 0; j < T; j++) acc += Aall[r * T + j] * v[j * dh + k];
        maxAll = Math.max(maxAll, Math.abs(acc - Call[r * dh + k]));
      }
      return { T, H, Hkv, q, dh, kv, A, vis, visList, Amax, sumA: A.reduce((a, b) => a + b, 0), top, v, oc, o, kStar, R, maxOne, maxAll, Call };
    }

    // bar labels: weights of 1% or more and the top three, largest first; a label is kept only if it
    // clears every label already kept (so crowded slots show the strongest numbers)
    function labelSet(d) {
      const kept = [];
      const cands = d.visList.filter((j) => d.A[j] >= 0.01 || d.top.includes(j));
      for (const j of cands.sort((a, b) => d.A[b] - d.A[a])) {
        const x = (j + 0.5) * geo.p, w = pct(d.A[j]).length * LABEL_CH + 4;
        if (kept.every((k) => Math.abs(x - k.x) >= (w + k.w) / 2 + 4)) kept.push({ j, x, w });
      }
      return new Set(kept.map((k) => k.j));
    }

    // ------------------------------------------------------------ static layout (rebuilt only when the geometry changes)
    function clearBuilt() {
      built.forEach((n) => { n.destroy?.(); n.remove(); });
      built = [];
      accVec?.destroy?.(); accVec = null;
      big?.destroy(); big = null;
      thumb?.destroy(); thumb = null;
      thumbSlot?.replaceChildren();
      colEls = []; barEls = []; keyEls = []; heroEls = []; bigEls = [];
      accWrap = null; readout = null; outLabel = null;
      geoKey = null; geo = null;
    }

    function build(d) {
      clearBuilt();
      const { T, dh } = d;
      geoKey = geometryKey(d);
      const shown = Math.min(dh, 17);                             // long vectors: first 12 … last 4
      const p = Math.min(44, Math.floor(KEYS_MAX / T));           // slot width per key
      const c = clamp(Math.min(p - 10, Math.floor((COL_MAX - 6 - (shown - 1) * 2) / shown)), 4, 12), g = c >= 8 ? 2 : 1;   // value cell and gap: 4 px at least
      const boxH = shown * c + (shown - 1) * g + 6;
      const bw = clamp(c + 6, 4, 22), wellW = c + 18;
      const groupW = GUT + T * p + GAP + wellW;
      const x0 = HERO.x + Math.max(0, (HERO.w - groupW) / 2) + GUT;   // left edge of key slot 0
      const xAcc = x0 + T * p + GAP + wellW / 2;
      const keysY = YCOL + boxH + 10;
      geo = { p, c, g, boxH, wellW, xAcc, keysY };

      // weights: one bar per key rising from the zero line; masked keys are hatched stubs
      const bars = h('div', { class: 'mx-bars', dataset: { trace: `L${l}.attn.probs` }, style: { width: `${T * p}px`, height: `${BAR_MAX + 28}px` } }, h('div', { class: 'mx-zero' }));
      for (let j = 0; j < T; j++) {
        const val = h('span', { class: 'mx-bar-val' });
        const bar = h('div', { class: 'mx-bar', dataset: { cell: `${d.q},${j},${hd}` }, style: { left: `${(j + 0.5) * p - bw / 2}px`, width: `${bw}px`, height: '0px' } }, val);
        bars.append(bar);
        barEls.push({ bar, val });
      }
      heroEls.push(put(bars, { x: x0, y: YZ - BAR_MAX - 28 }));
      heroEls.push(put(texLabel('\\Aw_{tj}', { size: 16, kind: 'attn' }), { x: x0 - 12, y: YZ - 50, anchor: 'rc' }));

      // values: v_j as a column of dh cells, one per key position
      for (let j = 0; j < T; j++) {
        const col = vec(d.v.subarray(j * dh, (j + 1) * dh), {
          orientation: 'v', cell: c, gap: g, kind: 'v', mode: 'diverging', range: d.R,
          trace: `L${l}.attn.v_heads`, cellOf: (k) => `${j},${k},${d.kv}`,
        });
        col.classList.add('mx-col', 'mx-fade');
        put(col, { x: x0 + (j + 0.5) * p, y: YCOL, anchor: 'tc' });
        colEls.push(col);
      }
      heroEls.push(put(texLabel('v_{j}', { size: 16, kind: 'v' }), { x: x0 - 12, y: YCOL + boxH / 2, anchor: 'rc' }));

      const keyRow = h('div', { class: 'mx-keys', style: { width: `${T * p}px` } });
      keyEls = Array.from({ length: T }, (_, j) => h('div', { class: 'mx-key', style: { width: `${p}px`, fontSize: T > 32 ? '11px' : '15px' } }, ctx.label(j)));
      keyRow.append(...keyEls);
      heroEls.push(put(keyRow, { x: x0, y: keysY }));

      // the running sum: a well under a sum sign, named o_q
      accWrap = h('div', { class: 'mx-acc' });
      const well = h('div', { class: 'mx-well', style: { width: `${wellW}px`, height: `${boxH + 16}px` } });
      heroEls.push(put(accWrap, { x: xAcc, y: YCOL, anchor: 'tc' }));
      heroEls.push(put(well, { x: xAcc, y: YCOL - 8, anchor: 'tc', layer: 'back' }));
      heroEls.push(put(op('\\sum', { kind: 'v', size: 48 }), { x: xAcc, y: YCOL - 22, anchor: 'bc' }));
      outLabel = texLabel(`o_{${d.q}}`, { size: 20, kind: 'v' });
      outLabel.dataset.q = String(d.q);
      heroEls.push(put(outLabel, { x: xAcc, y: keysY, anchor: 'tc' }));
      // the count sits above the sum sign, centred on the well and clear of the side column
      readout = put(h('div', { class: 'mx-count' }), { x: xAcc, y: YCOL - 84, anchor: 'bc' });
      heroEls.push(readout);
      heroEls.forEach((e) => e.classList.add('mx-fade'));

      // last step: the whole T x d_h output of this head, as the hero (the query's row is marked)
      const bigCell = clamp(Math.min(Math.floor(400 / T), Math.floor(960 / dh)), 4, 24);
      big = matrixCanvas({ rows: T, cols: dh, cell: bigCell, traceKey: `L${l}.attn.ctx`, head: hd, accent,
        label: `head ${hd} output at every position` });
      big.update({ data: d.Call, range: d.R, hiRow: d.q });
      // the label, then the matrix frame, centred in the hero band
      const cx = HERO.x + HERO.w / 2, frameH = T * bigCell + 18;     // the canvas plus the frame's padding and border
      const topY = BIG.y + Math.max(0, (BIG.h - (frameH + 36)) / 2);
      bigEls = [
        put(texLabel(`\\mathrm{ctx}^{(${hd})} \\in \\R^{${T}\\times ${dh}}`, { size: 18, kind: 'v' }), { x: cx, y: topY, anchor: 'tc' }),
        put(big.el, { x: cx, y: topY + 36, anchor: 'tc' }),
      ];
      bigEls.forEach((e) => e.classList.add('mx-fade'));

      // side thumbnail: the whole output of this head, the query's row marked
      const thumbCell = clamp(Math.min(Math.floor(170 / T), Math.floor(300 / dh)), 2, 6);
      thumb = matrixCanvas({ rows: T, cols: dh, cell: thumbCell, traceKey: `L${l}.attn.ctx`, head: hd, accent,
        label: `head ${hd} output at every position` });
      thumb.update({ data: d.Call, range: d.R, hiRow: -1 });
      thumbNote = h('div', { class: 'mx-thumb-note' });
      thumbSlot.replaceChildren(thumb.el, thumbNote);
    }

    // the same geometry with a new forward pass: update the values in place
    function refreshData(d) {
      colEls.forEach((col, j) => col.update(d.v.subarray(j * d.dh, (j + 1) * d.dh), { range: d.R }));
      big.update({ data: d.Call, range: d.R });
      thumb.update({ data: d.Call, range: d.R });
    }

    // the running sum for one state: zeros, or the final o_q
    function makeAcc(vals, d) {
      const next = vec(vals, { orientation: 'v', cell: geo.c, gap: geo.g, kind: 'v', mode: 'diverging', range: d.R, cellOf: (k) => `${d.q},${k},${hd}` });
      accWrap.replaceChildren(next);
      accVec?.destroy?.();
      accVec = next;
      return next;
    }

    // ------------------------------------------------------------ text
    const mainTex = (d) => [
      `\\V = ${inputTex()}\\,\\Wv^{\\top}${cfg().bias ? ' + b_v' : ''}`,
      `\\Aw_{${d.q},j} \\ge 0, \\qquad \\sum_{j} \\Aw_{${d.q},j} = 1`,
      `o_{${d.q}} = \\sum_{j} \\Aw_{${d.q},j}\\, v_j`,
      `\\mathrm{ctx}^{(${hd})} = \\Aw^{(${hd})}\\,\\V^{(${d.kv})}`,
    ][step];
    const words = (d) => {
      const nMasked = d.T - d.visList.length;
      return [
        `Each position <b>j</b> has a value vector <b>v<sub>j</sub></b> of length d<sub>h</sub> = ${d.dh}, read from KV head ${d.kv}.${nMasked ? ' Hatched columns are masked keys: this query may not see them.' : ''}`,
        `Query “<b>${label(d.q)}</b>” at position ${d.q} gives each key it may see a weight. The bars are the weights and sum to 1.${nMasked ? ' Masked keys get exactly 0.' : ' No key is masked, so every key gets a weight.'}`,
        `Each value enters the sum scaled by its weight. The sum is the head’s output <b>o<sub>${d.q}</sub></b>, which equals <b>L${l}.attn.ctx</b> for this query.`,
        `Doing this for every query gives the head’s output <b>A V</b>: (T×T)(T×d<sub>h</sub>) = (T×d<sub>h</sub>), one row per position. Row ${d.q} is the output for “${label(d.q)}”.`,
      ];
    };
    const ctxCheck = (d) => checkBadge(d.maxOne, 0, { label: `equals L${l}.attn.ctx[${hd}, ${d.q}]`, tol: 1e-9 });
    const allCheck = (d) => checkBadge(d.maxAll, 0, { label: `equals L${l}.attn.ctx, all positions`, tol: 1e-9 });

    // one live line per step, plus a small table of the top terms where the numbers need one
    function live(d) {
      const { q, dh, kv } = d;
      if (step === 0) {
        const first = Array.from(d.v.subarray(q * dh, q * dh + Math.min(4, dh)), (x) => num(x, 2));
        return [h('div', { html: texHTML(`v_{${q}} = (${first.join(',\\,')}${dh > 4 ? ',\\,\\dots' : ''})`) })];
      }
      if (step === 1) {
        const rest = d.visList.length - d.top.length;
        const rows = d.top.map((j) => ({ cells: [String(j), ctx.label(j), pct1(d.A[j])] }));
        if (rest > 0) rows.push({ cells: ['', `${rest} more`, pct1(d.sumA - d.top.reduce((a, j) => a + d.A[j], 0))], dim: true });
        return [
          h('div', { html: texHTML(`\\sum_{j} \\Aw_{${q},j} = ${num(d.sumA, 3)}`) }),
          table([{ t: 'key', w: '46px' }, { t: 'token', w: '1fr' }, { t: 'A', w: '72px', r: true }], rows),
        ];
      }
      if (step === 2) {
        const k = d.kStar, rest = d.visList.filter((j) => !d.top.includes(j));
        const restA = rest.reduce((a, j) => a + d.A[j], 0), restAV = rest.reduce((a, j) => a + d.A[j] * d.v[j * dh + k], 0);
        const rows = d.top.map((j) => ({ cells: [String(j), ctx.label(j), pct1(d.A[j]), fmt(d.v[j * dh + k], 2), fmt(d.A[j] * d.v[j * dh + k], 3)] }));
        if (rest.length) rows.push({ cells: ['', `${rest.length} more`, pct1(restA), '', fmt(restAV, 3)], dim: true });
        rows.push({ cells: ['', 'sum', pct1(d.sumA), '', fmt(d.oc[k], 3)], tot: true });
        return [
          h('div', { html: texHTML(`o_{${q},${k}} = \\sum_{j} \\Aw_{${q},j}\\, v_{j,${k}} = ${num(d.oc[k], 3)}`) }),
          table([{ t: 'j', w: '26px' }, { t: 'token', w: '72px' }, { t: 'A', w: '62px', r: true }, { t: 'v', w: '58px', r: true }, { t: 'A·v', w: '1fr', r: true }], rows),
        ];
      }
      return [h('div', { html: texHTML(`\\max_{t,k}\\big|\\mathrm{ctx}^{(${hd})} - \\Aw^{(${hd})}\\V^{(${kv})}\\big| = ${num(d.maxAll, 1)}`) })];
    }

    // ------------------------------------------------------------ paint one state
    function still() {
      st.root.classList.add('mx-still');
      requestAnimationFrame(() => requestAnimationFrame(() => st.root.classList.remove('mx-still')));
    }

    function setChips() {
      header?.el.querySelector('.k-eyebrow')?.replaceChildren(...chipsFor().map(([t, k]) => chip(t, k)));
    }

    function paint({ animate = true, d = read() } = {}) {
      const g = ++gen;
      timers.forEach(clearTimeout); timers = [];
      st.layers.front.querySelectorAll('.k-ghost').forEach((n) => n.remove());
      D = d;
      setChips();
      if (!d) {
        clearBuilt();
        cap.set(`This forward pass has no attention for layer ${l}, head ${hd}.`);
        checkSlot.replaceChildren();
        return;
      }
      if (geometryKey(d) !== geoKey) build(d);
      const instant = !animate || reducedMotion();
      if (instant) still();
      const q = d.q, N = d.visList.length, heroOn = step < 3;
      heroEls.forEach((e) => e.classList.toggle('mx-off', !heroOn));
      bigEls.forEach((e) => e.classList.toggle('mx-off', heroOn));
      thumbSlot.classList.toggle('mx-off', step === 3);
      colEls.forEach((col, j) => {
        const vis = d.vis[j], cells = col.querySelector('.k-cells');
        col.classList.toggle('mx-off', !heroOn);
        col.classList.toggle('mx-masked', !vis);
        col.classList.toggle('mx-q', j === q);
        cells.style.opacity = !vis ? '0.7' : step === 0 ? '1' : String(0.22 + 0.78 * (d.A[j] / (d.Amax || 1)));
      });
      const labelOn = labelSet(d);
      barEls.forEach(({ bar, val }, j) => {
        const vis = d.vis[j], on = step >= 1 && vis;
        bar.dataset.cell = `${q},${j},${hd}`;
        bar.classList.toggle('masked', !vis);
        bar.style.height = !vis ? '8px' : `${on ? Math.max(2, d.A[j] * BAR_MAX) : 0}px`;
        val.textContent = on && labelOn.has(j) ? pct(d.A[j]) : '';
      });
      keyEls.forEach((k, j) => {
        const name = ctx.label(j);                                  // the token text can change at the same T
        if (k.textContent !== name) k.textContent = name;
        k.classList.toggle('dim', !d.vis[j]); k.classList.toggle('cur', j === q);
      });

      // the running sum: zeros, playing, or final
      const finalSum = step >= 3 || (step === 2 && instant);
      const acc = makeAcc(finalSum ? d.oc : new Array(d.dh).fill(0), d);
      if (finalSum) acc.dataset.trace = `L${l}.attn.ctx`;
      readout.textContent = step === 2 ? `${finalSum ? N : 0} of ${N}\nadded` : '';
      // the output's name follows the query: build names it for the query it was built with
      if (outLabel.dataset.q !== String(q)) { outLabel.dataset.q = String(q); outLabel.innerHTML = texHTML(`o_{${q}}`); }

      header.setSub(step === 3
        ? 'Every position mixes the value vectors of the keys it may see. Each head makes one such mixture per token.'
        : `Query “<b>${label(q)}</b>” at position ${q} mixes the value vectors of the keys it may see. Every head makes one such mixture per token.`);
      cap.set(words(d)[step]);
      stepLabel.textContent = STEPS[step];
      eqMain.innerHTML = texHTML(mainTex(d), { display: true });
      eqLive.replaceChildren(...live(d));
      checkSlot.replaceChildren();
      if (step === 1) checkSlot.append(checkBadge(d.sumA, 1, { label: 'the row sums to 1', tol: 1e-9 }));
      if (step === 2 && finalSum) checkSlot.append(ctxCheck(d));
      if (step === 3) checkSlot.append(allCheck(d));
      thumb.update({ hiRow: step >= 2 ? q : -1 });
      thumb.el.style.opacity = step >= 2 ? '1' : '0.45';
      thumbNote.textContent = `head ${hd} output at every position · row ${q} is “${ctx.label(q)}”`;
      big.update({ hiRow: q });
      if (step === 2 && !instant) flights(g, d, acc);
    }

    // the weighted terms fly one by one into the sum; each arrival updates the running sum
    function flights(g, d, acc) {
      const N = d.visList.length, dh = d.dh, gap = gapMs(N);
      const partial = new Array(dh).fill(0);
      let n = 0;
      const settle = () => {
        if (g !== gen) return;
        acc.dataset.trace = `L${l}.attn.ctx`;
        checkSlot.replaceChildren(ctxCheck(d));
      };
      if (!N) { settle(); return; }
      d.visList.forEach((j, i) => {
        timers.push(setTimeout(() => {
          if (g !== gen) return;
          const landed = fly(st, colEls[j], acc, { duration: FLY_MS, arc: 40 });
          weightGhost(d.A[j]);
          landed.then(() => {
            if (g !== gen) return;
            for (let k = 0; k < dh; k++) partial[k] += d.A[j] * d.v[j * dh + k];
            n += 1;
            acc.update(partial, { range: d.R });
            readout.textContent = `${n} of ${N}\nadded`;
            if (n === N) settle();
          });
        }, i * gap));
      });
    }

    // the flying copy shrinks to its weight while it travels, so the sum visibly weights each term
    function weightGhost(w) {
      const ghost = st.layers.front.lastElementChild;
      const cells = ghost?.classList.contains('k-ghost') ? ghost.querySelector('.k-cells') : null;
      cells?.animate([{ transform: 'scaleY(1)' }, { transform: `scaleY(${w})`, offset: 0.4 }, { transform: `scaleY(${w})` }],
        { duration: FLY_MS, easing: MOTION.ease });
    }

    // ------------------------------------------------------------ steps and auto-play
    const stopAuto = () => { autoOn = false; clearTimeout(autoT); };
    function setStep(i) {
      if (!autoDriving) stopAuto();
      step = i;
      paint({ animate: true });
    }
    function scheduleNext(ms) {
      clearTimeout(autoT);
      if (!autoOn || step >= STEPS.length - 1) return;
      autoT = setTimeout(() => {
        autoDriving = true;
        try { ctx.setStep(step + 1); } finally { autoDriving = false; }
        scheduleNext(step === 2 && D ? flightMs(D.visList.length) : MOTION.step);
      }, ms);
    }

    // ------------------------------------------------------------ mount
    // The node's head becomes the app's head selection, so the story, maths and code drawers agree.
    if (ctx.state.sel.head !== hd) ctx.store.select({ head: hd });

    header = sceneHeader(st, { chips: chipsFor(), title: 'Weighted sum of values' });
    tp = tokenPicker(st, ctx, { label: 'query' });
    cap = sceneCaption(st, { w: ZONES.caption.w });

    side = h('div', { class: 'mx-side' });
    stepLabel = h('div', { class: 'mx-stepname' }, STEPS[0]);
    eqMain = h('div', { class: 'mx-eq-main' });
    eqLive = h('div', { class: 'mx-eq-live' });
    checkSlot = h('div', { style: { minHeight: '30px' } });
    thumbSlot = h('div', { class: 'mx-thumb mx-fade' });
    side.append(stepLabel, eqMain, eqLive, checkSlot, thumbSlot);
    st.add(side, { x: SIDE.x, y: SIDE.y });

    ctx.setSteps(STEPS, setStep);
    D = read();
    paint({ animate: true, d: D });
    const colGap = D ? colStagger(D.T) : MOTION.stagger;
    appear([header.el, tp.el, side, cap.el], { stagger: MOTION.stagger });
    appear(colEls, { stagger: colGap });
    // the first automatic step waits until the last column has appeared
    scheduleNext(Math.max(MOTION.step, D ? (D.T - 1) * colGap + MOTION.appear + 120 : 0));

    const offs = [
      ctx.on('trace', () => {
        if (!active) return;                                  // a hidden scene catches up on resume
        stopAuto();
        const nd = read();
        if (nd && geoKey !== null && geometryKey(nd) === geoKey) refreshData(nd);
        paint({ animate: true, d: nd });
      }),
      ctx.on('select', () => { if (!active) return; stopAuto(); paint({ animate: true }); }),
    ];
    el.addEventListener('pointerdown', stopAuto, { once: true });

    return {
      pause() { active = false; stopAuto(); gen++; timers.forEach(clearTimeout); timers = []; },
      resume() { active = true; paint({ animate: false }); },
      destroy() {
        stopAuto(); gen++; timers.forEach(clearTimeout); timers = [];
        offs.forEach((f) => f());
        offTheme();
        clearBuilt();
        st.destroy();
      },
    };
  },
};

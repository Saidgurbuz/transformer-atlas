// Scores QKᵀ for one head and one query, in four steps on the real trace:
// the query and every key as vectors → the dot product of the strongest key, element by element →
// every score as a bar, and the full S matrix → scaled by 1/√d_h, with a simulated illustration of why.
// Reference scene for contributors: see docs/ATLAS_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, vec, mat, checkBadge, appear, useStyle, MOTION, ZONES } from '../kit.js';
import { h, s, fmt, num, texHTML, diverging, heatmap, reducedMotion, cssVar } from '../../ui/index.js';

const STEPS = ['Queries and keys', 'Dot product', 'All scores', 'Scale by 1/√d_h'];

// Geometry on the 1600 × 900 canvas (docs/STYLE_GUIDE.md): hero x 110…1140, side column x 1180…1520
// (ZONES.side, 340 wide, the column lm-head.js uses). Anchors sit on the 10-unit grid.
const LEFT = ZONES.hero.x;                        // left edge of all hero text (x = 110)
const QX = LEFT + 20;                             // centre of the query vector
const KX0 = 210, KX1 = 1130;                      // key columns span this range
const TOK_Y = 530;                                // top of the token labels
const VEC_BOT = TOK_Y - 10;                       // every vector hangs from this line, just above its token
const BAND_Y = 600, BAND_H = 124, ZERO = 62, HALF = 50;   // score band: zero line and largest bar
const SIDE_X = ZONES.side.x, SIDE_W = ZONES.side.w, EQ_Y = ZONES.hero.y, MAT_Y = 490;     // equation column, then the S matrix below it
const PITCH_MAX = 64;                             // widest key pitch, so short sequences stay compact
const SIM = { seed: 20260611, trials: 60, keys: 16, dims: [4, 8, 16, 32, 64, 128, 256] };
let simCache = null;

// Key labels under the key columns. A word token (“␣liked”) can be wider than its column. A label that
// does not fit is turned 40° with its end on the column's centre, and the turned labels keep clear of each
// other. A turned label that would run down into the score band is cut to the longest width that clears
// it, with an ellipsis. When even turning is too dense, only every few labels is written, centred on its
// column, and the query's label is always one of them; every label keeps its full text on hover (title).
// Widths are measured in the page's own mono font, so any token size fits.
const KEY_TILT = (40 * Math.PI) / 180;
let measurer = null;
/**
 * 'flat': every label fits its slot. 'tilt': turned labels run parallel, slot × sin 40° apart, and `max` is the
 * widest one whose lower corner stays `room` px below the row's top. 'elide': too narrow to turn, so the row
 * writes every `step`-th label, which leaves each written label room for its full width.
 */
function keyLabelMode(labels, slot, px, room) {
  measurer ??= document.createElement('canvas').getContext('2d');
  measurer.font = `600 ${px}px ${cssVar('--font-mono') || 'monospace'}`;
  const widths = labels.map((t) => measurer.measureText(t).width);
  const need = Math.max(0, ...widths) + 4;                          // the widest label, with a margin
  if (need <= slot) return { mode: 'flat', widths, max: Infinity, step: 1 };
  if (slot * Math.sin(KEY_TILT) < px) return { mode: 'elide', widths, max: Infinity, step: Math.ceil(need / slot) };
  // a turned label's lowest corner lies 0.643 × width + 0.883 × px below the row's top
  return { mode: 'tilt', widths, max: Math.max(px, (room - 0.883 * px) / Math.sin(KEY_TILT)), step: 1 };
}

useStyle('scores', `
.scr-q .k-cells, .scr-col .k-cells { padding: 3px 2px; }
.scr-q .k-cell, .scr-col .k-cell { width: var(--cw); }
.scr-col .k-cell.ell { font-size: 10px; }
.scr-col { transition: opacity .4s ease; }
.scr-col.future { opacity: .35; }
.scr-col.dim { opacity: .28; }
.scr-col.focus .k-cells { box-shadow: 0 0 0 2px var(--c-k), 0 8px 18px rgba(0, 0, 0, .14); }
.scr-tok { display: grid; justify-items: center; font: 600 15px/1 var(--font-mono); color: var(--fg); white-space: nowrap; transition: opacity .4s ease; }
.scr-tok.xs { font-size: 12px; }
.scr-tok small { font: 500 12px/1 var(--font-mono); color: var(--fg-3); margin-top: 4px; }
.scr-tok.scr-tilt b { position: absolute; top: 0; right: 50%; transform-origin: 100% 50%; transform: rotate(-40deg); overflow: hidden; text-overflow: ellipsis; }
.scr-tok.scr-tilt small { margin-top: 22px; }      /* the index sits under the end of the turned word */
.scr-tok.future { opacity: .35; }
.scr-tok.dim { opacity: .28; }
.scr-keylab { font-size: 17px; }
.scr-band { position: absolute; }
.scr-zero { position: absolute; left: 0; right: 0; height: 0; border-top: 1px dashed var(--line-strong); }
.scr-bar { position: absolute; border-radius: 3px; box-shadow: 0 4px 10px rgba(0, 0, 0, .14);
  transition: top .6s cubic-bezier(.65,0,.35,1), height .6s cubic-bezier(.65,0,.35,1), opacity .4s ease, background-color .4s ease; }
.scr-bar.masked { background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--c-mask) 55%, transparent) 0 2px, transparent 2px 6px); box-shadow: none; }
.scr-val { position: absolute; left: 50%; translate: -50% 0; font: 600 12px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap;
  transition: top .6s cubic-bezier(.65,0,.35,1); }
.scr-prods { position: absolute; top: 0; transition: opacity .4s ease; }
.scr-pbar { position: absolute; width: 6px; border-radius: 2px; transition: top .45s cubic-bezier(.65,0,.35,1), height .45s cubic-bezier(.65,0,.35,1); }
.scr-ell { position: absolute; width: 9px; text-align: center; font: 600 13px/1 var(--font-mono); color: var(--fg-3); }
.scr-sum { position: absolute; translate: -50% 0; font: 600 17px/1 var(--font-mono); color: var(--fg); white-space: nowrap; transition: opacity .4s ease; }
.scr-sum span { color: var(--fg-2); font-weight: 500; }
.scr-mat { transition: opacity .6s ease; }
.scr-mat .k-label { margin-bottom: 4px; }
.scr-mat .k-mat-body { padding: 6px; }
.scr-eq { width: ${SIDE_W}px; display: grid; gap: 4px; }
.scr-eq > * { min-width: 0; }   /* the column stays 340 wide; an overlong line scrolls in its own box */
.scr-eq > .k-label { margin-bottom: 0; }
.scr-eq-main { font-size: 24px; }
.scr-eq .katex-display { margin: .05em 0; }
.scr-eq-live { font-size: 18px; padding: 6px 10px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.scr-eq-check { transition: opacity .4s ease; }
.scr-eq-note { font-size: 14px; line-height: 1.35; color: var(--fg-2); }
.scr-ghost { position: absolute; border-radius: 3px; border: 1px dashed color-mix(in srgb, var(--fg) 55%, transparent); background: transparent;
  transition: top .6s cubic-bezier(.65,0,.35,1), height .6s cubic-bezier(.65,0,.35,1), opacity .4s ease; }
.scr-inset { width: ${SIDE_W}px; box-sizing: border-box; padding: 10px 12px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent);
  border: 1px solid var(--line); box-shadow: 0 10px 26px rgba(0, 0, 0, .12); transition: opacity .5s ease; }
.scr-plots { display: grid; gap: 8px; }
.scr-xt { font: 500 13px var(--font-mono); fill: var(--fg-2); }
.scr-plot-title { font-size: 13px; line-height: 16px; color: var(--fg-2); white-space: nowrap; }
.scr-plot svg { display: block; overflow: visible; }
.scr-tick { font: 500 13px var(--font-mono); fill: var(--fg-3); }
.scr-axis { stroke: var(--line-strong); }
.scr-grid { stroke: var(--line); }
.scr-legend { display: flex; gap: 12px; align-items: center; margin-top: 8px; font: 500 13px/16px var(--font-mono); color: var(--fg-2); }
.scr-legend i { display: inline-block; width: 16px; border-top: 2px solid; vertical-align: middle; margin-right: 5px; }
.scr-sim { margin-top: 4px; font: 500 13px/16px var(--font-body); color: var(--fg-3); }
@media (prefers-reduced-motion: reduce) {
  .scr-col, .scr-tok, .scr-bar, .scr-ghost, .scr-val, .scr-prods, .scr-pbar, .scr-sum, .scr-mat, .scr-inset, .scr-eq-check { transition: none !important; }
}
`);

/** Visible slots of a vector of length n, elided the way vec() elides (first 12, ellipsis, last 4). */
function shownIdx(n) {
  if (n <= 17) return Array.from({ length: n }, (_, i) => i);
  return [...Array.from({ length: 12 }, (_, i) => i), -1, ...Array.from({ length: 4 }, (_, i) => n - 4 + i)];
}

/**
 * The dot product of the strongest key in two lines at most, so the live equation keeps its size:
 * the first two products, then the last product and the value. For d_h <= 4 every product is shown.
 * The sum is always over all d_h terms.
 */
function dotLines(dh, tm, value) {
  const t = Array.from({ length: dh }, (_, c) => tm(c));
  let rows;
  if (dh > 4) {
    rows = [`${t[0]} + ${t[1]}`, `+ \\cdots + ${t[dh - 1]} = ${value}`];
  } else {
    // two products on the first line, then one a line, so no line is wider than the column
    rows = [dh >= 2 ? `${t[0]} + ${t[1]}` : t[0]];
    for (let c = 2; c < dh; c++) rows.push(`+ ${t[c]}`);
    if (dh === 2) rows.push(`= ${value}`);
    else rows[rows.length - 1] += ` = ${value}`;
  }
  return `\\begin{gathered} ${rows.join(' \\\\ ')} \\end{gathered}`;
}

/**
 * SIMULATED ILLUSTRATION, not the model: seeded unit-variance Gaussian queries and keys for each d_h.
 * Returns per d_h: Var(q·k) over 16 keys, with and without 1/√d_h.
 */
function simulate(engine) {
  if (simCache) return simCache;
  const g = engine.gaussian(engine.mulberry32(SIM.seed));
  simCache = SIM.dims.map((d) => {
    const f = 1 / Math.sqrt(d);
    const q = new Float64Array(d), K = new Float64Array(SIM.keys * d);
    let vRaw = 0, vScaled = 0, n = 0;
    for (let tr = 0; tr < SIM.trials; tr++) {
      for (let c = 0; c < d; c++) q[c] = g();
      for (let i = 0; i < K.length; i++) K[i] = g();
      for (let j = 0; j < SIM.keys; j++) {
        let a = 0;
        for (let c = 0; c < d; c++) a += q[c] * K[j * d + c];
        vRaw += a * a;
        vScaled += (a * f) * (a * f);
        n++;
      }
    }
    return { dh: d, varRaw: vRaw / n, varScaled: vScaled / n };
  });
  return simCache;
}

/** A small line chart in SVG for the inset. x is log2(d_h) from 2 (d_h = 4) to 8 (d_h = 256), named under the ticks. */
function miniPlot(label, series, { yDomain, yTicks, yFmt }) {
  const W = 312, H = 76, m = { l: 28, r: 6, t: 4, b: 28 };
  const X = (x) => m.l + ((x - 2) / 6) * (W - m.l - m.r);
  const Y = (y) => H - m.b - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * (H - m.t - m.b);
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': label });
  for (const v of yTicks) {
    svg.append(s('line', { class: 'scr-grid', x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    svg.append(s('text', { class: 'scr-tick', x: m.l - 5, y: Y(v) + 4, 'text-anchor': 'end' }, yFmt(v)));
  }
  svg.append(s('line', { class: 'scr-axis', x1: m.l, x2: W - m.r, y1: Y(yDomain[0]), y2: Y(yDomain[0]) }));
  for (const [x, t, anchor] of [[2, '4', 'start'], [4, '16', 'middle'], [6, '64', 'middle'], [8, '256', 'end']]) {
    svg.append(s('text', { class: 'scr-tick', x: X(x), y: H - m.b + 14, 'text-anchor': anchor }, t));
  }
  svg.append(s('text', { class: 'scr-xt', x: (m.l + W - m.r) / 2, y: H - 1, 'text-anchor': 'middle' }, 'd', s('tspan', { 'baseline-shift': 'sub', 'font-size': '10px' }, 'h'), ', log2 scale'));
  for (const sr of series) {
    const d = sr.pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    svg.append(s('path', { d, fill: 'none', stroke: sr.color, 'stroke-width': sr.dashed ? 1.6 : 2.2, 'stroke-dasharray': sr.dashed ? '4 3' : null, 'stroke-linejoin': 'round' }));
    for (const p of sr.pts) svg.append(s('circle', { cx: X(p[0]).toFixed(1), cy: Y(p[1]).toFixed(1), r: 2, fill: sr.color }));
  }
  return svg;
}

/** Counts a number from 0 up to `to` over `ms` (ease-out). Returns a function that stops the count. */
function countUp(el, to, ms) {
  if (reducedMotion()) { el.textContent = fmt(to, 2); return () => {}; }
  const t0 = performance.now();
  let raf = 0;
  const tick = (now) => {
    const p = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(to * e, 2);
    if (p < 1) raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const { l, h: hd } = ctx.params;
    const word = (i) => (ctx.label(i) === '␣' ? 'space' : ctx.label(i));   // the space token reads as a word in captions
    let step = 0, D = null, timers = [], disposers = [], dyn = [], pmaxG = 1;
    let keyVecs = [], keyToks = [], bars = [], barVals = [], ghosts = [], pbars = [];
    let qVec = null, prodEl = null, sumEl = null, sumVal = null, matWrap = null, matEl = null, matHm = null;
    let curFull = null, insetBuilt = false, stopCount = null;   // stopCount cancels the step-1 count

    // ----------------------------------------------------- shared parts and the side column
    const header = sceneHeader(st, { chips: [[`Layer ${l}`, 'resid'], [`Head ${hd}`, 'attn']], title: 'Scores QKᵀ' });
    const picker = tokenPicker(st, ctx, { label: 'query', kind: 'attn' });
    const eqLabel = h('div', { class: 'k-label', style: { textAlign: 'left' } });
    const eqMain = h('div', { class: 'scr-eq-main' });
    const eqLive = h('div', { class: 'scr-eq-live' });
    const eqCheck = h('div', { class: 'scr-eq-check' });
    const eqNote = h('div', { class: 'scr-eq-note' });
    st.add(h('div', { class: 'scr-eq' }, eqLabel, eqMain, eqLive, eqCheck, eqNote), { x: SIDE_X, y: EQ_Y });
    const insetEl = h('div', { class: 'scr-inset', style: { opacity: '0', pointerEvents: 'none' } });
    st.add(insetEl, { x: SIDE_X, y: MAT_Y });            // step 3: the inset takes the matrix's place (one context view)
    const cap = sceneCaption(st, { y: ZONES.caption.y, w: ZONES.caption.w });

    const place = (node, o) => { st.add(node, o); dyn.push(node); return node; };
    const at = (ms, fn) => { timers.push(setTimeout(fn, reducedMotion() ? 0 : ms)); };
    const clearTimers = () => { timers.forEach(clearTimeout); timers = []; stopCount?.(); stopCount = null; };
    function teardown() {
      clearTimers();
      disposers.forEach((f) => f());
      disposers = [];
      dyn.forEach((n) => n.remove());
      dyn = [];
      keyVecs = []; keyToks = []; bars = []; barVals = []; ghosts = []; pbars = [];
      qVec = null; prodEl = null; sumEl = null; sumVal = null; matWrap = null; matEl = null; matHm = null;
    }

    /** Everything the scene needs from the trace, for the current query. Null until a trace exists. */
    function data() {
      const cur = ctx.state, cfg = cur.config, tr = cur.trace;
      const rope = cfg.pos_encoding === 'rope';
      const Qt = tr[`L${l}.attn.${rope ? 'q_rot' : 'q_heads'}`];
      const Kt = tr[`L${l}.attn.${rope ? 'k_rot' : 'k_heads'}`];
      const Sc = tr[`L${l}.attn.scores`];
      if (!Qt || !Kt || !Sc) return null;
      const Sa = tr[`L${l}.attn.scaled`] || null;                  // absent when attn_scale is off
      const Mk = tr.mask || null;                                  // [T, T], 1 = may attend
      const T = Sc.shape[1], dh = Qt.shape[2], H = cfg.n_head, Hkv = cfg.n_kv_head;
      const kvh = Math.floor(hd / (H / Hkv));                      // the KV head this query head reads
      const t = Math.min(cur.sel.token, T - 1);
      const q = Array.from(Qt.data.subarray((hd * T + t) * dh, (hd * T + t + 1) * dh));
      const K = Array.from({ length: T }, (_, j) => Array.from(Kt.data.subarray((kvh * T + j) * dh, (kvh * T + j + 1) * dh)));
      const row = Array.from(Sc.data.subarray((hd * T + t) * T, (hd * T + t + 1) * T));
      const full = Sc.data.subarray(hd * T * T, (hd + 1) * T * T);
      const rowS = Sa ? Array.from(Sa.data.subarray((hd * T + t) * T, (hd * T + t + 1) * T)) : null;
      const vis = Array.from({ length: T }, (_, j) => (Mk ? Mk.data[t * T + j] === 1 : true));
      const maskFull = Mk ? Mk.data : null;
      let jstar = -1;
      for (let j = 0; j < T; j++) if (vis[j] && (jstar < 0 || row[j] > row[jstar])) jstar = j;
      if (jstar < 0) jstar = 0;
      const prods = q.map((v, c) => v * K[jstar][c]);
      let dotLocal = 0;
      for (const p of prods) dotLocal += p;
      const loc = K.map((kj) => { let a = 0; for (let c = 0; c < dh; c++) a += q[c] * kj[c]; return a; });
      let rangeS = 0, rowMax = 0, rangeQ = 0, rangeK = 0;
      for (const v of full) if (Number.isFinite(v)) rangeS = Math.max(rangeS, Math.abs(v));
      for (const v of row) rowMax = Math.max(rowMax, Math.abs(v));
      // this head's own Q and K over the columns on screen and all tokens: the scale head.js uses for them
      const shownCols = shownIdx(dh).filter((c) => c >= 0);
      for (let r = 0; r < T; r++) for (const c of shownCols) {
        rangeQ = Math.max(rangeQ, Math.abs(Qt.data[(hd * T + r) * dh + c]));
        rangeK = Math.max(rangeK, Math.abs(Kt.data[(kvh * T + r) * dh + c]));
      }
      return {
        T, t, dh, hd, H, Hkv, kvh, rope, alibi: cfg.pos_encoding === 'alibi', Sa: !!Sa,
        q, K, row, full, rowS, vis, maskFull, jstar, prods, dotLocal, loc,
        rangeS: rangeS || 1, rowMax: rowMax || 1, rangeQ: rangeQ || 1, rangeK: rangeK || 1,
      };
    }

    /**
     * Display only: cells the causal mask hides are passed as −∞, so they draw hatched like every
     * masked value in the app. The values, and the trace they come from, are unchanged.
     */
    function hatched(vals) {
      if (!D.maskFull) return vals;
      return Array.from(vals, (v, k) => (D.maskFull[k] === 1 ? v : -Infinity));
    }

    /** Rebuild the DOM for the current query and trace. Animations are driven by paint(). */
    function build() {
      teardown();
      D = data();
      if (!D) return;
      const { T, t, dh } = D;
      const P = Math.min(PITCH_MAX, (KX1 - KX0) / T);
      const cellH = Math.max(6, Math.min(14, Math.floor(228 / Math.min(dh, 17)) - 2));
      const cw = Math.max(3, Math.min(cellH, Math.floor(P - 8)));
      const cx = (j) => KX0 + (j + 0.5) * P;
      const rot = D.rope ? '^{\\text{rot}}' : '';
      const sKey = `L${l}.attn.scores`;
      curFull = D.full;
      header.setSub(`Query <b>“${word(t)}”</b> at position ${t} is compared with every key it may see. A score is one dot product, q·k.`);

      // query and keys as columns of d_h cells: row c of every column is the same dimension.
      // Every column hangs from one line just above its token, so each stack stays with its label at any d_h.
      qVec = vec(D.q, { orientation: 'v', cell: cellH, gap: 2, kind: 'q', range: D.rangeQ, head: 12, tail: 4,
        tex: `q_t${rot}`, trace: `L${l}.attn.${D.rope ? 'q_rot' : 'q_heads'}`, cellOf: (i) => `${t},${i},${D.hd}` });
      qVec.classList.add('scr-q');
      qVec.style.setProperty('--cw', `${cellH}px`);
      const q0 = qVec;
      disposers.push(() => q0.destroy());
      place(qVec, { x: QX, y: VEC_BOT, anchor: 'bc' });
      place(h('div', { class: ['scr-tok', T > 32 && 'xs'] }, h('b', null, ctx.label(t)), h('small', null, `t = ${t}`)), { x: QX, y: TOK_Y, anchor: 'tc' });

      const tokLab = Array.from({ length: T }, (_, j) => ctx.label(j));
      // label size of .scr-tok (and .xs above 32 tokens); a turned label stays above the tallest bar of the band
      const key = keyLabelMode(tokLab, P, T > 32 ? 12 : 15, BAND_Y + ZERO - HALF - 4 - TOK_Y);
      for (let j = 0; j < T; j++) {
        const kv = vec(D.K[j], { orientation: 'v', cell: cellH, gap: 2, kind: 'k', range: D.rangeK, head: 12, tail: 4,
          trace: `L${l}.attn.${D.rope ? 'k_rot' : 'k_heads'}`, cellOf: (i) => `${j},${i},${D.kvh}` });
        kv.classList.add('scr-col');
        kv.style.setProperty('--cw', `${cw}px`);
        disposers.push(() => kv.destroy());
        place(kv, { x: cx(j), y: VEC_BOT, anchor: 'bc' });
        keyVecs.push(kv);
        const cut = key.mode === 'tilt' && key.widths[j] > key.max ? key.max : null;   // a turned label cut to the room above the band
        const written = key.mode === 'elide' && (j - t) % key.step !== 0 ? '' : tokLab[j];   // a narrow row writes every step-th label
        const tk = h('div', { class: ['scr-tok', T > 32 && 'xs', key.mode === 'tilt' && 'scr-tilt'], title: `${j}: ${tokLab[j]}`,
          style: key.mode === 'elide' ? { width: `${P}px`, height: '1em' } : null },   // the slot stays under the pointer for the title
          h('b', { style: cut ? { maxWidth: `${cut}px` } : null }, written), T <= 26 ? h('small', null, String(j)) : null);
        place(tk, { x: cx(j), y: TOK_Y, anchor: 'tc' });
        keyToks.push(tk);
      }
      void st.root.offsetHeight;                         // lay out, so the key stack has its height
      const keyTop = VEC_BOT - keyVecs[0].offsetHeight;
      place(h('div', { class: 'k-label scr-keylab', style: { color: 'var(--c-k)', textAlign: 'left' }, html: texHTML(`k_j${rot}`) }), { x: KX0, y: keyTop - 8, anchor: 'bl' });

      // score band: one bar per key, aligned under its column
      const band = place(h('div', { class: 'scr-band', style: { width: `${KX1 - KX0}px`, height: `${BAND_H}px` } },
        h('div', { class: 'scr-zero', style: { top: `${ZERO}px` } })), { x: KX0, y: BAND_Y });
      place(h('div', { class: 'k-label', style: { color: 'var(--c-attn)', textAlign: 'left' }, html: texHTML('S_{t,j}') }), { x: LEFT, y: BAND_Y + ZERO, anchor: 'lc' });
      const bw = Math.max(6, Math.min(26, P * 0.6));
      for (let j = 0; j < T; j++) {
        // step 3: dashed outline at the height before the factor, so the shrink is visible
        const gh = h('div', { class: 'scr-ghost', style: { left: `${cx(j) - KX0 - bw / 2}px`, width: `${bw}px`, top: `${ZERO}px`, height: '0px', opacity: '0' } });
        band.append(gh);
        ghosts.push(gh);
        const lab = h('div', { class: 'scr-val' });
        // each bar names the tensor it shows; paint() switches it between the raw scores and the scaled ones
        const b = h('div', { class: 'scr-bar', dataset: { cell: `${t},${j},${D.hd}`, trace: sKey },
          style: { left: `${cx(j) - KX0 - bw / 2}px`, width: `${bw}px`, top: `${ZERO}px`, height: '0px', opacity: '0' } }, lab);
        band.append(b);
        bars.push(b);
        barVals.push(lab);
      }

      // products of the strongest key, drawn as a small bar strip under its column (step 1)
      const shown = shownIdx(dh);
      const slot = 9, stripW = shown.length * slot - 3;
      pmaxG = Math.max(1e-9, ...shown.filter((i) => i >= 0).map((i) => Math.abs(D.prods[i])));
      const relx = Math.max(0, Math.min(cx(D.jstar) - KX0 - stripW / 2, KX1 - KX0 - stripW));
      prodEl = h('div', { class: 'scr-prods', style: { left: `${relx}px`, width: `${stripW}px`, height: `${BAND_H}px`, opacity: '0' } });
      shown.forEach((i, n) => {
        if (i < 0) { prodEl.append(h('div', { class: 'scr-ell', style: { left: `${n * slot}px`, top: `${ZERO - 9}px` } }, '⋯')); return; }
        const b = h('div', { class: 'scr-pbar', title: `q_${i} · k_${i} = ${fmt(D.prods[i], 3)}`,
          style: { left: `${n * slot}px`, top: `${ZERO}px`, height: '0px', background: diverging(D.prods[i] / pmaxG) } });
        prodEl.append(b);
        pbars.push({ i, v: D.prods[i], el: b });
      });
      sumVal = h('b', null, fmt(0, 2));
      sumEl = h('div', { class: 'scr-sum', style: { left: `${relx + stripW / 2}px`, top: `${BAND_H + 6}px`, opacity: '0' } }, h('span', null, 'S = '), sumVal);
      band.append(prodEl, sumEl);

      // the full S matrix of this head, row t highlighted (side column, from step 2).
      // Cells shrink as T grows (240 units in all, never below 4). Above 50 tokens a canvas heatmap draws it instead.
      const cellPx = Math.max(4, Math.min(7, Math.floor(240 / T)));
      const mLab = `S^{(${l},${D.hd})}`;
      matWrap = place(h('div', { class: 'scr-mat', style: { opacity: '0', pointerEvents: 'none' } }), { x: SIDE_X, y: MAT_Y });
      if (T <= 50) {
        matEl = mat(hatched(D.full), T, T, { cell: cellPx, gap: T > 24 ? 0 : 1, mode: 'diverging', range: D.rangeS, kind: 'attn', trace: sKey,
          tex: mLab, cellOf: (r, c) => `${r},${c},${D.hd}` });
        const m0 = matEl;
        disposers.push(() => m0.destroy());
        m0.highlightRow(t);
        m0.addEventListener('click', (e) => { const c = e.target.closest?.('[data-cell]'); if (c) ctx.store.select({ token: Number(c.dataset.cell.split(',')[0]) }); });
        matWrap.append(m0);
      } else {
        matWrap.append(h('div', { class: 'k-label', style: { color: 'var(--c-attn)' }, html: texHTML(mLab) }));
        matHm = heatmap({ data: hatched(D.full), rows: T, cols: T, mode: 'diverging', range: D.rangeS, kind: 'attn', cell: cellPx, showValues: false,
          highlight: { rows: [t] }, traceKey: sKey, ariaLabel: 'attention scores S', onClick: (r) => ctx.store.select({ token: r }),
          tooltip: (r, c) => {
            const k = r * T + c, v = curFull ? curFull[k] : NaN;
            return `[${r}, ${c}] = <b>${fmt(v, 4)}</b>${D.maskFull && D.maskFull[k] !== 1 ? ' · hidden by the mask' : ''}`;
          } });
        const hm0 = matHm;
        disposers.push(() => hm0.destroy());
        matWrap.append(hm0.el);
      }
    }

    /** Show step i. Each step sets the whole picture, so jumping between steps is safe. */
    function paint(i) {
      clearTimers();
      step = i;
      if (!D) return;
      const T = D.T, J = D.jstar, dh = D.dh, sqrtD = Math.sqrt(dh);
      const useScaled = i >= 3 && D.Sa;
      const rot = D.rope ? '^{\\text{rot}}' : '';
      const sKey = `L${l}.attn.scores`, saKey = `L${l}.attn.scaled`;
      const shownKey = useScaled ? saKey : sKey;           // the tensor the bars and the matrix show now

      // keys: step 1 focuses the strongest key; from step 2 the keys the mask hides are faded
      keyVecs.forEach((kv, j) => {
        const focus = i === 1 && j === J, dim = i === 1 && j !== J, fut = i >= 2 && !D.vis[j];
        kv.classList.toggle('focus', focus);
        kv.classList.toggle('dim', dim);
        kv.classList.toggle('future', fut);
        kv.highlight([]);
        keyToks[j].classList.toggle('dim', dim);
        keyToks[j].classList.toggle('future', fut);
      });
      qVec.highlight([]);

      // step 1: products of the strongest key appear one by one, then the sum counts up to the score
      prodEl.style.opacity = i === 1 ? '1' : '0';
      sumEl.style.opacity = i === 1 ? '1' : '0';
      pbars.forEach((p) => { p.el.style.top = `${ZERO}px`; p.el.style.height = '0px'; });
      eqCheck.style.opacity = i === 1 ? '0' : '1';
      if (i === 1) {
        sumVal.textContent = fmt(0, 2);
        pbars.forEach((p, n) => at(n * MOTION.stagger, () => {
          const hpx = Math.max(1.5, Math.abs(p.v) / pmaxG * HALF);
          p.el.style.top = `${p.v >= 0 ? ZERO - hpx : ZERO}px`;
          p.el.style.height = `${hpx}px`;
          qVec.highlight([p.i]);
          keyVecs[J].highlight([p.i]);
        }));
        const tEnd = pbars.length * MOTION.stagger;
        at(tEnd, () => {
          qVec.highlight([]);
          keyVecs[J].highlight([]);
          stopCount = countUp(sumVal, D.dotLocal, 700);
        });
        at(tEnd + 700, () => { eqCheck.style.opacity = '1'; });
      }

      // steps 2 and 3: one bar per key on the same scale; step 3 shrinks them by 1/√d_h
      const vals = useScaled ? D.rowS : D.row;
      bars.forEach((b, j) => {
        const v = vals[j], lab = barVals[j], masked = !D.vis[j];
        b.dataset.trace = shownKey;                         // hover and right-click name the tensor on screen
        b.style.pointerEvents = i < 2 ? 'none' : '';
        b.classList.toggle('masked', masked);   // a key the mask hides is hatched grey, as in S
        if (i < 2) { b.style.opacity = '0'; b.style.top = `${ZERO}px`; b.style.height = '0px'; lab.textContent = ''; return; }
        const hpx = v === 0 ? 0 : Math.max(1.5, Math.abs(v) / D.rowMax * HALF);
        b.style.opacity = '1';
        b.style.top = `${v >= 0 ? ZERO - hpx : ZERO}px`;
        b.style.height = `${hpx}px`;
        b.style.background = masked ? '' : diverging(v / D.rangeS);
        b.title = `${fmt(v, 4)}${masked ? ' · hidden by the mask' : ''}`;
        lab.textContent = !masked && (T <= 26 || j === J) ? fmt(v, 2) : '';
        lab.style.top = v >= 0 ? '-17px' : `${hpx + 4}px`;
      });

      // step 3: dashed outlines keep the scores before the factor, the solid bars show them after it
      ghosts.forEach((gh, j) => {
        const show = i === 3 && D.Sa, v = D.row[j];
        const hpx = show && v !== 0 ? Math.max(1.5, Math.abs(v) / D.rowMax * HALF) : 0;
        gh.style.opacity = show ? '1' : '0';
        gh.style.top = `${v >= 0 ? ZERO - hpx : ZERO}px`;
        gh.style.height = `${hpx}px`;
      });

      // the full matrix of raw scores is the context of step 2 (all scores); at step 3 the inset takes its place,
      // so the side column keeps one small context view
      if (matWrap) {
        const matOn = i === 2;
        matWrap.style.opacity = matOn ? '1' : '0';
        matWrap.style.pointerEvents = matOn ? '' : 'none';
      }
      insetEl.style.opacity = i === 3 ? '1' : '0';
      insetEl.style.pointerEvents = i === 3 ? '' : 'none';
      if (i === 3 && !insetBuilt) { buildInset(simulate(ctx.engine)); insetBuilt = true; }

      // equation, check and caption for this step (numbers come from the trace)
      const Sj = D.row[J];
      const tm = (c) => `(${num(D.q[c], 2)})(${num(D.K[J][c], 2)})`;
      const qL = word(D.t), kL = word(J);
      const hidden = D.vis.filter((v) => !v).length;
      const shownCount = shownIdx(dh).filter((x) => x >= 0).length;
      let check = null, note = '';
      eqLabel.textContent = i === 3 && !D.Sa ? 'Scale (off in this config)' : STEPS[i];
      if (i === 0) {
        eqMain.innerHTML = texHTML(`q_t${rot},\\; k_j${rot} \\in \\mathbb{R}^{d_h}`, { display: true });
        eqLive.innerHTML = texHTML(`d_h = ${dh},\\qquad j = 0,\\dots,${T - 1}`, { display: true });
        const parts = [];
        if (D.rope) parts.push('Vectors after RoPE: q_rot and k_rot.');
        if (D.Hkv < D.H) parts.push(`Head ${D.hd} reads KV head ${D.kvh}; ${D.H / D.Hkv} query heads share it.`);
        note = parts.join(' ');
        cap.set(`Query <b>“${qL}”</b> and every key are vectors of length <b>d<sub>h</sub> = ${dh}</b>. A score compares the query with one key.`);
      } else if (i === 1) {
        // \textstyle keeps the sum's limits beside it, so the main equation stays one line tall
        eqMain.innerHTML = texHTML('S_{tj} = q_t \\cdot k_j = \\textstyle\\sum_{c=1}^{d_h} q_{tc}\\,k_{jc}', { display: true });
        eqLive.innerHTML = texHTML(dotLines(dh, tm, num(Sj, 2)), { display: true });
        check = checkBadge(D.dotLocal, Sj);
        note = dh > 17 ? `Showing ${shownCount} of ${dh} products; all ${dh} are summed.` : '';
        cap.set(`The strongest visible key, <b>“${kL}”</b> at position ${J}, is compared element by element. The ${dh} products are summed into <b>S = ${fmt(Sj, 2)}</b>.`);
      } else if (i === 2) {
        const mx = Math.max(...D.row), mn = Math.min(...D.row);
        eqMain.innerHTML = texHTML('S = \\Q\\K^{\\top} \\in \\mathbb{R}^{T\\times T}', { display: true });
        eqLive.innerHTML = texHTML(`S_{${D.t},\\cdot} \\in [${num(mn, 2)},\\ ${num(mx, 2)}]`, { display: true });
        const matches = D.loc.filter((v, j) => Math.abs(v - D.row[j]) <= 1e-6 * Math.max(1, Math.abs(D.row[j]))).length;
        check = checkBadge(matches, T, { label: `all ${T} scores match the trace` });
        note = hidden ? `${hidden} of ${T} keys are hidden by the mask (hatched).` : 'Every key is visible to this query.';
        cap.set(hidden
          ? `Every key now has a score, so row <b>${D.t}</b> of S is complete. Hatched scores are computed first, then set to −∞ before the softmax, so they get weight 0.`
          : `Every key now has a score, so row <b>${D.t}</b> of S is complete. Under this mask every key is visible to the query.`);
      } else if (D.Sa) {
        const scaledHere = D.loc[J] / sqrtD;
        eqMain.innerHTML = texHTML('\\tilde S = \\Q\\K^{\\top} \\big/ \\sqrt{d_h}', { display: true });
        eqLive.innerHTML = texHTML(`\\begin{aligned} \\tilde S_{${D.t},${J}} &= ${num(D.loc[J], 2)} \\big/ \\sqrt{${dh}} \\\\ &= ${num(scaledHere, 2)} \\end{aligned}`, { display: true });
        check = checkBadge(scaledHere, D.rowS[J], { label: 'equals the traced scaled score' });
        note = D.alibi ? 'Dashed: before the factor. ALiBi’s bias comes after.' : 'Dashed outlines: the scores before the factor.';
        cap.set(`Dividing by <b>√d<sub>h</sub> = ${sqrtD.toFixed(2)}</b> shrinks every score by the same factor. The inset is a simulated illustration: without it, Var(q·k) grows with d<sub>h</sub>, and large dot products make the softmax peaked.`);
      } else {
        // attn_scale is off in this config: the model does not divide, so neither does the equation
        eqMain.innerHTML = texHTML('S = \\Q\\K^{\\top}', { display: true });
        eqLive.innerHTML = texHTML(`S_{${D.t},${J}} = ${num(D.loc[J], 2)}`, { display: true });
        check = checkBadge(D.loc[J], Sj, { label: 'equals the traced score' });
        note = 'attn_scale is off in this config, so no 1/√d_h factor is applied.';
        cap.set('Scaling is off in this config, so the bars keep their size. The inset is a simulated illustration of what the factor does.');
      }
      eqCheck.replaceChildren(...(check ? [check] : []));
      eqNote.textContent = note;
      eqNote.style.display = note ? '' : 'none';           // an empty note takes no room in the column
    }

    /** The inset: one chart from the simulation, Var(q·k) against d_h with and without 1/√d_h, and its legend. */
    function buildInset(rows) {
      const x = (r) => Math.log2(r.dh);
      const vPlot = miniPlot('variance of q·k against d_h, simulated illustration', [
        { color: 'var(--fg-2)', dashed: true, pts: rows.map((r) => [x(r), Math.log2(r.varRaw)]) },
        { color: 'var(--c-attn)', pts: rows.map((r) => [x(r), Math.log2(r.varScaled)]) },
      ], { yDomain: [-0.4, 8.4], yTicks: [0, 4, 8], yFmt: (v) => String(2 ** v) });
      insetEl.replaceChildren(
        h('div', { class: 'scr-plots' },
          h('div', null, h('div', { class: 'scr-plot-title', html: `${texHTML('\\mathrm{Var}(q\\cdot k)')}, simulated illustration` }), vPlot)),
        h('div', { class: 'scr-legend' },
          h('span', null, h('i', { style: { borderTopColor: 'var(--c-attn)' } }), 'with 1/√d_h'),
          h('span', null, h('i', { style: { borderTopColor: 'var(--fg-2)', borderTopStyle: 'dashed' } }), 'without')),
        h('div', { class: 'scr-sim' }, `${SIM.keys} keys, unit-variance Gaussian q and k`));
    }

    function rebuild() { build(); paint(step); }
    function setStep(i) { paint(i); }
    ctx.setSteps(STEPS, setStep);
    build();
    paint(0);
    appear([header.el, picker.el, eqLabel.parentElement], { stagger: MOTION.stagger });
    if (keyVecs.length) appear(keyVecs, { stagger: 0 });

    // auto-advance the first time the scene is shown, so it explains itself
    let auto = setTimeout(function next() {
      if (step < STEPS.length - 1) { ctx.setStep(step + 1); auto = setTimeout(next, MOTION.step); }
    }, 900);
    el.addEventListener('pointerdown', () => clearTimeout(auto), { once: true });
    el.addEventListener('atlas:interrupt', () => clearTimeout(auto), { once: true });

    const offs = [ctx.on('trace', rebuild), ctx.on('select', rebuild)];
    return {
      destroy() {
        clearTimeout(auto);
        clearTimers();
        offs.forEach((f) => f());
        teardown();
        st.destroy();
      },
    };
  },
};

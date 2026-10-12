// Animated matrix product on a canvas:  C = A · B   or   C = A · Bᵀ.
//
//   const mm = matmulAnim({
//     A: { data, rows, cols, name: 'X', kind: 'norm', shape: 'T×d', rowLabels },
//     B: { data, rows, cols, name: 'W_Q', kind: 'q', shape: 'd_h×d', bias, biasName: 'b_Q', rowLabels, colLabels },
//     transposeB: true,                                    // B is stored [n × k] (nn.Linear weight, keys), drawn as Bᵀ
//     C: { name: 'Q', kind: 'q', shape: 'T×d_h', data },   // data optional: reference values for maxError()
//     title, maxWidth: 720, maxCell: 56, maxHeight: null,
//     showValues: 'auto' | true | false, digits: 2, eqDigits, speed: 1,
//     hint: true,                              // false hides the idle 'hover a cell' text in the corner
//   });
//   parent.append(mm.el);
//
//   mm.focus(i, j, { animate: true })       mm.play()        mm.pause()       mm.reset()     mm.showAll()
//   mm.update({ A: { data }, C: { data } })  mm.setSpeed(2)   mm.cellRect(i, j)
//   mm.values()   mm.maxError()   mm.speed   mm.playing   mm.destroy()
//
// Layout (textbook arrangement): A at the left, B (drawn as Bᵀ, k × n) above C, C at the bottom
// right. Row i of A lines up with row i of C; column j of B lines up with column j of C. The empty
// top-left corner holds the product strip: one bar per k, aligned with column k of A, and the
// running sum. Shapes: A.shape and C.shape are the arrays as given; B.shape is the array as given,
// so with transposeB the label shows its transpose (W_Qᵀ  d×d_h).
//
// Rendering: A, Bᵀ and C are cached on offscreen canvases at device resolution. Each frame blits
// the caches and draws only the overlays, and a tween repaints only the matrices that move.
// Colours come from the diverging token ramp through a lookup table rebuilt on theme change.
// One requestAnimationFrame loop runs only while something moves. Reduced motion jumps to the
// end state of every change.

import { h, fmt, clamp } from './dom.js';
import { diverging, kindColor, cssVar, onThemeChange, parseColor } from './colors.js';
import { ease, reducedMotion } from './anim.js';
import { showTip, hideTip } from './tooltip.js';

const LUT_N = 128;          // colour steps across [−1, 1]
const STEP_MIN = 50;        // ms per k step at speed 1 (large inner dimension: readable bars, longer focus)
const STEP_MAX = 170;       // ms per k step (small inner dimension)
const TAIL_MS = 520;        // fly the running sum into C, reveal the cell, pop it
const FILL_MS = 480;        // showAll() fade
const TWEEN_MS = 350;       // update() colour tween
const DWELL_MS = 90;        // hover dwell before a focus animation starts
const CELL_PLAY_MS = 4200;  // whole cell-by-cell play at speed 1, shared by all cells
const ROW_PLAY_MS = [3000, 6000];  // row-by-row play, total at speed 1
const CELL_MODE_MAX = 36;   // play cell by cell up to this many cells, else row by row
const BAND_MAX = 180;       // tallest product strip (px)
const RATE_MAX = 24;        // play may compress a focus this much to keep the whole play short

const stepFor = (K) => clamp(1000 / K, STEP_MIN, STEP_MAX);
const focusDur = (K) => K * stepFor(K) + TAIL_MS;
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const easeOutC = (t) => 1 - Math.pow(1 - clamp01(t), 3);
const easeIO = (t) => ease.inOut(clamp01(t));
const lerp = (a, b, t) => a + (b - a) * t;
const clampSpeed = (v) => { const x = Number(v); return clamp(Number.isFinite(x) ? x : 1, 0.1, 8); };

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// Fixed-point text for values: never exponent form for small values (fmt() switched to it at digits 0).
// Zero never carries a minus sign (−0.00 reads 0.00).
function numText(v, d) {
  if (!Number.isFinite(v) || Math.abs(v) >= 1e5) return fmt(v, d);
  const t = v.toFixed(Math.min(10, Math.max(0, d | 0))).replace('-', '−');
  return /^−0(\.0*)?$/.test(t) ? t.slice(1) : t;
}

function maxAbs(arr) {
  let m = 0;
  for (let i = 0; i < arr.length; i++) {
    const a = Math.abs(arr[i]);
    if (a > m && Number.isFinite(a)) m = a;
  }
  return m || 1;
}

// Index into the colour lookup table for value v on the scale [−range, range].
function lutIndex(v, range) {
  let t = v / range;
  if (!(t >= -1)) t = -1;          // also catches NaN
  else if (t > 1) t = 1;
  return Math.round(((t + 1) * LUT_N) / 2);
}

function sameArr(a, b) {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let x = 0; x < a.length; x++) {
    const p = a[x], q = b[x];
    if (p !== q && !(Number.isNaN(p) && Number.isNaN(q))) return false;
  }
  return true;
}

const copyOf = (p) => ({ a: Float64Array.from(p.a), bd: Float64Array.from(p.bd), c: Float64Array.from(p.c) });

function roundRect(ctx, x, y, w, hh, r) {
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') ctx.roundRect(x, y, w, hh, r);
  else ctx.rect(x, y, w, hh);
}

// ----------------------------------------------------------------- theme

let tokCache = null;
const instances = new Set();
let measureCtx = null;

onThemeChange(() => {
  tokCache = null;
  for (const inst of instances) inst.onTheme();
});

function measurer() {
  if (!measureCtx) measureCtx = document.createElement('canvas').getContext('2d');
  return measureCtx;
}

function tokens() {
  if (tokCache) return tokCache;
  const bg = parseColor(cssVar('--bg')), fg = parseColor(cssVar('--fg'));
  const lum = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
  // Ink for values drawn inside cells: whichever of --fg / --bg is darker is the dark ink.
  const fgDarker = lum(fg) <= lum(bg);
  const inkDark = fgDarker ? cssVar('--fg') : cssVar('--bg');
  const inkLight = fgDarker ? cssVar('--bg') : cssVar('--fg');
  const lut = [], ink = [];
  for (let q = 0; q <= LUT_N; q++) {
    const col = diverging(-1 + (2 * q) / LUT_N);
    lut.push(col);
    ink.push(lum(parseColor(col)) > 120 ? inkDark : inkLight);
  }
  tokCache = {
    lut, ink,
    fg: cssVar('--fg'), fg2: cssVar('--fg-2'), fg3: cssVar('--fg-3'),
    surface: cssVar('--surface'), surface2: cssVar('--surface-2'),
    mono: cssVar('--font-mono') || 'monospace', sans: cssVar('--font-body') || 'sans-serif',
    kinds: new Map(),
  };
  return tokCache;
}

function kindOf(tk, kind) {
  if (!tk.kinds.has(kind)) tk.kinds.set(kind, kindColor(kind) || tk.fg2);
  return tk.kinds.get(kind);
}

// ------------------------------------------------------------------ data

// Normalise the options. The displayed B is always k × n. C = A·B is computed in the same order as
// the engine (bias first, then the products in k order), so the sums match the trace bit for bit.
function prepare(o) {
  const A = o.A, B = o.B;
  if (!A || !B) throw new Error('matmulAnim: opts.A and opts.B are required');
  const m = A.rows, k = A.cols, tB = !!o.transposeB;
  const n = tB ? B.rows : B.cols;
  const kB = tB ? B.cols : B.rows;
  if (!(m >= 1 && k >= 1 && n >= 1)) throw new Error('matmulAnim: every dimension must be at least 1');
  if (kB !== k) throw new Error(`matmulAnim: inner dimensions differ (A has ${k} columns, B has ${kB})`);
  if (!A.data || !B.data || A.data.length !== m * k || B.data.length !== B.rows * B.cols) {
    throw new Error('matmulAnim: A.data and B.data need rows × cols values each');
  }
  const a = Float64Array.from(A.data);
  const bd = new Float64Array(k * n);
  for (let kk = 0; kk < k; kk++) for (let j = 0; j < n; j++) bd[kk * n + j] = tB ? B.data[j * k + kk] : B.data[kk * n + j];
  const bias = B.bias ? Float64Array.from(B.bias) : null;
  if (bias && bias.length !== n) throw new Error('matmulAnim: B.bias needs one value per output column');
  const c = new Float64Array(m * n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    let s = bias ? bias[j] : 0;
    for (let kk = 0; kk < k; kk++) s += a[i * k + kk] * bd[kk * n + j];
    c[i * n + j] = s;
  }
  const ref = o.C && o.C.data ? Float64Array.from(o.C.data) : null;
  if (ref && ref.length !== m * n) throw new Error('matmulAnim: C.data needs rows × columns values');
  return { m, k, n, tB, a, bd, bias, c, ref };
}

// Display names and the live shape labels. A symbolic shape given for a transposed B is the stored
// one, so it is turned around for the label.
function names(o, P) {
  const label = (sym, numbers, turn) => {
    let s = sym ? String(sym) : '';
    if (s && turn) {
      const parts = s.split('×');
      if (parts.length === 2) s = `${parts[1]}×${parts[0]}`;
    }
    return s && s !== numbers ? `${s}  ${numbers}` : numbers;
  };
  return {
    C: o.C?.name ?? 'C',
    A: o.A.name ?? 'A',
    Bg: o.B.name ?? 'B',
    Bshow: (o.B.name ?? 'B') + (P.tB ? 'ᵀ' : ''),
    aKind: o.A.kind ?? 'norm',
    bKind: o.B.kind ?? 'k',
    cKind: o.C?.kind ?? 'attn',
    aRest: label(o.A.shape, `${P.m}×${P.k}`, false),
    bRest: label(o.B.shape, `${P.k}×${P.n}`, P.tB),
    cRest: label(o.C?.shape, `${P.m}×${P.n}`, false),
  };
}

// Focus record for cell (i, j): the k products, the running sums (starting at the bias) and timing.
function makeFocus(P, i, j, rate = 1) {
  const { k, n, a, bd, bias } = P;
  const pvals = new Float64Array(k), partial = new Float64Array(k + 1);
  let sum = bias ? bias[j] : 0;
  partial[0] = sum;
  let mx = 0;
  for (let kk = 0; kk < k; kk++) {
    const p = a[i * k + kk] * bd[kk * n + j];
    pvals[kk] = p;
    sum += p;
    partial[kk + 1] = sum;
    const ap = Math.abs(p);
    if (ap > mx) mx = ap;
  }
  const stepMs = stepFor(k);
  return { i, j, K: k, pvals, partial, rangeP: mx || 1, stepMs, t: 0, dur: k * stepMs + TAIL_MS, rate };
}

// Where a focus record is in its timeline.
function phaseOf(f) {
  const tS = f.K * f.stepMs;
  if (f.t >= f.dur) return { done: true, partial: f.partial[f.K] };
  if (f.t >= tS) {
    const t2 = f.t - tS;
    return {
      tail: true,
      fly: easeIO(t2 / (0.56 * TAIL_MS)),
      reveal: easeOutC((t2 - 0.34 * TAIL_MS) / (0.46 * TAIL_MS)),
      pop: clamp01((t2 - 0.5 * TAIL_MS) / (0.5 * TAIL_MS)),
      partial: f.partial[f.K],
    };
  }
  const s = Math.min(f.K - 1, Math.floor(f.t / f.stepMs));
  const local = (f.t - s * f.stepMs) / f.stepMs;
  const grow = clamp01((local - 0.28) / 0.52);
  return {
    step: s, local, grow,
    hl: clamp01(local / 0.28) * (1 - clamp01((local - 0.82) / 0.18)),
    partial: lerp(f.partial[s], f.partial[s + 1], easeIO(grow)),
  };
}

// Options that change the geometry or the label widths (a change here relayouts).
function layoutSig(x, P) {
  return JSON.stringify([
    P.m, P.k, P.n, !!P.tB, x.digits ?? 2, x.showValues ?? 'auto', x.maxWidth ?? 720, x.maxCell ?? 56, x.maxHeight ?? null,
    x.A.rowLabels ?? null, x.A.colLabels ?? null, x.B.rowLabels ?? null, x.B.colLabels ?? null,
  ]);
}

// Options that only change text (title, names, shape labels, the equation's digits).
function textSig(x) {
  return JSON.stringify([
    x.title ?? '', x.eqDigits ?? null,
    x.A.name ?? null, x.A.kind ?? null, x.A.shape ?? null,
    x.B.name ?? null, x.B.kind ?? null, x.B.shape ?? null, x.B.biasName ?? null,
    x.C?.name ?? null, x.C?.kind ?? null, x.C?.shape ?? null,
  ]);
}

// --------------------------------------------------------------- component

export function matmulAnim(opts) {
  if (!opts || !opts.A || !opts.B) throw new Error('matmulAnim: opts.A and opts.B are required');
  let o = { ...opts };
  let P = prepare(o);
  let N = names(o, P);
  const dig = () => o.digits ?? 2;
  const eqDigitsOf = () => o.eqDigits ?? Math.min(4, dig() + 1);
  let disp = copyOf(P);                       // displayed values (differs from P during a tween)
  const rng = { a: 1, bd: 1, c: 1 };          // colour range of each cached matrix (max |value|)
  let L = null, lastAvail = -1, lastDpr = 0;
  let rev = new Uint8Array(P.m * P.n);        // revealed C cells
  let F = null, RA = null, FA = null;         // focus, row wave, fill (display state)
  let tl = null, tw = null;                   // timeline, colour tween
  let speed = clampSpeed(opts.speed ?? 1);
  let pinned = false, hoverC = null, hoverA = null, hoverB = null;
  let dwell = 0, raf = 0, lastNow = 0, destroyed = false, eqKey = '', footKey = '';
  let ro = null, roFrame = 0, dprMQ = null;
  const offscreen = () => document.createElement('canvas');
  const cache = { a: offscreen(), bd: offscreen(), c: offscreen() };

  // ------------------------------------------------------------- DOM
  injectCss();
  const titleEl = h('div', { class: 'mm-title' });
  const scaleEl = h('span', {
    class: 'mm-scale', 'aria-hidden': 'true',
    title: 'Colour: blue negative, orange positive. Each matrix is scaled to its largest absolute value.',
  }, h('span', null, '−'), h('i'), h('span', null, '+'));
  const headEl = h('div', { class: 'mm-head' }, titleEl, scaleEl);
  const eqEl = h('div', { class: 'mm-eq', 'aria-live': 'polite' });
  const canvas = h('canvas', { tabindex: '0', role: 'application' });
  const stage = h('div', { class: 'mm-stage' }, canvas);
  const foot = h('div', { class: 'mm-foot' });
  const el = h('div', { class: 'mm' }, headEl, eqEl, stage, foot);
  const g = canvas.getContext('2d');

  const dprNow = () => window.devicePixelRatio || 1;
  // Numbers inside cells: 'auto' shows them when a cell is at least 24 px.
  const showNums = () => {
    const v = o.showValues ?? 'auto';
    if (v === true) return true;
    if (v === false) return false;
    return !!L && L.s >= 24;
  };

  // ---------------------------------------------------------- layout
  function computeLayout(avail) {
    const { m, k, n } = P, tk = tokens(), mg = measurer();
    const textW = (arr, font) => {
      if (!arr || !arr.length) return 0;
      mg.font = font;
      let w = 0;
      for (const t of arr) w = Math.max(w, mg.measureText(String(t)).width);
      return w;
    };
    const lblFont = `11px ${tk.mono}`;
    const lblA = o.A.rowLabels ? Math.ceil(textW(o.A.rowLabels, lblFont)) + 10 : 0;
    const gBL = o.B.rowLabels ? Math.ceil(textW(o.B.rowLabels, lblFont)) + 8 : 0;
    const gX = 26 + gBL;                    // gap between A and B / C
    const topH = o.B.colLabels ? 42 : 26;   // name row (+ column labels) above B
    const gY = 16;                          // gap between B and C
    const aGap = o.A.colLabels ? 18 : 10;   // strip to the top of A
    const botH = 30;                        // name row under A and C
    let s = Math.min(o.maxCell ?? 56, (avail - lblA - gX) / (k + n));
    if (o.maxHeight) s = Math.min(s, (o.maxHeight - topH - gY - botH) / (k + m));
    s = Math.max(1, s);
    const xA = Math.round(lblA);
    const xC = Math.round(xA + k * s + gX);
    const yB = topH;
    const yC = Math.round(yB + k * s + gY);
    const W = Math.ceil(xC + n * s) + 1;
    const H = Math.ceil(yC + m * s + botH);
    const bandBottom = yC - aGap;
    const bandTop = Math.max(30, bandBottom - BAND_MAX);
    const readY = bandTop > 40 ? bandTop - 12 : 13;
    const digits = dig();
    const fits = (px) => { mg.font = `${px}px ${tk.mono}`; return mg.measureText(numText(-10.5, digits)).width <= s - 4; };
    let numPx = clamp(Math.floor(s * 0.4), 6, 13);
    while (numPx > 6 && !fits(numPx)) numPx--;
    return {
      s, xA, xC, yB, yC, W, H,
      gap: s >= 5 ? 1 : 0,
      bw: clamp(s * 0.62, 2, 26),
      numPx,
      bandTop, bandBottom, readY,
    };
  }

  // Snapped cell box (x, y, w, h) of cell (r, c) of a matrix whose top-left is (x0, y0).
  function cellBox(x0, y0, r, c) {
    const { s, gap } = L;
    const xa = x0 + Math.round(c * s), xb = x0 + Math.round((c + 1) * s) - gap;
    const ya = y0 + Math.round(r * s), yb = y0 + Math.round((r + 1) * s) - gap;
    return [xa, ya, xb - xa, yb - ya];
  }

  function cellCentre(i, j) {
    const { s, xC, yC } = L;
    return [xC + (j + 0.5) * s, yC + (i + 0.5) * s];
  }

  function relayout(force = false) {
    if (destroyed) return;
    const maxW = o.maxWidth ?? 720;
    const avail = Math.max(120, Math.min(maxW, stage.clientWidth || maxW) - 2);
    const dpr = dprNow();
    if (!force && L && avail === lastAvail && dpr === lastDpr) return;
    lastAvail = avail;
    L = computeLayout(avail);
    lastDpr = dpr;
    const cw = Math.round(L.W * dpr), ch = Math.round(L.H * dpr);
    if (canvas.width !== cw) canvas.width = cw;      // resizing a canvas clears it: only when it changes
    if (canvas.height !== ch) canvas.height = ch;
    canvas.style.width = `${L.W}px`;
    canvas.style.aspectRatio = `${L.W} / ${L.H}`;
    paintAll();
    draw();
  }

  // ---------------------------------------------------------- caches

  // Paint one matrix into its cache. `mask` (C only): unrevealed cells stay blank.
  function paintMat(key, rows, cols, data, mask, withNums, range = maxAbs(data)) {
    const cv = cache[key];
    const { s, gap, numPx } = L, dpr = lastDpr, tk = tokens();
    const w = cols * s, hh = rows * s;
    const pw = Math.max(1, Math.ceil(w * dpr)), ph = Math.max(1, Math.ceil(hh * dpr));
    if (cv.width !== pw) cv.width = pw;
    if (cv.height !== ph) cv.height = ph;
    const cg = cv.getContext('2d');
    cg.setTransform(dpr, 0, 0, dpr, 0, 0);
    cg.clearRect(0, 0, w, hh);
    rng[key] = range;
    if (withNums) { cg.font = `${numPx}px ${tk.mono}`; cg.textAlign = 'center'; cg.textBaseline = 'middle'; }
    const d = dig();
    for (let r = 0; r < rows; r++) {
      const y0 = Math.round(r * s), y1 = Math.round((r + 1) * s) - gap;
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const x0 = Math.round(c * s), x1 = Math.round((c + 1) * s) - gap;
        if (mask && !mask[idx]) {
          cg.fillStyle = tk.surface2;
          cg.fillRect(x0, y0, x1 - x0, y1 - y0);
          continue;
        }
        const q = lutIndex(data[idx], range);
        cg.fillStyle = tk.lut[q];
        cg.fillRect(x0, y0, x1 - x0, y1 - y0);
        if (withNums) {
          cg.fillStyle = tk.ink[q];
          cg.fillText(numText(data[idx], d), (x0 + x1) / 2, (y0 + y1) / 2 + 0.5);
        }
      }
    }
  }

  function paintC(range) {
    if (!L) return;
    paintMat('c', P.m, P.n, disp.c, rev, showNums(), range);
  }

  function paintAll() {
    if (!L) return;
    const nums = showNums();
    paintMat('a', P.m, P.k, disp.a, null, nums);
    paintMat('bd', P.k, P.n, disp.bd, null, nums);
    paintC();
  }

  // One C cell into its cache (used when a cell is revealed, so nothing else is repainted).
  function paintCellC(i, j) {
    if (!L) return;
    const cv = cache.c, { s, gap, numPx } = L, dpr = lastDpr, tk = tokens();
    const cg = cv.getContext('2d');
    cg.setTransform(dpr, 0, 0, dpr, 0, 0);
    const x0 = Math.round(j * s), x1 = Math.round((j + 1) * s) - gap;
    const y0 = Math.round(i * s), y1 = Math.round((i + 1) * s) - gap;
    const v = disp.c[i * P.n + j];
    const q = lutIndex(v, rng.c);
    cg.fillStyle = tk.lut[q];
    cg.fillRect(x0, y0, x1 - x0, y1 - y0);
    if (showNums()) {
      cg.font = `${numPx}px ${tk.mono}`; cg.textAlign = 'center'; cg.textBaseline = 'middle';
      cg.fillStyle = tk.ink[q];
      cg.fillText(numText(v, dig()), (x0 + x1) / 2, (y0 + y1) / 2 + 0.5);
    }
  }

  function commitCell(i, j) {
    const idx = i * P.n + j;
    if (rev[idx]) return;
    rev[idx] = 1;
    paintCellC(i, j);
  }

  // ---------------------------------------------------------- drawing

  function drawName(tk, name, rest, x, y, colour) {
    g.textBaseline = 'middle';
    g.textAlign = 'left';
    g.font = `600 12px ${tk.sans}`;
    const w1 = g.measureText(name).width;
    g.font = `500 11px ${tk.mono}`;
    const w2 = g.measureText(rest).width;
    const xx = Math.max(0, Math.min(x, L.W - (w1 + 8 + w2) - 2));
    g.font = `600 12px ${tk.sans}`;
    g.fillStyle = colour;
    g.fillText(name, xx, y);
    g.font = `500 11px ${tk.mono}`;
    g.fillStyle = tk.fg3;
    g.fillText(rest, xx + w1 + 8, y);
  }

  function drawLabels(tk, row, col) {
    const { s, xA, xC, yB, yC } = L;
    const { m, k, n } = P;
    g.font = `11px ${tk.mono}`;
    if (o.A.rowLabels) {
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      const every = Math.max(1, Math.ceil(12 / s));
      o.A.rowLabels.forEach((t, i) => {
        if (i >= m || (i % every && i !== row)) return;
        g.fillStyle = i === row ? tk.fg : tk.fg3;
        g.fillText(String(t), xA - 6, yC + (i + 0.5) * s);
      });
    }
    if (o.A.colLabels) {
      g.textAlign = 'center';
      g.textBaseline = 'bottom';
      const every = Math.max(1, Math.ceil(24 / s));
      o.A.colLabels.forEach((t, kk) => {
        if (kk >= k || (kk % every && kk !== col)) return;
        g.fillStyle = tk.fg3;
        g.fillText(String(t), xA + (kk + 0.5) * s, yC - 4);
      });
    }
    if (o.B.colLabels) {
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      const every = Math.max(1, Math.ceil(24 / s));
      o.B.colLabels.forEach((t, j) => {
        if (j >= n || (j % every && j !== col)) return;
        g.fillStyle = j === col ? tk.fg : tk.fg3;
        g.fillText(String(t), xC + (j + 0.5) * s, yB - 12);
      });
    }
    if (o.B.rowLabels) {
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      const every = Math.max(1, Math.ceil(12 / s));
      o.B.rowLabels.forEach((t, kk) => {
        if (kk >= k || (kk % every && kk !== row)) return;
        g.fillStyle = tk.fg3;
        g.fillText(String(t), xC - 6, yB + (kk + 0.5) * s);
      });
    }
  }

  // The product strip: one bar per k, aligned with column k of A, plus the running sum.
  function drawBand(tk, f, ph, colour) {
    const { s, xA, bandTop, bandBottom, bw, readY } = L;
    const base = (bandTop + bandBottom) / 2;
    const half = Math.max(4, (bandBottom - bandTop) / 2 - 6);
    g.globalAlpha = 0.3;
    g.fillStyle = tk.fg3;
    g.fillRect(xA, Math.round(base) - 0.5, f.K * s, 1);
    g.globalAlpha = 1;
    if (ph.step != null) {
      g.globalAlpha = 0.09;
      g.fillStyle = colour;
      g.fillRect(xA + ph.step * s, bandTop, s, bandBottom - bandTop);
      g.globalAlpha = 1;
    }
    for (let kk = 0; kk < f.K; kk++) {
      const x = xA + (kk + 0.5) * s - bw / 2;
      let hf = 1;
      if (ph.step != null) hf = kk < ph.step ? 1 : kk === ph.step ? easeOutC(ph.grow) : 0;
      if (hf <= 0) {
        g.globalAlpha = 0.3;
        g.fillStyle = tk.fg3;
        g.fillRect(x, base - 1, bw, 2);
        g.globalAlpha = 1;
        continue;
      }
      const p = f.pvals[kk];
      const len = Math.max(1, (Math.abs(p) / f.rangeP) * half * hf);
      g.fillStyle = tk.lut[lutIndex(p, f.rangeP)];
      g.fillRect(x, p >= 0 ? base - len : base, bw, len);
    }

    // running sum, top left of the corner (it flies into C during the tail)
    g.textBaseline = 'middle';
    g.textAlign = 'left';
    const showing = !(ph.tail && ph.fly > 0.02);
    const val = ph.partial ?? f.partial[f.K];
    g.font = `500 12px ${tk.mono}`;
    g.fillStyle = tk.fg3;
    g.fillText('Σ', xA + 4, readY);
    const sigW = g.measureText('Σ  ').width;
    const txt = numText(val, eqDigitsOf());
    g.font = `600 13px ${tk.mono}`;
    const numW = g.measureText(txt).width;
    if (showing) {
      g.fillStyle = tk.fg;
      g.fillText(txt, xA + 4 + sigW, readY);
    }
    if (ph.tail && ph.fly > 0.02 && ph.fly < 0.98) {
      // the sum flies from the readout into its cell
      const fromX = xA + 4 + sigW + numW / 2, fromY = readY;
      const [cx, cy] = cellCentre(f.i, f.j);
      const x = lerp(fromX, cx, ph.fly), y = lerp(fromY, cy, ph.fly);
      const px = Math.round(lerp(13, L.numPx + 2, ph.fly));
      g.globalAlpha = 1 - clamp01((ph.fly - 0.8) / 0.2);
      g.font = `600 ${px}px ${tk.mono}`;
      g.fillStyle = tk.fg;
      g.textAlign = 'center';
      g.fillText(txt, x, y);
      g.globalAlpha = 1;
    }
    g.textAlign = 'left';
  }

  // The corner when no product is running: the zero line and a hint, or the row being filled.
  function drawCorner(tk, rowI) {
    if (o.hint === false && rowI < 0) return;   // scenes that explain the product themselves
    const { s, xA, bandTop, bandBottom, readY } = L;
    const width = P.k * s;
    const base = (bandTop + bandBottom) / 2;
    g.globalAlpha = 0.25;
    g.fillStyle = tk.fg3;
    g.fillRect(xA, Math.round(base) - 0.5, width, 1);
    g.globalAlpha = 1;
    g.textBaseline = 'middle';
    g.textAlign = 'left';
    g.font = `500 12px ${tk.mono}`;
    g.fillStyle = tk.fg3;
    g.fillText('Σ  —', xA + 4, readY);
    const cx = xA + width / 2;
    g.textAlign = 'center';
    if (rowI >= 0) {
      g.font = `600 12px ${tk.sans}`;
      g.fillStyle = tk.fg2;
      g.fillText(`row ${rowI + 1} of ${P.m}`, cx, base);
    } else if (width >= 180 && bandBottom - bandTop > 60) {
      g.font = `500 12px ${tk.sans}`;
      g.fillStyle = tk.fg3;
      g.fillText(`Hover a cell of ${N.C}.`, cx, base - 9);
      g.fillText('Its products appear here.', cx, base + 9);
    }
  }

  // Paint one C cell as an overlay (reveal, row wave, fill).
  function overlayC(tk, i, j, alpha, scale = 1) {
    const [x, y, w, hh] = cellBox(L.xC, L.yC, i, j);
    const v = disp.c[i * P.n + j];
    const q = lutIndex(v, rng.c);
    const cx = x + w / 2, cy = y + hh / 2, ww = w * scale, hgt = hh * scale;
    g.globalAlpha = alpha;
    g.fillStyle = tk.lut[q];
    g.fillRect(cx - ww / 2, cy - hgt / 2, ww, hgt);
    if (alpha > 0.6 && showNums()) {
      g.fillStyle = tk.ink[q];
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.font = `${Math.round(L.numPx * scale)}px ${tk.mono}`;
      g.fillText(numText(v, dig()), cx, cy + 0.5);
    }
    g.globalAlpha = 1;
  }

  // Small cells have no room for numbers: the focused cell gets a value badge once revealed.
  function drawBadge(tk, i, j, alpha, colour) {
    const [x, y, w, hh] = cellBox(L.xC, L.yC, i, j);
    const text = numText(disp.c[i * P.n + j], dig());
    g.font = `600 11px ${tk.mono}`;
    const bwid = Math.ceil(g.measureText(text).width) + 10, bh = 17;
    const cx = x + w / 2, cy = y + hh / 2;
    g.globalAlpha = alpha;
    g.fillStyle = tk.surface;
    roundRect(g, cx - bwid / 2, cy - bh / 2, bwid, bh, 5);
    g.fill();
    g.strokeStyle = colour;
    g.lineWidth = 1.5;
    g.stroke();
    g.fillStyle = tk.fg;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(text, cx, cy + 0.5);
    g.textAlign = 'left';
    g.globalAlpha = 1;
  }

  function outline(x, y, w, hh, colour, lw, alpha, dash) {
    g.strokeStyle = colour;
    g.lineWidth = lw;
    g.globalAlpha = alpha;
    g.setLineDash(dash || []);
    g.strokeRect(x + lw / 2, y + lw / 2, w - lw, hh - lw);
    g.setLineDash([]);
    g.globalAlpha = 1;
  }

  function draw() {
    if (destroyed || !L) return;
    const dpr = dprNow();
    if (dpr !== lastDpr) { relayout(true); return; }
    const tk = tokens();
    const { s, xA, xC, yB, yC, W, H } = L;
    const { m, k, n } = P;
    const kA = kindOf(tk, N.aKind), kB = kindOf(tk, N.bKind), kC = kindOf(tk, N.cKind);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    // 1. the three matrices, from cache
    g.drawImage(cache.a, xA, yC, k * s, m * s);
    g.drawImage(cache.bd, xC, yB, n * s, k * s);
    g.drawImage(cache.c, xC, yC, n * s, m * s);

    // 2. frames tinted by kind
    g.lineWidth = 1;
    g.globalAlpha = 0.6;
    g.strokeStyle = kA; g.strokeRect(xA - 0.5, yC - 0.5, k * s + 1, m * s + 1);
    g.strokeStyle = kB; g.strokeRect(xC - 0.5, yB - 0.5, n * s + 1, k * s + 1);
    g.strokeStyle = kC; g.strokeRect(xC - 0.5, yC - 0.5, n * s + 1, m * s + 1);
    g.globalAlpha = 1;

    const ph = F ? phaseOf(F) : null;
    const row = F ? F.i : RA ? RA.i : -1;
    const col = F ? F.j : -1;

    // 3. dim what the focus does not touch
    if (row >= 0) {
      g.globalAlpha = 0.5;
      g.fillStyle = tk.surface;
      if (row > 0) g.fillRect(xA, yC, k * s, row * s);
      if (row < m - 1) g.fillRect(xA, yC + (row + 1) * s, k * s, (m - row - 1) * s);
      if (col >= 0) {
        if (col > 0) g.fillRect(xC, yB, col * s, k * s);
        if (col < n - 1) g.fillRect(xC + (col + 1) * s, yB, (n - col - 1) * s, k * s);
      }
      g.globalAlpha = 1;
    }

    // 4. hover on A or B shows its row or column
    if (hoverA != null && hoverA !== row) {
      const [, ry, , rh] = cellBox(xA, yC, hoverA, 0);
      outline(xA, ry, k * s, rh, tk.fg3, 1, 0.7, [3, 3]);
    }
    if (hoverB != null && hoverB !== col) {
      const [cx, , cw] = cellBox(xC, yB, 0, hoverB);
      outline(cx, yB, cw, k * s, tk.fg3, 1, 0.7, [3, 3]);
    }

    // 5. crosshair: row i runs from A through C, column j from B through C
    if (row >= 0) {
      const cy = Math.round(yC + (row + 0.5) * s) + 0.5;
      g.setLineDash([3, 4]);
      g.strokeStyle = tk.fg3;
      g.globalAlpha = 0.6;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(xA, cy); g.lineTo(xC + n * s, cy); g.stroke();
      if (col >= 0) {
        const cx = Math.round(xC + (col + 0.5) * s) + 0.5;
        g.beginPath(); g.moveTo(cx, yB); g.lineTo(cx, yC + m * s); g.stroke();
      }
      g.setLineDash([]);
      g.globalAlpha = 1;

      // highlighted row of A and row of C, column of B
      const [, ry, , rh] = cellBox(xA, yC, row, 0);
      outline(xA, ry, k * s, rh, kA, 2, 0.95);
      outline(xC, ry, n * s, rh, kC, 1.5, 0.45);
    }
    if (col >= 0) {
      const [cx, , cw] = cellBox(xC, yB, 0, col);
      outline(cx, yB, cw, k * s, kB, 2, 0.95);
    }

    // 6. the product strip and running sum (or the corner hint when idle)
    if (F) drawBand(tk, F, ph, kA);
    else drawCorner(tk, RA ? RA.i : -1);

    // 7. the k step: the two source cells and the drop line to the strip
    if (F && ph.step != null && ph.hl > 0) {
      const kk = ph.step;
      const [ax, ay, aw, ah] = cellBox(xA, yC, F.i, kk);
      const [bx, by, bw2, bh2] = cellBox(xC, yB, kk, F.j);
      outline(ax, ay, aw, ah, tk.fg, 2, ph.hl);
      outline(bx, by, bw2, bh2, tk.fg, 2, ph.hl);
      const base = (L.bandTop + L.bandBottom) / 2;
      const gx = Math.round(xA + (kk + 0.5) * s) + 0.5;
      g.setLineDash([2, 3]);
      g.strokeStyle = kA;
      g.globalAlpha = ph.hl * 0.9;
      g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(gx, base); g.lineTo(gx, ay); g.stroke();
      g.setLineDash([]);
      g.globalAlpha = 1;
    }

    // 8. the focused cell of C: reveal, pop, outline and (small cells) its value
    if (F && ph.tail) {
      const sc = 1 + 0.12 * Math.sin(Math.PI * ph.pop);
      if (ph.reveal > 0) overlayC(tk, F.i, F.j, ph.reveal, sc);
      if (ph.pop > 0 && ph.pop < 1) {
        const [x, y, w, hh] = cellBox(xC, yC, F.i, F.j);
        const grow = 2 + 7 * ph.pop;
        outline(x - grow, y - grow, w + 2 * grow, hh + 2 * grow, kC, 2, (1 - ph.pop) * 0.9);
      }
    }
    if (F) {
      const [x, y, w, hh] = cellBox(xC, yC, F.i, F.j);
      outline(x, y, w, hh, kC, 2, 1);
      if (!showNums() && (ph.done || (ph.tail && ph.reveal > 0.5))) {
        drawBadge(tk, F.i, F.j, ph.done ? 1 : ph.reveal, kC);
      }
    }

    // 9. row wave: cells of row i reveal from left to right
    if (RA) {
      for (let j = 0; j < n; j++) {
        const tj = (j / n) * 0.5 * RA.dur;
        const local = clamp01((RA.t - tj) / (0.5 * RA.dur));
        if (local > 0 && !rev[RA.i * n + j]) overlayC(tk, RA.i, j, easeOutC(local));
      }
    }
    // 10. fill: every unrevealed cell fades in
    if (FA) {
      const a = easeOutC(FA.t / FA.dur);
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
        if (!rev[i * n + j]) overlayC(tk, i, j, a);
      }
    }

    // 11. hover on C
    if (hoverC && !(F && F.i === hoverC.i && F.j === hoverC.j)) {
      const [x, y, w, hh] = cellBox(xC, yC, hoverC.i, hoverC.j);
      outline(x, y, w, hh, tk.fg, 1.5, 0.8);
    }

    // 12. labels
    drawLabels(tk, row, col);
    drawName(tk, N.Bshow, N.bRest, xC, 11, kB);
    drawName(tk, N.A, N.aRest, xA, yC + m * s + 17, kA);
    drawName(tk, N.C, N.cRest, xC, yC + m * s + 17, kC);

    syncEq();
    syncFoot();
  }

  // ---------------------------------------------------- equation bar

  const sub = (x) => `<sub>${x}</sub>`;
  const chip = (text, idle) => `<span class="mm-chip${idle ? ' idle' : ''}">${esc(text)}</span>`;
  const biasName = () => esc(o.B.biasName ?? 'bias');
  const biasTail = (j) => (P.bias ? ` <span class="mm-op">+</span> ${biasName()}[${j}]` : '');
  const termB = (j, kk) => (P.tB ? `${esc(N.Bg)}[${j}, ${kk}]` : `${esc(N.Bg)}[${kk}, ${j}]`);

  function eqIdle() {
    return `<span class="mm-lhs">${esc(N.C)}[i, j]</span> <span class="mm-op">=</span> `
      + `<span class="mm-op">Σ${sub('k')}</span> ${esc(N.A)}[i, k] · ${termB('j', 'k')}${biasTail('j')}`
      + chip('hover or click a cell', true);
  }

  // One product and the running sum. The sum starts at the bias, so every step adds up on screen.
  function eqStep(f, kk) {
    const ed = eqDigitsOf();
    const av = P.a[f.i * P.k + kk], bv = P.bd[kk * P.n + f.j], pv = f.pvals[kk];
    const prev = f.partial[kk], next = f.partial[kk + 1];
    let sum;
    if (kk === 0) {
      sum = P.bias
        ? `Σ = ${numText(prev, ed)} (${biasName()}[${f.j}]) ${pv < 0 ? '−' : '+'} ${numText(Math.abs(pv), ed)} = ${numText(next, ed)}`
        : `Σ = ${numText(next, ed)}`;
    } else {
      sum = `Σ = ${numText(prev, ed)} ${pv < 0 ? '−' : '+'} ${numText(Math.abs(pv), ed)} = ${numText(next, ed)}`;
    }
    return `<span class="mm-lhs">${esc(N.C)}[${f.i}, ${f.j}]</span> <span class="mm-op">=</span> `
      + `<span class="mm-op">Σ${sub(`k = ${kk}`)}</span> ${esc(N.A)}[${f.i}, ${kk}] · ${termB(f.j, kk)} `
      + `<span class="mm-num">${numText(av, ed)} × ${numText(bv, ed)} = ${numText(pv, ed)}</span>`
      + chip(sum);
  }

  function eqFinal(f) {
    const ed = eqDigitsOf();
    return `<span class="mm-lhs">${esc(N.C)}[${f.i}, ${f.j}]</span> <span class="mm-op">=</span> `
      + `<span class="mm-op">Σ${sub('k')}</span> ${esc(N.A)}[${f.i}, k] · ${termB(f.j, 'k')}${biasTail(f.j)} `
      + `<span class="mm-op">=</span> <span class="mm-num">${numText(f.partial[f.K], ed)}</span>`
      + chip('stored in C');
  }

  function eqRow(i) {
    return `<span class="mm-lhs">${esc(N.C)}[${i}, :]</span> <span class="mm-op">=</span> `
      + `<span class="mm-op">Σ${sub('k')}</span> ${esc(N.A)}[${i}, k] · ${termB('j', 'k')}`
      + chip(`row ${i + 1} of ${P.m}`);
  }

  function eqAll() {
    return `<span class="mm-lhs">${esc(N.C)}</span> <span class="mm-op">=</span> ${esc(N.A)} · ${esc(N.Bshow)}`
      + chip('all cells');
  }

  function syncEq() {
    const busy = !!tl && !tl.paused;
    eqEl.setAttribute('aria-live', busy ? 'off' : 'polite');   // no announcements per step
    let key, html;
    if (RA) { key = `r${RA.i}`; html = eqRow(RA.i); }
    else if (FA) { key = 'f'; html = eqAll(); }
    else if (F) {
      const ph = phaseOf(F);
      if (ph.step != null) { key = `s${F.i},${F.j},${ph.step}`; html = eqStep(F, ph.step); }
      else { key = `d${F.i},${F.j}`; html = eqFinal(F); }
    } else { key = 'idle'; html = eqIdle(); }
    if (key === eqKey) return;
    const first = eqKey === '';
    eqKey = key;
    eqEl.innerHTML = html;
    if (!first && !key.startsWith('s') && !reducedMotion() && typeof eqEl.animate === 'function') {
      eqEl.animate(
        [{ boxShadow: 'inset 0 0 0 2px var(--accent)' }, { boxShadow: 'inset 0 0 0 2px transparent' }],
        { duration: 420, easing: 'ease-out' });
    }
  }

  // The status line describes the state as it is drawn (pinned cell, playing), not the last click.
  function syncFoot() {
    const playing = !!tl && tl.kind === 'play' && !tl.paused;
    const text = playing
      ? 'Playing. Space or a click pauses. Escape releases a pinned cell.'
      : pinned && F
        ? `Pinned to ${N.C}[${F.i}, ${F.j}]. Click the cell again or press Escape to release.`
        : 'Hover a cell of C to compute it. Click to pin it, arrow keys move, Space plays or pauses.';
    if (text === footKey) return;
    footKey = text;
    foot.textContent = text;
  }

  // ------------------------------------------------------- timeline

  // Before a new timeline replaces the running one, what has started is finished on screen.
  function supersede() {
    if (F && F.t > 0) commitCell(F.i, F.j);
    if (RA && RA.t > 0) {
      for (let j = 0; j < P.n; j++) rev[RA.i * P.n + j] = 1;
      paintC();
    }
    if (FA && FA.t > 0) { rev.fill(1); paintC(); }
    tl = null; F = null; RA = null; FA = null;
  }

  function startUnit(u) {
    if (u.type === 'focus') { F = makeFocus(P, u.i, u.j, u.rate ?? 1); RA = null; FA = null; return F; }
    if (u.type === 'row') { RA = { i: u.i, t: 0, dur: u.dur, rate: 1 }; F = null; FA = null; return RA; }
    FA = { t: 0, dur: FILL_MS, rate: 1 };
    F = null; RA = null;
    return FA;
  }

  function finishUnit(u) {
    if (u === F) commitCell(u.i, u.j);
    else if (u === RA) {
      for (let j = 0; j < P.n; j++) rev[RA.i * P.n + j] = 1;
      paintC();
    } else if (u === FA) {
      rev.fill(1);
      paintC();
    }
  }

  function nextUnit() {
    if (!tl) return;
    if (tl.idx >= tl.units.length) {
      tl = null;
      RA = null;
      FA = null;
      return;
    }
    tl.cur = startUnit(tl.units[tl.idx++]);
  }

  function buildPlayUnits() {
    const { m, n, k } = P;
    if (m * n <= CELL_MODE_MAX) {
      const todo = [];
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) if (!rev[i * n + j]) todo.push([i, j]);
      if (!todo.length) return [];
      const per = Math.max(120, CELL_PLAY_MS / todo.length);      // ms per cell at speed 1
      const rate = clamp(focusDur(k) / per, 0.4, RATE_MAX);
      return todo.map(([i, j]) => ({ type: 'focus', i, j, rate }));
    }
    const rows = [];
    for (let i = 0; i < m; i++) {
      let open = false;
      for (let j = 0; j < n; j++) if (!rev[i * n + j]) { open = true; break; }
      if (open) rows.push(i);
    }
    if (!rows.length) return [];
    const total = clamp(120 * m, ROW_PLAY_MS[0], ROW_PLAY_MS[1]);
    const dur = total / rows.length;
    return rows.map((i) => ({ type: 'row', i, dur }));
  }

  function advance(dt) {
    let busy = false;
    if (tw) {
      busy = true;
      tw.t += dt;
      if (tw.t >= TWEEN_MS) {
        disp = copyOf(P);
        tw = null;
        paintAll();
      } else {
        const e = easeIO(tw.t / TWEEN_MS);
        const rr = (key) => lerp(tw.range[key][0], tw.range[key][1], e);
        if (tw.moving.a) { for (let x = 0; x < P.a.length; x++) disp.a[x] = lerp(tw.from.a[x], P.a[x], e); paintMat('a', P.m, P.k, disp.a, null, showNums(), rr('a')); }
        if (tw.moving.bd) { for (let x = 0; x < P.bd.length; x++) disp.bd[x] = lerp(tw.from.bd[x], P.bd[x], e); paintMat('bd', P.k, P.n, disp.bd, null, showNums(), rr('bd')); }
        if (tw.moving.c) { for (let x = 0; x < P.c.length; x++) disp.c[x] = lerp(tw.from.c[x], P.c[x], e); paintC(rr('c')); }
      }
    }
    if (tl && !tl.paused) {
      busy = true;
      // Spend this frame's time on the units in order. What a finished unit leaves over goes to the
      // next one, so the timeline's length does not depend on the frame rate.
      let budget = dt * speed;
      while (tl && !tl.paused && budget > 0) {
        if (!tl.cur) nextUnit();
        if (!tl || !tl.cur) break;
        const cur = tl.cur, r = cur.rate || 1;
        const need = (cur.dur - cur.t) / r;
        if (budget >= need) {
          budget -= need;
          cur.t = cur.dur;
          finishUnit(cur);
          tl.cur = null;
          nextUnit();
        } else {
          cur.t += budget * r;
          budget = 0;
        }
      }
    }
    return busy;
  }

  function kick() {
    if (destroyed || raf) return;
    raf = requestAnimationFrame(frame);
  }

  function frame(now) {
    raf = 0;
    if (destroyed) return;
    const dt = lastNow ? Math.min(64, now - lastNow) : 16;
    lastNow = now;
    const busy = advance(dt);
    draw();
    if (busy) kick();
    else lastNow = 0;
  }

  // ----------------------------------------------------- public API

  const api = {
    el,

    focus(i, j, { animate = true } = {}) {
      if (destroyed) return;
      i = clamp(Math.round(i), 0, P.m - 1);
      j = clamp(Math.round(j), 0, P.n - 1);
      supersede();
      if (!animate || reducedMotion()) {
        F = makeFocus(P, i, j, 1);
        F.t = F.dur;
        commitCell(i, j);
        draw();
        return;
      }
      tl = { kind: 'focus', units: [{ type: 'focus', i, j, rate: 1 }], idx: 0, paused: false, cur: null };
      nextUnit();
      kick();
    },

    play() {
      if (destroyed) return;
      if (tl && tl.kind === 'play') { tl.paused = false; kick(); return; }
      if (reducedMotion()) { api.showAll(); return; }
      pinned = false;
      supersede();
      let units = buildPlayUnits();
      if (!units.length) {                 // everything is revealed: play again from the start
        rev.fill(0);
        paintC();
        units = buildPlayUnits();
      }
      if (!units.length) { kick(); return; }
      tl = { kind: 'play', units, idx: 0, paused: false, cur: null };
      nextUnit();
      kick();
    },

    pause() {
      if (destroyed) return;
      if (tl) tl.paused = true;
      kick();
    },

    reset() {
      if (destroyed) return;
      tl = null; F = null; RA = null; FA = null;
      pinned = false;
      rev.fill(0);
      paintC();
      kick();
    },

    showAll() {
      if (destroyed) return;
      supersede();
      if (reducedMotion()) { rev.fill(1); paintC(); kick(); return; }
      tl = { kind: 'fill', units: [{ type: 'fill' }], idx: 0, paused: false, cur: null };
      nextUnit();
      kick();
    },

    // New data, labels or text. Values tween to their new numbers. A change of shape or layout
    // relayouts at once. A text-only change just redraws the labels.
    update(patch = {}) {
      if (destroyed) return;
      const next = {
        ...o,
        ...patch,
        A: { ...o.A, ...(patch.A || {}) },
        B: { ...o.B, ...(patch.B || {}) },
        C: { ...(o.C || {}), ...(patch.C || {}) },
      };
      const P2 = prepare(next);                 // throws before anything changes
      if (patch.speed !== undefined) speed = clampSpeed(patch.speed);
      const sameShape = P2.m === P.m && P2.k === P.k && P2.n === P.n;
      const dataSame = sameShape && sameArr(P.a, P2.a) && sameArr(P.bd, P2.bd) && sameArr(P.c, P2.c) && sameArr(P.bias, P2.bias);
      const layoutSame = sameShape && layoutSig(next, P2) === layoutSig(o, P);
      const textSame = textSig(next) === textSig(o);
      if (dataSame && layoutSame && textSame) { o = next; P = P2; return; }   // nothing visible changed

      const keepF = F ? { i: F.i, j: F.j } : null;
      supersede();
      o = next;
      P = P2;
      N = names(o, P);
      eqKey = '';
      footKey = '';
      if (!sameShape) {
        rev = new Uint8Array(P.m * P.n);
        disp = copyOf(P);
        tw = null;
      } else if (reducedMotion()) {
        disp = copyOf(P);
        tw = null;
      } else {
        const from = disp;
        disp = { a: Float64Array.from(from.a), bd: Float64Array.from(from.bd), c: Float64Array.from(from.c) };
        tw = {
          t: 0, from,
          moving: { a: !sameArr(from.a, P.a), bd: !sameArr(from.bd, P.bd), c: !sameArr(from.c, P.c) },
          range: { a: [maxAbs(from.a), maxAbs(P.a)], bd: [maxAbs(from.bd), maxAbs(P.bd)], c: [maxAbs(from.c), maxAbs(P.c)] },
        };
      }
      if (sameShape && keepF && rev[keepF.i * P.n + keepF.j]) {
        // the focused cell stays on screen with its new numbers
        F = makeFocus(P, keepF.i, keepF.j, 1);
        F.t = F.dur;
      }
      titleEl.textContent = o.title ?? '';
      titleEl.hidden = !o.title;
      canvas.setAttribute('aria-label', `${N.C} = ${N.A} · ${N.Bshow}, a ${P.m} by ${P.n} matrix. Hover a cell, or use the arrow keys, to compute it.`);
      if (!layoutSame) {
        lastAvail = -1;
        relayout(true);                     // repaints every matrix from disp
      } else {
        if (reducedMotion()) paintAll();    // the tween branch repaints what moves
        draw();
      }
      kick();
    },

    setSpeed(v) { speed = clampSpeed(v); },

    // Cell box of C[i, j] in CSS px, relative to the canvas (for callouts placed by the page).
    cellRect(i, j) {
      if (!L) return null;
      const [x, y, w, hh] = cellBox(L.xC, L.yC, i, j);
      return { x, y, w, h: hh };
    },

    values() { return Float64Array.from(P.c); },

    // Largest |C - reference| over all cells, or null when no reference was given.
    maxError() {
      if (!P.ref) return null;
      let e = 0;
      for (let x = 0; x < P.c.length; x++) e = Math.max(e, Math.abs(P.c[x] - P.ref[x]));
      return e;
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      cancelAnimationFrame(raf);
      raf = 0;
      if (roFrame) cancelAnimationFrame(roFrame);
      clearTimeout(dwell);
      ro?.disconnect();
      dprMQ?.removeEventListener?.('change', onDprChange);
      instances.delete(inst);
      hideTip();
      tl = null; F = null; RA = null; FA = null; tw = null;
      el.remove();
    },

    get speed() { return speed; },
    get playing() { return !!tl && tl.kind === 'play' && !tl.paused; },
  };

  const inst = {
    onTheme() {
      if (destroyed) return;
      paintAll();
      draw();
    },
  };
  instances.add(inst);

  // ------------------------------------------------------ interaction

  function hitTest(ev) {
    if (!L) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = (ev.clientX - rect.left) * (L.W / rect.width);
    const y = (ev.clientY - rect.top) * (L.H / rect.height);
    const { s, xA, xC, yB, yC } = L;
    const { m, k, n } = P;
    if (x >= xC && x < xC + n * s && y >= yC && y < yC + m * s) {
      return { z: 'C', i: clamp(Math.floor((y - yC) / s), 0, m - 1), j: clamp(Math.floor((x - xC) / s), 0, n - 1) };
    }
    if (x >= xA && x < xA + k * s && y >= yC && y < yC + m * s) {
      return { z: 'A', i: clamp(Math.floor((y - yC) / s), 0, m - 1), kk: clamp(Math.floor((x - xA) / s), 0, k - 1) };
    }
    if (x >= xC && x < xC + n * s && y >= yB && y < yB + k * s) {
      return { z: 'B', kk: clamp(Math.floor((y - yB) / s), 0, k - 1), j: clamp(Math.floor((x - xC) / s), 0, n - 1) };
    }
    return null;
  }

  function scheduleDwell() {
    clearTimeout(dwell);
    if (!hoverC || pinned || (tl && tl.kind === 'play' && !tl.paused)) return;
    const { i, j } = hoverC;
    dwell = setTimeout(() => {
      if (destroyed || !hoverC || hoverC.i !== i || hoverC.j !== j) return;
      if (F && F.i === i && F.j === j) return;
      api.focus(i, j);
    }, DWELL_MS);
  }

  canvas.addEventListener('pointermove', (ev) => {
    if (destroyed) return;
    const hit = hitTest(ev);
    const nextC = hit && hit.z === 'C' ? { i: hit.i, j: hit.j } : null;
    const changedC = (nextC?.i ?? -1) !== (hoverC?.i ?? -1) || (nextC?.j ?? -1) !== (hoverC?.j ?? -1);
    const nextA = hit && hit.z === 'A' ? hit.i : null;
    const nextB = hit && hit.z === 'B' ? hit.j : null;
    const changed = changedC || nextA !== hoverA || nextB !== hoverB;
    if (changedC) {
      hoverC = nextC;
      scheduleDwell();
    }
    hoverA = nextA;
    hoverB = nextB;
    canvas.classList.toggle('mm-hot', !!nextC);
    if (hit) {
      let html = '';
      const d = dig();
      if (hit.z === 'C') {
        const idx = hit.i * P.n + hit.j;
        const val = rev[idx] ? `<b>${numText(P.c[idx], d)}</b>` : '<span class="muted">not computed yet</span>';
        html = `${esc(N.C)}[${hit.i}, ${hit.j}] = ${val}`;
      } else if (hit.z === 'A') {
        html = `${esc(N.A)}[${hit.i}, ${hit.kk}] = <b>${numText(P.a[hit.i * P.k + hit.kk], d)}</b>`;
      } else {
        const v = P.bd[hit.kk * P.n + hit.j];
        const idx = P.tB ? `[${hit.j}, ${hit.kk}]` : `[${hit.kk}, ${hit.j}]`;
        html = `${esc(N.Bg)}${idx} = <b>${numText(v, d)}</b>`;
      }
      showTip(ev.clientX, ev.clientY, html);
    } else {
      hideTip();
    }
    if (changed) kick();
  });

  canvas.addEventListener('pointerleave', () => {
    if (destroyed) return;
    hoverC = null; hoverA = null; hoverB = null;
    clearTimeout(dwell);
    canvas.classList.remove('mm-hot');
    hideTip();
    kick();
  });

  canvas.addEventListener('click', (ev) => {
    if (destroyed) return;
    const hit = hitTest(ev);
    if (!hit || hit.z !== 'C') return;
    if (tl && tl.kind === 'play' && !tl.paused) { api.pause(); return; }
    clearTimeout(dwell);
    if (pinned && F && F.i === hit.i && F.j === hit.j) { pinned = false; kick(); return; }
    pinned = true;
    if (!(F && F.i === hit.i && F.j === hit.j)) api.focus(hit.i, hit.j);
    kick();
    canvas.focus({ preventScroll: true });
  });

  const MOVES = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] };
  canvas.addEventListener('keydown', (ev) => {
    if (destroyed) return;
    const mv = MOVES[ev.key];
    if (mv) {
      ev.preventDefault();
      const i = F ? F.i + mv[0] : 0, j = F ? F.j + mv[1] : 0;
      pinned = true;
      api.focus(clamp(i, 0, P.m - 1), clamp(j, 0, P.n - 1));
      return;
    }
    if (ev.key === 'Escape') { pinned = false; kick(); return; }
    if (ev.key === ' ' || ev.key === 'Enter') {
      ev.preventDefault();
      if (tl && tl.kind === 'play' && !tl.paused) api.pause(); else api.play();
    }
  });

  // -------------------------------------------------------------- start

  function injectCss() {
    if (document.getElementById('mm-style')) return;
    document.head.append(h('style', { id: 'mm-style' }, MM_CSS));
  }

  // Redraw when the device pixel ratio changes (moving the window to another screen, zoom).
  function watchDpr() {
    if (destroyed || typeof matchMedia !== 'function') return;
    if (dprMQ) dprMQ.removeEventListener?.('change', onDprChange);
    dprMQ = matchMedia(`(resolution: ${dprNow()}dppx)`);
    dprMQ.addEventListener?.('change', onDprChange);
  }
  function onDprChange() {
    if (destroyed) return;
    watchDpr();
    relayout(true);
  }

  titleEl.textContent = o.title ?? '';
  titleEl.hidden = !o.title;
  canvas.setAttribute('aria-label', `${N.C} = ${N.A} · ${N.Bshow}, a ${P.m} by ${P.n} matrix. Hover a cell, or use the arrow keys, to compute it.`);
  if (typeof ResizeObserver === 'function') {
    ro = new ResizeObserver(() => {
      if (destroyed || roFrame) return;
      roFrame = requestAnimationFrame(() => { roFrame = 0; relayout(); });
    });
    ro.observe(stage);
  }
  watchDpr();
  relayout(true);
  document.fonts?.ready?.then(() => { if (!destroyed) relayout(true); });
  return api;
}

const MM_CSS = `
.mm{display:grid;gap:10px;min-width:0}
.mm-head{display:flex;align-items:center;gap:12px;min-width:0}
.mm-title{font:600 var(--fs-xs)/1.3 var(--font-mono);color:var(--fg-2);min-width:0}
.mm-title[hidden]{display:none}
.mm-scale{display:inline-flex;align-items:center;gap:6px;margin-left:auto;font:500 11px/1 var(--font-mono);color:var(--fg-3);white-space:nowrap}
.mm-scale i{display:block;width:64px;height:8px;border-radius:4px;background:linear-gradient(90deg,var(--heat-neg),var(--heat-mid),var(--heat-pos));box-shadow:inset 0 0 0 1px var(--line)}
.mm-eq{display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;min-height:44px;padding:9px 14px;border-radius:var(--r-md);background:var(--surface-2);border:1px solid var(--line);font:500 13px/1.5 var(--font-mono);color:var(--fg-2);min-width:0}
.mm-eq .mm-lhs{color:var(--fg);font-weight:600}
.mm-eq .mm-op{color:var(--fg-3)}
.mm-eq sub{font-size:.8em;line-height:0}
.mm-eq .mm-num{color:var(--fg);font-variant-numeric:tabular-nums}
.mm-eq .mm-chip{margin-left:auto;padding:3px 10px;border-radius:999px;background:var(--accent-soft);color:var(--fg);font-weight:600;white-space:nowrap;font-variant-numeric:tabular-nums}
.mm-eq .mm-chip.idle{background:transparent;border:1px dashed var(--line-strong);color:var(--fg-3);font-weight:500}
.mm-stage{min-width:0;max-width:100%}
.mm canvas{display:block;max-width:100%;height:auto;border-radius:4px;touch-action:pan-y}
.mm canvas.mm-hot{cursor:pointer}
.mm canvas:focus-visible{outline:2px solid var(--focus);outline-offset:3px}
.mm-foot{font-size:var(--fs-xs);color:var(--fg-3);min-height:1.3em}
`;

// Vision tokens: how an image becomes tokens for a VLM or a computer-use agent.
// The input is a procedurally drawn app window (or an upload), resized to a multiple of the
// patch size, cut into patches, flattened, embedded, joined with the text tokens and counted.
//
// Illustrations, always labelled: the encoder and projector matrices and the 2-D position
// vectors are seeded random, and the image block of the mask is a VLM design (this app's LLM has
// no image input). Everything the LLM computes is real: the text rows are its emb.out, the text
// block of the mask follows its own mask setting, and every count comes from the patch grid.
//
// Choices, stated in the captions: the encoder works at half the LLM width, so the projector
// widens it (real encoders are narrower than their LLM). First-view steps wait for their own
// animation (STEP_MS), the one documented exception to docs/STYLE_GUIDE.md section 6. The matrix
// fill is the slowest at 1.6 s; bar fills use MOTION.move.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, vec, op, checkBadge, texLabel, useStyle, MOTION, ZONES } from '../kit.js';
import { h, clamp, parseColor, diverging, onThemeChange, reducedMotion, showTip, hideTip, button, texHTML, num, fmt, heatmap } from '../../ui/index.js';
import { mulberry32, gaussian, maskAllows } from '../../engine/index.js';

const STEPS = ['Patches', 'Flatten', 'Embed', 'Into the LLM', 'How many tokens'];
const STEP_MS = [2600, 3600, 3200, 3000, 2800];   // first-view pace: the time each step's own animation needs
const SIZES = [224, 336];
const PATCH_SIZES = [14, 16, 32];
const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];   // CLIP's normalisation constants (RGB)
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];
const MEAN_FILL = 'rgb(124,116,104)';                    // pads a non-square upload to a square
const RANGE = 2;                                         // colour range of the patch vectors and the embedding matrix (±2)
const TEXT_SHOW = 8;                                     // text positions shown in the mask and the sequence: a window around the query
const SEED_W = 1729, SEED_PROJ = 4242, SEED_ROW = 9001, SEED_COL = 9013;   // seeds of the illustrative matrices
const POS_SCALE = 0.5;                                   // size of the random 2-D position vectors (illustration)
const SLICE_MS = 8;                                      // work per frame while a result is computed
const FXR = 2;                                           // backing-store scale of the animation canvas
const IMG = { x: 110, y: 262, s: 440 };                  // the input image (design units)
const STRIP_Y = 728;                                     // the image controls, under the image
const FX = { y: 260, h: 520 };                           // the animation layer under the header
const TILE = { x: 740, y: 296, s: 240 };                 // the patch, enlarged
const VEC_X = 600, VEC_Y = 556;                          // the flattened vector, right of the image (label above)
const EMB = { x: 800, y: 296, w: 240, maxH: 420 };       // the N × dv embedding matrix (step 2)
const MASK = { x: 110, y: 300, img: 176, cell: 20, labelW: 60, labelH: 44 };   // step 3
const SEQ = { x: 690, y: 296, w: 380 };                  // the stacked sequence (step 3)
const SEQ_IMG_H = 180;
const BARS = { x: 110, y0: 300, dy: 72 };                // the token bars (step 4)

useStyle('vision', `
.vz-group { position: absolute; left: 0; top: 0; width: 0; height: 0; opacity: 0; visibility: hidden; transition: opacity .35s var(--ease), visibility 0s linear .35s; }
.vz-group.on { opacity: 1; visibility: visible; transition: opacity .6s var(--ease) .12s, visibility 0s; }
.vz-fade { position: absolute; opacity: 0; visibility: hidden; transition: opacity .35s var(--ease), visibility 0s linear .35s; }
.vz-fade.on { opacity: 1; visibility: visible; transition: opacity .6s var(--ease) .12s, visibility 0s; }
.vz-k { font: 600 15px/1.3 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.vz-note { font-size: 16px; line-height: 1.5; color: var(--fg-2); max-width: 340px; }
.vz-note b { color: var(--fg); font-weight: 650; }
.vz-stack, .vz-eqs { display: grid; gap: 14px; justify-items: start; }
.vz-eqs .k-tex { color: var(--fg); }
.vz-card { display: grid; gap: 10px; padding: 12px 14px; border-radius: 12px; border: 1px solid var(--line); background: color-mix(in srgb, var(--surface) 85%, transparent); }
.vz-strip { display: flex; align-items: center; gap: 8px; }
.vz-pl { font: 600 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.vz-sep { width: 1px; height: 28px; background: var(--line); margin: 0 10px; }
.vz-pick { min-width: 44px; }
.vz-strip .btn { height: 36px; }
.vz-imgwrap { position: absolute; width: ${IMG.s}px; height: ${IMG.s}px; border-radius: 12px; overflow: hidden; background: var(--surface-2); border: 1px solid var(--line); box-shadow: var(--shadow-2); opacity: 0; visibility: hidden; transition: opacity .35s var(--ease), visibility 0s linear .35s; }
.vz-imgwrap.on { opacity: 1; visibility: visible; transition: opacity .6s var(--ease) .12s, visibility 0s; }
.vz-img { position: absolute; left: 0; top: 0; width: ${IMG.s}px; height: ${IMG.s}px; image-rendering: pixelated; }
.vz-cells { position: absolute; inset: 0; display: grid; }
.vz-cell { box-sizing: border-box; border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent); cursor: pointer; transition: background-color .2s; }
.vz-cell:hover { background: color-mix(in srgb, var(--accent) 16%, transparent); }
.vz-cell.sel { background: color-mix(in srgb, var(--accent) 22%, transparent); box-shadow: inset 0 0 0 2px var(--accent); }
.vz-scan { position: absolute; left: 0; top: 0; width: 3px; height: 100%; background: var(--accent); box-shadow: 0 0 16px 4px color-mix(in srgb, var(--accent) 45%, transparent); opacity: 0; }
.vz-fx { position: absolute; left: 0; top: ${FX.y}px; width: 1600px; height: ${FX.h}px; pointer-events: none; }
.vz-tex { position: absolute; image-rendering: pixelated; }
.vz-tile { position: absolute; image-rendering: pixelated; transition: opacity .4s var(--ease); }
.vz-tilelbl { font: 500 15px/1.5 var(--font-mono); color: var(--fg-3); white-space: nowrap; transition: opacity .4s var(--ease); }
.vz-arrow { position: absolute; height: 2px; background: var(--c-embed); }
.vz-arrow::after { content: ''; position: absolute; right: -3px; top: -4px; border-left: 8px solid var(--c-embed); border-top: 5px solid transparent; border-bottom: 5px solid transparent; }
.vz-band { position: absolute; border: 2px solid var(--accent); border-radius: 3px; opacity: 0; pointer-events: none; transition: opacity .2s; }
.vz-selrow { position: absolute; border: 2px solid var(--accent); border-radius: 3px; opacity: 0; pointer-events: none; transition: opacity .4s var(--ease); }
.vz-rowlbl { font: 500 15px/1.2 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.vz-selLbl { color: var(--fg); }
.vz-mask { position: absolute; }
.vz-mb { position: absolute; box-sizing: border-box; display: grid; place-content: center; text-align: center; border-radius: 8px; font: 600 15px/1.3 var(--font-mono); }
.vz-mb.allow { background: color-mix(in srgb, var(--c-attn) 82%, transparent); color: var(--surface); }
.vz-mb.hide { background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--c-mask) 70%, transparent) 0 2px, transparent 2px 7px); color: var(--fg-2); border: 1px dashed var(--line-strong); }
.vz-mb.hide b, .vz-mb.hide small { justify-self: center; background: var(--surface); color: var(--fg-2); border-radius: 6px; padding: 2px 6px; }
.vz-mb small { font: 500 15px/1.3 var(--font-mono); }
.vz-mc { position: absolute; box-sizing: border-box; border-radius: 3px; }
.vz-mc.allow { background: color-mix(in srgb, var(--c-attn) 82%, transparent); }
.vz-mc.hide { background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--c-mask) 70%, transparent) 0 2px, transparent 2px 7px); }
.vz-mc.sel { outline: 2px solid var(--accent); outline-offset: 1px; }
.vz-ml { position: absolute; font: 600 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.vz-row { position: absolute; display: grid; grid-template-columns: 360px 1fr 120px; align-items: center; height: 52px; }
.vz-rl { display: grid; gap: 3px; }
.vz-rl b { font: 600 17px/1.2 var(--font-body); }
.vz-rl small { font: 500 15px/1.2 var(--font-mono); color: var(--fg-3); }
.vz-track { position: relative; height: 26px; background: var(--surface-2); border-radius: 7px; }
.vz-fill { position: absolute; left: 0; top: 0; height: 100%; border-radius: 7px; transition: width 600ms cubic-bezier(.65,0,.35,1); }
.vz-row.hot .vz-track { box-shadow: 0 0 0 2px var(--accent); }
.vz-val { font: 600 18px/1 var(--font-mono); text-align: right; padding-right: 4px; }
.vz-kv { display: grid; gap: 3px; }
.vz-kv span { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); }
.vz-kv b { font: 600 19px/1.3 var(--font-mono); color: var(--fg); }
`);

// ------------------------------------------------------------------ numbers and helpers
const lerp = (a, b, t) => a + (b - a) * t;
const seg = (u, a, b) => clamp((u - a) / (b - a), 0, 1);
const easeIO = (t) => (t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2);
const grp = (n) => Math.round(n).toLocaleString('en-US').replace(/,/g, '{,}');   // TeX thousands
const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const maxAbs = (a) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m || 1; };
const rmsOf = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / Math.max(1, a.length)) || 1; };
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
const canvasOf = (w, hh) => { const c = document.createElement('canvas'); c.width = w; c.height = hh; return c; };
const tx = (latex, size = 22) => texLabel(latex, { size, display: true });
const kvLine = (k, v, w = 340) => h('div', { class: 'vz-kv', style: { width: `${w}px` } }, h('span', null, k), h('b', null, v));   // plain numbers, not maths
function place(parent, node, x, y, w, hh) {
  node.style.position = 'absolute';
  node.style.left = `${x}px`;
  node.style.top = `${y}px`;
  if (w != null) node.style.width = `${w}px`;
  if (hh != null) node.style.height = `${hh}px`;
  parent.append(node);
  return node;
}
// The encoder's width: half the LLM width. A choice: real encoders are narrower than their LLM.
const encWidth = (d) => Math.max(2, Math.floor(d / 2));
const dotAt = (A, ai, B, bj, K) => { let acc = 0; for (let k = 0; k < K; k++) acc += A[ai + k] * B[bj + k]; return acc; };

// Diverging colour lookup (256 entries) from the theme tokens; reset on theme change.
let LUT = null;
const palette = () => (LUT ||= Array.from({ length: 256 }, (_, i) => parseColor(diverging(i / 127.5 - 1))));
onThemeChange(() => { LUT = null; });

// Paint the first `upto` rows of a rows × cols matrix, one pixel per entry (CSS scales it).
function fillRows(can, data, rows, cols, range, upto = rows) {
  if (can.width !== cols) can.width = cols;
  if (can.height !== rows) can.height = rows;
  const cx = can.getContext('2d'), img = cx.createImageData(cols, rows), px = img.data, L = palette();
  const n = Math.min(upto, rows) * cols;
  for (let i = 0; i < n; i++) {
    const t = clamp(data[i] / range, -1, 1), c = L[Math.round((t + 1) * 127.5)], o = i * 4;
    px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
  }
  cx.clearRect(0, 0, cols, rows);
  cx.putImageData(img, 0, 0);
}

// Seeded random matrix, rows × cols, entries ~ N(0, scale²) (default scale 1/√cols): an illustration.
function randMatrix(rows, cols, seed, scale = 1 / Math.sqrt(cols)) {
  const next = gaussian(mulberry32(seed));
  const M = new Float64Array(rows * cols);
  for (let i = 0; i < M.length; i++) M[i] = next() * scale;
  return M;
}

// Small least-recently-used cache. Results are kept, not rebuilt, when the reader switches back.
function lru(limit) {
  const m = new Map();
  return {
    has: (k) => m.has(k),
    get(k) { if (!m.has(k)) return undefined; const v = m.get(k); m.delete(k); m.set(k, v); return v; },
    set(k, v) { m.delete(k); m.set(k, v); while (m.size > limit) m.delete(m.keys().next().value); return v; },
  };
}
const MATS = lru(12);                                    // seeded matrices, by shape and seed
const cachedMat = (key, make) => (MATS.has(key) ? MATS.get(key) : MATS.set(key, make()));

// Rounded rectangle path (no roundRect dependency).
function rrPath(x, X, Y, w, hh, r) {
  x.beginPath();
  x.moveTo(X + r, Y);
  x.arcTo(X + w, Y, X + w, Y + hh, r);
  x.arcTo(X + w, Y + hh, X, Y + hh, r);
  x.arcTo(X, Y + hh, X, Y, r);
  x.arcTo(X, Y, X + w, Y, r);
  x.closePath();
}

// The built-in sample: a generic app window with a title bar, a sign-in form, buttons,
// text lines and a small table, drawn at 448 px and resampled like any other image.
function drawSample() {
  const c = canvasOf(448, 448), x = c.getContext('2d');
  const bar = (X, Y, w, hh, col) => { rrPath(x, X, Y, w, hh, hh / 2); x.fillStyle = col; x.fill(); };
  x.fillStyle = '#e4e9f1';
  x.fillRect(0, 0, 448, 448);
  x.save();
  x.shadowColor = 'rgba(20,25,34,.28)'; x.shadowBlur = 16; x.shadowOffsetY = 5;
  rrPath(x, 28, 28, 392, 392, 16); x.fillStyle = '#ffffff'; x.fill();
  x.restore();
  x.save();
  rrPath(x, 28, 28, 392, 392, 16); x.clip();
  x.fillStyle = '#2a3142'; x.fillRect(28, 28, 392, 40);
  ['#ff5f57', '#febc2e', '#28c840'].forEach((col, i) => { x.beginPath(); x.arc(48 + i * 16, 48, 5, 0, Math.PI * 2); x.fillStyle = col; x.fill(); });
  bar(176, 44, 110, 8, '#58627a');
  x.fillStyle = '#f1f4f8'; x.fillRect(28, 68, 92, 352);
  rrPath(x, 40, 82, 68, 28, 8); x.fillStyle = '#4b54d6'; x.fill();
  for (let i = 0; i < 5; i++) bar(52, 88 + i * 40, 48, 8, i === 0 ? '#ffffff' : '#b9c2cf');
  x.fillStyle = '#1f2430'; x.font = '600 23px system-ui, sans-serif'; x.fillText('Sign in', 146, 108);
  bar(146, 120, 220, 7, '#c3cad6'); bar(146, 135, 190, 7, '#c3cad6'); bar(146, 150, 160, 7, '#c3cad6');
  for (const Y of [176, 222]) { rrPath(x, 146, Y, 250, 36, 8); x.fillStyle = '#ffffff'; x.fill(); x.strokeStyle = '#b9c2cf'; x.lineWidth = 1.5; x.stroke(); }
  bar(160, 189, 96, 8, '#8a94a6');
  for (let i = 0; i < 6; i++) { x.beginPath(); x.arc(162 + i * 11, 240, 3.2, 0, Math.PI * 2); x.fillStyle = '#8a94a6'; x.fill(); }
  rrPath(x, 146, 280, 120, 38, 9); x.fillStyle = '#4b54d6'; x.fill();
  rrPath(x, 280, 280, 116, 38, 9); x.fillStyle = '#ffffff'; x.fill(); x.strokeStyle = '#b9c2cf'; x.stroke();
  x.textAlign = 'center'; x.font = '600 15px system-ui, sans-serif';
  x.fillStyle = '#ffffff'; x.fillText('Sign in', 206, 304);
  x.fillStyle = '#3c4655'; x.fillText('Cancel', 338, 304);
  x.textAlign = 'left';
  rrPath(x, 146, 340, 250, 62, 10); x.fillStyle = '#f6f8fb'; x.fill(); x.strokeStyle = '#e1e6ee'; x.stroke();
  for (let i = 0; i < 2; i++) {
    const Y = 352 + i * 24;
    x.beginPath(); x.arc(164, Y + 6, 4.5, 0, Math.PI * 2); x.fillStyle = i ? '#f0b43a' : '#1f9a5d'; x.fill();
    bar(178, Y + 2, 120, 7, '#c3cad6'); bar(330, Y + 2, 46, 7, '#c3cad6');
  }
  x.restore();
  return c;
}

// Pad to a square with the mean colour (as LLaVA does), then resize to `side`.
function squareOf(src, side) {
  const c = canvasOf(side, side), x = c.getContext('2d');
  x.imageSmoothingQuality = 'high';
  x.fillStyle = MEAN_FILL;
  x.fillRect(0, 0, side, side);
  const sc = side / Math.max(src.width, src.height);
  x.drawImage(src, (side - src.width * sc) / 2, (side - src.height * sc) / 2, src.width * sc, src.height * sc);
  return c;
}

// Runs a generator one slice per frame (SLICE_MS of work each), then calls onDone with its result.
function sliced(gen, onDone) {
  let id = 0, dead = false;
  const tick = () => {
    if (dead) return;
    const t0 = performance.now();
    for (;;) {
      const r = gen.next();
      if (r.done) { onDone(r.value); return; }
      if (performance.now() - t0 >= SLICE_MS) break;
    }
    id = requestAnimationFrame(tick);
  };
  id = requestAnimationFrame(tick);
  return () => { dead = true; cancelAnimationFrame(id); };
}

// The whole pipeline for one (input size S, patch size P, source, LLM width d). It yields between
// slices of work, so no frame is blocked for long (docs/AGENT_GUIDE.md: no more than ~30 ms).
function* prepareSteps(S, P, src, d) {
  const g = Math.floor(S / P), E = g * P, N = g * g, D = 3 * P * P, dv = encWidth(d);
  const base = squareOf(src, S);                         // the input resolution
  const can = canvasOf(E, E), cx = can.getContext('2d', { willReadFrequently: true });
  cx.imageSmoothingQuality = 'high';
  cx.drawImage(base, 0, 0, E, E);                        // the nearest multiple of P (what the grid sees)
  const px = cx.getImageData(0, 0, E, E).data;
  yield;
  // patches: every pixel is read once, patch by patch (raster order inside each patch)
  const X = new Float64Array(N * D), contrast = new Float64Array(N);
  const perPatch = Math.max(1, Math.floor(20000 / (P * P)));
  let covered = 0;
  for (let i = 0; i < N; i++) {
    const pr = Math.floor(i / g), pc = i % g;
    let at = i * D, s1 = 0, s2 = 0;
    for (let y = 0; y < P; y++) {
      for (let x = 0; x < P; x++) {
        const o = ((pr * P + y) * E + pc * P + x) * 4;
        for (let ch = 0; ch < 3; ch++) X[at++] = (px[o + ch] / 255 - CLIP_MEAN[ch]) / CLIP_STD[ch];
        const lum = 0.299 * px[o] + 0.587 * px[o + 1] + 0.114 * px[o + 2];
        s1 += lum; s2 += lum * lum; covered++;
      }
    }
    contrast[i] = s2 - (s1 * s1) / (P * P);              // the most varied patch is the default pick
    if ((i + 1) % perPatch === 0) yield;
  }
  yield;
  // embedding: e_i = W_p x_i + p_i, where p_i is the row vector plus the column vector (illustrations)
  const Wp = cachedMat(`wp|${dv}|${D}|${P}`, () => randMatrix(dv, D, SEED_W + P));
  const Rp = cachedMat(`row|${g}|${dv}|${P}`, () => randMatrix(g, dv, SEED_ROW + P, POS_SCALE));
  const Cp = cachedMat(`col|${g}|${dv}|${P}`, () => randMatrix(g, dv, SEED_COL + P, POS_SCALE));
  yield;
  const Xv = new Float64Array(N * dv);
  const perRow = Math.max(1, Math.floor(150000 / (dv * D)));
  for (let i = 0; i < N; i++) {
    const r = Math.floor(i / g), c = i % g;
    for (let j = 0; j < dv; j++) Xv[i * dv + j] = dotAt(X, i * D, Wp, j * D, D) + Rp[r * dv + j] + Cp[c * dv + j];
    if ((i + 1) % perRow === 0) yield;
  }
  yield;
  // projector: dv → d, the LLM width (illustration)
  const Wq = cachedMat(`proj|${dv}|${d}`, () => randMatrix(d, dv, SEED_PROJ + d));
  const Hi = new Float64Array(N * d);
  const perHi = Math.max(1, Math.floor(150000 / (dv * d)));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < d; j++) Hi[i * d + j] = dotAt(Xv, i * dv, Wq, j * dv, dv);
    if ((i + 1) % perHi === 0) yield;
  }
  return { S, P, g, E, N, D, d, dv, can, X, Xv, Wp, Rp, Cp, Wq, Hi, wpMax: maxAbs(Wp), covered, pick: argmax(contrast) };
}

// The text positions shown in the mask and the sequence: TEXT_SHOW consecutive ones that hold the query.
function windowOf(T, q) {
  const Ts = Math.min(T, TEXT_SHOW), w0 = clamp(q - Math.floor(TEXT_SHOW / 2), 0, T - Ts);
  return { w0, Ts };
}

// What this model's own mask says about a text token's keys (the text block of the mask).
function textRule(cfg) {
  switch (cfg.mask) {
    case 'none': return { short: 'all visible (mask: none)', phrase: 'every text token (mask: none)' };
    case 'sliding': return { short: `sliding window ${cfg.window}`, phrase: `the last ${cfg.window} text tokens (sliding window)` };
    case 'prefix': return { short: `prefix ${cfg.prefix_len}, then causal`, phrase: `the first ${cfg.prefix_len} text tokens and earlier text (prefix)` };
    default: return { short: 'causal', phrase: 'earlier text only (causal)' };
  }
}

// ---------------------------------------------------------------- scene
export default {
  mount(el, ctx) {
    const st = stage2d(el);
    let step = 0, S = 224, P = 14, sel = 0, cur = null, curKey = '', srcVer = 0, userImg = null, stopped = false;
    let job = null, busy = null;                       // the slice job that computes `cur`, and its "still working" timer
    const cache = lru(4);                              // prepared results for the last four settings
    const live = new Set();                            // cancel functions of running animations
    const track = (fn) => { live.add(fn); return fn; };
    const cancelAll = () => { live.forEach((f) => { try { f(); } catch { /* already done */ } }); live.clear(); };
    const groupVecs = STEPS.map(() => []);             // vec() handles per step group, destroyed when the group is rebuilt
    const mkVec = (group, values, opts) => { const v = vec(values, opts); groupVecs[group].push(v); return v; };
    const releaseVecs = (group) => { groupVecs[group].forEach((v) => v.destroy?.()); groupVecs[group] = []; };
    let heat = null;                                    // the text heatmap, released when rebuilt
    let cellEls = [];
    const sample = drawSample();
    const offs = [ctx.on('trace', onTrace), ctx.on('select', onSelect), onThemeChange(() => { if (cur) render(); })];
    const cellLabel = (i) => `patch ${i} · row ${Math.floor(i / cur.g)}, column ${i % cur.g}`;

    // ---------------------------------------------------- header, pickers, caption (shared kit parts)
    sceneHeader(st, {
      chips: [['Image → tokens', 'embed'], ['ViT · VLM', 'pos']], title: 'Vision tokens',
      sub: 'A vision-language model turns pixels into vectors that sit in the same sequence as the text.',
    });
    const cap = sceneCaption(st, { y: ZONES.caption.y, w: 1060 });
    const picker = tokenPicker(st, ctx, { label: 'query' });   // step 3: which query row the mask shows
    picker.el.classList.add('vz-fade');

    // ---------------------------------------------------- the image controls, under the image
    const patchPick = h('div', { class: 'k-picker' });
    const sizePick = h('div', { class: 'k-picker' });
    const sampleBtn = button('Sample', () => { userImg = null; srcVer++; sync(true); render({ animate: true }); }, { variant: 'ghost', title: 'Go back to the drawn sample image' });
    sampleBtn.style.display = 'none';
    const fileIn = h('input', { type: 'file', accept: 'image/*', style: { display: 'none' } });
    const uploadBtn = button('Upload', () => fileIn.click(), { variant: 'ghost', title: 'Use your own image. It stays in this browser.' });
    fileIn.addEventListener('change', () => {
      const f = fileIn.files?.[0];
      fileIn.value = '';
      if (!f) return;
      if (!/^image\//.test(f.type)) { ctx.store.notice('Choose an image file (PNG, JPEG or WebP).', true); return; }
      const fr = new FileReader();
      fr.onerror = () => ctx.store.notice('The file could not be read.', true);
      fr.onload = () => {
        const img = new Image();
        img.onerror = () => ctx.store.notice('The browser could not decode that image.', true);
        img.onload = () => { if (stopped) return; userImg = img; srcVer++; sampleBtn.style.display = ''; sync(true); render({ animate: true }); };
        img.src = fr.result;
      };
      fr.readAsDataURL(f);
    });
    function pickRow(row, values, value, onPick) {
      row.replaceChildren(...values.map((v) => {
        const b = h('button', { type: 'button', class: ['k-pick', 'vz-pick', v === value && 'sel'], style: { '--kind': 'var(--c-embed)' }, 'aria-pressed': String(v === value) }, h('b', null, String(v)));
        b.addEventListener('click', () => onPick(v));
        return b;
      }));
    }
    function paintControls() {
      pickRow(patchPick, PATCH_SIZES, P, (v) => { P = v; sync(true); render({ animate: true }); });
      pickRow(sizePick, SIZES, S, (v) => { S = v; sync(true); render({ animate: true }); });
      sampleBtn.style.display = userImg ? '' : 'none';
    }
    const strip = h('div', { class: ['vz-fade', 'vz-strip'] },
      h('span', { class: 'vz-pl' }, 'patch'), patchPick,
      h('span', { class: 'vz-sep' }),
      h('span', { class: 'vz-pl' }, 'image'), sizePick,
      h('span', { class: 'vz-sep' }),
      uploadBtn, sampleBtn, fileIn);
    st.add(strip, { x: IMG.x, y: STRIP_Y });

    // ---------------------------------------------------- the image (steps 0 and 1)
    const wrap = h('div', { class: 'vz-imgwrap' });
    const imgCan = h('canvas', { class: 'vz-img', role: 'img', 'aria-label': 'the input image, resized to a multiple of the patch size' });
    const cellsEl = h('div', { class: 'vz-cells' });
    const scan = h('div', { class: 'vz-scan' });
    wrap.append(imgCan, cellsEl, scan);
    st.add(wrap, { x: IMG.x, y: IMG.y });

    // one group per step; only the active one is shown and rebuilt
    const G = STEPS.map((name) => h('div', { class: 'vz-group', dataset: { step: name } }));
    G.forEach((g) => st.add(g, { x: 0, y: 0 }));
    const BUILD = [buildPatches, buildFlatten, buildEmbed, buildLLM, buildCount];
    const CAP = [captionPatches, captionFlatten, captionEmbed, captionLLM, captionCount];

    // The prepared result for the current settings. The work runs a slice per frame; `cur` is null until it is done.
    function sync(force = false) {
      const d = ctx.cfg.d_model, key = `${S}|${P}|${srcVer}|${d}`;
      if (!force && (cur || job) && key === curKey) return false;
      curKey = key;
      cancelJob();
      cur = null;
      const hit = cache.get(key);
      if (hit) { adopt(hit); return true; }
      const jobKey = key;
      job = { cancel: sliced(prepareSteps(S, P, userImg || sample, d), (res) => {
        job = null;
        clearTimeout(busy);
        cache.set(jobKey, res);
        adopt(res);
        render({ animate: true });
      }) };
      busy = setTimeout(() => { if (job) cap.set('Computing the patches and their embeddings…'); }, 250);
      return true;
    }
    function cancelJob() { job?.cancel(); job = null; clearTimeout(busy); }
    function adopt(res) {
      cur = res;
      sel = res.pick;
      buildGrid();
    }

    function buildGrid() {
      const { g, N, can } = cur;
      cellsEl.replaceChildren();
      cellsEl.style.gridTemplateColumns = `repeat(${g}, 1fr)`;
      cellsEl.style.gridTemplateRows = `repeat(${g}, 1fr)`;
      cellEls = Array.from({ length: N }, (_, i) => {
        const c = h('div', { class: 'vz-cell', dataset: { i: String(i) } });
        c.addEventListener('click', () => choose(i));
        c.addEventListener('pointerenter', (e) => showTip(e.clientX, e.clientY, cellLabel(i)));
        c.addEventListener('pointerleave', hideTip);
        cellsEl.append(c);
        return c;
      });
      imgCan.width = cur.E;
      imgCan.height = cur.E;
      imgCan.getContext('2d').drawImage(can, 0, 0);
    }

    function choose(i) {
      if (!cur) return;
      sel = i;
      render({ animate: step === 1 });
    }

    // ---------------------------------------------------- step 0: the grid and N
    function buildPatches(G0) {
      const { S: s, E, g, N, P: p, D, covered } = cur;
      const resized = s === E ? null : `${s} px is not a multiple of ${p}, so the image is first resized to ${E} px.`;
      place(G0, h('div', { class: 'vz-stack' },
        h('div', { class: 'vz-k' }, 'patch grid'),
        kvLine('rows: H ÷ P', `${E} ÷ ${p} = ${g}`, 470),
        kvLine('columns: W ÷ P', `${E} ÷ ${p} = ${g}`, 470),
        kvLine('image tokens: rows × columns', `${g} × ${g} = ${N}`, 470),
        kvLine(`numbers per patch: ${p} × ${p} × 3`, `${D}`, 470)), 600, 296, 470);
      place(G0, h('div', { class: 'vz-eqs' },
        h('div', { class: 'vz-k' }, 'one token per patch'),
        tx(String.raw`N=\left\lfloor\tfrac{H}{P}\right\rfloor\left\lfloor\tfrac{W}{P}\right\rfloor=${N}`, 24),
        checkBadge(covered, E * E, { label: 'every pixel is in one patch' }),
        resized && h('div', { class: 'vz-note' }, resized)), 1180, 296, 340);
      return { start: null };
    }
    function captionPatches() {
      const { S: s, E, g, N, P: p, pick } = cur;
      const head = s === E
        ? `A ${s} px image is cut into ${g} × ${g} patches of ${p} × ${p} pixels: <b>${N} image tokens</b>.`
        : `A ${s} px image is resized to ${E} px, a multiple of ${p}, and cut into ${g} × ${g} patches: <b>${N} image tokens</b>.`;
      const outline = sel === pick
        ? `Patch ${sel}, the most varied, is outlined. Click any cell to choose another.`
        : `Patch ${sel} is outlined. Click any cell to choose another.`;
      return `${head} ${outline}`;
    }

    // ---------------------------------------------------- step 1: one patch unrolls into a vector
    function buildFlatten(G1, animate) {
      const { P: p, D, g } = cur;
      const r = Math.floor(sel / g), c = sel % g;
      const x = cur.X.subarray(sel * D, (sel + 1) * D);
      const img = cur.can.getContext('2d').getImageData(c * p, r * p, p, p);
      const tile = canvasOf(p, p);
      tile.getContext('2d').putImageData(img, 0, 0);
      tile.className = 'vz-tile';
      tile.style.opacity = animate ? '0' : '1';
      place(G1, tile, TILE.x, TILE.y, TILE.s, TILE.s);
      const lbl = place(G1, h('div', { class: 'vz-tilelbl' },
        `patch ${sel} · row ${r}, column ${c}`, h('br'),
        `pixels ${r * p}–${r * p + p - 1} (y) × ${c * p}–${c * p + p - 1} (x)`), TILE.x, TILE.y - 46);
      lbl.style.opacity = animate ? '0' : '1';
      const xv = place(G1, mkVec(1, x, { orientation: 'h', cell: 26, gap: 3, head: 12, tail: 4, range: RANGE, kind: 'embed', tex: String.raw`\mathbf{x}\in\mathbb{R}^{${D}}` }), VEC_X, VEC_Y);
      xv.style.opacity = animate ? '0' : '1';
      const note = place(G1, h('div', { class: 'vz-k' }, `${D} numbers, first 12 and last 4 shown · colour ±${RANGE}`), VEC_X, VEC_Y + 84);
      note.style.opacity = animate ? '0' : '1';
      place(G1, h('div', { class: 'vz-eqs' },
        h('div', { class: 'vz-k' }, `patch ${sel} → one vector`),
        tx(String.raw`\mathbf{x}=\operatorname{flatten}(\mathbf{p}_{${sel}})\in\mathbb{R}^{${D}}`, 22),
        checkBadge(x.length, 3 * p * p, { label: 'length is 3 · P · P' }),
        h('div', { class: 'vz-note' }, 'Read row by row, R, G, B inside each pixel. A patch embedding is a Conv2d with kernel P and stride P: one linear map on this vector.')), 1180, 296, 340);
      // slots of the vector's cells, measured on the page (design units)
      const slots = [...xv.querySelectorAll('.k-cell')].map((n) => {
        const q = st.center(n);
        return { x: q.x - q.w / 2, y: q.y - q.h / 2, w: q.w, h: q.h };
      });
      // the channel colours the numbers will land on, and the pixel colours they start from
      const L = palette();
      const chan = new Uint8ClampedArray(D * 3);
      for (let k = 0; k < D; k++) {
        const t = clamp(x[k] / RANGE, -1, 1), col = L[Math.round((t + 1) * 127.5)];
        chan[k * 3] = col[0]; chan[k * 3 + 1] = col[1]; chan[k * 3 + 2] = col[2];
      }
      const fx = place(G1, canvasOf(1600 * FXR, FX.h * FXR), 0, FX.y, 1600, FX.h);
      fx.classList.add('vz-fx');
      fx.width = 1600 * FXR; fx.height = FX.h * FXR;
      const parts = { tile, xv, lbl, note, slots, chan, fx, img: img.data, p, D, g, r, c, animate };
      parts.start = () => runFlatten(parts);
      if (!animate) finishFlatten(parts, 1);
      return parts;
    }
    function captionFlatten() {
      const { P: p, D } = cur;
      return `Patch <b>${sel}</b> is ${p} × ${p} pixels in 3 colours. Read row by row, it is one vector of <b>${D}</b> numbers, normalised with CLIP’s mean and standard deviation.`;
    }
    // Phases: the patch zooms out of the image (A), its rows open up (B), each pixel splits into
    // three colour numbers that slide into one line (C), and the line squeezes onto the vector's
    // cells while the numbers that are not shown fold into the ellipsis (D).
    function runFlatten(parts) {
      const { tile, xv, lbl, note, slots, chan, fx, img, p, D, g, r, c } = parts;
      const cx = fx.getContext('2d');
      const cell = IMG.s / g, srcX = IMG.x + c * cell, srcY = IMG.y + r * cell, sp = cell / p, tp = TILE.s / p;
      const top = slots[0].y, slotH = slots[0].h, left = slots[0].x, width = slots[16].x + slots[16].w - slots[0].x, cw = width / D;
      const draw = (u) => {
        cx.setTransform(FXR, 0, 0, FXR, 0, -FX.y * FXR);
        cx.clearRect(0, FX.y, 1600, FX.h);
        const eA = easeIO(seg(u, 0, 0.26)), eB = easeIO(seg(u, 0.26, 0.46));
        const uC = seg(u, 0.46, 0.86), eD = easeIO(seg(u, 0.86, 1));
        for (let rr = 0; rr < p; rr++) {
          const e = easeIO(clamp((uC - rr * (0.5 / p)) / 0.5, 0, 1));
          for (let cc = 0; cc < p; cc++) {
            const o = (rr * p + cc) * 4;
            const x1 = lerp(srcX + cc * sp, TILE.x + cc * tp, eA);
            const y1 = lerp(srcY + rr * sp, TILE.y + rr * (tp + 10 * eB), eA);
            const w1 = lerp(sp, tp, eA);
            if (uC <= 0) { cx.fillStyle = `rgb(${img[o]},${img[o + 1]},${img[o + 2]})`; cx.fillRect(x1, y1, w1, w1); continue; }
            for (let ch = 0; ch < 3; ch++) {
              const k = (rr * p + cc) * 3 + ch;
              const xr = left + k * cw, wr = cw * 0.84;               // on the line
              const si = k < 12 ? k : k >= D - 4 ? 13 + (k - (D - 4)) : 12;
              const sl = slots[si];
              let x = lerp(x1, xr, e), y = lerp(y1, top, e), w = lerp(w1, wr, e), hh = lerp(w1, slotH, e);
              x = lerp(x, sl.x, eD); y = lerp(y, sl.y, eD); w = lerp(w, sl.w, eD); hh = lerp(hh, sl.h, eD);
              const mid = si === 12;
              const col = [0, 1, 2].map((q) => Math.round(lerp(img[o + q], chan[k * 3 + q], e)));
              cx.globalAlpha = mid ? 1 - eD : 1;
              cx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
              cx.fillRect(x, y, w, hh);
            }
          }
        }
        cx.globalAlpha = 1;
        finishFlatten(parts, eD);
      };
      if (reducedMotion()) { draw(1); cx.clearRect(0, FX.y, 1600, FX.h); return; }
      const t0 = performance.now(), dur = 2600;
      let id = 0;
      const frame = (now) => {
        const u = clamp((now - t0) / dur, 0, 1);
        draw(u);
        if (u < 1) id = requestAnimationFrame(frame);
        else cx.clearRect(0, FX.y, 1600, FX.h);
      };
      id = requestAnimationFrame(frame);
      track(() => { cancelAnimationFrame(id); cx.clearRect(0, FX.y, 1600, FX.h); });
    }
    function finishFlatten(parts, e) {
      parts.tile.style.opacity = String(e);
      parts.lbl.style.opacity = String(e);
      parts.xv.style.opacity = String(e);
      parts.note.style.opacity = String(e);
    }

    // ---------------------------------------------------- step 2: W_p, the positions, the matrix X
    function buildEmbed(G2, animate) {
      const { D, dv, N, g, X, Xv, Wp, Rp, Cp, wpMax } = cur;
      const r = Math.floor(sel / g), c = sel % g;
      const xs = X.subarray(sel * D, (sel + 1) * D), es = Xv.subarray(sel * dv, (sel + 1) * dv);
      const wc = canvasOf(D, dv);
      fillRows(wc, Wp, dv, D, wpMax, dv);
      wc.className = 'vz-tex';
      place(G2, wc, 110, 452, 400, Math.max(40, 400 * dv / D));
      place(G2, h('div', { class: 'vz-k' }, `W_p  ${dv} × ${D}  ·  illustration · colour ±${fmt(wpMax, 2)}`), 110, 420);
      place(G2, op('\\times', { kind: 'embed', size: 44 }), 548 - 22, 500 - 22);
      place(G2, mkVec(2, xs, { orientation: 'v', cell: 13, gap: 2, head: 12, tail: 4, range: RANGE, kind: 'embed', tex: String.raw`\mathbf{x}_i` }), 600, 292);
      place(G2, mkVec(2, es, { orientation: 'v', cell: 13, gap: 2, head: 12, tail: 4, range: RANGE, kind: 'embed', tex: String.raw`\mathbf{e}_i` }), 700, 292);
      place(G2, h('div', { class: 'vz-arrow' }), 748, 470, 36, 2);
      // X: one row per patch, in raster order
      const rowH = Math.min(EMB.w / dv, EMB.maxH / N);
      const xc = canvasOf(dv, N);
      xc.className = 'vz-tex';
      place(G2, xc, EMB.x, EMB.y, EMB.w, rowH * N);
      place(G2, h('div', { class: 'vz-k' }, `X  ${N} × ${dv}  ·  colour ±${RANGE}`), EMB.x, EMB.y - 26);
      // a label every few patch rows; the selected patch's (row, column) sits beside its own row
      const every = g * Math.max(1, Math.ceil(18 / (rowH * g)));
      const selY = EMB.y + sel * rowH;
      for (let rI = 0; rI < N; rI += every) {
        const y = EMB.y + rI * rowH;
        if (Math.abs(y - selY) < 36) continue;
        place(G2, h('div', { class: 'vz-rowlbl' }, `row ${Math.floor(rI / g)}`), EMB.x + EMB.w + 10, y - 6);
      }
      place(G2, h('div', { class: 'vz-rowlbl vz-selLbl' }, h('b', null, `patch ${sel}`), h('br'), `row ${r} · col ${c}`), EMB.x + EMB.w + 10, selY - 14);
      const band = place(G2, h('div', { class: 'vz-band' }), EMB.x - 3, EMB.y, EMB.w + 6, rowH + 2);
      const selRow = place(G2, h('div', { class: 'vz-selrow' }), EMB.x - 3, selY - 1, EMB.w + 6, rowH + 2);
      if (!animate) { fillRows(xc, Xv, N, dv, RANGE, N); selRow.style.opacity = '1'; }
      const k = Math.min(5, dv - 1);
      const expect = dotAt(X, sel * D, Wp, k * D, D) + Rp[r * dv + k] + Cp[c * dv + k];
      place(G2, h('div', { class: 'vz-eqs' },
        h('div', { class: 'vz-k' }, `patch ${sel} → one row`),
        tx(String.raw`\mathbf{e}_i=W_p\,\mathbf{x}_i+\mathbf{p}_i\in\mathbb{R}^{${dv}}`, 22),
        checkBadge(Xv[sel * dv + k], expect, { label: `equals W_p x_i + p_i, component ${k}` }),
        h('div', { class: 'vz-note' }, 'p_i is the row vector plus the column vector of patch i. W_p and both position tables are seeded random (illustration). A trained encoder learns them; the shapes are the same.')), 1180, 296, 340);
      return { xc, N, dv, rowH, band, selRow, Xv, animate, start: () => runFill({ xc, N, dv, rowH, band, selRow, Xv }) };
    }
    function captionEmbed() {
      const { D, dv, d, N } = cur;
      return `A random linear map <b>W_p</b> takes each ${D}-vector to <b>${dv}</b> numbers, and a 2-D position is added. The ${N} rows stack into X. The width is a choice: half the LLM width of ${d}.`;
    }
    function runFill({ xc, N, dv, rowH, band, selRow, Xv }) {
      const t0 = performance.now(), dur = 1600;           // the one slower fill (documented above)
      let id = 0;
      const frame = (now) => {
        const u = clamp((now - t0) / dur, 0, 1), upto = Math.round((1 - (1 - u) ** 3) * N);
        fillRows(xc, Xv, N, dv, RANGE, upto);
        band.style.top = `${EMB.y + upto * rowH - 1}px`;
        band.style.opacity = u < 1 ? '1' : '0';
        if (u < 1) id = requestAnimationFrame(frame);
        else selRow.style.opacity = '1';
      };
      id = requestAnimationFrame(frame);
      track(() => cancelAnimationFrame(id));
    }

    // ---------------------------------------------------- step 3: one sequence with the text
    function buildLLM(G3, animate) {
      const { N, d, dv, Hi, Xv, Wq } = cur;
      const cfg = ctx.cfg, T = ctx.state.ids.length;
      let emb = ctx.state.trace?.['emb.out'] ?? null;
      if (emb && emb.shape[1] !== d) emb = null;
      const ratio = emb ? rmsOf(emb.data) / rmsOf(Hi) : 1;
      const Hs = Hi.map((v) => v * ratio);                // projector output on the scale of the text rows
      const range = Math.max(maxAbs(Hs), emb ? maxAbs(emb.data) : 0);
      const selTok = clamp(ctx.state.sel.token ?? 0, 0, T - 1);
      const { w0, Ts } = windowOf(T, selTok), q = selTok - w0;      // the window of text positions, and the query's row in it
      const rule = textRule(cfg);
      const fades = [];

      // The mask. Image rows and columns: bidirectional within the image, masked from the text.
      // Text rows and columns: this model's own mask (cfg.mask), on the window of text positions.
      const w = MASK.img, c = Math.max(MASK.cell, Math.round(160 / Ts)), Wt = Ts * c, L = MASK.labelW, Lh = MASK.labelH;
      place(G3, h('div', { class: 'vz-k' }, 'VLM mask · illustration: this app’s LLM has no image input'), MASK.x, MASK.y - 30);
      const box = place(G3, h('div', { class: 'vz-mask' }), MASK.x, MASK.y, L + w + Wt, Lh + w + Wt);
      const blk = (cls, X, Y, W, H, kids) => {
        const b = h('div', { class: ['vz-mb', cls], style: { left: `${X}px`, top: `${Y}px`, width: `${W}px`, height: `${H}px` } }, kids);
        box.append(b); fades.push(b); return b;
      };
      blk('allow', L, Lh, w, w, [h('b', null, 'bidirectional'), h('small', null, `N = ${N} tokens`)]);
      blk('hide', L + w, Lh, Wt, w, [h('b', null, 'masked'), h('small', null, 'image cannot see text')]);
      blk('allow', L, Lh + w, w, Wt, [h('b', null, 'text → image'), h('small', null, 'all visible')]);
      box.append(h('div', { class: 'vz-ml', style: { left: '0px', top: '0px', width: `${L - 10}px`, textAlign: 'right' } }, 'query'));
      box.append(h('div', { class: 'vz-ml', style: { left: `${L}px`, top: '0px', width: `${w}px`, textAlign: 'center' } }, 'keys: image'));
      box.append(h('div', { class: 'vz-ml', style: { left: `${L + w}px`, top: '0px', width: `${Wt}px`, textAlign: 'center' } }, 'keys: text →'));
      box.append(h('div', { class: 'vz-ml', style: { left: '0px', top: `${Lh + w / 2 - 8}px`, width: `${L - 10}px`, textAlign: 'right' } }, 'image'));
      for (let j = 0; j < Ts; j++) box.append(h('div', { class: 'vz-ml', style: { left: `${L + w + j * c}px`, top: `${Lh - 20}px`, width: `${c}px`, textAlign: 'center' } }, ctx.label(w0 + j)));
      for (let i = 0; i < Ts; i++) {
        box.append(h('div', { class: 'vz-ml', style: { left: '0px', top: `${Lh + w + i * c + 3}px`, width: `${L - 10}px`, textAlign: 'right', color: i === q ? 'var(--fg)' : null } }, ctx.label(w0 + i)));
        for (let j = 0; j < Ts; j++) {
          const qi = w0 + i, kj = w0 + j, ok = maskAllows(cfg, qi, kj);
          const cl = h('div', { class: ['vz-mc', ok ? 'allow' : 'hide', i === q && 'sel'], title: `query ${qi} → key ${kj}: ${ok ? 'visible' : 'masked'}`, style: { left: `${L + w + j * c}px`, top: `${Lh + w + i * c}px`, width: `${c - 1}px`, height: `${c - 1}px` } });
          box.append(cl);
          fades.push(cl);
        }
      }
      place(G3, h('div', { class: 'vz-k' }, `text → text: ${rule.short} · positions ${w0}–${w0 + Ts - 1} of ${T}, query ${selTok}`), MASK.x, MASK.y + Lh + w + Wt + 8);

      // one sequence: the image rows, then the text rows (the same window as the mask)
      const tc = canvasOf(d, N);
      fillRows(tc, Hs, N, d, range, N);
      tc.className = 'vz-tex';
      place(G3, tc, SEQ.x, SEQ.y, SEQ.w, SEQ_IMG_H);
      place(G3, h('div', { class: 'vz-k' }, `image tokens · ${N} × ${d} · illustration`), SEQ.x, SEQ.y - 26);
      place(G3, h('div', { class: 'vz-k' }, T > Ts ? `text tokens ${w0}–${w0 + Ts - 1} of ${T} · emb.out` : `text tokens · ${T} × ${d} · emb.out`), SEQ.x, SEQ.y + SEQ_IMG_H + 56);
      const tm = heatmap({
        data: emb ? emb.data.subarray(w0 * d, (w0 + Ts) * d) : new Float64Array(Ts * d), rows: Ts, cols: d,
        mode: 'diverging', showValues: false, range, cell: SEQ.w / d, cellH: 20, traceKey: 'emb.out',
        rowLabels: Array.from({ length: Ts }, (_, i) => ctx.label(w0 + i)), highlight: { rows: [q] },
        ariaLabel: 'text token embeddings, emb.out',
      });
      heat = tm;
      tm.el.style.maxWidth = 'none';                     // the kit's max-width:100% would resolve against the zero-width group
      place(G3, tm.el, SEQ.x, SEQ.y + SEQ_IMG_H + 80, 400);
      place(G3, kvLine('one sequence: image + text tokens', `${N} + ${T} = ${N + T} tokens of ${d}`), SEQ.x, SEQ.y + SEQ_IMG_H + 80 + 176);

      // the side: one equation, its check, one note
      place(G3, h('div', { class: 'vz-eqs' },
        h('div', { class: 'vz-k' }, 'one sequence'),
        tx(String.raw`\mathbf{x}_0=\begin{bmatrix}\hat{\mathbf{X}}\,W_{\text{proj}}^{\top}\\ \mathbf{E}_{\text{text}}\end{bmatrix}`, 22),
        checkBadge(Hi[sel * d + 3], dotAt(Xv, sel * dv, Wq, 3 * dv, dv), { label: 'equals the dot product, row 3' }),
        h('div', { class: 'vz-note' }, emb
          ? `Text rows are the model’s own emb.out, the x₀ of model.py. The projector is random, rescaled to the text rows. Both share one colour scale, ±${fmt(range, 2)}.`
          : 'Run a forward pass to see the text rows.')), 1180, 296, 340);
      return { fades, animate, start: () => fadeIn(fades) };
    }
    function captionLLM() {
      const { d, dv } = cur;
      return `A projector maps the ${dv}-wide rows to the LLM width <b>${d}</b>. Image tokens see each other both ways. Each text token sees every image token and, under this model’s mask, ${textRule(ctx.cfg).phrase}.`;
    }
    function fadeIn(els) {
      els.forEach((el, i) => {
        if (reducedMotion()) return;
        const a = el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 480, delay: 60 + i * 26, easing: MOTION.easeOut, fill: 'backwards' });
        track(() => a.cancel());
      });
    }

    // ---------------------------------------------------- step 4: how many tokens
    const COUNTS = (() => {
      const shot = (1920 * 1080) / (14 * 14);
      return [
        { label: 'ViT-L/14 · 224 px', sub: '16 × 16 patches', v: (224 / 14) ** 2 },
        { label: 'LLaVA-1.5 · 336 px', sub: '24 × 24 patches, CLS dropped', v: (336 / 14) ** 2 },
        { label: '1080p screenshot · 14 px', sub: '137.1 × 77.1 patches', v: shot, hot: true },
        { label: '2 × 2 merge (pixel shuffle)', sub: '÷ 4 tokens', v: shot / 4 },
        { label: '4 × 4 merge', sub: '÷ 16 tokens', v: shot / 16 },
      ];
    })();
    function buildCount(G4, animate) {
      const max = COUNTS[2].v;
      const rows = COUNTS.map((r, i) => {
        const fill = h('div', { class: 'vz-fill', style: { width: animate ? '0%' : `${(r.v / max) * 100}%`, background: 'var(--c-embed)' } });
        const val = h('div', { class: 'vz-val' }, animate ? '0' : fmtInt(r.v));
        place(G4, h('div', { class: ['vz-row', r.hot && 'hot'] },
          h('div', { class: 'vz-rl' }, h('b', null, r.label), h('small', null, r.sub)),
          h('div', { class: 'vz-track' }, fill), val), BARS.x, BARS.y0 + i * BARS.dy, 1030);
        return { r, fill, val };
      });
      const exact = COUNTS[2].v, kv = 2 * 32 * 4096 * 2;            // K and V, 32 layers, d = 4096, fp16: bytes per token
      const seqN = 20 * exact;
      const n1 = Math.round(exact), n2 = Math.round(exact / 4);
      const att1 = exact * exact, att2 = (exact / 4) * (exact / 4);
      place(G4, h('div', { class: 'vz-note', style: { maxWidth: '1030px' }, html:
        `Attention work per head and layer grows with N<sup>2</sup>, so the 2 × 2 merge cuts it ${Math.round(att1 / att2)}×: ${texHTML(`${grp(n1)}^2\\approx${num(att1, 2)}`)} entries before, ${texHTML(`${grp(n2)}^2\\approx${num(att2, 1)}`)} after.` }),
        BARS.x, BARS.y0 + 5 * BARS.dy + 14, 1030);
      place(G4, h('div', { class: 'vz-note', style: { maxWidth: '1030px' } },
        `If every screenshot stays in context, a 20-step episode holds about ${Math.round(seqN / 1000)}k image tokens, or ${Math.round(seqN / 4000)}k after the 2 × 2 merge.`), BARS.x, BARS.y0 + 5 * BARS.dy + 72, 1030);
      const gib = (n) => (n * kv / 2 ** 30).toFixed(1);
      place(G4, h('div', { class: 'vz-eqs' },
        h('div', { class: 'vz-k' }, 'one 1080p screenshot'),
        tx(String.raw`N=\frac{1920\cdot1080}{14^2}\approx${grp(exact)}`, 24),
        h('div', { class: 'vz-note' }, `Whole patches, after resizing to 1918 × 1078 px: ${Math.floor(1920 / 14)} × ${Math.floor(1080 / 14)} = ${fmtInt(Math.floor(1920 / 14) * Math.floor(1080 / 14))}.`),
        h('div', { class: 'vz-card' },
          h('div', { class: 'vz-k' }, 'KV cache · Llama-2-7B shape, fp16'),
          kvLine('per token: 2 × 32 × 4096 × 2 B', `${(kv / 2 ** 20).toFixed(1)} MiB`, 312),
          kvLine(`one screenshot, ${fmtInt(exact)} tokens`, `≈ ${gib(exact)} GiB`, 312),
          kvLine(`after 2 × 2 merge, ${fmtInt(exact / 4)} tokens`, `≈ ${gib(exact / 4)} GiB`, 312))), 1180, 296, 340);
      return {
        rows, animate,
        start: () => {
          requestAnimationFrame(() => rows.forEach(({ r, fill }) => { fill.style.width = `${(r.v / max) * 100}%`; }));
          rows.forEach(({ r, val }, i) => {
            const t0 = performance.now() + i * 90, dur = MOTION.move;
            let id = 0;
            const frame = (now) => {
              const u = clamp((now - t0) / dur, 0, 1), e = 1 - (1 - u) ** 3;
              val.textContent = fmtInt(r.v * e);
              if (u < 1) id = requestAnimationFrame(frame);
            };
            id = requestAnimationFrame(frame);
            track(() => cancelAnimationFrame(id));
          });
        },
      };
    }
    function captionCount() {
      return `A 1080p screenshot at 14 px is about <b>${(COUNTS[2].v / 1000).toFixed(1)}k</b> patches. Merging 2 × 2 patches into one token cuts the tokens 4× and the attention work 16×.`;
    }

    // ---------------------------------------------------- rendering and steps
    function render({ animate = false } = {}) {
      cancelAll();
      animate = animate && !reducedMotion();            // reduced motion: jump to each end state
      paintControls();
      picker.el.classList.toggle('on', step === 3);
      strip.classList.toggle('on', step <= 3);
      G.forEach((g, i) => g.classList.toggle('on', i === step && !!cur));
      wrap.classList.toggle('on', step <= 1 && !!cur);
      if (!cur) return;                                 // still computing: the busy caption says so
      cellEls.forEach((c, i) => c.classList.toggle('sel', i === sel));
      heat?.destroy();
      heat = null;
      releaseVecs(step);
      G[step].replaceChildren();
      const parts = BUILD[step](G[step], animate);
      cap.set(CAP[step]());
      if (animate && step === 0) wipe();
      if (animate && parts?.start) parts.start();
    }
    function wipe() {
      const a = cellsEl.animate([{ clipPath: 'inset(0 100% 0 0)' }, { clipPath: 'inset(0 0 0 0)' }], { duration: 1100, easing: MOTION.ease });
      const b = scan.animate([
        { transform: 'translateX(0px)', opacity: 1 },
        { transform: `translateX(${IMG.s - 3}px)`, opacity: 1, offset: 0.92 },
        { transform: `translateX(${IMG.s - 3}px)`, opacity: 0 },
      ], { duration: 1100, easing: MOTION.ease });
      track(() => { a.cancel(); b.cancel(); });
    }
    function goto(i, { animate = true } = {}) {
      step = i;
      render({ animate });
    }
    function onTrace() {
      const changed = sync();
      if (changed || step === 3) render({ animate: changed && step === 0 });
    }
    function onSelect() {
      if (step === 3) render();
    }

    // ---------------------------------------------------- boot
    sync(true);
    ctx.setSteps(STEPS, (i) => goto(i));
    goto(0, { animate: true });
    // First view only: walk the steps, each one given the time its animation needs.
    let auto = null;
    const advance = (i) => {
      if (stopped || i >= STEPS.length) return;
      ctx.setStep(i);
      auto = setTimeout(() => advance(i + 1), STEP_MS[i]);
    };
    auto = setTimeout(() => advance(1), STEP_MS[0]);
    el.addEventListener('pointerdown', () => clearTimeout(auto), { once: true });

    return {
      destroy() {
        stopped = true;
        clearTimeout(auto);
        cancelJob();
        cancelAll();
        heat?.destroy();
        groupVecs.forEach((list) => list.forEach((v) => v.destroy?.()));
        offs.forEach((f) => { try { f(); } catch { /* ignore */ } });
        fileIn.remove();
        hideTip();
        st.destroy();
      },
    };
  },
};

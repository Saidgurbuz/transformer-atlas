// Positional encoding: how the model knows where each token is.
//
// The scheme in use (cfg.pos_encoding) is the hero. Its numbers come from this model's own forward
// pass (ctx.state.trace) or from engine functions applied to its weights. The segmented control
// previews the other schemes with the same formulas: previews are illustrations and say so, except
// None, which is a real run of a clone with positions removed. Three steps per scheme.
// Layout: the controls band (y 196) holds the query tokens on the left and the scheme on the right.
// The side column holds the head chips (RoPE and ALiBi), the step name, one equation, the status and
// one or two sentences. Maths: src/core/math/positions.js and embed.js. model.py: sinusoidal_table,
// rope_angles, apply_rope, alibi_bias.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, vec, mat, op, fly, pulse, useStyle, ZONES, MOTION } from '../kit.js';
import { h, s, fmt, texHTML, num, heatmap, barChart, segmented, button, reducedMotion, clamp, diverging, onThemeChange } from '../../ui/index.js';
import { infer, cloneModel, createModel, ropeAngles, alibiSlopes, alibiBias, maskAllows, rope, Tensor, noGrad, mulberry32 } from '../../engine/index.js';

useStyle('positional', `
.sp-hero { position: relative; width: 1030px; height: 500px; }
.sp-layer { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
.sp-layer.sp-out { pointer-events: none; }
.sp-picker { width: 1032px; }
.sp-picker .k-picker { flex-wrap: wrap; row-gap: 4px; max-width: 960px; }
.sp-picker.dense .k-pick { height: 28px; }
.sp-scheme { width: 336px; display: flex; justify-content: flex-end; }
.sp-seg { width: max-content; max-width: none; flex-wrap: nowrap; }
.sp-seg button { white-space: nowrap; }
.sp-side { width: 340px; display: grid; gap: 12px; align-content: start; }
.sp-side > * { min-width: 0; max-width: 340px; }
.sp-side-body { display: grid; gap: 12px; align-content: start; }
.sp-heads { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; }
.sp-heads-label { flex-basis: 100%; font: 600 13px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 4px; }
.sp-heads .k-pick-head { width: 36px; height: 26px; font-size: 12px; }
.sp-status { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.sp-flag { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px; font: 600 13px/1 var(--font-mono); color: var(--fg-3); border: 1px dashed var(--line-strong); white-space: nowrap; }
.sp-flag.live { color: var(--c-pos); border: 1px solid color-mix(in srgb, var(--c-pos) 55%, transparent); background: color-mix(in srgb, var(--c-pos) 10%, transparent); }
.sp-eq { display: grid; gap: 10px; align-content: start; }
.sp-eq-main { font-size: 22px; }
.sp-eq-main .katex-display { margin: 0.2em 0; }
.sp-eq-live { display: grid; gap: 6px; font-size: 16px; padding: 12px 14px; border-radius: 14px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); }
.sp-note { font-size: 15px; line-height: 1.5; color: var(--fg-3); }
.sp-plate { position: relative; padding: 12px 14px 14px; border-radius: 14px; background: color-mix(in srgb, var(--surface) 74%, transparent); border: 1px solid color-mix(in srgb, var(--c-pos) 26%, var(--line)); box-shadow: 0 16px 36px rgba(0, 0, 0, .14); }
.sp-plate-t { font: 600 15px/1.2 var(--font-mono); color: var(--fg-2); margin-bottom: 8px; white-space: nowrap; }
.sp-thumbs { display: grid; gap: 12px 14px; justify-items: center; align-content: center; }
.sp-thumb { display: grid; justify-items: center; gap: 6px; padding: 6px; border-radius: 12px; border: 1px solid transparent; background: transparent; cursor: pointer; font: 600 15px/1 var(--font-mono); color: var(--fg-3); transition: border-color .25s, background-color .25s, color .25s; }
.sp-thumb:hover { background: color-mix(in srgb, var(--surface) 70%, transparent); color: var(--fg-2); }
.sp-thumb.on { border-color: var(--accent); color: var(--fg); }
.sp-thumb canvas { display: block; border-radius: 4px; }
.sp-abs { position: absolute; }
.sp-fig { position: absolute; left: 0; top: 0; overflow: visible; }
.sp-rowlab { position: absolute; left: 0; text-align: right; transform: translateY(-50%); font-size: 16px; color: var(--fg-2); white-space: nowrap; }
.sp-tick { position: absolute; transform: translateX(-50%); font: 500 15px/1 var(--font-mono); color: var(--fg-3); }
.sp-legend { position: absolute; left: 0; top: 0; display: flex; gap: 16px; font: 600 15px/1 var(--font-mono); }
.sp-scan, .sp-dot, .sp-hand { transition: transform .32s cubic-bezier(.65,0,.35,1); }
.sp-arc { fill: none; stroke: var(--c-pos); stroke-width: 2; opacity: .85; }
.sp-read { display: grid; gap: 10px; font-size: 20px; color: var(--fg-2); }
.sp-big { font: 600 34px/1 var(--font-mono); color: var(--fg); }
.sp-col { display: grid; justify-items: center; gap: 12px; }
.sp-stack { position: absolute; left: 0; right: 0; top: 96px; display: grid; justify-items: center; gap: 4px; }
.sp-table { width: 100%; border-collapse: collapse; font: 500 18px/1.2 var(--font-mono); color: var(--fg); }
.sp-table th { font: 600 14px/1.2 var(--font-mono); color: var(--fg-3); text-align: right; padding: 8px 12px; border-bottom: 1px solid var(--line-strong); }
.sp-table td { text-align: right; padding: 11px 12px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; transition: background .3s; }
.sp-table tbody tr { cursor: pointer; }
.sp-table tr.on td { background: var(--accent-soft); }
.sp-bars { position: absolute; left: 0; right: 0; bottom: 36px; display: flex; justify-content: center; align-items: flex-end; }
.sp-slope { display: grid; justify-items: center; gap: 6px; }
.sp-slope-m { font-size: 16px; line-height: 1.2; }
.sp-slope-v { font: 500 14px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.sp-bar { width: 84px; border-radius: 12px 12px 5px 5px; background: linear-gradient(180deg, color-mix(in srgb, var(--c-attn) 90%, white), var(--c-attn)); box-shadow: 0 12px 24px rgba(0, 0, 0, .18); transition: height .6s cubic-bezier(.65,0,.35,1); }
.sp-tile { position: absolute; box-sizing: border-box; display: grid; place-items: center; align-content: center; gap: 4px; height: 50px; border-radius: 10px; background: var(--surface); border: 1px solid color-mix(in srgb, var(--c-pos) 40%, var(--line)); box-shadow: 0 6px 14px rgba(0, 0, 0, .12); font: 600 16px/1 var(--font-mono); color: var(--fg); }
.sp-tile small { font: 500 12px/1 var(--font-mono); color: var(--fg-3); }
.sp-tag { position: absolute; font: 600 13px/1 var(--font-mono); color: var(--fg-3); letter-spacing: .04em; text-transform: uppercase; white-space: nowrap; }
`);

// ------------------------------------------------------------------ layout (design units)
const HERO = { x: 110, y: 260, w: 1030, h: 500 };     // the one visual; ZONES.side is the column to its right
const CTRL_Y = 196;                                   // controls band: query tokens on the left, the scheme on the right
const XR = 1490;                                      // right edge of the controls band and of the side column
const DENSE_T = 48;                                   // above this many tokens the tile row wraps to two rows
const SCHEMES = ['learned', 'sinusoidal', 'rope', 'alibi', 'none'];
const NAME = { learned: 'Learned', sinusoidal: 'Sinusoidal', rope: 'RoPE', alibi: 'ALiBi', none: 'None' };

// ------------------------------------------------------------------ content
const ONE = {
  learned: 'adds one trained vector per position to each token embedding.',
  sinusoidal: 'adds fixed sines and cosines of the position to each token embedding.',
  rope: 'turns each query and key by an angle that grows with the position, inside attention.',
  alibi: 'subtracts a penalty proportional to distance from the attention scores.',
  none: 'removes the position signal, so attention alone cannot see order.',
};
const NOTE = {
  learned: 'Illustration: a fresh random table, not this model’s trained rows.',
  sinusoidal: 'Illustration: the same formula, not this model’s weights.',
  rope: 'Illustration: RoPE applied to this model’s queries and keys.',
  alibi: 'Illustration: the same formula on this model’s scores.',
  none: 'A clone of this model with positions removed. A real run, not an illustration.',
};
const STEPS = {
  learned: ['The table', 'Added to a token', 'Compared pairwise'],
  sinusoidal: ['The waves', 'Read one position', 'Similarity by distance'],
  rope: ['One pair, one clock', 'Fast and slow pairs', 'Only the offset counts'],
  alibi: ['One slope per head', 'A distance penalty', 'Added to the scores'],
  none: ['Shuffle the input', 'Same rows, moved', 'The causal mask sees order'],
};
// A caption is a string, or a function of the scene state when the text depends on the config.
const CAPS = {
  learned: [
    (S) => (S.live
      ? 'Each position t has its own trained row of W_P. The highlighted row is the position of the selected token.'
      : 'An illustration: a random table with the shape of W_P. The highlighted row is the position of the selected token.'),
    (S) => (S.live
      ? 'The token’s row of W_E and its position row are added element by element. The sum is the input x_t to the first block.'
      : 'The token’s row of W_E is this model’s. The position row is random here, so the sum is an illustration.'),
    (S) => (S.live
      ? 'Each cell is the cosine of two position vectors. The diagonal is 1 by definition; the rest shows which positions the model treats as alike.'
      : 'An illustration: the cosines of a random table. Without training, the pattern is noise.'),
  ],
  sinusoidal: [
    'Each pair of dimensions (2p, 2p+1) holds a sine and a cosine of t·ω_p. Fast pairs wrap every few tokens; slow pairs barely move across this input.',
    'The scan line reads the whole vector PE[t] at each position: one sine and one cosine per pair. That vector is added to the token embedding.',
    'PE[s]·PE[t] is a sum of cosines of (s − t)·ω_p. It depends only on the distance: it falls off overall, with ripples from the fast pairs.',
  ],
  rope: [
    'The query pair (q₀, q₁) of this token is a 2-D vector. At position t it is turned by t·θ₀; with θ₀ = 1 the hand turns one radian per token.',
    'The fast pair and the slow pair turn at different speeds. Each pair of dimensions has its own θ, so positions are encoded at many scales.',
    'Turn the query to position m and the key to m + Δ. The angle between them is the same for every m, so q′·k′ depends only on Δ.',
  ],
  alibi: [
    'Head h has slope m_h = 2^(−8(h+1)/H). Each slope is a fixed fraction of the previous one, so head 0 is the steepest.',
    'The bias −m_h·|i − j| is zero on the diagonal and falls linearly with distance. Each head has its own slope, so its own range.',
    'The bias is added to the scores (scaled when attn_scale is on), before the mask and softmax. The penalty is linear in the score, so each extra token of distance multiplies the unnormalised weight by e^(−m_h): distant keys fall off exponentially.',
  ],
  none: [
    'Shuffle the input tokens. Without a position signal the model records only which token each one is, not where it was.',
    'Run the model on both orders, with no position signal and no causal mask (a clone, so the weights are untouched). Each output row equals the same token’s row, moved.',
    'Keep the causal mask and the rows change: row r now sees a different set of earlier tokens. The mask carries order on its own.',
  ],
};
const capText = (S, i) => { const c = CAPS[S.shown][i]; return typeof c === 'function' ? c(S) : c; };
const SHUFFLE_SEED = 2024;

// ------------------------------------------------------------------ numbers and maths
const labelsOf = (T) => Array.from({ length: T }, (_, i) => String(i));
const every = (T, max) => Math.max(1, Math.ceil(T / max));
const sparseLabels = (T, max = 16) => { const k = every(T, max); return Array.from({ length: T }, (_, i) => (i % k === 0 ? String(i) : '')); };
const absMax = (arrs) => { let m = 0; for (const a of arrs) for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m || 1; };
const maxAbsOf = (a) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };
const maxDiff = (a, b) => { let m = 0; const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const omega = (p, d) => Math.pow(10000, (-2 * p) / d);                // sinusoidal ω_p = 10000^(−2p/d)
const thetaOf = (p, dh, base) => Math.pow(base, (-2 * p) / dh);        // RoPE θ_p = base^(−2p/dh)
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));             // into (−π, π]
const rowOf = (flat, r, n) => flat.subarray(r * n, (r + 1) * n);
const headRow = (t, hd, r, T, n) => { const o = (hd * T + r) * n; return t.data.subarray(o, o + n); };   // [H, T, n]
const headMat = (t, hd, T) => t.data.subarray(hd * T * T, (hd + 1) * T * T);                              // [H, T, T]
const rotPair = (x0, x1, phi) => [x0 * Math.cos(phi) - x1 * Math.sin(phi), x0 * Math.sin(phi) + x1 * Math.cos(phi)];
const dotOf = (a, b) => { let acc = 0; for (let i = 0; i < a.length; i++) acc += a[i] * b[i]; return acc; };
const vals3 = (arr) => Array.from(arr.slice(0, 3), (v) => num(v, 3)).join(',\\,');
const pairNorm = (v, p) => Math.hypot(v[2 * p], v[2 * p + 1]);
const maxPairNorm = (v) => { let m = 0; for (let p = 0; p < v.length / 2; p++) m = Math.max(m, pairNorm(v, p)); return m || 1; };
// Values under 1e-9 are rounding noise: they read as below 10^-9, never as a result (TeX and plain text).
const nz = (v, digits = 2) => (Math.abs(v) < 1e-9 ? '< 10^{-9}' : num(v, digits));

// The pairs of dimensions drawn as waves: pair 0, then about 3/32, 1/4 and 1/2 of the way up.
function pairsShown(d) {
  const n = d / 2;
  return [...new Set([0, Math.round((n * 3) / 32), Math.round(n / 4), Math.round(n / 2)])].filter((p) => p < n).map((p) => ({ p, w: omega(p, d) }));
}
// A slow pair for the RoPE comparison: θ = 0.1 for head size 16 (one turn every ~63 tokens).
const slowPair = (dh) => Math.min(dh / 2 - 1, Math.max(1, Math.round(dh / 8)));

function unitRows(P, T, d) {
  const out = new Float64Array(T * d);
  for (let r = 0; r < T; r++) {
    let n2 = 0;
    for (let c = 0; c < d; c++) n2 += P[r * d + c] * P[r * d + c];
    const inv = 1 / (Math.sqrt(n2) || 1);
    for (let c = 0; c < d; c++) out[r * d + c] = P[r * d + c] * inv;
  }
  return out;
}

// A·Bᵀ for two [T, d] matrices, as a [T, T] array.
function gram(A, B, T, d) {
  const out = new Float64Array(T * T);
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    let acc = 0;
    for (let c = 0; c < d; c++) acc += A[i * d + c] * B[j * d + c];
    out[i * T + j] = acc;
  }
  return out;
}

// Seeded shuffle of 0..T-1 (never the identity for T ≥ 2).
function permute(T, seed) {
  const rnd = mulberry32(seed), p = Array.from({ length: T }, (_, i) => i);
  for (let i = T - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
  if (T > 1 && p.every((v, i) => v === i)) [p[0], p[1]] = [p[1], p[0]];
  return p;
}

// RoPE of one vector at an absolute position, through the engine's own rope() and angle tables.
function ropeAt(v, pos, base) {
  const dh = v.length, { cos, sin } = ropeAngles(1, dh, base, pos);
  return noGrad(() => rope(Tensor.from(v, [1, dh]), cos, sin)).data;
}

// Sine and cosine of one position, computed here from omega_p. The engine's table is checked against it.
function sinRow(t, d) {
  const out = new Float64Array(d);
  for (let p = 0; p < d / 2; p++) { const a = t * omega(p, d); out[2 * p] = Math.sin(a); out[2 * p + 1] = Math.cos(a); }
  return out;
}
function sinOwn(T, d) {
  const P = new Float64Array(T * d);
  for (let t = 0; t < T; t++) P.set(sinRow(t, d), t * d);
  return P;
}

// −m_h·|i − j| for one head, from the formula in this scene (not from the engine's alibiBias).
function ownBias(T, H, hd) {
  const mm = Math.pow(2, (-8 * (hd + 1)) / H), B = new Float64Array(T * T);
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) B[i * T + j] = -mm * Math.abs(i - j);
  return B;
}

// What a fresh learned table looks like before training (the engine's own initialisation).
let freshKey = '', freshData = null;
function freshLearned(S) {
  const c = S.cfg, key = [c.block_size, S.d, c.vocab_size, c.n_layer, c.n_head, c.n_kv_head, c.d_ff, S.seed].join('|');
  if (key !== freshKey) {
    freshKey = key;
    freshData = createModel({ ...c, pos_encoding: 'learned' }, S.seed).params['wpe.weight'].data;
  }
  return freshData;
}
const learnedTable = (S) => (S.live ? S.trace['emb.pos'].data : freshLearned(S).subarray(0, S.T * S.d));
const sinTable = (S) => (S.live ? S.trace['emb.pos'].data : sinOwn(S.T, S.d));

// Sizes: matrices shrink with the sequence. Hero matrices stay inside the hero's 500 units of height.
const cellW = (d) => clamp(Math.floor(Math.min(13, 900 / d)), 4, 13);
const cellH = (T) => clamp(Math.floor(Math.min(13, 380 / T)), 4, 13);
const cellHero = (T) => clamp(Math.floor(430 / T), 6, 26);
// Centre one visual across the hero.
const centred = (node, top = 0) => h('div', { class: 'sp-abs', style: { left: '0', top: `${top}px`, width: `${HERO.w}px`, display: 'flex', justifyContent: 'center' } }, node);
// Two visuals side by side, centred in the hero.
const pair = (...nodes) => h('div', { class: 'sp-abs', style: { left: '0', top: '0', width: `${HERO.w}px`, height: `${HERO.h}px`, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '36px' } }, ...nodes);

// ------------------------------------------------------------------ motion helpers
// Walks the scan from position 0 to the selected token, one integer position at a time.
// Holds at the selected token; reduced motion jumps to the end state.
function walk(S, onIndex) {
  const q = S.q;
  if (S.T < 2 || q === 0 || reducedMotion()) { onIndex(q, true); return; }
  const stepMs = clamp(Math.round(1400 / q), 45, 160);
  let t = 0, timer = null;
  const tick = () => {
    onIndex(t, t === q);
    if (t < q) { t += 1; timer = setTimeout(tick, stepMs); }
  };
  tick();
  S.cleanup(() => clearTimeout(timer));
}

// A traced value is only true at the traced position: data-trace is on while the shown values are that position.
const setTrace = (el, key) => { if (key) el.dataset.trace = key; else delete el.dataset.trace; };

// ------------------------------------------------------------------ shared pieces
const plate = (node, title) => h('div', { class: 'sp-plate' }, title ? h('div', { class: 'sp-plate-t', html: title }) : null, node);

function eqBlock(main, lines = [], check = null) {
  return h('div', { class: 'sp-eq' },
    h('div', { class: 'sp-eq-main', html: texHTML(main, { display: true }) }),
    lines.length ? h('div', { class: 'sp-eq-live' }, lines.map((l) => h('div', { html: texHTML(l, { display: false }) }))) : null,
    check);
}

function verdict(ok, text) {
  return h('span', { class: ['k-check', ok ? 'ok' : 'bad'] }, ok ? `✓ ${text}` : `✗ ${text}`);
}

// A canvas heatmap that tells the atlas which cell is under the pointer: data-cell="r,c,band" on the
// canvas gives the hover card that cell's live value, and the right-click menu its maths and code.
function heat(S, o, { trace = null, band = 0 } = {}) {
  let canvas = null;
  const hm = heatmap({
    ...o, traceKey: trace, band,
    onHover: (r, c, v) => { if (canvas) canvas.dataset.cell = `${r},${c},${band}`; o.onHover?.(r, c, v); },
    onLeave: () => { if (canvas) delete canvas.dataset.cell; o.onLeave?.(); },
  });
  canvas = hm.canvas;
  S.cleanup(() => hm.destroy());
  return hm;
}

// A small multiple: the whole T × T matrix drawn into a square of `size` units. It is a shrunk copy of
// the real matrix on the focus plate's colour scale, so it stays readable at any sequence length.
function thumbMap(M, T, range, size) {
  const dpr = window.devicePixelRatio || 1;
  const cv = h('canvas', { 'aria-hidden': 'true' });
  cv.width = Math.round(size * dpr);
  cv.height = Math.round(size * dpr);
  cv.style.width = `${size}px`;
  cv.style.height = `${size}px`;
  const paint = () => {
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size, size);
    const c = size / T;
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      g.fillStyle = diverging(M[i * T + j] / range);
      g.fillRect(j * c, i * c, c + 0.5, c + 0.5);
    }
  };
  paint();
  return { canvas: cv, paint };
}

// A matrix as mat (DOM cells, up to ~1600) or heatmap (canvas, larger). Both can highlight a row.
function table(S, data, rows, cols, o) {
  if (rows * cols <= 1600) {
    const m = mat(data, rows, cols, {
      cell: Math.min(o.cellW, o.cellH), gap: 1, kind: o.kind, mode: o.mode, range: o.range, rowLabels: o.rowLabels,
      trace: o.trace, cellOf: o.trace ? (r, c) => `${r},${c},0` : undefined,
    });
    S.cleanup(() => m.destroy?.());
    return { el: m, hi: (r) => m.highlightRow(r) };
  }
  const hm = heat(S, {
    data, rows, cols, mode: o.mode, kind: o.kind, range: o.range, cell: o.cellW, cellH: o.cellH,
    rowLabels: o.rowLabels, highlight: { rows: [o.hi] }, ariaLabel: o.ariaLabel, maxWidth: 1000,
  }, { trace: o.trace, band: 0 });
  return { el: hm.el, hi: (r) => hm.update({ highlight: { rows: [r] } }) };
}

// ------------------------------------------------------------------ clock (2-D planes for RoPE)
// SVG arc from local angle a0 to a1 at radius ra (math convention, y up).
function arcPath(ra, a0, a1) {
  let d = '';
  for (let k = 0; k <= 24; k++) {
    const psi = a0 + (a1 - a0) * (k / 24);
    d += `${k ? 'L' : 'M'}${(ra * Math.cos(psi)).toFixed(2)},${(-ra * Math.sin(psi)).toFixed(2)}`;
  }
  return d;
}

function buildClock(size) {
  const c = size / 2, R = c - 22;
  const svg = s('svg', { class: 'sp-clock', width: size, height: size, viewBox: `0 0 ${size} ${size}`, style: 'overflow: visible' });
  svg.append(s('circle', { cx: c, cy: c, r: R, fill: 'var(--surface)', stroke: 'var(--line-strong)', 'stroke-width': 1.2 }));
  svg.append(s('line', { x1: c - R, y1: c, x2: c + R, y2: c, stroke: 'var(--line)', 'stroke-dasharray': '3 4' }));
  svg.append(s('line', { x1: c, y1: c - R, x2: c, y2: c + R, stroke: 'var(--line)', 'stroke-dasharray': '3 4' }));
  svg.append(s('circle', { cx: c, cy: c, r: 4.5, style: 'fill: var(--fg)' }));
  return {
    el: svg, R, c,
    // A hand from the centre. len in px. Rotation is a CSS transform, so steps glide.
    hand({ len, color, ghost = false, width = 3.5, dash = null, arc = false }) {
      const g = s('g', { class: 'sp-hand' });
      let path = null;
      if (arc) { path = s('path', { class: 'sp-arc' }); g.append(path); }
      g.append(s('line', { x1: 0, y1: 0, x2: len, y2: 0, style: `stroke: ${color}`, 'stroke-width': width, 'stroke-linecap': 'round', 'stroke-dasharray': dash }));
      g.append(s('circle', { cx: len, cy: 0, r: ghost ? 3 : 5, style: `fill: ${color}` }));
      svg.append(g);
      return {
        setAngle(phi) { g.style.transform = `translate(${c}px, ${c}px) rotate(${(-phi * 180 / Math.PI).toFixed(3)}deg)`; },
        setArc(a0, a1) { if (path) path.setAttribute('d', arcPath(R * 0.3, a0, a1)); },
      };
    },
  };
}

// ------------------------------------------------------------------ learned
function learned0(S) {
  const { T, d, q } = S, P = learnedTable(S), row = rowOf(P, q, d);
  const tbl = table(S, P, T, d, {
    cellW: cellW(d), cellH: cellH(T), range: absMax([P]), mode: 'diverging', kind: 'pos', rowLabels: sparseLabels(T, 32),
    trace: S.live ? 'emb.pos' : null, hi: q, ariaLabel: 'position table',
  });
  S.hero.append(centred(plate(tbl.el, texHTML('W_P \\in \\R^{T \\times d}')), 0));
  tbl.hi(q);
  const check = S.live
    ? checkBadge(maxDiff(S.model.params['wpe.weight'].data.subarray(0, T * d), S.trace['emb.pos'].data), 0, { label: 'rows of W_P = traced emb.pos' })
    : null;
  S.side.append(eqBlock('\\Pe_t = W_P[t] \\in \\R^{d}', [
    `\\Pe_{${q}} = \\big(${vals3(row)},\\ \\dots\\big)`,
    `t = ${q}, \\quad d = ${d}`,
  ], check));
  S.note(`Query “${S.label(q)}” at position ${q}: row ${q} of the table.`);
}

function learned1(S) {
  const { d, q } = S, P = learnedTable(S);
  const tok = rowOf(S.trace['emb.tok'].data, q, d), pos = rowOf(P, q, d);
  const x = Float64Array.from(tok, (v, i) => v + pos[i]);
  const R = absMax([tok, pos, x]);
  const big = d > 96, cell = clamp(Math.floor(Math.min(13, 900 / d)), 4, 13);
  const strip = (vals, tex, trace) => {
    const v = vec(vals, {
      orientation: 'h', cell, gap: 1, head: big ? 12 : d, tail: big ? 4 : 0, mode: 'diverging', range: R, tex,
      trace: trace || undefined, cellOf: trace ? (i) => `${q},${i},0` : undefined,
    });
    S.cleanup(() => v.destroy());
    return v;
  };
  const sT = strip(tok, 'W_E[\\mathrm{id}_t]', 'emb.tok');
  const sP = strip(pos, '\\Pe_t', S.live ? 'emb.pos' : null);
  const sX = strip(x, 'x_t', S.live ? 'emb.out' : null);
  S.hero.append(h('div', { class: 'sp-stack' },
    sT, op('+', { size: 48, tex: false, kind: 'pos' }),
    sP, op('=', { size: 48, tex: false, kind: 'pos' }),
    sX));
  // The token row and the position row fly into the sum, which is revealed when they arrive.
  if (!reducedMotion() && S.st) {
    sX.style.opacity = '0';
    requestAnimationFrame(() => {
      Promise.all([
        fly(S.st, sT, sX, { duration: 760 }),
        fly(S.st, sP, sX, { duration: 760, delay: 160 }),
      ]).then(() => {
        sX.style.opacity = '';
        sX.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300, easing: MOTION.easeOut });
        pulse(sX);
      }).catch(() => {});
    });
  }
  const check = S.live ? checkBadge(maxDiff(x, rowOf(S.trace['emb.out'].data, q, d)), 0, { label: `equals traced emb.out[${q}]` }) : null;
  S.side.append(eqBlock('x_t = W_E[\\mathrm{id}_t] + \\Pe_t', [
    `x_{${q},0} = ${num(tok[0], 3)} + ${num(pos[0], 3)} = ${num(x[0], 3)}`,
    `x_{${q},${d - 1}} = ${num(tok[d - 1], 3)} + ${num(pos[d - 1], 3)} = ${num(x[d - 1], 3)}`,
  ], check));
  S.note(`Query “${S.label(q)}” at position ${q}: the position row is added to the token row.`);
}

function learned2(S) {
  const { T, d, q } = S, P = learnedTable(S), U = unitRows(P, T, d), C = gram(U, U, T, d), lab = sparseLabels(T, 16);
  const hm = heat(S, {
    data: C, rows: T, cols: T, mode: 'diverging', range: 1, cell: cellHero(T), rowLabels: lab, colLabels: lab,
    highlight: { rows: [q], cols: [q] }, maxWidth: 620, ariaLabel: 'cosine similarity of position vectors',
  });
  let nearest = -1;
  for (let t = 0; t < T; t++) if (t !== q && (nearest < 0 || C[q * T + t] > C[q * T + nearest])) nearest = t;
  const prof = barChart({
    values: Array.from(C.subarray(q * T, (q + 1) * T)), labels: lab, kind: 'pos', highlight: q, format: (v) => fmt(v, 2),
    width: 300, height: 200, ariaLabel: `cosine with position ${q}`,
  });
  S.hero.append(pair(
    plate(hm.el, texHTML('\\cos(\\Pe_s,\\Pe_t)')),
    plate(prof.el, texHTML(`\\cos(\\Pe_{${q}},\\Pe_t)`))));
  S.side.append(eqBlock('\\cos(\\Pe_s,\\Pe_t) = \\frac{\\Pe_s\\cdot\\Pe_t}{\\lVert\\Pe_s\\rVert\\,\\lVert\\Pe_t\\rVert}', [
    `\\cos(\\Pe_{${q}},\\Pe_{${q}}) = ${num(C[q * T + q], 3)}`,
    nearest >= 0 ? `\\text{nearest other: } t = ${nearest},\\ \\cos = ${num(C[q * T + nearest], 3)}` : '\\text{one position: nothing to compare}',
  ]));
  S.note(`The bar chart is row ${q} of the matrix: how alike each position is to position ${q}.`);
}

// ------------------------------------------------------------------ sinusoidal
// Waves over positions: one sine and one cosine per shown pair, with a scan line and dots.
function wavesFigure(S, { rowH, scanTo = null }) {
  const { T, d } = S, rows = pairsShown(d), n = rows.length;
  const top = 26, WX = 150, WW = 820, AMP = Math.floor(rowH * 0.36);
  const colW = WW / Math.max(1, T);
  const X = (t) => WX + (t + 0.5) * colW;
  const axisY = top + n * rowH + 4;
  const nS = Math.max(2, (T - 1) * 10 + 1);
  const svg = s('svg', { class: 'sp-fig', width: HERO.w, height: HERO.h, viewBox: `0 0 ${HERO.w} ${HERO.h}` });
  const labels = [], curves = [], dots = [];
  const path = (w, yc, fn) => {
    let dd = '';
    for (let k = 0; k < nS; k++) { const t = ((T - 1) * k) / (nS - 1); dd += `${k ? 'L' : 'M'}${X(t).toFixed(1)},${(yc - AMP * fn(t * w)).toFixed(2)}`; }
    return dd;
  };
  rows.forEach(({ p, w }, i) => {
    const yc = top + rowH * (i + 0.5);
    svg.append(s('line', { x1: WX, x2: WX + WW, y1: yc, y2: yc, stroke: 'var(--line)', 'stroke-dasharray': '3 5' }));
    const sinP = s('path', { d: path(w, yc, Math.sin), fill: 'none', style: 'stroke: var(--c-pos)', 'stroke-width': 2.4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
    const cosP = s('path', { d: path(w, yc, Math.cos), fill: 'none', style: 'stroke: var(--fg-2)', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', opacity: '0.8' });
    svg.append(sinP, cosP);
    curves.push(sinP, cosP);
    const dS = s('g', { class: 'sp-dot' }, s('circle', { cx: 0, cy: 0, r: 5, style: 'fill: var(--c-pos)', stroke: 'var(--surface)', 'stroke-width': 1.5 }));
    const dC = s('g', { class: 'sp-dot' }, s('circle', { cx: 0, cy: 0, r: 5, style: 'fill: var(--fg-2)', stroke: 'var(--surface)', 'stroke-width': 1.5 }));
    svg.append(dS, dC);
    dots.push({ dS, dC, w, yc });
    labels.push(h('div', { class: 'sp-rowlab', style: { top: `${yc}px`, width: `${WX - 26}px` }, html: texHTML(`\\omega_{${p}} = ${num(w, 3)}`) }));
  });
  svg.append(s('line', { x1: WX, x2: WX + WW, y1: axisY, y2: axisY, stroke: 'var(--line-strong)' }));
  const ticks = [];
  for (let t = 0, k = every(T, 12); t < T; t += k) ticks.push(h('div', { class: 'sp-tick', style: { left: `${X(t)}px`, top: `${axisY + 8}px` }, text: String(t) }));
  const scanG = s('g', { class: 'sp-scan' }, s('line', { x1: 0, x2: 0, y1: top - 12, y2: scanTo ?? axisY, style: 'stroke: var(--accent)', 'stroke-width': 1.6 }));
  svg.append(scanG);
  const setPos = (pos) => {
    const x = X(pos).toFixed(2);
    scanG.style.transform = `translate(${x}px, 0px)`;
    dots.forEach(({ dS, dC, w, yc }) => {
      dS.style.transform = `translate(${x}px, ${(yc - AMP * Math.sin(pos * w)).toFixed(2)}px)`;
      dC.style.transform = `translate(${x}px, ${(yc - AMP * Math.cos(pos * w)).toFixed(2)}px)`;
    });
  };
  setPos(0);
  const reveal = () => {
    if (reducedMotion()) return;
    curves.forEach((p, i) => {
      const L = Math.max(1, p.getTotalLength());
      p.style.strokeDasharray = `${L}`;
      p.animate([{ strokeDashoffset: L }, { strokeDashoffset: 0 }], {
        duration: 1500, delay: 80 * Math.floor(i / 2), easing: MOTION.easeOut, fill: 'backwards',
      }).onfinish = () => { p.style.strokeDasharray = ''; };
    });
  };
  const legend = h('div', { class: 'sp-legend' }, h('span', { style: { color: 'var(--c-pos)' } }, '— sin'), h('span', { style: { color: 'var(--fg-2)' } }, '— cos'));
  return { nodes: [svg, legend, ...labels, ...ticks], setPos, reveal, rows };
}

function sin0(S) {
  const fig = wavesFigure(S, { rowH: 104 });
  S.hero.append(...fig.nodes);
  fig.reveal();
  walk(S, (t) => fig.setPos(t));
  const ps = fig.rows, a = ps[1] ?? ps[0], b = ps[ps.length - 1];
  // Live: the engine's table (the trace) against this scene's own sines and cosines.
  const check = S.live ? checkBadge(maxDiff(sinOwn(S.T, S.d), S.trace['emb.pos'].data), 0, { label: 'equals the traced emb.pos' }) : null;
  S.side.append(eqBlock('\\Pe_{t,2p} = \\sin(t\\,\\omega_p)', [
    '\\Pe_{t,2p+1} = \\cos(t\\,\\omega_p), \\quad \\omega_p = 10000^{-2p/d}',
    `\\omega_{0} = ${num(ps[0].w, 3)}, \\quad \\omega_{${a.p}} = ${num(a.w, 3)}`,
    `\\text{pair } ${b.p} \\text{ turns once every } ${num(2 * Math.PI / b.w, 0)} \\text{ tokens}`,
  ], check));
  S.note(`Scan position: “${S.label(S.q)}” at position ${S.q}. Fast pairs repeat every few tokens.`);
}

function sin1(S) {
  const { T, d } = S, P = sinTable(S), R = absMax([P]);
  const fig = wavesFigure(S, { rowH: 74, scanTo: 436 });
  S.hero.append(...fig.nodes);
  fig.reveal();
  const big = d > 96, cell = clamp(Math.floor((820 - 12) / d) - 1, 4, 14);
  const vecEl = vec(rowOf(P, 0, d), {
    orientation: 'h', cell, gap: 1, head: big ? 12 : d, tail: big ? 4 : 0, mode: 'diverging', range: R,
    trace: S.live ? 'emb.pos' : undefined,
  });
  S.cleanup(() => vecEl.destroy());
  const label = h('div', { class: 'sp-abs', style: { left: '150px', top: '398px', font: '600 16px/1 var(--font-mono)' } });
  const vecWrap = h('div', { class: 'sp-abs', style: { left: '150px', top: '440px' } }, vecEl);
  S.hero.append(label, vecWrap);
  const live = h('div', { class: 'sp-eq-live' });
  const checkHost = h('div', {});
  S.side.append(h('div', { class: 'sp-eq' },
    h('div', { class: 'sp-eq-main', html: texHTML('\\Pe_t = \\big(\\sin(t\\,\\omega_p),\\ \\cos(t\\,\\omega_p)\\big)_{p}', { display: true }) }),
    live, checkHost));
  walk(S, (t) => {
    const row = rowOf(P, t, d);
    fig.setPos(t);
    vecEl.update(row, { range: R });
    label.innerHTML = texHTML(`\\Pe_{${t}} = \\big(${vals3(row)},\\ \\dots\\big)`);
    live.replaceChildren(
      h('div', { html: texHTML(`\\Pe_{${t},0} = \\sin(${t}\\,\\omega_0) = ${num(row[0], 3)}`) }),
      h('div', { html: texHTML(`\\Pe_{${t},1} = \\cos(${t}\\,\\omega_0) = ${num(row[1], 3)}`) }));
    checkHost.replaceChildren(S.live
      ? checkBadge(maxDiff(sinRow(t, d), row), 0, { label: `equals traced emb.pos[${t}]` })
      : h('span', { class: 'sp-note', text: `${T} positions; the scan reads PE[t] for each one.` }));
  });
}

function sin2(S) {
  const { T, d, q } = S, P = sinTable(S), G = gram(P, P, T, d), lab = sparseLabels(T, 16);
  // The colour runs from the smallest to the largest value of this matrix, so the decay shows.
  const lo = Math.min(...G), hi = Math.max(...G);
  const hm = heat(S, {
    data: G, rows: T, cols: T, mode: 'sequential', kind: 'pos', range: [lo, hi], cell: cellHero(T), rowLabels: lab, colLabels: lab,
    highlight: { rows: [q], cols: [q] }, maxWidth: 620, ariaLabel: 'dot products of encodings',
  });
  const prof = barChart({
    values: Array.from(G.subarray(q * T, (q + 1) * T)), labels: lab, kind: 'pos', highlight: q, format: (v) => fmt(v, 1),
    width: 300, height: 200, ariaLabel: `dot products with position ${q}`,
  });
  S.hero.append(pair(
    plate(hm.el, texHTML('\\Pe_s\\cdot\\Pe_t')),
    plate(prof.el, texHTML(`\\Pe_{${q}}\\cdot\\Pe_t`))));
  const j = Math.max(0, q - 3), off = q - j;
  let formula = 0;
  for (let p = 0; p < d / 2; p++) formula += Math.cos(off * omega(p, d));
  const direct = G[q * T + j], dd = Math.abs(direct - formula);
  S.side.append(eqBlock('\\Pe_s\\cdot\\Pe_t = \\sum_{p}\\cos\\big((s-t)\\,\\omega_p\\big)', [
    `\\Pe_{${q}}\\cdot\\Pe_{${q}} = ${num(G[q * T + q], 2)} = \\tfrac{d}{2}`,
    `s = ${q},\\ t = ${j}:\\quad \\Pe_s\\cdot\\Pe_t = ${num(direct, 3)}`,
    `\\sum_{p}\\cos\\big(${off}\\,\\omega_p\\big) = ${num(formula, 3)}`,
  ], verdict(dd < 1e-9, dd < 1e-9 ? 'cosine sum = dot product' : `differs by ${num(dd, 3)}`)));
  S.note(`The bar chart is row ${q}: the similarity of position ${q} with every position, “${S.label(q)}” included.`);
}

// ------------------------------------------------------------------ rope
function rope0(S) {
  const { T, dh, q, hd, cfg } = S, base = cfg.rope_base;
  const qv = headRow(S.trace['L0.attn.q_heads'], hd, q, T, dh), R0 = absMax([qv]) * 1.42;
  const clk = buildClock(340), sc = clk.R / maxPairNorm(qv);
  const phi0 = Math.atan2(qv[1], qv[0]), len = pairNorm(qv, 0) * sc, theta0 = thetaOf(0, dh, base);
  const ghost = clk.hand({ len, color: 'var(--fg-3)', ghost: true, width: 2, dash: '5 5' });
  ghost.setAngle(phi0);
  const hand = clk.hand({ len, color: 'var(--c-q)', arc: true });
  hand.setAngle(phi0);
  const cellOf = (i) => `${q},${i},${hd}`;
  const startVec = vec([qv[0], qv[1]], {
    orientation: 'h', cell: 30, gap: 6, head: 2, tail: 0, kind: 'q', mode: 'diverging', range: R0,
    tex: '(q_0,\\ q_1)_{\\text{start}}', trace: 'L0.attn.q_heads', cellOf,
  });
  const rotVec = vec([0, 0], { orientation: 'h', cell: 30, gap: 6, head: 2, tail: 0, kind: 'q', mode: 'diverging', range: R0, tex: '(\\Q_{t,0},\\ \\Q_{t,1})', cellOf });
  S.cleanup(() => { startVec.destroy(); rotVec.destroy(); });
  const bigT = h('div', { class: 'sp-big' });
  const thetaLine = h('div');
  const noteEl = h('div', { class: 'sp-note' });
  const readBox = h('div', { class: 'sp-abs sp-read', style: { left: '470px', top: '40px', width: '540px' } }, bigT, thetaLine, startVec, rotVec, noteEl);
  S.hero.append(h('div', { class: 'sp-abs', style: { left: '40px', top: '85px' } }, clk.el), readBox);
  const update = (t, done) => {
    const phi = t * theta0, [y0, y1] = rotPair(qv[0], qv[1], phi);
    hand.setAngle(phi0 + phi);
    hand.setArc(-wrapPi(phi), 0);
    bigT.textContent = `t = ${t}`;
    thetaLine.innerHTML = texHTML(`t\\,\\theta_0 = ${num(phi, 3)}\\ \\text{rad}\\qquad (\\text{mod } 2\\pi = ${num(wrapPi(phi), 3)})`);
    rotVec.update([y0, y1], { range: R0 });
    setTrace(rotVec, done && S.live ? 'L0.attn.q_rot' : null);
    noteEl.textContent = done
      ? (S.live
        ? `At position ${q}, the selected token, this pair is the traced q_rot.`
        : 'An illustration: this model’s query turned to this position. This config has no traced q_rot.')
      : 'The grey hand is the same query at position 0.';
  };
  walk(S, update);
  let check = null;
  if (S.live) {
    const tr = headRow(S.trace['L0.attn.q_rot'], hd, q, T, dh);
    let e = 0;
    for (let p = 0; p < dh / 2; p++) {
      const [y0, y1] = rotPair(qv[2 * p], qv[2 * p + 1], q * thetaOf(p, dh, base));
      e = Math.max(e, Math.abs(y0 - tr[2 * p]), Math.abs(y1 - tr[2 * p + 1]));
    }
    check = checkBadge(e, 0, { label: 'equals the traced q_rot' });
  }
  S.side.append(eqBlock('\\begin{aligned}\\Q_{t,0}&=q_0\\cos\\phi-q_1\\sin\\phi\\\\\\Q_{t,1}&=q_0\\sin\\phi+q_1\\cos\\phi\\end{aligned}', [
    `\\phi = t\\,\\theta_0,\\quad \\phi_{${q}} = ${num(q * theta0, 3)}`,
    `\\theta_p = \\text{base}^{-2p/d_h}`,
    `\\theta_0 = ${num(theta0, 3)},\\ \\text{base} = ${base},\\ d_h = ${dh}`,
  ], check));
  S.note(`The clock shows the first pair of the query of “${S.label(q)}”; each pair turns by its own angle.`);
}

function rope1(S) {
  const { T, dh, q, hd, cfg } = S, base = cfg.rope_base, n = dh / 2;
  const qv = headRow(S.trace['L0.attn.q_heads'], hd, q, T, dh);
  const ps = n >= 2 ? slowPair(dh) : 0;
  const pairs = [{ p: 0, label: 'fast pair' }, ...(n >= 2 ? [{ p: ps, label: 'slow pair' }] : [])];
  const CLOCK = 220, faceR = CLOCK / 2 - 22, sc = faceR / maxPairNorm(qv), R0 = absMax([qv]) * 1.42;
  const row = h('div', { class: 'sp-abs', style: { left: '0', right: '0', top: '0px', display: 'flex', justifyContent: 'center', gap: '90px' } });
  const cards = pairs.map(({ p, label }) => {
    const th = thetaOf(p, dh, base), clk = buildClock(CLOCK), len = pairNorm(qv, p) * sc, turn = 2 * Math.PI / th;
    const phi0 = Math.atan2(qv[2 * p + 1], qv[2 * p]);
    const ghost = clk.hand({ len, color: 'var(--fg-3)', ghost: true, width: 2, dash: '5 5' });
    ghost.setAngle(phi0);
    const hand = clk.hand({ len, color: 'var(--c-q)', arc: true });
    hand.setAngle(phi0);
    const rot = vec([0, 0], {
      orientation: 'h', cell: 26, gap: 6, head: 2, tail: 0, kind: 'q', mode: 'diverging', range: R0,
      tex: `(\\Q_{t,${2 * p}},\\ \\Q_{t,${2 * p + 1}})`, cellOf: (i) => `${q},${2 * p + i},${hd}`,
    });
    S.cleanup(() => rot.destroy());
    const box = h('div', { class: 'sp-read', style: { width: '260px', fontSize: '17px' } },
      h('div', { html: texHTML(`\\text{${label}}\\ \\ p = ${p}`) }),
      h('div', { html: texHTML(`\\theta_{${p}} = ${num(th, 4)}`) }),
      h('div', { html: texHTML(`\\text{one turn every } ${num(turn, turn < 100 ? 1 : 0)}\\ \\text{tokens}`) }),
      rot);
    row.append(h('div', { class: 'sp-col' }, clk.el, box));
    return { hand, th, p, rot, phi0 };
  });
  S.hero.append(row);
  const readout = h('div', { class: 'sp-abs', style: { left: '0', right: '0', top: '428px', textAlign: 'center', font: '500 18px/1 var(--font-mono)', color: 'var(--fg-2)' } });
  S.hero.append(readout);
  walk(S, (t, done) => {
    cards.forEach(({ hand, th, p, rot, phi0 }) => {
      const phi = t * th, [y0, y1] = rotPair(qv[2 * p], qv[2 * p + 1], phi);
      hand.setAngle(phi0 + phi);
      hand.setArc(-wrapPi(phi), 0);
      rot.update([y0, y1], { range: R0 });
      setTrace(rot, done && S.live ? 'L0.attn.q_rot' : null);
    });
    readout.textContent = done
      ? `t = ${t}: the fast pair has turned ${num(t * cards[0].th, 2)} rad, the slow pair ${num(t * cards[cards.length - 1].th, 2)} rad.`
      : `t = ${t}`;
  });
  S.side.append(eqBlock('\\theta_p = \\text{base}^{-2p/d_h}', [
    `\\theta_0 = 1 \\quad \\text{(the fast pair)}`,
    `\\theta_{${ps}} = ${num(thetaOf(ps, dh, base), 4)} \\quad \\text{(the slow pair)}`,
    '\\text{same direction, different speeds}',
  ]));
  S.note(`Both clocks show the query of “${S.label(q)}” at position ${q}, before and after the turn.`);
}

function rope2(S) {
  const { T, dh, q, hd, cfg } = S, base = cfg.rope_base, n = dh / 2;
  const kvh = Math.floor(hd / (S.H / cfg.n_kv_head));
  const qv = headRow(S.trace['L0.attn.q_heads'], hd, q, T, dh);
  const kv = headRow(S.trace['L0.attn.k_heads'], kvh, q, T, dh);
  const off = Math.min(3, Math.max(1, cfg.block_size - 2));
  const span = Math.max(0, cfg.block_size - 1 - off);
  const ms = [...new Set([0, Math.round(span / 3), Math.round((2 * span) / 3), span])];
  // Engine rope() at several absolute positions: the query at m, the key at m + off.
  const rowsData = ms.map((m) => {
    const qm = ropeAt(qv, m, base), kn = ropeAt(kv, m + off, base);
    return { m, n: m + off, qm, kn, dot: dotOf(qm, kn) };
  });
  const vals = rowsData.map((r) => r.dot), spread = Math.max(...vals) - Math.min(...vals);
  const plain = dotOf(qv, kv);
  const ps = n >= 2 ? slowPair(dh) : 0;
  const clk = buildClock(300);
  const pq = (r) => [r.qm[2 * ps], r.qm[2 * ps + 1]], pk = (r) => [r.kn[2 * ps], r.kn[2 * ps + 1]];
  const sc = clk.R / Math.max(pairNorm(rowsData[0].qm, ps), pairNorm(rowsData[0].kn, ps), 1e-9);
  const qHand = clk.hand({ len: Math.hypot(...pq(rowsData[0])) * sc, color: 'var(--c-q)', arc: true });
  const kHand = clk.hand({ len: Math.hypot(...pk(rowsData[0])) * sc, color: 'var(--c-k)' });
  // The angle from q′ to k′ is the same in every row: the arc lives on the query hand.
  const wedges = rowsData.map((r) => wrapPi(Math.atan2(pk(r)[1], pk(r)[0]) - Math.atan2(pq(r)[1], pq(r)[0])));
  const wSpread = Math.max(...wedges) - Math.min(...wedges);
  let aq = Math.atan2(pq(rowsData[0])[1], pq(rowsData[0])[0]), ak = Math.atan2(pk(rowsData[0])[1], pk(rowsData[0])[0]);
  const place = (r) => {
    const fq = Math.atan2(pq(r)[1], pq(r)[0]), fk = Math.atan2(pk(r)[1], pk(r)[0]);
    aq += wrapPi(fq - aq);
    ak += wrapPi(fk - ak);
    qHand.setAngle(aq);
    kHand.setAngle(ak);
  };
  qHand.setArc(0, wedges[0]);
  S.hero.append(h('div', { class: 'sp-abs', style: { left: '30px', top: '70px' } }, clk.el));
  const tbody = h('tbody');
  const trs = rowsData.map((r) => {
    const tr = h('tr', null, h('td', null, String(r.m)), h('td', null, String(r.n)), h('td', null, fmt(r.dot, 3)));
    tbody.append(tr);
    return tr;
  });
  const tableEl = h('table', { class: 'sp-table' },
    h('thead', null, h('tr', null, h('th', null, 'query at m'), h('th', null, `key at m + ${off}`), h('th', null, 'q′·k′'))),
    tbody);
  const readout = h('div', { class: 'sp-read', style: { fontSize: '18px' } });
  S.hero.append(h('div', { class: 'sp-abs', style: { left: '430px', top: '80px', width: '560px' } }, tableEl,
    h('div', { class: 'sp-note', style: { marginTop: '14px' }, text: `Without RoPE, the same two vectors give q·k = ${fmt(plain, 3)}: one number, with no position in it.` }),
    h('div', { class: 'sp-note', style: { marginTop: '6px' }, text: 'Illustration: the model’s own q and k, turned to other positions.' })));
  S.hero.append(h('div', { class: 'sp-abs', style: { left: '30px', top: '372px', width: '380px' } }, readout));
  const show = (i) => {
    const r = rowsData[i];
    place(r);
    trs.forEach((tr, k) => tr.classList.toggle('on', k === i));
    readout.replaceChildren(
      h('div', { html: texHTML(`m = ${r.m}, \\quad n = m + ${off} = ${r.n}`) }),
      h('div', { html: texHTML(`q'_m \\cdot k'_n = ${num(r.dot, 3)}`) }),
      h('div', { class: 'sp-note', text: `The angle from the query hand to the key hand: ${fmt(wedges[0], 3)} rad, the same in every row.` }));
  };
  // One pass through the rows, then rest on the first. Hovering or clicking a row stops the pass.
  let pass = null;
  const stopPass = () => { clearTimeout(pass); pass = null; };
  S.cleanup(stopPass);
  trs.forEach((tr, k) => {
    tr.addEventListener('pointerenter', () => { stopPass(); show(k); });
    tr.addEventListener('click', () => { stopPass(); show(k); });
  });
  tableEl.addEventListener('pointerenter', stopPass);
  if (!reducedMotion() && rowsData.length > 1) {
    let k = 1;
    const next = () => {
      show(k);
      k += 1;
      pass = setTimeout(k < rowsData.length ? next : () => show(0), 1300);
    };
    show(0);
    pass = setTimeout(next, 1300);
  } else {
    show(0);
  }
  S.side.append(eqBlock('\\langle R_m q,\\ R_n k\\rangle = \\langle q,\\ R_{n-m} k\\rangle', [
    `n - m = ${off} \\quad \\text{in every row}`,
    `\\max_m - \\min_m\\ q'_m\\cdot k'_{m+${off}} = ${nz(spread)}`,
    `\\text{angle spread } = ${nz(wSpread)}`,
  ], verdict(spread < 1e-9, spread < 1e-9 ? 'equal in every row' : 'the rows differ')));
}

// ------------------------------------------------------------------ alibi
function alibi0(S) {
  const { H } = S, m = alibiSlopes(H), top = 280;
  const colW = Math.min(120, Math.floor(HERO.w / H));
  const bars = m.map((mh, hh) => {
    const bar = h('div', { class: 'sp-bar', style: { height: '0px', width: `${Math.min(84, colW - 22)}px` } });
    return {
      bar, target: Math.max(3, (mh / m[0]) * top),
      col: h('div', { class: 'sp-slope', style: { width: `${colW}px` } },
        h('div', { class: 'sp-slope-m', html: texHTML(`m_{${hh}}`) }),
        h('div', { class: 'sp-slope-v', text: fmt(mh, 4) }),
        bar,
        h('div', { class: 'sp-note', text: `head ${hh}` })),
    };
  });
  S.hero.append(h('div', { class: 'sp-bars' }, bars.map((b) => b.col)));
  if (reducedMotion()) bars.forEach((b) => { b.bar.style.height = `${b.target}px`; });
  else requestAnimationFrame(() => requestAnimationFrame(() => bars.forEach((b) => { b.bar.style.height = `${b.target}px`; })));
  S.side.append(eqBlock('m_h = 2^{-8(h+1)/H}', [
    `m_{0} = 2^{-8/${H}} = ${num(m[0], 4)}`,
    `m_{${H - 1}} = 2^{-8} = ${num(m[H - 1], 4)}`,
    `\\frac{m_{h+1}}{m_h} = 2^{-8/H} = ${num(Math.pow(2, -8 / H), 4)}`,
    '\\text{the same ratio for every head}',
  ]));
}

function alibi1(S) {
  const { T, H, q, hd, cfg } = S;
  const m = alibiSlopes(H);
  const Bs = S.live ? null : alibiBias(cfg, T);                       // the formula, for previews only
  const bOf = (hh) => (S.live ? headMat(S.trace.alibi, hh, T) : Bs[hh]);  // the traced tensor when live
  // The colour scale is shared by every head: the steepest head (h = 0) at the largest distance.
  const range = Math.pow(2, -8 / H) * Math.max(1, T - 1);
  // Live cross-check against the formula, recomputed here cell by cell: the traced alibi tensor, and the
  // traced bias (biased minus the scores before it) for every head.
  let check = null;
  if (S.live) {
    const pre = S.trace[cfg.attn_scale ? 'L0.attn.scaled' : 'L0.attn.scores'], biased = S.trace['L0.attn.biased'];
    let e = 0;
    for (let hh = 0; hh < H; hh++) {
      const A = headMat(S.trace.alibi, hh, T), P0 = headMat(pre, hh, T), P1 = headMat(biased, hh, T);
      const mm = Math.pow(2, (-8 * (hh + 1)) / H);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        const own = -mm * Math.abs(i - j), k = i * T + j;
        e = Math.max(e, Math.abs(A[k] - own), Math.abs((P1[k] - P0[k]) - own));
      }
    }
    check = checkBadge(e, 0, { label: 'traced bias = −m_h·|i − j|' });
  }
  // The focus: the selected head at full size, with the query row outlined.
  const cf = clamp(Math.floor(430 / T), 4, 14);
  const focus = heat(S, {
    data: bOf(hd), rows: T, cols: T, mode: 'diverging', range, cell: cf, cellH: cf, highlight: { rows: [q] }, maxWidth: 480,
    ariaLabel: `ALiBi bias, head ${hd}`,
  }, { trace: S.live ? 'alibi' : null, band: hd });
  // The context: every head as a small multiple, on the same colour scale. Clicking one selects that head.
  const cols = Math.min(H, 4), rows = Math.ceil(H / cols);
  const size = clamp(Math.floor((HERO.h - rows * 46) / rows), 56, 100);
  const thumbs = [];
  for (let hh = 0; hh < H; hh++) {
    const { canvas, paint } = thumbMap(bOf(hh), T, range, size);
    S.cleanup(onThemeChange(paint));
    thumbs.push(h('button', {
      type: 'button', class: ['sp-thumb', hh === hd && 'on'], title: `head ${hh}: m = ${fmt(m[hh], 4)}`, 'aria-pressed': String(hh === hd),
      onClick: () => S.select(hh),
    }, canvas, h('span', { text: `h${hh}` })));
  }
  const grid = h('div', { class: 'sp-thumbs', style: { gridTemplateColumns: `repeat(${cols}, auto)` } }, ...thumbs);
  S.hero.append(h('div', { class: 'sp-abs', style: { left: '0', top: '0', width: `${HERO.w}px`, height: `${HERO.h}px`, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '48px' } },
    plate(focus.el, texHTML(`b_{${hd}}, \\; m_{${hd}} = ${num(m[hd], 4)}`)), grid));
  const dd = Math.min(4, q);
  const lines = dd > 0
    ? [`b_{0,${q},${q - dd}} = -${num(m[0], 4)} \\cdot ${dd} = ${num(bOf(0)[q * T + q - dd], 3)}`,
      `b_{${H - 1},${q},${q - dd}} = -${num(m[H - 1], 4)} \\cdot ${dd} = ${num(bOf(H - 1)[q * T + q - dd], 4)}`]
    : ['b_{h,i,i} = 0 \\quad \\text{on the diagonal}'];
  S.side.append(eqBlock('b_{h,ij} = -m_h\\,\\lvert i-j\\rvert', lines, check));
  S.note('Every head on one colour scale. The outlined head is the selected one; click a head to select it.');
}

function alibi2(S) {
  const { T, hd, q, cfg } = S;
  const Bown = ownBias(T, S.H, hd);
  const key = cfg.attn_scale ? 'L0.attn.scaled' : 'L0.attn.scores';
  const Sc = headMat(S.trace[key], hd, T);
  // S + b is computed here from the scores and the formula. When live, the traced biased tensor is the one shown.
  const Sbc = Float64Array.from(Sc, (v, k) => v + Bown[k]);
  const Sb = S.live ? headMat(S.trace['L0.attn.biased'], hd, T) : Sbc;
  const Bd = S.live ? headMat(S.trace.alibi, hd, T) : Bown;
  // The highlighted key: the best score after the bias among the keys the mask allows for this query.
  let j = -1;
  for (let k = 0; k < T; k++) if (maskAllows(cfg, q, k) && (j < 0 || Sb[q * T + k] > Sb[q * T + j])) j = k;
  if (j < 0) j = q;
  const range = absMax([Sc, Sb]), bRange = absMax([Bd]), c = clamp(Math.floor(240 / T), 3, 12), lab = sparseLabels(T, 16);
  const plateOf = (data, rng, title, aria, trace) => {
    const hm = heat(S, {
      data, rows: T, cols: T, mode: 'diverging', range: rng, cell: c, highlight: { rows: [q], cells: [[q, j]] }, maxWidth: 300,
      rowLabels: lab, ariaLabel: aria,
    }, { trace, band: hd });
    return plate(hm.el, texHTML(title));
  };
  const plateS = plateOf(Sc, range, cfg.attn_scale ? '\\tilde S_h' : 'S_h', 'scores', key);
  const plateB = plateOf(Bd, bRange, 'b_h', 'ALiBi bias', S.live ? 'alibi' : null);
  const plateSb = plateOf(Sb, range, 'S^{b}_h', 'scores with bias', S.live ? 'L0.attn.biased' : null);
  S.hero.append(h('div', { class: 'sp-abs', style: { left: '0', top: '0', width: `${HERO.w}px`, height: `${HERO.h}px`, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '14px' } },
    plateS, op('+', { size: 48, tex: false, kind: 'pos' }), plateB, op('=', { size: 48, tex: false, kind: 'pos' }), plateSb));
  let check = null;
  if (S.live) {
    const e = Math.max(maxDiff(Sbc, headMat(S.trace['L0.attn.biased'], hd, T)), maxDiff(Bown, headMat(S.trace.alibi, hd, T)));
    check = checkBadge(e, 0, { label: 'S + b equals traced biased' });
  }
  const lhs = cfg.attn_scale ? '\\frac{q_i\\cdot k_j}{\\sqrt{d_h}}' : 'q_i\\cdot k_j';
  S.side.append(eqBlock(`S^{b}_{h,ij} = ${lhs} - m_h\\lvert i-j\\rvert`, [
    `(i,j) = (${q}, ${j})`,
    `${num(Sc[q * T + j], 3)} + (${num(Bd[q * T + j], 3)}) = ${num(Sb[q * T + j], 3)}`,
    '\\text{the attention mask comes next}',
  ], check));
  S.note(`Head ${hd}, query “${S.label(q)}” at position ${q}. The outlined key has the highest score the mask allows after the bias.`);
}

// ------------------------------------------------------------------ none
function noneRuns(S, mask) {
  const perm = permute(S.T, SHUFFLE_SEED + S.shuffleN);
  const m = cloneModel(S.model);
  m.config = { ...m.config, pos_encoding: 'none', mask };
  const ids2 = perm.map((i) => S.ids[i]);
  const a = infer(m, S.ids, { trace: false }).logits, b = infer(m, ids2, { trace: false }).logits;
  return { perm, V: a.shape[1], a: a.data, b: b.data };
}

function none0(S) {
  const { T } = S, perm = permute(T, SHUFFLE_SEED + S.shuffleN);
  const colW = Math.min(HERO.w / T, 66), left = (HERO.w - T * colW) / 2, topY = 124, botY = 330, tileH = 50;
  const cx = (i) => left + i * colW + colW / 2;
  const svg = s('svg', { class: 'sp-fig', width: HERO.w, height: HERO.h, viewBox: `0 0 ${HERO.w} ${HERO.h}` });
  for (let r = 0; r < T; r++) {
    const x1 = cx(perm[r]), y1 = topY + tileH, x2 = cx(r), y2 = botY, ym = (y1 + y2) / 2;
    svg.append(s('path', { class: 'sp-arc', d: `M${x1.toFixed(1)},${y1} C${x1.toFixed(1)},${ym} ${x2.toFixed(1)},${ym} ${x2.toFixed(1)},${y2}` }));
  }
  // Long sequences: tiles shrink to ~10 px, so the index is dropped and the label gets smaller.
  const tile = (x, label, idx, y) => h('div', { class: 'sp-tile', style: { left: `${left + x * colW + 3}px`, top: `${y}px`, width: `${colW - 6}px` } },
    h('b', { style: { fontSize: T > 32 ? '12px' : '16px' }, text: label }), T > 32 ? null : h('small', null, String(idx)));
  const tiles = [];
  for (let i = 0; i < T; i++) tiles.push(tile(i, S.label(i), i, topY));
  for (let r = 0; r < T; r++) tiles.push(tile(r, S.label(perm[r]), perm[r], botY));
  S.hero.append(svg, ...tiles,
    h('div', { class: 'sp-tag', style: { left: `${left}px`, top: `${topY - 26}px` }, text: 'original order' }),
    h('div', { class: 'sp-tag', style: { left: `${left}px`, top: `${botY - 26}px` }, text: 'shuffled order' }));
  S.side.append(eqBlock('\\mathrm{out}(x_{\\pi}) = \\mathrm{out}(x)_{\\pi}', [
    `\\pi = \\big(${perm.slice(0, 10).join(', ')}${T > 10 ? ',\\ \\dots' : ''}\\big)`,
    '\\text{the same tokens, in a new order}',
  ]));
  S.side.append(button('Shuffle again', () => S.reshuffle(), { title: 'Draw another permutation' }));
}

function none1(S) {
  const { T, q } = S;
  if (T < 2) {
    S.hero.append(h('div', { class: 'sp-abs sp-note', style: { left: '0', top: '200px' }, text: 'Use at least two tokens to see a shuffle.' }));
    S.side.append(eqBlock('\\mathrm{out}(x_{\\pi}) = \\mathrm{out}(x)_{\\pi}'));
    return;
  }
  const { perm, V, a, b } = noneRuns(S, 'none');
  let maxD = 0;
  for (let r = 0; r < T; r++) for (let c = 0; c < V; c++) maxD = Math.max(maxD, Math.abs(b[r * V + c] - a[perm[r] * V + c]));
  // The change is judged against the output's own scale. A (near-)zero output does not depend on the input.
  const scale = maxAbsOf(a), flat = scale < 1e-9;
  const range = absMax([a, b]), lab = labelsOf(T);
  const cell = clamp(Math.min(Math.floor(360 / T), Math.floor(440 / V)), 3, 14);   // the plates fit the hero's width too
  const at = (tok) => perm.indexOf(tok);                   // the shuffled row that holds token tok
  let hA = null, hB = null;
  // The query's row lights up on both sides; hovering any row moves the light to that token.
  const light = (tok) => { hA.update({ highlight: { rows: [tok] } }); hB.update({ highlight: { rows: [at(tok)] } }); };
  hA = heat(S, {
    data: a, rows: T, cols: V, mode: 'diverging', range, cell, cellH: cell, rowLabels: lab, highlight: { rows: [q] }, maxWidth: 480,
    ariaLabel: 'outputs, original order', onHover: (r) => light(r), onLeave: () => light(q),
  });
  hB = heat(S, {
    data: b, rows: T, cols: V, mode: 'diverging', range, cell, cellH: cell, rowLabels: perm.map(String), highlight: { rows: [at(q)] }, maxWidth: 480,
    ariaLabel: 'outputs, shuffled input', onHover: (r) => light(perm[r]), onLeave: () => light(q),
  });
  S.hero.append(h('div', { class: 'sp-abs', style: { left: '0', top: '0', width: '1030px', display: 'flex', justifyContent: 'center', gap: '36px', alignItems: 'flex-start' } },
    plate(hA.el, texHTML('\\text{original input, row = token}')),
    plate(hB.el, texHTML('\\text{shuffled input, row = original index}'))));
  const check = flat ? null : checkBadge(maxD / scale, 0, { label: 'the same rows, moved', tol: 1e-6 });
  S.side.append(eqBlock('\\mathrm{out}_{r} = \\mathrm{out}_{\\pi(r)}', [
    `\\Delta = \\max_{r,c}\\big|\\mathrm{out}_{r,c}-\\mathrm{out}_{\\pi(r),c}\\big| = ${nz(maxD)}`,
  ], check));
  S.note(flat
    ? 'This configuration’s output does not depend on the input: every value is zero to within rounding.'
    : 'Hover a row on either side: the same token lights up on both sides.');
}

function none2(S) {
  const { T } = S;
  if (T < 2) {
    S.hero.append(h('div', { class: 'sp-abs sp-note', style: { left: '0', top: '200px' }, text: 'Use at least two tokens to see the mask at work.' }));
    S.side.append(eqBlock('\\text{causal mask}'));
    return;
  }
  const { perm, V, a, b } = noneRuns(S, 'causal');
  const vals = [];
  for (let r = 0; r < T; r++) {
    let mx = 0;
    for (let c = 0; c < V; c++) mx = Math.max(mx, Math.abs(b[r * V + c] - a[perm[r] * V + c]));
    vals.push(mx);
  }
  const mx = Math.max(...vals), scale = maxAbsOf(a), flat = scale < 1e-9;
  const bc = barChart({ values: vals, labels: labelsOf(T), kind: 'pos', format: (v) => (v < 1e-9 ? '0' : fmt(v, 3)), width: 900, height: 330, ariaLabel: 'change per row, causal mask kept' });
  S.hero.append(h('div', { class: 'sp-abs', style: { left: '50px', top: '0' } }, plate(bc.el,
    texHTML('\\max_c \\big|\\mathrm{out}_{r,c} - \\mathrm{out}_{\\pi(r),c}\\big| \\text{ per row } r, \\text{ causal mask kept}'))));
  const changed = !flat && mx > 1e-6 * scale;
  S.side.append(eqBlock('\\mathrm{keys}(r) = \\{0,\\dots,r\\}', [
    `\\max_{r}\\big|\\mathrm{out}_{r} - \\mathrm{out}_{\\pi(r)}\\big| = ${nz(mx)}`,
    '\\text{shuffled: } \\{\\pi(0),\\dots,\\pi(r)\\}',
  ], flat ? null : verdict(changed, changed ? 'rows change under the mask' : 'no change at this length: try a longer text')));
  S.note(flat ? 'This configuration’s output does not depend on the input, so the mask has nothing to show.' : 'Rows after the first see a different set of earlier tokens once the input is shuffled.');
}

// ------------------------------------------------------------------ tables
const STEP_FN = {
  learned: [learned0, learned1, learned2],
  sinusoidal: [sin0, sin1, sin2],
  rope: [rope0, rope1, rope2],
  alibi: [alibi0, alibi1, alibi2],
  none: [none0, none1, none2],
};

// Whether "Use this" changes a parameter shape (only the learned table appears or disappears). When it does, the store
// re-initialises every weight; when it does not, the current weights are kept.
const reinitText = (S) => ((S.cfg.pos_encoding === 'learned') !== (S.shown === 'learned')
  ? 'Using it re-initialises all weights with random values, because the parameter shapes change.'
  : 'Using it keeps the current weights: no parameter shape changes.');

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    let alive = true, paused = false, step = 0, preview = null, lastCurrent = ctx.cfg.pos_encoding;
    let shuffleN = 0, nSteps = 3, autoOn = true, auto = null, programmatic = false, cleanups = [], notes = [];

    // ---------------------------------------------------------------- frame (shared kit parts)
    const header = sceneHeader(st, { chips: [['Embedding', 'embed'], ['pos_encoding', 'pos']], title: 'Positional encoding' });
    const chipPos = header.el.querySelectorAll('.k-chip')[1];
    const seg = segmented({
      ariaLabel: 'Scheme to preview',
      options: SCHEMES.map((id) => ({ value: id, label: NAME[id], title: `Preview ${NAME[id]}` })),
      value: ctx.cfg.pos_encoding,
      onChange: (v) => { stopAuto(); preview = v === ctx.cfg.pos_encoding ? null : v; step = 0; rebuild(); },
    });
    seg.el.classList.add('sp-seg');
    // Controls band: the query tokens on the left (wrapping for long inputs), the scheme on the right.
    const picker = tokenPicker(st, ctx, { label: 'query', kind: 'pos', maxWidth: 1010 });
    picker.el.classList.add('sp-picker');
    const schemeBox = h('div', { class: 'sp-scheme' }, seg.el);
    st.add(schemeBox, { x: XR, y: CTRL_Y, anchor: 'tr' });
    // Side column: the head chips (RoPE and ALiBi, for the whole scheme), then the step's own content.
    const sideHead = h('div', { class: 'sp-heads' });
    const sideBody = h('div', { class: 'sp-side-body' });
    const side = h('div', { class: 'sp-side', style: { width: `${ZONES.side.w}px` } }, sideHead, sideBody);
    const hero = h('div', { class: 'sp-hero', style: { width: `${HERO.w}px`, height: `${HERO.h}px` } });
    const cap = sceneCaption(st, {});
    st.add(hero, { x: HERO.x, y: HERO.y });
    st.add(side, { x: ZONES.side.x, y: HERO.y });

    // ---------------------------------------------------------------- state
    function snapshot(layer) {
      const S0 = ctx.state, c = ctx.cfg;
      const T = S0.ids.length, d = c.d_model, H = c.n_head;
      const shown = preview ?? c.pos_encoding;
      return {
        st, T, d, H, dh: d / H, cfg: c, ids: S0.ids, trace: S0.trace, model: S0.model, seed: S0.seed, preset: !!S0.preset,
        q: clamp(S0.sel.token ?? T - 1, 0, T - 1), hd: clamp(S0.sel.head ?? 0, 0, H - 1),
        shown, live: shown === c.pos_encoding, shuffleN,
        hero: layer, side: sideBody, label: (i) => ctx.label(i),
        note: (text) => notes.push(text),
        select: (head) => ctx.store.select({ head }),
        cleanup: (fn) => cleanups.push(fn),
        reshuffle: () => { shuffleN++; renderStep(step); },
      };
    }

    function cleanupAll() {
      const list = cleanups;
      cleanups = [];
      list.forEach((f) => { try { f(); } catch { /* ignore */ } });
    }

    // A new layer for the hero. The previous layer fades out while the new one fades in.
    function newLayer() {
      const animate = !reducedMotion();
      for (const n of Array.from(hero.children)) {
        if (!animate || n.classList.contains('sp-out')) { n.remove(); continue; }
        n.classList.add('sp-out');
        n.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 300, easing: 'ease-out', fill: 'forwards' })
          .finished.then(() => n.remove(), () => n.remove());
      }
      const layer = h('div', { class: 'sp-layer' });
      if (animate) layer.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 420, delay: 140, easing: MOTION.easeOut, fill: 'backwards' });
      hero.append(layer);
      return layer;
    }

    function useScheme(v) {
      stopAuto();
      ctx.store.setConfig({ pos_encoding: v });    // re-initialises the weights when a parameter shape changes
    }

    // The status of the scheme, under the equation: live, or a preview with its one action.
    function statusRow(S) {
      if (S.live) return h('div', { class: 'sp-status' }, h('span', { class: 'sp-flag live' }, `in use · ${NAME[S.shown]}`));
      return h('div', { class: 'sp-status' },
        h('span', { class: 'sp-flag' }, S.shown === 'none' ? 'preview · clone' : 'preview · illustration'),
        button('Use this', () => useScheme(S.shown), { variant: 'primary', title: reinitText(S) }));
    }

    // Head chips for RoPE and ALiBi: one wrapping row, rebuilt whenever the trace or the selection changes.
    function renderHeads(S) {
      sideHead.replaceChildren(h('div', { class: 'sp-heads-label' }, 'head'), ...Array.from({ length: S.H }, (_, i) => {
        const b = h('button', { type: 'button', class: ['k-pick', 'k-pick-head', i === S.hd && 'sel'], title: `head ${i}`, style: { '--kind': 'var(--c-attn)' } }, h('b', null, `h${i}`));
        b.addEventListener('click', () => ctx.store.select({ head: i }));
        return b;
      }));
    }

    function renderStep(i) {
      if (!alive || paused) return;
      cleanupAll();
      const layer = newLayer();
      const S = snapshot(layer);
      notes = [];
      sideBody.replaceChildren(h('div', { class: 'k-label', style: { textAlign: 'left' } }, STEPS[S.shown][i]));
      try {
        STEP_FN[S.shown][i](S);
      } catch (err) {
        console.error('[positional] step failed', err);
        layer.replaceChildren(h('div', { class: 'callout warn' }, `This step could not be drawn: ${err.message}`));
      }
      sideBody.append(statusRow(S));
      if (!S.live) notes.unshift(NOTE[S.shown]);
      for (const t of notes) sideBody.append(h('div', { class: 'sp-note', text: t }));
      cap.set(capText(S, i));
      if (!reducedMotion()) sideBody.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 360, easing: MOTION.easeOut });
    }

    function stopAuto() { autoOn = false; clearTimeout(auto); }
    function scheduleAuto() {
      clearTimeout(auto);
      if (!autoOn || !alive || paused || reducedMotion()) return;
      auto = setTimeout(() => {
        if (step >= nSteps - 1) { autoOn = false; return; }
        programmatic = true;
        try { ctx.setStep(step + 1); } finally { programmatic = false; }
        scheduleAuto();
      }, MOTION.step);
    }

    // A step change from the story bar or a click stops auto-advance; programmatic moves do not.
    function onStep(i) {
      if (!programmatic) stopAuto();
      if (!alive || paused || i === step) return;
      step = i;
      renderStep(i);
    }

    // The header line: the question and the scheme in one sentence. None is stated, not asked.
    function subFor(S) {
      if (S.shown === 'none') return `<b>${NAME.none}</b> ${ONE.none}`;
      return `How does the model know where each token is? <b>${NAME[S.shown]}</b> ${ONE[S.shown]}`;
    }

    function rebuild() {
      if (!alive || paused || !ctx.state.trace) return;
      const c = ctx.cfg;
      if (lastCurrent !== c.pos_encoding) { lastCurrent = c.pos_encoding; preview = null; step = 0; }
      const S = snapshot(hero);
      nSteps = STEPS[S.shown].length;
      chipPos.textContent = `pos_encoding: ${c.pos_encoding}`;
      header.setSub(subFor(S));
      seg.set(S.shown);
      picker.el.classList.toggle('dense', ctx.state.ids.length > DENSE_T);
      const needHeads = S.shown === 'rope' || S.shown === 'alibi';
      sideHead.style.display = needHeads ? '' : 'none';
      if (needHeads) renderHeads(S); else sideHead.replaceChildren();
      ctx.setSteps(STEPS[S.shown], onStep);
      step = clamp(step, 0, nSteps - 1);
      programmatic = true;
      try { ctx.setStep(step); } finally { programmatic = false; }
      renderStep(step);
    }

    const offs = [ctx.on('trace', rebuild), ctx.on('select', rebuild)];
    el.addEventListener('pointerdown', stopAuto);

    rebuild();
    scheduleAuto();

    return {
      destroy() {
        alive = false;
        stopAuto();
        cleanupAll();
        offs.forEach((f) => f?.());
        st.destroy();
      },
      pause() { paused = true; stopAuto(); cleanupAll(); },
      resume() { paused = false; rebuild(); },
    };
  },
};

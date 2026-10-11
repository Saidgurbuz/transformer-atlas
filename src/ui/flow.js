// Motion kit: animated vector strips, flying clones, number tickers, stroke drawing,
// a step sequencer with its playbar, and small helpers. Plain ES module, no dependencies.
//
// Every animation
//   - returns a Promise that resolves true when it finishes and false when it is cancelled
//     or fails (the error is logged and the element is left in a consistent state);
//   - has .cancel() on that Promise and accepts { signal } (an AbortSignal);
//   - under prefers-reduced-motion jumps straight to its end state;
//   - runs on one shared requestAnimationFrame loop, so cancelling or destroying stops it.
//
//   const strip = vectorStrip({ values, cell: 6, mode: 'diverging' });   // canvas of colour cells
//   strip.update(next, { animate: true });                              // colours tween ~300 ms
//   await flyClone(rowCanvas, slotEl, { onDone: () => strip2.update(row) });
//   const t = ticker(normEl, 0, 4.21, { format: (v) => `‖x‖ ${v.toFixed(2)}` });
//   const seq = sequence([{ label: 'Query', duration: 900, run: (p) => paintQuery(p) }]);
//   sequencePlaybar(seq).el                                              // prev / play / next / pills
import { h, fmt, clamp } from './dom.js';
import { diverging, sequential, cssVar, kindColor, onThemeChange } from './colors.js';
import { showTip, hideTip } from './tooltip.js';
import { reducedMotion, ease } from './anim.js';

export { ease, reducedMotion };

// One injected stylesheet for the kit's own classes (fx-*). Colours come from tokens only,
// except the on-accent text colour, which falls back to white until main.css defines a token.
if (typeof document !== 'undefined' && !document.getElementById('fx-flow-style')) {
  const style = document.createElement('style');
  style.id = 'fx-flow-style';
  style.textContent = `
.fx-strip { display: block; width: max-content; max-width: 100%; position: relative; }
.fx-strip-title { font: 600 var(--fs-xs)/1.3 var(--font-mono); color: var(--fg-2); margin-bottom: 4px; }
.fx-canvas { display: block; max-width: 100%; height: auto; }
.fx-fly { border-radius: 4px; outline: 1px solid var(--line-strong); will-change: transform; }
.fx-playbar .steps { padding-bottom: 7px; }
.fx-pill { position: relative; }
.fx-playbar .fx-pill[aria-current="step"] { color: var(--on-accent, #fff); }
.fx-bar { position: absolute; left: 9px; right: 9px; bottom: -5px; height: 2px; border-radius: 2px; background: var(--accent); transform-origin: 0 50%; transform: scaleX(0); pointer-events: none; }
`;
  document.head.append(style);
}

// ----------------------------------------------------------------- easing

// Easing functions take t in [0, 1] and return a value that is 0 at 0 and 1 at 1.
export const easing = {
  linear: (t) => t,
  smooth: (t) => t * t * (3 - 2 * t),                       // smoothstep
  smoother: (t) => t * t * t * (t * (t * 6 - 15) + 10),     // smootherstep
  inOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  inOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,
  inCubic: (t) => t * t * t,
  outCubic: (t) => 1 - Math.pow(1 - t, 3),
  outQuart: (t) => 1 - Math.pow(1 - t, 4),
  outExpo: (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  outBack: (t) => { const c = 1.4; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); },
};

// ------------------------------------------------------------ frame loop

const frames = new Map();           // frame fn -> onError (or null)
let rafId = 0;

function runFrames(now) {
  rafId = 0;
  for (const [fn, onError] of Array.from(frames)) {
    if (!frames.has(fn)) continue;   // stopped by an earlier callback in this frame
    let keep = false;
    try {
      keep = fn(now);
    } catch (err) {
      frames.delete(fn);
      console.error('[flow] animation failed', err);
      try { onError?.(err); } catch (err2) { console.error('[flow]', err2); }
      continue;
    }
    if (!keep) frames.delete(fn);
  }
  if (frames.size && !rafId) rafId = requestAnimationFrame(runFrames);
}

// fn(now) returns true to keep running. If it throws, onError(err) runs and the frame is dropped.
// Returns a function that removes the frame.
function onFrame(fn, onError = null) {
  frames.set(fn, onError);
  if (!rafId) rafId = requestAnimationFrame(runFrames);
  return () => { frames.delete(fn); };
}

// ---------------------------------------------------------- theme tokens

let epoch = 0;                      // bumped on every theme change; caches key on it
onThemeChange(() => { epoch += 1; });

let tokCache = null, tokEpoch = -1;
function tokens() {
  if (!tokCache || tokEpoch !== epoch) {
    tokCache = {
      accent: cssVar('--accent'),
      fg: cssVar('--fg'),
      fg3: cssVar('--fg-3'),
      mask: cssVar('--c-mask'),
      surface2: cssVar('--surface-2'),
      font: cssVar('--font-mono') || 'monospace',
    };
    tokEpoch = epoch;
  }
  return tokCache;
}

// Colour lookup tables: 257 entries per (mode, kind), rebuilt on theme change.
const LUT_N = 256;
const luts = new Map();
let lutEpoch = -1;
function lutFor(mode, kind) {
  if (lutEpoch !== epoch) { luts.clear(); lutEpoch = epoch; }
  const key = `${mode}|${kind || ''}`;
  let lut = luts.get(key);
  if (!lut) {
    lut = new Array(LUT_N + 1);
    for (let i = 0; i <= LUT_N; i++) {
      const u = i / LUT_N;
      lut[i] = mode === 'sequential' ? sequential(u, kind || undefined) : diverging(u * 2 - 1);
    }
    luts.set(key, lut);
  }
  return lut;
}

// rng = [lo, hi]. Diverging ranges are symmetric: [-R, R].
function colorAt(v, lut, mode, rng) {
  const x = Number.isFinite(v) ? v : 0;
  let u;
  if (mode === 'sequential') u = (x - rng[0]) / ((rng[1] - rng[0]) || 1);
  else u = 0.5 + 0.5 * (x / (rng[1] || 1));
  return lut[Math.round(clamp(u, 0, 1) * LUT_N)];
}

function rangeFor(r, vals, mode) {
  if (mode === 'sequential') {
    if (Array.isArray(r)) return [Number(r[0]), Number(r[1]) || 1];
    if (typeof r === 'number') return [0, r || 1];
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo === Infinity) return [0, 1];
    return [Math.min(0, lo), hi > 0 ? hi : 1];
  }
  let R;
  if (typeof r === 'number') R = Math.abs(r);
  else if (Array.isArray(r)) R = Math.max(Math.abs(r[0]), Math.abs(r[1]));
  else {
    R = 0;
    for (const v of vals) if (Number.isFinite(v)) R = Math.max(R, Math.abs(v));
  }
  R = R || 1;
  return [-R, R];
}

const mixVal = (a, b, e) => {
  if (a === b) return a;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return e < 0.5 ? a : b;   // masked (-Infinity) cells switch mid-way
  return a + (b - a) * e;
};

const toF64 = (v) => Float64Array.from(v ?? []);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

// ---------------------------------------------------------- cancellable task

// Wraps animation work as a Promise with .cancel(). run(finish) starts the work and returns a
// stop() that releases timers and frames. onCancel() sets the end state. If run throws, the
// error is logged and the Promise resolves false.
function task(run, { signal, onCancel } = {}) {
  let done = false, stop = null, resolveFn;
  const promise = new Promise((res) => { resolveFn = res; });
  const finish = (value) => {
    if (done) return;
    done = true;
    signal?.removeEventListener('abort', cancel);
    const s = stop; stop = null;
    try { s?.(); } catch (err) { console.error('[flow]', err); }
    resolveFn(value);
  };
  function cancel() {
    if (done) return;
    try { onCancel?.(); } catch (err) { console.error('[flow]', err); }
    finish(false);
  }
  promise.cancel = cancel;
  if (signal?.aborted) { finish(false); return promise; }
  signal?.addEventListener('abort', cancel, { once: true });
  let s = null;
  try {
    s = run(finish);
  } catch (err) {
    console.error('[flow]', err);
    finish(false);
    return promise;
  }
  if (done) s?.();
  else stop = s || null;
  return promise;
}

// ------------------------------------------------------------ vectorStrip

const roundRectPath = (ctx, x, y, w, h, r) => { if (ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h); };

// A vector drawn as a row (orientation 'h') or column ('v') of colour cells on one canvas.
//   vectorStrip({ values, orientation: 'h', cell: 14, mode: 'diverging' | 'sequential', kind,
//                 labels, title, range, pad, thickness, duration, digits, onHover, onClick })
// Returns { el, canvas, values, update(values, { animate, range, duration }), pulse(),
//           highlight(indices), destroy() }.
// Masked cells (-Infinity) are drawn hatched. Hover shows index and value.
// pulse() draws a glow: a ring outside the cells when pad >= 3, or a tint with an inner
// outline when the strip has less padding than that.
export function vectorStrip(options = {}) {
  const o = {
    values: [], orientation: 'h', cell: 14, thickness: null, pad: 8, mode: 'diverging', kind: null,
    labels: null, title: null, range: null, duration: 300, digits: 3, ariaLabel: 'Vector strip',
    onHover: null, onClick: null, ...options,
  };
  const horizontal = o.orientation !== 'v';
  let values = toF64(o.values);
  let shown = values.slice();
  let rng = rangeFor(o.range, values, o.mode);
  let hover = -1;
  let hl = new Set();
  let pulseE = null;
  let destroyed = false;
  let stopTween = null, stopPulse = null, sizeW = -1, ringCache = null;

  const canvas = document.createElement('canvas');
  canvas.className = 'fx-canvas';
  // Strips are small and redrawn every frame. A CPU-backed context avoids a GPU flush per redraw.
  const ctx2d = () => canvas.getContext('2d', { willReadFrequently: true });
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', o.ariaLabel);
  const titleEl = h('div', { class: 'fx-strip-title', hidden: !o.title }, o.title || '');
  const el = h('div', { class: ['fx-strip', horizontal ? 'fx-h' : 'fx-v'] }, titleEl, canvas);

  function computeGeo() {
    const n = values.length, cell = o.cell, thick = o.thickness ?? cell, pad = o.pad;
    let labelBox = 0, labelW = 0;
    if (o.labels && o.labels.length) {
      const cx = ctx2d();
      cx.font = `11px ${tokens().font}`;
      if (horizontal) labelBox = 16;
      else labelW = Math.ceil(Math.max(...o.labels.map((l) => cx.measureText(String(l ?? '')).width))) + 8;
    }
    if (horizontal) {
      return { n, cw: cell, ch: thick, x0: pad, y0: pad, W: pad * 2 + n * cell, H: pad * 2 + thick + labelBox };
    }
    return { n, cw: thick, ch: cell, x0: pad + labelW, y0: pad, W: pad * 2 + labelW + thick, H: pad * 2 + n * cell };
  }
  let geo = computeGeo();

  const cellXY = (i) => (horizontal ? [geo.x0 + i * geo.cw, geo.y0] : [geo.x0, geo.y0 + i * geo.ch]);

  function ringColor() {
    if (!ringCache || ringCache.epoch !== epoch) {
      ringCache = { epoch, color: o.kind ? kindColor(o.kind) : tokens().accent };
    }
    return ringCache.color;
  }

  function draw() {
    if (destroyed) return;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(geo.W * dpr), bh = Math.round(geo.H * dpr);
    if (canvas.width !== bw) canvas.width = bw;
    if (canvas.height !== bh) canvas.height = bh;
    if (sizeW !== geo.W) { canvas.style.width = `${geo.W}px`; sizeW = geo.W; }
    const ctx = ctx2d();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, geo.W, geo.H);
    const t = tokens(), lut = lutFor(o.mode, o.kind);
    const { n, cw, ch, x0, y0 } = geo;
    const gap = Math.min(cw, ch) > 6 ? 1 : 0;
    const w1 = cw - gap, h1 = ch - gap;

    for (let i = 0; i < n; i++) {
      const [x, y] = cellXY(i);
      const v = shown[i];
      if (v === -Infinity) {
        ctx.fillStyle = t.surface2;
        ctx.fillRect(x, y, w1, h1);
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, w1, h1);
        ctx.clip();
        ctx.strokeStyle = t.mask;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let k = -h1; k < w1; k += 4) { ctx.moveTo(x + k, y + h1); ctx.lineTo(x + k + h1, y); }
        ctx.stroke();
        ctx.restore();
      } else {
        ctx.fillStyle = colorAt(v, lut, o.mode, rng);
        ctx.fillRect(x, y, w1, h1);
      }
    }

    for (const i of hl) {
      if (i < 0 || i >= n) continue;
      const [x, y] = cellXY(i);
      ctx.strokeStyle = t.accent;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, w1 - 2, h1 - 2);
    }
    if (hover >= 0 && hover < n) {
      const [x, y] = cellXY(hover);
      ctx.strokeStyle = t.fg;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x + 0.75, y + 0.75, w1 - 1.5, h1 - 1.5);
    }

    if (o.labels && o.labels.length) {
      ctx.font = `11px ${t.font}`;
      for (let i = 0; i < n; i++) {
        const [x, y] = cellXY(i);
        ctx.fillStyle = hl.has(i) ? t.fg : t.fg3;
        const txt = String(o.labels[i] ?? '');
        if (horizontal) {
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText(txt, x + cw / 2, y0 + ch + 3);
        } else {
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';
          ctx.fillText(txt, x0 - 6, y + ch / 2);
        }
      }
    }

    if (pulseE !== null) {
      const k = 1 - pulseE;
      const wAll = horizontal ? n * cw : cw, hAll = horizontal ? ch : n * ch;
      ctx.strokeStyle = ringColor();
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (o.pad >= 3) {
        // A ring outside the cells that grows and fades. It needs the padding to draw into.
        const grow = Math.max(0, o.pad - 3) * pulseE;
        ctx.globalAlpha = 0.9 * k;
        roundRectPath(ctx, x0 - 2 - grow, y0 - 2 - grow, wAll + 4 + 2 * grow, hAll + 4 + 2 * grow, 6);
      } else {
        // Too little padding for a ring outside the cells: tint the strip, then outline it from inside.
        ctx.globalAlpha = 0.3 * k;
        ctx.fillStyle = ringColor();
        ctx.fillRect(x0, y0, wAll, hAll);
        ctx.globalAlpha = 0.9 * k;
        roundRectPath(ctx, x0 + 1, y0 + 1, wAll - 2, hAll - 2, 3);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  function stopTweenNow() { stopTween?.(); stopTween = null; }
  function stopPulseNow() { stopPulse?.(); stopPulse = null; pulseE = null; }

  function update(next, opts = {}) {
    if (destroyed) return api;
    const { animate = true, range, duration = o.duration } = opts;
    const target = toF64(next);
    const toRng = rangeFor(range ?? o.range, target, o.mode);
    values = target;
    if (target.length !== shown.length) {
      stopTweenNow();
      shown = target.slice();
      rng = toRng;
      geo = computeGeo();
      draw();
      return api;
    }
    if (!animate || reducedMotion() || duration <= 0) {
      stopTweenNow();
      shown = target.slice();
      rng = toRng;
      draw();
      return api;
    }
    const from = shown.slice(), r0 = rng.slice();
    stopTweenNow();
    let t0 = null;
    stopTween = onFrame((now) => {
      if (destroyed) return false;
      if (t0 === null) t0 = now;
      const t = clamp((now - t0) / duration, 0, 1);
      const e = easing.inOutCubic(t);
      for (let i = 0; i < target.length; i++) shown[i] = mixVal(from[i], target[i], e);
      rng = [r0[0] + (toRng[0] - r0[0]) * e, r0[1] + (toRng[1] - r0[1]) * e];
      draw();
      if (t < 1) return true;
      stopTween = null;
      shown = target.slice();
      rng = toRng;
      draw();
      return false;
    }, () => { stopTween = null; shown = target.slice(); rng = toRng; draw(); });
    return api;
  }

  // A glow that grows out of the strip and fades. No-op under reduced motion.
  function pulse({ duration = 700 } = {}) {
    if (destroyed || reducedMotion()) return api;
    stopPulseNow();
    let t0 = null;
    stopPulse = onFrame((now) => {
      if (destroyed) return false;
      if (t0 === null) t0 = now;
      const t = clamp((now - t0) / duration, 0, 1);
      pulseE = easing.outCubic(t);
      draw();
      if (t < 1) return true;
      stopPulse = null;
      pulseE = null;
      draw();
      return false;
    }, () => { stopPulse = null; pulseE = null; draw(); });
    return api;
  }

  function highlight(indices) {
    if (destroyed) return api;
    hl = new Set(indices == null ? [] : (typeof indices === 'number' ? [indices] : Array.from(indices)));
    draw();
    return api;
  }

  function cellIndex(ev) {
    const r = canvas.getBoundingClientRect();
    if (!r.width || !r.height) return -1;
    const x = (ev.clientX - r.left) * (geo.W / r.width) - geo.x0;
    const y = (ev.clientY - r.top) * (geo.H / r.height) - geo.y0;
    const along = horizontal ? x : y, across = horizontal ? y : x;
    const crossSize = horizontal ? geo.ch : geo.cw, alongSize = horizontal ? geo.cw : geo.ch;
    if (across < 0 || across > crossSize || along < 0) return -1;
    const i = Math.floor(along / alongSize);
    return i < geo.n ? i : -1;
  }

  const tipHtml = (i) => {
    const lab = o.labels ? `${esc(o.labels[i] ?? '')} ` : '';
    const name = o.title ? `${esc(o.title)} · ` : '';
    return `${name}${lab}[${i}] = <b>${fmt(values[i], o.digits)}</b>`;
  };

  canvas.addEventListener('pointermove', (ev) => {
    const i = cellIndex(ev);
    if (i !== hover) { hover = i; draw(); }
    if (i < 0) { hideTip(); return; }
    showTip(ev.clientX, ev.clientY, tipHtml(i));
    o.onHover?.(i, values[i]);
  });
  canvas.addEventListener('pointerleave', () => {
    if (hover !== -1) { hover = -1; draw(); }
    hideTip();
  });
  canvas.addEventListener('click', (ev) => {
    const i = cellIndex(ev);
    if (i >= 0) o.onClick?.(i, values[i]);
  });
  if (o.onClick) canvas.style.cursor = 'pointer';

  const offTheme = onThemeChange(() => { if (!destroyed) draw(); });

  const api = {
    el,
    canvas,
    get values() { return values; },
    update,
    pulse,
    highlight,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      stopTweenNow();
      stopPulseNow();
      offTheme();
      if (hover >= 0) hideTip();
      el.remove();
    },
  };
  draw();
  return api;
}

// ------------------------------------------------------------------- flyClone

const centre = (r) => [r.left + r.width / 2, r.top + r.height / 2];
const quad = (S, C, E, e) => {
  const u = 1 - e;
  return [u * u * S[0] + 2 * u * e * C[0] + e * e * E[0], u * u * S[1] + 2 * u * e * C[1] + e * e * E[1]];
};

// Control point of the arc from S to E. It bows upward (or to the left for vertical moves).
function bowControl(S, E) {
  const dx = E[0] - S[0], dy = E[1] - S[1];
  const len = Math.hypot(dx, dy) || 1;
  let nx = -dy / len, ny = dx / len;
  if (ny > 0) { nx = -nx; ny = -ny; }
  const bow = Math.min(140, len * 0.3);
  return [(S[0] + E[0]) / 2 + nx * bow, (S[1] + E[1]) / 2 + ny * bow];
}

// SVG path data for the arc flyClone follows. Rects are in one coordinate system.
export function arcPathD(fromRect, toRect, { arc = true } = {}) {
  const S = centre(fromRect), E = centre(toRect);
  if (!arc) return `M${S[0]} ${S[1]} L${E[0]} ${E[1]}`;
  const C = bowControl(S, E);
  return `M${S[0]} ${S[1]} Q${C[0]} ${C[1]} ${E[0]} ${E[1]}`;
}

function rectOf(el) {
  if (!el || !el.isConnected) return null;
  const r = el.getBoundingClientRect();
  return r.width || r.height ? r : null;
}

// Bumped on every scroll (capture, so scrolling inside a container counts too) and resize.
// Flights take their rects again when it changes, instead of reading the layout every frame.
let viewEpoch = 0;
if (typeof window !== 'undefined') {
  window.addEventListener('scroll', () => { viewEpoch += 1; }, { capture: true, passive: true });
  window.addEventListener('resize', () => { viewEpoch += 1; }, { passive: true });
}

// Copies a canvas bitmap into another canvas of the same backing size.
function paintCanvasInto(copy, src) {
  copy.width = src.width;
  copy.height = src.height;
  copy.getContext('2d')?.drawImage(src, 0, 0);
}

function makeClone(src, r) {
  let node;
  if (src instanceof HTMLCanvasElement) {
    node = document.createElement('canvas');
    paintCanvasInto(node, src);
  } else {
    node = src.cloneNode(true);
    node.removeAttribute?.('id');
    node.querySelectorAll?.('[id]').forEach((n) => n.removeAttribute('id'));
    // cloneNode does not copy canvas bitmaps, so pair each live canvas with its copy in document order.
    const live = src.querySelectorAll?.('canvas') ?? [];
    const copies = node.querySelectorAll?.('canvas') ?? [];
    live.forEach((c, i) => { if (copies[i]) paintCanvasInto(copies[i], c); });
  }
  node.classList.add('fx-fly');
  node.setAttribute('aria-hidden', 'true');
  Object.assign(node.style, {
    position: 'fixed', left: '0px', top: '0px', width: `${r.width}px`, height: `${r.height}px`,
    margin: '0', transformOrigin: '0 0', pointerEvents: 'none', zIndex: '90',
  });
  return node;
}

// Snapshot fromEl (cloneNode for DOM and SVG, bitmap copy for canvas and for the canvases
// inside a DOM element), fly it along an arc to toEl while scaling to fit, then remove it.
// The ghost is fixed to the viewport. Its rects are taken again after any scroll or resize, so it
// still lands on the target when the page or a scroll container moves during the flight.
// Other layout changes during a flight are not followed. Options:
//   duration (600), easing (smooth), arc (true), lift (true), fadeOut (0..1, fraction of the
//   flight in which the clone fades out), onProgress(e, t), onDone() (runs before the clone
//   is removed, so the target can be updated with no flash), signal.
export function flyClone(fromEl, toEl, options = {}) {
  const { duration = 600, easing: ez = easing.smooth, onDone = null, onProgress = null,
    arc = true, lift = true, fadeOut = 0, signal } = options;
  return task((finish) => {
    let a = rectOf(fromEl), b = rectOf(toEl);
    if (!a || !b) { finish(false); return null; }
    if (reducedMotion()) { onDone?.(); finish(true); return null; }
    const clone = makeClone(fromEl, a);
    const w0 = a.width, h0 = a.height;
    document.body.append(clone);
    const place = (e) => {
      const S = centre(a), E = centre(b);
      const [cx, cy] = arc ? quad(S, bowControl(S, E), E, e) : [S[0] + (E[0] - S[0]) * e, S[1] + (E[1] - S[1]) * e];
      const k = lift ? 1 + 0.05 * Math.sin(Math.PI * e) : 1;
      const sx = w0 ? b.width / w0 : 1, sy = h0 ? b.height / h0 : 1;
      const scx = (1 + (sx - 1) * e) * k, scy = (1 + (sy - 1) * e) * k;
      const x = cx - (w0 * scx) / 2, y = cy - (h0 * scy) / 2;
      clone.style.transform = `translate3d(${x.toFixed(2)}px, ${y.toFixed(2)}px, 0) scale(${scx.toFixed(4)}, ${scy.toFixed(4)})`;
      if (fadeOut > 0) clone.style.opacity = String(1 - clamp((e - (1 - fadeOut)) / fadeOut, 0, 1));
    };
    place(0);
    let t0 = null, seen = viewEpoch;
    const stopFrame = onFrame((now) => {
      if (t0 === null) t0 = now;
      const t = clamp((now - t0) / duration, 0, 1);
      if (seen !== viewEpoch) {        // the page scrolled or resized: take the rects again
        seen = viewEpoch;
        a = rectOf(fromEl) || a;
        b = rectOf(toEl) || b;
      }
      const e = ez(t);
      place(e);
      onProgress?.(e, t);
      if (t < 1) return true;
      onDone?.();
      finish(true);
      return false;
    }, () => finish(false));
    return () => { stopFrame(); clone.remove(); };
  }, { signal });
}

// --------------------------------------------------------------------- ticker

const tickerOwner = new WeakMap();   // element -> { halt() } for the running ticker
const tickerValue = new WeakMap();   // element -> last number shown

// Animates the number in el from `from` to `to` (default: the last value shown, or `to`).
// Starting a new ticker on the same element replaces the running one.
export function ticker(el, from, to, options = {}) {
  const { duration = 700, digits = 2, format = null, easing: ez = easing.outCubic, signal } = options;
  const show = (v) => (format ? format(v) : fmt(v, digits));
  const write = (v) => { tickerValue.set(el, v); el.textContent = show(v); };
  const start = from ?? tickerValue.get(el) ?? to;
  tickerOwner.get(el)?.halt();
  return task((finish) => {
    const rec = { halt: () => finish(false) };
    tickerOwner.set(el, rec);
    const release = () => { if (tickerOwner.get(el) === rec) tickerOwner.delete(el); };
    if (reducedMotion() || duration <= 0 || start === to) {
      write(to);
      release();
      finish(true);
      return null;
    }
    let t0 = null;
    const stopFrame = onFrame((now) => {
      if (t0 === null) t0 = now;
      const t = clamp((now - t0) / duration, 0, 1);
      write(start + (to - start) * ez(t));
      if (t < 1) return true;
      release();
      finish(true);
      return false;
    }, () => { release(); finish(false); });
    return () => { stopFrame(); release(); };
  }, { signal, onCancel: () => write(to) });
}

// ------------------------------------------------------------------ drawPath

// Draws an SVG geometry element (path, line, polyline...) stroke by stroke.
export function drawPath(path, options = {}) {
  const { duration = 800, delay: wait = 0, easing: ez = easing.inOutCubic, signal } = options;
  return task((finish) => {
    const len = path && typeof path.getTotalLength === 'function' ? path.getTotalLength() : 0;
    if (!len || reducedMotion()) { finish(true); return null; }
    const prev = { dash: path.style.strokeDasharray, offset: path.style.strokeDashoffset };
    const restore = () => { path.style.strokeDasharray = prev.dash; path.style.strokeDashoffset = prev.offset; };
    path.style.strokeDasharray = `${len}px ${len}px`;
    path.style.strokeDashoffset = `${len}px`;
    let stopFrame = null, timer = null;
    const begin = () => {
      timer = null;
      let t0 = null;
      stopFrame = onFrame((now) => {
        if (t0 === null) t0 = now;
        const t = clamp((now - t0) / duration, 0, 1);
        path.style.strokeDashoffset = `${(len * (1 - ez(t))).toFixed(2)}px`;
        if (t < 1) return true;
        finish(true);
        return false;
      }, () => finish(false));
    };
    if (wait > 0) timer = setTimeout(begin, wait);
    else begin();
    return () => { if (timer) clearTimeout(timer); stopFrame?.(); restore(); };
  }, { signal });
}

// ------------------------------------------------------- delay, stagger, scope

// Cancellable wait. Under reduced motion it resolves at once.
export function delay(ms = 0, { signal } = {}) {
  const wait = reducedMotion() ? 0 : Math.max(0, ms);
  return task((finish) => {
    const t = setTimeout(() => finish(true), wait);
    return () => clearTimeout(t);
  }, { signal });
}

// fn(item, index) runs for each item, delayMs apart. If fn returns a Promise, the result
// waits for it. The result is true when every item completed. It is false when cancelled,
// when an item throws (the remaining items are not started), or when a returned Promise rejects.
// Cancelling cancels the pending calls and any returned task that has .cancel().
export function stagger(items, fn, delayMs = 80, { signal } = {}) {
  const list = Array.from(items ?? []);
  const gap = reducedMotion() ? 0 : Math.max(0, delayMs);
  return task((finish) => {
    if (!list.length) { finish(true); return null; }
    const timers = [], kids = [];
    let launched = 0, halted = false;
    const launch = (item, i) => {
      if (halted) return;
      let r;
      try {
        r = fn(item, i);
      } catch (err) {
        console.error('[flow] stagger item failed', err);
        halted = true;
        finish(false);
        return;
      }
      if (r && typeof r.then === 'function') kids.push(r);
      launched += 1;
      if (launched === list.length) {
        Promise.allSettled(kids).then((results) => {
          for (const res of results) if (res.status === 'rejected') console.error('[flow] stagger item rejected', res.reason);
          finish(results.every((res) => res.status === 'fulfilled' && res.value !== false));
        });
      }
    };
    list.forEach((item, i) => {
      if (halted) return;
      if (gap === 0) launch(item, i);
      else timers.push(setTimeout(() => launch(item, i), i * gap));
    });
    return () => {
      halted = true;
      timers.forEach(clearTimeout);
      kids.forEach((k) => k.cancel?.());
    };
  }, { signal });
}

// Collects tasks so one call cancels them all, e.g. when a view is torn down.
export function motionScope() {
  const set = new Set();
  return {
    add(p) {
      set.add(p);
      p.finally?.(() => set.delete(p));
      return p;
    },
    cancel() {
      for (const p of Array.from(set)) p.cancel?.();
      set.clear();
    },
    get size() { return set.size; },
  };
}

// ---------------------------------------------------------------- sequence

function listeners() {
  const map = new Map();
  return {
    on(ev, fn) {
      if (!map.has(ev)) map.set(ev, new Set());
      map.get(ev).add(fn);
      return () => { map.get(ev)?.delete(fn); };
    },
    emit(ev, payload) {
      for (const fn of Array.from(map.get(ev) || [])) {
        try { fn(payload); } catch (err) { console.error(`[flow] ${ev} listener failed`, err); }
      }
    },
    clear() { map.clear(); },
  };
}

// A timeline of steps. Each step is { label, duration (ms), run(progress, step), easing? }.
// run receives eased progress in [0, 1] and is called every frame while the step plays,
// with 0 when the step starts and 1 when it ends. Steps before the current index are kept
// at their end state and steps after it at their start state, so seek() always lands on
// a consistent picture.
//
//   const seq = sequence(steps, { onStep: (i, step) => {} });
//   seq.play(); seq.pause(); seq.next(); seq.prev(); seq.seek(3, 0.5); seq.refresh();
//   seq.on('step' | 'progress' | 'play' | 'pause' | 'end' | 'error', fn); seq.destroy();
//   await seq.stopped();   // true when the last step ended; false after a pause, an error or destroy
//
// next(), prev() and the playbar's step pills land on the END state of their step, so the
// picture and the caption agree. play() continues from the current position. A finished step
// replays from its start, and the last step replays the whole sequence.
// A step whose run() throws is logged and emits 'error'; playback then stops with 'pause'.
export function sequence(steps, options = {}) {
  const { onStep = null, easing: defaultEase = easing.inOutCubic } = options;
  const list = (steps || []).map((s, i) => ({ duration: 800, ...s, index: i }));
  const n = list.length;
  const bus = listeners();
  let index = 0, raw = 0, playing = false, dead = false, stopFrame = null, startT = 0;
  let waiters = [];
  const state = new Array(n).fill(null);
  const easeOf = (s) => (typeof s.easing === 'function' ? s.easing : defaultEase);
  const durOf = (k) => Math.max(0, list[k].duration || 0);

  function settled(value) {
    const w = waiters;
    waiters = [];
    w.forEach((res) => res(value));
  }
  function notify(fn) {
    try { fn(); } catch (err) { console.error('[flow] onStep failed', err); }
  }
  // Runs one step's run(). Returns false when it threw (the error is logged and emitted).
  function call(s, p) {
    try {
      s.run?.(p, s);
      return true;
    } catch (err) {
      console.error(`[flow] step "${s.label ?? s.index}" failed`, err);
      bus.emit('error', { index: s.index, error: err });
      return false;
    }
  }
  function paint(k, p) {
    const s = list[k];
    const q = clamp(p, 0, 1);
    state[k] = q >= 1 ? 'end' : q <= 0 ? 'start' : 'mid';
    return call(s, easeOf(s)(q));
  }
  function settle(k, which) {
    if (state[k] === which) return true;
    state[k] = which;
    return call(list[k], which === 'end' ? 1 : 0);
  }
  function jump(i, p) {
    const j = clamp(Math.round(i), 0, Math.max(0, n - 1));
    const before = index;
    for (let k = 0; k < j; k++) settle(k, 'end');
    for (let k = j + 1; k < n; k++) settle(k, 'start');
    index = j;
    raw = clamp(p, 0, 1);
    if (!n) return;
    paint(j, raw);
    if (j !== before) {
      bus.emit('step', { index: j, step: list[j] });
      notify(() => onStep?.(j, list[j]));
    }
    bus.emit('progress', { index: j, progress: raw });
  }

  for (let k = 0; k < n; k++) settle(k, 'start');

  // Stops playback after a failure. The frame that calls this returns false, so it is dropped.
  function halt() {
    playing = false;
    stopFrame?.();
    stopFrame = null;
    bus.emit('pause', { index, progress: raw });
    settled(false);
    return false;
  }

  function tick(now) {
    if (dead || !playing) return false;
    const d = durOf(index);
    raw = d === 0 ? 1 : clamp((now - startT) / d, 0, 1);
    if (!paint(index, raw)) return halt();
    bus.emit('progress', { index, progress: raw });
    if (raw < 1) return true;
    if (index < n - 1) {
      index += 1;
      raw = 0;
      startT = now;
      if (!paint(index, 0)) return halt();
      bus.emit('step', { index, step: list[index] });
      notify(() => onStep?.(index, list[index]));
      bus.emit('progress', { index, progress: 0 });
      return true;
    }
    playing = false;
    stopFrame = null;
    bus.emit('end', { index });
    settled(true);
    return false;
  }

  function play() {
    if (dead || playing || !n) return api;
    if (reducedMotion()) {
      // No motion: every step lands on its end state at once.
      jump(n - 1, 1);
      for (let k = 0; k < n; k++) settle(k, 'end');
      bus.emit('end', { index });
      settled(true);
      return api;
    }
    // A finished step replays from its start; the last step replays the whole sequence.
    if (raw >= 1) jump(index === n - 1 ? 0 : index, 0);
    playing = true;
    startT = performance.now() - raw * durOf(index);
    stopFrame = onFrame(tick, () => halt());
    bus.emit('play', { index });
    return api;
  }

  function pause() {
    if (!playing) return api;
    playing = false;
    stopFrame?.();
    stopFrame = null;
    bus.emit('pause', { index, progress: raw });
    settled(false);
    return api;
  }

  function seek(i, p = 0) {
    if (dead) return api;
    pause();
    jump(i, p);
    return api;
  }

  const api = {
    play,
    pause,
    seek,
    // Moving one step lands on the end state of the new step, so its picture and caption agree.
    next() { if (index < n - 1) seek(index + 1, 1); return api; },
    prev() { if (index > 0) seek(index - 1, 1); return api; },
    // Re-runs every step for the current index, e.g. after the data behind the steps changed.
    refresh() {
      if (dead) return api;
      state.fill(null);
      jump(index, raw);
      return api;
    },
    // Resolves at the next stop of playback (see the header comment).
    stopped() { return new Promise((res) => { waiters.push(res); }); },
    on: bus.on,
    destroy() {
      if (dead) return;
      pause();
      dead = true;
      bus.clear();
      settled(false);
    },
    get index() { return index; },
    get progress() { return raw; },
    get playing() { return playing; },
    get count() { return n; },
    get steps() { return list; },
  };
  if (n) paint(0, 0);
  return api;
}

// Playbar for a sequence(): prev / play / next, one pill per step, and a thin progress line
// under the current pill. Uses the playbar classes from main.css.
export function sequencePlaybar(ctl, options = {}) {
  const { labels = null } = options;
  const subs = [];
  const prev = h('button', { type: 'button', class: 'btn icon', title: 'Previous step', 'aria-label': 'Previous step', onClick: () => ctl.prev() }, '‹');
  const playBtn = h('button', { type: 'button', class: 'btn', title: 'Play from this step', onClick: () => (ctl.playing ? ctl.pause() : ctl.play()) }, '▶ Play');
  const nextBtn = h('button', { type: 'button', class: 'btn icon', title: 'Next step', 'aria-label': 'Next step', onClick: () => ctl.next() }, '›');
  const bars = [];
  const pills = ctl.steps.map((s, k) => {
    const bar = h('span', { class: 'fx-bar', 'aria-hidden': 'true' });
    bars.push(bar);
    return h('button', { type: 'button', class: 'step-dot fx-pill', onClick: () => ctl.seek(k, 1) },
      `${k + 1}. ${labels?.[k] ?? s.label ?? ''}`, bar);
  });
  const el = h('div', { class: 'playbar fx-playbar', role: 'group', 'aria-label': 'Step controls' },
    prev, playBtn, nextBtn, h('div', { class: 'steps' }, pills));

  // Pills and bars: finished steps are full, later steps are empty.
  function paintSteps() {
    const i = ctl.index;
    pills.forEach((p, k) => {
      if (k === i) p.setAttribute('aria-current', 'step');
      else p.removeAttribute('aria-current');
      p.classList.toggle('done', k < i);
      bars[k].style.transform = k < i ? 'scaleX(1)' : 'scaleX(0)';
    });
  }
  // Buttons. Writes only when a value changes, so this is cheap on every progress event.
  function paintButtons() {
    const i = ctl.index, last = ctl.count - 1;
    if (prev.disabled !== (i === 0)) prev.disabled = i === 0;
    if (nextBtn.disabled !== (i === last)) nextBtn.disabled = i === last;
    const label = ctl.playing ? '❚❚ Pause' : i === last && ctl.progress >= 1 ? '↻ Replay' : '▶ Play';
    if (playBtn.textContent !== label) playBtn.textContent = label;
  }

  subs.push(ctl.on('step', () => { paintSteps(); paintButtons(); }));
  subs.push(ctl.on('progress', ({ index, progress }) => { bars[index].style.transform = `scaleX(${progress})`; paintButtons(); }));
  subs.push(ctl.on('play', paintButtons));
  subs.push(ctl.on('pause', paintButtons));
  subs.push(ctl.on('end', paintButtons));
  subs.push(ctl.on('error', paintButtons));
  paintSteps();
  paintButtons();

  return {
    el,
    destroy() { subs.forEach((off) => off()); subs.length = 0; },
  };
}

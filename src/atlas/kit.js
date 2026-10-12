// 2.5D scene kit. Scenes are designed on a fixed 1600 × 900 canvas (like a slide)
// that scales to fit the screen. Everything here is plain DOM/SVG positioned in
// design units, so CSS transitions give smooth colour/position animation for free.
//
//   const st = stage2d(el);
//   st.add(vec(values, { kind: 'q', label: 'q_t' }), { x: 300, y: 400, anchor: 'c' });
//   st.arrow({ x: 360, y: 400 }, { x: 600, y: 400 }, { kind: 'q', flow: true });
//
// Elements carrying data-trace="<key>" (and data-cell="r,c,band" on cells) get the
// atlas hover card (equation + live numbers) and right-click menu (maths, code,
// inspect) automatically. Elements with data-node="<id>" zoom into that node on click.
import { h, s, fmt, diverging, sequential, onThemeChange, reducedMotion, texHTML } from '../ui/index.js';

export const W = 1600, H = 900;

// ------------------------------------------------------------------ stage

export function stage2d(el, { top = 64, bottom = 84, side = 24 } = {}) {
  const svg = s('svg', { class: 'k-svg', viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  const defs = s('defs');
  svg.append(defs);
  const layers = { back: h('div', { class: 'k-layer back' }), mid: h('div', { class: 'k-layer mid' }), front: h('div', { class: 'k-layer front' }) };
  const canvas = h('div', { class: 'k-canvas', style: { width: `${W}px`, height: `${H}px` } }, layers.back, svg, layers.mid, layers.front);
  const root = h('div', { class: 'k-stage' }, canvas);
  el.append(root);
  let k = 1;
  const scaleHooks = new Set();
  function fit() {
    const cw = el.clientWidth, ch = el.clientHeight;
    const aw = Math.max(200, cw - 2 * side), ah = Math.max(200, ch - top - bottom);
    k = Math.min(aw / W, ah / H);
    const x = (cw - W * k) / 2, y = top + (ah - H * k) / 2;
    canvas.style.transform = `translate(${x}px, ${y}px) scale(${k})`;
    scaleHooks.forEach((f) => f(k));
  }
  const ro = new ResizeObserver(fit);
  ro.observe(el);
  fit();
  const cleanups = [];

  const api = {
    root, canvas, svg, defs, layers,
    get scale() { return k; },
    onScale(fn) { scaleHooks.add(fn); return () => scaleHooks.delete(fn); },
    /** Position an element. anchor: 'tl' (default), 'c', 'tc', 'bc', 'lc', 'rc', 'tr', 'bl', 'br'. */
    add(node, { x = 0, y = 0, anchor = 'tl', layer = 'mid' } = {}) {
      node.style.position = 'absolute';
      node.style.left = `${x}px`;
      node.style.top = `${y}px`;
      const tx = { tl: '0,0', c: '-50%,-50%', tc: '-50%,0', bc: '-50%,-100%', lc: '0,-50%', rc: '-100%,-50%', tr: '-100%,0', bl: '0,-100%', br: '-100%,-100%' }[anchor] || '0,0';
      node.style.translate = tx.replace(',', ' ');
      layers[layer].append(node);
      return node;
    },
    move(node, x, y) { node.style.left = `${x}px`; node.style.top = `${y}px`; },
    /** Curved or straight arrow between two points (design units). */
    arrow(from, to, opts = {}) { const a = arrowPath(from, to, opts); svg.append(a.el); cleanups.push(a.destroy); return a; },
    /** Centre of an element in design units (after layout). */
    center(node) {
      const r = node.getBoundingClientRect(), c = canvas.getBoundingClientRect();
      return { x: (r.left + r.width / 2 - c.left) / k, y: (r.top + r.height / 2 - c.top) / k, w: r.width / k, h: r.height / k };
    },
    /** Edge points of an element: side = 'l' | 'r' | 't' | 'b'. */
    edge(node, side) {
      const b = api.center(node);
      return { l: { x: b.x - b.w / 2, y: b.y }, r: { x: b.x + b.w / 2, y: b.y }, t: { x: b.x, y: b.y - b.h / 2 }, b: { x: b.x, y: b.y + b.h / 2 } }[side];
    },
    onCleanup(fn) { cleanups.push(fn); },
    destroy() { ro.disconnect(); cleanups.forEach((f) => { try { f(); } catch { /* ignore */ } }); root.remove(); },
  };
  return api;
}

// ---------------------------------------------------------------- colours

function colorFor(v, { mode = 'diverging', range = 1, kind } = {}) {
  if (v === -Infinity) return null;
  if (mode === 'sequential') return sequential(Math.max(0, v) / (range || 1), kind);
  return diverging(v / (range || 1));
}

export function rangeOf(values, mode = 'diverging') {
  let m = 0;
  for (const v of values) if (Number.isFinite(v)) m = Math.max(m, mode === 'sequential' ? v : Math.abs(v));
  return m || 1;
}

// ----------------------------------------------------------------- vectors

/**
 * A vector as a strip of cells. Long vectors are elided in the middle (first `head`, last `tail`).
 * opts: {orientation 'v'|'h', cell (px), gap, kind, mode, range, label (html), tex (latex label),
 *        trace (key), cellOf (i -> 'r,c,b' for data-cell), head, tail, values shown on hover}
 * returns element with .update(values, {range}) and .highlight(indices)
 */
export function vec(values, opts = {}) {
  const o = { orientation: 'v', cell: 16, gap: 2, mode: 'diverging', head: 12, tail: 4, ...opts };
  const el = h('div', { class: ['k-vec', o.orientation === 'h' ? 'h' : 'v', o.kind && `kind-${o.kind}`], style: { '--cell': `${o.cell}px`, '--gap': `${o.gap}px`, '--kind': o.kind ? `var(--c-${o.kind})` : null } });
  if (o.trace) el.dataset.trace = o.trace;
  const strip = h('div', { class: 'k-cells' });
  const lab = o.label || o.tex ? h('div', { class: 'k-label', html: o.tex ? texHTML(o.tex) : o.label }) : null;
  if (lab) el.append(lab);
  el.append(strip);
  let cells = [], idx = [];
  function build(n) {
    strip.replaceChildren();
    cells = [];
    idx = n <= o.head + o.tail + 1 ? [...Array(n).keys()] : [...Array(o.head).keys(), -1, ...Array.from({ length: o.tail }, (_, i) => n - o.tail + i)];
    for (const i of idx) {
      const c = h('div', { class: i < 0 ? 'k-cell ell' : 'k-cell' }, i < 0 ? '⋮' : null);
      if (i >= 0 && o.cellOf) c.dataset.cell = o.cellOf(i);
      strip.append(c);
      cells.push(c);
    }
  }
  function paint(vals, range) {
    const r = range ?? o.range ?? rangeOf(vals, o.mode);
    idx.forEach((i, n) => {
      if (i < 0) return;
      const col = colorFor(vals[i], { mode: o.mode, range: r, kind: o.kind });
      cells[n].style.background = col ?? 'transparent';
      cells[n].classList.toggle('masked', col === null);
      cells[n].title = `${i}: ${fmt(vals[i], 4)}`;
    });
  }
  let current = Array.from(values);
  build(current.length);
  paint(current);
  // repaint on theme change; the listener removes itself once the element is gone
  const off = onThemeChange(() => { if (!el.isConnected) { off(); return; } paint(current); });
  el.update = (vals, { range } = {}) => { const v = Array.from(vals); if (v.length !== current.length) build(v.length); current = v; paint(v, range); };
  el.highlight = (set) => { const S = new Set(set); idx.forEach((i, n) => cells[n].classList.toggle('hl', S.has(i))); };
  el.values = () => current;
  el.destroy = off;
  return el;
}

// ---------------------------------------------------------------- matrices

/**
 * A matrix as a grid of cells (DOM, smooth colour transitions). For more than ~2500 cells
 * use the heatmap component instead. opts: {cell, gap, kind, mode, range, rowLabels, colLabels,
 * label/tex, trace, cellOf ((r,c) -> 'r,c,b'), tilt (2.5D perspective), showValues, digits}
 */
export function mat(data, rows, cols, opts = {}) {
  const o = { cell: 18, gap: 2, mode: 'diverging', digits: 2, ...opts };
  const el = h('div', { class: ['k-mat', o.tilt && 'tilt', o.kind && `kind-${o.kind}`], style: { '--cell': `${o.cell}px`, '--gap': `${o.gap}px`, '--kind': o.kind ? `var(--c-${o.kind})` : null } });
  if (o.trace) el.dataset.trace = o.trace;
  const grid = h('div', { class: 'k-grid', style: { gridTemplateColumns: `${o.rowLabels ? 'auto ' : ''}repeat(${cols}, var(--cell))` } });
  if (o.label || o.tex) el.append(h('div', { class: 'k-label', html: o.tex ? texHTML(o.tex) : o.label }));
  const rowBand = h('div', { class: 'k-row-band', hidden: true });
  el.append(h('div', { class: 'k-mat-body' }, grid, rowBand));
  let cells = [], current = Array.from(data);
  function build() {
    grid.replaceChildren();
    cells = [];
    if (o.colLabels) {
      if (o.rowLabels) grid.append(h('div'));
      o.colLabels.forEach((l) => grid.append(h('div', { class: 'k-axis col' }, l)));
    }
    for (let r = 0; r < rows; r++) {
      if (o.rowLabels) grid.append(h('div', { class: 'k-axis row' }, o.rowLabels[r]));
      for (let c = 0; c < cols; c++) {
        const cell = h('div', { class: 'k-cell' });
        const ref = o.cellOf ? o.cellOf(r, c) : null;
        if (ref != null) cell.dataset.cell = ref;
        grid.append(cell);
        cells.push(cell);
      }
    }
  }
  function paint(vals, range) {
    const rg = range ?? o.range ?? rangeOf(vals, o.mode);
    for (let i = 0; i < rows * cols; i++) {
      const col = colorFor(vals[i], { mode: o.mode, range: rg, kind: o.kind });
      const c = cells[i];
      c.style.background = col ?? 'transparent';
      c.classList.toggle('masked', col === null);
      if (o.showValues) c.textContent = vals[i] === -Infinity ? '−∞' : fmt(vals[i], o.digits);
    }
  }
  build();
  paint(current);
  const off = onThemeChange(() => { if (!el.isConnected) { off(); return; } paint(current); });
  el.update = (vals, { range } = {}) => { current = Array.from(vals); paint(current, range); };
  el.cellEl = (r, c) => cells[r * cols + c];
  // one band over the row (reads better than outlining many small cells)
  el.highlightRow = (r) => {
    const c0 = cells[r * cols], cN = cells[r * cols + cols - 1];
    if (!c0) { rowBand.hidden = true; return; }
    rowBand.hidden = false;
    rowBand.style.top = `${c0.offsetTop - 2}px`;
    rowBand.style.left = `${c0.offsetLeft - 3}px`;
    rowBand.style.width = `${cN.offsetLeft + cN.offsetWidth - c0.offsetLeft + 6}px`;
    rowBand.style.height = `${c0.offsetHeight + 4}px`;
  };
  el.highlightCol = (cc) => cells.forEach((c, i) => c.classList.toggle('hl', i % cols === cc));
  el.highlightCells = (list) => { const S = new Set(list.map(([r, c]) => r * cols + c)); cells.forEach((c, i) => c.classList.toggle('hl', S.has(i))); };
  el.clearHighlight = () => cells.forEach((c) => c.classList.remove('hl'));
  el.destroy = off;
  return el;
}

// --------------------------------------------------------------- operators

/** Operation node: ⊕, ×, softmax, √ ... (latex or text). */
export function op(content, { kind, size = 56, tex: isTex = true, title } = {}) {
  return h('div', { class: ['k-op', kind && `kind-${kind}`], title, style: { '--size': `${size}px`, '--kind': kind ? `var(--c-${kind})` : null }, html: isTex ? texHTML(content) : content });
}

// magnifier with a plus (drawn, so it never depends on a font having the glyph)
const ZOOM_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5 21 21M10.5 7.5v6M7.5 10.5h6"/></svg>';

/** Rounded 2.5D card; with `node` it becomes a zoom target. */
export function card({ title, sub, kind, node, w, hgt, body, trace } = {}) {
  const el = h('div', {
    class: ['k-card', node && 'zoomable', kind && `kind-${kind}`],
    style: { '--kind': kind ? `var(--c-${kind})` : null, width: w ? `${w}px` : null, height: hgt ? `${hgt}px` : null },
    dataset: node ? { node } : undefined,
  },
  title && h('div', { class: 'k-card-title' }, title),
  sub && h('div', { class: 'k-card-sub', html: sub }),
  body);
  if (trace) el.dataset.trace = trace;
  if (node) el.append(h('div', { class: 'k-zoom-badge', 'aria-hidden': 'true', html: ZOOM_ICON }));
  return el;
}

export function texLabel(latex, { size = 22, kind, display = false } = {}) {
  return h('div', { class: ['k-tex', kind && `kind-${kind}`], style: { fontSize: `${size}px`, '--kind': kind ? `var(--c-${kind})` : null }, html: texHTML(latex, { display }) });
}

export function caption(html, { w = 520 } = {}) {
  return h('div', { class: 'k-caption', style: { width: `${w}px` }, html });
}

export function chip(text, kind) {
  return h('span', { class: 'k-chip', style: { '--kind': kind ? `var(--c-${kind})` : null } }, text);
}

/** Token as a little tile: char + index. */
export function token(label, i, { selected = false, kind = 'embed' } = {}) {
  return h('div', { class: ['k-token', selected && 'sel'], style: { '--kind': `var(--c-${kind})` } }, h('b', null, label), h('small', null, i));
}

// ------------------------------------------------------------------ arrows

let arrowId = 0;
function arrowPath(from, to, { kind = 'resid', curve = 0, width = 2.5, flow = false, dash = false, head = true, opacity = 0.9, bend = 'auto' } = {}) {
  const id = `ka${++arrowId}`;
  const g = s('g', { class: `k-arrow kind-${kind}`, style: `--kind: var(--c-${kind})` });
  const dx = to.x - from.x, dy = to.y - from.y;
  let d;
  if (curve) {
    const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2;
    const nx = bend === 'auto' ? -dy : bend.x, ny = bend === 'auto' ? dx : bend.y;
    const len = Math.hypot(nx, ny) || 1;
    d = `M${from.x},${from.y} Q${mx + (nx / len) * curve},${my + (ny / len) * curve} ${to.x},${to.y}`;
  } else d = `M${from.x},${from.y} L${to.x},${to.y}`;
  const path = s('path', { d, fill: 'none', stroke: 'var(--kind)', 'stroke-width': width, 'stroke-linecap': 'round', opacity, 'stroke-dasharray': dash ? '6 6' : null, 'marker-end': head ? `url(#${id}h)` : null });
  const marker = s('marker', { id: `${id}h`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' },
    s('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'var(--kind)' }));
  g.append(s('defs', null, marker), path);
  let raf = null;
  const dots = [];
  if (flow && !reducedMotion()) {
    const total = path.getTotalLength?.() || Math.hypot(dx, dy);
    for (let i = 0; i < 3; i++) { const c = s('circle', { r: width * 1.3, fill: 'var(--kind)', opacity: 0.85 }); g.append(c); dots.push(c); }
    const t0 = performance.now();
    const tick = (now) => {
      const L = path.getTotalLength ? path.getTotalLength() : total;
      dots.forEach((c, i) => {
        const p = (((now - t0) / 1600 + i / dots.length) % 1);
        const pt = path.getPointAtLength(p * L);
        c.setAttribute('cx', pt.x);
        c.setAttribute('cy', pt.y);
        c.setAttribute('opacity', String(Math.sin(p * Math.PI) * 0.9));
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }
  return {
    el: g, path,
    /** Animate the stroke drawing in. */
    draw(duration = 600, delay = 0) {
      if (reducedMotion()) return Promise.resolve();
      const L = path.getTotalLength();
      path.style.strokeDasharray = `${L}`;
      path.style.strokeDashoffset = `${L}`;
      path.style.transition = `stroke-dashoffset ${duration}ms cubic-bezier(.2,.7,.2,1) ${delay}ms`;
      requestAnimationFrame(() => requestAnimationFrame(() => { path.style.strokeDashoffset = '0'; }));
      return new Promise((r) => setTimeout(() => { path.style.strokeDasharray = dash ? '6 6' : ''; r(); }, duration + delay + 30));
    },
    setOpacity(a) { g.style.opacity = String(a); },
    destroy() { if (raf) cancelAnimationFrame(raf); g.remove(); },
  };
}

// --------------------------------------------------------------- motion

/** Fade/rise elements in with a stagger. Returns a promise. */
export function appear(els, { stagger = 70, delay = 0, from = 'below', duration = 520 } = {}) {
  const list = [].concat(els).filter(Boolean);
  if (reducedMotion()) { list.forEach((e) => { e.style.opacity = ''; }); return Promise.resolve(); }
  list.forEach((e, i) => {
    e.animate([
      { opacity: 0, transform: from === 'scale' ? 'scale(.85)' : from === 'left' ? 'translateX(-24px)' : 'translateY(18px)' },
      { opacity: 1, transform: 'none' },
    ], { duration, delay: delay + i * stagger, easing: 'cubic-bezier(.2,.7,.2,1)', fill: 'backwards' });
  });
  return new Promise((r) => setTimeout(r, delay + list.length * stagger + duration));
}

export function pulse(el, { scale = 1.06 } = {}) {
  if (reducedMotion() || !el) return;
  el.animate([{ transform: 'scale(1)', filter: 'brightness(1)' }, { transform: `scale(${scale})`, filter: 'brightness(1.35)' }, { transform: 'scale(1)', filter: 'brightness(1)' }], { duration: 650, easing: 'ease-out' });
}

/** Move a clone of `fromEl` onto `toEl` (both inside the stage), then remove it. */
export function fly(stage, fromEl, toEl, { duration = 700, delay = 0, arc = 60 } = {}) {
  if (reducedMotion()) return Promise.resolve();
  const a = stage.center(fromEl), b = stage.center(toEl);
  const ghost = fromEl.cloneNode(true);
  ghost.classList.add('k-ghost');
  ghost.style.position = 'absolute';
  ghost.style.left = `${a.x}px`;
  ghost.style.top = `${a.y}px`;
  ghost.style.translate = '-50% -50%';
  stage.layers.front.append(ghost);
  const sx = b.w / Math.max(1, a.w), sy = b.h / Math.max(1, a.h);
  const anim = ghost.animate([
    { transform: 'translate(0,0) scale(1)', opacity: 0.95 },
    { transform: `translate(${(b.x - a.x) / 2}px, ${(b.y - a.y) / 2 - arc}px) scale(${(1 + sx) / 2}, ${(1 + sy) / 2})`, opacity: 1, offset: 0.5 },
    { transform: `translate(${b.x - a.x}px, ${b.y - a.y}px) scale(${sx}, ${sy})`, opacity: 0.2 },
  ], { duration, delay, easing: 'cubic-bezier(.45,0,.2,1)', fill: 'forwards' });
  return anim.finished.then(() => ghost.remove()).catch(() => ghost.remove());
}

/** Animate a number in an element. */
export function countTo(el, to, { from = 0, duration = 700, digits = 3 } = {}) {
  if (reducedMotion()) { el.textContent = fmt(to, digits); return; }
  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min(1, (now - t0) / duration), e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(from + (to - from) * e, digits);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

export const wait = (ms) => new Promise((r) => setTimeout(r, reducedMotion() ? 0 : ms));

/** Inject a scene's CSS once (selectors should be prefixed with the scene's class). */
const injected = new Set();
export function useStyle(id, css) {
  if (injected.has(id)) return;
  injected.add(id);
  document.head.append(h('style', { dataset: { scene: id } }, css));
}

// ------------------------------------------------------------ shared scene parts
// Every scene uses these so headers, captions, pickers and checks look and sit the same.
// Zones on the 1600 x 900 canvas (see docs/STYLE_GUIDE.md):
export const ZONES = {
  header: { x: 110, y: 28 },               // chips, title, one context line
  controls: { x: 110, y: 196 },            // token / head pickers (optional)
  hero: { x: 110, y: 270, w: 1380, h: 480 },
  side: { x: 1180, w: 340 },               // equation column when the hero leaves room
  caption: { x: 110, y: 790, w: 1060 },
};
export const MOTION = { move: 600, appear: 520, stagger: 60, step: 1700, ease: 'cubic-bezier(.65,0,.35,1)', easeOut: 'cubic-bezier(.2,.7,.2,1)' };

/** Eyebrow chips + title + one context line, top-left. Returns {el, setSub(html)}. */
export function sceneHeader(stage, { chips = [], title, sub = '' } = {}) {
  const eyebrow = h('div', { class: 'k-eyebrow' }, chips.map(([text, kind]) => chip(text, kind)));
  const t = h('div', { class: 'k-scene-title' }, title);
  const s = h('div', { class: 'k-scene-sub', html: sub });
  const el = h('div', { class: 'k-header' }, eyebrow, t, s);
  stage.add(el, { x: ZONES.header.x, y: ZONES.header.y });
  return {
    el, title: t,
    setSub(html) { s.innerHTML = html; },
    setChips(list) { eyebrow.replaceChildren(...list.map(([text, kind]) => chip(text, kind))); },
  };
}

/** One- or two-sentence caption in the bottom zone; text changes cross-fade. */
export function sceneCaption(stage, { x = ZONES.caption.x, y = ZONES.caption.y, w = ZONES.caption.w } = {}) {
  const el = h('div', { class: 'k-scene-caption', style: { width: `${w}px` } });
  stage.add(el, { x, y });
  let last = '';
  return {
    el,
    set(html) {
      if (html === last) return;
      last = html;
      if (reducedMotion()) { el.innerHTML = html; return; }
      el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 140 }).finished.then(() => {
        el.innerHTML = html;
        el.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 260, easing: MOTION.easeOut });
      }).catch(() => { el.innerHTML = html; });
    },
  };
}

/** Row of small token tiles; clicking selects the token for the whole app. */
export function tokenPicker(stage, ctx, { x = ZONES.controls.x, y = ZONES.controls.y, label = 'token', kind = 'attn', maxWidth = 1360, isDisabled = null } = {}) {
  const row = h('div', { class: 'k-picker' });
  const wrap = h('div', { class: 'k-picker-wrap' }, h('span', { class: 'k-picker-label' }, label), row);
  stage.add(wrap, { x, y });
  function render() {
    const s = ctx.state, T = s.ids.length, sel = Math.min(s.sel.token, T - 1);
    const w = Math.max(16, Math.min(40, (maxWidth - 70) / T - 4));
    row.replaceChildren(...s.ids.map((_, i) => {
      const t = h('button', { type: 'button', class: ['k-pick', i === sel && 'sel'], disabled: isDisabled?.(i), style: { width: `${w}px`, fontSize: T > 32 ? '12px' : '17px', '--kind': `var(--c-${kind})` }, title: `position ${i}` },
        h('b', null, ctx.label(i)), T <= 48 ? h('small', null, i) : null);
      t.addEventListener('click', () => ctx.store.select({ token: i }));
      return t;
    }));
  }
  render();
  const off = ctx.on('select', render);
  const off2 = ctx.on('trace', render);
  stage.onCleanup(() => { off(); off2(); });
  return { el: wrap, render };
}

/** Compact head chips (h0 … h{H-1}); selects the head for the whole app. */
export function headPicker(stage, ctx, { x, y, kind = 'attn' } = {}) {
  const row = h('div', { class: 'k-picker-wrap' });
  stage.add(row, { x, y });
  function render() {
    const H = ctx.cfg.n_head, cur = ctx.state.sel.head;
    row.replaceChildren(h('span', { class: 'k-picker-label' }, 'head'), ...Array.from({ length: H }, (_, i) => {
      const b = h('button', { type: 'button', class: ['k-pick', 'k-pick-head', i === cur && 'sel'], style: { '--kind': `var(--c-${kind})` } }, h('b', null, `h${i}`));
      b.addEventListener('click', () => ctx.store.select({ head: i }));
      return b;
    }));
  }
  render();
  const off = ctx.on('select', render);
  stage.onCleanup(off);
  return { el: row, render };
}

/** "✓ equals the traced value" badge (or a red ✗ with the difference). */
export function checkBadge(got, want, { label = 'equals the traced value', tol = 1e-6 } = {}) {
  const ok = (got === want) || Math.abs(got - want) <= tol * Math.max(1, Math.abs(want));
  return h('span', { class: ['k-check', ok ? 'ok' : 'bad'], title: `computed ${got} · traced ${want}` },
    ok ? `✓ ${label}` : `✗ differs from the traced value by ${fmt(Math.abs(got - want), 3)}`);
}

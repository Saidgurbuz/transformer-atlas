// Animated attention arcs: attention weights drawn as connections between tokens.
//
//   const arcs = arcDiagram({ labels, weights: { data, rows, cols }, mask, query: 3, layout: 'arc' });
//   parent.append(arcs.el);
//   arcs.setQuery(5);                // the arcs of query 5 draw in and the cursor slides to it
//   arcs.update({ weights: next });  // new data: arcs tween, new arcs draw in, old ones retract
//   await arcs.play();               // sweep the query left to right; resolves true at the end
//   arcs.destroy();
//
// Layouts. 'arc' puts every token on one row and draws each key-to-query arc above it, higher for
// longer distances. 'bipartite' puts queries on top and keys below, joined by curves. Two rows are
// also used when rows differ from cols or keyLabels is given (cross-attention).
// Encoding: width = 0.75 + 7.25 w px (w = 1 gives 8 px), opacity grows with w. A masked key of the
// selected query is a greyed token with a dashed stub ('arc') or a dashed ring ('bipartite').
// Interaction: hover a token to preview its arcs (in 'bipartite', hover a key to see the queries
// that read it). Click a query to select it, click it again to clear. Arrow keys step the query,
// Escape clears it. onSelect(i) runs for clicks, keys and play() steps, never for setQuery().
// onHover(i | null) reports the query token under the pointer; onHoverKey(j | null) reports the key
// under the pointer in 'bipartite'.
// Motion: one requestAnimationFrame loop. Arcs draw in nearest-first. A context arc that is already
// on screen keeps its line and draws its bright stroke over it, so nothing blinks. Arcs that leave
// the focus set retract toward their keys; width and opacity tween when weights change. Dots travel
// from key to query along the strongest arcs (dots: false turns them off), over the part of an arc
// that is already drawn. Reduced motion jumps to end states: no drawing, no dots, and play() goes
// straight to the last query. A query set from outside (setQuery, update({query})) stops play().
// When the query moves, a row that scrolls brings the query token into view.

import { h, s, clamp } from './dom.js';
import { kindVar, onThemeChange } from './colors.js';
import { reducedMotion, ease } from './anim.js';
import { showTip, hideTip } from './tooltip.js';

const PAD = 16;           // side padding inside the SVG, px
const MIN_PITCH = 9;      // narrowest token spacing; past this the row scrolls
const MAX_PITCH = 44;     // widest token spacing
const ROW_TOP = 56;       // y of the query row in two-row layouts (room for its index and label)
const ARC_FRAC = 0.55;    // arc area height, as a share of the longest arc span in the data
const DRAW_MS = 380;      // one arc draws in
const RETRACT_MS = 260;   // an arc leaves the focus set
const TWEEN_MS = 460;     // width and opacity follow new weights
const HOVER_MS = 200;     // hover previews are quicker
const SLIDE_MS = 320;     // the query cursor slides to its new token
const STAGGER_MS = 160;   // new arcs start in a wave spread over this window, nearest first
const STEP_MS = 600;      // play(): time per query token
const DOT_MAX = 6;        // travelling dots on the query's strongest arcs
const VALUE_MAX = 2;      // weight labels on the strongest arcs (the rest are in the tooltip)
const CHAR_W = 7.2;       // advance of one 12 px monospace glyph

const DEFAULTS = {
  labels: [], keyLabels: null, weights: null, mask: null, query: null,
  layout: 'arc', kind: 'attn', threshold: 0.02, maxWidth: 760,
  onSelect: null, onHover: null, onHoverKey: null, dots: true, speed: 1, showLabels: true, context: true,
};

// Visual style of each role: width (px) and opacity as functions of the weight w in [0, 1].
const ROLE = {
  focus: (w) => ({ w: 0.75 + 7.25 * w, o: 0.3 + 0.65 * w }),
  strong: (w) => ({ w: 0.75 + 7.25 * w, o: 0.35 + 0.5 * w }),
  faint: (w) => ({ w: 0.75 + 2.2 * w, o: 0.04 + 0.22 * w }),
  ghost: (w) => ({ w: 0.6 + 1.2 * w, o: 0.02 + 0.05 * w }),
  dim: (w) => ({ w: 0.75 + 7.25 * w, o: 0.06 + 0.2 * w }),
  stub: () => ({ w: 1, o: 0.8 }),
};

const STYLE = `
.arcs { display: grid; gap: 8px; min-width: 0; max-width: 100%; }
.arcs-scroll { overflow-x: auto; overflow-y: hidden; max-width: 100%; }
.arcs-svg { display: block; overflow: visible; border-radius: 8px; font-family: var(--font-mono); touch-action: pan-x pan-y; }
.arcs-svg:focus-visible { outline: 2px solid var(--focus); outline-offset: 4px; }
.arcs-svg text { font-family: var(--font-mono); font-size: 12px; fill: var(--fg-2); }
.arcs-svg .idx { font-size: 9.5px; fill: var(--fg-3); }
.arcs-svg .cap { font: 600 10px/1 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; fill: var(--fg-3); }
.arcs-svg .val { font-size: 10.5px; font-weight: 600; fill: var(--fg); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; stroke-linejoin: round; }
.arcs-svg .guide { stroke: var(--line-strong); stroke-width: 1; }
.arcs-svg .arc, .arcs-svg .tick { fill: none; stroke-linecap: round; }
.arcs-svg .stub { fill: none; stroke: var(--fg-3); stroke-width: 1; stroke-dasharray: 2 3; }
.arcs-svg .dot, .arcs-svg .dot-glow { pointer-events: none; }
.arcs-note { margin: 0; font-size: var(--fs-sm); color: var(--fg-3); }
`;

let styled = false;
function injectStyle() {
  if (styled || typeof document === 'undefined') return;
  styled = true;
  document.head.append(h('style', { id: 'arcs-style' }, STYLE));
}

const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const f2 = (v) => (Number.isFinite(v) ? v.toFixed(2) : '0.00');
const fmtW = (w) => (w >= 0.995 ? '1.00' : w.toFixed(2));
const same = (a, b) => Math.abs(a.p - b.p) < 1e-4 && Math.abs(a.w - b.w) < 1e-3
  && Math.abs(a.o - b.o) < 1e-4 && Math.abs(a.l - b.l) < 1e-4;

// Point on a quadratic (3 control points) or cubic (4) Bezier, pts = [x0, y0, x1, y1, ...].
function bez(pts, t, cubic) {
  const u = 1 - t;
  if (cubic) {
    const a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return [a * pts[0] + b * pts[2] + c * pts[4] + d * pts[6], a * pts[1] + b * pts[3] + c * pts[5] + d * pts[7]];
  }
  const a = u * u, b = 2 * u * t, c = t * t;
  return [a * pts[0] + b * pts[2] + c * pts[4], a * pts[1] + b * pts[3] + c * pts[5]];
}

// Token text that fits the spacing; the full text is always in the tooltip.
function fitLabel(text, pitch) {
  const str = String(text);
  const max = Math.max(1, Math.floor((pitch - 2) / CHAR_W));
  if (str.length <= max) return str;
  return max === 1 ? str.slice(0, 1) : `${str.slice(0, max - 1)}…`;
}

// True when value differs from the one last written under key (and records it). Used by paint.
function moved(shown, key, value, eps) {
  if (shown[key] !== undefined && Math.abs(shown[key] - value) <= eps) return false;
  shown[key] = value;
  return true;
}

function normalise(c) {
  const rows = c.weights?.rows | 0, cols = c.weights?.cols | 0;
  const data = c.weights?.data ?? [];
  const mask = c.mask ? (c.mask.data ?? c.mask) : null;
  return {
    ...c,
    labels: Array.from(c.labels ?? []),
    keyLabels: c.keyLabels ? Array.from(c.keyLabels) : null,
    weights: { data, rows, cols },
    maskData: mask && mask.length >= rows * cols ? mask : null,
    query: Number.isInteger(c.query) && c.query >= 0 && c.query < rows ? c.query : null,
    threshold: Number.isFinite(c.threshold) ? c.threshold : 0.02,
    speed: Number.isFinite(c.speed) && c.speed > 0 ? c.speed : 1,
    maxWidth: c.maxWidth > 0 ? c.maxWidth : 760,
  };
}

export function arcDiagram(options = {}) {
  injectStyle();

  let cfg = normalise({ ...DEFAULTS, ...options });
  let alive = true;
  let G = null;               // geometry for the current size and shape
  let geoKey = '';
  let shapeKey = '';
  let twoRows = false;
  let reach = 0;              // longest index distance of an arc in the data (sizes the arc area)
  let hover = null;           // { row: 'query' | 'key', index }
  let lastHoverQ = null, lastHoverK = null;
  let tipOn = false;
  let sweep = null;           // { t0, idx, resolve } while play() runs
  let raf = 0, lastFrame = 0, layoutRaf = 0;
  let measured = 0;           // content width from the ResizeObserver (0 until it reports)
  let tokens = { top: [], bot: [] };
  const records = new Map();  // key -> arc record (arcs, ticks, masked stubs)
  const active = new Set();   // records with a running tween or overlay
  let dots = [];              // travelling dots
  const cursor = { x: 0, o: 0, idx: 0, from: null, to: null, t0: 0, dur: 1 };

  const layers = {
    guide: s('g'), cursor: s('g'), arcs: s('g'), stubs: s('g'),
    dots: s('g'), values: s('g'), tokens: s('g'),
  };
  const cursorRect = s('rect', { class: 'cursor', rx: 7, 'fill-opacity': 0.14, 'stroke-opacity': 0.55, 'stroke-width': 1, opacity: 0 });
  layers.cursor.append(cursorRect);
  const svg = s('svg', {
    class: 'arcs-svg', role: 'group', tabindex: '0',
    'aria-label': 'Attention arcs. Arrow keys move the query, Escape clears it.',
  }, layers.guide, layers.cursor, layers.arcs, layers.stubs, layers.dots, layers.values, layers.tokens);
  const note = h('p', { class: 'arcs-note', hidden: true }, 'No attention weights to draw.');
  const scroll = h('div', { class: 'arcs-scroll' }, svg);
  const el = h('div', { class: 'arcs' }, scroll, note);

  svg.addEventListener('pointermove', onMove);
  svg.addEventListener('pointerleave', onLeave);
  svg.addEventListener('click', onClick);
  svg.addEventListener('keydown', onKey);

  const offTheme = onThemeChange(() => { if (alive) restyle(); });
  // The width comes from the observer, so refresh() never forces a layout on pointer moves.
  const ro = typeof ResizeObserver === 'function' ? new ResizeObserver((entries) => {
    measured = entries[0]?.contentRect.width ?? 0;
    if (layoutRaf || !alive) return;
    layoutRaf = requestAnimationFrame(() => { layoutRaf = 0; if (alive) refresh({ animate: false }); });
  }) : null;
  ro?.observe(el);

  reach = maxReach();
  refresh({ animate: false });

  // ------------------------------------------------------------ geometry

  function geometryFor(width) {
    const { rows, cols } = cfg.weights;
    const n = Math.max(rows, cols, 1);
    const avail = Math.min(width > 0 ? width : cfg.maxWidth, cfg.maxWidth);
    const pitch = clamp((avail - 2 * PAD) / n, MIN_PITCH, MAX_PITCH);
    const W = Math.ceil(Math.max(avail, n * pitch + 2 * PAD));
    const span = (count) => Array.from({ length: count }, (_, k) => (W - count * pitch) / 2 + (k + 0.5) * pitch);
    if (twoRows) {
      const gap = clamp(W * 0.26, 96, 180);
      const yTop = ROW_TOP, yBot = yTop + gap;
      return { W, H: yBot + 62, pitch, gap, xTop: span(rows), xBot: span(cols), yTop, yBot };
    }
    // The arc area is as tall as the longest arc in the data needs (stable across queries), up to a
    // share of the width.
    const cap = clamp(W * 0.2, 90, 170);
    const hMax = clamp(ARC_FRAC * reach * pitch, 44, cap);
    const yBase = hMax + 34;
    return { W, H: yBase + 44, pitch, hMax, yBase, x: span(n) };
  }

  // Path, midpoint and length of the arc from key j to query i (or the self tick / masked stub).
  function geomFor(kind, i, j) {
    if (twoRows) {
      const x0 = G.xBot[j], y0 = G.yBot - 6, x1 = G.xTop[i], y1 = G.yTop + 6, ym = (y0 + y1) / 2;
      const pts = [x0, y0, x0, ym, x1, ym, x1, y1];
      return {
        cubic: true, pts, chord: Math.hypot(x1 - x0, y1 - y0), mid: bez(pts, 0.5, true),
        d: `M${f2(x0)} ${f2(y0)}C${f2(x0)} ${f2(ym)} ${f2(x1)} ${f2(ym)} ${f2(x1)} ${f2(y1)}`,
      };
    }
    const yb = G.yBase;
    if (kind === 'tick') return { chord: 0, d: `M${f2(G.x[i])} ${f2(yb - 5)}V${f2(yb - 21)}` };
    if (kind === 'stub') return { chord: 0, d: `M${f2(G.x[j])} ${f2(yb - 6)}V${f2(yb - 17)}` };
    const x0 = G.x[j], x1 = G.x[i], dx = Math.abs(x1 - x0);
    const h = Math.min(G.hMax * (1 - Math.exp(-dx / (G.hMax * 0.8))), 0.6 * dx);
    const pts = [x0, yb, (x0 + x1) / 2, yb - 2 * h, x1, yb];
    return {
      cubic: false, pts, chord: dx, mid: bez(pts, 0.5, false),
      d: `M${f2(x0)} ${f2(yb)}Q${f2(pts[2])} ${f2(pts[3])} ${f2(x1)} ${f2(yb)}`,
    };
  }

  function xOf(index) { return twoRows ? G.xTop[index] : G.x[index]; }

  // Longest index distance between a query and a key whose arc is drawn (weight at or above the
  // threshold, and allowed by the mask). Data-only: it does not change when the query changes.
  function maxReach() {
    const { data, rows, cols } = cfg.weights;
    const thr = cfg.threshold, mask = cfg.maskData;
    let best = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (i === j) continue;
        const v = data[i * cols + j];
        if (!(v > 0) || v < thr) continue;
        if (mask && mask[i * cols + j] !== 1) continue;
        const d = Math.abs(i - j);
        if (d > best) best = d;
      }
    }
    return best;
  }

  // ------------------------------------------------------------ tokens

  function buildTokens() {
    layers.tokens.replaceChildren();
    layers.guide.replaceChildren();
    tokens = { top: [], bot: [] };
    const mk = (index) => {
      const node = s('circle', { class: 'node', r: 2.4 });
      const lab = s('text', { class: 'lab', 'text-anchor': 'middle' });
      const idx = s('text', { class: 'idx', 'text-anchor': 'middle' });
      layers.tokens.append(node, lab, idx);
      return { index, node, lab, idx };
    };
    for (let i = 0; i < cfg.weights.rows; i++) tokens.top.push(mk(i));
    if (twoRows) for (let j = 0; j < cfg.weights.cols; j++) tokens.bot.push(mk(j));
  }

  function place() {
    const p = G.pitch;
    const showIdx = cfg.showLabels && p >= 14;
    layers.guide.replaceChildren();
    const rowLine = (xs, y) => {
      if (xs.length) layers.guide.append(s('line', { class: 'guide', x1: f2(xs[0] - p / 2), x2: f2(xs[xs.length - 1] + p / 2), y1: f2(y), y2: f2(y) }));
    };
    // Query tokens (top row) carry their index above the label; keys carry it below.
    const setTok = (t, x, y, above) => {
      t.node.setAttribute('cx', f2(x));
      t.node.setAttribute('cy', f2(y));
      t.lab.setAttribute('x', f2(x));
      t.lab.setAttribute('y', f2(above ? y - 13 : y + 19));
      t.idx.setAttribute('x', f2(x));
      t.idx.setAttribute('y', f2(above ? y - 29 : y + 34));
      t.idx.textContent = showIdx ? String(t.index) : '';
    };
    if (twoRows) {
      rowLine(G.xTop, G.yTop);
      rowLine(G.xBot, G.yBot);
      layers.guide.append(
        s('text', { class: 'cap', x: f2(PAD), y: 12 }, 'queries'),
        s('text', { class: 'cap', x: f2(PAD), y: f2(G.yBot + 56) }, 'keys'));
      tokens.top.forEach((t, i) => setTok(t, G.xTop[i], G.yTop, true));
      tokens.bot.forEach((t, j) => setTok(t, G.xBot[j], G.yBot, false));
    } else {
      rowLine(G.x, G.yBase);
      tokens.top.forEach((t, i) => setTok(t, G.x[i], G.yBase, false));
    }
  }

  function syncLabels() {
    const p = G.pitch;
    const size = p < 12 ? '10.5px' : '';
    const kl = cfg.keyLabels ?? cfg.labels;
    tokens.top.forEach((t) => {
      t.lab.textContent = fitLabel(cfg.labels[t.index] ?? t.index, p);
      t.lab.style.fontSize = size;
    });
    tokens.bot.forEach((t) => {
      t.lab.textContent = fitLabel(kl[t.index] ?? t.index, p);
      t.lab.style.fontSize = size;
    });
  }

  // Colours and states of tokens follow the query, the hover and the mask.
  function styleTokens() {
    if (!tokens.top.length) return;
    const q = cfg.query, accent = kindVar(cfg.kind), cols = cfg.weights.cols;
    const hq = hover && hover.row === 'query' ? hover.index : -1;
    const hk = hover && hover.row === 'key' ? hover.index : -1;
    const masked = (j) => q != null && !!cfg.maskData && cfg.maskData[q * cols + j] !== 1;
    const paintNode = (t, { fill, blocked, hov, sel }) => {
      t.node.setAttribute('r', sel ? 4.2 : hov ? 3.6 : 2.4);
      t.node.style.fill = blocked ? 'none' : fill;
      t.node.style.stroke = blocked ? 'var(--fg-3)' : 'none';
      if (blocked) t.node.setAttribute('stroke-dasharray', '1.5 1.5');
      else t.node.removeAttribute('stroke-dasharray');
      t.lab.style.fill = sel || hov ? 'var(--fg)' : blocked ? 'var(--fg-3)' : 'var(--fg-2)';
      t.lab.setAttribute('font-weight', sel ? '700' : '500');
      t.lab.setAttribute('opacity', blocked ? '0.7' : '1');
    };
    tokens.top.forEach((t, i) => {
      const sel = i === q, hov = i === hq, blocked = !twoRows && masked(i);
      paintNode(t, { fill: sel ? accent : hov ? 'var(--fg)' : 'var(--fg-3)', blocked, hov, sel });
    });
    tokens.bot.forEach((t, j) => {
      const hov = j === hk, blocked = masked(j);
      paintNode(t, { fill: hov ? 'var(--fg)' : 'var(--fg-3)', blocked, hov, sel: false });
    });
  }

  function restyle() {
    const accent = kindVar(cfg.kind);
    cursorRect.style.fill = accent;
    cursorRect.style.stroke = accent;
    for (const r of records.values()) {
      if (r.kind !== 'stub') r.el.style.stroke = accent;
      if (r.ov) r.ov.el.style.stroke = accent;
    }
    for (const d of dots) { d.el.style.fill = accent; d.glow.style.fill = accent; }
    styleTokens();
  }

  // ------------------------------------------------------------ arc records

  function makeRecord(t) {
    const stub = t.kind === 'stub';
    const el = stub
      ? s('path', { class: 'stub' })
      : s('path', { class: t.kind, fill: 'none', pathLength: 1, 'stroke-dasharray': '1 1' });
    if (!stub) el.style.stroke = kindVar(cfg.kind);
    const r = {
      key: t.key, kind: t.kind, i: t.i, j: t.j, role: t.role, w: t.w, el, lab: null, geom: null,
      cur: { p: 0, w: 0, o: 0, l: 0 }, from: null, to: null, t0: 0, dur: 1, exiting: false,
      ov: null,           // overlay path: the bright stroke drawing over a context line
      shown: {},          // attribute values last written to el (see paint)
      labShown: null,     // the same for the value label
    };
    (stub ? layers.stubs : layers.arcs).append(el);
    records.set(t.key, r);
    relayoutRecord(r);
    paint(r);
    return r;
  }

  function relayoutRecord(r) {
    r.geom = geomFor(r.kind, r.i, r.j);
    r.el.setAttribute('d', r.geom.d);
    if (r.ov) r.ov.el.setAttribute('d', r.geom.d);
    placeLabel(r);
  }

  function removeRecord(r) {
    r.el.remove();
    r.ov?.el.remove();
    r.ov = null;
    r.lab?.remove();
    r.lab = null;
    records.delete(r.key);
    active.delete(r);
  }

  function makeLabel(r) {
    r.lab = s('text', { class: 'val' });
    r.labShown = {};
    layers.values.append(r.lab);
    placeLabel(r);
  }

  function placeLabel(r) {
    if (!r.lab || !r.geom?.mid) return;
    const [mx, my] = r.geom.mid;
    if (twoRows) {
      r.lab.setAttribute('x', f2(mx + 6));
      r.lab.setAttribute('y', f2(my + 3));
      r.lab.setAttribute('text-anchor', 'start');
    } else {
      r.lab.setAttribute('x', f2(mx));
      r.lab.setAttribute('y', f2(my - 7));
      r.lab.setAttribute('text-anchor', 'middle');
    }
  }

  // Value labels must not overprint. Sorted left to right, each label that would overlap an
  // earlier one moves up in 11 px steps. Labels that are fading out keep their default place.
  function avoidLabelCollisions() {
    const items = [...records.values()].filter((r) => r.lab && r.geom?.mid);
    items.sort((a, b) => a.geom.mid[0] - b.geom.mid[0]);
    const placed = [];
    for (const r of items) {
      const [mx, my] = r.geom.mid;
      const w = String(r.lab.textContent ?? '').length * 6.4 + 4;
      const start = twoRows;
      let x = start ? mx + 6 : mx;
      let y = start ? my + 3 : my - 7;
      const box = () => ({ x0: start ? x : x - w / 2, x1: start ? x + w : x + w / 2, y0: y - 9, y1: y + 3 });
      const live = !r.exiting && (r.to ? r.to.l > 0 : r.cur.l > 0.5);
      if (live) {
        for (let step = 0; step < 6; step++) {
          const b = box();
          if (!placed.some((p) => b.x0 < p.x1 && b.x1 > p.x0 && b.y0 < p.y1 && b.y1 > p.y0)) break;
          y -= 11;
        }
        placed.push(box());
      }
      r.lab.setAttribute('x', f2(x));
      r.lab.setAttribute('y', f2(y));
    }
  }

  function beginTween(r, to, dur, delay, now) {
    r.from = { ...r.cur };
    r.to = to;
    r.t0 = now + delay;
    r.dur = Math.max(1, dur);
    active.add(r);
    kick();
  }

  // An overlay is a second stroke over a line that is already on screen. It draws the bright
  // focus style in from the key, while the record's own line keeps its context style underneath.
  function startOverlay(r, style, dur, delay, now) {
    const el = s('path', {
      class: r.kind, fill: 'none', pathLength: 1, 'stroke-dasharray': '1 1',
      'stroke-dashoffset': 1, visibility: 'hidden',
    });
    el.setAttribute('d', r.geom.d);
    el.style.stroke = kindVar(cfg.kind);
    layers.arcs.append(el);
    r.ov = { el, to: style, t0: now + delay, dur: Math.max(1, dur), p: 0 };
    active.add(r);
    kick();
  }

  function stepOverlay(r, now) {
    const ov = r.ov;
    const t = clamp((now - ov.t0) / ov.dur, 0, 1);
    ov.p = ease.inOut(t);
    ov.el.setAttribute('visibility', ov.p > 0.002 ? 'visible' : 'hidden');
    ov.el.setAttribute('stroke-dashoffset', (1 - ov.p).toFixed(4));
    ov.el.setAttribute('stroke-width', ov.to.w.toFixed(3));
    ov.el.setAttribute('opacity', ov.to.o.toFixed(3));
    if (t >= 1) finishOverlay(r);
  }

  // The bright stroke is complete: the record's own line takes the focus style and the overlay goes.
  function finishOverlay(r) {
    const { to, el } = r.ov;
    el.remove();
    r.ov = null;
    r.cur.p = 1;
    r.cur.w = to.w;
    r.cur.o = to.o;
    if (r.to) { r.from = { ...r.cur }; r.to.p = 1; r.to.w = to.w; r.to.o = to.o; }
    paint(r);
    if (!r.to) active.delete(r);
  }

  function dropOverlay(r) {
    if (!r.ov) return;
    r.ov.el.remove();
    r.ov = null;
    if (!r.to) active.delete(r);
  }

  // Writes only the attributes whose text changed since the last paint. Most arcs on a given frame
  // are still waiting out their stagger or resting, so this keeps the per-frame DOM work small.
  // Numbers are compared first, so resting arcs cost no string work. The tolerances are far below
  // what the eye can see (a thousandth of a pixel or of an opacity step).
  function paint(r) {
    const { p, w, o, l } = r.cur;
    const sh = r.shown;
    if (r.kind === 'stub') {
      if (moved(sh, 'o', o, 5e-4)) r.el.setAttribute('opacity', o.toFixed(3));
    } else {
      const vis = p > 0.002;
      if (sh.v !== vis) { sh.v = vis; r.el.setAttribute('visibility', vis ? 'visible' : 'hidden'); }
      if (moved(sh, 'd', p, 5e-5)) r.el.setAttribute('stroke-dashoffset', (1 - p).toFixed(4));
      if (moved(sh, 'w', w, 5e-4)) r.el.setAttribute('stroke-width', w.toFixed(3));
      if (moved(sh, 'o', o, 5e-4)) r.el.setAttribute('opacity', o.toFixed(3));
    }
    if (r.lab && r.labShown && moved(r.labShown, 'l', l, 5e-4)) r.lab.setAttribute('opacity', l.toFixed(3));
  }

  function stepRecord(r, now) {
    if (r.to) {
      const t = clamp((now - r.t0) / r.dur, 0, 1);
      const e = ease.inOut(t);
      const { from, to, cur } = r;
      cur.p = from.p + (to.p - from.p) * e;
      cur.w = from.w + (to.w - from.w) * e;
      cur.o = from.o + (to.o - from.o) * e;
      cur.l = from.l + (to.l - from.l) * e;
      paint(r);
      if (t >= 1) {
        r.from = null;
        r.to = null;
        if (r.exiting) { removeRecord(r); return; }
        if (cur.l <= 0 && r.lab) { r.lab.remove(); r.lab = null; }
      }
    }
    if (r.ov) stepOverlay(r, now);
    if (!r.to && !r.ov) active.delete(r);
  }

  // Moves every record towards the wanted set. New arcs draw in, nearest to the anchor first. An
  // arc that becomes a focus arc while its line is on screen draws an overlay over that line. The
  // rest tween width and opacity, and arcs that are no longer wanted retract.
  function applyTargets(want, quick) {
    const now = performance.now();
    const instant = reducedMotion();
    const speed = clamp(cfg.speed, 0.05, 4);
    const drawMs = (quick ? HOVER_MS : DRAW_MS) / speed;
    const tweenMs = (quick ? HOVER_MS : TWEEN_MS) / speed;
    const retractMs = RETRACT_MS / speed;
    const ci = hover && hover.row === 'query' ? hover.index : (cfg.query ?? 0);
    const cj = hover && hover.row === 'key' ? hover.index : (cfg.query ?? 0);
    const dist = (x) => Math.abs(x.i - ci) + Math.abs(x.j - cj);

    for (const r of [...records.values()]) {
      if (want.has(r.key) || r.exiting) continue;
      r.exiting = true;
      dropOverlay(r);
      if (instant) { removeRecord(r); continue; }
      beginTween(r, { p: 0, w: r.cur.w, o: 0, l: 0 }, retractMs, 0, now);
    }

    const starters = [];
    for (const t of want.values()) {
      const st = ROLE[t.role](t.w);
      const to = { p: 1, w: st.w, o: st.o, l: t.label ? 1 : 0 };
      let r = records.get(t.key);
      const fresh = !r;
      if (fresh) { r = makeRecord(t); r.cur.w = to.w; }
      const prevRole = fresh ? null : r.role;
      r.role = t.role;
      r.w = t.w;
      r.exiting = false;
      // Incoming arcs of the query (or of a hovered token) are drawn above the context arcs.
      if (t.role === 'focus' && prevRole !== 'focus' && r.kind !== 'stub') layers.arcs.append(r.el);
      if (to.l > 0 && !r.lab) makeLabel(r);
      if (r.lab && to.l > 0) r.lab.textContent = fmtW(t.w);
      if (r.ov && t.role !== 'focus') dropOverlay(r);

      if (instant) {
        dropOverlay(r);
        r.cur = { ...to };
        r.from = null;
        r.to = null;
        active.delete(r);
        if (to.l <= 0 && r.lab) { r.lab.remove(); r.lab = null; }
        paint(r);
        continue;
      }
      const drawable = r.kind !== 'stub';
      // A context arc (ghost, faint or strong) that becomes focus. Its line is fully drawn already.
      const grows = drawable && t.role === 'focus' && prevRole !== 'focus' && prevRole !== 'dim';
      if (drawable && (fresh || (grows && r.cur.p < 0.999))) { starters.push({ r, to, overlay: false }); continue; }
      if (grows) { starters.push({ r, to, overlay: true }); continue; }
      if (r.ov) {
        // The bright stroke is already drawing: the line keeps its style and only the label moves.
        r.ov.to = { w: to.w, o: to.o };
        to.w = r.cur.w;
        to.o = r.cur.o;
      }
      if (same(r.to || r.cur, to)) continue;
      beginTween(r, to, r.cur.p < 0.999 ? drawMs : tweenMs, 0, now);
    }

    starters.sort((a, b) => dist(a.r) - dist(b.r));
    const n = starters.length;
    const spread = (quick ? 60 : STAGGER_MS) / speed;
    starters.forEach(({ r, to, overlay }, k) => {
      const delay = spread * (n > 1 ? k / (n - 1) : 0);
      if (overlay) {
        const style = { w: to.w, o: to.o };
        to.w = r.cur.w;
        to.o = r.cur.o;
        beginTween(r, to, drawMs, delay, now);
        startOverlay(r, style, drawMs, delay, now);
      } else {
        beginTween(r, to, drawMs, delay, now);
      }
    });
  }

  // Which arcs should be visible, and how strongly, for the current query, hover and options.
  function computeWant() {
    const want = new Map();
    const rowKeys = [];
    const { data, rows, cols } = cfg.weights;
    if (!G || !rows || !cols) return { want, rowKeys };
    const thr = cfg.threshold, q = cfg.query, mask = cfg.maskData, arc = !twoRows;
    const wAt = (i, j) => { const v = data[i * cols + j]; return v > 0 ? v : 0; };
    const ok = (i, j) => !mask || mask[i * cols + j] === 1;
    const put = (i, j, role) => {
      const kind = arc && i === j ? 'tick' : 'arc';
      const key = `a${i}_${j}`;
      want.set(key, { key, kind, i, j, role, w: wAt(i, j), label: false });
    };

    if (q == null) {
      for (let i = 0; i < rows; i++) {
        let best = -1, bestW = 0;
        for (let j = 0; j < cols; j++) if (ok(i, j) && wAt(i, j) > bestW) { bestW = wAt(i, j); best = j; }
        for (let j = 0; j < cols; j++) if (ok(i, j) && wAt(i, j) >= thr) put(i, j, j === best ? 'strong' : 'faint');
      }
    } else {
      if (cfg.context) {
        for (let i = 0; i < rows; i++) {
          if (i === q) continue;
          for (let j = 0; j < cols; j++) if (ok(i, j) && wAt(i, j) >= thr) put(i, j, 'ghost');
        }
      }
      for (let j = 0; j < cols; j++) {
        if (ok(q, j) && wAt(q, j) >= thr) put(q, j, 'focus');
        else if (arc && mask && !ok(q, j)) {
          const key = `s${q}_${j}`;
          want.set(key, { key, kind: 'stub', i: q, j, role: 'stub', w: 0, label: false });
        }
      }
    }

    if (hover && hover.row === 'query' && hover.index !== q) {
      const hq = hover.index;
      if (q != null) for (const t of want.values()) if (t.i === q && t.role === 'focus') t.role = 'dim';
      for (let j = 0; j < cols; j++) if (ok(hq, j) && wAt(hq, j) >= thr) put(hq, j, 'focus');
    } else if (hover && hover.row === 'key' && twoRows) {
      const k = hover.index;
      for (let i = 0; i < rows; i++) if (ok(i, k) && wAt(i, k) >= thr) put(i, k, 'focus');
    }

    // Value labels on the strongest arcs of whatever is in focus (the hover, else the query).
    let group = [];
    const arcsWhere = (pred) => [...want.values()].filter((t) => t.kind === 'arc' && pred(t));
    if (hover && hover.row === 'query' && hover.index !== q) group = arcsWhere((t) => t.i === hover.index);
    else if (hover && hover.row === 'key' && twoRows) group = arcsWhere((t) => t.j === hover.index);
    else if (q != null) group = arcsWhere((t) => t.i === q && t.role === 'focus');
    if (cfg.showLabels) {
      group.sort((a, b) => b.w - a.w).slice(0, VALUE_MAX).forEach((t) => { t.label = true; });
    }

    if (q != null) {
      const row = arcsWhere((t) => t.i === q && (t.role === 'focus' || t.role === 'dim')).sort((a, b) => b.w - a.w);
      for (const t of row.slice(0, DOT_MAX)) rowKeys.push(t.key);
    }
    return { want, rowKeys };
  }

  // ------------------------------------------------------------ dots and cursor

  function syncDots(keys) {
    if (reducedMotion()) {
      for (const d of dots) { d.el.remove(); d.glow.remove(); }
      dots = [];
      return;
    }
    const wanted = cfg.dots ? keys : [];
    for (const d of dots) d.target = wanted.includes(d.key) ? 1 : 0;
    wanted.forEach((key, k) => {
      if (dots.some((d) => d.key === key)) return;
      const r = records.get(key);
      const el = s('circle', { class: 'dot', r: 2.6 });
      const glow = s('circle', { class: 'dot-glow', r: 6 });
      el.style.fill = kindVar(cfg.kind);
      glow.style.fill = kindVar(cfg.kind);
      layers.dots.append(glow, el);
      dots.push({ key, el, glow, fade: 0, target: 1, phase: k / wanted.length, period: 1.8 + Math.min(1.2, (r?.geom?.chord ?? 200) / 500) });
    });
    if (dots.length) kick();
  }

  function stepDots(now, dt) {
    const speed = clamp(cfg.speed, 0.05, 4);
    const sec = now / 1000;
    const k = 1 - Math.exp(-dt / 140);
    const instant = reducedMotion();
    for (let n = dots.length - 1; n >= 0; n--) {
      const d = dots[n];
      if (instant) d.target = 0;
      d.fade += (d.target - d.fade) * k;
      if (d.target === 0 && d.fade < 0.01) {
        d.el.remove();
        d.glow.remove();
        dots.splice(n, 1);
        continue;
      }
      const r = records.get(d.key);
      if (!r?.geom?.pts) {
        d.target = 0;   // its arc is gone: fade the dot out and remove it
        d.el.setAttribute('opacity', '0');
        d.glow.setAttribute('opacity', '0');
        continue;
      }
      // A dot only travels over the part of its arc that is drawn, and its alpha goes to zero at the
      // loop point, so it never jumps while visible.
      const frac = r.ov ? r.ov.p : r.cur.p;
      const ur = (sec * speed / d.period + d.phase) % 1;
      const [x, y] = bez(r.geom.pts, ur * frac, r.geom.cubic);
      const o = r.ov ? r.ov.to.o : r.cur.o;
      const a = d.fade * Math.pow(Math.sin(Math.PI * ur), 0.6) * frac * Math.min(1, o * 1.6);
      d.el.setAttribute('cx', f2(x));
      d.el.setAttribute('cy', f2(y));
      d.el.setAttribute('opacity', a.toFixed(3));
      d.glow.setAttribute('cx', f2(x));
      d.glow.setAttribute('cy', f2(y));
      d.glow.setAttribute('opacity', (a * 0.28).toFixed(3));
    }
  }

  // Keeps the query token in view when the row scrolls (narrow screens, long inputs). It runs only
  // for animated query changes, and uses the measured width, so pointer moves never force a layout.
  function followQuery(q) {
    const view = measured;
    if (!view || G.W <= view + 1) return;
    const x = xOf(q);
    const left = scroll.scrollLeft;
    if (x >= left + 40 && x <= left + view - 40) return;
    scroll.scrollTo({ left: clamp(x - view / 2, 0, G.W - view), behavior: reducedMotion() ? 'auto' : 'smooth' });
  }

  function moveCursor(animate) {
    if (!G) return;
    const q = cfg.query;
    if (q != null) cursor.idx = q;
    const to = { x: q != null ? xOf(q) : cursor.x, o: q != null ? 1 : 0 };
    const ref = cursor.from ? cursor.to : { x: cursor.x, o: cursor.o };
    if (Math.abs(ref.x - to.x) < 0.01 && Math.abs(ref.o - to.o) < 0.001) return;
    // The row only scrolls when the query itself moves (this reads layout, so it stays off other paths).
    if (animate && q != null) followQuery(q);
    if (!animate || reducedMotion()) {
      cursor.from = null;
      cursor.x = to.x;
      cursor.o = to.o;
      paintCursor();
      return;
    }
    cursor.from = { x: cursor.x, o: cursor.o };
    cursor.to = to;
    cursor.t0 = performance.now();
    cursor.dur = SLIDE_MS / clamp(cfg.speed, 0.05, 4);
    kick();
  }

  function stepCursor(now) {
    const t = clamp((now - cursor.t0) / cursor.dur, 0, 1);
    const e = ease.inOut(t);
    cursor.x = cursor.from.x + (cursor.to.x - cursor.from.x) * e;
    cursor.o = cursor.from.o + (cursor.to.o - cursor.from.o) * e;
    if (t >= 1) cursor.from = null;
    paintCursor();
  }

  function paintCursor() {
    if (!G) return;
    const label = String(cfg.labels[cursor.idx] ?? '');
    const pw = clamp(Math.max(16, G.pitch - 4, label.length * CHAR_W + 8), 16, 60);
    cursorRect.setAttribute('x', f2(cursor.x - pw / 2));
    cursorRect.setAttribute('y', f2(twoRows ? G.yTop - 40 : G.yBase - 10));
    cursorRect.setAttribute('width', f2(pw));
    cursorRect.setAttribute('height', twoRows ? '49' : '38');
    cursorRect.setAttribute('opacity', cursor.o.toFixed(3));
  }

  // ------------------------------------------------------------ refresh

  function refresh({ animate = true, quick = false } = {}) {
    if (!alive) return;
    const { rows, cols } = cfg.weights;
    const two = cfg.layout === 'bipartite' || rows !== cols || !!cfg.keyLabels;
    const width = measured;
    scroll.hidden = !rows || !cols;
    note.hidden = !scroll.hidden;
    // Records whose tokens no longer exist (a shorter input) cannot be laid out: remove them now.
    for (const r of [...records.values()]) if (r.i >= rows || r.j >= cols) removeRecord(r);
    if (!rows || !cols) {
      for (const r of [...records.values()]) removeRecord(r);
      G = null;
      geoKey = '';
      shapeKey = '';
      syncDots([]);
      return;
    }
    const shape = `${two}|${rows}|${cols}`;
    if (shape !== shapeKey) {
      if (two !== twoRows) for (const r of [...records.values()]) removeRecord(r);
      twoRows = two;
      shapeKey = shape;
      buildTokens();
      geoKey = '';
    }
    let moved = false;
    const geo = `${shape}|${width}|${cfg.maxWidth}|${reach}`;
    if (geo !== geoKey) {
      geoKey = geo;
      moved = true;
      G = geometryFor(width);
      svg.setAttribute('width', String(G.W));
      svg.setAttribute('height', String(G.H));
      place();
      for (const r of records.values()) relayoutRecord(r);
    }
    syncLabels();
    const { want, rowKeys } = computeWant();
    applyTargets(want, quick);
    styleTokens();
    moveCursor(animate && !moved);
    paintCursor();
    syncDots(rowKeys);
    avoidLabelCollisions();
  }

  function kick() {
    if (alive && !raf) raf = requestAnimationFrame(frame);
  }

  function frame(now) {
    raf = 0;
    if (!alive) return;
    const dt = lastFrame ? Math.min(50, now - lastFrame) : 16;
    lastFrame = now;
    for (const r of active) stepRecord(r, now);
    if (cursor.from) stepCursor(now);
    if (sweep) stepSweep(now);
    if (dots.length) stepDots(now, dt);
    if (active.size || cursor.from || sweep || dots.length) kick();
    else lastFrame = 0;
  }

  // ------------------------------------------------------------ query and play

  function setQuery(q, { notify = false } = {}) {
    if (!alive) return;
    const n = cfg.weights.rows;
    const next = Number.isInteger(q) && q >= 0 && q < n ? q : null;
    // A query set from outside ends a running sweep. The sweep's own steps pass through.
    if (sweep && next !== sweep.idx) stop();
    if (next === cfg.query) return;
    cfg.query = next;
    refresh({ animate: true });
    if (notify) cfg.onSelect?.(next);
  }

  // Sweeps the query from the first token to the last, one STEP_MS each. Resolves true at the end and
  // false when stop() or an outside query change ends it. Reduced motion jumps to the last query.
  function play() {
    if (!alive || !cfg.weights.rows) return Promise.resolve(false);
    stop();
    if (reducedMotion()) {
      setQuery(cfg.weights.rows - 1, { notify: true });
      return Promise.resolve(true);
    }
    return new Promise((resolve) => {
      sweep = { t0: performance.now(), idx: -1, resolve };
      kick();
    });
  }

  function stop() {
    if (!sweep) return;
    const s0 = sweep;
    sweep = null;
    s0.resolve(false);
  }

  function stepSweep(now) {
    const s0 = sweep;
    const n = cfg.weights.rows;
    const step = STEP_MS / clamp(cfg.speed, 0.05, 4);
    const i = Math.max(0, Math.floor((now - s0.t0) / step));
    if (i >= n) {
      sweep = null;
      setQuery(n - 1, { notify: true });
      s0.resolve(true);
      return;
    }
    if (i !== s0.idx) {
      s0.idx = i;
      setQuery(i, { notify: true });
    }
  }

  // ------------------------------------------------------------ pointer and keyboard

  function hitAt(e) {
    if (!G) return null;
    const rect = svg.getBoundingClientRect();
    const sc = G.W / (rect.width || G.W);
    const x = (e.clientX - rect.left) * sc, y = (e.clientY - rect.top) * sc;
    const index = (xs) => {
      if (!xs.length) return null;
      const k = Math.floor((x - (xs[0] - G.pitch / 2)) / G.pitch);
      return k >= 0 && k < xs.length ? k : null;
    };
    if (twoRows) {
      if (y >= G.yTop - 44 && y <= G.yTop + 26) {
        const i = index(G.xTop);
        return i == null ? null : { row: 'query', index: i };
      }
      if (y >= G.yBot - 26 && y <= G.yBot + 40) {
        const j = index(G.xBot);
        return j == null ? null : { row: 'key', index: j };
      }
      return null;
    }
    if (y < G.yBase - 26 || y > G.yBase + 40) return null;
    const i = index(G.x);
    return i == null ? null : { row: 'query', index: i };
  }

  function tipFor(hit) {
    const { data, rows, cols } = cfg.weights;
    const wAt = (i, j) => { const v = data[i * cols + j]; return v > 0 ? v : 0; };
    const ok = (i, j) => !cfg.maskData || cfg.maskData[i * cols + j] === 1;
    const qNames = cfg.labels, kNames = cfg.keyLabels ?? cfg.labels;
    const pairs = [];
    let blocked = 0;
    if (hit.row === 'query') {
      for (let j = 0; j < cols; j++) {
        if (ok(hit.index, j)) pairs.push({ k: j, w: wAt(hit.index, j) });
        else blocked++;
      }
    } else {
      for (let i = 0; i < rows; i++) if (ok(i, hit.index)) pairs.push({ k: i, w: wAt(i, hit.index) });
    }
    pairs.sort((a, b) => b.w - a.w);
    const lines = hit.row === 'query'
      ? [`query <b>${esc(qNames[hit.index] ?? hit.index)}</b> (${hit.index}) reads`]
      : [`key <b>${esc(kNames[hit.index] ?? hit.index)}</b> (${hit.index}) is read by`];
    for (const p of pairs.slice(0, 3)) {
      if (p.w <= 0) continue;
      const name = hit.row === 'query' ? kNames[p.k] : qNames[p.k];
      lines.push(`${esc(name ?? p.k)} (${p.k}) <b>${fmtW(p.w)}</b>`);
    }
    if (blocked) lines.push(`${blocked} later keys masked`);
    return lines.join('<br>');
  }

  // Reports hover changes once each: onHover for the query row, onHoverKey for the bipartite keys.
  function notifyHover() {
    const q = hover && hover.row === 'query' ? hover.index : null;
    const k = hover && hover.row === 'key' ? hover.index : null;
    if (q !== lastHoverQ) { lastHoverQ = q; cfg.onHover?.(q); }
    if (k !== lastHoverK) { lastHoverK = k; cfg.onHoverKey?.(k); }
  }

  function onMove(e) {
    const hit = hitAt(e);
    const changed = hit ? !hover || hover.row !== hit.row || hover.index !== hit.index : !!hover;
    if (changed) {
      hover = hit;
      notifyHover();
      refresh({ animate: false, quick: true });
    }
    if (hit) { showTip(e.clientX, e.clientY, tipFor(hit)); tipOn = true; }
    else { hideTip(); tipOn = false; }
  }

  function onLeave() {
    if (!hover) return;
    hover = null;
    hideTip();
    tipOn = false;
    notifyHover();
    refresh({ animate: false, quick: true });
  }

  function onClick(e) {
    const hit = hitAt(e);
    if (!hit || hit.row !== 'query') return;
    stop();
    setQuery(cfg.query === hit.index ? null : hit.index, { notify: true });
  }

  function onKey(e) {
    const n = cfg.weights.rows;
    if (!n) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      const dir = e.key === 'ArrowRight' ? 1 : -1;
      const q = cfg.query;
      const next = q == null ? (dir > 0 ? 0 : n - 1) : clamp(q + dir, 0, n - 1);
      stop();
      setQuery(next, { notify: true });
    } else if (e.key === 'Escape' && cfg.query != null) {
      stop();
      setQuery(null, { notify: true });
    }
  }

  // ------------------------------------------------------------ update and destroy

  function update(patch = {}) {
    if (!alive) return;
    cfg = normalise({ ...cfg, ...patch });
    if (sweep && 'query' in patch && cfg.query !== sweep.idx) stop();
    if (hover) {
      const lim = hover.row === 'query' ? cfg.weights.rows : cfg.weights.cols;
      if (hover.index >= lim) { hover = null; notifyHover(); }
    }
    if ('kind' in patch) restyle();
    reach = maxReach();
    refresh({ animate: true });
  }

  function destroy() {
    if (!alive) return;
    stop();
    alive = false;
    cancelAnimationFrame(raf);
    cancelAnimationFrame(layoutRaf);
    raf = 0;
    layoutRaf = 0;
    ro?.disconnect();
    offTheme();
    if (tipOn) hideTip();
    records.clear();
    active.clear();
    dots = [];
    el.remove();
  }

  return {
    el,
    update,
    setQuery: (q) => setQuery(q, { notify: false }),
    play,
    stop,
    destroy,
    get query() { return cfg.query; },
    get playing() { return sweep !== null; },
  };
}

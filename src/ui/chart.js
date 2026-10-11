// Minimal SVG charts drawn to one scale, coloured from theme tokens.
import { s, h, fmt } from './dom.js';
import { showTip, hideTip } from './tooltip.js';

function niceTicks(lo, hi, n = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => span / st <= n) || 10 * mag;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

/**
 * lineChart({ series: [{label, kind | color, points: [[x, y], ...], dashed}],
 *             width, height, xLabel, yLabel, xDomain, yDomain, markers: [{x, label}] })
 */
export function lineChart(opts) {
  const el = h('div', { class: 'chart-wrap', style: { maxWidth: '100%', overflowX: 'auto' } });
  let o = opts;
  function render() {
    const W = o.width ?? 520, H = o.height ?? 220, m = { l: 46, r: 14, t: 12, b: o.xLabel ? 38 : 24 };
    const pts = o.series.flatMap((sr) => sr.points);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]).filter(Number.isFinite);
    const [x0, x1] = o.xDomain ?? [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = o.yDomain ?? [Math.min(0, ...ys), Math.max(...ys)];
    const X = (x) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
    const Y = (y) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
    const svg = s('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': o.ariaLabel || 'line chart' });
    const g = s('g', { class: 'grid' });
    for (const t of niceTicks(y0, y1, 4)) {
      g.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
      svg.append(s('text', { x: m.l - 6, y: Y(t) + 3, 'text-anchor': 'end' }, fmt(t, Math.abs(t) < 10 ? 2 : 0)));
    }
    for (const t of niceTicks(x0, x1, 6)) svg.append(s('text', { x: X(t), y: H - m.b + 14, 'text-anchor': 'middle' }, fmt(t, Number.isInteger(t) ? 0 : 2)));
    svg.prepend(g);
    svg.append(s('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: 'var(--line-strong)' }));
    if (o.xLabel) svg.append(s('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, o.xLabel));
    if (o.yLabel) svg.append(s('text', { x: 12, y: m.t + 8 }, o.yLabel));
    for (const mk of o.markers || []) {
      svg.append(s('line', { x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: H - m.b, stroke: 'var(--accent)', 'stroke-dasharray': '3 3' }));
      if (mk.label) svg.append(s('text', { x: X(mk.x) + 4, y: m.t + 10 }, mk.label));
    }
    for (const sr of o.series) {
      const color = sr.color ?? `var(--c-${sr.kind || 'resid'})`;
      const d = sr.points.filter((p) => Number.isFinite(p[1])).map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
      if (sr.area) {
        const first = sr.points[0], last = sr.points[sr.points.length - 1];
        svg.append(s('path', { d: `${d}L${X(last[0])},${Y(Math.max(y0, 0))}L${X(first[0])},${Y(Math.max(y0, 0))}Z`, fill: color, opacity: 0.12 }));
      }
      svg.append(s('path', { d, fill: 'none', stroke: color, 'stroke-width': sr.width ?? 2, 'stroke-dasharray': sr.dashed ? '5 4' : null, 'stroke-linejoin': 'round' }));
      for (const p of sr.dots ? sr.points : []) svg.append(s('circle', { cx: X(p[0]), cy: Y(p[1]), r: 3, fill: color }));
    }
    if (o.series.length > 1 && o.legend !== false) {
      let lx = m.l + 6;
      for (const sr of o.series) {
        const color = sr.color ?? `var(--c-${sr.kind || 'resid'})`;
        svg.append(s('rect', { x: lx, y: m.t, width: 10, height: 3, fill: color }));
        const t = s('text', { x: lx + 14, y: m.t + 4 }, sr.label);
        svg.append(t);
        lx += 24 + 7 * String(sr.label).length;
      }
    }
    svg.addEventListener('pointermove', (ev) => {
      const r = svg.getBoundingClientRect();
      const x = x0 + ((ev.clientX - r.left) * (W / r.width) - m.l) / (W - m.l - m.r) * (x1 - x0);
      const lines = o.series.map((sr) => {
        let best = sr.points[0];
        for (const p of sr.points) if (Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
        return `${sr.label ?? ''} ${fmt(best[0], 2)} → <b>${fmt(best[1], 3)}</b>`;
      });
      showTip(ev.clientX, ev.clientY, lines.join('<br>'));
    });
    svg.addEventListener('pointerleave', hideTip);
    el.replaceChildren(svg);
  }
  render();
  return { el, update(p) { o = { ...o, ...p }; render(); } };
}

/**
 * barChart({ values, labels, width, height, kind, highlight: index | Set, format, onClick, horizontal })
 * Vertical bars by default; good for a distribution over a small vocabulary.
 */
export function barChart(opts) {
  const el = h('div', { class: 'chart-wrap', style: { maxWidth: '100%', overflowX: 'auto' } });
  let o = opts;
  function render() {
    const n = o.values.length;
    const W = o.width ?? Math.max(240, Math.min(720, n * 22 + 50)), H = o.height ?? 180;
    const m = { l: 40, r: 8, t: 10, b: o.labels ? 26 : 10 };
    const vmax = o.max ?? Math.max(...o.values.filter(Number.isFinite), 1e-9);
    const vmin = Math.min(0, ...o.values.filter(Number.isFinite));
    const bw = (W - m.l - m.r) / n;
    const Y = (v) => H - m.b - ((v - vmin) / (vmax - vmin || 1)) * (H - m.t - m.b);
    const hl = o.highlight instanceof Set ? o.highlight : new Set(o.highlight == null ? [] : [o.highlight]);
    const svg = s('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': o.ariaLabel || 'bar chart' });
    for (const t of niceTicks(vmin, vmax, 3)) {
      svg.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: 'var(--line)' }));
      svg.append(s('text', { x: m.l - 5, y: Y(t) + 3, 'text-anchor': 'end' }, fmt(t, 2)));
    }
    const color = `var(--c-${o.kind || 'head'})`;
    o.values.forEach((v, i) => {
      const val = Number.isFinite(v) ? v : 0;
      const y = Y(Math.max(val, 0)), y0 = Y(Math.min(val, 0));
      const r = s('rect', {
        x: m.l + i * bw + bw * 0.12, y, width: Math.max(1, bw * 0.76), height: Math.max(0.5, y0 - y),
        rx: Math.min(3, bw * 0.2), fill: color, opacity: hl.size && !hl.has(i) ? 0.35 : 1,
      });
      r.addEventListener('pointermove', (ev) => showTip(ev.clientX, ev.clientY, `${o.labels ? o.labels[i] + ' ' : ''}<b>${o.format ? o.format(v) : fmt(v, 4)}</b>`));
      r.addEventListener('pointerleave', hideTip);
      if (o.onClick) { r.style.cursor = 'pointer'; r.addEventListener('click', () => o.onClick(i, v)); }
      svg.append(r);
      if (o.labels && (bw >= 9 || hl.has(i))) svg.append(s('text', { x: m.l + i * bw + bw / 2, y: H - m.b + 14, 'text-anchor': 'middle', style: hl.has(i) ? 'fill: var(--fg)' : null }, o.labels[i]));
    });
    svg.append(s('line', { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: 'var(--line-strong)' }));
    el.replaceChildren(svg);
  }
  render();
  return { el, update(p) { o = { ...o, ...p }; render(); } };
}

// Canvas heatmap for any 2-D slice of a tensor.
//
//   const hm = heatmap({ data, rows, cols, mode: 'sequential', kind: 'attn',
//                        rowLabels, colLabels, onClick: (r, c, v) => ... });
//   parent.append(hm.el);  hm.update({ data: next });
//
// Values of -Infinity (masked attention scores) are drawn as hatched cells.

import { h, fmt } from './dom.js';
import { diverging, sequential, cssVar, onThemeChange, parseColor } from './colors.js';
import { showTip, hideTip } from './tooltip.js';

const DEFAULTS = {
  mode: 'diverging',      // 'diverging' (signed values) | 'sequential' (0..max, e.g. probabilities)
  range: null,            // max |v| (diverging) or [lo, hi] / hi (sequential); null = auto
  kind: null,             // component hue for sequential ramps ('attn', 'q', ...)
  cell: null,             // fixed cell size in px; null = fit maxWidth
  cellH: null,            // separate cell height (defaults to cell width)
  maxWidth: 560,
  minCell: 3,
  maxCell: 34,
  gap: 1,
  rowLabels: null,
  colLabels: null,
  showValues: 'auto',     // draw numbers inside cells when they fit
  digits: 2,
  highlight: null,        // { rows: [i], cols: [j], cells: [[i, j]] }
  dimOutside: false,      // fade cells not in highlight.rows/cols
  title: null,
  tooltip: null,          // (r, c, v) => html
  onHover: null,
  onLeave: null,
  onClick: null,
  ariaLabel: 'heatmap',
};

export function heatmap(options) {
  let o = { ...DEFAULTS, ...options };
  const canvas = h('canvas', { role: 'img', 'aria-label': o.ariaLabel });
  const titleEl = h('div', { class: 'heatmap-title' });
  const el = h('div', { class: 'heatmap' }, titleEl, canvas);
  let hover = null, layout = null;

  function computeLayout() {
    const ctx = canvas.getContext('2d');
    ctx.font = `11px ${cssVar('--font-mono') || 'monospace'}`;
    const labelW = o.rowLabels ? Math.ceil(Math.max(...o.rowLabels.map((l) => ctx.measureText(String(l)).width), 0)) + 8 : 0;
    const labelH = o.colLabels ? 16 : 0;
    let cw = o.cell ?? Math.floor((o.maxWidth - labelW) / Math.max(1, o.cols));
    cw = Math.max(o.minCell, Math.min(o.maxCell, cw));
    const ch = o.cellH ?? cw;
    return { labelW, labelH, cw, ch, W: labelW + o.cols * cw, H: labelH + o.rows * ch };
  }

  function valueRange() {
    if (o.mode === 'sequential') {
      if (Array.isArray(o.range)) return o.range;
      if (typeof o.range === 'number') return [0, o.range];
      let lo = Infinity, hi = -Infinity;
      for (const v of o.data) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      return [Math.min(0, lo), hi > 0 ? hi : 1];
    }
    if (typeof o.range === 'number') return o.range;
    let m = 0;
    for (const v of o.data) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
    return m || 1;
  }

  function render() {
    titleEl.hidden = !o.title;
    if (o.title) titleEl.innerHTML = o.title;
    layout = computeLayout();
    const { labelW, labelH, cw, ch, W, H } = layout;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = `${W}px`;
    canvas.style.aspectRatio = `${W} / ${H}`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const range = valueRange();
    const color = o.mode === 'sequential'
      ? (v) => sequential((v - range[0]) / (range[1] - range[0] || 1), o.kind)
      : (v) => diverging(v / range);
    const maskColor = cssVar('--c-mask');
    const fg = cssVar('--fg'), fg3 = cssVar('--fg-3'), accent = cssVar('--accent');
    const gap = cw > 6 ? o.gap : 0;
    const hl = o.highlight || {};
    const hlRows = new Set(hl.rows || []), hlCols = new Set(hl.cols || []);
    const showVals = o.showValues === true || (o.showValues === 'auto' && cw >= 30 && ch >= 18);
    ctx.font = `${Math.min(11, Math.floor(ch * 0.42))}px ${cssVar('--font-mono') || 'monospace'}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let r = 0; r < o.rows; r++) {
      for (let c = 0; c < o.cols; c++) {
        const v = o.data[r * o.cols + c];
        const x = labelW + c * cw, y = labelH + r * ch;
        const dim = o.dimOutside && (hlRows.size || hlCols.size) && !(hlRows.has(r) || hlCols.has(c));
        ctx.globalAlpha = dim ? 0.28 : 1;
        if (v === -Infinity) {
          ctx.fillStyle = cssVar('--surface-2');
          ctx.fillRect(x, y, cw - gap, ch - gap);
          ctx.strokeStyle = maskColor;
          ctx.lineWidth = 1;
          ctx.save();
          ctx.beginPath();
          ctx.rect(x, y, cw - gap, ch - gap);
          ctx.clip();
          for (let k = -ch; k < cw; k += 5) { ctx.moveTo(x + k, y + ch); ctx.lineTo(x + k + ch, y); }
          ctx.stroke();
          ctx.restore();
        } else {
          ctx.fillStyle = color(v);
          ctx.fillRect(x, y, cw - gap, ch - gap);
          if (showVals) {
            const [rr, gg, bb] = parseColor(ctx.fillStyle);
            const lum = 0.299 * rr + 0.587 * gg + 0.114 * bb;
            ctx.fillStyle = lum > 150 ? '#141922' : '#f3f5f8';
            ctx.fillText(fmt(v, o.digits).replace('−0.00', '0.00'), x + (cw - gap) / 2, y + (ch - gap) / 2 + 0.5);
          }
        }
      }
    }
    ctx.globalAlpha = 1;

    // labels
    ctx.font = `11px ${cssVar('--font-mono') || 'monospace'}`;
    if (o.rowLabels) {
      ctx.textAlign = 'right';
      const every = Math.ceil(12 / ch);
      o.rowLabels.forEach((l, r) => {
        if (r % every && !hlRows.has(r)) return;
        ctx.fillStyle = hlRows.has(r) || hover?.r === r ? fg : fg3;
        ctx.fillText(String(l), labelW - 6, labelH + r * ch + ch / 2);
      });
    }
    if (o.colLabels) {
      ctx.textAlign = 'center';
      const every = Math.ceil(14 / cw);
      o.colLabels.forEach((l, c) => {
        if (c % every && !hlCols.has(c)) return;
        ctx.fillStyle = hlCols.has(c) || hover?.c === c ? fg : fg3;
        ctx.fillText(String(l), labelW + c * cw + cw / 2, labelH / 2);
      });
    }

    // highlight outlines
    ctx.strokeStyle = accent;
    ctx.lineWidth = 2;
    for (const r of hlRows) ctx.strokeRect(labelW + 1, labelH + r * ch + 1, o.cols * cw - 2, ch - 2);
    for (const c of hlCols) ctx.strokeRect(labelW + c * cw + 1, labelH + 1, cw - 2, o.rows * ch - 2);
    for (const [r, c] of hl.cells || []) ctx.strokeRect(labelW + c * cw + 1, labelH + r * ch + 1, cw - 2 - gap, ch - 2 - gap);
    if (hover) {
      ctx.strokeStyle = fg;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(labelW + hover.c * cw + 0.75, labelH + hover.r * ch + 0.75, cw - gap - 1.5, ch - gap - 1.5);
    }
  }

  function cellAt(ev) {
    if (!layout) return null;
    const rect = canvas.getBoundingClientRect();
    const sx = layout.W / rect.width, sy = layout.H / rect.height;
    const x = (ev.clientX - rect.left) * sx - layout.labelW, y = (ev.clientY - rect.top) * sy - layout.labelH;
    const c = Math.floor(x / layout.cw), r = Math.floor(y / layout.ch);
    if (r < 0 || c < 0 || r >= o.rows || c >= o.cols) return null;
    return { r, c, v: o.data[r * o.cols + c] };
  }

  canvas.addEventListener('pointermove', (ev) => {
    const cell = cellAt(ev);
    if (!cell) { if (hover) { hover = null; render(); hideTip(); o.onLeave?.(); } return; }
    if (!hover || hover.r !== cell.r || hover.c !== cell.c) {
      hover = cell;
      render();
      o.onHover?.(cell.r, cell.c, cell.v);
    }
    const rl = o.rowLabels ? ` ${o.rowLabels[cell.r]}` : '';
    const cl = o.colLabels ? ` ${o.colLabels[cell.c]}` : '';
    const html = o.tooltip ? o.tooltip(cell.r, cell.c, cell.v) : `[${cell.r}${rl}, ${cell.c}${cl}] = <b>${fmt(cell.v, 4)}</b>`;
    if (html) showTip(ev.clientX, ev.clientY, html);
  });
  canvas.addEventListener('pointerleave', () => { hover = null; hideTip(); render(); o.onLeave?.(); });
  canvas.addEventListener('click', (ev) => { const cell = cellAt(ev); if (cell) o.onClick?.(cell.r, cell.c, cell.v); });
  if (o.onClick) canvas.style.cursor = 'pointer';

  const offTheme = onThemeChange(render);
  render();

  return {
    el,
    canvas,
    update(patch) { o = { ...o, ...patch }; render(); },
    get options() { return o; },
    destroy() { offTheme(); hideTip(); },
  };
}

// Convenience: heatmap of slice [index] of a 3-D trace tensor (e.g. one head).
export function sliceOf(tensor, index) {
  const [, R, C] = tensor.shape;
  return { data: tensor.data.subarray(index * R * C, (index + 1) * R * C), rows: R, cols: C };
}

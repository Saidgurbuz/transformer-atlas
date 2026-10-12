// Code ↔ forward pass: walk reference/model.py in execution order. Each step is one
// traced tensor: its line is highlighted on the left, the live tensor, its shape,
// its equation with real numbers and its statistics are on the right.
// ← → step, Space plays, the timeline jumps anywhere.
import { h, fmt, heatmap, codeView, segmented, kindVar, texHTML } from '../../ui/index.js';
import { useStyle } from '../kit.js';
import { infoFor, shapePairs, parseKey } from '../../core/describe.js';
import { codeLines, lineForKey } from '../../core/codemap.js';
import { mathCard } from '../../core/mathview.js';
import { nodeForKey, nodeInfo } from '../nodes.js';

useStyle('stepper', `
.sp-root { position: absolute; inset: 0; padding: 76px 24px 24px; display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) auto; gap: 16px 22px; outline: none; }
.sp-code { min-height: 0; border-radius: var(--r-lg); overflow: hidden; border: 1px solid var(--line); background: color-mix(in srgb, var(--surface) 88%, transparent); display: grid; grid-template-rows: auto minmax(0, 1fr); }
.sp-code-head { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--line); font: 600 var(--fs-xs)/1 var(--font-mono); color: var(--fg-3); }
.sp-code .code { border: 0; border-radius: 0; max-height: none; height: 100%; background: transparent; font-size: 12.5px; }
.sp-side { min-height: 0; overflow-y: auto; display: grid; gap: 14px; align-content: start; padding-right: 4px; }
.sp-step { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.sp-count { font: 600 var(--fs-xs)/1 var(--font-mono); color: var(--fg-3); }
.sp-title { font: 700 32px/1.1 var(--font-display); letter-spacing: -.01em; }
.sp-what { color: var(--fg-2); font-size: 16px; max-width: 62ch; }
.sp-card { padding: 14px 16px; border-radius: var(--r-lg); border: 1px solid var(--line); background: color-mix(in srgb, var(--surface) 88%, transparent); display: grid; gap: 10px; min-width: 0; }
.sp-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.sp-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; font: 500 var(--fs-xs)/1.3 var(--font-mono); color: var(--fg-3); }
.sp-stats b { display: block; color: var(--fg); font-size: 13px; }
.sp-bar { grid-column: 1 / -1; display: grid; gap: 8px; padding: 10px 12px; border-radius: var(--r-lg); border: 1px solid var(--line); background: color-mix(in srgb, var(--surface) 88%, transparent); }
.sp-controls { display: flex; align-items: center; gap: 8px; }
.sp-timeline { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 2px; }
.sp-group { display: grid; gap: 3px; flex: none; }
.sp-glabel { font: 600 10px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .06em; }
.sp-cells { display: flex; gap: 2px; }
.sp-cell { width: 10px; height: 18px; border: 0; border-radius: 2px; padding: 0; opacity: .32; cursor: pointer; transition: opacity .2s, transform .2s; }
.sp-cell.done { opacity: .72; }
.sp-cell.cur { opacity: 1; transform: scaleY(1.3); box-shadow: 0 0 0 1.5px var(--fg); }
.sp-cell:hover { opacity: 1; }
@media (max-width: 900px) { .sp-root { grid-template-columns: minmax(0, 1fr); grid-template-rows: 38vh minmax(0, 1fr) auto; } }
`);

const SPEEDS = [0.5, 1, 2];

function groupOf(key) {
  const m = /^L(\d+)\./.exec(key);
  if (m) return `L${m[1]}`;
  return key.startsWith('emb.') || key === 'mask' || key === 'alibi' ? 'embed' : 'head';
}

export default {
  mount(el, ctx) {
    const root = h('div', { class: 'sp-root', tabindex: '0', dataset: { ownKeys: '1' } });
    el.append(root);
    const codeHead = h('div', { class: 'sp-code-head' }, h('span', null, 'reference/model.py'), h('span', { class: 'sp-line' }));
    const cv = codeView({ lines: codeLines, maxHeight: 99999, kindOf: (k) => infoFor(k).kind,
      onTraceClick: (k) => { const key = k.replace('{i}', String(layerOf(steps[idx]) ?? 0)); const n = steps.indexOf(key); if (n >= 0) go(n); } });
    root.append(h('section', { class: 'sp-code' }, codeHead, cv.el));
    const side = h('section', { class: 'sp-side' });
    root.append(side);

    const prev = h('button', { class: 'btn icon', type: 'button', 'aria-label': 'Previous line', onClick: () => { stop(); go(idx - 1); } }, '‹');
    const play = h('button', { class: 'btn primary', type: 'button', onClick: () => (timer ? stop() : start()) }, '▶ Play');
    const next = h('button', { class: 'btn icon', type: 'button', 'aria-label': 'Next line', onClick: () => { stop(); go(idx + 1); } }, '›');
    let speed = 1;
    const speedSeg = segmented({ options: SPEEDS.map((s) => ({ value: s, label: `${s}×` })), value: 1, ariaLabel: 'Speed', onChange: (v) => { speed = v; if (timer) { stop(); start(); } } });
    const counter = h('span', { class: 'sp-count' });
    const timeline = h('div', { class: 'sp-timeline' });
    root.append(h('div', { class: 'sp-bar' }, h('div', { class: 'sp-controls' }, prev, play, next, speedSeg.el, counter,
      h('span', { class: 'muted small', style: { marginLeft: 'auto' } }, '← → step · Space plays · click a cell to jump')), timeline));

    let steps = [], idx = 0, timer = null, hm = null, slice = 0;
    const layerOf = (key) => parseKey(key ?? '').layer;

    function collect() {
      const tr = ctx.state.trace;
      steps = Object.keys(tr);
      const want = ctx.state.focusKey;
      if (want && steps.includes(want)) { idx = steps.indexOf(want); delete ctx.state.focusKey; }
      idx = Math.max(0, Math.min(idx, steps.length - 1));
    }

    function renderTimeline() {
      const groups = [];
      for (const k of steps) {
        const g = groupOf(k);
        if (!groups.length || groups[groups.length - 1].g !== g) groups.push({ g, keys: [] });
        groups[groups.length - 1].keys.push(k);
      }
      timeline.replaceChildren(...groups.map(({ g, keys }) => h('div', { class: 'sp-group' },
        h('span', { class: 'sp-glabel' }, g === 'embed' ? 'embed' : g),
        h('div', { class: 'sp-cells' }, keys.map((k) => {
          const n = steps.indexOf(k);
          return h('button', { type: 'button', class: ['sp-cell', n === idx && 'cur', n < idx && 'done'], title: `${k} · ${infoFor(k).title}`,
            style: { background: kindVar(infoFor(k).kind) }, onClick: () => { stop(); go(n); } });
        })))));
      timeline.querySelector('.sp-cell.cur')?.scrollIntoView({ block: 'nearest', inline: 'center' });
    }

    function render() {
      const st = ctx.state, key = steps[idx];
      if (!key) return;
      const t = st.trace[key], info = infoFor(key), T = st.ids.length;
      const line = lineForKey(key, st.config);
      cv.setHighlight(line ? [line] : []);
      codeHead.lastChild.textContent = line ? `line ${line}` : 'no line (derived)';
      counter.textContent = `${idx + 1} / ${steps.length}`;

      // where we are, in words
      const { layer } = parseKey(key);
      const where = layer == null ? (groupOf(key) === 'embed' ? 'Embedding' : 'Output') : `Layer ${layer}`;
      const node = nodeForKey(key, st.sel);
      const nodeTitle = nodeInfo(node, st.config)?.title;
      side.replaceChildren(
        h('div', { class: 'sp-step' }, h('span', { class: 'sp-count' }, where), h('span', { class: 'kind-tag', style: { '--kind': kindVar(info.kind) } }, key)),
        h('div', { class: 'sp-title' }, info.title),
        info.what && h('div', { class: 'sp-what' }, info.what));

      // the tensor
      const card = h('div', { class: 'sp-card', dataset: { trace: key } });
      card.append(h('div', { class: 'sp-row' },
        h('span', { class: 'shape' }, t.shape.length ? shapePairs(key, st.config, T).map(([s, n]) => `${s}=${n}`).join(' × ') : 'scalar'),
        nodeTitle && h('button', { class: 'btn', type: 'button', onClick: () => ctx.go(node, card) }, `Zoom to “${nodeTitle}”`),
        h('button', { class: 'btn ghost', type: 'button', onClick: () => ctx.inspect(key) }, 'Every value')));
      if (t.shape.length >= 2) {
        let data = t.data, rows = t.shape[0], cols = t.shape[1];
        if (t.shape.length === 3) {
          slice = Math.min(slice, t.shape[0] - 1);
          card.append(h('div', { class: 'sp-row' }, h('span', { class: 'eyebrow' }, 'head'),
            segmented({ options: Array.from({ length: t.shape[0] }, (_, i) => ({ value: i, label: `h${i}` })), value: slice, ariaLabel: 'Head', onChange: (v) => { slice = v; render(); } }).el));
          [, rows, cols] = t.shape;
          data = t.data.subarray(slice * rows * cols, (slice + 1) * rows * cols);
        }
        const labels = st.ids.map((id) => st.tokenizer.label(id));
        const prob = /probs$/.test(key) || key === 'mask';
        hm?.destroy();
        hm = heatmap({ data, rows, cols, maxWidth: 560, mode: prob ? 'sequential' : 'diverging', kind: prob ? info.kind : null, traceKey: key,
          rowLabels: rows === T ? labels : null,
          colLabels: cols === T ? labels : cols === st.config.vocab_size ? st.tokenizer.vocab.map((c) => (c === ' ' ? '␣' : c)) : null,
          highlight: rows === T ? { rows: [Math.min(st.sel.token, T - 1)] } : null,
          onClick: (r) => { if (rows === T) ctx.store.select({ token: r }); } });
        card.append(h('div', { class: 'scroll-x' }, hm.el));
        let n = 0, sum = 0, sq = 0, mn = Infinity, mx = -Infinity;
        for (const v of data) if (Number.isFinite(v)) { n++; sum += v; sq += v * v; mn = Math.min(mn, v); mx = Math.max(mx, v); }
        const mean = sum / n, std = Math.sqrt(Math.max(0, sq / n - mean * mean));
        card.append(h('div', { class: 'sp-stats' }, ...[['mean', mean], ['std', std], ['min', mn], ['max', mx]].map(([k, v]) => h('div', null, k, h('b', null, fmt(v, 3))))));
      } else {
        card.append(h('div', { class: 'sp-title', html: texHTML(`${key === 'loss' ? '\\mathcal{L}' : key} = ${fmt(t.data[0], 4)}`) }));
      }
      side.append(card);

      // the maths, with the selected token's numbers
      const r = t.shape.length === 3 ? { band: slice, r: Math.min(st.sel.token, t.shape[1] - 1), x: 0 } : t.shape.length === 2 ? { band: 0, r: Math.min(st.sel.token, t.shape[0] - 1), x: 0 } : null;
      const mc = mathCard(st, key, r, { compact: true });
      if (mc) side.append(h('div', { class: 'sp-card' }, mc));

      renderTimeline();
    }

    function go(n) {
      if (!steps.length) return;
      idx = Math.max(0, Math.min(steps.length - 1, n));
      render();
    }
    function start() {
      play.textContent = '❚❚ Pause';
      if (idx >= steps.length - 1) idx = -1;
      timer = setInterval(() => { if (idx >= steps.length - 1) { stop(); return; } go(idx + 1); }, 1600 / speed);
    }
    function stop() { clearInterval(timer); timer = null; play.textContent = '▶ Play'; }

    root.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') { e.preventDefault(); stop(); go(idx + 1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); stop(); go(idx - 1); }
      else if (e.key === ' ') { e.preventDefault(); timer ? stop() : start(); }
    });
    el.addEventListener('atlas:interrupt', stop);

    const rebuild = () => { const key = steps[idx]; collect(); if (key && steps.includes(key)) idx = steps.indexOf(key); render(); };
    const offs = [ctx.on('trace', rebuild), ctx.on('select', render)];
    collect();
    render();
    requestAnimationFrame(() => root.focus({ preventScroll: true }));
    return {
      pause: stop,
      destroy() { stop(); hm?.destroy(); offs.forEach((f) => f()); root.remove(); },
    };
  },
};

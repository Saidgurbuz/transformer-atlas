// Panels shared by the app chrome: tensor inspector, model settings, notices, theme.
import { store } from './store.js';
import { infoFor, shapePairs } from './describe.js';
import { lineForKey, codeLines, sourceForKey } from './codemap.js';
import { normalizeCell } from './cells.js';
import { cellControls } from './cellview.js';
import { mathCard } from './mathview.js';
import { PRESETS } from '../presets/index.js';
import { CONFIG_CHOICES, countParams, dHead } from '../engine/index.js';
import {
  h, clear, fmt, fmtCount, heatmap, segmented, select, slider, toggle, button, shapeBadge, kindTag, codeView, notifyThemeChange,
} from '../ui/index.js';

// ------------------------------------------------------------- inspector

/** Fill `body` with everything about one traced tensor. onZoom(key) is optional. */
export function renderInspector(body, key, { slice = 0, cell = null, onCell, onSlice, onZoom, onCode, onStepper } = {}) {
  const st = store.state, t = st.trace?.[key];
  const info = infoFor(key);
  clear(body);
  if (!t) { body.append(h('p', { class: 'muted' }, `${key} is not produced by the current configuration.`)); return; }
  const T = st.ids.length;
  cell = normalizeCell(st, key, cell ?? (st.cellFocus?.key === key ? st.cellFocus.cell : null));
  if (cell) slice = cell.band;
  let hm;
  body.append(cellControls(st, key, cell, (c) => { store.focusCell(key, c); onCell?.(c); }));
  body.append(
    h('div', { class: 'row' }, kindTag(info.kind, key), shapeBadge(shapePairs(key, st.config, T))),
    info.what && h('p', null, info.what));
  let data = t.data, rows = t.shape[0], cols = t.shape[1] ?? 1;
  if (t.shape.length === 3) {
    slice = Math.min(slice, t.shape[0] - 1);
    const seg = segmented({ options: Array.from({ length: t.shape[0] }, (_, i) => ({ value: i, label: `h${i}` })), value: slice, ariaLabel: 'Head', onChange: (v) => onSlice?.(v) });
    body.append(h('div', { class: 'row' }, h('span', { class: 'eyebrow' }, 'slice'), seg.el));
    [, rows, cols] = t.shape;
    data = t.data.subarray(slice * rows * cols, (slice + 1) * rows * cols);
  } else if (t.shape.length === 0) {
    body.append(h('div', { class: 'formula num', style: { fontSize: '1.4rem' } }, fmt(t.data[0], 4)));
  }
  if (t.shape.length >= 2) {
    const labels = st.ids.map((id) => st.tokenizer.label(id));
    const isProb = /probs$/.test(key) || key === 'mask';
    hm = heatmap({
      data, rows, cols, maxWidth: 400, mode: isProb ? 'sequential' : 'diverging', kind: isProb ? info.kind : null, traceKey: key,
      band: slice, highlight: cell ? { cells: [[cell.r, cell.x]] } : null,
      rowLabels: rows === T ? labels : null,
      colLabels: cols === T ? labels : cols === st.config.vocab_size ? st.tokenizer.vocab.map((c) => (c === ' ' ? '␣' : c)) : null,
      onClick: (r, x) => { const c = { r, x, band: slice }; store.focusCell(key, c); onCell?.(c); },
    });
    body.append(h('div', { class: 'scroll-x' }, hm.el));
    let n = 0, sum = 0, sq = 0, mn = Infinity, mx = -Infinity;
    for (const v of data) if (Number.isFinite(v)) { n++; sum += v; sq += v * v; mn = Math.min(mn, v); mx = Math.max(mx, v); }
    const mean = sum / n, std = Math.sqrt(Math.max(0, sq / n - mean * mean));
    body.append(h('dl', { class: 'kv' }, ...[['mean', fmt(mean, 4)], ['std', fmt(std, 4)], ['min', fmt(mn, 4)], ['max', fmt(mx, 4)], ['finite values', String(n)]]
      .flatMap(([a, b]) => [h('dt', null, a), h('dd', null, b)])));
  }
  const mc = mathCard(st, key, cell);
  if (mc) body.append(mc);
  const line = lineForKey(key, st.config);
  if (line) {
    body.append(h('div', { class: 'eyebrow' }, `reference/model.py · line ${line}`));
    body.append(codeView({ lines: codeLines, from: line - 3, to: line + 2, highlight: [line] }).el);
    const source = sourceForKey(key, st.config);
    if (source.region) body.append(h('details', null, h('summary', null, `Inside ${source.helper}`),
      codeView({ lines: codeLines, ...source.region }).el));
  }
  body.append(h('div', { class: 'row' },
    onCode && button('Show in full source', () => onCode(key, cell)),
    onStepper && button('Follow in code stepper', () => onStepper(key, cell))));
  if (onZoom) body.append(h('div', { class: 'row' }, button('Zoom to this tensor', () => onZoom(key), { variant: 'primary' })));
  return () => hm?.destroy();
}

// -------------------------------------------------------------- settings

/** Model, weights and input controls. */
export function renderSettings(body) {
  const st = store.state, cfg = st.config;
  clear(body);
  const set = (patch) => { store.setConfig(patch); renderSettings(body); };
  const divisors = (n) => Array.from({ length: n }, (_, i) => i + 1).filter((k) => n % k === 0);
  const textArea = h('textarea', { id: 'settings-text', class: 'input-text', rows: 3, spellcheck: 'false', style: { height: 'auto', paddingBlock: '8px' } }, st.text);
  let typing;
  textArea.addEventListener('input', () => { clearTimeout(typing); typing = setTimeout(() => store.setText(textArea.value), 150); });
  body.append(
    h('div', { class: 'eyebrow' }, 'Input'),
    textArea,
    h('div', { class: 'small muted' }, `${st.ids.length} of ${cfg.block_size} tokens · ${st.tokenizer.kind === 'bpe'
      ? 'byte-level BPE: any text is representable'
      : 'characters outside the vocabulary are skipped'}`),
    h('div', { class: 'eyebrow' }, 'Weights'),
    select({ id: 'settings-preset', value: st.weights.startsWith('preset:') ? st.weights.slice(7) : 'random',
      options: [{ value: 'random', label: 'Random weights' }, ...PRESETS.map((p) => ({ value: p.id, label: p.title }))],
      onChange: async (v) => { if (v === 'random') store.reinitialize(); else await store.loadPreset(v); renderSettings(body); } }).el,
    st.preset && h('p', { class: 'small muted' }, st.preset.description),
    h('div', { class: 'kv' }, h('dt', null, 'parameters'), h('dd', null, fmtCount(countParams(st.model))), h('dt', null, 'head width dh'), h('dd', null, String(dHead(cfg)))),
    h('div', { class: 'eyebrow' }, 'Size'),
    slider({ id: 'cfg-n-layer', label: 'layers n_layer', min: 1, max: 8, value: cfg.n_layer, onInput: (v) => set({ n_layer: v }) }).el,
    select({ id: 'cfg-d-model', label: 'model width d_model', value: cfg.d_model, options: [8, 16, 24, 32, 48, 64, 96, 128], onChange: (v) => {
      const d = Number(v);
      const nh = d % cfg.n_head === 0 ? cfg.n_head : divisors(d).filter((k) => k <= 8).pop();
      set({ d_model: d, n_head: nh, n_kv_head: nh % cfg.n_kv_head === 0 ? cfg.n_kv_head : nh });
    } }).el,
    select({ id: 'cfg-n-head', label: 'query heads n_head', value: cfg.n_head, options: divisors(cfg.d_model).filter((k) => k <= 16), onChange: (v) => set({ n_head: Number(v), n_kv_head: Number(v) }) }).el,
    select({ id: 'cfg-n-kv-head', label: 'KV heads (MHA = n_head, MQA = 1)', value: cfg.n_kv_head, options: divisors(cfg.n_head), onChange: (v) => set({ n_kv_head: Number(v) }) }).el,
    select({ id: 'cfg-d-ff', label: 'MLP width d_ff', value: cfg.d_ff, options: [16, 32, 64, 128, 172, 256, 384, 512].concat(cfg.d_ff).filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b), onChange: (v) => set({ d_ff: Number(v) }) }).el,
    h('div', { class: 'eyebrow' }, 'Architecture'),
    ...[['pos_encoding', 'positions'], ['norm', 'normalisation'], ['norm_placement', 'norm placement'], ['activation', 'MLP activation'], ['mask', 'attention mask']].map(([k, label]) =>
      h('div', { class: 'field' }, h('span', { class: 'label' }, label), segmented({ options: CONFIG_CHOICES[k], value: cfg[k], ariaLabel: label, onChange: (v) => set({ [k]: v }) }).el)),
    cfg.mask === 'sliding' && slider({ id: 'cfg-window', label: 'window', min: 1, max: cfg.block_size, value: cfg.window, onInput: (v) => set({ window: v }) }).el,
    cfg.mask === 'prefix' && slider({ id: 'cfg-prefix', label: 'prefix length', min: 0, max: cfg.block_size, value: cfg.prefix_len, onInput: (v) => set({ prefix_len: v }) }).el,
    h('div', { class: 'eyebrow' }, 'Switches (keep trained weights)'),
    toggle({ id: 'cfg-residual', label: 'residual connections', checked: cfg.residual, onChange: (v) => set({ residual: v }) }).el,
    toggle({ id: 'cfg-scale', label: 'scale scores by 1/√dh', checked: cfg.attn_scale, onChange: (v) => set({ attn_scale: v }) }).el,
    toggle({ id: 'cfg-tie', label: 'tie embedding and LM head', checked: cfg.tie_weights, onChange: (v) => set({ tie_weights: v }) }).el,
    toggle({ id: 'cfg-bias', label: 'biases', checked: cfg.bias, onChange: (v) => set({ bias: v }) }).el,
    h('div', { class: 'eyebrow' }, 'Random weights'),
    slider({ id: 'cfg-seed', label: 'seed', min: 1, max: 9999, value: st.seed, onInput: (v) => store.reinitialize({ seed: v }) }).el,
    slider({ id: 'cfg-init', label: 'init scale ×', min: 1, max: 40, value: st.initScale, onInput: (v) => store.reinitialize({ initScale: v }) }).el,
    h('p', { class: 'small muted' }, 'Changes that keep every parameter’s shape (mask, window, residual, scaling, GELU ↔ ReLU) keep the current weights, so you can ablate a trained model. Other changes start from fresh random weights.'),
  );
}

// ----------------------------------------------------------------- theme

const THEME_KEY = 'te-theme';

export function currentTheme() { return document.documentElement.getAttribute('data-theme') || 'system'; }

export function applyTheme(t) {
  if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
  else document.documentElement.removeAttribute('data-theme');
  notifyThemeChange();
}

export function cycleTheme() {
  const next = { system: 'light', light: 'dark', dark: 'system' }[currentTheme()];
  try { localStorage.setItem(THEME_KEY, next); } catch { /* storage unavailable */ }
  applyTheme(next);
  return next;
}

export function loadTheme() {
  let saved = 'system';
  try { saved = localStorage.getItem(THEME_KEY) || 'system'; } catch { /* storage unavailable */ }
  applyTheme(saved);
  return saved;
}

// --------------------------------------------------------------- notices

export function mountNotices(parent) {
  const el = h('div', { class: 'notice', role: 'status', 'aria-live': 'polite' });
  parent.append(el);
  let timer;
  return store.on('notice', ({ text, error }) => {
    el.textContent = text;
    el.classList.toggle('error', !!error);
    el.classList.add('show');
    clearTimeout(timer);
    timer = setTimeout(() => el.classList.remove('show'), 3600);
  });
}

export { PRESETS, infoFor, lineForKey, codeLines };

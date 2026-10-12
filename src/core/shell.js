// App chrome: navigation, top bar, tensor inspector, model settings, notices.
import { store } from './store.js';
import { VIEWS, GROUPS } from '../views/registry.js';
import { infoFor, shapePairs } from './describe.js';
import { lineForKey, codeLines } from './codemap.js';
import { PRESETS } from '../presets/index.js';
import { mathCard } from './mathview.js';
import { CONFIG_CHOICES, countParams, dHead } from '../engine/index.js';
import {
  h, s, clear, fmt, fmtCount, heatmap, segmented, select, slider, toggle, button, shapeBadge, kindTag,
  codeView, notifyThemeChange, kindVar, openContextMenu,
} from '../ui/index.js';

const THEME_KEY = 'te-theme';

function logo() {
  // three stacked blocks with a residual line through them
  return s('svg', { width: 26, height: 26, viewBox: '0 0 26 26', 'aria-hidden': 'true' },
    s('rect', { x: 4, y: 3, width: 18, height: 5, rx: 1.5, fill: 'var(--c-q)' }),
    s('rect', { x: 4, y: 10.5, width: 18, height: 5, rx: 1.5, fill: 'var(--c-attn)' }),
    s('rect', { x: 4, y: 18, width: 18, height: 5, rx: 1.5, fill: 'var(--c-mlp)' }),
    s('line', { x1: 13, y1: 1, x2: 13, y2: 25, stroke: 'var(--fg)', 'stroke-width': 1.6, 'stroke-linecap': 'round' }));
}

export function modelSummary(cfg) {
  const pos = { learned: 'learned pos', sinusoidal: 'sin pos', rope: 'RoPE', alibi: 'ALiBi', none: 'no pos' }[cfg.pos_encoding];
  const attn = cfg.n_kv_head === cfg.n_head ? `${cfg.n_head}H` : `${cfg.n_head}H/${cfg.n_kv_head}KV`;
  return `${cfg.n_layer}L · ${attn} · d${cfg.d_model} · ${pos} · ${cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LN'}`;
}

export function buildShell(root, { onNavigate }) {
  const app = h('div', { class: 'app' });

  // ---------------------------------------------------------------- nav
  const nav = h('nav', { class: 'nav', 'aria-label': 'Sections' });
  nav.append(h('div', { class: 'brand' }, logo(), h('div', null, 'Transformer Atlas', h('small', null, 'every tensor, every line'))));
  const links = new Map();
  for (const g of GROUPS) {
    const items = VIEWS.filter((v) => v.group === g && !v.hidden);
    if (!items.length) continue;
    nav.append(h('div', { class: 'nav-group' }, h('div', { class: 'eyebrow' }, g), items.map((v) => {
      const a = h('a', { class: 'nav-link', href: `#${v.id}`, style: { '--kind': kindVar(v.kind) }, onClick: () => app.classList.remove('nav-open') },
        h('span', { class: 'dot' }), v.title);
      links.set(v.id, a);
      return a;
    })));
  }

  // ------------------------------------------------------------- top bar
  const menuBtn = h('button', { class: 'btn icon menu-btn', 'aria-label': 'Open navigation', onClick: () => app.classList.toggle('nav-open') }, '☰');
  const textInput = h('input', {
    id: 'input-text', class: 'input-text', type: 'text', spellcheck: 'false', autocomplete: 'off',
    'aria-label': 'Input text', placeholder: 'Type text to run through the model',
  });
  let typing;
  textInput.addEventListener('input', () => { clearTimeout(typing); typing = setTimeout(() => store.setText(textInput.value), 120); });
  const tokCount = h('span', { class: 'muted small num hide-narrow', style: { whiteSpace: 'nowrap' } });
  const presetSel = h('select', { id: 'preset-select', class: 'select hide-narrow', 'aria-label': 'Model weights' });
  presetSel.addEventListener('change', () => {
    if (presetSel.value === 'random') store.reinitialize();
    else store.loadPreset(presetSel.value);
  });
  const chip = h('button', { class: 'model-chip', title: 'Model settings', onClick: () => openSettings() });
  const themeBtn = h('button', { class: 'btn icon', 'aria-label': 'Toggle colour theme', title: 'Theme', onClick: cycleTheme });
  const topbar = h('header', { class: 'topbar' }, menuBtn,
    h('div', { class: 'grow row', style: { flexWrap: 'nowrap', gap: '8px' } }, textInput, tokCount),
    presetSel, chip, themeBtn);

  const main = h('main', { class: 'main', id: 'main' });
  const viewRoot = h('div', { class: 'view', id: 'view-root' });
  main.append(viewRoot);
  app.append(nav, topbar, main);

  // ------------------------------------------------------------- drawers
  const scrim = h('div', { class: 'scrim', onClick: () => closeDrawers() });
  const inspector = drawer('Tensor inspector');
  const settings = drawer('Model settings');
  const noticeEl = h('div', { class: 'notice', role: 'status', 'aria-live': 'polite' });
  root.append(app, scrim, inspector.el, settings.el, noticeEl);

  function drawer(title) {
    const body = h('div', { class: 'drawer-body' });
    const titleEl = h('h3', null, title);
    const el = h('aside', { class: 'drawer', 'aria-hidden': 'true' },
      h('div', { class: 'drawer-head' }, titleEl, h('button', { class: 'btn icon ghost', 'aria-label': 'Close', onClick: () => closeDrawers() }, '✕')),
      body);
    return { el, body, titleEl };
  }
  function openDrawer(d) {
    closeDrawers();
    d.el.classList.add('open');
    d.el.setAttribute('aria-hidden', 'false');
    scrim.classList.add('show');
  }
  function closeDrawers() {
    for (const d of [inspector, settings]) { d.el.classList.remove('open'); d.el.setAttribute('aria-hidden', 'true'); }
    scrim.classList.remove('show');
    inspectedKey = null;
  }
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawers(); });

  // ------------------------------------------------------ tensor inspector
  let inspectedKey = null, inspectedSlice = 0;
  function renderInspector(key) {
    const st = store.state, t = st.trace?.[key];
    const info = infoFor(key);
    inspector.titleEl.textContent = info.label || info.title;
    const body = clear(inspector.body);
    if (!t) { body.append(h('p', { class: 'muted' }, `${key} is not produced by the current configuration.`)); return; }
    const T = st.ids.length;
    body.append(
      h('div', { class: 'row' }, kindTag(info.kind, key), shapeBadge(shapePairs(key, st.config, T))),
      info.what && h('p', null, info.what),
      info.formula && h('div', { class: 'formula math', html: info.formula }));

    // 3-D tensors: pick a slice (head)
    let data = t.data, rows = t.shape[0], cols = t.shape[1] ?? 1;
    if (t.shape.length === 3) {
      inspectedSlice = Math.min(inspectedSlice, t.shape[0] - 1);
      const seg = segmented({
        options: Array.from({ length: t.shape[0] }, (_, i) => ({ value: i, label: `h${i}` })),
        value: inspectedSlice, ariaLabel: 'Head',
        onChange: (v) => { inspectedSlice = v; renderInspector(key); },
      });
      body.append(h('div', { class: 'row' }, h('span', { class: 'eyebrow' }, 'slice'), seg.el));
      [, rows, cols] = t.shape;
      data = t.data.subarray(inspectedSlice * rows * cols, (inspectedSlice + 1) * rows * cols);
    } else if (t.shape.length === 0) {
      body.append(h('div', { class: 'formula num', style: { fontSize: '1.4rem' } }, fmt(t.data[0], 4)));
    }
    if (t.shape.length >= 1) {
      const labels = st.ids.map((id) => st.tokenizer.label(id));
      const isProb = /probs$/.test(key) || key === 'mask';
      const hm = heatmap({
        data, rows, cols, maxWidth: 400, mode: isProb ? 'sequential' : 'diverging', kind: isProb ? info.kind : null,
        rowLabels: rows === T ? labels : null,
        colLabels: cols === T ? labels : cols === st.config.vocab_size ? st.tokenizer.vocab.map((c) => (c === ' ' ? '␣' : c)) : null,
        onClick: (r) => { if (rows === T) store.select({ token: r }); },
      });
      body.append(h('div', { class: 'scroll-x' }, hm.el));
      // stats
      let n = 0, sum = 0, sq = 0, mn = Infinity, mx = -Infinity;
      for (const v of data) if (Number.isFinite(v)) { n++; sum += v; sq += v * v; mn = Math.min(mn, v); mx = Math.max(mx, v); }
      const mean = sum / n, std = Math.sqrt(Math.max(0, sq / n - mean * mean));
      body.append(h('dl', { class: 'kv' },
        ...[['mean', fmt(mean, 4)], ['std', fmt(std, 4)], ['min', fmt(mn, 4)], ['max', fmt(mx, 4)], ['finite values', String(n)]]
          .flatMap(([a, b]) => [h('dt', null, a), h('dd', null, b)])));
    }
    // maths for the selected token's row (first column) when the tensor has a token axis
    const cellRow = t.shape.length === 3 ? { band: inspectedSlice, r: Math.min(st.sel.token, t.shape[1] - 1), x: 0 } : t.shape.length === 2 ? { band: 0, r: Math.min(st.sel.token, t.shape[0] - 1), x: 0 } : null;
    const mc = mathCard(st, key, cellRow);
    if (mc) body.append(mc);
    // code
    const line = lineForKey(key, st.config);
    if (line) {
      body.append(h('div', { class: 'eyebrow' }, `reference/model.py · line ${line}`));
      const cv = codeView({ lines: codeLines, from: line - 3, to: line + 2, highlight: [line] });
      body.append(cv.el);
    }
    const target = VIEWS.find((v) => v.id === info.view);
    body.append(h('div', { class: 'row' },
      target && button(`Open “${target.title}”`, () => { closeDrawers(); location.hash = target.id; }, { variant: 'primary' }),
      button('Find in code stepper', () => { closeDrawers(); store.state.focusKey = key; location.hash = 'stepper'; })));
  }
  store.on('inspect', (key) => { inspectedKey = key; inspectedSlice = store.state.sel.head; renderInspector(key); openDrawer(inspector); });
  store.on('trace', () => { if (inspectedKey) renderInspector(inspectedKey); });

  // -------------------------------------------------------- model settings
  function openSettings() { renderSettings(); openDrawer(settings); }
  function renderSettings() {
    const st = store.state, cfg = st.config;
    const body = clear(settings.body);
    const set = (patch) => { store.setConfig(patch); renderSettings(); };
    const divisors = (n) => Array.from({ length: n }, (_, i) => i + 1).filter((k) => n % k === 0);
    body.append(
      h('p', { class: 'small muted' }, 'Every change re-runs the forward pass. Changes that keep every parameter’s shape (mask, window, residual, 1/√dh scaling, GELU ↔ ReLU) keep the current weights, so you can ablate a trained model. Other changes start from fresh random weights.'),
      h('div', { class: 'kv' }, h('dt', null, 'parameters'), h('dd', null, fmtCount(countParams(st.model))), h('dt', null, 'head width dh'), h('dd', null, String(dHead(cfg))), h('dt', null, 'weights'), h('dd', null, st.weights)),
      h('div', { class: 'eyebrow' }, 'Size'),
      slider({ id: 'cfg-n-layer', label: 'layers n_layer', min: 1, max: 8, value: cfg.n_layer, onInput: (v) => set({ n_layer: v }) }).el,
      select({ id: 'cfg-d-model', label: 'model width d_model', value: cfg.d_model, options: [8, 16, 24, 32, 48, 64, 96, 128], onChange: (v) => {
        const d = Number(v);
        const nh = d % cfg.n_head === 0 ? cfg.n_head : divisors(d).filter((k) => k <= 8).pop();
        set({ d_model: d, n_head: nh, n_kv_head: nh % cfg.n_kv_head === 0 ? cfg.n_kv_head : nh });
      } }).el,
      select({ id: 'cfg-n-head', label: 'query heads n_head', value: cfg.n_head, options: divisors(cfg.d_model).filter((k) => k <= 16), onChange: (v) => set({ n_head: Number(v), n_kv_head: Number(v) }) }).el,
      select({ id: 'cfg-n-kv-head', label: 'KV heads n_kv_head (MHA = n_head, MQA = 1)', value: cfg.n_kv_head, options: divisors(cfg.n_head), onChange: (v) => set({ n_kv_head: Number(v) }) }).el,
      select({ id: 'cfg-d-ff', label: 'MLP width d_ff', value: cfg.d_ff, options: [16, 32, 64, 128, 172, 256, 384, 512].concat(cfg.d_ff).filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b), onChange: (v) => set({ d_ff: Number(v) }) }).el,
      select({ id: 'cfg-block', label: 'context length block_size', value: cfg.block_size, options: [8, 16, 24, 32, 48, 64].concat(cfg.block_size).filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b), onChange: (v) => set({ block_size: Number(v) }) }).el,
      h('div', { class: 'eyebrow' }, 'Architecture'),
      ...[['pos_encoding', 'positions'], ['norm', 'normalisation'], ['norm_placement', 'norm placement'], ['activation', 'MLP activation'], ['mask', 'attention mask']].map(([k, label]) =>
        h('div', { class: 'field' }, h('span', { class: 'label' }, label),
          segmented({ options: CONFIG_CHOICES[k], value: cfg[k], ariaLabel: label, onChange: (v) => set({ [k]: v }) }).el)),
      cfg.mask === 'sliding' && slider({ id: 'cfg-window', label: 'window', min: 1, max: cfg.block_size, value: cfg.window, onInput: (v) => set({ window: v }) }).el,
      cfg.mask === 'prefix' && slider({ id: 'cfg-prefix', label: 'prefix length', min: 0, max: cfg.block_size, value: cfg.prefix_len, onInput: (v) => set({ prefix_len: v }) }).el,
      h('div', { class: 'eyebrow' }, 'Switches'),
      toggle({ id: 'cfg-residual', label: 'residual connections', checked: cfg.residual, onChange: (v) => set({ residual: v }) }).el,
      toggle({ id: 'cfg-scale', label: 'scale scores by 1/√dh', checked: cfg.attn_scale, onChange: (v) => set({ attn_scale: v }) }).el,
      toggle({ id: 'cfg-tie', label: 'tie embedding and LM head', checked: cfg.tie_weights, onChange: (v) => set({ tie_weights: v }) }).el,
      toggle({ id: 'cfg-bias', label: 'biases', checked: cfg.bias, onChange: (v) => set({ bias: v }) }).el,
      h('div', { class: 'eyebrow' }, 'Random weights'),
      slider({ id: 'cfg-seed', label: 'seed', min: 1, max: 9999, value: st.seed, onInput: (v) => { store.reinitialize({ seed: v }); } }).el,
      slider({ id: 'cfg-init', label: 'init scale × (makes untrained attention less uniform)', min: 1, max: 40, value: st.initScale, onInput: (v) => { store.reinitialize({ initScale: v }); } }).el,
      h('div', { class: 'row' }, button('Reset to defaults', () => { store.useRandomVocab(); store.setConfig({ n_layer: 2, n_head: 4, n_kv_head: 4, d_model: 64, d_ff: 256, pos_encoding: 'learned', norm: 'layernorm', norm_placement: 'pre', activation: 'gelu', mask: 'causal', residual: true, attn_scale: true, tie_weights: false, bias: true, block_size: 32 }); renderSettings(); })),
    );
  }

  // ------------------------------------------- right-click on any traced element
  document.addEventListener('contextmenu', (e) => {
    const el = e.target.closest?.('[data-trace]');
    if (!el) return;
    e.preventDefault();
    const key = el.dataset.trace, info = infoFor(key), line = lineForKey(key, store.state.config);
    const target = VIEWS.find((v) => v.id === info.view);
    openContextMenu(e.clientX, e.clientY, [
      { label: `Show code${line ? ` · model.py:${line}` : ''}`, hint: 'Inspector with the producing line', run: () => store.inspect(key) },
      { label: 'Find in 3D explorer', run: () => { store.state.focusKey = key; location.hash = 'explorer'; } },
      { label: 'Step to it in the code stepper', run: () => { store.state.focusKey = key; location.hash = 'stepper'; } },
      target && { label: `Deep dive: ${target.title}`, run: () => { location.hash = target.id; } },
    ].filter(Boolean), { title: `${key} · ${info.title}` });
  });

  // ------------------------------------------------------------- notices
  let noticeTimer;
  store.on('notice', ({ text, error }) => {
    noticeEl.textContent = text;
    noticeEl.classList.toggle('error', !!error);
    noticeEl.classList.add('show');
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => noticeEl.classList.remove('show'), 3600);
  });

  // ---------------------------------------------------------------- theme
  function applyTheme(t) {
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
    else document.documentElement.removeAttribute('data-theme');
    themeBtn.textContent = t === 'dark' ? '☾' : t === 'light' ? '☀' : '◐';
    themeBtn.title = `Theme: ${t}`;
    notifyThemeChange();
  }
  function cycleTheme() {
    const cur = document.documentElement.getAttribute('data-theme') || 'system';
    const next = { system: 'light', light: 'dark', dark: 'system' }[cur];
    try { localStorage.setItem(THEME_KEY, next); } catch { /* storage unavailable */ }
    applyTheme(next);
  }
  let saved = 'system';
  try { saved = localStorage.getItem(THEME_KEY) || 'system'; } catch { /* storage unavailable */ }
  applyTheme(saved);

  // ------------------------------------------------------------- syncing
  function syncTop() {
    const st = store.state;
    if (document.activeElement !== textInput) textInput.value = st.text;
    tokCount.textContent = `${st.ids.length}/${st.config.block_size} tok`;
    chip.textContent = modelSummary(st.config);
    presetSel.replaceChildren(
      h('option', { value: 'random', selected: st.weights === 'random' }, 'Random weights'),
      ...PRESETS.map((p) => h('option', { value: p.id, selected: st.weights === `preset:${p.id}` }, p.title)));
    presetSel.hidden = !PRESETS.length;
  }
  store.on('trace', syncTop);

  return {
    viewRoot,
    main,
    setActive(id) {
      for (const [vid, a] of links) {
        if (vid === id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
      }
    },
    markStub(id, isStub) { const a = links.get(id); if (!a) return; a.querySelector('.stub')?.remove(); if (isStub) a.append(h('span', { class: 'stub' }, 'soon')); },
    closeDrawers,
    onNavigate,
  };
}

// The Atlas: a full-screen, zoomable hierarchy of scenes.
//   click a part (data-node) to zoom in · Esc / breadcrumb to zoom out ·
//   ← → (or the story bar) to walk the forward pass · right-click anything for maths and code.
// Scenes live in src/atlas/scenes/*.js and default-export {mount(el, ctx) -> {destroy, pause?, resume?, step?}}.
import { store } from '../core/store.js';
import { PATTERNS, nodeInfo, pathTo, storyOrder, validFor, nodeForKey, nodeForLine, EXTRAS, resolve } from './nodes.js';
import { renderInspector, renderSettings, cycleTheme, loadTheme, currentTheme, mountNotices, infoFor, lineForKey, codeLines } from '../core/panels.js';
import { mathCard, mathTipHTML, mathFor } from '../core/mathview.js';
import { h, clear, fmt, codeView, openContextMenu, reducedMotion, kindVar, texHTML } from '../ui/index.js';
import * as ui from '../ui/index.js';
import * as kit from './kit.js';
import * as engine from '../engine/index.js';
import { notesFor, topicsFor } from '../core/notes.js';
import { VERSION } from '../version.js';
import { REFERENCES, cite } from '../core/references.js';
import { keysForLine, sourceForKey } from '../core/codemap.js';
import { normalizeCell, cellValue, cellLabel } from '../core/cells.js';
import { cellControls } from '../core/cellview.js';

const DUR = 760;
const EASE = 'cubic-bezier(.65,0,.35,1)';

export function startAtlas(root) {
  // -------------------------------------------------------------- DOM
  const stageEl = h('div', { class: 'atlas-stage' });
  const crumbs = h('nav', { class: 'atlas-crumbs', 'aria-label': 'Where you are' });
  const tools = h('div', { class: 'atlas-tools' });
  const storyBar = h('div', { class: 'atlas-story', hidden: true });
  const hint = h('div', { class: 'atlas-hint' }, 'Click a part to zoom in · Esc to zoom out · → to follow the story · right-click for maths & code');
  const hover = h('div', { class: 'atlas-hover', role: 'tooltip' });
  const leftDrawer = h('aside', { class: 'atlas-drawer left', 'aria-hidden': 'true' });
  const rightDrawer = h('aside', { class: 'atlas-drawer right', 'aria-hidden': 'true' });
  const leftEdge = h('div', { class: 'atlas-edge left', title: 'Menu' }, h('span', null, '☰'));
  const rightEdge = h('div', { class: 'atlas-edge right', title: 'Code' }, h('span', null, '</>'));
  const app = h('div', { class: 'atlas' }, stageEl, h('header', { class: 'atlas-top' }, crumbs, tools), storyBar, hint, hover, leftEdge, rightEdge, leftDrawer, rightDrawer);
  root.append(app);
  mountNotices(app);
  loadTheme();

  const toolBtn = (icon, label, run, extra = {}) => h('button', { type: 'button', class: 'atlas-tool', title: label, 'aria-label': label, onClick: run, ...extra }, icon);
  const storyBtn = toolBtn('▶', 'Story: walk the forward pass (→)', () => toggleStory());
  const themeBtn = toolBtn('◐', 'Theme', () => { const t = cycleTheme(); themeBtn.textContent = t === 'dark' ? '☾' : t === 'light' ? '☀' : '◐'; });
  tools.append(
    storyBtn,
    toolBtn('∑', 'Maths for this view', () => openRight('maths')),
    toolBtn('</>', 'Code for this view', () => openRight('code')),
    toolBtn(h('span', { class: 'icon-i', 'aria-hidden': 'true' }, 'i'), 'Notes & sources: how to read this view, with references', () => openRight('notes')),
    toolBtn('⚙', 'Model, weights and input', () => openRight('model')),
    themeBtn,
    toolBtn('☰', 'Menu: every part of the model', () => openLeft()));
  themeBtn.textContent = currentTheme() === 'dark' ? '☾' : currentTheme() === 'light' ? '☀' : '◐';

  // ---------------------------------------------------------- scene stack
  /** stack[i] = {id, info, el, inst, origin (rect it was zoomed from, in stage coords), offs} */
  let stack = [];
  let busy = Promise.resolve();
  const cfg = () => store.state.config;

  function stageRect() { return stageEl.getBoundingClientRect(); }

  function sceneCtx(id, info, layer) {
    const offs = layer.offs;
    return {
      store, ui, kit, engine,
      get state() { return store.state; },
      get cfg() { return store.state.config; },
      node: info,
      params: Object.fromEntries(Object.entries(info.params).map(([k, v]) => [k, Number.isFinite(Number(v)) ? Number(v) : v])),
      on(event, fn) { const off = store.on(event, fn); offs.push(off); return off; },
      go: (target, origin) => go(target, { origin }),
      up: () => up(),
      showMath: (key, cell) => openRight('maths', { key, cell }),
      showCode: (keys) => openRight('code', typeof keys === 'string' ? { key: keys } : { keys }),
      inspect: (key, cell) => openRight('inspect', { key, cell }),
      label: (i) => store.state.tokenizer.label(store.state.ids[i]),
      /** Declare in-scene steps; story mode walks them before leaving the scene. */
      setSteps(labels, onStep) { layer.steps = { labels, onStep, index: 0 }; renderStory(); },
      setStep(i) { if (layer.steps) { layer.steps.index = i; layer.steps.onStep?.(i); renderStory(); } },
      layerEl: layer.el,
    };
  }

  async function mountLayer(id) {
    const info = nodeInfo(id, cfg());
    const selection = {};
    if (info.params.l != null) selection.layer = Number(info.params.l);
    if (info.params.h != null) selection.head = Number(info.params.h);
    store.select(selection, { keepFocus: true });
    const el = h('section', { class: 'atlas-layer', 'aria-label': info.title, dataset: { id } });
    stageEl.append(el);
    const layer = { id, info, el, inst: null, offs: [], steps: null };
    try {
      const mod = await info.def.scene();
      const scene = mod.default ?? mod;
      layer.inst = (await scene.mount(el, sceneCtx(id, info, layer))) || {};
    } catch (err) {
      console.error(`[atlas] scene ${id} failed`, err);
      el.append(h('div', { class: 'atlas-error' }, h('strong', null, `“${info.title}” failed to load.`), h('pre', null, String(err?.stack || err))));
      layer.inst = {};
    }
    return layer;
  }

  function disposeLayer(layer) {
    if (!layer) return;
    try { layer.inst?.destroy?.(); } catch (e) { console.error(e); }
    layer.offs.forEach((f) => f());
    layer.offs = [];
    layer.el?.remove();
  }

  function hideLayer(layer) { layer.el.classList.add('hidden'); layer.el.inert = true; try { layer.inst?.pause?.(); } catch { /* ignore */ } }
  function showLayer(layer) { layer.el.classList.remove('hidden'); layer.el.inert = false; try { layer.inst?.resume?.(); } catch { /* ignore */ } }

  function relRect(origin) {
    const S = stageRect();
    if (!origin) return { x: S.width * 0.4, y: S.height * 0.4, w: S.width * 0.2, h: S.height * 0.2 };
    const r = origin instanceof Element ? origin.getBoundingClientRect() : origin;
    return { x: r.left - S.left, y: r.top - S.top, w: Math.max(8, r.width), h: Math.max(8, r.height) };
  }

  function animateIn(newEl, oldEl, r) {
    const S = stageRect();
    if (reducedMotion()) return Promise.resolve();
    const s = Math.max(0.04, Math.min(r.w / S.width, r.h / S.height));
    const cx = r.x + r.w / 2 - S.width / 2, cy = r.y + r.h / 2 - S.height / 2;
    const zoomOld = Math.min(5, Math.max(1.6, 1 / s));
    const a = newEl.animate([
      { transform: `translate(${cx}px, ${cy}px) scale(${s})`, opacity: 0, filter: 'blur(4px)' },
      { opacity: 1, offset: 0.55 },
      { transform: 'none', opacity: 1, filter: 'blur(0)' },
    ], { duration: DUR, easing: EASE });
    if (oldEl) {
      oldEl.style.transformOrigin = `${r.x + r.w / 2}px ${r.y + r.h / 2}px`;
      oldEl.animate([{ transform: 'none', opacity: 1 }, { transform: `scale(${zoomOld})`, opacity: 0 }], { duration: DUR, easing: EASE, fill: 'forwards' });
    }
    return a.finished.catch(() => {});
  }

  function animateOut(curEl, parentEl, r) {
    const S = stageRect();
    if (reducedMotion()) return Promise.resolve();
    const s = Math.max(0.04, Math.min(r.w / S.width, r.h / S.height));
    const cx = r.x + r.w / 2 - S.width / 2, cy = r.y + r.h / 2 - S.height / 2;
    const zoomOld = Math.min(5, Math.max(1.6, 1 / s));
    parentEl.style.transformOrigin = `${r.x + r.w / 2}px ${r.y + r.h / 2}px`;
    parentEl.animate([{ transform: `scale(${zoomOld})`, opacity: 0 }, { transform: 'none', opacity: 1 }], { duration: DUR, easing: EASE });
    return curEl.animate([
      { transform: 'none', opacity: 1 },
      { opacity: 0.6, offset: 0.5 },
      { transform: `translate(${cx}px, ${cy}px) scale(${s})`, opacity: 0, filter: 'blur(4px)' },
    ], { duration: DUR, easing: EASE, fill: 'forwards' }).finished.catch(() => {});
  }

  function animateSlide(newEl, oldEl, dir) {
    if (reducedMotion()) return Promise.resolve();
    oldEl?.animate([{ transform: 'none', opacity: 1 }, { transform: `translateX(${-dir * 9}%)`, opacity: 0 }], { duration: 560, easing: EASE, fill: 'forwards' });
    return newEl.animate([{ transform: `translateX(${dir * 9}%)`, opacity: 0 }, { transform: 'none', opacity: 1 }], { duration: 560, easing: EASE }).finished.catch(() => {});
  }

  /** Navigate to a node. origin: element or DOMRect the zoom grows from. */
  let requested = null;     // latest navigation target, even while an earlier one is still animating
  function go(id, { origin = null, push = true } = {}) {
    requested = id;
    busy = busy.then(() => navigate(id, { origin, push })).catch((e) => console.error(e));
    return busy;
  }

  async function navigate(id, { origin, push }) {
    if (!resolve(id) || !validFor(id, cfg())) id = 'model';
    const cur = stack[stack.length - 1];
    if (cur?.id === id) return;
    const focus = store.state.cellFocus;
    if (focus && id !== 'stepper' && !nodeInfo(id, cfg()).keys.includes(focus.key) && nodeForKey(focus.key, store.state.sel) !== id) store.state.cellFocus = null;
    const path = pathTo(id, cfg());
    const curPath = stack.map((l) => l.id);
    let common = 0;
    while (common < Math.min(path.length, curPath.length) && path[common] === curPath[common]) common++;

    hideHover();
    if (cur && common === curPath.length) {
      // ---- deeper: zoom in (intermediate levels are recorded but not mounted)
      const r = relRect(origin);
      for (let i = common; i < path.length - 1; i++) stack.push({ id: path[i], info: nodeInfo(path[i], cfg()), el: null, inst: null, offs: [], origin: r, lazy: true });
      const layer = await mountLayer(id);
      layer.origin = r;
      stack.push(layer);
      await animateIn(layer.el, cur.el, r);
      if (cur.el) { hideLayer(cur); cur.el.getAnimations().forEach((a) => a.cancel()); }
    } else if (cur && common === path.length) {
      // ---- shallower: zoom out to an ancestor
      let target = stack[path.length - 1];
      const leaving = stack.slice(path.length);
      const firstLeaving = leaving[0];
      if (!target.el) {
        const fresh = await mountLayer(target.id);
        Object.assign(target, fresh, { origin: target.origin });
      }
      showLayer(target);
      await animateOut(cur.el, target.el, firstLeaving.origin || relRect(null));
      leaving.forEach(disposeLayer);
      stack = stack.slice(0, path.length);
    } else {
      // ---- elsewhere: slide (siblings) or fresh start
      const story = storyOrder(cfg(), { head: store.state.sel.head });
      const dir = cur && story.indexOf(id) < story.indexOf(cur.id) ? -1 : 1;
      const keep = stack.slice(0, common);
      const leaving = stack.slice(common);
      for (let i = common; i < path.length - 1; i++) keep.push({ id: path[i], info: nodeInfo(path[i], cfg()), el: null, inst: null, offs: [], origin: null, lazy: true });
      const layer = await mountLayer(id);
      layer.origin = leaving[0]?.origin ?? null;
      keep.push(layer);
      await animateSlide(layer.el, cur?.el, dir);
      leaving.forEach(disposeLayer);
      keep.slice(0, -1).forEach((l) => { if (l.el) { hideLayer(l); l.el.getAnimations().forEach((a) => a.cancel()); } });
      stack = keep;
    }
    const top = stack[stack.length - 1];
    if (location.hash !== `#${id}`) {
      if (push && cur) history.pushState(null, '', `#${id}`);
      else history.replaceState(null, '', `#${id}`);
    }
    document.title = top.id === 'model' ? 'Transformer Atlas' : `${top.info.title} · Transformer Atlas`;
    renderCrumbs();
    renderStory();
    if (rightDrawer.classList.contains('open')) renderRight();
    hint.classList.add('fade');
  }

  function up() {
    const cur = stack[stack.length - 1];
    if (cur?.info.parent) go(cur.info.parent);
  }

  // ------------------------------------------------------------- crumbs
  function renderCrumbs() {
    const items = [];
    stack.forEach((l, i) => {
      if (i) items.push(h('span', { class: 'sep', 'aria-hidden': 'true' }, '›'));
      const last = i === stack.length - 1;
      items.push(h('button', { type: 'button', class: ['crumb', last && 'cur'], style: { '--kind': kindVar(l.info.kind || 'resid') }, disabled: last, onClick: () => go(l.id) },
        h('i', null), l.info.title));
    });
    crumbs.replaceChildren(...items);
  }

  // -------------------------------------------------------------- story
  let storyOn = false, storyTimer = null, autoplay = false;
  function toggleStory(on = !storyOn) {
    storyOn = on;
    storyBtn.classList.toggle('on', on);
    if (!on) { autoplay = false; clearTimeout(storyTimer); }
    renderStory();
    if (on) {
      const order = storyOrder(cfg(), { head: store.state.sel.head });
      if (!order.includes(stack[stack.length - 1]?.id)) go(order[0]);
    }
  }
  // The reader took control: scenes stop their auto-advance. Scenes listen for
  // 'atlas:interrupt' on their element; older ones stop on pointerdown, so send both.
  function interrupt(layer) {
    if (!layer?.el) return;
    layer.el.dispatchEvent(new CustomEvent('atlas:interrupt'));
    try { layer.el.dispatchEvent(new PointerEvent('pointerdown')); } catch { /* old browsers */ }
  }

  function storyStep(delta) {
    const top = stack[stack.length - 1];
    interrupt(top);
    const st = top?.steps;
    if (st && st.labels.length) {
      const ni = st.index + delta;
      if (ni >= 0 && ni < st.labels.length) { st.index = ni; st.onStep?.(ni); renderStory(); return; }
    }
    const order = storyOrder(cfg(), { head: store.state.sel.head });
    const i = order.indexOf(top?.id);
    const next = order[i < 0 ? 0 : Math.max(0, Math.min(order.length - 1, i + delta))];
    if (next && next !== top?.id) {
      const nextPath = pathTo(next, cfg());
      // zoom in when the next node is a child of the current one, so the motion matches the hierarchy
      const childEl = nextPath.includes(top.id) ? top.el?.querySelector(`[data-node="${nextPath[nextPath.indexOf(top.id) + 1]}"]`) : null;
      go(next, { origin: childEl });
    }
  }
  function renderStory() {
    const top = stack[stack.length - 1];
    const st = top?.steps;
    const show = storyOn || (st && st.labels.length > 1);
    storyBar.hidden = !show;
    if (!show || !top) return;
    const order = storyOrder(cfg(), { head: store.state.sel.head });
    const i = order.indexOf(top.id);
    storyBar.replaceChildren(...[
      h('button', { type: 'button', class: 'atlas-tool', 'aria-label': 'Previous', onClick: () => storyStep(-1) }, '‹'),
      h('div', { class: 'story-text' },
        h('div', { class: 'story-title' }, top.info.title, storyOn && i >= 0 ? h('span', { class: 'story-count' }, `${i + 1} / ${order.length}`) : null),
        st && st.labels.length > 1
          ? h('div', { class: 'story-steps' }, st.labels.map((l, k) => h('button', { type: 'button', class: ['story-dot', k === st.index && 'cur', k < st.index && 'done'], onClick: () => { interrupt(top); st.index = k; st.onStep?.(k); renderStory(); } }, l)))
          : h('div', { class: 'story-blurb' }, top.info.blurb)),
      h('button', { type: 'button', class: 'atlas-tool', 'aria-label': 'Next', onClick: () => storyStep(1) }, '›'),
      storyOn && h('button', { type: 'button', class: ['atlas-tool', autoplay && 'on'], title: 'Auto-play', onClick: () => { autoplay = !autoplay; renderStory(); if (autoplay) tick(); } }, autoplay ? '❚❚' : '▶'),
    ].filter(Boolean));
  }
  function tick() {
    clearTimeout(storyTimer);
    if (!autoplay) return;
    storyTimer = setTimeout(() => { storyStep(1); tick(); }, 4200);
  }

  // ------------------------------------------------------------ drawers
  let rightMode = null, rightArg = {}, rightCleanup = null;
  function openRight(mode, arg = {}) {
    if (rightDrawer.classList.contains('open') && rightMode === mode && !arg.key && !arg.keys) { closeRight(); return; }
    if (['code', 'maths', 'inspect'].includes(mode) && !arg.key && !arg.keys && store.state.cellFocus) arg = { ...store.state.cellFocus };
    if (arg.key && arg.cell) store.focusCell(arg.key, arg.cell);
    rightMode = mode; rightArg = arg;
    app.classList.add('right-open');
    rightDrawer.classList.add('open');
    rightDrawer.setAttribute('aria-hidden', 'false');
    renderRight();
  }
  function closeRight() { rightCleanup?.(); rightCleanup = null; rightDrawer.classList.remove('open'); app.classList.remove('right-open'); rightDrawer.setAttribute('aria-hidden', 'true'); rightMode = null; }
  function renderRight() {
    rightCleanup?.(); rightCleanup = null;
    const top = stack[stack.length - 1];
    const body = h('div', { class: 'atlas-drawer-body' });
    const titles = { code: 'Code', maths: 'Maths', model: 'Model, weights & input', inspect: 'Tensor', notes: 'Notes & sources' };
    rightDrawer.replaceChildren(h('div', { class: 'atlas-drawer-head' }, h('h3', null, titles[rightMode]),
      h('button', { type: 'button', class: 'atlas-tool', 'aria-label': 'Close', onClick: closeRight }, '✕')), body);
    const st = store.state;
    if (rightMode === 'model') renderSettings(body);
    else if (rightMode === 'inspect') {
      const key = rightArg.key;
      const cell = normalizeCell(st, key, st.cellFocus?.key === key ? st.cellFocus.cell : rightArg.cell);
      rightCleanup = renderInspector(body, key, { cell,
        onSlice: (band) => store.focusCell(key, { ...cell, band }),
        onCode: (k, c) => openRight('code', { key: k, cell: c }),
        onStepper: (k, c) => { store.focusCell(k, c); st.focusKey = k; closeRight(); go('stepper'); },
        onZoom: (k) => { closeRight(); go(nodeForKey(k, st.sel)); } });
    }
    else if (rightMode === 'maths') {
      const keys = rightArg.key ? [rightArg.key] : (top?.info.keys || []);
      const present = keys.filter((k) => (st.trace[k] || k.startsWith('concept:')) && mathFor(k));
      if (!present.length) body.append(h('p', { class: 'muted' }, 'No equations for this view yet.'));
      for (const k of present) {
        const cell = normalizeCell(st, k, st.cellFocus?.key === k ? st.cellFocus.cell : rightArg.cell);
        const c = mathCard(st, k, cell);
        if (c) {
          const controls = cellControls(st, k, cell, (next) => store.focusCell(k, next));
          if (controls) body.append(controls);
          body.append(c);
          if (st.trace[k]) body.append(h('div', { class: 'row' }, ui.button('Show the code', () => openRight('code', { key: k, cell })),
            ui.button('Inspect every value', () => openRight('inspect', { key: k, cell }))));
        }
      }
    } else if (rightMode === 'notes') {
      const id = top?.id || 'model';
      const notes = notesFor(id);
      body.append(h('p', { class: 'small muted' }, 'Interpretation notes drafted from the sources below and fact-checked against them.'));
      for (const n of notes) {
        body.append(h('div', { class: 'note-card' },
          h('div', { class: 'eyebrow' }, n.placement),
          h('p', null, n.text),
          h('div', { class: 'small' }, 'Sources: ', n.refs.flatMap((r, i) => (i ? [', ', cite(r)] : [cite(r)])))));
      }
      if (!notes.length) body.append(h('p', { class: 'muted' }, 'No notes for this view yet.'));
      const topics = topicsFor(id);
      const refs = REFERENCES.filter((r) => r.views.some((v) => topics.includes(v)))
        .sort((a, b) => (a.type === 'paper') - (b.type === 'paper') || b.year - a.year);
      if (refs.length) {
        body.append(h('div', { class: 'eyebrow' }, `Further reading · ${refs.length}`));
        body.append(h('ul', { class: 'reading-list one-col' }, refs.map((r) => h('li', null,
          h('div', { class: 'reading-meta' }, h('span', { class: 'reading-type' }, r.type), h('span', null, /\d{4}/.test(r.org) ? r.org : `${r.org} · ${r.year}`)),
          h('a', { href: r.url, target: '_blank', rel: 'noopener' }, r.title),
          h('div', { class: 'small muted' }, r.authors),
          h('p', { class: 'small' }, r.why)))));
      }
    } else if (rightMode === 'code') {
      const keys = rightArg.keys || (rightArg.key ? [rightArg.key] : top?.info.keys || []);
      const lines = keys.map((k) => lineForKey(k, st.config)).filter(Boolean);
      const key = keys.length === 1 && st.trace[keys[0]] ? keys[0] : null;
      const cell = key ? normalizeCell(st, key, st.cellFocus?.key === key ? st.cellFocus.cell : rightArg.cell) : null;
      if (key) {
        body.append(cellControls(st, key, cell, (c) => store.focusCell(key, c)),
          h('div', { class: 'row' }, ui.button('Inspect this cell', () => openRight('inspect', { key, cell })),
            ui.button('Show the maths', () => openRight('maths', { key, cell })),
            ui.button('Follow in code stepper', () => { store.focusCell(key, cell); st.focusKey = key; closeRight(); go('stepper'); })));
        const source = sourceForKey(key, st.config);
        if (source.region) body.append(h('details', null, h('summary', null, `Inside ${source.helper}`), codeView({ lines: codeLines, ...source.region }).el));
      }
      body.append(h('p', { class: 'small muted' }, lines.length
        ? 'reference/model.py · highlighted: the lines behind this view. Click any line (or select a few) to zoom to the part of the model it computes.'
        : 'reference/model.py · click any line (or select a few) to zoom to the part of the model it computes.'));
      const cv = codeView({ lines: codeLines, highlight: lines, maxHeight: 4000, kindOf: (k) => infoFor(k).kind,
        keyForPattern: (k) => k.replace('{i}', String(lineCtx().layer)),
        isTraceActive: (k, n) => keysForLine(n, st.config, lineCtx().layer, st.trace).includes(k),
        onTraceClick: (k) => { const c = normalizeCell(st, k, st.cellFocus?.key === k ? st.cellFocus.cell : null); store.focusCell(k, c); openRight('inspect', { key: k, cell: c }); },
        onLineClick: (n) => { if (!window.getSelection()?.toString()) zoomToLine(n, cv.rows.get(n)); } });
      cv.el.classList.add('code-nav');
      wireCodeNav(cv);
      body.append(cv.el);
      requestAnimationFrame(() => {
        cv.setHighlight(lines, { scroll: false });
        // the drawer body is the scroller: bring this view's lines (or the clicked line) into view
        const n = rightArg.line ?? lines[0];
        rightArg.line = undefined;                 // only the navigation that the click caused keeps it
        cv.rows.get(n)?.scrollIntoView({ block: 'center', behavior: reducedMotion() ? 'auto' : 'smooth' });
      });
    }
  }

  // ---------------------------------------------------- code -> architecture
  function lineCtx() {
    const top = stack[stack.length - 1], p = top?.info.params || {};
    const key = rightMode === 'code' ? rightArg.key ?? (rightArg.keys?.length === 1 ? rightArg.keys[0] : null) : null;
    const layer = key && /^L(\d+)\./.exec(key)?.[1];
    return { layer: Number(layer ?? p.l ?? store.state.sel.layer) || 0, head: Number(p.h ?? store.state.sel.head) || 0 };
  }
  function zoomToLine(n, fromEl) {
    const key = keysForLine(n, cfg(), lineCtx().layer, store.state.trace)[0];
    if (key) store.focusCell(key, store.state.cellFocus?.key === key ? store.state.cellFocus.cell : null);
    const target = nodeForLine(n, cfg(), lineCtx());
    if (!target) { store.notice('This line is plumbing; it does not compute a part of the model.'); return; }
    if (target === stack[stack.length - 1]?.id) { pulseStage(); return; }
    rightArg = { ...rightArg, keys: undefined, key: undefined, line: n };     // keep the clicked line in view after the zoom
    go(target, { origin: fromEl || null });
  }
  function pulseStage() {
    const el = stack[stack.length - 1]?.el;
    el?.animate([{ filter: 'brightness(1)' }, { filter: 'brightness(1.25)' }, { filter: 'brightness(1)' }], { duration: 500 });
  }
  function wireCodeNav(cv) {
    // hover: tag the line with where it leads
    let tagged = null;
    const tag = h('span', { class: 'line-goto' });
    cv.el.addEventListener('pointermove', (e) => {
      const row = e.target.closest?.('.ln[data-line]');
      if (row === tagged) return;
      tagged = row;
      tag.remove();
      if (!row) return;
      const target = nodeForLine(Number(row.dataset.line), cfg(), lineCtx());
      if (!target) return;
      tag.textContent = target === stack[stack.length - 1]?.id ? '● here' : `→ ${nodeInfo(target, cfg())?.title}`;
      row.append(tag);
    });
    cv.el.addEventListener('pointerleave', () => { tag.remove(); tagged = null; });
    // a selection of several lines zooms to the first line that maps somewhere
    cv.el.addEventListener('mouseup', () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || !cv.el.contains(sel.anchorNode)) return;
      const lineOf = (node) => Number((node.nodeType === 1 ? node : node.parentElement)?.closest?.('.ln[data-line]')?.dataset.line);
      const a = lineOf(sel.anchorNode), b = lineOf(sel.focusNode);
      if (!a || !b) return;
      for (let n = Math.min(a, b); n <= Math.max(a, b); n++) {
        if (nodeForLine(n, cfg(), lineCtx())) { zoomToLine(n, cv.rows.get(n)); break; }
      }
    });
    // right-click: zoom, maths, inspect
    cv.el.addEventListener('contextmenu', (e) => {
      const row = e.target.closest?.('.ln[data-line]');
      if (!row) return;
      e.preventDefault();
      e.stopPropagation();
      const n = Number(row.dataset.line), target = nodeForLine(n, cfg(), lineCtx());
      const key = keysForLine(n, cfg(), lineCtx().layer, store.state.trace)[0];
      openContextMenu(e.clientX, e.clientY, [
        target && { label: `Zoom to “${nodeInfo(target, cfg())?.title}”`, run: () => zoomToLine(n, row) },
        key && store.state.trace[key] && mathFor(key) && { label: 'Show the maths', run: () => openRight('maths', { key }) },
        key && store.state.trace[key] && { label: 'Inspect every value', run: () => openRight('inspect', { key }) },
      ].filter(Boolean), { title: `model.py:${n}` });
    });
  }

  function openLeft() { renderMenu(); leftDrawer.classList.add('open'); leftDrawer.setAttribute('aria-hidden', 'false'); }
  function closeLeft() { leftDrawer.classList.remove('open'); leftDrawer.setAttribute('aria-hidden', 'true'); }
  function renderMenu() {
    const c = cfg(), top = stack[stack.length - 1]?.id;
    const item = (id, depth = 0) => {
      if (!validFor(id, c)) return null;
      const info = nodeInfo(id, c);
      return h('button', { type: 'button', class: ['menu-item', id === top && 'cur'], style: { '--depth': depth, '--kind': kindVar(info.kind) }, onClick: () => { closeLeft(); go(id); } },
        h('i', null), info.title);
    };
    const tree = [item('model'), item('tokens', 1), item('embedding', 1), item('token-embedding', 2), item('positional', 2)];
    for (let l = 0; l < c.n_layer; l++) {
      tree.push(item(`block.${l}`, 1));
      if (l === (stack.map((s) => s.info.params.l).find((x) => x !== undefined) ?? 0) * 1) {
        tree.push(item(`norm.${l}.1`, 2), item(`attention.${l}`, 2), item(`qkv.${l}`, 3));
        for (let hh = 0; hh < c.n_head; hh++) tree.push(item(`head.${l}.${hh}`, 3));
        const hh = store.state.sel.head;
        tree.push(item(`scores.${l}.${hh}`, 4), item(`mask.${l}.${hh}`, 4), item(`softmax.${l}.${hh}`, 4), item(`mix.${l}.${hh}`, 4), item(`out-proj.${l}`, 3), item(`gqa.${l}`, 3),
          item(`residual.${l}.1`, 2), item(`norm.${l}.2`, 2), item(`mlp.${l}`, 2), item(`activation.${l}`, 3), item(`neurons.${l}`, 3), item(`residual.${l}.2`, 2));
      }
    }
    tree.push(item('final-norm', 1), item('lm-head', 1), item('sampling', 2), item('generation', 1));
    leftDrawer.replaceChildren(
      h('div', { class: 'atlas-drawer-head' }, h('div', { class: 'atlas-brand' }, h('b', null, 'Transformer Atlas'), h('small', null, `v${VERSION} · every tensor, every line`)),
        h('button', { type: 'button', class: 'atlas-tool', 'aria-label': 'Close', onClick: closeLeft }, '✕')),
      h('div', { class: 'atlas-drawer-body' },
        h('div', { class: 'eyebrow' }, 'The model'), h('div', { class: 'menu-tree' }, tree.filter(Boolean)),
        h('div', { class: 'eyebrow' }, 'More'), h('div', { class: 'menu-tree' }, EXTRAS.map((id) => item(id)).filter(Boolean))));
  }

  // edge reveal: resting the pointer at a screen edge opens that drawer
  let edgeTimer;
  leftEdge.addEventListener('pointerenter', () => { edgeTimer = setTimeout(openLeft, 220); });
  rightEdge.addEventListener('pointerenter', () => { edgeTimer = setTimeout(() => openRight('code'), 220); });
  [leftEdge, rightEdge].forEach((e) => e.addEventListener('pointerleave', () => clearTimeout(edgeTimer)));
  leftEdge.addEventListener('click', openLeft);
  rightEdge.addEventListener('click', () => openRight('code'));
  leftDrawer.addEventListener('pointerleave', (e) => { if (e.relatedTarget && !leftDrawer.contains(e.relatedTarget)) closeLeft(); });

  // ------------------------------------------------- hover / click / menu
  function hideHover() { hover.classList.remove('show'); }
  function cellOf(el) {
    const c = el.closest('[data-cell]');
    if (!c) return null;
    const [r, x, band] = c.dataset.cell.split(',').map(Number);
    return { r, x, band: band || 0 };
  }
  stageEl.addEventListener('pointermove', (e) => {
    const traceEl = e.target.closest?.('[data-trace]');
    const nodeEl = e.target.closest?.('[data-node]');
    const st = store.state;
    let html = '';
    if (traceEl) {
      const key = traceEl.dataset.trace, info = infoFor(key), rawCell = cellOf(e.target), cell = rawCell && normalizeCell(st, key, rawCell), t = st.trace[key];
      let val = '';
      if (cell && t) {
        const v = cellValue(st, key, cell);
        val = `<div class="hv-val">${t.shape.length === 3 ? `head ${cell.band} · ` : ''}[${cell.r}, ${cell.x}] = <b>${fmt(v, 4)}</b></div>`;
      }
      html = `<div class="hv-title"><span style="color:${kindVar(info.kind)}">●</span> ${info.label || info.title}</div>${val}${mathTipHTML(st, key, cell)}`;
    } else if (nodeEl) {
      const info = nodeInfo(nodeEl.dataset.node, cfg());
      if (info) html = `<div class="hv-title">${info.title} <span class="hv-zoom">click to zoom in</span></div><div class="hv-blurb">${info.blurb}</div>`;
    }
    if (!html) { hideHover(); return; }
    hover.innerHTML = html;
    const pad = 16, r = hover.getBoundingClientRect();
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
    hover.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
    hover.classList.add('show');
  });
  stageEl.addEventListener('pointerleave', hideHover);
  stageEl.addEventListener('click', (e) => {
    const traceEl = e.target.closest?.('[data-trace]'), cell = cellOf(e.target);
    if (traceEl && cell) store.focusCell(traceEl.dataset.trace, cell);
    const nodeEl = e.target.closest?.('[data-node]');
    if (nodeEl && !e.defaultPrevented) { e.preventDefault(); go(nodeEl.dataset.node, { origin: nodeEl }); }
  });
  stageEl.addEventListener('contextmenu', (e) => {
    // a line of model.py shown inside a scene (e.g. the code stepper)
    const codeRow = e.target.closest?.('.code .ln[data-line]');
    if (codeRow) {
      e.preventDefault();
      const n = Number(codeRow.dataset.line), target = nodeForLine(n, cfg(), lineCtx());
      const key = keysForLine(n, cfg(), lineCtx().layer, store.state.trace)[0];
      openContextMenu(e.clientX, e.clientY, [
        target && { label: `Zoom to “${nodeInfo(target, cfg())?.title}” in the model`, run: () => zoomToLine(n, codeRow) },
        key && store.state.trace[key] && mathFor(key) && { label: 'Show the maths', run: () => openRight('maths', { key }) },
        key && store.state.trace[key] && { label: 'Inspect every value', run: () => openRight('inspect', { key }) },
      ].filter(Boolean), { title: `model.py:${n}` });
      return;
    }
    const traceEl = e.target.closest?.('[data-trace]');
    const nodeEl = e.target.closest?.('[data-node]');
    const top = stack[stack.length - 1];
    e.preventDefault();
    const items = [];
    let title = top?.info.title;
    if (traceEl) {
      const key = traceEl.dataset.trace, cell = normalizeCell(store.state, key, cellOf(e.target)), line = lineForKey(key, cfg());
      store.focusCell(key, cell);
      title = `${key} · ${infoFor(key).title}`;
      if (mathFor(key)) items.push({ label: 'Show the maths', hint: cell ? 'Equation with this cell’s numbers' : 'Equation, symbols, derivation', run: () => openRight('maths', { key, cell }) });
      items.push({ label: `Show the code${line ? ` · model.py:${line}` : ''}`, hint: cellLabel(store.state, key, cell), run: () => openRight('code', { key, cell }) });
      items.push({ label: 'Inspect every value', run: () => openRight('inspect', { key, cell }) });
      items.push({ label: 'Follow this cell in code stepper', run: () => { store.state.focusKey = key; closeRight(); go('stepper'); } });
      const target = nodeForKey(key, store.state.sel);
      if (target !== top?.id) items.push({ label: `Zoom to “${nodeInfo(target, cfg())?.title}”`, run: () => go(target, { origin: traceEl }) });
    }
    if (nodeEl) items.push({ label: `Zoom into “${nodeInfo(nodeEl.dataset.node, cfg())?.title}”`, run: () => go(nodeEl.dataset.node, { origin: nodeEl }) });
    if (!traceEl) {
      items.push({ label: 'Maths for this view', run: () => openRight('maths') }, { label: 'Code for this view', run: () => openRight('code') });
    }
    if (top?.info.parent) items.push({ label: 'Zoom out', hint: 'Esc', run: () => up() });
    openContextMenu(e.clientX, e.clientY, items, { title });
  });

  window.addEventListener('keydown', (e) => {
    if (e.target.closest?.('input, textarea, select, [contenteditable]')) return;
    // arrows and space belong to a focused component that handles them; Escape always zooms out
    if (e.key !== 'Escape' && (e.defaultPrevented || e.target.closest?.('[data-own-keys], [role="application"], [tabindex]:not(body):not(.atlas-tool)'))) return;
    if (e.key === 'Escape') {
      if (rightDrawer.classList.contains('open')) closeRight();
      else if (leftDrawer.classList.contains('open')) closeLeft();
      else up();
    } else if (e.key === 'ArrowRight') { if (!storyOn) toggleStory(true); else storyStep(1); }
    else if (e.key === 'ArrowLeft') storyStep(-1);
    else if (e.key === ' ' && storyOn) { e.preventDefault(); autoplay = !autoplay; renderStory(); if (autoplay) tick(); }
  });

  window.addEventListener('hashchange', () => {
    const id = location.hash.replace(/^#/, '') || 'model';
    if (id !== (requested ?? stack[stack.length - 1]?.id)) go(id, { push: false });
  });

  // when the model changes shape, nodes may disappear (e.g. layer 5 of a 2-layer model)
  store.on('config', () => {
    const top = stack[stack.length - 1];
    if (top && !validFor(top.id, cfg())) {
      // the open scene no longer exists for this model: stop it from reacting to the new trace
      for (const l of stack) if (!validFor(l.id, cfg())) { l.offs.forEach((f) => f()); l.offs = []; }
      go('model');
    }
    renderCrumbs();
    if (leftDrawer.classList.contains('open')) renderMenu();
  });
  store.on('trace', () => { if (rightDrawer.classList.contains('open') && rightMode !== 'model') renderRight(); });
  store.on('focus', (focus) => {
    if (rightArg.key === focus?.key) rightArg.cell = focus.cell;
    if (rightDrawer.classList.contains('open') && ['inspect', 'code', 'maths'].includes(rightMode)) renderRight();
  });

  const start = location.hash.replace(/^#/, '') || 'model';
  go(start);
  return { go, up, stack: () => stack };
}

export { PATTERNS };

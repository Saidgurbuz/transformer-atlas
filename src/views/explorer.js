// 3D Model Explorer: the forward pass as a tower of tensor sheets.
//   drag to orbit · scroll to zoom · click a layer to open it · right-click for code
// The cinematic plays the trace in execution order, flying the camera to each
// tensor while the matching line of reference/model.py lights up.
import { SceneEngine, THREE } from '../scene/engine3d.js';
import { Tower, groupOf } from '../scene/tower.js';
import { openContextMenu } from '../ui/contextmenu.js';
import { h, fmt, kindVar, codeView, showTip, hideTip, segmented } from '../ui/index.js';
import { infoFor, shapePairs, parseKey } from '../core/describe.js';
import { codeLines, lineForKey } from '../core/codemap.js';
import { VIEWS } from './registry.js';
import { mathCard, mathTipHTML, mathFor } from '../core/mathview.js';
import { tex } from '../ui/index.js';
import { PARTS } from '../scene/parts/index.js';
import { themeColor, label as sceneLabel, ease as sceneEase } from '../scene/engine3d.js';
import { CELL_X, CELL_Z } from '../scene/tower.js';

const SPEEDS = [0.5, 1, 2, 4];

export default {
  id: 'explorer',
  title: '3D model explorer',
  mount(root, ctx) {
    root.classList.add('view-full');
    const host = h('div', { class: 'scene-host' });
    const hud = h('div', { class: 'hud' });
    const stage = h('div', { class: 'stage' }, host, hud);
    const wrap = h('div', { class: 'explorer' }, stage);
    root.append(wrap);

    // ------------------------------------------------------------ engine
    const engine = new SceneEngine(host, {
      onHover: (info, ev) => hover(info, ev),
      onHoverMove: (info, ev) => hover(info, ev),
      onPick: (info) => pick(info),
      onDouble: (info) => { if (info?.traceKey) focusKey(info.traceKey, { close: true }); },
      onContext: (info, ev) => contextMenu(info, ev),
      onTheme: () => { tower.sync(ctx.state, { animate: false }); },
    });
    const tower = new Tower(engine);
    const hooks = { sync: new Set(), select: new Set(), step: new Set(), context: new Set() };
    const stepAnims = [];
    const towerSync = tower.sync.bind(tower);
    tower.sync = (...a) => { towerSync(...a); for (const f of hooks.sync) safe(f); };
    function safe(f, ...args) { try { return f(...args); } catch (e) { console.error('[explorer part]', e); return undefined; } }
    const setInsets = () => {
      const narrow = stage.clientWidth < 700;
      engine.setInsets({ left: narrow ? 0 : 196, top: 56, right: 8, bottom: narrow ? 150 : 236 });
    };
    new ResizeObserver(setInsets).observe(stage);
    setInsets();
    tower.onTokenClick = (t) => ctx.store.select({ token: t });

    // ------------------------------------------------------------ HUD
    const crumbs = h('div', { class: 'hud-crumbs' });
    const tools = h('div', { class: 'hud-tools' },
      hudBtn('Overview', 'Fly back to the whole model (Esc)', () => overview()),
      hudBtn('Open all layers', 'Show every traced tensor', () => { expandAll(true); }),
      hudBtn('Close all', 'Collapse every layer', () => { expandAll(false); }),
      hudBtn('Arcs', 'Toggle attention arcs into the selected token', () => { tower.showArcs = tower.showArcs === false; tower.syncArcs(); engine.invalidate(); }));
    const layerBox = h('div', { class: 'hud-layers' });
    const headBox = h('div', { class: 'hud-heads' });
    const legend = h('div', { class: 'hud-legend' },
      ...[['embed', 'embedding'], ['q', 'Q'], ['k', 'K'], ['v', 'V'], ['attn', 'attention'], ['o', 'O proj'], ['mlp', 'MLP'], ['norm', 'norm'], ['resid', 'residual'], ['head', 'LM head']]
        .map(([k, l]) => h('span', null, h('i', { style: { background: kindVar(k) } }), l)));
    const hint = h('div', { class: 'hud-hint' }, 'Drag to orbit · scroll to zoom · right-drag to pan · click a sheet to open its layer · double-click to fly in · right-click for code');

    // cinematic bar
    const caption = h('div', { class: 'hud-caption' });
    const playBtn = h('button', { class: 'btn primary', type: 'button', onClick: () => (playing ? pause() : play()) }, '▶ Play forward pass');
    const prevBtn = hudBtn('‹', 'Previous step', () => { pause(); goStep(stepIdx - 1); });
    const nextBtn = hudBtn('›', 'Next step', () => { pause(); goStep(stepIdx + 1); });
    let speed = 1;
    const speedSeg = segmented({ options: SPEEDS.map((s) => ({ value: s, label: `${s}×` })), value: 1, ariaLabel: 'Speed', onChange: (v) => { speed = v; } });
    const scrub = h('div', { class: 'hud-scrub' });
    const counter = h('span', { class: 'hud-counter num' });
    const cine = h('div', { class: 'hud-cine' },
      h('div', { class: 'row', style: { flexWrap: 'nowrap', gap: '6px' } }, prevBtn, playBtn, nextBtn, speedSeg.el, counter),
      scrub);

    // code dock
    const codeBody = h('div', { class: 'hud-code-body' });
    const codeTitle = h('div', { class: 'hud-code-title' }, 'reference/model.py');
    const code = h('aside', { class: 'hud-code' },
      h('div', { class: 'hud-code-head' }, codeTitle,
        h('button', { class: 'btn icon ghost', type: 'button', title: 'Hide code', 'aria-label': 'Hide code', onClick: () => code.classList.toggle('collapsed') }, '⇥')),
      codeBody);
    const cv = codeView({
      lines: codeLines, maxHeight: 2000,
      kindOf: (k) => infoFor(k).kind,
      onTraceClick: (k) => focusKey(k.includes('{i}') ? k.replace('{i}', ctx.state.sel.layer) : k),
    });
    codeBody.append(cv.el);

    hud.append(
      h('div', { class: 'hud-top' }, crumbs, tools),
      h('div', { class: 'hud-left' }, h('div', { class: 'eyebrow' }, 'Layers'), layerBox, h('div', { class: 'eyebrow' }, 'Head'), headBox, legend),
      h('div', { class: 'hud-bottom' }, caption, cine, hint));
    wrap.append(code);

    function hudBtn(text, title, onClick) { return h('button', { class: 'btn hud-btn', type: 'button', title, onClick }, text); }

    // ------------------------------------------------------------ state
    let steps = [], stepIdx = -1, playing = false, stepTimer = null, focusGroup = null, focusedKey = null;

    function render({ animate = true } = {}) {
      const st = ctx.state;
      tower.sync(st, { animate });
      steps = Object.keys(st.trace).filter((k) => st.trace[k].shape.length >= 2);
      renderScrub();
      renderLayers();
      renderHeads();
      renderCrumbs();
      if (focusedKey && !st.trace[focusedKey]) focusedKey = null;
      showStep(focusedKey ?? steps[stepIdx] ?? null, { fly: false });
    }

    function renderLayers() {
      const st = ctx.state;
      layerBox.replaceChildren(
        chip('Embed', 'embed'),
        ...Array.from({ length: st.config.n_layer }, (_, i) => chip(`L${i}`, `L${i}`)),
        chip('Head', 'head'));
      function chip(text, g) {
        const open = tower.expanded.has(g);
        return h('button', {
          type: 'button', class: ['hud-chip', open && 'open', focusGroup === g && 'focus'], title: `${open ? 'Close' : 'Open'} ${text}`,
          onClick: () => toggleGroup(g),
        }, h('span', null, text), h('i', null, open ? '−' : '+'));
      }
    }

    function renderHeads() {
      const st = ctx.state;
      headBox.replaceChildren(segmented({
        options: Array.from({ length: st.config.n_head }, (_, i) => ({ value: i, label: `h${i}` })),
        value: st.sel.head, ariaLabel: 'Head', onChange: (v) => ctx.store.select({ head: v }),
      }).el);
    }

    function renderCrumbs() {
      const parts = [h('button', { type: 'button', onClick: () => overview() }, 'Model')];
      if (focusGroup) parts.push(h('span', null, '›'), h('button', { type: 'button', onClick: () => focusOn(focusGroup) }, groupName(focusGroup)));
      if (focusedKey) {
        const info = infoFor(focusedKey);
        parts.push(h('span', null, '›'), h('b', { style: { color: kindVar(info.kind) } }, focusedKey.replace(/^L\d+\./, '')));
      }
      crumbs.replaceChildren(...parts);
    }

    function groupName(g) { return g === 'embed' ? 'Embedding' : g === 'head' ? 'LM head' : `Layer ${g.slice(1)}`; }

    function renderScrub() {
      const st = ctx.state;
      const groups = [];
      for (const k of steps) {
        const g = groupOf(k);
        if (!groups.length || groups[groups.length - 1].g !== g) groups.push({ g, keys: [] });
        groups[groups.length - 1].keys.push(k);
      }
      scrub.replaceChildren(...groups.map(({ g, keys }) => h('div', { class: 'scrub-group' },
        h('span', { class: 'scrub-label' }, g === 'embed' ? 'emb' : g),
        h('div', { class: 'scrub-cells' }, keys.map((k) => h('button', {
          type: 'button', class: ['scrub-cell', k === steps[stepIdx] && 'cur', steps.indexOf(k) < stepIdx && 'done'],
          style: { background: kindVar(infoFor(k).kind) }, title: `${k} · ${infoFor(k).title}`,
          onClick: () => { pause(); goStep(steps.indexOf(k)); },
        }))))));
      counter.textContent = stepIdx >= 0 ? `${stepIdx + 1} / ${steps.length}` : `${steps.length} steps`;
      void st;
    }

    // ------------------------------------------------------------ focus & camera
    function overview({ duration = 1100 } = {}) {
      focusGroup = null; focusedKey = null;
      tower.setFocusGroup(null);
      const box = tower.boxAll();
      if (!box.isEmpty()) engine.flyTo({ ...engine.framing(box, new THREE.Vector3(0.62, 0.38, 0.9), 0.66), duration });
      cv.setHighlight([]);
      renderCrumbs(); renderLayers();
      showStep(null);
    }

    function focusOn(g, { open = true } = {}) {
      focusGroup = g;
      if (open && !tower.expanded.has(g) && g !== 'head') { tower.expanded.add(g); tower.sync(ctx.state); }
      tower.setFocusGroup(g);
      const box = tower.boxOfGroup(g);
      if (!box.isEmpty()) engine.flyTo({ ...engine.framing(box, new THREE.Vector3(0.8, 0.32, 0.72), 0.95), duration: 1000 });
      renderLayers(); renderCrumbs();
    }

    function focusKey(key, { close = false } = {}) {
      const g = groupOf(key);
      if (!tower.expanded.has(g) && g !== 'head' && !tower.sheets.has(key)) { tower.expanded.add(g); tower.sync(ctx.state); }
      focusGroup = g;
      tower.setFocusGroup(g);
      const sh = tower.sheets.get(key);
      if (sh) {
        const box = tower.boxOf([key]);
        engine.flyTo({ ...engine.framing(box, new THREE.Vector3(0.45, close ? 0.95 : 0.7, 0.6), close ? 0.75 : 1.1), duration: 900 });
        sh.pulse(engine);
      }
      stepIdx = steps.indexOf(key);
      showStep(key);
      renderScrub(); renderLayers();
    }

    function toggleGroup(g) {
      if (tower.expanded.has(g)) {
        tower.expanded.delete(g);
        tower.sync(ctx.state);
        if (focusGroup === g) { focusGroup = null; tower.setFocusGroup(null); }
        renderLayers();
      } else focusOn(g);
    }

    function expandAll(on) {
      const n = ctx.state.config.n_layer;
      tower.expanded = new Set(on ? ['embed', ...Array.from({ length: n }, (_, i) => `L${i}`)] : []);
      tower.sync(ctx.state);
      setTimeout(() => overview(), 60);
    }

    // ------------------------------------------------------------ captions & code
    function showStep(key, { fly = false } = {}) {
      focusedKey = key;
      tower.setActiveKey(key);
      const st = ctx.state;
      if (!key || !st.trace[key]) {
        caption.replaceChildren(h('div', { class: 'cap-title' }, st.preset ? st.preset.title : 'Random weights'),
          h('div', { class: 'cap-what' }, st.preset ? st.preset.description : 'Load a trained preset from the top bar to see learned structure.'));
        renderCrumbs();
        return;
      }
      const info = infoFor(key);
      const line = lineForKey(key, st.config);
      caption.replaceChildren(
        h('div', { class: 'cap-row' },
          h('span', { class: 'kind-tag', style: { '--kind': kindVar(info.kind) } }, key),
          h('span', { class: 'shape' }, shapePairs(key, st.config, st.ids.length).map(([s, n]) => `${s}=${n}`).join(' × ')),
          line && h('span', { class: 'muted small mono' }, `model.py:${line}`)),
        h('div', { class: 'cap-title' }, info.label || info.title),
        mathFor(key) ? tex(mathFor(key).tex, { display: true, className: 'cap-tex' }) : h('div', { class: 'cap-formula math', html: info.formula }),
        h('div', { class: 'cap-what' }, info.what));
      if (line) cv.setHighlight([line]);
      renderCrumbs();
      if (fly) focusKey(key);
    }

    // ------------------------------------------------------------ cinematic
    function stepDuration() { return 1500 / speed; }

    async function goStep(i, { animateReveal = true } = {}) {
      if (!steps.length) return;
      i = Math.max(0, Math.min(steps.length - 1, i));
      stepIdx = i;
      const key = steps[i];
      // reveal: everything before the step is complete, everything after hidden
      const st = ctx.state;
      const g = groupOf(key);
      if (g !== 'head' && !tower.expanded.has(g)) {
        tower.expanded = new Set([g]);
        tower.sync(st);
      }
      const order = new Map(steps.map((k, n) => [k, n]));
      for (const [k, sh] of tower.sheets) {
        const n = order.get(k) ?? -1;
        sh.setReveal(n < i ? 1 : n > i ? 0 : animateReveal ? 0 : 1);
      }
      focusGroup = g;
      tower.setFocusGroup(g);
      const sh = tower.sheets.get(key);
      if (sh) {
        const box = tower.boxOf([key]);
        engine.flyTo({ ...engine.framing(box, new THREE.Vector3(0.5, 0.75, 0.75), 1.35), duration: Math.min(900, stepDuration() * 0.6) });
        const custom = animateReveal && stepAnims.find((a) => a.pattern.test(key));
        for (const f of hooks.step) safe(f, { key, index: i, sheet: sh, playing, duration: stepDuration() });
        if (custom) await Promise.resolve(safe(custom.fn, key, sh, { duration: stepDuration(), api })).catch(() => {}).then(() => sh.setReveal(1));
        else if (animateReveal) await engine.tween({ duration: stepDuration() * 0.75, delay: Math.min(300, stepDuration() * 0.2), easing: (t) => t, update: (e) => sh.setReveal(e) });
      }
      showStep(key);
      renderScrub(); renderLayers();
    }

    function play() {
      playing = true;
      playBtn.textContent = '❚❚ Pause';
      if (stepIdx >= steps.length - 1) stepIdx = -1;
      const run = async () => {
        if (!playing) return;
        if (stepIdx >= steps.length - 1) { pause(); revealAll(); return; }
        await goStep(stepIdx + 1);
        if (playing) stepTimer = setTimeout(run, stepDuration() * 0.25);
      };
      run();
    }

    function pause() {
      playing = false;
      clearTimeout(stepTimer);
      playBtn.textContent = '▶ Play forward pass';
    }

    function revealAll() { for (const sh of tower.sheets.values()) sh.setReveal(1); engine.invalidate(); }

    // ------------------------------------------------------------ pointer
    function hover(info, ev) {
      for (const sh of tower.sheets.values()) sh.setHover(-1, -1);
      if (!info || !ev) { hideTip(); return; }
      const st = ctx.state;
      const d = info.object.userData;
      if (d.sheet && info.cell) {
        d.sheet.setHover(info.cell.r, info.cell.zc);
        const c = info.cell, key = info.traceKey, sh = d.sheet;
        const tok = (i) => st.tokenizer.label(st.ids[i]);
        const colName = /probs$|scores|scaled|masked|biased|^mask$/.test(key) && !/^probs$/.test(key) ? `key ${tok(c.x)}(${c.x})`
          : key === 'logits' || key === 'probs' ? `next = “${st.tokenizer.label(c.x)}”` : `dim ${c.x}`;
        const band = sh.bands > 1 ? `head ${c.band} · ` : '';
        showTip(ev.clientX, ev.clientY, `<b>${key}</b><br>${band}token “${tok(c.r)}”(${c.r}) · ${colName}<br>value <b>${fmt(c.value, 4)}</b>${mathTipHTML(st, key, c)}`);
        const line = lineForKey(key, st.config);
        if (line && !playing) cv.setHighlight([line], { scroll: false });
      } else if (d.arc) {
        const a = d.arc, tok = (i) => st.tokenizer.label(st.ids[i]);
        showTip(ev.clientX, ev.clientY, `<b>L${a.layer} head ${a.head}</b><br>“${tok(a.q)}”(${a.q}) reads “${tok(a.k)}”(${a.k})<br>weight <b>${fmt(a.w, 3)}</b>`);
      } else if (d.bar) {
        showTip(ev.clientX, ev.clientY, `next token “${st.tokenizer.label(d.bar.token)}”<br>p = <b>${fmt(d.bar.p, 4)}</b>`);
      } else if (d.plate) {
        showTip(ev.clientX, ev.clientY, `<b>${d.plate.label}</b><br>click to open`);
      } else hideTip();
      engine.invalidate();
    }

    function pick(info) {
      if (!info) return;
      const d = info.object.userData;
      if (d.sheet && info.cell) {
        const patch = { token: info.cell.r };
        const { layer } = parseKey(info.traceKey);
        if (layer != null) patch.layer = layer;
        if (d.sheet.bands > 1 && /attn\./.test(info.traceKey)) patch.head = Math.min(info.cell.band, ctx.state.config.n_head - 1);
        ctx.store.select(patch);
        const g = groupOf(info.traceKey);
        if (!tower.expanded.has(g) && g !== 'head') focusOn(g);
        else { stepIdx = steps.indexOf(info.traceKey); showStep(info.traceKey); renderScrub(); }
      } else if (d.plate) {
        focusOn(`L${d.plate.layer}`);
      } else if (d.arc) {
        ctx.store.select({ token: d.arc.k });
      }
    }

    // floating maths panel (right-click > Show the math)
    const mathPanel = h('div', { class: 'hud-math', hidden: true });
    hud.append(mathPanel);
    function showMath(key, cell) {
      const card = mathCard(ctx.state, key, cell);
      if (!card) return;
      mathPanel.replaceChildren(
        h('div', { class: 'hud-math-head' }, h('span', { class: 'kind-tag', style: { '--kind': kindVar(infoFor(key).kind) } }, key),
          h('button', { class: 'btn icon ghost', type: 'button', 'aria-label': 'Close maths', onClick: () => { mathPanel.hidden = true; } }, '✕')),
        card);
      mathPanel.hidden = false;
    }

    function contextMenu(info, ev) {
      if (!info?.traceKey) {
        openContextMenu(ev.clientX, ev.clientY, [
          { label: 'Overview', run: () => overview() },
          { label: 'Play forward pass', run: () => play() },
        ]);
        return;
      }
      const key = info.traceKey, st = ctx.state, infoK = infoFor(key), line = lineForKey(key, st.config);
      const view = VIEWS.find((v) => v.id === infoK.view);
      const g = groupOf(key);
      openContextMenu(ev.clientX, ev.clientY, [
        { label: `Show code${line ? ` · model.py:${line}` : ''}`, hint: 'Highlights the line that computes this tensor', run: () => { code.classList.remove('collapsed'); showStep(key); } },
        mathFor(key) && { label: 'Show the math', hint: info.cell ? 'Equation with this cell’s numbers plugged in' : 'Equation, symbols and derivation', run: () => showMath(key, info.cell) },
        { label: 'Fly to this tensor', run: () => focusKey(key, { close: true }) },
        { label: 'Inspect tensor values', run: () => ctx.inspect(key) },
        view && { label: `Deep dive: ${view.title}`, run: () => ctx.navigate(view.id) },
        g !== 'head' && { label: tower.expanded.has(g) ? `Close ${groupName(g)}` : `Open ${groupName(g)}`, run: () => toggleGroup(g) },
        { label: 'Play forward pass from here', run: () => { stepIdx = Math.max(-1, steps.indexOf(key) - 1); play(); } },
        ...[...hooks.context].flatMap((f) => safe(f, info) || []),
      ].filter(Boolean), { title: `${key} · ${infoK.title}` });
    }

    const onKey = (e) => {
      if (e.target.closest('input, textarea, select')) return;
      if (e.key === 'Escape') overview();
      else if (e.key === 'ArrowRight') { pause(); goStep(stepIdx + 1); }
      else if (e.key === 'ArrowLeft') { pause(); goStep(stepIdx - 1); }
      else if (e.key === ' ') { e.preventDefault(); playing ? pause() : play(); }
    };
    window.addEventListener('keydown', onKey);

    // ------------------------------------------------------------ store
    ctx.on('trace', () => render());
    ctx.on('select', (sel) => { tower.select(sel); renderHeads(); for (const f of hooks.select) safe(f, sel); });
    // ------------------------------------------------------------ plug-ins
    const api = {
      THREE, engine, tower, store: ctx.store, ctx, hud, stage,
      get state() { return ctx.state; },
      themeColor, label: sceneLabel, ease: sceneEase, CELL_X, CELL_Z,
      on: (event, fn) => ctx.on(event, fn),
      onSync(fn) { hooks.sync.add(fn); return () => hooks.sync.delete(fn); },
      onSelect(fn) { hooks.select.add(fn); return () => hooks.select.delete(fn); },
      onStep(fn) { hooks.step.add(fn); return () => hooks.step.delete(fn); },
      addContextItems(fn) { hooks.context.add(fn); return () => hooks.context.delete(fn); },
      stepAnimation(pattern, fn) { const a = { pattern, fn }; stepAnims.push(a); return () => stepAnims.splice(stepAnims.indexOf(a), 1); },
      addTool({ label: text, title, run }) { const b = hudBtn(text, title, run); tools.append(b); return () => b.remove(); },
      focusKey: (k, o) => focusKey(k, o), focusGroup: (g) => focusOn(g), overview: () => overview(),
      play: () => play(), pause: () => pause(), isPlaying: () => playing, speed: () => speed,
    };
    const disposers = [];
    for (const part of PARTS) {
      try { const d = part.install(api); if (typeof d === 'function') disposers.push(d); } catch (e) { console.error(`[explorer part ${part.id}]`, e); }
    }

    tower.expanded = new Set();
    render({ animate: true });
    setTimeout(() => overview({ duration: 1600 }), 50);
    if (ctx.state.focusKey) { const k = ctx.state.focusKey; delete ctx.state.focusKey; setTimeout(() => focusKey(k), 700); }

    return () => {
      for (const d of disposers) safe(d);
      pause();
      window.removeEventListener('keydown', onKey);
      hideTip();
      tower.dispose();
      engine.dispose();
      root.classList.remove('view-full');
    };
  },
};

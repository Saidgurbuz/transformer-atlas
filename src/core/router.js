// Hash router: #<view-id>. Mounts one view at a time and cleans up the previous one.
import { VIEWS } from '../views/registry.js';
import { h } from '../ui/index.js';
import * as ui from '../ui/index.js';
import * as engine from '../engine/index.js';
import * as describe from './describe.js';
import * as codemap from './codemap.js';
import { store } from './store.js';

export function startRouter(shell) {
  let current = null, token = 0;

  async function show() {
    const id = (location.hash || '').replace(/^#/, '') || 'explorer';
    const def = VIEWS.find((v) => v.id === id) || VIEWS[0];
    const my = ++token;
    if (current) {
      try { current.cleanup?.(); } catch (e) { console.error(e); }
      current.offs.forEach((off) => off());
      current = null;
    }
    shell.closeDrawers();
    shell.setActive(def.id);
    const root = shell.viewRoot;
    root.replaceChildren();
    shell.main.scrollTop = 0;
    document.title = `${def.title} · Transformer Atlas`;
    const offs = [];
    const ctx = {
      store,
      get state() { return store.state; },
      engine, ui, describe, codemap,
      on(event, fn) { const off = store.on(event, fn); offs.push(off); return off; },
      inspect: (key) => store.inspect(key),
      navigate: (viewId) => { location.hash = viewId; },
      root,
    };
    try {
      const mod = await def.load();
      if (my !== token) return;
      const view = mod.default;
      shell.markStub(def.id, !!view.stub);
      const cleanup = await view.mount(root, ctx);
      current = { cleanup, offs };
    } catch (err) {
      console.error(`[view ${def.id}]`, err);
      current = { offs };
      root.append(h('div', { class: 'callout warn' },
        h('strong', null, `The “${def.title}” view failed to load.`),
        h('pre', { class: 'mono small', style: { whiteSpace: 'pre-wrap', margin: 0 } }, String(err?.stack || err))));
    }
  }

  window.addEventListener('hashchange', show);
  show();
}

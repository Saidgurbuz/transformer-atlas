// Wraps a v0.1 page (src/views/*) as an atlas scene.
import { VIEWS } from '../../views/registry.js';
import * as ui from '../../ui/index.js';
import * as engine from '../../engine/index.js';
import * as describe from '../../core/describe.js';
import * as codemap from '../../core/codemap.js';
import { nodeForKey } from '../nodes.js';

export function legacy(viewId) {
  return {
    default: {
      async mount(el, ctx) {
        const def = VIEWS.find((v) => v.id === viewId);
        const box = ui.h('div', { class: 'legacy-host' });
        const view = ui.h('div', { class: ['view', viewId === 'explorer' && 'view-full'] });
        box.append(view);
        el.append(box);
        const offs = [];
        const lctx = {
          store: ctx.store, get state() { return ctx.state; }, engine, ui, describe, codemap,
          on: (e, f) => { const off = ctx.store.on(e, f); offs.push(off); return off; },
          inspect: (key) => ctx.inspect(key),
          navigate: () => {},
          root: view,
        };
        const mod = await def.load();
        const cleanup = await mod.default.mount(view, lctx);
        return { destroy() { try { cleanup?.(); } catch { /* ignore */ } offs.forEach((f) => f()); box.remove(); } };
      },
    },
  };
}
export { nodeForKey };

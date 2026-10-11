// Fallback scene: the node's title and blurb, and one zoomable card per child.
// Used by levels whose dedicated scene is not built yet.
import { stage2d, card, appear, texLabel } from '../kit.js';
import { childrenOf, nodeInfo } from '../nodes.js';
import { h } from '../../ui/index.js';
import { mathFor } from '../../core/mathview.js';

export function autoScene() {
  return {
    mount(el, ctx) {
      const st = stage2d(el);
      st.root.classList.add('k-auto');      // marks an unbuilt scene (smoke tests report it as a stub)
      const info = ctx.node;
      const kids = childrenOf(info.id, ctx.cfg).map((id) => nodeInfo(id, ctx.cfg));
      st.add(h('div', { class: 'k-title' }, info.title), { x: 800, y: 120, anchor: 'tc' });
      st.add(h('div', { class: 'k-sub' }, info.blurb), { x: 800, y: 196, anchor: 'tc' });
      const eq = info.keys.map(mathFor).find(Boolean);
      if (eq) st.add(texLabel(eq.tex, { size: 26, display: true }), { x: 800, y: 270, anchor: 'tc' });
      const cols = Math.min(4, Math.max(1, kids.length));
      const cw = 300, gap = 36, x0 = 800 - (cols * cw + (cols - 1) * gap) / 2;
      const cards = kids.map((k, i) => st.add(card({ title: k.title, sub: k.blurb, kind: k.kind, node: k.id, w: cw }),
        { x: x0 + (i % cols) * (cw + gap), y: 400 + Math.floor(i / cols) * 190 }));
      if (!kids.length) st.add(h('div', { class: 'k-sub muted' }, 'This view is being built.'), { x: 800, y: 460, anchor: 'tc' });
      appear(cards, { stagger: 60 });
      return { destroy: () => st.destroy() };
    },
  };
}

export default autoScene();

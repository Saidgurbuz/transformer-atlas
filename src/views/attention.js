// Placeholder. Replaced by the view implementation.
import { h, viewHeader } from '../ui/index.js';

export default {
  id: 'attention',
  stub: true,
  mount(root) {
    root.append(viewHeader({ eyebrow: 'Coming together', title: 'attention' }),
      h('div', { class: 'placeholder-view' }, 'This view is being built.'));
  },
};

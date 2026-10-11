// Placeholder. Replaced by the view implementation.
import { h, viewHeader } from '../ui/index.js';

export default {
  id: 'residual',
  stub: true,
  mount(root) {
    root.append(viewHeader({ eyebrow: 'Coming together', title: 'residual' }),
      h('div', { class: 'placeholder-view' }, 'This view is being built.'));
  },
};

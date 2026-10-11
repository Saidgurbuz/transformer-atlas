// Row of token chips. Clicking a chip selects that position.
import { h } from './dom.js';

export function tokenStrip({ labels, selected = null, onSelect, onHover, color }) {
  let sel = selected;
  const chips = labels.map((l, i) => {
    const b = h('button', {
      type: 'button', class: 'token', 'aria-pressed': String(i === sel),
      title: `position ${i}`,
      onClick: () => { set(i); onSelect?.(i); },
      onMouseenter: () => onHover?.(i),
      onMouseleave: () => onHover?.(null),
    }, h('span', null, l), h('small', null, i));
    if (color) b.style.background = color(i);
    return b;
  });
  function set(i) { sel = i; chips.forEach((c, j) => c.setAttribute('aria-pressed', String(j === i))); }
  return { el: h('div', { class: 'token-strip' }, chips), set, chips };
}

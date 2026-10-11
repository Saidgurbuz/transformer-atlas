// Right-click menu shared by the 3D explorer and 2D views.
//   openContextMenu(x, y, [{label, hint, run, disabled}], {title})
import { h } from './dom.js';

let current = null;

export function closeContextMenu() {
  if (!current) return;
  current.el.remove();
  current.off();
  current = null;
}

export function openContextMenu(x, y, items, { title } = {}) {
  closeContextMenu();
  const buttons = items.map((it) => h('button', {
    type: 'button', class: 'ctx-item', disabled: it.disabled,
    onClick: () => { closeContextMenu(); it.run?.(); },
  }, h('span', null, it.label), it.hint && h('small', null, it.hint)));
  const el = h('div', { class: 'ctx-menu', role: 'menu' }, title && h('div', { class: 'ctx-title' }, title), buttons);
  document.body.append(el);
  const r = el.getBoundingClientRect();
  el.style.left = `${Math.max(8, Math.min(x, window.innerWidth - r.width - 8))}px`;
  el.style.top = `${Math.max(8, Math.min(y, window.innerHeight - r.height - 8))}px`;
  requestAnimationFrame(() => el.classList.add('show'));
  buttons[0]?.focus({ preventScroll: true });
  const onDown = (e) => { if (!el.contains(e.target)) closeContextMenu(); };
  const onKey = (e) => {
    if (e.key === 'Escape') closeContextMenu();
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const i = buttons.indexOf(document.activeElement);
      buttons[(i + (e.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length]?.focus();
    }
  };
  setTimeout(() => { document.addEventListener('pointerdown', onDown, true); }, 0);
  document.addEventListener('keydown', onKey, true);
  window.addEventListener('blur', closeContextMenu);
  current = { el, off: () => { document.removeEventListener('pointerdown', onDown, true); document.removeEventListener('keydown', onKey, true); window.removeEventListener('blur', closeContextMenu); } };
  return current;
}

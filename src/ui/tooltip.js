// One shared floating tooltip for the whole app.
import { h } from './dom.js';

let tip;
function el() {
  if (!tip) { tip = h('div', { class: 'tooltip', role: 'tooltip' }); document.body.append(tip); }
  return tip;
}

export function showTip(x, y, html) {
  const t = el();
  t.innerHTML = html;
  t.classList.add('show');
  const pad = 14, r = t.getBoundingClientRect();
  let left = x + pad, top = y + pad;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - pad;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - pad;
  t.style.left = `${Math.max(8, left)}px`;
  t.style.top = `${Math.max(8, top)}px`;
}

export function hideTip() { if (tip) tip.classList.remove('show'); }

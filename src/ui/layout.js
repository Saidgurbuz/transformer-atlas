// Layout building blocks shared by every view.
import { h } from './dom.js';
import { kindVar } from './colors.js';

// Page header for a view.
export function viewHeader({ eyebrow, title, lede }) {
  return h('header', { class: 'view-head' },
    eyebrow && h('div', { class: 'eyebrow' }, eyebrow),
    h('h1', null, title),
    lede && h('p', { html: lede }));
}

export function section({ title, lede, id }, ...children) {
  return h('section', { class: 'section', id },
    (title || lede) && h('div', { class: 'section-head' }, title && h('h2', null, title), lede && h('p', { html: lede })),
    children);
}

// A card. `kind` tints the title dot with a component colour.
export function card({ title, kind, actions, class: cls }, ...children) {
  return h('div', { class: ['card', cls] },
    (title || actions) && h('div', { class: 'card-head' },
      title && h('h4', null, kind && h('i', { class: 'kind-dot', style: { width: '9px', height: '9px', borderRadius: '2px', background: kindVar(kind), display: 'inline-block' } }), title),
      actions && h('div', { class: 'row' }, actions)),
    children);
}

export function grid(cols, ...children) {
  return h('div', { class: ['grid', `cols-${cols}`], style: { '--cols': cols } }, children);
}

export function callout(kind, ...children) {
  return h('div', { class: ['callout', kind] }, children);
}

// Definition list: [[label, value], ...]
export function kv(pairs) {
  return h('dl', { class: 'kv' }, pairs.flatMap(([k, v]) => [h('dt', null, k), h('dd', null, v)]));
}

// Shape badge. dims: [['T', 7], ['d', 64]] or [7, 64]
export function shapeBadge(dims) {
  const parts = [];
  dims.forEach((d, i) => {
    if (i) parts.push(h('span', { class: 'x' }, '×'));
    if (Array.isArray(d)) parts.push(h('span', null, `${d[0]}=`, h('b', null, String(d[1]))));
    else parts.push(h('b', null, String(d)));
  });
  return h('span', { class: 'shape' }, parts);
}

// Pill in a component colour, e.g. kindTag('q', 'W_Q')
export function kindTag(kind, text) {
  return h('span', { class: 'kind-tag', style: { '--kind': kindVar(kind) } }, text);
}

export function legend(items) {
  return h('div', { class: 'legend' }, items.map(([kind, label]) => h('span', null, h('i', { style: { '--kind': kindVar(kind) } }), label)));
}

// Tabs: items [{id, label}], returns {el, set}
export function tabs(items, value, onChange) {
  let cur = value;
  const btns = items.map((it) => h('button', { type: 'button', role: 'tab', 'aria-selected': String(it.id === cur), onClick: () => { set(it.id); onChange(it.id); } }, it.label));
  function set(id) { cur = id; btns.forEach((b, i) => b.setAttribute('aria-selected', String(items[i].id === id))); }
  return { el: h('div', { class: 'tabs', role: 'tablist' }, btns), set };
}

export function empty(text) { return h('div', { class: 'empty' }, text); }

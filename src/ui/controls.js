// Form controls with labels. Each returns {el, set(value), get()}.
import { h } from './dom.js';

let uid = 0;
const nextId = (p) => `${p}-${++uid}`;

export function slider({ id, label, min, max, step = 1, value, format = (v) => v, hint, onInput }) {
  const inputId = id || nextId('slider');
  const out = h('span', { class: 'value' }, format(value));
  const input = h('input', { id: inputId, type: 'range', min, max, step, value });
  input.addEventListener('input', () => { const v = Number(input.value); out.textContent = format(v); onInput?.(v); });
  const el = h('div', { class: 'field' }, h('label', { for: inputId }, h('span', null, label), out), input, hint && h('div', { class: 'hint' }, hint));
  return { el, input, set(v) { input.value = v; out.textContent = format(Number(v)); }, get: () => Number(input.value) };
}

// options: [{value, label}] or plain strings
export function select({ id, label, options, value, hint, onChange }) {
  const selId = id || nextId('select');
  const opts = options.map((o) => (typeof o === 'object' ? o : { value: o, label: String(o) }));
  const sel = h('select', { id: selId, class: 'select' }, opts.map((o) => h('option', { value: o.value, selected: String(o.value) === String(value) }, o.label)));
  sel.addEventListener('change', () => {
    const o = opts.find((x) => String(x.value) === sel.value);
    onChange?.(o ? o.value : sel.value);
  });
  const el = label
    ? h('div', { class: 'field' }, h('label', { for: selId }, label), sel, hint && h('div', { class: 'hint' }, hint))
    : sel;
  return { el, input: sel, set(v) { sel.value = String(v); }, get: () => sel.value };
}

export function toggle({ id, label, checked = false, onChange }) {
  const input = h('input', { id: id || nextId('toggle'), type: 'checkbox', checked });
  input.addEventListener('change', () => onChange?.(input.checked));
  const el = h('label', { class: 'toggle' }, input, h('span', null, label));
  return { el, input, set(v) { input.checked = !!v; }, get: () => input.checked };
}

// Segmented buttons: options [{value, label, title}] or strings.
export function segmented({ options, value, onChange, ariaLabel }) {
  const opts = options.map((o) => (typeof o === 'object' ? o : { value: o, label: String(o) }));
  let current = value;
  const buttons = opts.map((o) => h('button', {
    type: 'button',
    title: o.title,
    'aria-pressed': String(o.value === current),
    onClick: () => { set(o.value); onChange?.(o.value); },
  }, o.label));
  const el = h('div', { class: 'segmented', role: 'group', 'aria-label': ariaLabel }, buttons);
  function set(v) { current = v; buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(opts[i].value === v))); }
  return { el, set, get: () => current };
}

export function button(label, onClick, { variant = '', title, icon } = {}) {
  return h('button', { type: 'button', class: ['btn', variant], title, onClick }, icon, label);
}

export function numberInput({ id, label, value, min, max, step = 1, onChange }) {
  const inputId = id || nextId('num');
  const input = h('input', { id: inputId, type: 'number', class: 'select', value, min, max, step, style: { width: '100%' } });
  input.addEventListener('change', () => onChange?.(Number(input.value)));
  const el = h('div', { class: 'field' }, h('label', { for: inputId }, label), input);
  return { el, input, set(v) { input.value = v; }, get: () => Number(input.value) };
}

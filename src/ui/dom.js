// Tiny DOM helpers. h() builds HTML elements, s() builds SVG elements.
//   h('div', {class: 'card', onClick: fn, style: {'--kind': 'red'}}, 'text', child)

const SVG_NS = 'http://www.w3.org/2000/svg';

function applyProps(el, props, svg) {
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class' || k === 'className') {
      el.setAttribute('class', Array.isArray(v) ? v.filter(Boolean).join(' ') : v);
    } else if (k === 'style' && typeof v === 'object') {
      for (const [sk, sv] of Object.entries(v)) {
        if (sv == null) continue;
        if (sk.startsWith('--') || sk.includes('-')) el.style.setProperty(sk, sv);
        else el.style[sk] = sv;
      }
    } else if (k.startsWith('on') && typeof v === 'function') {
      el.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (k === 'dataset') {
      Object.assign(el.dataset, v);
    } else if (k === 'html') {
      el.innerHTML = v;
    } else if (k === 'text') {
      el.textContent = v;
    } else if (k === 'ref' && typeof v === 'function') {
      v(el);
    } else if (!svg && (k === 'value' || k === 'checked' || k === 'disabled' || k === 'selected')) {
      el[k] = v;
    } else {
      el.setAttribute(k, v === true ? '' : v);
    }
  }
}

function appendChildren(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false || c === true) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  applyProps(el, props, false);
  appendChildren(el, children);
  return el;
}

export function s(tag, props, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyProps(el, props, true);
  appendChildren(el, children);
  return el;
}

export function clear(el) { el.replaceChildren(); return el; }

// Number formatting for tensor values: fixed digits, exponent for tiny/huge, −∞ for masked.
export function fmt(v, digits = 3) {
  if (v === -Infinity) return '−∞';
  if (v === Infinity) return '∞';
  if (Number.isNaN(v)) return 'NaN';
  const a = Math.abs(v);
  if (a !== 0 && (a < 10 ** -digits || a >= 1e5)) return v.toExponential(Math.max(1, digits - 1)).replace('-', '−');
  return v.toFixed(digits).replace('-', '−');
}

// 1234 -> "1.2K", 7.2e9 -> "7.2B"
export function fmtCount(n) {
  const a = Math.abs(n);
  if (a >= 1e12) return `${(n / 1e12).toFixed(a >= 1e13 ? 0 : 1)}T`;
  if (a >= 1e9) return `${(n / 1e9).toFixed(a >= 1e10 ? 0 : 1)}B`;
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(a >= 1e4 ? 0 : 1)}K`;
  return String(Math.round(n));
}

export function fmtBytes(b) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(b >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}

export function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

// Calls fn at most once per animation frame.
export function rafThrottle(fn) {
  let queued = false, lastArgs;
  return (...args) => {
    lastArgs = args;
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; fn(...lastArgs); });
  };
}

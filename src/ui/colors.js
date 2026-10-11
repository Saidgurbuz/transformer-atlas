// Theme-aware colours: component hues come from CSS tokens, heatmap ramps
// interpolate between token colours so they follow light and dark mode.

export const KINDS = ['embed', 'pos', 'q', 'k', 'v', 'attn', 'o', 'mlp', 'norm', 'resid', 'head', 'mask'];

export function cssVar(name, el = document.documentElement) {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

// CSS value for a component kind, usable in style attributes: var(--c-q)
export function kindVar(kind) { return `var(--c-${kind})`; }

export function kindColor(kind) { return cssVar(`--c-${kind}`); }

let probe;
export function parseColor(str) {
  if (!probe) probe = document.createElement('canvas').getContext('2d');
  probe.fillStyle = '#000';
  probe.fillStyle = str;
  const v = probe.fillStyle;
  if (v.startsWith('#')) return [1, 3, 5].map((i) => parseInt(v.slice(i, i + 2), 16));
  const m = v.match(/[\d.]+/g) || [0, 0, 0];
  return m.slice(0, 3).map(Number);
}

const mix = (a, b, t) => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t));
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

let cache = null;
function ramps() {
  if (cache) return cache;
  cache = {
    neg: parseColor(cssVar('--heat-neg')),
    mid: parseColor(cssVar('--heat-mid')),
    pos: parseColor(cssVar('--heat-pos')),
    lo: parseColor(cssVar('--heat-lo')),
    hi: parseColor(cssVar('--heat-hi')),
  };
  return cache;
}

// t in [-1, 1]: blue (negative) through neutral to orange (positive)
export function diverging(t) {
  const r = ramps();
  const x = Math.max(-1, Math.min(1, t));
  const e = Math.pow(Math.abs(x), 0.75);
  return rgb(x < 0 ? mix(r.mid, r.neg, e) : mix(r.mid, r.pos, e));
}

// t in [0, 1]: neutral to violet, or to a component hue when `kind` is given
export function sequential(t, kind) {
  const r = ramps();
  const hi = kind ? parseColor(kindColor(kind)) : r.hi;
  const x = Math.max(0, Math.min(1, t));
  return rgb(mix(r.lo, hi, Math.pow(x, 0.8)));
}

export function withAlpha(color, a) {
  const c = parseColor(color);
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

const themeListeners = new Set();
export function onThemeChange(fn) { themeListeners.add(fn); return () => themeListeners.delete(fn); }
export function notifyThemeChange() { cache = null; themeListeners.forEach((fn) => fn()); }

if (typeof window !== 'undefined' && window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', notifyThemeChange);
}

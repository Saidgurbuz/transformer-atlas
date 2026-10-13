// Python source viewer with line numbers, light syntax colouring, highlighted
// lines and clickable `@trace` chips (the bridge from code to tensors).
import { h } from './dom.js';
import { kindVar } from './colors.js';
import { reducedMotion } from './anim.js';

const KW = new Set(['def', 'class', 'return', 'if', 'elif', 'else', 'for', 'in', 'not', 'and', 'or', 'is', 'None', 'True',
  'False', 'import', 'from', 'as', 'with', 'lambda', 'assert', 'raise', 'try', 'except', 'pass', 'while', 'yield', 'super']);

function esc(t) { return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

export function highlightPython(src) {
  const re = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(\d+\.?\d*(?:e-?\d+)?)\b|\b([A-Za-z_]\w*)\b(?=\s*\()|\b([A-Za-z_]\w*)\b/gm;
  let out = '', last = 0, m;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    const [tok, com, str, num, fn, word] = m;
    if (com) out += `<span class="com">${esc(com)}</span>`;
    else if (str) out += `<span class="str">${esc(str)}</span>`;
    else if (num) out += `<span class="num">${num}</span>`;
    else if (fn) out += KW.has(fn) ? `<span class="kw">${fn}</span>` : `<span class="fn">${fn}</span>`;
    else if (word) out += KW.has(word) ? `<span class="kw">${word}</span>` : word === 'self' || word === 'cfg' ? `<span class="self">${word}</span>` : word;
    else out += esc(tok);
    last = m.index + tok.length;
  }
  return out + esc(src.slice(last));
}

/**
 * codeView({ lines: [{n, text, keys: [traceKeyPattern]}], from, to, highlight: Set|number[],
 *            kindOf: (key) => kind, onTraceClick: (key, line) => ..., onLineClick, maxHeight })
 * `text` should already have its @trace marker stripped (see core/codemap.js).
 */
export function codeView({ lines, from = 1, to = Infinity, highlight = [], dimOthers = false, kindOf, onTraceClick, onLineClick, maxHeight, keyForPattern, isTraceActive, followResize = false }) {
  const el = h('div', { class: 'code', style: maxHeight ? { '--code-h': `${maxHeight}px` } : null });
  const rows = new Map(), traceButtons = [];
  for (const ln of lines) {
    if (ln.n < from || ln.n > to) continue;
    const chips = (ln.keys || []).map((pattern) => {
      const k = keyForPattern?.(pattern) ?? pattern;
      const active = isTraceActive?.(k, ln.n) !== false;
      const chip = h('button', {
      type: 'button', class: 'trace-chip', style: { '--kind': kindVar(kindOf?.(k) || 'norm') },
      disabled: !active, dataset: { traceKey: k },
      title: active ? `Open the live tensor ${k}` : `${k}: this branch is inactive in the current configuration`,
      onClick: (e) => { e.stopPropagation(); onTraceClick?.(keyForPattern?.(pattern) ?? pattern, ln.n); },
    }, k);
      traceButtons.push({ chip, pattern, line: ln.n });
      return chip;
    });
    const row = h('div', { class: 'ln', dataset: { line: ln.n }, onClick: onLineClick ? () => onLineClick(ln.n) : null },
      h(onLineClick ? 'button' : 'span', { class: 'no', ...(onLineClick ? { type: 'button', 'aria-label': `Open model.py line ${ln.n}` } : {}) }, ln.n), h('span', { class: 'src', html: highlightPython(ln.text) || ' ' }), h('span', { class: 'trace-chips' }, chips));
    rows.set(ln.n, row);
    el.append(row);
  }
  let highlighted = new Set();
  function scrollHighlight(behavior = reducedMotion() ? 'auto' : 'smooth') {
    const first = rows.get(Math.min(...highlighted));
    if (!first || !el.clientHeight) return;
    const rect = el.getBoundingClientRect(), scale = rect.height / el.offsetHeight || 1;
    const top = (first.getBoundingClientRect().top - rect.top) / scale + el.scrollTop - el.clientHeight / 3;
    el.scrollTo({ top: Math.max(0, top), behavior });
  }
  function setHighlight(hl, { scroll = true } = {}) {
    const set = new Set(hl instanceof Set ? hl : [].concat(hl));
    highlighted = set;
    for (const [n, row] of rows) {
      row.classList.toggle('hl', set.has(n));
      row.classList.toggle('dim', dimOthers && set.size > 0 && !set.has(n));
    }
    if (scroll && set.size) scrollHighlight();
  }
  setHighlight(highlight, { scroll: false });
  function setTraceContext() {
    for (const { chip, pattern, line } of traceButtons) {
      const key = keyForPattern?.(pattern) ?? pattern, active = isTraceActive?.(key, line) !== false;
      chip.textContent = key; chip.dataset.traceKey = key; chip.disabled = !active;
      chip.title = active ? `Open the live tensor ${key}` : `${key}: this branch is inactive in the current configuration`;
    }
  }
  const observer = followResize ? new ResizeObserver(() => scrollHighlight('auto')) : null;
  observer?.observe(el);
  return { el, setHighlight, setTraceContext, rows, destroy() { observer?.disconnect(); } };
}

// Python source viewer with line numbers, light syntax colouring, highlighted
// lines and clickable `@trace` chips (the bridge from code to tensors).
import { h } from './dom.js';
import { kindVar } from './colors.js';

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
export function codeView({ lines, from = 1, to = Infinity, highlight = [], dimOthers = false, kindOf, onTraceClick, onLineClick, maxHeight }) {
  const el = h('div', { class: 'code', style: maxHeight ? { '--code-h': `${maxHeight}px` } : null });
  const rows = new Map();
  for (const ln of lines) {
    if (ln.n < from || ln.n > to) continue;
    const chips = (ln.keys || []).map((k) => h('button', {
      type: 'button', class: 'trace-chip', style: { '--kind': kindVar(kindOf?.(k) || 'norm') },
      title: `Inspect ${k}`, onClick: (e) => { e.stopPropagation(); onTraceClick?.(k, ln.n); },
    }, k));
    const row = h('div', { class: 'ln', dataset: { line: ln.n }, onClick: onLineClick ? () => onLineClick(ln.n) : null },
      h('span', { class: 'no' }, ln.n), h('span', { class: 'src', html: highlightPython(ln.text) || ' ' }), h('span', null, chips));
    rows.set(ln.n, row);
    el.append(row);
  }
  function setHighlight(hl, { scroll = true } = {}) {
    const set = new Set(hl instanceof Set ? hl : [].concat(hl));
    for (const [n, row] of rows) {
      row.classList.toggle('hl', set.has(n));
      row.classList.toggle('dim', dimOthers && set.size > 0 && !set.has(n));
    }
    if (scroll && set.size) {
      const first = rows.get(Math.min(...set));
      if (first) {
        const top = first.offsetTop - el.clientHeight / 3;
        el.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
      }
    }
  }
  setHighlight(highlight, { scroll: false });
  return { el, setHighlight, rows };
}

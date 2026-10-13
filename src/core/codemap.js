// Parses reference/model.py (bundled as a string by tools/gen_codemap.mjs) into
// lines and finds the line that produces each trace key.
import { MODEL_PY } from '../generated/model_py.js';
import { parseKey } from './describe.js';

const MARK = /\s*#?\s*@trace\s+(\S+)\s*$/;

function build() {
  const lines = MODEL_PY.split('\n').map((raw, i) => {
    const m = raw.match(/#.*@trace\s+(\S+)/);
    let text = raw;
    if (m) {
      // drop the marker but keep any human comment before it
      text = raw.replace(MARK, '').replace(/\s+#\s*$/, '').replace(/\s+$/, '');
    }
    return { n: i + 1, raw, text, keys: m ? [m[1]] : [] };
  });
  const byPattern = new Map();
  for (const ln of lines) for (const k of ln.keys) {
    if (!byPattern.has(k)) byPattern.set(k, []);
    byPattern.get(k).push(ln.n);
  }
  // Named regions (class / def bodies) for views that show one component's code.
  const regions = {};
  let current = null;
  for (const ln of lines) {
    const m = ln.raw.match(/^(class|def)\s+(\w+)/);
    if (m) { current = m[2]; regions[current] = { from: ln.n, to: ln.n }; }
    else if (current && (ln.raw.startsWith(' ') || ln.raw.trim() === '')) regions[current].to = ln.n;
    else if (ln.raw.trim() && !ln.raw.startsWith(' ')) current = null;
  }
  for (const r of Object.values(regions)) {
    while (r.to > r.from && !lines[r.to - 1].raw.trim()) r.to--;
  }
  return { lines, byPattern, regions };
}

const MAP = build();

export const codeLines = MAP.lines;
export const codeRegions = MAP.regions;   // e.g. codeRegions.Attention -> {from, to}

// Lines (1-based) whose marker matches this trace key. Some keys appear on two
// alternative lines (e.g. emb.pos for learned vs sinusoidal); the config picks one.
export function linesForKey(key) {
  return MAP.byPattern.get(parseKey(key).pattern) || [];
}

export function lineForKey(key, cfg) {
  const ls = linesForKey(key);
  if (ls.length <= 1 || !cfg) return ls[0] ?? null;
  const { pattern } = parseKey(key);
  const pick = (needle) => ls.find((n) => MAP.lines[n - 1].raw.includes(needle)) ?? ls[0];
  if (pattern === 'emb.pos') return cfg.pos_encoding === 'sinusoidal' ? pick('sinusoidal_table') : pick('wpe');
  if (pattern === 'emb.out') {
    if (['learned', 'sinusoidal'].includes(cfg.pos_encoding)) return pick('tok + pe');
    return ls.find((n) => !MAP.lines[n - 1].raw.includes('tok + pe')) ?? ls[0];
  }
  if (pattern === 'L{i}.mlp.act') return pick(cfg.activation === 'swiglu' ? 'silu' : cfg.activation === 'gelu' ? 'gelu' : 'relu');
  if (pattern === 'L{i}.resid_mid' || pattern === 'L{i}.resid_out') {
    const pre = ls.find((n) => MAP.lines[n - 1].raw.includes('x = x +'));
    const post = ls.find((n) => !MAP.lines[n - 1].raw.includes('x = x +'));
    return (cfg.norm_placement === 'pre' ? pre : post) ?? ls[0];
  }
  return ls[0];
}

export function codeContext(line, before = 2, after = 2) {
  return MAP.lines.slice(Math.max(0, line - 1 - before), Math.min(MAP.lines.length, line + after));
}

/** Only return tensors produced by THIS branch under the current configuration. */
export function keysForLine(line, cfg, layer = 0, trace = null) {
  return (codeLines[line - 1]?.keys || []).map((k) => k.replace('{i}', String(layer)))
    .filter((key) => lineForKey(key, cfg) === line && (!trace || trace[key]));
}

/** The marked operation plus the helper implementation that explains its internals. */
export function sourceForKey(key, cfg) {
  const line = lineForKey(key, cfg), { pattern } = parseKey(key);
  let helper = null;
  if (/^(L\{i\}\.ln[12]|lnf)$/.test(pattern) || /^L\{i\}\.resid_(mid|out)$/.test(pattern) && cfg.norm_placement === 'post') {
    helper = cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';
  } else if (/\.attn\.[qk]_rot$/.test(pattern)) helper = 'apply_rope';
  else if (pattern === 'mask') helper = 'build_mask';
  else if (pattern === 'alibi') helper = 'alibi_bias';
  else if (pattern === 'emb.pos' && cfg.pos_encoding === 'sinusoidal') helper = 'sinusoidal_table';
  return { line, helper, region: helper ? codeRegions[helper] : null };
}

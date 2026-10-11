// Every catalogue equation must compile in KaTeX, and every live() instantiation
// must reproduce the traced value of the cell it explains.   node tests/math.mjs
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import katex from '../src/vendor/katex/katex.mjs';
import { MACROS } from '../src/ui/tex.js';
import { liveContext } from '../src/core/math/index.js';

// --only src/core/math/<topic>.js checks one topic file in isolation
const onlyArg = process.argv.indexOf('--only');
const MATH = onlyArg > 0
  ? (await import(pathToFileURL(join(dirname(fileURLToPath(import.meta.url)), '..', process.argv[onlyArg + 1])).href)).ENTRIES
  : (await import('../src/core/math/index.js')).MATH;
import { parseKey } from '../src/core/describe.js';
import { modelFromWeights, infer, makeTokenizer, mulberry32 } from '../src/engine/index.js';

globalThis.atob ??= (b) => Buffer.from(b, 'base64').toString('binary');
const here = dirname(fileURLToPath(import.meta.url));
let fails = 0, checks = 0;
const fail = (m) => { fails++; if (fails <= 60) console.log('  FAIL', m); };

function compile(where, src, display = true) {
  checks++;
  try {
    katex.renderToString(src, { displayMode: display, throwOnError: true, strict: 'ignore', macros: { ...MACROS }, trust: (c) => c.command === '\\htmlClass' });
  } catch (e) { fail(`${where}: ${e.message.split('\n')[0]}`); }
}

// 1. static LaTeX
for (const [k, e] of Object.entries(MATH)) {
  if (!e.title || !e.tex) fail(`${k}: missing title or tex`);
  compile(`${k}.tex`, e.tex);
  if (e.compact) compile(`${k}.compact`, e.compact, false);
  (e.symbols || []).forEach((s, i) => compile(`${k}.symbols[${i}]`, s.tex, false));
  (e.derivation || []).forEach((d, i) => d.steps.forEach((st, j) => compile(`${k}.derivation[${i}].steps[${j}]`, st)));
}

// 2. live instantiations on every fixture config and every preset
const states = [];
for (const f of readdirSync(join(here, 'fixtures')).filter((x) => x.endsWith('.json'))) {
  const fx = JSON.parse(readFileSync(join(here, 'fixtures', f), 'utf8'));
  const model = modelFromWeights(fx.config, fx.params);
  const tok = makeTokenizer('abcdefghijkl'.slice(0, fx.config.vocab_size));
  states.push({ name: fx.name, model, config: model.config, ids: fx.ids, tokenizer: tok, sel: { head: 0 } });
}
for (const f of readdirSync(join(here, '..', 'src', 'presets')).filter((x) => /^(charlm|llama|induction|reverse)\.js$/.test(x))) {
  const mod = (await import(pathToFileURL(join(here, '..', 'src', 'presets', f)).href)).default;
  const bin = atob(mod.weights), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const flat = new Float32Array(bytes.buffer), w = {};
  let off = 0;
  for (const [n, s] of mod.order) { const k = s.reduce((a, b) => a * b, 1); w[n] = { data: flat.subarray(off, off + k) }; off += k; }
  const model = modelFromWeights(mod.config, w), tok = makeTokenizer(mod.vocab);
  states.push({ name: mod.id, model, config: model.config, ids: tok.encode(mod.samples[0]).ids.slice(0, mod.config.block_size), tokenizer: tok, sel: { head: 0 } });
}
const rnd = mulberry32(7);
let liveRuns = 0;
for (const st of states) {
  st.trace = infer(st.model, st.ids).trace;
  for (const key of Object.keys(st.trace)) {
    const e = MATH[parseKey(key).pattern] ?? MATH[key];
    if (!e?.live) continue;
    const t = st.trace[key];
    if (t.shape.length < 2) continue;
    const three = t.shape.length === 3;
    const [B, R, C] = three ? t.shape : [1, ...t.shape];
    const cells = [[0, 0, 0], [B - 1, R - 1, C - 1], [0, R - 1, 0]];
    for (let n = 0; n < 4; n++) cells.push([Math.floor(rnd() * B), Math.floor(rnd() * R), Math.floor(rnd() * C)]);
    for (const [b, r, c] of cells) {
      const ctx = liveContext(st, key, { band: b, r, x: c });
      let out;
      try { out = e.live(ctx); } catch (err) { fail(`${st.name} ${key}[${b},${r},${c}] live threw: ${err.message}`); continue; }
      if (!out) continue;
      liveRuns++;
      compile(`${st.name} ${key}[${b},${r},${c}] live tex`, out.tex);
      checks++;
      const { result, expected } = out;
      const ok = (result === -Infinity && expected === -Infinity) || Math.abs(result - expected) <= 1e-6 * Math.max(1, Math.abs(expected));
      if (!ok) fail(`${st.name} ${key}[${b},${r},${c}] live result ${result} != traced ${expected}`);
    }
  }
}
console.log(`${Object.keys(MATH).length} entries, ${liveRuns} live instantiations on ${states.length} models`);
console.log(`${checks - fails}/${checks} checks passed`);
process.exit(fails ? 1 : 0);

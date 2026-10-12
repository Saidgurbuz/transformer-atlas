// The JS byte-level BPE must produce exactly the Python tokenizer's ids.
//   node tests/bpe.mjs [fixture]   (fixture from tools/train_bpe_preset.py)
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { makeBPETokenizer } from '../src/engine/index.js';

globalThis.atob ??= (b) => Buffer.from(b, 'base64').toString('binary');
const here = dirname(fileURLToPath(import.meta.url));
const presetFile = process.argv[3] || join(here, '..', 'src', 'presets', 'tinystories.js');
const fixture = process.argv[2] || join(here, 'fixtures', 'bpe_cases.json');
const mod = (await import(pathToFileURL(presetFile).href)).default;
const tok = makeBPETokenizer(mod.tokenizer);
const { cases } = JSON.parse(readFileSync(fixture, 'utf8'));
let bad = 0, n = 0;
for (const c of cases) {
  const got = tok.encode(c.text).ids;
  n += c.ids.length;
  if (got.join(',') !== c.ids.join(',')) {
    bad++;
    if (bad <= 5) console.log('MISMATCH', JSON.stringify(c.text.slice(0, 60)), '\n  js', got.slice(0, 20).join(','), '\n  py', c.ids.slice(0, 20).join(','));
  }
  if (tok.decode(c.ids) !== c.text) { bad++; if (bad <= 5) console.log('DECODE MISMATCH', JSON.stringify(c.text.slice(0, 60))); }
}
console.log(`${cases.length - bad}/${cases.length} texts identical (${n} tokens), vocab ${tok.size}, ${tok.merges} merges`);
console.log('example:', tok.encode('Lily had a red ball.').ids.map((i) => tok.label(i)).join(' | '));
process.exit(bad ? 1 : 0);

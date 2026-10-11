// Shipped preset weights: the JS engine must reproduce the PyTorch logits stored
// at export time (float32 weights, so tolerance 1e-4).   node tests/presets.mjs [dir]
import { readdirSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { modelFromWeights, infer } from '../src/engine/index.js';

globalThis.atob ??= (b) => Buffer.from(b, 'base64').toString('binary');
const dir = process.argv[2] ? resolve(process.argv[2]) : join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'presets');
const files = readdirSync(dir).filter((f) => f.endsWith('.js') && !['index.js', 'manifest.js'].includes(f));
let bad = 0;
for (const f of files) {
  const mod = (await import(pathToFileURL(join(dir, f)).href)).default;
  const bin = atob(mod.weights);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const flat = new Float32Array(bytes.buffer);
  const weights = {};
  let off = 0;
  for (const [name, shape] of mod.order) {
    const n = shape.reduce((a, b) => a * b, 1);
    weights[name] = { shape, data: flat.subarray(off, off + n) };
    off += n;
  }
  const model = modelFromWeights(mod.config, weights);
  const { logits } = infer(model, mod.check.ids);
  const V = mod.config.vocab_size, T = mod.check.ids.length;
  const last = logits.data.subarray((T - 1) * V, T * V);
  const err = Math.max(...mod.check.lastLogits.map((v, i) => Math.abs(v - last[i])));
  const ok = err < 1e-4;
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${mod.id.padEnd(10)} max |Δlogit| = ${err.toExponential(2)}  ${JSON.stringify(mod.metrics)}`);
}
process.exit(bad ? 1 : 0);

// Numerical parity: JS engine vs the PyTorch reference, key by key.
//   node tests/parity.mjs            (after tools/export_fixtures.py)
// Checks: same trace keys in the same order, every traced tensor, the loss,
// every parameter gradient, and KV-cache decoding vs the full forward pass.
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { modelFromWeights, forward, createCache, decodeStep, kvCacheSupported } from '../src/engine/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, 'fixtures');
const TOL = 1e-9;
let failures = 0, checks = 0;

function compare(label, got, want) {
  checks++;
  const g = got.data, w = want.data;
  if (g.length !== w.length) { failures++; console.log(`  FAIL ${label}: size ${g.length} vs ${w.length}`); return; }
  if (JSON.stringify(got.shape) !== JSON.stringify(want.shape)) {
    failures++; console.log(`  FAIL ${label}: shape [${got.shape}] vs [${want.shape}]`); return;
  }
  let worst = 0;
  for (let i = 0; i < w.length; i++) {
    const wv = w[i] === null ? -Infinity : w[i];
    if (wv === -Infinity || g[i] === -Infinity) {
      if (wv !== g[i]) { worst = Infinity; break; }
      continue;
    }
    worst = Math.max(worst, Math.abs(g[i] - wv) / Math.max(1, Math.abs(wv)));
  }
  if (!(worst <= TOL)) { failures++; console.log(`  FAIL ${label}: max rel err ${worst.toExponential(2)}`); }
}

const files = readdirSync(dir).filter((f) => f.endsWith('.json')).sort();
if (!files.length) { console.error('no fixtures; run tools/export_fixtures.py first'); process.exit(1); }

for (const f of files) {
  const fx = JSON.parse(readFileSync(join(dir, f), 'utf8'));
  const before = failures;
  const model = modelFromWeights(fx.config, fx.params);
  const { trace, loss } = forward(model, fx.ids, { targets: fx.targets });
  const keys = Object.keys(trace);
  checks++;
  if (keys.join('|') !== fx.order.join('|')) {
    failures++;
    console.log(`  FAIL ${fx.name}: trace keys differ\n    js:    ${keys.join(' ')}\n    torch: ${fx.order.join(' ')}`);
  }
  for (const k of fx.order) if (trace[k]) compare(`${fx.name} ${k}`, { shape: trace[k].shape, data: trace[k].data }, fx.trace[k]);
  loss.backward();
  for (const [name, g] of Object.entries(fx.grads)) {
    compare(`${fx.name} grad ${name}`, { shape: model.params[name].shape, data: model.params[name].grad }, g);
  }
  if (kvCacheSupported(model.config)) {
    const cache = createCache(model);
    const V = model.config.vocab_size;
    fx.ids.forEach((id, t) => {
      const { logits } = decodeStep(model, cache, id);
      compare(`${fx.name} kv-cache step ${t}`, { shape: [V], data: logits },
        { shape: [V], data: Array.from(trace.logits.data.subarray(t * V, (t + 1) * V)) });
    });
  }
  console.log(`${failures === before ? 'ok  ' : 'FAIL'} ${fx.name}  (${fx.order.length} traced tensors, ${Object.keys(fx.grads).length} grads)`);
}
console.log(`\n${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);

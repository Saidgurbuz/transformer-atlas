import { mkdirSync, writeFileSync } from 'node:fs';
import { createModel } from '../src/engine/index.js';
import { MINI_PROFILES } from '../src/lessons/training.js';
const inputs = Object.fromEntries(Object.entries(MINI_PROFILES).map(([id, p]) => {
  const model = createModel(p.config, 17);
  return [id, { config: model.config, params: Object.fromEntries(Object.entries(model.params).map(([n, t]) => [n, { shape: t.shape, data: Array.from(t.data) }])) }];
}));
mkdirSync(new URL('../tests/fixtures/lessons/', import.meta.url), { recursive: true });
writeFileSync(new URL('../tests/fixtures/lessons/training_inputs.json', import.meta.url), JSON.stringify(inputs));

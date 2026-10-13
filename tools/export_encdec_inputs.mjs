import { writeFileSync } from 'node:fs';
import { createEncoderDecoder } from '../src/lessons/encoder-decoder.js';
const m = createEncoderDecoder();
writeFileSync(new URL('../tests/fixtures/lessons/encdec_inputs.json', import.meta.url), JSON.stringify({ config: m.config,
  params: Object.fromEntries(Object.entries(m.params).map(([n, t]) => [n, { shape: t.shape, data: Array.from(t.data) }])) }));

// Bibliography, collapsed-study navigation and executable-source integrity.
import assert from 'node:assert/strict';
import { MATH } from '../src/core/math/index.js';
import { equationSources } from '../src/core/math/sources.js';
import { furtherStudy } from '../src/core/math/study.js';
import { REFERENCES, citeLabel } from '../src/core/references.js';
import { referenceSource } from '../src/core/codemap.js';
import { DEFAULT_CONFIG } from '../src/engine/index.js';
assert.equal(new Set(REFERENCES.map(r => r.id)).size, REFERENCES.length, 'reference IDs unique');
let links = 0, code = 0;
for (const [key, entry] of Object.entries(MATH)) {
  for (const cfg of [DEFAULT_CONFIG, { ...DEFAULT_CONFIG, norm: 'rmsnorm', activation: 'swiglu', pos_encoding: 'rope', n_kv_head: 2, tie_weights: true }]) {
    const related = furtherStudy(key, cfg);
    assert.equal(new Set(related).size, related.length, `${key}: no duplicate study links`);
    assert.ok(!related.includes(key), `${key}: no self link`);
    assert.ok(related.length <= 4, `${key}: focused next steps`);
    for (const target of related) { assert.ok(MATH[target], `${key}: missing related topic ${target}`); links++; }
  }
  for (const section of entry.derivation ?? []) if (section.reference) {
    const ref = section.reference;
    assert.ok(referenceSource(ref.file)?.regions[ref.region], `${key}: missing executable region ${ref.file}:${ref.region}`); code++;
  }
  for (const source of equationSources(key, DEFAULT_CONFIG)) {
    const reference = REFERENCES.find(r => r.id === source.id);
    assert.ok(reference && ['paper', 'documentation', 'code'].includes(reference.type));
    assert.ok(source.where && source.note, `${key}: source location and scope required`);
  }
}
for (const id of ['pytorch-sdpa', 'yarn-code']) assert.ok(!citeLabel(REFERENCES.find(r => r.id === id)).includes('null'));
for (const key of ['concept:qk-normalization', 'concept:rope-context', 'concept:mla']) assert.ok(MATH[key].scope.includes('main model'));
const study = (key, patch = {}) => furtherStudy(key, { ...DEFAULT_CONFIG, ...patch });
assert.deepEqual(study('unknown'), [], 'unmapped operations have no global fallback');
assert.deepEqual(study('concept:sampling'), [], 'no section when no relevant next step is curated');
assert.ok(study('L2.attn.probs').includes('concept:online-attention'), 'softmax links to its streamed recurrence');
assert.ok(!study('L2.attn.probs').includes('concept:mla'), 'softmax does not inherit cache architecture topics');
assert.ok(study('L2.attn.scores').includes('concept:qk-normalization'), 'QK normalization belongs beside scores');
assert.ok(!study('L2.attn.ctx').includes('concept:qk-normalization'), 'value mixing does not inherit score extensions');
assert.ok(!study('emb.pos').includes('concept:rope-context'), 'learned positions have no RoPE extension links');
assert.ok(study('L2.attn.q_rot', { pos_encoding: 'rope' }).includes('concept:rope-context'));
assert.ok(!study('concept:kv-cache').includes('concept:rope-context'));
assert.ok(study('concept:kv-cache', { pos_encoding: 'rope' }).includes('concept:rope-context'));
assert.ok(study('L2.ln1').includes('concept:layernorm-backward'));
assert.ok(!study('L2.ln1').includes('concept:rmsnorm-backward'));
assert.ok(study('L2.ln1', { norm: 'rmsnorm' }).includes('concept:rmsnorm-backward'));
assert.ok(!study('L2.ln1', { norm: 'rmsnorm' }).includes('concept:layernorm-backward'));
assert.ok(study('L2.mlp.act').includes('concept:gelu'));
assert.ok(!study('L2.mlp.act', { activation: 'swiglu' }).includes('concept:gelu'));
assert.ok(study('L2.mlp.act', { activation: 'swiglu' }).includes('concept:swiglu'));
assert.ok(!study('L2.mlp.act', { activation: 'relu' }).some(k => ['concept:gelu', 'concept:swiglu'].includes(k)));
assert.ok(study('emb.tok', { tie_weights: true }).includes('concept:weight-tying'));
assert.ok(!study('emb.tok').includes('concept:weight-tying'));
console.log(`${links} study links and ${code} executable code regions validated; all research variants have explicit scope.`);

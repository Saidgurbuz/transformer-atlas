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
  for (const related of furtherStudy(key)) { assert.ok(MATH[related], `${key}: missing related topic ${related}`); links++; }
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
console.log(`${links} study links and ${code} executable code regions validated; all research variants have explicit scope.`);

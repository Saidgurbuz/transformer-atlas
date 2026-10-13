import assert from 'node:assert/strict';
import katex from '../src/vendor/katex/katex.mjs';
import { MACROS } from '../src/ui/tex.js';
import { EXERCISES } from '../src/lessons/exercises.js';
import { evaluateExercise } from '../src/lessons/exercise-runner.js';
import { referenceSource } from '../src/core/codemap.js';
import { mathFor } from '../src/core/math/index.js';
assert.equal(new Set(EXERCISES.map(e => e.id)).size, EXERCISES.length);
for (const e of EXERCISES) {
  katex.renderToString(e.tex, { macros: MACROS, throwOnError: true });
  assert.ok(mathFor(e.math), `${e.id}: actual equation link`);
  assert.ok(referenceSource(e.reference.file)?.regions[e.reference.region], `${e.id}: actual Python operation`);
  if (e.type === 'Coding') {
    assert.ok(evaluateExercise(e.solution, e.tests).passed, e.id);
    assert.equal(evaluateExercise(e.starter, e.tests).passed, false, 'unfinished starter cannot pass');
  } else assert.ok(e.correct >= 0 && e.correct < e.choices.length && e.explanation.length > 40);
}
assert.ok(evaluateExercise('function broken(', []).error);
assert.equal(evaluateExercise('() => [NaN]', [{ name: 'NaN', args: [], expected: [0] }]).passed, false);
assert.equal(evaluateExercise('() => [1,2]', [{ name: 'Wrong shape', args: [], expected: [1] }]).passed, false);
console.log(`${EXERCISES.length} exercise definitions validated; all coding solutions pass and starters fail.`);

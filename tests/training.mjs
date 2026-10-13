import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { modelFromWeights } from '../src/engine/index.js';
import { trainingStep, optimizerState, trainingLesson } from '../src/lessons/training.js';
const inputs = JSON.parse(readFileSync(new URL('./fixtures/lessons/training_inputs.json', import.meta.url)));
const expected = JSON.parse(readFileSync(new URL('./fixtures/lessons/training.json', import.meta.url)));
let comparisons = 0;
const close = (a, b, label) => { assert.equal(a.length, b.length); for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-8 * Math.max(1, Math.abs(b[i])), `${label} ${i}`); comparisons++; };
for (const [profile, kinds] of Object.entries(expected)) for (const [kind, rounds] of Object.entries(kinds)) {
  const model = modelFromWeights(inputs[profile].config, inputs[profile].params), state = optimizerState();
  for (const want of rounds) {
    const got = trainingStep(model, state, kind);
    assert.ok(Math.abs(got.lossBefore - want.before) < 1e-9); assert.ok(Math.abs(got.lossAfter - want.after) < 1e-9);
    close(got.after.logits.data, want.afterLogits.data, 'new logits');
    close(got.activations.logits.data, want.logitsGradient.data, 'logit gradients');
    for (const [name, value] of Object.entries(want.gradients)) close(got.gradients[name], value.data, name);
    for (const [name, value] of Object.entries(want.paramsAfter)) close(model.params[name].data, value.data, name);
    const grad = got.activations.logits.data, p = got.trace.probs.data, targets = [2, 3, 4, -1];
    for (let t = 0; t < 4; t++) for (let v = 0; v < 5; v++) assert.ok(Math.abs(grad[t * 5 + v] - (targets[t] < 0 ? 0 : (p[t * 5 + v] - (targets[t] === v ? 1 : 0)) / 3)) < 1e-10);
  }
}
const lesson = trainingLesson();
assert.notDeepEqual(Array.from(lesson.initial.params['wte.weight'].data), Array.from(lesson.model.params['wte.weight'].data));
assert.throws(() => trainingLesson('missing'));
console.log(`${comparisons} PyTorch training comparisons passed, including ignored rows, shared weights, SGD and AdamW across three updates.`);

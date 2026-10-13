import { DEFAULT_CONFIG, createModel, cloneModel, forward, infer } from '../engine/index.js';

export const MINI_VOCAB = ['␣', 'c', 'a', 't', '.'];
export const MINI_IDS = [1, 2, 3, 4];
export const MINI_TARGETS = [2, 3, 4, -1];
export const MINI_PROFILES = {
  gpt: { title: 'GPT-style', config: { ...DEFAULT_CONFIG, vocab_size: 5, block_size: 8, n_layer: 1, d_model: 8, n_head: 2, n_kv_head: 2, d_ff: 32, tie_weights: true } },
  llama: { title: 'Llama-style · GQA', config: { ...DEFAULT_CONFIG, vocab_size: 5, block_size: 8, n_layer: 1, d_model: 8, n_head: 2, n_kv_head: 1, d_ff: 24,
    pos_encoding: 'rope', norm: 'rmsnorm', activation: 'swiglu', tie_weights: false, bias: false } },
};

export const OPT_DEFAULTS = { lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8, decay: 0.01 };
export function optimizerState() { return { step: 0, moments: {} }; }
export function updateParameters(model, gradients, state, optimizer = 'adamw', options = {}) {
  const opts = { ...OPT_DEFAULTS, ...options };
  if (!['sgd', 'adamw'].includes(optimizer) || !Number.isFinite(opts.lr) || opts.lr <= 0 || !Number.isFinite(opts.decay) || opts.decay < 0 || ![opts.beta1, opts.beta2].every((b) => b >= 0 && b < 1) || !Number.isFinite(opts.eps) || opts.eps <= 0) throw new Error('Invalid optimizer configuration');
  state.step++;
  for (const [name, p] of Object.entries(model.params)) {
    const grad = gradients[name];
    if (!grad || grad.length !== p.data.length) throw new Error(`Missing gradient for ${name}`);
    const decay = p.shape.length >= 2 ? opts.decay : 0; // gain/shift vectors are not decayed
    if (!state.moments[name]) state.moments[name] = { m: new Float64Array(grad.length), v: new Float64Array(grad.length) };
    const { m, v } = state.moments[name];
    for (let i = 0; i < grad.length; i++) {
      if (optimizer === 'sgd') p.data[i] -= opts.lr * grad[i];
      else {
        m[i] = opts.beta1 * m[i] + (1 - opts.beta1) * grad[i];
        v[i] = opts.beta2 * v[i] + (1 - opts.beta2) * grad[i] ** 2;
        const correctedM = m[i] / (1 - opts.beta1 ** state.step), correctedV = v[i] / (1 - opts.beta2 ** state.step);
        p.data[i] = p.data[i] * (1 - opts.lr * decay) - opts.lr * correctedM / (Math.sqrt(correctedV) + opts.eps);
      }
    }
  }
  return model;
}
export function trainingStep(model, state, optimizer = 'adamw', options = {}) {
  for (const p of Object.values(model.params)) p.zeroGrad();
  const before = forward(model, MINI_IDS, { targets: MINI_TARGETS });
  before.loss.backward();
  const gradients = Object.fromEntries(Object.entries(model.params).map(([n, p]) => [n, p.grad ? p.grad.slice() : new Float64Array(p.data.length)]));
  const activations = Object.fromEntries(Object.entries(before.trace).filter(([, t]) => t.grad).map(([n, t]) => [n, { shape: t.shape, data: t.grad.slice() }]));
  // Snapshot before mutating weights, so displayed probabilities and gradients
  // always refer to the same executed pass (including shared embedding gradients).
  const trace = Object.fromEntries(Object.entries(before.trace).map(([n, t]) => [n, { shape: t.shape, data: t.data.slice() }]));
  const lossBefore = before.loss.item();
  updateParameters(model, gradients, state, optimizer, options);
  const after = infer(model, MINI_IDS, { targets: MINI_TARGETS });
  return { trace, gradients, activations, lossBefore, lossAfter: after.loss.item(), after: after.trace, step: state.step };
}
export function trainingLesson(profile = 'gpt', optimizer = 'adamw', steps = 1) {
  if (!MINI_PROFILES[profile] || !Number.isInteger(steps) || steps < 1 || steps > 5) throw new Error('Invalid training lesson');
  const initial = createModel(MINI_PROFILES[profile].config, 17), model = cloneModel(initial), state = optimizerState();
  const rounds = [];
  for (let i = 0; i < steps; i++) rounds.push(trainingStep(model, state, optimizer));
  return { initial, model, state, rounds, round: rounds.at(-1), profile: MINI_PROFILES[profile] };
}

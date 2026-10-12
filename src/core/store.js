// Application state. One model, one input text, one forward trace, one selection.
// Views read `store.state` and subscribe with store.on(event, fn).
//
// Events
//   'trace'   forward pass recomputed (config, weights or text changed)
//   'select'  selection changed: {layer, head, token}
//   'config'  config changed (before the new trace)
//   'inspect' open the tensor inspector for a trace key
//   'notice'  short user-facing message {text, error}

import {
  createModel, modelFromWeights, infer, forward, validateConfig, paramSpecs, makeTokenizer, tokenizerFromPreset,
  DEFAULT_CONFIG, DEFAULT_VOCAB, noGrad,
} from '../engine/index.js';
import { PRESETS, loadPresetData } from '../presets/index.js';

const listeners = new Map();

function emit(event, payload) {
  for (const fn of listeners.get(event) || []) {
    try { fn(payload); } catch (err) { console.error(`[store] ${event} listener failed`, err); }
  }
}

const sameSpecs = (a, b) => JSON.stringify(paramSpecs(a)) === JSON.stringify(paramSpecs(b));

export const store = {
  state: {
    config: { ...DEFAULT_CONFIG },
    vocab: DEFAULT_VOCAB,
    tokenizer: makeTokenizer(DEFAULT_VOCAB),
    text: 'the cat sat on the mat',
    ids: [],
    dropped: [],
    model: null,
    weights: 'random',      // 'random' | 'preset:<id>'
    preset: null,           // loaded preset object (title, description, samples...)
    seed: 1337,
    initScale: 1,           // multiplies the random init std (makes untrained patterns visible)
    trace: null,
    logits: null,
    sel: { layer: 0, head: 0, token: 0 },
  },

  on(event, fn) {
    if (!listeners.has(event)) listeners.set(event, new Set());
    listeners.get(event).add(fn);
    return () => listeners.get(event)?.delete(fn);
  },

  emit,

  notice(text, error = false) { emit('notice', { text, error }); },

  // ----------------------------------------------------------------- text

  setText(text, { quiet = false } = {}) {
    const st = this.state;
    const { ids, dropped } = st.tokenizer.encode(text);
    // BPE presets have no padding token; keep at least one real token
    st.text = text;
    st.dropped = dropped;
    st.ids = ids.slice(0, st.config.block_size);
    if (ids.length > st.config.block_size && !quiet) this.notice(`Input cut to the ${st.config.block_size}-token context window.`);
    if (!st.ids.length) st.ids = [0];
    this.recompute();
  },

  // ---------------------------------------------------------------- model

  // Apply config changes. Weights survive when every parameter keeps its shape
  // (e.g. toggling the mask, residual or scaling), so trained presets can be ablated.
  setConfig(patch) {
    const st = this.state;
    const next = { ...st.config, ...patch };
    const errs = validateConfig(next);
    if (errs.length) { this.notice(errs[0], true); return errs; }
    const keep = st.model && sameSpecs(st.config, next);
    st.config = next;
    if (keep) {
      st.model = { config: next, params: st.model.params };
    } else {
      if (st.weights !== 'random') this.notice('Parameter shapes changed, so the model was re-initialised with random weights.');
      st.weights = 'random';
      st.preset = null;
      st.model = this.randomModel(next);
    }
    emit('config', next);
    this.setText(st.text);
    return [];
  },

  randomModel(cfg) {
    const m = createModel(cfg, this.state.seed);
    const k = this.state.initScale;
    if (k !== 1) {
      for (const [name, t] of Object.entries(m.params)) {
        if (name.endsWith('.bias') || /(ln1|ln2|ln_f)\.weight$/.test(name)) continue;
        for (let i = 0; i < t.data.length; i++) t.data[i] *= k;
      }
    }
    return m;
  },

  reinitialize({ seed, initScale } = {}) {
    const st = this.state;
    if (seed != null) st.seed = seed;
    if (initScale != null) st.initScale = initScale;
    st.weights = 'random';
    st.preset = null;
    st.model = this.randomModel(st.config);
    this.setText(st.text);
  },

  // Swap in an externally built model (same tokenizer).
  setModel(model, label = 'custom') {
    const st = this.state;
    st.model = model;
    st.config = { ...model.config };
    st.weights = label;
    emit('config', st.config);
    this.setText(st.text);
  },

  async loadPreset(id) {
    const meta = PRESETS.find((p) => p.id === id);
    if (!meta) { this.notice(`Unknown preset ${id}`, true); return; }
    const data = await loadPresetData(id);
    const st = this.state;
    st.vocab = data.vocab ?? null;
    st.tokenizer = tokenizerFromPreset(data);
    st.config = { ...DEFAULT_CONFIG, ...data.config };
    st.model = modelFromWeights(st.config, data.weights);
    st.weights = `preset:${id}`;
    st.preset = data;
    emit('config', st.config);
    // short samples keep the 3D tower readable; the user can paste longer text
    const sample = data.samples?.find((x) => x.length <= 40) ?? data.samples?.[0] ?? st.text;
    this.setText(sample.slice(0, st.config.block_size), { quiet: true });
  },

  useRandomVocab() {
    const st = this.state;
    st.vocab = DEFAULT_VOCAB;
    st.tokenizer = makeTokenizer(DEFAULT_VOCAB);
    this.setConfig({ vocab_size: st.tokenizer.size });
  },

  // ---------------------------------------------------------------- forward

  recompute() {
    const st = this.state;
    if (!st.model) st.model = this.randomModel(st.config);
    const out = infer(st.model, st.ids);
    st.trace = out.trace;
    st.logits = out.logits;
    const T = st.ids.length;
    st.sel = {
      layer: Math.min(st.sel.layer, st.config.n_layer - 1),
      head: Math.min(st.sel.head, st.config.n_head - 1),
      token: Math.min(st.sel.token ?? T - 1, T - 1),
    };
    emit('trace', st);
  },

  // Next-token loss on the current text plus gradients of every 2-D traced
  // activation and every parameter. Used by views that show backprop.
  gradients() {
    const st = this.state, ids = st.ids;
    const targets = ids.map((_, t) => (t + 1 < ids.length ? ids[t + 1] : -1));
    if (ids.length < 2) return null;
    for (const p of Object.values(st.model.params)) p.zeroGrad();
    const out = forward(st.model, ids, { targets });
    out.loss.backward();
    const act = {};
    for (const [k, t] of Object.entries(out.trace)) if (t.grad) act[k] = t.grad.slice();
    const params = {};
    for (const [k, t] of Object.entries(st.model.params)) { params[k] = t.grad ? t.grad.slice() : null; t.zeroGrad(); }
    return { loss: out.loss.item(), trace: out.trace, act, params, targets };
  },

  // ---------------------------------------------------------------- selection

  select(patch) {
    Object.assign(this.state.sel, patch);
    emit('select', this.state.sel);
  },

  inspect(key) { emit('inspect', key); },

  init() {
    this.state.model = this.randomModel(this.state.config);
    this.state.config.vocab_size = this.state.tokenizer.size;
    this.state.model = this.randomModel(this.state.config);
    this.setText(this.state.text);
    this.state.sel.token = this.state.ids.length - 1;
  },
};

export { noGrad };

// Decoder-only Transformer, a line-for-line port of reference/model.py.
// forward() records every `@trace` tensor under the same key, in the same order
// as the PyTorch reference, so views, the code panel and the parity test agree.

import {
  Tensor, noGrad, linear, matmul, matmulNT, add, mul, scale, addConst, embedding,
  layerNorm, rmsNorm, gelu, relu, silu, softmax, sliceCols, concatCols, rope,
  crossEntropy, stack,
} from './tensor.js';
import { mulberry32, gaussian } from './random.js';

export const DEFAULT_CONFIG = Object.freeze({
  vocab_size: 32,
  block_size: 32,
  n_layer: 2,
  n_head: 4,
  n_kv_head: 4,
  d_model: 64,
  d_ff: 256,
  pos_encoding: 'learned',   // learned | sinusoidal | rope | alibi | none
  norm: 'layernorm',         // layernorm | rmsnorm
  norm_placement: 'pre',     // pre | post
  activation: 'gelu',        // gelu | relu | swiglu
  mask: 'causal',            // causal | none | sliding | prefix
  window: 8,
  prefix_len: 4,
  tie_weights: false,
  bias: true,
  residual: true,
  attn_scale: true,
  rope_base: 10000,
  norm_eps: 1e-5,
});

export const CONFIG_CHOICES = Object.freeze({
  pos_encoding: ['learned', 'sinusoidal', 'rope', 'alibi', 'none'],
  norm: ['layernorm', 'rmsnorm'],
  norm_placement: ['pre', 'post'],
  activation: ['gelu', 'relu', 'swiglu'],
  mask: ['causal', 'none', 'sliding', 'prefix'],
});

export function dHead(cfg) { return cfg.d_model / cfg.n_head; }

// Returns a list of human-readable problems; empty means the config is valid.
export function validateConfig(cfg) {
  const errs = [];
  const posInt = (k) => { if (!Number.isInteger(cfg[k]) || cfg[k] < 1) errs.push(`${k} must be a positive integer`); };
  ['vocab_size', 'block_size', 'n_layer', 'n_head', 'n_kv_head', 'd_model', 'd_ff', 'window'].forEach(posInt);
  if (cfg.d_model % cfg.n_head) errs.push('d_model must be divisible by n_head');
  else if (dHead(cfg) % 2 && cfg.pos_encoding === 'rope') errs.push('RoPE needs an even head width (d_model / n_head)');
  if (cfg.n_head % cfg.n_kv_head) errs.push('n_head must be a multiple of n_kv_head');
  if (cfg.d_model % 2 && cfg.pos_encoding === 'sinusoidal') errs.push('sinusoidal encoding needs an even d_model');
  for (const [k, opts] of Object.entries(CONFIG_CHOICES)) if (!opts.includes(cfg[k])) errs.push(`${k} must be one of ${opts.join(', ')}`);
  if (!Number.isInteger(cfg.prefix_len) || cfg.prefix_len < 0) errs.push('prefix_len must be a non-negative integer');
  for (const k of ['rope_base', 'norm_eps']) if (!Number.isFinite(cfg[k]) || cfg[k] <= 0) errs.push(`${k} must be finite and positive`);
  for (const k of ['tie_weights', 'bias', 'residual', 'attn_scale']) if (typeof cfg[k] !== 'boolean') errs.push(`${k} must be a boolean`);
  return errs;
}

// ------------------------------------------------------------------ parameters

// Parameter names and shapes, in nn.Module.named_parameters() order.
export function paramSpecs(cfg) {
  const d = cfg.d_model, H = cfg.n_head, Hkv = cfg.n_kv_head, dh = dHead(cfg), F = cfg.d_ff, V = cfg.vocab_size;
  const specs = [['wte.weight', [V, d]]];
  if (cfg.pos_encoding === 'learned') specs.push(['wpe.weight', [cfg.block_size, d]]);
  const lin = (name, out, inp) => {
    specs.push([`${name}.weight`, [out, inp]]);
    if (cfg.bias) specs.push([`${name}.bias`, [out]]);
  };
  const norm = (name) => {
    specs.push([`${name}.weight`, [d]]);
    if (cfg.norm === 'layernorm' && cfg.bias) specs.push([`${name}.bias`, [d]]);
  };
  for (let i = 0; i < cfg.n_layer; i++) {
    const p = `blocks.${i}`;
    norm(`${p}.ln1`);
    lin(`${p}.attn.q_proj`, H * dh, d);
    lin(`${p}.attn.k_proj`, Hkv * dh, d);
    lin(`${p}.attn.v_proj`, Hkv * dh, d);
    lin(`${p}.attn.o_proj`, d, H * dh);
    norm(`${p}.ln2`);
    lin(`${p}.mlp.up_proj`, F, d);
    if (cfg.activation === 'swiglu') lin(`${p}.mlp.gate_proj`, F, d);
    lin(`${p}.mlp.down_proj`, d, F);
  }
  if (cfg.norm_placement === 'pre') norm('ln_f');
  if (!cfg.tie_weights) specs.push(['lm_head.weight', [V, d]]);
  return specs;
}

// GPT-2 initialisation: N(0, 0.02), residual writes scaled by 1/sqrt(2·n_layer).
export function createModel(config, seed = 1337) {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const errs = validateConfig(cfg);
  if (errs.length) throw new Error(errs.join('; '));
  const rand = gaussian(mulberry32(seed));
  const params = {};
  for (const [name, shape] of paramSpecs(cfg)) {
    const n = shape.reduce((a, b) => a * b, 1), data = new Float64Array(n);
    if (name.endsWith('.bias')) {
      // zeros
    } else if (/(ln1|ln2|ln_f)\.weight$/.test(name)) {
      data.fill(1);
    } else {
      const std = /(o_proj|down_proj)\.weight$/.test(name) ? 0.02 / Math.sqrt(2 * cfg.n_layer) : 0.02;
      for (let i = 0; i < n; i++) data[i] = rand() * std;
    }
    params[name] = Tensor.param(data, shape);
  }
  return { config: cfg, params };
}

// Load {name: {shape, data}} (data: number[] or Float64Array) into a model.
export function modelFromWeights(config, weights) {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const params = {};
  for (const [name, shape] of paramSpecs(cfg)) {
    const w = weights[name];
    if (!w) throw new Error(`missing weight ${name}`);
    const data = Float64Array.from(w.data ?? w);
    if (data.length !== shape.reduce((a, b) => a * b, 1)) throw new Error(`weight ${name} has wrong size`);
    params[name] = Tensor.param(data, shape);
  }
  return { config: cfg, params };
}

export function cloneModel(model) {
  const params = {};
  for (const [k, t] of Object.entries(model.params)) params[k] = Tensor.param(t.data.slice(), t.shape);
  return { config: { ...model.config }, params };
}

export function countParams(model) {
  return Object.values(model.params).reduce((s, t) => s + t.size, 0);
}

// ------------------------------------------------------ positions and masking

export function sinusoidalTable(T, d) {
  const pe = new Float64Array(T * d);
  for (let p = 0; p < T; p++) for (let i = 0; i < d; i += 2) {
    const freq = Math.exp(-Math.log(10000) * i / d);
    pe[p * d + i] = Math.sin(p * freq);
    pe[p * d + i + 1] = Math.cos(p * freq);
  }
  return pe;
}

// cos/sin tables [T, dh/2] for positions start..start+T-1
export function ropeAngles(T, dh, base, start = 0) {
  const half = dh / 2, cos = new Float64Array(T * half), sin = new Float64Array(T * half);
  for (let t = 0; t < T; t++) for (let k = 0; k < half; k++) {
    const theta = Math.pow(base, -(2 * k) / dh);
    const a = (start + t) * theta;
    cos[t * half + k] = Math.cos(a);
    sin[t * half + k] = Math.sin(a);
  }
  return { cos, sin };
}

export function alibiSlopes(H) {
  return Array.from({ length: H }, (_, h) => Math.pow(2, -8 * (h + 1) / H));
}

// [H][T*T] bias, -slope_h * |i - j|
export function alibiBias(cfg, T) {
  return alibiSlopes(cfg.n_head).map((m) => {
    const b = new Float64Array(T * T);
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) b[i * T + j] = -m * Math.abs(i - j);
    return b;
  });
}

// May query i attend to key j? Mirrors build_mask in model.py.
export function maskAllows(cfg, i, j) {
  switch (cfg.mask) {
    case 'none': return true;
    case 'sliding': return j <= i && j > i - cfg.window;
    case 'prefix': return j <= i || j < cfg.prefix_len;
    default: return j <= i;
  }
}

export function buildMask(cfg, T) {
  const m = new Uint8Array(T * T);
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) m[i * T + j] = maskAllows(cfg, i, j) ? 1 : 0;
  return m;
}

// ---------------------------------------------------------------------- forward

function normOp(model, name, x) {
  const cfg = model.config, P = model.params;
  if (cfg.norm === 'layernorm') return layerNorm(x, P[`${name}.weight`], P[`${name}.bias`] || null, cfg.norm_eps);
  return rmsNorm(x, P[`${name}.weight`], cfg.norm_eps);
}

function lin(model, name, x) {
  return linear(x, model.params[`${name}.weight`], model.params[`${name}.bias`] || null);
}

function attention(model, i, x, ctx) {
  const cfg = model.config, { rec, recStack, mask, ropeTab, alibi } = ctx;
  const p = `blocks.${i}.attn`, L = `L${i}.attn.`;
  const H = cfg.n_head, Hkv = cfg.n_kv_head, dh = dHead(cfg), T = x.shape[0];
  const q = rec(L + 'q', lin(model, `${p}.q_proj`, x));
  const k = rec(L + 'k', lin(model, `${p}.k_proj`, x));
  const v = rec(L + 'v', lin(model, `${p}.v_proj`, x));
  let qh = recStack(L + 'q_heads', Array.from({ length: H }, (_, h) => sliceCols(q, h * dh, dh)));
  let kh = recStack(L + 'k_heads', Array.from({ length: Hkv }, (_, h) => sliceCols(k, h * dh, dh)));
  let vh = recStack(L + 'v_heads', Array.from({ length: Hkv }, (_, h) => sliceCols(v, h * dh, dh)));
  if (ropeTab) {
    qh = recStack(L + 'q_rot', qh.map((t) => rope(t, ropeTab.cos, ropeTab.sin)));
    kh = recStack(L + 'k_rot', kh.map((t) => rope(t, ropeTab.cos, ropeTab.sin)));
  }
  if (Hkv < H) {
    const g = H / Hkv;
    kh = recStack(L + 'k_rep', Array.from({ length: H }, (_, h) => kh[Math.floor(h / g)]));
    vh = recStack(L + 'v_rep', Array.from({ length: H }, (_, h) => vh[Math.floor(h / g)]));
  }
  let scores = recStack(L + 'scores', qh.map((qq, h) => matmulNT(qq, kh[h])));
  if (cfg.attn_scale) scores = recStack(L + 'scaled', scores.map((s) => scale(s, 1 / Math.sqrt(dh))));
  if (alibi) scores = recStack(L + 'biased', scores.map((s, h) => addConst(s, alibi[h])));
  if (ctx.tracing) {
    rec(L + 'masked', stack(scores.map((s) => {
      const m = new Tensor(s.data.slice(), s.shape);
      for (let n = 0; n < m.data.length; n++) if (!mask[n]) m.data[n] = -Infinity;
      return m;
    })));
  }
  const probs = recStack(L + 'probs', scores.map((s) => softmax(s, mask)));
  const heads = recStack(L + 'ctx', probs.map((pr, h) => matmul(pr, vh[h])));
  const concat = rec(L + 'concat', concatCols(heads));
  void T;
  return rec(L + 'out', lin(model, `${p}.o_proj`, concat));
}

function mlp(model, i, x, ctx) {
  const cfg = model.config, { rec } = ctx;
  const p = `blocks.${i}.mlp`, L = `L${i}.mlp.`;
  const up = rec(L + 'up', lin(model, `${p}.up_proj`, x));
  let act;
  if (cfg.activation === 'swiglu') {
    const gate = rec(L + 'gate', lin(model, `${p}.gate_proj`, x));
    act = rec(L + 'act', mul(silu(gate), up));
  } else if (cfg.activation === 'gelu') {
    act = rec(L + 'act', gelu(up));
  } else {
    act = rec(L + 'act', relu(up));
  }
  return rec(L + 'out', lin(model, `${p}.down_proj`, act));
}

function block(model, i, x, ctx) {
  const cfg = model.config, { rec } = ctx, L = `L${i}.`, p = `blocks.${i}`;
  if (cfg.norm_placement === 'pre') {
    const h = rec(L + 'ln1', normOp(model, `${p}.ln1`, x));
    const a = attention(model, i, h, ctx);
    x = rec(L + 'resid_mid', cfg.residual ? add(x, a) : a);
    const h2 = rec(L + 'ln2', normOp(model, `${p}.ln2`, x));
    const m = mlp(model, i, h2, ctx);
    x = rec(L + 'resid_out', cfg.residual ? add(x, m) : m);
  } else {
    const a = attention(model, i, x, ctx);
    const s = rec(L + 'add1', cfg.residual ? add(x, a) : a);
    x = rec(L + 'resid_mid', normOp(model, `${p}.ln1`, s));
    const m = mlp(model, i, x, ctx);
    const s2 = rec(L + 'add2', cfg.residual ? add(x, m) : m);
    x = rec(L + 'resid_out', normOp(model, `${p}.ln2`, s2));
  }
  return x;
}

/**
 * Run the model on one sequence of token ids.
 * @returns {{logits: Tensor, probs: Tensor, loss: Tensor|null, trace: Object<string, Tensor>|null}}
 *   trace keys follow the `@trace` markers of reference/model.py with the batch axis removed.
 */
export function forward(model, ids, { targets = null, trace = true } = {}) {
  const cfg = model.config, P = model.params, T = ids.length, d = cfg.d_model;
  if (T < 1) throw new Error('forward: empty sequence');
  if (T > cfg.block_size) throw new Error(`forward: ${T} tokens exceed block_size ${cfg.block_size}`);
  const tr = trace ? {} : null;
  const rec = (key, t) => { if (tr) tr[key] = t; return t; };
  const recStack = (key, list) => { if (tr) tr[key] = stack(list); return list; };

  const tok = rec('emb.tok', embedding(P['wte.weight'], ids));
  let x;
  if (cfg.pos_encoding === 'learned' || cfg.pos_encoding === 'sinusoidal') {
    const pe = rec('emb.pos', cfg.pos_encoding === 'learned'
      ? embedding(P['wpe.weight'], Array.from({ length: T }, (_, t) => t))
      : new Tensor(sinusoidalTable(T, d), [T, d]));
    x = rec('emb.out', add(tok, pe));
  } else {
    x = rec('emb.out', tok);
  }
  const mask = buildMask(cfg, T);
  if (tr) rec('mask', new Tensor(Float64Array.from(mask), [T, T]));
  const ropeTab = cfg.pos_encoding === 'rope' ? ropeAngles(T, dHead(cfg), cfg.rope_base) : null;
  let alibi = null;
  if (cfg.pos_encoding === 'alibi') {
    alibi = alibiBias(cfg, T);
    if (tr) rec('alibi', stack(alibi.map((b) => new Tensor(b, [T, T]))));
  }
  const ctx = { rec, recStack, mask, ropeTab, alibi, tracing: !!tr };
  for (let i = 0; i < cfg.n_layer; i++) x = block(model, i, x, ctx);
  if (cfg.norm_placement === 'pre') x = rec('lnf', normOp(model, 'ln_f', x));
  const W = cfg.tie_weights ? P['wte.weight'] : P['lm_head.weight'];
  const logits = rec('logits', linear(x, W, null));
  const probs = rec('probs', noGrad(() => softmax(logits)));
  let loss = null;
  if (targets) loss = rec('loss', crossEntropy(logits, targets));
  return { logits, probs, loss, trace: tr };
}

// Inference-only forward (no autograd graph).
export function infer(model, ids, opts = {}) {
  return noGrad(() => forward(model, ids, opts));
}

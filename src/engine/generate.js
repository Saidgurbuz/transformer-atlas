// Autoregressive decoding: sampling rules and an incremental KV-cache decoder.
// decodeStep() must reproduce the last row of forward() for causal and sliding
// masks; tests/parity.mjs checks that.

import { Tensor, noGrad, linear, matmul, matmulNT, add, mul, scale, embedding, layerNorm, rmsNorm,
  gelu, relu, silu, softmax, sliceCols, concatCols, rope } from './tensor.js';
import { dHead, ropeAngles, sinusoidalTable, alibiSlopes, maskAllows } from './model.js';

// ---------------------------------------------------------------- sampling rules

// Returns the probability vector after temperature, top-k and top-p filtering.
export function samplingDistribution(logits, { temperature = 1, topK = 0, topP = 1 } = {}) {
  const V = logits.length;
  const t = Math.max(temperature, 1e-6);
  const z = Array.from(logits, (l) => l / t);
  const keep = new Array(V).fill(true);
  const order = [...z.keys()].sort((a, b) => z[b] - z[a]);
  if (topK > 0 && topK < V) order.slice(topK).forEach((i) => { keep[i] = false; });
  const mx = Math.max(...z);
  let p = z.map((v, i) => (keep[i] ? Math.exp(v - mx) : 0));
  let s = p.reduce((a, b) => a + b, 0);
  p = p.map((v) => v / s);
  if (topP < 1) {
    let cum = 0;
    for (const i of order) {
      if (!keep[i]) continue;
      if (cum >= topP) { keep[i] = false; p[i] = 0; } else cum += p[i];
    }
    s = p.reduce((a, b) => a + b, 0);
    p = p.map((v) => v / s);
  }
  return p;
}

export function sampleFrom(probs, uniform = Math.random) {
  let r = uniform(), acc = 0;
  for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (r < acc) return i; }
  return probs.length - 1;
}

export function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

// --------------------------------------------------------------------- KV cache

export function kvCacheSupported(cfg) { return cfg.mask === 'causal' || cfg.mask === 'sliding'; }

// One entry per layer: k and v are arrays (per kv-head) of row arrays, each row dh numbers.
export function createCache(model) {
  const cfg = model.config;
  return {
    length: 0,
    layers: Array.from({ length: cfg.n_layer }, () => ({
      k: Array.from({ length: cfg.n_kv_head }, () => []),
      v: Array.from({ length: cfg.n_kv_head }, () => []),
    })),
  };
}

// Bytes held by the cache for `len` tokens at `bytesPerValue` (2 for fp16/bf16).
export function cacheBytes(cfg, len, bytesPerValue = 2) {
  return 2 * cfg.n_layer * cfg.n_kv_head * dHead(cfg) * len * bytesPerValue;
}

function normRow(model, name, x) {
  const cfg = model.config, P = model.params;
  return cfg.norm === 'layernorm'
    ? layerNorm(x, P[`${name}.weight`], P[`${name}.bias`] || null, cfg.norm_eps)
    : rmsNorm(x, P[`${name}.weight`], cfg.norm_eps);
}

const lin = (model, name, x) => linear(x, model.params[`${name}.weight`], model.params[`${name}.bias`] || null);

/**
 * Feed one new token through the model, reusing cached keys/values.
 * Returns {logits: Float64Array(V), attn: per-layer [H][len] attention rows}.
 * Only valid for causal-style masks: with bidirectional attention, earlier
 * positions would change when a new token arrives, so their K/V could not be reused.
 */
export function decodeStep(model, cache, tokenId) {
  const cfg = model.config;
  if (!kvCacheSupported(cfg)) throw new Error('KV caching needs a causal or sliding-window mask');
  const pos = cache.length;
  if (pos >= cfg.block_size) throw new Error('context window is full');
  return noGrad(() => {
    const P = model.params, d = cfg.d_model, H = cfg.n_head, Hkv = cfg.n_kv_head, dh = dHead(cfg);
    let x = embedding(P['wte.weight'], [tokenId]);
    if (cfg.pos_encoding === 'learned') x = add(x, embedding(P['wpe.weight'], [pos]));
    else if (cfg.pos_encoding === 'sinusoidal') x = add(x, new Tensor(sinusoidalTable(pos + 1, d).slice(pos * d), [1, d]));
    const ropeTab = cfg.pos_encoding === 'rope' ? ropeAngles(1, dh, cfg.rope_base, pos) : null;
    const slopes = cfg.pos_encoding === 'alibi' ? alibiSlopes(H) : null;
    const attnRows = [];

    const attend = (i, h) => {
      const p = `blocks.${i}.attn`, layer = cache.layers[i];
      const q = lin(model, `${p}.q_proj`, h), k = lin(model, `${p}.k_proj`, h), v = lin(model, `${p}.v_proj`, h);
      for (let g = 0; g < Hkv; g++) {
        let kh = sliceCols(k, g * dh, dh);
        if (ropeTab) kh = rope(kh, ropeTab.cos, ropeTab.sin);
        layer.k[g].push(Array.from(kh.data));
        layer.v[g].push(Array.from(sliceCols(v, g * dh, dh).data));
      }
      const len = pos + 1, rows = [], heads = [];
      for (let hh = 0; hh < H; hh++) {
        const g = Math.floor(hh / (H / Hkv));
        let qh = sliceCols(q, hh * dh, dh);
        if (ropeTab) qh = rope(qh, ropeTab.cos, ropeTab.sin);
        const K = new Tensor(Float64Array.from(layer.k[g].flat()), [len, dh]);
        const Vt = new Tensor(Float64Array.from(layer.v[g].flat()), [len, dh]);
        let s = matmulNT(qh, K);
        if (cfg.attn_scale) s = scale(s, 1 / Math.sqrt(dh));
        if (slopes) for (let j = 0; j < len; j++) s.data[j] += -slopes[hh] * (pos - j);
        const m = Uint8Array.from({ length: len }, (_, j) => (maskAllows(cfg, pos, j) ? 1 : 0));
        const pr = softmax(s, m);
        rows.push(Array.from(pr.data));
        heads.push(matmul(pr, Vt));
      }
      attnRows.push(rows);
      return lin(model, `${p}.o_proj`, concatCols(heads));
    };

    const ffn = (i, h) => {
      const p = `blocks.${i}.mlp`, up = lin(model, `${p}.up_proj`, h);
      let a;
      if (cfg.activation === 'swiglu') a = mul(silu(lin(model, `${p}.gate_proj`, h)), up);
      else a = cfg.activation === 'gelu' ? gelu(up) : relu(up);
      return lin(model, `${p}.down_proj`, a);
    };

    for (let i = 0; i < cfg.n_layer; i++) {
      const b = `blocks.${i}`;
      if (cfg.norm_placement === 'pre') {
        const a = attend(i, normRow(model, `${b}.ln1`, x));
        x = cfg.residual ? add(x, a) : a;
        const m = ffn(i, normRow(model, `${b}.ln2`, x));
        x = cfg.residual ? add(x, m) : m;
      } else {
        const a = attend(i, x);
        x = normRow(model, `${b}.ln1`, cfg.residual ? add(x, a) : a);
        const m = ffn(i, x);
        x = normRow(model, `${b}.ln2`, cfg.residual ? add(x, m) : m);
      }
    }
    if (cfg.norm_placement === 'pre') x = normRow(model, 'ln_f', x);
    const W = cfg.tie_weights ? P['wte.weight'] : P['lm_head.weight'];
    const logits = linear(x, W, null);
    cache.length = pos + 1;
    return { logits: logits.data, attn: attnRows };
  });
}

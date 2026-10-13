// One complete 2017-style encoder/decoder block, including cross-attention.
// The lesson is untrained, dropout-free and uses a shared six-entry vocabulary.
import { DEFAULT_CONFIG, createModel, Tensor, noGrad, embedding, scale, add, linear, layerNorm, relu,
  sliceCols, concatCols, matmulNT, matmul, softmax, stack, sinusoidalTable } from '../engine/index.js';

export const ENCDEC_VOCAB = ['⟨bos⟩', 'a', 'cat', 'sat', '.', '⟨pad⟩'];
export const ENCDEC_CONFIG = { ...DEFAULT_CONFIG, vocab_size: 6, block_size: 8, n_layer: 1, d_model: 8, n_head: 2, n_kv_head: 2, d_ff: 16,
  norm_placement: 'post', pos_encoding: 'sinusoidal', activation: 'relu', tie_weights: true };
export function createEncoderDecoder(seed = 23) {
  const e = createModel(ENCDEC_CONFIG, seed), d = createModel(ENCDEC_CONFIG, seed + 1), c = createModel(ENCDEC_CONFIG, seed + 2);
  const params = { 'embed.weight': e.params['wte.weight'] };
  const copy = (model, from, to) => {
    for (const [name, p] of Object.entries(model.params)) if (name.startsWith(from + '.')) params[to + name.slice(from.length)] = p;
  };
  copy(e, 'blocks.0', 'encoder');
  copy(d, 'blocks.0.attn', 'self_attn'); copy(c, 'blocks.0.attn', 'cross_attn');
  copy(d, 'blocks.0.ln1', 'norm1'); copy(d, 'blocks.0.ln2', 'norm2'); copy(c, 'blocks.0.ln2', 'norm3');
  copy(d, 'blocks.0.mlp', 'ff');
  return { config: { ...ENCDEC_CONFIG }, params };
}
export function encoderDecoderForward(model, source, target, { padding = 5 } = {}) {
  const cfg = model.config, P = model.params, D = cfg.d_model, H = cfg.n_head, dh = D / H;
  if (!source.length || !target.length || source.length > cfg.block_size || target.length > cfg.block_size || !source.some((id) => id !== padding)
    || [...source, ...target].some((id) => !Number.isInteger(id) || id < 0 || id >= cfg.vocab_size)) throw new Error('Invalid source/target token sequence');
  return noGrad(() => {
    const trace = {}, rec = (key, value) => (trace[key] = value);
    const lin = (name, x) => linear(x, P[`${name}.weight`], P[`${name}.bias`] ?? null);
    const norm = (name, x) => layerNorm(x, P[`${name}.weight`], P[`${name}.bias`], cfg.norm_eps);
    const embed = (ids, name) => rec(name, add(scale(embedding(P['embed.weight'], ids), Math.sqrt(D)), new Tensor(sinusoidalTable(ids.length, D), [ids.length, D])));
    const attend = (name, query, memory, mask, prefix) => {
      const q = rec(`${prefix}.q`, lin(`${name}.q_proj`, query)), k = rec(`${prefix}.k`, lin(`${name}.k_proj`, memory)), v = rec(`${prefix}.v`, lin(`${name}.v_proj`, memory));
      const qs = [], ks = [], vs = [], scores = [], probs = [], heads = [];
      for (let h = 0; h < H; h++) {
        const qh = sliceCols(q, h * dh, dh), kh = sliceCols(k, h * dh, dh), vh = sliceCols(v, h * dh, dh);
        qs.push(qh); ks.push(kh); vs.push(vh);
        const s = scale(matmulNT(qh, kh), 1 / Math.sqrt(dh));
        for (let i = 0; i < s.data.length; i++) if (!mask[i]) s.data[i] = -Infinity;
        const a = softmax(s);
        scores.push(s); probs.push(a); heads.push(matmul(a, vh));
      }
      rec(`${prefix}.q_heads`, stack(qs)); rec(`${prefix}.k_heads`, stack(ks)); rec(`${prefix}.v_heads`, stack(vs));
      rec(`${prefix}.masked`, stack(scores)); rec(`${prefix}.probs`, stack(probs)); rec(`${prefix}.ctx`, stack(heads));
      return rec(`${prefix}.out`, lin(`${name}.o_proj`, concatCols(heads)));
    };
    const ffn = (name, x, prefix) => {
      const up = rec(`${prefix}.up`, lin(`${name}.up_proj`, x)), act = rec(`${prefix}.act`, relu(up));
      return rec(`${prefix}.out`, lin(`${name}.down_proj`, act));
    };
    const S = source.length, T = target.length;
    const sourceMask = Uint8Array.from({ length: S * S }, (_, i) => source[i % S] !== padding ? 1 : 0);
    const targetMask = Uint8Array.from({ length: T * T }, (_, i) => i % T <= Math.floor(i / T) && target[i % T] !== padding ? 1 : 0);
    if (!targetMask.slice(0, T).some(Boolean)) throw new Error('The first decoder token must be visible');
    const crossMask = Uint8Array.from({ length: T * S }, (_, i) => source[i % S] !== padding ? 1 : 0);
    rec('source.mask', new Tensor(Float64Array.from(sourceMask), [S, S]));
    rec('target.mask', new Tensor(Float64Array.from(targetMask), [T, T]));
    rec('cross.mask', new Tensor(Float64Array.from(crossMask), [T, S]));
    let enc = embed(source, 'source.embed');
    enc = rec('encoder.mid', norm('encoder.ln1', add(enc, attend('encoder.attn', enc, enc, sourceMask, 'encoder.attn'))));
    const memory = rec('encoder.memory', norm('encoder.ln2', add(enc, ffn('encoder.mlp', enc, 'encoder.mlp'))));
    let dec = embed(target, 'target.embed');
    dec = rec('decoder.self_out', norm('norm1', add(dec, attend('self_attn', dec, dec, targetMask, 'decoder.self'))));
    dec = rec('decoder.cross_out', norm('norm2', add(dec, attend('cross_attn', dec, memory, crossMask, 'decoder.cross'))));
    dec = rec('decoder.out', norm('norm3', add(dec, ffn('ff', dec, 'decoder.mlp'))));
    const logits = rec('logits', linear(dec, P['embed.weight']));
    rec('probs', softmax(logits));
    return { logits, trace, source: [...source], target: [...target] };
  });
}

// Catalogue of every traced tensor: what it is, its formula, its symbolic
// shape, its component colour and the view that explains it. Keys use the
// same patterns as the `@trace` markers in reference/model.py ({i} = layer).
import { dHead } from '../engine/index.js';

const T = 'T', d = 'd', H = 'H', Hkv = 'Hkv', dh = 'dh', F = 'F', V = 'V';

export const TRACE_INFO = {
  'emb.tok': { title: 'Token embeddings', kind: 'embed', view: 'tokens', dims: [T, d],
    formula: 'X<sub>tok</sub> = W<sub>E</sub>[ids]', what: 'Each token id selects one row of the V×d embedding matrix.' },
  'emb.pos': { title: 'Positional encoding', kind: 'pos', view: 'positions', dims: [T, d],
    formula: 'P[t] (learned row or sin/cos)', what: 'A vector per position, added so the model can tell order apart.' },
  'emb.out': { title: 'Block input x₀', kind: 'resid', view: 'tokens', dims: [T, d],
    formula: 'x<sub>0</sub> = X<sub>tok</sub> + P', what: 'Start of the residual stream: token identity plus position.' },
  mask: { title: 'Attention mask', kind: 'mask', view: 'masking', dims: [T, T],
    formula: 'M<sub>ij</sub> = 1 if query i may see key j', what: 'Which keys each query is allowed to attend to.' },
  alibi: { title: 'ALiBi distance bias', kind: 'pos', view: 'positions', dims: [H, T, T],
    formula: 'b<sub>h,ij</sub> = −m<sub>h</sub>·|i−j|', what: 'A per-head linear penalty on distance, added to the scores.' },
  'L{i}.ln1': { title: 'Norm before attention', kind: 'norm', view: 'residual', dims: [T, d],
    formula: 'LN(x) = γ ⊙ (x − μ)/σ + β', what: 'Pre-LN: normalise each token vector before it enters attention.' },
  'L{i}.attn.q': { title: 'Queries Q', kind: 'q', view: 'attention', dims: [T, `${H}·${dh}`],
    formula: 'Q = x W<sub>Q</sub><sup>⊤</sup> + b<sub>Q</sub>', what: 'What each position is looking for.' },
  'L{i}.attn.k': { title: 'Keys K', kind: 'k', view: 'attention', dims: [T, `${Hkv}·${dh}`],
    formula: 'K = x W<sub>K</sub><sup>⊤</sup> + b<sub>K</sub>', what: 'What each position offers to be matched against.' },
  'L{i}.attn.v': { title: 'Values V', kind: 'v', view: 'attention', dims: [T, `${Hkv}·${dh}`],
    formula: 'V = x W<sub>V</sub><sup>⊤</sup> + b<sub>V</sub>', what: 'The content each position hands over when attended to.' },
  'L{i}.attn.q_heads': { title: 'Q split into heads', kind: 'q', view: 'multihead', dims: [H, T, dh],
    formula: 'view(B,T,H,dh).transpose(1,2)', what: 'The d-wide Q is cut into H slices of width dh, one per head.' },
  'L{i}.attn.k_heads': { title: 'K split into heads', kind: 'k', view: 'multihead', dims: [Hkv, T, dh],
    formula: 'view(B,T,Hkv,dh).transpose(1,2)', what: 'Keys per KV head.' },
  'L{i}.attn.v_heads': { title: 'V split into heads', kind: 'v', view: 'multihead', dims: [Hkv, T, dh],
    formula: 'view(B,T,Hkv,dh).transpose(1,2)', what: 'Values per KV head.' },
  'L{i}.attn.q_rot': { title: 'Q after RoPE', kind: 'q', view: 'positions', dims: [H, T, dh],
    formula: 'q′ = R(t·θ) q', what: 'Each 2-D pair of q is rotated by an angle proportional to its position.' },
  'L{i}.attn.k_rot': { title: 'K after RoPE', kind: 'k', view: 'positions', dims: [Hkv, T, dh],
    formula: 'k′ = R(t·θ) k', what: 'Keys get the same rotation, so q′·k′ depends only on the offset.' },
  'L{i}.attn.k_rep': { title: 'K shared across query heads', kind: 'k', view: 'multihead', dims: [H, T, dh],
    formula: 'repeat_interleave(H/Hkv)', what: 'GQA/MQA: each KV head serves a group of query heads.' },
  'L{i}.attn.v_rep': { title: 'V shared across query heads', kind: 'v', view: 'multihead', dims: [H, T, dh],
    formula: 'repeat_interleave(H/Hkv)', what: 'GQA/MQA: values are shared the same way as keys.' },
  'L{i}.attn.scores': { title: 'Raw scores QKᵀ', kind: 'attn', view: 'attention', dims: [H, T, T],
    formula: 'S = Q K<sup>⊤</sup>', what: 'Dot product of every query with every key.' },
  'L{i}.attn.scaled': { title: 'Scaled scores', kind: 'attn', view: 'attention', dims: [H, T, T],
    formula: 'S / √d<sub>h</sub>', what: 'Dividing by √dh keeps score variance near 1 so softmax does not saturate.' },
  'L{i}.attn.biased': { title: 'Scores + ALiBi', kind: 'attn', view: 'positions', dims: [H, T, T],
    formula: 'S + b<sub>ALiBi</sub>', what: 'Distance penalty added before masking.' },
  'L{i}.attn.masked': { title: 'Masked scores', kind: 'mask', view: 'masking', dims: [H, T, T],
    formula: 'S<sub>ij</sub> = −∞ where M<sub>ij</sub> = 0', what: 'Disallowed keys get −∞, which softmax turns into exactly 0.' },
  'L{i}.attn.probs': { title: 'Attention weights', kind: 'attn', view: 'attention', dims: [H, T, T],
    formula: 'A = softmax(S) row-wise', what: 'How much each query reads from each key. Rows sum to 1.' },
  'L{i}.attn.ctx': { title: 'Per-head output A·V', kind: 'v', view: 'attention', dims: [H, T, dh],
    formula: 'head<sub>h</sub> = A<sub>h</sub> V<sub>h</sub>', what: 'Weighted average of value vectors, per head.' },
  'L{i}.attn.concat': { title: 'Heads concatenated', kind: 'attn', view: 'multihead', dims: [T, `${H}·${dh}`],
    formula: 'concat(head<sub>1</sub>, …, head<sub>H</sub>)', what: 'Head outputs placed side by side again.' },
  'L{i}.attn.out': { title: 'Attention output (O proj)', kind: 'o', view: 'multihead', dims: [T, d],
    formula: 'concat · W<sub>O</sub><sup>⊤</sup> + b<sub>O</sub>', what: 'Mixes the heads and writes back to model width d.' },
  'L{i}.resid_mid': { title: 'Residual after attention', kind: 'resid', view: 'residual', dims: [T, d],
    formula: 'x ← x + Attn(LN(x))', what: 'Attention adds its result onto the residual stream.' },
  'L{i}.add1': { title: 'Sum before post-norm (attn)', kind: 'resid', view: 'residual', dims: [T, d],
    formula: 'x + Attn(x)', what: 'Post-LN: the residual sum before it is normalised.' },
  'L{i}.ln2': { title: 'Norm before MLP', kind: 'norm', view: 'residual', dims: [T, d],
    formula: 'LN(x)', what: 'Pre-LN: normalise before the feed-forward network.' },
  'L{i}.mlp.up': { title: 'MLP up-projection', kind: 'mlp', view: 'mlp', dims: [T, F],
    formula: 'h = x W<sub>up</sub><sup>⊤</sup> + b', what: 'Expand each token to the wider hidden size F (often 4d).' },
  'L{i}.mlp.gate': { title: 'MLP gate (SwiGLU)', kind: 'mlp', view: 'mlp', dims: [T, F],
    formula: 'g = x W<sub>gate</sub><sup>⊤</sup>', what: 'Second projection that decides how much of each unit passes.' },
  'L{i}.mlp.act': { title: 'MLP activation', kind: 'mlp', view: 'mlp', dims: [T, F],
    formula: 'GELU(h) · or · SiLU(g) ⊙ h', what: 'The only position-wise non-linearity in the block.' },
  'L{i}.mlp.out': { title: 'MLP down-projection', kind: 'mlp', view: 'mlp', dims: [T, d],
    formula: 'a W<sub>down</sub><sup>⊤</sup> + b', what: 'Project back to width d to write into the residual stream.' },
  'L{i}.resid_out': { title: 'Block output', kind: 'resid', view: 'residual', dims: [T, d],
    formula: 'x ← x + MLP(LN(x))', what: 'Residual stream after the full block; input to the next layer.' },
  'L{i}.add2': { title: 'Sum before post-norm (MLP)', kind: 'resid', view: 'residual', dims: [T, d],
    formula: 'x + MLP(x)', what: 'Post-LN: the residual sum before the second norm.' },
  lnf: { title: 'Final norm', kind: 'norm', view: 'residual', dims: [T, d],
    formula: 'LN<sub>f</sub>(x)', what: 'One last normalisation before un-embedding (pre-LN models).' },
  logits: { title: 'Logits', kind: 'head', view: 'head', dims: [T, V],
    formula: 'z = x W<sub>U</sub><sup>⊤</sup>', what: 'One score per vocabulary entry, for every position.' },
  probs: { title: 'Next-token probabilities', kind: 'head', view: 'head', dims: [T, V],
    formula: 'p = softmax(z)', what: 'Row t is the predicted distribution for token t+1.' },
  loss: { title: 'Cross-entropy loss', kind: 'head', view: 'head', dims: [],
    formula: '−mean log p(target)', what: 'Average surprise at the true next token.' },
};

// 'L3.attn.q' -> {pattern: 'L{i}.attn.q', layer: 3}
export function parseKey(key) {
  const m = /^L(\d+)\.(.*)$/.exec(key);
  return m ? { pattern: `L{i}.${m[2]}`, layer: Number(m[1]) } : { pattern: key, layer: null };
}

export function infoFor(key) {
  const { pattern, layer } = parseKey(key);
  const info = TRACE_INFO[pattern];
  if (!info) return { title: key, kind: 'norm', view: 'stepper', dims: [], formula: '', what: '', pattern, layer };
  return { ...info, pattern, layer, label: layer == null ? info.title : `${info.title} · layer ${layer}` };
}

// Symbol values for a config and sequence length.
export function dimValues(cfg, seqLen) {
  return { T: seqLen, d: cfg.d_model, H: cfg.n_head, Hkv: cfg.n_kv_head, dh: dHead(cfg), F: cfg.d_ff, V: cfg.vocab_size };
}

// [['T', 7], ['d', 64]] style pairs for shapeBadge; composite dims like 'H·dh' are evaluated.
export function shapePairs(key, cfg, seqLen) {
  const vals = dimValues(cfg, seqLen);
  return infoFor(key).dims.map((sym) => [sym, sym.split('·').reduce((p, s) => p * (vals[s] ?? 1), 1)]);
}

// Optional depth, reached from an existing equation card; no extra scene text.
import { parseKey } from '../describe.js';
export function furtherStudy(key) {
  const { pattern } = parseKey(key);
  let keys = [];
  if (/\.attn\./.test(pattern) || pattern === 'mask' || pattern === 'concept:online-attention')
    keys = ['concept:attention-backward', 'concept:attention-invariants', 'concept:head-basis', 'concept:attention-circuits', 'concept:attention-numerics', 'concept:qk-normalization', 'concept:mla'];
  if (/\.ln[12]$|^lnf$|\.resid_|\.add[12]$/.test(pattern) || /concept:(layernorm-backward|pre-vs-post)/.test(pattern))
    keys = ['concept:layernorm-backward', 'concept:rmsnorm-backward', 'concept:norm-epsilon', 'concept:pre-vs-post'];
  if (/\.mlp\./.test(pattern) || /concept:(gelu|swiglu|key-value-memory)/.test(pattern))
    keys = ['concept:gelu', 'concept:swiglu', 'concept:mlp-interpretation', 'concept:training-gradient'];
  if (/emb\.pos|\.attn\.[qk]_rot|concept:(rope-relative|sinusoidal|alibi)/.test(pattern))
    keys.push('concept:rope-relative', 'concept:rope-context', 'concept:permutation', 'concept:head-basis');
  if (pattern === 'concept:kv-cache') keys = ['concept:cache-equivalence', 'concept:attention-numerics', 'concept:mla'];
  if (pattern === 'concept:training-gradient') keys = ['concept:attention-backward', 'concept:layernorm-backward', 'concept:rmsnorm-backward', 'concept:gelu', 'concept:mlp-interpretation'];
  if (pattern === 'concept:tensor-layout') keys = ['concept:attention-backward', 'concept:permutation', 'concept:head-basis'];
  if (pattern === 'concept:architecture-choices' || pattern === 'concept:encoder-decoder')
    keys = ['concept:attention-backward', 'concept:qk-normalization', 'concept:rope-context', 'concept:mla', 'concept:attention-circuits', 'concept:permutation'];
  return [...new Set(keys)].filter(k => k !== key);
}

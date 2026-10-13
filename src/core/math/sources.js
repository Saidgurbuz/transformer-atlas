// Paper locations attached to the equation they support, rather than a generic reading list.
import { parseKey } from '../describe.js';

export function equationSources(key, cfg) {
  const { pattern } = parseKey(key);
  if (pattern === 'concept:adamw') return [{ id: 'adamw2017', where: 'Algorithm 2', note: 'The decay is decoupled from the adaptive gradient update. Parameter grouping is an explicit lesson convention.' }];
  if (pattern === 'concept:training-gradient') return [{ id: 'vaswani2017', where: '§3.4, embeddings and softmax; §3.1, shifted decoder inputs', note: 'The derivatives are derived here and independently checked against PyTorch, including ignored targets and tied embedding gradients.' }];
  if (pattern === 'concept:tensor-layout') return [{ id: 'vaswani2017', where: '§3.2.2, Multi-head attention', note: 'The head decomposition is written with the batch axis explicit. View, stride and copying rules are implementation semantics explained by the executable tensor_ops.py reference.' }];
  const attention = { id: 'vaswani2017', where: '§3.2.1, Eq. (1)', note: 'Scaled dot products, softmax, and the weighted value sum. Masking is described in §3.2.3.' };
  const norm = cfg.norm === 'rmsnorm'
    ? { id: 'zhang2019rms', where: '§4, RMSNorm', note: 'Atlas adds norm_eps inside the square root; RMSNorm has a gain and no learned shift.' }
    : { id: 'ba2016', where: '§3, Layer normalization', note: 'Atlas normalizes each token across features, uses biased variance, and adds norm_eps for stability.' };
  if (/^(L\{i\}\.ln[12]|lnf|L\{i\}\.resid_(mid|out)|L\{i\}\.add[12])$/.test(pattern)) return [norm,
    { id: 'xiong2020', where: '§3, Table 1 and Figure 1', note: 'Pre-LN and post-LN place normalization on different sides of the residual addition.' }];
  if (/\.attn\.[qkv]_heads$/.test(pattern) || /\.attn\.[kv]_rep$/.test(pattern)) return [
    { id: 'vaswani2017', where: '§3.2.2, Multi-head attention', note: 'Atlas stores linear weights as [out, in], so row-vector equations use Wᵀ.' },
    ...(cfg.n_kv_head < cfg.n_head ? [{ id: 'ainslie2023gqa', where: '§2.2 and Figure 2', note: 'Each contiguous group of query heads shares one KV head.' }] : [])];
  if (/\.attn\.[qk]_rot$/.test(pattern) || pattern === 'concept:rope-relative') return [
    { id: 'su2021', where: '§3.2, Eqs. (13)–(16)', note: 'Atlas rotates interleaved coordinate pairs (2k, 2k+1); some implementations use a different coordinate ordering.' }];
  if (pattern === 'alibi' || /\.attn\.biased$/.test(pattern) || pattern === 'concept:alibi') return [
    { id: 'press2021', where: '§3 and Figure 3', note: 'Atlas uses −mₕ|i−j| and geometric slopes for every head count. For causal attention this equals the paper’s distance penalty; non-power-of-two slope ordering differs from the authors’ code.' }];
  if (/\.attn\.(scores|scaled|masked|probs|ctx)$/.test(pattern) || pattern === 'mask') return [attention];
  if (/\.attn\.(q|k|v|concat|out)$/.test(pattern)) return [
    { id: 'vaswani2017', where: '§3.2.2, Multi-head attention', note: 'The paper’s weight orientation differs from nn.Linear storage; Atlas uses y = xWᵀ + b.' }];
  if (/\.mlp\./.test(pattern)) return cfg.activation === 'swiglu'
    ? [{ id: 'shazeer2020', where: 'Eqs. (5)–(6)', note: 'Atlas uses SiLU on the gate projection (Swish with β=1), multiplied elementwise by the up projection.' }]
    : [{ id: 'vaswani2017', where: '§3.3, Eq. (2)', note: 'The original FFN uses ReLU. Atlas also offers GELU’s tanh approximation.' },
      ...(cfg.activation === 'gelu' ? [{ id: 'hendrycks2016', where: '§2, GELU approximation', note: 'The live implementation uses the tanh approximation rather than the exact Gaussian CDF.' }] : [])];
  if (/^emb\./.test(pattern) || /^(logits|probs)$/.test(pattern)) return [
    { id: 'vaswani2017', where: pattern === 'emb.pos' ? '§3.5, Positional encoding' : '§3.4, Embeddings and softmax', note: 'Atlas does not multiply token embeddings by √d_model; tying weights is configurable.' }];
  if (pattern === 'concept:kv-cache') return [{ id: 'ainslie2023gqa', where: '§2.2 and Figure 2', note: 'Reducing KV heads reduces cache storage. The byte-count equation is arithmetic for Atlas’s cache layout, not a copied paper equation.' }];
  return [];
}

const t = String.raw;
export const ENTRIES = {
  'concept:attention-invariants': {
    title: 'Softmax invariants and their boundaries',
    tex: t`\softmax(z+c\mathbf1)=\softmax(z)`,
    compact: t`\softmax(z+c\mathbf1)=\softmax(z)`,
    symbols: [{ tex: 'c', text: 'one finite constant per row, not a different constant per key' }],
    derivation: [
      { title: 'Shift invariance explains stable softmax', steps: [t`\frac{e^{z_j+c}}{\sum_k e^{z_k+c}}=\frac{e^{z_j}}{\sum_k e^{z_k}}`, t`J_{\mathrm{softmax}}\mathbf1=0`], note: 'This is an algebraic identity. Floating-point evaluation still needs max subtraction to avoid overflow. Multiplying logits by a constant generally changes the distribution; it changes the effective temperature.' },
      { title: 'A head output is a convex combination before dropout', steps: [t`A_{ij}\ge0,\quad\sum_jA_{ij}=1,\quad o_i=\sum_jA_{ij}v_j`], note: 'With no dropout and a visible key per row, o lies in the convex hull of the visible values. Attention dropout rescales retained weights without requiring their realized sum to be one. Output projection, head combination, residual addition and normalization change which vectors this statement describes.' },
      { title: 'Masked entries versus all-masked rows', steps: [t`M_{ij}=0\Longrightarrow A_{ij}=0`], note: 'This requires a well-defined normalizer from at least one visible score. A row of only −∞ has no softmax distribution. Different implementations reject, return NaN or adopt a zero-output convention; treat that policy explicitly rather than calling it a probability distribution.', reference: { file: 'attention_backward.py', region: 'attention_forward' } },
    ],
  },
  'concept:norm-epsilon': {
    title: 'Normalization: exact and approximate invariance',
    tex: t`\operatorname{Var}(\hat x)=\frac{\sigma^2}{\sigma^2+\varepsilon}`,
    compact: t`\operatorname{Var}(\hat x)=\sigma^2/(\sigma^2+\varepsilon)`,
    symbols: [{ tex: t`\hat x`, text: 'LayerNorm output before gain and shift: (x−μ)/√(σ²+ε)' }, { tex: t`\sigma^2`, text: 'biased feature variance of one token' }],
    derivation: [
      { title: 'Common shifts disappear exactly', steps: [t`\LN(x+c\mathbf1)=\LN(x)`, t`J_{\LN}\mathbf1=0`], note: 'The same featurewise gain/shift parameters and epsilon are used on both sides. LayerNorm subtracts the feature mean; RMSNorm generally does not have this shift invariance.' },
      { title: 'Epsilon leaves a radial derivative', steps: [t`u=x-\mu\mathbf1,\quad s=\sqrt{\sigma^2+\varepsilon}`, t`J_{\hat x}u=\frac{\varepsilon}{s^3}u`], note: 'This formula is before learned gain. For ε > 0, the centered direction is attenuated, not removed exactly. Scale invariance is an approximation when σ² is large relative to ε. If ε=0 and σ²>0, positive scaling is invariant; negative scaling flips the normalized vector.', reference: { file: 'numpy_transformer.py', region: 'layernorm_backward' } },
      { title: 'Gain/shift and RMSNorm change the claim', steps: [t`y=\gamma\odot\hat x+\beta`, t`\operatorname{mean}((x/r)^2)=\frac{\operatorname{mean}(x^2)}{\operatorname{mean}(x^2)+\varepsilon}`], note: 'The affine output need not have zero mean or unit variance. RMSNorm similarly has only approximate positive scale invariance with epsilon, and no general shift invariance. Constant or zero inputs require the epsilon term.' },
    ],
  },
  'concept:head-basis': {
    title: 'What is identifiable about an attention head?',
    scope: 'The unrestricted basis-change argument assumes a bias-free, unrotated head without QK normalization.',
    tex: t`W_{QK}=W_Q^\top W_K,\qquad W_{OV}=W_O^{(h)}W_V`,
    compact: t`W_{QK}=W_Q^\top W_K,\quad W_{OV}=W_O^{(h)}W_V`,
    symbols: [{ tex: t`W_Q,W_K,W_V`, text: 'one head’s projections stored [head width,model width]' }, { tex: t`W_O^{(h)}`, text: 'columns of W_O consuming this head: [model width,head width]' }],
    derivation: [
      { title: 'Equivalent parameters in a bias-free, unrotated head', steps: [t`W_Q'=CW_Q,\quad W_K'=C^{-\top}W_K`, t`W_V'=DW_V,\quad W_O'=W_OD^{-1}`], note: 'C and D are invertible head-space matrices, fixed across tokens. These changes preserve the QK and OV products, so individual channel coordinates are not uniquely identifiable from the function. C and D can be different; the code uses one change matrix for a small algebra check.', reference: { file: 'invariance_checks.py', region: 'head_basis_change' } },
      { title: 'RoPE and QK normalization restrict the equivalence', steps: [t`C^\top R_{j-i}C^{-\top}=R_{j-i}\quad\text{is required to preserve rotary scores}`], note: 'An arbitrary invertible C does not satisfy this condition. Nonlinear Q/K normalization also changes the argument, and shared KV heads constrain independent changes across query heads. This is a derived algebraic condition, not a claim that all production heads admit arbitrary reparameterization.' },
      { title: 'Rank is an architectural restriction', steps: [t`\operatorname{rank}(W_{QK})\le d_h,\qquad\operatorname{rank}(W_{OV})\le d_v`], note: 'These are per-head linear-product rank bounds. They do not bound the complete nonlinear network or imply that a head represents only one interpretable feature.' },
    ],
  },
  'concept:permutation': {
    title: 'Token permutations, positions and causal masks',
    tex: t`F(PX,PMP^\top)=PF(X,M)`,
    compact: t`F(PX,PMP^\top)=PF(X,M)`,
    symbols: [{ tex: 'P', text: 'permutation matrix on token rows' }, { tex: 'M', text: 'allowed-key mask, with both query and key axes reordered' }, { tex: 'F', text: 'attention with shared tokenwise projections and no index-tied position signal' }],
    derivation: [
      { title: 'The scores and weights permute consistently', steps: [t`Q'=PQ,\quad K'=PK,\quad V'=PV`, t`S'=PSP^\top,\quad A'=PAP^\top,\quad A'V'=PAV`], note: 'Tokenwise FFNs and norms with shared parameters also commute with row permutation. This is an exact architectural statement under these assumptions, not an empirical claim about learned language behavior.', reference: { file: 'invariance_checks.py', region: 'permutation_check' } },
      { title: 'A fixed causal mask is a counterexample', steps: [t`M_{ij}=\mathbf1[j\le i]`], note: 'Arbitrarily permuting X while retaining this fixed mask changes the allowed reads. Removing explicit positional embeddings therefore does not make causal attention permutation equivariant in general. Absolute position vectors assigned to fixed indices are another source of ordering.' },
    ],
  },
  'concept:cache-equivalence': {
    title: 'When cached decoding equals a full forward pass',
    tex: t`z_t^{\mathrm{cached}}\approx z_t^{\mathrm{full}}`,
    compact: t`z_t^{\mathrm{cached}}\approx z_t^{\mathrm{full}}`,
    symbols: [{ tex: t`\approx`, text: 'same real-arithmetic computation, compared within floating-point tolerances' }],
    derivation: [
      { title: 'Induction through causal layers', steps: [t`x_i^{(\ell)}\text{ depends only on }x_{\le i}^{(0)}`], note: 'With causal or causal-sliding reads, appending a token cannot change earlier hidden states. Their projected K/V remain valid at each layer. Evaluate with fixed weights, mask/position configuration, device/dtype and prompt prefix; disable dropout and stay within the supported context.', reference: { file: 'invariance_checks.py', region: 'cache_equivalence' } },
      { title: 'Positions and cache lifetime are part of the contract', steps: [t`q_t^{\mathrm{rot}}=R_tq_t,\quad k_i^{\mathrm{rot}}=R_ik_i`], note: 'The one-token input uses its absolute position t, not zero. Changing rotary frequencies or the prompt invalidates retained keys unless an implementation explicitly updates them. Bidirectional and incrementally built prefix attention can change earlier states. Atlas rejects those incremental cache modes and retains the full prefix under sliding masks.' },
      { title: 'Validate outputs, not just cache shapes', steps: [t`|z^{\mathrm{cached}}-z^{\mathrm{full}}|\le\mathrm{atol}+\mathrm{rtol}|z^{\mathrm{full}}|`], note: 'The small Python comparison checks every generated-prefix logit row. Near-zero values need an absolute tolerance. A matching cache shape alone does not establish correct masking, positional offsets or numerical behavior.' },
    ],
  },
};

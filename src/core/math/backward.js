const t = String.raw;
export const ENTRIES = {
  'concept:attention-backward': {
    title: 'Attention backward, end to end',
    tex: t`\begin{aligned}dQ&=\alpha\,dZK\\dK&=\alpha\,dZ^\top Q\\dV&=A^\top G\end{aligned}`,
    compact: t`dQ=\alpha dZK,\quad dK=\alpha dZ^\top Q,\quad dV=A^\top G`,
    symbols: [{ tex: 'G', text: 'upstream gradient at O = AV, before head concatenation and output projection; [T,dv]' },
      { tex: 'A,Z', text: 'attention probabilities and pre-mask scores Z = αQKᵀ + B; both [T,S]' },
      { tex: 'Q,K,V', text: 'single-head matrices [T,dh], [S,dh], [S,dv]; batch/head axes omitted' },
      { tex: t`\alpha`, text: '1/√dh, or 1 when scaling is disabled; B is an optional additive score bias' }],
    derivation: [
      { title: 'Assumptions and where the gradient enters', steps: [t`O=AV,\qquad A=\softmax(\mathrm{mask}(Z))`], note: 'Exact first-order derivatives for fixed masks and positions, no dropout, and at least one visible key per query. With dropout, differentiate the dropped/scaled probabilities before the softmax VJP. The isolated code checks rectangular attention and optional score bias against PyTorch SDPA/autograd.', reference: { file: 'attention_backward.py', region: 'attention_forward', label: 'Read the matching forward pass' } },
      { title: 'Values and attention weights', steps: [t`dV=A^\top G,\qquad dA=GV^\top`], note: 'A key/value participates in multiple query rows. Its gradient sums all those uses, so a blocked edge does not freeze the whole key or value.', reference: { file: 'attention_backward.py', region: 'attention_backward' } },
      { title: 'Softmax VJP, without constructing a Jacobian', steps: [t`dZ_{i,j}=A_{i,j}\left(dA_{i,j}-\sum_k A_{i,k}dA_{i,k}\right)`], note: 'The row reduction couples the visible scores. masked_fill has zero derivative on blocked entries. An additive bias receives dZ, reduced over any axes broadcast in the forward pass; only Q/K paths receive the multiplier α.', reference: { file: 'numpy_transformer.py', region: 'softmax_backward' } },
      { title: 'Projection gradients and the three input paths', steps: [t`Q=XW_Q^\top+b_Q,\quad dW_Q=dQ^\top X,\quad db_Q=\sum_i dQ_i`, t`dX=dQW_Q+dKW_K+dVW_V`], note: 'The last sum applies to self-attention, where all three projections read the same X. In cross-attention, the query gradient returns to the decoder and K/V gradients return to encoder memory. Weight matrices are stored [out,in].', reference: { file: 'numpy_transformer.py', region: 'linear_backward' } },
      { title: 'Shared KV heads accumulate their query-head uses', steps: [t`dK_g=\sum_{h:\,\mathrm{group}(h)=g}dK_h^{\mathrm{repeated}},\qquad dV_g=\sum_{h:\,\mathrm{group}(h)=g}dV_h^{\mathrm{repeated}}`], note: 'Undo contiguous head repetition by summing each group. Averaging would change the gradient. This reduction precedes undoing the KV RoPE rotation.', reference: { file: 'attention_backward.py', region: 'shared_kv_backward' } },
      { title: 'RoPE backward applies the inverse rotation', steps: [t`q^{\mathrm{rot}}=R_iq\quad\Longrightarrow\quad dq=R_i^\top dq^{\mathrm{rot}}`], note: 'Here q is a column vector for one rotary pair. Rᵀ = R⁻¹ because the rotation is orthogonal. Positions/frequencies are constants in the main model.', reference: { file: 'numpy_transformer.py', region: 'rope_backward' } },
      { title: 'Embedding lookup and tied weights', steps: [t`dW_E[v,:]=\sum_{i:\,\mathrm{id}_i=v}dX_i`, t`dW_E^{\mathrm{tied}}=dW_E^{\mathrm{lookup}}+dW_E^{\mathrm{head}}`], note: 'Repeated token IDs scatter-add into the same embedding row. A tied output matrix also receives the unembedding contribution before one optimizer update.', reference: { file: 'numpy_transformer.py', region: 'embedding_backward' } },
    ],
  },
  'concept:rmsnorm-backward': {
    title: 'RMSNorm backward and epsilon',
    tex: t`dx=\frac{g-u\,\operatorname{mean}(g\odot u)}{r}`,
    compact: t`dx=(g-u\,\operatorname{mean}(g\odot u))/r`,
    symbols: [{ tex: t`r`, text: '√(mean(x²)+ε), per token' }, { tex: 'u', text: 'x/r, before learned gain' }, { tex: t`g`, text: 'upstream gradient multiplied elementwise by γ' }],
    derivation: [
      { title: 'Differentiate the RMS denominator', steps: [t`\frac{\partial r}{\partial x_k}=\frac{x_k}{dr}`, t`\frac{\partial y_j}{\partial x_k}=\gamma_j\left(\frac{\delta_{jk}}r-\frac{x_jx_k}{dr^3}\right)`], note: 'Contract this Jacobian with the upstream gradient to obtain the displayed VJP. There is no mean subtraction as in LayerNorm; this does not mean that the input gradient is componentwise independent.', reference: { file: 'numpy_transformer.py', region: 'rmsnorm_backward' } },
      { title: 'Gain gradient and the zero-input case', steps: [t`d\gamma_c=\sum_i dy_{i,c}u_{i,c}`, t`x=0\quad\Longrightarrow\quad dx=g/\sqrt\varepsilon`], note: 'Sum gain gradients over all token/batch uses. Epsilon prevents division by zero but can make gradients large near zero; the scale depends on ε. This exact formula is checked against PyTorch, including zero inputs.' },
    ],
  },
};

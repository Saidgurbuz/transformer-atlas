const t = String.raw;
export const ENTRIES = {
  'concept:attention-numerics': {
    title: 'Attention in real arithmetic and in a library kernel',
    tex: t`|o_{\mathrm{reference}}-o_{\mathrm{kernel}}|\le\mathrm{atol}+\mathrm{rtol}|o_{\mathrm{reference}}|`,
    compact: t`|\Delta o|\le\mathrm{atol}+\mathrm{rtol}|o_{\mathrm{reference}}|`,
    symbols: [{ tex: t`\mathrm{atol},\mathrm{rtol}`, text: 'absolute and relative tolerances chosen for dtype, scale and the quantity being checked' }],
    derivation: [
      { title: 'Same attention operation does not mean identical bits', steps: [t`O=\softmax(\mathrm{mask}(QK^\top/\sqrt{d_h}+B))V`], note: 'Reduction order, fusion, rounding and intermediate precision can differ. Atlas’s browser engine uses Float64; this does not model every low-precision production kernel. The code compares the same masked operation against CPU SDPA in float64, float32 and bfloat16. Tolerances for those small tests are not universal error bounds.', reference: { file: 'sdpa_checks.py', region: 'compare_sdpa' } },
      { title: 'Boolean masks have API-specific meanings', steps: [t`\text{SDPA: True means allowed; blocked scores use }-\infty`], note: 'PyTorch SDPA Boolean True means allowed. nn.MultiheadAttention key_padding_mask uses True for ignored keys, so do not transfer a mask without checking the API. Float masks are additive score biases. This reference rejects rows with no visible key; its tested CPU SDPA path instead returns zero, an explicit convention rather than a probability distribution.', reference: { file: 'sdpa_checks.py', region: 'manual_attention' } },
      { title: 'Rectangular causal alignment and cached queries', steps: [t`M_{q,j}=\mathbf1[j\le q]\quad\text{with absolute positions}`], note: 'SDPA is_causal=True uses upper-left alignment for a rectangular mask. A one-query row over a longer cached prefix therefore cannot blindly use it: supply an offset-aware mask. The example checks a last-position query that should read the whole prefix against that incorrect shortcut.', reference: { file: 'sdpa_checks.py', region: 'cached_query_mask' } },
      { title: 'Dropout and grouped KV are implementation contracts', steps: [], note: 'SDPA applies the supplied dropout_p even during evaluation; pass zero explicitly when dropout is disabled. GQA requires compatible head counts and backend support. Explicit contiguous K/V repetition and enable_gqa are checked for equivalent outputs here; this CPU comparison makes no claim about GPU memory use or speed.', reference: { file: 'sdpa_checks.py', region: 'grouped_attention' } },
      { title: 'A focused debugging order', steps: [], note: 'First align layouts, scale/bias, masks, absolute positions, head mapping and dropout. Then compare outputs and gradients with dtype-appropriate tolerances. For large deviations or NaNs, inspect those contracts before widening tolerances. Near-zero reference values need an absolute tolerance.' },
    ],
  },
};

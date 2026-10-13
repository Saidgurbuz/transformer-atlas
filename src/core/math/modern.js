const t = String.raw;
export const ENTRIES = {
  'concept:qk-normalization': {
    title: 'QK normalization and score soft-capping',
    scope: 'Separate numerical reference; these variants are not active in the main model.',
    tex: t`S_{ij}=g\,\langle\hat q_i,\hat k_j\rangle`,
    compact: t`S_{ij}=g\langle\hat q_i,\hat k_j\rangle`,
    symbols: [{ tex: t`\hat q,\hat k`, text: 'L2-normalized query/key directions in this formula; other recipes use RMSNorm' }, { tex: 'g', text: 'scalar score gain; Henry et al. learn it instead of using the standard 1/√dh factor' }],
    derivation: [
      { title: 'L2 normalization: direction and score gain', steps: [t`\hat q=q/\max(\|q\|_2,\varepsilon),\qquad |S_{ij}|\le|g|`], note: 'Above the epsilon floor, positive rescaling of either vector leaves its direction unchanged. A large learned gain can still concentrate softmax; normalization is not a universal stability guarantee. The reference returns zero directions for zero inputs.', reference: { file: 'modern_attention.py', region: 'qk_l2_scores' } },
      { title: 'RMSNorm is a different normalization convention', steps: [t`\bar q=\gamma_q\odot\frac{q}{\sqrt{\operatorname{mean}(q^2)+\varepsilon}}`, t`S=\bar Q\bar K^\top/\sqrt{d_h}`], note: 'Gemma 3 uses a QK-normalized recipe. This small RMS example shows feature gains and an explicit score scale; it is not a Gemma checkpoint implementation. With unit gains and zero epsilon, RMS directions have norm √dh, so the displayed score is √dh times cosine similarity, not unit cosine similarity.', reference: { file: 'modern_attention.py', region: 'qk_rms_scores' } },
      { title: 'Smooth capping bounds scores but changes the gradient', steps: [t`s_{\mathrm{cap}}=c\tanh(s/c),\quad c>0`, t`\frac{ds_{\mathrm{cap}}}{ds}=1-\tanh^2(s/c)`], note: 'This is a smooth transform of scores, not hard clipping or Q/K normalization. Its derivative becomes small for very large |s|. Gemma 3 replaces Gemma 2’s soft-capping with QK-norm; those choices should not be conflated.', reference: { file: 'modern_attention.py', region: 'softcap_scores' } },
      { title: 'What to compare when changing the recipe', steps: [], note: 'Specify the normalized axis, L2 versus RMS denominator, epsilon, learned gain, position of RoPE and score scaling. These determine both forward scores and backward gradients. The reference checks the algebra and derivatives; it makes no trained-model quality or GPU performance claim.' },
    ],
  },
};

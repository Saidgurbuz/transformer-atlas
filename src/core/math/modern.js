const t = String.raw;
export const ENTRIES = {
  'concept:mla': {
    title: 'MLA: latent KV storage and absorbed projections',
    scope: 'Separate bias-free MLA algebra reference; the main model uses MHA/GQA, not MLA.',
    tex: t`\mathrm{cache\ bytes}=BLS(d_c+d_R)b`,
    compact: t`\mathrm{cache\ bytes}=BLS(d_c+d_R)b`,
    symbols: [{ tex: 'B,L,S', text: 'batch size, layer count and cached key-token count' }, { tex: 'd_c,d_R', text: 'latent KV width and shared rotary-key width' }, { tex: 'b', text: 'bytes per stored scalar; excludes weights and allocation overhead' }],
    derivation: [
      { title: 'What is stored and what is reconstructed', steps: [t`c_j=W^{DKV}x_j,\quad k_{j,h}^C=W_h^{UK}c_j,\quad v_{j,h}^C=W_h^{UV}c_j`, t`k_j^R=\mathrm{RoPE}(W^{KR}x_j)`], note: 'Cache c and the shared rotary key, rather than all expanded headwise K/V. The example accepts already projected queries, latent vectors and rotated keys; query compression and normalization are outside its scope.', reference: { file: 'mla_reference.py', region: 'mla_dense' } },
      { title: 'Content and position scores are separate', steps: [t`s_{ij,h}=\frac{(q_{i,h}^C)^\top W_h^{UK}c_j+(q_{i,h}^R)^\top k_j^R}{\sqrt{d_h+d_R}}`], note: 'Keeping the rotary path separate permits a fixed content projection to be absorbed into the query. Rotating the reconstructed content key would introduce a position-dependent matrix into that absorption.', reference: { file: 'mla_reference.py', region: '_probabilities' } },
      { title: 'Absorb linear projections without reconstructing the prefix', steps: [t`\tilde q_{i,h}=(W_h^{UK})^\top q_{i,h}^C`, t`o_i=\sum_h(W_O^{(h)}W_h^{UV})\sum_j A_{ij,h}c_j`], note: 'These are derived linear rearrangements for the same MLA parameters. Dense and absorbed outputs and gradients are checked independently. No nonlinear operation is inserted between the absorbed matrices.', reference: { file: 'mla_reference.py', region: 'mla_absorbed' } },
      { title: 'GQA comparison and the compression tradeoff', steps: [t`\mathrm{GQA\ cache\ bytes}=BLSH_{KV}(d_k+d_v)b`], note: 'GQA shares whole K/V heads; MLA learns a joint latent space. Exact rearrangement within one MLA parameterization does not mean that arbitrary MHA weights can be compressed losslessly. Savings depend on dimensions and backend; this example is not a quality or speed benchmark.', reference: { file: 'mla_reference.py', region: 'mla_cache_bytes' } },
    ],
  },
  'concept:rope-context': {
    title: 'Long-context RoPE: interpolation, YaRN and cache validity',
    scope: 'Separate fixed-frequency reference; context-extension methods are not active in the main model.',
    tex: t`\omega_i'=\left(\frac{1-\gamma_i}{s}+\gamma_i\right)\omega_i`,
    compact: t`\omega_i'=((1-\gamma_i)/s+\gamma_i)\omega_i`,
    symbols: [{ tex: t`\omega_i`, text: 'original rotary frequency base^(−2i/dh), with 0-based pair index i' }, { tex: 's', text: 'context extension factor ≥1' }, { tex: t`\gamma_i`, text: 'blend weight: retain the original frequency at 1, interpolate at 0' }],
    derivation: [
      { title: 'Position interpolation and frequency changes', steps: [t`\omega_i=\mathrm{base}^{-2i/d_h},\qquad\omega_i^{\mathrm{PI}}=\omega_i/s`], note: 'PI scales all frequencies equally. Changing the base affects frequencies unequally. Neither the relative-rotation identity nor a longer sine/cosine table proves useful length generalization.', reference: { file: 'rope_extensions.py', region: 'rope_frequencies' } },
      { title: 'The paper ramp and implementation ramp differ', steps: [t`r_i=L_0\omega_i/(2\pi),\qquad\gamma_i=\mathrm{clip}((r_i-\alpha)/(\beta-\alpha),0,1)`], note: 'This is the rotation-count ramp. The authors’ code instead ramps linearly in pair index between correction boundaries. The reference supports both explicitly and validates its index branch against the authors’ implementation.', reference: { file: 'rope_extensions.py', region: 'yarn_frequencies' } },
      { title: 'YaRN also changes score temperature', steps: [t`m=\sqrt{1/t}=1+0.1\ln s`, t`\langle mR_iq,mR_jk\rangle=m^2\langle q,R_{j-i}k\rangle`], note: 'The amplitude rule is an empirical recommendation for LLaMA-family models, not a mathematical necessity. A pure rotation preserves norms; multiplying it by m scales norms and logits.', reference: { file: 'rope_extensions.py', region: 'rotate_interleaved' } },
      { title: 'Cache compatibility and evaluation', steps: [], note: 'Changing frequencies invalidates already-rotated cached keys unless they are recomputed or stored unrotated. Keep positions absolute. Validate task performance and long-context behavior separately from algebra; this reference does not fine-tune a model or guarantee an extended context.' },
    ],
  },
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

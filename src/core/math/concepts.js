// Concept entries ('concept:*'): derivations and cost formulas not tied to one tensor.
export const ENTRIES = {
  'concept:encoder-decoder': {
    title: 'Cross-attention and the complete post-norm decoder',
    tex: '\\begin{aligned}D_1&=\\LN(D+\\mathrm{MaskedAttn}(D,D,D))\\\\ Q&=D_1W_Q^\\top,\\quad K=EW_K^\\top,\\quad V=EW_V^\\top\\\\ A&=\\softmax(\\mathrm{mask}(QK^\\top/\\sqrt{d_h}))\\in\\R^{T\\times S}\\\\ D_2&=\\LN(D_1+\\mathrm{CrossAttn}(D_1,E,E))\\\\ D_3&=\\LN(D_2+\\mathrm{FFN}(D_2))\\end{aligned}',
    compact: 'A=\\softmax(\\mathrm{mask}(QK^\\top/\\sqrt{d_h}))\\in\\R^{T\\times S}',
    symbols: [{ tex: 'E,D', text: 'encoder memory and decoder input; batch/head axes omitted' }, { tex: 'S,T', text: 'source and target lengths, which may differ' }],
    derivation: [{ title: 'Three distinct decoder sub-layers', steps: ['D\\to D_1\\to D_2\\to D_3'], note: 'Self-attention masks future target keys. Cross-attention reads all non-padding source positions. The separate lesson uses one encoder/decoder block, shared vocabulary weights, √d-scaled embeddings, sinusoidal positions, ReLU and post-LayerNorm, without dropout. Its random weights are not a translation model.' }],
  },
  'concept:online-attention': {
    title: 'Stable online softmax and the output accumulator',
    tex: '\\begin{aligned}m^+&=\\max(m,\\max_{j\\in J}s_j),\\quad\\alpha=e^{m-m^+}\\\\ \\tilde p_j&=e^{s_j-m^+}\\\\ \\ell^+&=\\alpha\\ell+\\sum_{j\\in J}\\tilde p_j\\\\ U^+&=\\alpha U+\\sum_{j\\in J}\\tilde p_jV_j\\\\ o&=U/\\ell\\end{aligned}',
    compact: 'U^+=\\alpha U+\\tilde p V_J,\\quad o=U/\\ell',
    symbols: [{ tex: 'J', text: 'current key tile for one query' }, { tex: 'm,\\ell,U', text: 'running maximum, exponential sum and unnormalised output accumulator' }, { tex: 's_j', text: 'scaled/biased and masked score; invisible keys are −∞' }],
    derivation: [{ title: 'Change the scale without changing the normalized output', steps: ['e^{s_j-m}=e^{s_j-m^+}e^{m^+-m}', 'e^{s_j-m^+}=\\alpha e^{s_j-m}'], note: 'Start with m = −∞, ℓ = 0 and U = 0. Skip wholly masked leading tiles. This unnormalised-U convention is algebraically equivalent to Algorithm 1, whose stored O is normalized. The lesson disables dropout and compares every prefix against dense PyTorch attention.' }],
  },
  'concept:training-gradient': {
    title: 'Mean next-token loss and the backward pass',
    tex: '\\begin{aligned} p_{t,v}&=\\softmax(z_t)_v \\\\ \\mathcal L&=-\\frac1M\\sum_{t\\in\\mathrm{valid}}\\log p_{t,y_t} \\\\ \\frac{\\partial\\mathcal L}{\\partial z_{t,v}}&=\\begin{cases}(p_{t,v}-\\mathbf1[v=y_t])/M&t\\in\\mathrm{valid}\\\\0&\\text{otherwise}\\end{cases} \\\\ \\nabla_W\\mathcal L&=G^\\top X,\\quad \\nabla_b\\mathcal L=\\sum_t G_t \\end{aligned}',
    compact: '\\partial\\mathcal L/\\partial z=(p-y)/M',
    symbols: [{ tex: 'M', text: 'number of supervised token positions, excluding ignored targets' }, { tex: 'G', text: 'upstream gradient at a linear layer output' }, { tex: 'W', text: 'linear weight stored as [out, in]; output is X Wᵀ + b' }],
    derivation: [{ title: 'Softmax plus cross-entropy', steps: ['-\\log p_y=-z_y+\\log\\sum_j e^{z_j}', '\\partial(-\\log p_y)/\\partial z_v=p_v-\\mathbf1[v=y]'], note: 'The mean contributes 1/M. The training lesson checks this formula against the actual logit gradients. Masked-out attention scores have zero gradient through masked_fill; residual branches add their gradient contributions.' },
      { title: 'Shared parameters receive the sum of their uses', steps: ['\\nabla_{W_E}\\mathcal L=\\nabla_{W_E}^{\\mathrm{lookup}}\\mathcal L+\\nabla_{W_E}^{\\mathrm{head}}\\mathcal L'], note: 'When embeddings and the LM head are tied, there is one parameter matrix and one optimizer update, after accumulating both contributions.' }],
  },
  'concept:adamw': {
    title: 'AdamW moments, bias correction and decoupled decay',
    tex: '\\begin{aligned} m_t&=\\beta_1m_{t-1}+(1-\\beta_1)g_t \\\\ v_t&=\\beta_2v_{t-1}+(1-\\beta_2)g_t^2 \\\\ \\hat m_t&=m_t/(1-\\beta_1^t),\\quad\\hat v_t=v_t/(1-\\beta_2^t) \\\\ \\theta_t&=(1-\\eta\\lambda)\\theta_{t-1}-\\eta\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon} \\end{aligned}',
    compact: '\\theta_t=(1-\\eta\\lambda)\\theta_{t-1}-\\eta\\hat m_t/(\\sqrt{\\hat v_t}+\\epsilon)',
    symbols: [{ tex: 'g_t', text: 'gradient at optimizer step t; powers, roots and products are elementwise' }, { tex: '\\eta,\\lambda', text: 'learning rate and decoupled weight decay' }, { tex: '\\beta_1,\\beta_2', text: 'first- and second-moment coefficients' }],
    derivation: [{ title: 'Decay is separate from adaptive gradient scaling', steps: ['\\theta\\leftarrow(1-\\eta\\lambda)\\theta', '\\theta\\leftarrow\\theta-\\eta\\hat m/(\\sqrt{\\hat v}+\\epsilon)'], note: 'The lesson decays matrices but not norm gain or bias vectors. Its SGD option has no decay. Three consecutive updates are checked against PyTorch to exercise moment accumulation and bias correction.' }],
  },
  'concept:tensor-layout': {
    title: 'Head layout, strides and contraction',
    tex: '\\begin{aligned} Q&\\in\\R^{B\\times T\\times d},\\qquad d=H d_h \\\\ Q^{\\mathrm{heads}}_{b,h,t,c}&=Q_{b,t,h d_h+c} \\\\ S_{b,h,t,s}&=\\sum_c Q^{\\mathrm{heads}}_{b,h,t,c}K^{\\mathrm{heads}}_{b,h,s,c} \\\\ \\mathrm{offset}(i)&=o+\\sum_a i_a\\,\\mathrm{stride}_a \\end{aligned}',
    compact: 'Q^{\\mathrm{heads}}_{b,h,t,c}=Q_{b,t,h d_h+c}',
    symbols: [{ tex: 'B,T,H,d_h', text: 'batch size, token count, head count and head width' }, { tex: 'o', text: 'storage offset of the tensor view' }],
    derivation: [{ title: 'Views change indexing; a contraction computes new values', steps: ['(B,T,d)\\xrightarrow{\\mathrm{view}}(B,T,H,d_h)\\xrightarrow{\\mathrm{transpose}}(B,H,T,d_h)'],
      note: 'These view operations share storage. A view must be compatible with its strides; reshape may copy when that is impossible. cat allocates a joined result, while einsum contracts the feature axis and computes dot products.' }],
  },
  'concept:kv-cache': {
    title: 'Incremental decoding and KV-cache storage',
    tex: '\\begin{aligned} K_{\\le t}^{(\\ell)}&=\\concat(K_{<t}^{(\\ell)},k_t^{(\\ell)}) \\\\ V_{\\le t}^{(\\ell)}&=\\concat(V_{<t}^{(\\ell)},v_t^{(\\ell)}) \\\\ o_t^{(\\ell)}&=\\softmax(\\hat S_{t,\\le t}^{(\\ell)})V_{\\le t}^{(\\ell)} \\\\ \\text{cache bytes}&=2LT H_{\\mathrm{KV}}d_h b \\end{aligned}',
    compact: '\\text{cache bytes}=2LT H_{\\mathrm{KV}}d_h b',
    symbols: [
      { tex: 'L', text: 'number of decoder blocks', dims: [] },
      { tex: 'T', text: 'cached token count', dims: [] },
      { tex: 'H_{\\mathrm{KV}}', text: 'KV head count; cache stores shared heads before repetition', dims: ['Hkv'] },
      { tex: 'd_h', text: 'head width', dims: ['dh'] },
      { tex: 'b', text: 'bytes per scalar (2 for fp16/bf16; JS uses float64 values)', dims: [] },
    ],
    derivation: [{ title: 'When cached keys and values remain valid', steps: ['\\frac{\\partial k_i^{(\\ell)}}{\\partial x_j}=\\frac{\\partial v_i^{(\\ell)}}{\\partial x_j}=0,\\qquad j>i'],
      note: 'For causal or causal sliding attention, future tokens cannot change earlier hidden states. Bidirectional and incrementally built prefix masks do not satisfy this condition. Atlas retains all cached tokens up to block_size; masking limits reads, and JavaScript object overhead is excluded from the byte count.' }],
  },
};

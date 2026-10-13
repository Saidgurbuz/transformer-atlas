// Concept entries ('concept:*'): derivations and cost formulas not tied to one tensor.
export const ENTRIES = {
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

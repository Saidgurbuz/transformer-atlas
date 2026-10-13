// Concept entries ('concept:*'): derivations and cost formulas not tied to one tensor.
export const ENTRIES = {
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

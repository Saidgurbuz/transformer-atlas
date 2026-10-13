// Maths for the attention scores, mask, softmax and weighted sum.
// Entry schema (see src/core/math/index.js):
//   title, tex (display), compact (one-line, for tooltips), symbols [{tex, text, dims}],
//   derivation [{title, steps: [tex], note}], live(ctx) -> {tex, result, expected, note}
import { at3, row3, kvHead, dotTerms, listTerms, num, dot, dh } from './util.js';

export const ENTRIES = {
  'L{i}.attn.scores': {
    title: 'Attention scores',
    tex: 'S = \\Q\\K^{\\top} \\in \\R^{T\\times T}, \\qquad S_{ij} = q_i \\cdot k_j = \\sum_{c=1}^{d_h} q_{ic}\\,k_{jc}',
    compact: 'S_{ij} = q_i \\cdot k_j',
    symbols: [
      { tex: '\\Q', text: 'queries of this head (after RoPE if used)', dims: ['T', 'dh'] },
      { tex: '\\K', text: 'keys of the KV head this query head reads', dims: ['T', 'dh'] },
      { tex: 'S_{ij}', text: 'how well query i matches key j', dims: [] },
    ],
    derivation: [{
      title: 'Why a dot product',
      steps: ['q_i\\cdot k_j = \\lVert q_i\\rVert\\,\\lVert k_j\\rVert\\cos\\theta_{ij}'],
      note: 'Large when the query and key point the same way and are long; the model learns W_Q and W_K so that the right pairs align.',
    }],
    live({ trace, cfg, layer, head, row, col }) {
      const q = trace[`L${layer}.attn.${cfg.pos_encoding === 'rope' ? 'q_rot' : 'q_heads'}`];
      const kKey = cfg.pos_encoding === 'rope' ? 'k_rot' : 'k_heads';
      const k = trace[`L${layer}.attn.${kKey}`];
      const qi = row3(q, head, row), kj = row3(k, kvHead(cfg, head), col);
      const result = dot(qi, kj);
      return {
        tex: `S_{${row},${col}} = ${dotTerms(qi, kj)} = ${num(result)}`,
        result, expected: at3(trace[`L${layer}.attn.scores`], head, row, col),
      };
    },
  },

  'L{i}.attn.scaled': {
    title: 'Scaled scores',
    tex: '\\tilde S = \\frac{\\Q\\K^{\\top}}{\\sqrt{d_h}}',
    compact: '\\tilde S_{ij} = S_{ij}/\\sqrt{d_h}',
    symbols: [{ tex: 'd_h', text: 'head width (d_model / n_head)', dims: [] }],
    derivation: [{
      title: 'Why divide by √d_h',
      steps: [
        'q_c, k_c \\sim \\text{iid},\\ \\mathbb{E}=0,\\ \\operatorname{Var}=1',
        '\\operatorname{Var}(q\\cdot k) = \\sum_{c=1}^{d_h} \\operatorname{Var}(q_c k_c) = d_h',
        '\\operatorname{Var}\\!\\left(\\tfrac{q\\cdot k}{\\sqrt{d_h}}\\right) = 1',
      ],
      note: 'Without the factor, scores grow like √d_h, softmax saturates to one-hot, and its gradients vanish.',
    }],
    live({ trace, cfg, layer, head, row, col }) {
      const s = at3(trace[`L${layer}.attn.scores`], head, row, col);
      const result = s / Math.sqrt(dh(cfg));
      return { tex: `\\tilde S_{${row},${col}} = \\frac{${num(s)}}{\\sqrt{${dh(cfg)}}} = ${num(result)}`, result, expected: at3(trace[`L${layer}.attn.scaled`], head, row, col) };
    },
  },

  'L{i}.attn.masked': {
    title: 'Masked scores',
    tex: '\\begin{aligned} Z &= \\alpha \\Q\\K^{\\top} + B,\\quad \\alpha=1/\\sqrt{d_h}\\text{ (or 1 if unscaled)} \\\\ \\hat S_{ij} &= \\begin{cases} Z_{ij} & \\Mk_{ij} = 1 \\\\ -\\infty & \\Mk_{ij} = 0 \\end{cases} \\end{aligned}',
    compact: '\\hat S_{ij} = -\\infty \\text{ where } \\Mk_{ij}=0',
    symbols: [{ tex: '\\Mk', text: 'Boolean mask: 1 where query i may see key j (causal: j ≤ i)', dims: ['T', 'T'] }, { tex: 'B', text: 'ALiBi bias when enabled, otherwise zero', dims: ['T', 'T'] }],
    derivation: [{ title: 'Why −∞', steps: ['e^{-\\infty} = 0 \\;\\Rightarrow\\; \\Aw_{ij} = 0 \\text{ exactly, and } \\partial \\Aw_{ij} / \\partial \\hat S_{ij} = 0'], note: 'Masked keys get exactly zero weight and zero gradient.' }],
    live({ trace, cfg, layer, head, row, col }) {
      const pre = trace[`L${layer}.attn.${cfg.pos_encoding === 'alibi' ? 'biased' : cfg.attn_scale ? 'scaled' : 'scores'}`];
      const allowed = trace.mask.data[row * trace.mask.shape[1] + col] === 1;
      const result = allowed ? at3(pre, head, row, col) : -Infinity;
      return { tex: `\\Mk_{${row},${col}} = ${allowed ? 1 : 0} \\Rightarrow \\hat S_{${row},${col}} = ${num(result)}`, result, expected: at3(trace[`L${layer}.attn.masked`], head, row, col) };
    },
  },

  'L{i}.attn.probs': {
    title: 'Attention weights (softmax)',
    tex: '\\begin{aligned} \\Aw &= \\softmax(\\hat S) \\\\ \\Aw_{ij} &= \\frac{e^{\\hat S_{ij}}}{\\sum_{k} e^{\\hat S_{ik}}} \\end{aligned}',
    compact: '\\Aw_{ij} = \\frac{e^{\\hat S_{ij}}}{\\sum_k e^{\\hat S_{ik}}}',
    symbols: [
      { tex: '\\hat S', text: 'scores after optional scaling, optional ALiBi, and masking', dims: ['T', 'T'] },
      { tex: '\\Aw_{ij}', text: 'share of query i’s attention on key j; each row sums to 1', dims: [] },
    ],
    derivation: [
      { title: 'Stable computation', steps: ['\\softmax(z)_j = \\frac{e^{z_j - m}}{\\sum_k e^{z_k - m}},\\quad m = \\max_k z_k'], note: 'Subtracting the row max changes nothing mathematically and prevents overflow.' },
      { title: 'Gradient (the Jacobian)', steps: ['\\frac{\\partial a_j}{\\partial z_k} = a_j(\\delta_{jk} - a_k)', '\\frac{\\partial L}{\\partial z_k} = a_k\\Big(g_k - \\sum_j a_j g_j\\Big),\\quad g = \\frac{\\partial L}{\\partial a}'], note: 'This is the backward rule the JS engine implements for softmax.' },
    ],
    live({ trace, layer, head, row, col }) {
      const s = row3(trace[`L${layer}.attn.masked`], head, row);
      const allowed = s.map((v, j) => [v, j]).filter(([v]) => v !== -Infinity);
      const m = Math.max(...allowed.map(([v]) => v));
      const Z = allowed.reduce((acc, [v]) => acc + Math.exp(v - m), 0);
      const result = s[col] === -Infinity ? 0 : Math.exp(s[col] - m) / Z;
      const den = listTerms(allowed, ([v]) => `e^{${num(v, 2)}}`);
      return {
        tex: s[col] === -Infinity
          ? `\\Aw_{${row},${col}} = \\frac{e^{-\\infty}}{\\cdots} = 0`
          : `\\Aw_{${row},${col}} = \\frac{e^{${num(s[col], 2)}}}{${den}} = ${num(result)}`,
        result, expected: at3(trace[`L${layer}.attn.probs`], head, row, col),
        note: `${allowed.length} visible keys; row sums to 1.`,
      };
    },
  },

  'L{i}.attn.ctx': {
    title: 'Weighted sum of values',
    tex: '\\text{head}_h = \\Aw_h \\V_h, \\qquad o_i = \\sum_{j} \\Aw_{ij}\\, v_j',
    compact: 'o_i = \\sum_j \\Aw_{ij} v_j',
    symbols: [
      { tex: '\\Aw_h', text: 'attention weights of head h', dims: ['T', 'T'] },
      { tex: '\\V_h', text: 'values of the KV head serving head h', dims: ['T', 'dh'] },
      { tex: 'o_i', text: 'head output for query i: a convex combination of value vectors', dims: ['dh'] },
    ],
    derivation: [],
    live({ trace, cfg, layer, head, row, col }) {
      const A = row3(trace[`L${layer}.attn.probs`], head, row);
      const V = trace[`L${layer}.attn.v_heads`], g = kvHead(cfg, head);
      const vc = A.map((_, j) => at3(V, g, j, col));
      const result = dot(A, vc);
      return { tex: `o_{${row},${col}} = \\sum_j \\Aw_{${row},j}\\,v_{j,${col}} = ${dotTerms(A, vc)} = ${num(result)}`, result, expected: at3(trace[`L${layer}.attn.ctx`], head, row, col) };
    },
  },
};

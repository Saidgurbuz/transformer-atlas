// Maths for the attention projections and head layout: Q, K and V, their per-head splits,
// the GQA sharing maps, the concatenation and the output projection, plus the parameter-count
// and KV-cache concepts. Notation as in docs/MATH_GUIDE.md; entry schema as in attention.js.
import { at2, at3, row2, wrow, attnInput, dh, kvHead, dotTerms, dot, num } from './util.js';

// Bias of an nn.Linear, or null when the model has none (cfg.bias = false).
const biasOf = (model, name, o) => (model.params[name] ? model.params[name].data[o] : null);

// The three input projections, y = X W^T + b.
const PROJ = {
  q: { sym: '\\Q', lin: 'q_proj', role: 'query head' },
  k: { sym: '\\K', lin: 'k_proj', role: 'KV head' },
  v: { sym: '\\V', lin: 'v_proj', role: 'KV head' },
};

// One output cell of X W^T + b, recomputed from the attention input and the weight row.
function projectionLive(kind) {
  const { sym, lin, role } = PROJ[kind];
  return ({ trace, cfg, model, layer, row, col }) => {
    const x = row2(attnInput(trace, cfg, layer), row);                 // token row x_t
    const w = wrow(model, `blocks.${layer}.attn.${lin}.weight`, col);  // row o of W
    const b = biasOf(model, `blocks.${layer}.attn.${lin}.bias`, col);
    const result = dot(x, w) + (b ?? 0);
    const width = dh(cfg);
    return {
      tex: `${sym}_{${row},${col}} = ${dotTerms(x, w)}${b == null ? '' : ` + (${num(b)})`} = ${num(result)}`,
      result,
      expected: at2(trace[`L${layer}.attn.${kind}`], row, col),
      note: `column ${col} is ${role} ${Math.floor(col / width)}, dimension ${col % width}`,
    };
  };
}

// Per-head reshape, (S_h)_{t,c} = S_{t, h d_h + c}: a pure index map, so the source cell is read.
function headSplitLive(kind) {
  const { sym, role } = PROJ[kind], name = kind.toUpperCase();
  return ({ trace, cfg, layer, head, row, col }) => {
    const width = dh(cfg), start = head * width, j = start + col;
    const result = at2(trace[`L${layer}.attn.${kind}`], row, j);
    return {
      tex: `(${sym}_{${head}})_{${row},${col}} = ${sym}_{${row},\\,${head}\\cdot${width}+${col}} = ${sym}_{${row},${j}} = ${num(result)}`,
      result,
      expected: at3(trace[`L${layer}.attn.${kind}_heads`], head, row, col),
      note: `${role} ${head} owns columns ${start} to ${start + width - 1} of ${name}`,
    };
  };
}

// GQA sharing: query head h reads KV head g = floor(h / G), with consecutive groups (repeat_interleave).
function repeatLive(kind) {
  const { sym } = PROJ[kind];
  return ({ trace, cfg, layer, head, row, col }) => {
    const g = kvHead(cfg, head), G = cfg.n_head / cfg.n_kv_head;
    // The reference repeats the rotated keys under RoPE, so read k_rot whenever it was traced.
    const rotated = kind === 'k' ? trace[`L${layer}.attn.k_rot`] : undefined;
    const src = rotated ?? trace[`L${layer}.attn.${kind}_heads`];
    const result = at3(src, g, row, col);
    // Under RoPE the copied tensor is the rotated keys K^rot, otherwise the projection K.
    const copied = rotated ? `${sym}^{\\mathrm{rot}}` : sym;
    return {
      tex: `${sym}^{\\mathrm{rep}}_{${head},${row},${col}} = ${copied}_{${g},${row},${col}} = ${num(result)}`,
      result,
      expected: at3(trace[`L${layer}.attn.${kind}_rep`], head, row, col),
      note: `query head ${head} reads KV head ${g} (group of ${G})${rotated ? ', after RoPE' : ''}`,
    };
  };
}

export const ENTRIES = {
  'L{i}.attn.q': {
    title: 'Queries Q (projection)',
    tex: '\\Q = X\\,\\Wq^{\\top} + b_Q, \\qquad \\Q_{t,o} = \\sum_{c=1}^{d} X_{t,c}\\,\\Wq_{o,c} + b_{Q,o}',
    compact: '\\Q_{t,o} = \\sum_{c} X_{t,c}\\,\\Wq_{o,c} + b_{Q,o}',
    symbols: [
      { tex: '\\Q', text: 'queries before RoPE: one row per token, H heads of width d_h side by side (under RoPE, attention reads trace key q_rot)', dims: ['T', 'H·dh'] },
      { tex: 'X', text: 'attention input: LN(x) for pre-LN, the block input for post-LN', dims: ['T', 'd'] },
      { tex: '\\Wq', text: 'query weight, stored as nn.Linear [out, in]', dims: ['H·dh', 'd'] },
      { tex: 'b_Q', text: 'query bias (absent when the model has no biases)', dims: ['H·dh'] },
    ],
    derivation: [
      {
        title: 'One matrix product for every token',
        steps: ['\\underbrace{\\Q}_{T\\times Hd_h} = \\underbrace{X}_{T\\times d}\\;\\underbrace{\\Wq^{\\top}}_{d\\times Hd_h} + \\underbrace{b_Q}_{\\text{added to each row}}'],
        note: 'Each query dimension is a learned linear read of the token vector. Vaswani et al. (2017) write X W^Q with W^Q of shape d × d_k; nn.Linear stores the transpose, so the code computes X W^⊤.',
      },
      {
        title: 'What one query head can match',
        steps: [
          '\\Q_h\\,\\K_g^{\\top} = X\\,\\big(\\Wq^{(h)}\\big)^{\\top}\\,\\Wk^{(g)}\\,X^{\\top}',
          '\\operatorname{rank}\\big(\\big(\\Wq^{(h)}\\big)^{\\top}\\Wk^{(g)}\\big) \\le d_h',
        ],
        note: 'W_Q^(h) is the d_h-row block of W_Q for head h. Ignoring biases and RoPE, a head scores token pairs with a d × d bilinear form of rank at most d_h (the QK circuit of Elhage et al., 2021), so each head compares tokens inside a d_h-dimensional subspace.',
      },
    ],
    live: projectionLive('q'),
  },

  'L{i}.attn.k': {
    title: 'Keys K (projection)',
    tex: '\\K = X\\,\\Wk^{\\top} + b_K, \\qquad \\K_{t,o} = \\sum_{c=1}^{d} X_{t,c}\\,\\Wk_{o,c} + b_{K,o}',
    compact: '\\K_{t,o} = \\sum_{c} X_{t,c}\\,\\Wk_{o,c} + b_{K,o}',
    symbols: [
      { tex: '\\K', text: 'keys before RoPE: one row per token, H_kv KV heads of width d_h (under RoPE, attention reads trace key k_rot)', dims: ['T', 'Hkv·dh'] },
      { tex: 'X', text: 'the same attention input as for Q', dims: ['T', 'd'] },
      { tex: '\\Wk', text: 'key weight, stored as nn.Linear [out, in]', dims: ['Hkv·dh', 'd'] },
      { tex: 'b_K', text: 'key bias', dims: ['Hkv·dh'] },
    ],
    derivation: [
      {
        title: 'Fewer key columns under GQA',
        steps: [
          '\\K \\in \\R^{T\\times H_{kv}d_h} = \\R^{T\\times \\frac{H_{kv}}{H}\\,d}',
          'H_{kv}=H:\\ \\K\\in\\R^{T\\times d}\\qquad H_{kv}=1:\\ \\K\\in\\R^{T\\times d_h}',
        ],
        note: 'The key width is H_kv·d_h, not H·d_h, so W_K is smaller and each group of query heads reads the same key columns. Only K and V are cached: a decoding step computes the new token’s q, k and v, appends k and v to the cache, and attends over all cached keys and values.',
      },
    ],
    live: projectionLive('k'),
  },

  'L{i}.attn.v': {
    title: 'Values V (projection)',
    tex: '\\V = X\\,\\Wv^{\\top} + b_V, \\qquad \\V_{t,o} = \\sum_{c=1}^{d} X_{t,c}\\,\\Wv_{o,c} + b_{V,o}',
    compact: '\\V_{t,o} = \\sum_{c} X_{t,c}\\,\\Wv_{o,c} + b_{V,o}',
    symbols: [
      { tex: '\\V', text: 'values: what a token hands over when it is attended to', dims: ['T', 'Hkv·dh'] },
      { tex: 'X', text: 'the same attention input as for Q and K', dims: ['T', 'd'] },
      { tex: '\\Wv', text: 'value weight, stored as nn.Linear [out, in]', dims: ['Hkv·dh', 'd'] },
      { tex: 'b_V', text: 'value bias', dims: ['Hkv·dh'] },
    ],
    derivation: [
      {
        title: 'What a head writes back',
        steps: [
          'o_i = \\sum_{j} \\Aw_{ij}\\,v_j, \\qquad \\sum_j \\Aw_{ij} = 1',
          '\\operatorname{rank}\\big(\\Wo^{(h)}\\,\\Wv^{(g)}\\big) \\le d_h',
        ],
        note: 'Each output is a weighted average of value rows, so a token’s value is copied into every position that attends to it. The OV circuit W_O^(h) W_V^(g) carries the attended residual vector into the stream along at most d_h directions (ignoring LayerNorm and biases; Elhage et al., 2021).',
      },
    ],
    live: projectionLive('v'),
  },

  'L{i}.attn.q_heads': {
    title: 'Queries split into heads',
    tex: '(\\Q_h)_{t,c} = \\Q_{t,\\;h\\,d_h + c}, \\qquad \\Q = \\big[\\,\\Q_0 \\mid \\Q_1 \\mid \\cdots \\mid \\Q_{H-1}\\,\\big]',
    compact: '(\\Q_h)_{t,c} = \\Q_{t,\\,h\\,d_h + c}',
    symbols: [
      { tex: '\\Q_h', text: 'queries of head h before RoPE: a T × d_h block of Q', dims: ['T', 'dh'] },
      { tex: 'h', text: 'query head index, 0 to H−1', dims: [] },
      { tex: 'c', text: 'position inside the head, 0 to d_h−1', dims: ['dh'] },
    ],
    derivation: [
      {
        title: 'Reshape, then transpose',
        steps: ['\\Q \\in \\R^{T\\times Hd_h} \\;\\longmapsto\\; \\Q^{\\text{heads}} \\in \\R^{H\\times T\\times d_h}'],
        note: 'The code calls view(B, T, H, d_h) and then transpose(1, 2): no arithmetic happens, only a relabelling of the same numbers. Afterwards each head is its own T × d_h matrix, so all heads run as one batched product.',
      },
      {
        title: 'Same budget, H attention patterns',
        steps: [
          'H\\,d_h = d \\;\\Rightarrow\\; \\text{params}(\\Wq) = H\\,d_h\\cdot d = d^{2}',
          'H\\cdot T^{2}\\,d_h = T^{2}\\,d \\quad\\text{score multiply-adds, the same as one head of width } d',
        ],
        note: 'Splitting costs no extra parameters or multiply-adds. The heads together use the weights of one wide head, but each gets its own T × T score matrix, so the model can attend in H different subspaces at once.',
      },
    ],
    live: headSplitLive('q'),
  },

  'L{i}.attn.k_heads': {
    title: 'Keys split into KV heads',
    tex: '(\\K_g)_{t,c} = \\K_{t,\\;g\\,d_h + c}, \\qquad \\K = \\big[\\,\\K_0 \\mid \\cdots \\mid \\K_{H_{kv}-1}\\,\\big]',
    compact: '(\\K_g)_{t,c} = \\K_{t,\\,g\\,d_h + c}',
    symbols: [
      { tex: '\\K_g', text: 'keys of KV head g before RoPE: a T × d_h block of K', dims: ['T', 'dh'] },
      { tex: 'g', text: 'KV head index, 0 to H_kv−1', dims: [] },
    ],
    derivation: [
      {
        title: 'Only H_kv blocks exist',
        steps: ['\\K \\in \\R^{T\\times H_{kv}d_h} \\;\\longmapsto\\; \\K^{\\text{heads}} \\in \\R^{H_{kv}\\times T\\times d_h}'],
        note: 'The same reshape as Q, over fewer blocks. This is where GQA saves on the key side: H_kv blocks are stored, and each block is read by G = H / H_kv query heads.',
      },
    ],
    live: headSplitLive('k'),
  },

  'L{i}.attn.v_heads': {
    title: 'Values split into KV heads',
    tex: '(\\V_g)_{t,c} = \\V_{t,\\;g\\,d_h + c}, \\qquad \\V = \\big[\\,\\V_0 \\mid \\cdots \\mid \\V_{H_{kv}-1}\\,\\big]',
    compact: '(\\V_g)_{t,c} = \\V_{t,\\,g\\,d_h + c}',
    symbols: [
      { tex: '\\V_g', text: 'values of KV head g: a T × d_h block of V', dims: ['T', 'dh'] },
      { tex: 'g', text: 'KV head index, 0 to H_kv−1', dims: [] },
    ],
    derivation: [
      {
        title: 'Values take the same route as keys',
        steps: ['\\V \\in \\R^{T\\times H_{kv}d_h} \\;\\longmapsto\\; \\V^{\\text{heads}} \\in \\R^{H_{kv}\\times T\\times d_h}'],
        note: 'Same reshape as K. RoPE rotates only Q and K, so v_heads feeds the value path unchanged, and v_rep is a copy of it.',
      },
    ],
    live: headSplitLive('v'),
  },

  'L{i}.attn.k_rep': {
    title: 'Keys shared by each query head',
    tex: '\\K^{\\mathrm{rep}}_{h} = \\begin{cases} \\K^{\\mathrm{rot}}_{g(h)} & \\text{if RoPE is used} \\\\ \\K_{g(h)} & \\text{otherwise} \\end{cases}, \\qquad g(h) = \\left\\lfloor \\frac{h}{G} \\right\\rfloor, \\qquad G = \\frac{H}{H_{kv}}',
    compact: '\\K^{\\mathrm{rep}}_h = \\K^{\\mathrm{rot}}_{g(h)} \\text{ if RoPE, else } \\K_{g(h)}, \\quad g(h) = \\lfloor h/G \\rfloor',
    symbols: [
      { tex: '\\K^{\\mathrm{rep}}_h', text: 'keys that query head h reads: keys after RoPE if used, of KV head g(h), repeated', dims: ['T', 'dh'] },
      { tex: '\\K^{\\mathrm{rot}}_g', text: 'keys of KV head g after RoPE (trace key k_rot, only with RoPE)', dims: ['T', 'dh'] },
      { tex: '\\K_g', text: 'keys of KV head g before RoPE (trace key k_heads)', dims: ['T', 'dh'] },
      { tex: 'g(h)', text: 'KV head that serves query head h', dims: [] },
      { tex: 'G', text: 'query heads per KV head, H / H_kv (1 for MHA)', dims: [] },
    ],
    derivation: [
      {
        title: 'Consecutive groups, not interleaved',
        steps: [
          'H = 8,\\;\\; H_{kv} = 2 \\;\\Rightarrow\\; G = 4:\\qquad (h=0,1,2,3)\\mapsto g=0,\\qquad (h=4,5,6,7)\\mapsto g=1',
        ],
        note: 'The reference uses repeat_interleave, which copies each KV head G times in a row; a plain repeat (tile) would give 0, 1, 0, 1, … and pair query heads with the wrong keys. The copy exists only for the matmul, while the cache stores H_kv heads.',
      },
      {
        title: 'Rotate, then copy',
        steps: [
          '\\K^{\\mathrm{rot}}_{g} = \\text{RoPE}(\\K_g) \\quad \\text{(each row } t \\text{ is rotated by its position, only under RoPE)}',
          '\\K^{\\mathrm{rep}}_{h} = \\K^{\\mathrm{rot}}_{g(h)} \\quad \\text{(under RoPE; without RoPE, } \\K^{\\mathrm{rep}}_{h} = \\K_{g(h)}\\text{)}',
        ],
        note: 'The reference applies RoPE to k before it repeats the heads, so every query head in a group receives the same rotated keys. Without RoPE the same copy is made from K itself.',
      },
    ],
    live: repeatLive('k'),
  },

  'L{i}.attn.v_rep': {
    title: 'Values shared by each query head',
    tex: '\\V^{\\mathrm{rep}}_{h} = \\V_{g(h)}, \\qquad g(h) = \\left\\lfloor \\frac{h}{G} \\right\\rfloor',
    compact: '\\V^{\\mathrm{rep}}_h = \\V_{g(h)}',
    symbols: [
      { tex: '\\V^{\\mathrm{rep}}_h', text: 'values that query head h reads (its KV head, repeated)', dims: ['T', 'dh'] },
      { tex: 'g(h)', text: 'KV head that serves query head h', dims: [] },
    ],
    derivation: [
      {
        title: 'One map for keys and values',
        steps: ['\\text{head}_h = \\Aw_h\\,\\V^{\\mathrm{rep}}_h = \\Aw_h\\,\\V_{g(h)}'],
        note: 'Query head h scores against K_g(h) and reads V_g(h), so the whole head is served by one KV head. The map comes from the head grouping, not from the data, and it is the same for keys and values.',
      },
    ],
    live: repeatLive('v'),
  },

  'L{i}.attn.concat': {
    title: 'Heads concatenated',
    tex: '\\concat_{t,\\;h\\,d_h + c} = (\\text{head}_h)_{t,c}, \\qquad \\concat = \\big[\\,\\text{head}_0 \\mid \\cdots \\mid \\text{head}_{H-1}\\,\\big]',
    compact: '\\concat_{t,\\,h\\,d_h + c} = (\\text{head}_h)_{t,c}',
    symbols: [
      { tex: '\\concat', text: 'all heads side by side, one row per token', dims: ['T', 'H·dh'] },
      { tex: '\\text{head}_h', text: 'output of head h, A_h V_g(h) (trace key ctx)', dims: ['T', 'dh'] },
    ],
    derivation: [
      {
        title: 'Side by side, no arithmetic',
        steps: ['\\concat \\in \\R^{T\\times Hd_h}, \\qquad \\text{head}_h \\text{ occupies columns } h\\,d_h \\text{ to } (h+1)\\,d_h - 1'],
        note: 'Concatenation only changes the layout: the transpose and reshape move numbers. The output projection that follows is the first place where the heads mix.',
      },
    ],
    live({ trace, cfg, layer, row, col }) {
      const width = dh(cfg), h = Math.floor(col / width), c = col % width;
      const result = at3(trace[`L${layer}.attn.ctx`], h, row, c);
      return {
        tex: `\\concat_{${row},${col}} = (\\text{head}_{${h}})_{${row},${c}} = ${num(result)}`,
        result,
        expected: at2(trace[`L${layer}.attn.concat`], row, col),
        note: `column ${col} = ${h}·${width} + ${c}, so it is dimension ${c} of head ${h}`,
      };
    },
  },

  'L{i}.attn.out': {
    title: 'Attention output (O projection)',
    tex: '\\text{out} = \\concat\\,\\Wo^{\\top} + b_O, \\qquad \\text{out}_{t,o} = \\sum_{j=1}^{Hd_h} \\concat_{t,j}\\,\\Wo_{o,j} + b_{O,o}',
    compact: '\\text{out}_{t,o} = \\sum_{j} \\concat_{t,j}\\,\\Wo_{o,j} + b_{O,o}',
    symbols: [
      { tex: '\\text{out}', text: 'attention output, written back to width d', dims: ['T', 'd'] },
      { tex: '\\Wo', text: 'output weight, stored as nn.Linear [out, in]', dims: ['d', 'H·dh'] },
      { tex: 'b_O', text: 'output bias', dims: ['d'] },
    ],
    derivation: [
      {
        title: 'Each head writes its own contribution',
        steps: [
          '\\text{out}_t = \\sum_{h=0}^{H-1} \\text{head}_{h,t}\\,\\big(\\Wo^{(h)}\\big)^{\\top} + b_O',
          '\\Wo^{(h)} = \\Wo\\big[\\,:,\\; h\\,d_h : (h+1)\\,d_h\\,\\big] \\in \\R^{d\\times d_h}',
        ],
        note: 'This is block multiplication: the columns of W_O are cut into H blocks of width d_h, one per head, and each head adds a d-dimensional vector to the residual stream. The output is the sum of these vectors plus the bias, and each write has rank at most d_h (the OV circuit of Elhage et al., 2021).',
      },
    ],
    live({ trace, cfg, model, layer, row, col }) {
      const x = row2(trace[`L${layer}.attn.concat`], row);
      const w = wrow(model, `blocks.${layer}.attn.o_proj.weight`, col);
      const b = biasOf(model, `blocks.${layer}.attn.o_proj.bias`, col);
      const result = dot(x, w) + (b ?? 0);
      // The same dot product grouped by head: head h contributes sum_c head_{h,t,c} W_O[o, h d_h + c].
      const width = dh(cfg);
      const parts = Array.from({ length: cfg.n_head }, (_, h) =>
        dot(x.slice(h * width, (h + 1) * width), w.slice(h * width, (h + 1) * width)));
      const top = parts.reduce((best, v, h) => (Math.abs(v) > Math.abs(parts[best]) ? h : best), 0);
      return {
        tex: `\\text{out}_{${row},${col}} = ${dotTerms(x, w)}${b == null ? '' : ` + (${num(b)})`} = ${num(result)}`,
        result,
        expected: at2(trace[`L${layer}.attn.out`], row, col),
        note: `largest single-head write is head ${top} (${num(parts[top], 2)}); the per-head sums add to the dot product above`,
      };
    },
  },

  'concept:mha-params': {
    title: 'Attention parameter count',
    tex: '\\begin{aligned} P_{\\text{attn}} &= \\underbrace{d\\,H\\,d_h}_{W_Q} + \\underbrace{2\\,d\\,H_{kv}\\,d_h}_{W_K,\\;W_V} + \\underbrace{H\\,d_h\\,d}_{W_O} \\\\ &= 2d^{2}\\Big(1+\\frac{H_{kv}}{H}\\Big) \\quad\\text{(plus biases } 2d\\Big(1+\\frac{H_{kv}}{H}\\Big)\\text{)} \\end{aligned}',
    compact: 'P_{\\text{attn}} = 2d^{2}\\,(1 + H_{kv}/H) \\;\\text{(weights only)}',
    symbols: [
      { tex: 'd', text: 'model width', dims: ['d'] },
      { tex: 'H', text: 'query heads', dims: ['H'] },
      { tex: 'H_{kv}', text: 'KV heads: 1 for multi-query, H for multi-head, in between for grouped', dims: ['Hkv'] },
      { tex: 'd_h', text: 'head width, d / H', dims: ['dh'] },
      { tex: 'P_{\\text{attn}}', text: 'attention weights (and biases) of one layer', dims: [] },
    ],
    derivation: [
      {
        title: 'Count each matrix',
        steps: [
          '\\text{params}(\\Wq) = (H\\,d_h)\\cdot d = d\\,H\\,d_h, \\qquad \\text{params}(\\Wk) = \\text{params}(\\Wv) = (H_{kv}\\,d_h)\\cdot d',
          '\\text{params}(\\Wo) = d\\cdot(H\\,d_h) = H\\,d_h\\,d',
        ],
        note: 'An nn.Linear(in, out) stores out × in weights. Under MHA, W_K and W_V match W_Q; under GQA only they shrink.',
      },
      {
        title: 'Use H·d_h = d',
        steps: [
          'P_{\\text{attn}} = d\\,H\\,d_h + 2\\,d\\,H_{kv}\\,d_h + H\\,d_h\\,d = 2d^{2} + 2d^{2}\\,\\frac{H_{kv}}{H} = 2d^{2}\\Big(1+\\frac{H_{kv}}{H}\\Big)',
        ],
        note: 'Since d_h = d / H, the head count cancels out of the total, and only d and the KV ratio H_kv / H matter. MHA gives 4d²; MQA gives 2d²(1 + 1/H).',
      },
      {
        title: 'Biases',
        steps: [
          '\\underbrace{H\\,d_h}_{b_Q} + \\underbrace{2\\,H_{kv}\\,d_h}_{b_K,\\;b_V} + \\underbrace{d}_{b_O} = 2d\\Big(1+\\frac{H_{kv}}{H}\\Big)',
        ],
        note: 'With biases and MHA the total is 4d² + 4d. The biases are exactly 1/d of the weight count, whatever H_kv is.',
      },
      {
        title: 'Worked example',
        steps: [
          'd = 64,\\; H = 4,\\; H_{kv} = 2:\\qquad 2\\cdot 64^{2}\\cdot\\tfrac{3}{2} = 12\\,288 \\text{ weights},\\qquad 2\\cdot 64\\cdot\\tfrac{3}{2} = 192 \\text{ biases}',
          'd = 64,\\; H = H_{kv} = 4:\\qquad 4\\cdot 64^{2} = 16\\,384 \\text{ weights},\\qquad 16\\,640 \\text{ with biases}',
        ],
        note: 'Vaswani et al. (2017) use h = 8 heads with d_k = d_v = 64 at d_model = 512: the same four matrices, here with H_kv = H.',
      },
    ],
  },

  'concept:gqa': {
    title: 'Grouped-query attention and the KV cache',
    tex: '\\begin{aligned} G &= \\frac{H}{H_{kv}} \\\\ M_{\\text{KV}}(T) &= \\underbrace{2}_{K,\\,V}\\; L\\; H_{kv}\\; d_h\\; T\\; s_{\\text{el}}\\ \\text{bytes} \\\\ \\frac{M^{\\text{GQA}}_{\\text{KV}}}{M^{\\text{MHA}}_{\\text{KV}}} &= \\frac{H_{kv}}{H} = \\frac{1}{G} \\end{aligned}',
    compact: 'M_{\\text{KV}}(T) = 2\\,L\\,H_{kv}\\,d_h\\,T\\,s_{\\text{el}}',
    symbols: [
      { tex: 'H_{kv}', text: 'KV heads: 1 is multi-query, H is multi-head, between is grouped', dims: ['Hkv'] },
      { tex: 'G', text: 'query heads that share one KV head, H / H_kv', dims: [] },
      { tex: 'L', text: 'number of layers (each layer keeps its own cache)', dims: [] },
      { tex: 'T', text: 'cached tokens (context length)', dims: ['T'] },
      { tex: 's_{\\text{el}}', text: 'bytes per cached element (2 for fp16 or bf16)', dims: [] },
    ],
    derivation: [
      {
        title: 'Sharing: one family of layouts',
        steps: [
          'g(h) = \\left\\lfloor \\frac{h}{G} \\right\\rfloor, \\qquad 1 \\le H_{kv} \\le H, \\quad H_{kv} \\mid H',
          'H_{kv}=H:\\ \\text{MHA}\\qquad H_{kv}=1:\\ \\text{MQA}\\qquad 1<H_{kv}<H:\\ \\text{GQA}',
        ],
        note: 'Shazeer (2019) proposed multi-query attention, with one K/V head shared by all query heads. Ainslie et al. (2023) generalised this to groups: they convert an MHA checkpoint by mean-pooling each group’s K/V heads, then uptrain briefly.',
      },
      {
        title: 'Bytes per token',
        steps: [
          'K \\text{ and } V \\text{ per token, per layer: } H_{kv}\\,d_h \\text{ values each}',
          '\\text{bytes per token} = 2\\,L\\,H_{kv}\\,d_h\\,s_{\\text{el}}',
          'L=80,\\; H_{kv}=8,\\; d_h=128,\\; s_{\\text{el}}=2:\\qquad 2\\cdot 80\\cdot 8\\cdot 128\\cdot 2 = 327\\,680 \\text{ bytes} \\approx 320 \\text{ KiB}',
        ],
        note: 'The cache stores K and V for every past token in every layer, so it grows linearly with T. It is a per-token cost, not a per-model one.',
      },
      {
        title: 'Cache size at a hypothetical 8K tokens',
        steps: [
          'M_{\\text{KV}}(8192) = 327\\,680 \\cdot 8192 = 2.5\\ \\text{GiB}, \\qquad \\text{with } H_{kv}=64 \\text{ (MHA)}: \\; 20\\ \\text{GiB}',
        ],
        note: 'The 8K length is a hypothetical cache for one sequence: Touvron et al. (2023) pretrain Llama 2 with a 4K context (Table 1) and use GQA for the 34B and 70B models (App. A.2.1). The 80 layers, 64 query heads and 8 KV heads (d_h = 128) are the released 70B configuration rather than figures from the paper, and caching 8 KV heads instead of 64 makes the GQA cache one eighth the size of an MHA cache.',
      },
    ],
  },
};

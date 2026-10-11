// Maths for the output head: the logits, next-token probabilities and the loss, then the
// concept entries behind training and inference cost (cross-entropy gradient, perplexity,
// sampling, KV cache, FLOPs, attention complexity). Matches reference/model.py: logits are
// x W_U^T with no bias, probs is a softmax over the vocabulary, and the loss is the mean
// cross-entropy over positions whose target is not -1.
// Entry schema (see src/core/math/index.js):
//   title, tex (display), compact (one line, for tooltips), symbols [{tex, text, dims}],
//   derivation [{title, steps: [tex], note}], live(ctx) -> {tex, result, expected, note}
// Only tex fields are typeset. Symbol text and notes are plain text, so they contain no LaTeX.
import { at2, row2, wrow, dot, dotTerms, listTerms, num } from './util.js';

// A vocabulary entry inside \text{}: printable characters are quoted and escaped; whitespace
// tokens get a name, because KaTeX has no glyph metrics for the visible-space symbol.
const NAMED = { '␣': 'space', ' ': 'space', '\n': 'newline', '\t': 'tab' };
const TEXT_ESC = {
  '\\': '\\textbackslash{}', '{': '\\{', '}': '\\}', '$': '\\$', '&': '\\&', '#': '\\#',
  '^': '\\textasciicircum{}', '_': '\\_', '%': '\\%', '~': '\\textasciitilde{}',
};
const vocabTeX = (ch) => NAMED[ch] ?? `"${[...String(ch)].map((c) => TEXT_ESC[c] ?? c).join('')}"`;

export const ENTRIES = {
  'logits': {
    title: 'Logits (unembedding)',
    tex: 'z_{t,v} = \\sum_{c=1}^{d} \\xres_{t,c}\\,\\WU_{v,c}, \\qquad z_t = \\xres_t\\,\\WU^{\\top} \\in \\R^{V}',
    compact: 'z_{t,v} = \\sum_{c} \\xres_{t,c}\\,\\WU_{v,c}',
    symbols: [
      { tex: '\\xres', text: 'final residual state of each token: after ln_f for pre-LN, the last block output for post-LN', dims: ['T', 'd'] },
      { tex: '\\WU', text: 'unembedding matrix, one row per vocabulary entry: lm_head.weight, or wte.weight when tied (no bias)', dims: ['V', 'd'] },
      { tex: 'z', text: 'logits: one unnormalised score per vocabulary entry, for every position', dims: ['T', 'V'] },
    ],
    derivation: [
      {
        title: 'One dot product per vocabulary entry',
        steps: ['z_{t,v} = \\langle \\xres_t,\\ \\WU_{v} \\rangle', 'z_t = \\xres_t\\,\\WU^{\\top}'],
        note: 'Each logit measures how well the final state of position t lines up with the output direction of token v. The softmax entry turns these scores into probabilities.',
      },
      {
        title: 'Tied weights',
        steps: ['\\WU = \\WE \\;\\Rightarrow\\; z_{t,v} = \\xres_t \\cdot \\WE_{v}'],
        note: 'With tied weights, the vector that embeds token v on the way in also scores it on the way out. This saves V·d parameters and ties the input and output geometry together.',
      },
    ],
    live({ trace, cfg, model, row, col, vocab }) {
      const x = cfg.norm_placement === 'pre' ? trace.lnf : trace[`L${cfg.n_layer - 1}.resid_out`];
      const xt = row2(x, row), wv = wrow(model, cfg.tie_weights ? 'wte.weight' : 'lm_head.weight', col);
      const result = dot(xt, wv);
      return {
        tex: `z_{${row},${col}} = ${dotTerms(xt, wv)} = ${num(result)} \\qquad \\text{column ${col}: ${vocabTeX(vocab(col))}}`,
        result, expected: at2(trace.logits, row, col),
        note: `Dot product of position ${row}’s final state with the output vector of column ${col}; the next entry turns row ${row} into probabilities.`,
      };
    },
  },

  'probs': {
    title: 'Next-token probabilities (softmax)',
    tex: 'p_{t,v} = \\frac{e^{z_{t,v}}}{\\sum_{k=1}^{V} e^{z_{t,k}}} = \\softmax(z_t)_v, \\qquad p_{t,v} > 0, \\quad \\sum_{v=1}^{V} p_{t,v} = 1',
    compact: 'p_{t,v} = \\frac{e^{z_{t,v}}}{\\sum_k e^{z_{t,k}}}',
    symbols: [
      { tex: 'z', text: 'logits from the previous entry', dims: ['T', 'V'] },
      { tex: 'p', text: 'probability that the token at position t+1 is vocabulary entry v; each row sums to 1. A genuine next-token prediction only where row t cannot see token t+1, as under causal or sliding masks', dims: ['T', 'V'] },
    ],
    derivation: [
      {
        title: 'Stable computation (max subtraction)',
        steps: ['m_t = \\max_k z_{t,k}', 'p_{t,v} = \\frac{e^{z_{t,v}-m_t}}{\\sum_k e^{z_{t,k}-m_t}}'],
        note: 'Subtracting the row maximum multiplies numerator and denominator by the same factor exp(−m_t), so the result is unchanged. Every exponent is now at most 0, nothing overflows, and the largest term is exactly 1.',
      },
      {
        title: 'Ratios depend only on logit gaps',
        steps: ['\\frac{p_{t,u}}{p_{t,v}} = e^{\\,z_{t,u}-z_{t,v}}'],
        note: 'A gap of 1 nat multiplies the odds by e ≈ 2.72. Every probability stays positive, so the softmax is a smooth, always differentiable “soft argmax”.',
      },
    ],
    live({ trace, row, col, vocab }) {
      const z = row2(trace.logits, row);
      const m = z.reduce((a, b) => Math.max(a, b), -Infinity);
      const gaps = z.map((v) => v - m).sort((a, b) => b - a);   // largest first: e^0 = 1 leads
      const Z = gaps.reduce((s, g) => s + Math.exp(g), 0);
      const result = Math.exp(z[col] - m) / Z;
      // Can row t see token t+1? That depends on the mask that was applied (trace.mask: 1 = visible).
      const T = trace.mask.shape[1];
      const seesNext = row + 1 < T && trace.mask.data[row * T + row + 1] === 1;
      const role = row + 1 === T
        ? `Row ${row} is the last position, so no token follows it in the input.`
        : seesNext
          ? `Under this mask row ${row} can see token ${row + 1}, so this row is not a next-token prediction.`
          : `Row ${row} predicts token ${row + 1}, which this mask hides from it.`;
      return {
        tex: `m_{${row}} = \\max_k z_{${row},k} = ${num(m)} \\qquad p_{${row},${col}} = \\frac{e^{${num(z[col] - m, 2)}}}{${listTerms(gaps, (g) => `e^{${num(g, 2)}}`)}} = ${num(result)}`,
        result, expected: at2(trace.probs, row, col),
        note: `${role} Column ${col} is “${vocab(col)}”. The logits are shifted by the row maximum, so the largest term is e^0 = 1.`,
      };
    },
  },

  'loss': {
    title: 'Cross-entropy loss',
    tex: '\\mathcal{L} = \\CE(z, y) = -\\frac{1}{|\\mathcal{P}|}\\sum_{t\\in\\mathcal{P}} \\log p_{t,\\,y_t} = \\frac{1}{|\\mathcal{P}|}\\sum_{t\\in\\mathcal{P}}\\Big(\\log\\sum_{k=1}^{V} e^{z_{t,k}} - z_{t,y_t}\\Big)',
    compact: '\\mathcal{L} = -\\frac{1}{|\\mathcal{P}|}\\sum_{t\\in\\mathcal{P}} \\log p_{t,y_t}',
    symbols: [
      { tex: '\\mathcal{L}', text: 'mean loss over the scored positions, in nats (natural log)', dims: [] },
      { tex: '\\mathcal{P}', text: 'positions with a target; targets equal to −1 are ignored', dims: ['T'] },
      { tex: '|\\mathcal{P}|', text: 'number of scored positions', dims: [] },
      { tex: 'y_t', text: 'target at position t: the token that follows position t in the training text', dims: ['T'] },
    ],
    derivation: [
      {
        title: 'Log-sum-exp form',
        steps: ['-\\log p_{t,y} = -z_{t,y} + \\log\\sum_{k} e^{z_{t,k}}'],
        note: 'The loss is the log-normaliser minus the score of the true token. It is never negative, and it is zero only when the true token has probability 1.',
      },
      {
        title: 'Maximum likelihood',
        steps: ['\\prod_{t\\in\\mathcal{P}} p_{t,y_t} = \\exp\\big(-|\\mathcal{P}|\\,\\mathcal{L}\\big)'],
        note: 'Minimising the mean cross-entropy maximises the likelihood of the training text, which is why it is the standard language-model objective. That reading needs each p_t to be a next-token prediction, which holds only where row t cannot see token t+1 (causal or sliding masks). Under mask “none”, and within the prefix of mask “prefix”, a row sees later tokens, so the product is a loss but not a likelihood of the text.',
      },
      {
        title: 'Uniform baseline',
        steps: ['z \\approx 0 \\;\\Rightarrow\\; p_{t,v} \\approx \\tfrac{1}{V} \\;\\Rightarrow\\; \\mathcal{L} \\approx \\log V'],
        note: 'A model that knows nothing loses ln V nats per position (ln 32 ≈ 3.47 for the 32-entry default vocabulary). A fresh random initialisation starts close to this value, and training should drive the loss well below it.',
      },
    ],
  },

  'concept:cross-entropy-gradient': {
    title: 'Gradient of the loss with respect to the logits',
    tex: '\\nabla_{z_t}\\mathcal{L} = \\frac{p_t - \\operatorname{onehot}(y_t)}{|\\mathcal{P}|}\\ \\ (t \\in \\mathcal{P}), \\qquad \\frac{\\partial \\mathcal{L}}{\\partial z_{t,c}} = \\frac{p_{t,c} - \\mathbf{1}[c = y_t]}{|\\mathcal{P}|}',
    compact: '\\nabla_{z_t}\\mathcal{L} = \\frac{p_t - \\operatorname{onehot}(y_t)}{|\\mathcal{P}|}',
    symbols: [
      { tex: 'p_t', text: 'predicted distribution at position t (softmax of z_t)', dims: ['V'] },
      { tex: '\\operatorname{onehot}(y_t)', text: 'one-hot vector of the target token at position t', dims: ['V'] },
      { tex: '|\\mathcal{P}|', text: 'number of scored positions; ignored positions get zero gradient', dims: [] },
      { tex: '\\eta', text: 'learning rate, used in the update below', dims: [] },
    ],
    derivation: [
      {
        title: 'One position',
        steps: [
          '\\ell_t = -\\log p_{t,y_t} = -z_{t,y_t} + \\log\\sum_{k} e^{z_{t,k}}',
          '\\frac{\\partial \\ell_t}{\\partial z_{t,c}} = -\\mathbf{1}[c = y_t] + \\frac{e^{z_{t,c}}}{\\sum_{k} e^{z_{t,k}}} = p_{t,c} - \\mathbf{1}[c = y_t]',
        ],
        note: 'Differentiating the log-normaliser gives back the softmax itself, so the two terms collapse to the predicted distribution minus the one-hot target.',
      },
      {
        title: 'Mean over scored positions',
        steps: ['\\frac{\\partial \\mathcal{L}}{\\partial z_{t,c}} = \\frac{1}{|\\mathcal{P}|}\\,\\frac{\\partial \\ell_t}{\\partial z_{t,c}}\\ \\ (t \\in \\mathcal{P}), \\qquad 0\\ \\ (t \\notin \\mathcal{P})'],
        note: 'The 1/|P| comes from the mean reduction. Every entry lies in [−1/|P|, 1/|P|], so the gradient scale does not grow with the number of scored positions.',
      },
      {
        title: 'What a gradient step does',
        steps: ['\\Delta z_{t,c} = -\\eta\\,\\frac{p_{t,c} - \\mathbf{1}[c = y_t]}{|\\mathcal{P}|}'],
        note: 'This is a step taken directly on the logits, treating z as free variables: the true token’s logit rises by η(1 − p)/|P| (p is its current probability), every other logit falls in proportion to its probability, and the step vanishes once the model is certain and right. A training step changes z only through the parameters (Δz ≈ (∂z/∂θ) Δθ by the chain rule), so the logits move by a different amount.',
      },
    ],
  },

  'concept:perplexity': {
    title: 'Perplexity',
    tex: '\\mathrm{PPL} = e^{\\mathcal{L}} = \\exp\\Big(-\\frac{1}{|\\mathcal{P}|}\\sum_{t\\in\\mathcal{P}} \\log p_{t,y_t}\\Big) = \\Big(\\prod_{t\\in\\mathcal{P}} p_{t,y_t}\\Big)^{-1/|\\mathcal{P}|}',
    compact: '\\mathrm{PPL} = e^{\\mathcal{L}}',
    symbols: [
      { tex: '\\mathcal{L}', text: 'mean cross-entropy in nats per scored position', dims: [] },
      { tex: '\\mathrm{PPL}', text: 'effective number of equally likely choices the model is torn between', dims: [] },
    ],
    derivation: [
      {
        title: 'Inverse geometric mean',
        steps: ['\\mathrm{PPL} = \\frac{1}{\\big(\\prod_{t\\in\\mathcal{P}} p_{t,y_t}\\big)^{1/|\\mathcal{P}|}}'],
        note: 'Perplexity is the reciprocal of the geometric mean of the probabilities given to the true tokens. A model that always gives the true token probability 1 has perplexity 1. As with the loss, this reads as next-token prediction only under a causal or sliding mask.',
      },
      {
        title: 'Uniform guessing',
        steps: ['p_{t,y_t} = \\tfrac{1}{V} \\;\\Rightarrow\\; \\mathcal{L} = \\log V \\;\\Rightarrow\\; \\mathrm{PPL} = V'],
        note: 'A model that guesses uniformly over the vocabulary has perplexity exactly V. A random initialisation is close to this, so an untrained model starts near PPL = V.',
      },
      {
        title: 'Units and comparisons',
        steps: ['\\frac{d\\,\\mathrm{PPL}}{\\mathrm{PPL}} = d\\mathcal{L}', '\\mathrm{PPL} = 2^{\\,\\text{bits per position}}\\ \\text{(if } \\log_2 \\text{ is used)}'],
        note: 'Lowering the loss by 0.1 nats cuts perplexity by about 10% (exp(−0.1) ≈ 0.905). Perplexity is per token of the chosen tokenizer, so compare models only when they share a vocabulary and tokenisation.',
      },
    ],
  },

  'concept:sampling': {
    title: 'Sampling: temperature, top-k and top-p',
    tex: '\\begin{aligned} p_v(\\tau) &= \\frac{e^{z_v/\\tau}}{\\sum_{j} e^{z_j/\\tau}} \\\\ S_k &= \\{\\,v : z_v \\ge z_{(k)}\\,\\} \\\\ p^{(k)}_v &= \\frac{p_v\\,\\mathbf{1}[v \\in S_k]}{\\sum_{u \\in S_k} p_u} \\\\ S_\\rho &= \\text{smallest prefix of the sorted } p^{(k)} \\text{ whose mass reaches } \\rho \\\\ S &= S_k \\cap S_\\rho \\\\ p\'_v &= \\frac{p_v\\,\\mathbf{1}[v \\in S]}{\\sum_{u \\in S} p_u} \\end{aligned}',
    compact: 'p_v(\\tau) \\propto e^{z_v/\\tau}',
    symbols: [
      { tex: '\\tau', text: 'temperature: τ → 0 is greedy decoding, τ > 1 flattens the distribution', dims: [] },
      { tex: 'k', text: 'top-k: keep the k largest logits (top_k in the code)', dims: [] },
      { tex: 'p^{(k)}', text: 'the distribution renormalised over the top-k set', dims: ['V'] },
      { tex: '\\rho', text: 'nucleus threshold in (0, 1] (top_p in the code)', dims: [] },
      { tex: 'S', text: 'tokens that survive both truncations: the intersection of the top-k and nucleus sets', dims: ['V'] },
    ],
    derivation: [
      {
        title: 'Temperature rescales logit gaps',
        steps: [
          '\\frac{p_u(\\tau)}{p_v(\\tau)} = e^{(z_u - z_v)/\\tau}',
          '\\lim_{\\tau\\to 0^{+}} p_v(\\tau) = \\mathbf{1}\\big[v = \\arg\\max_j z_j\\big] \\quad \\text{(unique maximum)}',
        ],
        note: 'Dividing by τ multiplies every logit gap by 1/τ. τ < 1 sharpens the distribution toward the most likely token, τ = 1 is the model’s own distribution, and τ > 1 flattens it toward uniform. Since τ > 0 keeps the order of the logits, temperature never changes which tokens top-k or top-p keep.',
      },
      {
        title: 'Truncate, then renormalise',
        steps: ['p\'_v = \\frac{p_v\\,\\mathbf{1}[v \\in S]}{\\sum_{u \\in S} p_u} = \\operatorname{softmax}_{v \\in S}\\big(z_v/\\tau\\big)'],
        note: 'Discarded tokens get logit −∞, so their probability is exp(−∞) = 0, and the final softmax renormalises over the kept set. Top-p runs after top-k, so the nucleus is measured on p^(k), the distribution already renormalised over the top-k set, as in reference/model.py and the app engine.',
      },
      {
        title: 'Top-k and top-p sets',
        steps: [
          'S_k = \\{\\,v : z_v \\ge z_{(k)}\\,\\}, \\quad z_{(k)} = \\text{k-th largest logit}',
          'S_\\rho = \\{\\,v_{(1)}, \\dots, v_{(i^\\star)}\\,\\}, \\quad i^\\star = \\min\\Big\\{\\,i : \\sum_{r=1}^{i} p^{(k)}_{(r)} \\ge \\rho\\,\\Big\\}',
          'S = S_k \\cap S_\\rho',
        ],
        note: 'Top-k keeps every token whose logit reaches the k-th largest. reference/model.py keeps all such ties, so it can keep more than k tokens, while the app engine (src/engine/generate.js) keeps exactly k. Top-p keeps the smallest set of most likely candidates whose mass reaches ρ, measured on p^(k), so the set is small when the model is confident and large when it is uncertain. S_ρ already lies inside S_k, so S = S_ρ, and the intersection just says so. reference/model.py also keeps the next token when the mass reaches ρ exactly; the app engine does not.',
      },
    ],
  },

  'concept:kv-cache': {
    title: 'KV cache: memory and decode cost',
    tex: '\\text{bytes} = 2\\cdot L\\cdot H_{kv}\\cdot d_h\\cdot T\\cdot b \\qquad \\text{per decode step: FLOPs} \\approx 2N + 4\\,L\\,T\\,d, \\quad \\text{bytes read} \\approx b\\,(N + 2\\,L\\,H_{kv}\\,d_h\\,T)',
    compact: '\\text{bytes} = 2\\,L\\,H_{kv}\\,d_h\\,T\\,b',
    symbols: [
      { tex: 'L', text: 'number of layers (n_layer)', dims: [] },
      { tex: 'H_{kv}', text: 'KV heads (n_kv_head): H for multi-head, 1 for multi-query, in between for grouped-query', dims: ['Hkv'] },
      { tex: 'd_h', text: 'head width d / H', dims: ['dh'] },
      { tex: 'T', text: 'cached positions, i.e. the context length so far', dims: ['T'] },
      { tex: 'b', text: 'bytes per stored value (4 for fp32, 2 for fp16 or bf16)', dims: [] },
      { tex: 'N', text: 'matmul parameters (see concept:flops)', dims: [] },
    ],
    derivation: [
      {
        title: 'What is cached',
        steps: [
          'k_t,\\,v_t \\in \\R^{H_{kv} d_h} \\quad \\text{per token, per layer}',
          '\\text{cache} = \\{\\,k_{\\le T}^{(\\ell)},\\ v_{\\le T}^{(\\ell)}\\,\\}_{\\ell=1}^{L}',
        ],
        note: 'Past keys and values never change, so each decode step computes q, k and v for the new token only, appends its k and v, and reuses everything else.',
      },
      {
        title: 'Bytes',
        steps: ['\\text{bytes} = \\underbrace{2}_{K,\\,V}\\cdot L\\cdot H_{kv}\\cdot d_h\\cdot T\\cdot b'],
        note: 'The cache grows linearly with T and is per sequence: a batch of B sequences multiplies it by B. For L = 32, H_kv = 32, d_h = 128 and T = 4096 in fp16 one sequence needs 2^31 bytes, i.e. 2 GiB, which is why fewer KV heads (grouped-query attention) matter at inference time.',
      },
      {
        title: 'Cost of one decode step',
        steps: [
          '\\text{FLOPs} \\approx \\underbrace{2N}_{\\text{weights}} + \\underbrace{4\\,L\\,T\\,d}_{\\text{attention over } T \\text{ cached keys}}',
          '\\text{bytes read} \\approx \\underbrace{b\\,N}_{\\text{weights}} + \\underbrace{2\\,b\\,L\\,H_{kv}\\,d_h\\,T}_{\\text{KV cache}}, \\qquad \\text{intensity} = \\frac{2N + 4\\,L\\,T\\,d}{b\\,(N + 2\\,L\\,H_{kv}\\,d_h\\,T)}',
          '\\text{MHA } (H_{kv} d_h = d):\\quad \\text{intensity} = \\frac{2(N + 2\\,L\\,T\\,d)}{b\\,(N + 2\\,L\\,T\\,d)} = \\frac{2}{b} \\ \\text{for every } T',
        ],
        note: 'One new row reads every weight once and the whole cache once. For an MHA model at batch size 1 the intensity is exactly 2/b, which is 1 FLOP per byte in fp16 and far below the balance point of modern accelerators, so decoding is memory-bound. Fewer KV heads (GQA, MQA) shrink the cache term by H/H_kv, so intensity rises toward (H/H_kv)·2/b when the cache dominates. Batching reads the weights once but the cache once per sequence, so it raises intensity. The attention term is 4LTd rather than the 2LTd of a prefill pass, because one query attends to all T keys with no causal halving.',
      },
    ],
  },

  'concept:flops': {
    title: 'FLOP budget: forward, training and whole runs',
    tex: 'C_{\\text{fwd}} \\approx 2N + 2\\,L\\,T\\,d, \\qquad C_{\\text{train}} \\approx 6N + 6\\,L\\,T\\,d \\ \\text{per token}, \\qquad C_{\\text{train}}(D) \\approx 6\\,N\\,D',
    compact: 'C_{\\text{fwd}} \\approx 2N, \\quad C_{\\text{train}} \\approx 6N \\ \\text{per token}',
    symbols: [
      { tex: 'N', text: 'matmul parameters: every weight that multiplies activations, including W_U; no embedding lookups, norms or biases', dims: [] },
      { tex: 'L', text: 'number of layers', dims: [] },
      { tex: 'T', text: 'sequence length of the training sequences. The attention term averages each token over its causal prefix (about T/2 keys); a decode step at context T costs 4LTd instead (see concept:kv-cache)', dims: ['T'] },
      { tex: 'd', text: 'model width', dims: ['d'] },
      { tex: 'C', text: 'floating-point operations; one multiply-add counts as 2', dims: [] },
      { tex: 'D', text: 'training tokens seen', dims: [] },
    ],
    derivation: [
      {
        title: 'Forward: two FLOPs per weight per token',
        steps: [
          'y = Wx,\\ W \\in \\R^{m\\times n}: \\quad mn \\text{ multiply-adds} = 2mn \\text{ FLOPs}',
          'C_{\\text{fwd}} = \\sum_{W} 2\\,|W| = 2N \\quad \\text{per token}',
        ],
        note: 'Each weight is used once per token, so the parameter matmuls cost 2N FLOPs for one token. Embedding lookups are memory reads rather than matmuls, which is why they are excluded from N.',
      },
      {
        title: 'Backward costs about twice the forward',
        steps: [
          '\\frac{\\partial \\mathcal{L}}{\\partial X} = \\frac{\\partial \\mathcal{L}}{\\partial Y}\\,W, \\qquad \\frac{\\partial \\mathcal{L}}{\\partial W} = \\Big(\\frac{\\partial \\mathcal{L}}{\\partial Y}\\Big)^{\\top} X',
          'C_{\\text{train}} = \\underbrace{2N}_{\\text{fwd}} + \\underbrace{4N}_{\\text{bwd}} = 6N \\quad \\text{per token}',
        ],
        note: 'Each linear layer needs two matmuls of the forward size: one for the input gradient and one for the weight gradient. Backward is therefore about 2× forward.',
      },
      {
        title: 'Attention term',
        steps: [
          '\\underbrace{2\\cdot\\tfrac{T}{2}\\cdot d}_{\\Q\\K^{\\top}} + \\underbrace{2\\cdot\\tfrac{T}{2}\\cdot d}_{\\Aw\\V} = 2\\,T\\,d \\quad \\text{per layer, per token}',
          'C_{\\text{fwd, attn}} = 2\\,L\\,T\\,d, \\qquad C_{\\text{train, attn}} = 3 \\times 2\\,L\\,T\\,d = 6\\,L\\,T\\,d',
        ],
        note: 'Attention has no weights, but each query scores against its causal prefix, about T/2 keys on average. Each key costs 2d FLOPs for the dot product with q and 2d for the weighted sum with v. Training repeats the same forward-and-backward pattern, so it costs 3× forward.',
      },
      {
        title: 'Whole runs',
        steps: [
          'C_{\\text{train}}(D) \\approx 6\\,N\\,D',
          '\\text{e.g. } N = 10^{9},\\ D = 2\\cdot 10^{10} \\Rightarrow 1.2\\cdot 10^{20}\\ \\text{FLOPs}',
        ],
        note: 'Planning rule: total training FLOPs ≈ 6ND. The attention term is small while T is well below about 12d (the usual 4d MLP), so the rule holds for typical contexts.',
      },
    ],
  },

  'concept:attention-complexity': {
    title: 'Attention cost and FlashAttention',
    tex: '\\text{time} = O(T^2 d), \\qquad \\text{score memory} = O(H\\,T^2)\\ \\ (\\text{per head } O(T^2)), \\qquad \\text{FlashAttention extra memory} = O(T)',
    compact: 'O(T^2 d)\\ \\text{time},\\quad O(T^2)\\ \\text{score memory}',
    symbols: [
      { tex: 'T', text: 'sequence length', dims: ['T'] },
      { tex: 'd', text: 'model width, d = H·d_h', dims: ['d'] },
      { tex: 'B_c', text: 'keys per block: the tile size of the online-softmax loop below', dims: [] },
      { tex: 's_j', text: 'masked, scaled score of query i against key j (−∞ where the mask hides j)', dims: [] },
      { tex: 'v_j', text: 'value vector of key j', dims: ['dh'] },
      { tex: 'm, \\ell', text: 'running row max and running softmax denominator, one pair per query', dims: ['T'] },
      { tex: 'O', text: 'running unnormalised output, one row per query', dims: ['T', 'dh'] },
    ],
    derivation: [
      {
        title: 'Where the T² comes from',
        steps: [
          '\\Q\\K^{\\top}: \\ (T \\times d_h)(d_h \\times T) \\Rightarrow T^2 d_h \\text{ multiply-adds per head}',
          '\\text{time} = \\underbrace{H\\,T^2 d_h}_{\\Q\\K^{\\top}} + \\underbrace{H\\,T^2 d_h}_{\\Aw\\V} = O(T^2 d)',
        ],
        note: 'Each of the T² query-key pairs costs one dot product of width d in total across heads, and the weighted sum over values costs the same again. Linear layers cost O(T d²), so attention takes over once T is a constant multiple of d.',
      },
      {
        title: 'Naive memory',
        steps: ['S,\\ A \\in \\R^{H\\times T\\times T} \\;\\Rightarrow\\; O(H\\,T^2) \\text{ values per layer}'],
        note: 'The score and weight matrices are materialised in high-bandwidth memory. For T = 4096 and H = 32 that is 32 · 4096² ≈ 5.4·10⁸ values per layer, which dwarfs the T·d activations.',
      },
      {
        title: 'Online softmax, one block at a time',
        steps: [
          '\\text{blk} = \\text{a block of } B_c \\text{ keys}, \\qquad m_{\\text{blk}} = \\max_{j\\in\\text{blk}} s_j, \\qquad m_{\\text{new}} = \\max(m,\\ m_{\\text{blk}})',
          '\\ell_{\\text{new}} = e^{\\,m-m_{\\text{new}}}\\,\\ell + \\sum_{j\\in\\text{blk}} e^{\\,s_j-m_{\\text{new}}}',
          'O_{\\text{new}} = e^{\\,m-m_{\\text{new}}}\\,O + \\sum_{j\\in\\text{blk}} e^{\\,s_j-m_{\\text{new}}}\\,v_j, \\qquad o_i = \\frac{O_{\\text{final}}}{\\ell_{\\text{final}}}',
        ],
        note: 'Each block of B_c keys is processed in turn, and its sums start relative to the running max. Re-basing them to m_new gives the same terms the full-row softmax uses, so o_i is exact up to floating-point rounding. Only m, ℓ and O are kept per query row, so the extra memory is O(T), and the T×T matrix never has to be written to slow memory. The FLOP count is unchanged; the saving is memory traffic.',
      },
      {
        title: 'Why the rescaling is exact',
        steps: ['\\ell_{\\text{old}} = \\sum_{j\\in\\text{old}} e^{\\,s_j-m} \\;\\Rightarrow\\; e^{\\,m-m_{\\text{new}}}\\,\\ell_{\\text{old}} = \\sum_{j\\in\\text{old}} e^{\\,s_j-m_{\\text{new}}}'],
        note: 'This one identity is the whole trick: multiplying by exp(m − m_new) moves a partial sum to the new reference point. Starting from m = −∞ and ℓ = 0 is safe, because exp(−∞) = 0 whenever m_new is finite (an empty partial sum rescales to zero), so the first block that has a visible key needs no special case. The one hazard is a block whose scores are all masked (for example keys outside a sliding window) while m is still −∞: then m_new = −∞ and exp(m − m_new) = exp(−∞ − (−∞)) is NaN. Skip such a block, or take its rescale factor as 0.',
      },
    ],
  },
};

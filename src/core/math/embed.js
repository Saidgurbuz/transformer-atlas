// Maths entries for the embedding stage: the token lookup, positions, the block input
// x_0 and the attention mask, plus two concepts (one-hot lookup, weight tying).
// Schema and checks: docs/MATH_GUIDE.md. Each live() recomputes its cell from the
// weights or from upstream traced tensors; tests/math.mjs compares it with the trace.
import { at2, num } from './util.js';

// A token label as LaTeX text: the specials are escaped, a lone space is spelled out, and the
// leading-space mark of a subword label (␣had) is drawn with the \vsp macro.
const TEXT_ESC = {
  '\\': '\\textbackslash{}', '{': '\\{', '}': '\\}', '$': '\\$', '&': '\\&', '#': '\\#',
  '%': '\\%', '_': '\\_', '^': '\\textasciicircum{}', '~': '\\textasciitilde{}',
  // display marks of byte-level labels that KaTeX's fonts lack: ⟨c3⟩ (a lone byte), ↵ and ⇥
  '⟨': '\\textless{}', '⟩': '\\textgreater{}', '↵': '\\textbackslash{}n', '⇥': '\\textbackslash{}t',
};
const isSpace = (label) => label === '␣' || label === ' ';
const tokenTeX = (label) => `\\text{${isSpace(label) ? 'space' : [...label].map((c) => TEXT_ESC[c] ?? (c === '␣' ? '\\vsp{}' : c)).join('')}}`;
const plain = (label) => (isSpace(label) ? 'space' : label);
// A number inside a sum: negatives are bracketed, as in "0.12 + (-0.03)".
const signed = (v, digits) => (v < 0 ? `(${num(v, digits)})` : num(v, digits));
// Live numbers: table entries and intermediate values show 6 decimals, so a hand check
// reproduces the chain. A computed result shows 4 decimals and is marked with \approx,
// because it is computed from rounded inputs.

export const ENTRIES = {
  'emb.tok': {
    title: 'Token embeddings (lookup)',
    tex: String.raw`X_{\mathrm{tok}} = \WE[\mathrm{ids}] \in \R^{T\times d}, \qquad X_{\mathrm{tok},t,c} = \WE[\mathrm{id}_t,\,c], \qquad \mathrm{id}_t \in \{0,\dots,V-1\}`,
    compact: String.raw`X_{\mathrm{tok},t,c} = \WE[\mathrm{id}_t,\,c]`,
    symbols: [
      { tex: String.raw`\mathrm{ids}`, text: 'token ids of the sequence, one per position', dims: ['T'] },
      { tex: String.raw`\mathrm{id}_t`, text: 'vocabulary index of the token at position t', dims: [] },
      { tex: String.raw`\WE`, text: 'token embedding matrix: one learned row per vocabulary entry', dims: ['V', 'd'] },
      { tex: String.raw`X_{\mathrm{tok}}`, text: 'token embeddings: row t is the vector of the token at position t', dims: ['T', 'd'] },
    ],
    derivation: [
      {
        title: 'Lookup is a one-hot product',
        steps: [
          String.raw`\mathrm{onehot}(v) \in \{0,1\}^{V}, \qquad \mathrm{onehot}(v)_u = \mathbf{1}[u = v]`,
          String.raw`\big(\mathrm{onehot}(\mathrm{id}_t)^{\top}\WE\big)_c = \sum_{u}\mathbf{1}[u=\mathrm{id}_t]\,\WE_{u,c} = \WE_{\mathrm{id}_t,\,c}`,
        ],
        note: 'Multiplying by a one-hot vector copies out one row, so the embedding layer is a linear map applied to one-hot inputs. Implementations index the row directly: d reads per token instead of V·d multiply-adds, with identical numbers.',
      },
      {
        title: 'Gradient: a scatter-add',
        steps: [
          String.raw`\frac{\partial L}{\partial \WE_{v,c}} = \sum_{t:\ \mathrm{id}_t = v} \frac{\partial L}{\partial X_{\mathrm{tok},t,c}}`,
        ],
        note: 'Only rows of tokens that occur in the input receive gradient from this path. Tied output weights add gradient to every row (see the weight-tying concept).',
      },
    ],
    live({ trace, model, ids, label, row, col }) {
      const W = model.params['wte.weight'], d = W.shape[1], id = ids[row];
      const result = W.data[id * d + col];
      return {
        tex: String.raw`X_{\mathrm{tok},${row},${col}} = \WE[\,\mathrm{id}_{${row}}{=}${id}\ (${tokenTeX(label(row))}),\ ${col}\,] = \WE_{${id},${col}} = ${num(result, 6)}`,
        result,
        expected: at2(trace['emb.tok'], row, col),
        note: `token ${row} is "${plain(label(row))}" (id ${id}): row ${id} of W_E is picked`,
      };
    },
  },

  'emb.pos': {
    title: 'Positional encoding',
    tex: String.raw`\Pe_{t,c} = \begin{cases} \kc{pos}{W_{P}}[t,\,c] & \text{learned} \\ \sin(\omega_i\,t) & \text{sinusoidal},\ c=2i \\ \cos(\omega_i\,t) & \text{sinusoidal},\ c=2i{+}1 \end{cases}, \qquad \omega_i = 10000^{-2i/d}`,
    compact: String.raw`\Pe_{t,2i}=\sin\!\big(\tfrac{t}{10000^{2i/d}}\big),\quad \Pe_{t,2i+1}=\cos\!\big(\tfrac{t}{10000^{2i/d}}\big)`,
    symbols: [
      { tex: String.raw`\Pe`, text: 'positional encoding: row t is the vector added to the token at position t', dims: ['T', 'd'] },
      { tex: String.raw`\kc{pos}{W_{P}}`, text: 'learned position table (learned models only), shape block_size × d; rows 0 to T−1 are used', dims: ['T', 'd'] },
      { tex: String.raw`t`, text: 'position index, 0 ≤ t < T', dims: [] },
      { tex: String.raw`i`, text: 'index of a sin/cos pair: dimensions 2i and 2i+1, 0 ≤ i < d/2', dims: [] },
      { tex: String.raw`\omega_i`, text: 'angular frequency of pair i (wavelength 2π/ω_i)', dims: [] },
    ],
    derivation: [
      {
        title: 'Learned table: a lookup with ids 0 … T−1',
        steps: [
          String.raw`\Pe = \mathrm{onehot}(0,1,\dots,T-1)\,\kc{pos}{W_{P}}`,
        ],
        note: 'A learned position is the same kind of embedding lookup, with the position index as the id. The table has block_size rows, so positions beyond block_size−1 have no vector (the model asserts T ≤ block_size).',
      },
      {
        title: 'Sinusoids: a rotation for every offset',
        steps: [
          String.raw`\Pe_{t,2i}=\sin(\omega_i t),\qquad \Pe_{t,2i+1}=\cos(\omega_i t),\qquad \lambda_i = \frac{2\pi}{\omega_i} = 2\pi\cdot 10000^{2i/d}`,
          String.raw`\begin{pmatrix}\Pe_{t+k,2i}\\ \Pe_{t+k,2i+1}\end{pmatrix} = \begin{pmatrix}\cos\omega_i k & \sin\omega_i k\\ -\sin\omega_i k & \cos\omega_i k\end{pmatrix}\begin{pmatrix}\Pe_{t,2i}\\ \Pe_{t,2i+1}\end{pmatrix}`,
        ],
        note: 'Each pair is a point on a circle that advances by ω_i per position; wavelengths run from 2π up to just under 2π·10⁴. A shift by k is one fixed 2×2 rotation for every t, so the encoding of t+k is a linear function of the encoding of t. Vaswani et al. (2017, §3.5) hypothesised that this makes relative attention easy to learn, and reported learned positions as nearly identical in quality. The same table is derived as concept:sinusoidal in positions.js, with the pair index written k.',
      },
    ],
    live({ trace, cfg, model, row, col }) {
      const t = row, c = col;
      if (cfg.pos_encoding === 'learned') {
        const W = model.params['wpe.weight'];
        const result = W.data[t * W.shape[1] + c];
        return {
          tex: String.raw`\Pe_{${t},${c}} = \kc{pos}{W_{P}}[${t},\,${c}] = ${num(result, 6)}`,
          result,
          expected: at2(trace['emb.pos'], t, c),
          note: `learned row ${t} of the position table, entry ${c}`,
        };
      }
      if (cfg.pos_encoding === 'sinusoidal') {
        const d = cfg.d_model, i2 = c - (c % 2);          // 2i: the even member of the pair
        const den = Math.pow(10000, i2 / d), angle = t / den;
        const even = c % 2 === 0, fn = even ? '\\sin' : '\\cos';
        const result = even ? Math.sin(angle) : Math.cos(angle);
        return {
          tex: String.raw`\Pe_{${t},${c}} = ${fn}\!\Big(\frac{${t}}{10000^{${i2}/${d}}}\Big) = ${fn}\!\Big(\frac{${t}}{${num(den, 6)}}\Big) = ${fn}(${num(angle, 6)}) \approx ${num(result, 4)}`,
          result,
          expected: at2(trace['emb.pos'], t, c),
          note: `pair ${i2 / 2} (dimensions ${i2} and ${i2 + 1}), ${even ? 'sine' : 'cosine'} branch; intermediates to 6 decimals`,
        };
      }
      return null; // RoPE, ALiBi and none add no positional table
    },
  },

  'emb.out': {
    title: 'Block input x₀',
    tex: String.raw`\xres_0 = \begin{cases} X_{\mathrm{tok}} + \Pe & \text{learned or sinusoidal} \\ X_{\mathrm{tok}} & \text{RoPE, ALiBi or none} \end{cases}`,
    compact: String.raw`\xres_{0} = X_{\mathrm{tok}} + \Pe \text{ (or } X_{\mathrm{tok}} \text{ with no table)}`,
    symbols: [
      { tex: String.raw`\xres_0`, text: 'block input: the residual stream before layer 0', dims: ['T', 'd'] },
      { tex: String.raw`X_{\mathrm{tok}}`, text: 'token embeddings', dims: ['T', 'd'] },
      { tex: String.raw`\Pe`, text: 'positional encoding (absent for RoPE, ALiBi and none)', dims: ['T', 'd'] },
      { tex: String.raw`R_t`, text: 'RoPE rotation for position t: one 2×2 block per pair (2k, 2k+1), angle t·θ_k (see concept:rope-relative)', dims: ['dh', 'dh'] },
      { tex: String.raw`m_h`, text: 'ALiBi slope of head h, 2^(−8(h+1)/H) (see the alibi entry)', dims: ['H'] },
      { tex: String.raw`\tilde S_{h,ij}`, text: 'scaled score q_i·k_j/√d_h (see L{i}.attn.scaled)', dims: ['H', 'T', 'T'] },
      { tex: String.raw`S^{b}_{h,ij}`, text: 'score after the ALiBi bias, before the mask (see L{i}.attn.biased)', dims: ['H', 'T', 'T'] },
    ],
    derivation: [
      {
        title: 'Why add, and where the gradient goes',
        steps: [
          String.raw`\xres_{0} = X_{\mathrm{tok}} + \Pe \quad\Longrightarrow\quad \frac{\partial L}{\partial X_{\mathrm{tok}}} = \frac{\partial L}{\partial \xres_0} = \frac{\partial L}{\partial \Pe}`,
          String.raw`\xres_{\ell}^{\mathrm{mid}} = \xres_{\ell} + \Attn_{\ell}\big(\LN_1(\xres_{\ell})\big), \qquad \xres_{\ell+1} = \xres_{\ell}^{\mathrm{mid}} + \MLP_{\ell}\big(\LN_2(\xres_{\ell}^{\mathrm{mid}})\big)`,
          String.raw`\xres_{L} = \xres_{0} + \sum_{\ell=0}^{L-1}\Big[\Attn_{\ell}\big(\LN_1(\xres_{\ell})\big) + \MLP_{\ell}\big(\LN_2(\xres_{\ell}^{\mathrm{mid}})\big)\Big] \qquad \text{(pre-LN, residual on)}`,
        ],
        note: 'Addition passes the gradient unchanged to both summands. X_tok then reaches W_E through the scatter-add of emb.tok, and a learned position table receives it through the lookup in emb.pos. The sinusoidal table is a constant, so the gradient stops there. Unrolled, the stream is x_0 plus every attention and MLP output, so token identity and position reach the final norm directly as well as through each block. Elhage et al. (2021) describe this stream as the shared channel that every layer reads from and writes to.',
      },
      {
        title: 'Without an additive position table',
        steps: [
          String.raw`\text{RoPE: } \Q_{t} = R_{t}\,q_{t}, \qquad \K_{t} = R_{t}\,k_{t}`,
          String.raw`\text{ALiBi: } S^{b}_{h,ij} = \tilde S_{h,ij} - m_h\,\lvert i-j\rvert`,
          String.raw`\text{none: } \xres_0 = X_{\mathrm{tok}}`,
        ],
        note: 'RoPE rotates the queries and keys and ALiBi adds a distance bias to the scores, so the residual stream carries token identity only. With no position signal and no mask, the block is permutation-equivariant: reordering the tokens reorders the outputs. A causal mask breaks that symmetry, so a decoder without positions can still infer order (Haviv et al., 2022). R_t and m_h are defined in the symbol list, and positions.js derives both.',
      },
    ],
    live({ trace, cfg, row, col }) {
      const tok = at2(trace['emb.tok'], row, col);
      const expected = at2(trace['emb.out'], row, col);
      if (cfg.pos_encoding === 'learned' || cfg.pos_encoding === 'sinusoidal') {
        const pos = at2(trace['emb.pos'], row, col), result = tok + pos;
        return {
          tex: String.raw`\xres_{0,${row},${col}} = X_{\mathrm{tok},${row},${col}} + \Pe_{${row},${col}} = ${num(tok, 6)} + ${signed(pos, 6)} \approx ${num(result, 4)}`,
          result,
          expected,
          note: `token and position vectors added element by element (${cfg.pos_encoding}); intermediates to 6 decimals`,
        };
      }
      return {
        tex: String.raw`\xres_{0,${row},${col}} = X_{\mathrm{tok},${row},${col}} = ${num(tok, 6)}`,
        result: tok,
        expected,
        note: `no positional table (${cfg.pos_encoding}): the block input is the token vector`,
      };
    },
  },

  mask: {
    title: 'Attention mask',
    tex: String.raw`\Mk_{ij} = \begin{cases} \mathbf{1}[j\le i] & \text{causal} \\ 1 & \text{none} \\ \mathbf{1}[j\le i]\land\mathbf{1}[j>i-w] & \text{sliding} \\ \mathbf{1}[j\le i]\lor\mathbf{1}[j<P_{\mathrm{pre}}] & \text{prefix} \end{cases}`,
    compact: String.raw`\Mk_{ij}=\mathbf{1}[\text{key } j \text{ visible to query } i]`,
    symbols: [
      { tex: String.raw`\Mk`, text: 'mask: 1 where query i may read key j, 0 where its score becomes −∞', dims: ['T', 'T'] },
      { tex: String.raw`i`, text: 'query position, 0-based (row of the mask): 0 ≤ i < T', dims: [] },
      { tex: String.raw`j`, text: 'key position, 0-based (column of the mask): 0 ≤ j < T', dims: [] },
      { tex: String.raw`w`, text: 'sliding window (cfg.window, at least 1): a query sees itself and w−1 earlier keys', dims: [] },
      { tex: String.raw`P_{\mathrm{pre}}`, text: 'prefix length (cfg.prefix_len, 0 to block_size): keys j < P are visible to every query; P ≥ T makes every key visible', dims: [] },
    ],
    derivation: [
      {
        title: 'Causal mask: why the future is hidden',
        steps: [
          String.raw`p(x_0,\dots,x_{T-1}) = \prod_{t=0}^{T-1} p(x_t \mid x_{<t})`,
          String.raw`\text{row } i \text{ may depend on } x_{\le i} \text{ only, so } \Aw_{ij} = 0 \text{ for } j>i`,
        ],
        note: 'The factorisation trains every next-token prediction in one pass only if position i cannot read x_{i+1}; without the mask it could copy the answer from its own input. Vaswani et al. (2017, §3.2.3) use this masking in the Transformer decoder.',
      },
      {
        title: 'Visible keys per query',
        steps: [
          String.raw`N_i = \sum_{j=0}^{T-1}\mathbf{1}[\Mk_{ij}=1]`,
          String.raw`\text{none: } N_i = T, \qquad \text{causal: } N_i = i+1, \qquad \text{sliding: } N_i = \min(i+1,\,w)`,
          String.raw`\text{prefix: } N_i = \min\big(T,\ \max(i+1,\,P_{\mathrm{pre}})\big)`,
        ],
        note: 'Softmax spreads each row over its N_i visible keys only, so the row sums to 1 over them. Keys run over 0 … T−1. The prefix keys are the first P_pre positions, so the visible set is the longer of the runs 0 … i and 0 … P_pre−1, capped at T: a prefix longer than the sequence makes every key visible. A sliding window keeps the work per query at w instead of i+1, the motivation for windowed attention (Beltagy et al., 2020; Mistral 7B, Jiang et al., 2023). Prefix-LM masks let the prompt be read bidirectionally while the continuation stays causal (Raffel et al., 2019).',
      },
    ],
    live({ trace, cfg, row: i, col: j }) {
      const T = trace.mask.shape[1];
      const expected = at2(trace.mask, i, j);
      const causal = j <= i ? 1 : 0;
      let tex, result;
      if (cfg.mask === 'none') {
        result = 1;
        tex = String.raw`\Mk_{${i},${j}} = 1`;
      } else if (cfg.mask === 'sliding') {
        const inWindow = j > i - cfg.window ? 1 : 0;
        result = causal * inWindow;
        tex = String.raw`\Mk_{${i},${j}} = \mathbf{1}[${j}\le${i}]\land\mathbf{1}[${j}>${i - cfg.window}] = ${causal}\land${inWindow} = ${result}`;
      } else if (cfg.mask === 'prefix') {
        const inPrefix = j < cfg.prefix_len ? 1 : 0;
        result = Math.max(causal, inPrefix);
        tex = String.raw`\Mk_{${i},${j}} = \mathbf{1}[${j}\le${i}]\lor\mathbf{1}[${j}<${cfg.prefix_len}] = ${causal}\lor${inPrefix} = ${result}`;
      } else {
        result = causal;
        tex = String.raw`\Mk_{${i},${j}} = \mathbf{1}[${j}\le${i}] = ${result}`;
      }
      let visible = 0;
      for (let k = 0; k < T; k++) visible += at2(trace.mask, i, k);
      return { tex, result, expected, note: `${visible} of ${T} keys are visible to query ${i}` };
    },
  },

  'concept:one-hot-lookup': {
    title: 'One-hot lookup',
    tex: String.raw`X_{\mathrm{tok}} = \mathrm{onehot}(\mathrm{ids})\,\WE, \qquad \mathrm{onehot}(\mathrm{ids})\in\{0,1\}^{T\times V}, \qquad \mathrm{onehot}(\mathrm{ids})_{t,u} = \mathbf{1}[u=\mathrm{id}_t]`,
    compact: String.raw`\mathrm{onehot}(\mathrm{ids})\,\WE = \WE[\mathrm{ids}]`,
    symbols: [
      { tex: String.raw`\mathrm{onehot}(\mathrm{ids})`, text: 'one row per position, with a single 1 in the column of its token id', dims: ['T', 'V'] },
      { tex: String.raw`\WE`, text: 'matrix of the linear map from V inputs to d outputs, applied to row vectors', dims: ['V', 'd'] },
      { tex: String.raw`\mathbf{1}[\cdot]`, text: 'indicator: 1 if the condition holds, else 0', dims: [] },
    ],
    derivation: [
      {
        title: 'A one-hot product selects a row',
        steps: [
          String.raw`\big(\mathrm{onehot}(v)^{\top}\WE\big)_c = \sum_{u}\mathbf{1}[u=v]\,\WE_{u,c} = \WE_{v,c}`,
          String.raw`\text{so row } t \text{ of } \mathrm{onehot}(\mathrm{ids})\,\WE \text{ equals row } \mathrm{id}_t \text{ of } \WE`,
        ],
        note: 'The embedding layer is therefore an ordinary linear layer on one-hot inputs. Bengio et al. (2003) introduced this word-feature lookup table for neural language models.',
      },
      {
        title: 'Cost: a lookup, not a multiply',
        steps: [
          String.raw`\underbrace{V\cdot d}_{\text{one-hot product, per token}}\ \text{multiply-adds} \qquad\text{versus}\qquad \underbrace{d}_{\text{row read}}\ \text{reads}`,
          String.raw`V = 32\,000,\ d = 4\,096:\qquad 32\,000 \cdot 4\,096 \approx 1.3\times10^{8}\ \text{multiply-adds per token}`,
        ],
        note: 'Both give identical numbers. Frameworks implement the lookup as a gather (index select); the one-hot product is only the mathematical description, and most of its multiplies are by zero.',
      },
      {
        title: 'Backward: a scatter-add',
        steps: [
          String.raw`\frac{\partial L}{\partial \WE_{v,c}} = \sum_{t:\ \mathrm{id}_t=v} \frac{\partial L}{\partial X_{\mathrm{tok},t,c}}`,
        ],
        note: 'A row of W_E gets gradient from the positions where its token occurs, so the lookup gradient is sparse: rows of unused tokens stay at zero from this path.',
      },
    ],
  },

  'concept:weight-tying': {
    title: 'Weight tying: logits = x W_E^⊤',
    tex: String.raw`z_{t,v} = \sum_{c=1}^{d} \xres_{\mathrm{f},t,c}\,\WU_{v,c}, \qquad z = \xres_{\mathrm{f}}\,\WU^{\top}, \qquad \WU = \WE \;\Rightarrow\; z = \xres_{\mathrm{f}}\,\WE^{\top}`,
    compact: String.raw`z = \xres_{\mathrm{f}}\,\WE^{\top} \text{ when } \WU = \WE`,
    symbols: [
      { tex: String.raw`\xres_{\mathrm{f}}`, text: 'final residual state of each position, after the final norm (ln_f for pre-LN; the last block’s LN2 output for post-LN)', dims: ['T', 'd'] },
      { tex: String.raw`z`, text: 'logits: one score per vocabulary entry for every position', dims: ['T', 'V'] },
      { tex: String.raw`\WU`, text: 'unembedding (lm_head): untied it is its own matrix; tied it is W_E', dims: ['V', 'd'] },
      { tex: String.raw`\WE`, text: 'token embedding matrix (the same parameter as W_U when tied)', dims: ['V', 'd'] },
    ],
    derivation: [
      {
        title: 'Parameter saving',
        steps: [
          String.raw`N_{\text{untied}} = \underbrace{Vd}_{\WE} + \underbrace{Vd}_{\WU} = 2Vd, \qquad N_{\text{tied}} = Vd, \qquad \Delta N = Vd`,
          String.raw`\text{charlm preset } (V=32,\ d=64):\quad 208\,256 \text{ parameters as shipped (untied)},\quad 206\,208 \text{ if tied},\quad \Delta N = 2\,048`,
        ],
        note: 'With V = 32 000 and d = 4 096 the saving is about 1.3 × 10⁸ parameters. It also removes a second V×d matrix that must be stored and optimised.',
      },
      {
        title: 'Why one matrix is a sensible prior',
        steps: [
          String.raw`z_{t,v} = \xres_{\mathrm{f},t} \cdot \WE_v = \lVert \xres_{\mathrm{f},t} \rVert\,\lVert \WE_v \rVert\,\cos\theta_{t,v}`,
        ],
        note: 'The logit of token v is large when the state points the same way as that token’s own embedding, the space the input lookup already uses. Press and Wolf (2016) showed that output embeddings can improve language models, and Vaswani et al. (2017, §3.4) shared the two matrices.',
      },
      {
        title: 'Gradients add up',
        steps: [
          String.raw`\frac{\partial L}{\partial \WE} = \underbrace{\mathrm{scatter}\!\Big(\frac{\partial L}{\partial X_{\mathrm{tok}}}\Big)}_{\text{input lookup}} + \underbrace{\Big(\frac{\partial L}{\partial z}\Big)^{\top} \xres_{\mathrm{f}}}_{\text{output head}}`,
        ],
        note: 'Here scatter adds each position’s gradient into the row of its token id. Both uses read the same parameter, so autograd sums the two contributions. The output term is dense: every vocabulary row receives a softmax-weighted gradient, so in a tied model every row of W_E gets gradient at every step. In an untied model an input row receives gradient only from positions holding that token.',
      },
    ],
  },
};

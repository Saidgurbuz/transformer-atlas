# Math catalogue guide

Entries live in `src/core/math/<topic>.js` as `export const ENTRIES = { '<pattern>': {...} }`.
Patterns are trace-key patterns (`L{i}.attn.q`, `emb.tok`, `logits`) or `concept:<name>`.
`src/core/math/attention.js` is the worked example: copy its style exactly.

## Entry schema

```js
{
  title: 'Attention weights (softmax)',
  tex: '...',            // display equation(s), the definitive statement
  compact: '...',        // one line for tooltips
  symbols: [{ tex: '\\Q', text: 'queries of this head', dims: ['T', 'dh'] }],  // dims use T d H Hkv dh F V
  derivation: [{ title: 'Why …', steps: ['tex', 'tex'], note: 'one or two plain sentences' }],
  live(ctx) {            // optional for tensors; omit for concept:* entries
    // ctx: {key, trace, cfg, model, ids, label(i), vocab(v), layer, head, row, col}
    //   2-D tensor [T, X]: row = token t, col = x.  3-D [B, T, X]: head = b, row = t, col = x.
    // Recompute THIS cell from upstream tensors / weights (never just read it back),
    // and render the equation with the actual numbers.
    return { tex: '…', result, expected: /* the traced value of this cell */, note: 'optional' };
  },
}
```

Helpers in `src/core/math/util.js`: `at2, at3, row2, row3, dh, kvHead, attnInput, mlpInput,
streamBefore, wrow (row of an nn.Linear weight), dotTerms, listTerms, dot, num`.

## Notation (use it everywhere)

- Row vectors: token t's state is x_t ∈ ℝ^d; matrices stack tokens as rows (T × d).
- nn.Linear stores W ∈ ℝ^{out×in}; write y = x W^⊤ + b. Element: y_{t,o} = Σ_c x_{t,c} W_{o,c} + b_o.
- Indices in live instantiations are the app's 0-based indices; general formulas use Σ_c, Σ_j
  without bounds or with explicit 1..d bounds. Be consistent inside one entry.
- Macros (coloured like the app): `\Q \K \V \Wq \Wk \Wv \Wo \WE \WU \Wup \Wdown \Wgate \Aw \Mk \Pe \xres`,
  operators `\softmax \LN \RMSNorm \GELU \SiLU \ReLU \concat \CE \Attn \MLP`, `\R` for ℝ.
  Use `\kc{q}{…}` to colour any expression with a component kind.
- Conventions that must match reference/model.py: LayerNorm uses the biased variance and
  ε = norm_eps; RMSNorm x / sqrt(mean(x²) + ε) · γ; GELU is the tanh approximation
  0.5x(1 + tanh(√(2/π)(x + 0.044715x³))); RoPE rotates interleaved pairs (2k, 2k+1) by
  t·θ_k, θ_k = base^(−2k/d_h); ALiBi slopes m_h = 2^(−8(h+1)/H), bias −m_h|i−j|; the causal
  mask is j ≤ i; pre-LN: x + Attn(LN(x)); post-LN: LN(x + Attn(x)); logits = x W_U^⊤ with
  W_U = W_E when tied; cross-entropy is the mean over positions with a target.

## Checks

```bash
node tests/math.mjs --only src/core/math/<topic>.js   # your file alone
node tests/math.mjs                                    # everything
```
The test compiles every LaTeX string with KaTeX (throwOnError) and runs every live() on 6
architecture fixtures and every trained preset, at several cells, requiring |result − expected|
≤ 1e-6·max(1, |expected|). All checks must pass.

// Interview cheat sheet: a scrollable reference for the model you have loaded (a k-doc, not a canvas).
//   1. Shapes: every traced tensor of the last forward pass, grouped, with its symbolic shape, the concrete
//      shape from the trace (checked against the symbols) and its line in reference/model.py. Rows zoom.
//   2. Formulas: the math catalogue, the forward pass first and then the concepts, which start collapsed. Tensor
//      cards recompute one value at the selected layer, head and query and check it against the trace. A card for
//      a part the config does not compute is hidden, so the forward pass lists only what this model runs.
//   3. Flashcards: 42 interview questions with a 3-D flip, tagged by topic, each linked to its node.
//   4. Quiz: a seeded shuffle, knew it or didn't, and a score.
// The four sections are the story steps: the bottom bar lists them, a step scrolls to its section, and scrolling
// moves the bar. The document scrolls between the atlas bands (the breadcrumb above, the story bar below).
// ATLAS_GUIDE (Design canvas) lets a reference scene scroll "still with the same restraint", so its type, colour and
// motion follow the guide. It has no caption zone: the live line above the formulas plays that role. It does not
// auto-advance, because a reference is read at the reader's pace; the story bar moves between sections on request.
import { sceneHeader, tokenPicker, headPicker, checkBadge, chip, card, appear, useStyle, MOTION } from '../kit.js';
import { h, texHTML, reducedMotion } from '../../ui/index.js';
import { mathFor, runLive } from '../../core/math/index.js';
import { infoFor, parseKey, shapePairs, TRACE_INFO } from '../../core/describe.js';
import { lineForKey } from '../../core/codemap.js';
import { nodeForKey, nodeInfo, validFor } from '../nodes.js';
import { paramSpecs, cacheBytes, dHead, mulberry32 } from '../../engine/index.js';

useStyle('cheatsheet', `
/* Every grid here gets a bounded track: a wide KaTeX line must scroll inside its card, not widen the page. */
.cs-doc .k-doc-inner, .cs-sec, .cs-fc, .cs-face, .cs-score, .cs-quiz, .cs-tile, .cs-live, .cs-grid .k-card, .cs-flip-inner { grid-template-columns: minmax(0, 1fr); }
/* The document scrolls in the band stage2d leaves for the atlas chrome: from under the breadcrumb (y 64) to above
   the story bar (y 816 of 900). Its column is the canvas content column (design x 110 … 1490 at the canvas scale k),
   so the text lines up with the canvas scenes, and the title is the canvas title at the same scale. */
.cs-doc { --cs-top: 64px; --cs-bottom: 84px; --cs-k: min(calc((100vw - 48px) / 1600), calc((100vh - 148px) / 900)); top: var(--cs-top); bottom: var(--cs-bottom); padding: 32px 24px 40px; }
.cs-doc:focus-visible { outline: 2px solid var(--focus); outline-offset: -4px; }
.cs-doc .k-doc-inner { gap: 64px; max-width: calc(1380 * var(--cs-k)); }
.cs-doc .k-header { width: auto; }
.cs-doc .k-scene-title { font-size: max(28px, calc(46 * var(--cs-k))); }
.cs-doc .k-scene-sub { max-width: 1060px; }
.cs-doc .k-eyebrow { flex-wrap: wrap; row-gap: 8px; min-height: 0; }
/* The last section is one band tall less the pin offset and the bottom padding, so every story step can pin its heading to the top. */
.cs-doc .k-doc-inner > .cs-sec:last-child { min-height: calc(100vh - var(--cs-top) - var(--cs-bottom) - 72px); }
.cs-btn { font: 600 14px/1 var(--font-body); padding: 10px 16px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--fg-2); cursor: pointer; transition: border-color .2s, color .2s, background-color .2s; }
.cs-btn:hover { color: var(--fg); border-color: var(--line-strong); }
.cs-btn.primary { background: var(--accent); border-color: var(--accent); color: var(--bg); }
.cs-btn.primary:hover { color: var(--bg); filter: brightness(1.08); }
.cs-sec { display: grid; gap: 22px; align-content: start; scroll-margin-top: 24px; min-width: 0; }
.cs-sec-head { display: grid; gap: 8px; align-content: start; }
.cs-sec-title { margin: 0; font: 650 30px/1.15 var(--font-display); letter-spacing: -.015em; }
.cs-lede { margin: 0; max-width: 70ch; font-size: 19px; line-height: 1.5; color: var(--fg-2); }
.cs-lede b { color: var(--fg); font-weight: 650; }
.cs-scroll { overflow-x: auto; max-width: 100%; padding: 2px; }
.cs-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }
.cs-tile { display: grid; gap: 8px; align-content: start; padding: 16px 18px; border-radius: 14px; background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow-1); }
.cs-tile-k { font: 600 12px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .08em; }
.cs-tile-v { font: 650 30px/1.1 var(--font-display); letter-spacing: -.01em; font-variant-numeric: tabular-nums; }
.cs-tile-s { font-size: 14px; line-height: 1.45; color: var(--fg-2); }
.cs-summary { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.cs-count { font: 500 13px/1 var(--font-mono); color: var(--fg-3); }
.cs-key-note { margin: 0; font: 500 13px/1.4 var(--font-mono); color: var(--fg-3); }
.cs-table-wrap { overflow-x: auto; border-radius: 14px; border: 1px solid var(--line); background: var(--surface); box-shadow: var(--shadow-1); }
.cs-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.cs-table th { text-align: left; font: 600 11px/1 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--fg-3); padding: 12px 14px; background: var(--surface-2); border-bottom: 1px solid var(--line); white-space: nowrap; }
.cs-table td { padding: 8px 14px; border-top: 1px solid var(--line); white-space: nowrap; vertical-align: middle; }
.cs-group td { padding: 16px 14px 7px; font: 650 12px/1 var(--font-mono); letter-spacing: .06em; text-transform: uppercase; color: var(--fg-3); background: var(--surface); }
.cs-row { cursor: zoom-in; transition: background-color .15s; }
.cs-row:hover td { background: var(--surface-2); }
.cs-row:focus-visible { outline: 2px solid var(--focus); outline-offset: -2px; }
.cs-key { font: 600 14px/1.2 var(--font-mono); color: var(--fg); }
.cs-key small { display: block; margin-top: 3px; font: 500 12px/1.2 var(--font-body); color: var(--fg-3); }
.cs-kind { display: inline-block; width: 8px; height: 8px; margin-right: 9px; border-radius: 2px; background: var(--kind, var(--fg-3)); vertical-align: 1px; }
.cs-mono { font: 500 14px/1.2 var(--font-mono); color: var(--fg-2); }
.cs-line { font: 500 13px/1.2 var(--font-mono); color: var(--fg-3); }
.cs-ok { color: var(--good); font-weight: 700; margin-left: 6px; }
.cs-bad { color: var(--bad); font-weight: 700; margin-left: 6px; }
.cs-pickers { display: flex; flex-wrap: wrap; gap: 14px 24px; align-items: center; }
.cs-pickers > * { min-width: 0; max-width: 100%; }
.cs-pickers .k-picker-wrap { flex-wrap: wrap; row-gap: 6px; }
.cs-pickers .k-picker { flex-wrap: wrap; row-gap: 4px; }
.cs-grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 380px), 1fr)); }
.cs-grid > .cs-wide { grid-column: 1 / -1; }
.cs-grid .k-card { display: grid; align-content: start; gap: 10px; min-width: 0; }
.cs-grid .k-card[hidden] { display: none; }
.cs-grid .k-card-title { font-size: 19px; line-height: 1.2; }
.cs-grid .k-card-sub { margin-top: 0; font: 500 12px/1.4 var(--font-mono); color: var(--fg-3); }
.cs-grid .k-card-sub span { color: var(--fg-2); }
.cs-math { overflow-x: auto; overflow-y: hidden; font-size: 17px; }
.cs-math .katex-display { margin: .2em 0; }
.cs-live { display: grid; gap: 6px; padding: 10px 12px; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--line); overflow-x: auto; min-width: 0; }
.cs-live .katex-display { margin: .1em 0; }
.cs-live-empty { font-size: 13px; line-height: 1.45; color: var(--fg-3); }
.cs-caption { font-size: 19px; line-height: 1.5; color: var(--fg-2); text-wrap: pretty; }
.cs-caption b { color: var(--fg); font-weight: 650; }
.cs-more > summary { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 6px 0; cursor: pointer; list-style: none; border-radius: 8px; }
.cs-more > summary::-webkit-details-marker { display: none; }
.cs-more > summary:focus-visible { outline: 2px solid var(--focus); outline-offset: 4px; }
.cs-more > summary::after { content: 'Show'; font: 600 12px/1 var(--font-mono); color: var(--fg-2); }
.cs-more[open] > summary::after { content: 'Hide'; }
.cs-filter-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; font: 600 12px/1 var(--font-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--fg-3); }
.cs-select { font: 600 14px/1 var(--font-body); letter-spacing: normal; text-transform: none; color: var(--fg); padding: 10px 36px 10px 14px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); appearance: none; -webkit-appearance: none; cursor: pointer; background-image: linear-gradient(45deg, transparent 50%, var(--fg-2) 50%), linear-gradient(135deg, var(--fg-2) 50%, transparent 50%); background-position: calc(100% - 19px) 52%, calc(100% - 14px) 52%; background-size: 5px 5px; background-repeat: no-repeat; transition: border-color .2s; }
.cs-select:hover { border-color: var(--line-strong); }
.cs-select:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.cs-deck { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 420px), 1fr)); }
.cs-fc { display: grid; grid-template-rows: 1fr auto; gap: 8px; min-width: 0; }
.cs-flip { position: relative; height: 100%; perspective: 1400px; cursor: pointer; border-radius: 16px; outline: none; }
.cs-flip:focus-visible { box-shadow: 0 0 0 2px var(--focus); }
.cs-flip-inner { display: grid; height: 100%; transform-style: preserve-3d; transition: transform ${MOTION.move}ms ${MOTION.ease}; }
.cs-flip.on .cs-flip-inner { transform: rotateY(180deg); }
.cs-face { grid-area: 1 / 1; display: grid; align-content: start; gap: 12px; padding: 18px 20px 16px; border-radius: 16px; background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow-1); backface-visibility: hidden; -webkit-backface-visibility: hidden; }
.cs-face.cs-back { transform: rotateY(180deg); background: var(--surface-2); border-color: color-mix(in srgb, var(--accent) 25%, var(--line)); }
.cs-face-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; font: 600 12px/1 var(--font-mono); color: var(--fg-3); }
.cs-q { font: 650 21px/1.25 var(--font-display); letter-spacing: -.01em; text-wrap: balance; }
.cs-a { margin: 0; font-size: 16px; line-height: 1.55; color: var(--fg); text-wrap: pretty; }
.cs-hint { font: 500 12px/1 var(--font-mono); color: var(--fg-3); }
.cs-fc-foot { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
.cs-topic { display: inline-flex; align-items: center; gap: 8px; font: 600 12px/1 var(--font-mono); color: var(--fg-2); }
.cs-topic i { width: 8px; height: 8px; border-radius: 50%; background: var(--fg-3); }
.cs-zoom { font: 600 12px/1 var(--font-mono); color: var(--fg-2); background: none; border: 0; padding: 7px 8px; border-radius: 8px; cursor: zoom-in; }
.cs-zoom:hover { color: var(--accent); background: var(--surface-2); }
.cs-quiz { display: grid; gap: 18px; width: 100%; max-width: 760px; margin-inline: auto; }
.cs-fc.big { gap: 14px; }
.cs-fc.big .cs-q { font-size: 28px; }
.cs-fc.big .cs-a { font-size: 18px; }
.cs-progress { display: grid; grid-template-columns: auto 1fr auto; gap: 14px; align-items: center; font: 600 13px/1 var(--font-mono); color: var(--fg-3); }
.cs-track { height: 6px; border-radius: 999px; background: var(--surface-3); overflow: hidden; }
.cs-track i { display: block; height: 100%; width: 0; border-radius: 999px; background: var(--accent); transition: width ${MOTION.move}ms ${MOTION.ease}; }
.cs-actions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; align-items: center; min-height: 46px; }
.cs-score { display: grid; gap: 14px; justify-items: center; text-align: center; padding: 30px 22px; border-radius: 18px; background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow-1); }
.cs-score-num { font: 700 56px/1 var(--font-display); letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.cs-missed { display: grid; gap: 6px; width: 100%; }
.cs-missed button { display: flex; gap: 10px; align-items: baseline; text-align: left; font: 500 15px/1.35 var(--font-body); color: var(--fg-2); background: var(--surface-2); border: 1px solid var(--line); border-radius: 10px; padding: 9px 12px; cursor: zoom-in; }
.cs-missed button:hover { color: var(--fg); border-color: var(--line-strong); }
.cs-missed b { font: 600 12px/1 var(--font-mono); color: var(--fg-3); flex: none; }
@media (max-width: 640px) {
  .cs-doc { --cs-top: 128px; --cs-bottom: 120px; padding: 24px 16px 40px; }   /* the breadcrumb wraps to two lines on a phone */
  .cs-doc .k-doc-inner { gap: 48px; max-width: none; }
  .cs-doc .k-scene-title { font-size: 30px; }
  .cs-sec-title { font-size: 26px; }
  .cs-q { font-size: 19px; }
  .cs-fc.big .cs-q { font-size: 22px; }
}
@media (prefers-reduced-motion: reduce) {
  .cs-flip-inner, .cs-track i { transition: none !important; }
}
`);

// ----------------------------------------------------------------- content

const TOPICS = ['Attention', 'Norms & residuals', 'Positions', 'Inference', 'Training', 'Architecture', 'Tokens', 'Vision', 'Interpretability'];
const OUTPUT_KEYS = new Set(['lnf', 'logits', 'probs', 'loss']);

// Flashcards. `node` is the part of the model the answer explains (a function when it depends on the config).
const DECK = [
  // Attention
  { topic: 'Attention', node: 'scores.0.0',
    q: 'Why divide the attention scores by √d_h?',
    a: 'If q and k have independent entries with zero mean and unit variance, their dot product has variance d_h, so raw scores grow like √d_h. Dividing by √d_h brings each score back to variance about 1. Without it, softmax saturates into an almost one-hot distribution and its gradients become tiny. The factor is model.py line 159, and attn_scale switches it off in this app.' },
  { topic: 'Attention', node: 'head.0.0',
    q: 'Why use several heads instead of one wide head?',
    a: 'Each head has its own W_Q, W_K and W_V slices of width d_h = d / H, so each can learn a different pattern, for example attending to the previous token in one head and to a matching earlier word in another. A single head of width d can form only one pattern per query. The total cost is about the same, because each head is narrower. The head outputs are concatenated and mixed back to width d by W_O.' },
  { topic: 'Attention', node: 'attention.0',
    q: 'What does self-attention cost as the sequence length T grows?',
    a: 'Forming QKᵀ and the weighted sum of values costs O(T²·d) operations per layer. The naive implementation also stores a T×T score matrix for every head, which is O(H·T²) memory. The projections cost only O(T·d²), so the quadratic term takes over as the context grows. That quadratic term is the reason for IO-aware kernels and for sparse or linear attention variants.' },
  { topic: 'Attention', node: 'attention.0',
    q: 'What does FlashAttention change, and what does it leave alone?',
    a: 'FlashAttention computes exactly the same softmax attention. It tiles Q, K and V into blocks that fit in on-chip SRAM and keeps a running softmax, so the T×T matrix is never written to GPU high-bandwidth memory. Extra memory falls from O(T²) to O(T), and wall-clock time improves because memory traffic falls. The arithmetic is still O(T²·d), and the result is not an approximation.' },
  { topic: 'Attention', node: (cfg) => (cfg.n_kv_head < cfg.n_head ? 'gqa.0' : 'attention.0'),
    q: 'How do MHA, GQA and MQA differ?',
    a: 'In multi-head attention (MHA) every query head has its own key and value head. Multi-query attention (MQA) shares one key and one value head across all query heads. Grouped-query attention (GQA) splits the query heads into groups, and each group shares one key and value head. Fewer KV heads shrink the KV cache and the memory traffic of every decode step, at a small cost in quality.' },
  // Norms & residuals
  { topic: 'Norms & residuals', node: 'block.0',
    q: 'Pre-LN or post-LN, and why do most LLMs use pre-LN?',
    a: 'Post-LN computes x ← LN(x + f(x)), so the residual stream is normalised after every addition. Pre-LN computes x ← x + f(LN(x)), which leaves the identity path unnormalised. Pre-LN trains stably at depth without a long learning-rate warm-up, which is why GPT-2, LLaMA and most later LLMs use it. Post-LN can match or slightly beat pre-LN when it trains, but it needs careful warm-up.' },
  { topic: 'Norms & residuals', node: 'residual.0.1',
    q: 'Why are residual connections essential in deep Transformers?',
    a: 'A residual connection adds a sub-layer’s output to its input. The identity path passes the signal and its gradient through unchanged, so each block only has to learn a correction. Without it, deep stacks are very hard to optimise. Here the residual stream is x₀ plus the running sum of attention and MLP outputs, and every layer reads from it and writes to it.' },
  { topic: 'Norms & residuals', node: 'norm.0.1',
    q: 'How does RMSNorm differ from LayerNorm, and why do LLaMA-style models use it?',
    a: 'LayerNorm subtracts the per-token mean, divides by the standard deviation, and applies a learned gain and bias. RMSNorm skips the mean subtraction and the bias and divides by the root mean square, leaving only a gain. It needs one reduction instead of two and has fewer parameters, and in practice it matches LayerNorm quality, so LLaMA and many later models use it.' },
  { topic: 'Norms & residuals', node: 'norm.0.1',
    q: 'Why is BatchNorm a poor fit for Transformers?',
    a: 'BatchNorm normalises each feature across the batch, so its statistics depend on the other examples in the batch. Text has variable lengths and padding, batches can be small, and decoding runs one token at a time, so batch statistics become unreliable. LayerNorm and RMSNorm normalise each token vector on its own, so training and inference behave the same.' },
  // Positions
  { topic: 'Positions', node: 'positional',
    q: 'How do absolute, RoPE and ALiBi encode position, and how do they handle longer inputs?',
    a: 'Absolute position vectors, learned or sinusoidal, are added to the token embeddings, so positions beyond those seen in training are out of distribution. RoPE rotates each query and key pair by an angle proportional to its position. ALiBi adds a head-specific penalty of −m_h·|i−j| to the scores and uses no position vectors at all. ALiBi was designed to extrapolate to longer inputs, and RoPE encodes relative offsets directly.' },
  { topic: 'Positions', node: 'positional',
    q: 'Why does RoPE make attention depend only on relative position?',
    a: 'RoPE applies a rotation R(tθ) to the query at position t and R(sθ) to the key at position s. Because rotations are orthogonal, the dot product becomes ⟨q, R((s−t)θ)k⟩, which depends only on the offset s−t. Each pair of dimensions rotates at its own frequency θ_k = base^(−2k/d_h). The model can therefore use fast and slow frequencies together.' },
  { topic: 'Positions', node: 'positional',
    q: 'How do position interpolation, NTK-aware scaling and YaRN extend a RoPE model’s context?',
    a: 'Position interpolation (PI) divides positions by the extension factor so that new positions fall inside the trained range, and it needs a short fine-tune. NTK-aware scaling raises the RoPE base instead, which stretches the slow frequencies and keeps the fast ones, so it needs less tuning. YaRN combines per-frequency interpolation with a temperature correction on the attention logits, and it is a widely used recipe for long contexts.' },
  // Inference
  { topic: 'Inference', node: 'generation',
    q: 'How much memory does the KV cache need, and what determines it?',
    a: 'The cache stores K and V for every layer, every KV head and every past token: bytes = 2 · L · H_kv · d_h · T · b, where b is bytes per value (2 for bf16). It grows linearly with context length and with batch size. It depends on the number of KV heads rather than query heads, which is why GQA and MQA shrink it.' },
  { topic: 'Inference', node: 'generation',
    q: 'Why do we cache keys and values but not queries?',
    a: 'Each decode step needs a query only for the newest token, and that query is used once, to attend over all earlier positions. The keys and values of earlier positions never change, because the causal mask means they depend only on earlier tokens. Caching them avoids recomputing the whole prefix at every step, while the new query costs one row of attention.' },
  { topic: 'Inference', node: 'sampling',
    q: 'What do temperature, top-k and top-p do when sampling?',
    a: 'Temperature divides the logits by τ before the softmax: τ below 1 sharpens the distribution, and τ approaching 0 becomes greedy decoding. Top-k keeps only the k largest logits and renormalises. Top-p keeps the smallest set of tokens whose probabilities sum to at least p. All three change only how the next token is chosen, not the model’s weights.' },
  { topic: 'Inference', node: 'sampling',
    q: 'What is beam search, and why is it rarely used for chat?',
    a: 'Beam search keeps the B highest-scoring partial sequences at every step, ranked by summed log-probability, and extends each of them. It finds high-likelihood outputs, which suits translation and summarisation. For open-ended chat it tends to produce short, generic text, so sampling with temperature or top-p is the usual choice.' },
  { topic: 'Inference', node: 'generation',
    q: 'How does speculative decoding speed up generation without changing the output distribution?',
    a: 'A small draft model proposes k tokens one at a time, which is cheap. The large target model then scores all k positions in a single parallel forward pass. Each proposed token is accepted with probability min(1, p_target / p_draft), and the first rejected token is replaced by a sample from an adjusted distribution. The output has exactly the target model’s distribution, and the speed-up depends on how often drafts are accepted.' },
  { topic: 'Inference', node: 'generation',
    q: 'What is the difference between prefill and decode, and which one is memory-bound?',
    a: 'Prefill processes the whole prompt in one parallel forward pass and fills the KV cache. It is dominated by large matrix multiplications, so it is usually compute-bound. Decode produces one token per step, reads all the weights and the cache for a single new position, and is usually memory-bandwidth-bound. Batching many decode requests together is the main way to raise throughput.' },
  // Training
  { topic: 'Training', node: 'mask.0.0',
    q: 'Why does the causal mask let a decoder train on a whole sequence in one pass?',
    a: 'The causal mask lets position t attend only to positions ≤ t, so each prediction depends only on the true tokens before it. During training the ground-truth sequence is fed in (teacher forcing), so all T next-token predictions come out of one forward pass, and each contributes a loss term. Generation still has to run one token at a time, which is why it is much slower than training.' },
  { topic: 'Training', node: 'embedding',
    q: 'Does GPT-2 use dropout, and does this reference model?',
    a: 'The Hugging Face GPT-2 configuration sets dropout 0.1 on the embeddings, on the attention weights and on each sub-layer output before it is added to the residual stream. Dropout is disabled in eval mode, so inference is deterministic. Many large pretraining runs now use no dropout, because the corpus is huge and most tokens are seen about once. This reference model has no dropout at all.' },
  { topic: 'Training', node: 'lm-head',
    q: 'What does label smoothing change?',
    a: 'Label smoothing replaces the one-hot target with a mixture: the true token keeps 1 − ε and the remaining ε is spread across the other tokens. The loss can no longer be driven to zero by pushing one logit to infinity, which keeps the model less overconfident. A typical value is ε = 0.1. It changes the loss, not the architecture.' },
  { topic: 'Training', node: 'block.0',
    q: 'How does gradient checkpointing save memory, and what does it cost?',
    a: 'Instead of keeping every intermediate activation of the forward pass, the model keeps only the inputs of selected blocks and recomputes the rest during the backward pass. Activation memory drops from every intermediate tensor of every block to the block inputs plus one block’s activations. The cost is roughly one extra forward pass, about a third more compute per step.' },
  { topic: 'Training', node: 'model',
    q: 'What is the difference between bf16 and fp16?',
    a: 'Both are 16-bit formats. fp16 has 5 exponent bits and 10 mantissa bits, so it is more precise, but its range is small (the maximum is 65,504) and training needs loss scaling to avoid underflow. bf16 has the same 8 exponent bits as fp32 and only 7 mantissa bits, so its range matches fp32 and it usually trains without loss scaling.' },
  { topic: 'Training', node: 'scaling',
    q: 'What did Chinchilla change about sizing a model for a compute budget?',
    a: 'Hoffmann et al. (2022) found that, for a fixed compute budget, parameters and training tokens should grow together, at roughly 20 training tokens per parameter. Many earlier large models were trained on far fewer tokens than that for their size. Training compute is about 6·N·D FLOPs for N parameters and D tokens, so the budget fixes the product of the two.' },
  { topic: 'Training', node: 'out-proj.0',
    q: 'Why scale the output projections of each block by 1/√(2N) at initialisation?',
    a: 'Each of the N blocks adds two contributions to the residual stream, one from attention and one from the MLP. Their variances add, so without scaling the stream’s variance would grow with depth. GPT-2 scales o_proj and down_proj by 1/√(2N), so the stream keeps a scale close to that of the embeddings at initialisation. In model.py this is lines 243 to 245.' },
  { topic: 'Training', node: 'lm-head',
    q: 'Why does the model output logits instead of probabilities?',
    a: 'The loss is computed from logits with log-softmax, which is numerically stable because it subtracts the maximum and never forms tiny probabilities explicitly. The softmax is applied where it is needed: in the loss, in sampling with a temperature, or when reading out probabilities. Logits are also the natural input for temperature and top-k, which act on the raw scores.' },
  { topic: 'Training', node: 'lm-head',
    q: 'What is perplexity, and how does it relate to the loss?',
    a: 'Perplexity is exp of the mean cross-entropy in nats, so it is the effective number of equally likely choices the model is hesitating between. A loss of 0 gives perplexity 1, and a uniform guess over V tokens gives perplexity V. It is a per-token measure, so it is only comparable between models that share a tokenisation and a dataset.' },
  { topic: 'Training', node: 'lm-head',
    q: 'What is the next-token loss, and why is the target shifted by one?',
    a: 'Row t of the logits predicts token t+1, so the target for position t is ids[t+1]. The loss is the average negative log-probability of those targets, and the last position is excluded because it has no next token. This shift is the entire training signal of a decoder-only model.' },
  // Architecture
  { topic: 'Architecture', node: 'encdec',
    q: 'What is the difference between decoder-only and encoder-decoder Transformers?',
    a: 'A decoder-only model is a stack of causal self-attention blocks trained to predict the next token, as in GPT and LLaMA. An encoder-decoder model has a bidirectional encoder for the input and a decoder that attends to the encoder through cross-attention, as in the original Transformer, T5 and BART. Decoder-only models are simpler to scale and serve, which is why most current LLMs use them.' },
  { topic: 'Architecture', node: 'encdec',
    q: 'In cross-attention, where do the queries, keys and values come from?',
    a: 'The queries come from the decoder’s hidden states, and the keys and values come from the encoder’s output. The attention matrix therefore has shape T_dec × T_enc. Because the encoder output does not change during decoding, its keys and values can be computed once and reused at every step.' },
  { topic: 'Architecture', node: 'mlp.0',
    q: 'How does a mixture-of-experts layer save compute?',
    a: 'A router scores E expert MLPs for each token and sends the token to only the top k. Compute per token scales with k, while total parameters scale with E, so capacity grows without a matching rise in FLOPs. Training needs a load-balancing term so tokens do not all go to the same expert, and serving needs enough memory to hold every expert.' },
  { topic: 'Architecture', node: 'block.0',
    q: 'Roughly how many parameters does one Transformer block have?',
    a: 'With standard multi-head attention and an MLP of width 4d, the block has four d×d attention matrices (W_Q, W_K, W_V, W_O) and two d×4d MLP matrices, so about 4d² + 8d² = 12d² parameters per layer. Biases and norm gains add only O(d). The token embedding adds V·d more, and weight tying lets the output projection reuse it.' },
  { topic: 'Architecture', node: 'lm-head',
    q: 'What does weight tying do?',
    a: 'Weight tying uses the same matrix for the token embedding W_E and the output projection W_U, so the logits are x·W_Eᵀ. It removes V·d parameters and ties the input and output representations of each token. In model.py, tie_weights makes lm_head.weight the same tensor as wte.weight.' },
  { topic: 'Architecture', node: 'neurons.0',
    q: 'What does the feed-forward (MLP) block do in a Transformer?',
    a: 'The MLP applies the same two-layer network to every token independently, and it holds most of the parameters in each block. Geva et al. read each hidden unit as a key-value memory: the input weights match a pattern, and the output weights write a vector back into the stream. Attention moves information between tokens, while the MLP transforms each token’s representation.' },
  { topic: 'Architecture', node: 'activation.0',
    q: 'Why do modern LLMs use SwiGLU instead of GELU?',
    a: 'SwiGLU multiplies a SiLU-gated projection by a second linear projection, so one unit gates another, which gives a more expressive non-linearity at the same width. It needs a third matrix, so the hidden width is usually reduced to about 8d/3, which keeps the parameter count close to a GELU MLP of width 4d. LLaMA and many later models use it for that quality per parameter.' },
  // Tokens
  { topic: 'Tokens', node: 'tokens',
    q: 'How does byte-pair encoding build its vocabulary?',
    // The answer depends on the loaded tokenizer: its merges and entries are read from the model, not written here.
    a: (tk) => (tk.kind === 'bpe'
      ? `BPE starts from a base alphabet of 256 byte values, so any text can be encoded. It repeatedly merges the most frequent adjacent pair of symbols in the training corpus into a new symbol, and each merge adds one entry to the vocabulary. This model learned ${fmtInt(tk.merges)} merges, so its ${fmtInt(tk.size)} entries are the 256 bytes, the merges${specialsText(tk)}. Common words become single tokens while rare words split into pieces, so there are no unknown tokens.`
      : `BPE starts from a base alphabet, usually bytes or characters. It repeatedly merges the most frequent adjacent pair of symbols in the corpus into a new symbol, and each merge adds one entry to the vocabulary. Common words become single tokens while rare words split into pieces, so there are no unknown tokens. This model uses one token per character: its ${fmtInt(tk.size)} entries are the characters, with no merges.`) },
  { topic: 'Tokens', node: 'tokens',
    q: 'Why do LLMs use subword tokens rather than characters or whole words?',
    a: 'Whole words need a huge vocabulary and produce unknown tokens for new words. Characters keep the vocabulary tiny but make sequences several times longer, and attention cost grows with the square of the length. Subwords sit between the two: a moderate vocabulary, shorter sequences and no unknown tokens.' },
  // Vision
  { topic: 'Vision', node: 'vision',
    q: 'How does a Vision Transformer turn an image into tokens?',
    a: 'The image is cut into a grid of fixed-size patches, for example 16×16 pixels. Each patch is flattened and linearly projected to width d, and a position embedding is added, so the image becomes a sequence of patch tokens. A standard Transformer encoder then processes them with bidirectional attention and no causal mask.' },
  { topic: 'Vision', node: 'vision',
    q: 'How do vision-language models feed images to a language model?',
    a: 'A vision encoder turns the image into patch or region features, and a small projector maps them into the language model’s embedding space. The projected features are inserted into the sequence as extra tokens, next to the text tokens. The language model then attends to them like any other tokens, so the rest of the decoder is unchanged.' },
  // Interpretability
  { topic: 'Interpretability', node: (cfg) => (cfg.n_layer > 1 ? 'attention.1' : 'attention.0'),
    q: 'What is an induction head?',
    a: 'An induction head completes repeated sequences. If the pattern [A][B] occurred earlier and A appears again, it attends to the token after the earlier A and copies B. In a two-layer model it usually works with a previous-token head in layer 0, which tags each position with the token before it, and an induction head in layer 1 that matches the current token against those tags.' },
  { topic: 'Interpretability', node: 'lm-head',
    q: 'What is the logit lens?',
    a: 'The logit lens applies the model’s own final norm and unembedding matrix to the residual stream after an intermediate layer. This gives a next-token guess at that depth, so watching it across layers shows where the answer forms. It is an approximation, because intermediate streams were not trained to be read this way.' },
  { topic: 'Interpretability', node: 'softmax.0.0',
    q: 'What is an attention sink?',
    a: 'In many trained LLMs much of the attention in a head lands on the first token, even when that token carries little content. The softmax must put its weight somewhere, so the first token becomes a sink that absorbs the surplus with a small value vector. Streaming-attention methods keep the first few tokens in the KV cache for this reason.' },
];

// Formula sheet. Each entry: [catalogue key (L{i} = the selected layer), node(layer, head, cfg), note, applies(cfg)].
// `applies` hides a card whose part the config does not compute (no position table under RoPE, no ln1 after a
// post-LN block). The core entries follow the forward pass; the concepts are the derivations and cost formulas.
const LEARNED_OR_SINUSOIDAL = (c) => c.pos_encoding === 'learned' || c.pos_encoding === 'sinusoidal';
const PRE_LN = (c) => c.norm_placement !== 'post';
const POST_LN = (c) => c.norm_placement === 'post';
const ROPE = (c) => c.pos_encoding === 'rope';
const ALIBI = (c) => c.pos_encoding === 'alibi';
const FORMULAS_CORE = [
  ['emb.tok', () => 'token-embedding'],
  ['emb.pos', () => 'positional', 'learned or sinusoidal', LEARNED_OR_SINUSOIDAL],
  ['emb.out', () => 'embedding'],
  ['mask', (l, hd) => `mask.${l}.${hd}`],
  ['L{i}.ln1', (l) => `norm.${l}.1`, null, PRE_LN],
  ['L{i}.attn.q', (l) => `qkv.${l}`],
  ['L{i}.attn.k', (l) => `qkv.${l}`],
  ['L{i}.attn.v', (l) => `qkv.${l}`],
  ['L{i}.attn.q_rot', () => 'positional', null, ROPE],
  ['L{i}.attn.scores', (l, hd) => `scores.${l}.${hd}`],
  ['alibi', () => 'positional', null, ALIBI],
  ['L{i}.attn.scaled', (l, hd) => `scores.${l}.${hd}`],
  ['L{i}.attn.masked', (l, hd) => `mask.${l}.${hd}`],
  ['L{i}.attn.probs', (l, hd) => `softmax.${l}.${hd}`],
  ['L{i}.attn.ctx', (l, hd) => `mix.${l}.${hd}`],
  ['L{i}.attn.out', (l) => `out-proj.${l}`],
  ['L{i}.add1', (l) => `residual.${l}.1`, null, POST_LN],
  ['L{i}.resid_mid', (l) => `residual.${l}.1`],
  ['L{i}.ln2', (l) => `norm.${l}.2`, null, PRE_LN],
  ['L{i}.mlp.up', (l) => `mlp.${l}`],
  ['L{i}.mlp.act', (l) => `activation.${l}`],
  ['L{i}.mlp.out', (l) => `neurons.${l}`],
  ['L{i}.add2', (l) => `residual.${l}.2`, null, POST_LN],
  ['L{i}.resid_out', (l) => `residual.${l}.2`],
  ['lnf', () => 'final-norm', null, PRE_LN],
  ['logits', () => 'lm-head'],
  ['probs', () => 'sampling'],
  ['loss', () => 'lm-head'],
];
const FORMULAS_CONCEPT = [
  ['concept:one-hot-lookup', () => 'token-embedding'],
  ['concept:sinusoidal', () => 'positional', 'sinusoidal'],
  ['concept:rope-relative', () => 'positional', 'RoPE'],
  ['concept:alibi', () => 'positional', 'ALiBi'],
  ['concept:weight-tying', () => 'lm-head'],
  ['concept:mha-params', (l) => `attention.${l}`],
  ['concept:gqa', (l, hd, c) => (c.n_kv_head < c.n_head ? `gqa.${l}` : `attention.${l}`)],
  ['concept:gelu', (l) => `activation.${l}`],
  ['concept:swiglu', (l) => `activation.${l}`, 'SwiGLU'],
  ['concept:mlp-params', (l) => `mlp.${l}`],
  ['concept:key-value-memory', (l) => `neurons.${l}`],
  ['concept:layernorm-backward', (l) => `norm.${l}.1`],
  ['concept:pre-vs-post', (l) => `block.${l}`],
  ['concept:residual-stream', () => 'model'],
  ['concept:cross-entropy-gradient', () => 'lm-head'],
  ['concept:perplexity', () => 'lm-head'],
  ['concept:sampling', () => 'sampling'],
  ['concept:kv-cache', () => 'generation'],
  ['concept:flops', () => 'scaling'],
  ['concept:attention-complexity', (l) => `attention.${l}`],
];

// ----------------------------------------------------------------- helpers

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmtInt = (v) => Math.round(v).toLocaleString('en-US');
/** The special tokens of a byte-level BPE vocabulary: its entries that are neither one of the 256 bytes nor a merge. */
const specialsText = (tk) => {
  const n = Math.max(0, tk.size - 256 - tk.merges);
  return n ? ` and ${n} special token${n === 1 ? '' : 's'}` : '';
};
/** Bytes in binary units, named exactly: 40,960 bytes is 40 KiB (the shared fmtBytes labels the same unit KB). */
const fmtKiB = (b) => (b < 1024 ? `${fmtInt(b)} bytes` : b < 1048576 ? `${+(b / 1024).toFixed(1)} KiB` : `${+(b / 1048576).toFixed(2)} MiB`);
const resolveNode = (id, cfg) => (validFor(id, cfg) ? id : 'model');
const cardNode = (c, cfg) => resolveNode(typeof c.node === 'function' ? c.node(cfg) : c.node, cfg);

/** The attention type of a config, in the notation of the chips and the maths. */
const attnName = (cfg) => (cfg.n_kv_head === cfg.n_head ? 'MHA' : cfg.n_kv_head === 1 ? `MQA ${cfg.n_head}:1` : `GQA ${cfg.n_head}:${cfg.n_kv_head}`);

/** The table group of a trace key: layers, embeddings, per-config constants, the output side. */
function groupOf(key) {
  const m = /^L(\d+)\./.exec(key);
  if (m) return `Layer ${m[1]}`;
  if (key.startsWith('emb.')) return 'Embeddings';
  if (key === 'mask' || key === 'alibi') return 'Constants';   // fixed by the config, not learned tensors
  if (OUTPUT_KEYS.has(key)) return 'Output';
  return 'Other';
}

/** A seeded Fisher-Yates shuffle: the same seed always gives the same order. */
function shuffled(list, seed) {
  const rnd = mulberry32(seed);
  const a = list.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Count a number up (or down) to `to`, formatted by `format`. Reduced motion jumps to the end. */
function tweenText(el, to, format) {
  cancelAnimationFrame(el._raf);
  const from = el._v ?? to;
  el._v = to;
  if (reducedMotion() || from === to) { el.textContent = format(to); return; }
  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min(1, (now - t0) / MOTION.move), e = 1 - (1 - p) ** 3;
    el.textContent = format(from + (to - from) * e);
    if (p < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}

// Trace keys that are attention matrices (per head: [H, T, T], or [T, T] for the mask). Their column is a
// key, so a live card reads the strongest key for its query rather than the diagonal.
const ATTN_KEYS = new Set(['L{i}.attn.scores', 'L{i}.attn.scaled', 'L{i}.attn.biased', 'L{i}.attn.masked', 'L{i}.attn.probs', 'alibi', 'mask']);

/** The key that query q of head hd (layer l) gives the most weight to. */
function strongestKey(st, l, hd, q) {
  const p = st.trace[`L${l}.attn.probs`], T = p.shape[1], off = (hd * T + q) * T;
  let best = 0;
  for (let j = 1; j < T; j++) if (p.data[off + j] > p.data[off + best]) best = j;
  return best;
}

/**
 * The trace cell a live formula is evaluated at: the selected head, the selected query row and a column
 * that means what the tensor's columns mean (a key, a vocabulary entry, the head's first projection column,
 * or the first dimension).
 */
function cellFor(st, key, l, t0, hd) {
  const t = st.trace[key];
  if (!t) return null;
  const { pattern } = parseKey(key), cfg = st.config, width = dHead(cfg);
  if (ATTN_KEYS.has(pattern)) return { band: hd, r: t0, x: strongestKey(st, l, hd, t0) };
  if (pattern === 'logits' || pattern === 'probs') {              // [T, V]: the predicted token at t0
    const C = t.shape[1];
    let best = 0;
    for (let v = 1; v < C; v++) if (t.data[t0 * C + v] > t.data[t0 * C + best]) best = v;
    return { band: hd, r: t0, x: best };
  }
  if (pattern === 'L{i}.attn.q') return { band: hd, r: t0, x: hd * width };               // the head's first column
  if (pattern === 'L{i}.attn.k' || pattern === 'L{i}.attn.v') {
    const g = Math.floor(hd / (cfg.n_head / cfg.n_kv_head));                           // the KV head that hd reads
    return { band: hd, r: t0, x: g * width };
  }
  return { band: hd, r: t0, x: 0 };                                                    // the first dimension
}

/**
 * The header chips: the parts that change how this model reads, one short chip each. Activation and mask are
 * named in the Shapes tiles; a switched-off residual or attention scale is named here, because it changes formulas.
 */
function configChips(cfg) {
  const attn = cfg.n_kv_head === cfg.n_head ? `${cfg.n_layer} layers · ${cfg.n_head} heads` : `${cfg.n_layer} layers · ${attnName(cfg)}`;
  const pos = { learned: 'learned positions', sinusoidal: 'sinusoidal positions', rope: 'RoPE', alibi: 'ALiBi', none: 'no positions' }[cfg.pos_encoding];
  const chips = [
    [attn, 'resid'], [pos, 'pos'],
    [`${cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm'} · ${cfg.norm_placement}-LN`, 'norm'],
  ];
  if (!cfg.residual) chips.push(['residual off', 'resid']);
  if (!cfg.attn_scale) chips.push(['no 1/√d_h', 'attn']);
  return chips;
}

// ----------------------------------------------------------------- scene

// A step scrolls its heading to this many px below the top of the document's band (y 96 on a desktop, where the band starts at y 64).
const PIN = 32;

export default {
  mount(el, ctx) {
    const cleanups = [];
    const host = (node) => ({ add: (n) => node.append(n), onCleanup: (f) => cleanups.push(f) });
    const tokName = (i) => (ctx.label(i) === '␣' ? 'space' : ctx.label(i));     // the space token reads as “space”, as in the LM-head scene
    // A flashcard answer: a string, or a function of the loaded tokenizer (a character or byte-level BPE model).
    const answerOf = (c) => (typeof c.a === 'function' ? c.a(ctx.state.tokenizer) : c.a);

    const docEl = h('div', { class: 'k-doc cs-doc', role: 'region', tabindex: 0, 'aria-label': 'Interview cheat sheet, a scrollable reference' });
    const inner = h('div', { class: 'k-doc-inner' });
    docEl.append(inner);
    el.append(docEl);

    // ------------------------------------------------------------ header
    const header = sceneHeader(host(inner), {
      chips: configChips(ctx.cfg), title: 'Interview cheat sheet',
      sub: 'Every number here comes from the model you have loaded. The flashcard answers are general.',
    });
    const eyebrow = header.el.querySelector('.k-eyebrow');

    // ------------------------------------------------------------ 1. shapes
    const tiles = {
      params: { v: h('div', { class: 'cs-tile-v' }), s: h('div', { class: 'cs-tile-s' }) },
      block: { v: h('div', { class: 'cs-tile-v' }), s: h('div', { class: 'cs-tile-s' }) },
      kv: { v: h('div', { class: 'cs-tile-v' }), s: h('div', { class: 'cs-tile-s' }) },
    };
    const tileEl = (label, t) => h('div', { class: 'cs-tile' }, h('div', { class: 'cs-tile-k' }, label), t.v, t.s);
    const summaryHost = h('div', { class: 'cs-summary' });
    const tbody = h('tbody');
    const table = h('table', { class: 'cs-table' },
      h('thead', null, h('tr', null, h('th', null, 'Tensor'), h('th', null, 'Symbolic'), h('th', null, 'Concrete'), h('th', null, 'Source'))),
      tbody);
    const secShapes = h('section', { class: 'cs-sec', 'aria-labelledby': 'cs-h-shapes' },
      h('div', { class: 'cs-sec-head' },
        h('h2', { class: 'cs-sec-title', id: 'cs-h-shapes' }, '1 · Shapes of this model'),
        h('p', { class: 'cs-lede' }, 'Every traced tensor of the last forward pass, in execution order. Click a row to zoom to the part that computes it.')),
      h('div', { class: 'cs-tiles' }, tileEl('Parameters', tiles.params), tileEl('One block', tiles.block), tileEl('KV cache', tiles.kv)),
      summaryHost,
      h('p', { class: 'cs-key-note' }, 'T tokens · d model width · H heads · H_kv KV heads · d_h head width · F MLP width · V vocabulary'),
      h('div', { class: 'cs-table-wrap' }, table));

    // ------------------------------------------------------------ 2. formulas
    const pickers = h('div', { class: 'cs-pickers' });
    const layerRow = h('div', { class: 'k-picker-wrap' });
    const headHost = h('div');
    const tokenHost = h('div', { class: 'cs-scroll' });
    pickers.append(layerRow, headHost, tokenHost);
    const hp = headPicker(host(headHost), ctx, { kind: 'attn' });
    tokenPicker(host(tokenHost), ctx, { label: 'query', kind: 'attn', maxWidth: 1080 });
    const caption = h('div', { class: 'cs-caption' });
    const formulas = [];

    function formulaCard(spec, entry) {
      const body = h('div', { class: 'cs-math', html: texHTML(entry.tex, { display: true }) });
      const live = entry.live ? h('div', { class: 'cs-live' }) : null;
      // `sub` must be truthy so the subtitle element exists; renderFormulas() fills it in.
      const el0 = card({ title: entry.title, sub: '&nbsp;', kind: 'norm', node: 'model', body: [body, live] });
      formulas.push({ spec, el: el0, math: body, sub: el0.querySelector('.k-card-sub'), live, entry, applies: spec[3] });
      return el0;
    }

    // A card spans the full row only when its equation is wider than one column.
    function measureWide() {
      for (const f of formulas) f.el.classList.remove('cs-wide');
      for (const f of formulas) if (f.math.scrollWidth > f.math.clientWidth + 2) f.el.classList.add('cs-wide');
    }

    const coreItems = [];
    const conceptItems = [];
    for (const spec of FORMULAS_CORE) {
      const entry = mathFor(spec[0].replace('{i}', '0'));
      if (entry) coreItems.push({ spec, entry });
    }
    for (const spec of FORMULAS_CONCEPT) {
      const entry = mathFor(spec[0]);
      if (entry) conceptItems.push({ spec, entry });
    }
    const formulaCore = h('div', { class: 'cs-grid' }, coreItems.map(({ spec, entry }) => formulaCard(spec, entry)));
    const formulaConcept = h('div', { class: 'cs-grid' }, conceptItems.map(({ spec, entry }) => formulaCard(spec, entry)));
    // The concepts are reference material, so they start collapsed. The forward pass is what the sheet is for.
    const conceptsEl = h('details', { class: 'cs-more' },
      h('summary', null, h('span', { class: 'cs-tile-k' }, `Concepts · ${conceptItems.length} cards`)),
      formulaConcept);
    conceptsEl.addEventListener('toggle', measureWide);
    const secFormulas = h('section', { class: 'cs-sec', 'aria-labelledby': 'cs-h-formulas' },
      h('div', { class: 'cs-sec-head' },
        h('h2', { class: 'cs-sec-title', id: 'cs-h-formulas' }, '2 · Formulas'),
        h('p', { class: 'cs-lede' }, 'Each card is the catalogue equation for one tensor. Tensor cards recompute one value and check it against the trace.')),
      pickers, caption,
      h('div', { class: 'cs-sec-head' }, h('div', { class: 'cs-tile-k' }, 'The forward pass')),
      formulaCore,
      conceptsEl);

    function renderLayerChips() {
      const cfg = ctx.cfg, cur = ctx.state.sel.layer;
      layerRow.replaceChildren(h('span', { class: 'k-picker-label' }, 'layer'), ...Array.from({ length: cfg.n_layer }, (_, i) => {
        const b = h('button', { type: 'button', class: ['k-pick', 'k-pick-head', i === cur && 'sel'], style: { '--kind': 'var(--c-resid)' } }, h('b', null, `L${i}`));
        b.addEventListener('click', () => ctx.store.select({ layer: i }));
        return b;
      }));
    }

    function liveChildren(res) {
      if (!res) return [h('div', { class: 'cs-live-empty' }, 'Not computed by this config.')];
      if (res.error || !res.tex) return [h('div', { class: 'cs-live-empty' }, 'No live value for this cell.')];
      const kids = [h('div', { html: texHTML(res.tex, { display: true }) })];
      if (res.expected !== undefined) kids.push(checkBadge(res.result, res.expected, { label: 'equals the traced value' }));
      if (res.note) kids.push(h('div', { class: 'cs-live-empty' }, res.note));
      return kids;
    }

    function renderFormulas() {
      const st = ctx.state, cfg = st.config, T = st.ids.length;
      const L = clamp(st.sel.layer ?? 0, 0, cfg.n_layer - 1);
      const hd = clamp(st.sel.head ?? 0, 0, cfg.n_head - 1);
      const t0 = clamp(st.sel.token ?? T - 1, 0, T - 1);
      const j = strongestKey(st, L, hd, t0);
      caption.replaceChildren(
        'Live at layer ', `${L}`, ', head ', `${hd}`, '. Query ', h('b', null, `“${tokName(t0)}”`), ' at position ', h('b', null, `${t0}`),
        '. Attention cards read its strongest key, ', h('b', null, `“${tokName(j)}”`), ` at position ${j}.`);
      for (const f of formulas) {
        f.el.hidden = !!f.applies && !f.applies(cfg);      // a part this config does not compute is not shown
        if (f.el.hidden) continue;
        const key = f.spec[0].replace('{i}', String(L));
        const id = resolveNode(f.spec[1](L, hd, cfg), cfg);
        f.el.dataset.node = id;
        f.el.style.setProperty('--kind', `var(--c-${nodeInfo(id, cfg).kind})`);
        f.sub.innerHTML = `${key}${f.spec[2] ? ` · <span>${f.spec[2]}</span>` : ''}`;
        if (f.live) {
          const cell = cellFor(st, key, L, t0, hd);
          const res = cell ? runLive(st, key, cell) : null;
          f.live.replaceChildren(...liveChildren(res));
        }
      }
      measureWide();
    }

    // ------------------------------------------------------------ 3. flashcards
    let topicFilter = null;
    const topicSel = h('select', { class: 'cs-select', 'aria-label': 'Topic', onChange: (e) => {
      topicFilter = e.target.value || null;
      quiz = null;                 // the deck changed, so a quiz in progress ends
      renderDeck();
      renderQuiz();
    } });
    const deckEl = h('div', { class: 'cs-deck' });
    const deckFor = (t) => DECK.filter((c) => !t || c.topic === t);
    const labelOf = (t) => t ?? 'All topics';

    function renderTopicSelect() {
      const items = [[null, 'All topics', DECK.length], ...TOPICS.map((t) => [t, t, deckFor(t).length])];
      topicSel.replaceChildren(...items.map(([t, label, count]) => h('option', { value: t ?? '', selected: topicFilter === t }, `${label} · ${count}`)));
    }

    function flashcard(c, { n, total, big = false, onFlip } = {}) {
      const cfg = ctx.cfg;
      const node = cardNode(c, cfg);
      const info = nodeInfo(node, cfg);
      const flip = h('div', { class: 'cs-flip', role: 'button', tabindex: 0, 'aria-pressed': 'false', 'aria-label': `Card ${n}: ${c.q} Press to reveal the answer.` },
        h('div', { class: 'cs-flip-inner' },
          h('div', { class: ['cs-face', 'cs-front'] },
            h('div', { class: 'cs-face-top' }, h('span', null, `${n} / ${total}`), h('span', null, 'Question')),
            h('div', { class: 'cs-q' }, c.q),
            h('div', { class: 'cs-hint' }, 'Press to reveal the answer')),
          h('div', { class: ['cs-face', 'cs-back'] },
            h('div', { class: 'cs-face-top' }, h('span', null, `${n} / ${total}`), h('span', null, 'Answer')),
            h('p', { class: 'cs-a' }, answerOf(c)))));
      const setFlipped = (on) => {
        flip.classList.toggle('on', on);
        flip.setAttribute('aria-pressed', String(on));
        onFlip?.(on);
      };
      flip.addEventListener('click', () => setFlipped(!flip.classList.contains('on')));
      flip.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); setFlipped(!flip.classList.contains('on')); }
      });
      const foot = h('div', { class: 'cs-fc-foot' },
        h('span', { class: 'cs-topic' }, h('i'), c.topic),
        h('button', { type: 'button', class: 'cs-zoom', dataset: { node }, title: `Open ${info.title}` }, `Zoom to ${info.title} →`));
      const el0 = h('article', { class: ['cs-fc', big && 'big'], dataset: { topic: c.topic } }, flip, foot);
      el0.setFlipped = setFlipped;
      return el0;
    }

    function renderDeck() {
      const list = deckFor(topicFilter);
      deckEl.replaceChildren(...list.map((c) => flashcard(c, { n: DECK.indexOf(c) + 1, total: DECK.length })));
    }

    const secCards = h('section', { class: 'cs-sec', 'aria-labelledby': 'cs-h-cards' },
      h('div', { class: 'cs-sec-head' },
        h('h2', { class: 'cs-sec-title', id: 'cs-h-cards' }, '3 · Flashcards'),
        h('p', { class: 'cs-lede' }, `${DECK.length} questions an interviewer is likely to ask. Press a card to reveal the answer.`)),
      h('label', { class: 'cs-filter-row' }, 'Topic', topicSel),
      deckEl);

    // ------------------------------------------------------------ 4. quiz
    const quizEl = h('div', { class: 'cs-quiz' });
    let quiz = null;                 // {order, i, known, missed, label, flipped, cardEl, actionsEl}
    let quizSeed = 1337;
    let scoreNum = null;             // the score element being counted up (its frame is cancelled on destroy)

    function begin(list, label) {
      quiz = { order: shuffled(list, quizSeed), i: 0, known: [], missed: [], label, flipped: false, cardEl: null, actionsEl: null };
      renderQuiz();
    }

    function answer(knew) {
      const c = quiz.order[quiz.i];
      (knew ? quiz.known : quiz.missed).push(c);
      quiz.i += 1;
      quiz.flipped = false;
      renderQuiz();
    }

    function renderActions() {
      if (!quiz?.actionsEl) return;
      // Build a plain list: replaceChildren() would turn a null argument into the text "null".
      const buttons = quiz.flipped
        ? [h('button', { type: 'button', class: 'cs-btn', onClick: () => answer(false) }, 'Didn’t know'),
          h('button', { type: 'button', class: 'cs-btn primary', onClick: () => answer(true) }, 'Knew it')]
        : [h('button', { type: 'button', class: 'cs-btn primary', onClick: () => quiz.cardEl.setFlipped(true) }, 'Show answer')];
      quiz.actionsEl.replaceChildren(...buttons);
    }

    function idlePanel() {
      const list = deckFor(topicFilter);
      return h('div', { class: 'cs-score' },
        h('div', { class: 'cs-tile-k' }, `${labelOf(topicFilter)} · ${list.length} cards`),
        h('p', { class: 'cs-lede' }, `The deck is shuffled with seed ${quizSeed}, so every start gives the same order. Show the answer, then mark whether you knew it. The score counts the cards you knew on the first pass.`),
        h('div', { class: 'cs-actions' },
          h('button', { type: 'button', class: 'cs-btn primary', onClick: () => begin(list, labelOf(topicFilter)) }, `Start quiz · ${list.length} cards`),
          h('button', { type: 'button', class: 'cs-btn', onClick: () => { quizSeed += 1; begin(list, labelOf(topicFilter)); } }, 'Start with another shuffle')));
    }

    function scorePanel() {
      const n = quiz.order.length, k = quiz.known.length;
      const scoreEl = h('div', { class: 'cs-score-num' }, `0 / ${n}`);
      scoreEl._v = 0;                // count up from zero each time a score is shown
      scoreNum = scoreEl;
      tweenText(scoreEl, k, (v) => `${Math.round(v)} / ${n}`);
      const cfg = ctx.cfg;
      const missed = quiz.missed.length
        ? h('div', { class: 'cs-missed' },
          h('div', { class: 'cs-tile-k' }, 'Missed · press to zoom to the part it explains'),
          quiz.missed.map((c) => h('button', { type: 'button', dataset: { node: cardNode(c, cfg) } },
            h('b', null, `${DECK.indexOf(c) + 1}`), c.q)))
        : h('p', { class: 'cs-lede' }, 'Every card was known on the first pass.');
      return h('div', { class: 'cs-score' },
        h('div', { class: 'cs-tile-k' }, `${quiz.label} · seed ${quizSeed}`),
        scoreEl,
        h('p', { class: 'cs-lede' }, `${n ? Math.round((k / n) * 100) : 0}% known on the first pass.`),
        missed,
        h('div', { class: 'cs-actions' },
          quiz.missed.length ? h('button', { type: 'button', class: 'cs-btn primary', onClick: () => { quizSeed += 1; begin(quiz.missed.slice(), 'Missed cards'); } }, `Quiz the ${quiz.missed.length} missed`) : null,
          h('button', { type: 'button', class: 'cs-btn', onClick: () => { quizSeed += 1; begin(deckFor(topicFilter), labelOf(topicFilter)); } }, 'New shuffle'),
          h('button', { type: 'button', class: 'cs-btn', onClick: () => { quiz = null; renderQuiz(); } }, 'Done')));
    }

    /** A new card can grow the page below the fold, so keep the answer row above the band's bottom edge. */
    function revealActions() {
      const a = quiz?.actionsEl;
      if (!a?.isConnected) return;
      const lim = docEl.getBoundingClientRect().bottom - 40, r = a.getBoundingClientRect();     // the document's own bottom padding
      if (r.bottom > lim) docEl.scrollBy({ top: r.bottom - lim, behavior: reducedMotion() ? 'auto' : 'smooth' });
    }

    function renderQuiz() {
      quizEl.replaceChildren();
      if (!quiz) { quizEl.append(idlePanel()); return; }
      if (quiz.i >= quiz.order.length) { quizEl.append(scorePanel()); return; }
      const total = quiz.order.length;
      const c = quiz.order[quiz.i];
      quiz.flipped = false;
      const cardEl = flashcard(c, {
        n: quiz.i + 1, total, big: true,
        onFlip: (on) => { if (quiz) { quiz.flipped = on; renderActions(); } },
      });
      const actionsEl = h('div', { class: 'cs-actions' });
      const fill = h('i', { style: { width: `${(quiz.i / total) * 100}%` } });
      quiz.cardEl = cardEl;
      quiz.actionsEl = actionsEl;
      quizEl.append(
        h('div', { class: 'cs-progress' },
          h('span', null, `${quiz.i} / ${total}`),
          h('div', { class: 'cs-track', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': total, 'aria-valuenow': quiz.i }, fill),
          h('button', { type: 'button', class: 'cs-zoom', onClick: () => { quiz = null; renderQuiz(); } }, 'Stop')),
        cardEl, actionsEl);
      renderActions();
      appear([cardEl], { duration: MOTION.appear });
      revealActions();
    }

    const secQuiz = h('section', { class: 'cs-sec', 'aria-labelledby': 'cs-h-quiz' },
      h('div', { class: 'cs-sec-head' },
        h('h2', { class: 'cs-sec-title', id: 'cs-h-quiz' }, '4 · Quiz'),
        h('p', { class: 'cs-lede' }, 'Mark each card knew it or didn’t. The topic filter sets the deck and ends a quiz in progress.')),
      quizEl);

    inner.append(secShapes, secFormulas, secCards, secQuiz);

    // ------------------------------------------------------------ story steps: the four sections
    // The bottom bar lists the sections. A step scrolls to its section, and the bar keeps that step until the
    // scroll arrives (it would otherwise read the old section while the document is still moving). After that,
    // scrolling moves the bar to the section the reader is in; the flag keeps that sync from scrolling again.
    const sectionEls = [secShapes, secFormulas, secCards, secQuiz];
    let activeStep = 0, syncing = false, scrollRaf = 0;
    let pinTop = null, pinTimer = 0;
    const scrollToSection = (i) => {
      const sec = sectionEls[i];
      if (!sec) return;
      pinTop = i === 0 ? 0 : Math.max(0, sec.offsetTop - PIN);
      clearTimeout(pinTimer);
      pinTimer = setTimeout(() => { pinTop = null; }, 1200);     // a clamped or interrupted scroll releases the pin
      docEl.scrollTo({ top: pinTop, behavior: reducedMotion() ? 'auto' : 'smooth' });
    };
    ctx.setSteps(['Shapes', 'Formulas', 'Flashcards', 'Quiz'], (i) => {
      activeStep = i;
      if (!syncing) scrollToSection(i);
    });
    // The document takes focus so that keyboard users can scroll it. The atlas leaves its own keys alone while
    // focus is inside the document, so the arrows step the sections here. The topic filter keeps its arrow keys.
    docEl.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'SELECT') return;
      const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!step) return;
      e.preventDefault();
      const next = clamp(activeStep + step, 0, sectionEls.length - 1);
      if (next !== activeStep) ctx.setStep(next);
    });
    function syncStep() {
      scrollRaf = 0;
      if (pinTop !== null) {
        if (Math.abs(docEl.scrollTop - pinTop) > 2) return;          // still travelling to the chosen step
        pinTop = null;
        clearTimeout(pinTimer);
      }
      const line = docEl.scrollTop + docEl.clientHeight * 0.35;      // the reading line: a third down the view
      let j = 0;
      sectionEls.forEach((s, i) => { if (s.offsetTop <= line) j = i; });
      if (docEl.scrollTop + docEl.clientHeight >= docEl.scrollHeight - 4) j = sectionEls.length - 1;
      if (j !== activeStep) { syncing = true; ctx.setStep(j); syncing = false; }
    }
    const onScroll = () => { if (!scrollRaf) scrollRaf = requestAnimationFrame(syncStep); };
    docEl.addEventListener('scroll', onScroll, { passive: true });

    // ------------------------------------------------------------ shapes (rebuilt only when the shapes change)
    let shapeSig = '';
    let cfgSig = '';
    function buildShapes(st) {
      const cfg = st.config, T = st.ids.length;
      const sig = `${JSON.stringify(cfg)}|${T}|${Object.entries(st.trace).map(([k, t]) => `${k}:${t.shape.join('x')}`).join(',')}`;
      if (sig === shapeSig) return;
      shapeSig = sig;
      const groups = new Map();
      for (const key of Object.keys(st.trace)) {
        const g = groupOf(key);
        if (!groups.has(g)) groups.set(g, []);
        groups.get(g).push(key);
      }
      let ok = 0, n = 0;
      const rows = [];
      for (const [g, keys] of groups) {
        rows.push(h('tr', { class: 'cs-group' }, h('td', { colspan: 4 }, `${g} · ${keys.length} ${keys.length === 1 ? 'tensor' : 'tensors'}`)));
        for (const key of keys) {
          const t = st.trace[key];
          const info = infoFor(key);
          const known = !!TRACE_INFO[parseKey(key).pattern];
          let check = null;
          if (known) {
            const want = shapePairs(key, cfg, T).map(([, v]) => v);
            const match = want.length === t.shape.length && want.every((v, i) => v === t.shape[i]);
            n += 1;
            if (match) ok += 1;
            check = h('span', { class: match ? 'cs-ok' : 'cs-bad', title: match ? 'matches the symbolic shape' : 'differs from the symbolic shape' }, match ? '✓' : '✗');
          }
          const sym = known ? (info.dims.length ? info.dims.join(' × ') : 'scalar') : '—';
          const line = lineForKey(key, cfg);
          const tr = h('tr', { class: 'cs-row', tabindex: 0, dataset: { trace: key } },
            h('td', null, h('div', { class: 'cs-key' }, h('i', { class: 'cs-kind', style: { '--kind': `var(--c-${info.kind})` } }), key, h('small', null, info.title))),
            h('td', { class: 'cs-mono' }, sym),
            h('td', { class: 'cs-mono' }, t.shape.length ? t.shape.join(' × ') : 'scalar', check),
            h('td', { class: 'cs-line' }, line ? `model.py:${line}` : '—'));
          const zoom = () => ctx.go(nodeForKey(key, ctx.state.sel), tr);
          tr.addEventListener('click', zoom);
          tr.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); zoom(); } });
          rows.push(tr);
        }
      }
      tbody.replaceChildren(...rows);
      summaryHost.replaceChildren(
        h('span', { class: ['k-check', ok === n ? 'ok' : 'bad'] }, ok === n ? `✓ all ${n} traced shapes match their symbols` : `✗ ${n - ok} of ${n} traced shapes differ from their symbols`),
        h('span', { class: 'cs-count' }, `${Object.keys(st.trace).length} tensors in the last forward pass`));
    }

    function renderTiles(st) {
      const cfg = st.config, T = st.ids.length, d = cfg.d_model;
      const specs = paramSpecs(cfg);
      const size = (sh) => sh.reduce((a, b) => a * b, 1);
      const total = specs.reduce((s, [, sh]) => s + size(sh), 0);
      const block = specs.reduce((s, [name, sh]) => (name.startsWith('blocks.0.') ? s + size(sh) : s), 0);
      // The weight matrices of block 0; whatever is left of the block is norm gains and biases.
      const weights = (part) => specs.reduce((s, [name, sh]) => (name.startsWith(`blocks.0.${part}.`) && name.endsWith('.weight') ? s + size(sh) : s), 0);
      const attnW = weights('attn'), mlpW = weights('mlp'), rest = block - attnW - mlpW;
      const textbook = cfg.n_kv_head === cfg.n_head && cfg.activation !== 'swiglu' && cfg.d_ff === 4 * d;
      const left = cfg.bias ? 'Biases and norm parameters' : 'Norm parameters';
      const act = { gelu: 'GELU', relu: 'ReLU', swiglu: 'SwiGLU' }[cfg.activation];
      const maskName = cfg.mask === 'none' ? 'no mask' : `${cfg.mask} mask`;
      const kv = cacheBytes(cfg, T, 2);
      tweenText(tiles.params.v, total, fmtInt);
      const vocab = st.tokenizer.kind === 'bpe'
        ? `${fmtInt(cfg.vocab_size)} tokens (byte-level BPE, ${fmtInt(st.tokenizer.merges)} merges)`
        : `${cfg.vocab_size} characters`;
      tiles.params.s.textContent = `${cfg.n_layer} blocks · vocabulary of ${vocab} · ${cfg.tie_weights ? 'output head tied to the embedding' : 'separate output head'}`;
      tweenText(tiles.block.v, block, fmtInt);
      const shape = textbook
        ? `12·d² = ${fmtInt(12 * d * d)} for MHA with d_ff = 4d. ${left} make up the other ${fmtInt(rest)}.`
        : `${attnName(cfg)}, d_ff ${cfg.d_ff}: attention ${fmtInt(attnW)} and MLP ${fmtInt(mlpW)} weights. ${left} make up the other ${fmtInt(rest)}.`;
      tiles.block.s.textContent = `${shape} ${act} activation, ${maskName}.`;
      tweenText(tiles.kv.v, kv, (v) => fmtKiB(Math.round(v)));
      tiles.kv.s.textContent = `${fmtInt(kv)} bytes for ${T} ${T === 1 ? 'token' : 'tokens'} in bf16: 2 · L · H_kv · d_h · T · 2 bytes`;
    }

    // ------------------------------------------------------------ wiring
    function onTrace() {
      const st = ctx.state;
      if (!st.trace) return;
      const tk = st.tokenizer;
      const sig = `${JSON.stringify(st.config)}|${tk.kind}|${tk.size}|${tk.merges ?? ''}`;   // the deck's answers read the tokenizer
      if (sig !== cfgSig) {
        cfgSig = sig;
        eyebrow.replaceChildren(...configChips(st.config).map(([t, k]) => chip(t, k)));
        renderDeck();
        renderQuiz();                // a quiz in progress shows its current card's answer for this model
      }
      renderTiles(st);
      buildShapes(st);
      renderLayerChips();
      hp.render();
      renderFormulas();
    }

    function onSelect() {
      if (!ctx.state.trace) return;
      renderLayerChips();
      hp.render();
      renderFormulas();
    }

    renderTopicSelect();
    renderDeck();
    renderQuiz();
    onTrace();
    const offs = [ctx.on('trace', onTrace), ctx.on('select', onSelect)];
    // Column widths change with the window; KaTeX fonts can arrive after the first layout.
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => measureWide()) : null;
    ro?.observe(docEl);
    document.fonts?.ready?.then(() => measureWide());

    appear([header.el, secShapes, secFormulas, secCards, secQuiz], { stagger: MOTION.stagger });

    return {
      destroy() {
        ro?.disconnect();
        docEl.removeEventListener('scroll', onScroll);
        cancelAnimationFrame(scrollRaf);
        clearTimeout(pinTimer);
        offs.forEach((f) => f());
        cleanups.forEach((f) => { try { f(); } catch { /* ignore */ } });
        for (const t of Object.values(tiles)) cancelAnimationFrame(t.v._raf);
        cancelAnimationFrame(scoreNum?._raf);
        docEl.remove();
      },
    };
  },
};

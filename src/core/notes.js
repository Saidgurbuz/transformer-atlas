// Interpretation notes per topic, drafted from sources and fact-checked against them
// (tools: sources workflow). Keys are topic ids; atlas nodes map onto them in notesFor().
export const NOTES = {
 "overview": [
  {
   "placement": "Architecture map: residual stream and norm placement in each block, plus the final norm before the LM head",
   "text": "Each block adds its attention and MLP outputs into one residual stream that stays d_model wide (64 in the character presets, 128 in the TinyStories preset) from embedding to LM head, and Elhage et al. describe both sub-layers as reading it 'by performing a linear projection' and writing back 'by adding a linear projection back in'. Every preset normalizes each sub-layer's input and applies a final norm (ln_f) before lm_head, as GPT-2 did after moving layer norm to the input of each sub-block. Xiong et al. show that in the original Post-LN Transformer the expected gradients of the parameters near the output layer are large at initialization, so a large learning rate makes training unstable and warm-up helps, while their Pre-LN experiments can omit warm-up; this is not a guarantee for every training recipe.",
   "refs": [
    "elhage2021",
    "radford2019",
    "touvron2023",
    "xiong2020"
   ]
  },
  {
   "placement": "Parameter breakdown: MLP versus attention share, GPT-2-style and LLaMA-style presets",
   "text": "The MLP holds most of the parameters in both presets: 63.6% of the GPT-2-style model (132,352 of 208,256) and 71.0% of the LLaMA-style model (132,096 of 185,920). The LLaMA-style share rises mainly because attention falls from 66,560 to 49,152 parameters, mostly through GQA, while its three SwiGLU matrices of width 172 cost about what two GELU matrices of width 256 do. Kaplan et al. approximate non-embedding parameters as N ≈ 2 <code>d_model</code> <code>n_layer</code> (2 <code>d_attn</code> + <code>d_ff</code>), which gives about 4d² for attention and 8d² for the MLP in each layer when d_attn = d_ff/4 = d_model.",
   "refs": [
    "kaplan2020"
   ]
  },
  {
   "placement": "Parameter breakdown: attention k_proj and v_proj shapes in the LLaMA-style preset (4 query heads, 2 key/value heads)",
   "text": "The LLaMA-style preset has 4 query heads sharing 2 key/value heads, so <code>k_proj</code> and <code>v_proj</code> are 32 × 64 instead of 64 × 64: 4,096 fewer weights per block and half the key/value cache per token. Ainslie et al. report that uptrained GQA 'achieves quality close to multi-head attention with comparable speed to MQA'. Llama 3 8B uses 32 query heads with 8 key/value heads (Table 3), the same idea at a 4:1 grouping rather than this preset's 2:1.",
   "refs": [
    "ainslie2023gqa",
    "shazeer2019mqa",
    "grattafiori2024"
   ]
  },
  {
   "placement": "Architecture map: the embedding stage (learned position rows) and the LLaMA-style preset's missing position table",
   "text": "The GPT-2-style, induction and reverse presets have a learned position table (wpe) with exactly block_size rows (64, 32 and 24), which is 4,096 parameters (about 2%) in the GPT-2-style model. In the reference PyTorch model (reference/model.py) the forward pass asserts T <= block_size and generation crops the context to block_size, so that table is a hard context limit in this code. The LLaMA-style preset has no such table, because LLaMA's authors write 'We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE)'; Su et al. describe RoPE as encoding absolute position with a rotation matrix while incorporating relative-position dependence into self-attention, and Press et al. report that current methods 'do not allow for efficient extrapolation'.",
   "refs": [
    "touvron2023",
    "su2021",
    "press2021"
   ]
  },
  {
   "placement": "Active preset: the train_loss_nats metric shown with each trained preset",
   "text": "The preset manifest records train_loss_nats of 0.5464 (GPT-2-style) and 0.2876 (LLaMA-style), which is cross-entropy in nats, the unit Kaplan et al. use (their Section 1.3 defines L as 'the cross entropy loss in nats'). These are training-set numbers: the metric is computed on the same Alice corpus the model is trained on, with no held-out split, whereas Kaplan et al. evaluate on a held-out test set, so these numbers measure fit to that text rather than generalization. As a reading aid, the per-character perplexity exp(0.5464) is about 1.73, against 32 for uniform guessing over the 32-symbol vocabulary.",
   "refs": [
    "kaplan2020"
   ]
  }
 ],
 "stepper": [
  {
   "placement": "attn.scores and attn.scaled rows (reference/model.py lines 157-159), in every preset",
   "text": "Read <code>attn.scaled</code> against <code>attn.scores</code>: each logit is divided by sqrt(d_head), which is sqrt(16) = 4 in the character presets (d_model 64 over 4 heads) and sqrt(32) in the TinyStories preset (d_model 128 over 4 heads). Vaswani et al. say they suspect that for large d_k the dot products grow in magnitude and push the softmax into regions of very small gradients, which is why they scale by 1/sqrt(d_k); the <code>attn_scale</code> switch in Config removes the factor.",
   "refs": [
    "vaswani2017",
    "mody-numpy"
   ]
  },
  {
   "placement": "attn.masked and attn.probs rows (lines 162-163) and the mask row (line 269); the upper triangle of attn.probs",
   "text": "<code>attn.masked</code> sets disallowed keys to <code>-inf</code> before the softmax, so the upper triangle of <code>attn.probs</code> is exactly 0 and each row sums to 1 over the visible keys. The zeros come from the softmax of the masked logits rather than from zeroing probabilities afterwards, so the remaining weights are renormalized over the allowed keys.",
   "refs": [
    "vaswani2017",
    "phuong2022"
   ]
  },
  {
   "placement": "resid_mid and resid_out rows (lines 211 and 214), the ln1, ln2 rows, and the final lnf row (line 277); every preset is pre-LN",
   "text": "With the residual switch on (the default, and the setting in every preset), resid_mid is the block input plus the attention output and resid_out adds the MLP output to that, while ln1 and ln2 are the normalized inputs to the attention and MLP and lnf is the final normalization before the LM head. GPT-2 moved layer norm to the input of each sub-block and added a final layer norm, and Xiong et al. report that, in the tasks they tested, the learning-rate warm-up could be removed with this pre-LN placement. Elhage et al. describe the residual stream as a communication channel that each attention and MLP layer reads from and writes to by adding a linear projection back in, so each residual row holds everything written to it so far.",
   "refs": [
    "radford2019",
    "xiong2020",
    "elhage2021",
    "mody-numpy"
   ]
  },
  {
   "placement": "attn.probs rows for layer 0 head 0 and the layer 1 heads (line 163) in the induction preset, next to its logits row",
   "text": "In the induction preset, layer 0 head 0 should put most of its weight on the previous token, the sub-diagonal of attn.probs, which Olsson et al. call a previous-token head. An induction head in layer 1 should attend to the token that followed an earlier copy of the current token (prefix matching), and its copying property says this attention raises that token's logit. Not every layer-1 head is an induction head: a head counts only if it has both properties.",
   "refs": [
    "olsson2022"
   ]
  },
  {
   "placement": "k and v (2 KV heads) versus k_rep and v_rep (4 query heads) rows (lines 154-156) in the llama preset, where n_head is 4 and n_kv_head is 2",
   "text": "In the llama preset, <code>k_rep</code> and <code>v_rep</code> repeat each of the 2 KV heads for two adjacent query heads: query heads 0 and 1 read KV head 0, and heads 2 and 3 read KV head 1. A decoding cache would store only the 2 KV heads, half the size of a 4-head multi-head cache. Ainslie et al. describe an intermediate number of groups as higher quality than multi-query attention but faster than multi-head attention.",
   "refs": [
    "ainslie2023gqa"
   ]
  }
 ],
 "tokens": [
  {
   "placement": "Toy BPE on the current text: the tokens-per-character figure and the merge table",
   "text": "Each toy merge shortens the sequence and creates one new symbol, so <code>tokens per character</code> falls as the slider moves; the GPT-2 paper describes BPE as <i>a practical middle ground between character and word level language modeling</i>. GPT-2’s byte-level BPE also blocks merges across character categories except spaces, which this toy does not do, so its merges can differ from GPT-2’s.",
   "refs": [
    "radford2019",
    "sennrich2016"
   ]
  },
  {
   "placement": "Does the LM head reuse W_E? (the tie diagram and the ‘saved by tying’ row)",
   "text": "Press and Wolf recommend tying the input and output embeddings for language models: in their PTB experiment, tying <i>significantly reduces perplexity on both the validation set and the test set</i>, which they read as less overfitting. The gain is model-dependent (they found tying <i>detrimental in word2vec</i>), so the <i>saved by tying</i> row here is a parameter count, not a quality measurement.",
   "refs": [
    "press2016tying"
   ]
  },
  {
   "placement": "Cosine similarity between rows: the heatmap and the mean-cosine table",
   "text": "Gao et al. report that, with the weight tying trick, most learnt word embeddings tend to degenerate and be distributed into a narrow cone, especially with big training datasets. In their machine-translation Transformer, the word embeddings are clustered together in such a cone, and their cosine similarities are positive for almost all pairs. They read this as degeneration rather than structure: the words huddle together and are not well separated in the embedding space.",
   "refs": [
    "gao2019"
   ]
  },
  {
   "placement": "PCA of the vocabulary rows: the scatter and the explained-variance bars",
   "text": "Elhage et al. report that, for models with large d_model, the PCA spectrum of token embeddings and unembeddings decays quickly, with them concentrated in a relatively small fraction of the dimensions. So if PC1 and PC2 hold only a small share of the variance here, this 2D scatter leaves out more of the structure than it would for such a model.",
   "refs": [
    "elhage2021"
   ]
  },
  {
   "placement": "Block input x₀: the emb.tok + emb.pos sum and the emb.out check",
   "text": "Before any block runs, the residual stream is just x₀; Elhage et al. describe the stream as <i>simply the sum of the output of all the previous layers and the original embedding</i>. Each block reads from it and adds its result back, so on the input side the token id reaches the model only through this vector.",
   "refs": [
    "elhage2021"
   ]
  }
 ],
 "positions": [
  {
   "placement": "The positional-encoding table (trace key emb.pos) for the sinusoidal and learned variants: rows are positions, columns are dimensions. Read the sin/cos column pairs from left to right.",
   "text": "The sinusoidal table is a fixed function of position: each sin/cos column pair turns at its own rate, and the wavelengths form a geometric progression from 2π to 10000·2π, so the first pair repeats about every six rows while the last pairs barely change over a few dozen positions. Vaswani et al. report that learned and sinusoidal embeddings gave nearly identical results in their experiments, and chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than those seen in training. The learned table in the same panel has no sin/cos pairs, so read those pairs only for the sinusoidal variant.",
   "refs": [
    "vaswani2017"
   ]
  },
  {
   "placement": "Q after RoPE and K after RoPE (trace keys L{i}.attn.q_rot and k_rot): each consecutive pair (x[2i], x[2i+1]) is rotated by an angle that grows with the row index.",
   "text": "Each pair (x₂ᵢ, x₂ᵢ₊₁) is rotated by t·θᵢ with θᵢ = 10000^(−2i/d_h), so the score between a rotated query and key depends only on their offset n−m when the underlying vectors are held fixed. RoFormer states the goal as an inner product that 'encodes position information only in the relative form', but in a trained model q and k also depend on the token, so the offset alone does not set the score. The app pairs consecutive entries, as RoFormer's Eq. 34 and EleutherAI's derivation do; EleutherAI's GPT-NeoX code instead pairs dimension i with i+d/2 (a half-split layout), and the two layouts differ by a fixed permutation of q and k, so projection weights trained in one must be reordered to run in the other (a derivation that neither source states).",
   "refs": [
    "su2021",
    "eleuther-rope"
   ]
  },
  {
   "placement": "ALiBi distance bias (trace key alibi, shape H×T×T) and the Scores + ALiBi panel (L{i}.attn.biased): each head has one fixed slope, and the bias depends only on the query-key distance.",
   "text": "ALiBi adds a fixed, unlearned penalty of −mₕ·(i−j) to each past key's score, with slopes from the geometric sequence that starts at 2^(−8/n) and uses 2^(−8/n) as its ratio. With 4 heads, as in the Alice presets, the slopes are 1/4, 1/16, 1/64 and 1/256, so a key 31 positions back starts 7.75 logits behind the query's own score in the steepest head and about 0.12 behind in the flattest. Press et al. report that, in their WikiText-103 experiments, a model trained on 512 tokens showed improved perplexity when handling more tokens than it saw in training, reaching 18.40 at length 3072.",
   "refs": [
    "press2021"
   ]
  },
  {
   "placement": "The no-positional-encoding setting and the permutation demo: compare outputs for shuffled or reversed inputs, with and without the causal mask.",
   "text": "With no positional encoding and no mask, self-attention is permutation-equivariant, so a permutation demo without the causal mask should return the same outputs, reordered; the causal mask breaks that symmetry because query t sees exactly t+1 keys. Haviv et al. find by probing that causal LMs without positional encodings carry an implicit notion of absolute position across layers, and conjecture that causal attention lets a token count its predecessors, which would essentially give that position; their direct test is a masked-LM corollary, not the counting mechanism itself. In Kazemnejad et al.'s experiments on reasoning and math tasks, NoPE outperforms other explicit encodings, although when trained with SGD it mostly resembles T5's relative attention patterns.",
   "refs": [
    "lee2019",
    "haviv2022",
    "kazemnejad2023"
   ]
  },
  {
   "placement": "The context-extension panel: positions past the trained length, for the learned table (trace key emb.pos) and for RoPE (q_rot and k_rot).",
   "text": "In reference/model.py the learned table has exactly block_size rows, so there is no embedding past the trained length, and Press et al. write that the learned approach 'does not have a way to encode positions greater than L' (footnote 4). For RoPE, Chen et al. warn that direct extrapolation may lead to 'catastrophically high attention scores', and their Position Interpolation scales positions down by L/L′ into the trained range, which in their LLaMA experiments extended the window up to 32768 with minimal fine-tuning (within 1000 steps).",
   "refs": [
    "press2021",
    "chen2023pi"
   ]
  }
 ],
 "attention": [
  {
   "placement": "Scaled scores (QK^T divided by √d_h) just before the softmax, and how peaked each row looks",
   "text": "The four character presets have d_h = 64 / 4 = 16, so their raw scores are divided by √16 = 4 before the softmax; the TinyStories preset has d_h = 128 / 4 = 32, so its scores are divided by √32. Vaswani et al. suspect that for large d_k the unscaled dot products grow in magnitude and push softmax into regions with extremely small gradients; the attn_scale ablation drops the factor, which multiplies every logit by 4 (or by √32 in TinyStories) and generally makes each row sharper.",
   "refs": [
    "vaswani2017"
   ]
  },
  {
   "placement": "The masked upper triangle of the score matrix, and the softmax rows that stop at the diagonal",
   "text": "Masked cells are set to −∞ before the softmax, not to 0, because exp(0) = 1 would still give them weight, while exp(−∞) = 0 removes them from the normalizer. That is why each visible row still sums to 1 over the tokens the query is allowed to see.",
   "refs": [
    "vaswani2017"
   ]
  },
  {
   "placement": "Induction preset: layer 0 head 0 (stripe just below the diagonal) and layer 1 heads 0, 2 and 3 (cells on the matching token)",
   "text": "In the induction preset, layer 0 head 0 acts as a previous-token head, and layer 1 heads 0, 2 and 3 act as induction heads that attend to the token following an earlier copy of the current token. Olsson et al. (2022) describe how this pair of heads makes the sequence [A][B] … [A] more likely to be completed with [B]. A heatmap shows where a head reads from, not what it writes: the OV circuit sets what is written (Elhage et al. 2021), and attention weights are only one of the two factors behind the output, the other being the norm of the transformed input vectors (Kobayashi et al. 2020).",
   "refs": [
    "olsson2022",
    "elhage2021",
    "kobayashi2020"
   ]
  },
  {
   "placement": "First column (token 0) of a head's attention map, when most rows concentrate there",
   "text": "When the first column is bright across most rows, the head is often acting as an attention sink: softmax must spread each row's mass over the allowed tokens, and the first token can absorb mass the head does not need. Xiao et al. report that in their Llama-2-7B experiments, beyond the bottom two layers, the model consistently focuses on the initial tokens across layers and heads; the 2-layer presets (induction, reverse) have no layers beyond the bottom two, so read the first column directly rather than assume it.",
   "refs": [
    "xiao2023"
   ]
  },
  {
   "placement": "The T×T matrix and the complexity readout, next to the FlashAttention description",
   "text": "Standard attention materialises a T×T score matrix for each head, which takes O(T²) memory per head (Dao et al. 2022, Sec. 2.2); its arithmetic is O(T²·d) per layer, where d is the representation dimension, i.e. the model width (Vaswani et al. 2017, Table 1). FlashAttention computes the same exact output with O(T) extra memory by tiling and recomputing on-chip; it does more FLOPs than standard attention, yet runs faster because it makes many times fewer HBM reads and writes (Dao et al. 2022).",
   "refs": [
    "vaswani2017",
    "dao2022flash"
   ]
  }
 ],
 "multihead": [
  {
   "placement": "Q, K and V split into heads (q_heads, k_heads, v_heads) and the O-projection per-head contributions",
   "text": "In the four character presets (64-wide, 4 heads) each query head is a 16-dimensional slice of Q, and K and V use the same 16-dimensional width per head; the llama preset has only 2 KV heads, which reference/model.py (lines 154-156) repeats to match its 4 query heads. Vaswani et al. define MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O, and Elhage et al. describe the concatenate-and-multiply form as mathematically equivalent to each head writing its output into the residual stream, so the per-head terms sum to the attention output; Elhage et al. do not consider biases, so in this app that sum is exact only up to the O-projection bias (reference/model.py line 139), which the charlm, induction and reverse presets have and llama does not. Kobayashi et al. show that attention weights alone are only one of the two factors that determine the output of attention, the other being the norm of the transformed input vectors, so a large weight on a token with a small transformed-vector norm can still contribute little.",
   "refs": [
    "vaswani2017",
    "elhage2021",
    "kobayashi2020"
   ]
  },
  {
   "placement": "Small multiples on the induction preset: previous-token and induction-score columns",
   "text": "On the induction preset, pair the layer-0 previous-token head with the layer-1 heads that score high on induction: Olsson et al. describe the circuit as a pair of heads in different layers, the first copying the previous token into each position. The induction score shows only the attention side of the pattern, attention to the token after an earlier copy of the current token. Olsson et al. also require copying, meaning the head's output raises the logit of the attended-to token, so check that effect before calling a head an induction head.",
   "refs": [
    "olsson2022"
   ]
  },
  {
   "placement": "Small multiples: first-token and entropy columns",
   "text": "A head that puts most of its weight on the first token may be using it as a sink: Xiao et al. report strong attention to initial tokens “as a ‘sink’ even if they are not semantically important”. Under the causal mask, row t (0-indexed) has only t+1 legal keys, so row 0's first-token and self shares are both 1 by construction, and a row's maximum entropy is log(t+1), not log T (the sequence length).",
   "refs": [
    "xiao2023",
    "vaswani2017"
   ]
  },
  {
   "placement": "All-heads small multiples as a whole, read together with the Ablation lab (A/B)",
   "text": "A clean pattern shows what a head reads, not how much the output depends on it. Michel et al. report, in their experiments, that a large percentage of attention heads can be removed at test time without significantly impacting performance, and Voita et al. observe that specialised heads are the last to be pruned, so test a head in the Ablation lab (A/B) before calling it essential.",
   "refs": [
    "michel2019",
    "voita2019"
   ]
  },
  {
   "placement": "KV sharing: k_rep and v_rep against k_heads and v_heads, and the KV-cache readout on the llama preset",
   "text": "On the llama preset (4 query heads, 2 KV heads), query heads 0 and 1 read KV head 0 and heads 2 and 3 read KV head 1, so <code>k_rep</code> and <code>v_rep</code> repeat each KV head, while <code>k_heads</code> and <code>v_heads</code> hold only the 2 KV slices. The repeats exist so the score matmul sees one K per query head; the cache stores only the 2 KV heads, half the size of the 4-KV-head (MHA) cache for this model. Shazeer reports, in their experiments, that sharing keys and values shrinks the tensors read during incremental decoding with only minor quality degradation, and Ainslie et al. report that uptrained GQA reaches quality close to multi-head attention with speed comparable to MQA.",
   "refs": [
    "shazeer2019mqa",
    "ainslie2023gqa"
   ]
  }
 ],
 "masking": [
  {
   "placement": "Masked-scores panel (L{i}.attn.masked) and the probability rows beside it, for every mask preset",
   "text": "Disallowed keys are set to −∞ before the softmax, so they come out as exactly 0 and each row sums to 1 over its allowed keys. A row with no allowed key gives NaN, because softmax over all −∞ is NaN; the built-in masks here always keep the diagonal, so this needs a padding-style mask. PyTorch PR #131060 ('Add a private _safe_softmax') specifies that such rows output 0 (out[masked_out_rows] = 0); GitHub shows it closed without a merge, but its commit appears in PyTorch's history, so it is no longer only a proposal.",
   "refs": [
    "vaswani2017",
    "pytorch-safe-softmax"
   ]
  },
  {
   "placement": "Mask grid for 'none' (bidirectional) beside 'causal', plus the leakage demo",
   "text": "Bidirectional attention (mask none) lets row i read column i+1, the token that row is supposed to predict, so a next-token loss can fall by copying the answer; that shortcut is what a leakage demo should expose. The causal triangle removes it. BERT's bidirectional encoder is a different setting: its MLM objective 'enables the representation to fuse the left and the right context', and BERT is not pre-trained as a left-to-right language model.",
   "refs": [
    "vaswani2017",
    "devlin2018"
   ]
  },
  {
   "placement": "Teacher-forced pass over the causal grid, and the one-row-per-token decode path (KV cache)",
   "text": "The causal mask is what lets one forward pass score every position: row i is trained to predict token i+1 from the true tokens before it. At generation time those true tokens are replaced by the model's own samples, a training/inference discrepancy that Bengio et al. describe as one that 'can yield errors that can accumulate quickly along the generated sequence.' In this app, the KV-cache decoder adds one row per generated token under the causal or sliding mask (kvCacheSupported is false for prefix and bidirectional masks), and tests/parity.mjs checks those rows against the full forward pass.",
   "refs": [
    "bengio2015",
    "vaswani2017"
   ]
  },
  {
   "placement": "Sliding preset: the band in the mask grid, the window slider (default 8), and column 0",
   "text": "With window W, row i sees keys i−W+1 through i, so column 0 goes dark for every row from W onward (row 8 at the default). Information still spreads further because each layer passes it on: in this app that is W−1 positions per layer, so 4 layers with W = 8 reach about 28 tokens back, whereas Mistral 7B's paper counts W per layer (its window spans i−W to i), giving about 131K for W = 4,096 over 32 layers. Xiao et al. report that, beyond the bottom two layers, attention consistently focuses on the initial tokens across all layers and heads (Sec. 3.1), and that a window-only cache collapses once the sequence length exceeds the cache size, even when only the first token's KV is evicted (Sec. 1).",
   "refs": [
    "jiang2023mistral",
    "xiao2023"
   ]
  },
  {
   "placement": "Prefix preset: the first columns of the grid and the prefix length slider (default 4)",
   "text": "The prefix length p makes the first p columns visible to every row, so prompt or image tokens get bidirectional context while the rest stays causal. PaliGemma uses this pattern for vision-language models: full attention over the image and prefix tokens, with the suffix (the answer) under an auto-regressive mask, including PAD tokens. T5 describes the same fully-visible prefix (Sec. 3.2.1), and its Table 2 shows the prefix LM beating a plain causal LM on every benchmark column under both objectives, but 'For all tasks, the encoder-decoder architecture with the denoising objective performed best' (Sec. 3.2.4), so the prefix mask was not T5's top architecture.",
   "refs": [
    "beyer2024",
    "raffel2019"
   ]
  }
 ],
 "mlp": [
  {
   "placement": "The up (T×F) and down (T×d) panels for the GPT-2-style and LLaMA-style presets, and the parameter counts that go with them.",
   "text": "<code>up</code> widens each token from d to F and <code>down</code> projects it back to d. With the usual F = 4d, Kaplan et al.'s count (N ≈ 12·n_layer·d²) puts two-thirds of each block's weight matrices in the MLP, which is 32,768 of 49,152 in the GPT-2-style preset (d=64, F=256). The LLaMA-style preset uses F=172 for its three SwiGLU matrices (33,024 weights, against 32,768 for the two-matrix MLP), close to the 2/3 width reduction Shazeer uses to keep parameters and compute fixed; its grouped-query K and V projections are small, so its MLP is about 73% of the block's weight matrices.",
   "refs": [
    "kaplan2020",
    "shazeer2020"
   ]
  },
  {
   "placement": "The act (T×F) panel in L{i}.mlp.act for the GELU presets (Alice GPT-2-style, induction, reverse), and the act in the LLaMA-style preset, which is SiLU(gate) ⊙ up, next to any sparsity readout.",
   "text": "GELU weights each pre-activation by its value rather than gating it by sign, so negative inputs give small negative outputs (minimum about -0.17, from the formula) instead of zeros, and a sparsity count on <code>act</code> depends on the threshold chosen. ReLU-type activations (ReLU, ReLU²) set negative inputs to exactly zero, and Zhang et al. handle non-ReLU LLMs by defining activation through an output-magnitude threshold, under which they still report sparse activation. Mirzadeh et al. (preprint) argue for reinstating ReLU, noting that GELU and SiLU are 'known for increased computation' and that using ReLU 'has a negligible impact on convergence and performance' in their experiments.",
   "refs": [
    "hendrycks2016",
    "zhang2024relu2",
    "mirzadeh2023",
    "shazeer2020"
   ]
  },
  {
   "placement": "The per-memory columns of L{i}.mlp.act, read alongside the up rows (keys) and down columns (values), and any active-share readout.",
   "text": "Geva et al. read each MLP as a key-value memory, FF(x) = f(xKᵀ)V with K, V ∈ ℝ^(d_m×d): each key is a d-dimensional vector matched against the input x, and each value is weighted by its key's activation in the sum; in this app, the keys are the rows of up's weight (F×d), the values are the columns of down's weight (d×F), and x is the normalised residual (ln2 output), which Geva et al. do not specify. In their experiments on a 16-layer WikiText-103 language model with 4,096 memories per layer, 'a typical example triggers hundreds of memories per layer (10%-50% of 4096 dimensions)', so the active share depends on the model and the threshold rather than on one rule. Li et al. measure sparsity in ReLU MLPs, where 'very few entries (e.g., 3.0% for T5-Base and 6.3% for ViT-B16) are nonzero for each input'; a GELU or SwiGLU count needs a stated magnitude threshold before it can be set beside those numbers.",
   "refs": [
    "geva2021",
    "li2023lazy"
   ]
  },
  {
   "placement": "The per-unit token ranking (one column of down projected through the output embedding), as opposed to a projection of the whole mlp.out vector.",
   "text": "Projecting one value vector through the output embedding (the unembedding here) ranks the tokens it promotes; Geva et al. report that a substantial portion (40%-70% in WikiLM and 20%-65% in GPT-2) of the top tokens were associated with well-defined concepts, while projecting the summed FFN output is 'not interpretable'. Their own caveats are that success may depend on the tokenization method, and that in the last layers 'the probability of the top-candidate is pushed down by dominant sub-updates'. Neurons also mix features ('Neural networks often pack many unrelated concepts into a single neuron', Elhage et al. 2022), so a unit's top-token list is a hypothesis to test with ablations rather than a label.",
   "refs": [
    "geva2022",
    "elhage2022"
   ]
  },
  {
   "placement": "The MoE panel, comparing total parameters with the parameters and FLOPs used per token. Every preset is dense, so this is a comparison rather than something they show.",
   "text": "In a Switch-style MoE layer, a router sends each token to a single expert, so the parameter count rises with the number of experts while FLOPs per example stay constant (Fedus et al. 2021, Secs. 2 and 2.1). Every expert's weights still have to be stored even though each token uses only one, and an auxiliary loss is added to encourage a balanced load across experts (Sec. 2.2).",
   "refs": [
    "fedus2021"
   ]
  }
 ],
 "residual": [
  {
   "placement": "Residual stream norms across depth (pre-LN presets)",
   "text": "In these pre-LN presets nothing normalises the residual stream between blocks: each attention or MLP output is added into it, as Elhage et al. (2021) describe, so resid_out can keep growing with depth. ln1 and ln2 normalise only the copy of the stream that each sub-layer reads (mean removed, divided by the standard deviation, then gain applied), and the stream itself is left unchanged. Sun et al. (2025) report that output variance grows with depth in large Pre-LN LLMs, so for these 2- and 4-layer presets compare the direction of the trend across layers, not its size, and check it in the resid_out traces.",
   "refs": [
    "elhage2021",
    "sun2025curse"
   ]
  },
  {
   "placement": "Pre-LN vs post-LN toggle, and gradient scale by layer",
   "text": "Post-LN normalises after each add, so resid_mid and resid_out keep a per-token scale set by the LayerNorm gain and bias instead of accumulating with depth. Xiong et al. find that at initialisation the expected gradient scale grows with layer index in Post-LN but 'almost keeps the same' across layers in Pre-LN, and they report that the warm-up stage is 'practically helpful for avoiding this problem'. Changing placement in the settings drawer re-initialises the weights (only pre-LN has ln_f), so compare placements from random init rather than on the trained presets.",
   "refs": [
    "xiong2020"
   ]
  },
  {
   "placement": "LayerNorm vs RMSNorm, step by step",
   "text": "LayerNorm subtracts the per-token mean before dividing by the standard deviation, so the normalised vector has zero mean before its gain; RMSNorm divides by the root mean square alone, so its output can keep a non-zero mean. Zhang and Sennrich hypothesise that re-centring is dispensable and report that RMSNorm reduces running time by 7% to 64% on different models in their experiments; the llama preset's norm has a gain but no bias.",
   "refs": [
    "zhang2019rms"
   ]
  },
  {
   "placement": "Gradient flow with and without residuals",
   "text": "With identity shortcuts, He et al. show that 'the gradient of a layer does not vanish even when the weights are arbitrarily small'. The residual-off switch keeps the trained weights, so it shows what the trained function loses without its skip path, not what training without skips would learn; Dong et al. find that without skip connections or MLPs 'the output converges doubly exponentially to a rank-1 matrix', a related but more extreme setting, since this view keeps the MLPs.",
   "refs": [
    "he2016identity",
    "dong2021rank"
   ]
  },
  {
   "placement": "Logit lens, layer by layer",
   "text": "In the logit lens the early layers 'sometimes look like nonsense', and the distributions 'gradually converge to the final distribution over the layers' (nostalgebraist). Belrose et al. call the lens 'often brittle' and define it as LayerNorm followed by the unembedding, so for pre-LN presets include <code>lnf</code> at every depth; post-LN has no <code>lnf</code>, so its block outputs go straight to the unembedding.",
   "refs": [
    "logit-lens",
    "belrose2023"
   ]
  }
 ],
 "head": [
  {
   "placement": "The lnf and logits traces, just before the next-token probability bars. src/views/head.js is still a stub at the time of this check, so this describes the planned panels.",
   "text": "In nanoGPT's forward pass the final LayerNorm (<code>ln_f</code>) runs right before <code>lm_head</code>, so the logits are the unembedding applied to the normalised final residual vector; every preset here is pre-norm and follows the same order. The logit lens applies the same final projection (multiplication by the transpose of the embedding matrix W, in GPT-2) to intermediate layers' activations, and in GPT-2 1558M it reports that 'at some point in the middle' the model has formed a 'pretty good guess' for the next token, which later layers seem to refine.",
   "refs": [
    "nanogpt",
    "logit-lens"
   ]
  },
  {
   "placement": "The temperature control and the probability bars it reshapes (the probs trace).",
   "text": "Hinton et al. write the class probabilities as q_i = exp(z_i/T) / (sum over j of exp(z_j/T)), where T is 'normally set to 1', and state that 'Using a higher value for T produces a softer probability distribution over classes.' Dividing the logits by T before the softmax therefore controls how peaked the next-token distribution is.",
   "refs": [
    "hinton2015"
   ]
  },
  {
   "placement": "The top-k and top-p controls, and the greedy (argmax) choice next to them.",
   "text": "Top-k keeps a fixed number of candidates whatever the distribution looks like, and Holtzman et al. point out that a fixed k can be a poor fit: the head of the next-word distribution 'can be flat across tens or hundreds of reasonable options' in some contexts, while in peaked contexts most of the mass sits on one or a few tokens. Nucleus (top-p) sampling keeps the smallest set whose cumulative probability reaches p, so the candidate set grows and shrinks with the model's uncertainty, and the paper reports that likelihood-maximising decoding gives text that is 'bland and strangely repetitive'.",
   "refs": [
    "holtzman2019"
   ]
  },
  {
   "placement": "The lm_head (unembedding) matrix in the parameter breakdown, next to the embedding matrix in the tokens view.",
   "text": "The four character presets do not tie their weights, so <code>lm_head</code> is a separate 32 x 64 matrix of 2,048 parameters, about 1 to 2% of each (2,048 of 208,256 for the charlm). The TinyStories preset ties them, so its output matrix is the 4,096 x 128 embedding itself. Press and Wolf recommend tying the input embedding and the output embedding, and their Table 5 gives PTB test perplexity of 74.3 with tying against 78.4 without, for the large LSTM with dropout (the 78.4 row is labelled 'Large [Zaremba et al. 2014]'). The logit lens describes GPT-2's final step as multiplying the final vector by the transpose of the same embedding matrix W.",
   "refs": [
    "press2016tying",
    "logit-lens"
   ]
  },
  {
   "placement": "The loss readout and the perplexity number beside it (the loss trace; the charlm manifest records 0.5464 nats).",
   "text": "The head's loss is the average negative log-probability of the true next token, measured in nats as in Kaplan et al., and perplexity is its exponential: the charlm's recorded 0.5464 nats is a perplexity of about 1.73, against 32 for a uniform guess over the 32 symbols (ln 32 = 3.47 nats). Because that number is measured on training text, it describes fit rather than generalisation, since perplexity is normally measured on held-out data. Perplexity also depends on the tokenizer, so these character-level values are not directly comparable with subword models.",
   "refs": [
    "kaplan2020",
    "meister2021",
    "nanogpt"
   ]
  }
 ],
 "generation": [
  {
   "placement": "Cached vs uncached cost comparison",
   "text": "Without a cache, each new token recomputes the keys and values of every earlier token; kipply notes that without a KV cache, sampling would be quadratic in time complexity as the number of tokens grows. With the cache, each step computes K and V for only the new token, but attention still reads every cached row, which the scaling book describes as 'a tiny amount of FLOPs while loading a massive KV cache', so per-step cost still grows with length. Pope et al. say that at larger batch sizes and sequence lengths (e.g. 2048+ tokens with batch size 512+), the time to load the KV cache dominates.",
   "refs": [
    "kipply",
    "scaling-book",
    "pope2022"
   ]
  },
  {
   "placement": "KV-cache memory: bytes per token for real models and the two presets",
   "text": "The engine's sliding-window mask (a config option that no shipped preset uses) hides old rows but never evicts them, so the cache grows linearly with the cached length until it holds block_size tokens, which is 64 in the two Alice presets (decodeStep then throws 'context window is full'). Mistral 7B's rolling buffer, by contrast, has 'a fixed size of W' and 'the size of the cache stops increasing'.",
   "refs": [
    "scaling-book",
    "jiang2023mistral",
    "ainslie2023gqa"
   ]
  },
  {
   "placement": "Prefill vs decode",
   "text": "Prefill runs the whole prompt in one pass, and the scaling book says that during prefill 'all matrix multiplications are basically always compute-bound'. Each decode step emits one token, and Pope et al. say that 'at small batch sizes and sequence lengths, the time to load weights dominates'. Using kipply's A100 figures (312e12 FLOP/s and 1.5e12 bytes/s) and his 16-bit assumption, the weight matmuls are memory-bandwidth bound below a batch of about 208; his threshold is derived for the weights, not for reads of the KV cache.",
   "refs": [
    "scaling-book",
    "pope2022",
    "kipply"
   ]
  },
  {
   "placement": "Paged attention: contiguous reservation vs fixed-size blocks",
   "text": "Reserving each request's maximum length up front wastes memory: the vLLM paper reports that existing systems used only 20.4% to 38.2% of their KV-cache memory for actual token states. PagedAttention allocates fixed-size blocks through a block table, and the paper says vLLM 'limits all the memory wastes for a request within one block', while parallel samples or beam candidates can map to the same physical blocks.",
   "refs": [
    "kwon2023paged"
   ]
  },
  {
   "placement": "Speculative decoding: draft, verify and speed-up",
   "text": "A cheaper approximation model proposes several tokens and the target model checks them in one parallel pass, so the output distribution is unchanged and the gain depends on how often the target accepts the proposals. In Leviathan et al.'s analysis the expected tokens per target call is (1 - α^(γ+1)) / (1 - α) and the walltime factor divides that by (γc + 1), so at low acceptance rates the factor can fall below 1; Chen et al. report a 2 to 2.5x decoding speedup for Chinchilla 70B in a distributed setup.",
   "refs": [
    "leviathan2022",
    "chen2023spec"
   ]
  }
 ],
 "lab": [
  {
   "placement": "The 'remove residuals' A/B result: the loss change reflects the whole stream, not any one block.",
   "text": "With residuals off, each block's output replaces the stream instead of being added to it, so an earlier layer's write reaches later layers only if a later sublayer recomputes it. Elhage et al. describe the residual stream as the shared channel through which components 'communicate with each other by reading and writing to different subspaces of the residual stream', so a large loss change reflects that channel more than any single block. Dong et al. show that pure attention stacks, with neither skip connections nor MLPs, converge doubly exponentially to a rank-1 matrix; this run keeps the MLPs, so it does not test that setting.",
   "refs": [
    "elhage2021",
    "dong2021rank"
   ]
  },
  {
   "placement": "The 1/√d_h switch: d_h is 16 in the character presets and 32 in TinyStories, so turning the scale off multiplies each score by √d_h.",
   "text": "Turning off the 1/√d_h scale multiplies every attention score by √d_h: by √16 = 4 in the character presets and by √32 ≈ 5.66 in TinyStories, so each softmax sharpens while the weights stay the same. Vaswani et al. motivate the scale by noting that, for large d_k, the dot products 'grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients'. At d_h = 16 or 32 the switch mainly acts like a temperature change on attention, so expect sharper heatmaps rather than a new mechanism.",
   "refs": [
    "vaswani2017"
   ]
  },
  {
   "placement": "The positions A/B: the GPT-2-style preset (learned table) against the LLaMA-style preset (RoPE).",
   "text": "Removing a learned position table from a model trained with one puts the network in a state it never saw, and a decoder trained without positional encoding (NoPE) is a different model. Kazemnejad et al. report that 'explicit position encodings are not essential for decoder-only Transformers to generalize well to longer sequences'; their theory says NoPE 'can represent both absolute and relative PEs', and the absolute-position proof (Appendix C.1) 'relies on the causal attention mask'. In this app, switching learned positions off changes the parameter list and re-initialises random weights, so zero the table instead to keep the trained weights; the RoPE preset has no table and keeps its weights.",
   "refs": [
    "kazemnejad2023"
   ]
  },
  {
   "placement": "Zero a head on the induction preset: layer-0 head 0 (95% of attention on the previous token) and layer-1 heads 0, 2 and 3 (60-84% on the induction target), as measured in this project.",
   "text": "Olsson et al. describe induction as 'a pair of attention heads in different layers' and report that when induction heads are knocked out at test time in small models, 'the amount of in-context learning greatly decreases'. Michel et al. report that 'a large percentage of attention heads can be removed at test time without significantly impacting performance', so zeroing one layer-1 head may change little; whether its sibling heads take over the same pattern is a hypothesis to test here, not a result of that paper. Wang et al. write that 'we find zero ablation to lead to noisy results in practice' and use mean ablation instead, so check any large drop against a mean-ablated run before reading it as a circuit.",
   "refs": [
    "olsson2022",
    "michel2019",
    "wang2022ioi"
   ]
  },
  {
   "placement": "The swap-layers A/B on the 2- and 4-layer presets: the loss change for each swapped pair of blocks.",
   "text": "Sun et al. find that 'the lower and final layers of pretrained transformers differ from middle layers' while 'middle layers have a surprising amount of uniformity', and they report robustness to 'running the layers in an order different from how they were trained' for some classes of problems. The models they study have 24 to 80 layers, far deeper than the 2 to 4 layer presets here, so treat their pattern as a hypothesis to test in this lab rather than a replication.",
   "refs": [
    "sun2024painters"
   ]
  }
 ],
 "scaling": [
  {
   "placement": "Parameter-count rows (GPT-2, GPT-3, LLaMA-2/3, Mistral)",
   "text": "Counts follow from shape: Kaplan et al. define N as non-embedding parameters and approximate <code>N ≈ 12·n_layer·d_model²</code> for d_attn = d_model and d_ff = 4·d_model, which gives about 174B for GPT-3 175B (96 layers, d_model 12288), so the blocks hold nearly all of the published total. For gated-MLP models such as LLaMA, the scaling book says the MLP block 'dominates the total parameter count', so the feed-forward layers usually hold most of the parameters.",
   "refs": [
    "kaplan2020",
    "brown2020",
    "scaling-book"
   ]
  },
  {
   "placement": "FLOPs-per-token figures (forward pass vs training)",
   "text": "Kaplan et al. estimate about 2N FLOPs per token for a forward pass and about 6N for training, so one trained token costs roughly three forward passes' worth of compute (for GPT-3 175B, about 350 GFLOPs forward and 1.05 TFLOPs training per token). Their extra attention term is a small fraction when d_model > n_ctx/12, so per-token FLOPs track the parameter count unless the context is long relative to the model width.",
   "refs": [
    "kaplan2020",
    "transformer-math",
    "scaling-book"
   ]
  },
  {
   "placement": "Training memory bars: weights and Adam state",
   "text": "With the usual fp16 mixed-precision recipe, Adam needs about 16 bytes of model state per parameter: 2 for fp16 weights, 2 for fp16 gradients and 12 for the fp32 master copy, momentum and variance, so a 7B model needs about 112 GB before activations. The optimizer's 12 bytes are six times the fp16 weights, and activations and temporary buffers are separate residual states that this figure leaves out.",
   "refs": [
    "rajbhandari2019",
    "transformer-math"
   ]
  },
  {
   "placement": "KV-cache memory bar",
   "text": "Per token, the KV cache stores 2 (keys and values) × bytes × layers × KV heads × head dimension, so it grows with batch size and context length and depends on the layer and KV-head counts rather than on the weight count. In Weng's example, a batch of 512 at 2,048 tokens needs 3 TB of KV cache, about three times the model size. Sharing key-value heads cuts this directly: Mistral 7B's 8 KV heads for 32 query heads make its cache four times smaller than full multi-head attention at the same head size, and Llama 2 uses GQA only at 34B and 70B.",
   "refs": [
    "scaling-book",
    "kipply",
    "weng-inference",
    "jiang2023mistral",
    "touvron2023b"
   ]
  },
  {
   "placement": "Chinchilla panel (tokens per parameter)",
   "text": "The rule of thumb of about 20 tokens per parameter is an implied, approximate compute-optimal ratio: Table 3's rows give about 20 to 22 (1B with 20.2B tokens, 10B with 205.1B, 67B with 1.5T), and the paper does not state the ratio in words. Deployed models sit far on both sides of it, from GPT-3 175B on 300B tokens (about 1.7 per parameter) to LLaMA-7B on 1.0T (about 140); Llama 3 says its smaller models 'perform better than compute-optimal models at the same inference budget', and Sardana et al. favour 'smaller and longer than Chinchilla-optimal' when inference demand is large. Besiroglu et al. (2024) call Hoffmann et al.'s reported third-approach confidence intervals 'implausibly narrow', but their own re-derivation of that approach is compatible with the first two approaches, and Table 3 comes from Approach 1, so treat 20 as a rough guide rather than a precise constant.",
   "refs": [
    "hoffmann2022",
    "brown2020",
    "touvron2023",
    "grattafiori2024",
    "sardana2024",
    "besiroglu2024"
   ]
  }
 ],
 "encdec": [
  {
   "placement": "The mask grids for BERT, GPT, T5 and the prefix LM, which are the first thing to compare. (src/views/encdec.js is still a placeholder stub, so this points at the planned panels; recheck when the view is built.)",
   "text": "BERT attends in both directions, so its grid has no causal triangle, while GPT's is lower-triangular because each token sees only earlier tokens. In T5's encoder-decoder, the encoder and the cross-attention are fully visible and only the decoder's self-attention is causal, and a prefix LM makes just the prefix fully visible.",
   "refs": [
    "devlin2018",
    "vaswani2017",
    "raffel2019"
   ]
  },
  {
   "placement": "The cross-attention block in the encoder-decoder diagram, read for which side supplies the queries and which supplies the keys and values.",
   "text": "In encoder-decoder attention the queries come from the previous decoder layer, while the memory keys and values come from the output of the encoder, so the score matrix has one row per target token and one column per source token. The encoder processes the whole input before decoding begins, and each decoder layer's encoder-decoder attention reads that same encoder output.",
   "refs": [
    "vaswani2017",
    "alammar-illustrated",
    "raffel2019"
   ]
  },
  {
   "placement": "The 'cross-attn' label in the view title and any BERT panel, where the same word can mean a different mechanism.",
   "text": "The BERT paper also uses 'bidirectional cross attention', but there it refers to the two sentences of one packed input inside a single self-attention stack, with no decoder. In this view, cross-attention means only the decoder-to-encoder block.",
   "refs": [
    "devlin2018"
   ]
  },
  {
   "placement": "Any side-by-side architecture comparison in the view. The GLUE column of T5's Table 2 is the one numeric anchor here.",
   "text": "On GLUE in T5's Table 2, the encoder-decoder with denoising scores 83.28, its weight-shared version 82.81, and the decoder-only language model 74.70; with the autoregressive (LM) objective, the decoder-only model scores 73.78 against 79.56 for the encoder-decoder. The encoder-decoder rows have 2P parameters against P for the language-model rows, and T5 says an encoder-decoder cannot be compared with a language model on parameters and compute at the same time, so these rows are not parameter-matched. Wang et al. 2022 found causal decoder-only models strongest for zero-shot generalization after purely unsupervised pretraining, so the ranking depends on the setting.",
   "refs": [
    "raffel2019",
    "wang2022arch"
   ]
  },
  {
   "placement": "Cross-attention as the VLM bridge: place it next to the cross-attention block for readers coming from vision-language work.",
   "text": "Flamingo keeps the pretrained language-model blocks frozen and inserts new gated cross-attention blocks, trained from scratch, whose keys and values come from visual features. Each gate is tanh(α) for a learnable scalar α initialized at 0, so at initialization the added branch is skipped and the model output matches the pretrained LM.",
   "refs": [
    "alayrac2022flamingo"
   ]
  }
 ],
 "vision": [
  {
   "placement": "Patch grid and visual-token count readout",
   "text": "With P×P patches, an H×W image gives <code>N = HW/P²</code> tokens, so a 336 px image with 14 px patches gives 24×24 = 576 patch tokens, and the ViT paper notes that the sequence length is 'inversely proportional to the square of the patch size'. Beyond cutting the image into patches (and interpolating position embeddings when the resolution changes), a ViT has little built-in 2D structure: its learned 1D position embeddings start with no 2D information, and the authors report no significant gain from 2D-aware ones.",
   "refs": [
    "dosovitskiy2020",
    "radford2021clip",
    "liu2023llava15"
   ]
  },
  {
   "placement": "Projector block between the vision encoder features and the LLM input",
   "text": "The projector maps the encoder's patch features into the LLM's word-embedding width, so each visual token has the same dimension as a text token. LLaVA used one trainable matrix W for this, and LLaVA-1.5 reports that a two-layer MLP connector can improve LLaVA's multimodal capabilities over that linear projection.",
   "refs": [
    "liu2023llava",
    "liu2023llava15"
   ]
  },
  {
   "placement": "CLIP encoder panel and its patch features",
   "text": "CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples, using a dataset of 400 million pairs. Its limitations section says that zero-shot performance is still quite weak on several kinds of tasks, that CLIP struggles with counting the number of objects in an image, and that it performs poorly on several fine-grained classification tasks such as differentiating models of cars, species of flowers and variants of aircraft. LLaVA-style models feed CLIP's patch features to the LLM, so checking counts and near-identical items against the picture is a sensible precaution; that last step is an inference, not a result either paper reports.",
   "refs": [
    "radford2021clip",
    "liu2023llava"
   ]
  },
  {
   "placement": "Token-merge control: visual token count before and after merging",
   "text": "Qwen2-VL runs a small MLP over each 2×2 block of ViT output tokens, so the LLM receives about a quarter as many visual tokens; its worked example gives 66 tokens for a 224×224 image with patch size 14 (256 patches merge to 64, plus the vision start and end special tokens the paper describes). Learned visual resamplers, which also cut the patch count, have a training cost: LLaVA-1.5 says they 'cannot achieve convergence as efficiently as LLaVA with a comparable amount of training data'. ToMe instead merges similar tokens inside the ViT, and its abstract reports 2x throughput on images (ViT-L@512 and ViT-H@518) for a 0.2–0.3% accuracy drop.",
   "refs": [
    "wang2024qwen2vl",
    "liu2023llava15",
    "bolya2022tome"
   ]
  }
 ],
 "cheatsheet": [
  {
   "placement": "Training-compute card: C ≈ 6ND and FLOPs per token",
   "text": "Training compute is about 6 FLOPs per parameter per token: roughly 2 for the forward pass and 4 for the backward pass. Kaplan et al. define N as non-embedding parameters (N ≈ 12 n_layer d_model² for the standard shape), and their forward pass adds an attention term, <code>2 n_layer n_ctx d_model</code>, that the 6N rule leaves out; at a 1,024-token context with d_model = 768 that term is about 11% of 2N (our arithmetic).",
   "refs": [
    "kaplan2020",
    "transformer-math"
   ]
  },
  {
   "placement": "Training-memory card: 16 bytes per parameter",
   "text": "In the ZeRO accounting, mixed-precision Adam needs 16 bytes per parameter: 2 for fp16 weights, 2 for fp16 gradients and 12 for optimizer states (an fp32 copy of the weights, momentum and variance), so a 7B-parameter model's model states take about 112 GB before activations (our arithmetic). Activation memory is extra and grows with batch size b and sequence length s: EleutherAI's estimate is linear in b·s plus an attention-score term that grows as s², which selective recomputation removes.",
   "refs": [
    "rajbhandari2019",
    "transformer-math"
   ]
  },
  {
   "placement": "KV-cache card and the MHA, MQA and GQA cards",
   "text": "The KV cache stores one key and one value per layer, per KV head and per token, and Kipply's formula, which uses n_heads and does not discuss shared K/V heads, gives 2 × 2 × n_layers × n_heads × d_head bytes per token at 16-bit precision. Sharing K/V heads shrinks the cache in proportion, so 8 KV heads serving 64 query heads give an 8× smaller cache than 64 KV heads (our arithmetic), and uptrained GQA (from multi-head checkpoints) reached performance close to MHA-XXL at speed close to MQA on T5-XXL (Ainslie et al., Sec. 3.2). Pope et al. report that, with appropriate partitioning, the lower memory requirements of multiquery attention enabled up to 32× longer context lengths.",
   "refs": [
    "kipply",
    "ainslie2023gqa",
    "shazeer2019mqa",
    "pope2022"
   ]
  },
  {
   "placement": "Attention card: what FlashAttention changes and what it does not",
   "text": "FlashAttention computes exact attention without writing the N × N score and probability matrices to GPU high-bandwidth memory, so its extra memory is O(N) rather than O(N²). The saving comes from fewer memory reads and writes, not fewer FLOPs: the backward pass recomputes S and P, and the authors report that the run is still faster despite the extra FLOPs.",
   "refs": [
    "dao2022flash"
   ]
  },
  {
   "placement": "Residual and norms card: pre-LN vs post-LN",
   "text": "Xiong et al. argue from mean-field analysis at initialization that post-LN (layer norm between the residual blocks) has large expected gradients near the output layer, which is why warm-up helps; pre-LN gradients are well-behaved there, and they remove warm-up when training pre-LN models. The reference model defaults to <code>norm_placement = \"pre\"</code> (reference/model.py, line 33), which its comment attributes to GPT-2 and LLaMA.",
   "refs": [
    "xiong2020"
   ]
  }
 ]
};

// atlas node pattern (first segment) -> topic ids whose notes apply
const TOPICS = {
  model: ['overview'], tokens: ['tokens'], embedding: ['tokens'], 'token-embedding': ['tokens'], positional: ['positions'],
  block: ['residual', 'overview'], norm: ['residual'], attention: ['attention', 'multihead'], qkv: ['attention'], head: ['multihead', 'attention'],
  scores: ['attention'], mask: ['masking'], softmax: ['attention'], mix: ['attention'], 'out-proj': ['multihead'], gqa: ['multihead'],
  residual: ['residual'], mlp: ['mlp'], activation: ['mlp'], neurons: ['mlp'], 'final-norm': ['residual'], 'lm-head': ['head'], sampling: ['head'],
  generation: ['generation'], cheatsheet: ['cheatsheet'], scaling: ['scaling'], lab: ['lab'], encdec: ['encdec'], vision: ['vision'],
  tower: ['overview', 'stepper'], stepper: ['stepper'],
};

export function topicsFor(nodeId) { return TOPICS[nodeId.split('.')[0]] || []; }

export function notesFor(nodeId) { return topicsFor(nodeId).flatMap((t) => NOTES[t] || []); }

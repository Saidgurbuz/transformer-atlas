# What the Python reference covers

`reference/model.py` is a complete runnable **educational decoder-only model** for the
architecture switches in the app. It includes embeddings, position encodings, MHA/GQA/MQA,
masked attention, normalization, residuals, MLPs, the vocabulary projection, next-token loss,
sampling, and batched incremental decoding. It uses ordinary PyTorch operations so every
intermediate calculation remains readable; it is not a reproduction of every production model.

The surrounding code is intentionally separate:

| Responsibility | Source |
| --- | --- |
| Autograd and parameter gradients | PyTorch; `src/engine/tensor.js` for the browser |
| Manual backward pass for studying derivatives | `reference/numpy_transformer.py` |
| Data, optimizer steps, checkpoints, trained presets | `tools/train_presets.py`, `tools/train_bpe_preset.py` |
| Character and byte-level BPE tokenization | `src/engine/tokenizer.js`, `src/engine/bpe.js`, BPE training tools |
| Browser sampling and incremental decoding | `src/engine/generate.js` |

Encoder–decoder attention and vision-token pages are conceptual illustrations. Padding-aware
batched training, dropout, fused/FlashAttention kernels, checkpoint-format interoperability,
quantization and distributed execution are outside this reference's implemented scope.

## Cached decoding

```python
model.eval()
cache = model.create_cache()
for t in range(prompt.shape[1]):
    logits = model.decode_step(prompt[:, t:t + 1], cache)  # [B, 1, V]
```

Caching supports causal and causal sliding masks. Keys are rotated before storage for RoPE,
and shared KV heads are stored before GQA repetition. The cache retains the prefix up to
`block_size`; sliding masks limit reads without evicting stored rows. Recreate the cache after
changing the prompt, weights, architecture, device or dtype. `generate()` is the simple full
forward implementation, with context cropping; `temperature=0` selects greedy decoding.

## Conventions and correspondence

- Linear weights use PyTorch's `[out, in]` storage: row-vector equations use `xWᵀ + b`.
- RoPE uses interleaved coordinate pairs. ALiBi uses geometric slopes for all head counts;
  the authors' non-power-of-two slope ordering differs. Atlas applies absolute-distance bias
  for its bidirectional/prefix experiments.
- LayerNorm uses biased variance; RMSNorm has no learned shift. Both include `norm_eps`.
- The app omits the batch axis when displaying tensors. Cache execution has different
  intermediate shapes; the standard inspector displays the full-sequence forward trace.
- `@trace` markers identify producer lines. The code viewer resolves alternatives from the
  current configuration and expands relevant helper implementations. The coordinates are
  zero-based and remain explicit when opening code, maths, or the inspector.

Run `node tools/gen_codemap.mjs` after changing the Python source. Builds and
`node tests/codemap.mjs` reject stale embedded source. The Python regression checks the current
source against stored traces and gradients and compares cached/full decoding in 60 combinations
of position encoding, mask, normalization placement, KV sharing, and float32/float64 dtype.

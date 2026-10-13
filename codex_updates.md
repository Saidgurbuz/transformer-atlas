# Codex updates

## 2026-10-13 · Encoder–decoder reference

Added a complete miniature encoder and decoder with bidirectional, causal and rectangular cross-attention, post-LayerNorm and shared embeddings. All traced tensors match PyTorch across four sequence configurations, with padding-invariance and causality checks.

## 2026-10-13 · Online attention reference

Added stable tiled softmax with an unnormalised output accumulator and per-tile state. Dense PyTorch checks cover every prefix, extreme logits and masked tiles; this CPU teaching recurrence makes no GPU speed claims.

## 2026-10-13 · Training lesson

Added a five-step backward-pass lesson with shifted targets, actual per-parameter gradients and consecutive optimizer updates, in the existing Atlas palette and motion. The mini model is isolated from the main model, and browser checks cover both profiles, both optimizers, source correspondence and card bounds.

## 2026-10-13 · Training numerical reference

Added a tiny isolated training model with mean next-token loss, captured gradients and real SGD/AdamW updates. Independent PyTorch checks cover every parameter, ignored targets and tied embeddings across three consecutive updates.

## 2026-10-13 · Tensor operations

Added an animated, shape-first tensor lab for arange, view, transpose, reshape, contiguous, cat, einsum and singleton axes, with offsets hidden by default. Its executable PyTorch counterpart verifies values, strides and storage sharing; browser checks cover every step, source links and panel bounds.

## 2026-10-13 · v1.3.0 · main

Improved traced cell ↔ Python source navigation with exact coordinates, active branches and helper implementations, and corrected configuration-specific equations with relevant paper locations. Added reference cached decoding and device/dtype handling, fixed source/drawer layout and browser history, reduced redundant rendering, and strengthened regression/build checks so the displayed computation remains correct and reproducible.

Validation: 549 numerical parity checks; 12,545 maths/reference checks; 323 source round trips; 60 cached-decoding configurations; 31 browser interaction checks; 66 route/theme checks; standalone and desktop/mobile screenshot inspection.

# Changelog

## 1.4.0 (2026-10-13)

- Sixteen derivation, shape, debugging and coding exercises with worked solutions, timed worker execution and browser-local progress.
- Architecture recipe comparison with explicit conventions, optional unique parameter counts and primary-paper links.
- Executable encoder–decoder lesson with rectangular cross-attention and source/target padding examples.
- Live-trace FlashAttention lesson with stable online softmax, masked tiles and explicit memory/performance conventions.
- Complete miniature training lesson with actual backward gradients, ignored targets, tied embeddings, SGD and AdamW.
- Shape-first tensor-operation lesson with animated element movement, copying/stride explanations and executable PyTorch correspondence.

- Corrected the remaining softmax display mask equation, preserved lesson source routing after cell selection, and made training gradient colors follow theme switches.

## 1.3.0 (2026-10-13)

- Exact cell ↔ Python source navigation, active-branch links, helper implementations, and consistent GQA/KV coordinates.
- Configuration-specific equations, corrected Boolean masking, deeper cache equations, and paper locations beside the maths.
- Reference PyTorch cached decoding, device/dtype-aware position tables, safer sampling and configuration validation.
- Wrapped source and symbol layouts, working browser Back/Forward, reusable stepper timelines, and inspector cleanup.
- Fixed standalone bundling, embedded-source freshness checks, and smoke tests that fail on errors, route mismatches, or overflow.


## 1.2.0 (2026-10-12)

- **A new 3D overview**, chosen over an independent alternative in a blind design review and
  refined over two more rounds. Every layer is an attention + MLP pair on a glass plate (the MLP
  raised: it runs second); the residual stream is a thick, labelled rod with a ⊕ where each
  sub-layer adds back. Live data drawn as geometry: each head's real attention pattern on its own
  screen, the MLP's most active units as sign-coloured bars, the real next-token distribution
  under a headline prediction. Saturated colour in both themes; in dark mode only data glows.
- **No flicker:** hover changes colour only, the camera never moves on its own, and a still scene
  measures 0 changed pixels (new `tools/frames.sh` checks this). One calm forward-pass animation:
  a light rises up the stream, each layer lights up and its units grow, the prediction appears.
- Phone layout with a colour legend; the alternative design stays viewable with `?root=b`.

## 1.1.0 (2026-10-12)

- **Subword model, now the default:** a GPT-2-style model trained on TinyStories with a
  4,096-token byte-level BPE vocabulary, so most words are one token (held-out loss 1.98
  nats/token). The browser tokenizer matches the Python one exactly. Every level handles the
  large vocabulary: top-k views for the LM head and sampling, word-sized tiles everywhere.
  The character models remain one click away; `?preset=<id>` links to any model.
- **Tokens, for real:** with the subword model the Tokens level shows the actual tokenizer:
  the word split, a ladder of each word's real merges from its bytes (type any word), the
  ids, and the composition of the vocabulary.
- **Read the code, zoom the model:** click, select or right-click any line of
  `reference/model.py` to fly to the part of the model it computes.
- **A new 3D overview** built on the logic of the original paper's figure: inputs at the
  bottom, a label rail, residual loops, one open layer with the others stacked as wafers, the
  real attention patterns, residual vectors and output distribution, and a forward-pass
  animation in which the predicted token flies back into the input.
- Verified: parity 549/549, every equation on every model (10,249 live checks), tokenizer
  parity on 204 texts, preset logits against PyTorch.

## 1.0.0 (2026-10-12)

First public release of **Transformer Atlas**.

- Full-screen, zoomable atlas of a decoder-only Transformer: a 3D model at the top, then one
  focused, animated scene per component (28 levels from tokens to sampling and the KV cache).
- Every number comes from a real forward pass of a small trained model running in the browser;
  scenes that recompute a value show a check against the model's own value.
- Hover any tensor for its equation with live numbers; right-click for the maths, the exact line
  of `reference/model.py`, or every value. Guided story mode with the arrow keys.
- Four trained presets: Alice char-LM (GPT-2 style and LLaMA style), induction heads, word reversal.
- Code stepper, interview cheat sheet, parameter/FLOP/memory calculator, ablation lab,
  encoder-decoder and vision-token chapters.
- Verified: JS engine vs PyTorch (549 checks), 57 equations with 9,031 live checks, and a NumPy
  implementation with hand-written backward passes matching PyTorch; 79 notes and 103 sources
  checked against their originals.

## 0.1 (git tag v0.1)

The earlier tensor-tower explorer (all tensors at once in 3D), still available from the menu.

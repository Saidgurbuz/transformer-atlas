# Changelog

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

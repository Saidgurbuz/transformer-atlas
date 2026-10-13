# Codex updates

## 2026-10-13 · Tensor operations

Added an animated, shape-first tensor lab for arange, view, transpose, reshape, contiguous, cat, einsum and singleton axes, with offsets hidden by default. Its executable PyTorch counterpart verifies values, strides and storage sharing; browser checks cover every step, source links and panel bounds.

## 2026-10-13 · v1.3.0 · main

Improved traced cell ↔ Python source navigation with exact coordinates, active branches and helper implementations, and corrected configuration-specific equations with relevant paper locations. Added reference cached decoding and device/dtype handling, fixed source/drawer layout and browser history, reduced redundant rendering, and strengthened regression/build checks so the displayed computation remains correct and reproducible.

Validation: 549 numerical parity checks; 12,545 maths/reference checks; 323 source round trips; 60 cached-decoding configurations; 31 browser interaction checks; 66 route/theme checks; standalone and desktop/mobile screenshot inspection.

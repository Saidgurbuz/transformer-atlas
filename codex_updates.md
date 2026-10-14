# Codex updates

## 2026-10-14 · Isolate analytics in live browser checks

Stubbed only external analytics in the browser harness so verification neither sends synthetic visits nor fails on blocked telemetry. All 95 live high-DPI render-budget and shadow/animation assertions pass with no browser errors; application analytics are unchanged.

## 2026-10-14 · v1.5.5 dark-mode performance

Published direct dark rendering, shadow reuse and GPU resource cleanup in separate tested updates. Verified retina/mobile and standalone layouts, stable resource budgets, zero idle renders, pixel-equivalent shadows, animation and existing prediction/focus behavior.

## 2026-10-14 · Reuse static shadows

Camera movement and hover now reuse the overview’s shadow map, while geometry rebuilds and forward-pass animation refresh it. Forty-six browser assertions confirm pixel-identical cached/fresh shadows and correct animation updates; the tested static frame drops from 202 to 157 actual GPU draw commands in either theme.

## 2026-10-14 · Direct dark-mode rendering

Removed the dark-only HDR/multisample bloom pipeline from both overviews, reducing the default scene from 15 renderer passes to one while retaining lighting, emissive colors and antialiasing. Fixed environment-target disposal; browser checks verify stable theme resources, high-DPI rendering, resizing and no idle GPU redraws, with before/after visual review.

## 2026-10-14 · v1.5.4 contextual study and project links

Prepared the scoped study navigation and quiet menu credit for release, with automatic deployment from main. Validation covers 279 study links, 33 executable regions, desktop/mobile and standalone study flows, existing cell/source navigation and visual review of the menu/footer.

## 2026-10-14 · Project links in the menu

Added a discreet menu footer with an inline GitHub mark, repository link and author credit linking to the verified personal website. Desktop, dark mobile and standalone checks confirm keyboard access, correct destinations and a visible footer below the scrolling navigation; screenshots were visually reviewed.

## 2026-10-14 · Equation fallback after preset changes

When a focused tensor disappears after an architecture change, Maths now falls back to the current view’s equations. The standalone study suite passes 198 assertions, including the learned-position to RoPE transition, and all 31 cell/source navigation regressions pass.

## 2026-10-14 · Contextual further study

Replaced broad topic lists with equation-specific links that respect the active architecture, and made open Maths drawers follow view/preset changes. Validated 279 links, 33 executable source regions and 392 desktop/mobile browser assertions, including explanation/source round trips; added link integrity to deployment checks.

## 2026-10-14 · Pages deployment from main

Added a main-only Actions workflow that checks the model/source correspondence, builds the existing app and publishes an artifact directly to Pages. Kept the manual deployment command as a workflow trigger and updated the development guide; publication files and the public URL stay the same.

## 2026-10-14 · Reusable Pages build

Extracted the publication files into a shared local/CI build command, preserving the current deployment's file layout. The package matches the deployed app byte for byte, apart from its source-commit metadata, and both shell scripts pass syntax checks.

## 2026-10-14 · v1.5.3 presentation update

Prepared the linked screenshot gallery and consistent page setup for publication. The build, focused integration checks, 47 source/standalone browser assertions and 65 documentation links pass; all captures and README layouts were visually reviewed.

## 2026-10-14 · Linked walkthrough screenshots

Added an actual 3D overview and a compact gallery covering attention, code/equations, backpropagation and FlashAttention, with each preview opening its live view. Visually reviewed all five captures and verified documentation links plus GitHub-rendered desktop/mobile layouts.

## 2026-10-14 · Hosted page setup

Unified hosted and standalone page initialization and reused the Atlas mark as the favicon, preserving local/offline isolation. Build checks, eight navigation assertions in each app variant and all 31 existing cell/source browser checks pass.

## 2026-10-14 · Public README and project preview

Rewrote the README around learning, teaching and research, with a matching Atlas mark, an actual product screenshot, linked features and optional advanced detail. All 57 documentation links/assets and GitHub-rendered desktop/mobile previews pass validation.

## 2026-10-14 · Public development guide

Added portable build, verification, extension and deployment instructions in a dedicated guide, with source links and explicit lesson conventions. Documentation links and embedded-source freshness checks pass.

## 2026-10-14 · v1.5.2 input and focus update

Published continuation dots, live prediction after input edits and optional token focus without zooming. State regressions, desktop/mobile/standalone prediction and gesture checks, existing source navigation and screenshot inspection pass.

## 2026-10-14 · Focus input tokens without zooming

Added Shift-click and right-click “Focus token” on overview input cards, preserving ordinary click-to-zoom and the camera framing. Seventy desktop/mobile browser assertions verify real token-card hits, both navigation paths and the corresponding probability rows, including normal animation.

## 2026-10-14 · Live continuation after input edits

Input edits now select the final retained token before emitting the new trace, while model/settings refreshes preserve exploratory selections. State regressions and 33 browser prediction checks pass, along with all 31 existing cell/source navigation checks.

## 2026-10-14 · Embedding continuation dots

Replaced preview omission sentences with aligned continuation dots, keeping row counts in hover/accessibility labels. Browser checks and screenshot inspection confirm the matrix remains readable and within bounds.

## 2026-10-14 · v1.5.1 citation update

Published five directly relevant sources and repaired citation access in existing detail drawers, preserving main content and visuals. Verified paper passages, source integrity, 12,695 maths checks and 122 desktop/mobile/standalone browser assertions.

## 2026-10-14 · Reachable lesson sources

Connected the six existing lesson routes to focused reading lists, and cited official tensor-view semantics and the online-softmax recurrence beside their equations. Desktop and mobile/standalone checks pass 122 citation, navigation and layout assertions; main lesson text is unchanged.

## 2026-10-14 · Research citation gaps

Linked Google’s Gemma 2 report to the existing soft-capping equation and Anthropic’s circuit-tracing methods to existing interpretation cautions. Verified the cited passages and source integrity without adding lesson content.

## 2026-10-14 · Foundational citations

Added the original Adam and neural word-embedding papers, and connected existing weight-tying references to their equations. Source sections were checked against the papers; bibliography/source integrity checks pass.

## 2026-10-13 · v1.5.0 research-content release

Completed all six content areas as optional equation/source depth and documented their assumptions and scope. Validation passes 549 parity checks, 12,695 maths checks, 34 Python tests, 36 author-code comparisons, 305 study links, 33 source regions and desktop/mobile/standalone browser interactions.

## 2026-10-13 · Rotary phase precision

Computed reference rotary phases before casting sine/cosine to the input dtype, preventing bfloat16 position rounding from merging adjacent long-context tokens. Added that counterexample and exercised sliding-cache checks with an active two-token window; all 34 Python tests pass.

## 2026-10-13 · Finite-row reference policy

Made the manual SDPA/backward references reject all-masked rows created by additive −∞ bias as well as Boolean masks. Focused PyTorch checks also cover NaN score bias, keeping the documented reference policy consistent with its implementation.

## 2026-10-13 · Jacobian conventions

Made residual Jacobian composition order explicit and qualified residual-enabled and basis-change assumptions. A noncommuting two-layer PyTorch counterexample verifies the order; normalization equations and NumPy reference checks pass.

## 2026-10-13 · MLA algebra and research navigation

Added a clearly scoped MLA cache/projection explanation with dense and absorbed Python paths, matching outputs and gradients. Connected related research sections through closed study links and verified all 305 links and 33 executable source regions.

## 2026-10-13 · Long-context RoPE conventions

Added optional PI/YaRN frequency, temperature and cache-validity explanations, explicitly separating the paper and implementation ramps. Fixed-frequency properties and stale-key counterexamples pass; 36 configurations match the inspected authors’ index-ramp implementation.

## 2026-10-13 · QK normalization reference

Added optional L2/RMS QK-normalization and smooth score-capping explanations with clearly labeled separate references and primary citations. PyTorch tests verify norm conventions, score bounds, zero-input behavior and derivatives; mobile drawer/source checks pass.

## 2026-10-13 · Attention implementation contracts

Added optional SDPA examples for dtype tolerances, mask conventions, cached-query alignment, dropout and GQA, verified on CPU. Also verified broadcast-gradient reductions and expanded browser checks for collapsed content, source links and drawer bounds.

## 2026-10-13 · Interpretation scope

Added optional QK/OV and MLP explanations that separate exact vector sums, trained-model observations and causal hypotheses. Checked the decompositions and counterexamples in PyTorch, and clarified the existing neuron lens caption without adding main-view text.

## 2026-10-13 · Invariants and counterexamples

Added optional explanations of softmax/normalization invariants, head-basis freedom, token permutations and cached/full-forward equivalence. Executable PyTorch checks include fixed-causal-mask and rotary counterexamples, plus ten position/mask cache configurations.

## 2026-10-13 · Optional backward explanations

Connected the full attention VJP, RMSNorm and implemented GELU derivatives to executable source through collapsed equation sections. Desktop/mobile checks verify source round trips and unchanged main-lesson text; derivative and equation checks pass.

## 2026-10-13 · Attention backward reference

Added a readable manual attention VJP and KV-sharing gradient reduction, including rectangular attention, additive score bias and masked edges. Independent SDPA/autograd comparisons verify the complete Q/K/V chain and contiguous head-group accumulation.

## 2026-10-13 · Explanation assumptions

Corrected normalization, masked-gradient, score-scaling and pre/post-norm explanations so exact identities and recipe-specific findings are distinguished. Small PyTorch counterexamples verify the epsilon and gradient qualifications; the full equation suite passes.

## 2026-10-13 · v1.4.0 validation and documentation

Documented the executable lesson conventions and prepared v1.4.0 after all routes passed in both themes. Validation passed 549 existing parity checks, 667 new PyTorch lesson comparisons, 12,582 maths checks, Python/NumPy references, cell/source round trips and standalone/desktop/mobile interaction checks.

## 2026-10-13 · Training theme updates

Made gradient heatmap colors repaint immediately when the theme changes, with listener cleanup on exit. The training browser check now exercises live theme switching alongside optimizer/profile controls.

## 2026-10-13 · Practice interface

Added one-task-at-a-time practice with worked reasoning, editable JavaScript, per-test feedback and local progress. Desktop/mobile and standalone browser checks cover every exercise, wrong answers, syntax errors, infinite loops and cancellation without changing the main model.

## 2026-10-13 · Exercise definitions and runner

Added 16 shape, derivation, debugging and coding tasks with worked equations and verified Python operation links. Coding tests run in a disposable worker with cancellation/time limits; all supplied solutions pass and unfinished starters fail.

## 2026-10-13 · Lesson source routing

Fixed an earlier selected model cell overriding a lesson’s explicit Python source or maths request. Browser checks verify that lesson navigation uses its own reference while preserving the original cell selection.

## 2026-10-13 · Architecture comparison

Added a source-linked comparison of 2017, GPT-2 and Llama/GQA recipes, with actual miniature parameter counts hidden by default. Norm placement, positional/head choices and gated FFNs are explained separately; browser checks cover every step and card bounds.

## 2026-10-13 · Softmax scene correctness

Removed a remaining display equation that added a Boolean mask as if it were an additive bias. Captions now follow the configured score/mask conventions and disclose floating-point roundoff.

## 2026-10-13 · Encoder–decoder lesson

Added real source/target attention matrices and residual states through all three decoder sub-layers, with selectable heads, queries and padding examples. Numerical/PyTorch source correspondence and desktop/mobile bounds were verified without modifying the main model.

## 2026-10-13 · FlashAttention lesson

Added live-model tile controls, memory-role illustrations and the running max/sum/output recurrence, with numbers optional and a full-stream equivalence check. Browser checks cover five configurations, tile navigation, one-token inputs and panel bounds; screenshots were inspected in both themes.

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

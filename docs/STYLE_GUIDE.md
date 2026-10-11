# Style guide

Every scene must look like it was designed by the same person on the same day. This page is
the shared visual language. When in doubt, open `src/atlas/scenes/softmax.js` and do what it does.

## 1. The feeling

Calm, exact, generous. Think of a 3Blue1Brown frame or a well-made physics textbook figure:
one idea on screen, lots of air, a few strong colours on a quiet background, and motion that
reveals the idea in the order you would explain it out loud. Nothing decorative. Every mark is
data, structure, or a label that names it.

## 2. The canvas and its zones (1600 × 900 design units)

```
y   20 ┌──────────────────────────────────────────────────────────────┐
       │ HEADER  chips · title (46px) · one context line (19px)        │  sceneHeader()
   180 ├──────────────────────────────────────────────────────────────┤
   196 │ CONTROLS (optional) token picker / head picker                │  tokenPicker(), headPicker()
   250 ├──────────────────────────────────────────┬───────────────────┤
       │ HERO  x 110 … 1140                        │ SIDE  x 1180…1520 │
       │ the one visual that carries the idea      │ equation for this │
       │                                           │ step, a check ✓,  │
       │                                           │ one small context │
       │                                           │ view              │
   760 ├──────────────────────────────────────────┴───────────────────┤
   790 │ CAPTION  1–2 sentences for the current step (19px)            │  sceneCaption()
   860 └──────────────────────────────────────────────────────────────┘
```
- Use `ZONES` from the kit. Left edge of all text: x = 110. Never place content above y = 20 or
  below y = 860 (the breadcrumb and story bar overlay those bands).
- The hero may use the full width (110 … 1490) when there is no side column.
- Align things on a 10-unit grid. Gaps between groups ≥ 40 units; inside a group 8–16.

## 3. Type

| role | how | size |
|---|---|---|
| scene title | `sceneHeader` | 46 px, Bricolage Grotesque 700 |
| context line | `sceneHeader` sub | 19 px, body, `--fg-2`, max one sentence (two at most) |
| caption | `sceneCaption` | 19 px, body, `--fg-2`; bold (`<b>`) only for the token or number that matters |
| labels on visuals | `.k-label` / kit `label`/`tex` options | 15–17 px mono, coloured by component |
| numbers in cells / on bars | mono 12–14 px | 2 decimals by default (`fmt(v, 2)`), percentages for probabilities |
| equations | `texLabel` / `texHTML` | 22–28 px for the step equation; at most two on screen |

Write like a careful teacher: short declarative sentences, the field's names (W_Q, d_h, KV cache),
no filler, no exclamation marks, no emoji, no "simply", no "just".

## 4. Colour

- Background, surfaces and text come from tokens (`--bg`, `--surface`, `--fg`, `--fg-2`, `--fg-3`,
  `--line`). Never hard-code hex.
- **Component colours carry meaning and never change:** Q `--c-q` orange, K `--c-k` blue, V
  `--c-v` green, attention `--c-attn` violet, O projection `--c-o` magenta, MLP `--c-mlp`
  amber, norm `--c-norm` slate, residual stream `--c-resid` indigo, embedding `--c-embed` teal,
  positions `--c-pos` sky, LM head `--c-head` rose, mask `--c-mask` grey (hatched).
- **Values** use the two ramps only: diverging (blue negative, neutral zero, orange positive) for
  signed activations and weights; sequential (neutral → component colour) for probabilities,
  attention weights and other non-negative quantities. Masked = hatched grey. Same value → same
  colour everywhere in the app.
- Accent (`--accent`) marks the current selection or the thing the step is about. Use it for
  one thing at a time.
- Semantic: `--good` for ✓ checks, `--bad` for ✗. Nothing else is green or red.

## 5. Shapes and depth (the 2.5D look)

- Tensors are cells: `vec` for vectors, `mat` for matrices, `heatmap`/`matmulAnim` when large.
  Cells are rounded 3 px with a hairline gap; containers are rounded 8–12 px with a soft shadow
  and a faint component-coloured border. Use `tilt: true` only for weight matrices that the
  data passes "through" (W_E, W_U, W_O), never for activations you ask the reader to read.
- Operations are round nodes (`op('\\oplus')`, `op('\\times')`, `op('\\softmax')`), 48–64 units.
- Flow is arrows (`stage.arrow`, `flow: true` for live dots) in the colour of what flows.
- Zoom targets are `card({node})` or a labelled shape with `data-node`; they lift on hover.
  Make the target the thing itself (the head's pattern, the MLP's neuron grid), not a link.
- Long vectors: show first 12 … last 4. Long sequences: shrink cells (min 4 px), keep labels on
  every k-th token, never overflow the hero zone.

## 6. Motion

| use | duration | easing |
|---|---|---|
| things moving or resizing (bars, cells, positions) | 600 ms (`MOTION.move`) | `MOTION.ease` = cubic-bezier(.65,0,.35,1) |
| things appearing | 520 ms, stagger 60 ms (`appear`) | `MOTION.easeOut` = cubic-bezier(.2,.7,.2,1) |
| flying a value from source to destination | 600–800 ms (`fly`) | ease in-out with a slight arc |
| auto-advancing steps on first view | 1700 ms per step (`MOTION.step`), stop on first pointer interaction | |

- Each step changes **one** thing. Previous steps' results stay visible (dimmed if needed) so the
  reader sees cause and effect.
- Animate the mathematics: a sum should visibly add up, a normalisation should visibly rescale,
  a lookup should visibly fetch. Never animate for decoration (no idle wobble, no bouncing).
- Under reduced motion, jump to end states; the static picture must still explain the step.

## 7. Composition templates (pick one per scene)

1. **Pipeline** (head, block, MLP, LM head): left-to-right stages, each a matrix or vector with a
   label and shape, joined by operation nodes and arrows; steps light the stages in order.
2. **Focus + context** (softmax, scores, mix, norm): the hero is one token's computation in big,
   readable form; the side column shows the full matrix with that token's row highlighted.
3. **Before / after** (residual, norm, lab): two states side by side, the change animated between
   them, a ✓ that the result equals the traced value.
4. **Build-up** (tokens, mask, positional): the structure assembles itself cell by cell or token
   by token, then the meaning is named.

## 8. Truth on screen

- Every number comes from the trace, the weights, or an engine call on them. When a scene
  recomputes something, show `checkBadge(computed, traced)`.
- Name the token and position you are showing ("query “s” at position 21").
- Illustrations that are not the model's own computation say "illustration" in their label.

## 9. Do / don't

| do | don't |
|---|---|
| one hero visual, one equation for this step | three charts competing for attention |
| the real matrix, shrunk to fit | a made-up matrix that "looks like" attention |
| a caption that says what changed in this step | a paragraph that explains everything at once |
| colour = component or value | colour for decoration or to "liven up" a card |
| controls next to what they change, small | a toolbar of options at the top of the scene |
| empty space | filling empty space |

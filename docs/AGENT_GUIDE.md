# Transformer Atlas: contributor guide

Read this whole file before writing code. It is the contract every view follows.

## What the app is

A self-contained, no-build web app (plain ES modules, no npm dependencies at runtime) that
explains a decoder-only Transformer by showing **every tensor of a real forward pass** and
tying each one to **the exact line of PyTorch** that produces it.

- `reference/model.py`: the ground-truth PyTorch model. Lines that produce a visible tensor end
  with `# ... @trace <key>` (`{i}` = layer index).
- `src/engine/`: a JS port of model.py with autograd. `forward()` records the same keys in the
  same order. `tests/parity.mjs` proves JS == PyTorch on every key and every gradient
  (549/549 checks, tolerance 1e-9) for 6 architecture variants. **Never edit the engine.**
- `src/core/`: store (state + events), router, shell (nav, inspector drawer, settings drawer),
  `describe.js` (catalogue of trace keys), `codemap.js` (model.py lines ↔ trace keys).
- `src/ui/`: the shared component kit (heatmap, charts, controls, code viewer, playbar...).
- `src/views/<id>.js`: one file per page. **This is the only kind of file you edit**, unless
  your task says otherwise.
- `src/views/_example.js`: a small, complete, working view. Copy its patterns.

Audience: an ML researcher (LLMs, VLMs, computer-use agents) preparing for research-engineer
interviews. Assume strong math; do not dumb anything down. Be exact: shapes, formulas, and
claims must match `reference/model.py` and the live numbers.

## Ground rules (shared login node)

- This runs on a **shared cluster login node**. Do not train models, run long loops, start
  servers you leave running, or install packages. Light commands only.
- Browser checks: **always** use `tools/smoke.sh` (it serialises Chromium with a lock).
- Do not edit files outside the ones your task names. If something in `src/core`, `src/ui` or
  `src/engine` is broken or missing, work around it inside your own file and report it under
  `core_issues` in your final answer.
- No external libraries, no CDN scripts, no fetch() of remote URLs. Everything is in-repo.
- Plain modern JavaScript (ES2022 modules). No TypeScript, no JSX, no build step.

## How to run and check your work

```bash
cd /proj/docling-vision/users/said/sandbox/transformer-explorer
node --check src/views/<id>.js                      # syntax
tools/smoke.sh --view <id> --out /path/shot.png --full                # light, 1440px
tools/smoke.sh --view <id> --theme dark --out /path/dark.png --full   # dark
tools/smoke.sh --view <id> --width 400 --out /path/phone.png --full  # phone width
tools/smoke.sh --view <id> --eval "document.querySelectorAll('canvas').length"
tools/smoke.sh --view <id> --eval "(document.querySelector('button.btn').click(), window.__te.store.state.sel)"
```

The smoke tool prints JSON: `errors` (console errors and page errors; must be empty),
`overflowX` / `viewOverflow` (must be false: the page must never scroll sideways),
`stub` (must be false once your view is in), `failed` (the router's error box).
**Open the PNG with your Read tool and look at it.** Judge it as a demanding designer would:
alignment, spacing, legibility in both themes, nothing clipped or overlapping.

`--eval` runs an expression in the page after load (`window.__te.store` is the store) and
returns its JSON value: use it to verify numbers and to click controls.

Write screenshots to your scratch directory, not into the repo.

## The view contract

```js
import { h, viewHeader, section, card, heatmap /* ... */ } from '../ui/index.js';

export default {
  id: 'attention',                 // must match the registry id
  title: 'Self-attention (one head)',
  mount(root, ctx) {               // root: empty <div class="view">; build into it
    // ...build DOM once...
    const render = () => { /* read ctx.state, update DOM */ };
    ctx.on('trace', render);       // forward pass changed (text, config, weights)
    ctx.on('select', render);      // layer / head / token selection changed
    render();
    return () => { /* cleanup: stop timers, animation loops, destroy heatmaps */ };
  },
};
```

`ctx` gives you:

| field | what |
|---|---|
| `ctx.state` | live store state (see below) |
| `ctx.store` | the store: `select`, `setConfig`, `setText`, `loadPreset`, `reinitialize`, `gradients`, `notice` |
| `ctx.on(event, fn)` | subscribe; auto-unsubscribed when the view unmounts |
| `ctx.inspect(key)` | open the tensor inspector for a concrete trace key, e.g. `'L1.attn.probs'` |
| `ctx.navigate(viewId)` | go to another view |
| `ctx.engine` | the engine module (same as `import * from '../engine/index.js'`) |
| `ctx.ui` | the UI kit module |
| `ctx.describe` | `infoFor(key)`, `shapePairs(key, cfg, T)`, `parseKey(key)`, `TRACE_INFO` |
| `ctx.codemap` | `codeLines`, `codeRegions`, `linesForKey(key)`, `lineForKey(key, cfg)`, `codeContext(line)` |

Rebuild only what changed. Heatmaps have `update(opts)`; keep the instance and update it
rather than recreating it on every event. Never block the main thread for more than ~30 ms.

## State (`ctx.state`)

| field | type | notes |
|---|---|---|
| `config` | object | see Config below |
| `tokenizer` | `{vocab: string[], size, encode(text)->{ids, dropped}, decode(ids), token(id), label(id)}` | char-level; `label` shows space as `␣` |
| `text`, `ids`, `dropped` | string, int[], string[] | ids are clipped to `block_size` |
| `model` | `{config, params: {name: Tensor}}` | **read-only for views**. Use `engine.cloneModel(model)` for experiments |
| `weights` | `'random'` or `'preset:<id>'` | |
| `preset` | object or null | `{id, title, description, samples[], metrics}` when a preset is loaded |
| `trace` | `{key: Tensor}` | every traced tensor of the last forward pass, **in execution order** |
| `logits` | Tensor `[T, V]` | |
| `sel` | `{layer, head, token}` | shared selection; set with `ctx.store.select({...})` |
| `focusKey` | string or undefined | set by the inspector's "Find in code stepper"; the stepper consumes it |

`store.gradients()` runs a next-token loss on the current text and backprops:
returns `{loss, trace, act: {key: Float64Array}, params: {name: Float64Array}, targets}`.
`act` holds gradients of the 2-D traced activations (residual stream, q/k/v, MLP...).

Presets (trained by the PyTorch reference, loaded at startup when present):
`charlm` (GPT-2 style char LM on Alice in Wonderland, 4L 4H d64), `llama` (same data, RoPE +
RMSNorm + SwiGLU + GQA 4Q/2KV, no bias), `induction` (2L, induction heads on repeated text),
`reverse` (2L, "word.drow", anti-diagonal attention). `import { PRESETS } from '../presets/index.js'`
lists those that exist; `ctx.store.loadPreset(id)` loads one. With random weights, attention is
nearly uniform; `ctx.store.reinitialize({initScale: 12})` makes untrained patterns visible.

## Config

```js
{ vocab_size, block_size, n_layer, n_head, n_kv_head, d_model, d_ff,
  pos_encoding: 'learned'|'sinusoidal'|'rope'|'alibi'|'none',
  norm: 'layernorm'|'rmsnorm', norm_placement: 'pre'|'post',
  activation: 'gelu'|'relu'|'swiglu', mask: 'causal'|'none'|'sliding'|'prefix',
  window, prefix_len, tie_weights, bias, residual, attn_scale, rope_base, norm_eps }
```
`dh = d_model / n_head`. Changing a shape-affecting field re-initialises random weights
(the store shows a notice). Mask, window, prefix_len, residual, attn_scale and activation
gelu↔relu keep weights, so they can be toggled on a trained preset.

## Trace keys (batch axis removed; T = tokens, V = vocab, F = d_ff)

| key | shape | when |
|---|---|---|
| `emb.tok` | [T, d] | always |
| `emb.pos` | [T, d] | learned or sinusoidal |
| `emb.out` | [T, d] | always (x₀) |
| `mask` | [T, T] | 1 = may attend |
| `alibi` | [H, T, T] | alibi |
| `L{i}.ln1` | [T, d] | pre-LN only |
| `L{i}.attn.q` / `.k` / `.v` | [T, H·dh] / [T, Hkv·dh] / [T, Hkv·dh] | |
| `L{i}.attn.q_heads` / `k_heads` / `v_heads` | [H,T,dh] / [Hkv,T,dh] / [Hkv,T,dh] | |
| `L{i}.attn.q_rot` / `k_rot` | [H,T,dh] / [Hkv,T,dh] | rope |
| `L{i}.attn.k_rep` / `v_rep` | [H, T, dh] | n_kv_head < n_head |
| `L{i}.attn.scores` | [H, T, T] | q·k (after RoPE when used) |
| `L{i}.attn.scaled` | [H, T, T] | attn_scale |
| `L{i}.attn.biased` | [H, T, T] | alibi |
| `L{i}.attn.masked` | [H, T, T] | −Infinity where masked |
| `L{i}.attn.probs` | [H, T, T] | rows sum to 1 |
| `L{i}.attn.ctx` | [H, T, dh] | probs @ v |
| `L{i}.attn.concat` | [T, H·dh] | |
| `L{i}.attn.out` | [T, d] | after o_proj |
| `L{i}.add1` | [T, d] | post-LN only (sum before ln1) |
| `L{i}.resid_mid` | [T, d] | stream after attention sub-layer |
| `L{i}.ln2` | [T, d] | pre-LN only |
| `L{i}.mlp.up` / `gate` / `act` / `out` | [T,F] / [T,F] (swiglu) / [T,F] / [T,d] | |
| `L{i}.add2` | [T, d] | post-LN only |
| `L{i}.resid_out` | [T, d] | block output |
| `lnf` | [T, d] | pre-LN only |
| `logits`, `probs` | [T, V] | row t predicts token t+1 |

Tensors are `{data: Float64Array (row-major), shape: number[]}`. For a 3-D tensor use
`sliceOf(tensor, h)` from the UI kit to get `{data, rows, cols}` of head `h`.
KV head for query head `h` is `Math.floor(h / (n_head / n_kv_head))`.
Weights use nn.Linear layout `[out, in]`: `y = x Wᵀ + b`. Names: `wte.weight`, `wpe.weight`,
`blocks.{i}.ln1.weight|bias`, `blocks.{i}.attn.{q,k,v,o}_proj.weight|bias`,
`blocks.{i}.ln2.*`, `blocks.{i}.mlp.{up,gate,down}_proj.weight|bias`, `ln_f.*`,
`lm_head.weight` (absent when tied: use `wte.weight`).

## Engine API (`ctx.engine` or `import ... from '../engine/index.js'`)

- Ops on 2-D Tensors (with autograd; wrap in `noGrad(() => ...)` for inference):
  `linear(x, W, b)`, `matmul(A, B)`, `matmulNT(A, B)` (A·Bᵀ), `add`, `mul`, `scale(A, s)`,
  `addConst(A, Float64Array)`, `embedding(W, ids)`, `layerNorm(x, w, b, eps)`, `rmsNorm(x, w, eps)`,
  `gelu`, `relu`, `silu`, `geluScalar(v)`, `siluScalar(v)`, `softmax(x, mask?)`,
  `sliceCols(x, start, len)`, `concatCols(list)`, `rope(x, cos, sin)`, `crossEntropy(logits, targets)`,
  `stack(list)`. `new Tensor(Float64Array, shape)`, `Tensor.from(array, shape)`.
- Model: `DEFAULT_CONFIG`, `CONFIG_CHOICES`, `dHead(cfg)`, `validateConfig(cfg)`, `paramSpecs(cfg)`,
  `createModel(cfg, seed)`, `modelFromWeights(cfg, weights)`, `cloneModel(model)`, `countParams(model)`,
  `sinusoidalTable(T, d)` (Float64Array T·d), `ropeAngles(T, dh, base, start=0)` → `{cos, sin}` [T, dh/2],
  `alibiSlopes(H)`, `alibiBias(cfg, T)`, `maskAllows(cfg, i, j)`, `buildMask(cfg, T)` (Uint8Array),
  `forward(model, ids, {targets, trace})`, `infer(model, ids, opts)` → `{logits, probs, loss, trace}`.
  To run a variant: `const m = cloneModel(state.model); m.config = {...m.config, residual: false}; infer(m, ids)`.
  (A clone keeps unused params, e.g. `wpe`, so switching to `pos_encoding: 'none'` works.)
- Decoding: `samplingDistribution(logits, {temperature, topK, topP})` → number[],
  `sampleFrom(probs, uniform)`, `argmax(arr)`, `kvCacheSupported(cfg)`, `createCache(model)`,
  `decodeStep(model, cache, tokenId)` → `{logits: Float64Array(V), attn: [layer][head][pos]}`,
  `cacheBytes(cfg, len, bytesPerValue=2)`.
- Randomness: `mulberry32(seed)` (uniform), `gaussian(uniform)`. Never use unseeded randomness
  for anything the user should be able to reproduce.

## UI kit (`../ui/index.js`)

- DOM: `h(tag, props, ...children)` (props: `class`, `style` object incl. CSS vars, `onClick`...,
  `html`, `text`, `dataset`), `s(tag, props, ...children)` for SVG, `clear(el)`,
  `fmt(v, digits)`, `fmtCount(n)`, `fmtBytes(n)`, `clamp`, `rafThrottle(fn)`.
- Colour: component kinds `embed pos q k v attn o mlp norm resid head mask`.
  `kindVar(kind)` → `'var(--c-q)'` for styles/SVG fills; `kindColor(kind)` → resolved colour
  for canvas; `diverging(t∈[-1,1])`, `sequential(t∈[0,1], kind?)`, `withAlpha(color, a)`,
  `onThemeChange(fn)` (re-draw canvases on theme change). **Never hard-code hex colours**;
  use `var(--fg)`, `var(--fg-2)`, `var(--fg-3)`, `var(--surface)`, `var(--surface-2)`,
  `var(--line)`, `var(--accent)`, `var(--good)`, `var(--warn)`, `var(--bad)` and the kind colours.
- `heatmap({data, rows, cols, mode: 'diverging'|'sequential', kind, range, maxWidth, cell, cellH,
  rowLabels, colLabels, showValues, digits, highlight: {rows, cols, cells}, dimOutside, title,
  tooltip(r,c,v), onHover, onLeave, onClick})` → `{el, update(patch), destroy()}`.
  −Infinity cells are drawn hatched (masked). `sliceOf(tensor3d, index)`.
- Charts: `lineChart({series: [{label, kind|color, points: [[x,y]], dashed, area, dots}], width,
  height, xLabel, yLabel, xDomain, yDomain, markers})`, `barChart({values, labels, kind, highlight,
  format, onClick, width, height, max})`. Both return `{el, update(patch)}`.
- Controls (each `{el, set, get}`): `slider({id, label, min, max, step, value, format, hint, onInput})`,
  `select({id, label, options, value, onChange})`, `toggle({id, label, checked, onChange})`,
  `segmented({options, value, onChange, ariaLabel})`, `numberInput(...)`,
  `button(label, onClick, {variant: 'primary'|'ghost'|'icon', title})` → element.
  Give every control a stable `id` prefixed with your view id (e.g. `attention-temp`).
- Layout: `viewHeader({eyebrow, title, lede})`, `section({title, lede}, ...children)`,
  `card({title, kind, actions}, ...children)`, `grid(cols, ...children)` (collapses on narrow
  screens), `callout('note'|'warn'|'', ...children)`, `kv([[k, v]])`, `shapeBadge([[sym, n]])`,
  `kindTag(kind, text)`, `legend([[kind, label]])`, `tabs(items, value, onChange)`, `empty(text)`.
- `tokenStrip({labels, selected, onSelect, onHover, color})` → `{el, set}`.
- `codeView({lines, from, to, highlight, dimOthers, kindOf, onTraceClick, onLineClick, maxHeight})`
  → `{el, setHighlight(lines)}`. Pass `lines: ctx.codemap.codeLines` and a region from
  `ctx.codemap.codeRegions` (`Config, LayerNorm, RMSNorm, make_norm, sinusoidal_table,
  rope_angles, apply_rope, alibi_bias, build_mask, Attention, MLP, Block, GPT`).
  `highlightPython(src)` for static snippets.
- Maths: `formula(html)`, `frac(num, den)`, `sqrt(x)`, `sub`, `sup`, `col(kind, text)`.
- Motion: `playbar({steps, onStep, interval})` (staged explanations), `tween({duration, onUpdate,
  onDone})`, `loop(fn)` → stop, `ease`, `reducedMotion()`. Respect reduced motion: animations
  must have a static end state that conveys the same information.
- Tooltips: `showTip(x, y, html)`, `hideTip()`.
- CSS classes you can use: `row`, `stack`, `scroll-x`, `muted`, `small`, `num`, `mono`,
  `eyebrow`, `formula`, `math`, `kv`, `shape`, `kind-tag`, `token`, `btn`, `btn primary`,
  `callout note|warn`, `empty`. For view-specific CSS, inject one `<style>` element from your
  module with selectors prefixed by a class unique to your view (e.g. `.v-attention ...`), and
  only use the CSS variables above for colour.

## Design and writing rules

- The bar is a polished product, not a demo. Clear hierarchy: view header, then 3 to 6
  sections, each answering one question. Real numbers from the live trace everywhere.
- Every major concept appears three ways: **the picture** (diagram/heatmap), **the maths**
  (formula), **the code** (the model.py lines via `codeView`, highlighted). Link tensors to the
  inspector (`ctx.inspect`) and to other views (`ctx.navigate`).
- Show shapes constantly (`shapeBadge` with live numbers from `describe.shapePairs`).
- Interactions must be obvious (buttons look like buttons) and give immediate feedback.
- Copy: plain, direct, exact. Short sentences. Name things the way the field does
  (W_Q, d_head, KV cache, pre-LN). No filler, no emoji, no exclamation marks.
- Layout must work at 400 px width: use `grid()` (collapses), put wide heatmaps in `scroll-x`.
- Both light and dark themes must look right (canvas colours come from tokens; re-render on
  `onThemeChange`).
- Interview angle: where it fits, add a short "Interview notes" card with crisp facts
  (complexities, parameter counts, common pitfalls). Keep them correct.

## Correctness rules

- Never invent numbers. Values come from `ctx.state.trace`, the model weights, or engine
  functions applied to them. Illustrations that are not the model's real computation (e.g. a
  toy BPE demo, a hypothetical cross-attention) must be labelled as illustrations.
- Formulas, shapes, and names must agree with `reference/model.py` (nn.Linear stores `[out, in]`;
  the app drops the batch axis; GELU is the tanh approximation; RoPE uses interleaved pairs
  `(2k, 2k+1)` with `θ_k = base^(−2k/dh)`; ALiBi slopes `2^(−8(h+1)/H)`; LayerNorm uses biased
  variance; the causal mask is `j ≤ i`).
- Handle every config: any `n_layer`, `n_head`, GQA/MQA, each `pos_encoding`, pre/post norm,
  swiglu, each mask, residual off, tied weights, T from 1 to `block_size`. Keys that do not exist
  for the current config must not crash the view; show a short note instead.

## Definition of done

1. `node --check` passes; smoke shows `errors: []`, `stub: false`, `failed: false`,
   no horizontal overflow at 1440 px and at 400 px, in light and dark.
2. You looked at the screenshots and fixed anything ugly, clipped, overlapping or illegible.
3. Every item of your task spec is implemented, or listed as skipped with a reason.
4. Switching layer/head/token, changing the input text, and changing the config (try
   `--eval "(window.__te.store.setConfig({pos_encoding:'rope', n_kv_head: 2}), 1)"`) re-render
   without errors.
5. Cleanup function stops every timer and animation loop.

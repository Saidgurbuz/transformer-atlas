# Atlas scene guide

The app is one full-screen, zoomable **Atlas** of the Transformer. The root is a 3D model
(`src/atlas/scenes/model.js`). Clicking a part zooms into its **scene**; scenes contain smaller
parts that zoom further (Model → Block 2 → Attention → Head 1 → Softmax). The hierarchy is in
`src/atlas/nodes.js`. Each scene is one file in `src/atlas/scenes/<name>.js`.

**Read `docs/STYLE_GUIDE.md` first** (zones, type, colour, motion, composition templates,
do/don't): every scene must follow it. Use the kit's shared parts: `sceneHeader`, `sceneCaption`,
`tokenPicker`, `headPicker`, `checkBadge`, `ZONES`, `MOTION`.

Read next: this file, `src/atlas/scenes/softmax.js` (the reference scene: copy its structure,
quality and restraint), `src/atlas/kit.js`, `src/atlas/nodes.js`, `docs/AGENT_GUIDE.md` (store,
trace keys, engine API, colours), `docs/MATH_GUIDE.md` (equations).

## The three rules (from the product owner)

1. **One clear focus per screen.** A scene answers exactly one question ("what does softmax do
   to one row of scores?"). At most three visual groups. Generous space. A newcomer must see
   what matters within two seconds. Anything secondary lives one level deeper (a child node),
   on hover, in the right-click menu, or in the drawers (maths, code, model settings).
2. **Minimal UI.** No sidebars, no dense controls, no paragraphs. A short title, one line of
   context, the visual, and a one- or two-sentence caption that changes with the step. Controls
   only when essential (a token picker, a head picker, a slider), small and close to what they
   change.
3. **Motion explains.** Every scene animates its idea in steps (3Blue1Brown style): values flow,
   bars morph, vectors fly into place, matrices fill. Transitions between steps are smooth
   (CSS transitions, the kit's helpers, Web Animations). Respect reduced motion (end states).

## Scene contract

```js
import { stage2d, vec, mat, op, card, token, texLabel, caption, chip, appear, fly, pulse, countTo, wait, useStyle } from '../kit.js';
import { h, fmt, texHTML, num } from '../../ui/index.js';

useStyle('myscene', `.ms-thing { ... }`);        // scene CSS, selectors prefixed with your scene's prefix

export default {
  mount(el, ctx) {
    const st = stage2d(el);                       // 1600 x 900 design canvas, scaled to fit
    // build once, positioned in design units: st.add(node, { x, y, anchor })
    // steps: ctx.setSteps(['Scores', 'Shift', ...], (i) => paint(i)) - story mode walks them
    // re-render on ctx.on('trace', ...) and ctx.on('select', ...)
    return { destroy() { /* clear timers, destroy components */ st.destroy(); } };
  },
};
```

`ctx`:

| member | what |
|---|---|
| `ctx.state` / `ctx.cfg` | live store state / config (see AGENT_GUIDE) |
| `ctx.params` | node parameters as numbers: `{l, h, n}` for e.g. `head.1.2` |
| `ctx.node` | `{id, title, blurb, kind, keys, parent}` |
| `ctx.store` | `select({token, head, layer})`, `setText`, `setConfig`, `loadPreset`, `notice` |
| `ctx.on(event, fn)` | `'trace'` (new forward pass), `'select'`; auto-removed on unmount |
| `ctx.setSteps(labels, onStep)`, `ctx.setStep(i)` | in-scene steps; the bottom bar shows them and story mode walks them before moving on |
| `el` event `'atlas:interrupt'` | the reader used the story bar or arrow keys: stop any auto-advance (`el.addEventListener('atlas:interrupt', stop)`) |
| `data-own-keys` attribute | put it on a focusable component that handles arrow keys itself, so the atlas does not also navigate |
| `ctx.go(nodeId, originElOrRect)` | zoom to another node (the kit does this for `data-node` elements automatically) |
| `ctx.showMath(key, cell)`, `ctx.showCode(keys)`, `ctx.inspect(key)` | open the drawers |
| `ctx.label(i)` | display label of token i |

## Design canvas and layout

- Design for **1600 × 900** units. The atlas reserves the top 64 px (breadcrumb) and bottom 84 px
  (story bar) of the screen; the canvas is scaled into the rest. Keep important content inside
  x ∈ [80, 1520], y ∈ [20, 860].
- Typography: title 44-48 px (`.k-title` with `fontSize` override, left-aligned like softmax), one
  context line 18-21 px (`.k-caption`), labels 15-17 px. KaTeX via `texLabel(latex)` or
  `texHTML`.
- Colours: component kinds (`q k v attn o mlp norm resid head embed pos mask`) through
  `var(--c-<kind>)`; tensor values through the kit (`vec`, `mat`) or `diverging`/`sequential`.
  Never hard-code hex colours. Both themes must look right.
- Reference scenes in the "More" menu (cheat sheet, scaling, lab, encdec, vision) may use a scrollable document instead of the canvas: `el.append(h("div", {class: "k-doc"}, h("div", {class: "k-doc-inner"}, ...)))`, still with the same restraint, cards and motion.
- Depth ("2.5D"): soft shadows and raised cells come with the kit; `mat(..., {tilt: true})`
  for a perspective plane; layering (back/mid/front) for parallax-like depth. No 3D engine in
  2.5D scenes.

## Making things interactive (automatic)

- `data-trace="<trace key>"` on a container and `data-cell="r,c,band"` on its cells: hovering
  shows the tensor's equation with that cell's live numbers; right-click offers *Show the maths*,
  *Show the code* (the model.py line), *Inspect every value*. `vec(..., {trace, cellOf})` and
  `mat(..., {trace, cellOf})` set these for you. Use the concrete key, e.g. `L1.attn.probs`.
- `data-node="<node id>"` on any element (e.g. `card({node: 'softmax.1.2'})`): it becomes a zoom
  target with a hover hint, and clicking zooms into that node from the element's rectangle.
  Every child node of your scene (see `childrenOf(id, cfg)` in nodes.js) must have a clear,
  visible zoom target inside the visual (not a list of links).

## Kit (src/atlas/kit.js)

`stage2d(el)` → `{add(node, {x, y, anchor, layer}), move, arrow(from, to, {kind, curve, flow, dash, width}) → {draw(ms), destroy}, center(node), edge(node, 'l'|'r'|'t'|'b'), onCleanup, destroy, svg, layers, scale}` ·
`vec(values, {orientation, cell, kind, mode, range, label, tex, trace, cellOf, head, tail})` (`.update(values)`, `.highlight(idx)`) ·
`mat(data, rows, cols, {cell, gap, kind, mode, range, rowLabels, colLabels, tex, trace, cellOf, tilt, showValues})` (`.update`, `.highlightRow(r)`, `.highlightCol`, `.highlightCells`, `.cellEl(r, c)`) — up to ~2500 cells; for bigger use `heatmap` from the UI kit with `traceKey` ·
`op(latex, {kind, size})` · `card({title, sub, kind, node, w})` · `token(label, i, {selected, kind})` · `texLabel(latex, {size, kind})` · `caption(html)` · `chip(text, kind)` ·
motion: `appear(els, {stagger, from})`, `pulse(el)`, `fly(stage, fromEl, toEl, {duration})`, `countTo(el, value)`, `wait(ms)` · `useStyle(id, css)` ·
shared parts: `sceneHeader(stage, {chips: [[text, kind]], title, sub}) -> {el, setSub(html), setChips(list)}`, `sceneCaption(stage, {x, y, w}) -> {set(html)}`, `tokenPicker(stage, ctx, {x, y, label, kind}) -> {el}`, `headPicker(stage, ctx, {x, y}) -> {el}`, `checkBadge(computed, traced, {label}) -> element`, `ZONES`, `MOTION`.

## Animated components (src/ui/*.js, import from '../../ui/index.js' or the file)

- `matmulAnim({A: {data, rows, cols, name, kind, shape, rowLabels}, B: {..., bias, biasName}, transposeB, C: {name, kind, data}, maxWidth, showValues, speed, hint})` (hint: false hides the idle corner text) → `{el, focus(i, j), play(), pause(), reset(), showAll(), update(patch), destroy()}`: the textbook A · Bᵀ animation, canvas based, smooth up to 64 × 64.
- `arcDiagram({labels, weights: {data, rows, cols}, mask, query, layout: 'arc' | 'bipartite', keyLabels, threshold, dots, onSelect})` → `{el, setQuery(i), update(patch), play(), stop(), destroy()}`: attention as arcs between tokens.
- `vectorStrip({values, orientation, cell, mode, kind, labels, title, range})` → `{el, update(values, {animate}), pulse(), highlight(idx), destroy()}`;
  `flyClone(fromEl, toEl, {duration})`, `ticker(el, from, to)`, `drawPath(path)`, `stagger`, `sequence(steps)` + `sequencePlaybar` (prefer `ctx.setSteps` for scene steps).
- `heatmap({data, rows, cols, mode, kind, traceKey, rowLabels, colLabels, onClick})`, `barChart`, `lineChart` (see AGENT_GUIDE).

Import from `../../ui/matmul.js`, `../../ui/arcs.js`, `../../ui/flow.js` directly (they may not be
re-exported by `ui/index.js`).

## Maths

`import { mathFor, runLive } from '../../core/math/index.js'` gives verified equations per trace
key (`mathFor('L1.attn.probs').tex`) and live instantiations (`runLive(state, key, {r, x, band})`
→ `{tex, result, expected}`). Show at most one or two equations on screen, chosen for the
current step; the full set is one click away in the maths drawer. Coloured macros: `\Q \K \V
\Wq \Wk \Wv \Wo \WE \WU \Wup \Wdown \Wgate \Aw \Mk \Pe \xres`, operators `\softmax \LN
\RMSNorm \GELU \SiLU \ReLU \concat \CE \Attn \MLP`.

## Tokenizers and vocabulary size

Two kinds of presets exist. Character presets (`charlm`, `llama`, `induction`, `reverse`): V = 32,
one character per token. The subword preset (`tinystories`, loaded by default): byte-level BPE
with V = 4,096, so most words are one token and labels are words like `␣Lily` (␣ marks the
leading space). `ctx.state.tokenizer.kind` is `'char'` or `'bpe'`; for BPE it also has
`words(text)` (pre-tokenized words) and `explain(word)` (the real merge sequence, for animating
BPE). Scenes must work for both:
- never draw all V rows or columns when V is large: show the top-k entries (by probability,
  logit, similarity, ...) or the rows the current text uses, and say how many are not shown;
- token labels can be several characters: size tiles to their text (the kit's `tokenPicker`
  already does), never assume one character per token;
- matrices with V rows (W_E, W_U, logits) show the rows for this text's tokens, or the top-k.

## Correctness

Every number on screen comes from `ctx.state.trace`, `ctx.state.model.params` or engine
functions applied to them. Where the scene recomputes something (a dot product, a norm), show a
small check that it equals the traced value (e.g. "✓ equals the traced value"). Handle every
config: any layer/head index, GQA/MQA, RoPE/ALiBi/none, post-LN (no `ln1`/`ln2`/`lnf` keys),
SwiGLU, every mask, residual off, T from 1 to 64 (elide long vectors: first 12 … last 4;
shrink cells for long sequences). Illustrations that are not the model's computation must say so.

## Checking

```bash
node --check src/atlas/scenes/<name>.js
tools/smoke.sh --view <node id> --out shot.png              # e.g. --view head.1.2 ; light, 1440x900
tools/smoke.sh --view <node id> --theme dark --out dark.png
tools/smoke.sh --view <node id> --width 1920 --height 1080 --out big.png
tools/smoke.sh --view <node id> --wait 6000 --out late.png  # after the step animation ran
tools/smoke.sh --view <node id> --eval "(async () => { window.__te.store.select({token: 3}); await new Promise(r => setTimeout(r, 800)); return 1 })()" --out sel.png
tools/smoke.sh --view <node id> --eval "(async () => { await window.__te.store.loadPreset('induction'); await new Promise(r => setTimeout(r, 1500)); return 1 })()" --out induction.png
```
Presets: `charlm` (loaded at start, 4 layers, 4 heads), `llama` (RoPE, RMSNorm, SwiGLU, GQA 4/2),
`induction` (2 layers; layer 1 heads 0/2/3 are induction heads), `reverse`. Try each one, and
`window.__te.store.setConfig({norm_placement: 'post'})`, `({mask: 'none'})`, `({pos_encoding: 'rope'})`.
`errors` must be `[]`, `stub` false, `failed` false. Look at every screenshot: is there ONE clear
focus? Would a newcomer understand it in two seconds? Is it beautiful in both themes?

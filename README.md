# Transformer Atlas

**Live:** https://saidgurbuz.github.io/transformer-atlas/ · version 1.3.0 ([changelog](CHANGELOG.md))

A full-screen, zoomable atlas of a decoder-only Transformer running a real (tiny, trained)
model in your browser. Start from a 3D view of the whole model, click a part to zoom into it
(Model → Block 2 → Attention → Head 1 → Softmax), and each level shows one idea, animated on the
real numbers. Hover anything for its equation with the live values plugged in; right-click for
the exact line of PyTorch that computes it. Arrow keys walk the forward pass as a guided story.

- **One focus per screen:** every node of the hierarchy (`src/atlas/nodes.js`) is a scene
  (`src/atlas/scenes/*.js`) that answers one question, in steps.
- **Numerically checked:** fixture traces and gradients match the PyTorch reference within
  the tested tolerances. Catalogue equations compile, and their live cell calculations are
  checked against traced values (`tests/math.mjs`); browser checks cover navigation and layout.
- **Cell ↔ code:** select an exact head, row and column in the inspector or code stepper,
  follow it into the Python source, and return to the same value. Inactive code branches are
  disabled, and norm/RoPE/mask helpers can be expanded. Equation cards link to specific papers
  and disclose implementation conventions.
- **v0.1** (git tag) is the earlier tensor-tower explorer; it is still available from the menu
  as "All tensors (v0.1)".

- **Ground truth:** `reference/model.py` is a readable PyTorch model (GPT-2 style by default,
  with switches for RoPE / ALiBi / sinusoidal / no positions, LayerNorm / RMSNorm, pre / post
  norm, GELU / ReLU / SwiGLU, MHA / GQA / MQA, causal / bidirectional / sliding / prefix masks,
  tied weights, and ablations for residuals and the 1/√d scale).
- **Live engine:** `src/engine/` is a JS port with reverse-mode autograd. It records every
  `@trace`-marked tensor of model.py under the same key and in the same order.
- **Proof they agree:** `tests/parity.mjs` compares JS against PyTorch on 6 architecture
  variants: every traced tensor, the loss, every parameter gradient, and KV-cache decoding
  (549/549 checks at 1e-9).
- **Trained demo models:** `src/presets/` ships five tiny models trained with the reference
  code: a subword model on TinyStories (byte-level BPE, 4,096 tokens, so most words are one
  token; held-out loss 1.98 nats/token), Alice char-LMs in GPT-2 and LLaMA style, induction
  heads, and word reversal. Link straight to one with `?preset=<id>`, e.g. `?preset=tinystories`.

## Interview prep

- Atlas levels: every component, animated on real numbers, with its equation and its line of code.
- **Code ↔ forward pass** (menu): walk `reference/model.py` line by line with the live tensors.
- **Interview cheat sheet** (menu): live shapes, formula sheet, flashcards and a quiz.
- `reference/numpy_transformer.py`: the whole model in NumPy with every backward pass written out
  and derived in comments (softmax, LayerNorm, attention, RoPE, GQA...). Its test
  (`tests/test_numpy_reference.py`) matches PyTorch on every tensor and gradient.

## Run it

Open `dist/transformer-atlas.html` (built by `node tools/build_single.mjs`) directly, or serve the
repository:

The app is plain ES modules with no build step. It must be served over HTTP (not `file://`):

```bash
cd transformer-atlas
python3 -m http.server 8000       # then open http://localhost:8000
```

From the cluster, forward the port (`ssh -L 8000:localhost:8000 <login-node>`) and open it locally.

## Deploy (GitHub Pages)

The app is static and is published with GitHub Pages at `https://saidgurbuz.github.io/transformer-atlas/`
(from the `site` branch; the source repository can stay private). To ship a new version: bump
`src/version.js`, add a CHANGELOG entry, tag `vX.Y.Z`, then run `tools/deploy_pages.sh`.

```bash
tools/deploy_pages.sh            # rebuild and update the `site` branch (private; nothing is published)
tools/deploy_pages.sh --enable   # also turn on Pages for that branch: the site becomes public
```
The script copies `index.html`, `styles/`, `src/` and the single-file build to the `site` branch
with a `.nojekyll` marker. It deliberately does not use a branch named `gh-pages`, because GitHub
publishes that branch automatically.

## Layout

```
reference/model.py              PyTorch ground truth with @trace markers
reference/numpy_transformer.py  NumPy implementation with a hand-written backward pass (interview study)
src/atlas/                      the zoomable explorer: host, node hierarchy, 2.5D kit, scenes/
src/scene/                      3D engine, tensor sheets, v0.1 tower and explorer plug-ins
src/engine/                     JS tensor ops + autograd, model, KV-cache decoding, tokenizer
src/core/                       store (state/events), router, shell (nav, inspector, settings),
                                describe.js (trace-key catalogue), codemap.js (code <-> keys)
src/ui/                         component kit: heatmap, charts, controls, code view, playbar...
src/views/                      one module per page; registry.js lists them
src/presets/                    trained weights (base64 float32) + manifest
styles/main.css                 design tokens (light + dark) and shared styles
tools/                          fixture export, preset training, code bundling, smoke tests
tests/                          parity (JS vs PyTorch), presets, NumPy reference tests
docs/ATLAS_GUIDE.md             how to write a scene (rules, contract, kit, checks)
docs/MATH_GUIDE.md              the math catalogue (schema, notation, conventions)
docs/AGENT_GUIDE.md             store, trace keys, engine and UI-kit reference
docs/SCENE_GUIDE.md             3D explorer plug-ins (v0.1 tower)
```

## Verify

```bash
.venv/bin/python tools/export_fixtures.py     # PyTorch -> tests/fixtures/*.json
node tests/parity.mjs                          # JS engine vs PyTorch (traces, grads, KV cache)
node tests/presets.mjs                         # shipped weights reproduce PyTorch logits
node tests/bpe.mjs                             # JS BPE tokenizer == the Python one (204 texts)
node tests/math.mjs                            # every equation compiles; live numbers match the trace
node tests/codemap.mjs                         # active branches, exact cell coordinates, embedded source freshness
node tools/gen_codemap.mjs --check              # fail if the browser's Python source is stale
.venv/bin/python tests/test_model_reference.py # current PyTorch source, batched KV cache, sampling and validation
.venv/bin/python tests/test_numpy_reference.py # NumPy forward + manual backward vs PyTorch
tools/smoke.sh --all                           # every atlas node in headless Chromium, light + dark
tools/smoke.sh --view head.1.2 --out shot.png  # one node + screenshot
tools/smoke.sh --view tokens --preset tinystories --out shot.png   # any node, any model
tools/smoke.sh --all --themes light --shard 0/3 --shots dir        # a third of a sweep, with screenshots
tools/smoke.sh --view stepper --preset llama --reduced-motion --eval-file tests/browser_cells.js
```

## Extend it

See [the reference model's coverage](docs/MODEL_REFERENCE.md) for what lives in `model.py`,
what belongs to the training/tokenizer tools, and the cache and architecture conventions.

**Add a level:** add a pattern to `src/atlas/nodes.js` (title, parent, kind, blurb, trace keys)
and write `src/atlas/scenes/<name>.js` (copy `softmax.js`; rules and API in
`docs/ATLAS_GUIDE.md`). Mark zoom targets with `data-node` and tensors with `data-trace` /
`data-cell`; the atlas adds hover maths, right-click code and zoom transitions.

**Add or change a model feature:**
1. Edit `reference/model.py`; put `# @trace <key>` on any new line whose result should be visible.
2. Mirror it in `src/engine/model.js` (and `generate.js` if decoding is affected) with `rec('<key>', ...)`.
3. Describe the key in `src/core/describe.js` (title, kind, view, dims, formula).
4. `node tools/gen_codemap.mjs` (bundles model.py for the code panels).
5. Add a case to `CASES` in `tools/export_fixtures.py`, re-export, and run `node tests/parity.mjs`.

**Retrain the presets** (cluster jobs, never on the login node; `tools/fetch_data.sh` gets the corpora):

```bash
bsub -G grp_vision_docling -q normal -n 8 -R "rusage[mem=8GB] span[hosts=1]" -W 120 \
     -o logs/presets.%J.log .venv/bin/python tools/train_presets.py
# subword preset: tokenizer + model, ~32 min on 32 CPU cores (a CUDA torch build also works)
STEPS=20000 bsub -G grp_vision_docling -q normal -n 32 -R "rusage[mem=32GB] span[hosts=1]" -W 240 \
     -o logs/bpe.%J.log .venv/bin/python tools/train_bpe_preset.py
node tests/presets.mjs && node tests/bpe.mjs
```

## Environment notes (handoff)

- Python env: `.venv` (Python 3.11) was created with `uv venv`, but packages were installed
  with **pip**, not uv. Inside the Claude Code command sandbox `uv pip install` fails at once with
  `Operation not permitted (os error 1)` for any cache location, and outside it the shared uv
  cache on GPFS fails with `Directory not empty (os error 39)` while unpacking wheels. Fallback
  used: `.venv/bin/python -m ensurepip` then
  `.venv/bin/python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu`
  and `pip install numpy`. Try uv first in a normal shell.
- Browser tests use Playwright 1.48.2 installed under `tools/node_modules` with its Chromium in
  the user's Playwright cache. On a login node `tools/smoke.sh` sends each run to the LSF
  interactive queue (~2 s to start; `SMOKE_LOCAL=1` forces a local run). `tools/smoke.mjs` sets
  `TMPDIR=/tmp`: the cluster's TMPDIR is on GPFS, where Chromium's cache made pages take ~30 s.
- Subword preset: `tokenizers` 0.23 (pip, same fallback as above) trains the BPE; the JS port in
  `src/engine/bpe.js` is checked against it by `tests/bpe.mjs`. TinyStories (Eldan & Li, 2023) is
  CDLA-Sharing-1.0; only the trained weights ship, not the data.
- Preset metrics: the char-LMs report training-set loss (there is no held-out split; the models
  memorise much of the book). Induction: 99.7% next-token accuracy on repeats. Reverse: 100%.
- Corpus: `data/alice_raw.txt`, *Alice's Adventures in Wonderland* from Project Gutenberg
  (public domain in the US).

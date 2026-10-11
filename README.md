# Transformer Explorer

An interactive, line-by-line tour of a decoder-only Transformer. Every tensor of a real
forward pass is visible, and each one is tied to the exact line of PyTorch that produces it.

- **Ground truth:** `reference/model.py` is a readable PyTorch model (GPT-2 style by default,
  with switches for RoPE / ALiBi / sinusoidal / no positions, LayerNorm / RMSNorm, pre / post
  norm, GELU / ReLU / SwiGLU, MHA / GQA / MQA, causal / bidirectional / sliding / prefix masks,
  tied weights, and ablations for residuals and the 1/√d scale).
- **Live engine:** `src/engine/` is a JS port with reverse-mode autograd. It records every
  `@trace`-marked tensor of model.py under the same key and in the same order.
- **Proof they agree:** `tests/parity.mjs` compares JS against PyTorch on 6 architecture
  variants: every traced tensor, the loss, every parameter gradient, and KV-cache decoding
  (549/549 checks at 1e-9).
- **Trained demo models:** `src/presets/` ships four tiny models trained with the reference
  code (Alice char-LM GPT-2 style, Alice char-LM LLaMA style, induction heads, word reversal).

## Run it

The app is plain ES modules with no build step. It must be served over HTTP (not `file://`):

```bash
cd transformer-explorer
python3 -m http.server 8000       # then open http://localhost:8000
```

From the cluster, forward the port (`ssh -L 8000:localhost:8000 <login-node>`) and open it locally.

## Layout

```
reference/model.py              PyTorch ground truth with @trace markers
reference/numpy_transformer.py  NumPy implementation with a hand-written backward pass
src/engine/                     JS tensor ops + autograd, model, KV-cache decoding, tokenizer
src/core/                       store (state/events), router, shell (nav, inspector, settings),
                                describe.js (trace-key catalogue), codemap.js (code <-> keys)
src/ui/                         component kit: heatmap, charts, controls, code view, playbar...
src/views/                      one module per page; registry.js lists them
src/presets/                    trained weights (base64 float32) + manifest
styles/main.css                 design tokens (light + dark) and shared styles
tools/                          fixture export, preset training, code bundling, smoke tests
tests/                          parity (JS vs PyTorch), presets, NumPy reference tests
docs/AGENT_GUIDE.md             the contributor contract (APIs, trace keys, design rules)
```

## Verify

```bash
.venv/bin/python tools/export_fixtures.py     # PyTorch -> tests/fixtures/*.json
node tests/parity.mjs                          # JS engine vs PyTorch (traces, grads, KV cache)
node tests/presets.mjs                         # shipped weights reproduce PyTorch logits
.venv/bin/python tests/test_numpy_reference.py # NumPy forward + manual backward vs PyTorch
tools/smoke.sh --all                           # every view in headless Chromium, light + dark
tools/smoke.sh --view attention --out shot.png # one view + screenshot
```

## Extend it

**Add a view:** create `src/views/<id>.js` exporting `{id, title, mount(root, ctx)}` (copy
`src/views/_example.js`), and add one line to `src/views/registry.js`. `docs/AGENT_GUIDE.md`
documents `ctx`, the store, the engine and the UI kit.

**Add or change a model feature:**
1. Edit `reference/model.py`; put `# @trace <key>` on any new line whose result should be visible.
2. Mirror it in `src/engine/model.js` (and `generate.js` if decoding is affected) with `rec('<key>', ...)`.
3. Describe the key in `src/core/describe.js` (title, kind, view, dims, formula).
4. `node tools/gen_codemap.mjs` (bundles model.py for the code panels).
5. Add a case to `CASES` in `tools/export_fixtures.py`, re-export, and run `node tests/parity.mjs`.

**Retrain the presets** (cluster job, not on the login node):

```bash
bsub -G grp_vision_docling -q normal -n 8 -R "rusage[mem=8GB] span[hosts=1]" -W 120 \
     -o logs/presets.%J.log .venv/bin/python tools/train_presets.py
node tests/presets.mjs
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
  the user's Playwright cache. `tools/smoke.sh` holds a lock so only one browser runs at a time.
- Preset metrics: the char-LMs report training-set loss (there is no held-out split; the models
  memorise much of the book). Induction: 99.7% next-token accuracy on repeats. Reverse: 100%.
- Corpus: `data/alice_raw.txt`, *Alice's Adventures in Wonderland* from Project Gutenberg
  (public domain in the US).

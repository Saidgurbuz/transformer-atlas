# Development and verification

The application runs as plain ES2022 modules. Its trained weights, JavaScript engine, KaTeX and Three.js assets are included in the repository; serving the source requires no npm installation.

## Build the standalone app

From the repository root, with Node.js and npm installed (the development tools are tested with Node.js 24):

```bash
npm ci --prefix tools
node tools/gen_codemap.mjs
node tools/build_single.mjs
```

Open `dist/transformer-atlas.html` directly in a browser. The build bundles the application, trained weights and equation fonts into one HTML file. Interface web fonts use local fallbacks when unavailable.

`gen_codemap.mjs` embeds the Python references and their trace markers for the code panels. The build rejects stale embedded source.

## Verify the computation and content

The stored PyTorch fixtures are included, so the main JavaScript parity checks can run without installing PyTorch:

```bash
node tests/parity.mjs
node tests/presets.mjs
node tests/bpe.mjs
node tests/codemap.mjs
node tests/store-selection.mjs
node tests/math.mjs
node tests/deep-content.mjs
node tools/gen_codemap.mjs --check
```

These cover traced outputs and gradients, cached decoding, shipped weights, tokenization, cell/source correspondence, input selection, equation rendering, live equation values and source links. Tolerances and cases are defined in each check.

The executable lessons have separate checks:

```bash
node tests/tensor-ops.mjs
node tests/training.mjs
node tests/flash.mjs
node tests/encoder-decoder.mjs
node tests/exercises.mjs
```

See [lesson conventions](LESSONS.md) and [research reference conventions](RESEARCH_CONTENT.md) for what each comparison establishes.

### Python references

The references are tested with Python 3.11, PyTorch and NumPy. With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv venv --python 3.11
uv pip install torch numpy tokenizers
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python tests/test_numpy_reference.py
```

The main [PyTorch reference](../reference/model.py) includes forward computation, next-token loss, sampling and batched cached decoding. The [NumPy reference](../reference/numpy_transformer.py) implements the backward pass explicitly.

To regenerate main-engine fixtures after a reference change, run `.venv/bin/python tools/export_fixtures.py` and then the parity checks. The lesson exporters in `tools/` regenerate their independent fixture sets; some consume inputs produced by the adjacent `*_inputs.mjs` scripts.

### Browser checks

Install the development dependencies above and the browser used by Playwright:

```bash
cd tools
npx playwright install chromium
cd ..
tools/smoke.sh --all
tools/smoke.sh --view model --preset tinystories --eval-file tests/browser_prediction.js
tools/smoke.sh --view model --preset tinystories --eval-file tests/browser_token_focus.js
tools/smoke.sh --view model --preset tinystories --reduced-motion --dpr 2 --eval-file tests/browser_render_budget.js
tools/smoke.sh --view model --preset tinystories --wait 5000 --eval-file tests/browser_shadows.js
tools/smoke.sh --view stepper --preset llama --reduced-motion --eval-file tests/browser_cells.js
```

The harness checks scene loading, browser errors, route handling and horizontal overflow. The interaction scripts verify prediction, selection and source navigation. Use `--theme dark`, `--width 400`, `--out /path/to/screenshot.png`, or `--file dist/transformer-atlas.html` for additional views and the standalone build. `browser_token_focus.js` uses the HTTP app's scene module to check real 3D token-card hits.

The render-budget check verifies one native scene render in both themes, stable GPU resources after theme switches, high-DPI resizing and no renders while the overview is settled. Use `--query root=b` to check the alternate overview; `--dpr 2` simulates a high-DPI display.

The shadow check counts actual WebGL draw commands, compares cached/fresh framebuffer pixels and verifies shadow updates through frozen samples and a completed forward-pass animation. Run it without `--reduced-motion`.

On the shared development cluster, use compute nodes for numerical work, browser checks and training. `tools/smoke.sh` dispatches browser work through the scheduler on cluster login nodes; the [contributor guide](AGENT_GUIDE.md) records the local environment rules.

## Extend the Atlas

- **Add a scene:** register a pattern in [`src/atlas/nodes.js`](../src/atlas/nodes.js) and implement a scene in `src/atlas/scenes/`. The [Atlas guide](ATLAS_GUIDE.md) covers hierarchy, controls, motion, trace links and cleanup.
- **Add equations or sources:** follow the [math catalogue guide](MATH_GUIDE.md) and [research content conventions](RESEARCH_CONTENT.md). Link primary sources to the claims they support, and keep deeper derivations behind the existing detail controls.
- **Change a model operation:** update the PyTorch reference and its JavaScript counterpart, add trace markers and catalogue entries, regenerate embedded source and independent fixtures, and verify parity. The [reference coverage document](MODEL_REFERENCE.md) describes the architecture and cache conventions.

## Train the educational presets

[`tools/train_presets.py`](../tools/train_presets.py) trains the character, induction and reversal presets. [`tools/train_bpe_preset.py`](../tools/train_bpe_preset.py) trains the byte-level BPE tokenizer and TinyStories model. Each script documents its inputs, output files and execution options.

Training data is downloaded separately. Character-model losses in the manifest are training-set metrics; TinyStories reports held-out metrics. Induction and reversal accuracy measure their respective synthetic tasks. These metrics describe different evaluation settings.

## Publish with GitHub Pages

[Deploy Pages](../.github/workflows/pages.yml) runs automatically after pushes to `main`, or manually from the repository's **Actions** tab. It installs the locked build dependencies, checks computation and source correspondence, builds the publication files and deploys them through GitHub Pages. The `github-pages` environment allows deployments from `main`.

[`tools/build_pages.sh`](../tools/build_pages.sh) prepares the same files locally in `dist/pages/`, including the standalone HTML and source-commit metadata. The workflow uploads this directory as an artifact; generated website files do not need their own branch.

To redeploy the current remote `main` with GitHub CLI authentication:

```bash
tools/deploy_pages.sh
gh run list --workflow pages.yml
```

Push local changes before requesting a deployment. In repository **Settings → Pages**, the publishing source must be **GitHub Actions**.

For an application release, update [`src/version.js`](../src/version.js) and the [changelog](../CHANGELOG.md), validate the change, and tag the release. Pushing `main` starts its deployment.

## Repository map

| Location | Purpose |
| --- | --- |
| `src/atlas/` | Scene hierarchy, navigation and interactive lessons |
| `src/engine/` | Browser tensor operations, autograd, model, tokenizer and generation |
| `src/core/` | State, tensor descriptions, source mapping, equations and citations |
| `src/scene/`, `src/ui/`, `styles/` | Rendering, shared controls and theme tokens |
| `src/presets/` | Trained weights and preset metadata |
| `reference/` | Executable PyTorch and NumPy references |
| `tests/` | Numerical fixtures, reference checks and browser interactions |
| `tools/` | Build, export, training, browser and deployment tools |

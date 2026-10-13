# Optional research depth

The main Atlas scenes retain their existing controls and illustrations. Open **Maths → Further study** to reach the new explanations; derivations and implementation links start collapsed. Code links include a return action to the explanation. Numerical examples are executable Python references, rather than extra numeric grids on the main stage.

| Material | Reach from | Reference / verification |
| --- | --- | --- |
| Complete attention backward chain, shared gradients and implemented activation derivatives | Training, attention, norms and MLP equations | `attention_backward.py`, `numpy_transformer.py`; SDPA/autograd and numerical-gradient checks |
| Softmax/normalization invariants, head basis changes, permutations and cache validity | Attention, norm, position and cache equations | `invariance_checks.py`; exact identities and explicit counterexamples |
| QK/OV writes, MLP directions and interpretation scope | Attention and MLP equations | `interpretation_checks.py`; vector decompositions and direct-lens counterexamples |
| SDPA masks, rectangular causal alignment, dropout, GQA and precision | Attention and cache equations | `sdpa_checks.py`; CPU float64/float32/bfloat16 comparisons |
| QK normalization and smooth score capping | Attention and architecture equations | `modern_attention.py`; L2/RMS distinctions, score bounds and gradients |
| PI/YaRN frequencies, temperature and cache compatibility | Positional and cache equations | `rope_extensions.py`; phase identities, stale-key and low-precision counterexamples |
| MLA latent storage and absorbed projections | Attention, architecture and cache equations | `mla_reference.py`; dense/absorbed output and gradient agreement |

Research extensions are explicitly labeled **separate references, not active main-model variants**. They do not load Gemma or DeepSeek checkpoints or establish trained-model quality, long-context performance or GPU speed. The MLA comparison uses the same MLA parameterization on both paths; it does not claim arbitrary MHA compression is lossless.

Explanations distinguish exact algebra, conditions needed by a derivation, implementation conventions and empirical paper findings. For example, epsilon changes normalization’s radial invariance, masked edges do not freeze a whole key/value vector, and an identity path is not a universal training-stability guarantee. The reference manual rejects all-masked rows; the tested CPU SDPA path adopts a zero-output convention. Neither convention defines a normalized probability distribution for that row.

The YaRN reference exposes both the paper’s rotation-count ramp and the authors’ pair-index ramp. The index branch matched 36 configurations from the inspected author implementation, whose SHA256 was `fa097066e5792239ff4375dd84ed65d6c01bce8a2b6f1083dd4504151210a4af`. The comparison covers frequencies and amplitude, not all rotary-layout or checkpoint conventions.

## Maintaining content

Math entries live in `src/core/math/`. A derivation may declare `reference: {file, region}` to link a bundled executable function. Use `scope` for an always-visible qualification in a detailed equation card; it does not add text to scene headers or compact tooltips. `study.js` connects topics through the existing collapsed sections. Equation sources live in `sources.js`, with paper sections or versioned documentation contracts and a statement of scope.

When adding a source, prefer the primary paper, authors’ code or official API documentation. Do not invent a publication year for software: nullable years are supported. Distinguish a paper’s empirical findings from locally derived identities and prototype choices. The main page should explain the operation briefly; detailed conditions and counterexamples belong in the equation sections.

## Checks

Follow the contributor guide’s compute-node rules for numerical/browser work:

```bash
node tests/math.mjs
node tests/deep-content.mjs
node tools/gen_codemap.mjs --check
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python tests/test_numpy_reference.py
tools/smoke.sh --view training --eval-file tests/browser_deep_content.js
tools/smoke.sh --view training --eval-file tests/browser_study.js
```

The catalogue check validates 305 study links, 33 executable source regions, source metadata and explicit extension scope. Browser checks visit every new chapter, expand its derivations, follow source links and return, check drawer bounds, and verify that the main lesson text is unchanged. Also check the standalone build and both desktop/mobile themes.

The optional `tools/check_yarn_reference.py` comparison accepts an inspected author-module path and performs no download. The author module is used only for validation and is not bundled into Atlas.

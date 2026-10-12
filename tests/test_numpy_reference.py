#!/usr/bin/env python3
"""Checks reference/numpy_transformer.py against the PyTorch float64 fixtures and numerical gradients.

Run from the project root with the project venv:
    .venv/bin/python tests/test_numpy_reference.py

For every tests/fixtures/*.json:
  * forward() must produce exactly the trace keys the fixture has;
  * every trace tensor and every parameter gradient must match the fixture to rtol = atol = 1e-8.
    JSON null means -inf, and -inf must sit in the same places.
Then central differences check the analytic gradients on tiny random models.
The exit status is 1 if any check fails and 0 otherwise.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import numpy_transformer as nt  # noqa: E402  (import after the path is set up)

FIXTURE_DIR = ROOT / "tests" / "fixtures"
RTOL = 1e-8
ATOL = 1e-8

# Central differences. The truncation error is about FD_STEP^2 f'''/6, roughly 1e-13, and the
# round-off error is about 1e-16 |loss| / FD_STEP, roughly 1e-9, at FD_STEP = 1e-6. These
# tolerances leave margin over that and still catch sign, scale and missing-term bugs.
FD_STEP = 1e-6
FD_RTOL = 1e-6
FD_ATOL = 1e-7


# ------------------------------------------------------------------ fixture checks

def decode(entry: dict) -> np.ndarray:
    """A fixture tensor {shape, data} as float64. JSON null stands for -inf."""
    flat = [-np.inf if v is None else v for v in entry["data"]]
    return np.asarray(flat, dtype=np.float64).reshape(entry["shape"])


def max_abs_error(got: np.ndarray, want: np.ndarray) -> float:
    """Largest |got - want| over entries that are finite on both sides (0.0 if there are none)."""
    both_finite = np.isfinite(got) & np.isfinite(want)
    if not both_finite.any():
        return 0.0
    return float(np.max(np.abs(got[both_finite] - want[both_finite])))


def compare(name: str, got, want: np.ndarray):
    """Return (max finite error, problem description or None). Shapes must match exactly."""
    got = np.asarray(got, dtype=np.float64)
    if got.shape != want.shape:
        return np.inf, f"{name}: shape {got.shape}, fixture has {want.shape}"
    err = max_abs_error(got, want)
    if np.allclose(got, want, rtol=RTOL, atol=ATOL, equal_nan=False):
        return err, None
    bad = np.argwhere(~np.isclose(got, want, rtol=RTOL, atol=ATOL, equal_nan=False))
    first = tuple(int(v) for v in bad[0])
    return err, (f"{name}: {len(bad)} entries outside tolerance, first at {first}: "
                 f"got {got[first]!r}, fixture {want[first]!r}")


def check_fixture(path: Path):
    """Run forward and backward on one fixture. Returns (passed, summary line, detail lines)."""
    fixture = json.loads(path.read_text())
    name = fixture["name"]
    cfg = nt.Config(**fixture["config"])
    params = {key: decode(entry) for key, entry in fixture["params"].items()}
    want_grads = {key: decode(entry) for key, entry in fixture["grads"].items()}
    want_trace = {key: decode(entry) for key, entry in fixture["trace"].items()}

    model = nt.NumpyGPT(cfg)
    _, loss, trace = model.forward(params, np.asarray(fixture["ids"]), np.asarray(fixture["targets"]))
    grads = model.backward()

    problems = []
    if set(trace) != set(want_trace):
        problems.append(f"trace keys differ: missing {sorted(set(want_trace) - set(trace))}, "
                        f"unexpected {sorted(set(trace) - set(want_trace))}")

    trace_err, n_trace_ok = 0.0, 0
    for key, want in want_trace.items():
        if key not in trace:
            continue
        err, problem = compare(key, trace[key], want)
        trace_err = max(trace_err, err)
        if problem is None:
            n_trace_ok += 1
        else:
            problems.append(problem)

    grad_err, n_grad_ok = 0.0, 0
    for key, want in want_grads.items():
        if key not in grads:
            problems.append(f"no gradient returned for {key}")
            continue
        err, problem = compare(f"grad {key}", grads[key], want)
        grad_err = max(grad_err, err)
        if problem is None:
            n_grad_ok += 1
        else:
            problems.append(problem)
    unexpected_grads = sorted(set(grads) - set(want_grads))
    if unexpected_grads:
        problems.append(f"gradients for names the fixture does not have: {unexpected_grads}")

    passed = not problems
    line = (f"{name:<24} {'PASS' if passed else 'FAIL'}  "
            f"trace {n_trace_ok}/{len(want_trace)}  grads {n_grad_ok}/{len(want_grads)}  "
            f"max|err| trace {trace_err:.1e} grads {grad_err:.1e}  loss {float(loss):.6f}")
    return passed, line, problems


def run_fixture(path: Path):
    try:
        return check_fixture(path)
    except Exception as exc:  # keep going so every fixture gets a line; the exit status records it
        tail = traceback.format_exc().strip().splitlines()[-3:]
        return False, f"{path.stem:<24} FAIL  exception", [f"{type(exc).__name__}: {exc}", *tail]


# ------------------------------------------------------------ numerical gradients

def check_gradients(cfg, names, rng, n_entries):
    """Analytic gradient vs central difference, on n_entries random entries of each named parameter.
    Returns (passed, worst |analytic - numeric|, lines describing failures)."""
    T = min(6, cfg.block_size)
    ids = rng.integers(0, cfg.vocab_size, size=T)
    targets = rng.integers(0, cfg.vocab_size, size=T)
    targets[0] = -1                                        # exercise ignore_index too
    params = {name: rng.normal(0.0, 0.5, size=shape) for name, shape in nt.param_shapes(cfg).items()}

    model = nt.NumpyGPT(cfg)
    model.forward(params, ids, targets)
    grads = model.backward()

    worst, failures = 0.0, []
    for name in names:
        if name not in params:
            failures.append(f"{name}: not a parameter of this config")
            continue
        if name not in grads:
            failures.append(f"{name}: backward() returned no gradient for it")
            continue
        if np.shape(grads[name]) != params[name].shape:
            failures.append(f"{name}: gradient shape {np.shape(grads[name])}, "
                            f"parameter shape {params[name].shape}")
            continue
        size = params[name].size
        for flat in rng.choice(size, size=min(n_entries, size), replace=False):
            index = np.unravel_index(flat, params[name].shape)
            loss_at = {}
            for sign in (+1.0, -1.0):
                bumped = params[name].copy()
                bumped[index] += sign * FD_STEP
                _, loss_at[sign], _ = model.forward({**params, name: bumped}, ids, targets)
            numeric = (loss_at[+1.0] - loss_at[-1.0]) / (2.0 * FD_STEP)
            analytic = grads[name][index]
            err = abs(numeric - analytic)
            worst = max(worst, err)
            if err > FD_ATOL + FD_RTOL * abs(analytic):
                failures.append(f"{name}{tuple(int(v) for v in index)}: "
                                f"analytic {analytic:.10g}, numeric {numeric:.10g}")
    return not failures, worst, failures


TINY = nt.Config(vocab_size=7, block_size=8, n_layer=2, n_head=4, n_kv_head=2, d_model=8, d_ff=12,
                 pos_encoding="rope", norm="layernorm", norm_placement="pre", activation="swiglu",
                 mask="sliding", window=3, tie_weights=False, bias=True, residual=True,
                 attn_scale=True)
TINY_NAMES = ["wte.weight", "blocks.0.attn.q_proj.weight", "blocks.0.attn.k_proj.weight",
              "blocks.1.attn.o_proj.bias", "blocks.1.mlp.gate_proj.weight", "blocks.0.ln1.bias",
              "lm_head.weight"]

# Option combinations that no fixture covers. Every parameter is checked in each.
_SMALL = dict(vocab_size=7, block_size=8, n_layer=2, n_head=4, d_model=8, d_ff=12)
SWEEP = {
    "learned+post+gelu+rmsnorm+tied": nt.Config(**_SMALL, n_kv_head=4, pos_encoding="learned",
                                                norm="rmsnorm", norm_placement="post",
                                                activation="gelu", mask="none", tie_weights=True),
    "alibi+pre+relu+prefix+MQA+noscale": nt.Config(**_SMALL, n_kv_head=1, pos_encoding="alibi",
                                                   norm="layernorm", norm_placement="pre",
                                                   activation="relu", mask="prefix", prefix_len=2,
                                                   bias=False, attn_scale=False),
    "sinusoidal+post+noresid+swiglu": nt.Config(**_SMALL, n_kv_head=2, pos_encoding="sinusoidal",
                                                norm="layernorm", norm_placement="post",
                                                activation="swiglu", mask="causal", bias=False,
                                                residual=False),
    "none+pre+rmsnorm+noresid+sliding+tied": nt.Config(**_SMALL, n_kv_head=4, pos_encoding="none",
                                                       norm="rmsnorm", norm_placement="pre",
                                                       activation="gelu", mask="sliding", window=2,
                                                       tie_weights=True, residual=False),
}


def report_gradient_check(head: str, cfg, names, rng, n_entries) -> bool:
    """Run check_gradients, then print one summary line and one line per failing entry.
    An exception inside a check counts as a failure of that check, so the remaining checks still run."""
    try:
        passed, worst, failures = check_gradients(cfg, names, rng, n_entries)
    except Exception as exc:
        print(f"{head}  FAIL  {type(exc).__name__}: {exc}")
        return False
    print(f"{head}  max|analytic - numeric| {worst:.2e}  {'PASS' if passed else 'FAIL'}")
    for line in failures:
        print(f"    {line}")
    return passed


def run_gradient_checks() -> bool:
    rng = np.random.default_rng(20240601)
    ok = report_gradient_check(
        f"tiny random config (rope, GQA 4/2, swiglu, sliding, untied): "
        f"{len(TINY_NAMES)} params x 3 entries",
        TINY, TINY_NAMES, rng, n_entries=3)
    for label, cfg in SWEEP.items():
        names = list(nt.param_shapes(cfg))
        ok &= report_gradient_check(f"{label}: all {len(names)} params x 2 entries",
                                    cfg, names, rng, n_entries=2)
    return ok


# ------------------------------------------------------------------------- main

def main() -> int:
    paths = sorted(FIXTURE_DIR.glob("*.json"))
    if not paths:
        print(f"no fixtures found in {FIXTURE_DIR}")
        return 1
    print(f"fixtures in {FIXTURE_DIR.relative_to(ROOT)}, rtol = atol = {RTOL:g}")
    all_ok = True
    for path in paths:
        passed, line, details = run_fixture(path)
        print(line)
        for detail in details:
            print(f"    {detail}")
        all_ok &= passed
    print("numerical gradient checks (central differences)")
    all_ok &= run_gradient_checks()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

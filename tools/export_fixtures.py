"""Run reference/model.py on small inputs and save every @trace tensor and gradient.

The tracer reads the `@trace <key>` markers in model.py, lets each marked line run,
then records the variable assigned on that line. No recording calls clutter the
model code. Output: tests/fixtures/<name>.json, consumed by tests/parity.mjs.

    .venv/bin/python tools/export_fixtures.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
from model import GPT, Config  # noqa: E402

MODEL_FILE = str((ROOT / "reference" / "model.py").resolve())
NO_BATCH_AXIS = {"emb.pos", "mask", "alibi", "loss"}


def parse_markers(path: str) -> dict[int, tuple[str, str]]:
    markers = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        m = re.search(r"#.*@trace\s+(\S+)", line)
        if not m:
            continue
        lhs = re.match(r"\s*([A-Za-z_]\w*)\s*=(?!=)", line)
        if not lhs:
            raise ValueError(f"model.py:{n} has a @trace marker but no single-name assignment")
        markers[n] = (m.group(1), lhs.group(1))
    return markers


MARKERS = parse_markers(MODEL_FILE)


def run_traced(fn):
    records: list[tuple[str, torch.Tensor]] = []
    pending: dict[int, int] = {}

    def capture(frame, lineno):
        key, var = MARKERS[lineno]
        if "{i}" in key:
            key = key.replace("{i}", str(frame.f_locals["self"].layer_idx))
        val = frame.f_locals[var]
        records.append((key, val.detach().clone()))

    def local(frame, event, arg):
        fid = id(frame)
        if event in ("line", "return"):
            prev = pending.pop(fid, None)
            if prev in MARKERS:
                capture(frame, prev)
            if event == "line":
                pending[fid] = frame.f_lineno
        return local

    def global_(frame, event, arg):
        return local if frame.f_code.co_filename == MODEL_FILE else None

    sys.settrace(global_)
    try:
        out = fn()
    finally:
        sys.settrace(None)
    return out, records


def encode(t: torch.Tensor) -> dict:
    t = t.detach().to(torch.float64)
    flat = [None if v == float("-inf") else v for v in t.flatten().tolist()]
    return {"shape": list(t.shape), "data": flat}


CASES = {
    "gpt2_style": dict(),
    "post_ln_bidirectional": dict(pos_encoding="sinusoidal", norm="rmsnorm", norm_placement="post",
                                  activation="relu", mask="none", bias=False),
    "llama_style_gqa": dict(pos_encoding="rope", norm="rmsnorm", activation="swiglu", n_kv_head=2,
                            mask="sliding", window=3, tie_weights=True, bias=False),
    "alibi_mqa_prefix": dict(pos_encoding="alibi", n_kv_head=1, mask="prefix", prefix_len=3, attn_scale=False),
    "no_pos_no_residual": dict(pos_encoding="none", residual=False, norm="rmsnorm"),
    "rope_post_three_layers": dict(pos_encoding="rope", norm_placement="post", n_layer=3, n_head=2, n_kv_head=2),
}
BASE = dict(vocab_size=12, block_size=10, n_layer=2, n_head=4, n_kv_head=4, d_model=16, d_ff=32)


def export(name: str, overrides: dict, seed: int) -> None:
    torch.manual_seed(seed)
    cfg = Config(**{**BASE, **overrides})
    model = GPT(cfg)
    for p in model.parameters():                     # non-trivial norms and biases
        with torch.no_grad():
            p.add_(0.05 * torch.randn_like(p))
    T = 7
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    targets = torch.randint(0, cfg.vocab_size, (1, T))
    targets[0, 0] = -1                               # exercise ignore_index
    (logits, loss), records = run_traced(lambda: model(idx, targets))
    loss.backward()
    trace, order = {}, []
    for key, val in records:
        if val.dtype == torch.bool:
            val = val.to(torch.float64)
        if key not in NO_BATCH_AXIS:
            val = val[0]
        trace[key] = encode(val)
        order.append(key)
    fixture = {
        "name": name,
        "config": {**cfg.__dict__},
        "ids": idx[0].tolist(),
        "targets": targets[0].tolist(),
        "params": {n: encode(p) for n, p in model.named_parameters()},
        "grads": {n: encode(p.grad) for n, p in model.named_parameters()},
        "trace": trace,
        "order": order,
    }
    out = ROOT / "tests" / "fixtures" / f"{name}.json"
    out.write_text(json.dumps(fixture))
    print(f"{name:26s} {len(order):3d} traced tensors  loss={loss.item():.6f}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    for n, (name, ov) in enumerate(CASES.items()):
        export(name, ov, seed=100 + n)

"""Train the tiny pretrained models that ship with the app, using reference/model.py.

Run as a cluster job (not on the login node):
    bsub -q normal -n 8 -R "rusage[mem=8GB]" -o logs/presets.%J.log \
        .venv/bin/python tools/train_presets.py

Writes src/presets/<id>.js (weights as base64 float32) and src/presets/manifest.js.
Each preset also stores a reference logit row so tests/presets.mjs can confirm the
JS engine reproduces PyTorch on the shipped weights.
"""
from __future__ import annotations

import base64
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
from model import GPT, Config  # noqa: E402

VOCAB = " abcdefghijklmnopqrstuvwxyz.,!?'"
STOI = {c: i for i, c in enumerate(VOCAB)}
OUT = ROOT / "src" / "presets"
SMOKE = "--smoke" in sys.argv          # a few steps per preset, written to --out DIR
if "--out" in sys.argv:
    OUT = Path(sys.argv[sys.argv.index("--out") + 1])


def encode(text: str) -> list[int]:
    return [STOI[c] for c in text if c in STOI]


def load_corpus() -> str:
    raw = (ROOT / "data" / "alice_raw.txt").read_text(encoding="utf-8")
    body = raw.split("*** START OF THE PROJECT GUTENBERG EBOOK", 1)[1].split("***", 1)[1]
    body = body.split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
    body = body.lower().replace("’", "'").replace("‘", "'").replace("“", "").replace("”", "")
    body = re.sub(r"[;:]", ",", body)
    body = re.sub(r"[\-—_*\[\]()]", " ", body)
    body = "".join(c if c in STOI else " " for c in body)
    return re.sub(r"\s+", " ", body).strip()


# ------------------------------------------------------------------ tasks

def charlm_batch(data: torch.Tensor, T: int, B: int):
    ix = torch.randint(0, len(data) - T - 1, (B,))
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


LETTERS = "abcdefghijklmnopqrstuvwxyz"


def induction_batch(T: int, B: int):
    """A random string followed by repeats of itself. Only the repeats are predictable,
    so the loss there can only be lowered by copying from the earlier occurrence."""
    xs, ys = [], []
    for _ in range(B):
        L = random.randint(5, 12)
        pat = [STOI[random.choice(LETTERS + " ")] for _ in range(L)]
        seq = (pat * (T // L + 2))[: T + 1]
        x, y = seq[:T], seq[1:T + 1]
        y = [t if i >= L - 1 else -1 for i, t in enumerate(y)]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs), torch.tensor(ys)


def reverse_batch(T: int, B: int):
    """'stressed.desserts' : letters, a period, then the letters reversed."""
    xs, ys = [], []
    for _ in range(B):
        L = random.randint(3, (T - 1) // 2)
        word = [random.choice(LETTERS) for _ in range(L)]
        s = encode("".join(word) + "." + "".join(reversed(word)))
        s = s + [STOI[" "]] * (T + 1 - len(s))
        x, y = s[:T], s[1:T + 1]
        y = [t if L <= i < 2 * L else -1 for i, t in enumerate(y)]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs), torch.tensor(ys)


# --------------------------------------------------------------- training

def train(cfg: Config, batch_fn, steps: int, lr: float, eval_fn, seed: int, log_every=250):
    if SMOKE:
        steps, log_every = 12, 6
    torch.manual_seed(seed)
    random.seed(seed)
    model = GPT(cfg)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.98))
    warm = min(200, steps // 10)
    t0 = time.time()
    for step in range(1, steps + 1):
        frac = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * max(frac, 0.05)
        x, y = batch_fn()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            print(f"  step {step:5d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    metrics = eval_fn(model)
    print(f"  eval: {metrics}", flush=True)
    return model, metrics


@torch.no_grad()
def masked_accuracy(model, batch_fn, n=20):
    hit = tot = 0
    for _ in range(n):
        x, y = batch_fn()
        logits, _ = model(x)
        m = y >= 0
        hit += (logits.argmax(-1)[m] == y[m]).sum().item()
        tot += m.sum().item()
    return round(hit / tot, 4)


@torch.no_grad()
def val_loss(model, data, T, n=40):
    losses = []
    for _ in range(n):
        x, y = charlm_batch(data, T, 64)
        losses.append(model(x, y)[1].item())
    return round(sum(losses) / len(losses), 4)


def export(pid: str, title: str, description: str, cfg: Config, model: GPT, samples: list[str], metrics: dict):
    order, chunks = [], []
    for name, p in model.named_parameters():
        order.append([name, list(p.shape)])
        chunks.append(p.detach().to(torch.float32).flatten())
    flat = torch.cat(chunks).numpy().tobytes()
    ids = encode(samples[0])[: cfg.block_size]
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]))
    module = {
        "id": pid,
        "title": title,
        "description": description,
        "vocab": VOCAB,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "samples": samples,
        "metrics": metrics,
        "check": {"ids": ids, "lastLogits": [round(v, 6) for v in logits[0, -1].tolist()]},
        "order": order,
    }
    js = ("// Generated by tools/train_presets.py. Do not edit.\n"
          f"const meta = {json.dumps(module)};\n"
          f"export default {{ ...meta, weights: \"{base64.b64encode(flat).decode()}\" }};\n")
    (OUT / f"{pid}.js").write_text(js)
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote src/presets/{pid}.js  ({n:,} params, {len(js) / 1e6:.2f} MB)", flush=True)
    return {"id": pid, "title": title, "description": description, "params": n, "metrics": metrics}


def main():
    torch.set_num_threads(2 if SMOKE else 8)
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    data = torch.tensor(encode(corpus))
    print(f"corpus: {len(corpus):,} chars", flush=True)
    manifest = []
    lm_samples = ["alice was beginning to get very tired of sitting by her sister on the bank",
                  "the queen of hearts, she made some tarts",
                  "curiouser and curiouser! cried alice"]

    print("charlm (GPT-2 style)", flush=True)
    cfg = Config(vocab_size=len(VOCAB), block_size=64, n_layer=4, n_head=4, n_kv_head=4, d_model=64, d_ff=256)
    m, met = train(cfg, lambda: charlm_batch(data, 64, 64), steps=6000, lr=3e-3, seed=1,
                   eval_fn=lambda mm: {"train_loss_nats": val_loss(mm, data, 64)})
    manifest.append(export("charlm", "Alice char-LM · GPT-2 style",
                           "4 layers, 4 heads, d=64, learned positions, pre-LayerNorm, GELU. Trained on Alice in Wonderland, one character per token (no held-out split: it has memorised much of the book).",
                           cfg, m, lm_samples, met))

    print("llama-style char LM", flush=True)
    cfg = Config(vocab_size=len(VOCAB), block_size=64, n_layer=4, n_head=4, n_kv_head=2, d_model=64, d_ff=172,
                 pos_encoding="rope", norm="rmsnorm", activation="swiglu", bias=False)
    m, met = train(cfg, lambda: charlm_batch(data, 64, 64), steps=6000, lr=3e-3, seed=2,
                   eval_fn=lambda mm: {"train_loss_nats": val_loss(mm, data, 64)})
    manifest.append(export("llama", "Alice char-LM · LLaMA style",
                           "Same data, modern recipe: RoPE, RMSNorm, SwiGLU, grouped-query attention (4 query heads, 2 KV heads), no biases.",
                           cfg, m, lm_samples, met))

    print("induction heads", flush=True)
    cfg = Config(vocab_size=len(VOCAB), block_size=32, n_layer=2, n_head=4, n_kv_head=4, d_model=64, d_ff=256)
    m, met = train(cfg, lambda: induction_batch(32, 64), steps=4000, lr=2e-3, seed=3,
                   eval_fn=lambda mm: {"repeat_accuracy": masked_accuracy(mm, lambda: induction_batch(32, 64))})
    manifest.append(export("induction", "Induction heads",
                           "2 layers trained to continue repeated random text. Look for a previous-token head in layer 0 and an induction head in layer 1.",
                           cfg, m, ["xqkmwrbz xqkmwrbz xqkmwrbz", "the cat sat. the cat sat. the cat", "hello world hello world hello"], met))

    print("reverse", flush=True)
    cfg = Config(vocab_size=len(VOCAB), block_size=24, n_layer=2, n_head=4, n_kv_head=4, d_model=64, d_ff=256)
    m, met = train(cfg, lambda: reverse_batch(24, 64), steps=5000, lr=2e-3, seed=4,
                   eval_fn=lambda mm: {"reversed_char_accuracy": masked_accuracy(mm, lambda: reverse_batch(24, 64))})
    manifest.append(export("reverse", "Reverse a word",
                           "2 layers trained on 'word.drow'. After the period, attention reads the input backwards: an anti-diagonal pattern.",
                           cfg, m, ["stressed.desserts", "transformer.remrofsnart", "attention.noitnetta"], met))

    (OUT / "manifest.js").write_text("// Written by tools/train_presets.py.\nexport const MANIFEST = "
                                     + json.dumps(manifest, indent=2) + ";\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()

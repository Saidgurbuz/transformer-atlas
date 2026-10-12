"""Train a subword (byte-level BPE) tokenizer and a small GPT-2-style model on TinyStories,
then export them as an app preset. Uses reference/model.py unchanged.

Run on a GPU node, not the login node:
    bsub -G grp_vision_docling -q normal -gpu "num=1" -n 8 -R "rusage[mem=32GB] span[hosts=1]" -W 180 \
        -o logs/bpe.%J.log .venv-gpu/bin/python tools/train_bpe_preset.py

Inputs:  data/tinystories/valid.txt, data/tinystories/train_slice.txt (TinyStoriesV2-GPT4)
Outputs: src/presets/tinystories.js (fp16 weights + tokenizer), src/presets/manifest.js (entry
         added first), tests/fixtures/bpe_cases.json (tokenizer parity cases for tests/bpe.mjs)
"""
from __future__ import annotations

import base64
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
from model import GPT, Config  # noqa: E402

DATA = ROOT / "data" / "tinystories"
OUT = ROOT / "src" / "presets"
EOT = "<|endoftext|>"
VOCAB = 4096
SMOKE = "--smoke" in sys.argv
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES = [
    "Lily had a red ball. She liked to play with it every day. One day, she",
    "Tom and his dog went to the park. The dog saw a big stick and",
    "Once upon a time, there was a little bird. The bird wanted to fly high in the sky. But the bird",
    "Mia wanted to bake a cake for her mom. She got the flour, the eggs and",
]


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", " - ").replace("–", "-").replace("…", "...")
    text = "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)
    return re.sub(r"[ \t]+", " ", text).strip()


def stories(path: Path, limit: int | None = None):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    out = []
    for s in raw.split(EOT):
        s = clean(s)
        if len(s) > 40:
            out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def train_tokenizer(texts):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOT], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer)
    return tok


def encode_all(tok, texts):
    eot = tok.token_to_id(EOT)
    ids = []
    for i in range(0, len(texts), 2000):
        for e in tok.encode_batch(texts[i:i + 2000]):
            ids.extend(e.ids)
            ids.append(eot)
    return np.array(ids, dtype=np.int64)


def main():
    t0 = time.time()
    torch.manual_seed(0)
    train_texts = stories(DATA / "train_slice.txt", 5000 if SMOKE else None)
    val_texts = stories(DATA / "valid.txt", 1000 if SMOKE else None)
    print(f"stories: train {len(train_texts):,}  val {len(val_texts):,}", flush=True)

    tok = train_tokenizer(train_texts[:200_000])
    spec = json.loads(tok.to_str())["model"]
    inv = {i: t for t, i in spec["vocab"].items()}
    vocab = [inv[i] for i in range(len(inv))]
    merges = [m if isinstance(m, str) else f"{m[0]} {m[1]}" for m in spec["merges"]]
    print(f"tokenizer: {len(vocab)} tokens, {len(merges)} merges ({time.time() - t0:.0f}s)", flush=True)

    train_ids = encode_all(tok, train_texts)
    val_ids = encode_all(tok, val_texts)
    chars = sum(len(t) for t in train_texts[:2000])
    toks = sum(len(e.ids) for e in tok.encode_batch(train_texts[:2000]))
    print(f"tokens: train {len(train_ids):,}  val {len(val_ids):,}  ({chars / toks:.2f} chars/token, {time.time() - t0:.0f}s)", flush=True)

    cfg = Config(vocab_size=len(vocab), block_size=64, n_layer=4, n_head=4, n_kv_head=4, d_model=128, d_ff=512,
                 tie_weights=True)
    torch.set_default_device(DEVICE)           # model.py builds positions and masks with default tensors
    torch.backends.cuda.matmul.allow_tf32 = True
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters on {DEVICE}", flush=True)

    T, B = cfg.block_size, 128
    steps = 60 if SMOKE else 30_000
    train_t = torch.as_tensor(train_ids, device=DEVICE)
    val_t = torch.as_tensor(val_ids, device=DEVICE)

    def batch(data):
        ix = torch.randint(0, len(data) - T - 1, (B,))
        off = torch.arange(T + 1)
        seq = data[ix[:, None] + off[None, :]]
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=2e-3, betas=(0.9, 0.95), fused=DEVICE == "cuda")
    warm = max(1, min(1000, steps // 10))

    @torch.no_grad()
    def val_loss(n=50):
        model.eval()
        ls = [model(*batch(val_t))[1].item() for _ in range(n)]
        model.train()
        return sum(ls) / len(ls)

    for step in range(1, steps + 1):
        frac = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g in opt.param_groups:
            g["lr"] = 2e-3 * max(frac, 0.03)
        x, y = batch(train_t)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == steps:
            print(f"  step {step:6d}  train {loss.item():.4f}  val {val_loss(20):.4f}  ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- export
    model = model.to("cpu").double()
    torch.set_default_device("cpu")
    with torch.no_grad():                       # ship fp16 weights; evaluate exactly what ships
        for p in model.parameters():
            p.copy_(p.half().double())
    torch.set_default_dtype(torch.float64)
    model.eval()
    with torch.no_grad():
        vl = []
        for _ in range(5 if SMOKE else 100):
            ix = np.random.randint(0, len(val_ids) - T - 1, size=32)
            xs = torch.as_tensor(np.stack([val_ids[i:i + T] for i in ix]))
            ys = torch.as_tensor(np.stack([val_ids[i + 1:i + T + 1] for i in ix]))
            vl.append(model(xs, ys)[1].item())
        vloss = float(np.mean(vl))
    print(f"held-out loss (fp16 weights): {vloss:.4f} nats/token, perplexity {math.exp(vloss):.2f}", flush=True)

    # greedy continuations, for the log and the preset
    def generate(prompt, n=24):
        ids = tok.encode(prompt).ids
        with torch.no_grad():
            for _ in range(n):
                logits, _ = model(torch.as_tensor([ids[-T:]]))
                ids.append(int(logits[0, -1].argmax()))
        return tok.decode(ids)
    for s in SAMPLES[:2]:
        print("  >", generate(s).replace("\n", " "), flush=True)

    order, chunks = [], []
    for name, p in model.named_parameters():
        order.append([name, list(p.shape)])
        chunks.append(p.detach().to(torch.float16).flatten().numpy())
    flat = np.concatenate(chunks).astype(np.float16).tobytes()
    check_ids = tok.encode(SAMPLES[0]).ids[:T]
    with torch.no_grad():
        logits, _ = model(torch.as_tensor([check_ids]))
    module = {
        "id": "tinystories",
        "title": "TinyStories · subword BPE · GPT-2 style",
        "description": "4 layers, 4 heads, d=128, a 4,096-token byte-level BPE vocabulary (most words are one token), tied embeddings. Trained on TinyStories; held-out loss "
                       f"{vloss:.2f} nats/token.",
        "config": {k: v for k, v in cfg.__dict__.items()},
        "tokenizer": {"type": "bpe", "vocab": vocab, "merges": merges, "special": [EOT]},
        "samples": SAMPLES,
        "metrics": {"val_loss_nats": round(vloss, 4), "val_perplexity": round(math.exp(vloss), 2), "params": n_params},
        "check": {"ids": check_ids, "lastLogits": [round(v, 6) for v in logits[0, -1].tolist()]},
        "order": order,
        "dtype": "f16",
    }
    js = ("// Generated by tools/train_bpe_preset.py. Do not edit.\n"
          f"const meta = {json.dumps(module)};\n"
          f"export default {{ ...meta, weights: \"{base64.b64encode(flat).decode()}\" }};\n")
    target = OUT / ("tinystories_smoke.js" if SMOKE else "tinystories.js")
    target.write_text(js)
    print(f"wrote {target.relative_to(ROOT)} ({len(js) / 1e6:.2f} MB)", flush=True)

    # tokenizer parity cases for the JS port
    cases = val_texts[:200] + ["Hello world!", "  two  spaces and\nnew lines\n\n", "Numbers 123 and 4567.", "don't can't I'm you're we'll"]
    (ROOT / "tests" / "fixtures" / ("bpe_cases_smoke.json" if SMOKE else "bpe_cases.json")).write_text(
        json.dumps({"cases": [{"text": c, "ids": tok.encode(c).ids} for c in cases]}))

    if not SMOKE:
        man_path = OUT / "manifest.js"
        src = man_path.read_text()
        manifest = json.loads(src[src.index("["): src.rindex("]") + 1])
        manifest = [m for m in manifest if m["id"] != "tinystories"]
        manifest.insert(0, {"id": "tinystories", "title": module["title"], "description": module["description"],
                            "params": n_params, "metrics": module["metrics"]})
        man_path.write_text("// Written by tools/train_presets.py and tools/train_bpe_preset.py.\nexport const MANIFEST = "
                            + json.dumps(manifest, indent=2) + ";\n")
        print("manifest updated", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

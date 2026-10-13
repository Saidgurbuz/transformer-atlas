"""Transformer Atlas: reference decoder-only Transformer in PyTorch.

This file is the ground truth for the app. A line that produces a tensor the app
can show ends with a `@trace <key>` marker; `{i}` stands for the layer index.
tools/export_fixtures.py records the marked variables while this code runs, and
tests/parity.mjs checks that the JS engine reproduces every one of them, plus
every parameter gradient.

Shape letters: B batch, T sequence length, d d_model, H n_head, Hkv n_kv_head,
dh = d / H (head width), F d_ff (MLP width), V vocab size.

Includes full-sequence training, sampling, and incremental causal/sliding decoding.
Tokenizer/data loading and optimizer loops live in tools/train_*; NumPy's manual
backward pass lives in numpy_transformer.py. This is an educational decoder,
not an encoder-decoder, a checkpoint loader, or a fused production kernel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 32
    block_size: int = 32            # longest sequence the model accepts
    n_layer: int = 2
    n_head: int = 4
    n_kv_head: int = 4              # = n_head: MHA · 1: MQA · in between: GQA
    d_model: int = 64
    d_ff: int = 256                 # MLP hidden width (4·d is the classic ratio)
    pos_encoding: str = "learned"   # learned | sinusoidal | rope | alibi | none
    norm: str = "layernorm"         # layernorm | rmsnorm
    norm_placement: str = "pre"     # pre (GPT-2, LLaMA) | post (2017 original)
    activation: str = "gelu"        # gelu | relu | swiglu
    mask: str = "causal"            # causal | none | sliding | prefix
    window: int = 8                 # sliding: a query sees itself and window-1 earlier keys
    prefix_len: int = 4             # prefix: the first prefix_len tokens see each other
    tie_weights: bool = False       # lm_head reuses the token embedding matrix
    bias: bool = True               # biases in Linear layers and LayerNorm
    residual: bool = True           # ablation: turn skip connections off
    attn_scale: bool = True         # ablation: drop the 1/sqrt(dh) factor
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_kv_head", "d_model", "d_ff", "window"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        choices = dict(pos_encoding=("learned", "sinusoidal", "rope", "alibi", "none"),
                       norm=("layernorm", "rmsnorm"), norm_placement=("pre", "post"),
                       activation=("gelu", "relu", "swiglu"), mask=("causal", "none", "sliding", "prefix"))
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.d_model % self.n_head or self.n_head % self.n_kv_head:
            raise ValueError("d_model must divide into query heads, and query heads into KV groups")
        if self.pos_encoding == "rope" and self.d_head % 2:
            raise ValueError("RoPE needs an even head width")
        if self.pos_encoding == "sinusoidal" and self.d_model % 2:
            raise ValueError("sinusoidal encoding needs an even d_model")
        if type(self.prefix_len) is not int or self.prefix_len < 0:
            raise ValueError("prefix_len must be a non-negative integer")
        for name in ("rope_base", "norm_eps"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("tie_weights", "bias", "residual", "attn_scale"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


# --------------------------------------------------------------------------- norms

class LayerNorm(nn.Module):
    def __init__(self, d: int, eps: float, bias: bool):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))                       # gamma
        self.bias = nn.Parameter(torch.zeros(d)) if bias else None      # beta

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)                       # per-token mean
        var = x.var(-1, keepdim=True, unbiased=False)       # per-token variance
        x_hat = (x - mu) / torch.sqrt(var + self.eps)       # zero mean; epsilon-adjusted variance
        y = x_hat * self.weight
        return y + self.bias if self.bias is not None else y


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # no mean subtraction
        return x / rms * self.weight


def make_norm(cfg: Config) -> nn.Module:
    if cfg.norm == "layernorm":
        return LayerNorm(cfg.d_model, cfg.norm_eps, cfg.bias)
    return RMSNorm(cfg.d_model, cfg.norm_eps)


# ------------------------------------------------------------- positions and masks

def sinusoidal_table(T: int, d: int, *, device=None, dtype=None):
    pos = torch.arange(T, device=device).unsqueeze(1)       # (T, 1)
    i = torch.arange(0, d, 2, device=device, dtype=dtype or torch.get_default_dtype())
    freq = torch.exp(-math.log(10000.0) * i / d)            # 1 / 10000^(i/d)
    pe = torch.zeros(T, d, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def rope_angles(T: int, dh: int, base: float, start: int = 0, *, device=None, dtype=None):
    k = torch.arange(0, dh, 2, device=device, dtype=dtype or torch.get_default_dtype())
    theta = base ** (-k / dh)                               # rotation speed of each pair
    ang = torch.arange(start, start + T, device=device).unsqueeze(1) * theta
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):                                # x: (B, H, T, dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]                     # the two coordinates of each pair
    y1 = x1 * cos - x2 * sin                                # rotate each 2-D pair by its angle
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)        # interleave back to (.., dh)


def alibi_bias(cfg: Config, T: int, *, device=None, dtype=None):
    H = cfg.n_head
    # Geometric extension for arbitrary H; the paper's non-power-of-two schedule differs.
    slopes = torch.tensor([2 ** (-8 * (h + 1) / H) for h in range(H)], device=device, dtype=dtype)
    pos = torch.arange(T, device=device)
    dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
    return -slopes.view(H, 1, 1) * dist                     # (H, T, T) farther = more negative


def build_mask(cfg: Config, T: int, *, device=None):        # True where query i may attend to key j
    i = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    if cfg.mask == "none":
        return torch.ones(T, T, dtype=torch.bool, device=device)
    if cfg.mask == "sliding":
        return (j <= i) & (j > i - cfg.window)              # causal, only the last `window` keys
    if cfg.mask == "prefix":
        return (j <= i) | (j < cfg.prefix_len)              # prefix-LM: prefix is fully visible
    return j <= i                                           # causal (GPT-style decoder)


# ---------------------------------------------------------------------- attention

class Attention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        d, H, Hkv, dh = cfg.d_model, cfg.n_head, cfg.n_kv_head, cfg.d_head
        self.q_proj = nn.Linear(d, H * dh, bias=cfg.bias)
        self.k_proj = nn.Linear(d, Hkv * dh, bias=cfg.bias)
        self.v_proj = nn.Linear(d, Hkv * dh, bias=cfg.bias)
        self.o_proj = nn.Linear(H * dh, d, bias=cfg.bias)

    def forward(self, x, mask, rope=None, alibi=None, cache=None):
        cfg = self.cfg
        B, T, _ = x.shape
        H, Hkv, dh = cfg.n_head, cfg.n_kv_head, cfg.d_head
        q = self.q_proj(x)                                  # (B, T, H*dh)      @trace L{i}.attn.q
        k = self.k_proj(x)                                  # (B, T, Hkv*dh)    @trace L{i}.attn.k
        v = self.v_proj(x)                                  # (B, T, Hkv*dh)    @trace L{i}.attn.v
        q = q.view(B, T, H, dh).transpose(1, 2)             # (B, H, T, dh)     @trace L{i}.attn.q_heads
        k = k.view(B, T, Hkv, dh).transpose(1, 2)           # (B, Hkv, T, dh)   @trace L{i}.attn.k_heads
        v = v.view(B, T, Hkv, dh).transpose(1, 2)           # (B, Hkv, T, dh)   @trace L{i}.attn.v_heads
        if rope is not None:
            q = apply_rope(q, *rope)                        # rotate by position @trace L{i}.attn.q_rot
            k = apply_rope(k, *rope)                        # rotate by position @trace L{i}.attn.k_rot
        if cache is not None:                              # cache BEFORE repeating KV heads
            k = torch.cat((cache["k"], k), dim=2) if cache["k"] is not None else k
            v = torch.cat((cache["v"], v), dim=2) if cache["v"] is not None else v
            cache["k"], cache["v"] = k, v
        if Hkv < H:                                         # GQA / MQA: query heads share K and V
            k = k.repeat_interleave(H // Hkv, dim=1)        # (B, H, T, dh)     @trace L{i}.attn.k_rep
            v = v.repeat_interleave(H // Hkv, dim=1)        # (B, H, T, dh)     @trace L{i}.attn.v_rep
        scores = q @ k.transpose(-2, -1)                    # (B, H, T, T)      @trace L{i}.attn.scores
        if cfg.attn_scale:
            scores = scores / math.sqrt(dh)                 # keep logits O(1)  @trace L{i}.attn.scaled
        if alibi is not None:
            scores = scores + alibi                         # distance penalty  @trace L{i}.attn.biased
        scores = scores.masked_fill(~mask, float("-inf"))   # hide disallowed keys @trace L{i}.attn.masked
        probs = F.softmax(scores, dim=-1)                   # rows sum to 1     @trace L{i}.attn.probs
        ctx = probs @ v                                     # (B, H, T, dh)     @trace L{i}.attn.ctx
        ctx = ctx.transpose(1, 2).reshape(B, T, H * dh)     # (B, T, H*dh)      @trace L{i}.attn.concat
        out = self.o_proj(ctx)                              # (B, T, d)         @trace L{i}.attn.out
        return out


# ---------------------------------------------------------------------------- MLP

class MLP(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        d, f = cfg.d_model, cfg.d_ff
        self.up_proj = nn.Linear(d, f, bias=cfg.bias)
        if cfg.activation == "swiglu":
            self.gate_proj = nn.Linear(d, f, bias=cfg.bias)
        self.down_proj = nn.Linear(f, d, bias=cfg.bias)

    def forward(self, x):
        up = self.up_proj(x)                                # (B, T, F)         @trace L{i}.mlp.up
        if self.cfg.activation == "swiglu":
            gate = self.gate_proj(x)                        # (B, T, F)         @trace L{i}.mlp.gate
            act = F.silu(gate) * up                         # gated unit        @trace L{i}.mlp.act
        elif self.cfg.activation == "gelu":
            act = F.gelu(up, approximate="tanh")            # (B, T, F)         @trace L{i}.mlp.act
        else:
            act = F.relu(up)                                # (B, T, F)         @trace L{i}.mlp.act
        out = self.down_proj(act)                           # (B, T, d)         @trace L{i}.mlp.out
        return out


# -------------------------------------------------------------------------- block

class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        self.ln1 = make_norm(cfg)
        self.attn = Attention(cfg, layer_idx)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg, layer_idx)

    def forward(self, x, mask, rope, alibi, cache=None):
        cfg = self.cfg
        if cfg.norm_placement == "pre":                     # x + f(norm(x))
            h = self.ln1(x)                                 # (B, T, d)         @trace L{i}.ln1
            a = self.attn(h, mask, rope, alibi, cache)
            x = x + a if cfg.residual else a                # residual add      @trace L{i}.resid_mid
            h = self.ln2(x)                                 # (B, T, d)         @trace L{i}.ln2
            m = self.mlp(h)
            x = x + m if cfg.residual else m                # residual add      @trace L{i}.resid_out
        else:                                               # norm(x + f(x))
            a = self.attn(x, mask, rope, alibi, cache)
            s = x + a if cfg.residual else a                # residual add      @trace L{i}.add1
            x = self.ln1(s)                                 # (B, T, d)         @trace L{i}.resid_mid
            m = self.mlp(x)
            s = x + m if cfg.residual else m                # residual add      @trace L{i}.add2
            x = self.ln2(s)                                 # (B, T, d)         @trace L{i}.resid_out
        return x


# ---------------------------------------------------------------------------- GPT

class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must split evenly across heads"
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_encoding == "learned":
            self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        if cfg.norm_placement == "pre":
            self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight           # one matrix: embed and un-embed
        self.apply(self._init_weights)
        for name, p in self.named_parameters():             # GPT-2: shrink residual writes
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        cfg = self.cfg
        B, T = idx.shape
        if not 1 <= T <= cfg.block_size:
            raise ValueError("sequence length must be between 1 and block_size")
        device, dtype = idx.device, self.wte.weight.dtype
        tok = self.wte(idx)                                 # (B, T, d)         @trace emb.tok
        if cfg.pos_encoding in ("learned", "sinusoidal"):
            if cfg.pos_encoding == "learned":
                pe = self.wpe(torch.arange(T, device=device))  # (T, d)         @trace emb.pos
            else:
                pe = sinusoidal_table(T, cfg.d_model, device=device, dtype=dtype)  # @trace emb.pos
            x = tok + pe                                    # (B, T, d)         @trace emb.out
        else:
            x = tok                                         # RoPE/ALiBi act inside attention @trace emb.out
        mask = build_mask(cfg, T, device=device)            # (T, T) bool       @trace mask
        rope = rope_angles(T, cfg.d_head, cfg.rope_base, device=device, dtype=dtype) if cfg.pos_encoding == "rope" else None
        alibi = None
        if cfg.pos_encoding == "alibi":
            alibi = alibi_bias(cfg, T, device=device, dtype=dtype)  # (H, T, T)   @trace alibi
        for block in self.blocks:
            x = block(x, mask, rope, alibi)
        if cfg.norm_placement == "pre":
            x = self.ln_f(x)                                # (B, T, d)         @trace lnf
        logits = self.lm_head(x)                            # (B, T, V)         @trace logits
        probs = F.softmax(logits, dim=-1)                   # next-token dist.  @trace probs
        loss = None
        if targets is not None:                             # -1 marks positions with no target
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1), ignore_index=-1)  # @trace loss
        return logits, loss

    def create_cache(self):
        if self.cfg.mask not in ("causal", "sliding"):
            raise ValueError("incremental caching requires a causal or sliding mask")
        return {"length": 0, "owner": self, "layers": [{"k": None, "v": None} for _ in self.blocks]}

    @torch.no_grad()
    def decode_step(self, idx, cache):
        """One new token per batch item; logits [B, 1, V]. No context rollover.

        Recreate the cache after changing weights/config/device/dtype or prompt.
        Sliding masks restrict reads; this cache retains the full prefix.
        """
        cfg = self.cfg
        if cfg.mask not in ("causal", "sliding") or cache.get("owner") is not self:
            raise ValueError("cache must belong to this model and use a causal or sliding mask")
        pos = cache["length"]
        if idx.ndim != 2 or idx.shape[1] != 1 or not 0 <= pos < cfg.block_size:
            raise ValueError("decode_step requires [B, 1] tokens and space in the context window")
        if pos and any(c["k"] is None or c["k"].shape[0] != idx.shape[0] or c["k"].shape[2] != pos
                       or c["k"].device != idx.device or c["k"].dtype != self.wte.weight.dtype for c in cache["layers"]):
            raise ValueError("cache shape/device/dtype no longer matches the model and batch")
        device, dtype = idx.device, self.wte.weight.dtype
        x = self.wte(idx)
        if cfg.pos_encoding == "learned":
            x = x + self.wpe(torch.tensor([pos], device=device))
        elif cfg.pos_encoding == "sinusoidal":
            x = x + sinusoidal_table(pos + 1, cfg.d_model, device=device, dtype=dtype)[pos:]
        rope = rope_angles(1, cfg.d_head, cfg.rope_base, pos, device=device, dtype=dtype) if cfg.pos_encoding == "rope" else None
        keys = torch.arange(pos + 1, device=device)
        mask = (keys > pos - cfg.window).unsqueeze(0) if cfg.mask == "sliding" else torch.ones(1, pos + 1, device=device, dtype=torch.bool)
        alibi = None
        if cfg.pos_encoding == "alibi":
            slopes = torch.tensor([2 ** (-8 * (h + 1) / cfg.n_head) for h in range(cfg.n_head)], device=device, dtype=dtype)
            alibi = -slopes[:, None, None] * (pos - keys)[None, None, :]
        for block, saved in zip(self.blocks, cache["layers"]):
            x = block(x, mask, rope, alibi, saved)
        if cfg.norm_placement == "pre":
            x = self.ln_f(x)
        logits = self.lm_head(x)
        cache["length"] = pos + 1
        return logits

    @torch.no_grad()
    def generate(self, idx, n_new: int, temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None):
        if type(n_new) is not int or n_new < 0 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("n_new must be non-negative and temperature finite and non-negative")
        if top_k is not None and (type(top_k) is not int or top_k < 0):
            raise ValueError("top_k must be a non-negative integer or None")
        if top_p is not None and (not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        for _ in range(n_new):
            idx_cond = idx[:, -self.cfg.block_size:]        # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                       # only the last position predicts
            if temperature == 0:                            # greedy; avoid division by zero
                idx = torch.cat([idx, logits.argmax(-1, keepdim=True)], dim=1)
                continue
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None:                           # nucleus: smallest set with mass >= p
                sorted_logits, order = torch.sort(logits, descending=True)
                cum = F.softmax(sorted_logits, dim=-1).cumsum(-1)
                drop = cum - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

"""Transformer Explorer: reference decoder-only Transformer in PyTorch.

This file is the ground truth for the app. A line that produces a tensor the app
can show ends with a `@trace <key>` marker; `{i}` stands for the layer index.
tools/export_fixtures.py records the marked variables while this code runs, and
tests/parity.mjs checks that the JS engine reproduces every one of them, plus
every parameter gradient.

Shape letters: B batch, T sequence length, d d_model, H n_head, Hkv n_kv_head,
dh = d / H (head width), F d_ff (MLP width), V vocab size.
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
        x_hat = (x - mu) / torch.sqrt(var + self.eps)       # zero mean, unit variance
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

def sinusoidal_table(T: int, d: int):
    pos = torch.arange(T).unsqueeze(1)                      # (T, 1)
    i = torch.arange(0, d, 2)                               # even dims 0, 2, 4, ...
    freq = torch.exp(-math.log(10000.0) * i / d)            # 1 / 10000^(i/d)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def rope_angles(T: int, dh: int, base: float):
    k = torch.arange(0, dh, 2)                              # 0, 2, 4, ... one per pair
    theta = base ** (-k / dh)                               # rotation speed of each pair
    ang = torch.arange(T).unsqueeze(1) * theta              # (T, dh/2) angle = pos * theta
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):                                # x: (B, H, T, dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]                     # the two coordinates of each pair
    y1 = x1 * cos - x2 * sin                                # rotate each 2-D pair by its angle
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)        # interleave back to (.., dh)


def alibi_bias(cfg: Config, T: int):
    H = cfg.n_head
    slopes = torch.tensor([2 ** (-8 * (h + 1) / H) for h in range(H)])  # one slope per head
    dist = (torch.arange(T).unsqueeze(1) - torch.arange(T).unsqueeze(0)).abs()
    return -slopes.view(H, 1, 1) * dist                     # (H, T, T) farther = more negative


def build_mask(cfg: Config, T: int):                        # True where query i may attend to key j
    i = torch.arange(T).unsqueeze(1)
    j = torch.arange(T).unsqueeze(0)
    if cfg.mask == "none":
        return torch.ones(T, T, dtype=torch.bool)           # bidirectional (BERT-style encoder)
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

    def forward(self, x, mask, rope=None, alibi=None):
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

    def forward(self, x, mask, rope, alibi):
        cfg = self.cfg
        if cfg.norm_placement == "pre":                     # x + f(norm(x))
            h = self.ln1(x)                                 # (B, T, d)         @trace L{i}.ln1
            a = self.attn(h, mask, rope, alibi)
            x = x + a if cfg.residual else a                # residual add      @trace L{i}.resid_mid
            h = self.ln2(x)                                 # (B, T, d)         @trace L{i}.ln2
            m = self.mlp(h)
            x = x + m if cfg.residual else m                # residual add      @trace L{i}.resid_out
        else:                                               # norm(x + f(x))
            a = self.attn(x, mask, rope, alibi)
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
        assert T <= cfg.block_size, "sequence longer than block_size"
        tok = self.wte(idx)                                 # (B, T, d)         @trace emb.tok
        if cfg.pos_encoding in ("learned", "sinusoidal"):
            if cfg.pos_encoding == "learned":
                pe = self.wpe(torch.arange(T))              # (T, d)            @trace emb.pos
            else:
                pe = sinusoidal_table(T, cfg.d_model)       # (T, d)            @trace emb.pos
            x = tok + pe                                    # (B, T, d)         @trace emb.out
        else:
            x = tok                                         # RoPE/ALiBi act inside attention @trace emb.out
        mask = build_mask(cfg, T)                           # (T, T) bool       @trace mask
        rope = rope_angles(T, cfg.d_head, cfg.rope_base) if cfg.pos_encoding == "rope" else None
        alibi = None
        if cfg.pos_encoding == "alibi":
            alibi = alibi_bias(cfg, T)                      # (H, T, T)         @trace alibi
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

    @torch.no_grad()
    def generate(self, idx, n_new: int, temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None):
        for _ in range(n_new):
            idx_cond = idx[:, -self.cfg.block_size:]        # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature         # only the last position predicts
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None:                           # nucleus: smallest set with mass >= p
                sorted_logits, order = torch.sort(logits, descending=True)
                cum = F.softmax(sorted_logits, dim=-1).cumsum(-1)
                drop = cum - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

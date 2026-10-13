"""Executable PyTorch reference for the corresponding Atlas lesson."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model import Config, GPT, Block, Attention, MLP, make_norm, sinusoidal_table

def attention_snapshot(module, query, memory, allowed):
    """Q comes from query; K and V come from memory. Lengths need not match."""
    B, T, d = query.shape
    S = memory.shape[1]
    H, dh = module.cfg.n_head, module.cfg.d_head
    q, k, v = module.q_proj(query), module.k_proj(memory), module.v_proj(memory)
    qh = q.view(B, T, H, dh).transpose(1, 2)
    kh = k.view(B, S, H, dh).transpose(1, 2)
    vh = v.view(B, S, H, dh).transpose(1, 2)
    scores = (qh @ kh.transpose(-2, -1) / math.sqrt(dh)).masked_fill(~allowed, float("-inf"))
    probs = scores.softmax(-1)
    ctx = probs @ vh
    out = module.o_proj(ctx.transpose(1, 2).contiguous().view(B, T, d))
    return out, dict(q=q, k=k, v=v, q_heads=qh, k_heads=kh, v_heads=vh,
                     masked=scores, probs=probs, ctx=ctx, out=out)


class TinyEncoderDecoder(nn.Module):
    """One encoder block and one decoder block; shared embeddings/unembedding.

    2017-style post-LayerNorm, ReLU and sinusoidal positions; no dropout.
    Random weights teach the computation, not translation quality.
    """
    def __init__(self):
        super().__init__()
        self.cfg = Config(vocab_size=6, block_size=8, n_layer=1, d_model=8, n_head=2,
                          n_kv_head=2, d_ff=16, norm_placement="post",
                          pos_encoding="sinusoidal", activation="relu", tie_weights=True)
        self.embed = nn.Embedding(6, 8)
        self.encoder = Block(self.cfg, 0)
        self.self_attn = Attention(self.cfg, 0)
        self.cross_attn = Attention(self.cfg, 0)
        self.norm1, self.norm2, self.norm3 = (make_norm(self.cfg) for _ in range(3))
        self.ff = MLP(self.cfg, 0)
        self.apply(GPT._init_weights)

    def forward(self, source, target, padding=5):
        cfg = self.cfg
        S, T = source.shape[1], target.shape[1]
        if not 1 <= S <= cfg.block_size or not 1 <= T <= cfg.block_size or not (source != padding).any(-1).all():
            raise ValueError("invalid source/target sequence")
        trace = {}
        def rec(key, value):
            trace[key] = value
            return value
        def emb(ids, name):
            pe = sinusoidal_table(ids.shape[1], cfg.d_model, device=ids.device, dtype=self.embed.weight.dtype)
            return rec(name, self.embed(ids) * math.sqrt(cfg.d_model) + pe)
        def attend(module, query, memory, allowed, prefix):
            out, values = attention_snapshot(module, query, memory, allowed)
            trace.update({prefix + "." + key: value for key, value in values.items()})
            return out
        def mlp(module, x, prefix):
            up = rec(prefix + ".up", module.up_proj(x))
            act = rec(prefix + ".act", F.relu(up))
            return rec(prefix + ".out", module.down_proj(act))
        src_mask = (source != padding)[:, None, None, :].expand(-1, 1, S, -1)
        causal = torch.ones(T, T, device=target.device, dtype=torch.bool).tril()
        tgt_mask = causal & (target != padding)[:, None, None, :]
        if not tgt_mask[:, :, 0].any(-1).all():
            raise ValueError("first decoder token must be visible")
        cross_mask = (source != padding)[:, None, None, :].expand(-1, 1, T, -1)
        rec("source.mask", src_mask[:, 0]); rec("target.mask", tgt_mask[:, 0]); rec("cross.mask", cross_mask[:, 0])
        enc = emb(source, "source.embed")
        enc = rec("encoder.mid", self.encoder.ln1(enc + attend(self.encoder.attn, enc, enc, src_mask, "encoder.attn")))
        memory = rec("encoder.memory", self.encoder.ln2(enc + mlp(self.encoder.mlp, enc, "encoder.mlp")))
        dec = emb(target, "target.embed")
        dec = rec("decoder.self_out", self.norm1(dec + attend(self.self_attn, dec, dec, tgt_mask, "decoder.self")))
        dec = rec("decoder.cross_out", self.norm2(dec + attend(self.cross_attn, dec, memory, cross_mask, "decoder.cross")))
        dec = rec("decoder.out", self.norm3(dec + mlp(self.ff, dec, "decoder.mlp")))
        logits = rec("logits", F.linear(dec, self.embed.weight))
        rec("probs", logits.softmax(-1))
        return logits, trace



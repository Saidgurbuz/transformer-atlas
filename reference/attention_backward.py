"""Manual attention VJP; batch/head axes broadcast as in matmul.

No dropout. Masks are constant, Boolean True means allowed, and each row must
have a visible key. Q/K have width dh; V may have a different width dv.
"""
import math
import torch

def attention_forward(q, k, v, allowed, scale=None, bias=None):
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    if not math.isfinite(scale) or not allowed.any(-1).all():
        raise ValueError('finite scale and at least one visible key per query required')
    z = (q @ k.transpose(-2, -1)) * scale
    if bias is not None:
        z = z + bias
    a = z.masked_fill(~allowed, float('-inf')).softmax(-1)
    return a @ v, (q, k, v, allowed, scale, a)

def attention_backward(g, cache):
    """G is the gradient at A @ V, before concatenation/output projection."""
    q, k, v, allowed, scale, a = cache
    dv = a.transpose(-2, -1) @ g
    da = g @ v.transpose(-2, -1)
    dz = a * (da - (a * da).sum(-1, keepdim=True))
    dz = dz.masked_fill(~allowed, 0.)
    dq = (dz @ k) * scale
    dk = (dz.transpose(-2, -1) @ q) * scale
    # An additive score bias receives dz, reduced across broadcast axes if needed.
    return dict(q=dq, k=dk, v=dv, scores=dz, probabilities=da)

def shared_kv_backward(repeated_gradient, kv_heads):
    """Undo repeat_interleave of contiguous KV groups; sum all query-head uses."""
    h, s, d = repeated_gradient.shape[-3:]
    if not 1 <= kv_heads <= h or h % kv_heads:
        raise ValueError('KV heads must divide query heads')
    return repeated_gradient.reshape(*repeated_gradient.shape[:-3], kv_heads, h // kv_heads, s, d).sum(-3)

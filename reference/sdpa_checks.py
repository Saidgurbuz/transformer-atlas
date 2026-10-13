"""Readable PyTorch attention checks. CPU validation, not a GPU benchmark."""
import math
import torch
from torch.nn import functional as F

def manual_attention(q, k, v, allowed, bias=None):
    """True means allowed. Deliberately reject all-masked rows."""
    if not allowed.any(-1).all():
        raise ValueError('no probability distribution for an all-masked row')
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if bias is not None:
        scores = scores + bias
    return scores.masked_fill(~allowed, float('-inf')).softmax(-1) @ v

def compare_sdpa(q, k, v, allowed, bias=None):
    """Use explicit masks; rectangular is_causal=True has upper-left alignment."""
    mask = allowed if bias is None else bias.masked_fill(~allowed, float('-inf'))
    expected = manual_attention(q, k, v, allowed, bias)
    got = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
    # Kernel choice, operation order and dtype can change rounding.
    return expected, got

def cached_query_mask(query_position, key_count, device=None):
    """Keys start at absolute position zero; not an upper-left 1×S causal mask."""
    return (torch.arange(key_count, device=device) <= query_position).reshape(1, -1)

def grouped_attention(q, k, v, allowed):
    """Portable explicit repetition and the SDPA GQA path represent the same groups."""
    groups = q.shape[-3] // k.shape[-3]
    explicit = F.scaled_dot_product_attention(q, k.repeat_interleave(groups, dim=-3), v.repeat_interleave(groups, dim=-3), attn_mask=allowed, dropout_p=0.)
    grouped = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0., enable_gqa=True)
    return explicit, grouped

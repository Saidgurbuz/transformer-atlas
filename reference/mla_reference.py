"""Bias-free MLA algebra: dense reconstruction versus absorbed projections.

Inputs are already projected: q_content/q_rotary [B,H,T,dh/dr], latent
[B,S,dc], shared rotated key [B,S,dr], up_k/up_v [H,dh/dv,dc].
No dropout, latent/query normalization or query-compression module is included.
This checks one MLA parameterization, not arbitrary MHA compression or a model.
"""
import math
import torch

def _probabilities(content_scores, q_rotary, k_rotary, allowed, content_width):
    if not allowed.any(-1).all():
        raise ValueError('each query needs a visible key')
    positional_scores = q_rotary @ k_rotary[:, None].transpose(-2, -1)
    scores = (content_scores + positional_scores) / math.sqrt(content_width + q_rotary.shape[-1])
    return scores.masked_fill(~allowed, float('-inf')).softmax(-1)

def mla_dense(q_content, q_rotary, latent, k_rotary, up_k, up_v, out_weight, allowed):
    """Reconstruct headwise content keys/values, then use ordinary attention."""
    k = torch.einsum('bsc,hkc->bhsk', latent, up_k)
    v = torch.einsum('bsc,hvc->bhsv', latent, up_v)
    a = _probabilities(q_content @ k.transpose(-2, -1), q_rotary, k_rotary, allowed, q_content.shape[-1])
    heads = a @ v
    merged = heads.transpose(1, 2).contiguous().flatten(-2)
    return merged @ out_weight.T

def mla_absorbed(q_content, q_rotary, latent, k_rotary, up_k, up_v, out_weight, allowed):
    """Read the latent cache directly, avoiding reconstructed K/V for all keys."""
    latent_query = torch.einsum('bhtk,hkc->bhtc', q_content, up_k)
    content_scores = latent_query @ latent[:, None].transpose(-2, -1)
    a = _probabilities(content_scores, q_rotary, k_rotary, allowed, q_content.shape[-1])
    latent_sum = a @ latent[:, None]
    h, dv, _ = up_v.shape
    split_out = out_weight.reshape(out_weight.shape[0], h, dv)
    combined = torch.einsum('dhv,hvc->dhc', split_out, up_v)
    return torch.einsum('bhtc,dhc->btd', latent_sum, combined)

def mla_cache_bytes(batch, layers, tokens, latent_width, rotary_width, bytes_per_scalar):
    """Only cached latent and shared rotary key; excludes weights/allocator overhead."""
    values = (batch, layers, tokens, latent_width, rotary_width, bytes_per_scalar)
    if not all(type(v) is int for v in values) or min(batch, layers, latent_width, bytes_per_scalar) < 1 or min(tokens, rotary_width) < 0:
        raise ValueError('invalid cache dimensions')
    return batch * layers * tokens * (latent_width + rotary_width) * bytes_per_scalar

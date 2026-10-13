"""Isolated research primitives, not active variants of Atlas's main model.

These small functions explain tensor algebra; they are not checkpoint loaders,
full Gemma/DeepSeek models or GPU kernels. Batch/head axes follow PyTorch matmul.
"""
import math
import torch
from torch.nn import functional as F

def qk_l2_scores(q, k, gain=1., eps=1e-12):
    """Henry et al. style: unit L2 directions and a scalar score gain."""
    q_unit = F.normalize(q, p=2, dim=-1, eps=eps)
    k_unit = F.normalize(k, p=2, dim=-1, eps=eps)
    return gain * (q_unit @ k_unit.transpose(-2, -1))

def qk_rms_scores(q, k, gain_q, gain_k, eps=1e-6):
    """Illustrate featurewise RMS gains; recipe-specific scale remains explicit."""
    q_norm = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps) * gain_q
    k_norm = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + eps) * gain_k
    return q_norm @ k_norm.transpose(-2, -1) / math.sqrt(q.shape[-1])

def softcap_scores(scores, cap):
    """Smoothly bound scalar scores; differentiate this function, not hard clipping."""
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('cap must be finite and positive')
    return cap * torch.tanh(scores / cap)

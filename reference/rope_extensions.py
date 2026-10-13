"""Fixed-frequency RoPE/PI/YaRN algebra, separate from the main decoder.

The paper uses a rotation-count ramp; the authors' implementation uses a ramp
in pair index. Both are explicit here. No automatic cache retuning or model
fine-tuning is performed. Positions are absolute and pairs are interleaved.
"""
import math
import torch

def rope_frequencies(width, base=10000., device=None):
    if width < 2 or width % 2 or base <= 1:
        raise ValueError('even positive rotary width and base > 1 required')
    pair = torch.arange(width // 2, dtype=torch.float64, device=device)
    return base ** (-2 * pair / width)

def yarn_frequencies(width, original_context, scale, base=10000., beta_fast=32., beta_slow=1., ramp='index'):
    """Authors' default index blend or paper rotation-count blend; default factors only."""
    if original_context < 1 or not math.isfinite(scale) or scale < 1 or not 0 < beta_slow < beta_fast:
        raise ValueError('positive context, scale >= 1 and 0 < slow < fast required')
    theta = rope_frequencies(width, base)
    if ramp == 'rotation-count':
        rotations = original_context * theta / (2 * math.pi)
        keep_original = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)
    elif ramp == 'index':
        def boundary(rotations):
            return width * math.log(original_context / (2 * math.pi * rotations)) / (2 * math.log(base))
        low = max(math.floor(boundary(beta_fast)), 0)
        high = min(math.ceil(boundary(beta_slow)), width - 1)
        if high < low:
            raise ValueError('context/rotation thresholds give an invalid correction interval')
        span = .001 if high == low else high - low
        pair = torch.arange(width // 2, dtype=theta.dtype)
        keep_original = 1 - ((pair - low) / span).clamp(0, 1)
    else:
        raise ValueError('choose index or rotation-count ramp')
    frequencies = theta * (keep_original + (1 - keep_original) / scale)
    amplitude = 1. if scale == 1 else 1 + .1 * math.log(scale)
    return frequencies, amplitude

def rotate_interleaved(x, positions, frequencies, amplitude=1.):
    """Fixed angles; scaled sine/cosine magnitude also scales the output norm."""
    if x.shape[-1] != 2 * frequencies.numel() or x.shape[-2] != positions.numel():
        raise ValueError('rotary width and position count must match')
    # Keep large absolute positions/frequencies in reference precision until
    # sine/cosine evaluation; bf16 indices can otherwise merge adjacent tokens.
    angles = positions.to(device=x.device, dtype=torch.float64)[:, None] * frequencies.to(device=x.device, dtype=torch.float64)[None, :]
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    result = torch.stack((pairs[..., 0] * c - pairs[..., 1] * s, pairs[..., 0] * s + pairs[..., 1] * c), -1)
    return amplitude * result.flatten(-2)

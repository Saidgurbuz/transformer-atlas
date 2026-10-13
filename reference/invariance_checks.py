"""Small algebra checks and counterexamples, not trained-model benchmarks."""
import math
import torch

def head_basis_change(q, k, v, out_weight, change):
    """Bias-free, no QK norm or rotary positions. Invertible changes preserve QK/OV."""
    inverse = torch.linalg.inv(change)
    scale = 1 / math.sqrt(q.shape[-1])
    a = (q @ k.T * scale).softmax(-1)
    original = (a @ v) @ out_weight.T
    q_new, k_new = q @ change.T, k @ inverse
    v_new, out_new = v @ change.T, out_weight @ inverse
    a_new = (q_new @ k_new.T * scale).softmax(-1)
    return original, (a_new @ v_new) @ out_new.T

def attention_with_mask(x, wq, wk, wv, allowed):
    q, k, v = x @ wq.T, x @ wk.T, x @ wv.T
    scores = (q @ k.T) / math.sqrt(q.shape[-1])
    return scores.masked_fill(~allowed, float('-inf')).softmax(-1) @ v

def permutation_check(x, wq, wk, wv, allowed, permutation):
    """Reorder BOTH mask axes. Holding a causal mask fixed is a counterexample."""
    original = attention_with_mask(x, wq, wk, wv, allowed)
    reordered_mask = allowed[permutation][:, permutation]
    equivariant = attention_with_mask(x[permutation], wq, wk, wv, reordered_mask)
    fixed_mask = attention_with_mask(x[permutation], wq, wk, wv, allowed)
    return original[permutation], equivariant, fixed_mask

def cache_equivalence(model, ids):
    """Eval mode, fixed config/weights, causal or causal-sliding attention, within block_size."""
    if model.training:
        raise ValueError('use eval mode for deterministic comparison')
    with torch.no_grad():
        full, _ = model(ids)
        cache = model.create_cache()
        cached = torch.cat([model.decode_step(ids[:, t:t+1], cache) for t in range(ids.shape[1])], dim=1)
    return full, cached

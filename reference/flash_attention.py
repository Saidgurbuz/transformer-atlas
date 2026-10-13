"""Executable PyTorch reference for the corresponding Atlas lesson."""
import torch

def online_attention(scores, values, tile_size=4):
    """One query's exact softmax-value product, using streamed key tiles.

    scores: [S], including -inf for masked keys; values: [S, dv].
    U is unnormalised. This explains the recurrence, not a fused GPU kernel.
    """
    m = scores.new_tensor(float("-inf"))
    ell = scores.new_tensor(0.0)
    U = values.new_zeros(values.shape[-1])
    for start in range(0, len(scores), tile_size):
        s = scores[start:start + tile_size]
        m_new = torch.maximum(m, s.max())
        if torch.isneginf(m_new):                # a wholly masked leading tile
            continue
        alpha = torch.exp(m - m_new)             # convert the previous accumulator's scale
        p = torch.exp(s - m_new)                 # masked entries become zero
        ell = alpha * ell + p.sum()
        U = alpha * U + p @ values[start:start + tile_size]
        m = m_new
    if ell == 0:
        raise ValueError("attention requires at least one visible key")
    return U / ell



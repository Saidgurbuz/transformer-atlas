"""Executable PyTorch reference for the corresponding Atlas lesson."""
import torch

def tensor_operations():
    B, T, H, dh = 2, 3, 2, 3
    arange = torch.arange(B * T * H * dh, dtype=torch.float64)
    x = arange.view(B, T, H * dh)
    split = x.view(B, T, H, dh)                  # split the feature axis; same storage
    heads = split.transpose(1, 2)               # [B, H, T, dh]; a strided view
    reshaped = heads.reshape(B, H * T, dh)      # these H,T strides cannot be merged: copies
    copied = heads.contiguous()                 # allocate packed storage in this order
    merged = heads.transpose(1, 2).contiguous().view(B, T, H * dh)
    concatenated = torch.cat([heads[:, h] for h in range(H)], dim=-1)
    positions = torch.arange(T)                  # integer position indices
    position_row = positions.unsqueeze(0)       # [1, T], broadcastable across a batch
    position_flat = position_row.squeeze(0)     # remove a singleton axis
    scores = torch.einsum("bhtd,bhsd->bhts", heads, heads)
    return dict(arange=arange, x=x, split=split, heads=heads, reshaped=reshaped, copied=copied, merged=merged,
                concatenated=concatenated, positions=positions, positionRow=position_row,
                positionFlat=position_flat, scores=scores)



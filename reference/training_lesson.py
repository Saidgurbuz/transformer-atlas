"""Executable PyTorch reference for the corresponding Atlas lesson."""
import torch
from model import Config

def architecture_profiles():
    """Tiny structural profiles, not production checkpoint configurations."""
    common = dict(vocab_size=5, block_size=8, n_layer=1, d_model=8, n_head=2)
    return {
        "gpt": Config(**common, n_kv_head=2, d_ff=32, tie_weights=True),
        "llama": Config(**common, n_kv_head=1, d_ff=24, pos_encoding="rope",
                        norm="rmsnorm", activation="swiglu", bias=False, tie_weights=False),
    }


def make_optimizer(model, kind="adamw"):
    if kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.01)  # no decay in this lesson's SGD
    if kind != "adamw":
        raise ValueError("unknown optimizer")
    groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 0.01},
              {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=0.01, betas=(0.9, 0.999), eps=1e-8, foreach=False)


def training_step(model, optimizer):
    inputs = torch.tensor([[1, 2, 3, 4]])         # c a t .
    targets = torch.tensor([[2, 3, 4, -1]])       # a t . ; last input has no next-token target
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(inputs, targets)
    logits.retain_grad()
    loss.backward()                             # chain rule all the way to the embeddings
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    logits_gradient = logits.grad.detach().clone()
    before = loss.item()
    optimizer.step()                            # shared embeddings get one combined update
    with torch.no_grad():
        after_logits, after_loss = model(inputs, targets)
    return dict(before=before, after=after_loss.item(), gradients=gradients,
                logitsGradient=logits_gradient[0], afterLogits=after_logits[0])



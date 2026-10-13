"""Exact vector decompositions; semantic or causal interpretations need evidence."""
import torch

def attention_contributions(probabilities, values, head_output_weight):
    """Single head, no dropout/bias: [T,S,D] contributions after this head's W_O."""
    writes = values @ head_output_weight.T
    return probabilities[..., None] * writes[None, ...]

def mlp_contributions(activation, down_weight):
    """[T,F,D] terms; sum units and then add down-projection bias separately."""
    return activation[..., None] * down_weight.T[None, ...]

def direct_logit_projection(write, unembedding):
    """A linear projection, not a full-model counterfactual through subsequent norms."""
    return write @ unembedding.T

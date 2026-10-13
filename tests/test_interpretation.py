import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reference'))
from interpretation_checks import attention_contributions, mlp_contributions, direct_logit_projection
torch.set_num_threads(1)

class Interpretation(unittest.TestCase):
    def test_vector_decomposition_and_weight_counterexample(self):
        a = torch.tensor([[.9, .1]], dtype=torch.float64)
        v = torch.tensor([[.01, 0.], [10., 0.]], dtype=torch.float64)
        wo = torch.eye(2, dtype=torch.float64)
        terms = attention_contributions(a, v, wo)
        torch.testing.assert_close(terms.sum(1), a @ v @ wo.T)
        self.assertLess(terms[0, 0].norm(), terms[0, 1].norm())

    def test_mlp_terms_and_bias(self):
        rng = torch.Generator().manual_seed(4)
        act = torch.randn(3, 8, generator=rng, dtype=torch.float64)
        down = torch.randn(4, 8, generator=rng, dtype=torch.float64)
        bias = torch.randn(4, generator=rng, dtype=torch.float64)
        terms = mlp_contributions(act, down)
        torch.testing.assert_close(terms.sum(1) + bias, act @ down.T + bias)

    def test_direct_lens_is_not_a_norm_counterfactual(self):
        x = torch.tensor([[1., 3., -2., 4.]], dtype=torch.float64)
        write = torch.tensor([[.1, -.2, .4, -.3]], dtype=torch.float64)
        unembed = torch.eye(4, dtype=torch.float64)
        raw = direct_logit_projection(write, unembed)
        effect = (torch.nn.functional.layer_norm(x + write, (4,)) - torch.nn.functional.layer_norm(x, (4,))) @ unembed.T
        self.assertGreater((raw - effect).norm().item(), .1)

if __name__ == '__main__':
    unittest.main()

import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reference'))
from invariance_checks import head_basis_change, permutation_check, cache_equivalence
from model import Config, GPT
torch.set_num_threads(1)

class Invariances(unittest.TestCase):
    def test_softmax_shift_and_convex_value_range(self):
        z = torch.tensor([[1., 3., -2.], [.5, -1., 2.]], dtype=torch.float64, requires_grad=True)
        a = z.softmax(-1)
        torch.testing.assert_close(a, (z + torch.tensor([[100.], [-100.]], dtype=torch.float64)).softmax(-1))
        da = torch.tensor([[2., -1., 4.], [1., 2., -2.]], dtype=torch.float64)
        dz, = torch.autograd.grad(a, z, da)
        torch.testing.assert_close(dz.sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
        v = torch.tensor([[1., -3.], [4., 2.], [-2., 5.]], dtype=torch.float64)
        out = a @ v
        self.assertTrue((out >= v.min(0).values).all() and (out <= v.max(0).values).all())

    def test_bias_free_head_change(self):
        rng = torch.Generator().manual_seed(42)
        q, k, v, wo = [torch.randn(n, 4, generator=rng, dtype=torch.float64) for n in (3, 5, 5, 8)]
        change = torch.diag(torch.tensor([2., .5, 1.5, .8], dtype=torch.float64))
        a, b = head_basis_change(q, k, v, wo, change)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        # A general change of basis does not commute with position-dependent RoPE.
        angle = .7
        rotation = torch.eye(4, dtype=torch.float64)
        rotation[:2, :2] = torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))], [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]])
        self.assertGreater((change.T @ rotation @ torch.linalg.inv(change).T - rotation).norm().item(), .1)

    def test_permutation_and_fixed_causal_counterexample(self):
        rng = torch.Generator().manual_seed(3)
        x, wq, wk, wv = [torch.randn(4, 4, generator=rng, dtype=torch.float64) for _ in range(4)]
        permutation = torch.tensor([3, 1, 0, 2])
        for mask in (torch.ones(4, 4, dtype=torch.bool), torch.ones(4, 4, dtype=torch.bool).tril()):
            a, reordered, fixed = permutation_check(x, wq, wk, wv, mask, permutation)
            torch.testing.assert_close(a, reordered, atol=1e-12, rtol=1e-12)
            if mask[0, 1]:
                torch.testing.assert_close(a, fixed)
            else:
                self.assertGreater((a - fixed).abs().max().item(), .01)

    def test_cache_conditions(self):
        for mask in ('causal', 'sliding'):
            for pos in ('learned', 'rope', 'alibi', 'sinusoidal', 'none'):
                model = GPT(Config(vocab_size=6, block_size=8, n_layer=2, d_model=8, n_head=2, n_kv_head=1, d_ff=16, mask=mask, pos_encoding=pos)).double().eval()
                a, b = cache_equivalence(model, torch.tensor([[1, 2, 3, 2, 4]]))
                torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
        model = GPT(Config(mask='none')).eval()
        with self.assertRaises(ValueError):
            cache_equivalence(model, torch.tensor([[1, 2]]))
        with self.assertRaises(ValueError):
            cache_equivalence(model.train(), torch.tensor([[1, 2]]))

if __name__ == '__main__':
    unittest.main()

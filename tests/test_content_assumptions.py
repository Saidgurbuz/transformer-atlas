"""Counterexamples and exact identities behind the explanatory corrections."""
import unittest
import torch
from torch.nn import functional as F
torch.set_num_threads(1)

class Assumptions(unittest.TestCase):
    def test_epsilon_changes_radial_invariance(self):
        x = torch.tensor([-2., -.5, 1., 3.], dtype=torch.float64)
        centered = x - x.mean()
        variance = centered.square().mean()
        for eps in (1e-5, .1, 1.):
            y = F.layer_norm(x, (4,), eps=eps)
            torch.testing.assert_close(y.square().mean(), variance / (variance + eps))
            j = torch.autograd.functional.jacobian(lambda t: F.layer_norm(t, (4,), eps=eps), x)
            torch.testing.assert_close(j @ torch.ones_like(x), torch.zeros_like(x), atol=1e-12, rtol=0)
            torch.testing.assert_close(j @ centered, eps / (variance + eps).pow(1.5) * centered)
            self.assertGreater((j @ centered).norm().item(), 0.)

    def test_masked_edge_is_not_a_frozen_key(self):
        q = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64, requires_grad=True)
        k = torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([[1., 0.], [0., 3.]], dtype=torch.float64, requires_grad=True)
        scores = q @ k.T
        scores.retain_grad()
        allowed = torch.tensor([[True, False], [True, True]])
        a = scores.masked_fill(~allowed, float('-inf')).softmax(-1)
        (a @ v).square().sum().backward()
        self.assertEqual(a[0, 1].item(), 0.)
        self.assertEqual(scores.grad[0, 1].item(), 0.)
        self.assertGreater(k.grad[1].norm().item(), 0.)
        self.assertGreater(v.grad[1].norm().item(), 0.)

    def test_large_common_logits_do_not_force_saturation(self):
        scores = torch.tensor([10000., 10000.], dtype=torch.float64)
        torch.testing.assert_close(scores.softmax(0), torch.tensor([.5, .5], dtype=torch.float64))

if __name__ == '__main__':
    unittest.main()

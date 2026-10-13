"""Counterexamples and exact identities behind the explanatory corrections."""
import unittest
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reference'))
from numpy_transformer import gelu_forward, gelu_backward, rmsnorm_forward, rmsnorm_backward
torch.set_num_threads(1)

class Assumptions(unittest.TestCase):
    def test_residual_jacobians_have_order(self):
        a = torch.tensor([[0., .2], [-.1, 0.]], dtype=torch.float64)
        b = torch.tensor([[0., 0.], [.3, .1]], dtype=torch.float64)
        x = torch.tensor([1., 2.], dtype=torch.float64)
        def layers(z):
            y = z + a @ z
            return y + b @ y
        jacobian = torch.autograd.functional.jacobian(layers, x)
        eye = torch.eye(2, dtype=torch.float64)
        torch.testing.assert_close(jacobian, (eye + b) @ (eye + a))
        self.assertGreater(((eye + b) @ (eye + a) - (eye + a) @ (eye + b)).norm().item(), .01)

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

    def test_implemented_gelu_derivative(self):
        x = torch.linspace(-5, 5, 31, dtype=torch.float64, requires_grad=True)
        wanted, = torch.autograd.grad(F.gelu(x, approximate='tanh').sum(), x)
        _, cache = gelu_forward(x.detach().numpy())
        got = gelu_backward(torch.ones_like(x).numpy(), cache)
        torch.testing.assert_close(torch.from_numpy(got), wanted, atol=1e-12, rtol=1e-12)

    def test_rmsnorm_backward_including_zero(self):
        x = torch.tensor([[0., 0., 0., 0.], [1., -2., .5, 3.]], dtype=torch.float64, requires_grad=True)
        gamma = torch.tensor([1., .5, 2., -1.], dtype=torch.float64, requires_grad=True)
        dy = torch.tensor([[1., 2., -1., .2], [-2., .4, 1., 3.]], dtype=torch.float64)
        for eps in (1e-5, .1):
            y = F.rms_norm(x, (4,), gamma, eps=eps)
            dx, dg = torch.autograd.grad(y, (x, gamma), dy)
            _, cache = rmsnorm_forward(x.detach().numpy(), gamma.detach().numpy(), eps)
            got = rmsnorm_backward(dy.numpy(), cache)
            for a, b in zip(got, (dx, dg)):
                torch.testing.assert_close(torch.from_numpy(a), b, atol=1e-11, rtol=1e-11)

if __name__ == '__main__':
    unittest.main()

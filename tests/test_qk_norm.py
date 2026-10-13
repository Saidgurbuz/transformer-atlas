import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reference'))
from modern_attention import qk_l2_scores, qk_rms_scores, softcap_scores
torch.set_num_threads(1)

class QKNorm(unittest.TestCase):
    def test_l2_bounds_zero_and_positive_rescaling(self):
        rng = torch.Generator().manual_seed(9)
        q = torch.randn(2, 3, 8, generator=rng, dtype=torch.float64)
        k = torch.randn(2, 5, 8, generator=rng, dtype=torch.float64)
        scores = qk_l2_scores(q, k, gain=3.)
        self.assertLessEqual(scores.abs().max().item(), 3.)
        torch.testing.assert_close(scores, qk_l2_scores(q * 10., k * .2, gain=3.), atol=1e-12, rtol=1e-12)
        self.assertTrue((qk_l2_scores(torch.zeros_like(q), k) == 0).all())

    def test_rms_matches_pytorch_rmsnorm(self):
        rng = torch.Generator().manual_seed(4)
        q = torch.randn(2, 3, 8, generator=rng, dtype=torch.float64)
        k = torch.randn(2, 5, 8, generator=rng, dtype=torch.float64)
        gains = torch.linspace(.5, 1.5, 8, dtype=torch.float64)
        wanted = F.rms_norm(q, (8,), gains, eps=1e-6) @ F.rms_norm(k, (8,), gains.flip(0), eps=1e-6).transpose(-2, -1) / 8**.5
        torch.testing.assert_close(qk_rms_scores(q, k, gains, gains.flip(0)), wanted, atol=1e-12, rtol=1e-12)

    def test_softcap_bounds_and_derivative(self):
        scores = torch.tensor([-1e4, -3., 0., 1., 1e4], dtype=torch.float64, requires_grad=True)
        y = softcap_scores(scores, 4.)
        self.assertTrue((y.abs() <= 4.).all())
        gradient, = torch.autograd.grad(y.sum(), scores)
        torch.testing.assert_close(gradient, 1 - torch.tanh(scores / 4.).square())
        for cap in (0., -1., float('inf')):
            with self.assertRaises(ValueError): softcap_scores(scores, cap)

if __name__ == '__main__':
    unittest.main()

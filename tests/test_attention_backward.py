import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reference'))
from attention_backward import attention_forward, attention_backward, shared_kv_backward
torch.set_num_threads(1)

class AttentionBackward(unittest.TestCase):
    def test_vjp_against_sdpa_autograd(self):
        rng = torch.Generator().manual_seed(41)
        for t, s in ((1, 1), (3, 5), (5, 3)):
            for scale in (None, 1.):
                for causal in (True, False):
                    q = torch.randn(2, 4, t, 6, generator=rng, dtype=torch.float64, requires_grad=True)
                    k = torch.randn(2, 4, s, 6, generator=rng, dtype=torch.float64, requires_grad=True)
                    v = torch.randn(2, 4, s, 3, generator=rng, dtype=torch.float64, requires_grad=True)
                    allowed = torch.ones(t, s, dtype=torch.bool).tril() if causal else torch.ones(t, s, dtype=torch.bool)
                    bias = torch.randn(t, s, generator=rng, dtype=torch.float64) * .1
                    out, cache = attention_forward(q, k, v, allowed, scale, bias)
                    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.masked_fill(~allowed, float('-inf')), dropout_p=0., scale=scale)
                    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
                    g = torch.randn(out.shape, generator=rng, dtype=torch.float64)
                    wanted = torch.autograd.grad(expected, (q, k, v), g)
                    got = attention_backward(g, cache)
                    for key, gradient in zip(('q', 'k', 'v'), wanted):
                        torch.testing.assert_close(got[key], gradient, atol=1e-11, rtol=1e-11)
                    self.assertTrue((got['scores'].masked_select(~allowed) == 0).all())

    def test_shared_heads_sum_their_gradients(self):
        rng = torch.Generator().manual_seed(7)
        for kv_heads in (1, 2, 4):
            x = torch.randn(2, kv_heads, 5, 3, generator=rng, dtype=torch.float64, requires_grad=True)
            expanded = x.repeat_interleave(4 // kv_heads, dim=-3)
            g = torch.randn(expanded.shape, generator=rng, dtype=torch.float64)
            want, = torch.autograd.grad(expanded, x, g)
            torch.testing.assert_close(shared_kv_backward(g, kv_heads), want)

    def test_broadcast_input_gradients_reduce_to_input_shapes(self):
        rng = torch.Generator().manual_seed(19)
        q = torch.randn(2, 4, 3, 6, generator=rng, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 1, 5, 6, generator=rng, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 1, 5, 3, generator=rng, dtype=torch.float64, requires_grad=True)
        allowed = torch.ones(3, 5, dtype=torch.bool)
        out, cache = attention_forward(q, k, v, allowed)
        oracle = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.)
        g = torch.randn(out.shape, generator=rng, dtype=torch.float64)
        expected = torch.autograd.grad(oracle, (q, k, v), g)
        got = attention_backward(g, cache)
        for key, wanted in zip(('q', 'k', 'v'), expected):
            self.assertEqual(got[key].shape, wanted.shape)
            torch.testing.assert_close(got[key], wanted, atol=1e-11, rtol=1e-11)

    def test_rejects_all_masked_row(self):
        x = torch.ones(1, 2, dtype=torch.float64)
        with self.assertRaises(ValueError):
            attention_forward(x, x, x, torch.zeros(1, 1, dtype=torch.bool))

if __name__ == '__main__':
    unittest.main()

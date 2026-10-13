import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reference'))
from sdpa_checks import compare_sdpa, cached_query_mask, grouped_attention, manual_attention
torch.set_num_threads(1)

class SDPA(unittest.TestCase):
    def test_manual_vs_library_across_shapes_and_dtypes(self):
        rng = torch.Generator().manual_seed(17)
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            for t, s in ((1, 7), (3, 5), (5, 3)):
                q = torch.randn(2, 4, t, 8, generator=rng).to(dtype)
                k = torch.randn(2, 4, s, 8, generator=rng).to(dtype)
                v = torch.randn(2, 4, s, 8, generator=rng).to(dtype)
                mask = torch.ones(t, s, dtype=torch.bool).tril()
                for bias in (None, torch.randn(t, s, generator=rng).to(dtype) * .1):
                    a, b = compare_sdpa(q, k, v, mask, bias)
                    tolerance = {torch.float64: 1e-12, torch.float32: 2e-6, torch.bfloat16: .04}[dtype]
                    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)

    def test_offset_mask_and_upper_left_counterexample(self):
        q = torch.zeros(1, 1, 1, 2)
        k = torch.zeros(1, 1, 4, 2)
        v = torch.arange(8.).reshape(1, 1, 4, 2)
        mask = cached_query_mask(3, 4)
        wanted = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        upper_left = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
        torch.testing.assert_close(wanted, v.mean(-2, keepdim=True))
        torch.testing.assert_close(upper_left, v[..., :1, :])
        self.assertFalse(torch.allclose(wanted, upper_left))

    def test_gqa_and_dropout_zero(self):
        q = torch.randn(1, 4, 3, 8, dtype=torch.float64)
        k, v = [torch.randn(1, 2, 3, 8, dtype=torch.float64) for _ in range(2)]
        mask = torch.ones(3, 3, dtype=torch.bool).tril()
        a, b = grouped_attention(q, k, v, mask)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(a, F.scaled_dot_product_attention(q, k.repeat_interleave(2, -3), v.repeat_interleave(2, -3), attn_mask=mask, dropout_p=0.), atol=0, rtol=0)

    def test_explicit_all_masked_policy(self):
        x = torch.ones(1, 1, 1, 2, dtype=torch.float64)
        mask = torch.zeros(1, 1, dtype=torch.bool)
        with self.assertRaises(ValueError): manual_attention(x, x, x, mask)
        # Observed for this CPU SDPA path; do not assume arbitrary kernels share a policy.
        y = F.scaled_dot_product_attention(x, x, x, attn_mask=mask, dropout_p=0.)
        torch.testing.assert_close(y, torch.zeros_like(y))

if __name__ == '__main__':
    unittest.main()

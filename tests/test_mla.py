import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reference'))
from mla_reference import mla_dense, mla_absorbed, mla_cache_bytes
torch.set_num_threads(1)

class MLA(unittest.TestCase):
    def test_dense_absorbed_outputs_and_gradients(self):
        rng = torch.Generator().manual_seed(52)
        for heads in (2, 4):
            for t, s in ((1, 1), (1, 7), (3, 5)):
                shapes = [(2, heads, t, 4), (2, heads, t, 2), (2, s, 3), (2, s, 2), (heads, 4, 3), (heads, 3, 3), (8, heads * 3)]
                inputs = [torch.randn(shape, generator=rng, dtype=torch.float64, requires_grad=True) * .2 for shape in shapes]
                allowed = torch.ones(t, s, dtype=torch.bool)
                if t > 1: allowed = allowed.tril()
                dense = mla_dense(*inputs, allowed)
                absorbed = mla_absorbed(*inputs, allowed)
                torch.testing.assert_close(dense, absorbed, atol=1e-12, rtol=1e-12)
                upstream = torch.randn(dense.shape, generator=rng, dtype=torch.float64)
                a = torch.autograd.grad(dense, inputs, upstream)
                b = torch.autograd.grad(absorbed, inputs, upstream)
                for x, y in zip(a, b): torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-11)

    def test_cached_bytes_count_the_two_stored_tensors(self):
        latent = torch.empty(2, 7, 3, dtype=torch.float64)
        rotary = torch.empty(2, 7, 2, dtype=torch.float64)
        actual = latent.numel() * latent.element_size() + rotary.numel() * rotary.element_size()
        self.assertEqual(mla_cache_bytes(2, 1, 7, 3, 2, 8), actual)
        self.assertEqual(mla_cache_bytes(2, 5, 7, 3, 2, 8), actual * 5)
        self.assertEqual(mla_cache_bytes(1, 1, 0, 3, 2, 2), 0)
        with self.assertRaises(ValueError): mla_cache_bytes(1, 1, -1, 3, 2, 2)

if __name__ == '__main__':
    unittest.main()

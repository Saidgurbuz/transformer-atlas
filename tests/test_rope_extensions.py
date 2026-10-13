import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reference'))
from rope_extensions import rope_frequencies, yarn_frequencies, rotate_interleaved
torch.set_num_threads(1)

class RopeExtensions(unittest.TestCase):
    def test_frequency_bounds_and_unit_scale(self):
        for width in (8, 16, 64, 128):
            theta = rope_frequencies(width)
            for mode in ('index', 'rotation-count'):
                unchanged, amplitude = yarn_frequencies(width, 2048, 1., ramp=mode)
                torch.testing.assert_close(unchanged, theta)
                self.assertEqual(amplitude, 1.)
                for scale in (4., 16.):
                    values, amplitude = yarn_frequencies(width, 2048, scale, ramp=mode)
                    self.assertTrue((values >= theta / scale).all() and (values <= theta).all())
                    self.assertGreater(amplitude, 1.)

    def test_relative_rotation_with_fixed_frequencies_and_amplitude(self):
        rng = torch.Generator().manual_seed(33)
        q, k = [torch.randn(4, 8, generator=rng, dtype=torch.float64) for _ in range(2)]
        positions = torch.arange(4) + 23
        for theta, amplitude in ((rope_frequencies(8), 1.), yarn_frequencies(8, 2048, 4.)):
            qr = rotate_interleaved(q, positions, theta, amplitude)
            kr = rotate_interleaved(k, positions, theta, amplitude)
            torch.testing.assert_close(qr.norm(dim=-1), q.norm(dim=-1) * amplitude, atol=1e-12, rtol=1e-12)
            shifted_q = rotate_interleaved(q, positions + 100, theta, amplitude)
            shifted_k = rotate_interleaved(k, positions + 100, theta, amplitude)
            torch.testing.assert_close(qr @ kr.T, shifted_q @ shifted_k.T, atol=1e-11, rtol=1e-11)

    def test_stale_rotated_keys_are_a_counterexample(self):
        rng = torch.Generator().manual_seed(5)
        q = torch.randn(1, 16, generator=rng, dtype=torch.float64)
        k = torch.randn(4, 16, generator=rng, dtype=torch.float64)
        old = rope_frequencies(16)
        new = old / 4
        qr = rotate_interleaved(q, torch.tensor([1004]), new)
        stale = rotate_interleaved(k, torch.arange(1000, 1004), old)
        refreshed = rotate_interleaved(k, torch.arange(1000, 1004), new)
        self.assertGreater((qr @ stale.T - qr @ refreshed.T).abs().max().item(), .1)

    def test_invalid_contracts(self):
        with self.assertRaises(ValueError): rope_frequencies(3)
        with self.assertRaises(ValueError): yarn_frequencies(8, 2048, 0.)
        with self.assertRaises(ValueError): yarn_frequencies(8, 2048, 4., ramp='unknown')

if __name__ == '__main__':
    unittest.main()

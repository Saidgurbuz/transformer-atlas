"""Exercise the actual Python reference, not just previously exported fixtures."""
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "tools"))
from model import Config, GPT
from export_fixtures import run_traced, NO_BATCH_AXIS

torch.set_num_threads(1)


class ReferenceTests(unittest.TestCase):
    def test_current_source_matches_fixtures(self):
        for path in sorted((ROOT / "tests/fixtures").glob("*.json")):
            with self.subTest(fixture=path.stem):
                fx = json.loads(path.read_text())
                model = GPT(Config(**fx["config"])).double()
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        expected = fx["params"][name]
                        parameter.copy_(torch.tensor(expected["data"], dtype=torch.float64).reshape(expected["shape"]))
                ids = torch.tensor([fx["ids"]])
                targets = torch.tensor([fx["targets"]])
                (_, loss), records = run_traced(lambda: model(ids, targets))
                self.assertEqual([k for k, _ in records], fx["order"])
                for key, value in records:
                    if key not in NO_BATCH_AXIS:
                        value = value[0]
                    expected = fx["trace"][key]
                    data = [float("-inf") if v is None else v for v in expected["data"]]
                    wanted = torch.tensor(data, dtype=torch.float64).reshape(expected["shape"])
                    torch.testing.assert_close(value.double(), wanted, atol=1e-9, rtol=1e-9, msg=key)
                loss.backward()
                for name, parameter in model.named_parameters():
                    expected = fx["grads"][name]
                    wanted = torch.tensor(expected["data"], dtype=torch.float64).reshape(expected["shape"])
                    torch.testing.assert_close(parameter.grad, wanted, atol=1e-9, rtol=1e-9, msg=name)

    def test_cache_matches_full_forward_and_preserves_kv_groups(self):
        for dtype in (torch.float32, torch.float64):
            for position in ("learned", "sinusoidal", "rope", "alibi", "none"):
                for mask in ("causal", "sliding"):
                    for placement, kv in (("pre", 4), ("post", 2), ("pre", 1)):
                        with self.subTest(dtype=dtype, position=position, mask=mask, placement=placement, kv=kv):
                            cfg = Config(vocab_size=12, block_size=6, n_layer=2, n_head=4, n_kv_head=kv,
                                         d_model=16, d_ff=24, pos_encoding=position, mask=mask, window=2,
                                         norm_placement=placement, norm="rmsnorm" if kv < 4 else "layernorm",
                                         activation="swiglu" if kv == 2 else "gelu", residual=kv != 1, tie_weights=True, bias=False)
                            model = GPT(cfg).to(dtype=dtype).eval()
                            ids = torch.tensor([[1, 5, 3, 2, 4, 6], [3, 2, 4, 8, 1, 0]])
                            cache = model.create_cache()
                            for t in range(ids.shape[1]):
                                got = model.decode_step(ids[:, t:t + 1], cache)
                                want, _ = model(ids[:, :t + 1])
                                tol = 2e-6 if dtype == torch.float32 else 1e-9
                                torch.testing.assert_close(got, want[:, -1:], atol=tol, rtol=tol)
                                for saved in cache["layers"]:
                                    self.assertEqual(saved["k"].shape, (2, kv, t + 1, cfg.d_head))
                                    self.assertEqual(saved["v"].dtype, dtype)
                            with self.assertRaises(ValueError):
                                model.decode_step(ids[:, :1], cache)

    def test_sampling_and_invalid_inputs(self):
        model = GPT(Config(vocab_size=8, block_size=4, d_model=8, d_ff=16)).eval()
        ids = torch.tensor([[1, 2]])
        logits, _ = model(ids)
        result = model.generate(ids, 1, temperature=0)
        self.assertEqual(result[0, -1].item(), logits[0, -1].argmax().item())
        for top_k in (0, 1, 100):
            self.assertEqual(model.generate(ids, 2, top_k=top_k, top_p=0.9).shape, (1, 4))
        for kwargs in ({"temperature": -1}, {"temperature": float("nan")}, {"top_k": -1}, {"top_p": 0}, {"top_p": 1.5}):
            with self.assertRaises(ValueError):
                model.generate(ids, 1, **kwargs)
        for patch in ({"norm_eps": 0}, {"rope_base": float("inf")}, {"prefix_len": -1}, {"activation": "oops"}):
            with self.assertRaises(ValueError):
                Config(**patch)
        for mask in ("none", "prefix"):
            with self.assertRaises(ValueError):
                GPT(Config(mask=mask)).create_cache()
        cache = model.create_cache()
        model.decode_step(ids[:, :1], cache)
        with self.assertRaises(ValueError):
            model.decode_step(ids.repeat(2, 1)[:, :1], cache)
        with self.assertRaises(ValueError):
            GPT(model.cfg).decode_step(ids[:, :1], cache)


if __name__ == "__main__":
    unittest.main()

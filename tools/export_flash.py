"""Independent dense PyTorch oracle for the streamed attention lesson."""
import json
import sys
import math
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'reference'))
from flash_attention import online_attention
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
cases = [([-math.inf, -math.inf, -math.inf, 1000, 999, -800, -math.inf, 1002], 3, [1, 2, 3, 4, 9]),
         ([0, 0, 0, 0, 0], 4, [1, 2, 4, 8]), ([3, -2, 1, -7, 8, 0, -math.inf], 2, [1, 3, 4])]
result = []
for raw, width, tiles in cases:
    scores = torch.tensor(raw, dtype=torch.float64)
    values = torch.tensor([math.sin(i + 1) for i in range(len(raw) * width)]).reshape(-1, width)
    dense = scores.softmax(0) @ values
    prefixes = {}
    for size in tiles:
        torch.testing.assert_close(online_attention(scores, values, size), dense, atol=1e-12, rtol=1e-12)
        prefixes[size] = []
        for end in range(size, len(raw) + size, size):
            s = scores[:min(end, len(raw))]
            prefixes[size].append((s.softmax(0) @ values[:len(s)]).tolist() if torch.isfinite(s).any() else [0.] * width)
    result.append(dict(scores=[None if v == -math.inf else v for v in raw], values=values.flatten().tolist(),
                       width=width, output=dense.tolist(), prefixes=prefixes))
(ROOT / 'tests/fixtures/lessons/flash.json').write_text(json.dumps(result))
print('Exported dense outputs and every tile prefix from PyTorch.')

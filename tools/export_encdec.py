"""Independent encoder/decoder traces, including rectangular cross-attention."""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'reference'))
from encoder_decoder import TinyEncoderDecoder
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
folder = ROOT / 'tests/fixtures/lessons'
spec = json.loads((folder / 'encdec_inputs.json').read_text())
model = TinyEncoderDecoder().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        value = spec['params'][name]
        p.copy_(torch.tensor(value['data']).reshape(value['shape']))
    result = []
    for source, target in (([2, 3, 4, 5], [0, 1, 2]), ([1, 2, 4], [0, 2]), ([2], [0]), ([2, 5, 5], [0, 1, 5])):
        _, trace = model(torch.tensor([source]), torch.tensor([target]))
        result.append(dict(source=source, target=target, trace={k: dict(shape=list(v[0].shape),
          data=[None if x == float('-inf') else x for x in v[0].double().flatten().tolist()]) for k, v in trace.items()}))
(folder / 'encdec.json').write_text(json.dumps(result))
print('Exported four complete encoder-decoder passes from PyTorch.')

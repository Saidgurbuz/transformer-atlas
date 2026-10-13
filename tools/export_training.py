import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
from model import GPT, Config
from training_lesson import make_optimizer, training_step
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
folder = ROOT / "tests/fixtures/lessons"
inputs = json.loads((folder / "training_inputs.json").read_text())
def encode(t): return dict(shape=list(t.shape), data=t.detach().flatten().tolist())
result = {}
for profile, spec in inputs.items():
    result[profile] = {}
    for kind in ("sgd", "adamw"):
        model = GPT(Config(**spec["config"]))
        with torch.no_grad():
            for name, p in model.named_parameters():
                value = spec["params"][name]
                p.copy_(torch.tensor(value["data"]).reshape(value["shape"]))
        optimizer = make_optimizer(model, kind)
        rounds = []
        for i in range(3):
            r = training_step(model, optimizer)
            r["gradients"] = {n: encode(g) for n, g in r["gradients"].items()}
            r["logitsGradient"] = encode(r["logitsGradient"])
            r["afterLogits"] = encode(r["afterLogits"])
            r["paramsAfter"] = {n: encode(p) for n, p in model.named_parameters()}
            rounds.append(r)
        result[profile][kind] = rounds
(folder / "training.json").write_text(json.dumps(result))
print('Exported 12 actual PyTorch training updates, including every parameter gradient.')

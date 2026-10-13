"""Export PyTorch's values, strides and storage aliases for the tensor lesson."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
from tensor_ops import tensor_operations

tensors = tensor_operations()
out = {key: dict(shape=list(t.shape), data=t.flatten().tolist(), strides=list(t.stride()),
                 sharesX=t.untyped_storage().data_ptr() == tensors["x"].untyped_storage().data_ptr()) for key, t in tensors.items()}
folder = ROOT / "tests/fixtures/lessons"
folder.mkdir(exist_ok=True)
(folder / "tensor_ops.json").write_text(json.dumps(out))
print('Exported 12 PyTorch tensor operations with strides and storage aliases.')

"""Optional comparison against the inspected authors' module supplied as a file.

Run on a compute node: .venv/bin/python tools/check_yarn_reference.py /path/to/author_module.py
No download occurs here; the external module is used only for validation.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'reference'))
from rope_extensions import yarn_frequencies
torch.set_num_threads(1)
path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('authors_yarn', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = 0
for width in (8, 16, 64, 128):
    for context in (128, 2048, 4096):
        for scale in (1., 4., 16.):
            author = module.LlamaYaRNScaledRotaryEmbedding(width, max_position_embeddings=4, original_max_position_embeddings=context, scale=scale)
            frequencies, amplitude = yarn_frequencies(width, context, scale)
            torch.testing.assert_close(frequencies, author.inv_freq.double(), atol=1e-12, rtol=3e-7)
            assert abs(amplitude - author.mscale) < 1e-12
            checks += 1
print(f'{checks} author-implementation comparisons passed; SHA256={hashlib.sha256(path.read_bytes()).hexdigest()}')

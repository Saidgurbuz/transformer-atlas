// Written by tools/train_presets.py.
export const MANIFEST = [
  {
    "id": "charlm",
    "title": "Alice char-LM \u00b7 GPT-2 style",
    "description": "4 layers, 4 heads, d=64, learned positions, pre-LayerNorm, GELU. Trained on Alice in Wonderland, one character per token (no held-out split: it has memorised much of the book).",
    "params": 208256,
    "metrics": {
      "train_loss_nats": 0.5464
    }
  },
  {
    "id": "llama",
    "title": "Alice char-LM \u00b7 LLaMA style",
    "description": "Same data, modern recipe: RoPE, RMSNorm, SwiGLU, grouped-query attention (4 query heads, 2 KV heads), no biases.",
    "params": 185920,
    "metrics": {
      "train_loss_nats": 0.2876
    }
  },
  {
    "id": "induction",
    "title": "Induction heads",
    "description": "2 layers trained to continue repeated random text. Look for a previous-token head in layer 0 and an induction head in layer 1.",
    "params": 106240,
    "metrics": {
      "repeat_accuracy": 0.9971
    }
  },
  {
    "id": "reverse",
    "title": "Reverse a word",
    "description": "2 layers trained on 'word.drow'. After the period, attention reads the input backwards: an anti-diagonal pattern.",
    "params": 105728,
    "metrics": {
      "reversed_char_accuracy": 1.0
    }
  }
];

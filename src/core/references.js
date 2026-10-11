// Curated reading list. Every entry is checked by tools/ (link resolves, title,
// authors and year match) before it ships. `views` lists the pages where the
// entry appears in "Further reading"; `why` says what it adds for that topic.
//
// Views render it with readingList(viewId) and cite inline with cite(id).

import { h } from '../ui/dom.js';

export const REFERENCES = [
  // ------------------------------------------------------------- foundations
  { id: 'vaswani2017', title: 'Attention Is All You Need', authors: 'Vaswani et al.', year: 2017, org: 'Google', type: 'paper',
    url: 'https://arxiv.org/abs/1706.03762', views: ['overview', 'attention', 'multihead', 'positions', 'encdec', 'cheatsheet'],
    why: 'The original Transformer: scaled dot-product attention, multi-head attention, sinusoidal positions, encoder-decoder.' },
  { id: 'radford2019', title: 'Language Models are Unsupervised Multitask Learners (GPT-2)', authors: 'Radford et al.', year: 2019, org: 'OpenAI', type: 'paper',
    url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf', views: ['overview', 'residual', 'tokens'],
    why: 'The decoder-only recipe this app follows: pre-LN blocks, final LayerNorm, residual-projection init scaled by depth, byte-level BPE.' },
  { id: 'brown2020', title: 'Language Models are Few-Shot Learners (GPT-3)', authors: 'Brown et al.', year: 2020, org: 'OpenAI', type: 'paper',
    url: 'https://arxiv.org/abs/2005.14165', views: ['scaling'],
    why: 'Model sizes from 125M to 175B parameters with their d_model, layers and heads.' },
  { id: 'touvron2023', title: 'LLaMA: Open and Efficient Foundation Language Models', authors: 'Touvron et al.', year: 2023, org: 'Meta', type: 'paper',
    url: 'https://arxiv.org/abs/2302.13971', views: ['overview', 'mlp', 'residual', 'positions'],
    why: 'The modern recipe: pre-normalisation with RMSNorm, SwiGLU MLP, rotary embeddings.' },
  { id: 'touvron2023b', title: 'Llama 2: Open Foundation and Fine-Tuned Chat Models', authors: 'Touvron et al.', year: 2023, org: 'Meta', type: 'paper',
    url: 'https://arxiv.org/abs/2307.09288', views: ['multihead', 'scaling'],
    why: 'Uses grouped-query attention in the 34B and 70B models to cut KV-cache memory.' },

  // ------------------------------------------------------ explainers & courses
  { id: 'annotated', title: 'The Annotated Transformer', authors: 'Rush; updated by Huang et al.', year: 2022, org: 'Harvard NLP', type: 'blog',
    url: 'https://nlp.seas.harvard.edu/annotated-transformer/', views: ['stepper', 'encdec', 'cheatsheet'],
    why: 'The 2017 paper as runnable PyTorch, line by line: the same idea as this app’s code stepper.' },
  { id: 'alammar-illustrated', title: 'The Illustrated Transformer', authors: 'Jay Alammar', year: 2018, org: 'Blog', type: 'blog',
    url: 'https://jalammar.github.io/illustrated-transformer/', views: ['overview', 'attention', 'encdec'],
    why: 'The classic visual walk-through of Q, K, V, multi-head attention and the encoder-decoder.' },
  { id: 'alammar-gpt2', title: 'The Illustrated GPT-2', authors: 'Jay Alammar', year: 2019, org: 'Blog', type: 'blog',
    url: 'https://jalammar.github.io/illustrated-gpt2/', views: ['masking', 'generation', 'head'],
    why: 'Decoder-only models, masked self-attention and token-by-token generation, drawn step by step.' },
  { id: 'weng-family', title: 'The Transformer Family Version 2.0', authors: 'Lilian Weng', year: 2023, org: 'Blog', type: 'blog',
    url: 'https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/', views: ['positions', 'masking', 'multihead', 'cheatsheet'],
    why: 'A dense survey of variants: positional encodings, sparse and local attention, efficient Transformers.' },
  { id: 'weng-inference', title: 'Large Transformer Model Inference Optimization', authors: 'Lilian Weng', year: 2023, org: 'Blog', type: 'blog',
    url: 'https://lilianweng.github.io/posts/2023-01-10-inference-optimization/', views: ['generation', 'scaling'],
    why: 'Why inference is memory-bound, KV-cache size, quantisation and other inference optimisations.' },
  { id: 'karpathy-gpt', title: 'Let’s build GPT: from scratch, in code, spelled out', authors: 'Andrej Karpathy', year: 2023, org: 'YouTube', type: 'video',
    url: 'https://www.youtube.com/watch?v=kCc8FmEb1nY', views: ['stepper', 'attention', 'cheatsheet'],
    why: 'Builds a character-level GPT like this app’s presets, deriving self-attention from a running average.' },
  { id: 'karpathy-tokenizer', title: 'Let’s build the GPT Tokenizer', authors: 'Andrej Karpathy', year: 2024, org: 'YouTube', type: 'video',
    url: 'https://www.youtube.com/watch?v=zduSFxRajkE', views: ['tokens'],
    why: 'Byte-pair encoding from scratch and the tokenizer quirks that leak into model behaviour.' },
  { id: 'nanogpt', title: 'nanoGPT', authors: 'Andrej Karpathy', year: 2023, org: 'GitHub', type: 'code',
    url: 'https://github.com/karpathy/nanoGPT', views: ['stepper', 'overview'],
    why: 'A minimal GPT-2 training and inference codebase; reference/model.py follows its structure.' },
  { id: 'mody-numpy', title: 'GPT in 60 Lines of NumPy', authors: 'Jay Mody', year: 2023, org: 'Blog', type: 'blog',
    url: 'https://jaykmody.com/blog/gpt-from-scratch/', views: ['cheatsheet', 'stepper'],
    why: 'A complete GPT-2 forward pass in NumPy that loads the real weights.' },
  { id: '3b1b-attention', title: 'Attention in transformers, step-by-step', authors: '3Blue1Brown (Grant Sanderson)', year: 2024, org: 'YouTube', type: 'video',
    url: 'https://www.youtube.com/watch?v=eMlx5fFNoYc', views: ['attention', 'multihead'],
    why: 'A geometric picture of queries, keys and values and why heads are low-rank.' },
  { id: '3b1b-mlp', title: 'How might LLMs store facts', authors: '3Blue1Brown (Grant Sanderson)', year: 2024, org: 'YouTube', type: 'video',
    url: 'https://www.youtube.com/watch?v=9-Jl0dxWQs8', views: ['mlp'],
    why: 'The MLP block as a bank of feature detectors whose outputs are added to the residual stream.' },
  { id: 'cs336', title: 'CS336: Language Modeling from Scratch', authors: 'Percy Liang, Tatsunori Hashimoto et al.', year: 2025, org: 'Stanford', type: 'course',
    url: 'https://stanford-cs336.github.io/', views: ['overview', 'scaling', 'cheatsheet'],
    why: 'Build and train an LM end to end: tokenizers, architectures, systems, scaling laws.' },
  { id: 'cs224n', title: 'CS224N: Natural Language Processing with Deep Learning', authors: 'Christopher Manning et al.', year: 2024, org: 'Stanford', type: 'course',
    url: 'https://web.stanford.edu/class/cs224n/', views: ['attention', 'encdec', 'tokens'],
    why: 'Lecture notes on self-attention, Transformers, pretraining and tokenisation.' },
  { id: 'scaling-book', title: 'How to Scale Your Model', authors: 'Austin et al.', year: 2025, org: 'Google DeepMind', type: 'blog',
    url: 'https://jax-ml.github.io/scaling-book/', views: ['scaling', 'generation'],
    why: 'Transformer FLOPs and memory arithmetic, KV caches, prefill vs decode and roofline analysis.' },
  { id: 'transformer-math', title: 'Transformer Math 101', authors: 'Anthony, Biderman, Schoelkopf', year: 2023, org: 'EleutherAI', type: 'blog',
    url: 'https://blog.eleuther.ai/transformer-math/', views: ['scaling'],
    why: 'C ≈ 6PD for training compute and the bytes-per-parameter of mixed-precision Adam.' },
  { id: 'kipply', title: 'Transformer Inference Arithmetic', authors: 'kipply', year: 2022, org: 'Blog', type: 'blog',
    url: 'https://kipp.ly/transformer-inference-arithmetic/', views: ['generation', 'scaling'],
    why: 'KV-cache sizing, FLOPs per token, and when decoding is memory-bandwidth bound.' },
  { id: 'raschka-arch', title: 'The Big LLM Architecture Comparison', authors: 'Sebastian Raschka', year: 2025, org: 'Ahead of AI', type: 'blog',
    url: 'https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison', views: ['overview', 'multihead', 'residual'],
    why: 'How recent open models differ: GQA vs MLA, norm placement, MoE, sliding-window attention.' },

  // ---------------------------------------------------------- interpretability
  { id: 'elhage2021', title: 'A Mathematical Framework for Transformer Circuits', authors: 'Elhage et al.', year: 2021, org: 'Anthropic', type: 'paper',
    url: 'https://transformer-circuits.pub/2021/framework/index.html', views: ['residual', 'multihead', 'attention', 'overview'],
    why: 'The residual stream as a shared communication channel; heads as independent QK and OV circuits that read from and write to it.' },
  { id: 'olsson2022', title: 'In-context Learning and Induction Heads', authors: 'Olsson et al.', year: 2022, org: 'Anthropic', type: 'paper',
    url: 'https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html', views: ['multihead', 'lab', 'attention'],
    why: 'Defines induction heads ([A][B] … [A] → attend to the token after the earlier [A] and copy it) and the previous-token heads that enable them.' },
  { id: 'elhage2022', title: 'Toy Models of Superposition', authors: 'Elhage et al.', year: 2022, org: 'Anthropic', type: 'paper',
    url: 'https://transformer-circuits.pub/2022/toy_model/index.html', views: ['mlp', 'tokens'],
    why: 'Why individual neurons are often polysemantic: models pack more features than dimensions.' },
  { id: 'bricken2023', title: 'Towards Monosemanticity: Decomposing Language Models With Dictionary Learning', authors: 'Bricken et al.', year: 2023, org: 'Anthropic', type: 'paper',
    url: 'https://transformer-circuits.pub/2023/monosemantic-features/index.html', views: ['mlp'],
    why: 'Sparse autoencoders recover interpretable features from MLP activations where neurons fail.' },
  { id: 'templeton2024', title: 'Scaling Monosemanticity', authors: 'Templeton et al.', year: 2024, org: 'Anthropic', type: 'paper',
    url: 'https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html', views: ['residual'],
    why: 'Dictionary learning on a production model’s residual stream finds millions of interpretable features.' },
  { id: 'geva2021', title: 'Transformer Feed-Forward Layers Are Key-Value Memories', authors: 'Geva et al.', year: 2021, org: 'EMNLP', type: 'paper',
    url: 'https://arxiv.org/abs/2012.14913', views: ['mlp'],
    why: 'Rows of the first MLP matrix act as keys matched to input patterns; columns of the second write values.' },
  { id: 'geva2022', title: 'Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space', authors: 'Geva et al.', year: 2022, org: 'EMNLP', type: 'paper',
    url: 'https://arxiv.org/abs/2203.14680', views: ['mlp', 'residual'],
    why: 'Projects each MLP value vector through the unembedding to read which tokens it promotes: the neuron view in this app.' },
  { id: 'logit-lens', title: 'interpreting GPT: the logit lens', authors: 'nostalgebraist', year: 2020, org: 'LessWrong', type: 'blog',
    url: 'https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens', views: ['residual', 'head'],
    why: 'Decode intermediate residual states with the final unembedding to watch the prediction form layer by layer.' },
  { id: 'belrose2023', title: 'Eliciting Latent Predictions from Transformers with the Tuned Lens', authors: 'Belrose et al.', year: 2023, org: 'arXiv', type: 'paper',
    url: 'https://arxiv.org/abs/2303.08112', views: ['residual'],
    why: 'A learned affine probe per layer that fixes the logit lens’s bias on early layers.' },
  { id: 'xiao2023', title: 'Efficient Streaming Language Models with Attention Sinks', authors: 'Xiao et al.', year: 2023, org: 'ICLR 2024', type: 'paper',
    url: 'https://arxiv.org/abs/2309.17453', views: ['attention', 'multihead', 'masking'],
    why: 'Many heads park attention on the first token as a no-op sink; keeping it in the cache stabilises sliding-window decoding.' },
  { id: 'clark2019', title: 'What Does BERT Look At? An Analysis of BERT’s Attention', authors: 'Clark et al.', year: 2019, org: 'BlackboxNLP', type: 'paper',
    url: 'https://arxiv.org/abs/1906.04341', views: ['multihead'],
    why: 'Catalogues head types: previous/next token, delimiter tokens, broad attention, syntactic relations.' },
  { id: 'michel2019', title: 'Are Sixteen Heads Really Better than One?', authors: 'Michel, Levy, Neubig', year: 2019, org: 'NeurIPS', type: 'paper',
    url: 'https://arxiv.org/abs/1905.10650', views: ['lab', 'multihead'],
    why: 'Ablating heads one at a time: many can be removed at test time with little loss.' },
  { id: 'voita2019', title: 'Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned', authors: 'Voita et al.', year: 2019, org: 'ACL', type: 'paper',
    url: 'https://arxiv.org/abs/1905.09418', views: ['lab', 'multihead'],
    why: 'A few specialised heads (positional, syntactic, rare-token) carry most of the function.' },
  { id: 'wang2022ioi', title: 'Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small', authors: 'Wang et al.', year: 2022, org: 'ICLR 2023', type: 'paper',
    url: 'https://arxiv.org/abs/2211.00593', views: ['lab'],
    why: 'Ablation and activation patching used to trace a full circuit of heads in a real model.' },
  { id: 'bills2023', title: 'Language models can explain neurons in language models', authors: 'Bills et al.', year: 2023, org: 'OpenAI', type: 'paper',
    url: 'https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html', views: ['mlp'],
    why: 'Automatically labels GPT-2 neurons from their top activations, and measures how often labels hold.' },

  // ------------------------------------------------------------- components
  { id: 'he2016', title: 'Deep Residual Learning for Image Recognition', authors: 'He et al.', year: 2015, org: 'Microsoft Research', type: 'paper',
    url: 'https://arxiv.org/abs/1512.03385', views: ['residual'],
    why: 'Residual connections: learn a correction to the identity, which keeps deep networks trainable.' },
  { id: 'ba2016', title: 'Layer Normalization', authors: 'Ba, Kiros, Hinton', year: 2016, org: 'arXiv', type: 'paper',
    url: 'https://arxiv.org/abs/1607.06450', views: ['residual'],
    why: 'Normalise each example over its features rather than over the batch.' },
  { id: 'zhang2019rms', title: 'Root Mean Square Layer Normalization', authors: 'Zhang, Sennrich', year: 2019, org: 'NeurIPS', type: 'paper',
    url: 'https://arxiv.org/abs/1910.07467', views: ['residual'],
    why: 'Drops the mean subtraction and bias: cheaper, and as good in practice.' },
  { id: 'xiong2020', title: 'On Layer Normalization in the Transformer Architecture', authors: 'Xiong et al.', year: 2020, org: 'ICML', type: 'paper',
    url: 'https://arxiv.org/abs/2002.04745', views: ['residual', 'overview'],
    why: 'Post-LN has large gradients near the output at initialisation, hence warmup; pre-LN trains without it.' },
  { id: 'liu2020admin', title: 'Understanding the Difficulty of Training Transformers', authors: 'Liu et al.', year: 2020, org: 'EMNLP', type: 'paper',
    url: 'https://arxiv.org/abs/2004.08249', views: ['residual'],
    why: 'Post-LN’s heavy reliance on the residual branch amplifies small parameter changes.' },
  { id: 'hendrycks2016', title: 'Gaussian Error Linear Units (GELUs)', authors: 'Hendrycks, Gimpel', year: 2016, org: 'arXiv', type: 'paper',
    url: 'https://arxiv.org/abs/1606.08415', views: ['mlp'],
    why: 'GELU and its tanh approximation, the activation used by GPT-2 and this app.' },
  { id: 'shazeer2020', title: 'GLU Variants Improve Transformer', authors: 'Noam Shazeer', year: 2020, org: 'arXiv', type: 'paper',
    url: 'https://arxiv.org/abs/2002.05202', views: ['mlp'],
    why: 'SwiGLU and friends; shrinks the hidden width to 2/3 to keep parameter count fixed.' },
  { id: 'shazeer2017moe', title: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer', authors: 'Shazeer et al.', year: 2017, org: 'ICLR', type: 'paper',
    url: 'https://arxiv.org/abs/1701.06538', views: ['mlp'],
    why: 'Routes each token to a few expert MLPs, decoupling parameter count from compute per token.' },
  { id: 'fedus2021', title: 'Switch Transformers', authors: 'Fedus, Zoph, Shazeer', year: 2021, org: 'JMLR', type: 'paper',
    url: 'https://arxiv.org/abs/2101.03961', views: ['mlp', 'scaling'],
    why: 'Top-1 expert routing at scale, with load balancing.' },

  // ------------------------------------------------------------- positions
  { id: 'su2021', title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding', authors: 'Su et al.', year: 2021, org: 'arXiv', type: 'paper',
    url: 'https://arxiv.org/abs/2104.09864', views: ['positions'],
    why: 'Rotates query/key pairs by position so their dot product depends only on the offset.' },
  { id: 'eleuther-rope', title: 'Rotary Embeddings: A Relative Revolution', authors: 'Biderman et al.', year: 2021, org: 'EleutherAI', type: 'blog',
    url: 'https://blog.eleuther.ai/rotary-embeddings/', views: ['positions'],
    why: 'An accessible derivation of RoPE and its comparison with learned and relative encodings.' },
  { id: 'press2021', title: 'Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation', authors: 'Press, Smith, Lewis', year: 2021, org: 'ICLR 2022', type: 'paper',
    url: 'https://arxiv.org/abs/2108.12409', views: ['positions'],
    why: 'ALiBi: a per-head linear distance penalty on the scores instead of position embeddings.' },
  { id: 'kazemnejad2023', title: 'The Impact of Positional Encoding on Length Generalization in Transformers', authors: 'Kazemnejad et al.', year: 2023, org: 'NeurIPS', type: 'paper',
    url: 'https://arxiv.org/abs/2305.19466', views: ['positions'],
    why: 'Decoder-only models without positional encoding (NoPE) can still learn position through the causal mask.' },
  { id: 'chen2023pi', title: 'Extending Context Window of Large Language Models via Positional Interpolation', authors: 'Chen et al.', year: 2023, org: 'Meta', type: 'paper',
    url: 'https://arxiv.org/abs/2306.15595', views: ['positions'],
    why: 'Extend RoPE models by scaling positions down into the trained range, then fine-tune briefly.' },
  { id: 'peng2023yarn', title: 'YaRN: Efficient Context Window Extension of Large Language Models', authors: 'Peng et al.', year: 2023, org: 'ICLR 2024', type: 'paper',
    url: 'https://arxiv.org/abs/2309.00071', views: ['positions'],
    why: 'Frequency-aware RoPE scaling that treats fast and slow rotation pairs differently.' },

  // ----------------------------------------------------- attention variants
  { id: 'shazeer2019mqa', title: 'Fast Transformer Decoding: One Write-Head is All You Need', authors: 'Noam Shazeer', year: 2019, org: 'arXiv', type: 'paper',
    url: 'https://arxiv.org/abs/1911.02150', views: ['multihead', 'generation'],
    why: 'Multi-query attention: one shared K/V head makes incremental decoding far less memory-bound.' },
  { id: 'ainslie2023gqa', title: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints', authors: 'Ainslie et al.', year: 2023, org: 'EMNLP', type: 'paper',
    url: 'https://arxiv.org/abs/2305.13245', views: ['multihead', 'generation'],
    why: 'Grouped-query attention interpolates between MHA and MQA; existing models can be uptrained to it.' },
  { id: 'dao2022flash', title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness', authors: 'Dao et al.', year: 2022, org: 'NeurIPS', type: 'paper',
    url: 'https://arxiv.org/abs/2205.14135', views: ['attention', 'cheatsheet'],
    why: 'Exact attention without materialising the T×T matrix: tiling plus an online softmax in on-chip memory.' },
  { id: 'milakov2018', title: 'Online normalizer calculation for softmax', authors: 'Milakov, Gimelshein', year: 2018, org: 'NVIDIA', type: 'paper',
    url: 'https://arxiv.org/abs/1805.02867', views: ['attention', 'head'],
    why: 'Computes the softmax max and normaliser in one pass, the trick FlashAttention builds on.' },
  { id: 'beltagy2020', title: 'Longformer: The Long-Document Transformer', authors: 'Beltagy, Peters, Cohan', year: 2020, org: 'AI2', type: 'paper',
    url: 'https://arxiv.org/abs/2004.05150', views: ['masking'],
    why: 'Sliding-window plus global attention: cost linear in sequence length.' },
  { id: 'jiang2023mistral', title: 'Mistral 7B', authors: 'Jiang et al.', year: 2023, org: 'Mistral AI', type: 'paper',
    url: 'https://arxiv.org/abs/2310.06825', views: ['masking', 'multihead', 'scaling'],
    why: 'Sliding-window attention with a rolling KV cache, plus GQA, in a 7B model.' },

  // -------------------------------------------------------------- decoding
  { id: 'holtzman2019', title: 'The Curious Case of Neural Text Degeneration', authors: 'Holtzman et al.', year: 2019, org: 'ICLR 2020', type: 'paper',
    url: 'https://arxiv.org/abs/1904.09751', views: ['head', 'generation'],
    why: 'Introduces nucleus (top-p) sampling and explains why greedy and beam search degenerate.' },
  { id: 'press2016tying', title: 'Using the Output Embedding to Improve Language Models', authors: 'Press, Wolf', year: 2016, org: 'EACL 2017', type: 'paper',
    url: 'https://arxiv.org/abs/1608.05859', views: ['head', 'tokens'],
    why: 'Weight tying: share the input embedding and the output projection.' },
  { id: 'kwon2023paged', title: 'Efficient Memory Management for Large Language Model Serving with PagedAttention', authors: 'Kwon et al.', year: 2023, org: 'SOSP', type: 'paper',
    url: 'https://arxiv.org/abs/2309.06180', views: ['generation'],
    why: 'vLLM: stores the KV cache in fixed-size pages to avoid fragmentation and share prefixes.' },
  { id: 'leviathan2022', title: 'Fast Inference from Transformers via Speculative Decoding', authors: 'Leviathan, Kalman, Matias', year: 2022, org: 'ICML 2023', type: 'paper',
    url: 'https://arxiv.org/abs/2211.17192', views: ['generation'],
    why: 'A small draft model proposes tokens that the large model verifies in one parallel pass.' },

  // ------------------------------------------------------------ tokenisation
  { id: 'sennrich2016', title: 'Neural Machine Translation of Rare Words with Subword Units', authors: 'Sennrich, Haddow, Birch', year: 2016, org: 'ACL', type: 'paper',
    url: 'https://arxiv.org/abs/1508.07909', views: ['tokens'],
    why: 'Byte-pair encoding for NLP: merge the most frequent adjacent pair, repeat.' },
  { id: 'kudo2018', title: 'SentencePiece: A simple and language independent subword tokenizer and detokenizer', authors: 'Kudo, Richardson', year: 2018, org: 'EMNLP', type: 'paper',
    url: 'https://arxiv.org/abs/1808.06226', views: ['tokens'],
    why: 'Tokenises raw text including spaces; used by LLaMA 1/2 and T5.' },

  // ------------------------------------------------------------- scaling
  { id: 'kaplan2020', title: 'Scaling Laws for Neural Language Models', authors: 'Kaplan et al.', year: 2020, org: 'OpenAI', type: 'paper',
    url: 'https://arxiv.org/abs/2001.08361', views: ['scaling'],
    why: 'Loss falls as a power law in parameters, data and compute; also the 2N FLOPs-per-token estimate.' },
  { id: 'hoffmann2022', title: 'Training Compute-Optimal Large Language Models (Chinchilla)', authors: 'Hoffmann et al.', year: 2022, org: 'DeepMind', type: 'paper',
    url: 'https://arxiv.org/abs/2203.15556', views: ['scaling'],
    why: 'For a fixed budget, scale parameters and tokens together: about 20 tokens per parameter.' },
  { id: 'rajbhandari2019', title: 'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models', authors: 'Rajbhandari et al.', year: 2019, org: 'Microsoft', type: 'paper',
    url: 'https://arxiv.org/abs/1910.02054', views: ['scaling'],
    why: 'Mixed-precision Adam needs about 16 bytes per parameter; ZeRO shards it across GPUs.' },
  { id: 'micikevicius2017', title: 'Mixed Precision Training', authors: 'Micikevicius et al.', year: 2017, org: 'NVIDIA / Baidu', type: 'paper',
    url: 'https://arxiv.org/abs/1710.03740', views: ['scaling', 'cheatsheet'],
    why: 'FP16 compute with an FP32 master copy of the weights and loss scaling.' },
  { id: 'chen2016ckpt', title: 'Training Deep Nets with Sublinear Memory Cost', authors: 'Chen et al.', year: 2016, org: 'arXiv', type: 'paper',
    url: 'https://arxiv.org/abs/1604.06174', views: ['scaling', 'cheatsheet'],
    why: 'Gradient checkpointing: recompute activations in the backward pass to save memory.' },

  // ----------------------------------------------------------- model families
  { id: 'devlin2018', title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', authors: 'Devlin et al.', year: 2018, org: 'Google', type: 'paper',
    url: 'https://arxiv.org/abs/1810.04805', views: ['encdec', 'masking'],
    why: 'Encoder-only, bidirectional attention, masked-language-model pretraining.' },
  { id: 'raffel2019', title: 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)', authors: 'Raffel et al.', year: 2019, org: 'Google', type: 'paper',
    url: 'https://arxiv.org/abs/1910.10683', views: ['encdec', 'masking'],
    why: 'Compares encoder-decoder, decoder-only and prefix-LM masks under one framework.' },
  { id: 'lewis2019bart', title: 'BART: Denoising Sequence-to-Sequence Pre-training', authors: 'Lewis et al.', year: 2019, org: 'Facebook AI', type: 'paper',
    url: 'https://arxiv.org/abs/1910.13461', views: ['encdec'],
    why: 'An encoder-decoder pretrained to reconstruct corrupted text.' },
  { id: 'tay2022ul2', title: 'UL2: Unifying Language Learning Paradigms', authors: 'Tay et al.', year: 2022, org: 'Google', type: 'paper',
    url: 'https://arxiv.org/abs/2205.05131', views: ['encdec', 'masking'],
    why: 'A mixture of denoising objectives; discusses causal vs prefix-LM vs encoder-decoder.' },

  // --------------------------------------------------------------- vision
  { id: 'dosovitskiy2020', title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)', authors: 'Dosovitskiy et al.', year: 2020, org: 'Google', type: 'paper',
    url: 'https://arxiv.org/abs/2010.11929', views: ['vision'],
    why: 'Split the image into patches, embed each linearly, add positions, run a Transformer encoder.' },
  { id: 'radford2021clip', title: 'Learning Transferable Visual Models From Natural Language Supervision (CLIP)', authors: 'Radford et al.', year: 2021, org: 'OpenAI', type: 'paper',
    url: 'https://arxiv.org/abs/2103.00020', views: ['vision'],
    why: 'Contrastive image-text pretraining; its ViT is the vision encoder of many VLMs.' },
  { id: 'alayrac2022flamingo', title: 'Flamingo: a Visual Language Model for Few-Shot Learning', authors: 'Alayrac et al.', year: 2022, org: 'DeepMind', type: 'paper',
    url: 'https://arxiv.org/abs/2204.14198', views: ['vision', 'encdec'],
    why: 'Injects vision into a frozen LM through gated cross-attention layers.' },
  { id: 'liu2023llava', title: 'Visual Instruction Tuning (LLaVA)', authors: 'Liu et al.', year: 2023, org: 'NeurIPS', type: 'paper',
    url: 'https://arxiv.org/abs/2304.08485', views: ['vision'],
    why: 'Project CLIP patch features into the LLM’s embedding space and train on instruction data.' },
  { id: 'liu2023llava15', title: 'Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)', authors: 'Liu et al.', year: 2023, org: 'CVPR 2024', type: 'paper',
    url: 'https://arxiv.org/abs/2310.03744', views: ['vision'],
    why: 'An MLP projector and 336 px CLIP ViT-L/14 input: 576 visual tokens per image.' },
  { id: 'wang2024qwen2vl', title: 'Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution', authors: 'Wang et al.', year: 2024, org: 'Alibaba', type: 'paper',
    url: 'https://arxiv.org/abs/2409.12191', views: ['vision', 'positions'],
    why: 'Native dynamic resolution, token merging of adjacent patches, and multimodal RoPE (M-RoPE).' },
];

const byId = new Map(REFERENCES.map((r) => [r.id, r]));

export function ref(id) { return byId.get(id); }

export function refsForView(viewId) { return REFERENCES.filter((r) => r.views.includes(viewId)); }

const TYPE_LABEL = { paper: 'Paper', blog: 'Article', course: 'Course', video: 'Video', code: 'Code' };

// Inline citation: "Olsson et al. 2022" linking to the source.
export function citeLabel(r) {
  const names = r.authors.split(/,\s*/);
  let who = r.authors;
  if (/et al\./.test(r.authors)) who = names[0];
  else if (names.length === 2) who = `${names[0]} & ${names[1]}`;
  else if (names.length > 2) who = `${names[0]} et al.`;
  return `${who} ${r.year}`;
}

export function cite(id, text) {
  const r = byId.get(id);
  if (!r) return h('span', { class: 'muted' }, text ?? id);
  return h('a', { class: 'cite', href: r.url, target: '_blank', rel: 'noopener', title: r.title }, text ?? citeLabel(r));
}

// "Further reading" card for a view: entries tagged with the view, papers last.
export function readingList(viewId, { title = 'Further reading', ids = null } = {}) {
  const items = (ids ? ids.map((i) => byId.get(i)).filter(Boolean) : refsForView(viewId))
    .sort((a, b) => (a.type === 'paper') - (b.type === 'paper') || b.year - a.year);
  return h('div', { class: 'card reading' },
    h('div', { class: 'card-head' }, h('h4', null, title), h('span', { class: 'muted small' }, `${items.length} sources`)),
    h('ul', { class: 'reading-list' }, items.map((r) => h('li', null,
      h('div', { class: 'reading-meta' }, h('span', { class: 'reading-type' }, TYPE_LABEL[r.type] || r.type), h('span', null, `${r.org} · ${r.year}`)),
      h('a', { href: r.url, target: '_blank', rel: 'noopener' }, r.title),
      h('div', { class: 'small muted' }, r.authors),
      h('p', { class: 'small' }, r.why)))));
}

// A short interpretation note backed by sources: note('text…', ['olsson2022'])
export function note(text, ids = [], { title = 'How to read this' } = {}) {
  return h('div', { class: 'callout note interp' },
    h('strong', null, title),
    h('span', { html: text }),
    ids.length ? h('span', { class: 'small' }, 'Source: ', ids.flatMap((id, i) => (i ? [', ', cite(id)] : [cite(id)]))) : null);
}

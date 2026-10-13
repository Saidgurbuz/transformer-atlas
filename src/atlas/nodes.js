import { codeLines, codeRegions, keysForLine } from '../core/codemap.js';
// The zoom hierarchy. A node id is dot-separated so it can live in the URL hash:
//   model · embedding · block.2 · attention.2 · head.2.1 · softmax.2.1 · mlp.2 ...
// Each pattern names its parameters (:l layer, :h head, :n which norm/residual).
// `keys(p, cfg)` lists the trace keys the scene is about (drives code highlights,
// the maths drawer and hover cards). `story: true` puts it on the guided path.

const T = (s) => () => s;

// ?root=<name> previews an alternative top-level scene (scenes/model-<name>.js), e.g. ?root=b.
function rootScene() {
  const v = typeof location === 'undefined' ? null : new URLSearchParams(location.search).get('root');
  return v === 'b' ? import('./scenes/model-b.js') : import('./scenes/model.js');
}

export const PATTERNS = [
  { pattern: 'model', title: T('Transformer'), parent: null, kind: 'resid', blurb: 'The whole decoder-only model, input to prediction.',
    scene: rootScene, keys: () => ['emb.out', 'logits', 'probs'] },

  { pattern: 'tokens', title: T('Tokens'), parent: () => 'model', kind: 'embed', blurb: 'Text becomes a sequence of integer ids.',
    scene: () => import('./scenes/tokens.js'), keys: () => [] },
  { pattern: 'embedding', title: T('Embedding'), parent: () => 'model', kind: 'embed', blurb: 'Each id becomes a d-dimensional vector, plus position.',
    scene: () => import('./scenes/embedding.js'), keys: () => ['emb.tok', 'emb.pos', 'emb.out'] },
  { pattern: 'token-embedding', title: T('Token lookup'), parent: () => 'embedding', kind: 'embed', blurb: 'A row of W_E per token: a lookup that equals a one-hot matmul.',
    scene: () => import('./scenes/token-embedding.js'), keys: () => ['emb.tok'] },
  { pattern: 'positional', title: T('Positional encoding'), parent: () => 'embedding', kind: 'pos', blurb: 'How the model knows where each token is.',
    scene: () => import('./scenes/positional.js'), keys: (p, cfg) => (cfg.pos_encoding === 'rope' ? ['L0.attn.q_rot', 'L0.attn.k_rot'] : cfg.pos_encoding === 'alibi' ? ['alibi', 'L0.attn.biased'] : ['emb.pos']) },

  { pattern: 'block.:l', title: (p) => `Block ${p.l}`, parent: () => 'model', kind: 'resid', blurb: 'Attention then MLP, each added onto the residual stream.',
    scene: () => import('./scenes/block.js'), keys: (p) => [`L${p.l}.ln1`, `L${p.l}.attn.out`, `L${p.l}.resid_mid`, `L${p.l}.ln2`, `L${p.l}.mlp.out`, `L${p.l}.resid_out`] },
  { pattern: 'norm.:l.:n', title: (p) => (p.n === '1' ? 'Norm before attention' : 'Norm before MLP'), parent: (p) => `block.${p.l}`, kind: 'norm', blurb: 'Re-centre and re-scale each token vector.',
    scene: () => import('./scenes/norm.js'), keys: (p) => [`L${p.l}.ln${p.n}`] },
  { pattern: 'attention.:l', title: T('Attention'), parent: (p) => `block.${p.l}`, kind: 'attn', blurb: 'Every token gathers information from the tokens it may see.',
    scene: () => import('./scenes/attention.js'), keys: (p) => [`L${p.l}.attn.q`, `L${p.l}.attn.k`, `L${p.l}.attn.v`, `L${p.l}.attn.probs`, `L${p.l}.attn.out`] },
  { pattern: 'qkv.:l', title: T('Q, K, V projections'), parent: (p) => `attention.${p.l}`, kind: 'q', blurb: 'Three learned linear maps of the same input.',
    scene: () => import('./scenes/qkv.js'), keys: (p) => [`L${p.l}.attn.q`, `L${p.l}.attn.k`, `L${p.l}.attn.v`] },
  { pattern: 'head.:l.:h', title: (p) => `Head ${p.h}`, parent: (p) => `attention.${p.l}`, kind: 'attn', blurb: 'One attention head, start to finish.',
    scene: () => import('./scenes/head.js'), keys: (p) => [`L${p.l}.attn.scores`, `L${p.l}.attn.probs`, `L${p.l}.attn.ctx`] },
  { pattern: 'scores.:l.:h', title: T('Scores QKᵀ'), parent: (p) => `head.${p.l}.${p.h}`, kind: 'attn', blurb: 'Dot products between every query and key, scaled by 1/√d_h.',
    scene: () => import('./scenes/scores.js'), keys: (p) => [`L${p.l}.attn.scores`, `L${p.l}.attn.scaled`] },
  { pattern: 'mask.:l.:h', title: T('Mask'), parent: (p) => `head.${p.l}.${p.h}`, kind: 'mask', blurb: 'Which keys each query is allowed to see.',
    scene: () => import('./scenes/mask.js'), keys: (p) => ['mask', `L${p.l}.attn.masked`] },
  { pattern: 'softmax.:l.:h', title: T('Softmax'), parent: (p) => `head.${p.l}.${p.h}`, kind: 'attn', blurb: 'Scores become weights that sum to one.',
    scene: () => import('./scenes/softmax.js'), keys: (p) => [`L${p.l}.attn.masked`, `L${p.l}.attn.probs`] },
  { pattern: 'mix.:l.:h', title: T('Weighted sum of values'), parent: (p) => `head.${p.l}.${p.h}`, kind: 'v', blurb: 'Each query’s output is a weighted average of value vectors.',
    scene: () => import('./scenes/mix.js'), keys: (p) => [`L${p.l}.attn.probs`, `L${p.l}.attn.v_heads`, `L${p.l}.attn.ctx`] },
  { pattern: 'out-proj.:l', title: T('Concat & output projection'), parent: (p) => `attention.${p.l}`, kind: 'o', blurb: 'Heads are placed side by side and mixed back to width d.',
    scene: () => import('./scenes/out-proj.js'), keys: (p) => [`L${p.l}.attn.concat`, `L${p.l}.attn.out`] },
  { pattern: 'gqa.:l', title: T('Grouped-query attention'), parent: (p) => `attention.${p.l}`, kind: 'k', blurb: 'Several query heads share one key/value head.',
    scene: () => import('./scenes/gqa.js'), keys: (p) => [`L${p.l}.attn.k_heads`, `L${p.l}.attn.v_heads`] },
  { pattern: 'residual.:l.:n', title: (p) => (p.n === '1' ? 'Residual add (attention)' : 'Residual add (MLP)'), parent: (p) => `block.${p.l}`, kind: 'resid', blurb: 'The sub-layer’s output is added to the stream.',
    scene: () => import('./scenes/residual.js'), keys: (p) => [p.n === '1' ? `L${p.l}.resid_mid` : `L${p.l}.resid_out`] },
  { pattern: 'mlp.:l', title: T('MLP'), parent: (p) => `block.${p.l}`, kind: 'mlp', blurb: 'A two-layer network applied to every token on its own.',
    scene: () => import('./scenes/mlp.js'), keys: (p) => [`L${p.l}.mlp.up`, `L${p.l}.mlp.act`, `L${p.l}.mlp.out`] },
  { pattern: 'activation.:l', title: T('Activation'), parent: (p) => `mlp.${p.l}`, kind: 'mlp', blurb: 'The non-linearity: GELU, ReLU or SwiGLU.',
    scene: () => import('./scenes/activation.js'), keys: (p) => [`L${p.l}.mlp.up`, `L${p.l}.mlp.act`] },
  { pattern: 'neurons.:l', title: T('Neurons as memories'), parent: (p) => `mlp.${p.l}`, kind: 'mlp', blurb: 'Each hidden unit writes its own direction into the stream.',
    scene: () => import('./scenes/neurons.js'), keys: (p) => [`L${p.l}.mlp.act`, `L${p.l}.mlp.out`] },

  { pattern: 'final-norm', title: T('Final norm'), parent: () => 'model', kind: 'norm', blurb: 'One last normalisation before predicting.',
    scene: () => import('./scenes/final-norm.js'), keys: () => ['lnf'] },
  { pattern: 'lm-head', title: T('LM head'), parent: () => 'model', kind: 'head', blurb: 'Scores for every vocabulary entry.',
    scene: () => import('./scenes/lm-head.js'), keys: () => ['logits', 'probs'] },
  { pattern: 'sampling', title: T('Sampling'), parent: () => 'lm-head', kind: 'head', blurb: 'Temperature, top-k and top-p turn scores into a choice.',
    scene: () => import('./scenes/sampling.js'), keys: () => ['probs'] },
  { pattern: 'generation', title: T('Generation & KV cache'), parent: () => 'model', kind: 'v', blurb: 'One token at a time, reusing cached keys and values.',
    scene: () => import('./scenes/generation.js'), keys: () => ['logits', 'concept:kv-cache'] },

  // Extras: reachable from the menu (and from related scenes)
  { pattern: 'tower', title: T('All tensors (v0.1)'), parent: () => 'model', extra: true, kind: 'resid', blurb: 'Every traced tensor as a 3D tower.',
    scene: () => import('./scenes/legacy.js').then((m) => m.legacy('explorer')), keys: () => [] },
  { pattern: 'stepper', title: T('Code ↔ forward pass'), parent: () => 'model', extra: true, kind: 'attn', blurb: 'Step through model.py line by line.',
    scene: () => import('./scenes/stepper.js'), keys: () => [] },
  { pattern: 'cheatsheet', title: T('Interview cheat sheet'), parent: () => 'model', extra: true, kind: 'head', blurb: 'Formulas, shapes and flashcards.',
    scene: () => import('./scenes/cheatsheet.js'), keys: () => [] },
  { pattern: 'scaling', title: T('Params, FLOPs & memory'), parent: () => 'model', extra: true, kind: 'mlp', blurb: 'What real models cost.',
    scene: () => import('./scenes/scaling.js'), keys: () => [] },
  { pattern: 'tensor-ops', title: T('Tensor operations'), parent: () => 'model', extra: true, kind: 'q', blurb: 'Shapes, strides, views, concatenation and einsum.',
    scene: () => import('./scenes/tensor-ops.js'), keys: () => ['concept:tensor-layout'], reference: { file: 'tensor_ops.py', region: 'tensor_operations' } },
  { pattern: 'training', title: T('One training step'), parent: () => 'model', extra: true, kind: 'head', blurb: 'Loss, backpropagation, gradients and optimizer updates.',
    scene: () => import('./scenes/training.js'), keys: () => ['concept:training-gradient', 'concept:adamw'], reference: { file: 'training_lesson.py', region: 'training_step' } },
  { pattern: 'flash', title: T('FlashAttention tiles'), parent: () => 'model', extra: true, kind: 'attn', blurb: 'Exact attention via stable online softmax and key/value tiles.',
    scene: () => import('./scenes/flash.js'), keys: () => ['concept:online-attention'], reference: { file: 'flash_attention.py', region: 'online_attention' } },
  { pattern: 'lab', title: T('Ablation lab'), parent: () => 'model', extra: true, kind: 'o', blurb: 'Switch parts off and watch the prediction change.',
    scene: () => import('./scenes/lab.js'), keys: () => [] },
  { pattern: 'encdec', title: T('Encoder–decoder'), parent: () => 'model', extra: true, kind: 'pos', blurb: 'Cross-attention and the other Transformer families.',
    scene: () => import('./scenes/encdec.js'), keys: () => [] },
  { pattern: 'encoder-decoder', title: T('Encoder–decoder computation'), parent: () => 'model', extra: true, kind: 'attn', blurb: 'Run bidirectional, causal and rectangular cross-attention.',
    scene: () => import('./scenes/encoder-decoder.js'), keys: () => ['concept:encoder-decoder'], reference: { file: 'encoder_decoder.py', region: 'TinyEncoderDecoder' } },
  { pattern: 'architectures', title: T('Architecture comparison'), parent: () => 'model', extra: true, kind: 'norm', blurb: '2017, GPT-2 and Llama-style structural choices and code.',
    scene: () => import('./scenes/architectures.js'), keys: () => ['concept:architecture-choices'], reference: { file: 'training_lesson.py', region: 'architecture_profiles' } },
  { pattern: 'exercises', title: T('Transformer practice'), parent: () => 'model', extra: true, kind: 'head', blurb: 'Derive, debug and run small coding exercises with feedback.',
    scene: () => import('./scenes/exercises.js'), keys: () => ['concept:training-gradient'] },
  { pattern: 'vision', title: T('Vision tokens'), parent: () => 'tokens', extra: true, kind: 'embed', blurb: 'How images become tokens: ViT patches to VLMs.',
    scene: () => import('./scenes/vision.js'), keys: () => [] },

  // Component demos (hidden from the menu; for development and smoke tests)
  { pattern: 'demo-matmul', title: T('Demo: matmul'), parent: () => 'model', extra: true, hidden: true, kind: 'q', blurb: 'Animated matrix product.',
    scene: () => import('./scenes/legacy.js').then((m) => m.legacy('_demo_matmul')), keys: () => [] },
  { pattern: 'demo-arcs', title: T('Demo: arcs'), parent: () => 'model', extra: true, hidden: true, kind: 'attn', blurb: 'Attention arcs.',
    scene: () => import('./scenes/legacy.js').then((m) => m.legacy('_demo_arcs')), keys: () => [] },
  { pattern: 'demo-flow', title: T('Demo: flow'), parent: () => 'model', extra: true, hidden: true, kind: 'resid', blurb: 'Motion kit.',
    scene: () => import('./scenes/legacy.js').then((m) => m.legacy('_demo_flow')), keys: () => [] },
];

/** Resolve an id like "head.1.2" to {def, params}. */
export function resolve(id) {
  const parts = id.split('.');
  for (const def of PATTERNS) {
    const pp = def.pattern.split('.');
    if (pp.length !== parts.length) continue;
    const params = {};
    let ok = true;
    pp.forEach((seg, i) => {
      if (seg.startsWith(':')) params[seg.slice(1)] = parts[i];
      else if (seg !== parts[i]) ok = false;
    });
    if (ok) return { def, params };
  }
  return null;
}

export function nodeInfo(id, cfg) {
  const r = resolve(id);
  if (!r) return null;
  const { def, params } = r;
  return {
    id, def, params,
    title: def.title(params),
    kind: def.kind,
    blurb: def.blurb,
    parent: def.parent ? def.parent(params) : null,
    keys: def.keys ? def.keys(params, cfg) : [],
    extra: !!def.extra,
    reference: def.reference ?? null,
  };
}

/** Is this node valid for the current config (layer/head in range, GQA present...)? */
export function validFor(id, cfg) {
  const r = resolve(id);
  if (!r) return false;
  const { def, params } = r;
  if (params.l !== undefined && !(Number(params.l) >= 0 && Number(params.l) < cfg.n_layer)) return false;
  if (params.h !== undefined && !(Number(params.h) >= 0 && Number(params.h) < cfg.n_head)) return false;
  if (def.pattern === 'gqa.:l' && cfg.n_kv_head === cfg.n_head) return false;
  if (def.pattern === 'final-norm' && cfg.norm_placement !== 'pre') return false;
  // post-LN has no norm in front of a sub-layer; its norms follow the adds (residual nodes)
  if (def.pattern === 'norm.:l.:n' && cfg.norm_placement !== 'pre') return false;
  return true;
}

export function pathTo(id, cfg) {
  const out = [];
  for (let cur = id; cur; cur = nodeInfo(cur, cfg)?.parent) out.unshift(cur);
  return out;
}

/** The guided story: the forward pass in order, layer 0 in depth, later layers at block level. */
export function storyOrder(cfg, { head = 0 } = {}) {
  const ids = ['model', 'tokens', 'embedding', 'token-embedding', 'positional'];
  for (let l = 0; l < cfg.n_layer; l++) {
    ids.push(`block.${l}`);
    if (l === 0) {
      if (cfg.norm_placement === 'pre') ids.push('norm.0.1');
      ids.push('attention.0', 'qkv.0', `head.0.${head}`, `scores.0.${head}`, `mask.0.${head}`, `softmax.0.${head}`, `mix.0.${head}`, 'out-proj.0');
      if (cfg.n_kv_head < cfg.n_head) ids.push('gqa.0');
      ids.push('residual.0.1');
      if (cfg.norm_placement === 'pre') ids.push('norm.0.2');
      ids.push('mlp.0', 'activation.0', 'neurons.0', 'residual.0.2');
    }
  }
  if (cfg.norm_placement === 'pre') ids.push('final-norm');
  ids.push('lm-head', 'sampling', 'generation');
  return ids.filter((id) => validFor(id, cfg));
}

export const EXTRAS = PATTERNS.filter((p) => p.extra && !p.hidden).map((p) => p.pattern);

/** The node that best explains a trace key (for "zoom to this tensor"). */
export function nodeForKey(key, sel = { head: 0 }) {
  const m = /^L(\d+)\.(.*)$/.exec(key);
  const hh = sel.head ?? 0;
  if (!m) {
    if (key === 'emb.tok') return 'token-embedding';
    if (key === 'emb.pos' || key === 'alibi') return 'positional';
    if (key === 'emb.out') return 'embedding';
    if (key === 'mask') return `mask.${sel.layer ?? 0}.${hh}`;
    if (key === 'lnf') return 'final-norm';
    return 'lm-head';
  }
  const l = m[1], k = m[2];
  if (k === 'ln1') return `norm.${l}.1`;
  if (k === 'ln2') return `norm.${l}.2`;
  if (k === 'resid_mid' || k === 'add1') return `residual.${l}.1`;
  if (k === 'resid_out' || k === 'add2') return `residual.${l}.2`;
  if (/^attn\.(q|k|v)$/.test(k)) return `qkv.${l}`;
  if (/^attn\.(q_rot|k_rot|biased)$/.test(k)) return 'positional';
  if (/^attn\.(k_rep|v_rep)$/.test(k)) return `gqa.${l}`;
  if (/^attn\.(q_heads|k_heads|v_heads)$/.test(k)) return `head.${l}.${hh}`;
  if (/^attn\.(scores|scaled)$/.test(k)) return `scores.${l}.${hh}`;
  if (k === 'attn.masked') return `mask.${l}.${hh}`;
  if (k === 'attn.probs') return `softmax.${l}.${hh}`;
  if (k === 'attn.ctx') return `mix.${l}.${hh}`;
  if (/^attn\.(concat|out)$/.test(k)) return `out-proj.${l}`;
  if (/^mlp\.(up|gate|act)$/.test(k)) return `activation.${l}`;
  if (k === 'mlp.out') return `neurons.${l}`;
  return `block.${l}`;
}

/** Every concrete node id for this config (layers, heads and norms expanded). */
export function allIds(cfg) {
  const ids = [];
  for (const def of PATTERNS) {
    const segs = def.pattern.split('.');
    let combos = [[]];
    for (const seg of segs) {
      const vals = seg === ':l' ? [...Array(cfg.n_layer).keys()] : seg === ':h' ? [...Array(cfg.n_head).keys()] : seg === ':n' ? [1, 2] : [seg];
      combos = combos.flatMap((c) => vals.map((v) => [...c, String(v)]));
    }
    for (const c of combos) { const id = c.join('.'); if (validFor(id, cfg)) ids.push(id); }
  }
  return ids;
}

/** Direct children of a node (not extras). */
export function childrenOf(id, cfg) {
  return allIds(cfg).filter((x) => { const n = nodeInfo(x, cfg); return n.parent === id && !n.extra; });
}

/**
 * The atlas node that best explains one line of reference/model.py.
 * Lines that produce a traced tensor map through that tensor; other lines map through
 * their class or function. `ctx` gives the layer/head to use for per-layer code.
 */
export function nodeForLine(n, cfg, { layer = 0, head = 0 } = {}) {
  const ln = codeLines[n - 1];
  if (!ln) return null;
  const l = Math.min(layer, cfg.n_layer - 1), hh = Math.min(head, cfg.n_head - 1);
  const pre = cfg.norm_placement === 'pre';
  if (ln.keys.length) {
    const key = keysForLine(n, cfg, l)[0];
    if (!key) return null;
    const target = nodeForKey(key, { head: hh, layer: l });
    return validFor(target, cfg) ? target : null;
  }
  const region = Object.entries(codeRegions).find(([, r]) => n >= r.from && n <= r.to)?.[0];
  const raw = ln.raw;
  const pick = (id) => (validFor(id, cfg) ? id : null);
  switch (region) {
    case 'LayerNorm': case 'RMSNorm': case 'make_norm':
      return pick(pre ? `norm.${l}.1` : `residual.${l}.1`) ?? `block.${l}`;
    case 'sinusoidal_table': case 'rope_angles': case 'apply_rope': case 'alibi_bias':
      return 'positional';
    case 'build_mask':
      return `mask.${l}.${hh}`;
    case 'Attention':
      if (/q_proj|k_proj|v_proj/.test(raw)) return `qkv.${l}`;
      if (/o_proj/.test(raw)) return `out-proj.${l}`;
      return `attention.${l}`;
    case 'MLP':
      return /gate|silu|gelu|relu/.test(raw) ? `activation.${l}` : `mlp.${l}`;
    case 'Block':
      if (/ln1|ln2/.test(raw)) return pick(pre ? `norm.${l}.${/ln2/.test(raw) ? 2 : 1}` : `residual.${l}.${/ln2/.test(raw) ? 2 : 1}`) ?? `block.${l}`;
      if (/attn/.test(raw)) return `attention.${l}`;
      if (/mlp/.test(raw)) return `mlp.${l}`;
      return `block.${l}`;
    case 'GPT': {
      const gen = codeRegions.GPT && raw && codeLines.slice(0, n).map((x) => x.raw).reverse().find((r) => /^\s+def /.test(r));
      if (gen && /def (create_cache|decode_step|generate)/.test(gen)) return /def generate/.test(gen) && /softmax|top_k|top_p|multinomial|temperature|sorted|scatter|kth/.test(raw) ? 'sampling' : 'generation';
      if (/wte|vocab_size, cfg.d_model\)/.test(raw) && !/lm_head/.test(raw)) return 'token-embedding';
      if (/wpe|pos_encoding|rope|alibi/.test(raw)) return 'positional';
      if (/build_mask/.test(raw)) return `mask.${l}.${hh}`;
      if (/ln_f/.test(raw)) return pick('final-norm') ?? 'lm-head';
      if (/lm_head|tie_weights/.test(raw)) return 'lm-head';
      if (/blocks/.test(raw)) return `block.${l}`;
      return 'model';
    }
    case 'Config':
      return 'model';
    default:
      return null;
  }
}

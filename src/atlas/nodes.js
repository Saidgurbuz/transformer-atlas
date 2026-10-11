// The zoom hierarchy. A node id is dot-separated so it can live in the URL hash:
//   model · embedding · block.2 · attention.2 · head.2.1 · softmax.2.1 · mlp.2 ...
// Each pattern names its parameters (:l layer, :h head, :n which norm/residual).
// `keys(p, cfg)` lists the trace keys the scene is about (drives code highlights,
// the maths drawer and hover cards). `story: true` puts it on the guided path.

const T = (s) => () => s;

export const PATTERNS = [
  { pattern: 'model', title: T('Transformer'), parent: null, kind: 'resid', blurb: 'The whole decoder-only model, input to prediction.',
    scene: () => import('./scenes/model.js'), keys: () => ['emb.out', 'logits', 'probs'] },

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
    scene: () => import('./scenes/generation.js'), keys: () => ['logits'] },

  // Extras: reachable from the menu (and from related scenes)
  { pattern: 'tower', title: T('All tensors (v0.1)'), parent: () => 'model', extra: true, kind: 'resid', blurb: 'Every traced tensor as a 3D tower.',
    scene: () => import('./scenes/legacy.js').then((m) => m.legacy('explorer')), keys: () => [] },
  { pattern: 'stepper', title: T('Code ↔ forward pass'), parent: () => 'model', extra: true, kind: 'attn', blurb: 'Step through model.py line by line.',
    scene: () => import('./scenes/legacy.js').then((m) => m.legacy('stepper')), keys: () => [] },
  { pattern: 'cheatsheet', title: T('Interview cheat sheet'), parent: () => 'model', extra: true, kind: 'head', blurb: 'Formulas, shapes and flashcards.',
    scene: () => import('./scenes/cheatsheet.js'), keys: () => [] },
  { pattern: 'scaling', title: T('Params, FLOPs & memory'), parent: () => 'model', extra: true, kind: 'mlp', blurb: 'What real models cost.',
    scene: () => import('./scenes/scaling.js'), keys: () => [] },
  { pattern: 'lab', title: T('Ablation lab'), parent: () => 'model', extra: true, kind: 'o', blurb: 'Switch parts off and watch the prediction change.',
    scene: () => import('./scenes/lab.js'), keys: () => [] },
  { pattern: 'encdec', title: T('Encoder–decoder'), parent: () => 'model', extra: true, kind: 'pos', blurb: 'Cross-attention and the other Transformer families.',
    scene: () => import('./scenes/encdec.js'), keys: () => [] },
  { pattern: 'vision', title: T('Vision tokens'), parent: () => 'tokens', extra: true, kind: 'embed', blurb: 'How images become tokens: ViT patches to VLMs.',
    scene: () => import('./scenes/vision.js'), keys: () => [] },
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

export const EXTRAS = PATTERNS.filter((p) => p.extra).map((p) => p.pattern);

/** The node that best explains a trace key (for "zoom to this tensor"). */
export function nodeForKey(key, sel = { head: 0 }) {
  const m = /^L(\d+)\.(.*)$/.exec(key);
  const hh = sel.head ?? 0;
  if (!m) {
    if (key === 'emb.tok') return 'token-embedding';
    if (key === 'emb.pos' || key === 'alibi') return 'positional';
    if (key === 'emb.out') return 'embedding';
    if (key === 'mask') return `mask.0.${hh}`;
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

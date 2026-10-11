// Every page of the app. Each view lives in its own module and is loaded on
// demand, so a failing view cannot break the others.
//   group: navigation section · kind: component colour for the nav dot

export const GROUPS = ['Start', 'Input', 'Attention', 'Inside the block', 'Output', 'Experiments', 'Beyond', 'Review'];

export const VIEWS = [
  { id: 'overview', title: 'Architecture map', group: 'Start', kind: 'resid', load: () => import('./overview.js') },
  { id: 'stepper', title: 'Code ↔ forward pass', group: 'Start', kind: 'attn', load: () => import('./stepper.js') },
  { id: 'tokens', title: 'Tokens & embeddings', group: 'Input', kind: 'embed', load: () => import('./tokens.js') },
  { id: 'positions', title: 'Positional encoding', group: 'Input', kind: 'pos', load: () => import('./positions.js') },
  { id: 'attention', title: 'Self-attention (one head)', group: 'Attention', kind: 'q', load: () => import('./attention.js') },
  { id: 'multihead', title: 'Multi-head, GQA & MQA', group: 'Attention', kind: 'k', load: () => import('./multihead.js') },
  { id: 'masking', title: 'Masking', group: 'Attention', kind: 'mask', load: () => import('./masking.js') },
  { id: 'mlp', title: 'Feed-forward (MLP)', group: 'Inside the block', kind: 'mlp', load: () => import('./mlp.js') },
  { id: 'residual', title: 'Residual stream & norms', group: 'Inside the block', kind: 'norm', load: () => import('./residual.js') },
  { id: 'head', title: 'LM head & sampling', group: 'Output', kind: 'head', load: () => import('./head.js') },
  { id: 'generation', title: 'Generation & KV cache', group: 'Output', kind: 'v', load: () => import('./generation.js') },
  { id: 'lab', title: 'Ablation lab (A/B)', group: 'Experiments', kind: 'o', load: () => import('./lab.js') },
  { id: 'scaling', title: 'Params, FLOPs & memory', group: 'Experiments', kind: 'mlp', load: () => import('./scaling.js') },
  { id: 'encdec', title: 'Encoder–decoder & cross-attn', group: 'Beyond', kind: 'pos', load: () => import('./encdec.js') },
  { id: 'vision', title: 'Vision tokens (ViT → VLM)', group: 'Beyond', kind: 'embed', load: () => import('./vision.js') },
  { id: '_example', title: 'Example view', group: 'Hidden', kind: 'attn', hidden: true, load: () => import('./_example.js') },
  { id: '_demo_matmul', title: 'Demo: matmul animation', group: 'Hidden', kind: 'q', hidden: true, load: () => import('./_demo_matmul.js') },
  { id: '_demo_arcs', title: 'Demo: attention arcs', group: 'Hidden', kind: 'attn', hidden: true, load: () => import('./_demo_arcs.js') },
  { id: '_demo_flow', title: 'Demo: flow & motion kit', group: 'Hidden', kind: 'resid', hidden: true, load: () => import('./_demo_flow.js') },
  { id: 'cheatsheet', title: 'Interview cheat sheet', group: 'Review', kind: 'head', load: () => import('./cheatsheet.js') },
];

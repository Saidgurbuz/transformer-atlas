// Params, FLOPs & memory: a scrollable calculator for one config at a time.
// Every figure is arithmetic on the config. The parameter formula mirrors engine.paramSpecs() and
// is checked against it exactly (the loaded model against engine.countParams); the KV cache is
// checked against engine.cacheBytes. Published configs are the reported architectures; anything
// estimated is marked ≈. Reference scene (More menu): a scrollable document, see docs/ATLAS_GUIDE.md.
import { countParams, paramSpecs, cacheBytes, DEFAULT_CONFIG } from '../../engine/index.js';
import { h, s, texHTML, num, segmented, slider, sequential, onThemeChange, reducedMotion, showTip, hideTip } from '../../ui/index.js';
import { chip, appear, checkBadge, useStyle } from '../kit.js';

// ------------------------------------------------------------------ constants

// Published architectures. ctx is the trained context window: it sets the learned position table
// where one exists, and the default T for the attention cost and the KV cache.
const PUBLISHED = [
  { id: 'gpt2', name: 'GPT-2 small', ctx: 1024, cfg: { vocab_size: 50257, block_size: 1024, n_layer: 12, n_head: 12, n_kv_head: 12, d_model: 768, d_ff: 3072, pos_encoding: 'learned', norm: 'layernorm', norm_placement: 'pre', activation: 'gelu', tie_weights: true, bias: true } },
  { id: 'gpt2xl', name: 'GPT-2 XL', ctx: 1024, cfg: { vocab_size: 50257, block_size: 1024, n_layer: 48, n_head: 25, n_kv_head: 25, d_model: 1600, d_ff: 6400, pos_encoding: 'learned', norm: 'layernorm', norm_placement: 'pre', activation: 'gelu', tie_weights: true, bias: true } },
  { id: 'gpt3', name: 'GPT-3 175B', ctx: 2048, cfg: { vocab_size: 50257, block_size: 2048, n_layer: 96, n_head: 96, n_kv_head: 96, d_model: 12288, d_ff: 49152, pos_encoding: 'learned', norm: 'layernorm', norm_placement: 'pre', activation: 'gelu', tie_weights: true, bias: true } },
  { id: 'llama2-7b', name: 'LLaMA-2 7B', ctx: 4096, cfg: { vocab_size: 32000, block_size: 4096, n_layer: 32, n_head: 32, n_kv_head: 32, d_model: 4096, d_ff: 11008, pos_encoding: 'rope', norm: 'rmsnorm', norm_placement: 'pre', activation: 'swiglu', tie_weights: false, bias: false } },
  { id: 'llama2-70b', name: 'LLaMA-2 70B', ctx: 4096, cfg: { vocab_size: 32000, block_size: 4096, n_layer: 80, n_head: 64, n_kv_head: 8, d_model: 8192, d_ff: 28672, pos_encoding: 'rope', norm: 'rmsnorm', norm_placement: 'pre', activation: 'swiglu', tie_weights: false, bias: false } },
  { id: 'llama3-8b', name: 'LLaMA-3 8B', ctx: 8192, cfg: { vocab_size: 128256, block_size: 8192, n_layer: 32, n_head: 32, n_kv_head: 8, d_model: 4096, d_ff: 14336, pos_encoding: 'rope', norm: 'rmsnorm', norm_placement: 'pre', activation: 'swiglu', tie_weights: false, bias: false } },
  { id: 'mistral-7b', name: 'Mistral 7B', ctx: 8192, cfg: { vocab_size: 32000, block_size: 8192, n_layer: 32, n_head: 32, n_kv_head: 8, d_model: 4096, d_ff: 14336, pos_encoding: 'rope', norm: 'rmsnorm', norm_placement: 'pre', activation: 'swiglu', tie_weights: false, bias: false } },
];

const LOG_LO = 6, LOG_HI = 17;                                  // context axis: 64 … 131,072 tokens
const CTX_TICKS = [64, 256, 1024, 4096, 16384, 65536, 131072];
const GPUS = { a100: { name: 'A100', tflops: 312 }, h100: { name: 'H100', tflops: 989 } };   // dense bf16 peak
const MFU = 0.4;                                                // assumed utilisation of the peak
const GPU_BUDGET = 80e9;                                        // one 80 GB GPU: reference line
const STEP_MS = 230;                                            // one term of the parameter sum
const CH = { W: 540, H: 280, l: 56, r: 18, t: 28, b: 40 };      // chart box, CSS px
const WEIGHT_TIERS = [['fp32', 4], ['bf16', 2], ['int8', 1], ['int4', 0.5]];
const TRAIN_TIERS = [['bf16 weights', 2], ['bf16 gradients', 2], ['fp32 master weights', 4], ['Adam m (fp32)', 4], ['Adam v (fp32)', 4]];

const NOTES = [
  ['Two FLOPs per weight, forward', 'A forward pass costs ≈2N FLOPs per token and training ≈6N: the backward pass computes a gradient for the input and for the weight of every matmul.'],
  ['KV cache per token', '2 · L · H_kv · d_h · bytes, per sequence. Grouped-query attention divides it by H / H_kv: LLaMA-2 70B has 64 query heads and 8 KV heads, an 8× saving.'],
  ['Where attention overtakes', 'The score FLOPs 2LTd equal the weight FLOPs of all blocks at T ≈ (N_attn + N_MLP) / (L · d). For LLaMA-2 7B that is about 49k tokens.'],
  ['Adam with mixed precision', 'About 16 bytes per parameter before activations: bf16 weights and gradients (2 + 2), fp32 master weights (4), and Adam’s two moments (4 + 4).'],
  ['Chinchilla is about training compute', 'About 20 training tokens per parameter minimise loss for a fixed compute budget (Hoffmann et al., 2022). Serving cost favours smaller models trained for longer.'],
];

const FOOT = '≈ marks approximations. N counts every parameter, although the embedding lookup costs no FLOPs, so 2N slightly overstates the forward cost. Attention scores use the causal average, 2·L·T·d per token (Kaplan et al., 2020). Memory excludes activations, fragmentation and temporary buffers. GPU-days assume 40% of peak dense bf16 throughput (A100 312 TFLOP/s, H100 989 TFLOP/s). Published configs are the reported architectures; GPT-3’s vocabulary and tied head follow GPT-2, and Mistral’s sliding-window attention is not modelled.';

const CSS = `
.k-doc.sc-doc { padding: 0 24px 110px; }
.k-doc-inner.sc-inner { max-width: 1160px; gap: 26px; padding-top: 84px; }
.sc-head { display: grid; gap: 10px; }
.sc-title { font: 700 46px/1.05 var(--font-display); letter-spacing: -.02em; margin: 0; }
.sc-lede { font-size: 19px; line-height: 1.5; color: var(--fg-2); margin: 0; max-width: 76ch; }
.sc-bar { position: sticky; top: 66px; z-index: 4; display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; padding: 10px 12px; border-radius: 14px;
  background: color-mix(in srgb, var(--bg) 84%, transparent); backdrop-filter: blur(12px) saturate(1.2); -webkit-backdrop-filter: blur(12px) saturate(1.2);
  border: 1px solid color-mix(in srgb, var(--line-strong) 60%, transparent); box-shadow: var(--shadow-1); }
.sc-bar .segmented button { font-size: 14px; padding: 5px 11px; }
.sc-readout { margin-left: auto; font: 600 15px/1 var(--font-mono); color: var(--fg); white-space: nowrap; font-variant-numeric: tabular-nums; }
.sc-readout small { font: 500 13px/1 var(--font-mono); color: var(--fg-3); margin-left: 6px; }
.sc-spec { display: grid; gap: 6px; padding: 12px 16px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); font: 500 14px/1.5 var(--font-mono); color: var(--fg-2); }
.sc-spec b { color: var(--fg); font-weight: 600; }
.sc-sec { display: grid; gap: 18px; min-width: 0; padding: 26px 30px 28px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 92%, transparent);
  border: 1px solid color-mix(in srgb, var(--kind, var(--line-strong)) 26%, var(--line)); box-shadow: var(--shadow-1); }
.sc-kicker { font: 600 13px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .08em; }
.sc-h2 { font: 650 30px/1.18 var(--font-display); letter-spacing: -.01em; margin: 0; text-wrap: balance; }
.sc-cap { font-size: 18px; line-height: 1.55; color: var(--fg-2); margin: 0; max-width: 90ch; }
.sc-read { font-size: 16px; line-height: 1.55; color: var(--fg-2); margin: 0; }
.sc-read b { color: var(--fg); font-weight: 600; }

.sc-stack { display: flex; height: 44px; border-radius: 10px; overflow: hidden; background: var(--surface-2); border: 1px solid var(--line); }
.sc-seg { display: block; height: 100%; width: 0; flex: none; background: var(--c); transition: width .6s cubic-bezier(.65,0,.35,1); }
.sc-seg + .sc-seg { box-shadow: -1px 0 0 color-mix(in srgb, var(--surface) 85%, transparent); }
.sc-table { display: grid; gap: 2px; min-width: 0; }
.sc-row, .sc-thead { display: grid; grid-template-columns: 12px minmax(0, 0.9fr) minmax(0, 1.6fr) minmax(0, 1.1fr) 156px 60px;
  grid-template-areas: "sw name tex sub count share"; gap: 14px; align-items: center; padding: 9px 8px; border-radius: 10px; min-width: 0; }
.sc-row { transition: opacity .35s ease, background .2s ease; }
.sc-row:hover { background: var(--surface-2); }
.sc-row.pending { opacity: .2; }
.sc-thead { padding: 2px 8px 0; font: 600 12px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .06em; }
.sc-sw { grid-area: sw; width: 12px; height: 12px; border-radius: 3px; background: var(--c); }
.sc-name { grid-area: name; font: 600 16px/1.3 var(--font-body); color: var(--fg); min-width: 0; }
.sc-name small { display: block; font: 500 13px/1.35 var(--font-mono); color: var(--fg-3); margin-top: 2px; }
.sc-tex-wrap { grid-area: tex; min-width: 0; display: grid; gap: 2px; }
.sc-tex { font-size: 18px; color: var(--fg); overflow-x: auto; overflow-y: hidden; white-space: nowrap; }
.sc-tex2 { font-size: 15px; color: var(--fg-3); overflow-x: auto; overflow-y: hidden; }
.sc-subst { grid-area: sub; font: 500 13px/1.45 var(--font-mono); color: var(--fg-2); min-width: 0; overflow-wrap: anywhere; }
.sc-count { grid-area: count; font: 600 16px/1 var(--font-mono); text-align: right; font-variant-numeric: tabular-nums; color: var(--fg); }
.sc-share { grid-area: share; font: 600 15px/1 var(--font-mono); text-align: right; color: var(--c); font-variant-numeric: tabular-nums; }
.sc-sum { margin-top: 6px; padding-top: 14px; border-top: 1px solid var(--line-strong); border-radius: 0; }
.sc-sum:hover { background: none; }
.sc-sum .sc-count { font-size: 20px; }
.sc-sum .sc-name { font-size: 17px; }

.sc-two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.04fr); gap: 26px; align-items: start; min-width: 0; }
.sc-col { display: grid; gap: 14px; min-width: 0; align-content: start; }
.sc-eq { display: grid; gap: 10px; padding: 14px 16px; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--line); min-width: 0; }
.sc-eq .sc-tex { font-size: 21px; }
.sc-nums { display: flex; flex-wrap: wrap; gap: 4px 14px; font: 500 15px/1.4 var(--font-mono); color: var(--fg-2); font-variant-numeric: tabular-nums; }
.sc-nums b { color: var(--fg); font-weight: 600; }
.sc-nums .sc-sc { color: var(--c-attn); }
.sc-bars { display: flex; height: 22px; border-radius: 7px; overflow: hidden; background: var(--surface-2); border: 1px solid var(--line); }
.sc-bars i { display: block; height: 100%; width: 0; flex: none; transition: width .6s cubic-bezier(.65,0,.35,1); }
.sc-legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font: 500 13px/1.3 var(--font-mono); color: var(--fg-2); }
.sc-legend span { display: inline-flex; align-items: center; gap: 7px; }
.sc-legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; background: var(--c); flex: none; }
.sc-chart { min-width: 0; }
.sc-chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.sc-chart text { font: 500 12px var(--font-mono); fill: var(--fg-3); }
.sc-mark { transition: transform .6s cubic-bezier(.65,0,.35,1); }
.sc-mark text { fill: var(--fg); font-weight: 600; }

.sc-mem { display: grid; gap: 12px; min-width: 0; }
.sc-mrow { display: grid; grid-template-columns: minmax(180px, 240px) minmax(0, 1fr) 124px; gap: 16px; align-items: center; min-width: 0; }
.sc-mlab { display: grid; gap: 6px; justify-items: start; min-width: 0; font: 600 16px/1.3 var(--font-body); color: var(--fg); }
.sc-mlab small { font: 500 13px/1.35 var(--font-mono); color: var(--fg-3); }
.sc-mx { display: grid; gap: 8px; justify-items: start; min-width: 0; }
.sc-track { position: relative; height: 30px; border-radius: 8px; background: var(--surface-2); border: 1px solid var(--line); }
.sc-fill { position: absolute; top: 0; bottom: 0; left: 0; width: 0; border-radius: 7px; transition: width .6s cubic-bezier(.65,0,.35,1), left .6s cubic-bezier(.65,0,.35,1); }
.sc-track .sc-fill + .sc-fill { box-shadow: -1px 0 0 color-mix(in srgb, var(--surface) 80%, transparent); }
.sc-ref { position: absolute; top: -8px; bottom: -8px; left: 0; border-left: 1px dashed var(--fg-2); transition: left .6s cubic-bezier(.65,0,.35,1); pointer-events: none; }
.sc-ref span { position: absolute; top: -17px; left: 5px; font: 600 12px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.sc-mval { font: 600 16px/1 var(--font-mono); text-align: right; font-variant-numeric: tabular-nums; color: var(--fg); white-space: nowrap; }

.sc-ctrl { display: flex; flex-wrap: wrap; gap: 10px 26px; align-items: center; }
.sc-ctrl > div { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.sc-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.sc-stat { display: grid; gap: 10px; align-content: start; padding: 16px 18px; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--line); min-width: 0; }
.sc-stat .lab { font: 600 13px/1.2 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .06em; }
.sc-stat .val { font: 700 30px/1.15 var(--font-mono); font-variant-numeric: tabular-nums; color: var(--fg); overflow-wrap: anywhere; }
.sc-stat .val small { font: 600 16px/1 var(--font-mono); color: var(--fg-3); margin-left: 6px; }
.sc-stat .sub { font-size: 15px; line-height: 1.45; color: var(--fg-2); }
.sc-eqc { padding: 14px 16px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); overflow-x: auto; }

.sc-notes { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px 30px; padding: 24px 30px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); }
.sc-note { display: grid; gap: 4px; align-content: start; min-width: 0; }
.sc-note b { font: 650 17px/1.3 var(--font-display); color: var(--fg); }
.sc-note p { margin: 0; font-size: 16px; line-height: 1.55; color: var(--fg-2); }
.sc-foot { margin: 0; font-size: 14px; line-height: 1.6; color: var(--fg-3); max-width: 110ch; }

@media (max-width: 960px) {
  .sc-two, .sc-notes { grid-template-columns: minmax(0, 1fr); }
  .sc-stats { grid-template-columns: minmax(0, 1fr); }
  .sc-title { font-size: 36px; }
  .sc-sec { padding: 22px 18px; }
  .sc-h2 { font-size: 25px; }
  .sc-row { grid-template-columns: 12px minmax(0, 1fr) auto; grid-template-areas: "sw name share" ". tex tex" ". sub count"; }
  .sc-thead { display: none; }
  .sc-mrow { grid-template-columns: minmax(0, 1fr); gap: 8px; }
  .sc-readout { margin-left: 0; }
  .sc-bar { top: 64px; }
}
`;

// ------------------------------------------------------------------ pure arithmetic

const fullCfg = (cfg) => ({ ...DEFAULT_CONFIG, ...cfg });

// The same tensors as engine.paramSpecs(): embeddings, learned positions, per-layer attention
// (Q, K, V, O) and MLP (up, gate with SwiGLU, down) with biases when cfg.bias, norms (weight, and
// bias for LayerNorm with bias), a final norm for pre-LN, and an LM head unless tied.
export function arch(cfg) {
  const d = cfg.d_model, H = cfg.n_head, Hkv = cfg.n_kv_head, F = cfg.d_ff, V = cfg.vocab_size, L = cfg.n_layer;
  const dh = d / H;
  const bias = !!cfg.bias, swiglu = cfg.activation === 'swiglu', tied = !!cfg.tie_weights;
  const learned = cfg.pos_encoding === 'learned', ctx = cfg.block_size;
  const lnBias = cfg.norm === 'layernorm' && bias;
  const pre = cfg.norm_placement === 'pre';
  const nNorms = 2 * L + (pre ? 1 : 0);
  const attnWl = d * H * dh + 2 * d * Hkv * dh + H * dh * d;          // per layer, weights
  const attnBl = bias ? H * dh + 2 * Hkv * dh + d : 0;                // per layer, biases
  const mlpWl = (swiglu ? 3 : 2) * d * F;                             // per layer, weights
  const mlpBl = bias ? (swiglu ? 2 * F : F) + d : 0;                  // per layer, biases
  const parts = {
    embed: V * d,
    pos: learned ? ctx * d : 0,
    attn: L * (attnWl + attnBl),
    mlp: L * (mlpWl + mlpBl),
    norms: nNorms * d * (lnBias ? 2 : 1),
    head: tied ? 0 : V * d,
  };
  const N = parts.embed + parts.pos + parts.attn + parts.mlp + parts.norms + parts.head;
  return {
    d, H, Hkv, dh, F, V, L, ctx, bias, swiglu, tied, learned, lnBias, pre, nNorms, attnWl, attnBl, mlpWl, mlpBl,
    normName: cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm', parts, N,
  };
}

// The engine's own count: the sum of every shape in paramSpecs(cfg). No tensors are allocated.
export const engineParams = (cfg) => paramSpecs(fullCfg(cfg)).reduce((acc, [, shape]) => acc + shape.reduce((x, y) => x * y, 1), 0);

// FLOPs per token. A matmul costs 2 FLOPs per weight; the score term is the causal average.
export const scoreFlops = (a, T) => 2 * a.L * T * a.d;
export const forwardFlops = (a, T) => 2 * a.N + scoreFlops(a, T);
export const trainingFlops = (a, T) => 3 * forwardFlops(a, T);
// Attention sub-layer (projections and scores) against the MLP sub-layer, per token at context T.
export const attnShare = (a, T) => { const A = 2 * a.parts.attn + scoreFlops(a, T), M = 2 * a.parts.mlp; return A / (A + M); };
// The T where score FLOPs equal the weight FLOPs of all blocks: 2LTd = 2 (attn + mlp).
export const scoresMatchBlocks = (a) => (a.parts.attn + a.parts.mlp) / (a.L * a.d);

// KV cache in bf16 (2 bytes): K and V, every layer, every KV head, every token of the batch.
const kvBytes = (cfg, T, batch) => cacheBytes(cfg, T * batch, 2);
export const kvFormula = (a, T, batch) => 2 * a.L * a.Hkv * a.dh * T * batch * 2;

// Chinchilla: D ≈ 20 N tokens (tpp tokens per parameter), C ≈ 6 N D, and GPU-days at MFU.
export function chinchilla(a, tpp, gpu) {
  const D = tpp * a.N, C = 6 * a.N * D, perGpu = GPUS[gpu].tflops * 1e12 * MFU;
  return { D, C, gpuDays: C / (perGpu * 86400) };
}

// The six terms of the parameter sum, with their TeX, the substituted numbers and the count.
export function paramRows(a) {
  const k = a.swiglu ? 3 : 2;
  return [
    { kind: 'embed', name: 'Token embeddings', sub: 'W_E', tex: 'V\\,d', tex2: '',
      subst: `${fmtInt(a.V)} × ${fmtInt(a.d)}`, n: a.parts.embed },
    { kind: 'pos', name: 'Learned positions', sub: 'W_P', tex: a.learned ? 'n_{\\mathrm{ctx}}\\,d' : '0',
      tex2: a.learned ? '' : '\\text{no learned table}', subst: a.learned ? `${fmtInt(a.ctx)} × ${fmtInt(a.d)}` : '0', n: a.parts.pos },
    { kind: 'attn', name: 'Attention', sub: 'W_Q, W_K, W_V, W_O',
      tex: 'L\\,(d\\,H\\,d_h + 2\\,d\\,H_{kv}\\,d_h + H\\,d_h\\,d)',
      tex2: a.bias ? '+\\; L\\,(H\\,d_h + 2\\,H_{kv}\\,d_h + d)' : '',
      subst: `${a.L} × (${fmtInt(a.attnWl)}${a.bias ? ` + ${fmtInt(a.attnBl)}` : ''})`, n: a.parts.attn },
    { kind: 'mlp', name: 'MLP', sub: a.swiglu ? 'W_gate, W_up, W_down' : 'W_up, W_down',
      tex: `${k}\\,L\\,d\\,F`, tex2: a.bias ? `+\\; L\\,(${a.swiglu ? '2F' : 'F'} + d)` : '',
      subst: `${a.L} × (${fmtInt(a.mlpWl)}${a.bias ? ` + ${fmtInt(a.mlpBl)}` : ''})`, n: a.parts.mlp },
    { kind: 'norm', name: 'Norms', sub: `${a.normName}: ${a.lnBias ? 'γ, β' : 'γ'}`, tex: a.pre ? '(2L{+}1)\\,d' : '2L\\,d',
      tex2: a.lnBias ? '\\times\\, 2' : '', subst: `${a.nNorms} × ${fmtInt(a.d)}${a.lnBias ? ' × 2' : ''}`, n: a.parts.norms },
    { kind: 'head', name: 'LM head', sub: a.tied ? 'tied to W_E' : 'W_U', tex: a.tied ? '0' : 'V\\,d',
      tex2: a.tied ? '\\text{shares } W_E' : '', subst: a.tied ? 'no new weights' : `${fmtInt(a.V)} × ${fmtInt(a.d)}`, n: a.parts.head },
  ];
}

// ------------------------------------------------------------------ formatting

const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
function fmtShort(n) {
  for (const [v, u] of [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) if (Math.abs(n) >= v) return `${(n / v).toPrecision(3)} ${u}`;
  return fmtInt(n);
}
function fmtGB(bytes) {
  const g = bytes / 1e9;
  if (g >= 1000) return `${(g / 1000).toPrecision(3)} TB`;
  if (g >= 0.01) return `${g.toPrecision(3)} GB`;
  return `${(bytes / 1e6).toPrecision(2)} MB`;
}
function fmtFlop(x) {
  for (const [v, u] of [[1e12, 'T'], [1e9, 'G'], [1e6, 'M']]) if (x >= v) return `${(x / v).toPrecision(3)} ${u}FLOP`;
  return `${fmtInt(x)} FLOP`;
}
function fmtGpu(days) {
  return days >= 1 ? `${fmtInt(days)} GPU-days` : `${(days * 24).toPrecision(2)} GPU-hours`;
}
const pct = (x) => (x > 0 && x < 0.001 ? '<0.1%' : `${(100 * x).toFixed(1)}%`);
const fmtTok = (T) => (T >= 1024 && T % 1024 === 0 ? `${T / 1024}k` : fmtInt(T));
const xOf = (T) => CH.l + ((Math.log2(T) - LOG_LO) / (LOG_HI - LOG_LO)) * (CH.W - CH.l - CH.r);
const yOf = (p) => CH.H - CH.b - p * (CH.H - CH.t - CH.b);
const tierColour = (k) => sequential(0.3 + 0.12 * k);

// ------------------------------------------------------------------ scene

export default {
  mount(el, ctx) {
    useStyle('scaling', CSS);
    let alive = true, timers = [];
    const later = (fn, ms) => { timers.push(setTimeout(() => { if (alive) fn(); }, ms)); };
    const clearTimers = () => { timers.forEach(clearTimeout); timers = []; };
    const moving = () => !reducedMotion();

    // selection: the app's loaded model, or a published configuration
    let choice = 'app', sel = null, a = null, lastKey = '';
    let T = 64, batch = 1, gpu = 'a100', tpp = 20;
    const pick = () => {
      if (choice === 'app') {
        const cfg = ctx.state.model?.config ?? ctx.state.config;
        return { name: 'App model', cfg, ctxDefault: cfg.block_size, loaded: true };
      }
      const p = PUBLISHED.find((x) => x.id === choice) ?? PUBLISHED[0];
      return { name: p.name, cfg: fullCfg(p.cfg), ctxDefault: p.ctx, loaded: false };
    };
    sel = pick();
    a = arch(sel.cfg);
    T = sel.ctxDefault;

    // counting animation: one tween per element, superseded by the next target
    function tween(node, to, fmt, { from, dur = 640, instant = false } = {}) {
      node._tok = (node._tok || 0) + 1;
      const tok = node._tok;
      const start = from ?? (typeof node._v === 'number' ? node._v : to);
      if (instant || !moving() || start === to) { node._v = to; node.textContent = fmt(to); return; }
      const t0 = performance.now();
      const step = (now) => {
        if (!alive || node._tok !== tok) return;
        const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
        node._v = start + (to - start) * e;
        node.textContent = fmt(node._v);
        if (p < 1) requestAnimationFrame(step);
        else node._v = to;
      };
      requestAnimationFrame(step);
    }

    // ---------------------------------------------------------- DOM, built once
    const R = {};

    R.readN = h('span', null, '');
    R.readout = h('div', { class: 'sc-readout' }, R.readN, h('small', null, 'parameters'));
    const modelSeg = segmented({
      ariaLabel: 'Model',
      value: choice,
      options: [{ value: 'app', label: 'App model', title: 'The model loaded in the app' },
        ...PUBLISHED.map((p) => ({ value: p.id, label: p.name, title: `${p.name}, published configuration` }))],
      onChange: (v) => { choice = v; T = pick().ctxDefault; ctxSlider.set(Math.log2(T)); update(); },
    });
    const bar = h('div', { class: 'sc-bar' }, h('span', { class: 'k-picker-label' }, 'model'), modelSeg.el, R.readout);
    R.spec = h('div', { class: 'sc-spec' });

    const head = h('header', { class: 'sc-head' },
      h('div', { class: 'k-eyebrow' }, chip('Scaling', 'mlp')),
      h('h1', { class: 'sc-title' }, 'Params, FLOPs & memory'),
      h('p', { class: 'sc-lede' }, 'Every figure here is arithmetic on a config. Pick a model; the parameter formula is checked against the engine’s own parameter list.'));

    // ---- 1 · parameters
    R.h2a = h('h2', { class: 'sc-h2' });
    R.stack = h('div', { class: 'sc-stack' });
    R.total = h('span', null, '0');
    R.checkSlot = h('div', { class: 'sc-subst' });
    R.table = h('div', { class: 'sc-table' });
    const thead = h('div', { class: 'sc-thead' },
      h('span', { style: { gridArea: 'name' } }, 'Term'),
      h('span', { style: { gridArea: 'tex' } }, 'Formula'),
      h('span', { style: { gridArea: 'sub' } }, 'Substituted'),
      h('span', { style: { gridArea: 'count', textAlign: 'right' } }, 'Count'),
      h('span', { style: { gridArea: 'share', textAlign: 'right' } }, 'Share'));
    R.sum = h('div', { class: 'sc-row sc-sum' },
      h('span', { style: { gridArea: 'sw' } }),
      h('div', { class: 'sc-name' }, 'Total N', h('small', null, 'the sum of the rows above')),
      h('div', { class: 'sc-tex-wrap' }, h('div', { class: 'sc-tex', html: texHTML('N = \\sum_k n_k') })),
      R.checkSlot,
      h('div', { class: 'sc-count' }, R.total),
      h('div', { class: 'sc-share' }, '100%'));
    const sec1 = h('section', { class: 'sc-sec', style: { '--kind': 'var(--c-mlp)' } },
      h('div', { class: 'sc-kicker' }, '1 · Parameters'),
      R.h2a,
      h('p', { class: 'sc-cap' }, 'Token embeddings and the LM head are V × d tables. Each layer holds four attention projections and an MLP of two d × F matrices (three with SwiGLU). Biases and norms are small.'),
      R.stack,
      R.table);

    // ---- 2 · compute per token
    R.h2b = h('h2', { class: 'sc-h2' });
    const ctxSlider = slider({
      id: 'scaling-ctx', label: 'Context length T', min: LOG_LO, max: LOG_HI, step: 0.05, value: Math.log2(T),
      format: (v) => `${fmtInt(Math.round(2 ** v))} tokens`,
      onInput: (v) => { T = Math.round(2 ** v); update(); },
    });
    const grey = 'color-mix(in srgb, var(--fg-3) 50%, var(--surface-3))';
    R.fTex = h('div', { class: 'sc-tex', html: texHTML('\\mathrm{FLOP}_{\\mathrm{fwd}}/\\mathrm{token} \\approx 2N + 2\\,L\\,T\\,d') });
    R.fW = h('b'); R.fS = h('b', { class: 'sc-sc' }); R.fT = h('b');
    R.fNums = h('div', { class: 'sc-nums' }, h('span', null, '2N = ', R.fW), h('span', null, '+ 2LTd = ', R.fS), h('span', null, '= ', R.fT));
    R.tTex = h('div', { class: 'sc-tex', html: texHTML('\\mathrm{FLOP}_{\\mathrm{train}}/\\mathrm{token} \\approx 6N + 6\\,L\\,T\\,d') });
    R.tW = h('b'); R.tS = h('b', { class: 'sc-sc' }); R.tT = h('b');
    R.tNums = h('div', { class: 'sc-nums' }, h('span', null, '6N = ', R.tW), h('span', null, '+ 6LTd = ', R.tS), h('span', null, '= ', R.tT));
    R.bW = h('i', { style: { background: grey } });
    R.bS = h('i', { style: { background: 'var(--c-attn)' } });
    const flopLegend = h('div', { class: 'sc-legend' },
      h('span', null, h('i', { style: { '--c': grey } }), '2N, weights'),
      h('span', null, h('i', { style: { '--c': 'var(--c-attn)' } }), '2LTd, attention scores'));
    R.svg = s('svg', { viewBox: `0 0 ${CH.W} ${CH.H}`, role: 'img', 'aria-label': 'Attention and MLP share of block FLOPs against context length' });
    R.chartG = s('g');
    R.mark = s('g', { class: 'sc-mark' });
    R.markLine = s('line', { x1: 0, x2: 0, y1: yOf(1), y2: yOf(0), style: 'stroke: var(--fg); stroke-width: 1.5; stroke-dasharray: 3 3' });
    R.markDot = s('circle', { cx: 0, cy: yOf(0.5), r: 4.5, style: 'fill: var(--c-attn); stroke: var(--surface); stroke-width: 2' });
    R.markTxt = s('text', { x: 8, y: yOf(1) + 14 }, '');
    R.mark.append(R.markLine, R.markDot, R.markTxt);
    R.svg.append(R.chartG, R.mark);
    R.svg.addEventListener('pointermove', (ev) => {
      const r = R.svg.getBoundingClientRect();
      const xv = ((ev.clientX - r.left) / r.width) * CH.W;
      const e = LOG_LO + ((xv - CH.l) / (CH.W - CH.l - CH.r)) * (LOG_HI - LOG_LO);
      const Tt = 2 ** Math.min(LOG_HI, Math.max(LOG_LO, e));
      const A = attnShare(a, Tt);
      showTip(ev.clientX, ev.clientY, `T = <b>${fmtInt(Math.round(Tt))}</b> · attention ${pct(A)} · MLP ${pct(1 - A)}`);
    });
    R.svg.addEventListener('pointerleave', hideTip);
    const chartLegend = h('div', { class: 'sc-legend' },
      h('span', null, h('i', { style: { '--c': 'var(--c-mlp)' } }), 'MLP sub-layer'),
      h('span', null, h('i', { style: { '--c': 'var(--c-attn)' } }), 'attention sub-layer: projections and scores'));
    R.scoreLine = h('p', { class: 'sc-read' });
    const sec2 = h('section', { class: 'sc-sec', style: { '--kind': 'var(--c-attn)' } },
      h('div', { class: 'sc-kicker' }, '2 · Compute per token'),
      R.h2b,
      h('p', { class: 'sc-cap' }, 'A matmul costs about 2 FLOPs per weight per token. The score matrix QKᵀ and the weighted sum over V add 2·L·T·d, so attention grows with context while the weights do not. The chart compares block FLOPs only.'),
      h('div', { class: 'sc-two' },
        h('div', { class: 'sc-col' },
          ctxSlider.el,
          h('div', { class: 'sc-eq' }, R.fTex, R.fNums),
          h('div', { class: 'sc-eq' }, R.tTex, R.tNums),
          h('div', { class: 'sc-bars' }, R.bW, R.bS),
          flopLegend),
        h('div', { class: 'sc-col' }, h('div', { class: 'sc-chart' }, R.svg), chartLegend, R.scoreLine)));

    // ---- 3 · memory
    R.h2c = h('h2', { class: 'sc-h2' });
    const memBatch = segmented({
      ariaLabel: 'Batch size', value: batch,
      options: [1, 8, 32, 128].map((v) => ({ value: v, label: String(v) })),
      onChange: (v) => { batch = v; update(); },
    });
    R.kvCheck = h('div', { class: 'sc-subst' });
    function makeRow({ label, sub, extra = null, fills = 1, refLabel = false }) {
      const fillEls = Array.from({ length: fills }, () => h('i', { class: 'sc-fill' }));
      const ref = h('i', { class: 'sc-ref' }, refLabel ? h('span', null, '80 GB, one GPU') : null);
      const trackEl = h('div', { class: 'sc-track' }, ...fillEls, ref);
      const val = h('span', null, '');
      const rowEl = h('div', { class: 'sc-mrow' },
        h('div', { class: 'sc-mlab' }, h('span', null, label), h('small', null, sub), extra),
        trackEl,
        h('div', { class: 'sc-mval' }, val));
      return { el: rowEl, fills: fillEls, ref, val, sub: rowEl.querySelector('.sc-mlab small') };
    }
    const memRows = {
      weights: WEIGHT_TIERS.map(([name, bytes], k) => makeRow({
        label: `${name} weights`, sub: `${bytes} ${bytes === 1 ? 'byte' : 'bytes'} per parameter`, refLabel: k === 0,
      })),
      kv: makeRow({ label: 'KV cache, bf16', sub: '', extra: h('div', { class: 'sc-mx' }, memBatch.el, R.kvCheck) }),
      train: makeRow({ label: 'Training with Adam', sub: '16 bytes per parameter', fills: TRAIN_TIERS.length }),
    };
    R.trainSw = TRAIN_TIERS.map(([name, bytes]) => h('i'));
    R.trainLegend = h('div', { class: 'sc-legend' }, TRAIN_TIERS.map(([name, bytes], k) =>
      h('span', null, R.trainSw[k], `${name}, ${bytes} B`)));
    const sec3 = h('section', { class: 'sc-sec', style: { '--kind': 'var(--c-norm)' } },
      h('div', { class: 'sc-kicker' }, '3 · Memory'),
      R.h2c,
      h('p', { class: 'sc-cap' }, 'Weights take N × bytes per value. The KV cache holds one key and one value per layer, per KV head, per token and per sequence in the batch. Adam keeps fp32 master weights and two fp32 moments.'),
      h('div', { class: 'sc-mem' },
        ...memRows.weights.map((r) => r.el), memRows.kv.el, memRows.train.el, R.trainLegend));

    // ---- 4 · training compute
    R.h2d = h('h2', { class: 'sc-h2' });
    const tppSeg = segmented({
      ariaLabel: 'Tokens per parameter', value: tpp,
      options: [5, 20, 100, 300].map((v) => ({ value: v, label: String(v) })),
      onChange: (v) => { tpp = v; update(); },
    });
    const gpuSeg = segmented({
      ariaLabel: 'GPU', value: gpu,
      options: Object.entries(GPUS).map(([k, g]) => ({ value: k, label: g.name, title: `${g.name}: ${g.tflops} TFLOP/s dense bf16` })),
      onChange: (v) => { gpu = v; update(); },
    });
    R.cDlab = h('span', null, '');
    R.cD = h('span', null, '');
    R.cDsub = h('div', { class: 'sub' });
    R.cC = h('div', { class: 'val' });
    R.cG = h('span', null, '');
    R.cGsub = h('div', { class: 'sub' });
    R.cEq = h('div', { class: 'sc-eqc' });
    const sec4 = h('section', { class: 'sc-sec', style: { '--kind': 'var(--c-resid)' } },
      h('div', { class: 'sc-kicker' }, '4 · Training compute'),
      R.h2d,
      h('p', { class: 'sc-cap' }, 'C ≈ 6ND: a forward and backward pass cost about 6 FLOPs per parameter per token. GPU-days divide C by the peak dense bf16 rate times 40% utilisation. Twenty tokens per parameter is the Chinchilla rule; 300 is about LLaMA-2 7B (2T tokens).'),
      h('div', { class: 'sc-ctrl' },
        h('div', null, h('span', { class: 'k-picker-label' }, 'tokens per parameter'), tppSeg.el),
        h('div', null, h('span', { class: 'k-picker-label' }, 'GPU'), gpuSeg.el)),
      h('div', { class: 'sc-stats' },
        h('div', { class: 'sc-stat' }, h('div', { class: 'lab' }, R.cDlab), h('div', { class: 'val' }, R.cD, h('small', null, 'tokens')), R.cDsub),
        h('div', { class: 'sc-stat' }, h('div', { class: 'lab' }, 'Compute C ≈ 6ND'), R.cC, h('div', { class: 'sub' }, 'FLOPs, forward plus backward')),
        h('div', { class: 'sc-stat' }, h('div', { class: 'lab' }, 'GPU time'), h('div', { class: 'val' }, R.cG), R.cGsub)),
      R.cEq);

    const notes = h('section', { class: 'sc-notes' },
      NOTES.map(([t, body]) => h('div', { class: 'sc-note' }, h('b', null, t), h('p', null, body))));
    const foot = h('p', { class: 'sc-foot' }, FOOT);

    el.append(h('div', { class: 'k-doc sc-doc' },
      h('div', { class: 'k-doc-inner sc-inner' }, head, bar, R.spec, sec1, sec2, sec3, sec4, notes, foot)));

    // ---------------------------------------------------------- paint
    function paintHeader() {
      const cfg = sel.cfg, b = (x) => h('b', null, String(x));
      tween(R.readN, a.N, fmtShort);
      const act = a.swiglu ? 'SwiGLU' : cfg.activation === 'relu' ? 'ReLU' : 'GELU';
      R.spec.replaceChildren(
        h('div', null, 'V ', b(fmtInt(a.V)), '   ctx ', b(fmtInt(a.ctx)), '   d ', b(a.d), '   L ', b(a.L),
          '   H ', b(a.H), '   H_kv ', b(a.Hkv), '   d_h ', b(a.dh), '   F ', b(fmtInt(a.F))),
        h('div', null, `${act} MLP, ${a.swiglu ? '3' : '2'}dF · ${a.normName}, ${a.pre ? 'pre' : 'post'}-norm · ${cfg.pos_encoding} positions · ${a.tied ? 'tied' : 'untied'} head · ${a.bias ? 'biases' : 'no biases'}`),
        sel.loaded ? h('div', null, `loaded in the app: ${ctx.state.preset?.title ?? 'random weights'}`) : null);
    }

    function rowEl(r, pending) {
      const count = h('div', { class: 'sc-count' }, pending ? '0' : fmtInt(r.n));
      const row = h('div', { class: ['sc-row', pending && 'pending'] },
        h('i', { class: 'sc-sw', style: { '--c': `var(--c-${r.kind})` } }),
        h('div', { class: 'sc-name' }, r.name, h('small', null, r.sub)),
        h('div', { class: 'sc-tex-wrap' },
          h('div', { class: 'sc-tex', html: texHTML(r.tex) }),
          r.tex2 ? h('div', { class: 'sc-tex2', html: texHTML(r.tex2) }) : null),
        h('div', { class: 'sc-subst' }, r.subst),
        count,
        h('div', { class: 'sc-share', style: { '--c': `var(--c-${r.kind})` } }, pct(r.n / a.N)));
      row._count = count;
      return row;
    }

    // Rebuilds the term rows and the stacked bar when the config changes; the sum adds up term by term.
    function buildParams(animate) {
      const rows = paramRows(a);
      R.h2a.innerHTML = `MLP blocks hold <b style="color:var(--c-mlp)">${pct(a.parts.mlp / a.N)}</b> of the parameters, attention <b style="color:var(--c-attn)">${pct(a.parts.attn / a.N)}</b>.`;
      R.checkSlot.replaceChildren(checkBadge(a.N, sel.loaded ? countParams(ctx.state.model) : engineParams(sel.cfg), {
        label: sel.loaded ? 'equals engine.countParams' : 'equals engine.paramSpecs', tol: 0,
      }));
      const segs = rows.map((r) => h('i', { class: 'sc-seg', style: { '--c': `var(--c-${r.kind})` } }));
      R.stack.replaceChildren(...segs);
      void R.stack.offsetWidth;
      const applyWidths = () => segs.forEach((sg, k) => {
        sg.style.width = `${(100 * rows[k].n) / a.N}%`;
        sg.style.transitionDelay = animate ? `${(k * STEP_MS) / 1000}s` : '0s';
        sg.style.transition = moving() ? '' : 'none';
      });
      if (animate) requestAnimationFrame(applyWidths); else applyWidths();

      const rowEls = rows.map((r) => rowEl(r, animate));
      R.table.replaceChildren(thead, ...rowEls, R.sum);
      if (animate) {
        R.total._v = 0;
        R.total.textContent = '0';
        let run = 0;
        rows.forEach((r, k) => later(() => {
          rowEls[k].classList.remove('pending');
          tween(rowEls[k]._count, r.n, fmtInt, { from: 0, dur: 420 });
          run += r.n;
          tween(R.total, run, fmtInt, { dur: 420 });
        }, k * STEP_MS));
      } else {
        rowEls.forEach((row, k) => { row.classList.remove('pending'); tween(row._count, rows[k].n, fmtInt, { instant: true }); });
        tween(R.total, a.N, fmtInt, { instant: true });
      }
    }

    function drawChart() {
      const xs = [], ms = [], as = [];
      for (let e = LOG_LO; e <= LOG_HI + 1e-9; e += 0.25) {
        const A = attnShare(a, 2 ** e);
        xs.push(xOf(2 ** e)); as.push(A); ms.push(1 - A);
      }
      const n = xs.length, P = (x, p) => `${x.toFixed(1)},${yOf(p).toFixed(1)}`;
      const mlpArea = [`M${P(xs[0], 0)}`, ...xs.map((x, i) => `L${P(x, ms[i])}`), `L${P(xs[n - 1], 0)}`, 'Z'].join(' ');
      const attArea = [`M${P(xs[0], 1)}`, ...xs.map((x) => `L${P(x, 1)}`),
        ...xs.map((x, i) => [x, i]).reverse().map(([x, i]) => `L${P(x, ms[i])}`), 'Z'].join(' ');
      const edge = xs.map((x, i) => `${i ? 'L' : 'M'}${P(x, ms[i])}`).join(' ');
      const kids = [];
      [0, 0.25, 0.5, 0.75, 1].forEach((p) => {
        kids.push(s('line', { x1: CH.l, x2: CH.W - CH.r, y1: yOf(p), y2: yOf(p), style: p === 0.5 ? 'stroke: var(--fg-3); stroke-dasharray: 4 4' : 'stroke: var(--line)' }));
        kids.push(s('text', { x: CH.l - 8, y: yOf(p) + 4, 'text-anchor': 'end' }, `${p * 100}%`));
      });
      CTX_TICKS.forEach((Tt) => kids.push(s('text', { x: xOf(Tt), y: CH.H - CH.b + 17, 'text-anchor': 'middle' }, fmtTok(Tt))));
      kids.push(s('path', { d: mlpArea, style: 'fill: var(--c-mlp); fill-opacity: .85' }));
      kids.push(s('path', { d: attArea, style: 'fill: var(--c-attn); fill-opacity: .85' }));
      kids.push(s('path', { d: edge, style: 'fill: none; stroke: var(--surface); stroke-width: 1.5' }));
      kids.push(s('text', { x: (CH.l + CH.W - CH.r) / 2, y: CH.H - 4, 'text-anchor': 'middle' }, 'context length T, tokens (log scale)'));
      R.chartG.replaceChildren(...kids);
      if (moving()) R.chartG.animate([{ opacity: 0.25 }, { opacity: 1 }], { duration: 520, easing: 'cubic-bezier(.2,.7,.2,1)' });
    }

    function paintCompute(changed) {
      const w = 2 * a.N, sc = scoreFlops(a, T), tot = w + sc;
      R.h2b.innerHTML = `At <b>${fmtInt(T)}</b> tokens, attention is <b style="color:var(--c-attn)">${pct(attnShare(a, T))}</b> of the block FLOPs.`;
      tween(R.fW, w, fmtFlop); tween(R.fS, sc, fmtFlop); tween(R.fT, tot, fmtFlop);
      tween(R.tW, 3 * w, fmtFlop); tween(R.tS, 3 * sc, fmtFlop); tween(R.tT, 3 * tot, fmtFlop);
      R.bW.style.width = `${(100 * w) / tot}%`;
      R.bS.style.width = `${(100 * sc) / tot}%`;
      const x = xOf(T), A = attnShare(a, T), right = x > CH.W - 220;
      R.mark.style.transform = `translate(${x}px, 0px)`;
      R.markTxt.setAttribute('x', right ? -8 : 8);
      R.markTxt.setAttribute('text-anchor', right ? 'end' : 'start');
      R.markTxt.textContent = `T = ${fmtInt(T)} · attention ${pct(A)}`;
      R.markDot.setAttribute('cy', yOf(A));
      if (changed) drawChart();
      R.scoreLine.innerHTML = `Score FLOPs equal the weight FLOPs of all blocks at T ≈ <b>${fmtInt(scoresMatchBlocks(a))}</b> tokens for this model.`;
    }

    function paintMemory() {
      const N = a.N, train = 16 * N, kv = kvBytes(sel.cfg, T, batch);
      const scale = Math.max(GPU_BUDGET, 4 * N, kv, train) * 1.08;
      const at = (bytes) => `${((100 * bytes) / scale).toFixed(3)}%`;
      const shade = (bytes) => sequential(0.25 + 0.75 * (bytes / scale));
      memRows.weights.forEach((row, k) => {
        const bytes = WEIGHT_TIERS[k][1] * N;
        row.fills[0].style.width = at(bytes);
        row.fills[0].style.background = shade(bytes);
        tween(row.val, bytes, fmtGB);
      });
      memRows.kv.fills[0].style.width = at(kv);
      memRows.kv.fills[0].style.background = shade(kv);
      tween(memRows.kv.val, kv, fmtGB);
      memRows.kv.sub.textContent = `T = ${fmtInt(T)} tokens · batch ${batch}`;
      R.kvCheck.replaceChildren(checkBadge(kvFormula(a, T, batch), kvBytes(sel.cfg, T, batch), { label: 'equals engine.cacheBytes', tol: 0 }));
      let cum = 0;
      TRAIN_TIERS.forEach(([, bytes], k) => {
        const seg = memRows.train.fills[k], w = bytes * N;
        seg.style.left = at(cum);
        seg.style.width = at(w);
        seg.style.background = tierColour(k);
        R.trainSw[k].style.background = tierColour(k);
        cum += w;
      });
      tween(memRows.train.val, train, fmtGB);
      [...memRows.weights, memRows.kv, memRows.train].forEach((r) => { r.ref.style.left = at(GPU_BUDGET); });
      R.h2c.innerHTML = `Training with Adam needs <b>${fmtGB(train)}</b> before activations.`;
    }

    function paintChinchilla() {
      const c = chinchilla(a, tpp, gpu), name = GPUS[gpu].name;
      R.h2d.innerHTML = `Compute-optimal training: <b>${fmtShort(c.D)}</b> tokens and about <b>${fmtGpu(c.gpuDays)}</b> on ${name}s.`;
      R.cDlab.textContent = `Tokens D ≈ ${tpp}N`;
      tween(R.cD, c.D, fmtShort);
      R.cDsub.textContent = tpp === 20 ? 'the Chinchilla rule (Hoffmann et al., 2022)' : `${tpp} tokens per parameter`;
      R.cC.innerHTML = texHTML(`${num(c.C, 2)}\\;\\mathrm{FLOP}`);
      tween(R.cG, c.gpuDays, fmtGpu);
      const hrs = (c.gpuDays * 24) / 1000;
      R.cGsub.textContent = `≈ ${hrs >= 1 ? `${hrs.toPrecision(2)} hours` : `${(hrs * 60).toPrecision(2)} minutes`} on 1,000 ${name}s at ${Math.round(MFU * 100)}% utilisation`;
      R.cEq.innerHTML = texHTML(`C \\approx 6\\,N\\,D = 6 \\times ${num(a.N, 2)} \\times ${num(c.D, 2)} \\approx ${num(c.C, 2)}\\;\\mathrm{FLOP}`, { display: true });
    }

    // changed: the model (or its shape) changed; otherwise only T, batch, GPU or tokens per parameter moved
    function update() {
      sel = pick();
      a = arch(sel.cfg);
      const key = JSON.stringify(sel.cfg);
      const changed = key !== lastKey;
      lastKey = key;
      if (changed) { clearTimers(); buildParams(moving()); }
      paintHeader();
      paintCompute(changed);
      paintMemory();
      paintChinchilla();
    }

    const offConfig = ctx.on('config', () => { if (choice === 'app') update(); });
    const offTheme = onThemeChange(() => { if (alive) update(); });

    update();
    appear([head, bar, sec1, sec2, sec3, sec4, notes], { stagger: 90 });

    return {
      destroy() {
        alive = false;
        clearTimers();
        offConfig();
        offTheme();
        hideTip();
      },
    };
  },
};

// Params, FLOPs & memory: one calculator for one config, walked in four steps: parameters, compute
// per token, memory, training compute. Every figure is arithmetic on the config and is checked
// against the engine: the parameter count against engine.paramSpecs / countParams, the KV cache
// against engine.cacheBytes. Published configs are the reported architectures; anything estimated
// is marked ≈. Reference scene (More menu); the notes and sources are in the Notes drawer.
import { countParams, paramSpecs, cacheBytes, DEFAULT_CONFIG } from '../../engine/index.js';
import { h, s, texHTML, num, segmented, slider, showTip, hideTip, reducedMotion } from '../../ui/index.js';
import { stage2d, sceneHeader, sceneCaption, appear, checkBadge, useStyle, ZONES, MOTION } from '../kit.js';

// ------------------------------------------------------------------ constants

// Published architectures. ctx is the trained context window: it sets the learned position table
// where one exists, and the default T for the attention cost and the KV cache.
const PUBLISHED = [
  { id: 'gpt2', name: 'GPT-2 small', ctx: 1024, cfg: { vocab_size: 50257, block_size: 1024, n_layer: 12, n_head: 12, n_kv_head: 12, d_model: 768, d_ff: 3072, pos_encoding: 'learned', norm: 'layernorm', norm_placement: 'pre', activation: 'gelu', tie_weights: true, bias: true } },
  { id: 'gpt2xl', name: 'GPT-2 XL', ctx: 1024, cfg: { vocab_size: 50257, block_size: 1024, n_layer: 48, n_head: 25, n_kv_head: 25, d_model: 1600, d_ff: 6400, pos_encoding: 'learned', norm: 'layernorm', norm_placement: 'pre', activation: 'gelu', tie_weights: true, bias: true } },
  { id: 'gpt3', name: 'GPT-3 175B', ctx: 2048, note: 'Its vocabulary and tied head follow GPT-2.', cfg: { vocab_size: 50257, block_size: 2048, n_layer: 96, n_head: 96, n_kv_head: 96, d_model: 12288, d_ff: 49152, pos_encoding: 'learned', norm: 'layernorm', norm_placement: 'pre', activation: 'gelu', tie_weights: true, bias: true } },
  { id: 'llama2-7b', name: 'LLaMA-2 7B', ctx: 4096, cfg: { vocab_size: 32000, block_size: 4096, n_layer: 32, n_head: 32, n_kv_head: 32, d_model: 4096, d_ff: 11008, pos_encoding: 'rope', norm: 'rmsnorm', norm_placement: 'pre', activation: 'swiglu', tie_weights: false, bias: false } },
  { id: 'llama2-70b', name: 'LLaMA-2 70B', ctx: 4096, cfg: { vocab_size: 32000, block_size: 4096, n_layer: 80, n_head: 64, n_kv_head: 8, d_model: 8192, d_ff: 28672, pos_encoding: 'rope', norm: 'rmsnorm', norm_placement: 'pre', activation: 'swiglu', tie_weights: false, bias: false } },
  { id: 'llama3-8b', name: 'LLaMA-3 8B', ctx: 8192, cfg: { vocab_size: 128256, block_size: 8192, n_layer: 32, n_head: 32, n_kv_head: 8, d_model: 4096, d_ff: 14336, pos_encoding: 'rope', norm: 'rmsnorm', norm_placement: 'pre', activation: 'swiglu', tie_weights: false, bias: false } },
  { id: 'mistral-7b', name: 'Mistral 7B', ctx: 8192, note: 'Its sliding-window attention is not modelled.', cfg: { vocab_size: 32000, block_size: 8192, n_layer: 32, n_head: 32, n_kv_head: 8, d_model: 4096, d_ff: 14336, pos_encoding: 'rope', norm: 'rmsnorm', norm_placement: 'pre', activation: 'swiglu', tie_weights: false, bias: false } },
];

const LOG_LO = 3, LOG_HI = 17;                                  // context axis: 8 … 131,072 tokens
const CTX_TICKS = [8, 64, 1024, 4096, 16384, 65536, 131072];
const GPUS = { a100: { name: 'A100', tflops: 312 }, h100: { name: 'H100', tflops: 989 } };   // dense bf16 peak
const MFU = 0.4;                                                // assumed utilisation of the peak
const GPU_BUDGET = 80e9;                                        // one 80 GB GPU: reference line
const STEP_MS = 230;                                            // one term of the parameter sum
const CH = { W: 840, H: 300, l: 64, r: 24, t: 26, b: 56 };      // chart box, design units
const WEIGHT_TIERS = [['fp32', 4], ['bf16', 2], ['int8', 1], ['int4', 0.5]];
const TRAIN_TIERS = [['bf16 weights', 2], ['bf16 gradients', 2], ['fp32 master weights', 4], ['Adam m (fp32)', 4], ['Adam v (fp32)', 4]];
const STEPS = ['Parameters', 'Compute', 'Memory', 'Training'];

const CSS = `
.sc-hero { position: relative; width: 1380px; height: 498px; }
.sc-panel { position: absolute; left: 0; top: 0; width: 1380px; height: 498px; min-width: 0;
  opacity: 0; visibility: hidden; transform: translateY(12px); transition: opacity .3s ease, transform .3s ease, visibility 0s linear .3s; }
.sc-panel.on { opacity: 1; visibility: visible; transform: none; transition: opacity .5s ease .15s, transform .55s cubic-bezier(.2,.7,.2,1) .15s, visibility 0s; }
.sc-p0 { display: grid; gap: 20px; align-content: start; }
.sc-p1 { display: grid; grid-template-columns: 840px 480px; column-gap: 60px; align-content: start; }
.sc-p2 { display: grid; gap: 26px; align-content: start; padding-top: 26px; }
.sc-p3 { display: grid; gap: 22px; align-content: start; }
.sc-col { display: grid; gap: 14px; align-content: start; min-width: 0; }
.sc-panel .k-check { white-space: nowrap; flex: none; }

.sc-stack { position: relative; display: flex; height: 40px; border-radius: 10px; overflow: hidden; background: var(--surface-2); border: 1px solid var(--line); }
.sc-seg { display: block; height: 100%; width: 0; flex: none; background: var(--c); transition: width .6s cubic-bezier(.65,0,.35,1); }
.sc-seg + .sc-seg { box-shadow: -1px 0 0 var(--surface); }
.sc-table { display: grid; gap: 4px; min-width: 0; }
.sc-row, .sc-thead { display: grid; grid-template-columns: 14px minmax(0, 230px) minmax(0, 1.1fr) minmax(0, 0.9fr) 176px 84px; gap: 16px; align-items: center; padding: 6px 10px; border-radius: 10px; min-width: 0; }
.sc-row { transition: opacity .35s ease; }
.sc-row.pending { opacity: .2; }
.sc-thead { padding-top: 0; padding-bottom: 4px; font: 600 14px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .06em; }
.sc-thead .r, .sc-row .r { text-align: right; }
.sc-sw { width: 14px; height: 14px; border-radius: 4px; background: var(--c); }
.sc-name { font: 600 17px/1.25 var(--font-body); color: var(--fg); min-width: 0; }
.sc-name small { display: block; font: 500 14px/1.35 var(--font-mono); color: var(--fg-3); margin-top: 2px; }
.sc-tex { font-size: 18px; color: var(--fg); overflow-x: auto; overflow-y: hidden; white-space: nowrap; }
.sc-subst { font: 500 15px/1.4 var(--font-mono); color: var(--fg-2); min-width: 0; overflow-wrap: anywhere; }
.sc-count { font: 600 17px/1 var(--font-mono); text-align: right; font-variant-numeric: tabular-nums; color: var(--fg); white-space: nowrap; }
.sc-share { font: 600 15px/1 var(--font-mono); text-align: right; color: var(--c, var(--fg)); font-variant-numeric: tabular-nums; }
.sc-sum { border-top: 1px solid var(--line-strong); border-radius: 0; padding-top: 12px; }
.sc-sum .sc-count { font-weight: 700; }

.sc-eq { display: grid; gap: 10px; padding: 16px 18px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow-1); min-width: 0; }
.sc-eq .sc-tex { font-size: 22px; }
.sc-nums { display: grid; gap: 4px; font: 500 15px/1.45 var(--font-mono); color: var(--fg-2); font-variant-numeric: tabular-nums; }
.sc-nums b { color: var(--fg); font-weight: 600; }
.sc-nums .sc-sc { color: var(--c-attn); }
.sc-bars { display: flex; height: 22px; border-radius: 7px; overflow: hidden; background: var(--surface-2); border: 1px solid var(--line); }
.sc-bars i { display: block; height: 100%; width: 0; flex: none; transition: width .6s cubic-bezier(.65,0,.35,1); }
.sc-legend { display: flex; flex-wrap: wrap; gap: 6px 22px; font: 500 15px/1.3 var(--font-mono); color: var(--fg-2); }
.sc-legend span { display: inline-flex; align-items: center; gap: 8px; }
.sc-legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; background: var(--c); flex: none; }
.sc-chart svg { display: block; width: 840px; height: 300px; overflow: visible; }
.sc-chart text { font: 500 15px var(--font-mono); fill: var(--fg-3); }
.sc-mark { transition: transform .6s cubic-bezier(.65,0,.35,1); }
.sc-mark text { fill: var(--fg); font: 600 15px var(--font-mono); }

.sc-mem { display: grid; gap: 12px; min-width: 0; }
.sc-mrow { display: grid; grid-template-columns: 270px minmax(0, 1fr) 150px; gap: 20px; align-items: center; min-width: 0; }
.sc-mlab { display: grid; gap: 6px; justify-items: start; min-width: 0; font: 600 17px/1.25 var(--font-body); color: var(--fg); }
.sc-mlab small { font: 500 14px/1.35 var(--font-mono); color: var(--fg-3); }
.sc-track { position: relative; height: 34px; border-radius: 8px; background: var(--surface-2); border: 1px solid var(--line); }
.sc-fill { position: absolute; top: 0; bottom: 0; left: 0; width: 0; border-radius: 7px; transition: width .6s cubic-bezier(.65,0,.35,1), left .6s cubic-bezier(.65,0,.35,1); }
.sc-fill.sc-tier { border-radius: 0; box-shadow: -2px 0 0 var(--surface); }
.sc-fill.sc-tier:first-child { border-radius: 7px 0 0 7px; box-shadow: none; }
.sc-fill.sc-tier:last-of-type { border-radius: 0 7px 7px 0; }
.sc-ref { position: absolute; top: -9px; bottom: -9px; left: 0; border-left: 1.5px dashed var(--fg-2); transition: left .6s cubic-bezier(.65,0,.35,1); pointer-events: none; }
.sc-ref span { position: absolute; top: -21px; left: 6px; font: 600 14px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.sc-ref.left span { left: auto; right: 6px; }
.sc-mval { font: 600 17px/1 var(--font-mono); text-align: right; font-variant-numeric: tabular-nums; color: var(--fg); white-space: nowrap; }
.sc-strip { display: flex; justify-content: space-between; align-items: center; gap: 24px; padding: 14px 18px; }

.sc-ctrl { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 14px 24px; align-items: center; }
.sc-ctrl > div { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.sc-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }
.sc-stat { display: grid; gap: 12px; align-content: start; padding: 20px 22px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow-1); min-width: 0; }
.sc-stat .lab { font: 600 14px/1.2 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .06em; }
.sc-stat .val { font: 700 36px/1.1 var(--font-mono); font-variant-numeric: tabular-nums; color: var(--fg); overflow-wrap: anywhere; }
.sc-stat .val small { font: 600 17px/1 var(--font-mono); color: var(--fg-3); margin-left: 8px; }
.sc-stat .sub { font: 500 15px/1.45 var(--font-body); color: var(--fg-2); }
.sc-eqc { padding: 18px 20px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow-1); overflow-x: auto; font-size: 22px; }

@media (prefers-reduced-motion: reduce) {
  .sc-panel, .sc-panel.on, .sc-seg, .sc-fill, .sc-ref, .sc-mark, .sc-row, .sc-bars i { transition: none !important; }
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
    mask: cfg.mask ?? DEFAULT_CONFIG.mask, window: cfg.window ?? DEFAULT_CONFIG.window, prefixLen: cfg.prefix_len ?? DEFAULT_CONFIG.prefix_len,
  };
}

// The engine's own count: the sum of every shape in paramSpecs(cfg). No tensors are allocated.
export const engineParams = (cfg) => paramSpecs(fullCfg(cfg)).reduce((acc, [, shape]) => acc + shape.reduce((x, y) => x * y, 1), 0);

// Mean number of keys a query may attend to over the T queries of a sequence. Mirrors maskAllows()
// in the engine: causal j ≤ i; none, every key; sliding j ≤ i and j > i − window; prefix j ≤ i or
// j < prefix_len. Closed forms, so the chart can call it at every point.
export function visibleKeys(a, T) {
  if (a.mask === 'none') return T;
  if (a.mask === 'sliding') return T <= a.window ? (T + 1) / 2 : a.window - (a.window * (a.window - 1)) / (2 * T);
  if (a.mask === 'prefix') {
    const p = Math.min(a.prefixLen, T), m1 = Math.min(T, Math.max(0, a.prefixLen - 1));   // queries whose prefix bound dominates
    return (m1 * p + (T * (T + 1)) / 2 - (m1 * (m1 + 1)) / 2) / T;
  }
  return (T + 1) / 2;                                                                       // causal
}

// FLOPs per token. A matmul costs 2 FLOPs per weight. The score term is QKᵀ and the weighted sum
// over V: 2 FLOPs per multiply-add, for every layer, head and visible key, so 4·L·d·K̄ per token.
export const scoreFlops = (a, T) => 4 * a.L * a.d * visibleKeys(a, T);
export const forwardFlops = (a, T) => 2 * a.N + scoreFlops(a, T);
export const trainingFlops = (a, T) => 3 * forwardFlops(a, T);
// Attention sub-layer (projections and scores) against the MLP sub-layer, per token at context T.
export const attnShare = (a, T) => { const A = 2 * a.parts.attn + scoreFlops(a, T), M = 2 * a.parts.mlp; return A / (A + M); };
// The T where the score FLOPs equal the weight FLOPs of all blocks, 4·L·d·K̄ = 2·(N_attn + N_MLP).
// null when the mask never gets there (a sliding window caps K̄ below the target).
export function scoresMatchBlocks(a) {
  const target = (a.parts.attn + a.parts.mlp) / (2 * a.L * a.d);
  if (visibleKeys(a, 2 ** 40) < target) return null;
  if (visibleKeys(a, 1) >= target) return 1;
  let lo = 1, hi = 2 ** 40;
  for (let i = 0; i < 90; i++) { const mid = Math.sqrt(lo * hi); if (visibleKeys(a, mid) >= target) hi = mid; else lo = mid; }
  return hi;
}

// KV cache in bf16 (2 bytes): K and V, every layer, every KV head, every token of the batch.
const kvBytes = (cfg, T, batch) => cacheBytes(cfg, T * batch, 2);
export const kvFormula = (a, T, batch) => 2 * a.L * a.Hkv * a.dh * T * batch * 2;

// Chinchilla: D = tpp · N tokens, C ≈ 6 N D, and GPU-days at MFU of the peak.
export function chinchilla(a, tpp, gpu) {
  const D = tpp * a.N, C = 6 * a.N * D, perGpu = GPUS[gpu].tflops * 1e12 * MFU;
  return { D, C, gpuDays: C / (perGpu * 86400) };
}

// The six terms of the parameter sum, with their TeX, the substituted numbers and the count.
export function paramRows(a) {
  const k = a.swiglu ? 3 : 2;
  return [
    { kind: 'embed', name: 'Token embeddings', sub: 'W_E', tex: 'V\\,d',
      subst: `${fmtInt(a.V)} × ${fmtInt(a.d)}`, n: a.parts.embed },
    { kind: 'pos', name: 'Learned positions', sub: 'W_P', tex: a.learned ? 'n_{\\mathrm{ctx}}\\,d' : '0',
      subst: a.learned ? `${fmtInt(a.ctx)} × ${fmtInt(a.d)}` : 'no learned table', n: a.parts.pos },
    { kind: 'attn', name: 'Attention', sub: `W_Q, W_K, W_V, W_O${a.bias ? ', biases' : ''}`,
      tex: 'L\\,(d\\,H\\,d_h + 2\\,d\\,H_{\\mathrm{kv}}\\,d_h + H\\,d_h\\,d)',
      subst: `${a.L} × (${fmtInt(a.attnWl)}${a.bias ? ` + ${fmtInt(a.attnBl)}` : ''})`, n: a.parts.attn },
    { kind: 'mlp', name: 'MLP', sub: `${a.swiglu ? 'W_gate, W_up, W_down' : 'W_up, W_down'}${a.bias ? ', biases' : ''}`,
      tex: `${k}\\,L\\,d\\,F`,
      subst: `${a.L} × (${fmtInt(a.mlpWl)}${a.bias ? ` + ${fmtInt(a.mlpBl)}` : ''})`, n: a.parts.mlp },
    { kind: 'norm', name: 'Norms', sub: `${a.normName}: ${a.lnBias ? 'γ, β' : 'γ'}`, tex: a.pre ? '(2L{+}1)\\,d' : '2L\\,d',
      subst: `${a.nNorms} × ${fmtInt(a.d)}${a.lnBias ? ' × 2' : ''}`, n: a.parts.norms },
    { kind: 'head', name: 'LM head', sub: a.tied ? 'tied to W_E' : 'W_U', tex: a.tied ? '0' : 'V\\,d',
      subst: a.tied ? 'shares W_E' : `${fmtInt(a.V)} × ${fmtInt(a.d)}`, n: a.parts.head },
  ];
}

// ------------------------------------------------------------------ formatting

const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const sigfig = (x, n) => Number(x.toPrecision(n));          // n significant figures, never an exponent in our ranges
function fmtShort(n) {
  for (const [v, u] of [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) if (Math.abs(n) >= v) return `${sigfig(n / v, 3)} ${u}`;
  return fmtInt(n);
}
function fmtBytes(b) {
  for (const [v, u] of [[1e12, 'TB'], [1e9, 'GB'], [1e6, 'MB'], [1e3, 'KB']]) if (b >= v) return `${sigfig(b / v, 3)} ${u}`;
  return `${fmtInt(b)} B`;
}
function fmtFlop(x) {
  for (const [v, u] of [[1e12, 'T'], [1e9, 'G'], [1e6, 'M']]) if (x >= v) return `${sigfig(x / v, 3)} ${u}FLOP`;
  return `${fmtInt(x)} FLOP`;
}
// A duration on a unit ladder: seconds below a minute, minutes below two hours, hours below two
// days, days otherwise. Two significant figures and digit grouping; no exponent.
function ladder(hours) {
  const sec = hours * 3600;
  if (sec < 1) return ['under 1', 'second'];
  if (sec < 60) return [String(sigfig(sec, 2)), 'second'];
  if (hours < 2) return [String(sigfig(hours * 60, 2)), 'minute'];
  if (hours < 48) return [fmtInt(sigfig(hours, 2)), 'hour'];
  return [fmtInt(sigfig(hours / 24, 2)), 'day'];
}
const plural = (n, unit) => (n === '1' || n.startsWith('under') ? unit : `${unit}s`);
const gpuSpan = (hours) => { const [n, u] = ladder(hours); return `${n} GPU-${plural(n, u)}`; };
const ladderAbout = (hours) => { const [n, u] = ladder(hours); return `${n.startsWith('under') ? n : `about ${n}`} ${plural(n, u)}`; };
const pct = (x) => (x > 0 && x < 0.001 ? '<0.1%' : `${(100 * x).toFixed(1)}%`);
// The table's shares by term kind, so the caption and the table print the same rounding.
const paramShares = (a) => { const rows = paramRows(a), sh = shares(rows.map((r) => r.n), a.N); return Object.fromEntries(rows.map((r, k) => [r.kind, sh[k]])); };
const fmtTok = (T) => (T >= 1024 && T % 1024 === 0 ? `${T / 1024}k` : fmtInt(T));
const xOf = (T) => CH.l + ((Math.log2(T) - LOG_LO) / (LOG_HI - LOG_LO)) * (CH.W - CH.l - CH.r);
const yOf = (p) => CH.H - CH.b - p * (CH.H - CH.t - CH.b);
// Shares in tenths of a percent that add up to exactly 100.0 (largest remainder).
function shares(values, total) {
  const raw = values.map((v) => (1000 * v) / total), base = raw.map(Math.floor);
  const left = 1000 - base.reduce((x, y) => x + y, 0);
  raw.map((r, i) => [r - base[i], i]).sort((p, q) => q[0] - p[0]).slice(0, left).forEach(([, i]) => { base[i] += 1; });
  return base.map((b) => `${(b / 10).toFixed(1)}%`);
}
// Memory magnitudes take a neutral ramp; the training tiers are five neutral steps with separators.
const memFill = (f) => `color-mix(in srgb, var(--fg-2) ${Math.round(30 + 40 * f)}%, var(--surface-2))`;
const tierFill = (k) => `color-mix(in srgb, var(--fg-2) ${22 + 16 * k}%, var(--surface-2))`;
const maskRule = (a) => {
  if (a.mask === 'none') return 'No mask: every query sees all keys.';
  if (a.mask === 'sliding') return `Sliding window ${a.window}: a query sees its last ${a.window} keys.`;
  if (a.mask === 'prefix') return `Prefix ${a.prefixLen}: the first ${a.prefixLen} keys are visible to every query.`;
  return 'Causal mask: query i sees i + 1 keys.';
};

// ------------------------------------------------------------------ scene

export default {
  mount(el, ctx) {
    useStyle('scaling', CSS);
    const st = stage2d(el);
    let alive = true, timers = [], auto = null, step = 0;
    const later = (fn, ms) => { timers.push(setTimeout(() => { if (alive) fn(); }, ms)); };
    const clearTimers = () => { timers.forEach(clearTimeout); timers = []; };
    const moving = () => !reducedMotion();

    // selection: the app's loaded model, or a published configuration
    let choice = 'app', sel = null, a = null, lastKey = '', lastCtx = null;
    let T = 64, batch = 1, gpu = 'a100', tpp = 20;
    const pick = () => {
      if (choice === 'app') {
        const cfg = fullCfg(ctx.state.model?.config ?? ctx.state.config);
        return { name: 'App model', cfg, ctxDefault: cfg.block_size, loaded: true };
      }
      const p = PUBLISHED.find((x) => x.id === choice) ?? PUBLISHED[0];
      return { name: p.name, cfg: fullCfg(p.cfg), ctxDefault: p.ctx, loaded: false };
    };
    sel = pick();
    a = arch(sel.cfg);
    T = sel.ctxDefault;
    lastCtx = sel.ctxDefault;

    // counting animation: one tween per element, superseded by the next target
    function tween(node, to, fmt, { from, dur = 640, instant = false } = {}) {
      node._tok = (node._tok || 0) + 1;
      const tok = node._tok;
      const start = from ?? (typeof node._v === 'number' ? node._v : to);
      if (instant || !moving() || start === to) { node._v = to; node.textContent = fmt(to); return; }
      const t0 = performance.now();
      const step1 = (now) => {
        if (!alive || node._tok !== tok) return;
        const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
        node._v = start + (to - start) * e;
        node.textContent = fmt(node._v);
        if (p < 1) requestAnimationFrame(step1);
        else node._v = to;
      };
      requestAnimationFrame(step1);
    }

    // ---------------------------------------------------------- DOM, built once
    const modelSeg = segmented({
      ariaLabel: 'Model', value: choice,
      options: [{ value: 'app', label: 'App model', title: 'The model loaded in the app' },
        ...PUBLISHED.map((p) => ({ value: p.id, label: p.name, title: p.note ? `${p.name}, published configuration. ${p.note}` : `${p.name}, published configuration` }))],
      onChange: (v) => { choice = v; T = pick().ctxDefault; ctxSlider.set(Math.log2(T)); update(); },
    });
    const header = sceneHeader(st, { chips: [['Scaling', 'resid']], title: 'Params, FLOPs & memory' });
    const picker = h('div', { class: 'k-picker-wrap' }, h('span', { class: 'k-picker-label' }, 'model'), modelSeg.el);
    st.add(picker, { x: ZONES.controls.x, y: ZONES.controls.y });

    const hero = h('div', { class: 'sc-hero' });
    st.add(hero, { x: ZONES.hero.x, y: 262 });
    const panels = [
      h('section', { class: 'sc-panel sc-p0', 'aria-label': 'Parameters' }),
      h('section', { class: 'sc-panel sc-p1', 'aria-label': 'Compute per token' }),
      h('section', { class: 'sc-panel sc-p2', 'aria-label': 'Memory' }),
      h('section', { class: 'sc-panel sc-p3', 'aria-label': 'Training compute' }),
    ];
    hero.append(...panels);
    const cap = sceneCaption(st, { x: ZONES.caption.x, y: ZONES.caption.y, w: 1380 });

    // ---- step 0 · parameters: the stacked bar and the six terms of the sum
    const stack = h('div', { class: 'sc-stack' });
    const checkSlot = h('div', { class: 'sc-subst' });
    const totalCount = h('span', null, '0');
    const thead = h('div', { class: 'sc-thead' }, h('span'), h('span', null, 'Term'), h('span', null, 'Formula'),
      h('span', null, 'Substituted'), h('span', { class: 'r' }, 'Count'), h('span', { class: 'r' }, 'Share'));
    const sumRow = h('div', { class: 'sc-row sc-sum' },
      h('span'),
      h('div', { class: 'sc-name' }, 'Total N', h('small', null, 'the sum of the rows above')),
      h('div', { class: 'sc-tex-wrap' }, h('div', { class: 'sc-tex', html: texHTML('N = \\sum_k n_k') })),
      checkSlot,
      h('div', { class: 'sc-count' }, totalCount),
      h('div', { class: 'sc-share' }, '100.0%'));
    const table = h('div', { class: 'sc-table' });
    panels[0].append(stack, table);

    function rowEl(r, share, pending) {
      const count = h('div', { class: 'sc-count' }, pending ? '0' : fmtInt(r.n));
      const row = h('div', { class: ['sc-row', pending && 'pending'] },
        h('i', { class: 'sc-sw', style: { '--c': `var(--c-${r.kind})` } }),
        h('div', { class: 'sc-name' }, r.name, h('small', null, r.sub)),
        h('div', { class: 'sc-tex-wrap' }, h('div', { class: 'sc-tex', html: texHTML(r.tex) })),
        h('div', { class: 'sc-subst' }, r.subst),
        count,
        h('div', { class: 'sc-share', style: { '--c': `var(--c-${r.kind})` } }, share));
      row._count = count;
      return row;
    }

    // Rebuilds the term rows and the stacked bar; with animate, the sum adds up term by term.
    function buildParams(animate) {
      clearTimers();
      const rows = paramRows(a), sh = shares(rows.map((r) => r.n), a.N);
      const loaded = sel.loaded && ctx.state.model;
      checkSlot.replaceChildren(checkBadge(a.N, loaded ? countParams(ctx.state.model) : engineParams(sel.cfg), {
        label: loaded ? 'equals engine.countParams' : 'equals engine.paramSpecs', tol: 0,
      }));
      const segs = rows.map((r) => h('i', { class: 'sc-seg', style: { '--c': `var(--c-${r.kind})` } }));
      stack.replaceChildren(...segs);
      void stack.offsetWidth;
      const applyWidths = () => segs.forEach((sg, k) => {
        sg.style.width = `${(100 * rows[k].n) / a.N}%`;
        sg.style.transitionDelay = animate ? `${(k * STEP_MS) / 1000}s` : '0s';
        sg.style.transition = moving() ? '' : 'none';
      });
      if (animate) requestAnimationFrame(applyWidths); else applyWidths();

      const rowEls = rows.map((r, k) => rowEl(r, sh[k], animate));
      table.replaceChildren(thead, ...rowEls, sumRow);
      if (animate) {
        totalCount._v = 0;
        totalCount.textContent = '0';
        let run = 0;
        rows.forEach((r, k) => later(() => {
          rowEls[k].classList.remove('pending');
          tween(rowEls[k]._count, r.n, fmtInt, { from: 0, dur: 420 });
          run += r.n;
          tween(totalCount, run, fmtInt, { dur: 420 });
        }, k * STEP_MS));
      } else {
        rowEls.forEach((row, k) => { row.classList.remove('pending'); tween(row._count, rows[k].n, fmtInt, { instant: true }); });
        tween(totalCount, a.N, fmtInt, { instant: true });
      }
    }

    // ---- step 1 · compute per token: the attention share against context, and one token's FLOPs
    const svg = s('svg', { viewBox: `0 0 ${CH.W} ${CH.H}`, role: 'img', 'aria-label': 'Attention and MLP share of block FLOPs against context length' });
    const chartG = s('g');
    const mark = s('g', { class: 'sc-mark' });
    const markLine = s('line', { x1: 0, x2: 0, y1: yOf(1), y2: yOf(0), style: 'stroke: var(--fg); stroke-width: 1.5; stroke-dasharray: 3 3' });
    const markDot = s('circle', { cx: 0, cy: yOf(0.5), r: 5, style: 'fill: var(--c-attn); stroke: var(--surface); stroke-width: 2' });
    const markTxt = s('text', { x: 8, y: yOf(1) + 16 }, '');
    mark.append(markLine, markDot, markTxt);
    svg.append(chartG, mark);
    svg.addEventListener('pointermove', (ev) => {
      const r = svg.getBoundingClientRect();
      const xv = ((ev.clientX - r.left) / r.width) * CH.W;
      const e = LOG_LO + ((xv - CH.l) / (CH.W - CH.l - CH.r)) * (LOG_HI - LOG_LO);
      const Tt = 2 ** Math.min(LOG_HI, Math.max(LOG_LO, e));
      const A = attnShare(a, Tt);
      showTip(ev.clientX, ev.clientY, `T = <b>${fmtInt(Math.round(Tt))}</b> · attention ${pct(A)} · MLP ${pct(1 - A)}`);
    });
    svg.addEventListener('pointerleave', hideTip);
    const chartLegend = h('div', { class: 'sc-legend' },
      h('span', null, h('i', { style: { '--c': 'var(--c-mlp)' } }), 'MLP sub-layer'),
      h('span', null, h('i', { style: { '--c': 'var(--c-attn)' } }), 'attention: projections and scores'));
    const ctxSlider = slider({
      id: 'scaling-ctx', label: 'Context length T', min: LOG_LO, max: LOG_HI, step: 0.05, value: Math.log2(T),
      format: (v) => `${fmtInt(Math.round(2 ** v))} tokens`,
      onInput: (v) => { T = Math.round(2 ** v); update(); },
    });
    const grey = 'color-mix(in srgb, var(--fg-3) 50%, var(--surface-3))';
    const R = {};
    R.fTex = h('div', { class: 'sc-tex', html: texHTML('\\mathrm{FLOP}_{\\mathrm{fwd}}\\,/\\,\\mathrm{token} \\approx 2N + 4\\,L\\,d\\,\\bar K') });
    R.fW = h('b'); R.fS = h('b', { class: 'sc-sc' }); R.fT = h('b');
    R.fNums = h('div', { class: 'sc-nums' }, h('div', null, '2N = ', R.fW), h('div', null, '4LdK = ', R.fS), h('div', null, 'total = ', R.fT));
    R.kLine = h('div', { class: 'sc-nums' });
    R.bW = h('i', { style: { background: grey } });
    R.bS = h('i', { style: { background: 'var(--c-attn)' } });
    const flopLegend = h('div', { class: 'sc-legend' },
      h('span', { title: 'N counts every parameter, including the embedding table, which costs no FLOPs' }, h('i', { style: { '--c': grey } }), '2N, weights'),
      h('span', null, h('i', { style: { '--c': 'var(--c-attn)' } }), '4LdK, attention scores'));
    R.scoreLine = h('div', { class: 'sc-nums' });
    R.trainLine = h('div', { class: 'sc-nums' });
    panels[1].append(
      h('div', { class: 'sc-col' }, h('div', { class: 'sc-chart' }, svg), chartLegend, ctxSlider.el),
      h('div', { class: 'sc-col' }, h('div', { class: 'sc-eq' }, R.fTex, R.fNums), R.kLine,
        h('div', { class: 'sc-bars' }, R.bW, R.bS), flopLegend, R.scoreLine, R.trainLine));

    // ---- step 2 · memory: the bars share one linear scale, the largest row fills the track
    const batchSeg = segmented({
      ariaLabel: 'Batch size', value: batch,
      options: [1, 8, 32, 128].map((v) => ({ value: v, label: String(v), title: `${v} sequence${v === 1 ? '' : 's'} in the batch` })),
      onChange: (v) => { batch = v; update(); },
    });
    function memRow({ label, sub, extra = null, fills = 1, refLabel = false }) {
      const fillEls = Array.from({ length: fills }, () => h('i', { class: ['sc-fill', fills > 1 && 'sc-tier'] }));
      const ref = h('div', { class: 'sc-ref' }, refLabel ? h('span', null, '80 GB, one GPU') : null);
      const track = h('div', { class: 'sc-track' }, ...fillEls, ref);
      const val = h('span', null, '');
      const el = h('div', { class: 'sc-mrow' },
        h('div', { class: 'sc-mlab' }, h('span', null, label), h('small', null, sub), extra), track,
        h('div', { class: 'sc-mval' }, val));
      return { el, fills: fillEls, ref, val, sub: el.querySelector('.sc-mlab small') };
    }
    const memRows = {
      weights: WEIGHT_TIERS.map(([name, bytes], k) => memRow({
        label: `${name} weights`, sub: `${bytes} ${bytes === 1 ? 'byte' : 'bytes'} per parameter`, refLabel: k === 0,
      })),
      kv: memRow({ label: 'KV cache, bf16', sub: '', extra: batchSeg.el }),
      train: memRow({ label: 'Training with Adam', sub: '16 bytes per parameter', fills: TRAIN_TIERS.length }),
    };
    R.kvTex = h('div', { class: 'sc-tex', html: texHTML('\\mathrm{KV} = 2\\,L\\,H_{\\mathrm{kv}}\\,d_h\\,T\\,\\mathrm{batch}\\cdot 2\\ \\mathrm{bytes}') });
    R.kvCheck = h('div');
    R.trainLegend = h('div', { class: 'sc-legend' }, TRAIN_TIERS.map(([name, bytes], k) =>
      h('span', null, h('i', { style: { '--c': tierFill(k) } }), `${name}, ${bytes} B`)));
    panels[2].append(
      h('div', { class: 'sc-mem' }, ...memRows.weights.map((r) => r.el), memRows.kv.el, memRows.train.el, R.trainLegend),
      h('div', { class: 'sc-eq sc-strip' }, R.kvTex, R.kvCheck));

    // ---- step 3 · training compute: tokens, compute and GPU time for tokens per parameter
    const tppSeg = segmented({
      ariaLabel: 'Tokens per parameter', value: tpp,
      options: [5, 20, 100, 300].map((v) => ({ value: v, label: String(v), title: v === 20 ? '20 tokens per parameter: the Chinchilla rule' : `${v} tokens per parameter` })),
      onChange: (v) => { tpp = v; update(); },
    });
    const gpuSeg = segmented({
      ariaLabel: 'GPU', value: gpu,
      options: Object.entries(GPUS).map(([k, g]) => ({ value: k, label: g.name, title: `${g.name}: ${g.tflops} TFLOP/s dense bf16` })),
      onChange: (v) => { gpu = v; update(); },
    });
    R.cDlab = h('span'); R.cD = h('span'); R.cDsub = h('div', { class: 'sub' });
    R.cC = h('div', { class: 'val' }); R.cG = h('span'); R.cGsub = h('div', { class: 'sub' });
    R.cEq = h('div', { class: 'sc-eqc' });
    panels[3].append(
      h('div', { class: 'sc-ctrl' },
        h('div', null, h('span', { class: 'k-picker-label' }, 'tokens per parameter'), tppSeg.el),
        h('div', null, h('span', { class: 'k-picker-label' }, 'GPU'), gpuSeg.el)),
      h('div', { class: 'sc-stats' },
        h('div', { class: 'sc-stat' }, h('div', { class: 'lab' }, R.cDlab), h('div', { class: 'val' }, R.cD, h('small', null, 'tokens')), R.cDsub),
        h('div', { class: 'sc-stat' }, h('div', { class: 'lab' }, 'Compute, C ≈ 6ND'), R.cC, h('div', { class: 'sub' }, 'FLOPs, forward plus backward')),
        h('div', { class: 'sc-stat' }, h('div', { class: 'lab' }, 'GPU time'), h('div', { class: 'val' }, R.cG), R.cGsub)),
      R.cEq);

    // ---------------------------------------------------------- paint
    function paintHeader() {
      const weights = sel.loaded ? ` · weights: ${ctx.state.preset?.title ?? 'random'}` : '';
      header.setSub(`${sel.name} · ${fmtShort(a.N)} parameters · ${a.L} layers · d ${a.d} · H ${a.H}, H_kv ${a.Hkv} · context ${fmtInt(a.ctx)}${weights}`);
    }

    function drawChart() {
      const xs = [], ms = [];
      for (let e = LOG_LO; e <= LOG_HI + 1e-9; e += 0.25) {
        const A = attnShare(a, 2 ** e);
        xs.push(xOf(2 ** e)); ms.push(1 - A);
      }
      const n = xs.length, P = (x, p) => `${x.toFixed(1)},${yOf(p).toFixed(1)}`;
      const mlpArea = [`M${P(xs[0], 0)}`, ...xs.map((x, i) => `L${P(x, ms[i])}`), `L${P(xs[n - 1], 0)}`, 'Z'].join(' ');
      const attArea = [`M${P(xs[0], 1)}`, ...xs.map((x) => `L${P(x, 1)}`),
        ...xs.map((x, i) => [x, i]).reverse().map(([x, i]) => `L${P(x, ms[i])}`), 'Z'].join(' ');
      const edge = xs.map((x, i) => `${i ? 'L' : 'M'}${P(x, ms[i])}`).join(' ');
      const kids = [];
      [0, 0.25, 0.5, 0.75, 1].forEach((p) => {
        kids.push(s('line', { x1: CH.l, x2: CH.W - CH.r, y1: yOf(p), y2: yOf(p), style: p === 0.5 ? 'stroke: var(--fg-3); stroke-dasharray: 4 4' : 'stroke: var(--line)' }));
        kids.push(s('text', { x: CH.l - 10, y: yOf(p) + 5, 'text-anchor': 'end' }, `${p * 100}%`));
      });
      CTX_TICKS.forEach((Tt) => kids.push(s('text', { x: xOf(Tt), y: CH.H - CH.b + 24, 'text-anchor': 'middle' }, fmtTok(Tt))));
      kids.push(s('path', { d: mlpArea, style: 'fill: var(--c-mlp); fill-opacity: .85' }));
      kids.push(s('path', { d: attArea, style: 'fill: var(--c-attn); fill-opacity: .85' }));
      kids.push(s('path', { d: edge, style: 'fill: none; stroke: var(--surface); stroke-width: 1.5' }));
      kids.push(s('text', { x: (CH.l + CH.W - CH.r) / 2, y: CH.H - 4, 'text-anchor': 'middle' }, 'context length T, tokens (log scale)'));
      chartG.replaceChildren(...kids);
      if (moving()) chartG.animate([{ opacity: 0.25 }, { opacity: 1 }], { duration: 520, easing: 'cubic-bezier(.2,.7,.2,1)' });
    }

    function paintCompute() {
      const w = 2 * a.N, sc = scoreFlops(a, T), tot = w + sc, A = attnShare(a, T), K = visibleKeys(a, T);
      tween(R.fW, w, fmtFlop); tween(R.fS, sc, fmtFlop); tween(R.fT, tot, fmtFlop);
      R.bW.style.width = `${(100 * w) / tot}%`;
      R.bS.style.width = `${(100 * sc) / tot}%`;
      const x = xOf(T), right = x > CH.W - 230;
      mark.style.transform = `translate(${x}px, 0px)`;
      markTxt.setAttribute('x', right ? -8 : 8);
      markTxt.setAttribute('text-anchor', right ? 'end' : 'start');
      markTxt.textContent = `T = ${fmtInt(T)} · attention ${pct(A)}`;
      markDot.setAttribute('cy', yOf(A));
      R.kLine.replaceChildren(
        h('div', null, maskRule(a)),
        h('div', null, 'mean K = ', h('b', null, K >= 100 ? fmtInt(K) : K.toFixed(1)), ` keys per query, over ${fmtInt(T)} queries`));
      const ks = scoresMatchBlocks(a);
      R.scoreLine.replaceChildren(ks == null
        ? h('div', null, `Scores never overtake the weights: the window caps K at ${a.window}.`)
        : h('div', null, 'Scores overtake the weights at T ≈ ', h('b', null, fmtInt(Math.round(ks))), ' tokens for this model and mask.'));
      R.trainLine.replaceChildren(h('div', null, 'Training: 3 × ', fmtFlop(tot), ' = ', h('b', null, fmtFlop(3 * tot)), ' per token.'));
    }

    function paintMemory() {
      const N = a.N, train = 16 * N, kv = kvBytes(sel.cfg, T, batch);
      const scale = Math.max(train, kv);
      const at = (bytes) => `${((100 * bytes) / scale).toFixed(3)}%`;
      const refOn = GPU_BUDGET < scale;
      const paintRef = (row) => {
        row.ref.style.display = refOn ? '' : 'none';
        row.ref.style.left = at(GPU_BUDGET);
        row.ref.classList.toggle('left', GPU_BUDGET / scale > 0.85);
      };
      memRows.weights.forEach((row, k) => {
        const bytes = WEIGHT_TIERS[k][1] * N;
        row.fills[0].style.width = at(bytes);
        row.fills[0].style.background = memFill(bytes / scale);
        tween(row.val, bytes, fmtBytes);
        paintRef(row);
      });
      memRows.kv.fills[0].style.width = at(kv);
      memRows.kv.fills[0].style.background = memFill(kv / scale);
      tween(memRows.kv.val, kv, fmtBytes);
      memRows.kv.sub.textContent = `T = ${fmtInt(T)} tokens · batch ${batch}`;
      paintRef(memRows.kv);
      R.kvCheck.replaceChildren(checkBadge(kvFormula(a, T, batch), kv, { label: 'equals engine.cacheBytes', tol: 0 }));
      let cum = 0;
      TRAIN_TIERS.forEach(([, bytes], k) => {
        const seg = memRows.train.fills[k], w = bytes * N;
        seg.style.left = at(cum);
        seg.style.width = at(w);
        seg.style.background = tierFill(k);
        cum += w;
      });
      tween(memRows.train.val, train, fmtBytes);
      paintRef(memRows.train);
    }

    function paintTraining() {
      const c = chinchilla(a, tpp, gpu), name = GPUS[gpu].name, hours = c.gpuDays * 24;
      R.cDlab.textContent = `Tokens D = ${tpp}N`;
      tween(R.cD, c.D, fmtShort);
      R.cDsub.textContent = tpp === 20 ? 'the Chinchilla rule (Hoffmann et al., 2022)' : `${tpp} tokens per parameter`;
      R.cC.innerHTML = texHTML(`${num(c.C, 2)}\\;\\mathrm{FLOP}`);
      R.cG.textContent = gpuSpan(hours);
      R.cGsub.textContent = `${ladderAbout(hours / 1000)} on 1,000 ${name}s at ${Math.round(MFU * 100)}% utilisation`;
      R.cEq.innerHTML = texHTML(`C \\approx 6\\,N\\,D = 6 \\times ${num(a.N, 2)} \\times ${num(c.D, 2)} \\approx ${num(c.C, 2)}\\;\\mathrm{FLOP}`, { display: true });
    }

    // One caption per step: one or two sentences, the number that matters in bold.
    const captions = [
      () => { const sh = paramShares(a); return `Each row counts one group of weights, and the bar adds them up to N. MLP blocks hold <b>${sh.mlp}</b> of N and attention <b>${sh.attn}</b>.`; },
      () => `A forward pass costs about 2N FLOPs of weight matmuls plus the attention scores. At <b>${fmtInt(T)}</b> tokens, attention is <b>${pct(attnShare(a, T))}</b> of the block FLOPs.`,
      () => `Weights take N × bytes per value. Training with Adam needs <b>${fmtBytes(16 * a.N)}</b> of state before activations, and the KV cache grows with T and the batch.`,
      () => `Training on <b>${tpp}</b> tokens per parameter takes <b>${fmtShort(tpp * a.N)}</b> tokens; ${tpp === 20 ? 'that is the compute-optimal point' : 'only 20 is compute-optimal'} (Hoffmann et al., 2022). GPU time assumes ${Math.round(MFU * 100)}% of peak dense bf16 throughput.`,
    ];

    // The selection, or the context of the loaded model, changed: rebuild what depends on the shape.
    function update() {
      sel = pick();
      a = arch(sel.cfg);
      if (sel.ctxDefault !== lastCtx) { lastCtx = sel.ctxDefault; T = sel.ctxDefault; ctxSlider.set(Math.log2(T)); }
      const key = JSON.stringify(sel.cfg), changed = key !== lastKey;
      lastKey = key;
      if (changed) { buildParams(step === 0 && moving()); drawChart(); }
      paintHeader();
      paintCompute();
      paintMemory();
      paintTraining();
      cap.set(captions[step]());
    }

    function showStep(i, { fresh = true } = {}) {
      step = i;
      panels.forEach((p, k) => { p.classList.toggle('on', k === i); p.toggleAttribute('inert', k !== i); });
      if (i === 0 && fresh) buildParams(moving());     // the sum adds up again each time it is entered
      cap.set(captions[i]());
    }

    const offs = [
      ctx.on('config', () => { if (alive && choice === 'app') update(); }),
      ctx.on('trace', () => { if (alive && choice === 'app') update(); }),   // weights may be re-initialised without a shape change
    ];

    ctx.setSteps(STEPS, (i) => { if (alive) showStep(i); });
    showStep(0, { fresh: false });
    update();
    appear([header.el, picker, hero], { stagger: 90 });

    // story: walk the steps on first view, unless the reader takes over
    const stopAuto = () => clearTimeout(auto);
    el.addEventListener('pointerdown', stopAuto, { once: true });
    el.addEventListener('atlas:interrupt', stopAuto, { once: true });
    const autoplay = () => {
      auto = setTimeout(() => {
        if (!alive || step >= STEPS.length - 1) return;
        ctx.setStep(step + 1);
        autoplay();
      }, step === 0 ? 2600 : MOTION.step);
    };
    autoplay();

    return {
      destroy() {
        alive = false;
        clearTimeout(auto);
        clearTimers();
        offs.forEach((f) => f());
        hideTip();
        st.destroy();
      },
    };
  },
};

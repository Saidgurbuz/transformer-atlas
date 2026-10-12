// Ablation lab (More menu, reference scene). One question: what breaks if one part of the model is removed?
// Every experiment edits a CLONE (engine.cloneModel) on the current text, with next-character targets, under noGrad.
// ctx.state.model is only read. Composition: before / after (the model and the copy).
// A mask that lets a position see the character it is scored on (no mask, or a prefix that covers it) leaks the
// answer: its losses are flagged, and the bidirectional row is marked as leakage instead of a model effect.
// Steps: the model, switch one part off, measure the loss, attention before and after, rank every ablation.
import { stage2d, sceneHeader, sceneCaption, headPicker, checkBadge, ZONES, MOTION, appear, countTo, useStyle } from '../kit.js';
import { h, fmt, texHTML, heatmap, diverging, onThemeChange, reducedMotion, tween, ease } from '../../ui/index.js';
import { cloneModel, infer, dHead, maskAllows } from '../../engine/index.js';

const STEPS = ['The model', 'Switch off', 'Measure', 'Attention', 'Rank'];
const HX = 110, HY = 300, HW = 1030;              // hero zone (design units): 1030 wide, 460 tall in the CSS
const SX = 1180, SY = 300;                         // side column (340 wide)
const CY = 196;                                    // controls row: experiment, layer, head (ZONES.controls)
const SEL_W = 440;                                 // width of the experiment select
const LW = 128;                                    // row-label width of the prediction strip
const PAT_W = 340;                                 // width of each attention heatmap (the query row sits above them)
const BARMAX = 110;                                // px, the longest per-position bar (either sign)
const SLICE_MS = 12;                               // time budget per macrotask while ablations run
const MOVE_MS = 600;                               // MOTION.move: values that morph or resize
const EPS = 1e-9;                                  // patterns closer than this count as identical
const MIN_SHOW = 0.3;                              // a small value is still drawn in a visible colour (matters in dark mode)
const TINY = 0.05;                                 // nats: when every effect is below this, the weights hardly matter
const NEED_TEXT = 'The lab needs at least two characters. Type them in the Input box of the ⚙ drawer, top right.';
const COPY_LEAK = 'With no mask each copy can see the character it is scored on, so this change includes leakage.';

useStyle('lab', `
.lb-select { width: 440px; height: 40px; padding: 0 12px; font: 600 16px/1 var(--font-body); border-radius: 10px; }
.lb-hero { position: absolute; width: 1030px; height: 460px; }
.lb-panel { position: absolute; inset: 0; opacity: 0; pointer-events: none; transition: opacity .5s cubic-bezier(.2,.7,.2,1); }
.lb-panel.on { opacity: 1; pointer-events: auto; }
.lb-bignote { display: grid; place-content: center; text-align: center; font-size: 20px; line-height: 1.5; color: var(--fg-2); }
.lb-qrow { position: absolute; left: 0; top: 0; height: 40px; display: flex; align-items: center; gap: 10px; }
.lb-qsel { width: 380px; height: 36px; padding: 0 10px; font: 600 15px/1 var(--font-mono); border-radius: 10px; }
.lb-strip { position: absolute; left: 0; top: 44px; width: 1030px; }
.lb-row { display: flex; align-items: center; height: 26px; margin-bottom: 8px; transition: opacity .4s ease; }
.lb-row.ghost { opacity: .3; }
.lb-r-d { height: 244px; align-items: stretch; margin-top: 20px; margin-bottom: 0; }
.lb-rl { flex: 0 0 ${LW}px; padding-right: 12px; text-align: right; font: 600 14px/1.2 var(--font-mono); color: var(--fg-3); white-space: nowrap; align-self: center; }
.lb-cells { display: flex; position: relative; height: 100%; }
.lb-c { width: var(--cw); margin-right: 1px; height: 26px; flex: none; display: grid; place-items: center; overflow: hidden; font: 600 var(--cf)/1 var(--font-mono); color: var(--fg); border-radius: 4px; background: color-mix(in srgb, var(--surface) 90%, transparent); box-shadow: inset 0 -1px 0 var(--line); transition: background-color .5s ease, color .5s ease, box-shadow .5s ease; }
.lb-r-next .lb-c { color: var(--fg-2); }
.lb-r-b .lb-c.flip { background: color-mix(in srgb, var(--c-head) 26%, var(--surface)); box-shadow: inset 0 0 0 1px var(--c-head); font-weight: 700; }
.lb-barzone { position: relative; flex: 1; height: 100%; }
.lb-hzero { position: absolute; left: 0; right: 0; top: 50%; border-top: 1px dashed var(--line-strong); }
.lb-cols { height: 100%; }
.lb-col { width: var(--cw); margin-right: 1px; height: 100%; flex: none; position: relative; }
.lb-bar { position: absolute; left: 16%; width: 68%; top: 50%; height: 0; border-radius: 3px; transition: top .6s cubic-bezier(.65,0,.35,1), height .6s cubic-bezier(.65,0,.35,1), background-color .4s ease; }
.lb-pat { position: absolute; left: 0; right: 0; top: 58px; display: flex; justify-content: center; gap: 52px; }
.lb-pcol { display: grid; gap: 10px; align-content: start; }
.lb-title { display: grid; gap: 2px; font: 600 15px/1.2 var(--font-mono); color: var(--fg-2); letter-spacing: .02em; white-space: nowrap; }
.lb-title small { font: 500 14px/1.2 var(--font-mono); color: var(--fg-3); }
.lb-rank { position: absolute; left: 0; top: 0; width: 1030px; display: grid; gap: 6px; align-content: start; }
.lb-rhead { font: 600 15px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
.lb-warn { font: 500 14px/1.45 var(--font-body); color: var(--warn); }
.lb-rrow { display: grid; grid-template-columns: 330px 1fr 92px; align-items: center; gap: 16px; height: 44px; padding: 0 12px; border-radius: 10px; cursor: pointer; transition: background-color .25s; }
.lb-rrow:hover { background: var(--surface-2); }
.lb-rrow:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.lb-rrow.sel { background: var(--accent-soft); }
.lb-rrow.na { cursor: default; opacity: .5; }
.lb-rrow.leak { opacity: .8; }
.lb-rname { font: 600 16px/1.2 var(--font-body); color: var(--fg); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lb-rname small { display: block; font: 500 14px/1.4 var(--font-mono); color: var(--fg-3); }
.lb-track { position: relative; height: 26px; }
.lb-vz { position: absolute; top: 0; bottom: 0; border-left: 1px solid var(--line-strong); }
.lb-rbar { position: absolute; top: 4px; bottom: 4px; border-radius: 5px; transform: scaleX(0); transition: transform .6s cubic-bezier(.65,0,.35,1); }
.lb-rval { font: 650 16px/1 var(--font-mono); text-align: right; color: var(--fg); font-variant-numeric: tabular-nums; }
.lb-rval.leak { font: 600 13px/1 var(--font-mono); letter-spacing: .05em; text-transform: uppercase; color: var(--warn); }
.lb-side { position: absolute; width: 340px; display: grid; gap: 12px; align-content: start; }
.lb-step { font: 600 17px/1.2 var(--font-mono); color: var(--fg-2); }
.lb-body { display: grid; gap: 12px; align-content: start; min-height: 0; }
.lb-box { border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); padding: 12px; display: grid; gap: 8px; }
.lb-eq { font-size: 18.2px; overflow-x: auto; overflow-y: hidden; }   /* KaTeX draws at 1.21em: about 22 px on screen */
.lb-eq .katex-display { margin: 0; }
.lb-tag { font: 600 15px/1.2 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .05em; }
.lb-rd-list { display: grid; border-top: 1px solid var(--line); }
.lb-rd { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; padding: 7px 0; border-bottom: 1px solid var(--line); }
.lb-rd-k { font: 500 14px/1.3 var(--font-mono); color: var(--fg-3); }
.lb-rd-v { font: 600 16px/1.3 var(--font-mono); color: var(--fg); text-align: right; font-variant-numeric: tabular-nums; }
.lb-note { font-size: 16px; line-height: 1.5; color: var(--fg-2); }
.lb-note b { color: var(--fg); }
.lb-status { font: 500 14px/1.2 var(--font-mono); color: var(--fg-3); min-height: 16px; }
.lb-grid { display: grid; gap: 4px; align-items: center; }
.lb-gl { font: 600 15px/1 var(--font-mono); color: var(--fg-3); text-align: center; }
.lb-gcell { height: 40px; border-radius: 7px; border: 1px solid transparent; font: 600 15px/1 var(--font-mono); color: var(--fg); background: var(--surface-2); cursor: pointer; transition: background-color .5s ease, border-color .2s, transform .2s; }
.lb-gcell:hover { border-color: var(--accent); transform: translateY(-1px); }
.lb-gcell.sel { border-color: var(--accent); }
@media (prefers-reduced-motion: reduce) {
  .lb-panel, .lb-c, .lb-bar, .lb-rbar, .lb-gcell, .lb-row { transition: none !important; }
}
`);

// ------------------------------------------------------------------ numerics

const sgn = (v, d = 2) => { const a = Math.abs(v).toFixed(d); return Number(a) === 0 ? a : (v < 0 ? '−' : '+') + a; };
const fix = (v, d = 2) => { if (!Number.isFinite(v)) return '–'; const s = v.toFixed(d); return (Number(s) === 0 ? (0).toFixed(d) : s).replace('-', '−'); };
const meanFinite = (xs) => { let s = 0, n = 0; for (const x of xs) if (Number.isFinite(x)) { s += x; n++; } return n ? s / n : NaN; };
const actName = (a) => (a === 'relu' ? 'ReLU' : a === 'gelu' ? 'GELU' : 'SwiGLU');
const normTex = (c) => (c.norm === 'rmsnorm' ? '\\RMSNorm' : '\\LN');
const aligned = (lines) => `\\begin{aligned}${lines.join(' \\\\ ')}\\end{aligned}`;
// Can position u see position u+1 under this mask? The engine's own rule, so a copy with no mask leaks.
function leaksAt(cfg, T) { for (let u = 0; u + 1 < T; u++) if (maskAllows(cfg, u, u + 1)) return true; return false; }
// Colour of a signed value in [-1, 1]; small values keep a visible tint.
const tint = (x) => diverging(Math.sign(x) * Math.max(Math.abs(x), MIN_SHOW));
// What the model's own mask does to its loss, in words (shown only when leaksAt is true for the model).
const modelLeakText = (cfg) => (cfg.mask === 'none'
  ? 'With no mask, each position can see the character it is scored on, so this loss does not measure prediction.'
  : 'This prefix mask lets some positions see the character they are scored on, so this loss does not measure prediction.');

// Per-position cross-entropy −log softmax(logits_t)[target_t]; NaN where there is no target.
function ceAll(logits, T, V, targets) {
  const out = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const tg = targets[t];
    if (tg < 0) { out[t] = NaN; continue; }
    const o = t * V;
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[o + v] > mx) mx = logits[o + v];
    let z = 0;
    for (let v = 0; v < V; v++) z += Math.exp(logits[o + v] - mx);
    out[t] = Math.log(z) + mx - logits[o + tg];
  }
  return out;
}

function top1All(logits, T, V) {
  const out = new Int32Array(T);
  for (let t = 0; t < T; t++) {
    const o = t * V;
    let best = 0;
    for (let v = 1; v < V; v++) if (logits[o + v] > logits[o + best]) best = v;
    out[t] = best;
  }
  return out;
}

// Which parameters differ between the model and a copy, and how many entries.
function diffParams(o, c) {
  const out = [];
  for (const k of Object.keys(o.params)) {
    const a = o.params[k].data, b = c.params[k].data;
    let n = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
    if (n) out.push([k, n]);
  }
  return out;
}

const sameValues = (a, b) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };

// Largest change between two attention patterns (flat [H·T·T] arrays) for one head, and the change per query row.
function patternDiff(A, B, off, T) {
  let max = 0, at = 0;
  const rowMax = new Float64Array(T);
  for (let r = 0; r < T; r++) for (let c = 0; c < T; c++) {
    const i = r * T + c, dd = Math.abs(B[off + i] - A[off + i]);
    if (dd > rowMax[r]) rowMax[r] = dd;
    if (dd > max) { max = dd; at = i; }
  }
  return { max, at, rowMax };
}

// ------------------------------------------------------------------ edits (applied to clones only)

function zeroHead(m, li, hi) {
  const cfg = m.config, dh = dHead(cfg), W = m.params[`blocks.${li}.attn.o_proj.weight`];
  const width = cfg.n_head * dh, c0 = hi * dh, c1 = c0 + dh;          // o_proj is [d, H·dh]: head h owns columns h·dh … (h+1)·dh−1
  for (let r = 0; r < cfg.d_model; r++) for (let c = c0; c < c1; c++) W.data[r * width + c] = 0;
}

function zeroMLP(m, li) {
  for (const k of [`blocks.${li}.mlp.down_proj.weight`, `blocks.${li}.mlp.down_proj.bias`]) m.params[k]?.data.fill(0);
}

function swapBlocks(m, a, b) {
  const pa = `blocks.${a}.`, pb = `blocks.${b}.`;
  for (const k of Object.keys(m.params)) {
    if (!k.startsWith(pa)) continue;
    const x = m.params[k], y = m.params[pb + k.slice(pa.length)];
    if (!y) continue;
    const tmp = x.data; x.data = y.data; y.data = tmp;
  }
}

function configCheck(key) {
  return (o, c) => {
    const keys = Object.keys(o.config).filter((k) => o.config[k] !== c.config[k]);
    const weights = diffParams(o, c).length;
    return { ok: keys.join() === key && weights === 0, text: `${keys.map((k) => `${k}: ${o.config[k]} → ${c.config[k]}`).join(', ')} · weights unchanged` };
  };
}

function headCheck(o, c, t) {
  const W = `blocks.${t.layer}.attn.o_proj.weight`, dh = dHead(o.config), width = o.config.n_head * dh;
  const c0 = t.head * dh, c1 = c0 + dh, d = o.config.d_model, data = c.params[W].data;
  let zero = true;
  for (let r = 0; r < d && zero; r++) for (let k = c0; k < c1; k++) if (data[r * width + k] !== 0) { zero = false; break; }
  const names = diffParams(o, c);
  const n = names.find(([k]) => k === W)?.[1] ?? 0;
  return { ok: zero && names.every(([k]) => k === W), text: `o_proj: ${n} entries set to 0, columns ${c0}–${c1 - 1}` };
}

function mlpCheck(o, c, t) {
  const pre = `blocks.${t.layer}.mlp.down_proj.`;
  const names = diffParams(o, c);
  const zeros = ['weight', 'bias'].every((p) => { const x = c.params[`${pre}${p}`]; return !x || x.data.every((v) => v === 0); });
  const n = names.reduce((s, [, k]) => s + k, 0);
  return { ok: zeros && names.every(([k]) => k.startsWith(pre)), text: `down_proj: ${n} entries set to 0` };
}

function swapCheck(o, c, t) {
  const pa = `blocks.${t.a}.`, pb = `blocks.${t.b}.`;
  let ok = diffParams(o, c).every(([k]) => k.startsWith(pa) || k.startsWith(pb));
  let n = 0;
  for (const k of Object.keys(o.params)) {
    if (!k.startsWith(pa)) continue;
    const other = pb + k.slice(pa.length);
    n++;
    if (!sameValues(c.params[k].data, o.params[other].data) || !sameValues(c.params[other].data, o.params[k].data)) ok = false;
  }
  return { ok, text: `${n} tensors per block exchanged between layers ${t.a} and ${t.b}` };
}

// Adjacent pair for the layer swap: the selected layer and the one after it (the one before it for the last layer).
const pairStart = (li, n) => (li < n - 1 ? li : Math.max(0, li - 1));
const tgtFor = (li, hi, n) => ({ layer: li, head: hi, a: pairStart(li, n), b: pairStart(li, n) + 1 });

// Each experiment: na(config, target) gives a reason when it does not apply; apply(clone, target) edits the clone.
// before / after give the equation of the part, for the model and for the copy, in the model's own norm and placement.
// leaks(config, T), when given, says whether the copy lets a position see the character it is scored on.
const EXPS = [
  {
    id: 'residual', title: 'Remove the residual connections', short: 'no residual connections', target: null,
    na: (c) => (c.residual ? null : 'the residual connections are already off'),
    apply: (m) => { m.config.residual = false; },
    before: (c) => {
      const N = normTex(c);
      return c.norm_placement === 'post'
        ? aligned([String.raw`y &= ${N}(x+\Attn(x))`, String.raw`x_{\ell+1} &= ${N}(y+\MLP(y))`])
        : aligned([String.raw`y &= x+\Attn(${N}(x))`, String.raw`x_{\ell+1} &= y+\MLP(${N}(y))`]);
    },
    after: (c) => {
      const N = normTex(c);
      return c.norm_placement === 'post'
        ? aligned([String.raw`y &= ${N}(\Attn(x))`, String.raw`x_{\ell+1} &= ${N}(\MLP(y))`])
        : aligned([String.raw`y &= \Attn(${N}(x))`, String.raw`x_{\ell+1} &= \MLP(${N}(y))`]);
    },
    check: configCheck('residual'),
  },
  {
    id: 'scale', title: 'Drop the 1/√d_h scaling', short: 'no 1/√d_h scaling', target: null,
    na: (c) => (c.attn_scale ? null : 'the 1/√d_h factor is already off'),
    apply: (m) => { m.config.attn_scale = false; },
    before: () => String.raw`\Aw=\softmax\!\big(\tfrac{\Q\K^\top}{\sqrt{d_h}}\big)`,
    after: () => String.raw`\Aw=\softmax\!\big(\Q\K^\top\big)`,
    check: configCheck('attn_scale'),
  },
  {
    id: 'bidir', title: 'Bidirectional mask', short: 'bidirectional mask', target: null,
    na: (c) => (c.mask === 'none' ? 'the mask is already off' : null),
    apply: (m) => { m.config.mask = 'none'; },
    leaks: (c, T) => leaksAt({ ...c, mask: 'none' }, T),
    before: (c) => (c.mask === 'sliding' ? String.raw`M_{ij}=\mathbf{1}[\,j\le i\,]\,\mathbf{1}[\,j>i-${c.window}\,]`
      : c.mask === 'prefix' ? String.raw`M_{ij}=\mathbf{1}[\,j\le i\,]\lor\mathbf{1}[\,j<${c.prefix_len}\,]`
        : String.raw`M_{ij}=\mathbf{1}[\,j\le i\,]`),
    after: () => String.raw`M_{ij}=1\quad\text{for all } i,j`,
    check: configCheck('mask'),
  },
  {
    id: 'nopos', title: 'No positional information', short: 'no positional information', target: null,
    na: (c) => (c.pos_encoding === 'none' ? 'the model has no positional information already' : null),
    apply: (m) => { m.config.pos_encoding = 'none'; },
    before: (c) => (c.pos_encoding === 'rope' ? String.raw`q_t=R_t\,W_Q x_t` : c.pos_encoding === 'alibi' ? String.raw`s_{ij}=\cdots-m_h|i-j|` : String.raw`x_t=E_{x_t}+P_t`),
    after: (c) => (c.pos_encoding === 'rope' ? String.raw`q_t=W_Q x_t` : c.pos_encoding === 'alibi' ? String.raw`s_{ij}=\cdots` : String.raw`x_t=E_{x_t}`),
    check: configCheck('pos_encoding'),
  },
  {
    id: 'act', title: 'Swap GELU and ReLU', short: 'GELU ↔ ReLU', target: null,
    na: (c) => (c.activation === 'swiglu' ? 'this model uses SwiGLU' : null),
    apply: (m) => { m.config.activation = m.config.activation === 'gelu' ? 'relu' : 'gelu'; },
    before: (c) => String.raw`\MLP(x)=\Wdown\,${actName(c.activation)}(\Wup x)`,
    after: (c) => String.raw`\MLP(x)=\Wdown\,${actName(c.activation === 'relu' ? 'gelu' : 'relu')}(\Wup x)`,
    check: configCheck('activation'),
  },
  {
    id: 'head', title: 'Zero one attention head', short: 'one attention head', target: 'head',
    na: () => null,
    apply: (m, t) => zeroHead(m, t.layer, t.head),
    before: (c, t) => String.raw`\Attn_{${t.layer}}(x)=\sum_{h'}\Aw^{(h')}\V^{(h')}\Wo^{(h')}`,
    after: (c, t) => String.raw`\Attn_{${t.layer}}(x)=\sum_{h'\neq ${t.head}}\Aw^{(h')}\V^{(h')}\Wo^{(h')}`,
    check: headCheck,
  },
  {
    id: 'mlp', title: 'Zero the MLP of one layer', short: 'the MLP of one layer', target: 'layer',
    na: () => null,
    apply: (m, t) => zeroMLP(m, t.layer),
    before: (c, t) => (c.norm_placement === 'post'
      ? String.raw`x_{${t.layer + 1}}=${normTex(c)}\big(y+\MLP(y)\big)`
      : String.raw`x_{${t.layer + 1}}=y+\MLP(${normTex(c)}(y))`),
    after: (c, t) => (c.norm_placement === 'post'
      ? String.raw`x_{${t.layer + 1}}=${normTex(c)}(y)`
      : String.raw`x_{${t.layer + 1}}=y+0=y`),
    check: mlpCheck,
  },
  {
    id: 'swap', title: 'Swap two adjacent layers', short: 'two adjacent layers', target: 'pair',
    na: (c) => (c.n_layer < 2 ? 'the model has one layer' : null),
    apply: (m, t) => swapBlocks(m, t.a, t.b),
    before: (c, t) => String.raw`x \leftarrow B_{${t.b}}\big(B_{${t.a}}(x)\big)`,
    after: (c, t) => String.raw`x \leftarrow B_{${t.a}}\big(B_{${t.b}}(x)\big)`,
    check: swapCheck,
  },
];
const HEAD_EXP = EXPS.findIndex((e) => e.id === 'head');

function targetText(exp, t) {
  if (exp.target === 'head') return `layer ${t.layer} · head ${t.head}`;
  if (exp.target === 'layer') return `layer ${t.layer}`;
  if (exp.target === 'pair') return `layers ${t.a} and ${t.b}`;
  return 'whole model';
}

// An experiment whose copy leaks the answer: it is shown, but it is not a model effect and is not ranked.
const leakRow = (i, cfg, T, t) => { const e = EXPS[i]; return !!e.leaks && !e.na(cfg, t) && e.leaks(cfg, T); };

const eqEl = (tex) => h('div', { class: 'lb-eq', html: texHTML(tex, { display: true }) });
const rd = (k, ...v) => h('div', { class: 'lb-rd' }, h('span', { class: 'lb-rd-k' }, k), h('span', { class: 'lb-rd-v' }, ...v));
const warn = (text) => h('div', { class: 'lb-warn' }, text);
const badge = (ok, good, bad) => h('span', { class: ['k-check', ok ? 'ok' : 'bad'] }, ok ? `✓ ${good}` : `✗ ${bad}`);

// Layer chips (L0 … L{n−1}) in the kit's picker style, the same shape as headPicker: selects the layer for the whole app.
function layerPicker(stage, ctx, { x, y } = {}) {
  const row = h('div', { class: 'k-picker-wrap lb-lpick' });
  stage.add(row, { x, y });
  function render() {
    const n = ctx.cfg.n_layer, cur = Math.min(ctx.state.sel.layer ?? 0, n - 1);
    row.replaceChildren(h('span', { class: 'k-picker-label' }, 'layer'), ...Array.from({ length: n }, (_, i) => {
      const b = h('button', { type: 'button', class: ['k-pick', 'k-pick-head', i === cur && 'sel'], style: { '--kind': 'var(--c-resid)' }, 'aria-pressed': i === cur ? 'true' : 'false' }, h('b', null, `L${i}`));
      b.addEventListener('click', () => ctx.store.select({ layer: i }));
      return b;
    }));
  }
  render();
  ctx.on('select', render);
  return { el: row, render };
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const S = () => ctx.state;
    const nL = () => ctx.cfg.n_layer, nH = () => ctx.cfg.n_head;
    const curL = () => Math.min(S().sel.layer ?? 0, nL() - 1);
    const curH = () => Math.min(S().sel.head ?? 0, nH() - 1);
    const lab = (id) => S().tokenizer.label(id);
    const labAt = (bs, i) => S().tokenizer.label(bs.ids[i]);
    // 'trained' only when a preset is loaded; the store re-initialises random weights when a shape changes
    const modelWord = () => { const w = String(S().weights); return w === 'random' ? 'random' : w.startsWith('preset:') ? 'trained' : 'loaded'; };
    const baseLeaks = () => leaksAt(S().config, S().ids.length);

    let step = 0, expIdx = 0, gen = 0;
    let base = null, det = null, rankRes = null, gridRes = null;
    let paused = false, lastStep = -1, lastLH = '';
    let queueId = 0, queueKey = '';
    let auto = null, autoTick = false, cancelMorph = null, morphKey = '';
    let hmA = null, hmB = null;
    let stripGen = -1, lastBarsKey = '', gridBuilt = '', gridCells = [];
    let cellsT = [], cellsN = [], cellsA = [], cellsB = [], barsD = [], rowB = null, rowD = null;

    // ------------------------------------------------------------- static parts (shared kit)
    const header = sceneHeader(st, {
      chips: [['Ablation', null], ['Next-character loss', 'head']], title: 'Ablation lab',
      sub: 'Each experiment edits a copy of the model, runs the same text and measures the change in loss. The model itself is never changed.',
    });

    // controls row: the experiment (one select), the layer and head (kit-style pickers)
    const expSel = h('select', { class: 'select lb-select', id: 'lab-experiment', 'aria-label': 'Experiment' });
    EXPS.forEach((_, i) => expSel.append(h('option', { value: String(i) })));
    expSel.addEventListener('change', () => pickExp(Number(expSel.value)));
    st.add(expSel, { x: ZONES.controls.x, y: CY });
    const layerPick = layerPicker(st, ctx, { x: ZONES.controls.x, y: CY + 4 });
    const headPick = headPicker(st, ctx, { x: ZONES.controls.x, y: CY + 4 });

    const hero = h('div', { class: 'lb-hero' });
    const panelStrip = h('div', { class: 'lb-panel' });
    const panelPattern = h('div', { class: 'lb-panel' });
    const panelRank = h('div', { class: 'lb-panel' });
    const noteBox = h('div', { class: 'lb-panel lb-bignote' });
    // the query row for the attention step, in the hero above the two patterns (a token picker needs up to 1360 units at 64 tokens)
    const qSel = h('select', { class: 'select lb-qsel', id: 'lab-query', 'aria-label': 'Query position' });
    qSel.addEventListener('change', () => ctx.store.select({ token: Number(qSel.value) }));
    const qRow = h('div', { class: 'lb-qrow' }, h('span', { class: 'k-picker-label' }, 'query'), qSel);
    hero.append(panelStrip, panelPattern, panelRank, noteBox, qRow);
    st.add(hero, { x: HX, y: HY });

    const side = h('div', { class: 'lb-side' });
    const stepTag = h('div', { class: 'lb-step' });
    const body = h('div', { class: 'lb-body' });
    const gridWrap = h('div', { class: 'lb-grid-wrap' });
    const statusEl = h('div', { class: 'lb-status' });
    side.append(stepTag, body);
    st.add(side, { x: SX, y: SY });

    const cap = sceneCaption(st, { y: 790, w: 1060 });
    // The kit cross-fades captions, and an older fade can write its text after a newer one. A newer caption here
    // cancels a fade still running, so only the latest text is ever written (the same pattern as lm-head.js).
    let capText = null;
    const capSet = (html) => {
      if (html === capText) return;
      capText = html;
      if (reducedMotion()) { cap.el.innerHTML = html; return; }
      cap.el.getAnimations().forEach((a) => a.cancel());
      cap.el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 140 }).finished.then(() => {
        if (html !== capText) return;
        cap.el.innerHTML = html;
        cap.el.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 260, easing: MOTION.easeOut });
      }).catch(() => {});
    };

    // the layer and head chips follow the selection's width (the layer row grows with n_layer)
    function placeControls() {
      const x = ZONES.controls.x + SEL_W + 24;
      st.move(layerPick.el, x, CY + 4);
      st.move(headPick.el, x + layerPick.el.offsetWidth + 28, CY + 4);
    }

    // ------------------------------------------------------------- data
    // Model reference for the current text (one traced forward, read from ctx.state).
    function getBase() {
      if (base && base.gen === gen) return base;
      const s = S(), ids = s.ids, T = ids.length;
      if (T < 2 || !s.logits) return null;
      const targets = ids.map((_, t) => (t + 1 < T ? ids[t + 1] : -1));
      const out = infer(s.model, ids, { targets, trace: false });
      const V = s.logits.data.length / T;
      const ceA = ceAll(s.logits.data, T, V, targets);
      base = { gen, T, V, ids: ids.slice(), targets, lossA: out.loss.item(), tracedA: meanFinite(ceA), ceA, top1A: top1All(s.logits.data, T, V) };
      return base;
    }

    // One experiment on the clone: loss, top-1 changes and the attention pattern of one head (cached per selection).
    // When this edit cannot reach the selected head (its layer reads the same input), the first later layer where the head changes is shown.
    function getDetail() {
      const bs = getBase();
      if (!bs) return null;
      const s = S(), li = curL(), hi = curH(), exp = EXPS[expIdx], t = tgtFor(li, hi, nL());
      const key = `${gen}|${expIdx}|${li}|${hi}`;
      if (det && det.key === key) return det;
      const why = exp.na(s.config, t);
      if (why) { det = { key, exp, t, layer: li, head: hi, na: why }; return det; }
      const m = cloneModel(s.model);
      exp.apply(m, t);
      const { T, V } = bs, n2 = T * T, off = hi * n2;
      const out = infer(m, bs.ids, { targets: bs.targets, trace: true });
      const ceB = ceAll(out.logits.data, T, V, bs.targets);
      const top1B = top1All(out.logits.data, T, V);
      const dCE = new Float64Array(T);
      let flips = 0, topPos = -1, topVal = -Infinity;
      for (let i = 0; i < T; i++) {
        dCE[i] = ceB[i] - bs.ceA[i];
        if (top1B[i] !== bs.top1A[i]) flips++;
        if (Number.isFinite(dCE[i]) && dCE[i] > topVal) { topVal = dCE[i]; topPos = i; }
      }
      const headDiff = (l) => patternDiff(s.trace[`L${l}.attn.probs`].data, out.trace[`L${l}.attn.probs`].data, off, T);
      let shown = li;
      if (headDiff(li).max < EPS) {
        for (let l = li + 1; l < nL(); l++) if (headDiff(l).max >= EPS) { shown = l; break; }
      }
      const probsA = s.trace[`L${shown}.attn.probs`].data.subarray(off, off + n2);
      const probsB = out.trace[`L${shown}.attn.probs`].data.subarray(off, off + n2);
      const pd = patternDiff(probsA, probsB, 0, T);
      let devB = 0;
      for (let r = 0; r < T; r++) { let sum = 0; for (let c = 0; c < T; c++) sum += probsB[r * T + c]; devB = Math.max(devB, Math.abs(sum - 1)); }
      det = {
        key, exp, t, layer: li, head: hi, shown, na: null,
        lossB: out.loss.item(), top1B, dCE, flips, topPos, topVal, meanDelta: meanFinite(dCE),
        probsA, probsB, maxD: pd.max, maxR: Math.floor(pd.at / T), maxC: pd.at % T, rowMax: pd.rowMax, devB,
        identical: pd.max < EPS, copy: exp.check(s.model, m, t),
      };
      return det;
    }

    // Loss change of one experiment on this text (loss only, no trace).
    function evalDelta(exp, t, bs) {
      const m = cloneModel(S().model);
      exp.apply(m, t);
      return infer(m, bs.ids, { targets: bs.targets, trace: false }).loss.item() - bs.lossA;
    }

    // ------------------------------------------------------------- time-sliced runs
    function cancelQueue() { queueId++; queueKey = ''; statusEl.textContent = ''; }

    function startQueue(jobs, key) {
      const id = ++queueId;
      queueKey = key;
      let i = 0;
      const slice = () => {
        if (id !== queueId) return;
        const t0 = performance.now();
        do { jobs[i](); i++; } while (i < jobs.length && performance.now() - t0 < SLICE_MS);
        if (i < jobs.length) { statusEl.textContent = `Running ${i} of ${jobs.length}…`; setTimeout(slice, 0); return; }
        queueKey = ''; statusEl.textContent = '';
        // text only: the ranking keeps its bars (and their growth), the caption and the grid note catch up
        if (step === 4) { const bs = getBase(); if (bs) { renderSide(bs, null); capSet(capFor(bs, null)); } }
      };
      statusEl.textContent = `Running 0 of ${jobs.length}…`;
      setTimeout(slice, 0);
    }

    // Ranking (8 experiments on the selected layer and head) and the layer × head grid, one forward per job.
    function runAll() {
      const bs = getBase();
      if (!bs) return;
      const li = curL(), hi = curH(), n = nL(), nh = nH(), key = `${gen}|${li}|${hi}`;
      const needRank = !(rankRes && rankRes.gen === gen && rankRes.layer === li && rankRes.head === hi && rankRes.vals.every((v) => v !== undefined));
      const needGrid = !(gridRes && gridRes.gen === gen && gridRes.vals.every((v) => v !== undefined));
      if (!needRank && !needGrid) return;
      if (queueKey === key) return;
      cancelQueue();
      const jobs = [];
      if (needRank) {
        const rr = { gen, layer: li, head: hi, vals: EXPS.map(() => undefined) };
        rankRes = rr;
        EXPS.forEach((exp, i) => jobs.push(() => {
          const t = tgtFor(li, hi, n);
          rr.vals[i] = exp.na(S().config, t) ? null : evalDelta(exp, t, bs);
          if (rr === rankRes && rr.vals.every((v) => v !== undefined)) paintRank(true);
        }));
      }
      if (needGrid) {
        if (!gridRes || gridRes.gen !== gen) gridRes = { gen, L: n, H: nh, vals: new Array(n * nh).fill(undefined) };
        const gr = gridRes;
        for (let a = 0; a < n; a++) for (let b = 0; b < nh; b++) {
          const k = a * nh + b;
          if (gr.vals[k] !== undefined) continue;
          jobs.push(() => { gr.vals[k] = evalDelta(EXPS[HEAD_EXP], tgtFor(a, b, n), bs); paintGrid(); });
        }
      }
      startQueue(jobs, key);
    }

    // ------------------------------------------------------------- strip: model and copy, token by token
    // The copy and the deltas are shown as faint rows before their step (the hero is filled from the start).
    function buildStrip(bs) {
      const T = bs.T;
      stripGen = gen;
      lastBarsKey = '';
      cellsT = []; cellsN = []; cellsA = []; cellsB = []; barsD = [];
      const cellW = Math.max(9, Math.min(45, Math.floor((HW - LW - 8) / T)));
      const cw = cellW - 1;
      const cf = cw >= 30 ? 17 : cw >= 22 ? 15 : cw >= 15 ? 13 : 11;
      const cellRow = (cls, label, store) => {
        const cells = h('div', { class: 'lb-cells' });
        for (let t = 0; t < T; t++) { const c = h('div', { class: 'lb-c' }); store.push(c); cells.append(c); }
        return h('div', { class: ['lb-row', cls] }, h('span', { class: 'lb-rl' }, label), cells);
      };
      const rT = cellRow('lb-r-text', 'input', cellsT);
      const rN = cellRow('lb-r-next', 'next', cellsN);
      const rA = cellRow('lb-r-a', 'model · top-1', cellsA);
      rA.dataset.trace = 'logits';                      // the model's own logits: hover and right-click work on each top-1 cell
      rowB = cellRow('lb-r-b', 'copy · top-1', cellsB); // the copy: not a traced tensor of the model, so no trace key
      const cols = h('div', { class: 'lb-cells lb-cols' });
      for (let t = 0; t < T; t++) { const bar = h('i', { class: 'lb-bar' }); barsD.push(bar); cols.append(h('div', { class: 'lb-col' }, bar)); }
      rowD = h('div', { class: ['lb-row', 'lb-r-d'] }, h('span', { class: 'lb-rl' }, 'Δ loss'), h('div', { class: 'lb-barzone' }, h('i', { class: 'lb-hzero' }), cols));
      panelStrip.replaceChildren(h('div', { class: 'lb-strip', style: { '--cw': `${cw}px`, '--cf': `${cf}px` } }, rT, rN, rA, rowB, rowD));
      for (let t = 0; t < T; t++) {
        cellsT[t].textContent = labAt(bs, t);
        cellsN[t].textContent = t + 1 < T ? labAt(bs, t + 1) : '–';
      }
    }

    function paintStrip(bs, d, stepNow) {
      const T = bs.T, A = bs.top1A, ok = !!d && !d.na;
      const showB = ok && stepNow >= 1, showD = ok && stepNow >= 2;
      rowB.classList.toggle('ghost', stepNow < 1);
      rowD.classList.toggle('ghost', stepNow < 2);
      for (let t = 0; t < T; t++) {
        cellsA[t].textContent = lab(A[t]);
        cellsA[t].dataset.cell = `${t},${A[t]},0`;
      }
      for (let t = 0; t < T; t++) {
        const cb = cellsB[t];
        if (showB) {
          cb.textContent = lab(d.top1B[t]);
          cb.classList.toggle('flip', d.top1B[t] !== A[t]);
          cb.title = `position ${t}: next “${t + 1 < T ? labAt(bs, t + 1) : '–'}” · model “${lab(A[t])}” · copy “${lab(d.top1B[t])}”`;
        } else { cb.textContent = ''; cb.classList.remove('flip'); cb.title = ''; }
      }
      const key = ok ? d.key : 'none';
      let mx = 0;
      if (ok) for (let t = 0; t < T - 1; t++) { const v = d.dCE[t]; if (Number.isFinite(v)) mx = Math.max(mx, Math.abs(v)); }
      const scale = Math.max(0.1, mx);
      const grow = showD && !reducedMotion() && (key !== lastBarsKey || (stepNow === 2 && lastStep !== 2));
      if (grow) {
        barsD.forEach((b) => { b.style.transition = 'none'; b.style.height = '0px'; });
        void rowD.offsetWidth;
        barsD.forEach((b) => { b.style.transition = ''; });
      }
      if (showD) lastBarsKey = key;
      for (let t = 0; t < T; t++) {
        const b = barsD[t], v = showD ? d.dCE[t] : NaN, finite = Number.isFinite(v);
        const hp = finite ? (Math.abs(v) / scale) * BARMAX : 0;
        b.style.transitionDelay = `${Math.min(600, t * 12)}ms`;
        b.style.height = `${hp}px`;
        b.style.top = finite && v >= 0 ? `calc(50% - ${hp}px)` : '50%';
        b.style.background = finite ? tint(v / scale) : 'transparent';
        b.title = finite ? `position ${t} (next “${labAt(bs, t + 1)}”): ${sgn(v)} nats` : '';
      }
    }

    // ------------------------------------------------------------- pattern: model and copy, morphing
    // The query row is a select above the two patterns; it lists every position of the text.
    function paintQuery(bs, q) {
      if (qSel.dataset.gen !== String(gen)) {
        qSel.dataset.gen = String(gen);
        qSel.replaceChildren(...Array.from({ length: bs.T }, (_, i) => h('option', { value: String(i) }, `${i} · ${labAt(bs, i)}`)));
      }
      qSel.value = String(q);
    }

    function paintPattern(bs, d) {
      const T = bs.T, q = Math.min(S().sel.token ?? T - 1, T - 1);
      paintQuery(bs, q);
      if (morphKey !== d.key) {
        morphKey = d.key;
        cancelMorph?.(); cancelMorph = null;
        hmA?.destroy(); hmB?.destroy();
        const A = d.probsA, B = d.probsB, buf = Float64Array.from(A);
        const tip = (r, c, v) => `query “${labAt(bs, r)}” at ${r} → key “${labAt(bs, c)}” at ${c}: <b>${fmt(v, 2)}</b>`;
        const onClick = (r) => ctx.store.select({ token: r });
        hmA = heatmap({ data: A, rows: T, cols: T, mode: 'sequential', kind: 'attn', range: [0, 1], maxWidth: PAT_W, digits: 2, highlight: { rows: [q] }, traceKey: `L${d.shown}.attn.probs`, tooltip: tip, onClick, ariaLabel: 'Model attention pattern' });
        hmB = heatmap({ data: buf, rows: T, cols: T, mode: 'sequential', kind: 'attn', range: [0, 1], maxWidth: PAT_W, digits: 2, highlight: { rows: [q] }, tooltip: tip, onClick, ariaLabel: 'Copy attention pattern' });
        const legend = 'rows are queries, columns are keys';
        panelPattern.replaceChildren(h('div', { class: 'lb-pat' },
          h('div', { class: 'lb-pcol' }, h('div', { class: 'lb-title' }, `model · ${modelWord()} · layer ${d.shown}, head ${d.head}`, h('small', null, legend)), hmA.el),
          h('div', { class: 'lb-pcol' }, h('div', { class: 'lb-title' }, `copy · ${d.exp.short}`, h('small', null, legend)), hmB.el)));
        cancelMorph = tween({ duration: MOVE_MS, easing: ease.inOut, onUpdate: (e) => {
          for (let i = 0; i < buf.length; i++) buf[i] = A[i] + (B[i] - A[i]) * e;
          hmB.update({ data: buf });
        } });
      } else {
        hmA.update({ highlight: { rows: [q] } });
        hmB.update({ highlight: { rows: [q] } });
      }
    }

    // ------------------------------------------------------------- ranking and grid
    const rankOf = (v) => (Number.isFinite(v) ? v : v === undefined ? -1e9 : -2e9);

    function paintRank(animate0) {
      const animate = animate0 && !reducedMotion();
      const cfg = S().config, T = S().ids.length;
      const li = curL(), hi = curH(), t0 = tgtFor(li, hi, nL());
      const valid = !!rankRes && rankRes.gen === gen && rankRes.layer === li && rankRes.head === hi;
      const vals = valid ? rankRes.vals : EXPS.map(() => undefined);
      const leak = EXPS.map((_, i) => leakRow(i, cfg, T, t0));
      const fin = vals.filter((v, i) => Number.isFinite(v) && !leak[i]);
      const lo = Math.min(0, ...fin), hiV = Math.max(0, ...fin);
      // bars share one scale with a floor of 0.1 nats, so a tiny effect cannot fill the track
      const span = Math.max(hiV - lo, 0.1), mAbs = Math.max(-lo, hiV, 0.1);
      const z = (-lo / span) * 100;
      const done = vals.every((v) => v !== undefined);
      // ranked effects first (largest first), then the leaking row, then the rows still running, then the rows that do not apply
      const cls = (i) => (vals[i] === null ? 3 : leak[i] ? 2 : Number.isFinite(vals[i]) ? 0 : 1);
      const order = EXPS.map((_, i) => i).sort((x, y) => cls(x) - cls(y) || rankOf(vals[y]) - rankOf(vals[x]) || x - y);
      const bars = [];
      const rows = order.map((i, k) => {
        const exp = EXPS[i], v = vals[i], isLeak = leak[i], na = v === null;
        const row = h('div', { class: ['lb-rrow', i === expIdx && 'sel', na && 'na', isLeak && 'leak'], title: exp.title, role: 'button', tabindex: na ? '-1' : '0' });
        const sub = na ? `not applicable: ${exp.na(cfg, t0)}` : isLeak ? 'not ranked: the copy can see the answer' : targetText(exp, t0);
        const track = h('div', { class: 'lb-track' }, h('i', { class: 'lb-vz', style: { left: `${z}%`, height: '100%' } }));
        if (Number.isFinite(v) && !isLeak) {
          const w = (Math.abs(v) / span) * 100;
          const bar = h('i', { class: 'lb-rbar', style: {
            left: `${v >= 0 ? z : z - w}%`, width: `${w}%`, background: tint(v / mAbs),
            transformOrigin: v >= 0 ? 'left' : 'right', transitionDelay: animate ? `${k * 70}ms` : '0ms',
            transform: animate ? 'scaleX(0)' : 'scaleX(1)',
          } });
          bars.push(bar);
          track.append(bar);
        }
        const value = isLeak ? h('div', { class: 'lb-rval leak' }, 'leak') : h('div', { class: 'lb-rval' }, Number.isFinite(v) ? sgn(v) : na ? '–' : '…');
        row.append(h('div', { class: 'lb-rname' }, exp.title, h('small', null, sub)), track, value);
        if (!na) {
          row.addEventListener('click', () => pickExp(i));
          row.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickExp(i); } });
        }
        return row;
      });
      const warns = [];
      if (baseLeaks()) warns.push(warn('This model’s mask lets positions see their targets, so every change here is measured on a leaky loss.'));
      if (done && fin.length && fin.every((v) => Math.abs(v) < TINY)) warns.push(warn(`Every effect is below ${TINY} nats${modelWord() === 'random' ? ': the weights are untrained.' : ' on this text.'}`));
      panelRank.replaceChildren(h('div', { class: 'lb-rank' },
        h('div', { class: 'lb-rhead' }, `Change in loss on this text, nats · layer ${li}, head ${hi}`), ...warns, ...rows));
      if (animate && bars.length) { void panelRank.offsetWidth; bars.forEach((b) => { b.style.transform = 'scaleX(1)'; }); }
    }

    function buildGrid() {
      const n = nL(), nh = nH();
      gridBuilt = `${n}x${nh}`;
      gridCells = [];
      const grid = h('div', { class: 'lb-grid', style: { gridTemplateColumns: `36px repeat(${nh}, 1fr)` } });
      grid.append(h('span'), ...Array.from({ length: nh }, (_, j) => h('span', { class: 'lb-gl' }, `h${j}`)));
      for (let a = 0; a < n; a++) {
        grid.append(h('span', { class: 'lb-gl' }, `L${a}`));
        for (let b = 0; b < nh; b++) {
          const c = h('button', { type: 'button', class: 'lb-gcell', title: `Zero head ${b} in layer ${a}` }, '…');
          c.addEventListener('click', () => pickHead(a, b));
          gridCells.push(c);
          grid.append(c);
        }
      }
      gridWrap.replaceChildren(h('div', { class: 'lb-rhead' }, 'Zero one head · Δ loss, nats'), grid);
    }

    function paintGrid() {
      if (!gridRes || gridRes.gen !== gen) { gridCells.forEach((c) => { c.textContent = '…'; c.style.background = ''; }); return; }
      const vals = gridRes.vals, nh = gridRes.H;
      let mx = 0.05;
      for (const v of vals) if (Number.isFinite(v)) mx = Math.max(mx, Math.abs(v));
      const li = curL(), hi = curH();
      vals.forEach((v, k) => {
        const c = gridCells[k];
        if (!c) return;
        const a = Math.floor(k / nh), b = k % nh, ok = Number.isFinite(v);
        c.textContent = ok ? sgn(v) : '…';
        c.style.background = ok ? tint(v / mx) : '';
        c.classList.toggle('sel', a === li && b === hi);
        c.title = ok ? `Zero head ${b} in layer ${a}: ${sgn(v)} nats` : `Zero head ${b} in layer ${a}`;
      });
    }

    function topHead() {
      if (!gridRes || gridRes.gen !== gen || !gridRes.vals.every((v) => Number.isFinite(v))) return null;
      let best = 0;
      gridRes.vals.forEach((v, k) => { if (v > gridRes.vals[best]) best = k; });
      return { layer: Math.floor(best / gridRes.H), head: best % gridRes.H, v: gridRes.vals[best] };
    }

    // ------------------------------------------------------------- controls
    function refreshExpSel() {
      const t = tgtFor(curL(), curH(), nL()), cfg = S().config;
      EXPS.forEach((exp, i) => {
        const why = exp.na(cfg, t), opt = expSel.options[i];
        opt.textContent = `${exp.title}${exp.target ? ` · ${targetText(exp, t)}` : ''}${why ? ' (not applicable)' : ''}`;
        opt.disabled = !!why;
        opt.title = why ? `${exp.title}: ${why}` : exp.title;
      });
      expSel.value = String(expIdx);
    }

    function pickExp(i) {
      expIdx = i;
      if (step === 0 || step === 4) goStep(1); else paint();
    }

    function pickHead(a, b) {
      expIdx = HEAD_EXP;
      goStep(1);                                   // leave the ranking first, so no ranking run starts for this selection
      ctx.store.select({ layer: a, head: b });
    }

    function goStep(i) { ctx.setStep(i); }
    function setStep(i) { if (!autoTick) stopAuto(); step = i; paint(); }

    function stopAuto() { clearTimeout(auto); auto = null; }

    // ------------------------------------------------------------- side column and caption
    const okD = (d) => !!d && !d.na;
    const copyLeaks = (d, bs) => !!d.exp.leaks?.(S().config, bs.T);

    function renderSide(bs, d) {
      if (!bs) { body.replaceChildren(); return; }
      if (step === 4) {
        const top = topHead();
        const note = h('div', { class: 'lb-note' }, top
          ? `Largest single-head effect: layer ${top.layer}, head ${top.head}, ${sgn(top.v)} nats. Click a cell to inspect that edit.`
          : 'Loss change when one head is zeroed on this text. Click a cell to inspect that edit.');
        body.replaceChildren(gridWrap, note, statusEl);
        return;
      }
      if (step === 0) {
        const lossEl = h('span', null, '');
        countTo(lossEl, bs.lossA, { digits: 2 });
        const induction = String(S().weights) === 'preset:induction';
        body.replaceChildren(
          h('div', { class: 'lb-box' }, h('div', { class: 'lb-tag' }, `${modelWord()} model`),
            eqEl(aligned([String.raw`\mathcal L &= -\frac{1}{${bs.T - 1}}\sum_{t}`, String.raw`&\log p_\theta\big(x_{t+1}\mid x_{\le t}\big)`]))),
          h('div', { class: 'lb-rd-list' }, rd('model loss', lossEl, ' nats')),
          checkBadge(bs.tracedA, bs.lossA, { label: 'equals the traced logits’ loss' }),
          baseLeaks() ? warn(modelLeakText(S().config)) : null,
          induction ? h('div', { class: 'lb-note' }, 'This preset repeats a string. Zero layer 0, head 0 and read the Δ row on the repeated part.') : null);
        return;
      }
      if (!okD(d)) { body.replaceChildren(); return; }
      if (step === 1) {
        const c = d.copy;
        body.replaceChildren(
          h('div', { class: 'lb-box' },
            h('div', { class: 'lb-tag' }, `${modelWord()} model`), eqEl(d.exp.before(S().config, d.t)),
            h('div', { class: 'lb-tag' }, 'copy'), eqEl(d.exp.after(S().config, d.t))),
          h('div', { class: 'lb-rd-list' }, rd('top-1 changed', `${d.flips} of ${bs.T}`), rd('copy loss', `${fix(d.lossB, 2)} nats`)),
          h('div', { class: 'lb-note' }, c.text),
          badge(c.ok, 'only the edit differs', 'the copy differs elsewhere'),
          copyLeaks(d, bs) ? warn(COPY_LEAK) : null);
        return;
      }
      if (step === 2) {
        const dl = d.lossB - bs.lossA;
        const color = dl >= 0 ? 'var(--heat-pos)' : 'var(--heat-neg)';
        body.replaceChildren(
          h('div', { class: 'lb-box' }, h('div', { class: 'lb-tag' }, 'change in loss'),
            eqEl(String.raw`\Delta\mathcal L=\tfrac{1}{${bs.T - 1}}\sum_{t}\big(\ell^B_t-\ell^A_t\big)`)),
          h('div', { class: 'lb-rd-list' },
            rd('model loss', `${fix(bs.lossA, 2)} nats`),
            rd('copy loss', `${fix(d.lossB, 2)} nats`),
            rd('Δ loss', h('b', { style: { color } }, `${sgn(dl)} nats`))),
          checkBadge(d.meanDelta, dl, { label: 'equals the mean of the bars' }),
          copyLeaks(d, bs) ? warn(COPY_LEAK) : null);
        return;
      }
      if (step === 3) {
        body.replaceChildren(
          h('div', { class: 'lb-box' }, h('div', { class: 'lb-tag' }, `layer ${d.shown}, head ${d.head} · model and copy`),
            eqEl(String.raw`\Aw^{(${d.shown},${d.head})}_{ij}=\frac{e^{\hat S_{ij}}}{\sum_k e^{\hat S_{ik}}}`)),
          h('div', { class: 'lb-rd-list' },
            rd('largest change, any row', fix(d.maxD, 2)),
            rd(d.identical ? 'pattern' : 'query → key', d.identical ? 'identical' : `${d.maxR} → ${d.maxC}`)),
          h('div', { class: 'lb-note' }, shiftNote(d)),
          checkBadge(d.devB, 0, { label: 'rows of the copy sum to 1' }));
      }
    }

    function shiftNote(d) {
      if (d.identical) return `No layer from ${d.layer} on changes for head ${d.head}, so the pattern is identical.`;
      if (d.shown !== d.layer) return `Layer ${d.layer} is unchanged: its attention reads input that this edit does not touch. Layer ${d.shown} is the first where head ${d.head} changes, so it is shown.`;
      return `The largest single change is at query position ${d.maxR} and key position ${d.maxC}.`;
    }

    function capFor(bs, d) {
      if (!bs) return 'A next-character loss needs at least two characters: each one is scored on the character that follows it.';
      if (step === 0) {
        const n = bs.T - 1;
        return `Each position is scored on the character that actually follows it. The loss is the average surprise over <b>${n}</b> ${n === 1 ? 'position' : 'positions'}: <b>${fix(bs.lossA, 2)}</b> nats.`;
      }
      if (step === 4) return 'Ranked by the change in loss on this text. Orange: removing the part raises the loss. Blue: removing it lowers the loss.';
      if (!okD(d)) return 'Pick another one above.';
      if (step === 1) return `Only one part changes. The copy predicts a different top-1 character at <b>${d.flips}</b> of ${bs.T} positions, shown in rose.`;
      if (step === 2) {
        const tail = d.topVal > 0 ? `The largest rise is at position <b>${d.topPos}</b>, where the next character is “<b>${labAt(bs, d.topPos + 1)}</b>”: ${sgn(d.topVal)} nats.` : 'No position’s loss rose.';
        return `Bars show how much each position’s loss rose (orange) or fell (blue). ${tail}`;
      }
      const q = Math.min(S().sel.token ?? bs.T - 1, bs.T - 1);
      if (d.identical) return `Query row <b>${q}</b> is unchanged: the copy gives this head exactly the same pattern as the model.`;
      if (d.shown !== d.layer) return `Layer ${d.layer} is unchanged, so layer ${d.shown} is shown. Query row <b>${q}</b> changes by up to ${fix(d.rowMax[q], 2)}.`;
      return `Query row <b>${q}</b> changes by up to ${fix(d.rowMax[q], 2)}. The largest change anywhere in the pattern is ${fix(d.maxD, 2)}, at query “${labAt(bs, d.maxR)}” and key “${labAt(bs, d.maxC)}”.`;
    }

    // ------------------------------------------------------------- one paint for every step
    // A newer caption cancels a fade still running (see capSet); the side and the strip are painted in the same pass.
    function invalidate() {
      gen++;
      base = null; det = null; rankRes = null; gridRes = null;
      cancelQueue();
    }

    function paint() {
      if (paused) return;
      const bs = getBase();
      const live = !!bs;
      const d = live ? getDetail() : null;
      if (live && stripGen !== gen) buildStrip(bs);
      const naHere = !!(d && d.na);
      stepTag.textContent = STEPS[step];
      expSel.style.display = live ? '' : 'none';
      layerPick.el.style.display = live ? '' : 'none';
      headPick.el.style.display = live ? '' : 'none';
      qRow.style.display = live && step === 3 && !naHere ? '' : 'none';
      if (live) { refreshExpSel(); layerPick.render(); headPick.render(); placeControls(); }
      const showStrip = live && step <= 2 && (step === 0 || !naHere);
      const showPattern = live && step === 3 && !!d && !naHere;
      const showRank = live && step === 4;
      const showNote = !live || (step >= 1 && step <= 3 && naHere);
      panelStrip.classList.toggle('on', showStrip);
      panelPattern.classList.toggle('on', showPattern);
      panelRank.classList.toggle('on', showRank);
      noteBox.classList.toggle('on', showNote);
      if (showNote) noteBox.replaceChildren(h('div', null, !live ? NEED_TEXT : `Not applicable to this model: ${d.na}.`));
      if (live && stripGen === gen) paintStrip(bs, d, step);
      if (showPattern) paintPattern(bs, d);
      if (showRank) paintRank(lastStep !== 4);
      paintGrid();
      renderSide(bs, d);
      capSet(capFor(bs, d));
      if (lastStep !== step && !reducedMotion()) body.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 380, easing: MOTION.easeOut });
      lastStep = step;
      if (step === 4 && live) runAll();
    }

    function rebuildAll() {
      if (gridBuilt !== `${nL()}x${nH()}`) buildGrid();
      paint();
    }

    // ------------------------------------------------------------- wiring
    buildGrid();
    lastLH = `${curL()}|${curH()}`;
    ctx.setSteps(STEPS, setStep);
    rebuildAll();
    appear([header.el, expSel, hero, side], { stagger: MOTION.stagger });

    const offTheme = onThemeChange(() => { if (!paused) paint(); });
    const offs = [
      ctx.on('trace', () => {
        invalidate();
        if (paused) return;
        rebuildAll();
      }),
      ctx.on('select', () => {
        const key = `${curL()}|${curH()}`;
        const changed = key !== lastLH;
        lastLH = key;
        if (paused) return;
        if (changed) { cancelQueue(); rankRes = null; }
        paint();
      }),
    ];

    // auto-advance the first time the scene is shown, so it explains itself; any user step (story bar, keys, controls) stops it
    const tick = () => {
      if (step < STEPS.length - 1) {
        autoTick = true;
        try { ctx.setStep(step + 1); } finally { autoTick = false; }
        auto = setTimeout(tick, MOTION.step);
      }
    };
    auto = setTimeout(tick, 900);
    el.addEventListener('pointerdown', stopAuto, { once: true });
    el.addEventListener('atlas:interrupt', stopAuto);

    return {
      destroy() {
        stopAuto();
        queueId++;
        cancelMorph?.();
        hmA?.destroy();
        hmB?.destroy();
        offTheme();
        offs.forEach((f) => f());
        st.destroy();
      },
      pause() { paused = true; stopAuto(); cancelQueue(); cancelMorph?.(); morphKey = ''; },
      resume() { paused = false; rebuildAll(); },
    };
  },
};

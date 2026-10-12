// One attention head, start to finish, on the real forward pass:
//   Q_h, K_h -> S = QKᵀ -> ÷√d_h -> mask -> softmax A -> A V_h -> head output.
// One left-to-right row of real matrices, every cell at least 4 units and never scaled below that
// for any sequence the trace allows. A head wider than 17 columns draws its first 12 and last 4 with
// an ellipsis column, as vec() does, and keeps the true indices in data-cell and in the maths.
// Each stage is a zoom target. Below the row: the step's equation, then four measurements of this
// head's pattern and one line that reads them (faded in from softmax on, where the pattern exists).
// Structure and restraint follow softmax.js; the shared parts come from the kit.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, mat, op, texLabel, appear, checkBadge, rangeOf, useStyle, MOTION, ZONES } from '../kit.js';
import { h, fmt, num, texHTML, sliceOf, clamp, reducedMotion } from '../../ui/index.js';
import { infer, mulberry32 } from '../../engine/index.js';

const STEPS = ['Queries and keys', 'Scores', 'Scale by 1/√d_h', 'Mask', 'Softmax', 'Mix with values'];
const OP = 48;          // operator node size (design units)
const OP_GAP = 60;      // room for an operator between two matrices: 48 plus clearance
const CHAIN_GAP = 144;  // room for the stacked stage pills between S and A
const PAD = 18;         // kit matrix container: 8 px padding and 1 px border on each side
const HEAD_KEEP = 12, TAIL_KEEP = 4;   // the d_h columns drawn: the first 12, an ellipsis, the last 4 (vec's rule)
const MID_Y = 429;      // midline of the pipeline: the tallest matrix (T rows) is centred here
const LABEL_GAP = 8;    // a tag (symbol and shape, 20 units tall) and the gap between it and its matrix
const MAX_SPAN = 256;   // the tallest column of cells: a cell never grows past this, so the band stays in reach
const MIN_CELL = 4, MAX_CELL = 14;   // design units; the row keeps MIN_CELL up to about 77 tokens
const ZX = ZONES.hero.x, ZW = ZONES.hero.w, CX = ZX + ZW / 2;   // the hero's centre line (x = 800)
const PILL_Y = [30, 62, 94];   // the stage pills hang below the flow arrow: their centres relative to MID_Y
const BAND_GAP = 30, EQ_H = 48, GROUP_GAP = 30;   // pipeline to equation; equation box; air above the metrics
const ANCHOR = { tl: '0 0', c: '-50% -50%', tc: '-50% 0', bc: '-50% -100%' };
const probes = new WeakMap();   // model -> Map(config|layer|head -> induction probe result)

useStyle('head', `
.hd-pipe > * { transition: opacity .6s cubic-bezier(.65,0,.35,1); }
.hd-pipe .is-dim { opacity: .22; }
.hd-pipe .k-cell.hl { outline: 2px solid var(--accent); outline-offset: 1px; box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 35%, transparent); z-index: 1; }
.hd-pipe .k-cell.hl-soft { outline: 2px solid color-mix(in srgb, var(--kind, var(--fg-3)) 50%, transparent); outline-offset: 1px; }
.hd-pipe .k-row-band.soft { border-color: color-mix(in srgb, var(--kind, var(--fg-3)) 55%, transparent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--kind, var(--fg-3)) 14%, transparent); }
.hd-small .k-cell { border-radius: 1px; }
.hd-zoom { cursor: zoom-in; }
.hd-zoom .k-mat-body { transition: transform .25s cubic-bezier(.2,.7,.2,1), box-shadow .25s; }
.hd-zoom:hover .k-mat-body { transform: translateY(-3px); box-shadow: 0 0 0 2px color-mix(in srgb, var(--kind, var(--accent)) 70%, transparent), 0 22px 46px rgba(0,0,0,.2); }
.hd-A .k-cell { box-shadow: inset 0 -1px 0 rgba(0,0,0,.14); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .hd-A .k-cell { box-shadow: inset 0 -1px 0 rgba(0,0,0,.14), inset 0 0 0 1px color-mix(in srgb, var(--c-attn) 30%, transparent); } }
:root[data-theme="dark"] .hd-A .k-cell { box-shadow: inset 0 -1px 0 rgba(0,0,0,.14), inset 0 0 0 1px color-mix(in srgb, var(--c-attn) 30%, transparent); }
.hd-tag { display: flex; align-items: center; gap: 8px; height: 20px; line-height: 1; white-space: nowrap; }
.hd-shape { font: 500 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.hd-pill { display: flex; align-items: center; justify-content: center; box-sizing: border-box; width: 136px; height: 28px; padding: 0 6px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--fg-2); font: 600 14px/1 var(--font-mono); white-space: nowrap; cursor: zoom-in; transition: opacity .6s cubic-bezier(.65,0,.35,1), border-color .25s, color .25s, box-shadow .25s; }
.hd-pill sub { font-size: 10px; }
.hd-pill:hover { border-color: var(--line-strong); color: var(--fg); }
.hd-pill.is-on { border-color: var(--c-attn); color: var(--fg); box-shadow: 0 0 0 3px color-mix(in srgb, var(--c-attn) 24%, transparent); }
.hd-eq { display: flex; align-items: center; gap: 18px; height: 48px; font-size: 24px; color: var(--fg); white-space: nowrap; }
.hd-eq .katex-display { margin: 0; }
.hd-metrics { display: flex; flex-direction: column; gap: 8px; width: 1380px; opacity: 0; transition: opacity .6s cubic-bezier(.65,0,.35,1); }
.hd-metrics.on { opacity: 1; }
.hd-chips { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.hd-chip { min-width: 0; padding: 8px 14px 9px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 82%, transparent); border: 1px solid var(--line); box-shadow: 0 8px 20px rgba(0,0,0,.10); }
.hd-chip-top { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; min-width: 0; }
.hd-chip-k { flex: 1 1 auto; min-width: 0; font: 600 12px/1.2 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .06em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hd-chip-v { flex: none; font: 650 18px/1.1 var(--font-mono); color: var(--fg); font-variant-numeric: tabular-nums; white-space: nowrap; }
.hd-bar { height: 3px; border-radius: 2px; background: var(--surface-2); margin-top: 7px; overflow: hidden; }
.hd-bar i { display: block; height: 100%; border-radius: 2px; background: var(--c-attn); transition: width .6s cubic-bezier(.65,0,.35,1); }
.hd-read { font-size: 19px; line-height: 1.4; color: var(--fg-2); white-space: nowrap; }
.hd-read b { color: var(--fg); }
@media (prefers-reduced-motion: reduce) { .hd-pipe > *, .hd-bar i, .hd-metrics { transition: none !important; } }
`);

const clamp01 = (v) => clamp(v, 0, 1);
const pct = (v) => (v == null || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`);
const col = (rows, c) => Array.from({ length: rows }, (_, r) => [r, c]);   // the cells of column c

// The d_h columns drawn: every column up to 17, otherwise the first 12, an ellipsis (-1) and the last 4.
function dimCols(dh) {
  if (dh <= HEAD_KEEP + TAIL_KEEP + 1) return Array.from({ length: dh }, (_, i) => i);
  return [...Array.from({ length: HEAD_KEEP }, (_, i) => i), -1, ...Array.from({ length: TAIL_KEEP }, (_, i) => dh - TAIL_KEEP + i)];
}

// Probe tokens for the induction score: L distinct tokens when the vocabulary has them, so the one
// aligned earlier copy is the only match; otherwise L random tokens, repeats allowed.
function probeSequence(cfg, L) {
  const rnd = mulberry32(20240611), V = cfg.vocab_size;
  if (L <= V) {
    const pool = Array.from({ length: V }, (_, i) => i);
    for (let i = 0; i < L; i++) { const j = i + Math.floor(rnd() * (V - i)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    return { seq: pool.slice(0, L), distinct: true };
  }
  return { seq: Array.from({ length: L }, () => Math.min(V - 1, Math.floor(rnd() * V))), distinct: false };
}

// Induction score on the real weights. A sequence of L tokens is repeated once; in the second copy,
// query i should put its weight on key i − L + 1, the token after the earlier copy of the current
// token. The score is the mean of that weight over the L queries of the second copy (cached per model).
function inductionScore(model, layer, head) {
  if (!model?.config) return null;
  const cfg = model.config, L = Math.floor(cfg.block_size / 2);
  if (L < 2) return null;
  const key = `${JSON.stringify(cfg)}|${layer}|${head}`;
  let memo = probes.get(model);
  if (!memo) { memo = new Map(); probes.set(model, memo); }
  if (!memo.has(key)) {
    try {
      const { seq, distinct } = probeSequence(cfg, L);
      const ids = seq.concat(seq), N = 2 * L;
      const P = infer(model, ids).trace[`L${layer}.attn.probs`].data;
      let sum = 0;
      for (let i = L; i < N; i++) sum += P[(head * N + i) * N + (i - L + 1)];
      memo.set(key, { score: sum / L, distinct, L });
    } catch (e) {
      memo.set(key, null);
    }
  }
  return memo.get(key);
}

// Mean over queries i = 1 … T−1 (query 0 has no earlier key): previous-token weight, position-0 weight,
// and the row entropy in nats next to its uniform value over the keys each query may see.
function metricsOf(P, M, T) {
  if (T < 3) return null;
  let prev = 0, pos0 = 0, ent = 0, uni = 0;
  for (let i = 1; i < T; i++) {
    const r = i * T;
    prev += P[r + i - 1];
    pos0 += P[r];
    let e = 0, n = 0;
    for (let j = 0; j < T; j++) {
      const p = P[r + j];
      if (p > 0) e -= p * Math.log(p);
      if (M[r + j] !== -Infinity) n++;
    }
    ent += e;
    uni += Math.log(Math.max(1, n));
  }
  const k = T - 1;
  return { prev: prev / k, pos0: pos0 / k, ent: ent / k, uni: uni / k };
}

// One interpretation, chosen by the numbers. An induction score of at least 0.4 names the induction
// head first. Otherwise the larger of the previous-token and position-0 weights names the head when it
// reaches 0.5; the entropy describes the pattern only when neither candidate dominates.
export function reading(m) {
  if (m.ind != null && m.ind >= 0.4) return 'Copies from earlier repeats: an induction head.';
  const best = m.pos0 > m.prev ? 'pos0' : 'prev';
  const v = m[best];
  if (best === 'prev' && v >= 0.5) return 'Mostly attends to the previous token: a previous-token head.';
  if (best === 'pos0' && v >= 0.5) return 'Most of the weight sits on position 0: an attention-sink head.';
  if (m.ent <= 0.5 * m.uni) return 'Sharp: each query reads from only a few of the keys it may see.';
  if (m.ent >= 0.85 * m.uni) return 'Diffuse: each query averages over most of the keys it may see.';
  return 'No single role dominates: a mixed pattern over the visible keys.';
}

// One measurement: the name and value on one row, with a bar for the ratio. Definitions live in the tooltip.
function metricChip(name, value, frac, title) {
  return h('div', { class: 'hd-chip', title },
    h('div', { class: 'hd-chip-top' }, h('div', { class: 'hd-chip-k' }, name), h('div', { class: 'hd-chip-v' }, value)),
    h('div', { class: 'hd-bar' }, h('i', { style: { width: `${(Number.isFinite(frac) ? clamp01(frac) : 0) * 100}%` } })));
}

function chipsFor(met, ind) {
  const none = '—', kind = ind?.distinct ? 'distinct' : 'random';
  return [
    metricChip('Previous token', met ? pct(met.prev) : none, met?.prev,
      'Mean weight a query puts on the token right before it: A[i, i−1], averaged over queries i ≥ 1.'),
    metricChip('Position 0', met ? pct(met.pos0) : none, met?.pos0,
      'Mean weight on the first token: A[i, 0], averaged over queries i ≥ 1. A large value is the attention-sink pattern.'),
    metricChip(ind ? `Induction · L ${ind.L} ${kind}` : 'Induction', ind ? pct(ind.score) : none, ind?.score,
      ind ? `Weight on the token after the earlier copy of the current token, averaged over the second copy of ${ind.L} ${kind} tokens repeated once.` : 'The probe needs a block size of at least 4.'),
    metricChip(met ? `Entropy · uniform ${met.uni.toFixed(2)}` : 'Entropy', met ? `${met.ent.toFixed(2)} nats` : none, met ? met.ent / Math.max(met.uni, 1e-9) : undefined,
      'Mean over queries of −Σ A ln A, in nats. The bar is the ratio to the uniform value over the visible keys, the entropy of an even spread.'),
  ];
}

// A matrix fills in top to bottom when its stage is first reached.
function wave(node) {
  if (reducedMotion()) return;
  node.querySelector('.k-grid')?.animate([{ clipPath: 'inset(0 0 100% 0)' }, { clipPath: 'inset(0 0 0 0)' }], { duration: MOTION.move, easing: MOTION.ease });
}

// Highlights: one accent focus per step; operands from earlier steps use the component colour.
function setRow(node, r, focus = true) {
  const band = node.querySelector('.k-row-band');
  if (r == null) { if (band) band.hidden = true; return; }
  node.highlightRow(r);
  band?.classList.toggle('soft', !focus);
}
function clearCells(node) { node.querySelectorAll('.k-cell.hl, .k-cell.hl-soft').forEach((c) => c.classList.remove('hl', 'hl-soft')); }
function mark(node, cells, cls) { cells.forEach(([r, c]) => node.cellEl(r, c)?.classList.add(cls)); }

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const { l, h: hd } = ctx.params;
    let step = 0, built = null, drawn = false;
    let x0 = 0, k = 1;   // the pipeline's placement on the canvas: local x maps to x0 + k·x

    // The menu's zoom targets for this head's keys follow the selected head: select this scene's head.
    ctx.store.select({ head: hd });

    // ---------------------------------------------------------------- shared parts and fixed frames
    const header = sceneHeader(st, { chips: [[`Layer ${l}`, 'resid'], [`Head ${hd}`, 'attn']], title: 'One attention head' });
    const picker = tokenPicker(st, ctx, { label: 'query', kind: 'attn' });
    const pipe = h('div', { class: 'hd-pipe' });
    const eqBox = h('div', { class: 'hd-eq' });
    const chipsEl = h('div', { class: 'hd-chips' });
    const readEl = h('div', { class: 'hd-read' });
    const metrics = h('div', { class: 'hd-metrics' }, chipsEl, readEl);
    st.add(pipe, { x: 0, y: 0 });
    st.add(eqBox, { x: ZX, y: 591 });
    st.add(metrics, { x: ZX, y: 679 });
    const cap = sceneCaption(st, {});
    const flow = st.arrow({ x: 0, y: MID_Y }, { x: 1, y: MID_Y }, { kind: 'attn', width: 2.5 });   // S to A, placed in build()

    // Place a node inside the pipeline (design units, local to the pipeline).
    function put(node, x, y, anchor = 'tl') {
      node.style.position = 'absolute';
      node.style.left = `${x}px`;
      node.style.top = `${y}px`;
      node.style.translate = ANCHOR[anchor];
      pipe.append(node);
      return node;
    }

    function teardown() {
      if (!built) return;
      Object.values(built.mats).forEach((m) => m.destroy());
      pipe.replaceChildren();
      built = null;
    }

    // Numbers for the query in view: its strongest key, the dot product, the masked row softmax.
    function derive(b) {
      const { T, dh, Q, K, V, S, Sc, M, A, O } = b;
      const q = Math.min(ctx.state.sel.token ?? T - 1, T - 1), r0 = q * T;
      let j = 0;
      for (let t = 1; t < T; t++) if (A[r0 + t] > A[r0 + j]) j = t;
      let dot = 0;
      for (let c = 0; c < dh; c++) dot += Q[q * dh + c] * K[j * dh + c];
      let mx = -Infinity;
      for (let t = 0; t < T; t++) if (Number.isFinite(M[r0 + t]) && M[r0 + t] > mx) mx = M[r0 + t];
      let Z = 0, ex = 0, rowSum = 0, nMask = 0, jm = -1;
      for (let t = 0; t < T; t++) {
        const v = M[r0 + t], e = Number.isFinite(v) ? Math.exp(v - mx) : 0;
        Z += e;
        rowSum += A[r0 + t];
        if (t === j) ex = e;
        if (v === -Infinity) { nMask++; if (jm < 0) jm = t; }
      }
      let oc = 0;
      for (let t = 0; t < T; t++) oc += A[r0 + t] * V[t * dh];
      return {
        q, j, dot, Sj: S[r0 + j], Scj: Sc[r0 + j], Mj: M[r0 + j], Aj: A[r0 + j],
        jm, Mjm: jm >= 0 ? M[r0 + jm] : null, Z, ex, p: Z > 0 ? ex / Z : 0, rowSum, nMask,
        oc, ocTr: O[q * dh],
      };
    }

    // One equation per step, with live numbers and a check against the traced value.
    function equation(b, d) {
      const { T, dh, cfg, slope } = b, { q, j } = d, sq = Math.sqrt(dh), alibi = cfg.pos_encoding === 'alibi';
      switch (step) {
        case 0:
          return { tex: `\\Q_{${hd}},\\;\\K_{${hd}} \\in \\R^{${T}\\times ${dh}}, \\qquad d_h = ${dh}`, check: null };
        case 1:
          return { tex: `S_{${q},${j}} = (\\Q_{${hd}}\\K_{${hd}}^{\\top})_{${q},${j}} = \\Q_{${hd}}[${q}] \\cdot \\K_{${hd}}[${j}] = ${num(d.dot, 2)}`, check: checkBadge(d.dot, d.Sj) };
        case 2: {
          const base = cfg.attn_scale ? d.Sj / sq : d.Sj;
          const calc = alibi ? base - slope * Math.abs(q - j) : base;
          const baseTeX = cfg.attn_scale ? `\\dfrac{${num(d.Sj, 2)}}{\\sqrt{${dh}}}` : num(d.Sj, 2);
          const tail = alibi ? ` - ${num(slope, 3)}\\,|${q}-${j}|` : '';
          return { tex: `\\hat S_{${q},${j}} = ${baseTeX}${tail} = ${num(d.Scj, 2)}`, check: checkBadge(calc, d.Scj) };
        }
        case 3:
          if (d.jm >= 0) return { tex: `\\text{masked}_{${q},${d.jm}} = -\\infty`, check: checkBadge(-Infinity, d.Mjm) };
          return { tex: `\\text{masked}_{${q},${j}} = \\hat S_{${q},${j}} = ${num(d.Mj, 2)}`, check: checkBadge(d.Scj, d.Mj) };
        case 4:
          return {
            tex: `A_{${q},${j}} = \\dfrac{e^{\\hat S_{${q},${j}} - m}}{\\sum_k e^{\\hat S_{${q},k} - m}} = \\dfrac{${num(d.ex, 3)}}{${num(d.Z, 3)}} = ${num(d.p, 3)}`,
            check: checkBadge(d.p, d.Aj),
          };
        default:
          return { tex: `\\text{ctx}_{${q},0} = \\sum_j A_{${q},j}\\,V_{j,0} = ${num(d.oc, 3)}`, check: checkBadge(d.oc, d.ocTr) };
      }
    }

    // One or two sentences per step, naming the real query token and its key.
    function caption(b, d) {
      const { dh, cfg } = b, qL = `“<b>${ctx.label(d.q)}</b>”`, jL = `“<b>${ctx.label(d.j)}</b>”`;
      switch (step) {
        case 0:
          return `Q and K are T × d<sub>h</sub> matrices. Row ${qL} of Q (position ${d.q}) and row ${jL} of K, its strongest key, are highlighted.`
            + (b.elided ? ` d<sub>h</sub> = ${dh} is elided: the first 12 and last 4 columns are drawn; the maths uses all ${dh}.` : '');
        case 1:
          return 'S = QKᵀ has one dot product for every query and key. The accent cell is the dot product of the two highlighted rows.';
        case 2: {
          const first = cfg.attn_scale
            ? `Dividing by √d<sub>h</sub> = √${dh} = ${Math.sqrt(dh).toFixed(2)} keeps the dot products from growing with the head width, so softmax does not saturate.`
            : 'Scaling is off in this config (an ablation), so the scores pass through unchanged.';
          const second = cfg.pos_encoding === 'alibi' ? ' ALiBi then adds the distance penalty −m<sub>h</sub>|q − j| to each allowed score.' : '';
          return first + second;
        }
        case 3: {
          const lead = {
            causal: 'The mask hides every key after the query’s own position',
            none: 'This config has no mask, so every query sees every key',
            sliding: `This config is sliding-window: a query sees itself and the ${cfg.window - 1} keys before it`,
            prefix: `This config is prefix-LM: every query sees the first ${cfg.prefix_len} tokens, and the rest is causal`,
          }[cfg.mask] || 'The mask hides the keys a query may not see';
          if (cfg.mask === 'none') return `${lead}. The mask pill opens its own scene.`;
          return `${lead}. ${d.nMask ? `For ${qL}, ${d.nMask} of ${b.T} keys are hatched` : `For ${qL}, no key is hatched`}; the mask pill opens its own scene.`;
        }
        case 4:
          return `Softmax turns each row into weights: subtract the row maximum m for stability, exponentiate, then divide by the row sum. The row for ${qL} now adds up to ${fmt(d.rowSum, 2)}.`;
        default:
          return 'Each output row is a weighted average of the value rows: ctx<sub>q</sub> = Σ<sub>j</sub> A<sub>qj</sub> V<sub>j</sub>. Concatenating the heads and applying W<sub>O</sub> comes next.';
      }
    }

    // Stage visibility, the morphing scores matrix, the highlights, the equation and the caption for this step.
    function paint() {
      if (!built) return;
      const b = built, m = b.mats, d = derive(b);

      // Stages: finished ones stay, later ones are dimmed; a matrix fills in when its stage is reached.
      for (const it of b.staged) {
        it.node.classList.toggle('is-dim', it.k > step);
        if (it.pill) it.node.classList.toggle('is-on', it.k === step);
        if (it.wave && it.k === step && !b.seen.has(it.node)) { b.seen.add(it.node); wave(it.node); }
      }
      // The measurements describe the finished pattern: they appear from softmax on.
      metrics.classList.toggle('on', step >= 4);

      // The scores matrix morphs in place: S, then the scaled (or biased) scores, then the masked ones.
      const src = step <= 1 ? 'S' : step === 2 ? 'Sc' : 'M';
      if (b.shown !== src) {
        b.shown = src;
        m.S.update(src === 'S' ? b.S : src === 'Sc' ? b.Sc : b.M, { range: b.R });
        m.S.dataset.trace = b.keys[src];
      }

      // One accent focus per step. Earlier operands (the query row, the key row) use the component colour.
      for (const node of Object.values(m)) { clearCells(node); setRow(node, null); }
      if (step === 0) { setRow(m.Q, d.q, true); setRow(m.K, d.j, false); }
      if (step === 1) { mark(m.S, [[d.q, d.j]], 'hl'); setRow(m.Q, d.q, false); setRow(m.K, d.j, false); }
      if (step === 2 || step === 3) setRow(m.S, d.q, true);
      if (step === 4) setRow(m.A, d.q, true);
      if (step === 5) { mark(m.O, [[d.q, 0]], 'hl'); setRow(m.O, d.q, false); setRow(m.A, d.q, false); mark(m.V, col(b.T, 0), 'hl-soft'); }

      const eq = equation(b, d);
      eqBox.replaceChildren(h('div', { html: texHTML(eq.tex) }), ...(eq.check ? [eq.check] : []));
      cap.set(caption(b, d));
    }

    function build() {
      teardown();
      const state = ctx.state, cfg = ctx.cfg, model = state.model, tr = state.trace;
      const T = state.ids.length, dh = cfg.d_model / cfg.n_head;
      const P = `L${l}.attn.`;
      const rope = cfg.pos_encoding === 'rope', alibi = cfg.pos_encoding === 'alibi', gqa = cfg.n_kv_head < cfg.n_head;
      // The trace keys that hold exactly what each stage shows for this config.
      const keys = {
        Q: P + (rope ? 'q_rot' : 'q_heads'),
        K: P + (gqa ? 'k_rep' : rope ? 'k_rot' : 'k_heads'),
        V: P + (gqa ? 'v_rep' : 'v_heads'),
        S: P + 'scores',
        Sc: alibi ? P + 'biased' : cfg.attn_scale ? P + 'scaled' : P + 'scores',
        M: P + 'masked',
        A: P + 'probs',
        O: P + 'ctx',
      };
      const hs = (key) => sliceOf(tr[key], hd).data;   // this head, flat row-major
      const Q = hs(keys.Q), K = hs(keys.K), V = hs(keys.V), S = hs(keys.S), Sc = hs(keys.Sc), M = hs(keys.M), A = hs(keys.A), O = hs(keys.O);
      const R = rangeOf(S);   // one colour scale for the scores through every step

      // The d_h columns on screen, and a copy of each token-by-d_h matrix that keeps only those columns.
      const dc = dimCols(dh), nd = dc.length, elided = nd < dh;
      const narrow = (src) => {
        const out = new Float64Array(T * nd);
        for (let r = 0; r < T; r++) for (let cc = 0; cc < nd; cc++) out[r * nd + cc] = dc[cc] < 0 ? 0 : src[r * dh + dc[cc]];
        return out;
      };

      // Geometry. The row holds four d_h-wide matrices (Q, K, V, O), two T-wide ones (S, A), four
      // operators and the chain between S and A. The cell is the largest size from 4 to 14 units whose
      // row fits the hero. Up to about 77 tokens the row fits at 4 units, so it is never scaled.
      const gapFor = (cc) => (cc >= 9 && T <= 24 ? 1 : 0);
      const box = (rows, cols, cc) => { const g = gapFor(cc); return { w: cols * cc + (cols - 1) * g + PAD, h: rows * cc + (rows - 1) * g + PAD }; };
      const rowWidth = (cc) => 4 * box(T, nd, cc).w + 2 * box(T, T, cc).w + 4 * OP_GAP + CHAIN_GAP;
      let c = MIN_CELL;
      for (let cc = MAX_CELL; cc >= MIN_CELL; cc--) if (rowWidth(cc) <= ZW && T * cc <= MAX_SPAN) { c = cc; break; }
      const gap = gapFor(c), W = rowWidth(c);
      k = Math.min(1, ZW / W);
      x0 = CX - (W * k) / 2;
      const toStage = (x, y) => ({ x: x0 + k * x, y: MID_Y + k * (y - MID_Y) });
      pipe.style.transformOrigin = `0px ${MID_Y}px`;
      pipe.style.transform = `translate(${x0}px, 0px) scale(${k})`;
      pipe.classList.toggle('hd-small', c < 8);

      const bQ = box(T, nd, c), bS = box(T, T, c);
      const xK = bQ.w + OP_GAP, xS = xK + bQ.w + OP_GAP, xA = xS + bS.w + CHAIN_GAP;
      const xV = xA + bS.w + OP_GAP, xO = xV + bQ.w + OP_GAP;
      const topOf = (bx) => MID_Y - bx.h / 2;

      // Matrices (kit cells). Each cell maps to the trace's own indices: (query, key, head) or (token, d_h, head).
      const mk = (data, rows, cols, o) => mat(data, rows, cols, { cell: c, gap, ...o });
      const cellT = (r, cc) => `${r},${cc},${hd}`;
      const cellD = (r, cc) => `${r},${dc[cc]},${hd}`;
      const narrowMat = (src, o) => {
        const m = mk(narrow(src), T, nd, { ...o, cellOf: cellD });
        for (let r = 0; r < T; r++) for (let cc = 0; cc < nd; cc++) {
          if (dc[cc] >= 0) continue;
          const cell = m.cellEl(r, cc);
          cell.classList.add('ell');
          cell.textContent = '⋮';
          cell.style.fontSize = `${Math.min(14, Math.max(4, c * 1.2))}px`;
          delete cell.dataset.cell;
        }
        return m;
      };
      const mQ = narrowMat(Q, { kind: 'q', trace: keys.Q });
      const mK = narrowMat(K, { kind: 'k', trace: keys.K });
      const mS = mk(S, T, T, { kind: 'attn', range: R, trace: keys.S, cellOf: cellT });
      const mA = mk(A, T, T, { kind: 'attn', mode: 'sequential', range: 1, trace: keys.A, cellOf: cellT });
      const mV = narrowMat(V, { kind: 'v', trace: keys.V });
      const mO = narrowMat(O, { kind: 'v', trace: keys.O });
      // Clicking a row of Q or K selects that token. The scores, the softmax and the mix are zoom targets.
      [mQ, mK].forEach((m) => m.addEventListener('click', (e) => { const cell = e.target.closest('[data-cell]'); if (cell) ctx.store.select({ token: Number(cell.dataset.cell.split(',')[0]) }); }));
      mS.classList.add('hd-zoom'); mS.dataset.node = `scores.${l}.${hd}`;
      mA.classList.add('hd-zoom', 'hd-A'); mA.dataset.node = `softmax.${l}.${hd}`;
      mV.classList.add('hd-zoom'); mV.dataset.node = `mix.${l}.${hd}`;
      mO.classList.add('hd-zoom'); mO.dataset.node = `mix.${l}.${hd}`;
      put(mQ, 0, topOf(bQ));
      put(mK, xK, topOf(bQ));
      put(mS, xS, topOf(bS));
      put(mA, xA, topOf(bS));
      put(mV, xV, topOf(bQ));
      put(mO, xO, topOf(bQ));

      // Each tag sits above its matrix: the symbol and the true shape on one line.
      // A short matrix (a few tokens) keeps its tag clear of the operator circles, which sit at MID_Y ± OP/2.
      const tag = (tex, shape, kind, bx, x) => put(h('div', { class: 'hd-tag' }, texLabel(tex, { size: 16, kind }), h('span', { class: 'hd-shape' }, shape)), x + bx.w / 2, Math.min(topOf(bx), MID_Y - OP / 2 - 4) - LABEL_GAP, 'bc');
      const tQ = tag(`\\Q_{${hd}}`, `${T} × ${dh}`, 'q', bQ, 0);
      const tK = tag(`\\K_{${hd}}`, `${T} × ${dh}`, 'k', bQ, xK);
      const tS = tag(`S_{${hd}}`, `${T} × ${T}`, 'attn', bS, xS);
      const tA = tag(`\\Aw_{${hd}}`, `${T} × ${T}`, 'attn', bS, xA);
      const tV = tag(`\\V_{${hd}}`, `${T} × ${dh}`, 'v', bQ, xV);
      const tO = tag(`\\text{ctx}_{${hd}}`, `${T} × ${dh}`, 'v', bQ, xO);

      // Operators between the matrices: Q Kᵀ = S, then A V = ctx.
      const opQK = put(op('\\times', { size: OP }), xK - OP_GAP / 2, MID_Y, 'c');
      const opEqS = put(op('=', { size: OP }), xS - OP_GAP / 2, MID_Y, 'c');
      const opAV = put(op('\\times', { size: OP }), xV - OP_GAP / 2, MID_Y, 'c');
      const opEqO = put(op('=', { size: OP }), xO - OP_GAP / 2, MID_Y, 'c');

      // The stage pills in the chain between S and A, hung below the flow arrow (so they never meet the tags
      // above a short matrix). Each is a zoom target.
      const pill = (html, node) => h('button', { type: 'button', class: 'hd-pill', dataset: { node }, html });
      const chainX = xS + bS.w + CHAIN_GAP / 2;
      const scaleHtml = cfg.attn_scale ? `÷ √d<sub>h</sub>${alibi ? ' + ALiBi' : ''}` : (alibi ? '+ ALiBi' : 'no scaling');
      const pScale = put(pill(scaleHtml, `scores.${l}.${hd}`), chainX, MID_Y + PILL_Y[0], 'c');
      const pMask = put(pill('mask', `mask.${l}.${hd}`), chainX, MID_Y + PILL_Y[1], 'c');
      const pSoft = put(pill('softmax', `softmax.${l}.${hd}`), chainX, MID_Y + PILL_Y[2], 'c');
      const a0 = toStage(xS + bS.w + 4, MID_Y), a1 = toStage(xA - 4, MID_Y);
      flow.path.setAttribute('d', `M${a0.x},${a0.y} L${a1.x},${a1.y}`);
      if (!drawn) { drawn = true; flow.draw(MOTION.move); }

      // Which step lights which stage. Tags dim with their matrix.
      const staged = [
        { node: mQ, k: 0, wave: true }, { node: tQ, k: 0 },
        { node: mK, k: 0, wave: true }, { node: tK, k: 0 }, { node: opQK, k: 0 },
        { node: opEqS, k: 1 }, { node: mS, k: 1, wave: true }, { node: tS, k: 1 },
        { node: pScale, k: 2, pill: true }, { node: pMask, k: 3, pill: true }, { node: pSoft, k: 4, pill: true },
        { node: mA, k: 4, wave: true }, { node: tA, k: 4 },
        { node: opAV, k: 5 }, { node: mV, k: 5, wave: true }, { node: tV, k: 5 },
        { node: opEqO, k: 5 }, { node: mO, k: 5, wave: true }, { node: tO, k: 5 },
      ];

      // Measurements on the real pattern (queries 1 … T−1), and the one line that reads them.
      const ind = inductionScore(model, l, hd);
      const met = metricsOf(A, M, T);
      chipsEl.replaceChildren(...chipsFor(met, ind));
      const text = met ? reading({ ...met, ind: ind?.score ?? null }) : 'Too few queries to read a pattern: a head needs at least three tokens.';
      const [lead, rest] = text.split(': ');
      readEl.innerHTML = rest ? `<b>${lead}</b>: ${rest}` : `<b>${text}</b>`;
      header.setSub(`Layer ${l}, head ${hd} on ${T} token${T === 1 ? '' : 's'}: each query scores the keys it may see, turns the scores into weights and mixes the values.`);

      // The lower band follows the pipeline and clears the stage pills that hang below the arrow: the
      // equation, then the measurements, with air between the groups. The caption keeps its zone unless
      // the band reaches it (only beyond 64 tokens).
      const pipeBottom = MID_Y + (bS.h * k) / 2, pillBottom = MID_Y + PILL_Y[2] + 14;
      const bandY = Math.max(pipeBottom + BAND_GAP, pillBottom + 10);
      st.move(eqBox, ZX, bandY);
      st.move(metrics, ZX, bandY + EQ_H + GROUP_GAP);
      st.move(cap.el, ZONES.caption.x, Math.max(ZONES.caption.y, bandY + EQ_H + GROUP_GAP + metrics.offsetHeight + 28));

      built = {
        T, dh, cfg, R, keys, staged, shown: 'S', seen: new Set(), elided,
        slope: 2 ** (-8 * (hd + 1) / cfg.n_head),
        Q, K, V, S, Sc, M, A, O,
        mats: { Q: mQ, K: mK, S: mS, A: mA, V: mV, O: mO },
      };
      paint();
    }

    function setStep(i) { step = i; paint(); }
    ctx.setSteps(STEPS, setStep);
    build();
    appear([header.el, picker.el, eqBox], { stagger: MOTION.stagger });

    // Auto-advance the first time the scene is shown, so it explains itself.
    let auto = setTimeout(function next() {
      if (step < STEPS.length - 1) { ctx.setStep(step + 1); auto = setTimeout(next, MOTION.step); }
    }, 900);
    el.addEventListener('pointerdown', () => clearTimeout(auto), { once: true });

    const offs = [ctx.on('trace', build), ctx.on('select', paint)];
    return {
      destroy() {
        clearTimeout(auto);
        offs.forEach((f) => f());
        teardown();
        st.destroy();
      },
    };
  },
};

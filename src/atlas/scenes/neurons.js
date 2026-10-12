// Neurons as memories: what each hidden unit of one MLP writes into the residual stream, for one token.
//   1. the most active units, ranked by |a|           activations, from the trace (L{l}.mlp.act)
//   2. what each writes: a_n · W_down[:, n]           weights, from state.model.params
//   3. the sum: all units + bias = mlp.out[t]         recomputed from 1 and 2, checked against the trace
//   4. what each direction promotes: W_U · W_down[:, n]   logit lens, an approximation
// Every number is traced or computed from the weights. Reference scene for contributors:
// see docs/ATLAS_GUIDE.md and docs/STYLE_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, op, appear, pulse, useStyle, ZONES, MOTION } from '../kit.js';
import { h, fmt, num, texHTML, diverging, onThemeChange, reducedMotion, clamp } from '../../ui/index.js';
import { geluScalar, siluScalar } from '../../engine/index.js';

const STEPS = ['Most active units', 'What each writes', 'The sum', 'What they promote'];
const K = 8;                                        // units shown: the most active by |a|
const LENS = 3;                                     // tokens listed per unit by the logit lens
const CELL = 12, GAP = 2;                           // cells of a strip (the kit's vec look)
const SPEC = { x: 110, y: 300, w: 1016, bh: 330 };  // spectrum box; its bars rise from y = 390 (box coordinates)
const PANEL = { x: 96, y: 274, w: 1044, h: 476 };   // the hero panel behind the visual
const SIDE_Y = 262, HEAD_Y = 302;
const ROW0 = 330, ROWP = 54;                        // rows: first centre y and pitch (last op ends above the panel edge)
const STACK0 = 330, STACKP = 30;                    // the stacked sum: first centre y and pitch
const X = { tag: 110, bar: 210, barW: 170, val: 392, times: 500, w: 540, eq: 826, s: 872, label: 440, plus: 512, sum: 540, chip: 856, chipP: 92 };

useStyle('neurons', `
.ns-panel { position: absolute; border-radius: 16px; background: color-mix(in srgb, var(--surface) 62%, transparent); border: 1px solid color-mix(in srgb, var(--c-mlp) 28%, var(--line)); box-shadow: 0 14px 34px rgba(0, 0, 0, .12), 0 2px 0 color-mix(in srgb, var(--c-mlp) 16%, transparent); pointer-events: none; }
.ns-mv { transition: left .6s cubic-bezier(.65, 0, .35, 1), top .6s cubic-bezier(.65, 0, .35, 1), opacity .42s ease; }
.ns-still, .ns-still * { transition: none !important; }
.ns-spec { position: absolute; width: 1016px; height: 420px; }
.ns-band { position: absolute; left: -5px; top: 40px; height: 350px; border-radius: 12px; background: color-mix(in srgb, var(--c-mlp) 9%, transparent); border: 1px solid color-mix(in srgb, var(--c-mlp) 22%, transparent); }
.ns-bars { position: absolute; left: 0; top: 60px; width: 1016px; height: 330px; }
.ns-bar { position: absolute; bottom: 0; border-radius: 3px 3px 1px 1px; transition: left .6s cubic-bezier(.65, 0, .35, 1), height .6s cubic-bezier(.65, 0, .35, 1), opacity .42s ease, background-color .42s ease; }
.ns-bar.rest { opacity: .34; }
.ns-base { position: absolute; left: 0; width: 1016px; top: 390px; border-top: 1px solid var(--line-strong); }
.ns-bracket { position: absolute; left: 0; top: 24px; height: 10px; border: 1.5px solid var(--c-mlp); border-bottom: 0; border-radius: 6px 6px 0 0; transition: width .6s cubic-bezier(.65, 0, .35, 1); }
.ns-top-lab, .ns-max-lab { position: absolute; top: 18px; font: 600 16px/1 var(--font-mono); color: var(--c-mlp); white-space: nowrap; }
.ns-max-lab { right: 0; color: var(--fg-3); font-weight: 500; }
.ns-axis { position: absolute; top: 400px; font: 500 14px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.ns-tag { position: absolute; font: 600 16px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.ns-track { position: absolute; width: 170px; height: 14px; }
.ns-zero { position: absolute; left: 85px; top: -6px; bottom: -6px; width: 1px; background: var(--line-strong); }
.ns-fill { position: absolute; top: 2px; height: 10px; border-radius: 3px; transition: left .6s cubic-bezier(.65, 0, .35, 1), width .6s cubic-bezier(.65, 0, .35, 1), background-color .42s ease; }
.ns-val { position: absolute; font: 500 15px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.ns-head { position: absolute; font: 600 14px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.ns-plus { position: absolute; font: 600 20px/1 var(--font-mono); color: var(--fg-3); }
.ns-rule { position: absolute; height: 0; border-top: 1.5px solid var(--line-strong); }
.ns-chip { position: absolute; box-sizing: border-box; width: 84px; height: 40px; display: flex; align-items: center; justify-content: space-between; padding: 0 10px; border-radius: 10px; background: var(--surface); border: 1px solid color-mix(in srgb, var(--c-head) 42%, var(--line)); box-shadow: 0 4px 10px rgba(0, 0, 0, .08); }
.ns-chip.neg { border-style: dashed; }
.ns-chip b { font: 600 19px/1 var(--font-mono); color: var(--fg); }
.ns-chip small { font: 500 13px/1 var(--font-mono); color: var(--fg-2); }
.ns-side { position: absolute; width: 340px; display: grid; gap: 12px; align-content: start; }
.ns-kick { font: 600 17px/1.2 var(--font-mono); color: var(--c-mlp); }
.ns-main { font-size: 22px; color: var(--fg); overflow-x: auto; }
.ns-main .katex-display { margin: 0; }
.ns-live { display: grid; gap: 6px; font-size: 16px; line-height: 1.35; padding: 10px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); box-shadow: 0 6px 16px rgba(0, 0, 0, .06); overflow-x: auto; }
.ns-lens { font: 600 16px/1.2 var(--font-mono); color: var(--fg); display: flex; gap: 10px; align-items: baseline; }
.ns-lens span { color: var(--fg-2); font-weight: 500; }
.ns-checks { display: flex; flex-wrap: wrap; gap: 8px; }
.ns-checks:empty, .ns-note:empty { display: none; }
.ns-note { font: 500 13px/1.45 var(--font-mono); color: var(--fg-3); }
`);

const esc = (x) => String(x).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);
const stripCells = (n) => (n <= 17 ? n : 17);
const stripWidth = (n) => stripCells(n) * CELL + (stripCells(n) - 1) * GAP + 12;

/** A vector as a strip of cells, with the kit's look: first 12 … last 4 when long. */
function strip(n, kind) {
  const idx = n <= 17 ? [...Array(n).keys()] : [...Array(12).keys(), -1, n - 4, n - 3, n - 2, n - 1];
  const cells = idx.map((i) => h('div', { class: i < 0 ? 'k-cell ell' : 'k-cell' }, i < 0 ? '⋮' : null));
  const el = h('div', { class: ['k-vec', 'h', `kind-${kind}`], style: { '--cell': `${CELL}px`, '--gap': `${GAP}px`, '--kind': `var(--c-${kind})` } },
    h('div', { class: 'k-cells' }, cells));
  return { el, idx, cells, width: stripWidth(n) };
}

/** Colour the cells of a strip. range sets the scale; cellId(i) names a cell for the hover card. */
function paintStrip(S, vals, range, cellId) {
  const r = range || 1e-12;
  S.idx.forEach((i, p) => {
    if (i < 0) return;
    const c = S.cells[p];
    c.style.background = diverging(vals[i] / r);
    c.title = `${i}: ${fmt(vals[i], 4)}`;
    if (cellId) c.dataset.cell = cellId(i);
  });
}

/** A vector as a short TeX tuple: first three entries, then dots. */
function vecTex(v) {
  const k = Math.min(3, v.length);
  return `(${Array.from(v.slice(0, k), (x) => num(x, 3)).join(',\\,')}${v.length > k ? ',\\,\\dots' : ''})`;
}

/** The k tokens whose logits a direction w raises most: w through the unembedding W_U [V, d]. */
function lensTop(WU, V, d, w, k) {
  const z = new Float64Array(V);
  for (let v = 0; v < V; v++) { let acc = 0; const b = v * d; for (let j = 0; j < d; j++) acc += WU[b + j] * w[j]; z[v] = acc; }
  return Array.from(z.keys()).sort((p, q) => z[q] - z[p] || p - q).slice(0, k).map((v) => ({ v, z: z[v] }));
}

/** Everything this scene needs for the selected token, read from the trace and the weights. */
function dataFor(ctx, L) {
  const S = ctx.state, c = S.config, T = S.ids?.length ?? 0, tr = S.trace;
  if (!T || !tr || !c) return null;
  const actT = tr[`L${L}.mlp.act`], outT = tr[`L${L}.mlp.out`];
  const inKey = tr[`L${L}.ln2`] ? `L${L}.ln2` : `L${L}.resid_mid`;
  const xT = tr[inKey];
  if (!actT || !outT || !xT) return null;
  const F = actT.shape[actT.shape.length - 1], d = outT.shape[outT.shape.length - 1];
  const t = clamp(S.sel.token ?? 0, 0, T - 1);
  const act = actT.data.subarray(t * F, (t + 1) * F);
  const out = outT.data.subarray(t * d, (t + 1) * d);
  const x = xT.data.subarray(t * d, (t + 1) * d);

  const P = S.model?.params ?? {};
  const W = (name) => P[name]?.data ?? null;
  const sw = c.activation === 'swiglu', relu = c.activation === 'relu', bias = !!c.bias;
  const Wd = W(`blocks.${L}.mlp.down_proj.weight`), Wu = W(`blocks.${L}.mlp.up_proj.weight`);
  const Wg = sw ? W(`blocks.${L}.mlp.gate_proj.weight`) : null;
  const bd = bias ? W(`blocks.${L}.mlp.down_proj.bias`) : null;
  const bu = bias ? W(`blocks.${L}.mlp.up_proj.bias`) : null;
  const bg = bias && sw ? W(`blocks.${L}.mlp.gate_proj.bias`) : null;
  const WU = W(c.tie_weights ? 'wte.weight' : 'lm_head.weight');
  if (!Wd || !Wu || (sw && !Wg) || !WU || Wd.length !== d * F || Wu.length !== F * d || WU.length % d) return null;
  const V = WU.length / d;

  // every hidden pre-activation from the input, then every activation: checked against the trace below
  const dotRow = (M, r, cols, v) => { let acc = 0; const b = r * cols; for (let k = 0; k < cols; k++) acc += M[b + k] * v[k]; return acc; };
  const hPre = new Float64Array(F), gPre = sw ? new Float64Array(F) : null;
  for (let f = 0; f < F; f++) {
    hPre[f] = dotRow(Wu, f, d, x) + (bu ? bu[f] : 0);
    if (sw) gPre[f] = dotRow(Wg, f, d, x) + (bg ? bg[f] : 0);
  }
  let errAct = 0;
  for (let f = 0; f < F; f++) {
    const a = sw ? siluScalar(gPre[f]) * hPre[f] : relu ? Math.max(0, hPre[f]) : geluScalar(hPre[f]);
    errAct = Math.max(errAct, Math.abs(a - act[f]));
  }

  // rank the units by |a|; the top K are shown
  const order = Array.from({ length: F }, (_, n) => n).sort((p, q) => Math.abs(act[q]) - Math.abs(act[p]) || p - q);
  const rank = new Int32Array(F);
  order.forEach((n, r) => { rank[n] = r; });
  const KK = Math.min(K, F), top = order.slice(0, KK);
  let sumAbs = 0, topAbs = 0, rAct = 0, nZero = 0;
  for (let n = 0; n < F; n++) { sumAbs += Math.abs(act[n]); rAct = Math.max(rAct, Math.abs(act[n])); if (act[n] === 0) nZero++; }
  for (const n of top) topAbs += Math.abs(act[n]);
  rAct ||= 1;
  const share = sumAbs ? topAbs / sumAbs : 0;
  let rW = 0; for (const v of Wd) rW = Math.max(rW, Math.abs(v)); rW ||= 1;

  // write directions: column n of W_down is unit n's write direction (length d)
  const colOf = (n) => { const v = new Float64Array(d); for (let j = 0; j < d; j++) v[j] = Wd[j * F + n]; return v; };
  const scaled = top.map((n) => Float64Array.from(colOf(n), (w) => act[n] * w));
  const topSum = new Float64Array(d), rest = new Float64Array(d);
  const inTop = new Uint8Array(F);
  top.forEach((n) => { inTop[n] = 1; });
  for (let n = 0; n < F; n++) {
    const tgt = inTop[n] ? topSum : rest, a = act[n];
    for (let j = 0; j < d; j++) tgt[j] += a * Wd[j * F + n];
  }
  const biasV = bd ? Float64Array.from(bd) : new Float64Array(d);
  let errSum = 0;
  for (let j = 0; j < d; j++) errSum = Math.max(errSum, Math.abs(topSum[j] + rest[j] + biasV[j] - out[j]));

  // one common scale for every vector that is added up, so the sum reads on the same colours
  let rSc = 0;
  const scan = (v) => { for (const q of v) rSc = Math.max(rSc, Math.abs(q)); };
  scaled.forEach(scan); scan(topSum); scan(rest); scan(biasV); scan(out);
  rSc ||= 1;

  // logit lens for the top units
  const lens = top.map((n) => lensTop(WU, V, d, colOf(n), LENS));

  return {
    L, t, d, F, V, KK, sw, relu, bias, act, out, top, rank, share, sumAbs, rAct, rW, rSc, nZero,
    hPre, gPre, errAct, errSum, scaled, topSum, rest, biasV, colOf, lens,
    sig: [L, d, F, V, KK, sw, relu, bias].join('|'),
  };
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const L = Number(ctx.params.l);
    let step = 0, built = null, parts = [], arrows = [], R = null, destroyed = false, auto = null, pulseT = null;
    const lab = (v) => ctx.state.tokenizer.label(v);

    // ------------------------------------------------------------ persistent parts: header, picker, panel, side, caption
    const header = sceneHeader(st, { chips: [[`Layer ${L}`, 'resid'], ['MLP', 'mlp']], title: 'Neurons as memories' });
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'mlp' });
    const cap = sceneCaption(st, {});
    const panel = h('div', { class: 'ns-panel', style: { width: `${PANEL.w}px`, height: `${PANEL.h}px` } });
    st.add(panel, { x: PANEL.x, y: PANEL.y, anchor: 'tl', layer: 'back' });
    const side = h('div', { class: 'ns-side' });
    st.add(side, { x: ZONES.side.x, y: SIDE_Y, anchor: 'tl' });

    /** Build the elements for one configuration. They start in their neutral state; paint() moves them. */
    function build(D) {
      parts.forEach((p) => p.remove());
      arrows.forEach((a) => a.destroy());
      parts = []; arrows = [];
      const place = (node, x, y, anchor = 'lc') => { parts.push(st.add(node, { x, y, anchor, layer: 'mid' })); return node; };
      const { d, F, KK, bias } = D;
      const mv = (node) => { node.classList.add('ns-mv'); return node; };
      const mvStrip = (x, y) => { const S = strip(d, 'mlp'); place(S.el, x, y); mv(S.el); return S; };

      // spectrum of all F units, ranked by |a| (step 0)
      const pitch = SPEC.w / F, bw = Math.max(1.5, pitch * 0.7);
      const barEls = Array.from({ length: F }, () => h('div', { class: 'ns-bar', style: { width: `${bw}px`, height: '0px' } }));
      const band = h('div', { class: 'ns-band' });            // shades the top K, behind the bars
      const bracket = h('div', { class: 'ns-bracket' });
      const topLab = h('div', { class: 'ns-top-lab' });
      const maxLab = h('div', { class: 'ns-max-lab' });
      const spec = mv(h('div', { class: 'ns-spec' }, band, bracket, topLab, maxLab,
        h('div', { class: 'ns-bars', dataset: { trace: `L${L}.mlp.act` } }, barEls),
        h('div', { class: 'ns-base' }),
        h('div', { class: 'ns-axis', style: { left: '0px' } }, '0'),
        h('div', { class: 'ns-axis', style: { right: '0px' } }, `F = ${F} hidden units, ranked by |a|`)));
      place(spec, SPEC.x, SPEC.y, 'tl');

      // rows: one per top unit (steps 1 to 3)
      const rows = [];
      for (let k = 0; k < KK; k++) {
        const y = ROW0 + k * ROWP;
        const fill = h('span', { class: 'ns-fill' });
        const track = mv(place(h('div', { class: 'ns-track', dataset: { trace: `L${L}.mlp.act` } }, h('span', { class: 'ns-zero' }), fill), X.bar, y));
        const W = strip(d, 'mlp');
        place(W.el, X.w, y); mv(W.el);
        const S = strip(d, 'mlp');
        place(S.el, X.s, y); mv(S.el);
        const arrow = st.arrow({ x: X.w + W.width + 8, y }, { x: X.chip - 8, y }, { kind: 'mlp', width: 2 });
        arrow.el.classList.add('ns-mv');
        arrows.push(arrow);
        rows.push({
          tag: mv(place(h('div', { class: 'ns-tag' }), X.tag, y)),
          fill, track,
          val: mv(place(h('div', { class: 'ns-val' }), X.val, y)),
          times: mv(place(op('\\times', { kind: 'mlp', size: 48 }), X.times, y, 'c')),
          W, S,
          eq: mv(place(op('=', { kind: 'mlp', size: 48 }), X.eq, y, 'c')),
          chips: Array.from({ length: LENS }, (_, j) => mv(place(h('div', { class: 'ns-chip' }, h('b'), h('small')), X.chip + j * X.chipP, y))),
          arrow,
        });
      }

      // the stacked sum (step 2): the top units, then the rest, then the bias, then the traced result
      const restOn = F > KK, biasOn = !!bias;
      const restY = STACK0 + KK * STACKP;
      const biasY = STACK0 + (KK + (restOn ? 1 : 0)) * STACKP;
      const lastY = biasOn ? biasY : restOn ? restY : STACK0 + (KK - 1) * STACKP;
      const resultY = (STACK0 + lastY) / 2;
      const restG = restOn ? { tag: mv(place(h('div', { class: 'ns-tag' }, 'rest'), X.label, restY)), plus: mv(place(h('div', { class: 'ns-plus' }, '+'), X.plus, restY)), S: mvStrip(X.sum, restY) } : null;
      const biasG = biasOn ? { tag: mv(place(h('div', { class: 'ns-tag' }, 'bias'), X.label, biasY)), plus: mv(place(h('div', { class: 'ns-plus' }, '+'), X.plus, biasY)), S: mvStrip(X.sum, biasY) } : null;
      const rule = mv(place(h('div', { class: 'ns-rule', style: { width: `${stripWidth(d)}px` } }), X.sum, lastY + 17));
      const resS = strip(d, 'mlp');
      resS.el.dataset.trace = `L${L}.mlp.out`;     // the traced result: its cells carry the maths on hover
      place(resS.el, X.s, resultY); mv(resS.el);
      const res = {
        S: resS,
        lab: mv(place(h('div', { class: 'ns-tag', style: { color: 'var(--c-mlp)' }, html: texHTML('\\text{mlp.out}[t]') }), X.s, resultY - 34)),
        eq: mv(place(op('=', { kind: 'mlp', size: 48 }), X.eq, resultY, 'c')),
      };

      // column headings
      const heads = {
        tag: mv(place(h('div', { class: 'ns-head', html: 'unit' }), X.tag, HEAD_Y)),
        bar: mv(place(h('div', { class: 'ns-head', html: `activation ${texHTML('a_n')}` }), X.bar, HEAD_Y)),
        w: mv(place(h('div', { class: 'ns-head', html: `write direction ${texHTML('\\Wdown_{:,n}')}` }), X.w, HEAD_Y)),
        s: mv(place(h('div', { class: 'ns-head', html: texHTML('a_n\\,\\Wdown_{:,n}') }), X.s, HEAD_Y)),
        chip: mv(place(h('div', { class: 'ns-head', html: 'promotes (approx.)' }), X.chip, HEAD_Y)),
      };

      R = { spec, barEls, band, bracket, topLab, maxLab, pitch, bw, rows, restG, biasG, res, rule, restY, biasY, lastY, resultY, heads };

      // neutral start: everything hidden, so the first paint animates in
      [spec, ...rows.flatMap((r) => [r.tag, r.track, r.val, r.times, r.W.el, r.eq, r.S.el, ...r.chips]), ...(restG ? [restG.tag, restG.plus, restG.S.el] : []),
        ...(biasG ? [biasG.tag, biasG.plus, biasG.S.el] : []), rule, res.S.el, res.lab, res.eq, ...Object.values(heads)]
        .forEach((e) => { e.style.opacity = '0'; e.style.pointerEvents = 'none'; });
      rows.forEach((r) => { r.fill.style.width = '0px'; });
      st.root.offsetWidth;   // flush the neutral styles so the transitions start from them
    }

    /** Set one element's position and visibility; transitions carry the change. */
    const at = (el, x, y, on, dly = 0) => {
      st.move(el, x, y);
      el.style.opacity = on ? '1' : '0';
      el.style.pointerEvents = on ? '' : 'none';
      el.style.transitionDelay = `${dly}ms`;
    };

    /** Paint step i for the selected token: the spectrum, the rows, the stack, the side column and the caption. */
    function paint(D, i, { animate = true } = {}) {
      st.root.classList.toggle('ns-still', !animate || reducedMotion());
      const { t, F, KK, act, out, rAct, rW, rSc, top, rank, share, lens, colOf, scaled, rest, biasV, topSum } = D;
      const { pitch, bw } = R;
      const tok = esc(ctx.label(t));
      const cascade = (k) => (i === 1 || i === 2 ? k * 60 : 0);

      // step 0: every unit as a bar, ranked by |a|; the top K stand out
      at(R.spec, SPEC.x, SPEC.y, i === 0, 0);
      R.barEls.forEach((b, n) => {
        const r = rank[n], a = act[n];
        b.style.left = `${r * pitch + (pitch - bw) / 2}px`;
        b.style.height = `${Math.max(0.5, (Math.abs(a) / rAct) * SPEC.bh)}px`;
        b.style.background = diverging(a / rAct);
        b.classList.toggle('rest', r >= KK);
        b.dataset.cell = `${t},${n},0`;
        b.title = `unit ${n}: ${fmt(a, 4)}`;
      });
      R.band.style.width = `${KK * pitch + 10}px`;
      R.bracket.style.width = `${KK * pitch - 3}px`;
      R.topLab.style.left = `${KK * pitch + 12}px`;
      R.topLab.textContent = `the top ${KK} · ${Math.round(share * 100)}% of Σ|a|`;
      R.maxLab.textContent = `max |a| = ${fmt(rAct, 2)}`;

      // steps 1 to 3: the rows. In step 2 the scaled vectors move into the stack.
      const stacked = i === 2;
      R.rows.forEach((Rw, k) => {
        const n = top[k], a = act[n], y = ROW0 + k * ROWP, ys = STACK0 + k * STACKP, dl = cascade(k);
        const half = X.barW / 2, len = (Math.abs(a) / rAct) * half;
        Rw.fill.style.left = `${a >= 0 ? half : half - len}px`;
        Rw.fill.style.width = `${len}px`;
        Rw.fill.style.background = diverging(a / rAct);
        Rw.track.dataset.cell = `${t},${n},0`;
        Rw.tag.textContent = `n ${n}`;
        Rw.val.textContent = fmt(a, 2);
        paintStrip(Rw.W, colOf(n), rW);
        paintStrip(Rw.S, scaled[k], rSc);
        at(Rw.tag, stacked ? X.label : X.tag, stacked ? ys : y, i >= 1, dl);
        at(Rw.track, X.bar, y, i === 1, dl);
        at(Rw.val, X.val, y, i === 1, dl);
        at(Rw.times, X.times, y, i === 1, dl);
        at(Rw.W.el, X.w, y, i === 1 || i === 3, dl);
        at(Rw.eq, X.eq, y, i === 1, dl);
        at(Rw.S.el, stacked ? X.sum : X.s, stacked ? ys : y, i === 1 || i === 2, dl);
        Rw.chips.forEach((c, j) => {
          const e = lens[k][j];
          c.firstChild.textContent = lab(e.v);
          c.lastChild.textContent = fmt(e.z, 2);
          c.classList.toggle('neg', a < 0);
          at(c, X.chip + j * X.chipP, y, i === 3, dl);
        });
        Rw.arrow.setOpacity(i === 3 ? 1 : 0);
        Rw.arrow.el.style.transitionDelay = `${dl}ms`;
        if (i === 3 && animate) Rw.arrow.draw(480, dl);
      });

      // column headings
      const H = R.heads;
      at(H.tag, X.tag, HEAD_Y, i === 1 || i === 3, 0);
      at(H.bar, X.bar, HEAD_Y, i === 1, 0);
      at(H.w, X.w, HEAD_Y, i === 1 || i === 3, 0);
      at(H.s, X.s, HEAD_Y, i === 1, 0);
      at(H.chip, X.chip, HEAD_Y, i === 3, 0);

      // step 2: the stack, the rule and the traced result
      const on2 = i === 2;
      if (R.restG) {
        at(R.restG.tag, X.label, R.restY, on2, 0);
        at(R.restG.plus, X.plus, R.restY, on2, 0);
        paintStrip(R.restG.S, rest, rSc);
        at(R.restG.S.el, X.sum, R.restY, on2, 0);
      }
      if (R.biasG) {
        at(R.biasG.tag, X.label, R.biasY, on2, 0);
        at(R.biasG.plus, X.plus, R.biasY, on2, 0);
        paintStrip(R.biasG.S, biasV, rSc);
        at(R.biasG.S.el, X.sum, R.biasY, on2, 0);
      }
      at(R.rule, X.sum, R.lastY + 17, on2, 0);
      paintStrip(R.res.S, out, rSc, (j) => `${t},${j},0`);
      at(R.res.S.el, X.s, R.resultY, on2, 120);
      at(R.res.lab, X.s, R.resultY - 34, on2, 120);
      at(R.res.eq, X.eq, R.resultY, on2, 120);
      if (on2 && animate) { clearTimeout(pulseT); pulseT = setTimeout(() => pulse(R.res.S.el), 700); }

      // text for this step: side column, caption, header line
      side.replaceChildren(...sideItems(D, i).flat().filter(Boolean));
      cap.set(words(D)[i]);
      header.setSub(`Token “<b>${tok}</b>” at position <b>${t}</b> is written into the stream by <b>${F}</b> hidden units, each adding one fixed direction.`);
    }

    /** Side column: the equation for this step, the live numbers, the check and one note. */
    function sideItems(D, i) {
      const { t, F, KK, sw, relu, bias, act, top, errAct, errSum, hPre, gPre, lens, colOf, scaled, rest, biasV, topSum, out, V, nZero, share } = D;
      const f = top[0], a0 = act[f];
      const fnName = relu ? 'ReLU' : sw ? 'SiLU' : 'GELU';
      const bu = bias ? ' + b_n' : '';
      const kick = h('div', { class: 'ns-kick' }, STEPS[i]);
      const main = (lines) => lines.map((tex) => h('div', { class: 'ns-main', html: texHTML(tex, { display: true }) }));
      const liveBox = (lines) => h('div', { class: 'ns-live' }, lines.filter(Boolean).map((tex) => h('div', { html: texHTML(tex) })));
      const checks = (...c) => h('div', { class: 'ns-checks' }, c);
      const note = (txt) => h('div', { class: 'ns-note', html: txt });

      if (i === 0) {
        const eq = sw
          ? [`g_n = (\\Wgate\\,\\hat{x})_n${bias ? ' + b_n' : ''}`, `u_n = (\\Wup\\,\\hat{x})_n${bu}`, `a_n = \\SiLU(g_n)\\,u_n`]
          : [`h_n = (\\Wup\\,\\hat{x})_n${bu}`, `a_n = \\${fnName}(h_n)`];
        const numbers = sw
          ? [`n^\\star = ${f}`, `g_{${f}} = ${num(gPre[f], 2)}, \\quad u_{${f}} = ${num(hPre[f], 2)}`, `a_{${f}} = ${num(a0, 2)}`]
          : [`n^\\star = ${f}`, `h_{${f}} = ${num(hPre[f], 2)}`, `a_{${f}} = \\${fnName}(h_{${f}}) = ${num(a0, 2)}`];
        return [kick, main(eq),
          liveBox([...numbers, `\\frac{\\sum_{\\text{top }${KK}}|a_n|}{\\sum_{n}|a_n|} = ${Math.round(share * 100)}\\%`]),
          checks(checkBadge(errAct, 0, { label: `equals L${L}.mlp.act, all ${F} units` })),
          note(relu ? `${nZero} of ${F} activations are exactly 0.` : sw ? 'The gate SiLU(g) scales the up-value u.' : 'GELU keeps negative values above −0.17.')];
      }
      if (i === 1) {
        return [kick, main([`\\mathbf{v}_n = a_n\\,\\Wdown_{:,n} \\in \\R^{${D.d}}`]),
          liveBox([`n^\\star = ${f}, \\quad a_{${f}} = ${num(a0, 2)}`, `\\Wdown_{:,${f}} = ${vecTex(colOf(f))}`, `\\mathbf{v}_{${f}} = ${vecTex(scaled[0])}`]),
          note('Each row is one hidden unit: a row of W_up and a column of W_down.')];
      }
      if (i === 2) {
        return [kick, main([`\\mathbf{m}_t = \\sum_{n=1}^{${F}} a_n\\,\\Wdown_{:,n}${bias ? ' + b_{\\text{down}}' : ''}`]),
          liveBox([`\\sum_{\\text{top }${KK}} a_n\\,\\Wdown_{:,n} = ${vecTex(topSum)}`, F > KK ? `\\sum_{\\text{rest}} = ${vecTex(rest)}` : '',
            bias ? `b_{\\text{down}} = ${vecTex(biasV)}` : '', `\\text{mlp.out}[t] = ${vecTex(out)}`]),
          checks(checkBadge(errSum, 0, { label: bias ? 'sum of all units + bias equals mlp.out[t]' : 'sum of all units equals mlp.out[t]' })),
          note(`The traced tensor is L${L}.mlp.out, row ${t}.`)];
      }
      // step 3: the logit lens on the strongest unit
      const lensNodes = lens[0].map(({ v, z }) => h('div', { class: 'ns-lens' }, h('b', null, lab(v)), h('span', null, `ℓ = ${fmt(z, 2)}`)));
      return [kick, main([`\\ell_n = \\WU\\,\\Wdown_{:,n} \\in \\R^{${V}}`]),
        h('div', { class: 'ns-live' }, h('div', { html: texHTML(`n^\\star = ${f}`) }), ...lensNodes),
        note('Approximation: the lens ignores later layers and the final norm.'),
        a0 < 0 ? note(`a = ${fmt(a0, 2)} is negative here, so these tokens are pushed down.`) : null];
    }

    function words(D) {
      const { F, KK, share, bias } = D;
      return [
        `Each unit has one activation <b>a</b> for this token. The top ${KK} by |a| are shown; together they carry <b>${Math.round(share * 100)}%</b> of Σ|a| over all ${F} units.`,
        'Unit <b>n</b> writes its activation times one fixed direction, the column <b>W_down[:, n]</b>. A negative activation writes the opposite direction.',
        `The scaled vectors, the other units${bias ? ' and the bias' : ''} add up to the traced <b>mlp.out[t]</b>, to rounding error.`,
        'Projected through the unembedding, each direction favours some tokens. This is an approximation: it ignores later layers and the final norm.',
      ];
    }

    /** Make the DOM match the current trace (rebuilding when the shapes change), then paint the current step. */
    function refresh({ animate = true } = {}) {
      if (destroyed) return;
      const D = dataFor(ctx, L);
      if (!D) { cap.set('This layer has no MLP trace for the selected token.'); return; }
      if (built !== D.sig) { build(D); built = D.sig; }
      paint(D, step, { animate });
    }

    function setStep(i) {
      step = i;
      refresh({ animate: true });
    }

    ctx.setSteps(STEPS, setStep);
    const offs = [ctx.on('trace', () => refresh({ animate: true })), ctx.on('select', () => refresh({ animate: true }))];
    const offTheme = onThemeChange(() => refresh({ animate: false }));
    refresh({ animate: true });
    appear([header.el, picker.el, side, panel], { stagger: MOTION.stagger });

    // auto-advance the first time the scene is shown, so it explains itself (not under reduced motion)
    if (!reducedMotion()) {
      auto = setTimeout(function next() {
        if (step < STEPS.length - 1) { ctx.setStep(step + 1); auto = setTimeout(next, MOTION.step); }
      }, 900);
      el.addEventListener('pointerdown', () => clearTimeout(auto), { once: true });
    }

    return {
      destroy() {
        destroyed = true;
        clearTimeout(auto);
        clearTimeout(pulseT);
        offs.forEach((f) => f());
        offTheme();
        st.destroy();
      },
    };
  },
};

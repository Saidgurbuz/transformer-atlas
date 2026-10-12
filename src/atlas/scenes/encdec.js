// Encoder–decoder in three steps, one composition throughout: a focus in the hero, its context in the side column.
//   1. The 2017 Transformer, drawn in. The figure is the focus; the side compares the paper with this model.
//   2. Cross-attention, live. The arc diagram is the focus. The figure moves to the side, dimmed, with the
//      cross-attention block lit. An illustration: the weights are layer 0’s own projections, never trained for it.
//   3. The families. Four mask matrices are the focus; this model’s is its traced mask. The figure stays in the side
//      with the self-attention blocks lit, because that is where the mask acts.
// Reference scene for contributors: see docs/ATLAS_GUIDE.md and softmax.js.
import { stage2d, sceneHeader, sceneCaption, headPicker, mat, vec, chip, checkBadge, pulse, useStyle, MOTION, ZONES } from '../kit.js';
import { h, s, num, texHTML, reducedMotion } from '../../ui/index.js';
import { arcDiagram } from '../../ui/arcs.js';
import { cloneModel, infer, maskAllows } from '../../engine/index.js';

const STEPS = ['The 2017 Transformer', 'Cross-attention', 'Families'];

const SUB = [
  'Vaswani et al. (2017): an encoder reads the source; a decoder writes the target.',
  'The decoder asks the encoder’s memory what to read. Each decoder token takes a weighted mix of the source tokens.',
  'The mask decides what each token may see, and so what a model is for.',
];

const CAP = [
  'The encoder reads the source with bidirectional self-attention. The decoder writes the target with masked self-attention, then cross-attention reads the encoder output.',
  'Queries come from the decoder’s layer-0 input through W_Q. Keys and values come from the encoder memory through W_K and W_V, so every source token is visible and each row sums to 1.',
  'Encoder-only models read both ways; decoder-only models see only the past. Encoder–decoder and prefix-LM models combine the two: a bidirectional part for the input, a causal part for the output.',
];

const FAMILIES = [
  { id: 'enc', name: 'Encoder-only', ex: 'BERT · masked LM', rule: 'Bidirectional: every token sees every token.' },
  { id: 'dec', name: 'Decoder-only', ex: 'GPT, LLaMA · causal LM', rule: 'Causal: each token sees itself and the past.' },
  { id: 'encdec', name: 'Encoder–decoder', ex: 'T5, BART · seq2seq', rule: 'The encoder sees all of the source; the decoder is causal and reads all of the encoder.' },
  { id: 'prefix', name: 'Prefix-LM', ex: 'UL2, PaLI · input side', rule: 'The prefix (input, image or prompt) is visible both ways; the rest is causal.' },
];
const FAMILY_OF_MASK = { none: 'enc', causal: 'dec', sliding: 'dec', prefix: 'prefix' };

const SRC0 = 'a cat sat';          // default source phrase
const N_MINI = 8;                  // tokens in each mini mask
const PREFIX_ILL = 4;              // prefix length of the illustrative prefix-LM mask (DEFAULT_CONFIG.prefix_len)
const MINI = { cell: 16, gap: 1, mode: 'sequential', range: 1, kind: 'attn' };
const STAGGER = 80;                // ms between build-up items
const BW = 300, FW = 340;          // block and frame widths, design units
const E = 360, D = 890;            // column centres: encoder, decoder
const FIG = { x: 190, y: 262, w: 870, h: 486 };     // the figure’s box at full size (hero position)
const CTX = { x: 1180, y: 300 };                    // where the figure sits as context, in the side column
const CTX_K = 340 / FIG.w;                          // and its scale there
const HERO = ZONES.hero;                            // x 110, y 270

useStyle('encdec', `
.ed-group { position: absolute; left: 0; top: 0; width: 0; height: 0; }
.ed-fade { transition: opacity .5s var(--ease), visibility .5s; }
.ed-off { opacity: 0 !important; visibility: hidden; pointer-events: none; }
.ed-arch { transform-origin: 0 0; transition: transform .6s cubic-bezier(.65,0,.35,1); }
.ed-cross { width: 1000px; }
.ed-cross .arcs { max-width: none; }
.ed-spacer { height: 228px; }
.ed-svg { position: absolute; left: 0; top: 0; width: 1600px; height: 900px; overflow: visible; pointer-events: none; }
.ed-line { fill: none; stroke-width: 2; stroke-linecap: round; transition: opacity .6s; }
.ed-frame { box-sizing: border-box; border-radius: 16px; border: 1px dashed var(--line-strong); background: color-mix(in srgb, var(--surface-2) 40%, transparent); transition: opacity .6s; }
.ed-frame-l { position: absolute; left: 14px; top: -10px; padding: 0 8px; background: var(--bg); font: 600 15px/18px var(--font-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--fg-3); white-space: nowrap; transition: opacity .6s; }
.ed-block { box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; padding: 4px 14px 4px 20px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 92%, transparent); border: 1px solid color-mix(in srgb, var(--kind) 45%, var(--line)); box-shadow: 0 10px 22px rgba(0, 0, 0, .13), 0 1px 0 color-mix(in srgb, var(--kind) 30%, transparent); transition: opacity .6s, box-shadow .6s, border-color .6s; }
.ed-block::before { content: ""; position: absolute; left: 0; top: 8px; bottom: 8px; width: 3px; border-radius: 0 3px 3px 0; background: var(--kind); }
.ed-block.ed-hl { border-color: var(--accent); box-shadow: 0 0 0 7px color-mix(in srgb, var(--accent) 40%, transparent), 0 10px 22px rgba(0, 0, 0, .13); }
.ed-block-t { font: 650 17px/1.15 var(--font-display); color: var(--fg); white-space: nowrap; transition: opacity .6s; }
.ed-block-s { font: 500 15px/1.25 var(--font-mono); color: var(--fg-3); white-space: nowrap; margin-top: 2px; transition: opacity .6s; }
.ed-pill { box-sizing: border-box; display: grid; place-items: center; border-radius: 999px; background: color-mix(in srgb, var(--kind) 16%, var(--surface)); border: 1px solid color-mix(in srgb, var(--kind) 55%, var(--line)); font: 600 15px/1 var(--font-mono); color: var(--fg); white-space: nowrap; box-shadow: var(--shadow-1); transition: opacity .6s, color .6s; }
.ed-tag { position: absolute; translate: -50% -100%; font: 600 15px/1 var(--font-mono); color: var(--c-resid); white-space: nowrap; transition: opacity .6s; }
.ed-ctx .ed-block { opacity: .3; }
.ed-ctx .ed-block.ed-hl { opacity: 1; }
.ed-ctx .ed-block-t, .ed-ctx .ed-block-s, .ed-ctx .ed-frame-l, .ed-ctx .ed-tag { opacity: 0; }
.ed-ctx .ed-pill { opacity: .3; color: transparent; }
.ed-ctx .ed-frame { opacity: .5; }
.ed-ctx .ed-line, .ed-ctx .ed-head { opacity: .45; }
.ed-input { width: 340px; box-sizing: border-box; font: 600 20px/1.2 var(--font-mono); color: var(--fg); background: var(--surface); border: 1px solid var(--line-strong); border-radius: 10px; padding: 8px 12px; box-shadow: var(--shadow-1); outline: none; }
.ed-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 28%, transparent); }
.ed-src { display: flex; align-items: center; gap: 14px; white-space: nowrap; }
.ed-src-l { font: 600 15px/1 var(--font-mono); color: var(--fg-3); text-transform: uppercase; letter-spacing: .08em; }
.ed-info { font: 500 15px/1.3 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.ed-note { font-size: 16px; line-height: 1.45; color: var(--fg-3); }
.ed-hero-l { font: 600 17px/1.2 var(--font-mono); color: var(--c-attn); white-space: nowrap; }
.ed-side { box-sizing: border-box; width: 340px; display: grid; gap: 14px; }
.ed-lbl { font: 600 15px/1.35 var(--font-mono); color: var(--fg-2); }
.ed-cap { font-size: 16px; line-height: 1.45; color: var(--fg-3); }
.ed-eqbox { display: grid; gap: 6px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 88%, transparent); border: 1px solid var(--line); box-shadow: var(--shadow-1); }
.ed-eq { font-size: 22px; overflow-x: auto; }
.ed-live-eq { font: 600 16px/1.35 var(--font-mono); color: var(--fg); }
.ed-live-sub { font: 500 14px/1.3 var(--font-mono); color: var(--fg-3); }
.ed-qrow { display: flex; align-items: center; gap: 10px; }
.ed-qlab { font: 600 17px/1 var(--font-mono); color: var(--c-q); }
.ed-checks { display: grid; gap: 8px; justify-items: start; }
.ed-tbl { display: grid; grid-template-columns: 1.4fr 1fr 1fr; font: 500 16px/1 var(--font-mono); }
.ed-tbl > div { padding: 11px 4px; border-bottom: 1px solid var(--line); color: var(--fg-2); }
.ed-tbl > .ed-th { font: 600 13px/1 var(--font-mono); letter-spacing: .04em; text-transform: uppercase; white-space: nowrap; color: var(--fg-3); padding-top: 0; }
.ed-tbl > .ed-paper { color: var(--fg-3); }
.ed-tbl > .ed-mine { color: var(--fg); font-weight: 600; }
.ed-legend { display: flex; gap: 18px; align-items: center; font: 500 15px/1.2 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.ed-sw { display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 8px; vertical-align: -2px; }
.ed-sw.on { background: var(--c-attn); }
.ed-sw.off { background: repeating-linear-gradient(135deg, var(--c-mask) 0 2px, transparent 2px 5px); box-shadow: inset 0 0 0 1px var(--c-mask); }
.ed-grid { display: grid; grid-template-columns: 500px 500px; gap: 20px; width: 1020px; }
.ed-tile { box-sizing: border-box; position: relative; display: grid; grid-template-columns: 156px minmax(0, 1fr); gap: 22px; align-items: center; padding: 20px 22px; border-radius: 16px; background: color-mix(in srgb, var(--surface) 90%, transparent); border: 1px solid var(--line); box-shadow: 0 16px 38px rgba(0, 0, 0, .12); }
.ed-tile.is-this { border-color: var(--accent); box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 50%, transparent), 0 16px 38px rgba(0, 0, 0, .14); }
.ed-tile-mat { justify-self: center; }
.ed-tile-text { min-width: 0; }
.ed-tile-chip { position: absolute; left: 18px; top: -12px; }
.ed-tile-name { font: 700 27px/1.1 var(--font-display); letter-spacing: -.01em; color: var(--fg); }
.ed-tile-ex { font: 600 14px/1.2 var(--font-mono); color: var(--fg-3); margin-top: 6px; }
.ed-tile-rule { font-size: 16px; line-height: 1.4; color: var(--fg-2); margin-top: 10px; }
.ed-tile-note { font: 500 14px/1.3 var(--font-mono); color: var(--fg-2); margin-top: 8px; }
.ed-tile-check { margin-top: 8px; }
.ed-tile-note:empty, .ed-tile-check:empty { display: none; }
@media (prefers-reduced-motion: reduce) {
  .ed-arch, .ed-fade, .ed-line, .ed-frame, .ed-frame-l, .ed-block, .ed-block-t, .ed-block-s, .ed-pill, .ed-tag { transition: none !important; }
}
`);

// Place a node at (x, y) in design units inside a parent that sits at the canvas origin.
const place = (parent, node, x, y, w, hh) => {
  node.style.position = 'absolute';
  node.style.left = `${x}px`;
  node.style.top = `${y}px`;
  if (w != null) node.style.width = `${w}px`;
  if (hh != null) node.style.height = `${hh}px`;
  parent.append(node);
  return node;
};

// Arrow head: a triangle with its tip at (tx, ty), pointing along the unit vector (ux, uy).
const headD = (tx, ty, ux, uy) => {
  const bx = tx - ux * 10, by = ty - uy * 10, nx = -uy * 5, ny = ux * 5;
  return `M${tx} ${ty}L${bx + nx} ${by + ny}L${bx - nx} ${by - ny}Z`;
};

// Linear map y = x Wᵀ + b, with X [rows, din] and W [dout, din] stored row-major (nn.Linear layout).
const project = (X, rows, din, W, dout, b) => {
  const Y = new Float64Array(rows * dout);
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < dout; o++) {
      let acc = b ? b[o] : 0;
      for (let i = 0; i < din; i++) acc += X[r * din + i] * W[o * din + i];
      Y[r * dout + o] = acc;
    }
  }
  return Y;
};

// Illustrative mini mask, N × N: 1 where a query may see a key, −∞ (hatched by the kit) where not.
const miniMask = (id, prefixLen = PREFIX_ILL) => {
  const N = N_MINI, half = N / 2, out = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let ok;
      if (id === 'enc') ok = maskAllows({ mask: 'none' }, i, j);
      else if (id === 'dec') ok = maskAllows({ mask: 'causal' }, i, j);
      else if (id === 'prefix') ok = maskAllows({ mask: 'prefix', prefix_len: prefixLen }, i, j);
      else if (i < half) ok = j < half;                                // encoder rows: the source, both ways
      else if (j < half) ok = true;                                    // decoder rows: cross-attention to all of it
      else ok = maskAllows({ mask: 'causal' }, i - half, j - half);    // decoder rows: causal among themselves
      out[i * N + j] = ok ? 1 : -Infinity;
    }
  }
  return out;
};

const maskName = (cfg) => (cfg.mask === 'none' ? 'no mask'
  : cfg.mask === 'sliding' ? `sliding window ${cfg.window}`
    : cfg.mask === 'prefix' ? `prefix ${cfg.prefix_len}` : 'causal mask');
const ruleNote = (cfg) => (cfg.mask === 'sliding' ? `window ${cfg.window}`
  : cfg.mask === 'prefix' ? `prefix ${cfg.prefix_len}` : '');

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const header = sceneHeader(st, { chips: [], title: 'Encoder–decoder', sub: SUB[0] });
    const eyebrow = header.el.querySelector('.k-eyebrow');
    const cap = sceneCaption(st);
    const timers = [], owned = new Set();
    const own = (c) => { owned.add(c); return c; };                 // components that hold theme listeners
    const release = (c) => { if (c) { owned.delete(c); c.destroy?.(); } };
    let step = -1, walking = false, auto = null;
    let cross = null, srcText = SRC0, dropped = [];
    let lastTrace, lastMine, lastN = -1;     // sentinels: the first families paint always builds the four masks
    let qStrip = null;

    // One group per part, placed in design units. The figure is a single group that moves between hero and side.
    const group = (cls) => { const g = h('div', { class: ['ed-group', cls] }); st.add(g, { x: 0, y: 0 }); return g; };
    const gArch = group('ed-arch'), gCross = group('ed-cross ed-fade'), gFam = group('ed-fade');
    const side = (cls, y) => { const d = h('div', { class: ['ed-side', 'ed-fade', cls] }); st.add(d, { x: ZONES.side.x, y }); return d; };
    const sArch = side('ed-s-arch', 318), sCtx = side('ed-s-ctx', 262), sCross = side('ed-s-cross', 530);

    // ---------------------------------------------------------------- step 1: the 2017 figure
    // Built bottom to top, each item with its own timing; arrows draw in as the stacks rise.
    const seq = [];
    const svg = s('svg', { class: 'ed-svg', width: 1600, height: 900, viewBox: '0 0 1600 900' });
    gArch.append(svg);
    const frame = (x, top, hh, label) => {
      const f = place(gArch, h('div', { class: 'ed-frame' }, h('div', { class: 'ed-frame-l' }, label)), x, top, FW, hh);
      seq.push({ node: f, fade: true });
      return f;
    };
    const block = (cx, top, hh, title, sub, kind) => {
      const n = place(gArch, h('div', { class: 'ed-block', style: { '--kind': `var(--c-${kind})` } },
        h('div', { class: 'ed-block-t' }, title), sub ? h('div', { class: 'ed-block-s' }, sub) : null), cx - BW / 2, top, BW, hh);
      seq.push({ node: n });
      return n;
    };
    const pill = (cx, top, hh, text, kind, w) => {
      const n = place(gArch, h('div', { class: 'ed-pill', style: { '--kind': `var(--c-${kind})` } }, text), cx - w / 2, top, w, hh);
      seq.push({ node: n });
      return n;
    };
    const arrow = (x1, y1, x2, y2) => {
      const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
      const p = s('path', { class: 'ed-line', d: `M${x1} ${y1}L${x2 - ux * 8} ${y2 - uy * 8}`, style: { stroke: 'var(--c-resid)' } });
      const t = s('path', { class: 'ed-head', d: headD(x2, y2, ux, uy), style: { fill: 'var(--c-resid)' } });
      svg.append(p, t);
      p.style.strokeDasharray = `${L}`;
      seq.push({ arrow: true, p, t, L });
    };
    const memArrow = (d, tx, ty, ux, uy) => {
      const p = s('path', { class: 'ed-line', d, style: { stroke: 'var(--c-resid)' } });
      const t = s('path', { class: 'ed-head', d: headD(tx, ty, ux, uy), style: { fill: 'var(--c-resid)' } });
      svg.append(p, t);
      const L = p.getTotalLength() || 320;
      p.style.strokeDasharray = `${L}`;
      seq.push({ arrow: true, p, t, L });
    };

    // encoder (left): self-attention and feed-forward, each followed by Add & Norm (post-LN, as in 2017)
    frame(E - FW / 2, 428, 216, 'Encoder ×6');
    pill(E, 716, 26, 'Inputs', 'embed', 140);
    arrow(E, 716, E, 700);
    block(E, 666, 34, 'Embedding + positions', null, 'embed');
    arrow(E, 666, E, 630);
    const encSA = block(E, 584, 46, 'Self-attention', 'bidirectional · no mask', 'attn');
    arrow(E, 584, E, 568);
    pill(E, 544, 24, 'Add & Norm', 'norm', 180);
    arrow(E, 544, E, 528);
    block(E, 482, 46, 'Feed-forward', 'position-wise MLP', 'mlp');
    arrow(E, 482, E, 466);
    pill(E, 442, 24, 'Add & Norm', 'norm', 180);
    arrow(E, 442, E, 412);
    pill(E, 386, 26, 'encoder output (memory)', 'resid', 240);

    // decoder (right): masked self-attention, cross-attention to the memory, feed-forward
    frame(D - FW / 2, 326, 318, 'Decoder ×6');
    pill(D, 716, 26, 'Outputs (shifted right)', 'embed', 230);
    arrow(D, 716, D, 700);
    block(D, 666, 34, 'Embedding + positions', null, 'embed');
    arrow(D, 666, D, 630);
    const decSA = block(D, 584, 46, 'Masked self-attention', 'causal · no peeking ahead', 'attn');
    arrow(D, 584, D, 568);
    pill(D, 544, 24, 'Add & Norm', 'norm', 180);
    arrow(D, 544, D, 528);
    const crossBlock = block(D, 482, 46, 'Cross-attention', 'Q: decoder · K, V: encoder', 'attn');
    arrow(D, 482, D, 466);
    pill(D, 442, 24, 'Add & Norm', 'norm', 180);
    arrow(D, 442, D, 426);
    block(D, 380, 46, 'Feed-forward', 'position-wise MLP', 'mlp');
    arrow(D, 380, D, 364);
    pill(D, 340, 24, 'Add & Norm', 'norm', 180);
    arrow(D, 340, D, 310);
    block(D, 274, 36, 'Linear + softmax', null, 'head');
    memArrow('M480 399H630Q660 399 660 429V475Q660 505 690 505H730', 740, 505, 1, 0);
    seq.push({ node: place(gArch, h('div', { class: 'ed-tag' }, 'memory (K, V)'), 555, 388) });
    const archMs = seq.length * STAGGER + 480;
    const sa = [encSA, decSA];

    // The figure’s place: the hero at full size, or the side column as dimmed context.
    const placeFig = (asContext) => {
      const k = asContext ? CTX_K : 1;
      const x = asContext ? CTX.x : FIG.x, y = asContext ? CTX.y : FIG.y;
      gArch.style.transform = `translate(${x - FIG.x * k}px, ${y - FIG.y * k}px) scale(${k})`;
    };
    const lit = (node, on) => node.classList.toggle('ed-hl', on);

    // ---------------------------------------------------------------- step 2: cross-attention (illustration)
    const input = h('input', { class: 'ed-input', type: 'text', value: srcText, autocomplete: 'off', spellcheck: 'false', 'aria-label': 'Source phrase for the encoder' });
    const srcInfo = h('div', { class: 'ed-info' });
    const xTitle = h('div', { class: 'ed-hero-l' });
    // The query row of this diagram is the one query control: a click on a query selects its token.
    const arcs = arcDiagram({
      labels: [], keyLabels: [], weights: { data: [], rows: 0, cols: 0 }, layout: 'bipartite', kind: 'attn', maxWidth: 1000,
      // A click on the selected query toggles it off inside the widget. The selection belongs to the scene, so keep it.
      onSelect: (i) => (i == null ? paintCross() : ctx.store.select({ token: i })),
    });
    arcs.el.style.width = '1000px';
    place(gCross, xTitle, HERO.x, 252);
    place(gCross, arcs.el, HERO.x, 290);
    place(gCross, h('div', { class: 'ed-src' }, h('span', { class: 'ed-src-l' }, 'source'), input), HERO.x, 622);
    place(gCross, srcInfo, HERO.x, 672);
    place(gCross, h('div', { class: 'ed-note' },
      'Illustration: layer 0’s own projections, never trained for cross-attention. The memory is this model run with mask “none”. No rotary or position bias is applied across the two streams.'),
    HERO.x, 706, 1000);
    input.addEventListener('input', () => { srcText = input.value; prepare(); paint(); });

    const live = h('div', { class: 'ed-live' });
    const qRow = h('div', { class: 'ed-qrow' });
    const checks = h('div', { class: 'ed-checks' });
    sCross.append(
      h('div', { class: 'ed-eqbox' },
        h('div', { class: 'ed-eq', html: texHTML('\\Aw = \\softmax\\!\\Big(\\frac{\\Q\\K^{\\top}}{\\sqrt{d_h}}\\Big)', { display: true }) }),
        h('div', { class: 'ed-cap' }, 'No mask: every source key is visible to every decoder token.')),
      live, qRow, checks);

    const hp = headPicker(st, ctx, { x: ZONES.controls.x, y: ZONES.controls.y, kind: 'attn' });
    hp.el.classList.add('ed-fade');

    // ---------------------------------------------------------------- side column for step 1 and the figure’s context
    const ctxNote = h('div', { class: 'ed-cap' });
    sCtx.append(h('div', { class: 'ed-lbl' }, 'In the 2017 model'), h('div', { class: 'ed-spacer' }), ctxNote);
    const tbl = h('div', { class: 'ed-tbl' });
    sArch.append(h('div', { class: 'ed-lbl' }, 'Base model (Vaswani et al., 2017) and this model'), tbl,
      h('div', { class: 'ed-cap' }, 'This model has a decoder stack only: no encoder, so no cross-attention.'));

    // ---------------------------------------------------------------- step 3: the families
    place(gFam, h('div', { class: 'ed-legend' },
      h('span', null, h('i', { class: 'ed-sw on' }), 'visible'),
      h('span', null, h('i', { class: 'ed-sw off' }), 'masked'),
      h('span', null, `illustrative masks: ${N_MINI} tokens, prefix ${PREFIX_ILL} · rows are queries, columns are keys`)),
    HERO.x, 250);
    const grid = place(gFam, h('div', { class: 'ed-grid' }), HERO.x, 292);
    const tiles = {};
    FAMILIES.forEach((f) => {
      const miniWrap = h('div', { class: 'ed-tile-mat' });
      const chipWrap = h('div', { class: 'ed-tile-chip', hidden: true }, chip('this model', 'attn'));
      const note = h('div', { class: 'ed-tile-note' });
      const check = h('div', { class: 'ed-tile-check' });
      const tile = h('div', { class: 'ed-tile', dataset: { family: f.id } },
        chipWrap, miniWrap,
        h('div', { class: 'ed-tile-text' },
          h('div', { class: 'ed-tile-name' }, f.name),
          h('div', { class: 'ed-tile-ex' }, f.ex),
          h('div', { class: 'ed-tile-rule' }, f.rule),
          note, check));
      grid.append(tile);
      tiles[f.id] = { el: tile, miniWrap, chipWrap, note, check, illus: own(mat(miniMask(f.id), N_MINI, N_MINI, MINI)), traced: null };
    });

    // ---------------------------------------------------------------- data for step 2
    // Cross-attention for the current input: queries from the decoder’s layer-0 input, keys from the encoder memory,
    // all through layer 0’s own q and k projections.
    function prepare() {
      const S0 = ctx.state, cfg = S0.config, M = S0.model, tr = S0.trace;
      cross = null;
      if (!M || !tr || !tr['L0.attn.q']) return;
      const T = S0.ids.length, H = cfg.n_head, Hkv = cfg.n_kv_head, d = cfg.d_model, dh = d / H;
      const src = S0.tokenizer.encode(srcText);
      dropped = src.dropped;
      const srcIds = src.ids.slice(0, cfg.block_size), S = srcIds.length;
      const proj = (name) => ({ w: M.params[`${name}.weight`].data, b: M.params[`${name}.bias`]?.data ?? null });
      const wq = proj('blocks.0.attn.q_proj'), wk = proj('blocks.0.attn.k_proj');
      // the decoder’s stream at layer 0’s attention (ln1 when pre-LN, the embedding when post-LN)
      const X = (tr['L0.ln1'] ?? tr['emb.out']).data;
      const Q = project(X, T, d, wq.w, H * dh, wq.b);
      const Tq = tr['L0.attn.q'].data;                     // the same projection, as the model traced it
      let qdiff = 0;
      for (let k = 0; k < Q.length; k++) qdiff = Math.max(qdiff, Math.abs(Q[k] - Tq[k]));
      let K = new Float64Array(0);
      if (S) {
        // the memory: this model with mask “none” over the source; its last stream is the encoder output
        const enc = cloneModel(M);
        enc.config = { ...enc.config, mask: 'none' };
        const etr = infer(enc, srcIds).trace;
        const mem = (etr.lnf ?? etr[`L${cfg.n_layer - 1}.resid_out`]).data;
        K = project(mem, S, d, wk.w, Hkv * dh, wk.b);
      }
      cross = { T, S, H, Hkv, dh, Q, K, qdiff, srcIds, decIds: S0.ids.slice(), tk: S0.tokenizer };
    }

    // Attention weights of the selected head: T × S, rows sum to 1, no mask.
    function pattern() {
      const C = cross;
      if (!C || !C.T || !C.S) return null;
      const hd = Math.min(Math.max(0, ctx.state.sel.head ?? 0), C.H - 1), kvh = Math.floor(hd / (C.H / C.Hkv));
      const qs = C.H * C.dh, ks = C.Hkv * C.dh, sc = ctx.state.config.attn_scale ? 1 / Math.sqrt(C.dh) : 1;
      const P = new Float64Array(C.T * C.S), row = new Float64Array(C.S);
      for (let i = 0; i < C.T; i++) {
        let m = -Infinity;
        for (let j = 0; j < C.S; j++) {
          let acc = 0;
          for (let c = 0; c < C.dh; c++) acc += C.Q[i * qs + hd * C.dh + c] * C.K[j * ks + kvh * C.dh + c];
          row[j] = acc * sc;
          if (row[j] > m) m = row[j];
        }
        let z = 0;
        for (let j = 0; j < C.S; j++) { row[j] = Math.exp(row[j] - m); z += row[j]; }
        for (let j = 0; j < C.S; j++) P[i * C.S + j] = row[j] / z;
      }
      return { hd, P };
    }

    // This model’s own mask over its first n tokens, cell by cell from the trace (data-trace "mask").
    const tracedMini = (tr, T, n) => mat(
      Array.from({ length: n * n }, (_, k) => (tr.mask.data[Math.floor(k / n) * T + (k % n)] === 1 ? 1 : -Infinity)),
      n, n, { ...MINI, trace: 'mask', cellOf: (r, c) => `${r},${c}` });

    // ---------------------------------------------------------------- paint
    function paintArch() {
      const cfg = ctx.state.config;
      const row = (k, paper, mine) => [h('div', null, k), h('div', { class: 'ed-paper' }, paper), h('div', { class: 'ed-mine' }, String(mine))];
      tbl.replaceChildren(
        h('div', { class: 'ed-th' }, ''), h('div', { class: 'ed-th' }, 'paper'), h('div', { class: 'ed-th' }, 'this model'),
        ...row('layers N', '6', cfg.n_layer),
        ...row('d_model', '512', cfg.d_model),
        ...row('heads h', '8', cfg.n_head),
        ...row('d_ff', '2048', cfg.d_ff),
        ...row('norm', 'post-LN', cfg.norm_placement === 'post' ? 'post-LN' : 'pre-LN'),
        ...row('stacks', 'enc + dec', 'dec'));
    }

    function paintCross() {
      const C = cross, pat = pattern();
      if (!C || !pat) {
        arcs.el.style.visibility = 'hidden';
        arcs.update({ labels: [], keyLabels: [], weights: { data: new Float64Array(0), rows: 0, cols: 0 }, query: null });
        xTitle.textContent = 'illustration · cross-attention';
        srcInfo.textContent = !C || !C.T ? 'Needs at least one decoder token.' : 'Type a source phrase with at least one character from the vocabulary.';
        live.replaceChildren();
        release(qStrip);
        qStrip = null;
        qRow.replaceChildren();
        checks.replaceChildren();
        return;
      }
      const { P, hd } = pat;
      const S = C.S, tk = C.tk, q = Math.max(0, Math.min(ctx.state.sel.token ?? 0, C.T - 1));
      arcs.el.style.visibility = '';
      arcs.update({
        labels: C.decIds.map((id) => tk.label(id)),
        keyLabels: C.srcIds.map((id) => tk.label(id)),
        weights: { data: P, rows: C.T, cols: S },
        query: q,
      });
      let jStar = 0;
      for (let j = 1; j < S; j++) if (P[q * S + j] > P[q * S + jStar]) jStar = j;
      const qLab = tk.label(C.decIds[q]), kLab = tk.label(C.srcIds[jStar]);
      xTitle.textContent = `illustration · cross-attention, head ${hd} · query “${qLab}” at position ${q}`;
      live.replaceChildren(
        h('div', { class: 'ed-live-eq' }, `A(“${qLab}”, “${kLab}”) = ${num(P[q * S + jStar], 2)}`),
        h('div', { class: 'ed-live-sub' }, `largest of ${S} weights in this row`));
      // the query row of this head, from the trace (L0.attn.q): hover it for the traced numbers
      const tq = ctx.state.trace['L0.attn.q'], dh = C.dh, Hq = C.H * dh;
      release(qStrip);
      qStrip = null;
      qRow.replaceChildren();
      if (tq) {
        qStrip = own(vec(Array.from({ length: dh }, (_, c) => tq.data[q * Hq + hd * dh + c]), {
          orientation: 'h', cell: 14, gap: 2, mode: 'diverging', kind: 'q', trace: 'L0.attn.q',
          cellOf: (c) => `${q},${hd * dh + c}`,
        }));
        qRow.append(h('div', { class: 'ed-qlab', html: texHTML(`\\Q_{${q}}`) }), qStrip);
      }
      checks.replaceChildren(checkBadge(C.qdiff, 0, { label: 'Q equals the traced L0.attn.q' }));
      srcInfo.textContent = `${S} source token${S === 1 ? '' : 's'} · memory: this model with mask “none”${dropped.length ? ` · dropped ${dropped.map((c) => `“${c}”`).join(' ')}` : ''}`;
    }

    function paintFamilies() {
      const S0 = ctx.state, cfg = S0.config, tr = S0.trace, T = S0.ids.length;
      const mine = FAMILY_OF_MASK[cfg.mask] ?? null;
      const n = Math.min(N_MINI, T), traced = !!tr?.mask && n > 0;
      let bad = 0;
      if (traced) {
        // the traced mask, cell by cell, against the rule this model’s mask names
        for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) if ((tr.mask.data[i * T + j] === 1) !== maskAllows(cfg, i, j)) bad++;
      }
      const rebuild = tr !== lastTrace || mine !== lastMine || n !== lastN;
      lastTrace = tr; lastMine = mine; lastN = n;
      for (const f of FAMILIES) {
        const t = tiles[f.id], isMine = f.id === mine;
        t.el.classList.toggle('is-this', isMine);
        t.chipWrap.hidden = !isMine;
        const parts = isMine ? [traced ? (T > n ? `first ${n} of ${T}` : `all ${T}`) : '', ruleNote(cfg)] : [];
        t.note.textContent = parts.filter(Boolean).join(' · ');
        t.check.replaceChildren(...(isMine && traced ? [checkBadge(bad, 0, { label: 'traced mask follows the rule' })] : []));
        if (rebuild) {
          release(t.traced);
          t.traced = isMine && traced ? own(tracedMini(tr, T, n)) : null;
          t.miniWrap.replaceChildren(t.traced ?? t.illus);
        }
      }
    }

    // Build-up: the mask fills in cell by cell, tile after tile.
    const buildIn = (node, delay0) => {
      node.querySelectorAll('.k-cell').forEach((c, i) => c.animate([{ opacity: 0 }, { opacity: 1 }],
        { duration: 220, delay: delay0 + i * 9, easing: MOTION.easeOut, fill: 'backwards' }));
    };
    function playFamilies() {
      if (reducedMotion()) return;
      FAMILIES.forEach((f, k) => buildIn(tiles[f.id].miniWrap, k * 150));
    }

    // Build-up on entry: every item fades or draws in, bottom to top.
    function playArch() {
      if (reducedMotion()) return;
      seq.forEach((it, i) => {
        const delay = i * STAGGER;
        if (it.arrow) {
          it.p.animate([{ strokeDashoffset: it.L }, { strokeDashoffset: 0 }], { duration: 420, delay, easing: MOTION.easeOut, fill: 'backwards' });
          it.t.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, delay: delay + 340, fill: 'backwards' });
        } else if (it.fade) {
          it.node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 600, delay, easing: MOTION.easeOut, fill: 'backwards' });
        } else {
          it.node.animate([{ opacity: 0, transform: 'translateY(12px)' }, { opacity: 1, transform: 'none' }],
            { duration: 460, delay, easing: MOTION.easeOut, fill: 'backwards' });
        }
      });
    }

    function paint() {
      const on = (node, yes) => node.classList.toggle('ed-off', !yes);
      on(sArch, step === 0);
      on(gCross, step === 1); on(sCross, step === 1); on(hp.el, step === 1);
      on(gFam, step === 2);
      on(sCtx, step > 0);
      gArch.classList.toggle('ed-ctx', step > 0);
      placeFig(step > 0);
      lit(crossBlock, step === 1);
      sa.forEach((b) => lit(b, step === 2));
      eyebrow.replaceChildren(chip(`This model · ${maskName(ctx.state.config)}`, 'resid'));
      header.setSub(SUB[step]);
      cap.set(CAP[step]);
      ctxNote.textContent = step === 2 ? 'Highlighted: self-attention, where the mask acts.' : '';
      if (step === 0) paintArch();
      if (step === 1) paintCross();
      if (step === 2) paintFamilies();
    }

    function setStep(i) {
      const was = step;
      step = i;
      paint();
      if (i === 0 && was !== 0) {
        playArch();
        timers.push(setTimeout(() => pulse(crossBlock), archMs - 200));
      }
      if (i === 2 && was !== 2) playFamilies();
    }

    // The figure draws itself in, then the story moves on by itself, once. Any step change by the reader
    // (the story bar, the keyboard) or any pointer press stops the walk.
    const stopWalk = () => { clearTimeout(auto); auto = null; };
    const walk = (ms) => {
      auto = setTimeout(() => {
        if (step >= STEPS.length - 1) return;
        walking = true;
        try { ctx.setStep(step + 1); } finally { walking = false; }
        walk(MOTION.step);
      }, ms);
    };
    ctx.setSteps(STEPS, (i) => { if (!walking) stopWalk(); setStep(i); });
    ctx.on('trace', () => { prepare(); paint(); });
    ctx.on('select', () => paint());
    document.addEventListener('pointerdown', stopWalk, true);

    prepare();
    setStep(0);
    walk(archMs + MOTION.step);

    return {
      destroy() {
        stopWalk();
        timers.forEach((t) => clearTimeout(t));
        document.removeEventListener('pointerdown', stopWalk, true);
        arcs.destroy();
        owned.forEach((c) => c.destroy?.());
        owned.clear();
        st.destroy();
      },
    };
  },
};

// Embedding: how each token id becomes a vector, and where its position vector joins.
// Left to right: the token tiles (the ids) -> W_E, the lookup table -> X_tok, one row per token.
// Below X_tok: P, the position vectors -> plus -> x₀, the block input. Without a position table
// (RoPE, ALiBi, none) there is no P: x₀ = X_tok, and a note says where position enters instead.
// Layout on the 1600 × 900 canvas (docs/STYLE_GUIDE.md): the hero spans x 110 … 1140 and the
// equation column sits at ZONES.side.x. Deviations, with reasons: the hero starts at y 222, not
// 270, because X_tok and P must both fit above the caption; the token tiles are the scene's
// picker, so a tokenPicker row would only repeat them; the tiles are 18 units tall because
// twelve rows of two stacked matrices fill the hero. Long sequences elide rows (first 8, ⋮, last
// 3). Matrices wider than 16 columns elide columns (first 12, ⋮, last 4), so x₀ stays in the hero.
// Reference scene for contributors: see docs/ATLAS_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, mat, op, card, chip, texLabel, checkBadge, appear, pulse, fly, wait, useStyle, ZONES, MOTION } from '../kit.js';
import { h, fmt, num, texHTML, diverging, onThemeChange, reducedMotion } from '../../ui/index.js';
import { mathFor, runLive } from '../../core/math/index.js';

const STEPS = ['Ids', 'Look up rows', 'Add positions', 'x₀'];
const HEAD = 8, TAIL = 3, MAX_ROWS = 12;             // rows: the first 8, ⋮, the last 3
const COL_HEAD = 12, COL_TAIL = 4, MAX_COLS = 16;     // columns: past 16, the first 12, ⋮, the last 4
const TOP = 222;                                      // hero top (the labels sit above it)
const HERO_BOTTOM = 770;                              // the caption starts at ZONES.caption.y (790)
const PAD = 9;                                        // .k-mat-body: 8px padding + 1px border
const ROW_H = 18, ROW_GAP = 2, PITCH = ROW_H + ROW_GAP;   // X_tok, P and x₀ rows
const GAP_V = 40;                                     // X_tok to P: room for the P label
const COL_W = 203;                                    // a matrix is about this wide (17 columns at most)
const X_IDS = ZONES.header.x, TILE_W = 118, LAB_W = 22, STUB_W = 66, STUB_LAB_W = 52;
const X_WTE = 268;                                    // W_E: its row labels, then the cells
const PLUS_D = 48;
const ANCHOR = { tl: '0 0', c: '-50% -50%', bl: '0 -100%', br: '-100% -100%' };

useStyle('embedding', `
.em-vis { position: absolute; }
.em-tok { position: absolute; box-sizing: border-box; width: ${TILE_W}px; height: ${ROW_H}px; display: flex; align-items: center; gap: 8px;
  padding: 0 8px 0 3px; border-radius: 6px; background: var(--surface); border: 1px solid color-mix(in srgb, var(--c-embed) 34%, var(--line));
  box-shadow: var(--shadow-1); font: 500 13px/1 var(--font-mono); color: var(--fg-3); cursor: pointer;
  transition: background-color .3s, border-color .3s, transform .35s cubic-bezier(.2,.7,.2,1); }
.em-tok b { display: grid; place-items: center; min-width: 22px; height: ${ROW_H - 4}px; padding: 0 4px; border-radius: 4px;
  background: color-mix(in srgb, var(--c-embed) 16%, transparent); color: var(--fg); font: 700 13px/1 var(--font-mono); }
.em-tok:hover { border-color: var(--c-embed); }
.em-tok:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.em-tok.sel { border-color: var(--accent); }
.em-tok.lit { background: var(--accent-soft); border-color: var(--accent); transform: translateX(8px); }
.em-ellt { position: absolute; box-sizing: border-box; width: ${TILE_W}px; height: ${ROW_H}px; padding-left: 6px; display: flex; align-items: center;
  font: 500 13px/1 var(--font-mono); color: var(--fg-3); }
.em-ellm { position: absolute; height: ${ROW_H}px; display: flex; align-items: center; justify-content: center; font: 500 13px/1 var(--font-mono);
  color: var(--fg-3); pointer-events: none; white-space: nowrap; }
.em-slot { position: absolute; pointer-events: none; }
.em-stub { position: absolute; display: flex; overflow: hidden; border-radius: 4px; box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--fg) 8%, transparent);
  transition: opacity .35s ease; }
.em-stub > span { flex: 1 1 0; min-width: 0; }
.em-stub-lab { position: absolute; height: ${ROW_H}px; font: 500 13px/${ROW_H}px var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.em-lift { position: absolute; z-index: 3; display: flex; gap: var(--gx, 1px); border-radius: 3px; pointer-events: none;
  box-shadow: 0 0 0 2px var(--accent), 0 0 16px color-mix(in srgb, var(--accent) 50%, transparent); }
.em-lift > span { flex: 0 0 var(--cw, 4px); }
.em-zlab, .em-zoom { cursor: zoom-in; }
.em-zoom .k-mat-body { transition: box-shadow .25s ease; }
.em-zoom:hover .k-mat-body { box-shadow: 0 0 0 2px var(--accent), 0 16px 36px rgba(0, 0, 0, .18); }
.em-wte.tilt .k-mat-body { transform: perspective(1100px) rotateX(16deg); }
.em-wte .k-grid { row-gap: var(--ep, 2px); column-gap: 1px; }
.em-wte .k-cell { height: var(--eh, 14px); transition: background-color .5s ease, opacity .4s ease; }
.em-wte .k-axis.row { box-sizing: border-box; width: ${LAB_W}px; height: var(--eh, 14px); padding: 0 6px 0 0; font-size: 13px; line-height: var(--eh, 14px);
  transition: opacity .3s, color .3s; }
.em-wte.em-focus .k-cell { opacity: .3; }
.em-wte.em-focus .k-axis.row { opacity: .5; }
.em-wte .k-cell.em-on { opacity: 1; }
.em-wte .k-axis.row.em-on { opacity: 1; color: var(--accent); font-weight: 700; }
.em-x .k-grid { row-gap: ${ROW_GAP}px; column-gap: 1px; }
.em-x .k-cell { height: ${ROW_H}px; transition: background-color .5s ease, opacity .45s ease; }
.em-x .k-cell.em-pend { opacity: 0; }
.em-x .k-mat-body { transition: background-color .5s ease, box-shadow .5s ease; }
.em-x.em-idle .k-mat-body { background: transparent; box-shadow: none; border-style: dashed; border-color: color-mix(in srgb, var(--line-strong) 85%, transparent); }
.em-x .k-cell.em-ell, .em-wte .k-cell.em-ell { background: none !important; box-shadow: none; color: var(--fg-3); font-size: 13px; }
.em-x .k-cell.em-ell { line-height: ${ROW_H}px; }
.em-wte .k-cell.em-ell { line-height: var(--eh, 14px); }
.em-wte.em-focus .k-cell.em-ell { opacity: 1; }
.em-note .k-card-title { font-size: 19px; }
.em-note .k-card-sub { font-size: 15px; }
.em-eq { width: 340px; display: grid; gap: 8px; }
.em-eq-main { font-size: 22px; }
.em-eq-main .katex-display { margin: 0; }
.em-eq-live { font: 500 15px/1.45 var(--font-mono); color: var(--fg-2); padding: 9px 12px; border-radius: 12px; overflow-x: auto;
  background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); }
.em-read { width: 340px; display: grid; gap: 7px; }
.em-vals { display: flex; flex-wrap: wrap; gap: 4px; }
.em-v { box-sizing: border-box; width: 48px; padding: 5px 0; text-align: center; border-radius: 6px; white-space: nowrap; font: 600 11px/1 var(--font-mono); color: var(--fg);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--fg) 8%, transparent); }
.em-v.hl { box-shadow: inset 0 0 0 2px var(--accent); }
.em-v.gap { width: 16px; background: none; box-shadow: none; color: var(--fg-3); }
@media (prefers-reduced-motion: reduce) {
  .em-tok, .em-stub, .em-wte .k-cell, .em-x .k-cell, .em-wte .k-axis.row { transition: none !important; }
}
`);

// Rows to draw for T tokens: every row up to 12, else the first 8, a gap, and the last 3.
const rowsFor = (T) => (T <= MAX_ROWS
  ? Array.from({ length: T }, (_, t) => ({ t }))
  : [...Array.from({ length: HEAD }, (_, t) => ({ t })), { ell: true, n: T - HEAD - TAIL, from: HEAD, to: T - TAIL - 1 },
    ...Array.from({ length: TAIL }, (_, k) => ({ t: T - TAIL + k }))]);

// Columns to draw for d: all of them up to 16, else the first 12, a gap (-1), and the last 4.
const colsFor = (d) => (d <= MAX_COLS
  ? Array.from({ length: d }, (_, c) => c)
  : [...Array.from({ length: COL_HEAD }, (_, c) => c), -1, ...Array.from({ length: COL_TAIL }, (_, k) => d - COL_TAIL + k)]);

// Cell size for a matrix of n drawn columns: about COL_W units across, so the layout does not grow with d.
function cellGeom(n) {
  const gap = 1, cw = Math.max(4, Math.min(12, Math.floor((COL_W - (n - 1) * gap) / n)));
  return { cw, gap, rowW: n * cw + (n - 1) * gap };
}

// W_E: one row per vocabulary entry, shrunk so the whole table fits the hero.
function wteGeom(V) {
  const pitch = Math.max(8, Math.min(PITCH, Math.floor((HERO_BOTTOM - TOP - 2 * PAD + 2) / V)));
  const gap = pitch >= 12 ? 2 : 1;
  return { pitch, gap, eh: pitch - gap };
}

// The ⋮ row for elided tokens. It names the selected token when that token is hidden in the gap.
const gapText = (r, q) => (q >= r.from && q <= r.to ? `⋮ ${r.n} tokens, incl. ${q}` : `⋮ ${r.n} tokens`);

const sci = (v) => (v === 0 ? '0' : v.toExponential(1));
const absMax = (...arrs) => { let m = 0; for (const a of arrs) if (a) for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m || 1; };
// No position table (rope, alibi, none): the note that takes the place of P.
const NO_POS = {
  rope: ['Position enters inside attention', 'RoPE rotates q and k inside each attention layer.', 'RoPE applies position inside attention, so x₀ is the token vector alone.'],
  alibi: ['Position enters inside attention', 'ALiBi subtracts a distance penalty from each attention score.', 'ALiBi applies position inside attention, so x₀ is the token vector alone.'],
  none: ['No position vector', 'No position signal is added, here or inside attention.', 'No position signal is added anywhere, so x₀ is the token vector alone.'],
};

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const vis = h('div', { class: 'em-vis' });
    st.add(vis, { x: 0, y: 0 });
    const header = sceneHeader(st, {
      title: 'Embedding',
      sub: 'Each token id reads one row of W_E. The position vector for its slot is then added, and the sum is x₀.',
    });
    const cap = sceneCaption(st);
    const eqBox = h('div', { class: 'em-eq' });
    const readBox = h('div', { class: 'em-read' });
    st.add(eqBox, { x: ZONES.side.x, y: TOP });
    st.add(readBox, { x: ZONES.side.x, y: 560 });

    let destroyed = false, cur = 0, run = 0, autoOn = true, autoTimer = null, autoStep = false, animating = false;
    let g = null, hoverWte = null;                    // hoverWte: the W_E entry under the pointer, {r, c}
    const alive = (my) => !destroyed && my === run;
    const put = (node, x, y, anchor = 'tl') => {
      Object.assign(node.style, { position: 'absolute', left: `${x}px`, top: `${y}px`, translate: ANCHOR[anchor] });
      vis.append(node);
      return node;
    };
    const zoomLabel = (text, color, node) => h('div', { class: 'k-label em-zlab', dataset: { node }, style: { textAlign: 'left', marginBottom: '0', color } }, text);

    // W_E is weights, not a trace, so it has no hover card. Its value shows in the readout instead.
    const showRead = () => { if (g) readBox.replaceChildren(...readParts()); };
    function wteOver(e) {
      const cell = e.target.closest?.('[data-wte]');
      if (!cell) { if (hoverWte) { hoverWte = null; showRead(); } return; }
      const [r, c] = cell.dataset.wte.split(',').map(Number);
      if (hoverWte && hoverWte.r === r && hoverWte.c === c) return;
      hoverWte = { r, c };
      showRead();
    }
    function wteOut() { if (hoverWte) { hoverWte = null; showRead(); } }

    // ------------------------------------------------------------- build
    // Everything that depends on the forward pass is built here; a new pass rebuilds it.
    function build() {
      if (g) { g.mats.forEach((m) => m.destroy()); g.arrows.forEach((a) => a.destroy()); }
      g = null; hoverWte = null;
      vis.replaceChildren();
      const s = ctx.state, cfg = s.config, T = s.ids.length, d = cfg.d_model, V = cfg.vocab_size;
      header.el.querySelector('.k-eyebrow')?.replaceChildren(chip(`V = ${V}`, 'embed'), chip(`d = ${d}`, 'embed'), chip(`T = ${T}`, 'embed'));
      if (!T || !s.trace['emb.tok']) {
        put(h('div', { class: 'k-sub', style: { width: '900px', textAlign: 'left' } }, 'Type some text to see its tokens.'), X_IDS, TOP);
        return;
      }
      const enc = cfg.pos_encoding;
      const hasP = (enc === 'learned' || enc === 'sinusoidal') && !!s.trace['emb.pos'];
      header.setSub(hasP
        ? 'Each token id reads one row of W_E. The position vector for its slot is then added, and the sum is x₀.'
        : enc === 'none'
          ? 'Each token id reads one row of W_E, and that row is x₀. No position signal is added anywhere.'
          : 'Each token id reads one row of W_E, and that row is x₀. Position enters inside attention, not here.');

      const ids = s.ids, rows = rowsFor(T), R = rows.length;
      const cols = colsFor(d), n = cols.length;
      const tok = s.trace['emb.tok'].data, out = s.trace['emb.out'].data;
      const pos = hasP ? s.trace['emb.pos'].data : null;
      const W = s.model.params['wte.weight'].data;
      const { cw, gap, rowW } = cellGeom(n);
      const ew = wteGeom(V);
      const RANGE = absMax(W, tok, out, pos);            // one colour scale for the whole scene
      const color = (v) => diverging(v / RANGE);
      const bodyW = rowW + 2 * PAD, bodyH = R * ROW_H + (R - 1) * ROW_GAP + 2 * PAD;
      const wteW = LAB_W + 1 + rowW + 2 * PAD;
      const xT = X_WTE + wteW + 48, xTR = xT + bodyW;           // X_tok: left and right edges
      const yBot = TOP + bodyH, yP = yBot + GAP_V, yOp = yBot + GAP_V / 2;
      const plusC = xTR + 64, xO = plusC + PLUS_D / 2 + 36;     // the plus, then x₀
      const yX0 = yOp - bodyH / 2, yMid = TOP + bodyH / 2;
      const rowY = (top, k) => top + PAD + k * PITCH;

      // checks: the lookup and the sum, over every entry (the trace must reproduce them exactly)
      let dTok = 0, dOut = 0;
      for (let t = 0; t < T; t++) for (let c = 0; c < d; c++) {
        dTok = Math.max(dTok, Math.abs(tok[t * d + c] - W[ids[t] * d + c]));
        dOut = Math.max(dOut, Math.abs(out[t * d + c] - (tok[t * d + c] + (hasP ? pos[t * d + c] : 0))));
      }

      // displayed rows and columns only (elided rows and columns are not drawn, so they carry no value)
      const disp = (src) => {
        const o = new Float64Array(R * n);
        rows.forEach((r, k) => { if (!r.ell) cols.forEach((c, j) => { if (c >= 0) o[k * n + j] = src[r.t * d + c]; }); });
        return o;
      };
      const dispW = new Float64Array(V * n);
      for (let v = 0; v < V; v++) cols.forEach((c, j) => { if (c >= 0) dispW[v * n + j] = W[v * d + c]; });
      const cellOf = (k, j) => (rows[k].ell || cols[j] < 0 ? null : `${rows[k].t},${cols[j]},0`);
      const opts = { cell: cw, gap, mode: 'diverging', range: RANGE };

      // W_E is a table of weights, not a trace: no data-trace, but a zoom into the lookup
      const wte = mat(dispW, V, n, { ...opts, kind: 'embed', rowLabels: Array.from({ length: V }, (_, v) => s.tokenizer.label(v)), tilt: true });
      const xTok = mat(disp(tok), R, n, { ...opts, kind: 'embed', trace: 'emb.tok', cellOf });
      const P = hasP ? mat(disp(pos), R, n, { ...opts, kind: 'pos', trace: 'emb.pos', cellOf }) : null;
      const x0 = mat(disp(out), R, n, { ...opts, kind: 'resid', trace: 'emb.out', cellOf });
      const mats = [wte, xTok, P, x0].filter(Boolean);

      // gaps: the ⋮ row and column carry no value and no hover card
      const glyph = (cell) => { cell.classList.add('em-ell'); cell.removeAttribute('data-cell'); };
      const gapCol = (cell) => { glyph(cell); cell.textContent = '⋮'; };
      [xTok, P, x0].filter(Boolean).forEach((m) => rows.forEach((r, k) => cols.forEach((c, j) => {
        if (r.ell) glyph(m.cellEl(k, j)); else if (c < 0) gapCol(m.cellEl(k, j));
      })));
      // every W_E entry names itself, so the readout can show it on hover
      for (let v = 0; v < V; v++) cols.forEach((c, j) => {
        const cell = wte.cellEl(v, j);
        if (c < 0) gapCol(cell); else cell.dataset.wte = `${v},${c}`;
      });
      wte.classList.add('em-wte', 'em-zoom');
      wte.dataset.node = 'token-embedding';
      wte.style.setProperty('--eh', `${ew.eh}px`);
      wte.style.setProperty('--ep', `${ew.gap}px`);
      wte.addEventListener('pointerover', wteOver);
      wte.addEventListener('pointerleave', wteOut);
      xTok.classList.add('em-x');
      x0.classList.add('em-x');
      if (P) { P.classList.add('em-x', 'em-zoom'); P.dataset.node = 'positional'; }

      put(wte, X_WTE, TOP);
      put(xTok, xT, TOP);
      if (P) put(P, xT, yP);
      put(x0, xO, yX0);

      const label = (text, color2, x, y) => put(h('div', { class: 'k-label', style: { textAlign: 'left', marginBottom: '0', color: color2 } }, text), x, y, 'bl');
      label('token ids', 'var(--fg-2)', X_IDS, TOP - 10);
      put(texLabel('\\WE \\in \\R^{V\\times d}', { size: 17, kind: 'embed' }), X_WTE, TOP - 10, 'bl');
      put(zoomLabel('token lookup', 'var(--c-embed)', 'token-embedding'), X_WTE + wteW, TOP - 10, 'br');
      put(texLabel('X_{\\mathrm{tok}} \\in \\R^{T\\times d}', { size: 17, kind: 'embed' }), xT, TOP - 10, 'bl');
      put(texLabel('\\xres_0 \\in \\R^{T\\times d}', { size: 17, kind: 'resid' }), xO, yX0 - 10, 'bl');
      if (hasP) {
        put(texLabel('\\Pe \\in \\R^{T\\times d}', { size: 17, kind: 'pos' }), xT, yP - 10, 'bl');
        put(zoomLabel('positions', 'var(--c-pos)', 'positional'), X_IDS, yP - 10, 'bl');
      }

      // token tiles (the ids), position bars, and the slots that the flights land in
      const sel = Math.min(s.sel.token, T - 1);
      const rowOf = new Map(), tiles = [], stubs = [], slots = [], pSlots = [], gaps = [];
      rows.forEach((r, k) => {
        if (r.ell) {
          put(h('div', { class: 'em-ellt' }, `⋮ ${r.n} more`), X_IDS, rowY(TOP, k));
          if (hasP) put(h('div', { class: 'em-ellt' }, '⋮'), X_IDS, rowY(yP, k));
          gaps.push({ el: put(h('div', { class: 'em-ellm', style: { width: `${rowW}px` } }, gapText(r, sel)), xT + PAD, rowY(TOP, k)), r });
          if (hasP) put(h('div', { class: 'em-ellm', style: { width: `${rowW}px` } }, `⋮ ${r.n} positions`), xT + PAD, rowY(yP, k));
          put(h('div', { class: 'em-ellm', style: { width: `${rowW}px` } }, `⋮ ${r.n} tokens`), xO + PAD, rowY(yX0, k));
          return;
        }
        rowOf.set(r.t, k);
        const tile = h('button', { type: 'button', class: ['em-tok', r.t === sel && 'sel'], title: `position ${r.t}` },
          h('b', null, ctx.label(r.t)), h('span', null, `id ${ids[r.t]}`));
        tile.addEventListener('click', () => ctx.store.select({ token: r.t }));
        tiles[r.t] = put(tile, X_IDS, rowY(TOP, k));
        if (hasP) {
          stubs[k] = put(h('div', { class: 'em-stub', style: { width: `${STUB_W}px`, height: `${ROW_H}px` } },
            cols.map((c) => h('span', { style: { background: c < 0 ? 'transparent' : color(pos[r.t * d + c]) } }))), X_IDS + TILE_W - STUB_W, rowY(yP, k));
          put(h('div', { class: 'em-stub-lab', style: { width: `${STUB_LAB_W}px` } }, `pos ${r.t}`), X_IDS, rowY(yP, k));
          pSlots[k] = put(h('div', { class: 'em-slot', style: { width: `${rowW}px`, height: `${ROW_H}px` } }), xT + PAD, rowY(yP, k));
        }
        slots[k] = put(h('div', { class: 'em-slot', style: { width: `${rowW}px`, height: `${ROW_H}px` } }), xT + PAD, rowY(TOP, k));
      });

      // arrows and the plus
      const arr = {};
      arr.ids = st.arrow({ x: X_IDS + TILE_W + 4, y: yMid }, { x: X_WTE - 4, y: yMid }, { kind: 'embed', width: 2 });
      arr.wt = st.arrow({ x: X_WTE + wteW + 4, y: yMid }, { x: xT - 4, y: yMid }, { kind: 'embed', width: 2.2 });
      arr.tp = st.arrow({ x: xTR + 4, y: yBot - 14 }, { x: plusC - PLUS_D / 2 - 4, y: yOp - 6 }, { kind: 'embed', width: 2 });
      if (hasP) arr.pp = st.arrow({ x: xTR + 4, y: yP + 14 }, { x: plusC - PLUS_D / 2 - 4, y: yOp + 6 }, { kind: 'pos', width: 2 });
      arr.out = st.arrow({ x: plusC + PLUS_D / 2 + 4, y: yOp }, { x: xO - 4, y: yOp }, { kind: 'resid', width: 2.2 });
      const arrows = Object.values(arr);
      arrows.forEach((a) => { a.el.style.transition = 'opacity .5s ease'; });
      const plus = put(op(hasP ? '\\oplus' : '=', { kind: 'resid', size: PLUS_D, tex: hasP }), plusC, yOp, 'c');

      // no position table: a note takes the place of P, and zooms to the positional scene
      if (!hasP) {
        const [title, sub] = NO_POS[enc] || NO_POS.none;
        put(card({ title, sub, kind: 'pos', node: 'positional', w: bodyW }), xT, yP).classList.add('em-note');
      }

      g = {
        T, R, n, rows, cols, d, ids, hasP, cw, gap, rowW, RANGE, color, W, tok, out, pos, dTok, dOut,
        wte, xTok, P, x0, mats, arrows, arr, plus, tiles, stubs, slots, pSlots, rowOf, gaps,
        gapRow: rows.find((r) => r.ell), sel, dispTok: disp(tok), dispOut: disp(out), ew,
      };
      markSel(cur);
      refreshGaps();
    }

    // the ⋮ rows follow the selection: a selected token that is elided is named in its gap
    function refreshGaps() { if (g) g.gaps.forEach(({ el, r }) => { el.textContent = gapText(r, g.sel); }); }

    // ------------------------------------------------------------- states
    const setRow = (m, k, on) => { for (let j = 0; j < g.n; j++) if (g.cols[j] >= 0) m.cellEl(k, j).classList.toggle('em-pend', !on); };

    // W_E: dim every row but the one being looked up; its label lights up too
    function litRow(id) {
      if (!g) return;
      const { wte, n } = g;
      wte.querySelectorAll('.em-on').forEach((e) => e.classList.remove('em-on'));
      wte.classList.toggle('em-focus', id != null);
      if (id == null) return;
      for (let j = 0; j < n; j++) wte.cellEl(id, j).classList.add('em-on');
      wte.querySelectorAll('.k-axis.row')[id]?.classList.add('em-on');
    }

    // a copy of W_E row `id`, drawn over the row and lifted into the flight
    function liftRow(id) {
      const { wte, n, cols, cw, gap, W, d, color } = g;
      const c0 = wte.cellEl(id, 0), cN = wte.cellEl(id, n - 1);
      const lift = h('div', {
        class: 'em-lift',
        style: {
          left: `${c0.offsetLeft}px`, top: `${c0.offsetTop}px`,
          width: `${cN.offsetLeft + cN.offsetWidth - c0.offsetLeft}px`, height: `${c0.offsetHeight}px`,
          '--cw': `${cw}px`, '--gx': `${gap}px`,
        },
      }, cols.map((c) => h('span', { style: { background: c < 0 ? 'transparent' : color(W[id * d + c]) } })));
      wte.querySelector('.k-mat-body').append(lift);
      return lift;
    }

    // the arrows, the plus and the selected row band: the parts that change with the step
    function applyChrome(i) {
      if (!g) return;
      const { arr, plus, tiles, xTok, P, x0 } = g;
      const a = (on) => (on ? 1 : 0.3);
      arr.ids.setOpacity(a(i >= 1));
      arr.wt.setOpacity(a(i >= 1));
      arr.tp.setOpacity(a(i >= 2));
      if (arr.pp) arr.pp.setOpacity(a(i >= 2));
      arr.out.setOpacity(a(i >= 2));
      plus.style.opacity = i >= 2 ? '1' : '0.35';
      // the empty frames of X_tok, P and x₀ read as placeholders until their values arrive
      xTok.classList.toggle('em-idle', i < 1);
      if (P) P.classList.toggle('em-idle', i < 2);
      x0.classList.toggle('em-idle', i < 2);
      tiles.forEach((t) => t?.classList.remove('lit'));
      markSel(i);
    }

    // every step's end state, for jumps, back-steps and the non-animated path
    function applyStatic(i) {
      if (!g) return;
      const { rows, xTok, P, x0, stubs, dispOut, RANGE } = g;
      applyChrome(i);
      rows.forEach((r, k) => {
        if (r.ell) return;
        setRow(xTok, k, i >= 1);
        if (P) setRow(P, k, i >= 2);
        setRow(x0, k, i >= 2);
      });
      x0.update(dispOut, { range: RANGE });
      stubs.forEach((b) => { if (b) b.style.opacity = i >= 2 ? '0' : '1'; });
      litRow(i >= 1 ? g.ids[g.sel] : null);
    }

    // the selected token: its tile, its band in X_tok (from step 1) and x₀ (from step 2)
    function markSel(i = cur) {
      if (!g) return;
      const k = g.rowOf.get(g.sel) ?? -1;
      g.tiles.forEach((t, idx) => t?.classList.toggle('sel', idx === g.sel));
      g.xTok.highlightRow(i >= 1 ? k : -1);
      g.x0.highlightRow(i >= 2 ? k : -1);
    }

    // ---------------------------------------------------------- animations
    // Step 1 (lookup): for each token, its tile and W_E row light up; a copy of the row flies
    // into its row of X_tok, which then fills in. Elided tokens are passed in one pause.
    async function animateLookup(my) {
      const { T, ids, rowOf, tiles, slots, xTok, wte } = g;
      let paused = false;
      for (let t = 0; t < T; t++) {
        if (!alive(my)) return;
        const k = rowOf.get(t);
        if (k === undefined) {
          if (!paused) { litRow(null); paused = true; await wait(260); }
          continue;
        }
        paused = false;
        const id = ids[t], tile = tiles[t];
        litRow(id);
        tile.classList.add('lit');
        const from = st.center(tile), to = st.center(wte.querySelectorAll('.k-axis.row')[id]);
        const link = st.arrow({ x: from.x + from.w / 2 + 4, y: from.y }, { x: to.x - to.w / 2 - 3, y: to.y },
          { kind: 'embed', width: 1.6, dash: true, opacity: 0.9 });
        setTimeout(() => link.destroy(), 520);
        const lift = liftRow(id);
        fly(st, lift, slots[k], { duration: 640, arc: 56 }).then(() => { if (alive(my)) setRow(xTok, k, true); });
        lift.remove();
        await wait(260);
        tile.classList.remove('lit');
      }
      if (!alive(my)) return;
      litRow(ids[g.sel]);
      await wait(520);
      // a flight that has not reported back cannot leave its row blank
      if (alive(my)) g.rows.forEach((r, k) => { if (!r.ell) setRow(xTok, k, true); });
    }

    // Step 2 (positions): each position bar flies into its row of P; then the plus merges X_tok
    // and P into x₀: the cells start at the X_tok colours and move to the sums.
    async function animatePositions(my) {
      const { rows, P, x0, stubs, pSlots, dispTok, dispOut, RANGE, plus } = g;
      if (P) {
        for (let k = 0; k < rows.length; k++) {
          if (!alive(my)) return;
          const bar = stubs[k];
          if (!bar) continue;
          fly(st, bar, pSlots[k], { duration: 700, arc: 40 }).then(() => { if (alive(my)) setRow(P, k, true); });
          bar.style.opacity = '0';                  // after fly(), which clones the bar as it is
          await wait(150);
        }
        await wait(420);
      }
      if (!alive(my)) return;
      pulse(plus);
      x0.update(dispTok, { range: RANGE });
      rows.forEach((r, k) => { if (!r.ell) setRow(x0, k, true); });
      await wait(560);
      if (!alive(my)) return;
      x0.update(dispOut, { range: RANGE });
      await wait(300);
      if (alive(my) && P) rows.forEach((r, k) => { if (!r.ell) setRow(P, k, true); });
    }

    async function animate(i, my) {
      animating = true;
      if (i === 1) await animateLookup(my);
      else if (i === 2) await animatePositions(my);
      else if (i === 3) { if (g) pulse(g.x0); await wait(500); }
      if (alive(my)) animating = false;
    }

    // ----------------------------------------------------------- auto-advance
    // First view: a step every MOTION.step, waiting for any animation to finish first.
    function stopAuto() { autoOn = false; clearTimeout(autoTimer); }
    function armAuto() {
      clearTimeout(autoTimer);
      if (!autoOn || cur >= STEPS.length - 1) return;
      const tick = () => {
        if (destroyed || !autoOn) return;
        if (animating) { autoTimer = setTimeout(tick, 200); return; }
        autoStep = true;
        ctx.setStep(cur + 1);
        autoStep = false;
      };
      autoTimer = setTimeout(tick, MOTION.step);
    }

    // ------------------------------------------------------------- text
    function capText(i) {
      const cfg = ctx.state.config, enc = cfg.pos_encoding, { hasP } = g;
      const posLine = hasP
        ? 'Each position vector joins the token at that position, so the same character gets a different x₀ at every position.'
        : (NO_POS[enc] || NO_POS.none)[2];
      return [
        `Each character is one token. Its id, from 0 to ${cfg.vocab_size - 1}, is its index in the vocabulary and names one row of W_E.`,
        'Each id copies its row of W_E into that token’s row of X_tok. A one-hot vector times W_E gives the same row, so the lookup is a matrix product.',
        posLine,
        `${hasP ? 'x₀ = X_tok + P' : 'x₀ = X_tok'} is the block input for every token. ${cfg.residual
          ? 'The residual stream starts here: each attention and MLP output is added onto it, layer after layer.'
          : 'Without residual connections, x₀ feeds block 0 only, because each sub-layer replaces the stream.'}`,
      ][i];
    }

    function eqParts(i) {
      const s = ctx.state, { d, ids, hasP, W, tok, out, pos, dTok, dOut, T, sel: q } = g;
      const id = ids[q];
      const name = i === 2 && !hasP ? 'No position table' : STEPS[i];
      const parts = [h('div', { class: 'k-label', style: { textAlign: 'left', marginBottom: '0', color: i === 3 ? 'var(--c-resid)' : 'var(--c-embed)' } },
        `${i + 1} · ${name}`)];
      const tex = (t) => h('div', { class: 'em-eq-main', html: texHTML(t, { display: true }) });
      const live = (t) => h('div', { class: 'em-eq-live', html: texHTML(t) });
      const note = (text) => h('div', { class: 'k-caption', style: { fontSize: '14px' } }, text);
      // the engine's own live maths for this entry (the maths drawer shows the same), with a direct fallback
      const viaEngine = i === 0 ? null : runLive(s, i >= 2 ? 'emb.out' : 'emb.tok', { r: q, band: 0, x: 0 });
      const lv = viaEngine && viaEngine.result !== undefined ? viaEngine
        : (i >= 2 ? { result: tok[q * d] + (hasP ? pos[q * d] : 0), expected: out[q * d] } : { result: W[id * d], expected: tok[q * d] });
      if (i === 0) {
        parts.push(tex(mathFor('emb.tok').compact),
          h('div', { class: 'em-eq-live' }, `“${ctx.label(q)}” at position ${q} is id ${id}, so it reads row ${id} of W_E.`));
      } else if (i === 1) {
        parts.push(tex(mathFor('concept:one-hot-lookup').compact),
          live(`X_{\\mathrm{tok},${q},0} = \\WE_{${id},0} = ${num(lv.result, 4)}`),
          checkBadge(lv.result, lv.expected, { label: 'equals the traced value' }),
          note(`all ${T * d} entries of X_tok: max |Δ| = ${sci(dTok)}`));
      } else if (i === 2) {
        const p0 = hasP ? pos[q * d] : 0;
        parts.push(tex(hasP ? '\\xres_0 = X_{\\mathrm{tok}} + \\Pe' : '\\xres_0 = X_{\\mathrm{tok}}'),
          live(hasP
            ? `\\xres_{0,${q},0} = ${num(tok[q * d], 4)} ${p0 < 0 ? '-' : '+'} ${num(Math.abs(p0), 4)} = ${num(lv.result, 4)}`
            : `\\xres_{0,${q},0} = ${num(tok[q * d], 4)}`),
          checkBadge(lv.result, lv.expected, { label: 'equals the traced value' }),
          note(hasP
            ? (s.config.pos_encoding === 'learned' ? 'P is a learned table, one row per position.' : 'P is a fixed sine and cosine table.')
            : 'No positional table: x₀ is the token vector.'));
      } else {
        parts.push(tex(hasP ? '\\xres_{0,t,c} = X_{\\mathrm{tok},t,c} + \\Pe_{t,c}' : '\\xres_{0,t,c} = X_{\\mathrm{tok},t,c}'),
          checkBadge(dOut, 0, { label: `all ${T * d} entries of x₀ = ${hasP ? 'X_tok + P' : 'X_tok'}` }),
          note(`max |Δ| over the traced values = ${sci(dOut)}`));
      }
      return parts;
    }

    // the readout: the selected token's W_E row, or the row of the entry under the pointer
    function readParts() {
      const { d, n, cols, ids, W, color, sel: q, gapRow } = g;
      const hv = hoverWte;
      const r = hv ? hv.r : ids[q];
      const parts = [];
      if (ctx.state.config.tie_weights) parts.push(chip('tied: this matrix is also the output head', 'head'));
      parts.push(h('div', { class: 'k-label', style: { textAlign: 'left', marginBottom: '0', color: 'var(--c-embed)' } },
        hv ? `W_E[${hv.r}, ${hv.c}] · “${ctx.state.tokenizer.label(hv.r)}”` : `row ${r} of W_E · “${ctx.label(q)}”`));
      parts.push(h('div', { class: 'em-vals' }, cols.map((c) => (c < 0
        ? h('span', { class: 'em-v gap' }, '⋮')
        : h('span', { class: ['em-v', hv && c === hv.c && 'hl'], style: { background: color(W[r * d + c]) } }, fmt(W[r * d + c], 2))))));
      parts.push(h('div', { class: 'k-caption', style: { fontSize: '15px' } }, hv
        ? `column ${hv.c} of ${d}`
        : `${d > MAX_COLS ? 'first 12 … last 4' : 'all'} of ${d} values · position ${q}`));
      if (!hv && gapRow && q >= gapRow.from && q <= gapRow.to) {
        parts.push(h('div', { class: 'k-caption', style: { fontSize: '14px' } }, `Position ${q} is in the gap, so its row is not drawn in X_tok.`));
      }
      parts.push(h('div', { class: 'k-caption', style: { fontSize: '14px' } }, 'Hover an entry of W_E to read it. Click a token to select it.'));
      return parts;
    }

    function showCards(i) {
      if (!g) { eqBox.replaceChildren(); readBox.replaceChildren(); cap.set(''); return; }
      eqBox.replaceChildren(...eqParts(i));
      cap.set(capText(i));
      readBox.replaceChildren(...readParts());
    }

    // ------------------------------------------------------------- steps
    function setStep(i) {
      if (!autoStep) stopAuto();
      const prev = cur;
      const forward = i === prev + 1 && !reducedMotion() && !!g;
      if (animating) applyStatic(prev);      // an interrupted step is completed at once, then the new one runs
      cur = i;
      const my = ++run;
      animating = false;
      if (forward) { applyChrome(i); animate(i, my); } else applyStatic(i);
      showCards(i);
      armAuto();
    }

    function onSelect() {
      if (!g) return;
      g.sel = Math.min(ctx.state.sel.token, g.T - 1);
      markSel(cur);
      if (!animating) litRow(cur >= 1 ? g.ids[g.sel] : null);
      refreshGaps();
      showCards(cur);
    }

    function rebuild() {
      run++;
      animating = false;
      build();
      applyStatic(cur);
      showCards(cur);
    }

    // ---------------------------------------------------------------- mount
    ctx.setSteps(STEPS, setStep);
    rebuild();
    appear([header.el, vis, eqBox, readBox, cap.el], { stagger: MOTION.stagger });
    el.addEventListener('pointerdown', stopAuto, { once: true });
    const offs = [ctx.on('trace', () => { stopAuto(); rebuild(); }), ctx.on('select', onSelect)];
    const themeOff = onThemeChange(() => setTimeout(() => { if (!destroyed) rebuild(); }, 0));
    armAuto();

    return {
      pause() { run++; animating = false; stopAuto(); },
      resume() { if (!destroyed) { applyStatic(cur); showCards(cur); } },
      destroy() {
        destroyed = true;
        run++;
        clearTimeout(autoTimer);
        offs.forEach((f) => f());
        themeOff();
        g?.mats.forEach((m) => m.destroy());
        st.destroy();
      },
    };
  },
};

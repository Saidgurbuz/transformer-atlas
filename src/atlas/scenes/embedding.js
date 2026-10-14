// Embedding: how one token id becomes the block input x₀, followed for the selected token.
// Focus + context (docs/STYLE_GUIDE.md, template 2). The hero holds that token's row at each stage,
// top to bottom: W_E[id], X_tok[t], P[t] and x₀[t], with the sum drawn as ⊕ and =. Each step lights
// one more stage and dims the rest; the accent marks the row the step is about. The side column holds
// the step's equation and check, then one small context table (W_E, P or x₀) with the selected row
// marked. Long sequences show the first 8 rows, the selected token with its neighbours, and the last 3.
// Without a position table (RoPE, ALiBi, none) there is no P: x₀ = X_tok, and a note in the P slot
// says where position enters instead.
// Reference scene for contributors: see docs/ATLAS_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, vec, op, checkBadge, chip, appear, fly, pulse, wait, useStyle, ZONES, MOTION } from '../kit.js';
import { h, fmt, num, texHTML, diverging, onThemeChange, reducedMotion } from '../../ui/index.js';
import { mathFor, runLive } from '../../core/math/index.js';

const STEPS = ['Token ids', 'Look up rows', 'Positions', 'x₀'];
const CELL = 42, CELL_GAP = 4, ROW_H = 60;          // hero cells are 42 wide, 4 apart; a stage row is 60 tall
const HEAD_COLS = 12, TAIL_COLS = 4;                // columns: all of d up to 17, else the first 12, ⋮, the last 4
const CTX_ALL = 36, CTX_HEAD = 8, CTX_TAIL = 3;     // context rows: every row up to 36, else a window
const CTX_IDS = 6;                                  // W_E rows for a vocabulary above CTX_ALL: this text's rows nearest the selected one
const CTX_W = 340, CTX_LAB = 34, CTX_CELL = 16;     // the side column is 340 wide
const LABW_MAX = 120, LABEL_CHAR = 9.2;             // a context label column fits its longest label (up to LABW_MAX), 15 px mono
const CTX_Y = 520, CTX_ROWS_Y = 550, CTX_BOTTOM = 760, CTX_AVAIL = CTX_BOTTOM - CTX_ROWS_Y, GAP_ROW = 16;
const LABEL_PX = 15;                                // the label size: a context label every k rows keeps this much room
const SIDE_Y = 260;
const LAB_R = 240;                                  // stage labels end here, right-aligned
const OP_X = 280, OP_D = 40;                        // the operator column: ⊕ and =
const STRIP_X = 320;                                // the hero strips start here
const Y_IDX = 270, Y_A = 290, Y_B = 430, Y_C = 550, Y_D = 660;   // the hero starts at ZONES.hero.y; rows: W_E[id], X_tok[t], P[t], x₀[t]
const ANCHOR = { tl: '0 0', c: '-50% -50%', lc: '0 -50%', rc: '-100% -50%' };
const HINT = 'Hover to read a value.';
// No position table (rope, alibi, none): the note that takes the place of P, with its caption sentence.
const NO_POS = {
  rope: ['Position enters inside attention', 'RoPE rotates q and k inside each attention layer.', 'RoPE applies position inside attention, so x₀ is the token vector alone.'],
  alibi: ['Position enters inside attention', 'ALiBi subtracts a distance penalty from each attention score.', 'ALiBi applies position inside attention, so x₀ is the token vector alone.'],
  none: ['No position vector', 'No position signal is added, here or inside attention.', 'No position signal is added anywhere, so x₀ is the token vector alone.'],
};

useStyle('embedding', `
.em-idx { position: absolute; display: flex; gap: ${CELL_GAP}px; padding: 0 6px; font: 500 13px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.em-idx > span { width: ${CELL}px; text-align: center; }
.em-strip { transition: opacity .45s ease; }
.em-strip .k-cell { height: var(--em-ch, ${CELL}px); }
.em-lab { position: absolute; text-align: right; white-space: nowrap; transition: opacity .45s ease; }
.em-lab-main { font-size: 18px; line-height: 1.2; }
.em-lab-main .katex { font-size: 1em; }
.em-lab-sub { font: 500 15px/1.3 var(--font-mono); color: var(--fg-2); }
.em-lab[data-node] { cursor: zoom-in; }
.em-zlab { position: absolute; font: 600 15px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; cursor: zoom-in; transition: opacity .45s ease; }
.em-dim { opacity: .22; }
.em-focus .k-cells { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 24%, transparent), 0 6px 16px rgba(0, 0, 0, .12); }
.em-op { transition: opacity .45s ease; }
.em-note { position: absolute; box-sizing: border-box; height: ${ROW_H}px; display: grid; align-content: center; gap: 3px; padding: 8px 18px;
  border-radius: 12px; background: color-mix(in srgb, var(--surface) 86%, transparent); border: 1px solid color-mix(in srgb, var(--c-pos) 45%, var(--line));
  box-shadow: 0 10px 26px rgba(0, 0, 0, .12); cursor: zoom-in; transition: opacity .45s ease, box-shadow .25s ease; }
.em-note:hover { box-shadow: 0 0 0 2px color-mix(in srgb, var(--c-pos) 70%, transparent), 0 16px 36px rgba(0, 0, 0, .18); }
.em-note-t { font: 650 17px/1.2 var(--font-display); color: var(--fg); }
.em-note-s { font-size: 15px; line-height: 1.35; color: var(--fg-2); }
.em-side { width: ${CTX_W}px; display: grid; gap: 10px; align-content: start; }
.em-steplab { font: 600 17px/1.2 var(--font-mono); color: var(--fg-2); }
.em-eq-main { font-size: 22px; min-height: 40px; }
.em-eq-main .katex-display { margin: 0; }
.em-eq-live { font-size: 18px; padding: 9px 12px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
.em-eq-live.plain { font: 500 16px/1.45 var(--font-body); color: var(--fg); }
.em-eq-live .katex-display { margin: 0; }
.em-check { min-height: 28px; display: flex; align-items: center; }
.em-note-txt { font-size: 15px; line-height: 1.4; color: var(--fg-2); }
.em-ctxbox { width: ${CTX_W}px; display: grid; gap: 6px; }
.em-ctx-tag { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; height: 22px; font: 500 14px/1.3 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.em-ctx-read { color: var(--fg-2); }
.em-ctx { padding-top: 4px; }
.em-ctx-row { display: flex; align-items: center; height: var(--pc); }
.em-ctx-lab { flex: none; width: var(--lw, ${CTX_LAB}px); padding-right: 6px; text-align: right; font: 500 ${LABEL_PX}px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.em-ctx-cells { display: flex; gap: 1px; }
.em-ctx-c { flex: none; width: var(--cw); height: calc(var(--pc) - 1px); border-radius: 2px; }
.em-ctx-c.ell { display: grid; place-items: center; background: none !important; font: 500 12px/1 var(--font-mono); color: var(--fg-3); }
.em-ctx-row.on .em-ctx-lab { color: var(--fg); font-weight: 700; }
.em-ctx-row.on .em-ctx-cells { outline: 2px solid var(--kind); outline-offset: 2px; border-radius: 3px; }
.em-ctx-gap { display: flex; align-items: center; height: ${GAP_ROW}px; }
.em-ctx-gap .em-ctx-lab { color: var(--fg-3); }
@media (prefers-reduced-motion: reduce) {
  .em-strip, .em-lab, .em-op, .em-note, .em-zlab, .em-focus .k-cells, .em-strip .k-cell { transition: none !important; }
}
`);

// Columns drawn for d: the same rule as vec(): all of them up to 17, else the first 12, a gap (-1), the last 4.
const colsFor = (d) => (d <= HEAD_COLS + TAIL_COLS + 1
  ? Array.from({ length: d }, (_, c) => c)
  : [...Array.from({ length: HEAD_COLS }, (_, c) => c), -1, ...Array.from({ length: TAIL_COLS }, (_, k) => d - TAIL_COLS + k)]);

// Rows of a context table: every row up to CTX_ALL. Beyond that: the first CTX_HEAD, the anchor token with one
// neighbour on each side, and the last CTX_TAIL. A gap row stands for the tokens skipped between them.
function ctxRows(n, anchor) {
  if (n <= CTX_ALL) return Array.from({ length: n }, (_, t) => ({ t }));
  const keep = new Set();
  for (let t = 0; t < CTX_HEAD; t++) keep.add(t);
  for (let t = anchor - 1; t <= anchor + 1; t++) if (t >= 0 && t < n) keep.add(t);
  for (let t = n - CTX_TAIL; t < n; t++) keep.add(t);
  const rows = [];
  let prev = -1;
  for (const t of [...keep].sort((a, b) => a - b)) {
    if (t - prev > 1) rows.push({ ell: true, n: t - prev - 1 });
    rows.push({ t });
    prev = t;
  }
  return rows;
}

// A vocabulary too large to list (W_E with V > CTX_ALL): the rows of this text nearest the selected position, up to
// CTX_IDS distinct ids, in id order. A gap row counts the vocabulary entries between two rows, and the ends too.
function idRows(V, ids, q, n = CTX_IDS) {
  const seen = [];
  for (let k = 0; seen.length < n && (q - k >= 0 || q + k < ids.length); k++) {
    for (const t of k === 0 ? [q] : [q - k, q + k]) {
      if (t >= 0 && t < ids.length && seen.length < n && !seen.includes(ids[t])) seen.push(ids[t]);
    }
  }
  const rows = [];
  let prev = -1;
  for (const id of seen.sort((a, b) => a - b)) {
    if (id - prev > 1) rows.push({ ell: true, n: id - prev - 1 });
    rows.push({ t: id });
    prev = id;
  }
  if (V - 1 - prev > 0) rows.push({ ell: true, n: V - 1 - prev });
  return rows;
}
const fmtN = (n) => Math.round(n).toLocaleString('en-US');

const absMax = (...arrs) => { let m = 0; for (const a of arrs) if (a) for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m || 1; };
const sci = (v) => (v === 0 ? '0' : v.toExponential(1));

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const header = sceneHeader(st, {
      title: 'Embedding',
      sub: 'Each token id reads one row of W_E, and those rows form the block input x₀.',
    });
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'embed' });
    const hero = h('div', { class: 'em-hero' });
    st.add(hero, { x: 0, y: 0 });
    const stepLab = h('div', { class: 'em-steplab' });
    const eqMain = h('div', { class: 'em-eq-main' });
    const eqLive = h('div', { class: 'em-eq-live' });
    const eqCheck = h('div', { class: 'em-check' });
    const eqNote = h('div', { class: 'em-note-txt' });
    const side = h('div', { class: 'em-side' }, stepLab, eqMain, eqLive, eqCheck, eqNote);
    st.add(side, { x: ZONES.side.x, y: SIDE_Y });
    const ctxBox = h('div', { class: 'em-ctxbox' });
    st.add(ctxBox, { x: ZONES.side.x, y: CTX_Y });
    const cap = sceneCaption(st);

    let destroyed = false, cur = 0, run = 0, animating = false, autoOn = true, autoTimer = null, autoStep = false;
    let g = null, readEl = null;                    // g: the built geometry and data for the current pass
    const alive = (my) => !destroyed && my === run;
    const lab = (v) => ctx.state.tokenizer.label(v);                                     // vocabulary entry v
    const name = (v) => { const L = lab(v); return L === '␣' || L === ' ' ? 'space' : L; };
    const tokName = (t) => { const L = ctx.label(t); return L === '␣' || L === ' ' ? 'space' : L; };   // token at position t
    const put = (node, x, y, anchor = 'tl') => {
      Object.assign(node.style, { position: 'absolute', left: `${x}px`, top: `${y}px`, translate: ANCHOR[anchor] });
      hero.append(node);
      return node;
    };

    // ------------------------------------------------------------- build (per forward pass and selection)
    function build() {
      g?.arrow?.destroy();
      g = null;
      hero.replaceChildren();
      const s = ctx.state, cfg = s.config, T = s.ids.length, d = cfg.d_model, V = cfg.vocab_size;
      header.el.querySelector('.k-eyebrow')?.replaceChildren(chip(`V = ${V}`, 'embed'), chip(`d = ${d}`, 'embed'), chip(`T = ${T}`, 'embed'));
      if (!T || !s.trace['emb.tok'] || !s.trace['emb.out']) {
        put(h('div', { class: 'k-sub', style: { width: '900px', textAlign: 'left' } }, 'Type some text to see its tokens.'), 110, 300);
        ctxBox.replaceChildren();
        stepLab.textContent = ''; eqMain.innerHTML = ''; eqLive.innerHTML = ''; eqCheck.replaceChildren(); eqNote.textContent = '';
        cap.set('');
        return;
      }
      const enc = cfg.pos_encoding;
      const hasP = (enc === 'learned' || enc === 'sinusoidal') && !!s.trace['emb.pos'];
      header.setSub(hasP
        ? 'Each token id reads one row of W_E. The position vector for its slot is then added, and the sum is x₀.'
        : enc === 'none'
          ? 'Each token id reads one row of W_E, and that row is x₀. No position signal is added anywhere.'
          : 'Each token id reads one row of W_E, and that row is x₀. Position enters inside attention, not here.');

      const tok = s.trace['emb.tok'].data, out = s.trace['emb.out'].data;
      const pos = hasP ? s.trace['emb.pos'].data : null;
      const W = s.model.params['wte.weight'].data, ids = s.ids;
      const q = Math.min(Math.max(0, s.sel.token ?? T - 1), T - 1);
      const RANGE = absMax(W, tok, out, pos);            // one colour scale for the whole scene
      const cols = colsFor(d), n = cols.length, stripW = n * CELL + (n - 1) * CELL_GAP + 12;
      // checks: the lookup and the sum, over every entry (the trace must reproduce them exactly)
      let dTok = 0, dOut = 0;
      for (let t = 0; t < T; t++) for (let c = 0; c < d; c++) {
        dTok = Math.max(dTok, Math.abs(tok[t * d + c] - W[ids[t] * d + c]));
        dOut = Math.max(dOut, Math.abs(out[t * d + c] - (tok[t * d + c] + (hasP ? pos[t * d + c] : 0))));
      }
      g = { T, V, d, ids, q, hasP, enc, tok, out, pos, W, RANGE, cols, n, stripW, dTok, dOut, arrow: null };
      buildHero();
      applyStatic(cur);
    }

    function buildHero() {
      const { d, cols, stripW, q, ids, RANGE, W, tok, out, pos, hasP, enc } = g;
      const id = ids[q];
      put(h('div', { class: 'em-idx', style: { width: `${stripW}px` } }, cols.map((c) => h('span', null, c < 0 ? '⋮' : String(c)))), STRIP_X, Y_IDX);
      // one row of d values as a strip; a row from the trace carries data-trace and data-cell, so it hovers into the maths
      const strip = (vals, kind, trace, y) => {
        const el = vec(Array.from(vals), {
          orientation: 'h', cell: CELL, gap: CELL_GAP, mode: 'diverging', range: RANGE, kind,
          head: HEAD_COLS, tail: TAIL_COLS, trace, cellOf: trace ? (c) => `${q},${c},0` : undefined,
        });
        el.classList.add('em-strip');
        el.style.setProperty('--em-ch', `${ROW_H - 10}px`);     // the cells fill the row, inside the tray's padding
        return put(el, STRIP_X, y);
      };
      // a stage: its strip, and its label on the left (main line in the stage colour, sub line in text colour)
      const stage = (y, vals, kind, trace, main, sub, zoom) => {
        const el = strip(vals, kind, trace, y);
        const label = put(h('div', { class: 'em-lab', dataset: zoom ? { node: zoom } : undefined },
          h('div', { class: 'em-lab-main', style: { color: `var(--c-${kind})` }, html: texHTML(main) }),
          h('div', { class: 'em-lab-sub' }, sub)), LAB_R, y + ROW_H / 2, 'rc');
        return { el, lab: label, y };
      };
      g.A = stage(Y_A, W.subarray(id * d, id * d + d), 'embed', null, `\\WE[${id}]`, `“${tokName(q)}” · id ${id}`);
      g.B = stage(Y_B, tok.subarray(q * d, q * d + d), 'embed', 'emb.tok', `X_{\\mathrm{tok},${q}}`, `position ${q}`);
      if (hasP) {
        g.C = stage(Y_C, pos.subarray(q * d, q * d + d), 'pos', 'emb.pos', `\\Pe_{${q}}`, 'positions', 'positional');
      } else {
        const [title, sub] = NO_POS[enc] || NO_POS.none;
        const note = h('div', { class: 'em-note', dataset: { node: 'positional' }, style: { width: `${stripW}px` } },
          h('div', { class: 'em-note-t' }, title), h('div', { class: 'em-note-s' }, sub));
        const label = put(h('div', { class: 'em-lab' },
          h('div', { class: 'em-lab-main', style: { color: 'var(--fg-2)' }, html: texHTML(`\\Pe_{${q}}`) }),
          h('div', { class: 'em-lab-sub' }, 'no table')), LAB_R, Y_C + ROW_H / 2, 'rc');
        g.C = { el: put(note, STRIP_X, Y_C), lab: label, y: Y_C };
      }
      g.D = stage(Y_D, out.subarray(q * d, q * d + d), 'resid', 'emb.out', `\\xres_{0,${q}}`, `position ${q}`);
      // the lookup arrow from the W_E row down to the X_tok row, its label beside it (a zoom into the lookup)
      g.arrow = st.arrow({ x: OP_X, y: Y_A + ROW_H + 6 }, { x: OP_X, y: Y_B - 6 }, { kind: 'embed', width: 2.2 });
      g.arrow.el.style.transition = 'opacity .45s ease';
      g.look = put(h('div', { class: 'em-zlab', dataset: { node: 'token-embedding' } }, 'token lookup'), STRIP_X, (Y_A + ROW_H + Y_B) / 2, 'lc');
      g.plus = hasP ? put(op('\\oplus', { kind: 'resid', size: OP_D }), OP_X, (Y_B + ROW_H + Y_C) / 2, 'c') : null;
      g.eqOp = put(op('=', { kind: 'resid', size: OP_D }), OP_X, (Y_C + ROW_H + Y_D) / 2, 'c');
      [g.plus, g.eqOp].forEach((node) => node?.classList.add('em-op'));
    }

    // ------------------------------------------------------------- states
    function setStage(s, on) { s.el.classList.toggle('em-dim', !on); s.lab.classList.toggle('em-dim', !on); }

    // The end state of step i: which stages are lit, where the accent is, the operators, the side and the context.
    function applyStatic(i) {
      if (!g) return;
      const { A, B, C, D, plus, eqOp, arrow, look, hasP } = g;
      setStage(A, true);
      setStage(B, i >= 1);
      if (C) setStage(C, i >= 2);
      setStage(D, i >= 3);
      const focus = i === 0 ? A : i === 1 ? B : i === 2 ? (hasP ? C : null) : D;   // one accent per step
      [A, B, C, D].forEach((s) => s?.el.classList.toggle('em-focus', s === focus));
      look.classList.toggle('em-dim', i < 1);
      arrow.setOpacity(i >= 1 ? 1 : 0.25);
      if (plus) plus.style.opacity = i >= 2 ? '1' : '0.25';
      eqOp.style.opacity = i >= 3 ? '1' : '0.25';
      renderCtx(i);
      renderSide(i);
    }

    // The side context for step i: the table the step reads (W_E, P or x₀), with the selected row marked.
    function tableFor(i) {
      const { T, V, d, q, ids, hasP, W, pos, out } = g;
      if (i <= 1) return {
        kind: 'embed', n: V, anchor: ids[q], hi: i === 1 ? ids[q] : null, wte: true, word: 'id',
        rows: V > CTX_ALL ? idRows(V, ids, q) : null,            // a large vocabulary shows the rows of this text
        value: (r, c) => W[r * d + c], labelOf: (r) => lab(r), tex: '\\WE', shape: `${V} × ${d}`,
      };
      if (i === 2 && hasP) return {
        kind: 'pos', n: T, anchor: q, hi: q, word: 'position', ref: true, trace: 'emb.pos',
        value: (t, c) => pos[t * d + c], labelOf: (t) => ctx.label(t), tex: '\\Pe', shape: `${T} × ${d}`,
      };
      if (i === 2) return null;
      return {
        kind: 'resid', n: T, anchor: q, hi: q, word: 'position', ref: true, trace: 'emb.out',
        value: (t, c) => out[t * d + c], labelOf: (t) => ctx.label(t),
        tex: hasP ? '\\xres_0' : '\\xres_0 = X_{\\mathrm{tok}}', shape: `${T} × ${d}`,
      };
    }

    function renderCtx(i) {
      ctxBox.replaceChildren();
      readEl = null;
      const tab = g && tableFor(i);
      if (!tab) return;
      const { cols, RANGE } = g;
      const rows = tab.rows ?? ctxRows(tab.n, tab.anchor);
      const gaps = rows.filter((r) => r.ell).length, toks = rows.length - gaps;
      // a short table (the rows of a large vocabulary) has room for a taller row, so every label fits its own row
      const pitch = Math.max(5, Math.min(tab.rows ? 16 : 14, Math.floor((CTX_AVAIL - gaps * GAP_ROW) / Math.max(1, toks))));
      const k = Math.max(1, Math.ceil(LABEL_PX / pitch));    // a label every k-th row keeps one label height between labels
      const hiRow = tab.hi == null ? -1 : rows.findIndex((r) => r.t === tab.hi);
      // the label column is as wide as its longest label (a subword such as “␣internationalization” is many characters)
      const longest = Math.max(0, ...rows.filter((r) => !r.ell).map((r) => [...tab.labelOf(r.t)].length));
      const labW = Math.min(LABW_MAX, Math.max(CTX_LAB, Math.ceil(longest * LABEL_CHAR) + 10));
      const cw = Math.min(CTX_CELL, Math.floor((CTX_W - labW - 6 - (cols.length - 1)) / cols.length));
      const body = h('div', { class: 'em-ctx', style: { '--kind': `var(--c-${tab.kind})`, '--pc': `${pitch}px`, '--cw': `${cw}px`, '--lw': `${labW}px` } });
      if (tab.trace) body.dataset.trace = tab.trace;
      rows.forEach((r, ri) => {
        if (r.ell) {
          const continuation = `${fmtN(r.n)} more ${tab.wte ? 'vocabulary' : 'sequence'} ${r.n === 1 ? 'row' : 'rows'}`;
          body.append(h('div', { class: 'em-ctx-gap', role: 'img', 'aria-label': continuation, title: continuation },
            h('span', { class: 'em-ctx-lab', 'aria-hidden': 'true' }, '⋮'),
            h('div', { class: 'em-ctx-cells', 'aria-hidden': 'true' }, cols.map((c) => h('span', { class: 'em-ctx-c ell' }, c < 0 ? '⋱' : '⋮')))));
          return;
        }
        const on = ri === hiRow;
        const shown = on || (ri % k === 0 && (hiRow < 0 || Math.abs(ri - hiRow) >= k));
        body.append(h('div', { class: ['em-ctx-row', on && 'on'] },
          h('span', { class: 'em-ctx-lab', title: `${tab.word} ${r.t}${tab.wte ? `: ${name(r.t)}` : ''}` }, shown ? tab.labelOf(r.t) : ''),
          h('div', { class: 'em-ctx-cells' }, cols.map((c) => {
            if (c < 0) return h('span', { class: 'em-ctx-c ell' }, '⋮');
            const cell = h('span', { class: 'em-ctx-c', style: { background: diverging(tab.value(r.t, c) / RANGE) } });
            if (tab.ref) cell.dataset.cell = `${r.t},${c},0`;
            if (tab.wte) cell.dataset.wte = `${r.t},${c}`;
            return cell;
          }))));
      });
      readEl = h('span', { class: 'em-ctx-read' }, tab.wte && i === 1 ? HINT : '');
      ctxBox.append(h('div', { class: 'em-ctx-tag' }, h('span', { html: `${texHTML(tab.tex)} · ${tab.shape}` }), readEl), body);
    }

    // The equation, the check and the note for step i, from the trace and the engine's own live maths.
    function renderSide(i) {
      if (!g) return;
      const { T, V, d, q, ids, hasP, enc, tok, out, pos, W, dTok, dOut } = g;
      const s = ctx.state, id = ids[q];
      const liveOf = (key) => { const lv = runLive(s, key, { r: q, band: 0, x: 0 }); return lv && lv.result !== undefined ? lv : null; };
      let main = '', live = '', plain = false, note = '', check = null;
      stepLab.textContent = i === 2 && !hasP ? 'No position table' : STEPS[i];
      if (i === 0) {
        main = `\\mathrm{id}_t \\in \\{0,\\dots,${V - 1}\\}`;
        plain = true;
        live = `“${tokName(q)}” at position ${q} is id ${id}, so it reads row ${id} of W_E.`;
        note = `The vocabulary has ${fmtN(V)} entries, and an id is the index of its token in that list.`
          + (s.config.tie_weights ? ' Tied weights: W_E is also W_U.' : '');   // short: the side column has three lines here
      } else if (i === 1) {
        const lv = liveOf('emb.tok') ?? { result: W[id * d], expected: tok[q * d] };
        main = mathFor('emb.tok').compact;
        live = `X_{\\mathrm{tok},${q},0} = \\WE[${id},\\,0] = ${num(lv.result, 4)}`;
        check = checkBadge(lv.result, lv.expected, { label: 'equals the traced value' });
        note = `all ${T * d} entries of X_tok: max |Δ| = ${sci(dTok)}`;
      } else if (i === 2) {
        if (hasP) {
          const lv = liveOf('emb.pos');
          main = enc === 'learned' ? '\\Pe_{t,c} = W_P[t,\\,c]' : '\\Pe_{t,2i} = \\sin\\!\\Big(\\frac{t}{10000^{2i/d}}\\Big)';
          live = `\\Pe_{${q},0} = ${num(lv ? lv.result : pos[q * d], 4)}`;
          if (lv) check = checkBadge(lv.result, lv.expected, { label: 'equals the traced value' });
          note = enc === 'learned' ? 'P is a learned table, one row per position.' : 'P is a fixed sine and cosine table.';
        } else {
          main = '\\text{no } \\Pe \\text{ table}';
          plain = true;
          live = (NO_POS[enc] || NO_POS.none)[2];
        }
      } else {
        const lv = liveOf('emb.out') ?? { result: tok[q * d] + (hasP ? pos[q * d] : 0), expected: out[q * d] };
        const p0 = hasP ? pos[q * d] : 0;
        main = hasP ? '\\xres_{0,t,c} = X_{\\mathrm{tok},t,c} + \\Pe_{t,c}' : '\\xres_{0,t,c} = X_{\\mathrm{tok},t,c}';
        live = hasP
          ? `\\begin{aligned}\\xres_{0,${q},0} &= ${num(tok[q * d], 4)} ${p0 < 0 ? '-' : '+'} ${num(Math.abs(p0), 4)} \\\\ &= ${num(lv.result, 4)}\\end{aligned}`
          : `\\xres_{0,${q},0} = ${num(tok[q * d], 4)}`;
        check = checkBadge(lv.result, lv.expected, { label: 'equals the traced value' });
        note = `max |Δ| over all ${T * d} entries: ${sci(dOut)}`;
      }
      eqMain.innerHTML = texHTML(main, { display: true });
      eqLive.classList.toggle('plain', plain);
      eqLive.innerHTML = plain ? '' : texHTML(live, { display: true });
      if (plain) eqLive.textContent = live;
      eqCheck.replaceChildren(...(check ? [check] : []));
      eqNote.textContent = note;
      cap.set(capText(i));
    }

    function capText(i) {
      const { V, q, ids, hasP, enc } = g;
      const cfg = ctx.state.config;
      if (i === 0) {
        const unit = ctx.state.tokenizer.kind === 'char'
          ? 'Each character is one token.'
          : 'Each token is a piece of text, often a whole word and sometimes part of one.';
        return `${unit} Its id, from 0 to ${fmtN(V - 1)}, is its index in the vocabulary, and it names one row of W_E.`;
      }
      if (i === 1) return `The id copies row ${ids[q]} of W_E into row ${q} of X_tok, the row for “<b>${tokName(q)}</b>”. A one-hot vector times W_E gives the same row, so the lookup is a matrix product.`;
      if (i === 2) return hasP
        ? 'Each position vector joins the token at that position, so the same token gets a different x₀ at every position.'
        : (NO_POS[enc] || NO_POS.none)[2];
      const tail = cfg.residual
        ? 'The residual stream starts here: each attention and MLP output is added onto it, layer after layer.'
        : 'Without residual connections, x₀ feeds block 0 only, because each sub-layer replaces the stream.';
      return `${hasP ? 'x₀ = X_tok + P' : 'x₀ = X_tok'} is the block input for every token. ${tail}`;
    }

    // ------------------------------------------------------------- motion
    // Step 1 (lookup): a copy of the W_E row flies down into its row of X_tok, which lights up when it lands.
    async function lookup(my) {
      const { A, B, arrow, look } = g;
      arrow.setOpacity(1);
      look.classList.remove('em-dim');
      renderCtx(1); renderSide(1);
      A.el.classList.remove('em-focus');                 // the accent moves to X_tok as the copy lands
      // the copy is of the whole strip: its cell sizes are CSS variables on the strip, which a copy of the tray would lose
      const flight = fly(st, A.el, B.el, { duration: 640, arc: 24 });
      await wait(560);
      if (!alive(my)) return;
      setStage(B, true);
      B.el.classList.add('em-focus');
      await flight;
    }

    // Step 2 (positions): the position row fills in from left to right.
    async function positions(my) {
      const { B, C, q, d, pos, RANGE } = g;
      renderCtx(2); renderSide(2);
      setStage(C, true);
      B.el.classList.remove('em-focus');
      C.el.classList.add('em-focus');
      const cells = [...C.el.querySelectorAll('.k-cell:not(.ell)')];
      cells.forEach((c) => { c.style.transition = 'none'; c.style.background = 'transparent'; });
      void C.el.offsetWidth;                                       // reflow, so the colours start from transparent
      cells.forEach((c, k) => { c.style.transition = ''; c.style.transitionDelay = `${k * 26}ms`; });
      C.el.update(pos.subarray(q * d, q * d + d), { range: RANGE });
      await wait(cells.length * 26 + 520);
      if (!alive(my)) return;
      cells.forEach((c) => { c.style.transitionDelay = ''; });
    }

    // Step 3 (sum): x₀ starts as the X_tok row, then the sum with P lands in it; the plus pulses.
    async function sum(my) {
      const { B, C, D, plus, eqOp, q, d, tok, out, RANGE } = g;
      renderCtx(3); renderSide(3);
      setStage(D, true);
      B.el.classList.remove('em-focus');
      C?.el.classList.remove('em-focus');
      D.el.classList.add('em-focus');
      D.el.update(tok.subarray(q * d, q * d + d), { range: RANGE });
      await wait(560);
      if (!alive(my)) return;
      D.el.update(out.subarray(q * d, q * d + d), { range: RANGE });
      eqOp.style.opacity = '1';
      if (plus) pulse(plus);
      await wait(560);
    }

    async function animate(i, my) {
      animating = true;
      if (i === 1) await lookup(my);
      else if (i === 2 && g.hasP) await positions(my);
      else if (i === 3) await sum(my);
      if (!alive(my)) return;
      animating = false;
      applyStatic(i);
    }

    // ------------------------------------------------------------- steps
    function setStep(i) {
      if (!autoStep) stopAuto();
      const prev = cur, forward = i === prev + 1 && !reducedMotion() && !!g;
      cur = i;
      const my = ++run;
      animating = false;
      if (forward) { applyStatic(prev); animate(i, my); } else applyStatic(i);
      armAuto();
    }

    // First view: a step every MOTION.step, waiting for any motion to finish first.
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

    function rebuild() { run++; animating = false; build(); }

    // ---------------------------------------------------------------- mount
    ctx.setSteps(STEPS, setStep);
    rebuild();
    appear([header.el, hero, side, ctxBox, cap.el], { stagger: MOTION.stagger });
    el.addEventListener('pointerdown', stopAuto, { once: true });
    el.addEventListener('atlas:interrupt', stopAuto, { once: true });   // the story bar or the arrow keys took over
    // the W_E readout: the entry under the pointer, shown only while the lookup step is on
    ctxBox.addEventListener('pointerover', (e) => {
      const cell = e.target.closest?.('[data-wte]');
      if (!cell || !g || !readEl || cur !== 1) return;
      const [r, c] = cell.dataset.wte.split(',').map(Number);
      readEl.textContent = `W_E[${r}, ${c}] = ${fmt(g.W[r * g.d + c], 3)} · “${name(r)}”`;
    });
    ctxBox.addEventListener('pointerleave', () => { if (readEl) readEl.textContent = cur === 1 ? HINT : ''; });
    const offs = [ctx.on('trace', () => { stopAuto(); rebuild(); }), ctx.on('select', () => { if (!destroyed) rebuild(); })];
    const themeOff = onThemeChange(() => setTimeout(() => { if (!destroyed) rebuild(); }, 0));
    armAuto();

    return {
      pause() { run++; animating = false; stopAuto(); },
      resume() { if (!destroyed) applyStatic(cur); },
      destroy() {
        destroyed = true;
        run++;
        clearTimeout(autoTimer);
        offs.forEach((f) => f());
        themeOff();
        g?.arrow?.destroy();
        st.destroy();
      },
    };
  },
};

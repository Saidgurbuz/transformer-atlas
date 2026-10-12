// Sampling: how the next token is chosen from the model's distribution.
// Hero: one bar per entry of the vocabulary, sorted by score, for the selected position. Up to 64 entries (the char
// model) keeps its own layout, with an ellipsis past 40 bars. A longer vocabulary (the subword model, 4,096 entries) shows the top 20 by
// probability, with their labels in two staggered rows, and a note counts the rest and bounds their probability.
// Temperature, top-k and top-p reshape the full distribution through engine.samplingDistribution; removed entries
// fade. The Draw step is inverse-CDF sampling: a seeded uniform U lands on a unit strip cut into the final
// probabilities, and the segment it lands in is the token chosen. The strip has one segment per bar and one tail
// segment for the rest, so a draw can land past the top bars and still name its token. The strip, the histogram and
// the draw controls appear on that step only, so before it the bars and the step equation are the only focus.
// The histogram counts draws from one distribution. It is marked stale when the position, the settings, the
// vocabulary or the logits row change.
// Each step owns the controls it teaches. A step sets its own settings when it is chosen; moving a control on an
// earlier step moves the scene to the step that shows it, so every caption and equation describes the bars on
// screen. A bar carries data-trace (the atlas hover card and maths) only while it equals the traced softmax
// (tau 1, no filters). Otherwise a tooltip gives its sampling weight.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, appear, pulse, useStyle, ZONES, MOTION } from '../kit.js';
import { samplingDistribution, sampleFrom, argmax, mulberry32 } from '../../engine/index.js';
import { h, fmt, num, texHTML, slider, button, reducedMotion, ease, showTip, hideTip, sequential } from '../../ui/index.js';

useStyle('sampling', `
.sp-plot { position: absolute; border-radius: 12px; background: color-mix(in srgb, var(--surface) 74%, transparent);
  border: 1px solid color-mix(in srgb, var(--c-head) 26%, var(--line));
  box-shadow: 0 16px 36px rgba(0, 0, 0, .14), 0 2px 0 color-mix(in srgb, var(--c-head) 16%, transparent); }
.sp-plot.still * { transition: none !important; }
.sp-grid { position: absolute; border-top: 1px dashed var(--line-strong); opacity: .7; transition: top .6s cubic-bezier(.65, 0, .35, 1); }
.sp-grid.sp-base { border-top: 1px solid var(--line-strong); opacity: 1; }
.sp-tick { position: absolute; width: 46px; text-align: right; font: 500 12px/1 var(--font-mono); color: var(--fg-3); transform: translateY(-50%);
  transition: top .6s cubic-bezier(.65, 0, .35, 1); }
.sp-sigma { font: 600 14px/1 var(--font-mono); color: var(--fg-2); }
.sp-col { position: absolute; transition: height .6s cubic-bezier(.65, 0, .35, 1); }
.sp-fill { position: absolute; left: 16%; right: 16%; bottom: 0; border-radius: 4px 4px 2px 2px; background: var(--c-head);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .22), 0 6px 12px rgba(0, 0, 0, .12);
  transition: height .6s cubic-bezier(.65, 0, .35, 1), background-color .4s, opacity .4s, box-shadow .4s; }
.sp-fill.off { background: color-mix(in srgb, var(--fg-3) 26%, transparent); box-shadow: none; }
.sp-fill.pick { outline: 2px solid var(--accent); outline-offset: 2px; }
.sp-val { position: absolute; left: -12px; right: -12px; text-align: center; font: 600 12px/1 var(--font-mono); color: var(--fg-2);
  white-space: nowrap; transition: bottom .6s cubic-bezier(.65, 0, .35, 1); }
.sp-val.top { color: var(--fg); }
.sp-frame { position: absolute; border: 1px dashed var(--fg-3); border-radius: 8px; pointer-events: none;
  transition: top .6s cubic-bezier(.65, 0, .35, 1), height .6s cubic-bezier(.65, 0, .35, 1); }
.sp-flab { position: absolute; font: 500 12px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.sp-chars { position: absolute; transition: top .6s cubic-bezier(.65, 0, .35, 1); }
.sp-char { position: absolute; top: 0; height: 18px; line-height: 18px; text-align: center; white-space: nowrap;
  font: 600 var(--sp-lfs, 15px)/18px var(--font-mono); color: var(--fg); transition: color .3s, opacity .3s; }
.sp-char.off { color: var(--fg-3); opacity: .5; }
.sp-char.pick { color: var(--accent); }
.sp-char.sp-ell { color: var(--fg-3); }
.sp-group { transition: opacity .4s ease, visibility .4s; }
.sp-group.sp-off { opacity: 0; visibility: hidden; }
.sp-track { position: absolute; border-radius: 6px; background: var(--surface-2); border: 1px solid var(--line); overflow: hidden; }
.sp-seg { position: absolute; top: 0; bottom: 0; background: var(--c-head); box-shadow: inset -1.5px 0 0 var(--surface);
  transition: left .6s cubic-bezier(.65, 0, .35, 1), width .6s cubic-bezier(.65, 0, .35, 1), background-color .3s; }
.sp-seg.alt { background: color-mix(in srgb, var(--c-head) 70%, var(--surface)); }
.sp-seg.hot { background: color-mix(in srgb, var(--c-head) 55%, var(--fg)); }
.sp-seg.pick { background: var(--accent); }
.sp-tail { background: color-mix(in srgb, var(--c-head) 22%, var(--surface));
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--c-head) 40%, transparent);
  transition: left .6s cubic-bezier(.65, 0, .35, 1), width .6s cubic-bezier(.65, 0, .35, 1); }
.sp-tail[hidden], .sp-tpick[hidden] { display: none; }
.sp-tpick { background: var(--accent); box-shadow: none;
  transition: left .6s cubic-bezier(.65, 0, .35, 1), width .6s cubic-bezier(.65, 0, .35, 1), background-color .3s; }
.sp-tpick.hot { background: color-mix(in srgb, var(--accent) 70%, var(--fg)); }
.sp-marker { position: absolute; width: 2px; border-radius: 1px; background: var(--accent); opacity: 0;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent); transition: opacity .3s; }
.sp-pill { position: absolute; transform: translateX(-50%); padding: 3px 8px; border-radius: 999px; opacity: 0;
  font: 600 12px/1.2 var(--font-mono); color: var(--surface); background: var(--accent); white-space: nowrap;
  transition: opacity .3s, top .6s cubic-bezier(.65, 0, .35, 1); }
.sp-hcol { position: absolute; }
.sp-hbar { position: absolute; left: 16%; right: 16%; bottom: 0; border-radius: 3px 3px 1px 1px; background: var(--c-head); opacity: .9;
  transition: height .6s cubic-bezier(.65, 0, .35, 1), opacity .4s; }
.sp-hcol.stale .sp-hbar { opacity: .28; }
.sp-hexp { position: absolute; left: 8%; right: 8%; height: 2px; border-radius: 1px; background: var(--fg); opacity: 0;
  transition: bottom .6s cubic-bezier(.65, 0, .35, 1), opacity .4s; }
.sp-hcnt { position: absolute; left: -10px; right: -10px; text-align: center; font: 600 11px/1 var(--font-mono); color: var(--fg-2);
  transition: bottom .6s cubic-bezier(.65, 0, .35, 1); }
.sp-hcap { position: absolute; font: 500 12px/1.2 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.sp-more { position: absolute; text-align: right; font: 500 13px/1.3 var(--font-mono); color: var(--fg-2); white-space: nowrap;
  pointer-events: none; }
.sp-more sup { font-size: 9px; line-height: 0; }
.sp-note { position: absolute; font: 500 13px/1.2 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.sp-note[hidden] { display: none; }
.sp-note::before { content: ''; display: inline-block; width: 7px; height: 7px; margin: 0 8px 1px 0; border-radius: 50%;
  background: var(--c-mask); vertical-align: middle; }
.sp-note.point::before { background: var(--fg-3); }
.sp-go { background: none; border: 0; padding: 0; font: inherit; color: var(--fg); cursor: pointer; text-decoration: underline; text-underline-offset: 4px; }
.sp-go:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
.sp-side { position: absolute; display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; align-content: start; }
.sp-side > * { min-width: 0; }
.sp-eq-main { font-size: 22px; min-height: 58px; overflow-x: auto; }
.sp-eq-live { font-size: 16px; padding: 8px 12px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent);
  border: 1px solid var(--line); overflow-x: auto; }
.sp-check-row { min-height: 24px; display: flex; align-items: center; }
.sp-stats { display: grid; gap: 3px; }
.sp-stat { font: 500 14px/1.4 var(--font-mono); color: var(--fg-2); }
.sp-stat b { color: var(--fg); font-weight: 600; }
.sp-ctl { display: grid; gap: 4px; }
.sp-ctl .field { gap: 2px; }
#sampling-temp, #sampling-topk, #sampling-topp { accent-color: var(--c-head); }
.sp-btns { display: flex; gap: 8px; align-items: center; }
.sp-seed { margin-left: auto; font: 500 12px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
`);

const STEPS = ['Distribution', 'Temperature', 'Top-k and top-p', 'Draw'];
const DRAW_STEP = 3;                                        // the strip, the histogram and the draw controls show from here on
const SEED = 1337;                                          // mulberry32 seed: the same draws on every visit
const ZOOM_MS = 800;                                        // the zoom into this scene takes about 760 ms
const PLOT = { x: ZONES.hero.x, y: ZONES.hero.y, w: 1000, h: 470 };
const SIDE = { x: ZONES.side.x, y: ZONES.hero.y, w: ZONES.side.w };
const GL = 56, GR = 24, PW = PLOT.w - GL - GR;              // left gutter, right margin, width of every row
const TOP = 56, BH_PLAIN = 344, BH_DRAW = 204;              // the 1.0 line; bar height before the draw, and on the draw step
const LIST_MAX = 64;                                        // up to this many entries the chart keeps its own layout (the char model; an ellipsis past 40 bars)
const MAX_BARS = 40;                                        // 41 to 64 entries: the first ones, an ellipsis, the last 4
const TOP_BARS = 20;                                        // longer vocabularies (the subword model): the top bars by probability
const LBL_GAP = 22, LBL_ONE = 18, LBL_TWO = LBL_GAP + 18;   // label rows: one row, or two (the odd bars on the lower row)
const BH_DRAW_TWO = 176;                                    // draw-step bar height with two label rows: the U pill still fits above the strip
const STRIP = { y: 312, h: 22 };                            // cumulative strip, [0, 1] (draw step)
const HIST = { y: 352, h: 72 };                             // histogram of 100 draws (draw step)
const NEAR = 0.9;                                           // a top probability this high is "nearly certain" (top-p 0.9 keeps one token then)
// What a step sets when it is chosen from the step bar. The draw uses the settings on screen.
const SETTINGS = [{ tau: 1, k: 0, p: 1 }, { tau: 2, k: 0, p: 1 }, { k: 8, p: 0.9 }];

const px = (n) => `${n}px`;
const r2 = (x) => Math.round(x * 100) / 100;
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
const word = (label) => (label === '␣' ? 'space' : esc(label));           // a token as it reads in a caption
// A probability as the reader sees it. "100%" only for values that round to it at two decimals; a value just short of 1 reads >99.9%.
const pct = (p) => (p >= 0.99995 ? '100%' : p >= 0.999 ? '>99.9%' : p >= 0.99 ? `${(p * 100).toFixed(1)}%`
  : p >= 0.1 ? `${Math.round(p * 100)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : p > 0 ? '<0.1%' : '0%');
// A probability bound for the note: four decimals, or a mantissa and a power of ten below 0.0001.
const probHtml = (p) => {
  if (p === 0) return '0';
  if (p >= 0.0001) return p.toFixed(4);
  let e = Math.floor(Math.log10(p)), m = (p / 10 ** e).toFixed(1);
  if (m === '10.0') { e += 1; m = (p / 10 ** e).toFixed(1); }
  return `${m}×10<sup>−${-e}</sup>`;
};
const fmtInt = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const entropy = (p) => -p.reduce((a, x) => (x > 0 ? a + x * Math.log(x) : a), 0);
const plural = (n, noun) => `${fmtInt(n)} ${noun}${n === 1 ? '' : 's'}`;
const shownCount = (V) => (V > LIST_MAX ? TOP_BARS : V);         // the ranks with their own bar and strip segment
const unitOf = (s) => (s.tokenizer.kind === 'char' ? 'character' : 'token');   // the char model keeps its own word
const TEX_ESC = { '\\': '\\backslash', '{': '\\{', '}': '\\}', '$': '\\$', '&': '\\&', '#': '\\#', '^': '\\textasciicircum',
  '_': '\\_', '%': '\\%', '~': '\\textasciitilde', "'": "\\text{'}", ' ': '\\text{space}', '\n': '\\text{newline}', '\t': '\\text{tab}' };
// A token as typewriter TeX, one escape per character. A leading space is a short gap; the char model's space keeps its word.
const texChar = (s) => (s === ' ' ? '\\mathtt{\\text{space}}'
  : `\\mathtt{${Array.from(s, (c) => (c === ' ' ? '\\ ' : TEX_ESC[c] ?? (c.charCodeAt(0) > 126 ? `\\text{${c}}` : c))).join('')}}`);
const put = (el, tex) => { if (el.dataset.tex !== tex) { el.dataset.tex = tex; el.innerHTML = texHTML(tex, { display: true }); } };

/** Indices to draw for an axis of length n: all of them, or the first ones, -1 (an ellipsis), the last 4. */
function elide(n, max) {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const tail = 4, head = max - tail - 1;
  return [...Array.from({ length: head }, (_, i) => i), -1, ...Array.from({ length: tail }, (_, i) => n - tail + i)];
}
/** Where the bars stop for a bar height: the baseline, the first label row, and the height of the label block. */
const geoOf = (BH, two) => ({ BH, BASE: TOP + BH, CHR: TOP + BH + 6, LBL: two ? LBL_TWO : LBL_ONE });
/** U and the segment it falls in, at one precision: six decimals, more only where U sits within a rounding step of a
 *  bound, so the comparison on screen is the one the draw made. */
function segText(U, lo, hi) {
  for (let d = 6; d <= 12; d++) {
    const a = lo.toFixed(d), b = U.toFixed(d), c = hi.toFixed(d);
    if (Number(a) <= Number(b) && Number(b) < Number(c)) return [a, b, c];
  }
  return [lo.toFixed(12), U.toFixed(12), hi.toFixed(12)];
}
/** The note above a long vocabulary's bars: how many entries the bars leave out, their share of the mass, and the
 *  largest of them as a bound. Filtered entries show their weight before the filters, as their bars do. */
function moreHtml(D, NS) {
  const N = D.V - NS, r0 = D.order[NS];
  const thr = D.pFin[r0] > 0 ? D.pFin[r0] : D.pRaw[r0];
  let mass = 0, kept = 0;
  for (let r = NS; r < D.V; r++) { const v = D.order[r]; if (D.pFin[v] > 0) { mass += D.pFin[v]; kept++; } }
  if (kept === 0) return `+ ${fmtInt(N)} more, all removed by the filters (p ≤ ${probHtml(thr)} before them)`;
  const what = kept === N ? `${pct(mass)} of the mass`
    : `${fmtInt(kept)} kept (${pct(mass)} of the mass), ${fmtInt(N - kept)} removed`;
  return `+ ${fmtInt(N)} more, ${what}, p ≤ ${probHtml(thr)}`;
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const rng = mulberry32(SEED);
    const view = { tau: 1, k: 0, p: 1, step: 0, lastU: null, hist: null };
    let cur = null, builtV = -1, anim = 0, auto = 0, autoDriving = false, keeping = false;
    let lastNote = '', capText = null, rowMemo = null, spreadMemo = null;
    let plot, grid = [], cols = [], chars = [], segs = [], hcols = [], drawEls = [];
    let frame, chRow, track, marker, pill, histCap, tail, tailPick, more;
    let longVocab = false;                                  // the bars are the top TOP_BARS of a long vocabulary (labels in two rows)
    const stopAuto = () => { clearTimeout(auto); auto = 0; };

    // ------------------------------------------------------------------ shared parts
    const header = sceneHeader(st, { chips: [['LM head', 'head']], title: 'Sampling' });
    const picker = tokenPicker(st, ctx, { label: 'after', kind: 'head' });
    const cap = sceneCaption(st);
    // The kit's caption fades but does not cancel a fade already running, so an older text can land last. As in
    // lm-head.js, a newer caption cancels the running fade and only the latest text is ever written.
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

    plot = h('div', { class: 'sp-plot', style: { width: px(PLOT.w), height: px(PLOT.h) } });
    st.add(plot, { x: PLOT.x, y: PLOT.y });
    // One line above the plot, right-aligned to it: the bidirectional-mask note, or the pointer to a spread position.
    const note = h('div', { class: 'sp-note', hidden: true });
    st.add(note, { x: PLOT.x + PLOT.w, y: PLOT.y - 6, anchor: 'br' });

    // A bar that is not the traced softmax has no data-trace, so the atlas card would show the wrong number.
    // Its tooltip gives the sampling weight instead. Leaving the plot hides it.
    plot.addEventListener('pointermove', (e) => {
      const col = e.target.closest?.('.sp-col');
      if (!col || col.dataset.trace || !cur) { hideTip(); return; }
      showTip(e.clientX, e.clientY, sampleTip(cur, Number(col.dataset.v)));
    });
    plot.addEventListener('pointerleave', hideTip);

    // ------------------------------------------------------------------ side column
    // A control moved on an earlier step moves the scene to the step that shows it (the Temperature step for tau,
    // the filters step for top-k and top-p). The values the reader set are kept.
    const own = (step) => { if (view.step < step) jumpTo(step); else paint(); };
    const jumpTo = (i) => { keeping = true; try { ctx.setStep(i); } finally { keeping = false; } };
    const stepLab = h('div', { class: 'k-label', style: { textAlign: 'left' } });    // the step's name, as in softmax.js
    const eqMain = h('div', { class: 'sp-eq-main' }), eqLive = h('div', { class: 'sp-eq-live' });
    const eqCheck = h('div', { class: 'sp-check-row' });
    const statH = h('div', { class: 'sp-stat' }), statG = h('div', { class: 'sp-stat' }), statU = h('div', { class: 'sp-stat sp-group' });
    const tauS = slider({ id: 'sampling-temp', label: 'temperature τ', min: 0.1, max: 2, step: 0.05, value: view.tau,
      format: (v) => v.toFixed(2), onInput: (v) => { stopAuto(); view.tau = r2(v); own(1); } });
    const kS = slider({ id: 'sampling-topk', label: 'top-k', min: 0, max: ctx.state.logits.shape[1], step: 1, value: view.k,
      format: (v) => (v ? String(v) : 'off'), onInput: (v) => { stopAuto(); view.k = v; own(2); } });
    const pS = slider({ id: 'sampling-topp', label: 'top-p', min: 0.1, max: 1, step: 0.05, value: view.p,
      format: (v) => (v < 1 ? v.toFixed(2) : 'off'), onInput: (v) => { stopAuto(); view.p = r2(v); own(2); } });
    const btnDraw = button('Sample', () => { stopAuto(); draw(); }, { title: 'Draw one token from the current distribution' });
    const btnMany = button('Sample 100', () => { stopAuto(); sampleMany(100); }, { title: 'Draw 100 tokens and fill the histogram' });
    const btns = h('div', { class: 'sp-btns sp-group' }, btnDraw, btnMany,
      h('span', { class: 'sp-seed', title: 'U comes from the mulberry32 generator with this seed' }, `seed ${SEED}`));
    const side = h('div', { class: 'sp-side', style: { width: px(SIDE.w) } },
      stepLab, eqMain, eqLive, eqCheck,
      h('div', { class: 'sp-stats' }, statH, statG, statU),
      h('div', { class: 'sp-ctl' }, tauS.el, kS.el, pS.el),
      btns);
    st.add(side, { x: SIDE.x, y: SIDE.y });
    const sideDraw = [statU, btns];                         // the draw readouts and buttons: the Draw step only

    // A position named in the note can be clicked to select it.
    const onPick = (e) => {
      const go = e.target.closest?.('[data-pos]');
      if (!go) return;
      stopAuto();
      ctx.store.select({ token: Number(go.dataset.pos) });
    };
    el.addEventListener('click', onPick);

    // ------------------------------------------------------------------ maths of the step
    // The histogram is about one distribution: the position, the vocabulary, the settings and the logits row.
    function keyOf(D) { return `${D.t}|${D.V}|${view.tau}|${view.k}|${view.p}|${D.z.join(',')}`; }

    // Everything the scene shows, for the selected position: the scores, the distributions and the order.
    function compute() {
      const s = ctx.state, T = s.logits.shape[0], V = s.logits.shape[1];
      const t = Math.min(s.sel.token ?? T - 1, T - 1);
      view.k = Math.min(view.k, V);
      const row = (r) => s.logits.data.subarray(r * V, (r + 1) * V);
      const z = row(t);
      // What depends only on the logits row (the untempered softmax, its check against the trace, the sorted order) is
      // kept until the logits or the position change, so a slider move sorts just the two tempered distributions.
      let R = rowMemo;
      if (!R || R.logits !== s.logits || R.t !== t) {
        const pRef = samplingDistribution(z, { temperature: 1 });                           // untouched softmax
        const traced = s.trace.probs.data.subarray(t * V, (t + 1) * V);
        let maxDiff = 0;
        for (let v = 0; v < V; v++) maxDiff = Math.max(maxDiff, Math.abs(pRef[v] - traced[v]));
        // Sorting by score is the same at every temperature, so the bars keep their order when tau changes.
        const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => z[b] - z[a] || a - b);
        R = rowMemo = { logits: s.logits, t, pRef, traced, maxDiff, order };
      }
      const { pRef, traced, maxDiff, order } = R;
      const pRaw = samplingDistribution(z, { temperature: view.tau, topK: 0, topP: 1 });   // softmax(z / tau)
      const pFin = samplingDistribution(z, { temperature: view.tau, topK: view.k, topP: view.p });
      const rankOf = new Array(V), start = new Array(V);
      let acc = 0;
      order.forEach((v, r) => { rankOf[v] = r; start[r] = acc; acc += pFin[v]; });
      const kept = order.filter((v) => pFin[v] > 0);
      const mass = kept.reduce((a, v) => a + pRaw[v], 0);
      const mask = s.trace.mask;
      const g = argmax(z);
      const topFin = pFin[g];                                                                // the greedy token is kept
      const exact = pFin.every((x, v) => Math.abs(x - pRef[v]) < 1e-9);                      // the bars are the traced softmax
      // Nearly certain here: the most spread-out other position that is not nearly certain, at these settings.
      // The search sorts every position's row, so it is kept until the logits, the position or the settings change.
      let spread = null;
      if (topFin >= NEAR) {
        const m = spreadMemo;
        if (m && m.logits === s.logits && m.t === t && m.tau === view.tau && m.k === view.k && m.p === view.p) spread = m.spread;
        else {
          let bestH = -1;
          for (let r = 0; r < T; r++) {
            if (r === t) continue;
            const q = samplingDistribution(row(r), { temperature: view.tau, topK: view.k, topP: view.p });
            const top = Math.max(...q);
            if (top >= NEAR) continue;
            const H = entropy(q);
            if (H > bestH) { bestH = H; spread = { r, top }; }
          }
          spreadMemo = { logits: s.logits, t, tau: view.tau, k: view.k, p: view.p, spread };
        }
      }
      return {
        s, t, T, V, z, pRaw, pFin, pRef, traced, order, rankOf, start, kept, mass, maxDiff, exact,
        greedy: g, topFin, H: entropy(pFin), H1: entropy(pRef),
        runner: V > 1 ? order[1] : null,
        seesNext: t + 1 < T && !!mask && mask.data[t * T + t + 1] === 1,
        spread,
      };
    }

    // The draw: the same rule as engine.sampleFrom, applied to the kept tokens in rank order.
    function pickFor(D, u) {
      const probs = D.kept.map((v) => D.pFin[v]);
      return D.kept[sampleFrom(probs, () => u)];
    }

    // Tooltip for a bar without data-trace: its sampling weight, the settings, and the traced softmax next to it.
    function sampleTip(D, v) {
      const set = [`τ = ${view.tau.toFixed(2)}`, view.k ? `top-k ${view.k}` : null, view.p < 1 ? `top-p ${view.p.toFixed(2)}` : null]
        .filter(Boolean).join(', ');
      const weight = D.pFin[v] > 0 ? `sampling weight ${fmt(D.pFin[v], 3)}`
        : `removed by the filters (weight before them ${fmt(D.pRaw[v], 3)})`;
      return `<b>“${esc(D.s.tokenizer.label(v))}”</b> · ${weight}<br>${set}<br>model softmax, traced: ${fmt(D.traced[v], 3)}`;
    }

    // ------------------------------------------------------------------ DOM of the plot (rebuilt when V changes)
    function buildPlot(V) {
      builtV = V;
      longVocab = V > LIST_MAX;
      const slots = longVocab ? Array.from({ length: TOP_BARS }, (_, i) => i) : elide(V, MAX_BARS);   // the ranks drawn, left to right; -1 is an ellipsis
      const colW = PW / slots.length;
      const kids = [];
      grid = [];
      for (const [label, key, base] of [['1.0', 'top', false], ['0.5', 'mid', false], ['0', 'base', true]]) {
        const line = h('div', { class: base ? 'sp-grid sp-base' : 'sp-grid', style: { left: px(GL), width: px(PW) } });
        const tick = h('div', { class: 'sp-tick', style: { left: px(4) } }, label);
        grid.push({ key, line, tick });
        kids.push(line, tick);
      }
      // the draw parts: the strip, the histogram and their labels, shown on the Draw step
      const sigma = h('div', { class: 'sp-tick sp-sigma sp-group', style: { left: px(4), top: px(STRIP.y + STRIP.h / 2) } }, 'Σp');
      const drawsTick = h('div', { class: 'sp-tick sp-group', style: { left: px(4), top: px(HIST.y + HIST.h / 2) } }, 'draws');
      const histBase = h('div', { class: 'sp-grid sp-base sp-group', style: { left: px(GL), width: px(PW), top: px(HIST.y + HIST.h) } });
      // the greedy column is the argmax (the first column, since the bars are sorted by score), framed and labelled
      frame = h('div', { class: 'sp-frame', style: { left: px(GL - 4), width: px(colW + 8), top: px(TOP - 26) } });
      const frameLab = h('div', { class: 'sp-flab', style: { left: px(GL - 4), top: px(12) } }, 'greedy');
      cols = new Array(V).fill(null); chars = new Array(V).fill(null); hcols = new Array(V).fill(null);
      const colEls = [], charEls = [], histEls = [];
      slots.forEach((r, i) => {
        if (r < 0) { charEls.push(h('div', { class: 'sp-char sp-ell', style: { left: px(i * colW), width: px(colW) } }, '⋯')); return; }
        const fill = h('div', { class: 'sp-fill' }), val = h('div', { class: 'sp-val' });
        const el = h('div', { class: 'sp-col', style: { left: px(GL + i * colW), width: px(colW), top: px(TOP) } }, fill, val);
        cols[r] = { fill, val, el };
        colEls.push(el);
        // a label is as wide as its bar's column; on the two-row layout it may take the width of two columns, centred on its bar
        const box = longVocab ? { left: px((i + 0.5) * colW - (colW - 2)), width: px(2 * colW - 4), top: px(i % 2 ? LBL_GAP : 0) }
          : { left: px(i * colW), width: px(colW) };
        chars[r] = h('div', { class: 'sp-char', style: box });
        charEls.push(chars[r]);
        const bar = h('div', { class: 'sp-hbar' }), exp = h('div', { class: 'sp-hexp' }), cnt = h('div', { class: 'sp-hcnt' });
        const hel = h('div', { class: 'sp-hcol sp-group', style: { left: px(GL + i * colW), width: px(colW), top: px(HIST.y), height: px(HIST.h) } }, bar, exp, cnt);
        hcols[r] = { el: hel, bar, exp, cnt };
        histEls.push(hel);
      });
      // one strip segment per shown rank; past the top bars, one tail segment holds the rest of the strip
      segs = Array.from({ length: shownCount(V) }, () => h('div', { class: 'sp-seg' }));
      tail = h('div', { class: 'sp-seg sp-tail', hidden: true });
      tailPick = h('div', { class: 'sp-seg sp-tpick', hidden: true });
      chRow = h('div', { class: 'sp-chars', dataset: { trace: 'logits' }, style: { left: px(GL), width: px(PW), height: px(longVocab ? LBL_TWO : LBL_ONE) } }, charEls);
      track = h('div', { class: 'sp-track sp-group', style: { left: px(GL), top: px(STRIP.y), width: px(PW), height: px(STRIP.h) } }, segs, tail, tailPick);
      marker = h('div', { class: 'sp-marker', style: { top: px(STRIP.y - 8), height: px(STRIP.h + 16) } });
      pill = h('div', { class: 'sp-pill' });
      histCap = h('div', { class: 'sp-hcap sp-group', style: { left: px(GL), top: px(HIST.y + HIST.h + 12) } });
      more = h('div', { class: 'sp-more', hidden: true, style: { right: px(GR), top: px(TOP + 6) } });
      drawEls = [sigma, drawsTick, histBase, track, histCap, ...histEls];
      plot.replaceChildren(...kids, sigma, drawsTick, histBase, frame, frameLab, ...colEls, chRow, track, marker, pill, ...histEls, histCap, more);
    }

    /** The bars' height depends on the step: taller while the draw is hidden. The lines, labels and frame follow it. */
    function placeBars(g) {
      const at = { top: TOP, mid: TOP + g.BH / 2, base: g.BASE };
      for (const q of grid) { q.line.style.top = px(at[q.key]); q.tick.style.top = px(at[q.key]); }
      for (const c of cols) if (c) c.el.style.height = px(g.BH);
      chRow.style.top = px(g.CHR);
      frame.style.height = px(g.CHR + 24 - (TOP - 26));
      pill.style.top = px(g.CHR + g.LBL + 6);
    }

    /** The draw landed past the top bars: its exact entry gets its own sliver of the strip, sized to its probability. */
    function placeTailPick(D, r) {
      const on = r >= segs.length;                              // r = -1 (no draw on screen) is never past the bars
      tailPick.hidden = !on;
      if (!on) return;
      const v = D.order[r];
      tailPick.style.left = px(D.start[r] * PW);
      tailPick.style.width = px(Math.max(2, D.pFin[v] * PW));
    }

    /** Move the U marker along the strip; with D, the segment under it is highlighted. */
    function placeMarker(x, D) {
      const on = x != null;
      marker.style.opacity = on ? '1' : '0';
      pill.style.opacity = on ? '1' : '0';
      segs.forEach((sg) => sg.classList.remove('hot'));
      tailPick.classList.remove('hot');
      if (!on) return;
      marker.style.left = px(GL + x * PW - 1);
      pill.style.left = px(GL + Math.min(PW - 40, Math.max(40, x * PW)));
      pill.textContent = `U = ${x.toFixed(6)}`;
      if (!D) return;
      const r = D.rankOf[pickFor(D, x)];
      if (r < segs.length) segs[r].classList.add('hot');
      else { placeTailPick(D, r); tailPick.classList.add('hot'); }
    }

    // ------------------------------------------------------------------ paint
    function paint() {
      const D = compute();
      cur = D;
      if (D.V !== builtV) buildPlot(D.V);
      const drawing = view.step === DRAW_STEP;
      const NS = shownCount(D.V);
      const BH = drawing ? (longVocab ? BH_DRAW_TWO : BH_DRAW) : BH_PLAIN;
      placeBars(geoOf(BH, longVocab));
      [...drawEls, ...sideDraw].forEach((e) => e.classList.toggle('sp-off', !drawing));
      kS.input.max = String(D.V);
      if (kS.get() !== view.k) kS.set(view.k);
      plot.classList.toggle('still', reducedMotion());
      const sweeping = anim !== 0;
      const hist = view.hist, stale = !!hist && hist.key !== keyOf(D);
      const U = drawing && !sweeping && view.lastU != null ? view.lastU : null;      // the draw on screen, once it has landed
      const pick = U == null ? -1 : D.rankOf[pickFor(D, U)];
      // The labels of a long vocabulary share one size: the longest label on screen must fit its two-column slot.
      if (longVocab) {
        let longest = 1;
        for (let r = 0; r < NS; r++) longest = Math.max(longest, Array.from(D.s.tokenizer.label(D.order[r])).length);
        const fs = Math.max(11, Math.min(15, Math.floor((2 * (PW / TOP_BARS) - 6) / (0.6 * longest))));
        chRow.style.setProperty('--sp-lfs', px(fs));
      }
      let ki = 0;                                              // index among the kept tokens (alternating tints)
      for (let r = 0; r < NS; r++) {
        const v = D.order[r], kept = D.pFin[v] > 0, p = kept ? D.pFin[v] : D.pRaw[v];
        const cell = `${D.t},${v},0`;
        const sg = segs[r];
        sg.classList.toggle('alt', kept && ki++ % 2 === 1);
        sg.classList.toggle('pick', r === pick);
        sg.classList.remove('hot');
        sg.style.left = px(D.start[r] * PW);
        sg.style.width = px(D.pFin[v] * PW);
        const c = cols[r];
        if (c) {
          c.el.dataset.v = String(v);
          if (D.exact) { c.el.dataset.trace = 'probs'; c.el.dataset.cell = cell; }
          else { delete c.el.dataset.trace; delete c.el.dataset.cell; }
          c.fill.style.height = px(p * BH);
          c.fill.style.background = kept ? sequential(p, 'head') : '';     // kept bars on the sequential ramp; removed ones grey
          c.fill.classList.toggle('off', !kept);
          c.fill.classList.toggle('pick', r === pick);
          c.val.style.bottom = px(p * BH + 4);
          c.val.classList.toggle('top', r === 0);
          c.val.textContent = kept && (r === 0 || r === pick || (r < 3 && p >= 0.05)) ? pct(p) : '';
          const ch = chars[r];
          ch.textContent = D.s.tokenizer.label(v);
          ch.dataset.cell = cell;
          ch.classList.toggle('off', !kept);
          ch.classList.toggle('pick', r === pick);
        }
        const hc = hcols[r];
        if (hc) {
          const n = hist ? hist.n : 0, cnt = hist ? hist.counts.get(v) ?? 0 : 0, f = n ? cnt / n : 0;
          hc.el.classList.toggle('stale', stale);
          hc.bar.style.height = px(f * HIST.h);
          hc.cnt.textContent = cnt ? String(cnt) : '';
          hc.cnt.style.bottom = px(f * HIST.h + 3);
          hc.exp.style.bottom = px(D.pFin[v] * HIST.h);
          hc.exp.style.opacity = kept && D.pFin[v] >= 0.001 ? '1' : '0';
        }
      }
      // The tail: every entry past the top bars, as one strip segment whose width is their share of the mass.
      let tailMass = 0;
      for (let r = NS; r < D.V; r++) tailMass += D.pFin[D.order[r]];
      const hasTail = NS < D.V && tailMass > 0;
      tail.hidden = !hasTail;
      if (hasTail) {
        tail.style.left = px(D.start[NS] * PW); tail.style.width = px(tailMass * PW);
        tail.title = `the other ${fmtInt(D.V - NS)} ${unitOf(D.s)}s, ${pct(tailMass)} of the mass`;
      }
      placeTailPick(D, pick);
      more.hidden = NS >= D.V;
      if (!more.hidden) more.innerHTML = moreHtml(D, NS);
      const inBars = hist ? D.order.slice(0, NS).reduce((a, v) => a + (hist.counts.get(v) ?? 0), 0) : 0;
      histCap.textContent = !hist ? 'Sample 100 fills this histogram. The ticks mark the probabilities.'
        : stale ? `${hist.n} draws from an earlier distribution. Press Sample 100 to redraw.`
          : NS < D.V && hist.n > inBars ? `${hist.n} draws: ${inBars} in the top ${NS}, ${hist.n - inBars} in the other ${fmtInt(D.V - NS)}. The ticks mark the probabilities.`
            : NS < D.V ? `${hist.n} draws, all in the top ${NS}. The ticks mark the probabilities.`
              : `${hist.n} draws. The ticks mark the probabilities.`;
      if (!sweeping) placeMarker(U, null);
      paintText(D, U, sweeping, drawing);
    }

    /** The note above the plot: the mask note, or the pointer to a spread position where the bars move. One line. */
    function noteHtml(D, step) {
      const unit = unitOf(D.s);
      if (D.seesNext) return { cls: 'sp-note mask', html: `Bidirectional mask: position ${D.t} can see the next ${unit}, so these bars are not a prediction.` };
      if (D.topFin < NEAR) return null;
      const word1 = word(D.s.tokenizer.label(D.greedy));
      const lead = step === DRAW_STEP ? `Position ${D.t} is nearly certain, so most draws pick <b>“${word1}”</b>.`
        : step === 0 ? `Position ${D.t} is nearly certain.` : `Position ${D.t} is nearly certain at these settings.`;
      if (!D.spread) return { cls: 'sp-note point', html: `${lead} No other position has a top probability below ${pct(NEAR)}.` };
      const why = step === DRAW_STEP ? 'to see spread-out draws' : step === 0 ? 'to see a spread-out distribution' : 'to see the settings act';
      return { cls: 'sp-note point', html: `${lead} Select <button type="button" class="sp-go" data-pos="${D.spread.r}">position ${D.spread.r}</button> (top ${pct(D.spread.top)}) ${why}.` };
    }

    function paintText(D, U, sweeping, drawing) {
      const { tau, k, p } = view, step = view.step, hist = view.hist;
      const kOn = k > 0, pOn = p < 1, p2 = p.toFixed(2), nF = D.kept.length, NS = shownCount(D.V);
      const unit = unitOf(D.s);
      const g = D.greedy, top = `<b>“${word(D.s.tokenizer.label(g))}”</b>`;
      const nm = (v) => word(D.s.tokenizer.label(v));
      const tok = (i) => word(ctx.label(i));
      const tex = (v) => texChar(D.s.tokenizer.token(v));
      const pv = U == null ? null : pickFor(D, U), pr = pv == null ? -1 : D.rankOf[pv];
      const seg = pv == null ? null : segText(U, D.start[pr], D.start[pr] + D.pFin[pv]);
      // The live equations subtract the largest score, as lm-head does, so the greedy token's exponent is 0.
      const m = D.z[g];
      const Zs = D.z.reduce((a, x) => a + Math.exp(x - m), 0);
      const Zt = D.z.reduce((a, x) => a + Math.exp((x - m) / tau), 0);

      // Step 0: the softmax at tau 1 with no filters (the step's own settings: they hold until another step is chosen).
      const s0 = D.topFin >= NEAR
        ? `${top} has ${pct(D.topFin)} of the mass, so the choice is nearly certain.`
        : `${top} has ${pct(D.topFin)}${D.runner == null ? '' : `, and the runner-up <b>“${nm(D.runner)}”</b> has ${pct(D.pFin[D.runner])}`}.`;
      // Step 1: the temperature with no filters. The numbers are the bars' own.
      const dH = D.H.toFixed(2) === D.H1.toFixed(2);                      // the same at the two decimals shown
      const change = tau === 1
        ? `At τ = 1 the bars are the plain softmax, with entropy ${D.H1.toFixed(2)} nats.`
        : `The top ${unit}, ${top}, moves from ${pct(D.pRef[g])} at τ = 1 to ${pct(D.topFin)} at τ = ${tau.toFixed(2)}, and ${dH
          ? `the entropy stays at ${D.H.toFixed(2)} nats.` : `the entropy from ${D.H1.toFixed(2)} to ${D.H.toFixed(2)} nats.`}`;
      // Step 2: the filters at the current temperature. The engine applies top-k first, then top-p inside the top-k set.
      const kTxt = k === 1 ? `keeps the most likely ${unit}` : `keeps the ${k} most likely ${unit}s`;
      let filt;
      if (!kOn && !pOn) filt = 'Top-k and top-p are both off.';
      else if (!pOn) filt = `Top-k ${kTxt}; top-p is off.`;
      else if (!kOn) filt = nF === 1
        ? `Top-p keeps the smallest set reaching ${p2}. Here that is ${top} alone, since it already reaches ${p2}.`
        : `Top-p keeps the smallest set reaching ${p2}, which is ${plural(nF, unit)}.`;
      else if (k === 1) filt = `Top-k ${kTxt}, so top-p does not bind.`;
      else if (nF === 1) filt = `Top-k ${kTxt}, and top-p cuts that set to ${top}, which reaches ${p2} on its own.`;
      else if (nF < k) filt = `Top-k ${kTxt}, and top-p cuts that set to the ${plural(nF, unit)} reaching ${p2}.`;
      else filt = `Top-k ${kTxt}, and top-p at ${p2} cuts none of them.`;
      const cut = nF < D.V ? 'Faded bars are removed, and the rest are renormalised to sum to 1.' : 'Nothing is removed, so the bars are unchanged.';
      // Step 3: the draw, with the settings it uses named.
      const setTxt = `τ = ${tau.toFixed(2)}${kOn || pOn ? `, ${[kOn ? `top-k ${k}` : '', pOn ? `top-p ${p2}` : ''].filter(Boolean).join(', ')}` : ', no filters'}`;
      const draws = U != null
        ? `Draws use ${setTxt}. U = ${seg[1]} falls in the segment of <b>“${nm(pv)}”</b>${pr >= NS ? `, beyond the top ${NS} bars,` : ','} so that ${unit} comes next; greedy would always take the tallest bar.`
        : sweeping
          ? `Draws use ${setTxt}. A seeded uniform number U lands on the strip, and the segment under it is the ${unit} chosen.`
          : `Draws use ${setTxt}. Press Sample to draw one ${unit}, or Sample 100 to fill the histogram.`;
      capSet([
        `Softmax turns the ${fmtInt(D.V)} scores after position ${D.t}, <b>“${tok(D.t)}”</b>, into probabilities that sum to 1. ${s0}`,
        `Dividing every score by τ before the softmax: below 1 sharpens the bars, above 1 flattens them. ${change}`,
        `${filt} ${cut}`,
        draws,
      ][step]);

      const main = [
        String.raw`\textstyle p_v = \frac{e^{z_v}}{\sum_k e^{z_k}}`,
        String.raw`\textstyle p_v(\tau) = \frac{e^{z_v / \tau}}{\sum_k e^{z_k / \tau}}`,
        String.raw`\textstyle p'_v = \frac{p_v}{\sum_{u \in S} p_u} \quad (v \in S)`,
        String.raw`\textstyle \sum_{i<j} p'_i \le U < \sum_{i \le j} p'_i`,
      ][step];
      const live = [
        String.raw`\textstyle p_{${tex(g)}} = \frac{e^{0}}{${num(Zs, 3)}} = ${num(D.pRef[g], 2)}`,
        String.raw`\textstyle p_{${tex(g)}}(\tau = ${tau.toFixed(2)}) = \frac{e^{0}}{${num(Zt, 3)}} = ${num(D.pRaw[g], 2)}`,
        String.raw`\sum_{u \in S} p_u = ${num(D.mass, 3)}, \quad |S| = ${nF}`,
        seg == null ? String.raw`U \sim \mathrm{Uniform}[0, 1)`
          : String.raw`\begin{aligned} ${seg[0]} &\le U = ${seg[1]} \\ &< ${seg[2]} \;\Rightarrow\; ${tex(pv)} \end{aligned}`,
      ][step];
      put(eqMain, main);
      put(eqLive, live);
      stepLab.textContent = STEPS[step];
      eqCheck.replaceChildren(checkBadge(D.maxDiff, 0, { label: 'untempered softmax = traced probs' }));
      btnDraw.title = `Draw one ${unit} from the current distribution`;
      btnMany.title = `Draw 100 ${unit}s and fill the histogram`;

      statH.innerHTML = `Entropy <b>${D.H.toFixed(2)}</b> nats, about <b>${Math.exp(D.H).toFixed(1)}</b> equally likely ${unit}s.`;
      statG.innerHTML = `Greedy (argmax) takes ${top}, at <b>${pct(D.topFin)}</b>.`;
      statU.innerHTML = !drawing ? '' : sweeping ? 'Drawing U from the seeded generator…'
        : U == null ? (hist ? `The histogram holds ${hist.n} draws. Press <b>Sample</b> to draw one and see its U.` : 'No draw yet. Press <b>Sample</b>.')
          : `U = <b>${seg[1]}</b> lands on “<b>${nm(pv)}</b>”, the segment from ${seg[0]} to ${seg[2]}${pr >= NS ? `, beyond the top ${NS} bars` : ''}.`;
      header.setSub(`Position <b>${D.t}</b> (<b>“${tok(D.t)}”</b>) scores all ${fmtInt(D.V)} vocabulary entries. Sampling decides which one comes next.`);

      const nt = noteHtml(D, step);
      const key = nt ? `${nt.cls}|${nt.html}` : '';
      if (key !== lastNote) {
        lastNote = key;
        if (nt) { note.className = nt.cls; note.innerHTML = nt.html; }
      }
      note.hidden = !nt;
    }

    // ------------------------------------------------------------------ draws
    // Sample: one U from the seeded generator, swept along the strip, then the pick lands.
    function draw() {
      cancelAnimationFrame(anim);
      const u = rng();
      view.lastU = u;
      if (reducedMotion()) { anim = 0; paint(); return; }
      anim = -1;                               // sweeping: the old pick is cleared, the readouts say "Drawing"
      paint();
      const t0 = performance.now();
      const sweep = (now) => {
        const q = Math.min(1, Math.max(0, (now - t0) / 900));
        if (q < 1) {
          placeMarker(u * ease.inOut(q), cur);
          anim = requestAnimationFrame(sweep);
          return;
        }
        anim = 0;
        paint();                               // landed: the pick is shown and its bar pulses
        pulse(cols[cur.rankOf[pickFor(cur, u)]]?.fill);
      };
      anim = requestAnimationFrame(sweep);
    }

    // Sample 100: 100 draws from the same generator, tallied per token. The single-draw readouts are
    // cleared, because the pick and U belong to one draw and the histogram to the hundred.
    function sampleMany(n = 100) {
      cancelAnimationFrame(anim);
      anim = 0;
      const D = cur, counts = new Map();
      for (let i = 0; i < n; i++) {
        const v = pickFor(D, rng());
        counts.set(v, (counts.get(v) ?? 0) + 1);
      }
      view.hist = { counts, n, key: keyOf(D) };
      view.lastU = null;
      paint();
    }

    function setSettings(patch) {
      Object.assign(view, patch);
      tauS.set(view.tau);
      kS.set(view.k);
      pS.set(view.p);
      paint();
    }

    // Steps set one thing each: the raw softmax, a temperature (flatter), the filters at that temperature, then a
    // draw. A step the reader picks (story dot, arrow or key) ends the walk. A jump made by a control keeps the
    // values the reader has set (see own()).
    function setStep(i) {
      if (!autoDriving && !keeping) stopAuto();
      if (i !== DRAW_STEP) { cancelAnimationFrame(anim); anim = 0; }     // leaving the draw ends its sweep
      const from = view.step;
      view.step = i;
      if (i === DRAW_STEP) {
        // Reached before the filters step with the filters off: the draw uses the filtered distribution, as the walk does.
        if (from < 2 && view.k === 0 && view.p >= 1) setSettings(SETTINGS[2]);
        draw();
        return;
      }
      if (keeping) paint();
      else setSettings(SETTINGS[i]);
    }

    // ------------------------------------------------------------------ mount
    ctx.setSteps(STEPS, setStep);
    paint();
    appear([header.el, picker.el, plot, side], { stagger: MOTION.stagger });

    // The walk runs once on first view, so the scene explains itself: one step every MOTION.step ms, the first one
    // read for a full step after the zoom has finished (as lm-head.js does).
    function walk() {
      auto = 0;
      if (view.step >= STEPS.length - 1) return;
      autoDriving = true;
      try { ctx.setStep(view.step + 1); } finally { autoDriving = false; }
      if (view.step < STEPS.length - 1) auto = setTimeout(walk, MOTION.step);
    }
    auto = setTimeout(walk, ZOOM_MS + MOTION.step);
    el.addEventListener('pointerdown', stopAuto);
    el.addEventListener('atlas:interrupt', stopAuto);

    const offs = [ctx.on('trace', paint), ctx.on('select', paint)];
    return {
      destroy() {
        stopAuto();
        cancelAnimationFrame(anim);
        el.removeEventListener('pointerdown', stopAuto);
        el.removeEventListener('atlas:interrupt', stopAuto);
        el.removeEventListener('click', onPick);
        hideTip();
        offs.forEach((f) => f());
        st.destroy();
      },
    };
  },
};

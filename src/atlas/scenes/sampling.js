// Sampling: how the next character is chosen from the model's distribution.
// Hero: one bar per vocabulary entry, sorted by score, for the selected position. Temperature,
// top-k and top-p reshape the bars through engine.samplingDistribution; removed characters turn
// grey. A draw is inverse-CDF sampling: a seeded uniform U (mulberry32) lands on a unit strip cut
// into the final probabilities, and the segment it lands in is the character that is chosen.
// The histogram counts draws from one distribution. It is marked stale when the position, the
// settings, the vocabulary or the logits row change.
// Each step owns the controls it teaches. A step sets its own settings when it is chosen; moving a
// control on an earlier step moves the scene to the step that shows it, so every caption and equation
// describes the bars on screen. A bar carries data-trace (the atlas hover card and maths) only while it
// equals the traced softmax (tau 1, no filters). Otherwise a tooltip gives its sampling weight.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, checkBadge, appear, pulse, useStyle, ZONES, MOTION } from '../kit.js';
import { samplingDistribution, sampleFrom, argmax, mulberry32 } from '../../engine/index.js';
import { h, fmt, num, texHTML, slider, button, reducedMotion, ease, showTip, hideTip } from '../../ui/index.js';

useStyle('sampling', `
.sp-plot { position: absolute; border-radius: 12px; background: color-mix(in srgb, var(--surface) 74%, transparent);
  border: 1px solid color-mix(in srgb, var(--c-head) 26%, var(--line));
  box-shadow: 0 16px 36px rgba(0, 0, 0, .14), 0 2px 0 color-mix(in srgb, var(--c-head) 16%, transparent); }
.sp-plot.still * { transition: none !important; }
.sp-grid { position: absolute; border-top: 1px dashed var(--line-strong); opacity: .7; }
.sp-grid.sp-base { border-top: 1px solid var(--line-strong); opacity: 1; }
.sp-tick { position: absolute; width: 46px; text-align: right; font: 500 12px/1 var(--font-mono); color: var(--fg-3); transform: translateY(-50%); }
.sp-sigma { font: 600 14px/1 var(--font-mono); color: var(--fg-2); }
.sp-col { position: absolute; }
.sp-fill { position: absolute; left: 16%; right: 16%; bottom: 0; border-radius: 4px 4px 2px 2px; background: var(--c-head);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .22), 0 6px 12px rgba(0, 0, 0, .12);
  transition: height .6s cubic-bezier(.65, 0, .35, 1), background-color .4s, opacity .4s, box-shadow .4s; }
.sp-fill.off { background: var(--c-mask); opacity: .42; box-shadow: none; }
.sp-fill.pick { outline: 2px solid var(--accent); outline-offset: 2px; }
.sp-val { position: absolute; left: -12px; right: -12px; text-align: center; font: 600 12px/1 var(--font-mono); color: var(--fg-2);
  white-space: nowrap; transition: bottom .6s cubic-bezier(.65, 0, .35, 1); }
.sp-val.top { color: var(--fg); }
.sp-frame { position: absolute; border: 1px dashed var(--fg-3); border-radius: 8px; pointer-events: none; }
.sp-flab { position: absolute; font: 500 12px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.sp-chars { position: absolute; }
.sp-char { position: absolute; top: 0; height: 18px; line-height: 18px; text-align: center; font: 600 15px/18px var(--font-mono);
  color: var(--fg); transition: color .3s, opacity .3s; }
.sp-char.off { color: var(--fg-3); opacity: .5; }
.sp-char.pick { color: var(--accent); }
.sp-track { position: absolute; border-radius: 6px; background: var(--surface-2); border: 1px solid var(--line); overflow: hidden; }
.sp-seg { position: absolute; top: 0; bottom: 0; background: var(--c-head); box-shadow: inset -1.5px 0 0 var(--surface);
  transition: left .6s cubic-bezier(.65, 0, .35, 1), width .6s cubic-bezier(.65, 0, .35, 1), background-color .3s; }
.sp-seg.alt { background: color-mix(in srgb, var(--c-head) 70%, var(--surface)); }
.sp-seg.hot { background: color-mix(in srgb, var(--c-head) 55%, var(--fg)); }
.sp-seg.pick { background: var(--accent); }
.sp-marker { position: absolute; width: 2px; border-radius: 1px; background: var(--accent); opacity: 0;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent); transition: opacity .3s; }
.sp-pill { position: absolute; transform: translateX(-50%); padding: 3px 8px; border-radius: 999px; opacity: 0;
  font: 600 12px/1.2 var(--font-mono); color: var(--surface); background: var(--accent); white-space: nowrap; transition: opacity .3s; }
.sp-hcol { position: absolute; }
.sp-hbar { position: absolute; left: 16%; right: 16%; bottom: 0; border-radius: 3px 3px 1px 1px; background: var(--c-head); opacity: .9;
  transition: height .6s cubic-bezier(.65, 0, .35, 1), opacity .4s; }
.sp-hcol.stale .sp-hbar { opacity: .28; }
.sp-hexp { position: absolute; left: 8%; right: 8%; height: 2px; border-radius: 1px; background: var(--fg); opacity: 0;
  transition: bottom .6s cubic-bezier(.65, 0, .35, 1), opacity .4s; }
.sp-hcnt { position: absolute; left: -10px; right: -10px; text-align: center; font: 600 11px/1 var(--font-mono); color: var(--fg-2);
  transition: bottom .6s cubic-bezier(.65, 0, .35, 1); }
.sp-hcap { position: absolute; font: 500 12px/1.2 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.sp-note { position: absolute; display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
  font: 500 13px/1.2 var(--font-mono); color: var(--fg-2); white-space: nowrap;
  background: color-mix(in srgb, var(--c-mask) 16%, var(--surface)); border: 1px solid color-mix(in srgb, var(--c-mask) 45%, transparent); }
.sp-note[hidden] { display: none; }
.sp-note::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: var(--c-mask); flex: none; }
.sp-point { display: block; white-space: normal; max-width: 640px; line-height: 1.45; padding: 8px 14px; border-radius: 12px;
  background: color-mix(in srgb, var(--surface) 88%, transparent); border-color: var(--line); color: var(--fg-2); }
.sp-point::before { display: inline-block; vertical-align: middle; margin: -2px 8px 0 0; background: var(--accent); }
.sp-point[hidden] { display: none; }
.sp-check { position: absolute; }
.sp-go { background: none; border: 0; padding: 0; font: inherit; color: var(--accent); cursor: pointer;
  text-decoration: underline; text-underline-offset: 4px; }
.sp-go:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
.sp-side { position: absolute; display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; align-content: start; }
.sp-side > * { min-width: 0; }
.sp-ctl { display: grid; gap: 4px; }
.sp-ctl .field { gap: 2px; }
.sp-btns { display: flex; gap: 8px; align-items: center; }
.sp-seed { margin-left: auto; font: 500 12px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; }
.sp-stats { display: grid; gap: 3px; }
.sp-stat { font: 500 14px/1.4 var(--font-mono); color: var(--fg-2); }
.sp-stat b { color: var(--fg); font-weight: 600; }
.sp-eq-main { font-size: 22px; overflow-x: auto; }
.sp-eq-live { font-size: 16px; padding: 8px 12px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent);
  border: 1px solid var(--line); overflow-x: auto; }
`);

const STEPS = ['Distribution', 'Temperature', 'Top-k and top-p', 'Draw'];
const SEED = 1337;                                          // mulberry32 seed: the same draws on every visit
const PLOT = { x: ZONES.hero.x, y: ZONES.hero.y, w: 1000, h: 470 };
const SIDE = { x: ZONES.side.x, y: ZONES.hero.y, w: ZONES.side.w };
const GL = 56, GR = 24, PW = PLOT.w - GL - GR;              // left gutter, right margin, width of every row
const TOP = 56, BH = 204, BASE = TOP + BH;                  // bars: top edge, height, baseline
const CHR = BASE + 6;                                       // character labels under the bars
const PILL = CHR + 24;                                      // readout of U, above the strip
const STRIP = { y: 312, h: 22 };                            // cumulative strip, [0, 1]
const HIST = { y: 352, h: 72 };                             // histogram of 100 draws
const NEAR = 0.9;                                           // a top probability this high is "nearly certain" (top-p 0.9 keeps one character then)
// What a step sets when it is chosen from the step bar. Step 3 draws with the settings on screen.
const SETTINGS = [{ tau: 1, k: 0, p: 1 }, { tau: 2, k: 0, p: 1 }, { k: 8, p: 0.9 }];

const px = (n) => `${n}px`;
const r2 = (x) => Math.round(x * 100) / 100;
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
const pct = (p) => (p >= 0.995 ? '100%' : p >= 0.1 ? `${Math.round(p * 100)}%` : p >= 0.001 ? `${(p * 100).toFixed(1)}%` : p > 0 ? '<0.1%' : '0%');
const entropy = (p) => -p.reduce((a, x) => (x > 0 ? a + x * Math.log(x) : a), 0);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const TEX_ESC = { '\\': '\\backslash', '{': '\\{', '}': '\\}', '$': '\\$', '&': '\\&', '#': '\\#', '^': '\\textasciicircum',
  '_': '\\_', '%': '\\%', '~': '\\textasciitilde', "'": "\\text{'}", ' ': '\\text{space}', '\n': '\\text{newline}', '\t': '\\text{tab}' };
const texChar = (ch) => `\\mathtt{${TEX_ESC[ch] ?? ch}}`;

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const rng = mulberry32(SEED);
    const view = { tau: 1, k: 0, p: 1, step: 0, lastU: null, hist: null };
    let cur = null, builtV = -1, anim = 0, auto = 0, autoDriving = false, keeping = false, lastPointer = '';
    let plot, cols = [], chars = [], segs = [], hcols = [], marker, pill, histCap;
    const stopAuto = () => { clearTimeout(auto); auto = 0; };

    // ------------------------------------------------------------------ shared parts
    const header = sceneHeader(st, {
      chips: [['LM head', 'head']], title: 'Sampling',
      sub: 'Every vocabulary entry gets a score. Sampling decides which one becomes the next character.',
    });
    const picker = tokenPicker(st, ctx, { label: 'after', kind: 'head' });
    const cap = sceneCaption(st);

    plot = h('div', { class: 'sp-plot', style: { width: px(PLOT.w), height: px(PLOT.h) } });
    st.add(plot, { x: PLOT.x, y: PLOT.y });
    // Top right of the plot: the mask note (a bidirectional mask applies) or the pointer to a spread position.
    const note = h('div', { class: 'sp-note', hidden: true });
    const ptr = h('div', { class: 'sp-note sp-point', hidden: true });
    const check = h('div', { class: 'sp-check' });
    st.add(note, { x: PLOT.x + PLOT.w - 16, y: PLOT.y + 14, anchor: 'tr' });
    st.add(ptr, { x: PLOT.x + PLOT.w - 16, y: PLOT.y + 14, anchor: 'tr' });
    st.add(check, { x: PLOT.x + PLOT.w - 16, y: PLOT.y + PLOT.h - 14, anchor: 'br' });

    // A bar that is not the traced softmax has no data-trace, so the atlas card would show the wrong number.
    // Its tooltip gives the sampling weight instead. Leaving the plot hides it.
    plot.addEventListener('pointermove', (e) => {
      const col = e.target.closest?.('.sp-col');
      if (!col || col.dataset.trace || !cur) { hideTip(); return; }
      showTip(e.clientX, e.clientY, sampleTip(cur, Number(col.dataset.v)));
    });
    plot.addEventListener('pointerleave', hideTip);

    // ------------------------------------------------------------------ side column
    // A control moved on an earlier step moves the scene to the step that shows it (the Temperature step for
    // tau, the filters step for top-k and top-p). The values the reader set are kept.
    const own = (step) => { if (view.step < step) jumpTo(step); else paint(); };
    const jumpTo = (i) => { keeping = true; try { ctx.setStep(i); } finally { keeping = false; } };
    const tauS = slider({ id: 'sampling-temp', label: 'temperature τ', min: 0.1, max: 2, step: 0.05, value: view.tau,
      format: (v) => v.toFixed(2), onInput: (v) => { stopAuto(); view.tau = r2(v); own(1); } });
    const kS = slider({ id: 'sampling-topk', label: 'top-k', min: 0, max: ctx.state.logits.shape[1], step: 1, value: view.k,
      format: (v) => (v ? String(v) : 'off'), onInput: (v) => { stopAuto(); view.k = v; own(2); } });
    const pS = slider({ id: 'sampling-topp', label: 'top-p', min: 0.1, max: 1, step: 0.05, value: view.p,
      format: (v) => v.toFixed(2), onInput: (v) => { stopAuto(); view.p = r2(v); own(2); } });
    const statH = h('div', { class: 'sp-stat' }), statG = h('div', { class: 'sp-stat' }), statU = h('div', { class: 'sp-stat' });
    const eqMain = h('div', { class: 'sp-eq-main' }), eqLive = h('div', { class: 'sp-eq-live' });
    const side = h('div', { class: 'sp-side', style: { width: px(SIDE.w) } },
      h('div', { class: 'sp-ctl' }, tauS.el, kS.el, pS.el),
      h('div', { class: 'sp-btns' },
        button('Sample', () => { stopAuto(); draw(); }, { variant: 'primary', title: 'Draw one character from the current distribution' }),
        button('Sample 100', () => { stopAuto(); sampleMany(100); }, { title: 'Draw 100 characters and fill the histogram' }),
        h('span', { class: 'sp-seed', title: 'U comes from the mulberry32 generator with this seed' }, `seed ${SEED}`)),
      h('div', { class: 'sp-stats' }, statH, statG, statU),
      eqMain, eqLive);
    st.add(side, { x: SIDE.x, y: SIDE.y });

    // A position named in the caption or in the pointer can be clicked to select it.
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
      const pRaw = samplingDistribution(z, { temperature: view.tau, topK: 0, topP: 1 });   // softmax(z / tau)
      const pFin = samplingDistribution(z, { temperature: view.tau, topK: view.k, topP: view.p });
      const pRef = samplingDistribution(z, { temperature: 1 });                             // untouched softmax
      const traced = s.trace.probs.data.subarray(t * V, (t + 1) * V);
      let maxDiff = 0;
      for (let v = 0; v < V; v++) maxDiff = Math.max(maxDiff, Math.abs(pRef[v] - traced[v]));
      // Sorting by score is the same at every temperature, so the bars keep their order when tau changes.
      const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => z[b] - z[a] || a - b);
      const rankOf = new Array(V), start = new Array(V);
      let acc = 0;
      order.forEach((v, r) => { rankOf[v] = r; start[r] = acc; acc += pFin[v]; });
      const kept = order.filter((v) => pFin[v] > 0);
      const mass = kept.reduce((a, v) => a + pRaw[v], 0);
      const mask = s.trace.mask;
      const g = argmax(z);
      const topFin = pFin[g];                                                                // the greedy character is kept
      const exact = pFin.every((x, v) => Math.abs(x - pRef[v]) < 1e-9);                      // the bars are the traced softmax
      // Nearly certain here: the most spread-out other position that is not nearly certain, at these settings.
      let spread = null;
      if (topFin >= NEAR) {
        let bestH = -1;
        for (let r = 0; r < T; r++) {
          if (r === t) continue;
          const q = samplingDistribution(row(r), { temperature: view.tau, topK: view.k, topP: view.p });
          const top = Math.max(...q);
          if (top >= NEAR) continue;
          const H = entropy(q);
          if (H > bestH) { bestH = H; spread = { r, top }; }
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

    // The draw: the same rule as engine.sampleFrom, applied to the kept characters in rank order.
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
      const colW = PW / V;
      const kids = [];
      for (const [y, base, label] of [[TOP, false, '1.0'], [TOP + BH / 2, false, '0.5'], [BASE, true, '0']]) {
        kids.push(h('div', { class: base ? 'sp-grid sp-base' : 'sp-grid', style: { left: px(GL), width: px(PW), top: px(y) } }));
        kids.push(h('div', { class: 'sp-tick', style: { left: px(4), top: px(y) } }, label));
      }
      kids.push(h('div', { class: 'sp-tick sp-sigma', style: { left: px(4), top: px(STRIP.y + STRIP.h / 2) } }, 'Σp'));
      // the histogram band has a baseline and a label, so it reads as a frame for the counts before the first draw
      kids.push(h('div', { class: 'sp-grid sp-base', style: { left: px(GL), width: px(PW), top: px(HIST.y + HIST.h) } }));
      kids.push(h('div', { class: 'sp-tick', style: { left: px(4), top: px(HIST.y + HIST.h / 2) } }, 'draws'));
      // the greedy column is the argmax (the first column, since the bars are sorted by score), framed and labelled
      const frame = h('div', { class: 'sp-frame', style: { left: px(GL - 4), width: px(colW + 8), top: px(TOP - 26), height: px(CHR + 24 - (TOP - 26)) } });
      const frameLab = h('div', { class: 'sp-flab', style: { left: px(GL - 4), top: px(12) } }, 'greedy');
      cols = []; chars = []; segs = []; hcols = [];
      for (let r = 0; r < V; r++) {
        const x = GL + r * colW;
        const fill = h('div', { class: 'sp-fill' }), val = h('div', { class: 'sp-val' });
        const col = h('div', { class: 'sp-col', style: { left: px(x), width: px(colW), top: px(TOP), height: px(BH) } }, fill, val);
        cols.push({ fill, val, el: col });
        chars.push(h('div', { class: 'sp-char', style: { left: px(r * colW), width: px(colW) } }));
        segs.push(h('div', { class: 'sp-seg' }));
        const bar = h('div', { class: 'sp-hbar' }), exp = h('div', { class: 'sp-hexp' }), cnt = h('div', { class: 'sp-hcnt' });
        hcols.push({ el: h('div', { class: 'sp-hcol', style: { left: px(x), width: px(colW), top: px(HIST.y), height: px(HIST.h) } }, bar, exp, cnt), bar, exp, cnt });
      }
      const chRow = h('div', { class: 'sp-chars', dataset: { trace: 'logits' }, style: { left: px(GL), top: px(CHR), width: px(PW), height: '18px' } }, chars);
      const track = h('div', { class: 'sp-track', style: { left: px(GL), top: px(STRIP.y), width: px(PW), height: px(STRIP.h) } }, segs);
      marker = h('div', { class: 'sp-marker', style: { top: px(STRIP.y - 8), height: px(STRIP.h + 16) } });
      pill = h('div', { class: 'sp-pill', style: { top: px(PILL) } });
      histCap = h('div', { class: 'sp-hcap', style: { left: px(GL), top: px(HIST.y + HIST.h + 12) } });
      plot.replaceChildren(...kids, frame, frameLab, ...cols.map((c) => c.el), chRow, track, marker, pill, ...hcols.map((c) => c.el), histCap);
    }

    // Move the U marker along the strip; with D, the segment under it is highlighted.
    function placeMarker(x, D) {
      const on = x != null;
      marker.style.opacity = on ? '1' : '0';
      pill.style.opacity = on ? '1' : '0';
      segs.forEach((sg) => sg.classList.remove('hot'));
      if (!on) return;
      marker.style.left = px(GL + x * PW - 1);
      pill.style.left = px(GL + Math.min(PW - 40, Math.max(40, x * PW)));
      pill.textContent = `U = ${x.toFixed(4)}`;
      if (D) segs[D.rankOf[pickFor(D, x)]].classList.add('hot');
    }

    // ------------------------------------------------------------------ paint
    function paint() {
      const D = compute();
      cur = D;
      if (D.V !== builtV) buildPlot(D.V);
      kS.input.max = String(D.V);
      if (kS.get() !== view.k) kS.set(view.k);
      plot.classList.toggle('still', reducedMotion());
      const sweeping = anim !== 0;
      const hist = view.hist, stale = !!hist && hist.key !== keyOf(D);
      const pick = view.lastU == null || sweeping ? -1 : D.rankOf[pickFor(D, view.lastU)];
      let ki = 0;                                              // index among the kept characters (alternating tints)
      for (let r = 0; r < D.V; r++) {
        const v = D.order[r], kept = D.pFin[v] > 0, p = kept ? D.pFin[v] : D.pRaw[v];
        const c = cols[r], ch = chars[r], sg = segs[r], hc = hcols[r], cell = `${D.t},${v},0`;
        sg.classList.toggle('alt', kept && ki++ % 2 === 1);
        c.el.dataset.v = String(v);
        if (D.exact) { c.el.dataset.trace = 'probs'; c.el.dataset.cell = cell; }
        else { delete c.el.dataset.trace; delete c.el.dataset.cell; }
        c.fill.style.height = px(p * BH);
        c.fill.classList.toggle('off', !kept);
        c.fill.classList.toggle('pick', r === pick);
        c.val.style.bottom = px(p * BH + 4);
        c.val.classList.toggle('top', r === 0);
        c.val.textContent = kept && (r === 0 || r === pick || (r < 3 && p >= 0.05)) ? pct(p) : '';
        ch.textContent = D.s.tokenizer.label(v);
        ch.dataset.cell = cell;
        ch.classList.toggle('off', !kept);
        ch.classList.toggle('pick', r === pick);
        sg.style.left = px(D.start[r] * PW);
        sg.style.width = px(D.pFin[v] * PW);
        sg.classList.toggle('pick', r === pick);
        sg.classList.remove('hot');
        const n = hist ? hist.n : 0, cnt = hist ? (hist.counts.get(v) ?? 0) : 0, f = n ? cnt / n : 0;
        hc.el.classList.toggle('stale', stale);
        hc.bar.style.height = px(f * HIST.h);
        hc.cnt.textContent = cnt ? String(cnt) : '';
        hc.cnt.style.bottom = px(f * HIST.h + 3);
        hc.exp.style.bottom = px(D.pFin[v] * HIST.h);
        hc.exp.style.opacity = kept && D.pFin[v] >= 0.001 ? '1' : '0';
      }
      histCap.textContent = !hist ? 'Sample 100 fills this histogram. The ticks mark the probabilities.'
        : stale ? `${hist.n} draws from an earlier distribution. Press Sample 100 to redraw.`
          : `${hist.n} draws. The ticks mark the probabilities.`;
      note.hidden = !D.seesNext;
      if (D.seesNext) note.textContent = `Bidirectional mask: position ${D.t} can see the next character, so these bars are not a prediction.`;
      if (!sweeping) placeMarker(view.lastU, null);
      paintText(D, sweeping);
    }

    function paintText(D, sweeping) {
      const lab = (v) => esc(D.s.tokenizer.label(v));
      const tex = (v) => texChar(D.s.tokenizer.token(v));
      const g = D.greedy, tau = view.tau, step = view.step, hist = view.hist;
      const U = sweeping ? null : view.lastU, pv = U == null ? null : pickFor(D, U), pr = pv == null ? -1 : D.rankOf[pv];
      const Z1 = D.z.reduce((a, x) => a + Math.exp(x), 0);
      const Zt = D.z.reduce((a, x) => a + Math.exp(x / tau), 0);
      const at = esc(ctx.label(D.t));
      const p2 = view.p.toFixed(2), k = view.k, kOn = k > 0, pOn = view.p < 1, nF = D.kept.length;
      const top = `<b>“${lab(g)}”</b>`;

      // Step 0, the softmax at tau 1 with no filters (the step's own settings: they hold until another step is chosen).
      const s0 = D.topFin >= NEAR
        ? `${top} has ${pct(D.topFin)} of the mass, so the choice is nearly certain.`
        : `${top} has ${pct(D.topFin)}${D.runner == null ? '' : `, and the runner-up <b>“${lab(D.runner)}”</b> has ${pct(D.pFin[D.runner])}`}.`;

      // Step 1, the temperature with no filters. The numbers are the bars' own.
      const change = tau === 1
        ? `At τ = 1 the bars are the plain softmax, with entropy ${D.H1.toFixed(2)} nats.`
        : `The top character, ${top}, moves from ${pct(D.pRef[g])} at τ = 1 to ${pct(D.topFin)} at τ = ${tau.toFixed(2)}, and the entropy from ${D.H1.toFixed(2)} to ${D.H.toFixed(2)} nats.`;

      // Step 2, the filters at the current temperature. The engine applies top-k first, then top-p inside the top-k set.
      const kTxt = k === 1 ? 'keeps the most likely character' : `keeps the ${k} most likely characters`;
      let filt;
      if (!kOn && !pOn) filt = 'Top-k and top-p are both off.';
      else if (!pOn) filt = `Top-k ${kTxt}; top-p is off.`;
      else if (!kOn) filt = nF === 1
        ? `Top-p keeps the smallest set reaching ${p2}. Here that is ${top} alone, since it already reaches ${p2}.`
        : `Top-p keeps the smallest set reaching ${p2}, which is ${plural(nF, 'character')}.`;
      else if (k === 1) filt = `Top-k ${kTxt}, so top-p does not bind.`;
      else if (nF === 1) filt = `Top-k ${kTxt}, and top-p cuts that set to ${top}, which reaches ${p2} on its own.`;
      else if (nF < k) filt = `Top-k ${kTxt}, and top-p cuts that set to the ${plural(nF, 'character')} reaching ${p2}.`;
      else filt = `Top-k ${kTxt}, and top-p at ${p2} cuts none of them.`;
      const cut = nF < D.V ? 'Grey bars are removed, and the rest are renormalised to sum to 1.' : 'Nothing is removed, so the bars are unchanged.';

      const captions = [
        `Softmax turns the ${D.V} scores after <b>“${at}”</b> into probabilities that sum to 1. ${s0}`,
        `Dividing every score by τ before the softmax: below 1 sharpens the bars, above 1 flattens them. ${change}`,
        `${filt} ${cut}`,
        U == null
          ? 'A seeded uniform number U picks the character whose segment on the strip contains it. Greedy would always take the tallest bar.'
          : `U = ${U.toFixed(4)} falls in the segment of <b>“${lab(pv)}”</b>, so that character comes next. Greedy would always take the tallest bar.`,
      ];
      cap.set(captions[step]);

      const main = [
        String.raw`\textstyle p_v = \frac{e^{z_v}}{\sum_k e^{z_k}}`,
        String.raw`\textstyle p_v(\tau) = \frac{e^{z_v / \tau}}{\sum_k e^{z_k / \tau}}`,
        String.raw`\textstyle p'_v = \frac{p_v}{\sum_{u \in S} p_u} \quad (v \in S)`,
        String.raw`\textstyle \sum_{i<j} p'_i \le U < \sum_{i \le j} p'_i`,
      ][step];
      const zg = D.z[g];
      const live = [
        String.raw`\textstyle p_{${tex(g)}} = \frac{e^{${num(zg, 2)}}}{${num(Z1, 2)}} = ${num(D.pRef[g], 2)}`,
        String.raw`\textstyle p_{${tex(g)}}(\tau = ${tau.toFixed(2)}) = \frac{e^{${num(zg / tau, 2)}}}{${num(Zt, 2)}} = ${num(D.pRaw[g], 2)}`,
        String.raw`\sum_{u \in S} p_u = ${num(D.mass, 3)}, \quad |S| = ${D.kept.length}`,
        U == null ? String.raw`U \sim \mathrm{Uniform}[0, 1)`
          : String.raw`${num(D.start[pr], 2)} \le U = ${num(U, 4)} < ${num(D.start[pr] + D.pFin[pv], 2)} \Rightarrow ${tex(pv)}`,
      ][step];
      eqMain.innerHTML = texHTML(main, { display: true });
      eqLive.innerHTML = texHTML(live, { display: true });

      statH.innerHTML = `Entropy <b>${D.H.toFixed(2)}</b> nats, about <b>${Math.exp(D.H).toFixed(1)}</b> equally likely characters.`;
      statG.innerHTML = `Greedy (argmax) takes ${top}, at <b>${pct(D.topFin)}</b>.`;
      statU.innerHTML = sweeping ? 'Drawing U from the seeded generator…'
        : U == null ? (hist ? `The histogram holds ${hist.n} draws. Press <b>Sample</b> to draw one and see its U.` : 'No draw yet. Press <b>Sample</b>.')
          : `U = <b>${U.toFixed(4)}</b> lands on “<b>${lab(pv)}</b>”, the segment from ${pct(D.start[pr])} to ${pct(D.start[pr] + D.pFin[pv])}.`;

      // The pointer, on every step: the selected position is nearly certain at these settings, so the bars hardly move
      // there. It points to a position where they do move. Hidden when a bidirectional mask owns the note.
      let pointer = '';
      if (!D.seesNext && D.topFin >= NEAR) {
        pointer = D.spread
          ? `Position ${D.t} is nearly certain at these settings. Select <button type="button" class="sp-go" data-pos="${D.spread.r}">position ${D.spread.r}</button> (top ${pct(D.spread.top)}) to see the settings act.`
          : `Position ${D.t} is nearly certain at these settings. No other position in this text has a top probability below ${pct(NEAR)}.`;
      }
      ptr.hidden = !pointer;
      if (pointer !== lastPointer) { lastPointer = pointer; ptr.innerHTML = pointer; }

      check.replaceChildren(checkBadge(D.maxDiff, 0, { label: 'untempered softmax = traced probs' }));
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
        paint();                               // landed: the pick is shown and the bar pulses
        pulse(cols[cur.rankOf[pickFor(cur, u)]]?.fill);
      };
      anim = requestAnimationFrame(sweep);
    }

    // Sample 100: 100 draws from the same generator, tallied per character. The single-draw readouts are
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

    // Steps set one thing each: the raw softmax, a temperature (flatter), the filters at that temperature,
    // then a draw. A step the reader picks (story dot, arrow or key) ends the walk. A jump made by a control
    // keeps the values the reader has set (see own()).
    function setStep(i) {
      if (!autoDriving && !keeping) stopAuto();
      view.step = i;
      if (i === 3) { paint(); draw(); return; }
      if (keeping) paint();
      else setSettings(SETTINGS[i]);
    }

    // ------------------------------------------------------------------ mount
    ctx.setSteps(STEPS, setStep);
    paint();
    appear([header.el, picker.el, plot, side], { stagger: MOTION.stagger });

    // The walk runs once on first view, so the scene explains itself: one step every MOTION.step ms.
    function walk() {
      auto = 0;
      if (view.step >= STEPS.length - 1) return;
      autoDriving = true;
      try { ctx.setStep(view.step + 1); } finally { autoDriving = false; }
      if (view.step < STEPS.length - 1) auto = setTimeout(walk, MOTION.step);
    }
    auto = setTimeout(walk, 900);
    el.addEventListener('pointerdown', stopAuto);

    const offs = [ctx.on('trace', paint), ctx.on('select', paint)];
    return {
      destroy() {
        stopAuto();
        cancelAnimationFrame(anim);
        el.removeEventListener('pointerdown', stopAuto);
        el.removeEventListener('click', onPick);
        hideTip();
        offs.forEach((f) => f());
        st.destroy();
      },
    };
  },
};

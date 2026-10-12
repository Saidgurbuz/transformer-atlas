// Softmax for one query row of one head, in five steps on the real numbers:
//   scores → subtract the max → exponentiate → normalise → the full pattern.
// Reference scene for contributors: see docs/ATLAS_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, mat, appear, pulse, countTo, useStyle, MOTION } from '../kit.js';
import { h, fmt, texHTML, num } from '../../ui/index.js';

const STEPS = ['Scores', 'Subtract max', 'Exponentiate', 'Normalise', 'The pattern'];
const BX = 110, BY = 290, BW = 960, BH = 390;     // bar area in design units

useStyle('softmax', `
.sm-bars { position: relative; border-left: 1px solid var(--line-strong); }
.sm-zero { position: absolute; left: 0; right: 0; height: 0; border-top: 1px dashed var(--line-strong); transition: top .6s cubic-bezier(.65,0,.35,1); }
.sm-bar { position: absolute; border-radius: 4px 4px 2px 2px; background: linear-gradient(180deg, color-mix(in srgb, var(--c-attn) 92%, white), var(--c-attn));
  box-shadow: 0 6px 14px rgba(0,0,0,.18); transition: top .6s cubic-bezier(.65,0,.35,1), height .6s cubic-bezier(.65,0,.35,1), background .4s; }
.sm-bar.neg { background: linear-gradient(0deg, color-mix(in srgb, var(--heat-neg) 85%, white), var(--heat-neg)); border-radius: 2px 2px 4px 4px; }
.sm-bar.max { outline: 2px solid var(--accent); outline-offset: 2px; }
.sm-bar.masked { background: repeating-linear-gradient(135deg, var(--c-mask) 0 2px, transparent 2px 5px); box-shadow: none; }
.sm-val { position: absolute; left: 50%; translate: -50% 0; font: 600 12px/1 var(--font-mono); color: var(--fg-2); white-space: nowrap; transition: top .6s cubic-bezier(.65,0,.35,1); }
.sm-keys { display: flex; }
.sm-key { text-align: center; font: 600 16px/1 var(--font-mono); color: var(--fg-2); transition: opacity .3s; }
.sm-key.dim { opacity: .3; }
.sm-stack { position: relative; width: 34px; border-radius: 8px; background: var(--surface-2); border: 1px solid var(--line); overflow: hidden; }
.sm-seg { position: absolute; left: 0; right: 0; background: var(--c-attn); border-top: 1px solid color-mix(in srgb, var(--bg) 60%, transparent); transition: height .6s cubic-bezier(.65,0,.35,1), bottom .6s cubic-bezier(.65,0,.35,1); }
.sm-eq { width: 380px; display: grid; gap: 10px; }
.sm-eq-main { font-size: 26px; }
.sm-eq-live { font-size: 19px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid var(--line); overflow-x: auto; }
`);

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const { l, h: hd } = ctx.params;
    let step = 0;

    // ---------------------------------------------------------------- header, picker, caption (shared kit parts)
    const header = sceneHeader(st, { chips: [[`Layer ${l}`, 'resid'], [`Head ${hd}`, 'attn']], title: 'Softmax' });
    const picker = tokenPicker(st, ctx, { label: 'query' });

    // bars
    const bars = h('div', { class: 'sm-bars', style: { width: `${BW}px`, height: `${BH}px` } });
    const zero = h('div', { class: 'sm-zero' });
    const keyRow = h('div', { class: 'sm-keys', style: { width: `${BW}px` } });
    bars.append(zero);
    st.add(bars, { x: BX, y: BY });
    st.add(keyRow, { x: BX, y: BY + BH + 10 });
    st.add(h('div', { class: 'k-label', style: { color: 'var(--fg-3)' } }, 'keys →'), { x: BX + BW + 8, y: BY + BH + 14 });

    // stacked "sums to 1" column
    const stack = h('div', { class: 'sm-stack', style: { height: `${BH}px` } });
    const stackLab = h('div', { class: 'k-label' }, 'Σ = 1');
    st.add(stack, { x: BX + BW + 50, y: BY });
    st.add(stackLab, { x: BX + BW + 64, y: BY - 34, anchor: 'tc' });

    // right column: equation and readouts
    const eq = h('div', { class: 'sm-eq' });
    st.add(eq, { x: 1180, y: BY - 10 });
    const patternBox = h('div');
    st.add(patternBox, { x: 1180, y: 548 });
    const cap = sceneCaption(st, { w: 960 });

    let barEls = [], keyEls = [], stackEls = [], patternMat = null;

    function data() {
      const s = ctx.state, T = s.ids.length, q = Math.min(s.sel.token, T - 1);
      const masked = s.trace[`L${l}.attn.masked`], probs = s.trace[`L${l}.attn.probs`];
      const off = (hd * T + q) * T;
      const sc = Array.from(masked.data.subarray(off, off + T));
      const p = Array.from(probs.data.subarray(off, off + T));
      const vis = sc.filter((v) => v !== -Infinity);
      const m = Math.max(...vis);
      const ex = sc.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
      const Z = ex.reduce((a, b) => a + b, 0);
      return { T, q, sc, p, m, ex, Z, nVis: vis.length };
    }

    function buildStatic() {
      const s = ctx.state, d = data();
      const colW = BW / d.T;
      bars.querySelectorAll('.sm-bar').forEach((b) => b.remove());
      barEls = d.sc.map((_, j) => {
        const b = h('div', { class: 'sm-bar', style: { left: `${j * colW + colW * 0.15}px`, width: `${colW * 0.7}px` }, dataset: { cell: `${d.q},${j},${hd}` } },
          h('span', { class: 'sm-val' }));
        bars.append(b);
        return b;
      });
      keyEls = d.sc.map((_, j) => h('div', { class: 'sm-key', style: { width: `${colW}px`, fontSize: d.T > 32 ? '11px' : '16px' } }, ctx.label(j)));
      keyRow.replaceChildren(...keyEls);
      stackEls = d.p.map((_, j) => h('div', { class: 'sm-seg', dataset: { cell: `${d.q},${j},${hd}` } }));
      stack.replaceChildren(...stackEls);
      stack.dataset.trace = `L${l}.attn.probs`;
      // full pattern of this head (the row we are building is highlighted)
      const probs = s.trace[`L${l}.attn.probs`];
      const T = d.T, cell = Math.max(4, Math.min(16, Math.floor(330 / T)));
      patternMat = mat(probs.data.subarray(hd * T * T, (hd + 1) * T * T), T, T, {
        cell, gap: T > 24 ? 0 : 1, mode: 'sequential', range: 1, kind: 'attn', trace: `L${l}.attn.probs`,
        tex: `\\Aw^{(${l},${hd})}`, cellOf: (r, c) => `${r},${c},${hd}`,
      });
      patternMat.addEventListener('click', (e) => { const c = e.target.closest('[data-cell]'); if (c) ctx.store.select({ token: Number(c.dataset.cell.split(',')[0]) }); });
      patternBox.replaceChildren(patternMat);
      header.setSub(`Query “<b>${ctx.label(d.q)}</b>” at position ${d.q} has a score for each of the ${d.nVis} keys it may see. Softmax turns them into weights that sum to 1.`);
      paint({ animate: false });
    }

    function paint({ animate = true } = {}) {
      const d = data();
      bars.dataset.trace = step <= 1 ? `L${l}.attn.masked` : `L${l}.attn.probs`;
      const vals = step === 0 ? d.sc : step === 1 ? d.sc.map((v) => v - d.m) : step === 2 ? d.ex : d.p;
      const finite = vals.filter((v) => Number.isFinite(v));
      const lo = Math.min(0, ...finite), hi = Math.max(0, ...finite);
      const span = hi - lo || 1;
      const y0 = BH * (hi / span);               // zero line from the top
      zero.style.top = `${y0}px`;
      const showVals = d.T <= 26;
      barEls.forEach((b, j) => {
        const v = vals[j];
        b.style.transitionDuration = animate ? '' : '0s';
        b.classList.toggle('masked', v === -Infinity);
        b.classList.toggle('max', step === 1 && d.sc[j] === d.m);
        if (v === -Infinity) { b.style.top = `${y0 - 4}px`; b.style.height = '8px'; b.firstChild.textContent = showVals ? '−∞' : ''; return; }
        const hpx = Math.max(2, (Math.abs(v) / span) * BH);
        b.style.top = `${v >= 0 ? y0 - hpx : y0}px`;
        b.style.height = `${hpx}px`;
        b.classList.toggle('neg', v < 0);
        b.firstChild.textContent = showVals || j === d.q ? (step >= 3 ? `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%` : fmt(v, 2)) : '';
        b.firstChild.style.top = v >= 0 ? '-22px' : `${hpx + 4}px`;
      });
      keyEls.forEach((k, j) => k.classList.toggle('dim', d.sc[j] === -Infinity));
      // stacked column fills in when the weights are known
      let acc = 0;
      stackEls.forEach((sgm, j) => {
        const p = step >= 3 ? d.p[j] : 0;
        sgm.style.height = `${p * BH}px`;
        sgm.style.bottom = `${acc * BH}px`;
        sgm.style.opacity = String(0.35 + 0.65 * (j === argmax(d.p) ? 1 : p * 2));
        acc += p;
      });
      stackLab.style.opacity = step >= 3 ? '1' : '0.25';
      patternBox.style.opacity = step >= 4 ? '1' : '0.35';
      patternMat?.highlightRow(d.q);
      if (step === 4) pulse(patternMat);

      // equation for the current step, with the numbers of the strongest key
      const j = argmax(d.p);
      const parts = [
        `\\hat S_{${d.q},j}\\;:\\; \\hat S_{${d.q},${j}} = ${num(d.sc[j], 2)},\\quad ${d.T - d.nVis} \\text{ masked} = -\\infty`,
        `m = \\max_j \\hat S_{${d.q},j} = ${num(d.m, 2)}, \\qquad \\hat S_{${d.q},j} - m \\le 0`,
        `e^{\\hat S_{${d.q},j} - m}\\;:\\; e^{${num(d.sc[j] - d.m, 2)}} = ${num(d.ex[j], 3)}, \\quad e^{-\\infty} = 0`,
        `\\Aw_{${d.q},${j}} = \\frac{e^{\\hat S_{${d.q},${j}} - m}}{\\sum_k e^{\\hat S_{${d.q},k} - m}} = \\frac{${num(d.ex[j], 3)}}{${num(d.Z, 3)}} = ${num(d.p[j], 3)}`,
        `\\Aw = \\softmax\\!\\Big(\\tfrac{\\Q\\K^{\\top}}{\\sqrt{d_h}} + \\Mk\\Big) \\in \\R^{T\\times T}`,
      ];
      eq.replaceChildren(
        h('div', { class: 'k-label', style: { textAlign: 'left' } }, STEPS[step]),
        h('div', { class: 'sm-eq-main', html: texHTML('\\Aw_{ij} = \\frac{e^{\\hat S_{ij}}}{\\sum_k e^{\\hat S_{ik}}}', { display: true }) }),
        h('div', { class: 'sm-eq-live', html: texHTML(parts[step], { display: true }) }));
      if (step === 3) {
        const z = h('b'); countTo(z, d.Z, { digits: 3 });
        eq.append(h('div', { class: 'k-caption' }, 'Z = Σ e^{…} = ', z));
      }
      const words = [
        'Each bar is a scaled dot product q·k for one key. Masked keys (the future, for a causal model) are −∞.',
        'Subtracting the row maximum changes nothing mathematically, but keeps every exponent ≤ 0 so e^x cannot overflow.',
        'Exponentiating makes everything positive and stretches gaps: a score 2 higher becomes e² ≈ 7.4 times larger. Masked keys become exactly 0.',
        `Dividing by the sum Z gives weights that add up to exactly 1. The strongest key, “${ctx.label(j)}”, gets ${(d.p[j] * 100).toFixed(1)}%.`,
        'Doing this for every query gives the head’s full attention pattern: one row per query, causal triangle included.',
      ];
      cap.set(words[step]);
    }

    function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

    function setStep(i) { step = i; paint(); }
    ctx.setSteps(STEPS, setStep);
    buildStatic();
    appear([header.el, picker.el, bars, keyRow, eq], { stagger: MOTION.stagger });

    // auto-advance the first time the scene is shown, so it explains itself
    let auto = setTimeout(function next() {
      if (step < STEPS.length - 1) { ctx.setStep(step + 1); auto = setTimeout(next, MOTION.step); }
    }, 900);
    el.addEventListener('pointerdown', () => clearTimeout(auto), { once: true });
    el.addEventListener('atlas:interrupt', () => clearTimeout(auto), { once: true });

    const offs = [ctx.on('trace', buildStatic), ctx.on('select', buildStatic)];
    return { destroy() { clearTimeout(auto); offs.forEach((f) => f()); st.destroy(); } };
  },
};

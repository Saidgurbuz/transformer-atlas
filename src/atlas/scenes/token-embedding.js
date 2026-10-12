// Token lookup: row t of W_E is the embedding of token t. The one-hot vector of its id, multiplied by
// W_E, selects that row, so a lookup and a matrix product are one operation. Steps on the real weights:
//   one-hot    the vector e in focus (every entry, the lit one labelled); W_E dimmed as the table
//   multiply   the product widget: each entry of x is a sum with a single live term
//   lookup     row t of W_E copied into x, entry by entry, checked against emb.tok[t]
//   embedding  a PCA map of every row of W_E, the tokens of this text highlighted
// Reference scene for contributors: see docs/ATLAS_GUIDE.md.
import { stage2d, sceneHeader, sceneCaption, tokenPicker, vec, chip, checkBadge, appear, useStyle, MOTION, ZONES } from '../kit.js';
import { h, s, fmt, num, texHTML, clamp, showTip, hideTip, reducedMotion, heatmap } from '../../ui/index.js';
import { matmulAnim } from '../../ui/matmul.js';
import { mulberry32 } from '../../engine/random.js';

const STEPS = ['One-hot', 'Multiply', 'Same as a lookup', 'Embedding space'];
const HERO_W = 1030, HERO_H = 480;      // hero: x 110 … 1140, y 270 … 750 (ZONES.hero, without the side column)
const MM_MAX_H = HERO_H - 80;           // product widget: the hero less its title and gaps
const MAX_DRAW_V = 96;                  // larger vocabularies skip the lookup and the widget: their cells would be too small
const MAX_MAP_V = 1500;                 // larger vocabularies skip the map
const MAP_PAD = 64;                     // room around the map for its labels
const PCA_SEED = 20261011;
// Lookup layout, in hero units: the one-hot row, the table W_E (context), the product row.
const E_TOP = 46, W_TOP = 160, X_TOP = 392;
const COPY_LEAD = 120, COPY_STEP = 14, COPY_FADE = 260;   // the product row fills left to right (ms)

useStyle('token-embedding', `
.te-group { transition: opacity .5s var(--ease), transform .6s var(--ease); }
.te-group.off { opacity: 0; transform: translateY(14px); visibility: hidden; pointer-events: none; transition: opacity .35s ease, transform .5s var(--ease), visibility 0s linear .5s; }
.te-look { position: relative; width: ${HERO_W}px; height: ${HERO_H}px; }
.te-look canvas { display: block; }
.te-look .heatmap { display: block; line-height: 0; }
.te-look .heatmap-title { display: none; }
.te-abs { position: absolute; }
.te-fade { transition: opacity .5s var(--ease); }
.te-hide { opacity: 0 !important; visibility: hidden; transition: opacity .35s ease, visibility 0s linear .35s; }
.te-band { position: absolute; box-sizing: border-box; border: 2px solid var(--c-embed); border-radius: 4px; background: color-mix(in srgb, var(--c-embed) 12%, transparent); pointer-events: none; }
.te-cap { position: absolute; font: 500 15px/1.2 var(--font-mono); color: var(--fg-2); white-space: nowrap; }
.te-cap.embed { color: var(--c-embed); }
.te-idx { position: absolute; translate: -50% 0; font: 500 15px/1 var(--font-mono); color: var(--fg-3); white-space: nowrap; pointer-events: none; }
.te-idx.lit { color: var(--accent); font-weight: 700; }
.te-mm { position: relative; width: ${HERO_W}px; }
.te-mm .mm-eq, .te-mm .mm-foot, .te-mm .mm-chip.idle { display: none; }
.te-side { width: 340px; display: grid; gap: 14px; align-content: start; }
.te-side .katex-display, .te-side .katex-display > .katex { margin: 0; text-align: left; }
.te-label { text-align: left; color: var(--c-embed); }
.te-eq { text-align: left; color: var(--fg); }
.te-live { font-size: 18px; padding: 12px 14px; border-radius: 12px; background: color-mix(in srgb, var(--surface) 85%, transparent); border: 1px solid color-mix(in srgb, var(--c-embed) 30%, var(--line)); overflow-x: auto; }
.te-check { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; opacity: 0; transform: translateY(6px); transition: opacity .5s var(--ease), transform .5s var(--ease); }
.te-check.on { opacity: 1; transform: none; }
.te-note { font-size: 16px; line-height: 1.5; }
.te-legend { display: grid; gap: 8px; }
.te-legend-row { display: flex; align-items: center; gap: 10px; font: 500 16px/1.3 var(--font-body); color: var(--fg-2); }
.te-sw { width: 12px; height: 12px; border-radius: 50%; flex: none; background: var(--fg-3); opacity: .5; }
.te-sw.in { background: var(--c-embed); opacity: 1; }
.te-sw.sel { background: var(--accent); opacity: 1; box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 25%, transparent); }
.te-map { position: relative; width: ${HERO_W}px; height: ${HERO_H}px; }
.te-svg { position: absolute; left: 0; top: 0; overflow: visible; }
.te-axis { stroke: var(--line-strong); stroke-width: 1; stroke-dasharray: 4 5; }
.te-axis-lab { font: 500 15px var(--font-mono); fill: var(--fg-3); }
.te-pt .te-dot { fill: var(--fg-3); opacity: .45; transform-box: fill-box; transform-origin: center; transition: transform .45s var(--ease), opacity .45s, fill .45s; }
.te-pt .te-lab { font: 500 15px var(--font-mono); fill: var(--fg-3); opacity: 0; transition: fill .45s, opacity .45s; paint-order: stroke; stroke: var(--bg); stroke-width: 3px; stroke-linejoin: round; }
.te-pt .te-lead { stroke: var(--fg-3); stroke-width: 1; opacity: 0; transition: opacity .3s; }
.te-pt.lead .te-lead { opacity: .9; }
.te-pt .te-sel-ring { fill: none; stroke: var(--accent); stroke-width: 2; opacity: 0; transition: opacity .45s; }
.te-pt:hover .te-dot { opacity: 1; }
.te-pt:hover .te-lab { opacity: 1; fill: var(--fg); }
.te-pt.in { cursor: pointer; }
.te-pt.in .te-dot { fill: var(--c-embed); opacity: 1; transform: scale(1.45); }
.te-pt.in .te-lab { font: 700 15px var(--font-mono); fill: var(--fg); opacity: 1; }
.te-pt.sel .te-dot { fill: var(--accent); transform: scale(1.9); }
.te-pt.sel .te-lab { fill: var(--accent); }
.te-pt.sel .te-sel-ring { opacity: 1; }
@media (prefers-reduced-motion: reduce) {
  .te-group, .te-fade, .te-hide, .te-check, .te-pt .te-dot, .te-pt .te-lab, .te-pt .te-lead, .te-pt .te-sel-ring { transition: none !important; }
}
`);

const pct = (v) => `${Math.round(v * 100)}%`;
const px = (v) => `${v}px`;
const legendRow = (sw, text) => h('div', { class: 'te-legend-row' }, sw, text);
const idxLab = (left, top, text) => h('div', { class: 'te-idx', style: { left: px(left), top: px(top) } }, text);
const fadeIn = (node) => {
  if (reducedMotion() || !node.animate) return;
  node.animate([{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'none' }], { duration: MOTION.appear, easing: MOTION.easeOut });
};
const maxAbs = (a) => { let m = 0; for (const v of a) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v)); return m || 1; };
// Timings of matmulAnim at speed 1 (src/ui/matmul.js): a focus runs K steps of clamp(1000/K, 50, 170) ms plus a 520 ms tail.
const focusMs = (D) => D.V * clamp(1000 / D.V, 50, 170) + 520;
// The product focus runs up to 3x so that it fits one step (MOTION.step) for typical vocabularies.
const mmSpeed = (D) => clamp(focusMs(D) / MOTION.step, 1, 3);
const copyMs = (D) => COPY_LEAD + (D.d - 1) * COPY_STEP + COPY_FADE;

/** Top two principal components of the rows of X (V × d): centre the rows, then power iteration on XᵀX. */
function pca2(X, V, d) {
  const mean = new Float64Array(d);
  for (let i = 0; i < V; i++) for (let c = 0; c < d; c++) mean[c] += X[i * d + c] / V;
  const Xc = new Float64Array(V * d);
  let total = 0;
  for (let i = 0; i < V; i++) for (let c = 0; c < d; c++) {
    const v = X[i * d + c] - mean[c];
    Xc[i * d + c] = v;
    total += v * v;
  }
  const times = (v) => {                       // Xc · v: one value per row
    const u = new Float64Array(V);
    for (let i = 0; i < V; i++) {
      let a = 0;
      for (let c = 0; c < d; c++) a += Xc[i * d + c] * v[c];
      u[i] = a;
    }
    return u;
  };
  const timesT = (u) => {                      // Xcᵀ · u: one value per dimension
    const w = new Float64Array(d);
    for (let i = 0; i < V; i++) for (let c = 0; c < d; c++) w[c] += Xc[i * d + c] * u[i];
    return w;
  };
  const basis = [];
  const orthoUnit = (v) => {                   // remove the components found so far, then normalise
    for (const p of basis) {
      let dot = 0;
      for (let c = 0; c < d; c++) dot += v[c] * p[c];
      for (let c = 0; c < d; c++) v[c] -= dot * p[c];
    }
    let n = 0;
    for (let c = 0; c < d; c++) n += v[c] * v[c];
    n = Math.sqrt(n);
    if (!(n > 1e-12)) return null;
    for (let c = 0; c < d; c++) v[c] /= n;
    return v;
  };
  const rnd = mulberry32(PCA_SEED);
  const eig = [];
  for (let k = 0; k < Math.min(2, d); k++) {
    let v = orthoUnit(Float64Array.from({ length: d }, () => rnd() - 0.5));
    for (let it = 0; v && it < 500; it++) {
      const w = orthoUnit(timesT(times(v)));
      if (!w) break;
      let move = 0;
      for (let c = 0; c < d; c++) move = Math.max(move, Math.abs(w[c] - v[c]));
      v = w;
      if (move < 1e-12) break;
    }
    if (!v) break;
    basis.push(v);
    let lam = 0;
    for (const u of times(v)) lam += u * u;
    eig.push(lam);
  }
  const xy = new Float64Array(2 * V);
  basis.forEach((v, k) => {
    const p = times(v);
    for (let i = 0; i < V; i++) xy[2 * i + k] = p[i];
  });
  const ratio = [0, 1].map((k) => (total > 0 && eig[k] ? eig[k] / total : 0));
  return { xy, ratio };
}

/** The k other rows of X that point most nearly the same way as row id (cosine over all d dimensions). */
function nearest(X, V, d, id, k = 3) {
  const norms = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    let n = 0;
    for (let c = 0; c < d; c++) n += X[i * d + c] ** 2;
    norms[i] = Math.sqrt(n);
  }
  const out = [];
  if (!(norms[id] > 0)) return out;
  for (let j = 0; j < V; j++) {
    if (j === id || !(norms[j] > 0)) continue;
    let dot = 0;
    for (let c = 0; c < d; c++) dot += X[id * d + c] * X[j * d + c];
    out.push({ j, cos: dot / (norms[id] * norms[j]) });
  }
  return out.sort((a, b) => b.cos - a.cos).slice(0, k);
}

export default {
  mount(el, ctx) {
    const st = stage2d(el);
    const offs = [];
    let step = 0, cur = null, destroyed = false, ticking = false;
    let mm = null, mmWE = null, mmSig = '';     // step 1: the product widget, and what it was built for
    let lk = null;                               // steps 0 and 2: the lookup composition, rebuilt on each refresh
    let map = null, mapWE = null;                // step 3: the PCA map, and the W_E it was computed from
    let checkEl = null;                          // the ✓ in the side column (step 2)
    let auto = null, autoOn = !reducedMotion();
    let timers = [];

    // ------------------------------------------------------------ header, token picker, caption (shared parts)
    const header = sceneHeader(st, { chips: [], title: 'Token lookup' });
    const picker = tokenPicker(st, ctx, { label: 'token', kind: 'embed' });
    const cap = sceneCaption(st);

    // ------------------------------------------------------------ hero: three layers, one visible at a time
    const lookNote = h('div', { class: 'k-caption', style: { width: px(HERO_W), display: 'none' } });
    const lookHost = h('div', { class: 'te-look' }, lookNote);
    const gLook = h('div', { class: 'te-group' }, lookHost);              // steps 0 and 2
    st.add(gLook, { x: ZONES.hero.x, y: ZONES.hero.y });
    const mmSlot = h('div');
    const gMul = h('div', { class: 'te-group off te-mm' }, mmSlot);       // step 1
    st.add(gMul, { x: ZONES.hero.x, y: ZONES.hero.y });
    const mapHost = h('div', { class: 'te-map' });
    const mapNote = h('div', { class: 'k-caption', style: { width: px(HERO_W), display: 'none' } });
    const gMap = h('div', { class: 'te-group off' }, mapHost, mapNote);   // step 3
    st.add(gMap, { x: ZONES.hero.x, y: ZONES.hero.y });

    // ------------------------------------------------------------ right column: this step's equation, live numbers, check
    const side = h('div', { class: 'te-side' });
    st.add(side, { x: ZONES.side.x, y: ZONES.hero.y });

    // ------------------------------------------------------------ helpers
    const at = (i) => ctx.label(i);                          // the token at position i
    const lab = (v) => ctx.state.tokenizer.label(v);         // vocabulary entry v
    const later = (ms, fn) => timers.push(setTimeout(fn, ms));
    const stopTimers = () => { timers.forEach(clearTimeout); timers = []; };
    const stopAuto = () => { autoOn = false; clearTimeout(auto); auto = null; };
    const pointTip = (p) => {
      const name = lab(p.v) === '␣' ? 'space' : lab(p.v);
      return `<b>${name}</b> · id ${p.v}${p.positions.length ? ` · in this text at position ${p.positions.join(', ')}` : ' · not in this text'}`;
    };

    /** Everything the scene shows, read from the store. */
    function data() {
      const cs = ctx.state;
      const T = cs.ids.length;
      const WE = cs.model.params['wte.weight'];
      const [V, d] = WE.shape;
      const t = clamp(Number.isFinite(cs.sel.token) ? cs.sel.token : T - 1, 0, T - 1);
      const id = cs.ids[t];
      const row = cs.trace['emb.tok'].data;                  // [T, d]: row t is the traced embedding of token t
      const x = Float64Array.from({ length: d }, (_, c) => row[t * d + c]);
      const onehot = new Float64Array(V);
      if (id < V) onehot[id] = 1;
      let jStar = 0;                                         // the output entry of largest magnitude
      for (let c = 1; c < d; c++) if (Math.abs(x[c]) > Math.abs(x[jStar])) jStar = c;
      // e · W_E computed as the sum it is (one live term per entry), compared with the traced row
      let maxDelta = 0;
      for (let c = 0; c < d; c++) {
        let a = 0;
        for (let k = 0; k < V; k++) a += onehot[k] * WE.data[k * d + c];
        maxDelta = Math.max(maxDelta, Math.abs(a - x[c]));
      }
      return { cfg: cs.config, ids: cs.ids, T, WE, V, d, t, id, x, onehot, jStar, maxDelta };
    }

    // ------------------------------------------------------------ step 1: the product widget, e · W_E = x
    function dropProduct() {
      mm?.destroy();
      mm = null; mmWE = null; mmSig = '';
      mmSlot.replaceChildren();
    }

    function syncProduct(D) {
      if (D.V > MAX_DRAW_V) { dropProduct(); return; }
      const sig = `${D.V}x${D.d}`;
      if (mm && mmWE === D.WE && mmSig === sig) {
        mm.update({ A: { data: D.onehot }, B: { data: D.WE.data }, C: { data: D.x }, speed: mmSpeed(D) });
      } else {
        dropProduct();
        mm = matmulAnim({
          A: { data: D.onehot, rows: 1, cols: D.V, name: 'e', kind: 'embed' },
          B: { data: D.WE.data, rows: D.V, cols: D.d, name: 'W_E', kind: 'embed' },
          C: { data: D.x, name: 'x', kind: 'embed' },
          maxWidth: HERO_W, maxHeight: MM_MAX_H, maxCell: 56, digits: 3, speed: mmSpeed(D),
        });
        mmSlot.append(mm.el);
        mmWE = D.WE; mmSig = sig;
      }
    }

    // ------------------------------------------------------------ steps 0 and 2: the lookup, drawn at readable size
    function dropLook() {
      lk?.destroy();
      lk = null;
    }

    /**
     * The one-hot row e (the focus of step 0), the table W_E (dimmed context, row t framed at step 2) and the
     * product row x (filled entry by entry at step 2). The rows are sized to the hero, not to W_E.
     */
    function buildLook(D) {
      dropLook();
      const { V, d, WE, id, t, x, onehot } = D;
      if (V > MAX_DRAW_V) {
        lookNote.style.display = '';
        lookNote.textContent = `W_E has ${V} rows, too many to draw entry by entry. The lookup is still exact: the one-hot row picks row ${id}, which is emb.tok[${t}].`;
        lookHost.replaceChildren(lookNote);
        return;
      }
      lookNote.style.display = 'none';
      // cell pitches: the one-hot row and the product row fill the hero width; the table stays small
      const pe = Math.min(24, Math.floor(HERO_W / V)), ge = pe >= 8 ? 2 : 1, ce = pe - ge;
      const pxd = Math.min(15, Math.floor((HERO_W - 24) / d)), gx = pxd >= 8 ? 2 : 1, cx = pxd - gx;
      const cw = Math.max(2, Math.min(8, Math.floor(480 / d), Math.floor(150 / V)));
      const eW = V * pe - ge + 12, xW = d * pxd - gx + 12, wW = d * cw;   // strips: 6 px of padding and border each side
      const eX = (HERO_W - eW) / 2, xX = (HERO_W - xW) / 2, wX = (HERO_W - wW) / 2;
      const rW = maxAbs(WE.data);                                           // one colour scale for W_E and x
      const eH = ce + 12, xH = cx + 12;

      // the one-hot row: every entry shown, index labels every 8 and at the lit entry
      const eVec = vec(onehot, { orientation: 'h', cell: ce, gap: ge, kind: 'embed', mode: 'sequential', range: 1, head: V, tail: 0 });
      eVec.highlight([id]);
      const eLabs = [];
      for (let j = 0; j < V; j += 8) if (Math.abs(j - id) >= 3) eLabs.push(idxLab(6 + j * pe + ce / 2, eH + 6, String(j)));
      const litAlign = id < 4 ? '0 0' : id > V - 5 ? '-100% 0' : '-50% 0';
      eLabs.push(h('div', { class: 'te-idx lit', style: { left: px(6 + id * pe + ce / 2), top: px(eH + 6), translate: litAlign } }, `${id} · “${at(t)}”`));
      const eWrap = h('div', { class: 'te-abs te-fade', style: { left: px(eX), top: px(E_TOP) } }, eVec,
        h('div', { class: 'te-cap', style: { left: '0px', top: px(-30) } }, `one-hot e, ${V} entries`), ...eLabs);

      // the table: a kit heatmap at a small cell size, the row of this token framed at step 2
      const heat = heatmap({ data: WE.data, rows: V, cols: d, mode: 'diverging', range: rW, cell: cw, minCell: 2, maxCell: 8, showValues: false, maxWidth: wW + 40 });
      const wWrap = h('div', { class: 'te-abs te-fade', style: { left: px(wX), top: px(W_TOP) } }, heat.el,
        h('div', { class: 'te-cap', style: { left: '0px', top: px(-30) } }, `W_E (${V} × ${d}), the table`));
      const band = h('div', { class: 'te-band te-hide', style: { left: px(wX - 3), top: px(W_TOP + id * cw - 3), width: px(wW + 6), height: px(cw + 6) } });

      // the product row: the traced emb.tok[t], so the hover and right-click menus of emb.tok apply to it
      const xVec = vec(x, { orientation: 'h', cell: cx, gap: gx, kind: 'embed', mode: 'diverging', range: rW, head: d, tail: 0, trace: 'emb.tok', cellOf: (c) => `${t},${c},0` });
      const xLabs = [];
      for (let j = 0; j < d; j += 8) xLabs.push(idxLab(6 + j * pxd + cx / 2, xH + 6, String(j)));
      const xWrap = h('div', { class: 'te-abs te-hide', style: { left: px(xX), top: px(X_TOP) } }, xVec,
        h('div', { class: 'te-cap embed', style: { left: '0px', top: px(-30) } }, `x = e · W_E: row ${id} of W_E, copied`), ...xLabs);

      lookHost.replaceChildren(lookNote, eWrap, wWrap, band, xWrap);
      lk = {
        eWrap, wWrap, band, xWrap, xVec,
        destroy() { eVec.destroy(); xVec.destroy(); heat.destroy(); },
      };
    }

    /** Which layer of the lookup is in focus at this step. */
    function showLook() {
      if (!lk) return;
      const on = step === 2;
      lk.eWrap.style.opacity = on ? '.4' : '1';
      lk.wWrap.style.opacity = '.55';
      lk.band.classList.toggle('te-hide', !on);
      lk.xWrap.classList.toggle('te-hide', !on);
    }

    /** Step 2: row t of W_E is copied into x, one entry after the other. */
    function copyRow() {
      if (!lk) return;
      fadeIn(lk.band);
      if (reducedMotion()) return;
      lk.xVec.querySelectorAll('.k-cell').forEach((c, j) => c.animate(
        [{ opacity: 0, transform: 'translateY(-8px)' }, { opacity: 1, transform: 'none' }],
        { duration: COPY_FADE, delay: COPY_LEAD + j * COPY_STEP, easing: MOTION.easeOut, fill: 'backwards' }));
    }

    // ------------------------------------------------------------ step 3: the embedding space, PCA of every row of W_E
    function buildMap(D) {
      const { xy, ratio } = pca2(D.WE.data, D.V, D.d);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let v = 0; v < D.V; v++) {
        x0 = Math.min(x0, xy[2 * v]); x1 = Math.max(x1, xy[2 * v]);
        y0 = Math.min(y0, xy[2 * v + 1]); y1 = Math.max(y1, xy[2 * v + 1]);
      }
      const k = Math.min((HERO_W - 2 * MAP_PAD) / (x1 - x0 || 1), (HERO_H - 2 * MAP_PAD) / (y1 - y0 || 1));
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const toX = (x) => HERO_W / 2 + (x - cx) * k, toY = (y) => HERO_H / 2 + (y - cy) * k;   // one scale for both axes
      const ox = toX(0), oy = toY(0);                                                       // the PCA origin: the mean row
      const svg = s('svg', { class: 'te-svg', viewBox: `0 0 ${HERO_W} ${HERO_H}`, width: HERO_W, height: HERO_H },
        s('line', { class: 'te-axis', x1: 0, y1: oy, x2: HERO_W, y2: oy }),
        s('line', { class: 'te-axis', x1: ox, y1: 0, x2: ox, y2: HERO_H }),
        s('text', { class: 'te-axis-lab', x: HERO_W - 4, y: oy - 8, 'text-anchor': 'end' }, `PC1 · ${pct(ratio[0])} of variance`),
        s('text', { class: 'te-axis-lab', x: ox + 8, y: 18 }, `PC2 · ${pct(ratio[1])} of variance`));
      const pts = [];
      for (let v = 0; v < D.V; v++) {
        const x = toX(xy[2 * v]), y = toY(xy[2 * v + 1]);
        const lead = s('line', { class: 'te-lead', x1: 0, y1: 0, x2: 0, y2: 0 });
        const lbl = s('text', { class: 'te-lab', x: 10, y: 5 }, lab(v));
        const g = s('g', { class: 'te-pt' },
          s('circle', { class: 'te-sel-ring', r: 11, cx: 0, cy: 0 }),
          lead,
          s('circle', { class: 'te-dot', r: 5.5, cx: 0, cy: 0 }),
          lbl);
        const p = { v, x, y, g, lbl, lead, positions: [], delay: 0, dist: Math.hypot(x - ox, y - oy) };
        g.addEventListener('pointerenter', (e) => showTip(e.clientX, e.clientY, pointTip(p)));
        g.addEventListener('pointerleave', hideTip);
        g.addEventListener('click', () => { if (p.positions.length) ctx.store.select({ token: p.positions[0] }); });
        pts.push(p);
        svg.append(g);
      }
      const far = Math.max(1, ...pts.map((p) => p.dist));
      pts.forEach((p) => { p.delay = 60 + Math.round((620 * p.dist) / far); });
      mapHost.replaceChildren(svg);
      map = { svg, pts, ox, oy, ratio };
    }

    /** Rebuild the map only when W_E itself changes. Returns true when it was rebuilt. */
    function syncMap(D) {
      const big = D.V > MAX_MAP_V;
      mapHost.style.display = big ? 'none' : '';
      mapNote.style.display = big ? '' : 'none';
      if (big) {
        mapNote.textContent = `The map shows up to ${MAX_MAP_V} vocabulary rows; this model has ${D.V}.`;
        map = null; mapWE = null;
        return false;
      }
      if (map && mapWE === D.WE) return false;
      buildMap(D);
      mapWE = D.WE;
      return true;
    }

    /** Points fly out from the centre, nearest first. */
    function bloom() {
      if (!map) return;
      const still = reducedMotion();
      for (const p of map.pts) {
        p.g.style.transition = 'none';
        p.g.style.transform = `translate(${map.ox}px, ${map.oy}px)`;
        p.g.style.opacity = still ? '1' : '0';
      }
      void map.svg.getBoundingClientRect();     // commit the start state before the transition
      for (const p of map.pts) {
        p.g.style.transition = still ? 'none' : `transform 1s cubic-bezier(.2,.7,.2,1) ${p.delay}ms, opacity .6s ease ${p.delay}ms`;
        p.g.style.transform = `translate(${p.x}px, ${p.y}px)`;
        p.g.style.opacity = '1';
      }
    }

    /** Highlights on the map: the tokens of this text, and the selected token. */
    function paintMap(D) {
      const where = new Map();
      D.ids.forEach((id, i) => where.set(id, [...(where.get(id) || []), i]));
      for (const p of map.pts) {
        p.positions = where.get(p.v) || [];
        p.g.classList.toggle('in', p.positions.length > 0);
        p.g.classList.toggle('sel', p.v === D.id);
      }
      const sel = map.pts.find((p) => p.v === D.id);
      if (sel) map.svg.append(sel.g);          // the selected token is drawn last, above its neighbours
      const inText = map.pts.filter((p) => p.positions.length > 0 && p !== sel).sort((a, b) => a.positions[0] - b.positions[0]);
      placeLabels([...(sel ? [sel] : []), ...inText]);
    }

    /**
     * Labels for the tokens of this text (and the selected one): each takes the first free spot around its
     * point, and a spot away from the point gets a leader line. The other labels show on hover only.
     */
    function placeLabels(list) {
      const LH = 16, CW = 9.4;                  // 15 px mono: line box and advance, in hero units
      for (const p of map.pts) {
        p.lbl.setAttribute('x', 10); p.lbl.setAttribute('y', 5);
        p.lead.setAttribute('x2', 0); p.lead.setAttribute('y2', 0);
        p.g.classList.remove('lead');
      }
      const boxes = [];
      for (const p of list) {
        const w = CW * [...p.lbl.textContent].length + 6;
        const cands = [                          // [left, top, leader?] of the label box, nearest first
          [p.x + 10, p.y - 8, 0], [p.x - 10 - w, p.y - 8, 0],
          [p.x - w / 2, p.y - 12 - LH, 0], [p.x - w / 2, p.y + 12, 0],
          [p.x + 8, p.y - 12 - LH, 0], [p.x + 8, p.y + 12, 0],
          [p.x - 8 - w, p.y - 12 - LH, 0], [p.x - 8 - w, p.y + 12, 0],
          [p.x + 30, p.y - 8, 1], [p.x - 30 - w, p.y - 8, 1], [p.x - w / 2, p.y - 34, 1], [p.x - w / 2, p.y + 30, 1],
        ];
        const free = ([bx, by]) => bx >= 0 && by >= 0 && bx + w <= HERO_W && by + LH <= HERO_H
          && boxes.every(([qx, qy, qw]) => bx < qx + qw + 3 && bx + w + 3 > qx && by < qy + LH + 3 && by + LH + 3 > qy);
        const [bx, by, leader] = cands.find(free) || cands[0];
        boxes.push([bx, by, w]);
        p.lbl.setAttribute('x', bx - p.x);
        p.lbl.setAttribute('y', by + 12 - p.y);
        p.lead.setAttribute('x2', bx + w / 2 - p.x);
        p.lead.setAttribute('y2', by + LH / 2 - p.y);
        p.g.classList.toggle('lead', !!leader);
      }
    }

    // ------------------------------------------------------------ the right column, the caption, the header
    /** The equation of the current step with its numbers, and the check that ties it to the trace. */
    function renderSide(D) {
      const { id, t, V, d, WE, x, jStar: j, cfg, ids, maxDelta } = D;
      const eq = (tex, size = 24) => h('div', { class: 'te-eq', style: { fontSize: `${size}px` }, html: texHTML(tex, { display: true }) });
      const live = (tex) => h('div', { class: 'te-live', html: texHTML(tex, { display: true }) });
      const kids = [h('div', { class: 'k-label te-label' }, STEPS[step])];
      checkEl = null;
      if (step === 0) {
        kids.push(
          eq(String.raw`\mathbf{e}_{${id}} \in \{0,1\}^{${V}}`),
          live(String.raw`\begin{aligned} (\mathbf{e}_{${id}})_u &= \mathbf{1}[u = ${id}] \\ &= (0,\dots,0,\,\underbrace{1}_{u=${id}},\,0,\dots,0) \end{aligned}`));
      } else if (step === 1) {
        kids.push(
          eq(String.raw`x_{${j}} = \sum_{u} e_u\,\WE_{u,${j}}`),
          live(String.raw`\begin{aligned} &= \cdots + 1\cdot\WE_{${id},${j}} + \cdots \\ &= ${num(WE.data[id * d + j], 3)} \end{aligned}`));
      } else if (step === 2) {
        kids.push(
          eq(String.raw`\mathbf{x} = \WE_{${id},:}`),
          live(String.raw`\begin{aligned} x_{${j}} &= \WE_{${id},${j}} = ${num(WE.data[id * d + j], 3)} \\ X_{\mathrm{tok},${t},${j}} &= ${num(x[j], 3)} \end{aligned}`));
        checkEl = h('div', { class: 'te-check' },
          checkBadge(maxDelta, 0, { label: `equals emb.tok[${t}]` }),
          h('span', { class: 'k-caption', style: { fontSize: '15px' } }, `all ${d} entries · max |Δ| = ${maxDelta === 0 ? '0' : fmt(maxDelta, 3)}`));
        kids.push(checkEl);
      } else {
        const near = V >= 2 ? nearest(WE.data, V, d, id, 3) : [];
        const distinct = new Set(ids).size;
        kids.push(
          eq(String.raw`\begin{aligned} \WE &\in \R^{${V}\times${d}} \\ &\xrightarrow{\text{PCA}} \R^{${V}\times 2} \end{aligned}`, 22),
          h('div', { class: 'te-legend' },
            legendRow(h('span', { class: 'te-sw' }), 'vocabulary entry'),
            legendRow(h('span', { class: 'te-sw in' }), `token in this text (${distinct} distinct)`),
            legendRow(h('span', { class: 'te-sw sel' }), `selected: “${at(t)}” at position ${t}`)),
          h('div', { class: 'k-caption te-note', html: cfg.tie_weights
            ? 'Weight tying is on: the output layer reuses this matrix, logits = x<sub>f</sub> W<sub>E</sub><sup>T</sup>, where x<sub>f</sub> is the final hidden state (after ln_f for pre-LN models). The same points therefore also score the next token.'
            : 'Weight tying is off: the output layer has its own matrix W<sub>U</sub>, so this map shows the input side only.' }));
        if (near.length) {
          kids.push(h('div', { class: 'k-caption te-note', html: `Closest to “${at(t)}” by cosine over all ${d} dimensions: ${near.map((n) => `“${lab(n.j)}” ${fmt(n.cos, 2)}`).join(' · ')}.` }));
        }
      }
      side.replaceChildren(...kids);
    }

    function captionFor(D) {
      const { id, t, V, d, jStar: j, x, maxDelta } = D;
      const drawn = V <= MAX_DRAW_V;
      if (step === 0) return `The one-hot vector of token <b>“${at(t)}”</b> has a 1 at index ${id} and 0 everywhere else, one entry per vocabulary item. W_E, the table, is shown dimmed.`;
      if (step === 1) return `Each entry of the output is a sum over the ${V} rows of W_E, and every term but one is multiplied by 0. So <b>x<sub>${j}</sub> = W_E[${id}, ${j}] = ${fmt(x[j], 3)}</b>.`;
      if (step === 2) {
        const check = `It matches emb.tok[${t}] entry by entry (max |Δ| = ${maxDelta === 0 ? '0' : fmt(maxDelta, 3)}), so a lookup is a one-hot product.`;
        return drawn
          ? `The product is row ${id} of W_E, copied out. ${check}`
          : `The product is row ${id} of W_E, the vector of this token. ${check}`;
      }
      return `Each row of W_E is a point in ${d} dimensions. PCA keeps the two directions of greatest spread, so points close together tend to have similar vectors. The map is a projection, so distances are approximate.`;
    }

    function render(D) {
      cur = D;
      const { id, t, V, d } = D;
      header.el.querySelector('.k-eyebrow')?.replaceChildren(chip(`position ${t}`, 'embed'), chip(`id ${id}`, 'embed'), chip(`W_E ${V}×${d}`, 'embed'));
      header.setSub(`Token <b>“${at(t)}”</b> at position ${t} has id ${id}. Multiplying its one-hot vector by W_E selects row ${id} of the table, which is this token’s vector.`);
      renderSide(D);
      cap.set(captionFor(D));
      if (map) paintMap(D);
    }

    function setGroups() {
      const product = step === 1 && !!mm;
      gLook.classList.toggle('off', step === 3 || product);
      gMul.classList.toggle('off', !product);
      gMap.classList.toggle('off', step !== 3);
    }

    // ------------------------------------------------------------ steps and motion
    /** How long a step stays on screen when the scene plays itself: long enough for its motion. */
    function holdMs(D) {
      if (step === 0) return MOTION.step;
      if (step === 1) return mm ? focusMs(D) / mmSpeed(D) + 150 : MOTION.step;
      if (step === 2) return Math.max(MOTION.step, copyMs(D) + 300);
      return 0;
    }

    /** The motion of the current step, from its start. */
    function playStep({ entering = false } = {}) {
      stopTimers();
      if (step !== 1) mm?.reset();
      if (step === 1) {
        if (mm) { mm.reset(); mm.focus(0, cur.jStar); }
      } else if (step === 2) {
        copyRow();
        later(reducedMotion() ? 0 : copyMs(cur) - 80, () => checkEl?.classList.add('on'));
      } else if (step === 3 && entering) {
        bloom();
      }
    }

    function scheduleNext() {
      clearTimeout(auto);
      if (!autoOn || destroyed) return;
      if (step >= STEPS.length - 1) { autoOn = false; return; }
      auto = setTimeout(() => {
        ticking = true;
        try { ctx.setStep(step + 1); } finally { ticking = false; }
        scheduleNext();
      }, holdMs(cur));
    }

    function enterStep(i) {
      if (destroyed) return;
      if (!ticking) stopAuto();                // a step chosen by hand ends the auto-play
      step = i;
      const D = data();
      setGroups();
      showLook();
      render(D);
      playStep({ entering: true });
      fadeIn(side);
    }

    /** Store events: a new token, a new forward pass, a new W_E. The current step plays again. */
    function refresh() {
      if (destroyed) return;
      const D = data();
      syncProduct(D);
      buildLook(D);
      const rebuilt = syncMap(D);
      setGroups();
      showLook();
      render(D);
      playStep({ entering: rebuilt });
    }

    // ------------------------------------------------------------ start
    ctx.setSteps(STEPS, enterStep);
    refresh();
    appear([header.el, picker.el, gLook, side], { stagger: MOTION.stagger });
    if (autoOn) scheduleNext();
    el.addEventListener('pointerdown', stopAuto);
    offs.push(ctx.on('trace', refresh), ctx.on('select', refresh));

    return {
      pause() { stopAuto(); stopTimers(); mm?.pause(); },
      resume() { if (!destroyed) { render(data()); playStep({ entering: true }); } },
      destroy() {
        destroyed = true;
        stopAuto();
        stopTimers();
        offs.forEach((off) => off());
        dropLook();
        mm?.destroy();
        mm = null;
        hideTip();
        st.destroy();
      },
    };
  },
};

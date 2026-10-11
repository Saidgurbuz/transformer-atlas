// Demo page for the animated matrix product (src/ui/matmul.js).
//   1. Queries: Q = X · W_Qᵀ (+ b_Q) for one head, from the live trace. Checked against L0.attn.q_heads.
//   2. Scores: S = Q · Kᵀ for the same head. Checked against L0.attn.scores (before the scale and the mask).
//   3. A 3×4 · 4×2 product with hand-picked integers. An illustration, not model data. Checked against a plain triple loop.
import { h, clamp, fmt } from '../ui/dom.js';
import { viewHeader, section, card, callout, shapeBadge } from '../ui/layout.js';
import { button, segmented, slider } from '../ui/controls.js';
import { matmulAnim } from '../ui/matmul.js';
import { dHead } from '../engine/model.js';

const TINY_A = [[1, 2, 0, -1], [0, 1, 3, 2], [2, -1, 1, 0]];   // 3 × 4
const TINY_B = [[1, 0], [2, -1], [0, 3], [-2, 1]];             // 4 × 2
// Reference for the tiny product: a plain triple loop, independent of the component.
const TINY_C = TINY_A.map((row) => [0, 1].map((j) => row.reduce((s, v, k) => s + v * TINY_B[k][j], 0)));

// Head or layer slice of a 3-D trace tensor [H, R, C] -> {data, rows, cols}.
function slice3(t, idx) {
  const [, R, C] = t.shape;
  return { data: t.data.subarray(idx * R * C, (idx + 1) * R * C), rows: R, cols: C };
}

const count = (n) => n.toLocaleString('en-US');

// Pass/fail tag for a comparison with a reference. err is null when there is nothing to compare.
function checkTag(err, ref, entries) {
  if (err == null) return h('span', { class: 'kind-tag', style: { '--kind': 'var(--fg-3)' } }, 'illustration, not checked');
  const ok = err < 1e-9;
  const text = ok
    ? `matches ${ref} on ${count(entries)} entries: max |Δ| = ${err === 0 ? '0' : fmt(err, 2)}`
    : `differs from ${ref}: max |Δ| = ${fmt(err, 3)}`;
  // The tag may wrap: on a phone the reference name is longer than the card.
  return h('span', { class: 'kind-tag', style: { '--kind': ok ? 'var(--good)' : 'var(--bad)', whiteSpace: 'normal', maxWidth: '100%', lineHeight: '1.4', textAlign: 'left' } }, text);
}

export default {
  id: '_demo_matmul',
  title: 'Demo: matmul animation',
  mount(root, ctx) {
    const mms = { q: null, s: null, t: null };
    const live = () => Object.values(mms).filter(Boolean);
    window.__matmul = mms;   // for smoke tests; removed on unmount

    // ---- shared controls
    const speed = slider({
      id: 'matmul-demo-speed', label: 'Speed', min: 0.25, max: 3, step: 0.25, value: 1,
      format: (v) => `${v}×`, onInput: (v) => live().forEach((m) => m.setSpeed(v)),
    });
    const headHost = h('div');
    let headSeg = null, headsBuilt = -1;
    const costEl = h('span');

    // ---- section 1: Q
    const qLede = h('p', { class: 'small muted' });
    const qBadge = h('div', { class: 'row' });
    const qHost = h('div', { class: 'scroll-x' });

    // ---- section 2: scores
    const sLede = h('p', { class: 'small muted' });
    const sBadge = h('div', { class: 'row' });
    const sHost = h('div', { class: 'scroll-x' });

    // ---- section 3: the tiny product
    const tHost = h('div', { class: 'scroll-x' });
    const tBadge = h('div', { class: 'row' });

    function controlsFor(key) {
      return h('div', { class: 'row' },
        button('▶ Play', () => mms[key]?.play(), { variant: 'primary', title: 'Fill C cell by cell' }),
        button('❚❚ Pause', () => mms[key]?.pause()),
        button('↺ Reset', () => mms[key]?.reset()),
        button('Fill all', () => mms[key]?.showAll(), { variant: 'ghost', title: 'Show every entry at once' }));
    }

    root.append(
      viewHeader({
        eyebrow: 'Demo',
        title: 'Matrix product, animated',
        lede: 'Hover a cell of C to compute it, or click to pin it. Each entry is a sum of products: the bars show the products, the readout above them is the running sum, and the sum flies into its cell. Play fills C in order.',
      }),
      h('div', { class: 'row' }, h('span', { class: 'eyebrow' }, 'head'), headHost, h('span', { class: 'eyebrow' }, 'speed'), speed.el),
      section({ title: 'Queries: Q = X · W_Qᵀ (+ b_Q)' },
        card({},
          qLede,
          qBadge,
          qHost,
          controlsFor('q'))),
      section({ title: 'Scores: S = Q · Kᵀ' },
        card({},
          sLede,
          sBadge,
          sHost,
          controlsFor('s'))),
      section({ title: 'A small product, with numbers' },
        card({},
          h('p', { class: 'small muted' }, 'Illustration. These integers are hand-picked and are not model data. A is 3 × 4 and B is 4 × 2, so C is 3 × 2. Each entry is a row of A times a column of B: four products and their sum.'),
          tBadge,
          tHost,
          controlsFor('t'))),
      callout('note', h('strong', null, 'Cost. '), costEl),
    );

    // The tiny example never changes.
    mms.t = matmulAnim({
      A: { data: TINY_A.flat(), rows: 3, cols: 4, name: 'A', kind: 'embed', shape: '3×4', rowLabels: ['r0', 'r1', 'r2'], colLabels: ['k0', 'k1', 'k2', 'k3'] },
      B: { data: TINY_B.flat(), rows: 4, cols: 2, name: 'B', kind: 'mlp', shape: '4×2', rowLabels: ['k0', 'k1', 'k2', 'k3'], colLabels: ['c0', 'c1'] },
      transposeB: false,
      C: { name: 'C', kind: 'o', shape: '3×2', data: TINY_C.flat() },
      title: 'C = A · B  (illustration, hand-picked integers)',
      maxWidth: 720, maxCell: 56, showValues: true, digits: 0, eqDigits: 0,
    });
    mms.t.setSpeed(speed.get());
    tHost.replaceChildren(mms.t.el);
    tBadge.replaceChildren(checkTag(mms.t.maxError(), 'a plain triple loop', 3 * 2), shapeBadge([['m', 3], ['k', 4], ['n', 2]]));

    // Create the component on first use, then update it in place (its update() ignores no-op changes).
    function upsert(key, opts, host) {
      if (!mms[key]) {
        mms[key] = matmulAnim(opts);
        mms[key].setSpeed(speed.get());
        host.replaceChildren(mms[key].el);
      } else {
        mms[key].update(opts);
      }
    }

    // ---- data-driven parts (rebuilt or updated on every trace and selection change)
    function render() {
      const st = ctx.state, cfg = st.config;
      if (!st.trace || !st.model) return;
      const T = st.ids.length, d = cfg.d_model, H = cfg.n_head, dh = dHead(cfg);
      const Hkv = cfg.n_kv_head, grp = H / Hkv;
      const head = clamp(st.sel.head ?? 0, 0, H - 1);
      const tr = st.trace;
      const labels = st.ids.map((id) => st.tokenizer.label(id));
      const pre = cfg.norm_placement !== 'post';
      const rope = cfg.pos_encoding === 'rope';
      const normName = cfg.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm';

      // Head selector: built once per head count. Later renders only move its state (rebuilding it loses keyboard focus).
      if (headsBuilt !== H) {
        headsBuilt = H;
        headSeg = segmented({
          options: Array.from({ length: H }, (_, i) => ({ value: i, label: `h${i}` })),
          value: head, ariaLabel: 'Head',
          onChange: (v) => ctx.store.select({ head: v }),
        });
        headHost.replaceChildren(headSeg.el);
      } else {
        headSeg.set(head);
      }

      // Section 1: Q = X W_Qᵀ + b_Q for this head, against the trace of q_heads.
      const X = tr['L0.ln1'] ?? tr['emb.out'];
      const Wq = st.model.params['blocks.0.attn.q_proj.weight'];
      const bq = st.model.params['blocks.0.attn.q_proj.bias'] ?? null;
      const wq = Wq.data.subarray(head * dh * d, (head + 1) * dh * d);      // dh rows of W_Q, each of length d
      const bqh = bq ? bq.data.subarray(head * dh, (head + 1) * dh) : null;
      const qRef = slice3(tr['L0.attn.q_heads'], head);
      const opts1 = {
        A: { data: X.data, rows: T, cols: d, name: 'X', kind: 'norm', shape: 'T×d', rowLabels: labels },
        B: { data: wq, rows: dh, cols: d, name: 'W_Q', kind: 'q', shape: 'd_h×d', bias: bqh, biasName: 'b_Q' },
        transposeB: true,
        C: { name: 'Q', kind: 'q', shape: 'T×d_h', data: qRef.data },
        title: `layer 0, head ${head}:  Q = X · W_Qᵀ${bq ? ' + b_Q' : ''},  W_Q = rows ${head * dh}–${(head + 1) * dh - 1} of blocks.0.attn.q_proj.weight`,
        maxWidth: 720, maxCell: 40, maxHeight: 600, showValues: 'auto', digits: 2,
      };
      upsert('q', opts1, qHost);
      qLede.textContent = `Row i of X is the input of token i: ${pre ? `L0.ln1, the ${normName} output` : 'emb.out, the block input (post-LN)'}. `
        + `Entry Q[i, j] is that row dotted with row j of W_Q${bq ? ', plus b_Q[j]' : ''}. `
        + `The strip shows the ${d} products of the focused entry. Its running sum starts at ${bq ? 'b_Q[j]' : '0'} and becomes the entry.`;
      qBadge.replaceChildren(checkTag(mms.q.maxError(), `trace L0.attn.q_heads, head ${head}`, T * dh), shapeBadge([['T', T], ['d', d], ['d_h', dh]]));

      // Section 2: S = Q Kᵀ for this head, against the trace of scores.
      const kvh = Math.floor(head / grp);
      const qSrc = tr['L0.attn.q_rot'] ?? tr['L0.attn.q_heads'];
      const kSrc = tr['L0.attn.k_rot'] ?? tr['L0.attn.k_heads'];
      const Qh = slice3(qSrc, head), Kh = slice3(kSrc, kvh), Sref = slice3(tr['L0.attn.scores'], head);
      const opts2 = {
        A: { data: Qh.data, rows: T, cols: dh, name: 'Q', kind: 'q', shape: 'T×d_h', rowLabels: labels },
        B: { data: Kh.data, rows: T, cols: dh, name: 'K', kind: 'k', shape: 'T×d_h', colLabels: labels },
        transposeB: true,
        C: { name: 'S', kind: 'attn', shape: 'T×T', data: Sref.data },
        title: `layer 0, head ${head}:  S = Q · Kᵀ  (before the 1/√d_h scale and the mask)`,
        maxWidth: 720, maxCell: 40, maxHeight: 600, showValues: 'auto', digits: 2,
      };
      upsert('s', opts2, sHost);
      sLede.textContent = `Entry S[i, j] is query i dotted with key j. Keys come from key head ${kvh}${grp > 1 ? `, which ${grp} query heads share` : ''}.${rope ? ' Both vectors are after RoPE (q_rot, k_rot).' : ''} The reference is L0.attn.scores.`;
      sBadge.replaceChildren(checkTag(mms.s.maxError(), `trace L0.attn.scores, head ${head}`, T * T), shapeBadge([['T', T], ['d_h', dh]]));

      costEl.textContent = `One head's Q takes T·d·d_h = ${T}·${d}·${dh} = ${count(T * d * dh)} multiply-adds, and its scores take T·T·d_h = ${count(T * T * dh)}. Q grows linearly with T, the scores quadratically.`;
    }

    ctx.on('trace', render);
    ctx.on('select', render);
    render();

    return () => {
      live().forEach((m) => m.destroy());
      mms.q = mms.s = mms.t = null;
      if (window.__matmul === mms) delete window.__matmul;
    };
  },
};

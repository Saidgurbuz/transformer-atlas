import { h, fmt } from '../../ui/index.js';
import { mat, tokenPicker, useStyle } from '../kit.js';
import { streamAttentionRow } from '../../lessons/flash.js';
import { lessonFrame, lessonButton, equation, sourceLink, morphContent } from './_lesson.js';

const STEPS = ['Dense result', 'Load a tile', 'Rescale', 'Accumulate', 'Normalize'];
useStyle('flash-lesson', `
.fl-panels { display: grid; grid-template-columns: 290px 420px 265px; gap: 20px; }
.fl-panel { height: 440px; transition: border-color .6s, box-shadow .6s; }
.fl-panel.on { border-color: var(--c-attn); box-shadow: 0 0 0 4px color-mix(in srgb, var(--c-attn) 12%, transparent); }
.fl-panel .ls-title { font-size: 23px; }
.fl-panel .ls-copy { font-size: 16px; margin-top: 12px; }
.fl-panel .k-label { white-space: normal; font-size: 14px; }
.fl-grids > .k-mat { min-width: 0; width: 105px; }
.fl-grids { display: flex; gap: 22px; margin: 20px 0; }
.fl-state { margin-top: 22px; display: grid; gap: 16px; }
.fl-scalar { white-space: pre-line; font: 500 16px/1.4 var(--font-mono); color: var(--fg-2); }
`);
export default {
  mount(el, ctx) {
    const f = lessonFrame(el, { title: 'FlashAttention, tile by tile', sub: 'Keep the exact softmax result while streaming keys and values.', chips: [['Live model trace', 'attn'], ['CPU recurrence · conceptual GPU memory', 'norm']] });
    let step = 0, tileSize = 4, tileIndex = 0, numbers = false, grids = [];
    const select = (id, label, fn) => h('select', { id, class: 'ls-select', 'aria-label': label, onChange: fn });
    const layer = select('flash-layer', 'Layer', e => { ctx.store.select({ layer: Number(e.target.value) }); });
    const head = select('flash-head', 'Query head', e => { ctx.store.select({ head: Number(e.target.value) }); });
    const size = select('flash-size', 'Tile width', e => { tileSize = Number(e.target.value); tileIndex = 0; paint(); });
    size.append(...[2, 4, 8].map(n => h('option', { value: n, selected: n === 4 }, `${n} keys / tile`)));
    const next = lessonButton('Next tile', () => { tileIndex++; paint(); }, { id: 'flash-next' });
    const toggle = lessonButton('Show numbers', () => { numbers = !numbers; paint(); }, { id: 'flash-numbers', 'aria-pressed': 'false' });
    f.controls.append(layer, head, size, next, toggle, lessonButton('PyTorch code', () => ctx.showReference('flash_attention.py', 'online_attention')));
    tokenPicker(f.stage, ctx, { y: 260, label: 'query', maxWidth: 1000 });
    const panels = h('div', { class: 'fl-panels', dataset: { flashResult: 'true' } });
    f.stage.add(panels, { x: 110, y: 310 });
    const summary = f.stage.add(h('div', { class: 'ls-meta' }), { x: 110, y: 755 });
    function grid(data, rows, cols, kind, label, mode = 'diverging') {
      const m = mat(data, rows, cols, { cell: cols <= 3 && rows > 1 ? 18 : 24, gap: 3, kind, label, mode, range: mode === 'sequential' ? [0, 1] : undefined, showValues: numbers }); grids.push(m); return m;
    }
    function paint() {
      grids.forEach(g => g.destroy()); grids = [];
      const { n_head: H, n_kv_head: Hkv, d_model: D, n_layer: L } = ctx.cfg, T = ctx.state.ids.length, dh = D / H;
      const l = Math.min(ctx.state.sel.layer, L - 1), hq = Math.min(ctx.state.sel.head, H - 1), q = Math.min(ctx.state.sel.token, T - 1), kv = Math.floor(hq / (H / Hkv));
      layer.replaceChildren(...Array.from({ length: L }, (_, i) => h('option', { value: i, selected: i === l }, `Layer ${i}`)));
      head.replaceChildren(...Array.from({ length: H }, (_, i) => h('option', { value: i, selected: i === hq }, `Head ${i}`)));
      const tr = ctx.state.trace, scores = tr[`L${l}.attn.masked`].data.slice((hq * T + q) * T, (hq * T + q + 1) * T);
      const v = tr[`L${l}.attn.v_heads`].data.slice(kv * T * dh, (kv + 1) * T * dh);
      const out = streamAttentionRow(scores, v, dh, tileSize);
      tileIndex %= out.tiles.length;
      const tile = out.tiles[tileIndex], rows = tile.end - tile.start, cols = Math.min(dh, 6), kvCols = Math.min(dh, 3);
      const ref = tr[`L${l}.attn.ctx`].data.slice((hq * T + q) * dh, (hq * T + q + 1) * dh);
      const error = Math.max(...out.output.map((a, i) => Math.abs(a - ref[i])));
      const matched = error <= 1e-9 * Math.max(1, ...Array.from(ref, Math.abs));
      const k = tr[`L${l}.attn.${ctx.cfg.pos_encoding === 'rope' ? 'k_rot' : 'k_heads'}`].data;
      const slice = (data, offset = 0) => Float64Array.from({ length: rows * kvCols }, (_, i) => data[offset + (tile.start + Math.floor(i / kvCols)) * dh + i % kvCols]);
      toggle.textContent = numbers ? 'Hide numbers' : 'Show numbers'; toggle.setAttribute('aria-pressed', String(numbers));
      next.textContent = tileIndex === out.tiles.length - 1 ? 'First tile' : 'Next tile';
      const stateVector = step === 2 ? Float64Array.from(tile.oldAccumulator, x => tile.correction * x) : step === 4 ? tile.output : tile.accumulator;
      const card = (title, meta, content, copy, on) => h('div', { class: ['ls-panel', 'fl-panel', on && 'on'] }, h('div', { class: 'ls-title' }, title), h('div', { class: 'ls-meta' }, meta), content, h('div', { class: 'ls-copy' }, copy));
      morphContent(panels, () => [
        card('Keys & values', `KV head ${kv} · [${T} × ${dh}]`, h('div', { class: 'fl-grids' },
          grid(slice(k, kv * T * dh), rows, kvCols, 'k', `K · ${tile.start}–${tile.end - 1}`),
          grid(slice(v), rows, kvCols, 'v', 'V tile')),
        `HBM → SRAM: load ${rows} key/value rows. ${kvCols < dh ? `First ${kvCols} features displayed.` : 'All head features displayed.'}`, step === 1),
        card('Tile workspace', `query ${q} · tile ${tileIndex + 1} / ${out.tiles.length}`, h('div', { class: 'fl-state' },
          grid(scores.slice(tile.start, tile.end), 1, rows, 'attn', 'Masked scores'),
          grid(tile.weights, 1, rows, 'attn', 'p̃ = exp(scores − new max)', 'sequential'),
          h('div', { class: 'fl-scalar' }, numbers ? `m: ${fmt(tile.oldMax, 3)} → ${fmt(tile.max, 3)}\nα = ${fmt(tile.correction, 4)} · ℓ = ${fmt(tile.sum, 4)}` : 'Running maximum m\nNormalizer ℓ\nRescaling factor α')),
        'p̃ is unnormalized. When the maximum rises, rescale the previous sum and output accumulator before adding this tile.', step === 2 || step === 3),
        card(step === 4 ? 'Prefix output' : 'Output accumulator', `[${dh}] · constant in key count`, h('div', { class: 'fl-state' },
          grid(stateVector.slice(0, cols), 1, cols, 'v', step === 2 ? 'α · old U' : step === 4 ? 'U / ℓ' : 'U (unnormalized)'),
          grid(out.output.slice(0, cols), 1, cols, 'v', 'Full output')),
        tileIndex === out.tiles.length - 1 ? 'All tiles read: normalize U by ℓ to obtain the full attention output.' : 'This prefix has read only the highlighted tiles. The full result below includes every key.', step === 4),
      ], f.stage, { track: false });
      const formulas = ['o=\\softmax(\\hat S_q)V', '\\tilde p=\\exp(\\hat S_{q,J}-m^+)',
        'm^+=\\max(m,\\max_{j\\in J}\\hat S_{q,j})\\quad\\alpha=e^{m-m^+}',
        '\\ell^+=\\alpha\\ell+\\sum_j\\tilde p_j\\quad U^+=\\alpha U+\\tilde p V_J', 'o_{\\mathrm{prefix}}=U/\\ell'];
      f.side.replaceChildren(equation(formulas[step]), h('div', { class: 'ls-copy' }, [
        'Ordinary attention stores a quadratic score/probability matrix. The streamed recurrence gives the same selected query output.',
        'The GPU algorithm tiles both queries and keys. Here one query exposes the recurrence and its small workspace.',
        'm starts at −∞, ℓ and U at zero. A wholly masked leading tile contributes nothing; its output remains zero until a visible key arrives.',
        'U is unnormalized. Rescaling only ℓ, or mixing normalized and unnormalized accumulators, produces the wrong result.',
        'Exact refers to the attention operation. Floating-point results can differ between GPU kernels; dropout is disabled in this demonstration.'
      ][step]), h('div', { class: 'ls-check' }, matched ? '✓ full stream matches the live dense trace' : 'Comparison failed'),
      sourceLink('FlashAttention', 'https://arxiv.org/abs/2205.14135', 'Algorithm 1 · §3.1'), lessonButton('Detailed maths', () => ctx.showMath('concept:online-attention')));
      summary.textContent = `Score workspace per query: ${T} → ≤${tileSize} entries · dot-product work remains quadratic for the whole sequence.`;
      panels.dataset.matched = String(matched); panels.dataset.tile = String(tileIndex); panels.dataset.tiles = String(out.tiles.length); panels.dataset.error = String(error);
      f.caption.set('This browser lesson retains the dense trace and tile snapshots for verification. It illustrates the GPU algorithm’s memory roles; it does not measure GPU speed or implement a fused kernel.');
    }
    ctx.setSteps(STEPS, i => { step = i; paint(); }); ctx.on('trace', () => { tileIndex = 0; paint(); }); ctx.on('select', paint); paint();
    return { destroy() { grids.forEach(g => g.destroy()); f.destroy(); } };
  },
};

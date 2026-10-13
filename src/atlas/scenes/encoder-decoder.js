import { h } from '../../ui/index.js';
import { mat, useStyle } from '../kit.js';
import { createEncoderDecoder, encoderDecoderForward, ENCDEC_VOCAB } from '../../lessons/encoder-decoder.js';
import { lessonFrame, lessonButton, equation, sourceLink, flowBox, morphContent } from './_lesson.js';
const STEPS = ['Encoder', 'Decoder self-attention', 'Cross-attention', 'Feed-forward', 'Output'];
const EXAMPLES = [ [[2, 3, 4, 5], [0, 1, 2]], [[1, 2, 4], [0, 2]], [[2], [0]], [[2, 5, 5], [0, 1, 5]] ];
useStyle('encoder-decoder-live', `
.ec-panels { display: grid; grid-template-columns: 290px 420px 265px; gap: 20px; }
.ec-panel { height: 350px; }
.ec-panel .ls-title { font-size: 23px; }
.ec-panel .ls-copy { margin-top: 14px; font-size: 16px; }
.ec-panel .ls-meta { font-size: 14px; }
.ec-matrix { display: flex; justify-content: center; margin-top: 25px; }
.ec-tokens { font: 600 18px/1.6 var(--font-mono); color: var(--c-embed); margin-top: 16px; }
.ec-tokens.target { color: var(--c-head); }
.ec-panel .k-label { white-space: normal; font-size: 14px; }
`);
export default {
  mount(el, ctx) {
    const f = lessonFrame(el, { title: 'Encoder–decoder computation', sub: 'Follow a real decoder query into the encoder’s keys and values.', chips: [['2017-style · post-LayerNorm', 'norm'], ['Seeded, untrained miniature', 'resid']] });
    const model = createEncoderDecoder();
    let step = 0, example = 0, head = 0, query = 0, numbers = false, grids = [];
    const picker = (id, label, fn) => h('select', { id, class: 'ls-select', 'aria-label': label, onChange: fn });
    const examples = picker('encdec-example', 'Sequence example', e => { example = Number(e.target.value); query = 0; paint(); });
    examples.append(...['Padded source · 4 → 3', 'Different lengths · 3 → 2', 'Single token · 1 → 1', 'Padded source & target'].map((s, i) => h('option', { value: i }, s)));
    const heads = picker('encdec-head', 'Attention head', e => { head = Number(e.target.value); paint(); }); heads.append(h('option', { value: 0 }, 'Head 0'), h('option', { value: 1 }, 'Head 1'));
    const queries = picker('encdec-query', 'Query position', e => { query = Number(e.target.value); paint(); });
    const toggle = lessonButton('Show numbers', () => { numbers = !numbers; paint(); }, { id: 'encdec-numbers', 'aria-pressed': 'false' });
    f.controls.append(examples, heads, queries, toggle, lessonButton('PyTorch code', () => ctx.showReference('encoder_decoder.py', step === 2 ? 'attention_snapshot' : 'TinyEncoderDecoder')));
    const encoder = f.stage.add(flowBox('Encoder', 'Bidirectional attention → FFN', 'embed', 290), { x: 110, y: 285 });
    const cross = f.stage.add(flowBox('Cross-attention', 'Q: decoder · K, V: encoder', 'attn', 420), { x: 420, y: 285 });
    const decoder = f.stage.add(flowBox('Decoder', 'Self → cross → FFN', 'head', 265), { x: 860, y: 285 });
    const panels = h('div', { class: 'ec-panels', dataset: { encdecResult: 'true' } }); f.stage.add(panels, { x: 110, y: 410 });
    function grid(t, label, kind, row, headIndex = null, mode = 'diverging', mask = null) {
      const [rows, fullCols] = t.shape.slice(-2), cols = Math.min(fullCols, 8), offset = headIndex === null ? 0 : headIndex * rows * fullCols;
      const values = Float64Array.from({ length: rows * cols }, (_, i) => t.data[offset + Math.floor(i / cols) * fullCols + i % cols]);
      if (mask) values.forEach((_, i) => { if (!mask.data[i]) values[i] = -Infinity; });
      const m = mat(values, rows, cols, { cell: 22, gap: 3, kind, label, showValues: numbers, mode, range: mode === 'sequential' ? [0, 1] : undefined });
      grids.push(m); if (row != null) m.highlightRow(row); return m;
    }
    function paint() {
      grids.forEach(g => g.destroy()); grids = [];
      const [source, target] = EXAMPLES[example], { trace: tr } = encoderDecoderForward(model, source, target);
      const encoderStep = step === 0, prefix = encoderStep ? 'encoder.attn' : step === 1 ? 'decoder.self' : 'decoder.cross';
      const mask = tr[encoderStep ? 'source.mask' : step === 1 ? 'target.mask' : 'cross.mask'];
      const nQueries = encoderStep ? source.length : target.length; query = Math.min(query, nQueries - 1);
      queries.replaceChildren(...Array.from({ length: nQueries }, (_, i) => h('option', { value: i, selected: i === query }, `Query ${i}: ${ENCDEC_VOCAB[(encoderStep ? source : target)[i]]}`)));
      queries.hidden = step > 2; heads.hidden = step > 2;
      toggle.textContent = numbers ? 'Hide numbers' : 'Show numbers'; toggle.setAttribute('aria-pressed', String(numbers));
      encoder.classList.toggle('on', step === 0 || step === 2); cross.classList.toggle('on', step === 2); decoder.classList.toggle('on', step !== 0);
      const card = (title, meta, content, copy) => h('div', { class: 'ls-panel ec-panel' }, h('div', { class: 'ls-title' }, title), h('div', { class: 'ls-meta' }, meta), content, h('div', { class: 'ls-copy' }, copy));
      const attention = step <= 2;
      const state = tr[step === 0 ? 'encoder.memory' : step === 1 ? 'decoder.self_out' : step === 2 ? 'decoder.cross_out' : 'decoder.out'];
      morphContent(panels, () => [
        card('Two token sequences', `source S=${source.length} · target T=${target.length}`, h('div', null,
          h('div', { class: 'ec-tokens' }, source.map(i => ENCDEC_VOCAB[i]).join(' ')), h('div', { class: 'ec-tokens target' }, target.map(i => ENCDEC_VOCAB[i]).join(' '))),
          'The target begins with BOS. Padding is excluded as a key, while padded query rows are still computed.'),
        card(attention ? ['Encoder attention', 'Decoder self-attention', 'Decoder cross-attention'][step] : step === 3 ? 'ReLU feed-forward' : 'Next-token probabilities',
          attention ? `A [${tr[prefix + '.probs'].shape.slice(-2).join(' × ')}] · selected row ${query}` : step === 3 ? 'per token: 8 → 16 → 8' : `[${target.length} × 6] · shared output weights`,
          h('div', { class: 'ec-matrix' }, attention ? grid(tr[prefix + '.probs'], 'Rows: queries · columns: keys', 'attn', query, head, 'sequential', mask)
            : step === 3 ? grid(tr['decoder.mlp.act'], 'First 8 of 16 hidden features', 'mlp', null) : grid(tr.probs, 'Vocabulary distribution', 'head', null, null, 'sequential')),
          attention ? ['Each source token may read every non-padding source token.', 'Each target token reads only its visible prefix.', 'Rectangular attention: decoder queries read all non-padding source positions.'][step] : step === 3 ? 'Each token is transformed independently after the cross-attention residual and normalization.' : 'The weights are untrained: these distributions illustrate computation, not translation quality.'),
        card('Residual state', `[${state.shape.join(' × ')}] · after add + LayerNorm`, h('div', { class: 'ec-matrix' }, grid(state, 'All 8 model features', 'resid', attention ? query : null)),
          step === 0 ? 'Encoder memory stays fixed while decoder queries change.' : 'Each decoder sub-layer has its own add + post-LayerNorm.'),
      ], f.stage, { track: false });
      const formulas = ['\\begin{aligned}E_1&=\\LN(E_0+\\mathrm{Attn}(E_0,E_0,E_0))\\\\ E&=\\LN(E_1+\\mathrm{FFN}(E_1))\\end{aligned}', 'D_1=\\LN(D+\\mathrm{MaskedAttn}(D,D,D))',
        '\\begin{aligned}Q&=D_1W_Q^\\top\\\\ K&=EW_K^\\top\\\\ V&=EW_V^\\top\\end{aligned}', 'D_3=\\LN(D_2+\\mathrm{FFN}(D_2))', 'p=\\softmax(D_3W_E^\\top)'];
      f.side.replaceChildren(equation(formulas[step]), h('div', { class: 'ls-copy' }, ['Scaled token embeddings plus sinusoidal positions enter one full encoder block.',
        'The decoder receives the shifted target prefix. Future target keys are masked with −∞ before softmax.',
        'Q and K/V come from different streams. Attention has shape [H, T, S]; T and S can differ.',
        'FFN(x) = ReLU(x W₁ᵀ + b₁) W₂ᵀ + b₂. The original Transformer normalizes after each residual addition.',
        'One shared vocabulary embedding is used for source lookup, target lookup and output projection.'][step]),
        h('div', { class: 'ls-check' }, '✓ every tensor checked against PyTorch'), sourceLink('Attention Is All You Need', 'https://arxiv.org/abs/1706.03762', '§3.1–3.5'),
        lessonButton('Detailed maths', () => ctx.showMath('concept:encoder-decoder')));
      panels.dataset.shape = tr['decoder.cross.probs'].shape.slice(-2).join(','); panels.dataset.paddedZero = String(source.every((id, s) => id !== 5 || Array.from({ length: 2 * target.length }, (_, i) => tr['decoder.cross.probs'].data[i * source.length + s]).every(v => v === 0)));
      f.caption.set('One complete encoder block and one complete decoder block run locally. This separate model uses the 2017 conventions, including √d embedding scaling, post-LayerNorm, ReLU and three decoder sub-layers.');
    }
    ctx.setSteps(STEPS, i => { step = i; paint(); }); paint();
    return { destroy() { grids.forEach(g => g.destroy()); f.destroy(); } };
  },
};

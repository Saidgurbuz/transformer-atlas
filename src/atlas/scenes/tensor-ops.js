import { h } from '../../ui/index.js';
import { coordinates, storageIndex, tensorExample, isContiguous } from '../../lessons/tensors.js';
import { lessonFrame, lessonButton, equation, sourceLink, morphContent } from './_lesson.js';

const STEPS = ['arange', 'view', 'transpose', 'reshape', 'contiguous', 'cat', 'einsum', 'unsqueeze', 'squeeze'];
const EXAMPLE = tensorExample();
const SPEC = [
  ['arange', 'One flat allocation', 'torch.arange(B * T * H * dh)', 'A range creates the values in one packed buffer. The identity labels are hidden until you ask for them.'],
  ['split', 'Split the feature axis', 'x.view(B, T, H, dh)', 'view keeps the same storage and divides each token’s feature axis into heads. No multiplication and no copy happen.'],
  ['heads', 'Move the head axis', 'split.transpose(1, 2)', 'transpose swaps the T and H axes by changing strides. Each head now indexes all token positions, while the storage stays shared.'],
  ['reshaped', 'Reshape a strided view', 'heads.reshape(B, H * T, dh)', 'This reshape preserves logical order but copies because H and T cannot be merged with the current strides. reshape may return a view when the strides are compatible.'],
  ['copied', 'Pack the new order', 'heads.contiguous()', 'contiguous copies this strided view into packed head-major storage. The logical values stay the same; their physical offsets change.'],
  ['concatenated', 'Join head features', 'torch.cat([heads[:, h]\n  for h in range(H)], dim=-1)', 'cat allocates a result and places the head feature vectors side by side. It reconstructs B × T × d, with the original feature order.'],
  ['scores', 'Contract the feature axis', 'torch.einsum(\n  "bhtd,bhsd->bhts", heads, heads)', 'einsum sums over d. Batch and head stay separate, while query t and key s become the two attention-score axes.'],
  ['positionRow', 'Insert a singleton axis', 'positions.unsqueeze(0)', 'unsqueeze changes T into 1 × T without copying. This row can broadcast across a batch; broadcasting does not create independent learned positions.'],
  ['positionFlat', 'Remove a singleton axis', 'position_row.squeeze(0)', 'squeeze removes the size-one axis. Specify the axis: an unrestricted squeeze can accidentally remove the batch axis when B = 1.'],
];

export default {
  mount(el, ctx) {
    const f = lessonFrame(el, { title: 'Tensor operations', sub: 'Follow axes, strides and storage through the operations behind multi-head attention.', chips: [['Layout example', 'q'], ['B = 2 · T = 3 · H = 2', 'resid']] });
    let step = 0, labels = false, selected = 0;
    const left = h('div', { class: 'ls-panel', style: { width: '475px', height: '340px' } });
    const right = h('div', { class: 'ls-panel', style: { width: '475px', height: '340px' }, dataset: { lessonResult: 'tensor-ops' } });
    f.stage.add(left, { x: 110, y: 280 }); f.stage.add(right, { x: 650, y: 280 });
    f.stage.arrow({ x: 601, y: 438 }, { x: 630, y: 438 }, { kind: 'q' });
    const buffer = h('div', { class: 'ls-panel', style: { width: '1015px', height: '115px' } });
    f.stage.add(buffer, { x: 110, y: 648 });
    const toggle = lessonButton('Show offsets', () => { labels = !labels; toggle.setAttribute('aria-pressed', String(labels)); render(false); }, { id: 'tensor-ops-labels', 'aria-pressed': 'false' });
    f.controls.append(toggle, lessonButton('PyTorch code', () => ctx.showReference('tensor_ops.py', 'tensor_operations')));

    function picture(t, name, kind = 'q') {
      const rows = t.shape.length >= 2 ? t.shape.at(-2) : 1, cols = t.shape.at(-1), groups = t.data.length / (rows * cols);
      const result = h('div', { class: 'ls-pictures', style: { gridTemplateColumns: `repeat(${groups === 4 ? 4 : Math.min(groups, 3)}, max-content)` } });
      for (let g = 0; g < groups; g++) {
        const grid = h('div', { class: 'ls-grid', style: { gridTemplateColumns: `repeat(${Math.min(cols, 12)}, 27px)` } });
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          const logical = (g * rows + r) * cols + c, coord = coordinates(logical, t.shape), offset = storageIndex(t, coord), value = t.data[offset];
          const sharedIdentity = name !== 'scores' && !name.startsWith('position') ? value : null;
          grid.append(h('button', { type: 'button', class: ['ls-slot', labels && 'labelled', sharedIdentity === selected && 'selected'],
            style: { '--kind': `var(--c-${kind})` }, title: `[${coord.join(', ')}] · storage offset ${offset}`,
            'aria-label': `${name} coordinate ${coord.join(', ')}; storage offset ${offset}`,
            dataset: { identity: `${name === 'scores' ? 'score' : name.startsWith('position') ? 'position' : 'q'}:${sharedIdentity ?? logical}`, offset },
            onClick: () => { if (sharedIdentity != null) { selected = sharedIdentity; render(false); } } }, labels ? String(offset) : null));
        }
        const prefix = coordinates(g, t.shape.slice(0, -2));
        const axes = name === 'split' ? ['B', 'T'] : ['B', 'H'];
        result.append(h('div', { class: 'ls-tensor' }, groups > 1 ? h('div', { class: 'ls-meta' }, prefix.map((v, i) => `${axes[i]}${v}`).join(' · ')) : null, grid));
      }
      return result;
    }
    function render(animate = true) {
      const [key, title, code, copy] = SPEC[step], t = EXAMPLE[key];
      const positions = key.startsWith('position'), input = positions ? EXAMPLE.positions : EXAMPLE.x;
      left.replaceChildren(h('div', { class: 'ls-title' }, positions ? 'Position indices' : 'Before head splitting'),
        h('div', { class: 'ls-meta' }, `shape [${input.shape.join(', ')}] · packed storage`), picture(input, positions ? 'positions' : 'x', positions ? 'pos' : 'q'));
      morphContent(right, () => h('div', null, h('div', { class: 'ls-title' }, title), h('div', { class: 'ls-meta' }, `shape [${t.shape.join(', ')}]`), picture(t, key, positions ? 'pos' : 'q')), f.stage, { track: animate && key !== 'scores' });
      right.dataset.operation = STEPS[step]; right.dataset.shape = t.shape.join(','); right.dataset.contiguous = String(isContiguous(t));
      const slots = h('div', { class: 'ls-buffer' });
      for (let i = 0; i < (positions ? 3 : 36); i++) slots.append(h('div', { class: ['ls-slot', labels && 'labelled', i === selected && 'selected'], style: { '--kind': positions ? 'var(--c-pos)' : 'var(--c-q)' }, title: `storage offset ${i}` }, labels ? String(i) : null));
      buffer.replaceChildren(h('div', { class: 'ls-meta' }, positions ? 'Position-index storage · unsqueeze and squeeze share this buffer' : ['reshaped', 'copied', 'concatenated', 'scores'].includes(key) ? 'Original Q storage · this operation allocates separate result storage' : 'Original Q storage · the views index this same buffer'), slots);
      f.side.replaceChildren(equation(key === 'scores' ? 'S_{b,h,t,s}=\\sum_c Q_{b,h,t,c}K_{b,h,s,c}' : positions ? 'p\\in\\R^{T}\\;\\longleftrightarrow\\;p[\\mathrm{None},:]\\in\\R^{1\\times T}' : 'd = H\\,d_h'),
        h('pre', { class: 'ls-code' }, code), h('div', { class: 'ls-copy' }, `strides [${t.strides.join(', ')}] · ${isContiguous(t) ? 'contiguous' : 'strided'}`),
        h('div', { class: 'ls-check' }, '✓ values, shapes and strides match PyTorch'),
        sourceLink('PyTorch tensor views', 'https://docs.pytorch.org/docs/stable/tensor_view.html'),
        h('div', { class: 'ls-source' }, 'This transpose needs a copy to flatten its H × T axes. A view is valid when the requested shape fits its stride pattern.'));
      f.caption.set(copy);
    }
    ctx.setSteps(STEPS, (i) => { step = i; render(); });
    render(false);
    return { destroy: f.destroy };
  },
};

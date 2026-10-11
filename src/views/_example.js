// Reference view for contributors (reachable at #_example, hidden from the nav).
// Shows the conventions every view follows:
//   - read tensors only from ctx.state.trace (the live forward pass)
//   - re-render on 'trace' and 'select' via ctx.on (auto-unsubscribed on unmount)
//   - show the matching lines of reference/model.py and let users open the inspector
//   - colours come from component kinds, never hard-coded hex values
import {
  h, viewHeader, section, card, grid, heatmap, sliceOf, segmented, tokenStrip, codeView, playbar,
  formula, frac, sqrt, col, kindTag, shapeBadge, button, callout, fmt,
} from '../ui/index.js';

export default {
  id: '_example',
  title: 'Example view',
  mount(root, ctx) {
    const { codemap, describe } = ctx;

    // ---- static scaffolding (built once)
    const layerSeg = h('div');
    const headSeg = h('div');
    const tokensBox = h('div');
    const heatBox = h('div', { class: 'scroll-x' });
    const readout = h('div', { class: 'stack' });

    // Code panel: the Attention class, with the line producing `probs` highlighted.
    const region = codemap.codeRegions.Attention;
    const code = codeView({
      lines: codemap.codeLines, from: region.from, to: region.to, maxHeight: 360,
      kindOf: (k) => describe.infoFor(k).kind,
      onTraceClick: (k) => ctx.inspect(k.replace('{i}', ctx.state.sel.layer)),
    });

    // A staged explanation: each step highlights one line of code.
    const steps = [
      { label: 'scores', key: 'L{i}.attn.scores' },
      { label: 'scale', key: 'L{i}.attn.scaled' },
      { label: 'mask', key: 'L{i}.attn.masked' },
      { label: 'softmax', key: 'L{i}.attn.probs' },
    ];
    const pb = playbar({
      steps: steps.map((s) => s.label),
      onStep: (i) => code.setHighlight(codemap.linesForKey(steps[i].key)),
    });

    root.append(
      viewHeader({
        eyebrow: 'Example',
        title: 'Attention pattern of one head',
        lede: 'Pick a layer, a head and a query token. The heatmap is the live <code>probs</code> tensor from the forward pass.',
      }),
      section({ title: 'Pattern' },
        h('div', { class: 'row' }, h('span', { class: 'eyebrow' }, 'layer'), layerSeg, h('span', { class: 'eyebrow' }, 'head'), headSeg),
        tokensBox,
        grid(2,
          card({ title: 'Attention weights', kind: 'attn', actions: [button('Inspect tensor', () => ctx.inspect(`L${ctx.state.sel.layer}.attn.probs`))] },
            heatBox, readout),
          card({ title: 'The code that made it', kind: 'attn' },
            formula(`A = softmax(${frac(`${col('q', 'Q')}${col('k', 'K')}<sup>⊤</sup>`, sqrt('d<sub>h</sub>'))} + M)`),
            pb.el, code.el))),
      callout('note', h('strong', null, 'Try it: '), 'switch the mask to “none” in Model settings and watch the upper triangle fill in.'),
    );

    // ---- dynamic parts (rebuilt when the trace or selection changes)
    let hm = null;
    function render() {
      const st = ctx.state, { layer, head, token } = st.sel, cfg = st.config;
      const labels = st.ids.map((id) => st.tokenizer.label(id));

      layerSeg.replaceChildren(segmented({
        options: Array.from({ length: cfg.n_layer }, (_, i) => ({ value: i, label: `L${i}` })),
        value: layer, ariaLabel: 'Layer', onChange: (v) => ctx.store.select({ layer: v }),
      }).el);
      headSeg.replaceChildren(segmented({
        options: Array.from({ length: cfg.n_head }, (_, i) => ({ value: i, label: `h${i}` })),
        value: head, ariaLabel: 'Head', onChange: (v) => ctx.store.select({ head: v }),
      }).el);
      tokensBox.replaceChildren(tokenStrip({ labels, selected: token, onSelect: (i) => ctx.store.select({ token: i }) }).el);

      const probs = st.trace[`L${layer}.attn.probs`];
      const { data, rows, cols } = sliceOf(probs, head);
      const opts = {
        data, rows, cols, mode: 'sequential', kind: 'attn', range: 1,
        rowLabels: labels, colLabels: labels, highlight: { rows: [token] },
        tooltip: (r, c, v) => `query <b>${labels[r]}</b>(${r}) → key <b>${labels[c]}</b>(${c}) = ${fmt(v, 3)}`,
        onClick: (r) => ctx.store.select({ token: r }),
      };
      if (hm) hm.update(opts); else { hm = heatmap(opts); heatBox.append(hm.el); }

      const row = Array.from(data.subarray(token * cols, (token + 1) * cols));
      const best = row.indexOf(Math.max(...row));
      readout.replaceChildren(
        h('div', { class: 'row' }, kindTag('attn', `L${layer}.attn.probs`), shapeBadge(describe.shapePairs(`L${layer}.attn.probs`, cfg, st.ids.length))),
        h('p', { class: 'small' }, `Query “${labels[token]}” at position ${token} puts ${fmt(row[best] * 100, 1)}% of its attention on “${labels[best]}” at position ${best}.`));
    }

    ctx.on('trace', render);
    ctx.on('select', render);
    render();

    return () => { pb.destroy(); hm?.destroy(); };
  },
};

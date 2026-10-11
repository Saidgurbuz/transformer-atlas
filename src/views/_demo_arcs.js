// Demo page for the animated attention arcs (src/ui/arcs.js). Hidden from the nav: open #_demo_arcs.
// Every number on the page is measured from the live trace: attention weights of the selected head,
// and the induction score of each head (weight on the token after an earlier copy of the query token).
import {
  h, viewHeader, section, card, grid, segmented, toggle, button, callout, sliceOf, fmt, tokenStrip,
} from '../ui/index.js';
import { arcDiagram } from '../ui/arcs.js';
import { PRESETS } from '../presets/index.js';

// For each position t: the latest earlier copy of its token (prev) and the token after it (target).
// Null when the token has no earlier copy, or when that copy is the token just before t.
function inductionTargets(ids) {
  const last = new Map();
  return ids.map((id, t) => {
    const prev = last.get(id);
    last.set(id, t);
    return prev !== undefined && prev + 1 < t ? { prev, target: prev + 1 } : null;
  });
}

// Index of the largest weight in row t of one head's attention: the query's strongest key.
function topKeyOf(weights, t) {
  const { data, cols } = weights;
  let best = 0, bestW = -1;
  for (let j = 0; j < cols; j++) {
    const v = data[t * cols + j];
    if (v > bestW) { bestW = v; best = j; }
  }
  return best;
}

// Mean weight one head puts on the induction targets, and how often the target is its top key.
function inductionStats(weights, targets) {
  const { data, cols } = weights;
  let n = 0, mass = 0, top = 0;
  targets.forEach((tg, t) => {
    if (!tg) return;
    n += 1;
    mass += data[t * cols + tg.target];
    if (topKeyOf(weights, t) === tg.target) top += 1;
  });
  return { n, mass: n ? mass / n : 0, top };
}

function bestIndex(stats) {
  let best = -1, bestMass = -1;
  stats.forEach((s, k) => {
    if (s.n && s.mass > bestMass) { bestMass = s.mass; best = k; }
  });
  return best;
}

export default {
  id: '_demo_arcs',
  title: 'Demo: attention arcs',
  mount(root, ctx) {
    let showAll = false;
    let sweeping = false;
    let pickerKey = '';
    let smallKey = '';
    let smalls = [];           // one { diagram, stat, card, hd } per head of the selected layer
    let layerSeg = null, headSeg = null;

    const layerBox = h('div');
    const headBox = h('div');
    const modeBox = h('div');
    const speedBox = h('div');
    const dotsBox = h('div');
    const textIn = h('input', { class: 'input-text', type: 'text', 'aria-label': 'Input text', spellcheck: 'false' });
    const stripBox = h('div');
    const readBig = h('p', { class: 'small' });
    const readBip = h('p', { class: 'small muted' });
    const inductionText = h('p', { class: 'small' });
    const smallBox = grid(2);

    const allDiagrams = () => [big, bip, ...smalls.map((x) => x.diagram)];

    // A query chosen in the store; a cleared query shows every query. The one-row diagram's onSelect
    // also runs for every sweep step, so it must not stop the sweep. Clicks from anywhere else go
    // through userPick, which stops the sweep first.
    const pick = (q) => {
      if (q == null) { showAll = true; render(); return; }
      showAll = false;
      ctx.store.select({ token: q });
    };
    const userPick = (q) => { big.stop(); pick(q); };

    // maxWidth lets each diagram fill its card; the arc area is sized from the data.
    const big = arcDiagram({ layout: 'arc', maxWidth: 1600, onSelect: pick });
    const bip = arcDiagram({ layout: 'bipartite', maxWidth: 1600, onSelect: userPick });

    const modeSeg = segmented({
      options: [
        { value: 'one', label: 'One query', title: 'Show the arcs of the selected query' },
        { value: 'all', label: 'All queries', title: 'Show every query at once' },
      ],
      value: 'one',
      ariaLabel: 'Queries shown',
      onChange: (v) => {
        if (v === 'all') big.stop();
        showAll = v === 'all';
        render();
      },
    });
    const speedSeg = segmented({
      options: [
        { value: 1, label: '1×', title: 'Normal speed' },
        { value: 0.25, label: '¼×', title: 'Slow motion: a quarter of the speed' },
      ],
      value: 1,
      ariaLabel: 'Animation speed',
      onChange: (v) => allDiagrams().forEach((d) => d.update({ speed: v })),
    });
    const dotsToggle = toggle({
      id: '_demo_arcs-dots',
      label: 'Travelling dots',
      checked: true,
      onChange: (v) => { big.update({ dots: v }); bip.update({ dots: v }); },
    });

    const playBtn = button('▶ Play sweep', () => {
      if (sweeping) { big.stop(); return; }
      sweeping = true;
      showAll = false;
      playBtn.textContent = '❚❚ Stop';
      render();
      big.play().then(() => {
        sweeping = false;
        playBtn.textContent = '▶ Play sweep';
      });
    }, { variant: 'primary', title: 'Move the query from left to right, one token every 0.6 s' });

    const presetBtn = button('Load induction preset', async () => {
      presetBtn.disabled = true;
      big.stop();
      try {
        await ctx.store.loadPreset('induction');
        const st = ctx.state;
        const layer = st.config.n_layer - 1;
        const targets = inductionTargets(st.ids);
        const probs = st.trace[`L${layer}.attn.probs`];
        const stats = Array.from({ length: st.config.n_head }, (_, hd) => inductionStats(sliceOf(probs, hd), targets));
        const head = Math.max(0, bestIndex(stats));
        // Prefer the last query whose strongest key is its induction target (and not a space).
        const W = sliceOf(probs, head);
        const isSpace = (id) => st.tokenizer.token(id) === ' ';
        const usable = (t) => targets[t] && !isSpace(st.ids[targets[t].target]);
        let token = -1;
        for (let t = targets.length - 1; t >= 0 && token < 0; t--) {
          if (usable(t) && topKeyOf(W, t) === targets[t].target) token = t;
        }
        for (let t = targets.length - 1; t >= 0 && token < 0; t--) {
          if (usable(t)) token = t;
        }
        if (token < 0) token = st.ids.length - 1;
        showAll = false;
        ctx.store.select({ layer, head, token });
      } finally {
        presetBtn.disabled = false;
      }
    }, { title: 'Load the 2-layer model trained on repeated random text' });
    if (!PRESETS.some((p) => p.id === 'induction')) presetBtn.disabled = true;

    textIn.addEventListener('change', () => ctx.store.setText(textIn.value));

    function ensurePickers(NL, H, layer, head) {
      const key = `${NL}x${H}`;
      if (key !== pickerKey) {
        pickerKey = key;
        layerSeg = segmented({
          options: Array.from({ length: NL }, (_, i) => ({ value: i, label: `L${i}`, title: `Layer ${i}` })),
          value: layer,
          ariaLabel: 'Layer',
          onChange: (v) => ctx.store.select({ layer: v }),
        });
        headSeg = segmented({
          options: Array.from({ length: H }, (_, i) => ({ value: i, label: `h${i}`, title: `Head ${i}` })),
          value: head,
          ariaLabel: 'Head',
          onChange: (v) => ctx.store.select({ head: v }),
        });
        layerBox.replaceChildren(layerSeg.el);
        headBox.replaceChildren(headSeg.el);
      } else {
        layerSeg.set(layer);
        headSeg.set(head);
      }
    }

    function ensureSmalls(H) {
      const key = String(H);
      if (key === smallKey) return;
      smallKey = key;
      for (const x of smalls) x.diagram.destroy();
      smalls = [];
      for (let hd = 0; hd < H; hd++) {
        const diagram = arcDiagram({ layout: 'arc', maxWidth: 540, showLabels: false, dots: false, onSelect: userPick });
        const stat = h('p', { class: 'small muted' });
        const cardEl = card({ title: `Head h${hd}`, kind: 'attn' }, diagram.el, stat);
        smalls.push({ diagram, stat, card: cardEl, hd });
      }
      smallBox.replaceChildren(...smalls.map((x) => x.card));
    }

    function render() {
      const st = ctx.state, cfg = st.config;
      const T = st.ids.length;
      const NL = cfg.n_layer, H = cfg.n_head;
      const layer = Math.min(st.sel.layer, NL - 1);
      const head = Math.min(st.sel.head, H - 1);
      const token = Math.min(st.sel.token ?? T - 1, T - 1);
      const labels = st.ids.map((id) => st.tokenizer.label(id));
      const mask = { data: ctx.engine.buildMask(cfg, T) };
      const probs = st.trace[`L${layer}.attn.probs`];
      const headWeights = (hd) => sliceOf(probs, hd);
      const query = showAll ? null : token;

      ensurePickers(NL, H, layer, head);
      modeSeg.set(showAll ? 'all' : 'one');
      if (document.activeElement !== textIn) textIn.value = st.text;
      stripBox.replaceChildren(tokenStrip({
        labels,
        selected: showAll ? null : token,
        onSelect: (i) => { big.stop(); showAll = false; ctx.store.select({ token: i }); },
      }).el);
      ensureSmalls(H);

      big.update({ labels, mask, query, weights: headWeights(head) });
      bip.update({ labels, mask, query, weights: headWeights(head) });
      smalls.forEach((x) => x.diagram.update({ labels, mask, query, weights: headWeights(x.hd) }));

      const targets = inductionTargets(st.ids);
      const stats = Array.from({ length: H }, (_, hd) => inductionStats(headWeights(hd), targets));
      const W = headWeights(head);
      const tg = targets[token];
      const quote = (k) => `“${labels[k]}”`;
      const strongest = Array.from({ length: W.cols }, (_, j) => ({ j, w: W.data[token * W.cols + j] }))
        .filter((x) => x.w > 0)
        .sort((a, b) => b.w - a.w)
        .slice(0, 3)
        .map((x) => `${quote(x.j)} (${x.j}) ${x.w.toFixed(2)}`)
        .join(', ');
      if (showAll) {
        readBig.textContent = 'Every query is drawn at once. Click a token to select one query.';
      } else {
        const top = topKeyOf(W, token);
        // An adjacent repeat has an earlier copy, but the token after that copy is the query itself.
        const adjacent = token > 0 && st.ids[token - 1] === st.ids[token];
        const induction = tg
          ? `Query ${quote(token)} at position ${token}. Its earlier copy is at ${tg.prev}, so the induction target is ${tg.target} (${quote(tg.target)}). Head h${head} puts ${fmt(W.data[token * W.cols + tg.target] * 100, 1)}% of its weight there${top === tg.target ? ', and that is its strongest key.' : `, and its strongest key is ${quote(top)} (${top}).`}`
          : adjacent
            ? `Query ${quote(token)} at position ${token} repeats the token just before it. Its earlier copy is the previous token, so the token after that copy is the query itself, and there is no induction target.`
            : `Query ${quote(token)} at position ${token} has no earlier copy, so it has no induction target.`;
        readBig.textContent = `${induction} Strongest keys: ${strongest}.`;
      }
      readBip.textContent = 'Queries are on the top row and keys on the bottom row. Hover a key to see which queries read it.';

      const here = stats[head];
      if (!here.n) {
        inductionText.textContent = 'This text has no repeated tokens, so no induction score can be measured. Load the induction preset to see one.';
      } else {
        const best = bestIndex(stats);
        inductionText.textContent = `Layer ${layer}, head h${head}: at ${here.n} repeated positions the head puts ${fmt(here.mass * 100, 0)}% of its weight on the induction target, on average. The target is its strongest key at ${here.top} of ${here.n} positions. ${best === head ? 'It is the strongest induction head in this layer.' : `The strongest induction head in this layer is h${best}.`}`;
      }
      const best = bestIndex(stats);
      smalls.forEach((x) => {
        const s = stats[x.hd];
        x.stat.textContent = s.n
          ? `induction ${fmt(s.mass, 2)} · top key at ${s.top} of ${s.n}${x.hd === best ? ' · strongest' : ''}`
          : 'no repeated tokens to measure';
      });
    }

    // ---- static scaffolding
    modeBox.append(modeSeg.el);
    speedBox.append(speedSeg.el);
    dotsBox.append(dotsToggle.el);

    root.append(
      viewHeader({
        eyebrow: 'Demo',
        title: 'Attention arcs',
        lede: 'Each arc is one attention weight, from the key token it reads to the query token that reads it. Width and brightness follow the weight, and the dots run from key to query. Hover a token to preview its arcs, and click a query to select it.',
      }),
      card({ title: 'Input and selection', kind: 'attn' },
        h('div', { class: 'stack' },
          h('div', { class: 'row' },
            h('span', { class: 'eyebrow' }, 'layer'), layerBox,
            h('span', { class: 'eyebrow' }, 'head'), headBox,
            h('span', { class: 'eyebrow' }, 'queries'), modeBox),
          h('div', { class: 'row' }, playBtn, presetBtn,
            h('span', { class: 'eyebrow' }, 'speed'), speedBox, dotsBox),
          stripBox,
          textIn)),
      section({
        title: 'One token row',
        lede: 'Arcs rise from each key to the query, higher for longer distances. Masked keys are greyed with a dashed stub, so the causal triangle is visible.',
      }, card({ title: 'Selected head, one row', kind: 'attn' }, big.el, readBig)),
      section({
        title: 'Queries above, keys below',
        lede: 'The same head as two rows. The query is the token on top and the curves come from the keys below.',
      }, card({ title: 'Selected head, two rows', kind: 'attn' }, bip.el, readBip)),
      section({
        title: 'Every head in the selected layer',
        lede: 'The selected query in each head of this layer. Induction scores come from the trace.',
      }, callout('note', inductionText), smallBox),
    );

    render();
    ctx.on('trace', render);
    ctx.on('select', render);
    window.__arcsDemo = { big, bip, smalls: () => smalls.map((x) => x.diagram), render };

    return () => {
      big.destroy();
      bip.destroy();
      for (const x of smalls) x.diagram.destroy();
      smalls = [];
      if (window.__arcsDemo) delete window.__arcsDemo;
    };
  },
};

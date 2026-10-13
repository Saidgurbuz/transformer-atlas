import { h, fmt, diverging, reducedMotion, onThemeChange } from '../../ui/index.js';
import { MINI_VOCAB, MINI_IDS, MINI_TARGETS, trainingLesson } from '../../lessons/training.js';
import { useStyle, MOTION } from '../kit.js';
import { lessonFrame, lessonButton, flowBox, equation, sourceLink } from './_lesson.js';

const STEPS = ['Predict', 'Loss', 'Backpropagate', 'Update', 'Predict again'];
const PARAMETERS = [['W_E', 'wte.weight', 'embed'], ['W_Q', 'blocks.0.attn.q_proj.weight', 'q'], ['W_K', 'blocks.0.attn.k_proj.weight', 'k'], ['W_V', 'blocks.0.attn.v_proj.weight', 'v'],
  ['W_O', 'blocks.0.attn.o_proj.weight', 'o'], ['W_up', 'blocks.0.mlp.up_proj.weight', 'mlp'], ['W_down', 'blocks.0.mlp.down_proj.weight', 'mlp']];
useStyle('training', `
.tr-graph { width: 1015px; height: 282px; }
.tr-cards { display: grid; grid-template-columns: 300px 340px 335px; gap: 20px; }
.tr-card { height: 206px; padding: 18px; }
.tr-loss { font: 700 34px/1.2 var(--font-display); color: var(--c-head); margin: 8px 0; font-variant-numeric: tabular-nums; }
.tr-card:first-child .ls-meta { font-size: 14px; }
.tr-token-line { display: flex; gap: 12px; align-items: center; }
.tr-token { display: inline-grid; place-items: center; width: 42px; height: 36px; border-radius: 6px; border: 1px solid var(--line); background: color-mix(in srgb, var(--c-embed) 12%, var(--surface)); font: 600 18px var(--font-mono); }
.tr-token.target { background: color-mix(in srgb, var(--c-head) 12%, var(--surface)); }
.tr-token.ignored { border-style: dashed; color: var(--fg-3); background: var(--surface-2); }
.tr-targets { display: grid; gap: 12px; }
.tr-prob-row { display: grid; grid-template-columns: 28px 1fr 55px; gap: 10px; align-items: center; font: 500 16px var(--font-mono); }
.tr-track { height: 10px; background: var(--surface-2); border-radius: 5px; overflow: hidden; }
.tr-fill { height: 100%; width: 0; border-radius: inherit; background: var(--c-head); transition: width .6s var(--ease); }
.tr-grad { display: grid; gap: 3px; margin-top: 10px; }
.tr-grad i { display: block; height: 10px; border-radius: 2px; border: 1px solid var(--line); }
.tr-zero { font: 500 13px/1.4 var(--font-mono); color: var(--fg-3); margin-top: 8px; }
.tr-select { border: 1px solid var(--line); border-radius: 8px; background: var(--surface); color: var(--fg); padding: 8px 12px; font: 600 16px var(--font-mono); }
`);

export default {
  mount(el, ctx) {
    const f = lessonFrame(el, { title: 'One training step', sub: 'Predict the next token, propagate its error, and update the same model.', chips: [['Isolated miniature model', 'head'], ['1 block · 4 tokens · d = 8', 'resid']] });
    let profile = 'gpt', optimizer = 'adamw', rounds = 1, step = 0, parameter = 'blocks.0.attn.q_proj.weight';
    let data = trainingLesson(profile, optimizer, rounds), animations = [], previousProbs = null;
    const gpt = lessonButton('GPT-style', () => reset('gpt'), { id: 'training-gpt', 'aria-pressed': 'true' });
    const llama = lessonButton('Llama-style', () => reset('llama'), { id: 'training-llama', 'aria-pressed': 'false' });
    const opt = h('select', { id: 'training-optimizer', class: 'tr-select', 'aria-label': 'Optimizer', onChange: (e) => { optimizer = e.target.value; rounds = 1; recompute(); } },
      h('option', { value: 'adamw' }, 'AdamW'), h('option', { value: 'sgd' }, 'SGD'));
    const param = h('select', { id: 'training-parameter', class: 'tr-select', 'aria-label': 'Parameter gradient', onChange: (e) => { parameter = e.target.value; paint(); } },
      PARAMETERS.map(([label, value]) => h('option', { value, selected: value === parameter }, label)));
    const next = lessonButton('Next update', () => { rounds = rounds === 5 ? 1 : rounds + 1; recompute(); }, { id: 'training-next' });
    f.controls.append(gpt, llama, opt, param, next, lessonButton('PyTorch code', () => ctx.showReference('training_lesson.py', step === 3 ? 'make_optimizer' : 'training_step')));

    const tokens = h('div', { class: 'tr-targets' });
    f.stage.add(tokens, { x: 130, y: 280 });
    const titles = ['Embedding', 'Norm → Attention', 'Norm → MLP', 'Norm → Head'];
    const kinds = ['embed', 'attn', 'mlp', 'head'], xs = [130, 365, 620, 875], widths = [190, 210, 210, 230];
    const nodes = titles.map((title, i) => f.stage.add(flowBox(title, ['x₀', 'Mix · residual add', 'Transform · add', 'Next-token logits'][i], kinds[i], widths[i]), { x: xs[i], y: 412 }));
    const forwardArrows = [], backwardArrows = [];
    for (let i = 0; i < 3; i++) {
      forwardArrows.push(f.stage.arrow({ x: xs[i] + widths[i] + 6, y: 456 }, { x: xs[i + 1] - 12, y: 456 }, { kind: kinds[i + 1] }));
      backwardArrows.push(f.stage.arrow({ x: xs[i + 1] - 12, y: 515 }, { x: xs[i] + widths[i] + 6, y: 515 }, { kind: kinds[i + 1] }));
    }
    const gradLabel = f.stage.add(h('div', { class: 'ls-meta', style: { color: 'var(--c-head)' } }, '← chain rule through the residual branches and the sub-layers'), { x: 400, y: 545 });
    const cards = h('div', { class: 'tr-cards', dataset: { trainingResult: 'true' } });
    f.stage.add(cards, { x: 110, y: 575 });
    const loss = h('div', { class: 'ls-panel tr-card' }), confidence = h('div', { class: 'ls-panel tr-card' }), gradient = h('div', { class: 'ls-panel tr-card' });
    cards.append(loss, confidence, gradient);

    function reset(value) { profile = value; rounds = 1; recompute(); }
    function recompute() { data = trainingLesson(profile, optimizer, rounds); paint(); }
    function gradientGrid(values, shape) {
      const rows = Math.min(shape[0], 6), cols = Math.min(shape[1], 8), max = Math.max(...Array.from(values, Math.abs), 1e-12);
      const grid = h('div', { class: 'tr-grad', style: { gridTemplateColumns: `repeat(${cols}, 1fr)` } });
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) grid.append(h('i', { style: { background: diverging(values[r * shape[1] + c] / max) }, title: `[${r}, ${c}] ${fmt(values[r * shape[1] + c], 5)}` }));
      return grid;
    }
    function paint() {
      animations.forEach((a) => a.cancel()); animations = [];
      const r = data.round, after = step === 4;
      const allParameters = Object.keys(data.model.params);
      if (!allParameters.includes(parameter)) parameter = 'blocks.0.attn.q_proj.weight';
      param.replaceChildren(...allParameters.map((name) => h('option', { value: name, selected: name === parameter }, PARAMETERS.find(([, n]) => n === name)?.[0] ?? name.replace('blocks.0.', ''))));
      gpt.setAttribute('aria-pressed', String(profile === 'gpt')); llama.setAttribute('aria-pressed', String(profile === 'llama'));
      param.hidden = step < 2; next.hidden = step < 3;
      next.textContent = rounds === 5 ? 'Reset updates' : 'Next update';
      tokens.replaceChildren(
        h('div', { class: 'tr-token-line' }, h('span', { class: 'ls-meta', style: { width: '100px' } }, 'inputs'), MINI_IDS.map((id) => h('span', { class: 'tr-token' }, MINI_VOCAB[id]))),
        h('div', { class: 'tr-token-line', style: { opacity: step >= 1 ? '1' : '.3', transition: 'opacity .5s' } }, h('span', { class: 'ls-meta', style: { width: '100px' } }, 'targets'), MINI_TARGETS.map((id) => h('span', { class: ['tr-token', 'target', id < 0 && 'ignored'], title: id < 0 ? 'No next-token target; this row contributes zero loss and zero logit gradient' : 'The next input token' }, id < 0 ? '—' : MINI_VOCAB[id]))));
      const reverse = step === 2;
      forwardArrows.forEach((a) => a.setOpacity(reverse ? .2 : 1)); backwardArrows.forEach((a) => a.setOpacity(reverse ? 1 : 0));
      gradLabel.style.opacity = reverse ? '1' : '0';
      nodes.forEach((n, i) => { n.classList.toggle('on', reverse || step === 3 || i === 3 && step >= 1); n.classList.toggle('dim', step === 1 && i < 3); });
      if (reverse && !reducedMotion()) backwardArrows.slice().reverse().forEach((a, i) => a.draw(MOTION.move, i * 130));
      loss.replaceChildren(h('div', { class: 'ls-title', style: { fontSize: '21px' } }, after ? 'Loss after the update' : 'Mean next-token loss'),
        h('div', { class: 'tr-loss' }, fmt(after ? r.lossAfter : r.lossBefore, 4)), h('div', { class: 'ls-meta' }, `3 supervised positions · update ${rounds}`),
        after ? h('div', { class: 'tr-zero' }, `before this update: ${fmt(r.lossBefore, 4)}`) : h('div', { class: 'tr-zero' }, 'Last input has no target.'));
      const probs = (after ? r.after : r.trace).probs.data;
      const rows = MINI_TARGETS.slice(0, 3).map((id, t) => {
        const p = probs[t * 5 + id], fill = h('div', { class: 'tr-fill', style: { width: `${p * 100}%` } });
        if (!reducedMotion()) animations.push(fill.animate([{ width: `${(previousProbs?.[t] ?? p) * 100}%` }, { width: `${p * 100}%` }], { duration: MOTION.move, easing: MOTION.ease }));
        return h('div', { class: 'tr-prob-row' }, MINI_VOCAB[id], h('div', { class: 'tr-track' }, fill), `${(p * 100).toFixed(1)}%`);
      });
      previousProbs = MINI_TARGETS.slice(0, 3).map((id, t) => probs[t * 5 + id]);
      confidence.replaceChildren(h('div', { class: 'ls-title', style: { fontSize: '21px' } }, 'Correct next-token probability'), h('div', { class: 'tr-targets', style: { marginTop: '18px' } }, rows));
      const kind = parameter.startsWith('lm_head') ? 'head' : parameter.startsWith('wpe') ? 'pos' : /ln/.test(parameter) ? 'norm' : /mlp/.test(parameter) ? 'mlp' : /k_proj/.test(parameter) ? 'k' : /v_proj/.test(parameter) ? 'v' : /o_proj/.test(parameter) ? 'o' : 'q';
      const chosen = PARAMETERS.find(([, name]) => name === parameter) ?? [parameter.replace('blocks.0.', ''), parameter, kind];
      const values = r.gradients[parameter], originalShape = data.model.params[parameter].shape, shape = originalShape.length === 1 ? [1, originalShape[0]] : originalShape;
      const shown = step >= 2;
      gradient.style.opacity = shown ? '1' : '.25'; gradient.style.transition = 'opacity .5s';
      gradient.replaceChildren(h('div', { class: 'ls-title', style: { fontSize: '21px', color: `var(--c-${chosen[2]})` } }, `∇ ${chosen[0]}`),
        h('div', { class: 'ls-meta' }, `[${originalShape.join(' × ')}] · ${step === 3 ? 'used by the optimizer' : 'from this loss'}`), gradientGrid(values, shape),
        h('div', { class: 'tr-zero' }, `${Math.min(shape[0], 6)} × ${Math.min(shape[1], 8)} shown · per-tensor color scale`));
      const formulas = ['z=x_f\\,\\WU^\\top', '\\mathcal L=-\\frac1M\\sum_{t\\in\\mathrm{valid}}\\log p_{t,y_t}', '\\frac{\\partial\\mathcal L}{\\partial z_{t,v}}=\\frac{p_{t,v}-\\mathbf1[v=y_t]}{M}',
        optimizer === 'sgd' ? '\\theta^+=\\theta-\\eta\\,\\nabla_\\theta\\mathcal L' : '\\theta^+=(1-\\eta\\lambda)\\theta-\\eta\\frac{\\hat m}{\\sqrt{\\hat v}+\\epsilon}', 'p^+=\\softmax(z^+)'];
      f.side.replaceChildren(equation(formulas[step]), h('div', { class: 'ls-copy' }, ['Each row predicts the token after its input position.', 'M = 3. The mean excludes the unlabelled final row.', 'The ignored row has zero logit gradient. Shared embedding weights accumulate both lookup and output-projection gradients.',
        optimizer === 'sgd' ? 'SGD follows the gradient directly. This lesson uses learning rate 0.01 and no weight decay.' : 'AdamW tracks first and second moments with bias correction. Decay applies to weight matrices, not norm gains or biases.', 'Run the same inputs with the updated weights and measure the new loss.'][step]),
        h('div', { class: 'ls-check' }, '✓ gradients and updates checked against PyTorch'),
        sourceLink('AdamW', 'https://arxiv.org/abs/1711.05101', 'Algorithm 2'),
        lessonButton('Detailed maths', () => ctx.showMath(step === 3 && optimizer === 'adamw' ? 'concept:adamw' : 'concept:training-gradient')));
      cards.dataset.before = String(r.lossBefore); cards.dataset.after = String(r.lossAfter); cards.dataset.profile = profile; cards.dataset.optimizer = optimizer; cards.dataset.round = String(rounds);
      f.caption.set(['The input “cat.” produces four next-token distributions. A tiny, seeded model keeps the computation inspectable.',
        'Shift the targets by one position. Only “a”, “t” and “.” supervise the loss; the final row is ignored.',
        'The error travels backward through the head, MLP, attention and embeddings. Every trainable parameter receives its own accumulated gradient.',
        'The optimizer uses these gradients to update each unique parameter once. Weight tying combines the two contributions before the update.',
        'A fresh forward pass shows what changed. A single step is an update on this example, not evidence of generalization.'][step]);
    }
    ctx.setSteps(STEPS, (i) => { step = i; paint(); });
    paint();
    const offTheme = onThemeChange(paint);
    return { pause() { animations.forEach((a) => a.cancel()); }, destroy() { offTheme(); animations.forEach((a) => a.cancel()); f.destroy(); } };
  },
};

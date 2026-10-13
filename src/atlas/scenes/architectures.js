import { h } from '../../ui/index.js';
import { useStyle } from '../kit.js';
import { createModel, countParams } from '../../engine/index.js';
import { MINI_PROFILES } from '../../lessons/training.js';
import { createEncoderDecoder } from '../../lessons/encoder-decoder.js';
import { lessonFrame, lessonButton, equation, sourceLink } from './_lesson.js';
const STEPS = ['Topology & norm', 'Positions & heads', 'MLP & weights'];
useStyle('architecture-comparison', `
.ac-cards { display: grid; grid-template-columns: repeat(3, 325px); gap: 20px; }
.ac-card { height: 475px; }
.ac-card .ls-title { font-size: 25px; }
.ac-row { border-left: 3px solid transparent; padding: 8px 8px 8px 12px; margin-top: 12px; transition: border-color .6s, background .6s; }
.ac-row.on { border-color: var(--kind); background: color-mix(in srgb, var(--kind) 8%, transparent); border-radius: 0 8px 8px 0; }
.ac-row strong { font: 650 19px/1.3 var(--font-display); color: var(--kind); display: block; margin-bottom: 6px; }
.ac-row .ls-copy { font-size: 16px; line-height: 1.4; }
.ac-count { margin-top: 8px; font: 500 13px/1.4 var(--font-mono); color: var(--fg-3); }
`);
export default {
  mount(el, ctx) {
    const f = lessonFrame(el, { title: 'Three Transformer recipes', sub: 'Compare the structural choices that change equations and code.', chips: [['Architecture families', 'resid'], ['Miniatures · not checkpoint specifications', 'norm']] });
    let step = 0, counts = false;
    const models = [createEncoderDecoder(), createModel(MINI_PROFILES.gpt.config, 17), createModel(MINI_PROFILES.llama.config, 17)];
    const cards = h('div', { class: 'ac-cards' }); f.stage.add(cards, { x: 110, y: 280 });
    const toggle = lessonButton('Show mini-model counts', () => { counts = !counts; paint(); }, { id: 'architectures-counts', 'aria-pressed': 'false' });
    f.controls.append(toggle, lessonButton('Run encoder–decoder', () => ctx.go('encoder-decoder')), lessonButton('Compare training profiles', () => ctx.go('training')));
    const names = ['2017-style', 'GPT-2-style', 'Llama-style · GQA'];
    function paint() {
      toggle.setAttribute('aria-pressed', String(counts)); toggle.textContent = counts ? 'Hide mini-model counts' : 'Show mini-model counts';
      cards.replaceChildren(...models.map((model, i) => {
        const c = model.config, encoder = i === 0;
        const rows = [
          [encoder ? 'Encoder + decoder' : 'Decoder only', `${c.norm_placement === 'post' ? 'Post' : 'Pre'}-${c.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm'} · ${encoder ? '3 decoder sub-layers' : '2 sub-layers + final norm'}`, 'norm'],
          [c.pos_encoding === 'learned' ? 'Learned positions' : c.pos_encoding === 'rope' ? 'Rotary Q & K' : 'Sinusoidal positions', `${c.n_head} query heads · ${c.n_kv_head} KV heads${encoder ? ' · plus cross-attention' : ''}`, 'attn'],
          [c.activation === 'swiglu' ? 'Gated SwiGLU' : c.activation.toUpperCase(), `FFN width ${c.d_ff} · ${c.bias ? 'with' : 'without'} biases · ${c.tie_weights ? 'tied' : 'untied'} output weights`, 'mlp'],
        ];
        const card = h('div', { class: 'ls-panel ac-card', dataset: { architecture: i } }, h('div', { class: 'ls-title' }, names[i]),
          h('div', { class: 'ls-meta' }, encoder ? '1 encoder + 1 decoder block' : '1 decoder block'),
          ...rows.map(([title, text, kind], r) => h('div', { class: ['ac-row', r === step && 'on'], style: { '--kind': `var(--c-${kind})` } }, h('strong', null, title), h('div', { class: 'ls-copy' }, text))),
          counts ? h('div', { class: 'ac-count' }, `${countParams(model).toLocaleString()} params · V${c.vocab_size} · d${c.d_model}`) : null,
          lessonButton('Source', () => ctx.showReference(encoder ? 'encoder_decoder.py' : 'training_lesson.py', encoder ? 'TinyEncoderDecoder' : 'architecture_profiles'), { style: { marginTop: '12px', fontSize: '15px', padding: '7px 12px' } }));
        return card;
      }));
      const formulas = ['\\begin{aligned}\\text{pre: }&x+f(N(x))\\\\ \\text{post: }&N(x+f(x))\\end{aligned}',
        'h_{\\mathrm{KV}}=\\left\\lfloor h/(H/H_{\\mathrm{KV}})\\right\\rfloor', '\\begin{aligned}h_g&=\\mathrm{SiLU}(xW_g^\\top)\\\\ h_u&=xW_u^\\top\\\\ \\mathrm{FFN}(x)&=(h_g\\odot h_u)W_d^\\top\\end{aligned}'];
      f.side.replaceChildren(equation(formulas[step]), h('div', { class: 'ls-copy' }, [
        'Normalization placement changes both the forward path and the gradient path. Encoder–decoder blocks also add a separate cross-attention sub-layer.',
        'Learned and sinusoidal positions are added to embeddings. RoPE rotates Q/K inside attention. GQA shares K/V across a group of query heads.',
        'A gated FFN has three projection matrices; a conventional FFN has two. Tied embeddings share one parameter and accumulate gradients from both uses.'
      ][step]), sourceLink('Original Transformer', 'https://arxiv.org/abs/1706.03762', '§3'),
      sourceLink('GPT-2', 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf', '§2.3'),
      sourceLink('LLaMA', 'https://arxiv.org/abs/2302.13971', '§2.2'), sourceLink('Llama 3', 'https://arxiv.org/abs/2407.21783', '§3.2 · GQA'),
      lessonButton('Relevant maths', () => ctx.showMath('concept:architecture-choices')));
      f.caption.set('These are small structural examples, not replicas of every production convention. Llama versions differ; the GQA recipe combines the original LLaMA choices with later GQA. Miniature counts include their explicit vocabulary and FFN sizes, so they are not a controlled efficiency comparison.');
    }
    ctx.setSteps(STEPS, i => { step = i; paint(); }); paint(); return { destroy: f.destroy };
  },
};

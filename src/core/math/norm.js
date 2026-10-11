// Maths for the normalisation layers and the residual stream: the LayerNorm / RMSNorm
// blocks (ln1, ln2, lnf), the residual sums (resid_mid, resid_out), the post-LN sums
// (add1, add2), and three concept entries (LayerNorm backward, pre- vs post-LN blocks,
// the residual stream). Entry schema (see src/core/math/index.js):
//   title, tex (display), compact (one-line, for tooltips), symbols [{tex, text, dims}],
//   derivation [{title, steps: [tex], note}], live(ctx) -> {tex, result, expected, note}
import { at2, row2, streamBefore, num } from './util.js';

// ------------------------------------------------------------------ live helpers

// Gain, and shift when the norm has one, of the norm named e.g. 'blocks.0.ln1'.
// RMSNorm has no shift; LayerNorm has one only with cfg.bias (model.js creates the parameter only then).
function normParams(model, name, cfg) {
  const b = model.params[`${name}.bias`];
  return {
    gamma: Array.from(model.params[`${name}.weight`].data),
    beta: cfg.norm === 'layernorm' && cfg.bias && b ? Array.from(b.data) : null,
  };
}

// The display lines as one left-aligned block.
const aligned = (lines) => `\\begin{aligned}${lines.map((l) => ` &${l}`).join(' \\\\')} \\end{aligned}`;

// One output cell of a norm, recomputed from the whole input row x (d numbers).
// sym: TeX name of the input tensor; out: TeX name of the output tensor.
function normCell({ x, row, col, cfg, P, sym, out }) {
  const d = x.length, eps = cfg.norm_eps, xc = x[col], g = P.gamma[col], E = num(eps, 5);
  if (cfg.norm === 'rmsnorm') {
    const ms = x.reduce((s, v) => s + v * v, 0) / d;
    const rms = Math.sqrt(ms + eps);
    const result = (g * xc) / rms;
    return {
      result,
      lines: [
        `\\operatorname{ms}_{${row}} = \\frac{1}{${d}}\\sum_c {${sym}_{${row},c}}^{2} = ${num(ms, 4)}`,
        `\\operatorname{rms}_{${row}} = \\sqrt{${num(ms, 4)}+${E}} = ${num(rms, 4)}`,
        `${out}_{${row},${col}} = \\gamma_{${col}}\\,\\frac{${sym}_{${row},${col}}}{\\operatorname{rms}_{${row}}} = (${num(g, 4)})\\,\\frac{(${num(xc, 4)})}{(${num(rms, 4)})} = ${num(result, 4)}`,
      ],
    };
  }
  const mu = x.reduce((s, v) => s + v, 0) / d;
  const v2 = x.reduce((s, v) => s + (v - mu) ** 2, 0) / d;
  const xhat = (xc - mu) / Math.sqrt(v2 + eps);
  const b = P.beta ? P.beta[col] : 0;
  const result = g * xhat + b;
  return {
    result,
    lines: [
      `\\mu_{${row}} = \\frac{1}{${d}}\\sum_c ${sym}_{${row},c} = ${num(mu, 4)}`,
      `\\sigma^2_{${row}} = \\frac{1}{${d}}\\sum_c \\left({${sym}_{${row},c}}-\\mu_{${row}}\\right)^{2} = ${num(v2, 4)}`,
      `\\hat x_{${row},${col}} = \\frac{${sym}_{${row},${col}}-\\mu_{${row}}}{\\sqrt{\\sigma^2_{${row}}+\\varepsilon}} = \\frac{(${num(xc, 4)})-(${num(mu, 4)})}{\\sqrt{${num(v2, 4)}+${E}}} = ${num(xhat, 4)}`,
      `${out}_{${row},${col}} = \\gamma_{${col}}\\,\\hat x_{${row},${col}}${P.beta ? ` + \\beta_{${col}}` : ''} = (${num(g, 4)})(${num(xhat, 4)})${P.beta ? ` + (${num(b, 4)})` : ''} = ${num(result, 4)}`,
    ],
  };
}

export const ENTRIES = {
  'L{i}.ln1': {
    title: 'Norm before attention',
    tex: '\\begin{aligned} &\\mu_t = \\frac{1}{d}\\sum_{c=1}^{d}\\xres_{t,c}, \\qquad \\sigma_t^2 = \\frac{1}{d}\\sum_{c=1}^{d}\\left(\\xres_{t,c}-\\mu_t\\right)^2 \\\\ &\\text{LayerNorm:}\\quad h_{t,c} = \\gamma_c\\,\\frac{\\xres_{t,c}-\\mu_t}{\\sqrt{\\sigma_t^2+\\varepsilon}} + \\beta_c \\qquad \\text{(}\\beta\\text{ only if the model has biases)} \\\\ &\\text{RMSNorm:}\\quad h_{t,c} = \\gamma_c\\,\\frac{\\xres_{t,c}}{\\operatorname{rms}_t}, \\qquad \\operatorname{rms}_t = \\sqrt{\\frac{1}{d}\\sum_{c=1}^{d}\\xres_{t,c}^{2}+\\varepsilon} \\end{aligned}',
    compact: 'h_{t,c} = \\gamma_c\\,\\frac{x_{t,c}-\\mu_t}{\\sqrt{\\sigma_t^2+\\varepsilon}}+\\beta_c\\,\\text{(only if biases)}',
    symbols: [
      { tex: '\\xres', text: 'residual stream entering the block (the stream before layer i)', dims: ['T', 'd'] },
      { tex: 'h', text: 'normalised stream, the input to attention', dims: ['T', 'd'] },
      { tex: '\\mu_t', text: 'mean of token t’s d features', dims: ['T'] },
      { tex: '\\sigma_t^2', text: 'biased variance of token t’s features (divide by d)', dims: ['T'] },
      { tex: '\\operatorname{rms}_t', text: 'root mean square of token t, the RMSNorm denominator (RMSNorm only)', dims: ['T'] },
      { tex: '\\gamma', text: 'learned gain, one per feature', dims: ['d'] },
      { tex: '\\beta', text: 'learned shift, LayerNorm with bias only', dims: ['d'] },
      { tex: '\\varepsilon', text: 'norm_eps, keeps the division finite', dims: [] },
    ],
    derivation: [
      {
        title: 'Each token is normalised on its own',
        steps: [
          '\\mu_t = \\frac{1}{d}\\sum_{c=1}^{d}\\xres_{t,c}, \\qquad \\sigma_t^2 = \\frac{1}{d}\\sum_{c=1}^{d}\\left(\\xres_{t,c}-\\mu_t\\right)^2',
          '\\hat x_{t,c} = \\frac{\\xres_{t,c}-\\mu_t}{\\sqrt{\\sigma_t^2+\\varepsilon}} \\;\\Rightarrow\\; \\sum_{c=1}^{d}\\hat x_{t,c} = 0, \\qquad \\frac{1}{d}\\sum_{c=1}^{d}\\hat x_{t,c}^{2} = \\frac{\\sigma_t^2}{\\sigma_t^2+\\varepsilon} \\approx 1',
        ],
        note: 'The statistics use only token t’s own features, so no other token or batch element enters. The variance is biased (divided by d), as in PyTorch and reference/model.py. Ba, Kiros and Hinton (2016) introduced this per-example normalisation, unlike BatchNorm, which pools over the batch.',
      },
      {
        title: 'Gain and shift',
        steps: ['h_{t,c} = \\gamma_c\\,\\hat x_{t,c} + \\beta_c'],
        note: 'γ and β are per feature, not per token, so they cannot undo the per-token normalisation. They only set each feature’s scale and offset, so attention sees inputs whose scale is set by γ, however large the residual stream has grown. β exists only in LayerNorm with biases.',
      },
      {
        title: 'RMSNorm: rescale without re-centring',
        steps: ['\\operatorname{rms}_t = \\sqrt{\\frac{1}{d}\\sum_{c=1}^{d}\\xres_{t,c}^{2}+\\varepsilon}, \\qquad h_{t,c} = \\gamma_c\\,\\frac{\\xres_{t,c}}{\\operatorname{rms}_t}'],
        note: 'Zhang and Sennrich (2019) tested dropping the mean subtraction: rescaling alone performed comparably at lower cost. LLaMA-family models use this form.',
      },
    ],
    live({ trace, cfg, model, layer, row, col }) {
      const n = normCell({ x: row2(streamBefore(trace, layer), row), row, col, cfg, P: normParams(model, `blocks.${layer}.ln1`, cfg), sym: '\\xres', out: 'h' });
      return {
        tex: aligned(n.lines), result: n.result, expected: at2(trace[`L${layer}.ln1`], row, col),
        note: `The statistics use all ${cfg.d_model} features of token ${row}.`,
      };
    },
  },

  'L{i}.ln2': {
    title: 'Norm before MLP',
    tex: '\\begin{aligned} &\\mu_t = \\frac{1}{d}\\sum_{c=1}^{d}x^{\\text{mid}}_{t,c}, \\qquad \\sigma_t^2 = \\frac{1}{d}\\sum_{c=1}^{d}\\left(x^{\\text{mid}}_{t,c}-\\mu_t\\right)^2 \\\\ &\\text{LayerNorm:}\\quad h_{t,c} = \\gamma_c\\,\\frac{x^{\\text{mid}}_{t,c}-\\mu_t}{\\sqrt{\\sigma_t^2+\\varepsilon}} + \\beta_c \\qquad \\text{(}\\beta\\text{ only if the model has biases)} \\\\ &\\text{RMSNorm:}\\quad h_{t,c} = \\gamma_c\\,\\frac{x^{\\text{mid}}_{t,c}}{\\operatorname{rms}_t}, \\qquad \\operatorname{rms}_t = \\sqrt{\\frac{1}{d}\\sum_{c=1}^{d}\\left(x^{\\text{mid}}_{t,c}\\right)^{2}+\\varepsilon} \\end{aligned}',
    compact: 'h_{t,c} = \\gamma_c\\,\\frac{x^{\\text{mid}}_{t,c}-\\mu_t}{\\sqrt{\\sigma_t^2+\\varepsilon}}+\\beta_c\\,\\text{(only if biases)}',
    symbols: [
      { tex: 'x^{\\text{mid}}', text: 'stream after attention, the residual input of the MLP sub-layer', dims: ['T', 'd'] },
      { tex: 'h', text: 'normalised stream, the input to the MLP', dims: ['T', 'd'] },
      { tex: '\\mu_t', text: 'mean of token t’s d features', dims: ['T'] },
      { tex: '\\sigma_t^2', text: 'biased variance of token t’s features (divide by d)', dims: ['T'] },
      { tex: '\\operatorname{rms}_t', text: 'root mean square of token t, the RMSNorm denominator (RMSNorm only)', dims: ['T'] },
      { tex: '\\gamma', text: 'gain of this block’s second norm (blocks.i.ln2.weight)', dims: ['d'] },
      { tex: '\\beta', text: 'shift of the second norm, LayerNorm with bias only', dims: ['d'] },
      { tex: '\\varepsilon', text: 'norm_eps, keeps the division finite', dims: [] },
      { tex: '\\Wup', text: 'MLP up-projection weight, shape F × d', dims: ['F', 'd'] },
      { tex: '\\sigma_W', text: 'standard deviation of the random initialisation of W_up (0.02 by default)', dims: [] },
    ],
    derivation: [
      {
        title: 'Same norm, second position in the block',
        steps: ['h_{t,c} = \\gamma_c\\,\\frac{x^{\\text{mid}}_{t,c}-\\mu_t}{\\sqrt{\\sigma_t^2+\\varepsilon}} + \\beta_c'],
        note: 'Each block has two norms with separate parameters (blocks.i.ln1 and blocks.i.ln2). The MLP reads the normalised stream after attention, so its input scale does not depend on depth. β is present only when the model has biases.',
      },
      {
        title: 'Why the MLP input scale matters at initialisation',
        steps: ['\\operatorname{Var}\\big((\\Wup h)_o\\big) = \\sigma_W^2\\sum_{c=1}^{d}h_{t,c}^{2} \\approx d\\,\\sigma_W^2'],
        note: 'The variance is over the random initialisation of W_up (zero-mean, i.i.d. Gaussian entries with standard deviation σ_W), with h fixed. With γ = 1 the mean square of h is about 1, and σ_W = 0.02 by default. An up-projection output then has standard deviation about √d·σ_W (0.16 for d = 64) at every layer, so the GELU sees a predictable range.',
      },
    ],
    live({ trace, cfg, model, layer, row, col }) {
      const n = normCell({ x: row2(trace[`L${layer}.resid_mid`], row), row, col, cfg, P: normParams(model, `blocks.${layer}.ln2`, cfg), sym: 'x^{\\text{mid}}', out: 'h' });
      return {
        tex: aligned(n.lines), result: n.result, expected: at2(trace[`L${layer}.ln2`], row, col),
        note: `The statistics use all ${cfg.d_model} features of token ${row} after attention.`,
      };
    },
  },

  lnf: {
    title: 'Final norm',
    tex: '\\begin{aligned} &\\mu_t = \\frac{1}{d}\\sum_{c=1}^{d}\\xres_{t,c}, \\qquad \\sigma_t^2 = \\frac{1}{d}\\sum_{c=1}^{d}\\left(\\xres_{t,c}-\\mu_t\\right)^2 \\\\ &\\text{LayerNorm:}\\quad h^{f}_{t,c} = \\gamma_c\\,\\frac{\\xres_{t,c}-\\mu_t}{\\sqrt{\\sigma_t^2+\\varepsilon}} + \\beta_c \\qquad \\text{(}\\beta\\text{ only if the model has biases)} \\\\ &\\text{RMSNorm:}\\quad h^{f}_{t,c} = \\gamma_c\\,\\frac{\\xres_{t,c}}{\\operatorname{rms}_t}, \\qquad \\operatorname{rms}_t = \\sqrt{\\frac{1}{d}\\sum_{c=1}^{d}\\xres_{t,c}^{2}+\\varepsilon} \\\\ &\\text{logits}_{t,v} = \\sum_{c=1}^{d} h^{f}_{t,c}\\,(\\WU)_{v,c} \\end{aligned}',
    compact: 'h^{f}_{t,c} = \\gamma_c\\,\\frac{x_{t,c}-\\mu_t}{\\sqrt{\\sigma_t^2+\\varepsilon}}+\\beta_c\\,\\text{(only if biases)}',
    symbols: [
      { tex: '\\xres', text: 'stream entering the final norm: the output of the last block (the embedding if there are no blocks)', dims: ['T', 'd'] },
      { tex: 'h^{f}', text: 'final normalised stream, the input to the unembedding', dims: ['T', 'd'] },
      { tex: '\\mu_t', text: 'mean of token t’s d features', dims: ['T'] },
      { tex: '\\sigma_t^2', text: 'biased variance of token t’s features (divide by d)', dims: ['T'] },
      { tex: '\\operatorname{rms}_t', text: 'root mean square of token t, the RMSNorm denominator (RMSNorm only)', dims: ['T'] },
      { tex: '\\gamma', text: 'final gain (ln_f.weight)', dims: ['d'] },
      { tex: '\\beta', text: 'final shift (ln_f.bias), LayerNorm with bias only', dims: ['d'] },
      { tex: '\\varepsilon', text: 'norm_eps, keeps the division finite', dims: [] },
      { tex: '\\WU', text: 'unembedding matrix (tied to W_E when tie_weights)', dims: ['V', 'd'] },
    ],
    derivation: [{
      title: 'Why normalise before the unembedding',
      steps: ['\\text{logits}_{t,v} = \\sum_{c=1}^{d} h^{f}_{t,c}\\,(\\WU)_{v,c}'],
      note: 'Each logit is a dot product of h^f with one row of W_U, so the logits scale with the length of h^f. The final norm keeps that length set by the gain and by d, not by how far the residual stream has grown across layers. Pre-LN streams are never normalised in place, so this step is needed.',
    }],
    live({ trace, cfg, model, row, col }) {
      const n = normCell({ x: row2(streamBefore(trace, cfg.n_layer), row), row, col, cfg, P: normParams(model, 'ln_f', cfg), sym: '\\xres', out: 'h^{f}' });
      return {
        tex: aligned(n.lines), result: n.result, expected: at2(trace.lnf, row, col),
        note: cfg.n_layer ? `Input: the output of block ${cfg.n_layer - 1}.` : 'Input: the embedding (no blocks).',
      };
    },
  },

  'L{i}.resid_mid': {
    title: 'Residual after attention',
    tex: '\\begin{aligned} &\\text{pre-LN:}\\quad x^{\\text{mid}} = \\xres + \\Attn(\\LN_1(\\xres)) \\\\ &\\text{post-LN:}\\quad x^{\\text{mid}} = \\LN_1(\\xres + \\Attn(\\xres)) \\\\ &\\text{residuals off:}\\quad x^{\\text{mid}} = a \\;\\text{(pre-LN)}, \\qquad x^{\\text{mid}} = \\LN_1(a) \\;\\text{(post-LN)} \\end{aligned}',
    compact: 'x^{\\text{mid}}_{t,c} = x_{t,c} + a_{t,c} \\;\\;(\\text{post-LN: } \\LN(x+a))',
    symbols: [
      { tex: '\\xres', text: 'stream entering the block', dims: ['T', 'd'] },
      { tex: 'a', text: 'attention output after W_O (trace L{i}.attn.out)', dims: ['T', 'd'] },
      { tex: 'x^{\\text{mid}}', text: 'stream after the attention sub-layer (trace L{i}.resid_mid)', dims: ['T', 'd'] },
      { tex: 's^{(1)}', text: 'post-LN only: the sum x + a before the first norm (trace L{i}.add1)', dims: ['T', 'd'] },
      { tex: '\\hat s', text: 'post-LN only: s^{(1)} normalised per token, (s − μ_t)/√(σ_t² + ε)', dims: ['T', 'd'] },
      { tex: '\\mu_t', text: 'post-LN only: mean of the d features of s^{(1)} for token t', dims: ['T'] },
      { tex: '\\sigma_t^2', text: 'post-LN only: biased variance of the features of s^{(1)} for token t', dims: ['T'] },
      { tex: '\\varepsilon', text: 'norm_eps, keeps the division finite', dims: [] },
      { tex: '\\gamma', text: 'gain of ln1, the first norm of the block', dims: ['d'] },
      { tex: 'J_{\\mathrm{LN}}', text: 'Jacobian of the norm, diag(γ)(I − 1·1ᵀ/d − ŝŝᵀ/d)/√(σ²+ε) (concept:pre-vs-post)', dims: ['d', 'd'] },
      { tex: 'I', text: 'identity matrix', dims: ['d', 'd'] },
      { tex: '\\mathbf{1}', text: 'all-ones vector', dims: ['d'] },
    ],
    derivation: [
      {
        title: 'Pre-LN: the residual add is an identity path',
        steps: ['\\frac{\\partial x^{\\text{mid}}}{\\partial \\xres} = I + \\frac{\\partial a}{\\partial \\xres}'],
        note: 'In pre-LN, the sum passes the gradient through unchanged (the I term) and adds the attention correction on top. He et al. (2016) introduced this skip connection so that a layer only has to learn a correction to its input.',
      },
      {
        title: 'Post-LN: the norm sits on the sum',
        steps: [
          '\\frac{\\partial x^{\\text{mid}}}{\\partial \\xres} = J_{\\mathrm{LN}}\\Big(I + \\frac{\\partial a}{\\partial \\xres}\\Big)',
          '\\frac{\\partial \\LN(s)}{\\partial s} = \\frac{\\operatorname{diag}(\\gamma)}{\\sqrt{\\sigma_t^2+\\varepsilon}}\\Big(I - \\frac{\\mathbf{1}\\mathbf{1}^{\\top}}{d} - \\frac{\\hat s\\,\\hat s^{\\top}}{d}\\Big)',
        ],
        note: 'With s^(1) = x + a, the identity term is multiplied by this Jacobian: each feature is rescaled by γ/√(σ²+ε), and the mean and the normalised direction are projected out. See concept:pre-vs-post.',
      },
    ],
    live({ trace, cfg, model, layer, row, col }) {
      const expected = at2(trace[`L${layer}.resid_mid`], row, col);
      const inp = row2(streamBefore(trace, layer), row), att = row2(trace[`L${layer}.attn.out`], row);
      if (cfg.norm_placement === 'pre') {
        const result = cfg.residual ? inp[col] + att[col] : att[col];
        return {
          tex: cfg.residual
            ? `x^{\\text{mid}}_{${row},${col}} = \\xres_{${row},${col}} + a_{${row},${col}} = (${num(inp[col], 4)}) + (${num(att[col], 4)}) = ${num(result, 4)}`
            : `x^{\\text{mid}}_{${row},${col}} = a_{${row},${col}} = ${num(result, 4)}`,
          result, expected,
          note: cfg.residual ? 'Pre-LN: a plain sum, so the stream is not normalised here.' : 'Residuals are off: the attention output replaces the stream.',
        };
      }
      const s = cfg.residual ? inp.map((v, c) => v + att[c]) : att;
      const n = normCell({ x: s, row, col, cfg, P: normParams(model, `blocks.${layer}.ln1`, cfg), sym: 's^{(1)}', out: 'x^{\\text{mid}}' });
      const sum = cfg.residual
        ? `s^{(1)}_{${row},${col}} = \\xres_{${row},${col}} + a_{${row},${col}} = (${num(inp[col], 4)}) + (${num(att[col], 4)}) = ${num(s[col], 4)}`
        : `s^{(1)}_{${row},${col}} = a_{${row},${col}} = ${num(s[col], 4)}`;
      return {
        tex: aligned([sum, ...n.lines]), result: n.result, expected,
        note: 'Post-LN: the sum is normalised by ln1, so the stream is normalised after this step.',
      };
    },
  },

  'L{i}.resid_out': {
    title: 'Block output',
    tex: '\\begin{aligned} &\\text{pre-LN:}\\quad x^{\\text{out}} = x^{\\text{mid}} + \\MLP(\\LN_2(x^{\\text{mid}})) \\\\ &\\text{post-LN:}\\quad x^{\\text{out}} = \\LN_2(x^{\\text{mid}} + \\MLP(x^{\\text{mid}})) \\\\ &\\text{residuals off:}\\quad x^{\\text{out}} = m \\;\\text{(pre-LN)}, \\qquad x^{\\text{out}} = \\LN_2(m) \\;\\text{(post-LN)} \\end{aligned}',
    compact: 'x^{\\text{out}}_{t,c} = x^{\\text{mid}}_{t,c} + m_{t,c} \\;\\;(\\text{post-LN: } \\LN(x^{\\text{mid}}+m))',
    symbols: [
      { tex: 'x^{\\text{mid}}', text: 'stream after attention (trace L{i}.resid_mid)', dims: ['T', 'd'] },
      { tex: 'm', text: 'MLP output after the down-projection (trace L{i}.mlp.out)', dims: ['T', 'd'] },
      { tex: 'x^{\\text{out}}', text: 'block output, the input to the next block (trace L{i}.resid_out)', dims: ['T', 'd'] },
      { tex: 's^{(2)}', text: 'post-LN only: the sum x^mid + m before the second norm (trace L{i}.add2)', dims: ['T', 'd'] },
      { tex: '\\hat s', text: 'post-LN only: s^{(2)} normalised per token, (s − μ_t)/√(σ_t² + ε)', dims: ['T', 'd'] },
      { tex: '\\mu_t', text: 'post-LN only: mean of the d features of s^{(2)} for token t', dims: ['T'] },
      { tex: '\\sigma_t^2', text: 'post-LN only: biased variance of the features of s^{(2)} for token t', dims: ['T'] },
      { tex: '\\varepsilon', text: 'norm_eps, keeps the division finite', dims: [] },
      { tex: '\\gamma', text: 'gain of ln2, the second norm of the block', dims: ['d'] },
      { tex: 'J_{\\mathrm{LN}}', text: 'Jacobian of the norm, diag(γ)(I − 1·1ᵀ/d − ŝŝᵀ/d)/√(σ²+ε) (concept:pre-vs-post)', dims: ['d', 'd'] },
      { tex: 'I', text: 'identity matrix', dims: ['d', 'd'] },
      { tex: '\\mathbf{1}', text: 'all-ones vector', dims: ['d'] },
    ],
    derivation: [
      {
        title: 'Pre-LN: the stream grows by two terms per block',
        steps: ['\\xres_{\\ell+1}-\\xres_\\ell = a_\\ell + m_\\ell'],
        note: 'In pre-LN, each block adds exactly its two sub-layer outputs to the stream, and nothing else changes it. Elhage et al. (2021) call this the residual stream. Post-LN breaks the pattern, since the norm rescales the whole sum after each block.',
      },
      {
        title: 'Pre-LN: identity path through the second add',
        steps: ['\\frac{\\partial x^{\\text{out}}}{\\partial x^{\\text{mid}}} = I + \\frac{\\partial m}{\\partial x^{\\text{mid}}}'],
        note: 'Same structure as the first add in pre-LN: the gradient reaches x^{mid} directly, plus the MLP correction. See concept:residual-stream.',
      },
      {
        title: 'Post-LN: the second norm sits on the sum',
        steps: [
          '\\frac{\\partial x^{\\text{out}}}{\\partial x^{\\text{mid}}} = J_{\\mathrm{LN}}\\Big(I + \\frac{\\partial m}{\\partial x^{\\text{mid}}}\\Big)',
          '\\frac{\\partial \\LN(s)}{\\partial s} = \\frac{\\operatorname{diag}(\\gamma)}{\\sqrt{\\sigma_t^2+\\varepsilon}}\\Big(I - \\frac{\\mathbf{1}\\mathbf{1}^{\\top}}{d} - \\frac{\\hat s\\,\\hat s^{\\top}}{d}\\Big)',
        ],
        note: 'With s^(2) = x^mid + m, the identity term is multiplied by this Jacobian: each feature is rescaled by γ/√(σ²+ε), and the mean and the normalised direction are projected out. See concept:pre-vs-post.',
      },
    ],
    live({ trace, cfg, model, layer, row, col }) {
      const expected = at2(trace[`L${layer}.resid_out`], row, col);
      const mid = row2(trace[`L${layer}.resid_mid`], row), mlp = row2(trace[`L${layer}.mlp.out`], row);
      if (cfg.norm_placement === 'pre') {
        const result = cfg.residual ? mid[col] + mlp[col] : mlp[col];
        return {
          tex: cfg.residual
            ? `x^{\\text{out}}_{${row},${col}} = x^{\\text{mid}}_{${row},${col}} + m_{${row},${col}} = (${num(mid[col], 4)}) + (${num(mlp[col], 4)}) = ${num(result, 4)}`
            : `x^{\\text{out}}_{${row},${col}} = m_{${row},${col}} = ${num(result, 4)}`,
          result, expected,
          note: cfg.residual ? 'Pre-LN: the block output is the stream after both sub-layers.' : 'Residuals are off: the MLP output replaces the stream.',
        };
      }
      const s = cfg.residual ? mid.map((v, c) => v + mlp[c]) : mlp;
      const n = normCell({ x: s, row, col, cfg, P: normParams(model, `blocks.${layer}.ln2`, cfg), sym: 's^{(2)}', out: 'x^{\\text{out}}' });
      const sum = cfg.residual
        ? `s^{(2)}_{${row},${col}} = x^{\\text{mid}}_{${row},${col}} + m_{${row},${col}} = (${num(mid[col], 4)}) + (${num(mlp[col], 4)}) = ${num(s[col], 4)}`
        : `s^{(2)}_{${row},${col}} = m_{${row},${col}} = ${num(s[col], 4)}`;
      return {
        tex: aligned([sum, ...n.lines]), result: n.result, expected,
        note: 'Post-LN: the sum is normalised by ln2, which becomes the input of the next block.',
      };
    },
  },

  'L{i}.add1': {
    title: 'Sum before post-norm (attn)',
    tex: 's^{(1)} = \\xres + \\Attn(\\xres), \\qquad x^{\\text{mid}} = \\LN_1\\big(s^{(1)}\\big) \\qquad \\text{(residuals off: }s^{(1)} = a\\text{)}',
    compact: 's^{(1)}_{t,c} = x_{t,c} + a_{t,c}',
    symbols: [
      { tex: 's^{(1)}', text: 'sum before the first post-norm (trace L{i}.add1)', dims: ['T', 'd'] },
      { tex: '\\xres', text: 'stream entering the block, the raw input to attention', dims: ['T', 'd'] },
      { tex: 'a', text: 'attention output on the raw stream, a = Attn(x)', dims: ['T', 'd'] },
      { tex: 'x^{\\text{mid}}', text: 'output of the first post-norm (trace L{i}.resid_mid)', dims: ['T', 'd'] },
      { tex: 'J_{\\mathrm{LN}}', text: 'Jacobian of the norm, diag(γ)(I − 1·1ᵀ/d − x̂ x̂ᵀ/d)/√(σ²+ε) (concept:pre-vs-post)', dims: ['d', 'd'] },
      { tex: 'I', text: 'identity matrix', dims: ['d', 'd'] },
    ],
    derivation: [
      {
        title: 'Add, then normalise',
        steps: ['s^{(1)} = \\xres + a, \\qquad x^{\\text{mid}} = \\LN_1(s^{(1)})'],
        note: 'Vaswani et al. (2017) call this Add & Norm: a residual sum followed by LayerNorm. Every post-LN block ends with a norm, so from layer 1 on the stream entering each block is normalised; only the embedding is not.',
      },
      {
        title: 'The gradient goes through the norm',
        steps: ['\\frac{\\partial x^{\\text{mid}}}{\\partial \\xres} = J_{\\mathrm{LN}}\\Big(I + \\frac{\\partial a}{\\partial \\xres}\\Big)'],
        note: 'The residual identity is no longer passed on directly: it first goes through the norm Jacobian J_LN, which rescales and projects it (concept:pre-vs-post).',
      },
    ],
    live({ trace, cfg, layer, row, col }) {
      const inp = at2(streamBefore(trace, layer), row, col), att = at2(trace[`L${layer}.attn.out`], row, col);
      const result = cfg.residual ? inp + att : att;
      return {
        tex: cfg.residual
          ? `s^{(1)}_{${row},${col}} = \\xres_{${row},${col}} + a_{${row},${col}} = (${num(inp, 4)}) + (${num(att, 4)}) = ${num(result, 4)}`
          : `s^{(1)}_{${row},${col}} = a_{${row},${col}} = ${num(result, 4)}`,
        result, expected: at2(trace[`L${layer}.add1`], row, col),
        note: cfg.residual ? 'Post-LN: attention reads the raw stream, and its output is added to that stream.' : 'Residuals are off: the sum is the attention output alone.',
      };
    },
  },

  'L{i}.add2': {
    title: 'Sum before post-norm (MLP)',
    tex: 's^{(2)} = x^{\\text{mid}} + \\MLP(x^{\\text{mid}}), \\qquad x^{\\text{out}} = \\LN_2\\big(s^{(2)}\\big) \\qquad \\text{(residuals off: }s^{(2)} = m\\text{)}',
    compact: 's^{(2)}_{t,c} = x^{\\text{mid}}_{t,c} + m_{t,c}',
    symbols: [
      { tex: 's^{(2)}', text: 'sum before the second post-norm (trace L{i}.add2)', dims: ['T', 'd'] },
      { tex: 'x^{\\text{mid}}', text: 'output of the first post-norm, the MLP input (trace L{i}.resid_mid)', dims: ['T', 'd'] },
      { tex: 'm', text: 'MLP output, m = MLP(x^{mid})', dims: ['T', 'd'] },
      { tex: 'x^{\\text{out}}', text: 'output of the second post-norm (trace L{i}.resid_out)', dims: ['T', 'd'] },
    ],
    derivation: [{
      title: 'The second block sum',
      steps: ['s^{(2)} = x^{\\text{mid}} + m, \\qquad x^{\\text{out}} = \\LN_2(s^{(2)})'],
      note: 'The MLP reads x^{mid} directly, with no norm in front of it, and its output is added and normalised. Each block therefore ends with normalised features (before γ and β), which is the defining property of post-LN.',
    }],
    live({ trace, cfg, layer, row, col }) {
      const mid = at2(trace[`L${layer}.resid_mid`], row, col), mlp = at2(trace[`L${layer}.mlp.out`], row, col);
      const result = cfg.residual ? mid + mlp : mlp;
      return {
        tex: cfg.residual
          ? `s^{(2)}_{${row},${col}} = x^{\\text{mid}}_{${row},${col}} + m_{${row},${col}} = (${num(mid, 4)}) + (${num(mlp, 4)}) = ${num(result, 4)}`
          : `s^{(2)}_{${row},${col}} = m_{${row},${col}} = ${num(result, 4)}`,
        result, expected: at2(trace[`L${layer}.add2`], row, col),
        note: cfg.residual ? 'Post-LN: the MLP output is added to the normalised stream from the first sum.' : 'Residuals are off: the sum is the MLP output alone.',
      };
    },
  },

  'concept:layernorm-backward': {
    title: 'LayerNorm backward',
    tex: '\\frac{\\partial L}{\\partial x_{t,c}} = \\frac{1}{s_t}\\Big(g_{t,c} - \\frac{1}{d}\\sum_{k=1}^{d} g_{t,k} - \\hat x_{t,c}\\,\\frac{1}{d}\\sum_{k=1}^{d} g_{t,k}\\,\\hat x_{t,k}\\Big), \\qquad g_{t,c} = \\frac{\\partial L}{\\partial h_{t,c}}\\,\\gamma_c, \\qquad s_t = \\sqrt{\\sigma_t^2+\\varepsilon}',
    compact: '\\frac{\\partial L}{\\partial x_{t,c}} = \\frac{1}{s_t}\\left(g_{t,c} - \\operatorname{mean}(g_t) - \\hat x_{t,c}\\,\\operatorname{mean}(g_t\\hat x_t)\\right)',
    symbols: [
      { tex: 'x_{t,c}', text: 'input to the norm (the stream), one token row', dims: ['T', 'd'] },
      { tex: '\\hat x_{t,c}', text: 'normalised input before the gain', dims: ['T', 'd'] },
      { tex: '\\mu_t', text: 'mean of token t’s d features', dims: ['T'] },
      { tex: '\\gamma_c', text: 'gain, one per feature (enters g)', dims: ['d'] },
      { tex: 'g_{t,c}', text: 'gradient with respect to x̂: upstream gradient times γ', dims: ['T', 'd'] },
      { tex: 's_t', text: '√(σ_t² + ε), the per-token scale', dims: ['T'] },
      { tex: '\\frac{\\partial L}{\\partial h}', text: 'upstream gradient, arriving from the next sub-layer', dims: ['T', 'd'] },
    ],
    derivation: [
      {
        title: 'Derivatives of the statistics',
        steps: [
          '\\frac{\\partial \\mu_t}{\\partial x_{t,c}} = \\frac{1}{d}, \\qquad \\frac{\\partial \\sigma_t^2}{\\partial x_{t,c}} = \\frac{2}{d}\\left(x_{t,c}-\\mu_t\\right)',
          '\\frac{\\partial s_t}{\\partial x_{t,c}} = \\frac{1}{2s_t}\\,\\frac{\\partial \\sigma_t^2}{\\partial x_{t,c}} = \\frac{x_{t,c}-\\mu_t}{d\\,s_t} = \\frac{\\hat x_{t,c}}{d}',
        ],
        note: 'Changing one feature moves the mean by 1/d and the variance by 2(x − μ)/d. The mean’s own contribution to the variance vanishes, because the deviations from the mean sum to zero.',
      },
      {
        title: 'Derivative of x̂ (quotient rule)',
        steps: [
          '\\frac{\\partial \\hat x_{t,k}}{\\partial x_{t,c}} = \\frac{\\left(\\delta_{kc}-\\frac{1}{d}\\right)s_t - \\left(x_{t,k}-\\mu_t\\right)\\frac{\\hat x_{t,c}}{d}}{s_t^{2}}',
          '= \\frac{1}{s_t}\\Big(\\delta_{kc} - \\frac{1}{d} - \\frac{\\hat x_{t,k}\\,\\hat x_{t,c}}{d}\\Big)',
        ],
        note: 'The last step uses x_{t,k} − μ_t = s_t x̂_{t,k}. The −1/d and −x̂x̂/d terms are how a change in one feature leaks into all the others through μ and σ².',
      },
      {
        title: 'Chain rule',
        steps: [
          '\\frac{\\partial L}{\\partial x_{t,c}} = \\sum_{k=1}^{d} g_{t,k}\\,\\frac{\\partial \\hat x_{t,k}}{\\partial x_{t,c}} = \\frac{1}{s_t}\\Big(g_{t,c} - \\frac{1}{d}\\sum_{k=1}^{d} g_{t,k} - \\hat x_{t,c}\\,\\frac{1}{d}\\sum_{k=1}^{d} g_{t,k}\\hat x_{t,k}\\Big)',
        ],
        note: 'Here g_{t,k} = ∂L/∂x̂_{t,k} = γ_k ∂L/∂h_{t,k}. The two sums are computed once per token, so the backward pass costs about as much as the forward pass.',
      },
      {
        title: 'What the backward pass removes',
        steps: ['\\sum_{c=1}^{d}\\frac{\\partial L}{\\partial x_{t,c}} = 0'],
        note: 'Adding the same constant to every feature of a token leaves LayerNorm’s output unchanged, so the gradient must sum to zero over features. The two subtracted terms enforce this. RMSNorm has no mean, so its backward pass drops the first subtraction and keeps the x̂ term.',
      },
      {
        title: 'Gain and shift gradients',
        steps: ['\\frac{\\partial L}{\\partial \\gamma_c} = \\sum_{t}\\frac{\\partial L}{\\partial h_{t,c}}\\,\\hat x_{t,c}, \\qquad \\frac{\\partial L}{\\partial \\beta_c} = \\sum_{t}\\frac{\\partial L}{\\partial h_{t,c}}'],
        note: 'The sums run over the tokens of the batch. These are the gradients the engine accumulates for the gain and the shift.',
      },
    ],
  },

  'concept:pre-vs-post': {
    title: 'Pre-LN vs post-LN blocks',
    tex: '\\begin{aligned} &\\text{pre-LN:}\\quad x^{\\text{mid}} = \\xres + \\Attn(\\LN_1(\\xres)), \\qquad x^{\\text{out}} = x^{\\text{mid}} + \\MLP(\\LN_2(x^{\\text{mid}})) \\\\ &\\text{post-LN:}\\quad x^{\\text{mid}} = \\LN_1(\\xres + \\Attn(\\xres)), \\qquad x^{\\text{out}} = \\LN_2(x^{\\text{mid}} + \\MLP(x^{\\text{mid}})) \\end{aligned}',
    compact: '\\text{pre: } x + \\mathcal{F}(\\LN(x)) \\quad\\text{post: } \\LN(x + \\mathcal{F}(x))',
    symbols: [
      { tex: '\\xres', text: 'block input, the residual stream', dims: ['T', 'd'] },
      { tex: 'x^{\\text{mid}}', text: 'stream after the attention sub-layer', dims: ['T', 'd'] },
      { tex: 'x^{\\text{out}}', text: 'stream after the MLP sub-layer, the next block’s input', dims: ['T', 'd'] },
      { tex: '\\LN_1, \\LN_2', text: 'the block’s two norms, each with its own γ and β', dims: [] },
      { tex: '\\mathcal{F}', text: 'generic sub-layer, either Attn or MLP (not the dimension F)', dims: [] },
      { tex: 'J_k', text: 'Jacobian of block k’s residual increment, ∂(x_{k+1} − x_k)/∂x_k', dims: ['d', 'd'] },
      { tex: 'J_{\\mathrm{LN}}', text: 'Jacobian of a norm, diag(γ)(I − 1·1ᵀ/d − x̂ x̂ᵀ/d)/√(σ²+ε)', dims: ['d', 'd'] },
      { tex: 'I', text: 'identity matrix', dims: ['d', 'd'] },
    ],
    derivation: [
      {
        title: 'Where the norm sits',
        steps: [
          '\\text{pre-LN:}\\quad \\xres \\mapsto \\xres + \\mathcal{F}\\big(\\LN(\\xres)\\big)',
          '\\text{post-LN:}\\quad \\xres \\mapsto \\LN\\big(\\xres + \\mathcal{F}(\\xres)\\big)',
        ],
        note: 'Pre-LN normalises only what a sub-layer reads, so the stream is never normalised in place. Post-LN normalises the sum, so every block output is normalised. The original Transformer (Vaswani et al. 2017) is post-LN. GPT-2 (Radford et al. 2019) and LLaMA (Touvron et al. 2023) are pre-LN.',
      },
      {
        title: 'Pre-LN keeps an identity path',
        steps: [
          '\\frac{\\partial \\xres_{\\ell+1}}{\\partial \\xres_\\ell} = I + J_\\ell',
          '\\frac{\\partial \\xres_{L}}{\\partial \\xres_{\\ell}} = \\prod_{k=\\ell}^{L-1}\\big(I + J_k\\big) = I + \\sum_{k=\\ell}^{L-1} J_k + \\cdots',
        ],
        note: 'J_k = ∂(x_{k+1} − x_k)/∂x_k is the Jacobian of block k’s residual increment. The identity term is always present, so the gradient reaches layer ℓ even when the J_k are small. Consistent with Xiong et al. (2020), who show with mean-field analysis that Pre-LN gradients are well-behaved at initialisation, so warm-up can be dropped. The identity path is an intuitive picture, not their argument.',
      },
      {
        title: 'Post-LN has no bare identity',
        steps: [
          '\\frac{\\partial x^{\\text{out}}}{\\partial \\xres} = J_{\\mathrm{LN}_2}\\,\\big(I+J_{\\MLP}\\big)\\,J_{\\mathrm{LN}_1}\\,\\big(I+J_{\\Attn}\\big)',
          'J_{\\mathrm{LN}} = \\frac{\\operatorname{diag}(\\gamma)}{\\sqrt{\\sigma_t^2+\\varepsilon}}\\Big(I - \\frac{\\mathbf{1}\\mathbf{1}^{\\top}}{d} - \\frac{\\hat x\\,\\hat x^{\\top}}{d}\\Big)',
        ],
        note: 'Each norm rescales the identity per feature by γ/√(σ_t²+ε) and projects out the mean and the normalised direction, so no bare identity survives a block. Xiong et al. (2020) show that the gradients near the output are large at initialisation in Post-LN, which is why it needs warm-up.',
      },
      {
        title: 'The final norm',
        steps: ['\\text{logits} = \\LN_f(\\xres_L)\\,\\WU^{\\top}'],
        note: 'The pre-LN stream grows with depth and is never normalised in place, so the model needs a final LN_f before the unembedding. A post-LN block already ends with a norm, so it needs none.',
      },
    ],
  },

  'concept:residual-stream': {
    title: 'The residual stream',
    tex: '\\begin{aligned} &\\text{pre-LN, residuals on:}\\quad \\xres_L = \\xres_0 + \\sum_{\\ell=0}^{L-1}\\left(a_\\ell + m_\\ell\\right) \\\\ &a_\\ell = \\Attn_\\ell\\big(\\LN_1(\\xres_\\ell)\\big), \\qquad m_\\ell = \\MLP_\\ell\\big(\\LN_2(\\xres_\\ell + a_\\ell)\\big) \\end{aligned}',
    compact: '\\xres_L = \\xres_0 + \\sum_{\\ell}\\left(a_\\ell + m_\\ell\\right) \\;(\\text{pre-LN, residuals on})',
    symbols: [
      { tex: '\\xres_0', text: 'embedding output (trace emb.out): token plus position for learned or sinusoidal, token only for rope, alibi and none', dims: ['T', 'd'] },
      { tex: '\\xres_\\ell', text: 'stream before block ℓ', dims: ['T', 'd'] },
      { tex: 'a_\\ell', text: 'attention output of block ℓ (trace L{ℓ}.attn.out)', dims: ['T', 'd'] },
      { tex: 'm_\\ell', text: 'MLP output of block ℓ (trace L{ℓ}.mlp.out)', dims: ['T', 'd'] },
      { tex: 'L', text: 'number of blocks (n_layer)', dims: [] },
      { tex: '\\LN_1, \\LN_2', text: 'the block’s two norms, applied to what each sub-layer reads', dims: [] },
    ],
    derivation: [
      {
        title: 'Telescoping sum',
        steps: [
          '\\xres_{\\ell+1}-\\xres_\\ell = a_\\ell + m_\\ell',
          '\\xres_L - \\xres_0 = \\sum_{\\ell=0}^{L-1}\\big(\\xres_{\\ell+1}-\\xres_\\ell\\big) = \\sum_{\\ell=0}^{L-1}\\big(a_\\ell + m_\\ell\\big)',
        ],
        note: 'Each sub-layer reads the stream through a norm and writes its output back by addition. Nothing else changes the stream, so the final stream is the embedding plus one term per sub-layer. This holds for pre-LN with residuals on.',
      },
      {
        title: 'A shared channel',
        steps: ['\\text{logits} = \\LN_f(\\xres_L)\\,\\WU^{\\top}'],
        note: 'Elhage et al. (2021) call this the residual stream: each attention head and MLP reads from it and writes to it. Between the sum and the logits there is only the final norm and a linear map. That norm centres each token and divides it by the token’s own σ_t, so each contribution is read off up to this shared per-token scale. Centring, gain and the linear map are linear, so the contributions still add up; β adds a constant offset.',
      },
      {
        title: 'Ablation: residuals off',
        steps: ['\\xres_L = m_{L-1} \\qquad \\text{(pre-LN, residuals off)}'],
        note: 'Each sub-layer then replaces the stream instead of adding to it, so only the last block’s MLP output reaches the end.',
      },
      {
        title: 'Growth with depth',
        steps: ['\\operatorname{Var}(\\xres_L) \\approx \\operatorname{Var}(\\xres_0) + \\sum_{\\ell=0}^{L-1}\\big(\\operatorname{Var}(a_\\ell) + \\operatorname{Var}(m_\\ell)\\big)'],
        note: 'If the terms are roughly uncorrelated, the stream’s variance grows about linearly with depth. The pre-LN norms keep each sub-layer’s input at a set scale regardless, but the stream itself is never rescaled, so the final LN_f is needed.',
      },
    ],
  },
};

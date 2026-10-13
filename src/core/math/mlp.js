// Maths for the MLP block: the up, gate, activation and down projections (tensor entries),
// plus concept entries for GELU, SwiGLU, parameter counts and the key-value reading.
// Entry schema (see src/core/math/index.js and docs/MATH_GUIDE.md):
//   title, tex (display), compact (one line, for tooltips), symbols [{tex, text, dims}],
//   derivation [{title, steps: [tex], note}], live(ctx) -> {tex, result, expected, note}
// Indices: t runs over tokens, f over the F hidden units, o over the d outputs, c over the
// components of a d-vector. Symbol text and notes are plain text (the card does not typeset them).
import { at2, row2, wrow, dot, dotTerms, num, mlpInput } from './util.js';

// Same operations, in the same order, as src/engine/tensor.js (GELU_C = sqrt(2/pi)).
const GELU_C = Math.sqrt(2 / Math.PI);
const geluTanh = (h) => 0.5 * h * (1 + Math.tanh(GELU_C * (h + 0.044715 * h * h * h)));
const siluOf = (g) => g / (1 + Math.exp(-g));
const hasBias = (model, name) => Boolean(model.params[`${name}.bias`]);

// One output unit of an nn.Linear: y_j = sum_c x_c W[j,c] + b_j, with the terms written out.
// name is the module path (e.g. 'blocks.0.mlp.up_proj'); the bias is absent when cfg.bias is off.
function linearUnit(model, name, x, j) {
  const w = wrow(model, `${name}.weight`, j);
  const withBias = hasBias(model, name);
  const b = withBias ? model.params[`${name}.bias`].data[j] : 0;
  const result = dot(x, w) + b;
  return { tex: `${dotTerms(x, w)}${withBias ? ` + (${num(b)})` : ''} = ${num(result)}`, result };
}

export const ENTRIES = {
  'L{i}.mlp.up': {
    title: 'MLP up-projection',
    tex: 'h = x\\,\\Wup^{\\top} + b_{\\text{up}}, \\qquad h_{t,f} = \\sum_{c=1}^{d} x_{t,c}\\,(\\Wup)_{f,c} + (b_{\\text{up}})_{f}',
    compact: 'h_{t,f} = x_t \\cdot (\\Wup)_f + b_f',
    symbols: [
      { tex: 'x', text: 'MLP input: for pre-LN, the output of norm 2 (LayerNorm or RMSNorm); for post-LN, the normalised residual after attention', dims: ['T', 'd'] },
      { tex: '\\Wup', text: 'up-projection weight, one row per hidden unit', dims: ['F', 'd'] },
      { tex: 'b_{\\text{up}}', text: 'up-projection bias (absent when cfg.bias is off)', dims: ['F'] },
      { tex: 'h_{t,f}', text: 'pre-activation of hidden unit f at token t (for SwiGLU, the value branch)', dims: ['T', 'F'] },
    ],
    derivation: [
      {
        title: 'Every token, same weights',
        steps: ['H = X\\,\\Wup^{\\top} + \\mathbf{1}\\,b_{\\text{up}}^{\\top}, \\qquad X \\in \\R^{T\\times d},\\ H \\in \\R^{T\\times F}'],
        note: 'The MLP is position-wise: every token passes through the same weights independently. Tokens only interact in attention.',
      },
      {
        title: 'Each unit is a detector',
        steps: ['h_{t,f} = w_f^{\\top} x_t + b_f, \\qquad w_f = (\\Wup)_{f,:} \\in \\R^{d}'],
        note: 'Each row of the up-projection weight is a direction in the d-dimensional stream, and the pre-activation of unit f at token t measures how strongly token t points along it. With F about 4d there are many more such detectors than the d coordinates of the stream.',
      },
    ],
    live({ trace, cfg, model, layer, row, col }) {
      const x = row2(mlpInput(trace, cfg, layer), row);
      const { tex, result } = linearUnit(model, `blocks.${layer}.mlp.up_proj`, x, col);
      return {
        tex: `h_{${row},${col}} = ${tex}`,
        result, expected: at2(trace[`L${layer}.mlp.up`], row, col),
        note: `Sum over d = ${x.length} inputs of token ${row}.`,
      };
    },
  },

  'L{i}.mlp.gate': {
    title: 'MLP gate projection (SwiGLU)',
    tex: 'g = x\\,\\Wgate^{\\top} + b_{\\text{gate}}, \\qquad g_{t,f} = \\sum_{c=1}^{d} x_{t,c}\\,(\\Wgate)_{f,c} + (b_{\\text{gate}})_{f}',
    compact: 'g_{t,f} = x_t \\cdot (\\Wgate)_f + b_f',
    symbols: [
      { tex: 'x', text: 'MLP input (the same tensor the up-projection reads)', dims: ['T', 'd'] },
      { tex: '\\Wgate', text: 'gate-projection weight; exists only when activation = swiglu', dims: ['F', 'd'] },
      { tex: 'g_{t,f}', text: 'gate pre-activation of hidden unit f at token t', dims: ['T', 'F'] },
    ],
    derivation: [
      {
        title: 'A second learned view of the same input',
        steps: ['a_{t,f} = \\SiLU(g_{t,f})\\,h_{t,f}'],
        note: 'The value h and the gate g come from the same x through two matrices, so each unit learns both what to pass (h) and how much of it to let through (SiLU of g). The price is one extra d×F matrix, so SwiGLU uses a narrower hidden width, F = 8d/3, to keep the parameter count equal (concept:swiglu).',
      },
    ],
    live({ trace, cfg, model, layer, row, col }) {
      const x = row2(mlpInput(trace, cfg, layer), row);
      const { tex, result } = linearUnit(model, `blocks.${layer}.mlp.gate_proj`, x, col);
      return {
        tex: `g_{${row},${col}} = ${tex}`,
        result, expected: at2(trace[`L${layer}.mlp.gate`], row, col),
        note: `Sum over d = ${x.length} inputs of token ${row}.`,
      };
    },
  },

  'L{i}.mlp.act': {
    title: 'MLP activation',
    tex: 'a_{t,f} = \\begin{cases} \\GELU(h_{t,f}) & \\text{gelu} \\\\ \\ReLU(h_{t,f}) = \\max(0,\\,h_{t,f}) & \\text{relu} \\\\ \\SiLU(g_{t,f})\\,h_{t,f} = g_{t,f}\\,\\sigma(g_{t,f})\\,h_{t,f} & \\text{swiglu} \\end{cases}',
    compact: 'a_{t,f} = \\GELU(h_{t,f}) \\text{ or } \\ReLU(h_{t,f}) \\text{ or } \\SiLU(g_{t,f})\\,h_{t,f}',
    symbols: [
      { tex: 'h_{t,f}', text: 'up-projection output (the pre-activation for gelu and relu; the value branch for swiglu)', dims: ['T', 'F'] },
      { tex: 'g_{t,f}', text: 'gate output (swiglu only)', dims: ['T', 'F'] },
      { tex: 'a_{t,f}', text: 'activation of hidden unit f at token t; this is what the down-projection reads', dims: ['T', 'F'] },
    ],
    derivation: [
      {
        title: 'Why a non-linearity at all',
        steps: ['\\big(x\\,\\Wup^{\\top}+b_{\\text{up}}\\big)\\,\\Wdown^{\\top}+b_{\\text{down}} = x\\,(\\Wdown\\,\\Wup)^{\\top} + \\big(b_{\\text{up}}\\,\\Wdown^{\\top}+b_{\\text{down}}\\big)'],
        note: 'Without the activation the two projections collapse into one affine map, so the MLP would add no expressive power. The activation bends each unit’s response, which lets stacked layers represent curved functions.',
      },
      {
        title: 'GELU is a smooth ReLU',
        steps: ['\\GELU(h) = h\\,\\Phi(h) \\approx \\tfrac12\\,h\\Big(1+\\tanh\\Big(\\sqrt{\\tfrac{2}{\\pi}}\\,(h+0.044715\\,h^{3})\\Big)\\Big)'],
        note: 'GELU scales h by the probability that a standard normal falls below h: close to zero for very negative h, close to the identity for large positive h, and smooth in between. The model evaluates the tanh form in the second step (concept:gelu).',
      },
    ],
    live({ trace, cfg, layer, row, col }) {
      const h = at2(trace[`L${layer}.mlp.up`], row, col);
      const expected = at2(trace[`L${layer}.mlp.act`], row, col);
      if (cfg.activation === 'swiglu') {
        const g = at2(trace[`L${layer}.mlp.gate`], row, col);
        const s = siluOf(g);
        const result = s * h;
        return {
          tex: `a_{${row},${col}} = \\SiLU(${num(g)})\\,(${num(h)}) = \\frac{${num(g)}}{1+e^{${num(-g)}}}\\,(${num(h)}) = ${num(s)}\\cdot (${num(h)}) = ${num(result)}`,
          result, expected,
          note: 'g and h are the traced gate and up values at this cell.',
        };
      }
      if (cfg.activation === 'gelu') {
        const u = GELU_C * (h + 0.044715 * h * h * h);
        const result = geluTanh(h);
        return {
          tex: `\\GELU(${num(h)}) = \\tfrac12\\,(${num(h)})\\Big(1+\\tanh\\Big(\\sqrt{\\tfrac{2}{\\pi}}\\big[(${num(h)})+0.044715\\,(${num(h)})^{3}\\big]\\Big)\\Big) = \\tfrac12\\,(${num(h)})\\big(1+\\tanh(${num(u)})\\big) = ${num(result)}`,
          result, expected,
          note: 'h is the traced up-projection value at this cell; tanh of the bracketed argument.',
        };
      }
      const result = Math.max(0, h);
      return {
        tex: `\\ReLU(${num(h)}) = \\max(0,\\,${num(h)}) = ${num(result)}`,
        result, expected,
        note: 'h is the traced up-projection value at this cell.',
      };
    },
  },

  'L{i}.mlp.out': {
    title: 'MLP down-projection',
    tex: 'y = a\\,\\Wdown^{\\top} + b_{\\text{down}}, \\qquad y_{t,o} = \\sum_{f=1}^{F} a_{t,f}\\,(\\Wdown)_{o,f} + (b_{\\text{down}})_{o}',
    compact: 'y_{t,o} = a_t \\cdot (\\Wdown)_o + b_o',
    symbols: [
      { tex: 'a', text: 'hidden activations: the input of the down-projection', dims: ['T', 'F'] },
      { tex: '\\Wdown', text: 'down-projection weight, one column per hidden unit', dims: ['d', 'F'] },
      { tex: 'b_{\\text{down}}', text: 'down-projection bias (absent when cfg.bias is off)', dims: ['d'] },
      { tex: 'y_{t,o}', text: 'MLP output for token t, written into the residual stream', dims: ['T', 'd'] },
    ],
    derivation: [
      {
        title: 'Each unit writes its column',
        steps: ['y_t = \\sum_{f=1}^{F} a_{t,f}\\,(\\Wdown)_{:,f} + b_{\\text{down}}'],
        note: 'The output is a weighted sum of F learned vectors of width d: unit f adds its column of the down-projection weight, scaled by its activation. Under the key-value reading (concept:key-value-memory) the columns are the values.',
      },
    ],
    live({ trace, model, layer, row, col }) {
      const a = row2(trace[`L${layer}.mlp.act`], row);
      const { tex, result } = linearUnit(model, `blocks.${layer}.mlp.down_proj`, a, col);
      return {
        tex: `y_{${row},${col}} = ${tex}`,
        result, expected: at2(trace[`L${layer}.mlp.out`], row, col),
        note: `Sum over F = ${a.length} hidden units of token ${row}.`,
      };
    },
  },

  'concept:gelu': {
    title: 'GELU: exact and tanh forms',
    tex: '\\GELU(x) = x\\,\\Phi(x) = \\frac{x}{2}\\Big(1+\\operatorname{erf}\\Big(\\frac{x}{\\sqrt{2}}\\Big)\\Big) \\approx \\frac{x}{2}\\Big(1+\\tanh\\Big(\\sqrt{\\tfrac{2}{\\pi}}\\,\\big(x+0.044715\\,x^{3}\\big)\\Big)\\Big)',
    compact: '\\GELU(x) = x\\,\\Phi(x)',
    symbols: [
      { tex: 'x', text: 'pre-activation of one hidden unit (a scalar, such as h[t,f])', dims: [] },
      { tex: '\\Phi(x)', text: 'standard normal CDF: Pr[Z ≤ x] for Z ~ N(0,1)', dims: [] },
      { tex: '\\varphi(x)', text: 'standard normal density: exp(-x²/2) / √(2π)', dims: [] },
    ],
    derivation: [
      {
        title: 'Keep x with the probability Φ(x)',
        steps: ['x\\,\\Phi(x) = x\\cdot \\Pr[Z \\le x], \\qquad Z \\sim \\mathcal{N}(0,1)'],
        note: 'GELU scales its input by how likely a standard normal is to lie below it: a soft version of the ReLU threshold at 0 (Hendrycks and Gimpel, 2016).',
      },
      {
        title: 'Derivative of exact GELU',
        steps: ['\\frac{d}{dx}\\GELU(x) = \\Phi(x) + x\\,\\varphi(x), \\qquad \\varphi(x) = \\frac{e^{-x^{2}/2}}{\\sqrt{2\\pi}}'],
        note: 'The slope is 1/2 at zero, tends to 1 for large positive x and to 0 for large negative x. It is slightly negative for x below about -0.75, where GELU dips to its minimum of about -0.17. Backpropagation multiplies the upstream gradient by this slope.',
      },
      {
        title: 'Tanh approximation',
        steps: ['\\Phi(x) \\approx \\tfrac12\\Big(1+\\tanh\\Big(\\sqrt{\\tfrac{2}{\\pi}}\\,(x+0.044715\\,x^{3})\\Big)\\Big)'],
        note: 'The tanh form avoids erf and is the approximation used by GPT-2 and by the reference model. Its largest error against the exact GELU is about 4.7e-4 (on [-8, 8], near x = -2.7).',
      },
      {
        title: 'Derivative of the tanh form used by the code',
        steps: [String.raw`c=\sqrt{2/\pi},\quad a=0.044715,\quad u=c(x+ax^3),\quad t=\tanh(u)`, String.raw`\frac{dy}{dx}=\tfrac12(1+t)+\tfrac12x(1-t^2)c(1+3ax^2)`],
        note: 'Differentiate the implemented approximation itself. The Φ(x)+xφ(x) expression above is the derivative of exact GELU; it is not the exact derivative of this tanh computation. Backpropagation multiplies the upstream gradient by the displayed slope.',
        reference: { file: 'numpy_transformer.py', region: 'gelu_backward' },
      },
    ],
  },

  'concept:swiglu': {
    title: 'SwiGLU: gated linear unit with SiLU',
    tex: '\\MLP_{\\text{SwiGLU}}(x) = \\big(\\SiLU(x\\,\\Wgate^{\\top})\\odot x\\,\\Wup^{\\top}\\big)\\,\\Wdown^{\\top}, \\qquad \\SiLU(z) = z\\,\\sigma(z) = \\frac{z}{1+e^{-z}}, \\qquad \\text{(bias terms omitted)}',
    compact: '\\MLP(x) = \\big(\\SiLU(g)\\odot h\\big)\\,\\Wdown^{\\top}',
    symbols: [
      { tex: 'h', text: 'value branch: x times the up-projection weight, transposed (plus the up bias when cfg.bias is on)', dims: ['T', 'F'] },
      { tex: 'g', text: 'gate branch: x times the gate-projection weight, transposed (plus the gate bias when cfg.bias is on)', dims: ['T', 'F'] },
      { tex: 'F', text: 'hidden width d_ff; 8d/3 at matched parameters (in practice rounded, as in LLaMA)', dims: [] },
      { tex: '\\sigma(z)', text: 'logistic sigmoid 1/(1+exp(-z))', dims: [] },
    ],
    derivation: [
      {
        title: 'From GLU to SwiGLU',
        steps: [
          '\\operatorname{GLU}(x) = (xW+b)\\odot\\sigma(xV+c)',
          '\\operatorname{SwiGLU}(x) = \\big(\\SiLU(x\\,\\Wgate^{\\top})\\odot x\\,\\Wup^{\\top}\\big)\\,\\Wdown^{\\top}',
        ],
        note: 'Dauphin et al. (2017) gated one linear branch by a sigmoid of another. Shazeer (2020) tried several gates in Transformer FFNs; the SiLU (Swish) gate, SwiGLU, was among the best and beat the ReLU and GELU baselines. SiLU is smooth, unbounded above, and dips to about -0.278 near z = -1.28.',
      },
      {
        title: 'Matching the parameter budget',
        steps: [
          '\\text{classic: } 2\\cdot d\\cdot 4d = 8d^{2}',
          '\\text{SwiGLU: } 3\\cdot d\\cdot F = 8d^{2} \\;\\Rightarrow\\; F = \\tfrac{8d}{3}',
        ],
        note: 'SwiGLU has three d×F matrices instead of two, so its hidden width is 2/3 of the classic 4d: F = 8d/3. The weight count, and with it the matmul FLOPs per token, stays equal to the classic 4d MLP. LLaMA-7B uses d = 4096 and F = 11008, which is 8d/3 ≈ 10,923 rounded up to the next multiple of 256.',
      },
    ],
  },

  'concept:mlp-params': {
    title: 'MLP parameter count and share of a block',
    tex: 'P_{\\text{GELU, ReLU}}\\,(\\text{biases on}) = \\underbrace{dF+F}_{\\text{up}} + \\underbrace{Fd+d}_{\\text{down}} = 2dF+F+d, \\qquad P_{\\text{SwiGLU}}\\,(\\text{biases on}) = 3dF+2F+d',
    compact: 'P_{\\text{MLP}} = 2dF + F + d \\;(\\text{GELU, ReLU; biases on})',
    symbols: [
      { tex: 'P_{\\text{MLP}}', text: 'parameters of one MLP, weights and biases', dims: [] },
      { tex: 'P_{\\text{attn}}', text: 'parameters of one attention sub-layer', dims: [] },
      { tex: '\\rho', text: 'MLP share of the block: MLP parameters / (MLP parameters + attention parameters)', dims: [] },
    ],
    derivation: [
      {
        title: 'Count the matrices',
        steps: [
          '\\underbrace{dF}_{\\Wup}+\\underbrace{F}_{b_{\\text{up}}}+\\underbrace{Fd}_{\\Wdown}+\\underbrace{d}_{b_{\\text{down}}}=2dF+F+d',
          '\\underbrace{dF}_{\\Wgate}+\\underbrace{F}_{b_{\\text{gate}}}+\\underbrace{dF}_{\\Wup}+\\underbrace{F}_{b_{\\text{up}}}+\\underbrace{Fd}_{\\Wdown}+\\underbrace{d}_{b_{\\text{down}}}=3dF+2F+d \\quad(\\text{SwiGLU})',
        ],
        note: 'With bias = false the bias terms vanish and the counts are 2dF, or 3dF for SwiGLU with its extra gate matrix. Norm parameters are small by comparison (2d per LayerNorm with a bias).',
      },
      {
        title: 'Share of a block when F = 4d',
        steps: [
          'P_{\\text{MLP}} = 2d(4d) + 4d + d = 8d^{2}+5d',
          'P_{\\text{attn}} = 4d^{2}+4d \\quad (\\text{MHA, with biases})',
          '\\rho = \\frac{8d^{2}+5d}{12d^{2}+9d} \\approx \\frac{8}{12} = \\frac{2}{3}',
        ],
        note: 'About two thirds of each block sits in the MLP and a third in attention. Grouped-query attention shrinks the K and V projections, so the MLP share rises. GPT-2 small (d = 768, F = 3072, 12 layers) has 56.6M MLP weights out of 124M parameters.',
      },
    ],
  },

  'concept:key-value-memory': {
    title: 'Feed-forward layers as key-value memories',
    tex: '\\MLP(x) = \\sum_{f=1}^{F} m_f(x)\\,v_f + b_{\\text{down}}, \\qquad v_f = (\\Wdown)_{:,f}, \\qquad m_f(x) = \\begin{cases} \\GELU(k_f\\cdot x + b_f) & \\text{GELU} \\\\ \\ReLU(k_f\\cdot x + b_f) & \\text{ReLU} \\\\ \\SiLU(k^{g}_f\\cdot x + b^{g}_f)\\,(k_f\\cdot x + b_f) & \\text{SwiGLU} \\end{cases}',
    compact: '\\MLP(x) = \\sum_f m_f(x)\\,v_f + b_{\\text{down}}',
    symbols: [
      { tex: 'k_f', text: 'key of hidden unit f: row f of the up-projection weight (for SwiGLU, the up key)', dims: ['d'] },
      { tex: 'b_f', text: 'bias of hidden unit f: element f of the up-projection bias (for SwiGLU, the up bias); absent when cfg.bias is off', dims: [] },
      { tex: 'k^{g}_f', text: 'SwiGLU gate key: row f of the gate-projection weight', dims: ['d'] },
      { tex: 'b^{g}_f', text: 'SwiGLU gate bias: element f of the gate-projection bias', dims: [] },
      { tex: 'v_f', text: 'value of hidden unit f: column f of the down-projection weight', dims: ['d'] },
      { tex: 'm_f(x)', text: 'coefficient of unit f for input x; when x is the vector of token t, it is the activation of unit f at token t', dims: [] },
      { tex: '\\phi', text: 'the one-key activation, GELU or ReLU', dims: [] },
    ],
    derivation: [
      {
        title: 'Keys match, values are written',
        steps: [
          '\\MLP(x) = \\phi\\big(x\\,\\Wup^{\\top}+b_{\\text{up}}\\big)\\,\\Wdown^{\\top}+b_{\\text{down}}, \\qquad \\phi \\in \\{\\GELU, \\ReLU\\}',
          '= \\sum_{f=1}^{F} \\phi(k_f\\cdot x+b_f)\\,v_f + b_{\\text{down}}',
          '\\MLP(x) = \\sum_{f=1}^{F} \\SiLU(k^{g}_f\\cdot x+b^{g}_f)\\,(k_f\\cdot x+b_f)\\,v_f + b_{\\text{down}} \\qquad (\\text{SwiGLU})',
          'y_t = \\sum_{f=1}^{F} a_{t,f}\\,(\\Wdown)_{:,f} + b_{\\text{down}}, \\qquad a_{t,f} = m_f(x_t)',
        ],
        note: 'Read the up-projection weight row by row as F keys, and the down-projection weight column by column as F values. A unit fires strongly when its key matches the token, and then adds its value to the residual stream. In SwiGLU each unit has a gate key and an up key, and its coefficient is the product of the two matches. The last line holds for every activation.',
      },
      {
        title: 'What the values do to the logits',
        steps: ['\\text{logit contribution of unit } f \\;\\approx\\; m_f(x)\\,\\WU\\,v_f \\in \\R^{V}'],
        note: 'Ignoring the final norm, projecting a value through the unembedding matrix shows which vocabulary entries it promotes. Geva et al. (2021) found that keys often match human-readable textual patterns and that values tend to promote related output tokens.',
      },
    ],
  },
};

// Maths for positional information: the ALiBi distance bias, RoPE on queries and keys,
// the score with the ALiBi bias added, and the concepts behind sinusoidal encodings,
// RoPE and ALiBi. Entry schema: docs/MATH_GUIDE.md. Style: src/core/math/attention.js.
import { at3, row3, kvHead, dh, dot } from './util.js';

// ALiBi slope of head h, as in reference/model.py alibi_bias: m_h = 2^(-8(h+1)/H).
const slope = (cfg, h) => Math.pow(2, -8 * (h + 1) / cfg.n_head);
// RoPE frequency of pair p, as in reference/model.py rope_angles: theta_p = base^(-2p/d_h).
const thetaOf = (cfg, p) => Math.pow(cfg.rope_base, -(2 * p) / dh(cfg));

// Display rounding. Each printed number is rounded here (half away from zero), and each line of an
// equation is computed from the rounded values printed above it, so a hand check reproduces it.
const rnd = (v, dp) => { const f = 10 ** dp; return Math.sign(v) * Math.round(Math.abs(v) * f + 1e-9) / f; };
// Fixed decimals for LaTeX, with no "-0.0000".
const fix = (v, dp) => { const r = rnd(v, dp); return r === 0 ? (0).toFixed(dp) : r.toFixed(dp); };
// Trailing zeros dropped after the decimal point: "0.010000" -> "0.01".
const trimZeros = (s) => (s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s);
// v to n significant digits, in fixed notation (RoPE frequencies can be small).
const sig = (v, n) => (v === 0 ? '0' : trimZeros(fix(v, Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(v)))))));
const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
const ROUND_NOTE = 'Numbers are rounded as printed, and each line uses the rounded values above it, so the printed result can differ from the exact value in its last digit.';

// m_h = 2^(-8(h+1)/H): the exponent as written, then as an exact fraction, then m_h to 8 decimals.
function slopeTex(cfg, h) {
  const H = cfg.n_head, n = -8 * (h + 1), g = gcd(Math.abs(n), H);
  const top = n / g, bot = H / g;
  const exact = bot === 1 ? `${top}` : `${top < 0 ? '-' : ''}\\frac{${Math.abs(top)}}{${bot}}`;
  return `m_{${h}} &= 2^{-\\frac{8\\cdot ${h + 1}}{${H}}} = 2^{${exact}} = ${fix(slope(cfg, h), 8)}`;
}

// (a_c)(b_c) for every column, at 6 decimals. Long sums keep the first three terms and the last.
function productsTex(a, b) {
  const term = (c) => `(${fix(a[c], 6)})(${fix(b[c], 6)})`;
  const n = a.length;
  if (n <= 4) return Array.from({ length: n }, (_, c) => term(c)).join(' + ');
  return `${[0, 1, 2].map(term).join(' + ')} + \\cdots + ${term(n - 1)}`;
}

// live() body shared by q_rot and k_rot: one rotated cell, with the numbers plugged in.
// The result is the exact traced value; the printed chain uses the rounded numbers.
function ropeLive(cfg, x, t, c, { name, src, expected }) {
  const p = Math.floor(c / 2), even = c % 2 === 0, i0 = 2 * p, dhv = dh(cfg);
  const theta = thetaOf(cfg, p), ang = t * theta, cs = Math.cos(ang), sn = Math.sin(ang);
  const x0 = x[i0], x1 = x[i0 + 1];
  const y0 = x0 * cs - x1 * sn, y1 = x0 * sn + x1 * cs;
  const result = even ? y0 : y1;
  // Printed chain: theta to 8 significant digits, the angle, cos, sin and the factors to 6 decimals
  // (guard digits), and the result to 4 decimals, the precision of the cell's value.
  const thetaTex = sig(theta, 8), angTex = trimZeros(fix(t * Number(thetaTex), 6));
  const angS = Number(angTex), csS = rnd(Math.cos(angS), 6), snS = rnd(Math.sin(angS), 6);
  const x0S = rnd(x0, 6), x1S = rnd(x1, 6);
  const resS = rnd(even ? x0S * csS - x1S * snS : x0S * snS + x1S * csS, 4);
  const lines = [
    p === 0 ? `\\theta_{0} &= ${cfg.rope_base}^{0} = ${thetaTex}`
      : `\\theta_{${p}} &= ${cfg.rope_base}^{-${i0}/${dhv}} = ${thetaTex}`,
    `t\\,\\theta_{${p}} &= ${t}\\cdot ${thetaTex} = ${angTex}`,
    `\\cos(t\\theta_{${p}}) &= \\cos(${angTex}) = ${fix(csS, 6)}`,
    `\\sin(t\\theta_{${p}}) &= \\sin(${angTex}) = ${fix(snS, 6)}`,
    even
      ? `${name}_{${t},${c}} &= ${src}_{${t},${i0}}\\cos(t\\theta_{${p}}) - ${src}_{${t},${i0 + 1}}\\sin(t\\theta_{${p}})`
      : `${name}_{${t},${c}} &= ${src}_{${t},${i0}}\\sin(t\\theta_{${p}}) + ${src}_{${t},${i0 + 1}}\\cos(t\\theta_{${p}})`,
    even
      ? `&= (${fix(x0S, 6)})(${fix(csS, 6)}) - (${fix(x1S, 6)})(${fix(snS, 6)}) = ${fix(resS, 4)}`
      : `&= (${fix(x0S, 6)})(${fix(snS, 6)}) + (${fix(x1S, 6)})(${fix(csS, 6)}) = ${fix(resS, 4)}`,
  ];
  const len = (a, b) => Math.hypot(a, b).toFixed(4);
  return {
    tex: `\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`,
    result, expected,
    note: `Pair (${i0}, ${i0 + 1}) keeps its length: ${len(x0, x1)} before, ${len(y0, y1)} after. ${ROUND_NOTE}`,
  };
}

export const ENTRIES = {
  'alibi': {
    title: 'ALiBi distance bias',
    tex: 'b_{h,ij} = -m_h\\,\\lvert i-j\\rvert, \\qquad m_h = 2^{-8(h+1)/H}',
    compact: 'b_{h,ij} = -m_h\\,\\lvert i-j\\rvert',
    symbols: [
      { tex: 'b_{h,ij}', text: 'additive bias on the score of query i for key j in head h', dims: ['H', 'T', 'T'] },
      { tex: 'm_h', text: 'slope of head h, positive and shrinking as h grows', dims: ['H'] },
      { tex: 'h', text: 'head index, 0 ≤ h < H', dims: ['H'] },
      { tex: 'i', text: 'query position (the row of the score matrix)', dims: ['T'] },
      { tex: 'j', text: 'key position (the column of the score matrix)', dims: ['T'] },
      { tex: '\\lvert i-j\\rvert', text: 'distance between query i and key j (0 on the diagonal)', dims: ['T', 'T'] },
    ],
    derivation: [
      {
        title: 'A distance penalty is a geometric decay in the softmax',
        steps: ['\\exp\\big(b_{h,ij}\\big) = \\exp\\big(-m_h\\lvert i-j\\rvert\\big) = \\big(e^{-m_h}\\big)^{\\lvert i-j\\rvert}'],
        note: 'Adding the bias to a score multiplies its unnormalised softmax weight by e^(-m_h) for every step of distance. Nearby keys dominate unless the content match is strong enough to outweigh the decay.',
      },
      {
        title: 'Head slopes form a geometric ladder',
        steps: ['m_h = 2^{-8(h+1)/H} = \\big(2^{-8/H}\\big)^{h+1}', 'm_0 > m_1 > \\cdots > m_{H-1} > 0'],
        note: 'Head 0 is the steepest and looks only a little way back, while the last head decays slowest and reaches furthest. The slopes are fixed, not learned, and do not depend on the sequence length.',
      },
    ],
    live({ trace, cfg, head, row, col }) {
      const m = slope(cfg, head), dist = Math.abs(row - col), result = -m * dist;
      // The printed bias uses the slope rounded to 8 decimals, as the slope line shows it.
      const mS = rnd(m, 8), biasS = rnd(-mS * dist, 4);
      return {
        tex: `\\begin{aligned} ${slopeTex(cfg, head)} \\\\ b_{${head},${row},${col}} &= -m_{${head}}\\,\\lvert ${row}-${col}\\rvert = -(${fix(mS, 8)})(${dist}) = ${fix(biasS, 4)} \\end{aligned}`,
        result,
        expected: at3(trace.alibi, head, row, col),
        note: `The slope is rounded to 8 decimals, and the printed bias is computed from that rounded slope. ${ROUND_NOTE}`,
      };
    },
  },

  'L{i}.attn.q_rot': {
    title: 'Queries after RoPE',
    tex: `\\begin{aligned}
      \\Q_{t,2p} &= q_{t,2p}\\cos(t\\theta_p) - q_{t,2p+1}\\sin(t\\theta_p) \\\\
      \\Q_{t,2p+1} &= q_{t,2p}\\sin(t\\theta_p) + q_{t,2p+1}\\cos(t\\theta_p) \\\\
      \\theta_p &= \\text{base}^{-2p/d_h}
    \\end{aligned}`,
    compact: '\\Q_t = R_t\\,q_t',
    symbols: [
      { tex: '\\Q_{t,c}', text: 'queries of one head after RoPE (row t, column c)', dims: ['T', 'dh'] },
      { tex: 'q_{t,c}', text: 'the same queries before RoPE (q_heads)', dims: ['T', 'dh'] },
      { tex: 'R_t', text: 'block-diagonal rotation by position t, one 2×2 block per pair', dims: ['dh', 'dh'] },
      { tex: 't', text: 'token position, 0 ≤ t < T', dims: ['T'] },
      { tex: 'p', text: 'index of the column pair (2p, 2p+1), 0-based: 0 ≤ p < d_h/2', dims: [] },
      { tex: '\\theta_p', text: 'frequency of pair p, base^(-2p/d_h); pair 0 is the fastest', dims: [] },
    ],
    derivation: [
      {
        title: 'Each pair is a 2-D rotation',
        steps: ['\\begin{pmatrix}\\Q_{t,2p}\\\\ \\Q_{t,2p+1}\\end{pmatrix} = \\begin{pmatrix}\\cos t\\theta_p & -\\sin t\\theta_p\\\\ \\sin t\\theta_p & \\cos t\\theta_p\\end{pmatrix}\\begin{pmatrix}q_{t,2p}\\\\ q_{t,2p+1}\\end{pmatrix}'],
        note: 'Columns are grouped as (0,1), (2,3), and so on. Each pair is a 2-D vector turned by an angle proportional to the token position t.',
      },
      {
        title: 'Length is preserved, so position lives in the angle',
        steps: ['\\Q_{t,2p}^2 + \\Q_{t,2p+1}^2 = \\big(\\cos^{2}(t\\theta_p) + \\sin^{2}(t\\theta_p)\\big)\\big(q_{t,2p}^2 + q_{t,2p+1}^2\\big) = q_{t,2p}^2 + q_{t,2p+1}^2'],
        note: 'A rotation is orthogonal, so RoPE changes the direction of each query pair and not its size.',
      },
      {
        title: 'One frequency per pair',
        steps: ['\\theta_p = \\text{base}^{-2p/d_h}, \\qquad \\theta_0 = 1', '\\lambda_p = \\frac{2\\pi}{\\theta_p} = 2\\pi\\,\\text{base}^{2p/d_h}'],
        note: 'Pair 0 turns one radian per token. Wavelengths grow geometrically, from 2π for the first pair to 2π·base^(1−2/d_h) for the last pair (about 2π·3162 when d_h = 16 and 2π·7499 when d_h = 64), like clock hands at different speeds.',
      },
    ],
    live({ trace, cfg, layer, head, row, col }) {
      const x = row3(trace[`L${layer}.attn.q_heads`], head, row);
      return ropeLive(cfg, x, row, col, { name: '\\Q', src: 'q', expected: at3(trace[`L${layer}.attn.q_rot`], head, row, col) });
    },
  },

  'L{i}.attn.k_rot': {
    title: 'Keys after RoPE',
    tex: `\\begin{aligned}
      \\K_{t,2p} &= k_{t,2p}\\cos(t\\theta_p) - k_{t,2p+1}\\sin(t\\theta_p) \\\\
      \\K_{t,2p+1} &= k_{t,2p}\\sin(t\\theta_p) + k_{t,2p+1}\\cos(t\\theta_p) \\\\
      \\theta_p &= \\text{base}^{-2p/d_h}
    \\end{aligned}`,
    compact: '\\K_t = R_t\\,k_t',
    symbols: [
      { tex: '\\K_{t,c}', text: 'keys of one KV head after RoPE (row t, column c)', dims: ['T', 'dh'] },
      { tex: 'k_{t,c}', text: 'the same keys before RoPE (k_heads)', dims: ['T', 'dh'] },
      { tex: 'R_t', text: 'block-diagonal rotation by position t, one 2×2 block per pair', dims: ['dh', 'dh'] },
      { tex: 't', text: 'token position, 0 ≤ t < T', dims: ['T'] },
      { tex: 'p', text: 'index of the column pair (2p, 2p+1), 0-based: 0 ≤ p < d_h/2', dims: [] },
      { tex: '\\theta_p', text: 'frequency of pair p, base^(-2p/d_h); the same table as for queries', dims: [] },
    ],
    derivation: [
      {
        title: 'Keys turn by the same angles as queries',
        steps: ['\\begin{pmatrix}\\K_{t,2p}\\\\ \\K_{t,2p+1}\\end{pmatrix} = R(t\\theta_p)\\begin{pmatrix}k_{t,2p}\\\\ k_{t,2p+1}\\end{pmatrix}'],
        note: 'Queries and keys share one rotation family, which is what makes the paired dot product depend only on the offset (see concept:rope-relative).',
      },
      {
        title: 'Only the angles carry position',
        steps: ['\\lVert \\K_{t,2p}, \\K_{t,2p+1}\\rVert = \\lVert k_{t,2p}, k_{t,2p+1}\\rVert'],
        note: 'The rotation keeps each key pair the same length, so position changes the angle between queries and keys and never the size of a key.',
      },
    ],
    live({ trace, cfg, layer, head, row, col }) {
      // k_rot is [Hkv, T, dh], so the band is already the KV head index.
      const x = row3(trace[`L${layer}.attn.k_heads`], head, row);
      return ropeLive(cfg, x, row, col, { name: '\\K', src: 'k', expected: at3(trace[`L${layer}.attn.k_rot`], head, row, col) });
    },
  },

  'L{i}.attn.biased': {
    title: 'Scores with the ALiBi bias',
    tex: 'S^{b}_{h,ij} = \\tilde S_{h,ij} + b_{h,ij}, \\qquad \\tilde S_{h,ij} = \\frac{q_i\\cdot k_j}{\\sqrt{d_h}}',
    compact: 'S^{b}_{h,ij} = \\tilde S_{h,ij} + b_{h,ij}',
    symbols: [
      { tex: 'S^{b}_{h,ij}', text: 'score of query i on key j after the ALiBi bias, before the mask', dims: ['H', 'T', 'T'] },
      { tex: '\\tilde S_{h,ij}', text: 'scaled score q_i·k_j/√d_h (the raw score when attn_scale is off)', dims: ['H', 'T', 'T'] },
      { tex: 'b_{h,ij}', text: 'ALiBi bias, -m_h·|i-j|', dims: ['H', 'T', 'T'] },
      { tex: 'm_h', text: 'ALiBi slope of head h', dims: ['H'] },
      { tex: 'q_i, k_j', text: 'query of row i and key of column j in one head (no RoPE in ALiBi models)', dims: ['dh'] },
    ],
    derivation: [
      {
        title: 'The bias is added before the mask',
        steps: [
          'S^{b}_{h,ij} = \\tilde S_{h,ij} + b_{h,ij}',
          '\\hat S_{h,ij} = \\begin{cases} S^{b}_{h,ij} & \\Mk_{ij} = 1 \\\\ -\\infty & \\Mk_{ij} = 0 \\end{cases}',
        ],
        note: 'The bias is finite everywhere, so the mask still decides visibility: a masked key stays at -∞ whatever its bias. For ALiBi models this S^b is the score that the mask and softmax read.',
      },
      {
        title: 'Multiplicative view',
        steps: ['\\exp\\big(S^{b}_{h,ij}\\big) = \\exp\\big(\\tilde S_{h,ij}\\big)\\,\\big(e^{-m_h}\\big)^{\\lvert i-j\\rvert}'],
        note: 'Softmax exponentiates its inputs, so an additive bias is a multiplicative decay on each unnormalised weight.',
      },
    ],
    live({ trace, cfg, layer, head, row, col }) {
      // ALiBi models have no RoPE, so q_heads and k_heads are the scores' inputs.
      const q = row3(trace[`L${layer}.attn.q_heads`], head, row);
      const k = row3(trace[`L${layer}.attn.k_heads`], kvHead(cfg, head), col);
      const dhv = dh(cfg), m = slope(cfg, head), dist = Math.abs(row - col);
      const raw = dot(q, k);
      const result = (cfg.attn_scale ? raw / Math.sqrt(dhv) : raw) - m * dist;
      // Printed chain: factors, raw and scaled scores to 6 decimals (guard digits), the slope to
      // 8, and the result to 4. Each line is computed from the printed values above it.
      const qS = q.map((v) => rnd(v, 6)), kS = k.map((v) => rnd(v, 6));
      const rawS = rnd(dot(qS, kS), 6);
      const mS = rnd(m, 8);
      const prod = productsTex(qS, kS);
      const lines = [slopeTex(cfg, head)];
      if (cfg.attn_scale) {
        const sdTex = trimZeros(fix(Math.sqrt(dhv), 6)), sdS = Number(sdTex);
        const scaledS = rnd(rawS / sdS, 6), resultS = rnd(scaledS - mS * dist, 4);
        lines.push(
          `S^{b}_{${head},${row},${col}} &= \\frac{${prod}}{\\sqrt{${dhv}}} - m_{${head}}\\lvert ${row}-${col}\\rvert`,
          `&= \\frac{${fix(rawS, 6)}}{\\sqrt{${dhv}}} = \\frac{${fix(rawS, 6)}}{${sdTex}} = ${fix(scaledS, 6)}`,
          `&= ${fix(scaledS, 6)} - (${fix(mS, 8)})(${dist}) = ${fix(resultS, 4)}`,
        );
      } else {
        const resultS = rnd(rawS - mS * dist, 4);
        lines.push(
          `S^{b}_{${head},${row},${col}} &= ${prod} - m_{${head}}\\lvert ${row}-${col}\\rvert`,
          `&= ${fix(rawS, 6)} - (${fix(mS, 8)})(${dist}) = ${fix(resultS, 4)}`,
        );
      }
      const note = [
        cfg.attn_scale ? 'Scaled by 1/√d_h, then the ALiBi bias is added.' : 'attn_scale is off, so the raw dot product is used.',
        dhv > 4 && 'The middle terms of the sum are elided; the raw score sums all d_h of them.',
        ROUND_NOTE,
      ].filter(Boolean).join(' ');
      return {
        tex: `\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`,
        result,
        expected: at3(trace[`L${layer}.attn.biased`], head, row, col),
        note,
      };
    },
  },

  'concept:sinusoidal': {
    title: 'Sinusoidal positional encoding',
    tex: '\\Pe_{t,2p} = \\sin(t\\,\\omega_p), \\qquad \\Pe_{t,2p+1} = \\cos(t\\,\\omega_p), \\qquad \\omega_p = 10000^{-2p/d}',
    compact: '\\Pe_{t,2p} = \\sin(t\\omega_p),\\ \\Pe_{t,2p+1} = \\cos(t\\omega_p)',
    symbols: [
      { tex: '\\Pe_{t}', text: 'encoding vector of position t, added to the token embedding', dims: ['d'] },
      { tex: 'p', text: 'index of a sine/cosine pair, 0-based: 0 ≤ p < d/2', dims: [] },
      { tex: '\\omega_p', text: 'angular frequency of pair p; pair 0 is the fastest', dims: [] },
      { tex: 't', text: 'token position, 0 ≤ t < T', dims: ['T'] },
      { tex: '\\Delta', text: 'a fixed shift in positions (an integer offset)', dims: [] },
      { tex: 'M_{\\Delta}', text: 'block-diagonal map for a shift by Δ: one 2×2 block per pair; its blocks depend on Δ and the fixed ω_p, not on t', dims: ['d', 'd'] },
    ],
    derivation: [
      {
        title: 'Shifting by Δ is a fixed linear map',
        steps: [
          '\\begin{pmatrix}\\sin((t+\\Delta)\\omega_p)\\\\ \\cos((t+\\Delta)\\omega_p)\\end{pmatrix} = \\begin{pmatrix}\\cos\\Delta\\omega_p & \\sin\\Delta\\omega_p\\\\ -\\sin\\Delta\\omega_p & \\cos\\Delta\\omega_p\\end{pmatrix}\\begin{pmatrix}\\sin t\\omega_p\\\\ \\cos t\\omega_p\\end{pmatrix}',
          '\\Pe_{t+\\Delta} = M_{\\Delta}\\,\\Pe_t',
        ],
        note: 'The angle-addition identities give the first line. M_Δ depends on Δ and the fixed ω_p, not on t, so one fixed linear map moves any position by Δ. Vaswani et al. (2017) motivate the encoding this way.',
      },
      {
        title: 'Similarity depends only on the offset',
        steps: ['\\Pe_s\\cdot\\Pe_t = \\sum_{p=0}^{d/2-1}\\big(\\sin s\\omega_p\\sin t\\omega_p + \\cos s\\omega_p\\cos t\\omega_p\\big) = \\sum_{p=0}^{d/2-1}\\cos\\big((s-t)\\omega_p\\big)'],
        note: 'Each sine/cosine pair traces a circle as t grows, and a shift is a rotation of that circle. So the dot product of two encodings is a function of their distance alone. The sums run over the 0-based pair index p.',
      },
    ],
  },

  'concept:rope-relative': {
    title: 'RoPE: scores depend only on relative position',
    tex: '\\begin{aligned} \\langle R_{m}\\,q,\\; R_{n}\\,k\\rangle &= \\langle q,\\; R_{n-m}\\,k\\rangle \\\\ R_{t} &= \\operatorname{diag}\\big(R(t\\theta_0),\\ldots,R(t\\theta_{d_h/2-1})\\big) \\\\ R(\\varphi) &= \\begin{pmatrix}\\cos\\varphi & -\\sin\\varphi\\\\ \\sin\\varphi & \\cos\\varphi\\end{pmatrix} \\end{aligned}',
    compact: '\\langle R_m q,\\, R_n k\\rangle = \\langle q,\\, R_{n-m} k\\rangle',
    symbols: [
      { tex: 'R_{t}', text: 'block-diagonal rotation by position t: one 2×2 block per pair', dims: ['dh', 'dh'] },
      { tex: 'R(\\varphi)', text: '2×2 rotation matrix by angle φ', dims: [] },
      { tex: 'q, k', text: 'query vector q and key vector k of one head, before RoPE', dims: ['dh'] },
      { tex: 'm', text: 'position of the query q', dims: ['T'] },
      { tex: 'n', text: 'position of the key k', dims: ['T'] },
      { tex: '\\theta_p', text: 'frequency of pair p: base^(-2p/d_h), with 0-based pair index p', dims: [] },
    ],
    derivation: [
      {
        title: 'One pair: rotations add their angles',
        steps: [
          'R(\\alpha)^{\\top} = R(-\\alpha), \\qquad R(\\alpha)\\,R(\\beta) = R(\\alpha+\\beta)',
          '\\langle R(m\\theta)\\,q,\\; R(n\\theta)\\,k\\rangle = q^{\\top}R(m\\theta)^{\\top}R(n\\theta)\\,k = q^{\\top}R\\big((n-m)\\theta\\big)\\,k',
        ],
        note: 'R(α) is orthogonal, so its transpose is its inverse. Turning the query by mθ and the key by nθ combines into one turn by (n−m)θ, so the logit depends only on the offset n−m.',
      },
      {
        title: 'Whole head: blockwise',
        steps: [
          'R_{m}^{\\top}R_{n} = \\operatorname{diag}\\big(R((n-m)\\theta_0),\\ldots\\big) = R_{n-m}',
          's_{mn} = (R_{m}q)^{\\top}(R_{n}k) = q^{\\top}R_{n-m}\\,k',
        ],
        note: 'Each pair keeps its own θ_p, and the 2×2 blocks never mix, so the identity holds block by block. Only the relative offset survives.',
      },
      {
        title: 'Only the angles carry position',
        steps: ['\\lVert R_{t}\\,q\\rVert = \\lVert q\\rVert'],
        note: 'RoPE is applied to queries and keys but not to values. Norms are unchanged, so position alters the attention pattern without changing the size of the vectors.',
      },
    ],
  },

  'concept:alibi': {
    title: 'ALiBi: a linear distance penalty',
    tex: 'S^{b}_{h,ij} = \\tilde S_{h,ij} - m_h\\,\\lvert i-j\\rvert, \\qquad \\tilde S_{h,ij} = \\frac{q_i\\cdot k_j}{\\sqrt{d_h}}, \\qquad m_h = 2^{-8(h+1)/H}',
    compact: 'S^{b}_{h,ij} = \\tilde S_{h,ij} - m_h\\,\\lvert i-j\\rvert',
    symbols: [
      { tex: 'm_h', text: 'slope of head h; fixed, not learned', dims: ['H'] },
      { tex: '\\lvert i-j\\rvert', text: 'distance between query i and key j (equals i-j under the causal mask)', dims: ['T', 'T'] },
      { tex: '\\tilde S_{h,ij}', text: 'scaled score q_i·k_j/√d_h (with attn_scale off, the raw dot product q_i·k_j)', dims: ['H', 'T', 'T'] },
      { tex: 'S^{b}_{h,ij}', text: 'score with the distance penalty, before the mask and softmax', dims: ['H', 'T', 'T'] },
    ],
    derivation: [
      {
        title: 'The penalty depends only on distance',
        steps: ['b_{h,ij} = f_h(i-j), \\qquad f_h(\\Delta) = -m_h\\lvert\\Delta\\rvert'],
        note: 'No position vector is added to the stream and no positional parameters are learned. The same rule applies at every position, so no embedding table for positions is needed.',
      },
      {
        title: 'Geometric decay in the softmax',
        steps: ['\\exp\\big(S^{b}_{h,ij}\\big) = \\exp\\big(\\tilde S_{h,ij}\\big)\\,\\big(e^{-m_h}\\big)^{\\lvert i-j\\rvert}', 'm_h = \\big(2^{-8/H}\\big)^{h+1}'],
        note: 'Each extra step back multiplies the unnormalised weight by e^(-m_h), so steep heads attend locally and shallow heads reach further. The rule has no length limit, which is why ALiBi was proposed for running on inputs longer than those seen in training (Press et al., 2022). With attn_scale off, the first term is the raw dot product q_i·k_j.',
      },
    ],
  },
};

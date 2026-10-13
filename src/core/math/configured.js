// The primary equation follows the selected architecture. General alternatives
// and derivations remain available in the catalogue.
import { parseKey } from '../describe.js';

export function configuredEquation(key, cfg, entry, compact = false) {
  const { pattern } = parseKey(key), norm = cfg.norm === 'rmsnorm' ? '\\RMSNorm' : '\\LN';
  const aligned = (lines) => `\\begin{aligned} ${lines.join(' \\\\ ')} \\end{aligned}`;
  if (/^(L\{i\}\.ln[12]|lnf)$/.test(pattern)) {
    const input = pattern === 'L{i}.ln2' ? 'x^{\\text{mid}}' : '\\xres', out = pattern === 'lnf' ? 'h^f' : 'h';
    if (cfg.norm === 'rmsnorm') return `${out}_{t,c}=\\gamma_c\\,\\frac{${input}_{t,c}}{\\sqrt{\\frac1d\\sum_j (${input}_{t,j})^{2}+\\varepsilon}}`;
    const result = `${out}_{t,c}=\\gamma_c\\,\\frac{${input}_{t,c}-\\mu_t}{\\sqrt{\\sigma_t^2+\\varepsilon}}${cfg.bias ? '+\\beta_c' : ''}`;
    return compact ? result : aligned([
      `\\mu_t &= \\frac1d\\sum_j ${input}_{t,j}`,
      `\\sigma_t^2 &= \\frac1d\\sum_j (${input}_{t,j}-\\mu_t)^2`, result.replace('=', '&='),
    ]);
  }
  if (/^L\{i\}\.(resid_(mid|out)|add[12])$/.test(pattern)) {
    const second = /out|add2/.test(pattern), input = second ? 'x^{\\text{mid}}' : '\\xres';
    const branch = second ? 'm' : 'a', output = second ? 'x^{\\text{out}}' : 'x^{\\text{mid}}', n = second ? 2 : 1;
    const sum = cfg.residual ? `${input}+${branch}` : branch;
    if (/add[12]$/.test(pattern)) return `s^{(${n})}=${sum}`;
    const result = `${output}=${cfg.norm_placement === 'pre' ? sum : `${norm}_${n}(${sum})`}`;
    return compact ? result : aligned([
      `${branch} &= ${second ? '\\MLP' : '\\Attn'}(${cfg.norm_placement === 'pre' ? `${norm}_${n}(${input})` : input})`,
      result.replace('=', '&='),
    ]);
  }
  if (pattern === 'L{i}.attn.masked' && !compact) return aligned([
    `Z &= ${cfg.attn_scale ? '\\Q\\K^\\top/\\sqrt{d_h}' : '\\Q\\K^\\top'}${cfg.pos_encoding === 'alibi' ? '+B' : ''}`,
    '\\hat S_{ij} &= \\begin{cases} Z_{ij} & \\Mk_{ij}=1 \\\\ -\\infty & \\Mk_{ij}=0 \\end{cases}',
  ]);
  if (pattern === 'L{i}.mlp.act') return `a_{t,f}=${cfg.activation === 'swiglu' ? '\\SiLU(g_{t,f})\\,h_{t,f}' : cfg.activation === 'gelu' ? '\\GELU(h_{t,f})' : '\\ReLU(h_{t,f})'}`;
  if (pattern === 'L{i}.attn.scores' && !compact) return aligned(['S &= \\Q\\K^\\top', 'S_{ij} &= \\sum_c q_{ic}k_{jc}']);
  if (pattern === 'L{i}.attn.ctx' && !compact) return aligned(['\\text{head}_h &= \\Aw_h\\V_h', 'o_i &= \\sum_j \\Aw_{ij}v_j']);
  return compact ? entry.compact || entry.tex : entry.tex;
}

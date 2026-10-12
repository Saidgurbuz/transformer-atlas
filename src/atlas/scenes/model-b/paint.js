// Canvas painting for the root view (variant b): token keycap faces, attention patterns and the
// soft contact shadow. Everything here is drawn once per build into mip-mapped textures, so it
// stays crisp when near and calm (no shimmer) when far.
import { THREE } from '../../../scene/engine3d.js';

const MONO = '"JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace';
const hex = (c) => `#${c.getHexString()}`;      // THREE.Color (linear) -> sRGB hex for canvas

function texture(canvas, renderer) {
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  return t;
}

/**
 * The face of a token keycap: the token text centred; a leading '␣' (space marker) drawn quieter.
 * w, h in world units; ppu = canvas pixels per unit.
 */
export function keycapTexture(renderer, text, { w, h, fg, dim, bg = null, ppu = 300 }) {
  const c = document.createElement('canvas');
  c.width = Math.max(16, Math.round(w * ppu));
  c.height = Math.max(16, Math.round(h * ppu));
  const g = c.getContext('2d');
  if (bg) { g.fillStyle = hex(bg); g.fillRect(0, 0, c.width, c.height); }
  const px = Math.round(h * ppu * 0.64);
  g.font = `600 ${px}px ${MONO}`;
  g.textBaseline = 'middle';
  const chars = Array.from(String(text));
  const lead = chars[0] === '␣' ? '␣' : '';
  const rest = lead ? chars.slice(1).join('') : chars.join('');
  const wl = lead ? g.measureText(lead).width * 0.8 : 0, wr = g.measureText(rest).width;
  const total = wl + wr, fit = Math.min(1, (c.width * 0.9) / Math.max(1, total));
  g.save();
  g.translate(c.width / 2, c.height / 2 + px * 0.04);
  g.scale(fit, fit);
  let x = -total / 2;
  if (lead) {
    g.fillStyle = hex(dim);
    g.save(); g.translate(x, 0); g.scale(0.8, 0.8); g.fillText(lead, 0, 0); g.restore();
    x += wl;
  }
  g.fillStyle = hex(fg);
  g.fillText(rest, x, 0);
  g.restore();
  return texture(c, renderer);
}

/**
 * One head's attention pattern, rows = queries (top = first token), columns = keys.
 * Values are the real probabilities, shown on a square-root ramp (sequential: lo -> hi).
 * Masked cells are left as the screen colour so the causal triangle reads at a glance.
 */
export function patternTexture(renderer, probs, mask, hh, { lo, hi, masked = lo, gamma = 0.5 }) {
  const T = probs.shape[1];
  const S = T <= 16 ? 128 : 256;
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d');
  g.fillStyle = hex(masked);
  g.fillRect(0, 0, S, S);
  const cell = S / T, col = new THREE.Color();
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      if (mask && !(mask.data[i * T + j] > 0)) continue;
      const p = probs.data[(hh * T + i) * T + j];
      const v = Math.pow(Math.max(0, Math.min(1, p)), gamma);
      col.copy(lo).lerp(hi, v);
      g.fillStyle = hex(col);
      // integer-aligned cells with no seams
      const x0 = Math.round(j * cell), y0 = Math.round(i * cell), x1 = Math.round((j + 1) * cell), y1 = Math.round((i + 1) * cell);
      g.fillRect(x0, y0, x1 - x0, y1 - y0);
    }
  }
  return texture(c, renderer);
}

/** A soft elliptical contact shadow (white = opaque; used as an alpha map). */
export function shadowTexture(renderer) {
  const S = 256, c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.35, 'rgba(255,255,255,0.55)');
  grad.addColorStop(0.7, 'rgba(255,255,255,0.14)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, S, S);
  const t = texture(c, renderer);
  t.colorSpace = THREE.NoColorSpace;
  return t;
}


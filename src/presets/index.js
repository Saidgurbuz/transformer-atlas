// Pretrained tiny models shipped with the app (trained by tools/train_presets.py
// with the PyTorch reference, exported as base64 float32 by the same script).
// Each preset module default-exports {id, title, description, vocab, config,
// samples, order: [[name, shape]], weights: base64 string}.

import { MANIFEST } from './manifest.js';

// [{id, title, description}] — written by tools/train_presets.py
export const PRESETS = MANIFEST;

function decodeBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// IEEE half -> double
function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function decodeWeights(b64, dtype = 'f32') {
  const bytes = decodeBytes(b64);
  if (dtype !== 'f16') return new Float32Array(bytes.buffer);
  const u16 = new Uint16Array(bytes.buffer), out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

const cache = new Map();

export async function loadPresetData(id) {
  if (cache.has(id)) return cache.get(id);
  const mod = (await import(`./${id}.js`)).default;
  const flat = decodeWeights(mod.weights, mod.dtype);
  const weights = {};
  let off = 0;
  for (const [name, shape] of mod.order) {
    const n = shape.reduce((a, b) => a * b, 1);
    weights[name] = { shape, data: flat.subarray(off, off + n) };
    off += n;
  }
  const data = { ...mod, weights };
  cache.set(id, data);
  return data;
}

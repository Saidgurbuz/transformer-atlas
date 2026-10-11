// Pretrained tiny models shipped with the app (trained by tools/train_presets.py
// with the PyTorch reference, exported as base64 float32 by the same script).
// Each preset module default-exports {id, title, description, vocab, config,
// samples, order: [[name, shape]], weights: base64 string}.

import { MANIFEST } from './manifest.js';

// [{id, title, description}] — written by tools/train_presets.py
export const PRESETS = MANIFEST;

function decodeFloat32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

const cache = new Map();

export async function loadPresetData(id) {
  if (cache.has(id)) return cache.get(id);
  const mod = (await import(`./${id}.js`)).default;
  const flat = decodeFloat32(mod.weights);
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

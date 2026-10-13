// Small strided tensors for explaining PyTorch's view/transpose/contiguous semantics.
// Shared-storage views carry offsets; no values are invented by the presentation.
export const sizeOf = (shape) => shape.reduce((n, d) => n * d, 1);
export function contiguousStrides(shape) {
  let n = 1;
  return shape.map((_, i) => i).reverse().map((i) => { const s = n; n *= shape[i]; return s; }).reverse();
}
export function tensor(data, shape) {
  if (shape.some((d) => !Number.isInteger(d) || d < 1) || sizeOf(shape) !== data.length) throw new Error('Tensor shape does not match storage');
  return { data: Float64Array.from(data), shape: [...shape], strides: contiguousStrides(shape), offset: 0 };
}
export function isContiguous(t) {
  let expected = 1;
  for (let i = t.shape.length - 1; i >= 0; i--) {
    if (t.shape[i] !== 1 && t.strides[i] !== expected) return false;
    expected *= t.shape[i];
  }
  return true;
}
export function coordinates(i, shape) {
  const c = Array(shape.length);
  for (let d = shape.length - 1; d >= 0; d--) { c[d] = i % shape[d]; i = Math.floor(i / shape[d]); }
  return c;
}
export function storageIndex(t, coord) {
  if (coord.length !== t.shape.length || coord.some((v, i) => !Number.isInteger(v) || v < 0 || v >= t.shape[i])) throw new Error('Coordinate out of range');
  return t.offset + coord.reduce((s, v, i) => s + v * t.strides[i], 0);
}
export function logicalValues(t) {
  return Float64Array.from({ length: sizeOf(t.shape) }, (_, i) => t.data[storageIndex(t, coordinates(i, t.shape))]);
}
export function view(t, shape) {
  if (!isContiguous(t) || sizeOf(shape) !== sizeOf(t.shape)) throw new Error('This lesson view requires contiguous storage and an unchanged element count');
  return { ...t, shape: [...shape], strides: contiguousStrides(shape) };
}
export function transpose(t, a, b) {
  const shape = [...t.shape], strides = [...t.strides];
  if (![a, b].every((i) => Number.isInteger(i) && i >= 0 && i < shape.length)) throw new Error('Invalid transpose axis');
  [shape[a], shape[b]] = [shape[b], shape[a]];
  [strides[a], strides[b]] = [strides[b], strides[a]];
  return { ...t, shape, strides };
}
export function contiguous(t) { return isContiguous(t) ? t : tensor(logicalValues(t), t.shape); }
export function reshape(t, shape) { return isContiguous(t) ? view(t, shape) : tensor(logicalValues(t), shape); }
export function unsqueeze(t, axis) {
  if (!Number.isInteger(axis) || axis < 0 || axis > t.shape.length) throw new Error('Invalid unsqueeze axis');
  const shape = [...t.shape], strides = [...t.strides];
  strides.splice(axis, 0, axis < shape.length ? shape[axis] * strides[axis] : 1);
  shape.splice(axis, 0, 1);
  return { ...t, shape, strides };
}
export function squeeze(t, axis) {
  const shape = [...t.shape], strides = [...t.strides];
  if (!Number.isInteger(axis) || axis < 0 || axis >= shape.length) throw new Error('Invalid squeeze axis');
  if (shape[axis] !== 1) return t; // torch.squeeze(dim) leaves non-singleton axes unchanged
  shape.splice(axis, 1); strides.splice(axis, 1);
  return { ...t, shape, strides };
}
export function cat(tensors, axis) {
  if (!tensors.length) throw new Error('cat needs at least one tensor');
  const shape = [...tensors[0].shape];
  if (!Number.isInteger(axis) || axis < 0 || axis >= shape.length) throw new Error('Invalid cat axis');
  for (const t of tensors) if (t.shape.length !== shape.length || t.shape.some((d, i) => i !== axis && d !== shape[i])) throw new Error('cat requires matching non-concatenated dimensions');
  shape[axis] = tensors.reduce((n, t) => n + t.shape[axis], 0);
  return tensor(Float64Array.from({ length: sizeOf(shape) }, (_, i) => {
    const c = coordinates(i, shape); let j = c[axis];
    for (const t of tensors) { if (j < t.shape[axis]) { c[axis] = j; return t.data[storageIndex(t, c)]; } j -= t.shape[axis]; }
  }), shape);
}
export function headScores(q, k) {
  const [B, H, T, D] = q.shape, S = k.shape[2];
  if (q.shape.length !== 4 || k.shape.length !== 4 || k.shape[0] !== B || k.shape[1] !== H || k.shape[3] !== D) throw new Error('Head shapes do not match');
  const out = new Float64Array(B * H * T * S);
  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) for (let s = 0; s < S; s++) {
    let v = 0;
    for (let d = 0; d < D; d++) v += q.data[storageIndex(q, [b, h, t, d])] * k.data[storageIndex(k, [b, h, s, d])];
    out[((b * H + h) * T + t) * S + s] = v;
  }
  return tensor(out, [B, H, T, S]);
}
export function tensorExample() {
  const arange = tensor(Array.from({ length: 36 }, (_, i) => i), [36]), x = view(arange, [2, 3, 6]);
  const split = view(x, [2, 3, 2, 3]), heads = transpose(split, 1, 2);
  const copied = contiguous(heads), reshaped = reshape(heads, [2, 6, 3]), merged = view(contiguous(transpose(heads, 1, 2)), [2, 3, 6]);
  const pieces = Array.from({ length: 2 }, (_, h) => ({ ...heads, shape: [2, 3, 3], strides: [heads.strides[0], heads.strides[2], heads.strides[3]], offset: h * heads.strides[1] }));
  const positions = tensor([0, 1, 2], [3]);
  return { arange, x, split, heads, copied, reshaped, merged, concatenated: cat(pieces, 2), positions,
    positionRow: unsqueeze(positions, 0), positionFlat: squeeze(unsqueeze(positions, 0), 0), scores: headScores(heads, heads) };
}

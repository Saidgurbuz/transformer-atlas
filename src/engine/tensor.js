// Small reverse-mode autograd over 2-D Float64 tensors.
// Every op mirrors one PyTorch call in reference/model.py. Gradients are only
// tracked while grad mode is on and at least one input requires a gradient.

let GRAD_ENABLED = true;

export function noGrad(fn) {
  const prev = GRAD_ENABLED;
  GRAD_ENABLED = false;
  try { return fn(); } finally { GRAD_ENABLED = prev; }
}

export class Tensor {
  constructor(data, shape) {
    this.data = data;
    this.shape = shape;
    this.grad = null;
    this.requiresGrad = false;
    this._parents = null;
    this._backward = null;
  }

  get size() { return this.data.length; }
  get rows() { return this.shape[0]; }
  get cols() { return this.shape[this.shape.length - 1]; }

  static zeros(shape) { return new Tensor(new Float64Array(shape.reduce((a, b) => a * b, 1)), shape.slice()); }

  static from(values, shape) { return new Tensor(Float64Array.from(values), shape.slice()); }

  static param(data, shape) {
    const t = new Tensor(data instanceof Float64Array ? data : Float64Array.from(data), shape.slice());
    t.requiresGrad = true;
    return t;
  }

  item() { return this.data[0]; }

  at(...idx) {
    let off = 0;
    for (let a = 0; a < idx.length; a++) off = off * this.shape[a] + idx[a];
    return this.data[off];
  }

  // Row r of a 2-D tensor (or the [r] slice of the first axis) as a subarray view.
  row(r) {
    const w = this.size / this.shape[0];
    return this.data.subarray(r * w, (r + 1) * w);
  }

  ensureGrad() {
    if (!this.grad) this.grad = new Float64Array(this.data.length);
    return this.grad;
  }

  zeroGrad() { if (this.grad) this.grad.fill(0); }

  backward(seed = 1) {
    const order = [];
    const seen = new Set();
    const visit = (t) => {
      if (seen.has(t)) return;
      seen.add(t);
      if (t._parents) for (const p of t._parents) visit(p);
      order.push(t);
    };
    visit(this);
    this.ensureGrad().fill(seed);
    for (let n = order.length - 1; n >= 0; n--) {
      const t = order[n];
      if (t._backward) t._backward();
    }
    // Free intermediate graph references so memory can be reclaimed.
    for (const t of order) {
      if (t._parents) { t._parents = null; t._backward = null; }
    }
  }
}

function result(data, shape, parents, backward) {
  const out = new Tensor(data, shape);
  if (GRAD_ENABLED && parents.some((p) => p && p.requiresGrad)) {
    out.requiresGrad = true;
    out._parents = parents.filter(Boolean);
    out._backward = () => backward(out.grad || out.ensureGrad());
  }
  return out;
}

const need = (t) => t && t.requiresGrad;

// y = x @ W^T + b   (x: [T, I], W: [O, I] stored like nn.Linear, b: [O] or null)
export function linear(x, W, b = null) {
  const [T, I] = x.shape;
  const O = W.shape[0];
  if (W.shape[1] !== I) throw new Error(`linear: x has ${I} features but W expects ${W.shape[1]}`);
  const xd = x.data, wd = W.data, out = new Float64Array(T * O);
  for (let t = 0; t < T; t++) {
    const xo = t * I;
    for (let o = 0; o < O; o++) {
      const wo = o * I;
      let s = b ? b.data[o] : 0;
      for (let i = 0; i < I; i++) s += xd[xo + i] * wd[wo + i];
      out[t * O + o] = s;
    }
  }
  return result(out, [T, O], [x, W, b], (g) => {
    if (need(x)) {
      const gx = x.ensureGrad();
      for (let t = 0; t < T; t++) for (let o = 0; o < O; o++) {
        const gv = g[t * O + o];
        if (gv === 0) continue;
        const wo = o * I, xo = t * I;
        for (let i = 0; i < I; i++) gx[xo + i] += gv * wd[wo + i];
      }
    }
    if (need(W)) {
      const gw = W.ensureGrad();
      for (let t = 0; t < T; t++) for (let o = 0; o < O; o++) {
        const gv = g[t * O + o];
        if (gv === 0) continue;
        const wo = o * I, xo = t * I;
        for (let i = 0; i < I; i++) gw[wo + i] += gv * xd[xo + i];
      }
    }
    if (need(b)) {
      const gb = b.ensureGrad();
      for (let t = 0; t < T; t++) for (let o = 0; o < O; o++) gb[o] += g[t * O + o];
    }
  });
}

// C = A @ B   (A: [M, K], B: [K, N])
export function matmul(A, B) {
  const [M, K] = A.shape, N = B.shape[1];
  if (B.shape[0] !== K) throw new Error('matmul: inner dimensions differ');
  const a = A.data, b = B.data, out = new Float64Array(M * N);
  for (let m = 0; m < M; m++) for (let k = 0; k < K; k++) {
    const av = a[m * K + k];
    if (av === 0) continue;
    for (let n = 0; n < N; n++) out[m * N + n] += av * b[k * N + n];
  }
  return result(out, [M, N], [A, B], (g) => {
    if (need(A)) {
      const ga = A.ensureGrad();
      for (let m = 0; m < M; m++) for (let k = 0; k < K; k++) {
        let s = 0;
        for (let n = 0; n < N; n++) s += g[m * N + n] * b[k * N + n];
        ga[m * K + k] += s;
      }
    }
    if (need(B)) {
      const gb = B.ensureGrad();
      for (let m = 0; m < M; m++) for (let k = 0; k < K; k++) {
        const av = a[m * K + k];
        if (av === 0) continue;
        for (let n = 0; n < N; n++) gb[k * N + n] += av * g[m * N + n];
      }
    }
  });
}

// C = A @ B^T   (A: [M, K], B: [N, K])  used for q @ k^T
export function matmulNT(A, B) {
  const [M, K] = A.shape, N = B.shape[0];
  if (B.shape[1] !== K) throw new Error('matmulNT: inner dimensions differ');
  const a = A.data, b = B.data, out = new Float64Array(M * N);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += a[m * K + k] * b[n * K + k];
    out[m * N + n] = s;
  }
  return result(out, [M, N], [A, B], (g) => {
    if (need(A)) {
      const ga = A.ensureGrad();
      for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
        const gv = g[m * N + n];
        if (gv === 0) continue;
        for (let k = 0; k < K; k++) ga[m * K + k] += gv * b[n * K + k];
      }
    }
    if (need(B)) {
      const gb = B.ensureGrad();
      for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
        const gv = g[m * N + n];
        if (gv === 0) continue;
        for (let k = 0; k < K; k++) gb[n * K + k] += gv * a[m * K + k];
      }
    }
  });
}

export function add(A, B) {
  if (A.size !== B.size) throw new Error('add: size mismatch');
  const out = new Float64Array(A.size);
  for (let i = 0; i < out.length; i++) out[i] = A.data[i] + B.data[i];
  return result(out, A.shape.slice(), [A, B], (g) => {
    if (need(A)) { const ga = A.ensureGrad(); for (let i = 0; i < g.length; i++) ga[i] += g[i]; }
    if (need(B)) { const gb = B.ensureGrad(); for (let i = 0; i < g.length; i++) gb[i] += g[i]; }
  });
}

export function mul(A, B) {
  if (A.size !== B.size) throw new Error('mul: size mismatch');
  const out = new Float64Array(A.size);
  for (let i = 0; i < out.length; i++) out[i] = A.data[i] * B.data[i];
  return result(out, A.shape.slice(), [A, B], (g) => {
    if (need(A)) { const ga = A.ensureGrad(); for (let i = 0; i < g.length; i++) ga[i] += g[i] * B.data[i]; }
    if (need(B)) { const gb = B.ensureGrad(); for (let i = 0; i < g.length; i++) gb[i] += g[i] * A.data[i]; }
  });
}

export function scale(A, s) {
  const out = new Float64Array(A.size);
  for (let i = 0; i < out.length; i++) out[i] = A.data[i] * s;
  return result(out, A.shape.slice(), [A], (g) => {
    const ga = A.ensureGrad();
    for (let i = 0; i < g.length; i++) ga[i] += g[i] * s;
  });
}

// A + C where C is a constant Float64Array of the same size (ALiBi bias).
export function addConst(A, C) {
  const out = new Float64Array(A.size);
  for (let i = 0; i < out.length; i++) out[i] = A.data[i] + C[i];
  return result(out, A.shape.slice(), [A], (g) => {
    const ga = A.ensureGrad();
    for (let i = 0; i < g.length; i++) ga[i] += g[i];
  });
}

// Row lookup: W[ids[t]] for each t   (W: [V, d]) -> [T, d]
export function embedding(W, ids) {
  const d = W.shape[1], T = ids.length, out = new Float64Array(T * d);
  for (let t = 0; t < T; t++) {
    const id = ids[t];
    if (id < 0 || id >= W.shape[0]) throw new Error(`embedding: id ${id} out of range`);
    out.set(W.data.subarray(id * d, (id + 1) * d), t * d);
  }
  return result(out, [T, d], [W], (g) => {
    const gw = W.ensureGrad();
    for (let t = 0; t < T; t++) {
      const o = ids[t] * d;
      for (let j = 0; j < d; j++) gw[o + j] += g[t * d + j];
    }
  });
}

export function layerNorm(x, w, b, eps) {
  const [T, d] = x.shape, xd = x.data, out = new Float64Array(T * d);
  const xhat = new Float64Array(T * d), rstd = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const o = t * d;
    let mu = 0;
    for (let j = 0; j < d; j++) mu += xd[o + j];
    mu /= d;
    let v = 0;
    for (let j = 0; j < d; j++) { const c = xd[o + j] - mu; v += c * c; }
    v /= d;
    const r = 1 / Math.sqrt(v + eps);
    rstd[t] = r;
    for (let j = 0; j < d; j++) {
      const h = (xd[o + j] - mu) * r;
      xhat[o + j] = h;
      out[o + j] = h * w.data[j] + (b ? b.data[j] : 0);
    }
  }
  return result(out, [T, d], [x, w, b], (g) => {
    if (need(w)) { const gw = w.ensureGrad(); for (let i = 0; i < g.length; i++) gw[i % d] += g[i] * xhat[i]; }
    if (need(b)) { const gb = b.ensureGrad(); for (let i = 0; i < g.length; i++) gb[i % d] += g[i]; }
    if (need(x)) {
      const gx = x.ensureGrad();
      for (let t = 0; t < T; t++) {
        const o = t * d;
        let m1 = 0, m2 = 0;
        for (let j = 0; j < d; j++) { const gh = g[o + j] * w.data[j]; m1 += gh; m2 += gh * xhat[o + j]; }
        m1 /= d; m2 /= d;
        for (let j = 0; j < d; j++) {
          const gh = g[o + j] * w.data[j];
          gx[o + j] += rstd[t] * (gh - m1 - xhat[o + j] * m2);
        }
      }
    }
  });
}

export function rmsNorm(x, w, eps) {
  const [T, d] = x.shape, xd = x.data, out = new Float64Array(T * d), rinv = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const o = t * d;
    let ms = 0;
    for (let j = 0; j < d; j++) ms += xd[o + j] * xd[o + j];
    const r = 1 / Math.sqrt(ms / d + eps);
    rinv[t] = r;
    for (let j = 0; j < d; j++) out[o + j] = xd[o + j] * r * w.data[j];
  }
  return result(out, [T, d], [x, w], (g) => {
    if (need(w)) { const gw = w.ensureGrad(); for (let i = 0; i < g.length; i++) gw[i % d] += g[i] * xd[i] * rinv[(i / d) | 0]; }
    if (need(x)) {
      const gx = x.ensureGrad();
      for (let t = 0; t < T; t++) {
        const o = t * d, r = rinv[t];
        let s = 0;
        for (let j = 0; j < d; j++) s += g[o + j] * w.data[j] * xd[o + j];
        for (let j = 0; j < d; j++) gx[o + j] += r * w.data[j] * g[o + j] - xd[o + j] * r * r * r * s / d;
      }
    }
  });
}

const GELU_C = Math.sqrt(2 / Math.PI);

// tanh-approximate GELU, the GPT-2 form (F.gelu(x, approximate="tanh"))
export function geluScalar(v) { return 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v))); }

export function gelu(x) {
  const out = new Float64Array(x.size), th = new Float64Array(x.size);
  for (let i = 0; i < out.length; i++) {
    const v = x.data[i];
    th[i] = Math.tanh(GELU_C * (v + 0.044715 * v * v * v));
    out[i] = 0.5 * v * (1 + th[i]);
  }
  return result(out, x.shape.slice(), [x], (g) => {
    const gx = x.ensureGrad();
    for (let i = 0; i < g.length; i++) {
      const v = x.data[i], t = th[i];
      const d = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * v * v);
      gx[i] += g[i] * d;
    }
  });
}

export function relu(x) {
  const out = new Float64Array(x.size);
  for (let i = 0; i < out.length; i++) out[i] = x.data[i] > 0 ? x.data[i] : 0;
  return result(out, x.shape.slice(), [x], (g) => {
    const gx = x.ensureGrad();
    for (let i = 0; i < g.length; i++) if (x.data[i] > 0) gx[i] += g[i];
  });
}

export function siluScalar(v) { return v / (1 + Math.exp(-v)); }

export function silu(x) {
  const out = new Float64Array(x.size), sg = new Float64Array(x.size);
  for (let i = 0; i < out.length; i++) {
    const v = x.data[i];
    sg[i] = 1 / (1 + Math.exp(-v));
    out[i] = v * sg[i];
  }
  return result(out, x.shape.slice(), [x], (g) => {
    const gx = x.ensureGrad();
    for (let i = 0; i < g.length; i++) gx[i] += g[i] * sg[i] * (1 + x.data[i] * (1 - sg[i]));
  });
}

// Row-wise softmax over the last axis. `mask` (Uint8Array, 1 = keep) zeroes
// disallowed entries exactly like masked_fill(-inf) followed by softmax.
export function softmax(x, mask = null) {
  const [R, C] = x.shape, out = new Float64Array(R * C);
  for (let r = 0; r < R; r++) {
    const o = r * C;
    let mx = -Infinity;
    for (let c = 0; c < C; c++) if (!mask || mask[o + c]) mx = Math.max(mx, x.data[o + c]);
    let s = 0;
    for (let c = 0; c < C; c++) {
      const e = !mask || mask[o + c] ? Math.exp(x.data[o + c] - mx) : 0;
      out[o + c] = e;
      s += e;
    }
    for (let c = 0; c < C; c++) out[o + c] /= s;
  }
  return result(out, [R, C], [x], (g) => {
    const gx = x.ensureGrad();
    for (let r = 0; r < R; r++) {
      const o = r * C;
      let dot = 0;
      for (let c = 0; c < C; c++) dot += out[o + c] * g[o + c];
      for (let c = 0; c < C; c++) gx[o + c] += out[o + c] * (g[o + c] - dot);
    }
  });
}

// Columns [start, start+len) of a [T, C] tensor -> [T, len]
export function sliceCols(x, start, len) {
  const [T, C] = x.shape, out = new Float64Array(T * len);
  for (let t = 0; t < T; t++) out.set(x.data.subarray(t * C + start, t * C + start + len), t * len);
  return result(out, [T, len], [x], (g) => {
    const gx = x.ensureGrad();
    for (let t = 0; t < T; t++) for (let j = 0; j < len; j++) gx[t * C + start + j] += g[t * len + j];
  });
}

// Concatenate [T, c_k] tensors along columns -> [T, sum c_k]
export function concatCols(list) {
  const T = list[0].shape[0], C = list.reduce((s, t) => s + t.shape[1], 0), out = new Float64Array(T * C);
  let off = 0;
  for (const p of list) {
    const c = p.shape[1];
    for (let t = 0; t < T; t++) out.set(p.data.subarray(t * c, (t + 1) * c), t * C + off);
    off += c;
  }
  return result(out, [T, C], list, (g) => {
    let o2 = 0;
    for (const p of list) {
      const c = p.shape[1];
      if (need(p)) {
        const gp = p.ensureGrad();
        for (let t = 0; t < T; t++) for (let j = 0; j < c; j++) gp[t * c + j] += g[t * C + o2 + j];
      }
      o2 += c;
    }
  });
}

// Rotary embedding on interleaved pairs (2k, 2k+1) of a [T, dh] tensor.
// cos, sin: Float64Array of shape [T, dh/2] (position offset already applied).
export function rope(x, cos, sin) {
  const [T, dh] = x.shape, half = dh / 2, out = new Float64Array(T * dh);
  for (let t = 0; t < T; t++) for (let k = 0; k < half; k++) {
    const c = cos[t * half + k], s = sin[t * half + k];
    const x1 = x.data[t * dh + 2 * k], x2 = x.data[t * dh + 2 * k + 1];
    out[t * dh + 2 * k] = x1 * c - x2 * s;
    out[t * dh + 2 * k + 1] = x1 * s + x2 * c;
  }
  return result(out, [T, dh], [x], (g) => {
    const gx = x.ensureGrad();
    for (let t = 0; t < T; t++) for (let k = 0; k < half; k++) {
      const c = cos[t * half + k], s = sin[t * half + k];
      const g1 = g[t * dh + 2 * k], g2 = g[t * dh + 2 * k + 1];
      gx[t * dh + 2 * k] += g1 * c + g2 * s;
      gx[t * dh + 2 * k + 1] += -g1 * s + g2 * c;
    }
  });
}

// Mean cross-entropy over positions whose target is >= 0 (-1 = ignore).
export function crossEntropy(logits, targets) {
  const [T, V] = logits.shape, probs = new Float64Array(T * V);
  let loss = 0, n = 0;
  for (let t = 0; t < T; t++) {
    const o = t * V;
    let mx = -Infinity;
    for (let v = 0; v < V; v++) mx = Math.max(mx, logits.data[o + v]);
    let s = 0;
    for (let v = 0; v < V; v++) { probs[o + v] = Math.exp(logits.data[o + v] - mx); s += probs[o + v]; }
    for (let v = 0; v < V; v++) probs[o + v] /= s;
    if (targets[t] >= 0) { loss += -Math.log(probs[o + targets[t]]); n++; }
  }
  const out = Float64Array.of(n ? loss / n : 0);
  return result(out, [], [logits], (g) => {
    if (!n) return;
    const gl = logits.ensureGrad(), k = g[0] / n;
    for (let t = 0; t < T; t++) {
      if (targets[t] < 0) continue;
      const o = t * V;
      for (let v = 0; v < V; v++) gl[o + v] += k * (probs[o + v] - (v === targets[t] ? 1 : 0));
    }
  });
}

// Stack same-shape tensors along a new leading axis (trace snapshots only, no grad).
export function stack(list) {
  const inner = list[0].shape, n = list[0].size, out = new Float64Array(list.length * n);
  list.forEach((t, i) => out.set(t.data, i * n));
  return new Tensor(out, [list.length, ...inner]);
}

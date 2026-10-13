"""NumPy reference for reference/model.py: the same decoder-only Transformer, with no torch and no autograd.

Every op has an explicit forward, which returns (output, cache), and an explicit backward, which
takes (upstream gradient, cache) and returns the gradients of its inputs. Parameter gradients are
accumulated by hand. tests/test_numpy_reference.py checks the results against PyTorch float64
fixtures and against central differences.

Conventions
  * One sequence at a time. The batch axis of model.py is dropped, as in the fixtures: their ids
    are a flat list and their trace tensors carry no batch axis.
  * Shape letters: T sequence length, d d_model, H n_head, Hkv n_kv_head, dh = d / H head width,
    G = H / Hkv query heads per KV head, F d_ff, V vocab size.
  * nn.Linear layout: weight is (out, in) and y = x @ weight.T + bias.
  * backward() differentiates the mean cross-entropy loss, which is what loss.backward() does
    when the fixtures are exported.
  * NumpyGPT(cfg) mirrors GPT(cfg): forward(params, ids, targets) returns (logits, loss, trace),
    and backward() returns one gradient per parameter for the last forward() call.

Deliberate differences from model.py
  * An unknown option string raises ValueError. model.py would silently fall back to a default.
  * A sequence with no target != -1 raises ValueError. PyTorch returns NaN.
  * generate() is not ported; it is sampling code, not part of the forward/backward contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------------------ config

@dataclass
class Config:
    """Same fields, defaults and meaning as Config in reference/model.py."""

    vocab_size: int = 32
    block_size: int = 32            # longest sequence the model accepts
    n_layer: int = 2
    n_head: int = 4
    n_kv_head: int = 4              # = n_head: MHA; 1: MQA; in between: GQA
    d_model: int = 64
    d_ff: int = 256                 # MLP hidden width (4 * d is the classic ratio)
    pos_encoding: str = "learned"   # learned | sinusoidal | rope | alibi | none
    norm: str = "layernorm"         # layernorm | rmsnorm
    norm_placement: str = "pre"     # pre (GPT-2, LLaMA) | post (2017 original)
    activation: str = "gelu"        # gelu | relu | swiglu
    mask: str = "causal"            # causal | none | sliding | prefix
    window: int = 8                 # sliding: a query sees itself and window-1 earlier keys
    prefix_len: int = 4             # prefix: the first prefix_len tokens see each other
    tie_weights: bool = False       # lm_head reuses the token embedding matrix
    bias: bool = True               # biases in Linear layers and LayerNorm
    residual: bool = True           # ablation: turn skip connections off
    attn_scale: bool = True         # ablation: drop the 1/sqrt(dh) factor
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


_CHOICES = {
    "pos_encoding": ("learned", "sinusoidal", "rope", "alibi", "none"),
    "norm": ("layernorm", "rmsnorm"),
    "norm_placement": ("pre", "post"),
    "activation": ("gelu", "relu", "swiglu"),
    "mask": ("causal", "none", "sliding", "prefix"),
}


def _validate_config(cfg: Config) -> None:
    for field, allowed in _CHOICES.items():
        value = getattr(cfg, field)
        if value not in allowed:
            raise ValueError(f"{field}={value!r}: expected one of {allowed}")
    if cfg.d_model % cfg.n_head != 0:
        raise ValueError("d_model must split evenly across heads")
    if cfg.n_head % cfg.n_kv_head != 0:
        raise ValueError("n_head must be a multiple of n_kv_head")
    if cfg.pos_encoding == "rope" and cfg.d_head % 2 != 0:
        raise ValueError("RoPE rotates interleaved pairs, so the head width must be even")
    if cfg.pos_encoding == "sinusoidal" and cfg.d_model % 2 != 0:
        raise ValueError("the sinusoidal table needs an even d_model")


def param_shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    """Every parameter that model.py's GPT(cfg) registers, by name, with its shape."""
    d, H, Hkv, dh, F, V = cfg.d_model, cfg.n_head, cfg.n_kv_head, cfg.d_head, cfg.d_ff, cfg.vocab_size
    shapes: dict[str, tuple[int, ...]] = {"wte.weight": (V, d)}
    if cfg.pos_encoding == "learned":
        shapes["wpe.weight"] = (cfg.block_size, d)

    def norm(prefix: str) -> None:
        shapes[f"{prefix}.weight"] = (d,)
        if cfg.norm == "layernorm" and cfg.bias:       # RMSNorm has no beta
            shapes[f"{prefix}.bias"] = (d,)

    def linear(prefix: str, n_out: int, n_in: int, bias: bool) -> None:
        shapes[f"{prefix}.weight"] = (n_out, n_in)     # nn.Linear layout: (out, in)
        if bias:
            shapes[f"{prefix}.bias"] = (n_out,)

    for i in range(cfg.n_layer):
        b = f"blocks.{i}"
        norm(f"{b}.ln1")
        linear(f"{b}.attn.q_proj", H * dh, d, cfg.bias)
        linear(f"{b}.attn.k_proj", Hkv * dh, d, cfg.bias)
        linear(f"{b}.attn.v_proj", Hkv * dh, d, cfg.bias)
        linear(f"{b}.attn.o_proj", d, H * dh, cfg.bias)
        norm(f"{b}.ln2")
        linear(f"{b}.mlp.up_proj", F, d, cfg.bias)
        if cfg.activation == "swiglu":
            linear(f"{b}.mlp.gate_proj", F, d, cfg.bias)
        linear(f"{b}.mlp.down_proj", d, F, cfg.bias)
    if cfg.norm_placement == "pre":
        norm("ln_f")
    if not cfg.tie_weights:
        shapes["lm_head.weight"] = (V, d)              # nn.Linear(..., bias=False)
    return shapes


# -------------------------------------------------------------- linear and embedding

def linear_forward(x, weight, bias):
    """y = x @ W.T + b.   x: (T, in)   W: (out, in)   b: (out,) or None   ->   y: (T, out)"""
    y = x @ weight.T                                    # (T, out)
    if bias is not None:
        y = y + bias                                    # (out,) broadcasts over the T rows
    return y, (x, weight)


def linear_backward(dy, cache):
    """dy: (T, out) -> dx: (T, in), dW: (out, in), db: (out,).

    y[t, o] = sum_i x[t, i] W[o, i] + b[o]. Each input feeds every output, so its gradient sums
    over o. Each weight W[o, i] multiplies x[t, i] in every row t, so its gradient sums over t:
        dx[t, i] = sum_o dy[t, o] W[o, i]      ->  dx = dy @ W
        dW[o, i] = sum_t dy[t, o] x[t, i]      ->  dW = dy.T @ x
        db[o]    = sum_t dy[t, o]              ->  db = dy.sum(axis=0)
    """
    x, weight = cache
    dx = dy @ weight                                    # (T, in)
    dweight = dy.T @ x                                  # (out, in)
    dbias = dy.sum(axis=0)                              # (out,)
    return dx, dweight, dbias


def embedding_forward(ids, table):
    """Row lookup, i.e. one_hot(ids) @ table without building the one-hot matrix.
    ids: (T,) int64   table: (V, d)   ->   (T, d)"""
    return table[ids], ids                              # table[ids]: (T,) -> (T, d); cache is ids


def embedding_backward(dout, ids, table_shape):
    """dout: (T, d) -> dtable: (V, d).

    A token can occur several times in one sequence, and every occurrence adds its gradient to the
    same row. `dtable[ids] += dout` would keep only one of them, because repeated indices in a
    fancy-index assignment are not accumulated. np.add.at does accumulate them.
    """
    dtable = np.zeros(table_shape)                      # (V, d)
    np.add.at(dtable, ids, dout)                        # dtable[ids[t]] += dout[t] for every t
    return dtable


# --------------------------------------------------------------------------- norms

def layernorm_forward(x, gamma, beta, eps):
    """Per token, normalise over the d features with the biased variance (divide by d).
    x: (T, d)   gamma: (d,)   beta: (d,) or None   ->   y: (T, d)"""
    mu = x.mean(axis=-1, keepdims=True)                   # (T, 1)
    centered = x - mu                                     # (T, d)
    var = np.mean(centered ** 2, axis=-1, keepdims=True)  # (T, 1)  biased: divide by d, not d - 1
    inv_std = 1.0 / np.sqrt(var + eps)                    # (T, 1)
    xhat = centered * inv_std                             # (T, d)  zero mean; variance = var / (var + eps)
    y = xhat * gamma                                      # (T, d)
    if beta is not None:
        y = y + beta
    return y, (xhat, inv_std, gamma)


def layernorm_backward(dy, cache):
    """dy: (T, d) -> dx: (T, d), dgamma: (d,), dbeta: (d,).

    Derivation for one token, with s = sqrt(var + eps) and xhat_j = (x_j - mu) / s:
        d mu / d x_k      = 1 / d
        d var / d x_k     = (2 / d) (x_k - mu)          (the sum of (x_j - mu) is 0, so the
                                                         1/d term from mu drops out)
        d s / d x_k       = (x_k - mu) / (d s)
        d xhat_j / d x_k  = (delta_jk - 1/d - xhat_j xhat_k / d) / s
    Contracting with dxhat_j = dy_j gamma_j gives
        dx = (1/s) (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)).
    The common-shift direction is removed exactly. With eps > 0, the radial direction
    is attenuated by eps / (var + eps) rather than removed exactly.
    """
    xhat, inv_std, gamma = cache
    dgamma = (dy * xhat).sum(axis=0)                      # (d,)  summed over the T tokens
    dbeta = dy.sum(axis=0)                                # (d,)
    dxhat = dy * gamma                                    # (T, d)
    dx = inv_std * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                    - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))   # (T, d)
    return dx, dgamma, dbeta


def rmsnorm_forward(x, gamma, eps):
    """y = x / rms(x) * gamma, with rms = sqrt(mean(x^2) + eps). No mean subtraction, no bias.
    x: (T, d) -> y: (T, d)"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)   # (T, 1)
    u = x / rms                                            # (T, d)  mean square = mean(x^2) / (mean(x^2) + eps)
    y = u * gamma                                          # (T, d)
    return y, (u, rms, gamma)


def rmsnorm_backward(dy, cache):
    """dy: (T, d) -> dx: (T, d), dgamma: (d,).

    With r = sqrt(mean(x^2) + eps), dr/dx_k = x_k / (d r). Then
        y_j = gamma_j x_j / r   ->   dy_j / dx_k = gamma_j (delta_jk / r - x_j x_k / (d r^3)).
    Summing dy_j times that over j and writing u = x / r gives
        dx = (gamma * dy - u * mean(gamma * dy * u)) / r.
    """
    u, rms, gamma = cache
    dgamma = (dy * u).sum(axis=0)                          # (d,)
    dyg = dy * gamma                                       # (T, d)
    dx = (dyg - u * np.mean(dyg * u, axis=-1, keepdims=True)) / rms   # (T, d)
    return dx, dgamma


# --------------------------------------------------------------------- activations

_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_K = 0.044715


def gelu_forward(x):
    """Tanh-approximated GELU, the formula of F.gelu(approximate='tanh'). x: (T, F) -> (T, F)"""
    t = np.tanh(_GELU_C * (x + _GELU_K * x ** 3))          # (T, F)
    y = 0.5 * x * (1.0 + t)                                # (T, F)
    return y, (x, t)


def gelu_backward(dy, cache):
    """y = 0.5 x (1 + t) with t = tanh(u), u = c (x + k x^3), so du/dx = c (1 + 3 k x^2).
    Product rule:  dy/dx = 0.5 (1 + t) + 0.5 x (1 - t^2) c (1 + 3 k x^2)."""
    x, t = cache
    local = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * _GELU_C * (1.0 + 3.0 * _GELU_K * x ** 2)
    return dy * local


def relu_forward(x):
    return np.maximum(x, 0.0), x


def relu_backward(dy, x):
    """d/dx max(x, 0) is 1 for x > 0 and 0 otherwise, the same convention as PyTorch at x = 0."""
    return dy * (x > 0)


def sigmoid(x):
    """Logistic function that only exponentiates non-positive numbers, so it cannot overflow."""
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def silu_forward(x):
    """silu(x) = x * sigmoid(x)."""
    s = sigmoid(x)
    return x * s, (x, s)


def silu_backward(dy, cache):
    """s = sigmoid(x) and s' = s (1 - s). Product rule on x s:
        d/dx [x s] = s + x s (1 - s) = s (1 + x (1 - s))."""
    x, s = cache
    return dy * s * (1.0 + x * (1.0 - s))


def swiglu_forward(gate, up):
    """act = silu(gate) * up.   gate, up: (T, F) -> act: (T, F)"""
    a, silu_cache = silu_forward(gate)                     # (T, F)
    return a * up, (a, up, silu_cache)                     # (T, F)


def swiglu_backward(dact, cache):
    """act = a * up with a = silu(gate):
        d act / d up   = a                  so  dup   = dact * a
        d act / d gate = up * silu'(gate)   so  dgate = silu_backward(dact * up)
    """
    a, up, silu_cache = cache
    dup = dact * a                                         # (T, F)
    dgate = silu_backward(dact * up, silu_cache)           # (T, F)
    return dgate, dup


# ----------------------------------------------------------------- softmax and loss

def softmax_forward(z):
    """Row-wise softmax over the last axis. Subtracting the row maximum changes nothing and avoids
    overflow. A masked logit of -inf gives exp(-inf) = 0. Every row needs one finite logit."""
    shifted = z - z.max(axis=-1, keepdims=True)            # (..., N)
    e = np.exp(shifted)                                    # (..., N)
    p = e / e.sum(axis=-1, keepdims=True)                  # (..., N)  rows sum to 1
    return p, p


def softmax_backward(dprobs, p):
    """Jacobian-vector product for p = softmax(z) over the last axis. Returns dL/dz.

    dprobs is the upstream gradient g = dL/dp. The softmax Jacobian has entries
    dp_i / dz_j = p_i (delta_ij - p_j), so by the chain rule
        dL/dz_j = sum_i g_i (dp_i / dz_j) = sum_i g_i p_i (delta_ij - p_j) = p_j g_j - p_j sum_i g_i p_i,
    that is, dL/dz = p * (g - sum(g * p)) over the last axis. The N x N Jacobian is never formed.
    Masked entries have p = 0, so they receive zero gradient without any special case.
    """
    return p * (dprobs - np.sum(dprobs * p, axis=-1, keepdims=True))     # (..., N)


def log_softmax(z):
    """log p = z - logsumexp(z), computed from shifted logits for stability."""
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def cross_entropy_forward(logits, targets):
    """Mean negative log-likelihood over the positions whose target is not -1 (ignore_index).

    logits: (T, V)   targets: (T,) integers in [-1, V)   ->   loss (scalar), cache
    """
    valid = targets != -1                                  # (T,) bool
    n_valid = int(valid.sum())                             # scalar: number of counted targets
    if n_valid == 0:
        raise ValueError("cross-entropy needs at least one target != -1")
    safe = np.where(valid, targets, 0)                     # (T,)  placeholder index for ignored rows
    log_p = log_softmax(logits)                            # (T, V)
    nll = -log_p[np.arange(len(targets)), safe]            # (T,)
    loss = np.sum(nll * valid) / n_valid                   # scalar: sum over valid rows / count
    return loss, (np.exp(log_p), safe, valid, n_valid)


def cross_entropy_backward(cache):
    """loss = (1/N) sum over valid t of -log p[t, y_t], with p = softmax(logits).

    Applying the softmax Jacobian to -1/p_y gives d(-log p_y)/dz_k = p_k - delta_ky. Hence
        dlogits[t] = (p[t] - onehot(y_t)) / N   for valid t, and 0 for ignored t.
    """
    probs, safe, valid, n_valid = cache
    dlogits = probs.copy()                                 # (T, V)
    dlogits[np.arange(len(safe)), safe] -= 1.0             # p - onehot(y)
    return dlogits * (valid[:, None] / n_valid)            # (T, V), ignored rows zeroed


# ----------------------------------------------------------- positions and masks

def sinusoidal_table(T, d):
    """(T, d) with PE[pos, 2k] = sin(pos f_k) and PE[pos, 2k+1] = cos(pos f_k), f_k = 10000^(-2k/d).
    It is a constant with no parameters, so it takes no gradient."""
    pos = np.arange(T)[:, None]                            # (T, 1)
    even_dims = np.arange(0, d, 2)                         # (d/2,)  the values 2k
    freq = np.exp(-math.log(10000.0) * even_dims / d)      # (d/2,)  = 10000^(-2k/d)
    pe = np.zeros((T, d))                                  # (T, d)
    pe[:, 0::2] = np.sin(pos * freq)                       # (T, d/2)
    pe[:, 1::2] = np.cos(pos * freq)                       # (T, d/2)
    return pe


def rope_tables(T, dh, base):
    """cos and sin of angle[t, k] = t * theta_k, with theta_k = base^(-2k/dh) for k = 0 .. dh/2 - 1.
    Each returned array has shape (T, dh/2)."""
    even_dims = np.arange(0, dh, 2)                        # (dh/2,)  the values 2k
    theta = base ** (-even_dims / dh)                      # (dh/2,)  rotation speed of each pair
    angle = np.arange(T)[:, None] * theta[None, :]         # (T, dh/2)
    return np.cos(angle), np.sin(angle)


def rope_forward(x, cos, sin):
    """Rotate each interleaved pair (x[2k], x[2k+1]) by its own angle.

    x: (..., T, dh)   cos, sin: (T, dh/2), broadcast over the leading head axis
        y1 = x1 cos - x2 sin,    y2 = x1 sin + x2 cos,    where x1 = x[..., 0::2], x2 = x[..., 1::2]
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]                    # (..., T, dh/2)
    y1 = x1 * cos - x2 * sin                               # (..., T, dh/2)
    y2 = x1 * sin + x2 * cos                               # (..., T, dh/2)
    return np.stack((y1, y2), axis=-1).reshape(x.shape)    # (..., T, dh/2, 2) -> (..., T, dh)


def rope_backward(dy, cos, sin):
    """Each pair is multiplied by a 2x2 rotation R(angle). Rotations are orthogonal, so the input
    gradient is R(angle)^T dy = R(-angle) dy:
        dx1 = dy1 cos + dy2 sin,      dx2 = dy2 cos - dy1 sin.
    dy: (..., T, dh) -> dx: (..., T, dh)
    """
    dy1, dy2 = dy[..., 0::2], dy[..., 1::2]               # (..., T, dh/2)
    dx = np.empty_like(dy)                                 # (..., T, dh)
    dx[..., 0::2] = dy1 * cos + dy2 * sin                  # (..., T, dh/2)
    dx[..., 1::2] = dy2 * cos - dy1 * sin                  # (..., T, dh/2)
    return dx


def alibi_bias(H, T):
    """(H, T, T) with bias[h, i, j] = -m_h |i - j| and slope m_h = 2^(-8 (h+1) / H). A constant."""
    slopes = np.array([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)])     # (H,)
    dist = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])          # (T, T)
    return -slopes[:, None, None] * dist[None, :, :]                       # (H, T, T)


def build_mask(cfg, T):
    """(T, T) bool, True where query i may attend to key j."""
    i = np.arange(T)[:, None]                              # (T, 1)
    j = np.arange(T)[None, :]                              # (1, T)
    if cfg.mask == "none":
        return np.ones((T, T), dtype=bool)                 # (T, T)  bidirectional
    if cfg.mask == "sliding":
        return (j <= i) & (j > i - cfg.window)             # (T, T)  causal, last `window` keys
    if cfg.mask == "prefix":
        return (j <= i) | (j < cfg.prefix_len)             # (T, T)  prefix-LM: prefix fully visible
    return j <= i                                          # (T, T)  causal


# ------------------------------------------------------------------- GQA head sharing

def repeat_kv_forward(kv, group):
    """(Hkv, T, dh) -> (H, T, dh) with H = Hkv * group. Query head h reads KV head h // group,
    which is np.repeat along the head axis (the same as torch's repeat_interleave)."""
    return np.repeat(kv, group, axis=0)                 # (Hkv, T, dh) -> (H, T, dh)


def repeat_kv_backward(drep, group):
    """(H, T, dh) -> (Hkv, T, dh). Each KV head was copied into `group` query heads, and every copy
    depends on the same weights, so the gradient of the KV head is the sum of its copies.
    Head h = kv * group + g, so reshaping the head axis to (Hkv, group) puts each KV head's copies
    together, and summing over the group axis adds them up."""
    H, T, dh = drep.shape
    return drep.reshape(H // group, group, T, dh).sum(axis=1)           # (H, T, dh) -> (Hkv, T, dh)


# ------------------------------------------------------------------- param plumbing

def _linear_wb(params, prefix):
    """(weight, bias or None) of the nn.Linear named `prefix`."""
    return params[f"{prefix}.weight"], params.get(f"{prefix}.bias")


def _accumulate_linear(grads, prefix, dweight, dbias):
    grads[f"{prefix}.weight"] += dweight
    if f"{prefix}.bias" in grads:                          # the layer only has a bias if it exists
        grads[f"{prefix}.bias"] += dbias


def _norm_forward(cfg, params, prefix, x):
    if cfg.norm == "layernorm":
        return layernorm_forward(x, params[f"{prefix}.weight"], params.get(f"{prefix}.bias"), cfg.norm_eps)
    return rmsnorm_forward(x, params[f"{prefix}.weight"], cfg.norm_eps)


def _norm_backward(cfg, prefix, dy, cache, grads):
    if cfg.norm == "layernorm":
        dx, dgamma, dbeta = layernorm_backward(dy, cache)
        grads[f"{prefix}.weight"] += dgamma
        if f"{prefix}.bias" in grads:
            grads[f"{prefix}.bias"] += dbeta
    else:
        dx, dgamma = rmsnorm_backward(dy, cache)
        grads[f"{prefix}.weight"] += dgamma
    return dx


def _check_params(cfg, params):
    expected = param_shapes(cfg)
    missing = sorted(set(expected) - set(params))
    unexpected = sorted(set(params) - set(expected))
    if missing or unexpected:
        raise KeyError(f"parameters do not match the config: missing {missing}, unexpected {unexpected}")
    for name, shape in expected.items():
        if np.shape(params[name]) != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {np.shape(params[name])}")


def _as_sequence(values, name, low, high):
    """Integers shaped (T,) or (1, T), each in [low, high). A batch axis of size 1 is dropped."""
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]                                       # (1, T) -> (T,)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be integers shaped (T,) or (1, T); got {arr.dtype} {arr.shape}")
    if arr.size and (arr.min() < low or arr.max() >= high):
        raise ValueError(f"{name} must lie in [{low}, {high})")
    return arr.astype(np.int64)


# --------------------------------------------------------------------------- model

class NumpyGPT:
    """NumPy twin of model.py's GPT.

    As with GPT(cfg), the config is fixed at construction, because it decides which ops run.
    Parameters are passed to forward(params, ids, targets), so the same object can be re-run at
    perturbed parameters for numerical checks.
    """

    def __init__(self, cfg: Config):
        _validate_config(cfg)
        self.cfg = cfg
        self._tape = None                                  # what forward() saved for backward()

    # ----------------------------------------------------------------- forward

    def forward(self, params, ids, targets=None):
        """Run the model on one sequence.

        params:  dict name -> float64 array, with the names and shapes of param_shapes(cfg)
        ids:     (T,) or (1, T) token ids
        targets: same length as ids, -1 where there is no target
        returns: logits (T, V), loss (np.float64, or None when targets is None), trace dict

        The trace has exactly the keys that model.py's @trace markers record for this config,
        with the batch axis dropped.
        """
        cfg = self.cfg
        _check_params(cfg, params)
        ids = _as_sequence(ids, "ids", 0, cfg.vocab_size)                 # (T,)
        T = ids.shape[0]
        if T > cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {cfg.block_size}")
        trace = {}

        # Embeddings: x = tok (+ pos). RoPE and ALiBi act inside attention instead.
        tok, _ = embedding_forward(ids, params["wte.weight"])             # (T, d)
        trace["emb.tok"] = tok
        if cfg.pos_encoding == "learned":
            pos = params["wpe.weight"][:T]                                # (T, d)  wpe(arange(T))
        elif cfg.pos_encoding == "sinusoidal":
            pos = sinusoidal_table(T, cfg.d_model)                        # (T, d)
        else:
            pos = None
        if pos is not None:
            trace["emb.pos"] = pos
            x = tok + pos                                                 # (T, d)
        else:
            x = tok                                                       # (T, d)
        trace["emb.out"] = x

        # Mask and position signals shared by every layer.
        mask = build_mask(cfg, T)                                         # (T, T) bool
        trace["mask"] = mask
        rope = rope_tables(T, cfg.d_head, cfg.rope_base) if cfg.pos_encoding == "rope" else None
        alibi = None
        if cfg.pos_encoding == "alibi":
            alibi = alibi_bias(cfg.n_head, T)                             # (H, T, T)
            trace["alibi"] = alibi

        block_caches = []
        for i in range(cfg.n_layer):
            x, block_cache = self._block_forward(params, i, x, mask, rope, alibi, trace)
            block_caches.append(block_cache)

        lnf_cache = None
        if cfg.norm_placement == "pre":
            x, lnf_cache = _norm_forward(cfg, params, "ln_f", x)          # (T, d)
            trace["lnf"] = x
        head_name = "wte.weight" if cfg.tie_weights else "lm_head.weight"
        logits, head_cache = linear_forward(x, params[head_name], None)   # (T, V)
        trace["logits"] = logits
        probs, _ = softmax_forward(logits)                                # (T, V)
        trace["probs"] = probs

        loss, ce_cache = None, None
        if targets is not None:
            targets = _as_sequence(targets, "targets", -1, cfg.vocab_size)  # (T,)
            if targets.shape != ids.shape:
                raise ValueError("targets must have the same length as ids")
            loss, ce_cache = cross_entropy_forward(logits, targets)
            trace["loss"] = np.float64(loss)

        self._tape = dict(params=params, ids=ids, rope=rope, mask=mask, block_caches=block_caches,
                          lnf_cache=lnf_cache, head_name=head_name, head_cache=head_cache,
                          ce_cache=ce_cache)
        return logits, loss, trace

    def _block_forward(self, params, i, x, mask, rope, alibi, trace):
        """One transformer block. Returns (x_out (T, d), cache).

        pre-norm:   x_mid = x + A(LN1(x)),       x_out = x_mid + M(LN2(x_mid))
        post-norm:  x_mid = LN1(x + A(x)),       x_out = LN2(x_mid + M(x_mid))
        where A is attention and M is the MLP. With residual=False, the "x +" terms are dropped.
        """
        cfg = self.cfg
        if cfg.norm_placement == "pre":
            h1, c_ln1 = _norm_forward(cfg, params, f"blocks.{i}.ln1", x)          # (T, d)
            trace[f"L{i}.ln1"] = h1
            a, c_attn = self._attention_forward(params, i, h1, mask, rope, alibi, trace)
            x_mid = x + a if cfg.residual else a                                   # (T, d)
            trace[f"L{i}.resid_mid"] = x_mid
            h2, c_ln2 = _norm_forward(cfg, params, f"blocks.{i}.ln2", x_mid)      # (T, d)
            trace[f"L{i}.ln2"] = h2
            m, c_mlp = self._mlp_forward(params, i, h2, trace)                     # (T, d)
            x_out = x_mid + m if cfg.residual else m                               # (T, d)
            trace[f"L{i}.resid_out"] = x_out
            cache = dict(placement="pre", ln1=c_ln1, attn=c_attn, ln2=c_ln2, mlp=c_mlp)
        else:
            a, c_attn = self._attention_forward(params, i, x, mask, rope, alibi, trace)
            s1 = x + a if cfg.residual else a                                      # (T, d)
            trace[f"L{i}.add1"] = s1
            x_mid, c_ln1 = _norm_forward(cfg, params, f"blocks.{i}.ln1", s1)      # (T, d)
            trace[f"L{i}.resid_mid"] = x_mid
            m, c_mlp = self._mlp_forward(params, i, x_mid, trace)                  # (T, d)
            s2 = x_mid + m if cfg.residual else m                                  # (T, d)
            trace[f"L{i}.add2"] = s2
            x_out, c_ln2 = _norm_forward(cfg, params, f"blocks.{i}.ln2", s2)      # (T, d)
            trace[f"L{i}.resid_out"] = x_out
            cache = dict(placement="post", attn=c_attn, ln1=c_ln1, mlp=c_mlp, ln2=c_ln2)
        return x_out, cache

    def _attention_forward(self, params, i, h, mask, rope, alibi, trace):
        """Multi-head self-attention on h: (T, d). Returns (out: (T, d), cache)."""
        cfg = self.cfg
        T = h.shape[0]
        H, Hkv, dh = cfg.n_head, cfg.n_kv_head, cfg.d_head
        group = H // Hkv                                                       # query heads per KV head
        name = f"blocks.{i}.attn"

        # Project, then split the feature axis into heads: (T, heads * dh) -> (heads, T, dh).
        q2, c_q = linear_forward(h, *_linear_wb(params, f"{name}.q_proj"))     # (T, H*dh)
        k2, c_k = linear_forward(h, *_linear_wb(params, f"{name}.k_proj"))     # (T, Hkv*dh)
        v2, c_v = linear_forward(h, *_linear_wb(params, f"{name}.v_proj"))     # (T, Hkv*dh)
        trace[f"L{i}.attn.q"] = q2
        trace[f"L{i}.attn.k"] = k2
        trace[f"L{i}.attn.v"] = v2
        q = q2.reshape(T, H, dh).transpose(1, 0, 2)                            # (H, T, dh)
        k = k2.reshape(T, Hkv, dh).transpose(1, 0, 2)                          # (Hkv, T, dh)
        v = v2.reshape(T, Hkv, dh).transpose(1, 0, 2)                          # (Hkv, T, dh)
        trace[f"L{i}.attn.q_heads"] = q
        trace[f"L{i}.attn.k_heads"] = k
        trace[f"L{i}.attn.v_heads"] = v

        if rope is not None:                                                   # rotate by position
            cos, sin = rope
            q = rope_forward(q, cos, sin)                                      # (H, T, dh)
            k = rope_forward(k, cos, sin)                                      # (Hkv, T, dh)
            trace[f"L{i}.attn.q_rot"] = q
            trace[f"L{i}.attn.k_rot"] = k

        if Hkv < H:                                                            # GQA / MQA sharing
            k = repeat_kv_forward(k, group)                                    # (H, T, dh)
            v = repeat_kv_forward(v, group)                                    # (H, T, dh)
            trace[f"L{i}.attn.k_rep"] = k
            trace[f"L{i}.attn.v_rep"] = v

        scores = q @ k.transpose(0, 2, 1)                                      # (H, T, T)
        trace[f"L{i}.attn.scores"] = scores
        scaled = scores
        if cfg.attn_scale:
            scaled = scores / math.sqrt(dh)                                    # (H, T, T)
            trace[f"L{i}.attn.scaled"] = scaled
        biased = scaled
        if alibi is not None:
            biased = scaled + alibi                                            # (H, T, T)
            trace[f"L{i}.attn.biased"] = biased
        masked = np.where(mask, biased, -np.inf)                               # (H, T, T)
        trace[f"L{i}.attn.masked"] = masked
        probs, _ = softmax_forward(masked)                                     # (H, T, T)
        trace[f"L{i}.attn.probs"] = probs
        ctx = probs @ v                                                        # (H, T, dh)
        trace[f"L{i}.attn.ctx"] = ctx
        concat = ctx.transpose(1, 0, 2).reshape(T, H * dh)                     # (T, H*dh)
        trace[f"L{i}.attn.concat"] = concat
        out, c_o = linear_forward(concat, *_linear_wb(params, f"{name}.o_proj"))  # (T, d)
        trace[f"L{i}.attn.out"] = out

        cache = dict(q=c_q, k=c_k, v=c_v, o=c_o, mask=mask, rope=rope,
                     q_rot=q, k_rep=k, v_rep=v, probs=probs)
        return out, cache

    def _mlp_forward(self, params, i, h, trace):
        """MLP on h: (T, d). Returns (out: (T, d), cache)."""
        cfg = self.cfg
        name = f"blocks.{i}.mlp"
        up, c_up = linear_forward(h, *_linear_wb(params, f"{name}.up_proj"))   # (T, F)
        trace[f"L{i}.mlp.up"] = up
        c_gate = None
        if cfg.activation == "swiglu":
            gate, c_gate = linear_forward(h, *_linear_wb(params, f"{name}.gate_proj"))  # (T, F)
            trace[f"L{i}.mlp.gate"] = gate
            act, act_cache = swiglu_forward(gate, up)                          # (T, F)
        elif cfg.activation == "gelu":
            act, act_cache = gelu_forward(up)                                  # (T, F)
        else:
            act, act_cache = relu_forward(up)                                  # (T, F)
        trace[f"L{i}.mlp.act"] = act
        out, c_down = linear_forward(act, *_linear_wb(params, f"{name}.down_proj"))  # (T, d)
        trace[f"L{i}.mlp.out"] = out
        return out, dict(up=c_up, gate=c_gate, act=act_cache, down=c_down)

    # ---------------------------------------------------------------- backward

    def backward(self):
        """Gradients of the mean cross-entropy loss of the last forward() call, one per parameter.

        The chain rule runs in reverse: unembedding -> final norm -> blocks (last to first) ->
        embeddings. A shared weight receives the sum of its contributions. For tied weights,
        wte gets both the unembedding gradient and the embedding-lookup gradient.
        """
        tape = self._tape
        if tape is None or tape["ce_cache"] is None:
            raise RuntimeError("backward() needs the last forward() call to have received targets")
        cfg, params, ids = self.cfg, tape["params"], tape["ids"]
        grads = {name: np.zeros_like(value) for name, value in params.items()}

        dlogits = cross_entropy_backward(tape["ce_cache"])                    # (T, V)
        dx, dhead, _ = linear_backward(dlogits, tape["head_cache"])           # (T, d), (V, d)
        grads[tape["head_name"]] += dhead
        if tape["lnf_cache"] is not None:
            dx = _norm_backward(cfg, "ln_f", dx, tape["lnf_cache"], grads)    # (T, d)

        for i in reversed(range(cfg.n_layer)):
            dx = self._block_backward(i, dx, tape["block_caches"][i], grads)  # (T, d)

        # x = tok (+ pos): the sum passes dx unchanged to each summand.
        grads["wte.weight"] += embedding_backward(dx, ids, params["wte.weight"].shape)   # (V, d)
        if cfg.pos_encoding == "learned":
            grads["wpe.weight"][: len(ids)] += dx                             # (T, d) into rows 0..T-1
        return grads

    def _block_backward(self, i, dx_out, cache, grads):
        """Gradient w.r.t. the block input (T, d), given the gradient w.r.t. its output.

        r is 1 with residuals and 0 without, so each "+ r * ..." term is a skip connection.
        """
        cfg = self.cfg
        r = 1.0 if cfg.residual else 0.0
        if cache["placement"] == "pre":
            # x_out = x_mid + M(h2) and h2 = LN2(x_mid), so the MLP path and skip path both reach x_mid.
            d_h2 = self._mlp_backward(i, dx_out, cache["mlp"], grads)                       # (T, d)
            d_xmid = _norm_backward(cfg, f"blocks.{i}.ln2", d_h2, cache["ln2"], grads)      # (T, d)
            d_xmid = d_xmid + r * dx_out
            # x_mid = x + A(h1) and h1 = LN1(x)
            d_h1 = self._attention_backward(i, d_xmid, cache["attn"], grads)                # (T, d)
            d_x = _norm_backward(cfg, f"blocks.{i}.ln1", d_h1, cache["ln1"], grads)         # (T, d)
            return d_x + r * d_xmid
        # post-norm: x_out = LN2(s2), s2 = x_mid + M(x_mid), x_mid = LN1(s1), s1 = x + A(x)
        d_s2 = _norm_backward(cfg, f"blocks.{i}.ln2", dx_out, cache["ln2"], grads)          # (T, d)
        d_xmid = self._mlp_backward(i, d_s2, cache["mlp"], grads) + r * d_s2               # (T, d)
        d_s1 = _norm_backward(cfg, f"blocks.{i}.ln1", d_xmid, cache["ln1"], grads)          # (T, d)
        return self._attention_backward(i, d_s1, cache["attn"], grads) + r * d_s1           # (T, d)

    def _attention_backward(self, i, dout, cache, grads):
        """dout: (T, d) -> gradient w.r.t. the attention input h: (T, d).

        Forward, in order:  Q, K, V projections -> split heads -> RoPE -> repeat KV heads
                            -> scores = Q K^T -> scale -> + ALiBi -> mask -> softmax
                            -> ctx = probs V -> merge heads -> output projection.
        Backward runs the same steps in reverse. For the softmax step, see softmax_backward.

        Scores: S = Q K^T.  Taking dS as given,
            dQ = dS K   and   dK = dS^T Q        (per head, because S[i, j] = sum_d Q[i, d] K[j, d]).
        Context: C = P V.  Taking dC as given,
            dP = dC V^T   and   dV = P^T dC.
        Masked entries have P = 0, so dS is 0 there. ALiBi is a constant and adds nothing to the
        gradient. The scale 1/sqrt(dh) divides the gradient on its way back.
        """
        cfg = self.cfg
        name = f"blocks.{i}.attn"
        T = dout.shape[0]
        H, Hkv, dh = cfg.n_head, cfg.n_kv_head, cfg.d_head
        probs, v_rep, k_rep, q_rot = cache["probs"], cache["v_rep"], cache["k_rep"], cache["q_rot"]

        d_concat, dW, db = linear_backward(dout, cache["o"])                  # (T, H*dh)
        _accumulate_linear(grads, f"{name}.o_proj", dW, db)
        d_ctx = d_concat.reshape(T, H, dh).transpose(1, 0, 2)                 # (H, T, dh)

        d_probs = d_ctx @ v_rep.transpose(0, 2, 1)                            # (H, T, T)
        d_v_rep = probs.transpose(0, 2, 1) @ d_ctx                            # (H, T, dh)

        d_masked = softmax_backward(d_probs, probs)                           # (H, T, T)
        d_biased = np.where(cache["mask"], d_masked, 0.0)                     # (H, T, T)  masked_fill
        d_scores = d_biased / math.sqrt(dh) if cfg.attn_scale else d_biased   # (H, T, T)

        d_q_rot = d_scores @ k_rep                                            # (H, T, dh)
        d_k_rep = d_scores.transpose(0, 2, 1) @ q_rot                         # (H, T, dh)

        # Undo the forward steps in reverse: first the KV repeat, then the RoPE rotation.
        d_v, d_k_rot = d_v_rep, d_k_rep
        if Hkv < H:
            group = H // Hkv
            d_v = repeat_kv_backward(d_v_rep, group)                          # (Hkv, T, dh)
            d_k_rot = repeat_kv_backward(d_k_rep, group)                      # (Hkv, T, dh)
        d_q, d_k = d_q_rot, d_k_rot
        if cache["rope"] is not None:
            cos, sin = cache["rope"]
            d_q = rope_backward(d_q_rot, cos, sin)                            # (H, T, dh)
            d_k = rope_backward(d_k_rot, cos, sin)                            # (Hkv, T, dh)

        # Back to the (T, heads * dh) layout of the projection outputs.
        d_q2 = d_q.transpose(1, 0, 2).reshape(T, H * dh)                      # (T, H*dh)
        d_k2 = d_k.transpose(1, 0, 2).reshape(T, Hkv * dh)                    # (T, Hkv*dh)
        d_v2 = d_v.transpose(1, 0, 2).reshape(T, Hkv * dh)                    # (T, Hkv*dh)

        d_in_q, dW, db = linear_backward(d_q2, cache["q"])                    # (T, d)
        _accumulate_linear(grads, f"{name}.q_proj", dW, db)
        d_in_k, dW, db = linear_backward(d_k2, cache["k"])                    # (T, d)
        _accumulate_linear(grads, f"{name}.k_proj", dW, db)
        d_in_v, dW, db = linear_backward(d_v2, cache["v"])                    # (T, d)
        _accumulate_linear(grads, f"{name}.v_proj", dW, db)
        return d_in_q + d_in_k + d_in_v                                       # h feeds Q, K and V

    def _mlp_backward(self, i, dout, cache, grads):
        """dout: (T, d) -> gradient w.r.t. the MLP input h: (T, d)."""
        cfg = self.cfg
        name = f"blocks.{i}.mlp"
        d_act, dW, db = linear_backward(dout, cache["down"])                  # (T, F)
        _accumulate_linear(grads, f"{name}.down_proj", dW, db)
        d_h_gate = 0.0
        if cfg.activation == "swiglu":
            d_gate, d_up = swiglu_backward(d_act, cache["act"])               # (T, F) each
            d_h_gate, dW, db = linear_backward(d_gate, cache["gate"])         # (T, d)
            _accumulate_linear(grads, f"{name}.gate_proj", dW, db)
        elif cfg.activation == "gelu":
            d_up = gelu_backward(d_act, cache["act"])                         # (T, F)
        else:
            d_up = relu_backward(d_act, cache["act"])                         # (T, F)
        d_h_up, dW, db = linear_backward(d_up, cache["up"])                   # (T, d)
        _accumulate_linear(grads, f"{name}.up_proj", dW, db)
        return d_h_up + d_h_gate

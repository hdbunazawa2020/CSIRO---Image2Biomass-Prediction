import torch
import torch.nn.functional as F

# 数値安定用の微小量
eps = 1e-8


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Compute mean over valid (masked) elements.

    Args:
        x (Tensor): Values to be averaged. Shape is broadcastable with `m`.
        m (Tensor): Binary mask with 1 for valid, 0 for padded/invalid.
            Can be (..., T) or (..., T, 1); must be broadcastable to `x`.

    Returns:
        Tensor: Scalar tensor (0-d) = sum(x*m) / (sum(m) + eps).

    Note:
        - `m` is treated as weights {0,1}. If you want soft-weights, you can pass real-valued `m`.
        - Adds a small `eps` to avoid division by zero when mask is empty.
    """
    # 形状が違ってもブロードキャストで掛けられるように、そのまま計算
    return (x * m).sum() / (m.sum() + eps)


def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error with masking.

    Args:
        pred (Tensor): Predictions. Shape (..., T[, C]).
        tgt  (Tensor): Targets. Shape broadcastable to `pred`.
        m    (Tensor): Mask. Shape broadcastable to `pred` (1=valid, 0=invalid).

    Returns:
        Tensor: Scalar masked MSE.
    """
    return _masked_mean((pred - tgt) ** 2, m)


def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (L1) with masking.

    Args:
        pred (Tensor): Predictions.
        tgt  (Tensor): Targets.
        m    (Tensor): Mask (1=valid, 0=invalid).

    Returns:
        Tensor: Scalar masked L1.
    """
    return _masked_mean((pred - tgt).abs(), m)


def masked_huber(
    pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """Huber loss with masking.

    Huber(delta) = 0.5 * e^2                 if |e| <= delta
                 = delta * (|e| - 0.5*delta) otherwise

    Args:
        pred (Tensor): Predictions.
        tgt  (Tensor): Targets.
        m    (Tensor): Mask (1=valid, 0=invalid).
        delta (float): Transition point between L2 and L1 regions.

    Returns:
        Tensor: Scalar masked Huber loss.

    Tips:
        - 大きめの外れ値に対して勾配が穏やかになり、L2よりもロバストです。
        - データスケールに応じて delta を調整してください（例: 標準偏差の ~0.5–2 倍）。
    """
    diff = pred - tgt
    abs_diff = diff.abs()
    loss = torch.where(
        abs_diff <= delta,
        0.5 * abs_diff ** 2,
        delta * (abs_diff - 0.5 * delta),
    )
    return _masked_mean(loss, m)


def masked_smooth_l1(
    pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """Smooth L1 (a.k.a. L1 Huber variant) with masking.

    SmoothL1(beta) = 0.5 * (|e|^2) / beta   if |e| < beta
                   = |e| - 0.5 * beta       otherwise

    Args:
        pred (Tensor): Predictions.
        tgt  (Tensor): Targets.
        m    (Tensor): Mask (1=valid, 0=invalid).
        beta (float): Width of the quadratic region (beta->0 で L1 に近づく).

    Returns:
        Tensor: Scalar masked Smooth L1.
    """
    diff = (pred - tgt).abs()
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return _masked_mean(loss, m)


def masked_charbonnier(
    pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor, epsilon: float = 1e-3
) -> torch.Tensor:
    """Charbonnier (pseudo-Huber) loss with masking.

    loss = sqrt(e^2 + epsilon^2)

    Args:
        pred (Tensor): Predictions.
        tgt  (Tensor): Targets.
        m    (Tensor): Mask (1=valid, 0=invalid).
        epsilon (float): Smoothing constant to keep gradient finite near zero.

    Returns:
        Tensor: Scalar masked Charbonnier loss.

    Note:
        - L1 に似たロバスト性を保ちつつ、ゼロ付近で滑らかにします。
    """
    diff = pred - tgt
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    return _masked_mean(loss, m)


def masked_log_cosh(pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Log-cosh loss with masking.

    loss = log(cosh(e))

    Args:
        pred (Tensor): Predictions.
        tgt  (Tensor): Targets.
        m    (Tensor): Mask (1=valid, 0=invalid).

    Returns:
        Tensor: Scalar masked log-cosh loss.

    Note:
        - 小さい誤差では L2 に、誤差が大きい領域では L1 に近い振る舞いをします。
    """
    diff = pred - tgt
    loss = torch.log(torch.cosh(diff + 1e-12))  # 数値安定のため微小量を加算
    return _masked_mean(loss, m)


def masked_tukey_biweight(
    pred: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor, c: float = 4.685
) -> torch.Tensor:
    """Tukey's biweight (bisquare) loss with masking.

    For residual r and u = r / c:
        loss = (c^2 / 6) * (1 - (1 - u^2)^3)   if |u| <= 1
             = (c^2 / 6)                        otherwise

    Args:
        pred (Tensor): Predictions.
        tgt  (Tensor): Targets.
        m    (Tensor): Mask (1=valid, 0=invalid).
        c (float): Tuning constant controlling the cutoff of influence.

    Returns:
        Tensor: Scalar masked Tukey biweight loss.

    Warning:
        - |r| > c の外れ値はほぼ無視（勾配が極小）されます。
        - 外れ値割合が多すぎる場合は学習が進みにくくなる可能性があります。
    """
    r = pred - tgt
    u = r / (c + eps)  # ゼロ割防止
    w = torch.clamp(1 - u**2, min=0.0)
    loss = (c**2 / 6.0) * (1 - w**3)
    loss = torch.where(u.abs() <= 1, loss, (c**2) / 6.0)
    return _masked_mean(loss, m)

def masked_huber_temporal(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    m: torch.Tensor,
    delta: float = 1.0,
    time_decay: float = 0.03,
) -> torch.Tensor:
    """時間減衰付き Huber ロス（マスク対応）。
    pred, tgt: (..., T, C)
    m:        (..., T) or (..., T, 1)  (1=valid)
    time_decay > 0 のとき、t が大きいほど weight = exp(-time_decay * t) で軽くなる。
    """
    diff = pred - tgt                         # (..., T, C)
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff <= delta,
        0.5 * diff * diff,
        delta * (abs_diff - 0.5 * delta),
    )                                         # (..., T, C)
    # 時間減衰の重みを作成（時間次元 = -2 を想定）
    T = pred.size(-2)
    if time_decay > 0.0:
        t = torch.arange(T, device=pred.device, dtype=pred.dtype)  # (T,)
        w_t = torch.exp(-time_decay * t).view(
            *([1] * (huber.ndim - 2)), T, 1
        )  # 例: (1, 1, T, 1)
        huber = huber * w_t
        # mask も同じ重みでスケーリング
        if m.ndim == huber.ndim - 1:
            m_w = m.unsqueeze(-1) * w_t
        else:
            m_w = m * w_t
    else:
        if m.ndim == huber.ndim - 1:
            m_w = m.unsqueeze(-1)
        else:
            m_w = m
    return (huber * m_w).sum() / (m_w.sum() + eps)

def build_masked_loss(kind: str, **kw):
    """Factory for masked regression losses.

    Args:
        kind (str): Loss name.
            One of: {"mse", "l1"/"mae", "huber", "smoothl1"/"smooth_l1",
                     "charbonnier"/"pseudohuber"/"pseudo_huber",
                     "logcosh"/"log_cosh",
                     "tukey"/"tukey_biweight"/"biweight"}.
        **kw: Hyperparameters for each loss:
            huber_delta (float): For "huber".
            smoothl1_beta (float): For "smoothl1".
            charbonnier_eps (float): For "charbonnier".
            tukey_c (float): For "tukey".

    Returns:
        Callable[[Tensor, Tensor, Tensor], Tensor]:
            A function `f(pred, tgt, m) -> scalar loss`.

    Raises:
        ValueError: If `kind` is unknown.

    Examples:
        >>> loss_fn = build_masked_loss("huber", huber_delta=0.5)
        >>> loss = loss_fn(pred, target, mask)
    """
    kind = (kind or "mse").lower()

    if kind == "mse":
        return lambda p, t, m: masked_mse(p, t, m)

    if kind in ("l1", "mae"):
        return lambda p, t, m: masked_l1(p, t, m)

    if kind == "huber":
        delta = float(kw.get("huber_delta", 1.0))
        return lambda p, t, m: masked_huber(p, t, m, delta=delta)

    if kind in ("huber_time", "temporal_huber"):
        delta = float(kw.get("huber_delta", 1.0))
        time_decay = float(kw.get("time_decay", 0.03))
        return lambda p, t, m: masked_huber_temporal(
            p, t, m, delta=delta, time_decay=time_decay
        )

    if kind in ("smoothl1", "smooth_l1"):
        beta = float(kw.get("smoothl1_beta", 1.0))
        return lambda p, t, m: masked_smooth_l1(p, t, m, beta=beta)

    if kind in ("charbonnier", "pseudohuber", "pseudo_huber"):
        epsilon = float(kw.get("charbonnier_eps", 1e-3))
        return lambda p, t, m: masked_charbonnier(p, t, m, epsilon=epsilon)

    if kind in ("logcosh", "log_cosh"):
        return lambda p, t, m: masked_log_cosh(p, t, m)

    if kind in ("tukey", "tukey_biweight", "biweight"):
        c = float(kw.get("tukey_c", 4.685))
        return lambda p, t, m: masked_tukey_biweight(p, t, m, c=c)

    raise ValueError(f"Unknown loss kind: {kind}")

import torch
import torch.nn as nn
class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
    
    def forward(self, pred, target, mask):
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(abs_err <= self.delta, 0.5 * err * err, 
                           self.delta * (abs_err - 0.5 * self.delta))
        
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1) * weight
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)
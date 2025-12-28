from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

def global_weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """公式定義に合わせた「グローバル重み付きR²」を計算する。

    Args:
        y_true: 正解値 (N, K)
        y_pred: 予測値 (N, K)
        weights: ターゲット重み (K,)
            target_cols の順序に対応させる（重要）

    Returns:
        float: global weighted R²

    Notes:
        - 全(N*K)行を一つに並べて、行ごとに target type の重みを適用した R²。
        - μ（平均との差）は「重み付き平均」を使用。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    n, k = y_true.shape
    if weights.shape[0] != k:
        raise ValueError(f"weights length mismatch: expected {k}, got {weights.shape[0]}")

    # flatten（row-major）に合わせて重みも N回繰り返す
    w_flat = np.tile(weights, n)                 # (N*K,)
    yt = y_true.reshape(-1)                      # (N*K,)
    yp = y_pred.reshape(-1)                      # (N*K,)

    # 重み付き平均
    mu = np.sum(w_flat * yt) / np.sum(w_flat)

    ss_res = np.sum(w_flat * (yt - yp) ** 2)
    ss_tot = np.sum(w_flat * (yt - mu) ** 2)

    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """ターゲットごとのR²（デバッグ/分析用）"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n, k = y_true.shape
    out = []
    for j in range(k):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out.append(r2)
    return np.asarray(out, dtype=np.float64)
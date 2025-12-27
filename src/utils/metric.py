# -*- coding: utf-8 -*-
"""
metric.py

本コンペで使用する評価指標の計算関数をまとめます。
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """ターゲットごとのR2と、その重み付き平均（weighted_r2）を計算する。

    Args:
        y_true: 正解値。shape = (N, K)
        y_pred: 予測値。shape = (N, K)
        weights: 各ターゲットの重み。shape = (K,)
            None の場合はコンペ既定値 [0.1, 0.1, 0.1, 0.2, 0.5] を使用。

    Returns:
        weighted_r2: 重み付きR2（float）
        r2_scores: ターゲットごとのR2。shape = (K,)

    Notes:
        - ss_tot が 0（全て同じ値）になるターゲットは R2 を 0.0 とします。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true.shape != y_pred.shape: {y_true.shape} vs {y_pred.shape}")

    n, k = y_true.shape

    if weights is None:
        weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if weights.shape[0] != k:
        raise ValueError(f"weights length mismatch: expected {k}, got {weights.shape[0]}")

    # 各ターゲットのR2をベクトル化して計算
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean_true = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - mean_true) ** 2, axis=0)

    r2_scores = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)

    weighted_r2 = float(np.sum(r2_scores * weights) / np.sum(weights))
    return weighted_r2, r2_scores.astype(np.float64)
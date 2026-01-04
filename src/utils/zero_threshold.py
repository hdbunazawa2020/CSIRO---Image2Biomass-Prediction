# -*- coding: utf-8 -*-
"""
ゼロしきい値（0落とし）用の後処理＆OOFからのしきい値推定。

想定:
    - 入力は raw(g) 予測 / raw(g) 正解
    - Clover / Dead の「小さい予測を 0 に落とす」ことで
      ゼロ過剰（true=0が多い）に対するFPを抑える

注意:
    - しきい値は OOF に最適化するので過学習リスクはあります
    - まずは "delta" モード（GDM/Totalを差分だけ調整）が無難です
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np


def global_weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Sequence[float],
) -> float:
    """コンペ想定の global weighted R2 を numpy だけで計算する。

    定義（よくある形）:
        R2 = 1 - (Σ_j w_j * Σ_i (y_ij - p_ij)^2) / (Σ_j w_j * Σ_i (y_ij - mean_j)^2)

    Args:
        y_true: shape (N,K)
        y_pred: shape (N,K)
        weights: length K

    Returns:
        score: float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    # 念のため非負
    y_pred = np.clip(y_pred, 0.0, None)

    mean = y_true.mean(axis=0, keepdims=True)
    sse = ((y_true - y_pred) ** 2).sum(axis=0)  # (K,)
    sst = ((y_true - mean) ** 2).sum(axis=0)    # (K,)

    num = float((w * sse).sum())
    den = float((w * sst).sum())
    if den <= 0:
        return 0.0
    return 1.0 - num / den


def apply_zero_thresholds(
    preds_raw: np.ndarray,
    *,
    target_cols: List[str],
    thresholds: Dict[str, float],
    mode: str = "delta",
    clip_nonneg: bool = True,
) -> np.ndarray:
    """予測に「0落とし」しきい値を適用する。

    mode:
      - "none":
          Clover/Dead等の指定列だけ 0 に落とす。GDM/Totalは触らない。
      - "delta"（おすすめ）:
          0落としで減った分(delta)だけ GDM/Total を減算する（破壊が小さい）
      - "sum_fix":
          GDM=Green+Clover, Total=GDM+Dead で作り直す（強制整合）

    Args:
        preds_raw: shape (N,K) raw(g) 予測
        target_cols: 列順（target_cols）
        thresholds: {"Dry_Clover_g": 0.2, "Dry_Dead_g": 0.5} など
        mode: "none" | "delta" | "sum_fix"
        clip_nonneg: Trueなら最後に0以上へclip

    Returns:
        preds_pp: shape (N,K) 後処理後予測
    """
    cols = list(target_cols)
    preds = np.asarray(preds_raw, dtype=np.float64).copy()

    before = preds.copy()  # delta計算用

    # しきい値適用（指定列だけ）
    for name, thr in thresholds.items():
        if name not in cols:
            continue
        i = cols.index(name)
        t = float(thr)
        preds[preds[:, i] < t, i] = 0.0

    if mode == "delta":
        # Clover/Dead を落とした分だけ GDM/Total を減らす（必要列がある場合のみ）
        def _idx(n: str) -> Optional[int]:
            return cols.index(n) if n in cols else None

        i_green = _idx("Dry_Green_g")
        i_clov  = _idx("Dry_Clover_g")
        i_dead  = _idx("Dry_Dead_g")
        i_gdm   = _idx("GDM_g")
        i_total = _idx("Dry_Total_g")

        if i_clov is not None:
            delta_clov = before[:, i_clov] - preds[:, i_clov]
        else:
            delta_clov = 0.0

        if i_dead is not None:
            delta_dead = before[:, i_dead] - preds[:, i_dead]
        else:
            delta_dead = 0.0

        if i_gdm is not None and i_clov is not None:
            preds[:, i_gdm] = preds[:, i_gdm] - delta_clov

        if i_total is not None:
            # Totalは Clover/Dead の分だけ減らす（greenは変えてないので）
            if i_clov is not None:
                preds[:, i_total] = preds[:, i_total] - delta_clov
            if i_dead is not None:
                preds[:, i_total] = preds[:, i_total] - delta_dead

    elif mode == "sum_fix":
        # 物理整合を強制
        if ("Dry_Green_g" in cols) and ("Dry_Clover_g" in cols) and ("Dry_Dead_g" in cols):
            i_green = cols.index("Dry_Green_g")
            i_clov  = cols.index("Dry_Clover_g")
            i_dead  = cols.index("Dry_Dead_g")

            if "GDM_g" in cols:
                i_gdm = cols.index("GDM_g")
                preds[:, i_gdm] = preds[:, i_green] + preds[:, i_clov]

            if "Dry_Total_g" in cols:
                i_total = cols.index("Dry_Total_g")
                # GDMが無いなら green+clover を使う
                gdm = (preds[:, cols.index("GDM_g")] if "GDM_g" in cols else (preds[:, i_green] + preds[:, i_clov]))
                preds[:, i_total] = gdm + preds[:, i_dead]

    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if clip_nonneg:
        preds = np.clip(preds, 0.0, None)

    return preds


def _build_threshold_grid_from_true_zero(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx: int,
    *,
    q_start: float = 0.50,
    q_end: float = 0.995,
    n: int = 30,
    add_zero: bool = True,
) -> np.ndarray:
    """true==0 のときの予測分布から threshold 候補gridを作る。

    Args:
        y_true: (N,K)
        y_pred: (N,K)
        idx: 対象列index
        q_start/q_end/n: quantileの範囲と分割数
        add_zero: 0.0を候補に含める

    Returns:
        grid: 1D ndarray（昇順、重複除去済み）
    """
    mask0 = (y_true[:, idx] == 0.0)
    pred0 = y_pred[mask0, idx]

    if pred0.size == 0:
        return np.array([0.0], dtype=np.float64)

    qs = np.linspace(q_start, q_end, int(n))
    cand = np.quantile(pred0, qs)
    cand = np.unique(np.round(cand.astype(np.float64), 6))

    if add_zero:
        cand = np.unique(np.concatenate([np.array([0.0], dtype=np.float64), cand]))

    # 念のため非負のみ
    cand = cand[cand >= 0.0]
    return cand


def fit_zero_thresholds_grid(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    *,
    target_cols: List[str],
    weights: Sequence[float],
    targets: Tuple[str, str] = ("Dry_Clover_g", "Dry_Dead_g"),
    mode: str = "delta",
    grid_n: int = 30,
) -> Dict[str, float]:
    """OOFから Clover/Dead の最適しきい値をgrid searchで推定する（2D探索）。

    Args:
        y_true_raw: (N,K) raw(g) 正解
        y_pred_raw: (N,K) raw(g) 予測
        target_cols: 列順
        weights: global weighted r2 の重み
        targets: 最適化する2列（例: Clover, Dead）
        mode: apply_zero_thresholds のmode
        grid_n: 各列の候補数（大きいほど探索は増える）

    Returns:
        best_thresholds: {"Dry_Clover_g": t1, "Dry_Dead_g": t2}
    """
    cols = list(target_cols)

    # ベーススコア（thresholdなし）
    base_pred = apply_zero_thresholds(
        y_pred_raw,
        target_cols=cols,
        thresholds={targets[0]: 0.0, targets[1]: 0.0},
        mode=mode,
    )
    base_score = global_weighted_r2_score(y_true_raw, base_pred, weights)

    # grid構築（true==0条件の予測分布から作る）
    if targets[0] not in cols or targets[1] not in cols:
        raise KeyError(f"targets {targets} must exist in target_cols: {cols}")

    i1 = cols.index(targets[0])
    i2 = cols.index(targets[1])

    g1 = _build_threshold_grid_from_true_zero(y_true_raw, y_pred_raw, i1, n=grid_n)
    g2 = _build_threshold_grid_from_true_zero(y_true_raw, y_pred_raw, i2, n=grid_n)

    best_score = base_score
    best_t1 = 0.0
    best_t2 = 0.0

    # 2D grid search（Nが小さいので現実的）
    for t1 in g1:
        for t2 in g2:
            thr = {targets[0]: float(t1), targets[1]: float(t2)}
            pred_pp = apply_zero_thresholds(
                y_pred_raw,
                target_cols=cols,
                thresholds=thr,
                mode=mode,
            )
            score = global_weighted_r2_score(y_true_raw, pred_pp, weights)
            if score > best_score:
                best_score = score
                best_t1 = float(t1)
                best_t2 = float(t2)

    # 参考: どれだけ上がったかを表示したい場合は呼び出し側で比較してください
    return {targets[0]: best_t1, targets[1]: best_t2}
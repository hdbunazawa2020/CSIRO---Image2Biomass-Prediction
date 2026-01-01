
from __future__ import annotations

import os
import gc
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

import matplotlib
matplotlib.use("Agg")  # headless環境でも保存できるように
import matplotlib.pyplot as plt
from PIL import Image

# ===================================
# path / optional utils
# ===================================
_DEFAULT_PROJECT_ROOT = Path("/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction/")
_THIS_DIR = Path("/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction/src/script")
_SRC_DIR = Path("/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction/src")

import sys
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

# --- utils.data（無い場合はfallback）---
from utils.data import sep, show_df, set_seed  # type: ignore
# --- utils.metric（無い場合はfallback）---
from utils.metric import global_weighted_r2_score, r2_per_target  # type: ignore

# ===================================
# plotting helpers
# ===================================
def _prepare_xy_for_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    space: str,
    clip_min: float = 0.0,
    log_clip_min: float = -20.0,
    log_clip_max: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """散布図用に (x, y) を整形する。"""
    if space not in ("raw", "log1p"):
        raise ValueError(f"space must be 'raw' or 'log1p', got: {space}")

    if space == "raw":
        x = np.log1p(np.clip(y_true, clip_min, None))
        y = np.log1p(np.clip(y_pred, clip_min, None))
        return x, y

    x = np.clip(y_true, log_clip_min, log_clip_max)
    y = np.clip(y_pred, log_clip_min, log_clip_max)
    return x, y


def plot_scatter_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    *,
    folds: Optional[np.ndarray] = None,
    space: str = "raw",
    nrows: int = 2,
    ncols: int = 3,
    s: float = 10.0,
    alpha: float = 0.55,
    sample_per_fold: Optional[int] = None,
    show_legend: bool = True,
    title_prefix: str = "OOF scatter",
    savedir: Optional[Path] = None,
) -> None:
    """ターゲット別散布図を 2x3 で可視化（fold色分け対応）."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "y_true と y_pred のshapeが一致しません"
    assert y_true.shape[1] == len(target_cols), "target_cols と列数が一致しません"

    n, _k = y_true.shape

    if folds is not None:
        folds = np.asarray(folds).astype(int)
        assert len(folds) == n, "folds の長さが N と一致しません"
        fold_values = np.unique(folds)
        fold_values = np.sort(fold_values)
    else:
        fold_values = None

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    axes_flat = axes.ravel()

    cmap = plt.get_cmap("tab10")

    legend_handles = None
    legend_labels = None

    for j, t in enumerate(target_cols):
        ax = axes_flat[j]

        yt = y_true[:, j]
        yp = y_pred[:, j]

        x_all, y_all = _prepare_xy_for_scatter(yt, yp, space=space)

        if fold_values is not None:
            handles = []
            labels = []
            for fi, f in enumerate(fold_values):
                m = (folds == f)
                if not np.any(m):
                    continue

                x = x_all[m]
                y = y_all[m]

                if sample_per_fold is not None and len(x) > sample_per_fold:
                    idx = np.random.choice(len(x), size=int(sample_per_fold), replace=False)
                    x = x[idx]
                    y = y[idx]

                sc = ax.scatter(x, y, s=s, alpha=alpha, color=cmap(fi % 10), label=f"fold{int(f)}")
                handles.append(sc)
                labels.append(f"fold{int(f)}")

            if legend_handles is None:
                legend_handles, legend_labels = handles, labels

        else:
            ax.scatter(x_all, y_all, s=s, alpha=alpha)

        mn = float(min(np.min(x_all), np.min(y_all)))
        mx = float(max(np.max(x_all), np.max(y_all)))
        ax.plot([mn, mx], [mn, mx])

        ax.set_title(f"{title_prefix} ({space}): {t}")
        ax.set_xlabel("x=true")
        ax.set_ylabel("y=pred")
        ax.grid(True, alpha=0.2)

    for kk in range(len(target_cols), nrows * ncols):
        fig.delaxes(axes_flat[kk])

    if show_legend and legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(5, len(legend_labels)),
            frameon=False,
        )

    fig.tight_layout()

    if savedir is not None:
        savedir.mkdir(parents=True, exist_ok=True)
        out_path = savedir / f"scatter_grid_{space}.jpg"
        plt.savefig(out_path, dpi=200)
    plt.close(fig)


def build_image_score_df(
    ids: np.ndarray,
    folds: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    weights: Sequence[float],
    target_cols: List[str],
    df_pivot_pl: pl.DataFrame,
    *,
    presence_targets: Optional[List[str]] = None,
    presence_threshold_g: float = 0.0,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """OOFから画像単位の診断スコアを作る（r2_imgは使わない版）。"""
    ids = np.asarray(ids).astype(str)
    folds = np.asarray(folds).astype(int)
    targets = np.asarray(targets, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n, k = targets.shape
    assert preds.shape == targets.shape, "preds/targets shape mismatch"
    assert len(target_cols) == k, "target_cols length mismatch"
    assert len(weights) == k, "weights length mismatch"
    assert len(ids) == n and len(folds) == n, "ids/folds length mismatch"

    err = preds - targets
    abs_err = np.abs(err)
    sq_err = err ** 2

    wsq_err = sq_err * weights[None, :]
    wabs_err = abs_err * weights[None, :]

    weighted_sse_img = np.sum(wsq_err, axis=1)
    weighted_mae_img = np.sum(wabs_err, axis=1)

    mu = float(np.sum(targets * weights[None, :]) / (n * np.sum(weights)))
    sst_total = float(np.sum(weights[None, :] * (targets - mu) ** 2))
    impact_on_global_r2 = weighted_sse_img / max(sst_total, eps)

    dominant_idx = np.argmax(wsq_err, axis=1)
    dominant_target = [target_cols[i] for i in dominant_idx]
    dominant_wsse = wsq_err[np.arange(n), dominant_idx]

    if presence_targets is None:
        presence_targets = target_cols[:3]

    presence_indices = [target_cols.index(t) for t in presence_targets if t in target_cols]

    if len(presence_indices) > 0:
        thr = float(presence_threshold_g)
        true_pos = targets[:, presence_indices] > thr
        pred_pos = preds[:, presence_indices] > thr

        fp = (~true_pos) & (pred_pos)
        fn = (true_pos) & (~pred_pos)

        presence_fp_count = np.sum(fp, axis=1).astype(int)
        presence_fn_count = np.sum(fn, axis=1).astype(int)
        fp_any = np.any(fp, axis=1)
        fn_any = np.any(fn, axis=1)
    else:
        presence_fp_count = np.zeros(n, dtype=int)
        presence_fn_count = np.zeros(n, dtype=int)
        fp_any = np.zeros(n, dtype=bool)
        fn_any = np.zeros(n, dtype=bool)

    score_df = pd.DataFrame(
        {
            "image_id": ids,
            "fold": folds,
            "row_idx": np.arange(n, dtype=np.int64),
            "weighted_sse_img": weighted_sse_img.astype(np.float64),
            "impact_on_global_r2": impact_on_global_r2.astype(np.float64),
            "weighted_mae_img": weighted_mae_img.astype(np.float64),
            "dominant_target": dominant_target,
            "dominant_wsse": dominant_wsse.astype(np.float64),
            "presence_fp_count": presence_fp_count,
            "presence_fn_count": presence_fn_count,
            "presence_fp_any": fp_any.astype(bool),
            "presence_fn_any": fn_any.astype(bool),
        }
    )

    for j, t in enumerate(target_cols):
        score_df[f"true_{t}"] = targets[:, j]
        score_df[f"pred_{t}"] = preds[:, j]
        score_df[f"err_{t}"] = err[:, j]
        score_df[f"abs_err_{t}"] = abs_err[:, j]
        score_df[f"wsse_{t}"] = wsq_err[:, j]

    meta_cols = [c for c in ["image_id", "image_path", "Pre_GSHH_NDVI", "Height_Ave_cm", "State", "Sampling_Date", "Species"] if c in df_pivot_pl.columns]
    meta_pd = df_pivot_pl.select(meta_cols).to_pandas()
    score_df = score_df.merge(meta_pd, on="image_id", how="left")

    score_df["label"] = score_df.apply(
        lambda r: f"{r['image_id']} | fold{int(r['fold'])} | wsse={r['weighted_sse_img']:.2f}",
        axis=1,
    )

    return score_df


def plot_image_and_bars_grid(
    subset_df: pd.DataFrame,
    preds: np.ndarray,
    targets: np.ndarray,
    target_cols: List[str],
    input_dir: Union[str, Path],
    title_prefix: str,
    *,
    use_log1p_y: bool = False,
    score_col: str = "weighted_sse_img",
    out_name: str = "best.jpg",
    savedir: Optional[Path] = None,
) -> None:
    """Nx2（左=画像、右=target/pred棒グラフ）で可視化する."""
    n = len(subset_df)
    if n == 0:
        print("[WARN] subset_df is empty.")
        return

    if savedir is None:
        raise ValueError("savedir is required for saving plots.")

    savedir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(input_dir)

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)

    x = np.arange(len(target_cols))
    width = 0.35

    for i, row in enumerate(subset_df.itertuples(index=False)):
        idx = int(getattr(row, "row_idx"))
        image_id = str(getattr(row, "image_id"))
        fold = int(getattr(row, "fold"))
        score_val = float(getattr(row, score_col))

        ax_img = axes[i, 0]
        img_path = None
        if hasattr(row, "image_path") and getattr(row, "image_path") is not None:
            img_path = input_dir / str(getattr(row, "image_path"))

        if img_path is not None and img_path.exists():
            try:
                with Image.open(img_path) as im:
                    img = im.convert("RGB")
                ax_img.imshow(img)
            except Exception as e:
                ax_img.text(0.5, 0.5, f"Failed to open image\n{img_path}\n{e}", ha="center", va="center")
        else:
            ax_img.text(0.5, 0.5, f"Image not found\n{img_path}", ha="center", va="center")
        ax_img.axis("off")
        ax_img.set_title(f"{title_prefix}{i+1}: {image_id} Fold{fold} | {score_col}={score_val:.2f}")

        ax_bar = axes[i, 1]
        y_true = targets[idx].copy()
        y_pred = preds[idx].copy()

        if use_log1p_y:
            y_true_plot = np.log1p(np.clip(y_true, 0, None))
            y_pred_plot = np.log1p(np.clip(y_pred, 0, None))
            ax_bar.set_ylabel("log1p(g)")
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
            ax_bar.set_ylabel("g")

        ax_bar.bar(x - width / 2, y_true_plot, width, label="target")
        ax_bar.bar(x + width / 2, y_pred_plot, width, label="pred")

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(target_cols, rotation=45, ha="right")
        ax_bar.grid(True, axis="y", alpha=0.25)
        ax_bar.legend()

    fig.tight_layout()
    plt.savefig(savedir / out_name, dpi=200)
    plt.close(fig)


def plot_presence_fp_fn_by_fold(
    score_df: pd.DataFrame,
    component_targets: List[str],
    *,
    threshold: float = 0.0,
    savedir: Optional[Path] = None,
    out_name: str = "plot_presence_fp_fn_by_fold.jpg",
) -> pd.DataFrame:
    """fold別に presence の FP/FN を集計し、棒で可視化する。"""
    if savedir is None:
        raise ValueError("savedir is required for saving plots.")

    rows = []
    for f, g in score_df.groupby("fold"):
        for t in component_targets:
            yt = g[f"true_{t}"].values
            yp = g[f"pred_{t}"].values
            true_pos = yt > threshold
            pred_pos = yp > threshold

            fp = np.mean((~true_pos) & (pred_pos))
            fn = np.mean((true_pos) & (~pred_pos))
            zero_rate = np.mean(~true_pos)

            rows.append(
                {
                    "fold": int(f),
                    "target": t,
                    "n": int(len(g)),
                    "true_zero_rate": float(zero_rate),
                    "fp_rate": float(fp),
                    "fn_rate": float(fn),
                }
            )

    summary_df = pd.DataFrame(rows)

    folds_sorted = sorted(summary_df["fold"].unique())
    fig, axes = plt.subplots(1, len(component_targets), figsize=(5 * len(component_targets), 4), squeeze=False)
    axes = axes.ravel()

    for i, t in enumerate(component_targets):
        ax = axes[i]
        sub = summary_df[summary_df["target"] == t].sort_values("fold")

        x = np.arange(len(folds_sorted))
        ax.bar(x - 0.2, sub["true_zero_rate"].values, width=0.2, label="true_zero_rate")
        ax.bar(x + 0.0, sub["fp_rate"].values, width=0.2, label="FP rate")
        ax.bar(x + 0.2, sub["fn_rate"].values, width=0.2, label="FN rate")

        ax.set_xticks(x)
        ax.set_xticklabels([f"fold{f}" for f in folds_sorted])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Presence diagnostics: {t}")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend()

    plt.tight_layout()
    savedir.mkdir(parents=True, exist_ok=True)
    plt.savefig(savedir / out_name, dpi=200)
    plt.close(fig)

    return summary_df


def plot_abs_error_boxplot_by_fold(
    score_df: pd.DataFrame,
    target_cols: List[str],
    *,
    savedir: Path,
    out_name: str = "plot_abs_error_boxplot_by_fold.jpg",
) -> None:
    """ターゲット別に abs error の分布を foldごとの箱ひげで見る。"""
    folds_sorted = sorted(score_df["fold"].unique())
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    axes = axes.ravel()

    for j, t in enumerate(target_cols):
        ax = axes[j]
        data = [score_df.loc[score_df["fold"] == f, f"abs_err_{t}"].values for f in folds_sorted]
        ax.boxplot(data, labels=[f"f{f}" for f in folds_sorted], showfliers=False)
        ax.set_title(f"abs error by fold: {t}")
        ax.set_ylabel("|pred - true|")
        ax.grid(True, axis="y", alpha=0.2)

    for kk in range(len(target_cols), nrows * ncols):
        fig.delaxes(axes[kk])

    fig.tight_layout()
    savedir.mkdir(parents=True, exist_ok=True)
    plt.savefig(savedir / out_name, dpi=200)
    plt.close(fig)


def plot_true_distribution_by_fold(
    score_df: pd.DataFrame,
    target_cols: List[str],
    *,
    use_log1p: bool = True,
    bins: int = 20,
    savedir: Optional[Path] = None,
    out_name: str = "true_distribution_by_fold.jpg",
) -> None:
    """ターゲットの真値分布をfold別に重ねて描く（分布差の確認用）."""
    if savedir is None:
        raise ValueError("savedir is required for saving plots.")

    folds_sorted = sorted(score_df["fold"].unique())
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    axes = axes.ravel()

    for j, t in enumerate(target_cols):
        ax = axes[j]
        for f in folds_sorted:
            x = score_df.loc[score_df["fold"] == f, f"true_{t}"].values
            if use_log1p:
                x = np.log1p(np.clip(x, 0, None))
            ax.hist(x, bins=bins, alpha=0.35, label=f"fold{f}")

        ax.set_title(f"true distribution by fold: {t} ({'log1p' if use_log1p else 'raw'})")
        ax.grid(True, alpha=0.2)
        ax.legend()

    for kk in range(len(target_cols), nrows * ncols):
        fig.delaxes(axes[kk])

    fig.tight_layout()
    savedir.mkdir(parents=True, exist_ok=True)
    plt.savefig(savedir / out_name, dpi=200)
    plt.close(fig)


# ===================================
# additional diagnostics (State / Total-bin)
# ===================================
def state_global_r2(
    df: pd.DataFrame,
    target_cols: List[str],
    weights: np.ndarray,
    *,
    state_col: str = "State",
) -> pd.DataFrame:
    """State別 global weighted R2."""
    if state_col not in df.columns:
        raise KeyError(f"{state_col} not found in df.")
    rows = []
    for st, g in df.groupby(state_col, dropna=False):
        y_true = g[[f"true_{t}" for t in target_cols]].values
        y_pred = g[[f"pred_{t}" for t in target_cols]].values
        r2 = global_weighted_r2_score(y_true, y_pred, weights)
        rows.append({"State": st, "n": int(len(g)), "global_weighted_r2": float(r2)})
    out = pd.DataFrame(rows).sort_values("global_weighted_r2").reset_index(drop=True)
    return out


def state_target_metrics(
    df: pd.DataFrame,
    target_cols: List[str],
    weights: np.ndarray,
    *,
    state_col: str = "State",
) -> pd.DataFrame:
    """State×target で SSE / wSSE / RMSE / MAE を集計."""
    if state_col not in df.columns:
        raise KeyError(f"{state_col} not found in df.")
    rows = []
    for st, g in df.groupby(state_col, dropna=False):
        for j, t in enumerate(target_cols):
            e = (g[f"pred_{t}"].values - g[f"true_{t}"].values).astype(np.float64)
            sse = float(np.sum(e ** 2))
            wsse = float(weights[j] * sse)
            rmse = float(np.sqrt(np.mean(e ** 2)))
            mae = float(np.mean(np.abs(e)))
            rows.append(
                {
                    "State": st,
                    "target": t,
                    "n": int(len(g)),
                    "SSE": sse,
                    "wSSE": wsse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "weight": float(weights[j]),
                }
            )
    return pd.DataFrame(rows)


def total_bin_metrics(
    df: pd.DataFrame,
    target_cols: List[str],
    weights: np.ndarray,
    *,
    total_target: str = "Dry_Total_g",
    n_bins: int = 6,
) -> pd.DataFrame:
    """true_total の大きさ（qcut）でbin分けして誤差を集計する。"""
    df = df.copy()
    true_total_col = f"true_{total_target}"
    pred_total_col = f"pred_{total_target}"

    if true_total_col not in df.columns:
        raise KeyError(f"{true_total_col} not found in df.")
    if pred_total_col not in df.columns:
        raise KeyError(f"{pred_total_col} not found in df.")

    df["total_bin"] = pd.qcut(df[true_total_col], q=n_bins, duplicates="drop")

    rows = []
    for b, g in df.groupby("total_bin", observed=True):
        y_true = g[[f"true_{t}" for t in target_cols]].values
        y_pred = g[[f"pred_{t}" for t in target_cols]].values
        r2 = global_weighted_r2_score(y_true, y_pred, weights)

        e_total = (g[pred_total_col].values - g[true_total_col].values).astype(np.float64)
        mae_total = float(np.mean(np.abs(e_total)))
        rmse_total = float(np.sqrt(np.mean(e_total ** 2)))

        rows.append(
            {
                "bin": str(b),
                "n": int(len(g)),
                "true_total_mean": float(g[true_total_col].mean()),
                "true_total_median": float(g[true_total_col].median()),
                "pred_total_mean": float(g[pred_total_col].mean()),
                "MAE_total": mae_total,
                "RMSE_total": rmse_total,
                "mean_weighted_sse_img": float(g["weighted_sse_img"].mean()),
                "global_weighted_r2_in_bin": float(r2),
            }
        )

    out = pd.DataFrame(rows).sort_values("true_total_median").reset_index(drop=True)
    return out


def plot_error_vs_true_total(
    df: pd.DataFrame,
    *,
    total_target: str = "Dry_Total_g",
    savedir: Path,
    out_scatter: str = "error_vs_true_total.jpg",
) -> None:
    """true_total vs weighted_sse_img の散布図（診断）."""
    true_total_col = f"true_{total_target}"
    if true_total_col not in df.columns:
        print(f"[WARN] {true_total_col} not found. Skip plot_error_vs_true_total.")
        return

    savedir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(df[true_total_col].values, df["weighted_sse_img"].values, s=10, alpha=0.6)
    plt.xlabel(f"true {total_target}")
    plt.ylabel("weighted_sse_img")
    plt.grid(True, alpha=0.2)
    plt.title("Error vs true_total (diagnostic)")
    plt.tight_layout()
    plt.savefig(savedir / out_scatter, dpi=200)
    plt.close()


def plot_binned_error_trend(
    bin_df: pd.DataFrame,
    *,
    savedir: Path,
    out_name: str = "binned_error_trend_true_total.jpg",
) -> None:
    """bin中央値 vs mean_weighted_sse_img の折れ線."""
    if len(bin_df) == 0:
        print("[WARN] bin_df is empty. Skip plot_binned_error_trend.")
        return

    savedir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(bin_df["true_total_median"].values, bin_df["mean_weighted_sse_img"].values, marker="o")
    plt.xlabel("true_total median (per bin)")
    plt.ylabel("mean weighted_sse_img (per bin)")
    plt.grid(True, alpha=0.2)
    plt.title("Binned error trend (true_total)")
    plt.tight_layout()
    plt.savefig(savedir / out_name, dpi=200)
    plt.close()


def summarize_true_total_by_state(
    df: pd.DataFrame,
    *,
    total_target: str = "Dry_Total_g",
    state_col: str = "State",
) -> pd.DataFrame:
    """State別 true_total 分布サマリ（分位点など）."""
    true_total_col = f"true_{total_target}"
    if true_total_col not in df.columns:
        raise KeyError(f"{true_total_col} not found in df.")
    if state_col not in df.columns:
        raise KeyError(f"{state_col} not found in df.")

    def _q(p: float):
        return lambda x: float(np.quantile(x, p))

    out = (
        df.groupby(state_col)[true_total_col]
        .agg(
            n="count",
            mean="mean",
            median="median",
            p75=_q(0.75),
            p90=_q(0.90),
            p95=_q(0.95),
            max="max",
            min="min",
        )
        .reset_index()
        .sort_values("median")
        .reset_index(drop=True)
    )
    return out


def plot_true_total_distribution_by_state(
    df: pd.DataFrame,
    *,
    total_target: str = "Dry_Total_g",
    state_col: str = "State",
    bins: int = 20,
    savedir: Path,
    out_hist: str = "state_true_total_hist.jpg",
    out_box: str = "state_true_total_boxplot.jpg",
) -> None:
    """State別 true_total 分布（ヒスト+箱ひげ）を保存."""
    true_total_col = f"true_{total_target}"
    if true_total_col not in df.columns or state_col not in df.columns:
        print("[WARN] Required columns missing. Skip plot_true_total_distribution_by_state.")
        return

    states = [s for s in sorted(df[state_col].dropna().unique())]
    if len(states) == 0:
        print("[WARN] No states found. Skip plot_true_total_distribution_by_state.")
        return

    savedir.mkdir(parents=True, exist_ok=True)

    n_states = len(states)
    ncols = min(4, n_states)
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True, squeeze=False)
    axes = axes.ravel()

    for ax, st in zip(axes, states):
        x = df.loc[df[state_col] == st, true_total_col].values
        ax.hist(x, bins=bins, alpha=0.9)
        ax.set_title(f"{st} (n={len(x)})")
        ax.set_xlabel("true_total")
        ax.grid(True, alpha=0.2)

    for ax in axes[len(states):]:
        ax.axis("off")

    axes[0].set_ylabel("count")
    fig.suptitle(f"State-wise distribution of true {total_target}")
    fig.tight_layout()
    plt.savefig(savedir / out_hist, dpi=200)
    plt.close(fig)

    plt.figure(figsize=(max(6, 1.2 * len(states)), 4))
    df.boxplot(column=true_total_col, by=state_col)
    plt.title("State-wise true_total (boxplot)")
    plt.suptitle("")
    plt.ylabel("true_total")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(savedir / out_box, dpi=200)
    plt.close()


def focus_state_worst(
    df: pd.DataFrame,
    *,
    focus_state: str,
    worst_n: int = 15,
    state_col: str = "State",
) -> pd.DataFrame:
    """特定Stateの worst（weighted_sse_img 上位）一覧."""
    if state_col not in df.columns:
        return pd.DataFrame()
    sub = df[df[state_col] == focus_state].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    worst = sub.sort_values("weighted_sse_img", ascending=False).head(worst_n).reset_index(drop=True)
    return worst


def state_bin_metrics(
    df: pd.DataFrame,
    target_cols: List[str],
    weights: np.ndarray,
    *,
    total_target: str = "Dry_Total_g",
    n_bins: int = 6,
    state_col: str = "State",
) -> pd.DataFrame:
    """State×true_total bin の誤差テーブル."""
    true_total_col = f"true_{total_target}"
    pred_total_col = f"pred_{total_target}"
    if true_total_col not in df.columns or pred_total_col not in df.columns:
        raise KeyError("true_total/pred_total columns not found.")
    if state_col not in df.columns:
        raise KeyError(f"{state_col} not found.")

    df_bins = df.copy()
    df_bins["total_bin"] = pd.qcut(df_bins[true_total_col], q=n_bins, duplicates="drop")

    def _group_metrics(g: pd.DataFrame) -> pd.Series:
        y_true = g[[f"true_{t}" for t in target_cols]].values
        y_pred = g[[f"pred_{t}" for t in target_cols]].values
        r2 = global_weighted_r2_score(y_true, y_pred, weights)

        e_total = (g[pred_total_col].values - g[true_total_col].values).astype(np.float64)
        mae_total = float(np.mean(np.abs(e_total)))
        rmse_total = float(np.sqrt(np.mean(e_total ** 2)))

        return pd.Series({
            "n": int(len(g)),
            "true_total_median": float(g[true_total_col].median()),
            "mean_weighted_sse_img": float(g["weighted_sse_img"].mean()),
            "sum_weighted_sse_img": float(g["weighted_sse_img"].sum()),
            "MAE_total": mae_total,
            "RMSE_total": rmse_total,
            "global_weighted_r2_in_group": float(r2),
        })

    out = (
        df_bins.groupby([state_col, "total_bin"], observed=True)
        .apply(_group_metrics)
        .reset_index()
        .sort_values([state_col, "true_total_median"])
        .reset_index(drop=True)
    )
    out["total_bin"] = out["total_bin"].astype(str)
    return out


def plot_binned_error_trend_by_state(
    state_bin_df: pd.DataFrame,
    *,
    state_col: str = "State",
    savedir: Path,
    out_name: str = "binned_error_trend_by_state.jpg",
) -> None:
    """Stateごとの bin推移（mean_weighted_sse_img）を同一図に描く."""
    if len(state_bin_df) == 0:
        print("[WARN] state_bin_df is empty. Skip plot.")
        return

    savedir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for st in sorted(state_bin_df[state_col].dropna().unique()):
        g = state_bin_df[state_bin_df[state_col] == st].sort_values("true_total_median")
        plt.plot(g["true_total_median"].values, g["mean_weighted_sse_img"].values, marker="o", label=str(st))
    plt.xlabel("true_total median (per bin)")
    plt.ylabel("mean weighted_sse_img")
    plt.title("Binned error trend by State (true_total)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir / out_name, dpi=200)
    plt.close()


# ===================================
# postprocess helper
# ===================================
def apply_sum_fix(preds: np.ndarray, target_cols: List[str]) -> np.ndarray:
    """
    目的変数に GDM / Total が含まれる場合、
    - GDM = Green + Clover
    - Total = Green + Clover + Dead
    を強制する後処理。
    """
    preds_fix = np.asarray(preds, dtype=np.float64).copy()
    name_to_idx = {t: i for i, t in enumerate(target_cols)}

    def _find(keys: List[str]) -> Optional[int]:
        for k in keys:
            if k in name_to_idx:
                return name_to_idx[k]
        return None

    idx_green = _find(["Dry_Green_g", "Green", "Dry_Green"])
    idx_clover = _find(["Dry_Clover_g", "Clover", "Dry_Clover"])
    idx_dead = _find(["Dry_Dead_g", "Dead", "Dry_Dead"])
    idx_gdm = _find(["GDM_g", "GDM"])
    idx_total = _find(["Dry_Total_g", "Total_g", "Dry_Total", "Total"])

    if idx_green is not None and idx_clover is not None and idx_gdm is not None:
        preds_fix[:, idx_gdm] = preds_fix[:, idx_green] + preds_fix[:, idx_clover]

    if idx_green is not None and idx_clover is not None and idx_dead is not None and idx_total is not None:
        preds_fix[:, idx_total] = preds_fix[:, idx_green] + preds_fix[:, idx_clover] + preds_fix[:, idx_dead]

    return preds_fix


# ===================================
# main
# ===================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF評価・診断スクリプト（scatter / best-worst / State×bin など）")
    parser.add_argument("EXP", help="実験番号（experiments/<EXP> を参照）")

    parser.add_argument(
        "--project_root",
        type=str,
        default=os.environ.get("PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT)),
        help="プロジェクトルート。デフォルトは env:PROJECT_ROOT またはスクリプトの親ディレクトリ。",
    )
    parser.add_argument("--focus_state", type=str, default="NSW", help="worstを確認したいState")
    parser.add_argument("--worst_n_state", type=int, default=15, help="focus_state の worst 件数")
    parser.add_argument("--n_bins", type=int, default=6, help="true_total のbin数（qcut）")

    parser.add_argument("--no_sum_fix", action="store_true", help="GDM/Totalの和制約fixを無効化する")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # date
    date = datetime.now().strftime("%Y%m%d")
    print(f"TODAY is {date}")
    print("[ARGS]", vars(args))

    set_seed(args.seed)

    project_root = Path(args.project_root).expanduser().resolve()
    exp_dir = project_root / "experiments" / args.EXP
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")

    # --------------------
    # config
    # --------------------
    cfg_path = exp_dir / "yaml" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    folds_cfg = cfg.get("folds")
    if folds_cfg is None:
        raise KeyError("cfg['folds'] not found in config.yaml")

    target_cols = cfg.get("target_cols") or [
        "Dry_Green_g",
        "Dry_Clover_g",
        "Dry_Dead_g",
        "GDM_g",
        "Dry_Total_g",
    ]
    target_cols = list(target_cols)

    if "metric" in cfg and isinstance(cfg["metric"], dict) and "weights" in cfg["metric"]:
        weights = np.asarray(cfg["metric"]["weights"], dtype=np.float64)
    else:
        # fallback（コンペ設定）
        weights = np.asarray([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)

    # output dir（cfgに合わせて保存したい場合にも対応）
    if "output_dir" in cfg and "exp" in cfg:
        savedir = Path(cfg["output_dir"]) / str(cfg["exp"]) / "eval"
    else:
        savedir = exp_dir / "eval"
    savedir.mkdir(parents=True, exist_ok=True)

    sep("CONFIG")
    print("project_root:", project_root)
    print("exp_dir     :", exp_dir)
    print("savedir     :", savedir)
    print("folds       :", folds_cfg)
    print("targets     :", target_cols)
    print("weights     :", weights)

    # --------------------
    # df_pivot（meta）
    # --------------------
    pp_dir = Path(cfg.get("pp_dir", project_root / "preprocess"))
    preprocess_ver = cfg.get("preprocess_ver", "")
    data_ppdir = pp_dir / preprocess_ver if preprocess_ver else pp_dir
    df_pivot_path = data_ppdir / "df_pivot.csv"
    if not df_pivot_path.exists():
        raise FileNotFoundError(f"df_pivot.csv not found: {df_pivot_path}")
    df_pivot = pl.read_csv(df_pivot_path)
    sep("df_pivot")
    show_df(df_pivot, 5)

    # --------------------
    # OOF load
    # --------------------
    ids_list: List[np.ndarray] = []
    folds_list: List[np.ndarray] = []
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    for fold in folds_cfg:
        oof_path = exp_dir / "oof" / f"oof_fold{fold}.npz"
        if not oof_path.exists():
            raise FileNotFoundError(f"OOF file not found: {oof_path}")

        oof = np.load(oof_path)

        if "ids" not in oof or "preds" not in oof or "targets" not in oof:
            raise KeyError(f"npz keys must include ids/preds/targets. got={list(oof.keys())}")

        _ids = oof["ids"]
        _preds = oof["preds"]
        _targets = oof["targets"]

        ids_list.append(_ids.astype(str))
        folds_list.append(np.full(len(_ids), int(fold), dtype=int))
        preds_list.append(_preds.astype(np.float64))
        targets_list.append(_targets.astype(np.float64))

        print(f"[INFO] Loaded OOF: {oof_path}  n={len(_ids)}")

    ids = np.concatenate(ids_list).astype(str)
    folds = np.concatenate(folds_list).astype(int)
    preds = np.concatenate(preds_list).astype(np.float64)
    targets = np.concatenate(targets_list).astype(np.float64)

    # log1p（表示用）
    targs_log = np.log1p(np.clip(targets, 0, None))

    # --------------------
    # postprocess（sum fix）
    # --------------------
    r2_before = global_weighted_r2_score(targets, preds, weights)
    preds_eval = preds
    if not args.no_sum_fix:
        preds_eval = apply_sum_fix(preds, target_cols)
        r2_after = global_weighted_r2_score(targets, preds_eval, weights)
        print(f"[INFO] Sum-fix applied. R2 before={r2_before:.6f} after={r2_after:.6f}")
    else:
        print(f"[INFO] Sum-fix disabled. R2={r2_before:.6f}")

    preds_log_eval = np.log1p(np.clip(preds_eval, 0, None))

    # per target r2（参考）
    try:
        r2_targets = r2_per_target(targets, preds_eval)
        print("[INFO] R2 per target:")
        for t, r2v in zip(target_cols, r2_targets):
            print(f"  {t:>15s}: {r2v:.6f}")
    except Exception as e:
        print("[WARN] r2_per_target failed:", e)

    # --------------------
    # 保存用：OOF予測の表（polars）
    # --------------------
    pred_df = pl.DataFrame({"image_id": ids, "fold": folds})
    for j, t in enumerate(target_cols):
        pred_df = pred_df.with_columns(
            pl.Series(f"preds_{t}", preds_eval[:, j]),
            pl.Series(f"targets_{t}", targets[:, j]),
            pl.Series(f"preds_log_{t}", preds_log_eval[:, j]),
            pl.Series(f"targets_log_{t}", targs_log[:, j]),
        )
    pred_df_path = savedir / "oof_pred_df.csv"
    pred_df.write_csv(pred_df_path)
    print(f"[INFO] Saved: {pred_df_path}")

    # --------------------
    # scatter grid
    # --------------------
    sep("scatter_grid (raw->log1p view)")
    plot_scatter_grid(
        y_true=targets,
        y_pred=preds_eval,
        target_cols=target_cols,
        folds=folds,
        space="raw",
        nrows=2,
        ncols=3,
        show_legend=True,
        savedir=savedir,
    )
    plot_scatter_grid(
        y_true=targs_log,
        y_pred=preds_log_eval,
        target_cols=target_cols,
        folds=folds,
        space="log1p",
        nrows=2,
        ncols=3,
        show_legend=True,
        savedir=savedir,
    )
    print(f"[INFO] Saved scatter plots under: {savedir}")

    # --------------------
    # image-level score df
    # --------------------
    presence_thr = float(cfg.get("loss", {}).get("presence_threshold_g", 0.0)) if isinstance(cfg.get("loss", {}), dict) else 0.0

    score_df = build_image_score_df(
        ids=ids,
        folds=folds,
        targets=targets,
        preds=preds_eval,
        weights=weights,
        target_cols=target_cols,
        df_pivot_pl=df_pivot,
        presence_threshold_g=presence_thr,
    )
    score_df_path = savedir / "score_df.csv"
    score_df.to_csv(score_df_path, index=False)
    print(f"[INFO] Saved: {score_df_path}")

    # best / worst
    TOP_N = 8
    WORST_N = 8
    best_df = score_df.sort_values("weighted_sse_img", ascending=True).head(TOP_N)
    worst_df = score_df.sort_values("weighted_sse_img", ascending=False).head(WORST_N)

    sep("BEST/WORST (table)")
    cols_preview = ["image_id", "fold", "State", "Sampling_Date", "weighted_sse_img", "impact_on_global_r2", "dominant_target"]
    cols_preview = [c for c in cols_preview if c in score_df.columns]
    print("[BEST]")
    show_df(best_df[cols_preview], TOP_N)
    print("[WORST]")
    show_df(worst_df[cols_preview], WORST_N)

    # images + bars
    input_dir = cfg.get("input_dir")
    if input_dir is None:
        cand = project_root / "input"
        if cand.exists():
            input_dir = str(cand)

    if input_dir is None:
        print("[WARN] cfg['input_dir'] not found, and fallback input/ not found. Skip image plotting.")
    else:
        try:
            plot_image_and_bars_grid(
                subset_df=best_df,
                preds=preds_eval,
                targets=targets,
                target_cols=target_cols,
                input_dir=input_dir,
                title_prefix="TOP",
                use_log1p_y=False,
                out_name="best.jpg",
                savedir=savedir,
            )
            plot_image_and_bars_grid(
                subset_df=worst_df,
                preds=preds_eval,
                targets=targets,
                target_cols=target_cols,
                input_dir=input_dir,
                title_prefix="WORST",
                use_log1p_y=False,
                out_name="worst.jpg",
                savedir=savedir,
            )
        except Exception as e:
            print("[WARN] plot_image_and_bars_grid failed:", e)

    # presence by fold
    component_targets = [t for t in ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"] if t in target_cols]
    if len(component_targets) > 0:
        presence_summary = plot_presence_fp_fn_by_fold(score_df, component_targets, threshold=0.0, savedir=savedir)
        presence_summary.to_csv(savedir / "presence_summary_by_fold.csv", index=False)
        print(f"[INFO] Saved: {savedir / 'presence_summary_by_fold.csv'}")
    else:
        print("[WARN] component targets not found in target_cols. Skip presence diagnostics.")

    # abs error boxplot
    plot_abs_error_boxplot_by_fold(score_df, target_cols, savedir=savedir)

    # true distribution
    plot_true_distribution_by_fold(score_df, target_cols, use_log1p=True, savedir=savedir)

    # --------------------
    # additional diagnostics: State / total bin
    # --------------------
    sep("State / Total-bin diagnostics")
    if "State" in score_df.columns and "weighted_sse_img" in score_df.columns and "true_Dry_Total_g" in score_df.columns:
        # State global R2
        state_r2_df = state_global_r2(score_df, target_cols, weights)
        state_r2_df.to_csv(savedir / "state_global_weighted_r2.csv", index=False)
        print("[INFO] Saved: state_global_weighted_r2.csv")
        show_df(state_r2_df, 10)

        # State×target metrics
        state_sse_df = state_target_metrics(score_df, target_cols, weights)
        state_sse_df.to_csv(savedir / "state_target_metrics.csv", index=False)
        print("[INFO] Saved: state_target_metrics.csv")

        # pivot wSSE
        wsse_pivot = state_sse_df.pivot(index="State", columns="target", values="wSSE")
        wsse_pivot.to_csv(savedir / "state_target_wSSE_pivot.csv")
        print("[INFO] Saved: state_target_wSSE_pivot.csv")

        # total bin metrics (overall)
        bin_df = total_bin_metrics(score_df, target_cols, weights, n_bins=args.n_bins)
        bin_df.to_csv(savedir / "total_bin_metrics.csv", index=False)
        print("[INFO] Saved: total_bin_metrics.csv")
        show_df(bin_df, 10)

        plot_error_vs_true_total(score_df, savedir=savedir)
        plot_binned_error_trend(bin_df, savedir=savedir)

        # State-wise true_total distribution
        state_total_summary = summarize_true_total_by_state(score_df)
        state_total_summary.to_csv(savedir / "state_true_total_summary.csv", index=False)
        print("[INFO] Saved: state_true_total_summary.csv")
        plot_true_total_distribution_by_state(score_df, savedir=savedir)

        # focus state worst
        worst_state_df = focus_state_worst(score_df, focus_state=args.focus_state, worst_n=args.worst_n_state)
        if len(worst_state_df) == 0:
            print(f"[WARN] focus_state='{args.focus_state}' not found or empty.")
        else:
            worst_state_df.to_csv(savedir / f"{args.focus_state}_worst_{args.worst_n_state}.csv", index=False)
            print(f"[INFO] Saved: {args.focus_state}_worst_{args.worst_n_state}.csv")

            if input_dir is not None:
                try:
                    plot_image_and_bars_grid(
                        subset_df=worst_state_df.head(min(8, len(worst_state_df))),
                        preds=preds_eval,
                        targets=targets,
                        target_cols=target_cols,
                        input_dir=input_dir,
                        title_prefix=f"{args.focus_state}_WORST",
                        use_log1p_y=False,
                        out_name=f"{args.focus_state}_worst.jpg",
                        savedir=savedir,
                    )
                except Exception as e:
                    print("[WARN] focus_state worst plot failed:", e)

        # State×bin table
        state_bin_df = state_bin_metrics(score_df, target_cols, weights, n_bins=args.n_bins)
        state_bin_df.to_csv(savedir / "state_totalbin_metrics.csv", index=False)
        print("[INFO] Saved: state_totalbin_metrics.csv")

        # pivot outputs
        pivot_mean = state_bin_df.pivot(index="State", columns="total_bin", values="mean_weighted_sse_img")
        pivot_n = state_bin_df.pivot(index="State", columns="total_bin", values="n")
        pivot_r2 = state_bin_df.pivot(index="State", columns="total_bin", values="global_weighted_r2_in_group")
        pivot_mean.to_csv(savedir / "state_totalbin_pivot_mean_weighted_sse.csv")
        pivot_n.to_csv(savedir / "state_totalbin_pivot_n.csv")
        pivot_r2.to_csv(savedir / "state_totalbin_pivot_global_r2.csv")
        print("[INFO] Saved: state_totalbin pivot tables (mean/n/r2)")

        plot_binned_error_trend_by_state(state_bin_df, savedir=savedir)

    else:
        print("[WARN] State / true_Dry_Total_g not found. Skip State/Total diagnostics.")

    gc.collect()
    print("\n✅ DONE. outputs ->", savedir)


if __name__ == "__main__":
    main()

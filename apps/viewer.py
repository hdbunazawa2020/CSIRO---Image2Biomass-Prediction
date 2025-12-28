"""
Streamlit OOF Viewer

2ページ構成:
1) 個別のデータ可視化
   - "image_id | r2=..." のドロップダウンから1件選択
   - 左: 画像、右: target/pred の棒グラフ

2) 全体のデータ可視化
   - global weighted R²（公式定義）
   - 散布図グリッド（log1p）
   - TOP / WORST サンプル可視化

起動例:
streamlit run app/viewer.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import streamlit as st
# original
import sys
sys.path.append("/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction/src")
from utils.metric import global_weighted_r2_score  # 公式global
# ※ r2_per_target は必要なら import してください

PROJECT_ROOT = Path("/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction")
# =========================================================
# Cache loaders
# =========================================================
@st.cache_data(show_spinner=False)
def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_pivot_csv(pp_dir: str, preprocess_ver: str, pivot_csv_name: str) -> pd.DataFrame:
    path = Path(pp_dir) / preprocess_ver / pivot_csv_name
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_oof_npz(npz_path: str) -> Dict[str, np.ndarray]:
    npz = np.load(npz_path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def load_oof(exp_dir: Path, fold: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """foldを指定して OOF を読む。fold="ALL"なら全fold結合。"""
    oof_dir = exp_dir / "oof"
    files = sorted(oof_dir.glob("oof_fold*.npz"))
    if len(files) == 0:
        raise FileNotFoundError(f"OOF files not found: {oof_dir}")

    if fold != "ALL":
        f = oof_dir / f"oof_fold{fold}.npz"
        d = load_oof_npz(str(f))
        return d["ids"].astype(str), d["preds"], d["targets"]

    # ALL: concat
    ids_all, preds_all, targs_all = [], [], []
    for f in files:
        d = load_oof_npz(str(f))
        ids_all.append(d["ids"].astype(str))
        preds_all.append(d["preds"])
        targs_all.append(d["targets"])
    return np.concatenate(ids_all), np.concatenate(preds_all), np.concatenate(targs_all)


def build_image_score_df(
    ids: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    weights: List[float],
    pivot_df: pd.DataFrame,
) -> pd.DataFrame:
    """画像単位の診断スコア r2_img を計算してメタ情報と結合する。"""
    w = np.asarray(weights, dtype=np.float64)
    n = targets.shape[0]

    # globalと整合する mean
    mu = np.sum(targets * w[None, :]) / (n * np.sum(w))

    sse = np.sum(w[None, :] * (targets - preds) ** 2, axis=1)
    sst = np.sum(w[None, :] * (targets - mu) ** 2, axis=1)
    r2_img = np.where(sst > 0, 1.0 - sse / sst, np.nan)

    df = pd.DataFrame(
        {
            "image_id": ids.astype(str),
            "row_idx": np.arange(n, dtype=np.int64),
            "r2_img": r2_img,
            "sse": sse,
            "sst": sst,
        }
    )

    meta_cols = [c for c in ["image_id", "image_path", "State", "Sampling_Date", "month"] if c in pivot_df.columns]
    df = df.merge(pivot_df[meta_cols], on="image_id", how="left")
    df["label"] = df.apply(lambda r: f"{r['image_id']} | r2={r['r2_img']:.3f}", axis=1)
    return df


def make_bar_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    use_log1p_y: bool,
    title: str,
):
    """target/pred棒グラフを作る（matplotlib）。"""
    x = np.arange(len(target_cols))
    width = 0.35

    if use_log1p_y:
        y_true_plot = np.log1p(np.clip(y_true, 0, None))
        y_pred_plot = np.log1p(np.clip(y_pred, 0, None))
        ylabel = "log1p(g)"
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        ylabel = "g"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, y_true_plot, width, label="target")
    ax.bar(x + width / 2, y_pred_plot, width, label="pred")
    ax.set_xticks(x)
    ax.set_xticklabels(target_cols, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter_grid(targets: np.ndarray, preds: np.ndarray, target_cols: List[str]) -> plt.Figure:
    """ターゲット別散布図（log1p）を 2x3 で表示する。"""
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4), squeeze=False)
    axes_flat = axes.ravel()

    for j, t in enumerate(target_cols):
        ax = axes_flat[j]
        x = np.log1p(np.clip(targets[:, j], 0, None))
        y = np.log1p(np.clip(preds[:, j], 0, None))
        ax.scatter(x, y, s=8, alpha=0.6)
        mn = float(min(x.min(), y.min()))
        mx = float(max(x.max(), y.max()))
        ax.plot([mn, mx], [mn, mx])
        ax.set_title(t)
        ax.set_xlabel("log1p(true)")
        ax.set_ylabel("log1p(pred)")
        ax.grid(True, alpha=0.2)

    # 余り枠を消す
    for k in range(len(target_cols), nrows * ncols):
        fig.delaxes(axes_flat[k])

    fig.tight_layout()
    return fig


def plot_top_worst_figure(
    score_df: pd.DataFrame,
    preds: np.ndarray,
    targets: np.ndarray,
    target_cols: List[str],
    input_dir: str,
    mode: str,
    n: int,
    use_log1p_y: bool,
) -> plt.Figure:
    """TOP/WORST を Nx2 でまとめて描画する（Streamlit用）。"""
    df = score_df.dropna(subset=["r2_img"]).copy()
    if mode == "TOP":
        df = df.sort_values("r2_img", ascending=False).head(n)
    else:
        df = df.sort_values("r2_img", ascending=True).head(n)

    fig, axes = plt.subplots(len(df), 2, figsize=(14, 4 * len(df)), squeeze=False)

    for i, row in enumerate(df.itertuples(index=False)):
        idx = int(row.row_idx)
        img_path = Path(input_dir) / str(row.image_path)
        img = Image.open(img_path).convert("RGB")

        ax_img = axes[i, 0]
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(f"{mode}{i+1}: {row.image_id} | r2={row.r2_img:.3f}")

        ax_bar = axes[i, 1]
        y_true = targets[idx]
        y_pred = preds[idx]

        # same bar logic but inline（速度のため）
        x = np.arange(len(target_cols))
        width = 0.35
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
        ax_bar.set_title(f"SSE(w)={row.sse:.2f}")

    fig.tight_layout()
    return fig


# =========================================================
# UI
# =========================================================
def main():
    st.set_page_config(page_title="CSIRO OOF Viewer", layout="wide")

    # experiments から選択
    exp_root = PROJECT_ROOT / "experiments"
    exp_candidates = sorted([p.name for p in exp_root.glob("*") if p.is_dir()])
    if len(exp_candidates) == 0:
        st.error(f"experiments dir not found or empty: {exp_root}")
        return

    st.sidebar.header("設定")
    exp_name = st.sidebar.selectbox("Experiment", exp_candidates, index=0)
    exp_dir = exp_root / exp_name

    cfg_path = exp_dir / "yaml" / "config.yaml"
    if not cfg_path.exists():
        st.error(f"config.yaml not found: {cfg_path}")
        return

    cfg = load_yaml(str(cfg_path))
    target_cols = list(cfg["target_cols"])
    weights = list(cfg["metric"]["weights"])

    # fold選択（存在する oof_fold*.npz を見て選択肢を作る）
    oof_dir = exp_dir / "oof"
    fold_files = sorted(oof_dir.glob("oof_fold*.npz"))
    fold_ids = []
    for f in fold_files:
        # "oof_fold0.npz" -> "0"
        name = f.stem.replace("oof_fold", "")
        fold_ids.append(name)
    fold_options = ["ALL"] + fold_ids
    fold = st.sidebar.selectbox("Fold", fold_options, index=0)

    use_log1p_y = st.sidebar.checkbox("棒グラフのY軸をlog1p表示", value=False)
    top_n = st.sidebar.slider("TOP/WORST 表示数", min_value=3, max_value=30, value=8, step=1)

    page = st.sidebar.radio("ページ", ["1) 個別のデータ可視化", "2) 全体のデータ可視化"])

    # pivot と oof をロード
    pivot_df = load_pivot_csv(cfg["pp_dir"], cfg["preprocess_ver"], cfg["pivot_csv_name"])
    ids, preds, targets = load_oof(exp_dir, fold)

    # 非負クリップ（表示/評価の安定化）
    preds = np.clip(preds, 0.0, None)

    # 公式 global R²（表示用）
    global_r2 = global_weighted_r2_score(targets, preds, np.asarray(weights, dtype=np.float64))

    score_df = build_image_score_df(ids, targets, preds, weights, pivot_df)
    score_df = score_df.dropna(subset=["r2_img"]).sort_values("r2_img", ascending=False).reset_index(drop=True)

    st.title(f"OOF Viewer: {exp_name} (fold={fold})")
    st.write(f"**Global weighted R² (official style):** {global_r2:.5f}")

    # =====================================================
    # Page 1: Individual
    # =====================================================
    if page.startswith("1"):
        st.subheader("個別のデータ可視化")

        # selectbox は index を選ばせると扱いやすい
        idx = st.selectbox(
            "サンプル選択（image_id | r2=...）",
            options=list(range(len(score_df))),
            format_func=lambda i: score_df.loc[i, "label"],
        )
        row = score_df.loc[idx]
        row_idx = int(row["row_idx"])

        # meta
        st.caption(
            f"image_id={row['image_id']} / r2={row['r2_img']:.3f} / "
            f"State={row.get('State','-')} / Date={row.get('Sampling_Date','-')}"
        )

        # 画像 + 棒グラフ（横並び）
        col_img, col_bar = st.columns([1, 1])
        img_path = Path(cfg["input_dir"]) / str(row["image_path"])
        img = Image.open(img_path).convert("RGB")
        col_img.image(img, caption=str(img_path), use_container_width=True)

        fig = make_bar_figure(
            y_true=targets[row_idx],
            y_pred=preds[row_idx],
            target_cols=target_cols,
            use_log1p_y=use_log1p_y,
            title=f"{row['image_id']} | r2={row['r2_img']:.3f}",
        )
        col_bar.pyplot(fig, clear_figure=True)

    # =====================================================
    # Page 2: Overview
    # =====================================================
    else:
        st.subheader("全体のデータ可視化")

        # Scatter grid
        st.markdown("### ターゲット別散布図（log1p空間）")
        fig_scatter = plot_scatter_grid(targets, preds, target_cols)
        st.pyplot(fig_scatter, clear_figure=True)

        # Histogram of per-image r2
        st.markdown("### 画像スコア（r2_img）の分布（診断用）")
        fig_hist, ax = plt.subplots(figsize=(8, 3))
        ax.hist(score_df["r2_img"].values, bins=30)
        ax.set_xlabel("r2_img")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
        fig_hist.tight_layout()
        st.pyplot(fig_hist, clear_figure=True)

        # Top/Worst
        st.markdown("### TOP / WORST サンプル（Nx2表示）")
        col_top, col_worst = st.columns(2)

        fig_top = plot_top_worst_figure(
            score_df=score_df,
            preds=preds,
            targets=targets,
            target_cols=target_cols,
            input_dir=cfg["input_dir"],
            mode="TOP",
            n=top_n,
            use_log1p_y=use_log1p_y,
        )
        col_top.pyplot(fig_top, clear_figure=True)

        fig_worst = plot_top_worst_figure(
            score_df=score_df,
            preds=preds,
            targets=targets,
            target_cols=target_cols,
            input_dir=cfg["input_dir"],
            mode="WORST",
            n=top_n,
            use_log1p_y=use_log1p_y,
        )
        col_worst.pyplot(fig_worst, clear_figure=True)

        # テーブルも出す（軽く）
        st.markdown("### TOP/WORST 一覧（テーブル）")
        col1, col2 = st.columns(2)
        col1.dataframe(score_df.head(top_n)[["label", "State", "Sampling_Date", "sse"]], use_container_width=True)
        col2.dataframe(score_df.tail(top_n).sort_values("r2_img")[["label", "State", "Sampling_Date", "sse"]], use_container_width=True)


if __name__ == "__main__":
    main()
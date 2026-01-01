from __future__ import annotations

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

# original
import sys
sys.path.append(r"..")

from utils.data import (
    sep, show_df, set_seed,
    save_config_yaml,
)

date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")


# ============================================================
# Helpers: config access
# ============================================================
def cfg_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """dictのネストを 'a.b.c' で安全に取る."""
    cur = d
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ============================================================
# Helpers: group key / stratify label
# ============================================================
def build_group_array(
    df: pd.DataFrame,
    group_cols: List[str],
    sampling_date_group: str = "day",
) -> np.ndarray:
    """
    group_cols を結合して groups を作る。
    sampling_date_group:
      - "day": Sampling_Dateそのまま
      - "month": year_month を使う
    """
    parts = []
    for c in group_cols:
        if c == "Sampling_Date" and sampling_date_group == "month":
            if "year_month" not in df.columns:
                raise KeyError("year_month is missing. date decomposition failed.")
            parts.append(df["year_month"].astype(str).values)
        else:
            if c not in df.columns:
                raise KeyError(f"group col '{c}' not in df columns.")
            parts.append(df[c].astype(str).values)

    groups = np.array(["_".join(xs) for xs in zip(*parts)], dtype=object)
    return groups


def build_stratify_labels(
    df: pd.DataFrame,
    mode: str,
    state_col: str,
    target_col: str,
    use_log1p: bool,
    n_bins: int,
    presence_targets: List[str],
    presence_threshold_g: float,
    add_state: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    StratifiedGroupKFold 用 y を作る。

    mode:
      - "target"      : targetのbin + presence flags (+ optional state)
      - "state"       : stateのみ
      - "state_target": state + target bin (+ presence flags)

    add_state:
      mode=target でも state を label に混ぜたい場合に true
      (ただしカテゴリ数が爆発して失敗しやすいので、基本は score 側で state バランスを見るのが安定)
    """
    mode = str(mode)

    # --- state id ---
    states = sorted(df[state_col].dropna().unique().tolist())
    state2id = {s: i for i, s in enumerate(states)}
    state_id = df[state_col].map(state2id).fillna(-1).astype(int).values
    n_states = len(states)

    info: Dict[str, Any] = {"mode": mode, "states": states, "n_states": n_states}

    if mode == "state":
        y = state_id
        info["used_bins"] = None
        info["n_classes"] = int(len(np.unique(y)))
        info["total_bin"] = None
        return y, info

    # --- target bin ---
    if target_col not in df.columns:
        raise KeyError(f"target_col='{target_col}' not found in df columns.")
    y_main = df[target_col].astype(float).values
    y_main_for_bin = np.log1p(np.clip(y_main, 0, None)) if bool(use_log1p) else y_main

    total_bin = pd.qcut(
        y_main_for_bin,
        q=int(n_bins),
        labels=False,
        duplicates="drop",
    )
    total_bin = np.asarray(total_bin, dtype=int)
    used_bins = int(pd.Series(total_bin).nunique())

    # --- presence flags ---
    flags = np.zeros(len(df), dtype=int)
    for i, t in enumerate(presence_targets):
        if t not in df.columns:
            raise KeyError(f"presence target '{t}' not found in df columns.")
        present = (df[t].fillna(0).astype(float).values > float(presence_threshold_g)).astype(int)
        flags |= (present << i)

    base = total_bin * (2 ** len(presence_targets)) + flags

    # --- combine with state if needed ---
    if mode == "state_target":
        y = base * n_states + state_id
    else:
        # mode == "target"
        if add_state:
            y = base * n_states + state_id
        else:
            y = base

    info["used_bins"] = used_bins
    info["n_classes"] = int(len(np.unique(y)))
    info["total_bin"] = total_bin
    return y, info


# ============================================================
# Split evaluation metrics (Plan1 core)
# ============================================================
def _l1_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))


def compute_split_metrics(
    df: pd.DataFrame,
    folds: np.ndarray,
    n_splits: int,
    state_col: str,
    target_col: str,
    total_bin: Optional[np.ndarray],
    zero_targets: List[str],
    high_total_quantile: float = 0.85,
) -> Dict[str, Any]:
    """
    split の “偏り” を数値化する。
    - mean と max を両方返す（fold1-3も含めて事故foldを抑えるため）
    """
    out: Dict[str, Any] = {}

    # --- fold sizes ---
    fold_counts = np.bincount(folds, minlength=n_splits).astype(float)
    out["fold_min"] = int(fold_counts.min())
    out["fold_max"] = int(fold_counts.max())
    out["size_cv"] = float(np.std(fold_counts) / (np.mean(fold_counts) + 1e-9))

    # --- state distribution L1 ---
    states = sorted(df[state_col].dropna().unique().tolist())
    overall_state = (
        df[state_col].value_counts(normalize=True)
        .reindex(states, fill_value=0)
        .values.astype(float)
    )

    state_l1_per_fold = []
    missing_state_pairs = 0
    for f in range(n_splits):
        sub = df.iloc[np.where(folds == f)[0]]
        vc = sub[state_col].value_counts(normalize=True).reindex(states, fill_value=0).values.astype(float)
        state_l1_per_fold.append(_l1_dist(vc, overall_state))
        # fold内に存在しないstate数
        missing_state_pairs += int((sub[state_col].value_counts().reindex(states, fill_value=0) == 0).sum())

    out["state_l1_mean"] = float(np.mean(state_l1_per_fold))
    out["state_l1_max"] = float(np.max(state_l1_per_fold))
    out["missing_state_pairs"] = int(missing_state_pairs)

    # --- target bin distribution L1（参考）---
    if total_bin is not None:
        bins = sorted(pd.Series(total_bin).unique().tolist())
        overall_bin = (
            pd.Series(total_bin).value_counts(normalize=True)
            .reindex(bins, fill_value=0)
            .values.astype(float)
        )
        bin_l1_per_fold = []
        for f in range(n_splits):
            idx = np.where(folds == f)[0]
            vc = (
                pd.Series(total_bin[idx]).value_counts(normalize=True)
                .reindex(bins, fill_value=0)
                .values.astype(float)
            )
            bin_l1_per_fold.append(_l1_dist(vc, overall_bin))
        out["bin_l1_mean"] = float(np.mean(bin_l1_per_fold))
        out["bin_l1_max"] = float(np.max(bin_l1_per_fold))
    else:
        out["bin_l1_mean"] = np.nan
        out["bin_l1_max"] = np.nan

    # --- zero rate deviations (mean / max) ---
    # true=0 が多いターゲット（clover/dead）の偏り抑制
    zero_rate_mae_list = []
    zero_rate_maxdev_list = []
    for t in zero_targets:
        if t not in df.columns:
            continue
        z_overall = float((df[t].fillna(0).astype(float).values <= 0.0).mean())
        devs = []
        for f in range(n_splits):
            sub = df.iloc[np.where(folds == f)[0]]
            z = float((sub[t].fillna(0).astype(float).values <= 0.0).mean())
            devs.append(abs(z - z_overall))
        zero_rate_mae_list.append(float(np.mean(devs)))
        zero_rate_maxdev_list.append(float(np.max(devs)))

    out["zero_rate_mae"] = float(np.mean(zero_rate_mae_list)) if len(zero_rate_mae_list) else 0.0
    out["zero_rate_maxdev"] = float(np.mean(zero_rate_maxdev_list)) if len(zero_rate_maxdev_list) else 0.0

    # --- high total rate deviations (mean / max) ---
    if target_col in df.columns:
        y = df[target_col].astype(float).values
        thr = float(np.quantile(y, float(high_total_quantile)))
        overall_rate = float((y >= thr).mean())

        devs = []
        for f in range(n_splits):
            sub = df.iloc[np.where(folds == f)[0]]
            rate = float((sub[target_col].astype(float).values >= thr).mean())
            devs.append(abs(rate - overall_rate))

        out["high_rate_mae"] = float(np.mean(devs))
        out["high_rate_maxdev"] = float(np.max(devs))
        out["high_total_thr"] = float(thr)
    else:
        out["high_rate_mae"] = 0.0
        out["high_rate_maxdev"] = 0.0
        out["high_total_thr"] = np.nan

    return out


def compute_split_score(metrics: Dict[str, Any], w: Dict[str, float]) -> float:
    """
    小さいほど良い（偏りが少ない）。
    meanだけでなく maxdev も評価することで、fold1-3も事故りにくくする。
    """
    score = 0.0
    score += float(w.get("size_cv", 1.0)) * float(metrics.get("size_cv", 0.0))

    score += float(w.get("state_l1_mean", 1.0)) * float(metrics.get("state_l1_mean", 0.0))
    score += float(w.get("state_l1_max", 0.7)) * float(metrics.get("state_l1_max", 0.0))
    score += float(w.get("missing_state_pairs", 2.0)) * float(metrics.get("missing_state_pairs", 0.0))

    # binは補助（StratifiedGroupKFoldを使うので基本小さいが、maxは事故検知に効く）
    score += float(w.get("bin_l1_mean", 0.3)) * float(metrics.get("bin_l1_mean", 0.0) if pd.notna(metrics.get("bin_l1_mean")) else 0.0)
    score += float(w.get("bin_l1_max", 0.3)) * float(metrics.get("bin_l1_max", 0.0) if pd.notna(metrics.get("bin_l1_max")) else 0.0)

    score += float(w.get("zero_rate_mae", 0.7)) * float(metrics.get("zero_rate_mae", 0.0))
    score += float(w.get("zero_rate_maxdev", 0.7)) * float(metrics.get("zero_rate_maxdev", 0.0))

    score += float(w.get("high_rate_mae", 0.7)) * float(metrics.get("high_rate_mae", 0.0))
    score += float(w.get("high_rate_maxdev", 0.7)) * float(metrics.get("high_rate_maxdev", 0.0))
    return float(score)


def relabel_folds_make_fold0_representative(
    df: pd.DataFrame,
    folds: np.ndarray,
    n_splits: int,
    state_col: str,
    target_col: str,
    zero_targets: List[str],
    high_total_quantile: float,
) -> np.ndarray:
    """
    fold番号を付け替えて、fold0が全体分布に最も近いfoldになるようにする。
    ※ foldの中身は変わらず「番号だけ」変える。
    """
    # overall stats
    states = sorted(df[state_col].dropna().unique().tolist())
    overall_state = (
        df[state_col].value_counts(normalize=True)
        .reindex(states, fill_value=0)
        .values.astype(float)
    )
    y = df[target_col].astype(float).values
    thr = float(np.quantile(y, float(high_total_quantile)))
    overall_high = float((y >= thr).mean())

    overall_zero = {}
    for t in zero_targets:
        if t in df.columns:
            overall_zero[t] = float((df[t].fillna(0).astype(float).values <= 0.0).mean())

    rep_scores = []
    for f in range(n_splits):
        sub = df.iloc[np.where(folds == f)[0]]

        # state L1
        vc = sub[state_col].value_counts(normalize=True).reindex(states, fill_value=0).values.astype(float)
        s_l1 = _l1_dist(vc, overall_state)

        # high total dev
        high = float((sub[target_col].astype(float).values >= thr).mean())
        high_dev = abs(high - overall_high)

        # zero dev
        z_devs = []
        for t, z_all in overall_zero.items():
            z = float((sub[t].fillna(0).astype(float).values <= 0.0).mean())
            z_devs.append(abs(z - z_all))
        z_dev = float(np.mean(z_devs)) if len(z_devs) else 0.0

        rep = s_l1 + 0.5 * high_dev + 0.5 * z_dev
        rep_scores.append(rep)

    order = np.argsort(rep_scores)  # 小さいほど代表
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    new_folds = np.array([mapping[int(f)] for f in folds], dtype=int)
    return new_folds


# ============================================================
# Fold builder (single seed)
# ============================================================
def build_folds_for_seed(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
    groups: np.ndarray,
    stratify_cfg: Dict[str, Any],
    fallback_to_groupkfold: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    1 seed で split を作る。
    """
    bins_candidates: List[int] = list(stratify_cfg.get("bins_candidates", [8, 6, 5, 4, 3]))
    mode = str(stratify_cfg.get("mode", "target"))
    state_col = str(stratify_cfg.get("state_col", "State"))
    target_col = str(stratify_cfg.get("target", "Dry_Total_g"))
    use_log1p = bool(stratify_cfg.get("use_log1p", True))
    presence_targets = list(stratify_cfg.get("presence_targets", ["Dry_Clover_g", "Dry_Dead_g"]))
    presence_threshold_g = float(stratify_cfg.get("presence_threshold_g", 0.0))
    add_state = bool(stratify_cfg.get("add_state", False))

    # mode=state なら bins を使わない
    if mode == "state":
        y, info = build_stratify_labels(
            df=df,
            mode=mode,
            state_col=state_col,
            target_col=target_col,
            use_log1p=use_log1p,
            n_bins=1,
            presence_targets=[],
            presence_threshold_g=presence_threshold_g,
            add_state=False,
        )
        # ★ 追加：クラス最小数チェック（これでwarningが出る設定を弾く）
        vc = pd.Series(y).value_counts()
        min_count = int(vc.min())
        info["min_class_count"] = min_count
        if min_count < n_splits:
            raise ValueError(
                f"stratify label too sparse: min_class_count={min_count} < n_splits={n_splits} "
                f"(mode={mode}, presence_targets={presence_targets})"
            )

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        folds = np.full(len(df), -1, dtype=int)
        for fold, (_, va_idx) in enumerate(sgkf.split(X=np.zeros((len(df), 1)), y=y, groups=groups)):
            folds[va_idx] = fold
        info["seed"] = int(seed)
        return folds, info

    # mode=target / state_target
    last_err = None
    for nb in bins_candidates:
        try:
            y, info = build_stratify_labels(
                df=df,
                mode=mode,
                state_col=state_col,
                target_col=target_col,
                use_log1p=use_log1p,
                n_bins=int(nb),
                presence_targets=presence_targets,
                presence_threshold_g=presence_threshold_g,
                add_state=add_state,
            )
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            folds = np.full(len(df), -1, dtype=int)
            for fold, (_, va_idx) in enumerate(sgkf.split(X=np.zeros((len(df), 1)), y=y, groups=groups)):
                folds[va_idx] = fold
            info["seed"] = int(seed)
            info["requested_bins"] = int(nb)
            return folds, info
        except ValueError as e:
            last_err = str(e)

    if fallback_to_groupkfold:
        gkf = GroupKFold(n_splits=n_splits)
        folds = np.full(len(df), -1, dtype=int)
        # GroupKFoldは seed を使わない
        for fold, (_, va_idx) in enumerate(gkf.split(X=np.zeros((len(df), 1)), y=np.zeros(len(df)), groups=groups)):
            folds[va_idx] = fold
        info = {
            "seed": int(seed),
            "fallback": "GroupKFold",
            "error": last_err,
            "used_bins": None,
            "total_bin": None,
            "mode": "groupkfold",
        }
        return folds, info

    return None, {"seed": int(seed), "error": last_err}


# ============================================================
# Seed search (Plan1)
# ============================================================
def generate_seed_candidates(base_seed: int, cfg: Dict[str, Any]) -> List[int]:
    method = str(cfg.get("method", "random"))
    n_trials = int(cfg.get("n_trials", 200))
    seed_start = int(cfg.get("seed_start", 0))
    seed_end = int(cfg.get("seed_end", 50000))

    if "seed_list" in cfg and cfg["seed_list"] is not None and len(cfg["seed_list"]) > 0:
        return [int(x) for x in cfg["seed_list"]]

    if method == "grid":
        return [seed_start + i for i in range(n_trials)]

    # random
    rng = np.random.default_rng(int(base_seed))
    seeds = rng.integers(seed_start, seed_end + 1, size=n_trials).tolist()
    # base_seed も混ぜる（再現性）
    seeds = [int(base_seed)] + [int(s) for s in seeds]
    # unique preserving order
    seen = set()
    uniq = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# ============================================================
# main
# ============================================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Plan1:
      - pivot作成
      - group/stratify 設定に沿って seed search で “偏りの少ないsplit” を選ぶ
      - fold0が代表になるよう番号を入れ替える（任意）
      - df_pivot.csv 保存 + split_search_report.csv 保存
    """
    cfg_dict = OmegaConf.to_container(cfg["001_preprocess_v2"], resolve=True)

    exp = str(cfg_dict.get("exp", "001_preprocess_verXX"))
    base_seed = int(cfg_dict.get("seed", 1129))
    debug = bool(cfg_dict.get("debug", False))

    input_dir = Path(str(cfg_dict["input_dir"]))
    output_dir = Path(str(cfg_dict["output_dir"]))
    pivot_csv_name = str(cfg_dict.get("pivot_csv_name", "df_pivot.csv"))

    n_splits = int(cfg_dict.get("n_splits", 4))
    fold_col = str(cfg_dict.get("fold_col", "Fold"))

    group_cols = list(cfg_dict.get("group_cols", ["State", "Sampling_Date"]))
    sampling_date_group = str(cfg_dict.get("sampling_date_group", "day"))  # day/month

    # stratify settings
    stratify_cfg = dict(cfg_dict.get("stratify", {}))
    # backward compat (old flat config)
    if len(stratify_cfg) == 0:
        stratify_cfg = {
            "mode": "target",
            "state_col": "State",
            "target": str(cfg_dict.get("stratify_target", "Dry_Total_g")),
            "use_log1p": bool(cfg_dict.get("stratify_use_log1p", True)),
            "bins_candidates": list(cfg_dict.get("stratify_bins_candidates", [8, 6, 5, 4, 3])),
            "presence_targets": ["Dry_Clover_g"] if bool(cfg_dict.get("stratify_add_clover_flag", True)) else [],
            "presence_threshold_g": 0.0,
            "add_state": False,
        }

    # seed search
    seed_search_cfg = dict(cfg_dict.get("seed_search", {}))
    seed_search_enabled = bool(seed_search_cfg.get("enabled", True))
    top_k = int(seed_search_cfg.get("top_k", 30))
    score_weights = dict(seed_search_cfg.get("score_weights", {}))
    make_fold0_rep = bool(seed_search_cfg.get("make_fold0_representative", True))

    # score metrics cfg
    zero_targets = list(seed_search_cfg.get("zero_targets", ["Dry_Clover_g", "Dry_Dead_g"]))
    high_total_quantile = float(seed_search_cfg.get("high_total_quantile", 0.85))

    # debug用
    if debug:
        exp = "001_pp_debug"
        seed_search_cfg["n_trials"] = min(int(seed_search_cfg.get("n_trials", 50)), 50)
        top_k = min(top_k, 10)

    # seed
    set_seed(base_seed)

    # make savedir
    savedir = output_dir / exp
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir / "yaml", exist_ok=True)

    # YAMLとして保存
    save_config_yaml(cfg_dict, savedir / "yaml" / "config.yaml")
    print(f"[INFO] Config saved to {(savedir / 'yaml' / 'config.yaml').resolve()}")

    # ==================
    # read data
    # ==================
    sep("Session 1: データの読み込み")
    train_df = pl.read_csv(input_dir / "train.csv")
    sep("train_df"); show_df(train_df, 3, True)

    test_df = pl.read_csv(input_dir / "test.csv")
    sep("test_df"); show_df(test_df, 3, False)

    # ==================
    # preprocessing
    # ==================
    sep("Session 2: 前処理（pivot / date / species）")

    print("\n=== Basic Information ===")
    print(f"Total samples: {len(train_df)}")
    print(f"Unique images: {train_df['image_path'].n_unique()}")
    print(f"Date range: {train_df['Sampling_Date'].min()} to {train_df['Sampling_Date'].max()}")
    print(f"States: {train_df['State'].unique().to_list()}")
    print(f"Unique Species strings: {train_df['Species'].n_unique()}")

    # 2-2. pivot
    print("\n=== Creating pivoted biomass table ===")
    train_df_with_id = train_df.with_columns(
        pl.col("sample_id").str.extract(r"(ID\d+)").alias("image_id")
    )

    train_pivot_df = train_df_with_id.pivot(
        values="target",
        index=[
            "image_id", "image_path", "Sampling_Date", "State", "Species",
            "Pre_GSHH_NDVI", "Height_Ave_cm"
        ],
        on="target_name",
    )

    print(f"Pivoted dataframe shape: {train_pivot_df.shape}")
    show_df(train_pivot_df, 3)

    # null check
    print("\n=== Checking for null values ===")
    show_df(train_pivot_df.null_count())

    # 2-3. Sampling_Date decomposition
    print("\n=== Decomposing Sampling_Date ===")
    train_pivot_df = train_pivot_df.with_columns([
        pl.col("Sampling_Date").str.strptime(pl.Date, "%Y/%m/%d").alias("date"),
    ]).with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.weekday().alias("weekday"),
    ])

    # year_month string (group by month用)
    train_pivot_df = train_pivot_df.with_columns(
        (pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Int64).cast(pl.Utf8)).alias("year_month")
    )

    # weekday name
    weekday_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    train_pivot_df = train_pivot_df.with_columns(
        pl.col("weekday").map_elements(lambda x: weekday_map.get(x, "Unknown"), return_dtype=pl.Utf8).alias("weekday_name")
    )
    show_df(train_pivot_df, 3)

    # 2-4. Species one-hot（そのまま維持）
    sep("Creating One-hot encoding for Species")
    all_species = set()
    for species_str in train_pivot_df["Species"].to_list():
        if species_str and species_str != "null":
            all_species.update(str(species_str).split("_"))
    all_species = sorted(all_species)

    for sp in all_species:
        col_name = f"is_{sp}"
        train_pivot_df = train_pivot_df.with_columns(
            pl.col("Species").cast(pl.Utf8).str.contains(sp).fill_null(False).alias(col_name)
        )

    print(f"[INFO] one-hot species cols: {len(all_species)}")

    # polars -> pandas（split searchはpandasで計算する）
    df_pd = train_pivot_df.to_pandas()

    # ==================
    # Build groups
    # ==================
    sep("Session 3: Group keys")
    groups = build_group_array(df_pd, group_cols=group_cols, sampling_date_group=sampling_date_group)
    n_groups = len(np.unique(groups))
    print(f"[INFO] unique groups = {n_groups}")

    if n_groups < n_splits:
        print(f"[WARN] n_splits={n_splits} > unique groups={n_groups}. reduce n_splits -> {n_groups}")
        n_splits = n_groups
    if n_splits < 2:
        raise ValueError(f"n_splits must be >=2, got {n_splits}")

    # ==================
    # Seed search
    # ==================
    sep("Session 4: Seed search for best split (Plan1)")

    target_col = str(stratify_cfg.get("target", "Dry_Total_g"))
    state_col = str(stratify_cfg.get("state_col", "State"))

    if seed_search_enabled:
        seeds = generate_seed_candidates(base_seed, seed_search_cfg)
        print(f"[INFO] seed_search enabled. candidates={len(seeds)}")
    else:
        seeds = [base_seed]
        print("[INFO] seed_search disabled. use base seed only.")

    rows = []
    best = {"score": 1e18, "seed": None, "folds": None, "info": None, "metrics": None}

    for i, sd in tqdm(enumerate(seeds), total=len(seeds)):
        folds, info = build_folds_for_seed(
            df=df_pd,
            n_splits=n_splits,
            seed=int(sd),
            groups=groups,
            stratify_cfg=stratify_cfg,
            fallback_to_groupkfold=True,
        )
        if folds is None:
            continue

        total_bin = info.get("total_bin", None)
        metrics = compute_split_metrics(
            df=df_pd,
            folds=folds,
            n_splits=n_splits,
            state_col=state_col,
            target_col=target_col,
            total_bin=total_bin,
            zero_targets=zero_targets,
            high_total_quantile=high_total_quantile,
        )
        score = compute_split_score(metrics, score_weights)

        row = {
            "seed": int(sd),
            "score": float(score),
            "mode": str(info.get("mode")),
            "used_bins": info.get("used_bins"),
            "requested_bins": info.get("requested_bins", None),
            "n_classes": info.get("n_classes", None),
            **metrics,
        }
        rows.append(row)

        if score < best["score"]:
            best = {"score": float(score), "seed": int(sd), "folds": folds.copy(), "info": info, "metrics": metrics}

        if debug and i >= 20:
            break

    if len(rows) == 0 or best["folds"] is None:
        raise RuntimeError("No valid split found. Check stratify/group settings.")

    report_df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    report_path = savedir / "split_search_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"[INFO] split search report saved: {report_path}")

    print("\n[INFO] TOP candidates:")
    print(report_df.head(min(top_k, len(report_df)))[
        ["seed", "score", "mode", "used_bins", "size_cv", "state_l1_mean", "state_l1_max",
         "zero_rate_mae", "zero_rate_maxdev", "high_rate_mae", "high_rate_maxdev", "missing_state_pairs"]
    ])

    folds_best = best["folds"]
    print(f"\n[INFO] BEST seed = {best['seed']}  score={best['score']:.6f}")
    print("[INFO] BEST metrics:", best["metrics"])

    # fold0を代表に（番号を付け替え）
    if make_fold0_rep:
        folds_best = relabel_folds_make_fold0_representative(
            df=df_pd,
            folds=folds_best,
            n_splits=n_splits,
            state_col=state_col,
            target_col=target_col,
            zero_targets=zero_targets,
            high_total_quantile=high_total_quantile,
        )
        print("[INFO] relabeled folds to make fold0 representative.")

    # add fold column (pandas -> polars)
    df_pd[fold_col] = folds_best.astype(int)
    df_out = pl.from_pandas(df_pd)

    # ==================
    # Save
    # ==================
    sep("Session 5: Save df_pivot")
    df_out.write_csv(savedir / pivot_csv_name)
    print(f"[INFO] saved: {(savedir / pivot_csv_name).resolve()}")

    # summary
    print("\nFold counts:")
    print(pd.Series(folds_best).value_counts().sort_index().to_dict())
    print("\nFold x State:")
    print(pd.crosstab(df_pd[fold_col], df_pd["State"]))

    # leak check (group_colsベースで確認)
    group_key = groups
    leak = (
        pd.DataFrame({"group_key": group_key, "fold": folds_best})
        .groupby("group_key")["fold"].nunique()
    )
    bad = leak[leak > 1]
    print("\nLeak groups (should be empty):")
    print(bad.head(10))
    if len(bad) > 0:
        print(f"[WARN] leak groups detected: {len(bad)} (group config issue?)")
    else:
        print("[OK] No leakage across folds for given group_cols.")


if __name__ == "__main__":
    main()
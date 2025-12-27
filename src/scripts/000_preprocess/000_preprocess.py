import os, gc, re, yaml, glob, pickle, warnings
import time
import random, math
import joblib, itertools
from pathlib import Path
from datetime import datetime

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
import utils
from utils.data import (
    sep, show_df, glob_walk, set_seed, format_time,
    save_config_yaml, dict_to_namespace
)

date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")

# ===================================
# main
# ===================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Preprocess:
      - train.csv を image単位にpivot
      - Sampling_Date分解
      - Species one-hot
      - StratifiedGroupKFoldで Fold 列付与
      - df_pivot.csv 保存
    """
    # ------------------
    # set config
    # ------------------
    config_dict = OmegaConf.to_container(cfg["000_preprocess"], resolve=True)
    config = dict_to_namespace(config_dict)

    if config.debug:
        config.exp = "000_pp_debug"  # debug用の保存先

    # seed
    set_seed(config.seed)

    # make savedir
    savedir = Path(config.output_dir) / config.exp
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir / "yaml", exist_ok=True)

    # YAMLとして保存
    output_path = Path(savedir / "yaml" / "config.yaml")
    save_config_yaml(config, output_path)
    print(f"Config saved to {output_path.resolve()}")

    # ==================
    # read data
    # ==================
    sep("Session 1: データの読み込み")
    train_df = pl.read_csv(Path(config.input_dir) / "train.csv")
    sep("train_df"); show_df(train_df, 3, True)

    test_df = pl.read_csv(Path(config.input_dir) / "test.csv")
    sep("test_df"); show_df(test_df, 3, False)

    # ==================
    # preprocessing
    # ==================
    sep("Session 2: データの前処理")

    # 2-1. basic info
    print("\n=== Basic Information ===")
    print(f"Total samples: {len(train_df)}")
    print(f"Unique images: {train_df['image_path'].n_unique()}")
    print(f"Date range: {train_df['Sampling_Date'].min()} to {train_df['Sampling_Date'].max()}")
    print(f"\nStates: {train_df['State'].unique().to_list()}")
    print(f"Number of unique species combinations: {train_df['Species'].n_unique()}")

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
    null_counts = train_pivot_df.null_count()
    show_df(null_counts)

    # 2-3. Sampling_Date decomposition
    print("\n=== Decomposing Sampling_Date ===")
    train_pivot_df = train_pivot_df.with_columns([
        pl.col("Sampling_Date").str.strptime(pl.Date, "%Y/%m/%d").alias("date"),
    ]).with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.weekday().alias("weekday"),  # 0=Mon, 6=Sun
    ])

    weekday_map = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}
    train_pivot_df = train_pivot_df.with_columns(
        pl.col("weekday").map_elements(
            lambda x: weekday_map.get(x, "Unknown"),
            return_dtype=pl.Utf8
        ).alias("weekday_name")
    )
    show_df(train_pivot_df, 3)

    # 2-4. Species one-hot
    sep("Creating One-hot encoding for Species")

    print("\n=== Extracting unique species ===")
    all_species = set()
    for species_str in train_pivot_df["Species"].to_list():
        if species_str and species_str != "null":
            all_species.update(species_str.split("_"))
    all_species = sorted(all_species)

    print(f"Number of unique species: {len(all_species)}")
    print(f"Species list: {all_species}")

    print("\n=== Creating one-hot encoding ===")
    for species in all_species:
        col_name = f"is_{species}"
        train_pivot_df = train_pivot_df.with_columns(
            pl.col("Species").str.contains(species).fill_null(False).alias(col_name)
        )

    species_cols = [f"is_{species}" for species in all_species]
    print(f"\nCreated {len(species_cols)} species indicator columns")
    show_df(train_pivot_df.select(["image_id", "Species"] + species_cols[:5]), 3)

    print("\n=== Species frequency ===")
    for species in all_species:
        count = int(train_pivot_df[f"is_{species}"].sum())
        print(f"{species}: {count} images ({count / len(train_pivot_df) * 100:.1f}%)")

    # 2-5. Fold creation
    sep("Session 2-5: Creating CV folds (StratifiedGroupKFold)")

    # config-driven parameters
    n_splits = int(getattr(config, "n_splits", 5))
    fold_col = str(getattr(config, "fold_col", "Fold"))

    group_cols = list(getattr(config, "group_cols", ["State", "Sampling_Date"]))

    stratify_target = str(getattr(config, "stratify_target", "Dry_Total_g"))
    stratify_use_log1p = bool(getattr(config, "stratify_use_log1p", True))
    bins_candidates = list(getattr(config, "stratify_bins_candidates", [8, 6, 5, 4, 3]))

    add_clover_flag = bool(getattr(config, "stratify_add_clover_flag", True))
    clover_target = str(getattr(config, "clover_target", "Dry_Clover_g"))

    # ---- group key ----
    # groups = State + Sampling_Date など（config.group_colsで指定）
    group_lists = [train_pivot_df[c].to_list() for c in group_cols]
    groups = np.array(
        ["_".join(map(str, vals)) for vals in zip(*group_lists)],
        dtype=object
    )

    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        print(f"[WARN] n_splits={n_splits} > unique groups={n_groups}. Reduce n_splits -> {n_groups}")
        n_splits = n_groups
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    # ---- stratify label ----
    if stratify_target not in train_pivot_df.columns:
        raise KeyError(f"stratify_target='{stratify_target}' not found in pivoted df columns.")

    y_main = np.asarray(train_pivot_df[stratify_target].to_list(), dtype=float)
    y_main_for_bin = np.log1p(y_main) if stratify_use_log1p else y_main

    if add_clover_flag:
        if clover_target not in train_pivot_df.columns:
            raise KeyError(f"clover_target='{clover_target}' not found in pivoted df columns.")
        y_clover = np.asarray(train_pivot_df[clover_target].to_list(), dtype=float)
        clover_flag = (y_clover > 0).astype(int)
    else:
        clover_flag = None

    folds = None
    used_bins = None

    # try stratified-group with several bins; fallback to GroupKFold if fail
    for n_bins in bins_candidates:
        try:
            total_bin = pd.qcut(
                y_main_for_bin,
                q=int(n_bins),
                labels=False,
                duplicates="drop"
            )
            total_bin = np.asarray(total_bin, dtype=int)

            if add_clover_flag:
                y_strat = total_bin * 2 + clover_flag
            else:
                y_strat = total_bin

            sgkf = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=config.seed
            )

            folds = np.full(len(train_pivot_df), -1, dtype=np.int16)
            for fold, (_, va_idx) in enumerate(sgkf.split(
                X=np.zeros((len(train_pivot_df), 1)),
                y=y_strat,
                groups=groups
            )):
                folds[va_idx] = fold

            used_bins = int(n_bins)
            break

        except ValueError as e:
            print(f"[WARN] StratifiedGroupKFold failed with n_bins={n_bins}: {e}")
            folds = None

    if folds is None:
        print("[WARN] fallback to GroupKFold (no stratification)")
        gkf = GroupKFold(n_splits=n_splits)
        folds = np.full(len(train_pivot_df), -1, dtype=np.int16)
        for fold, (_, va_idx) in enumerate(gkf.split(
            X=np.zeros((len(train_pivot_df), 1)),
            y=y_main,
            groups=groups
        )):
            folds[va_idx] = fold

    # add Fold column
    train_pivot_df = train_pivot_df.with_columns(
        pl.Series(name=fold_col, values=folds.astype(int).tolist())
    )

    # summary
    print(f"\nFold created: n_splits={n_splits}, seed={config.seed}, used_bins={used_bins}, fold_col='{fold_col}'")
    fold_counts = pd.Series(folds).value_counts().sort_index()
    print("Fold counts:", fold_counts.to_dict())

    tmp_pd = pd.DataFrame({
        fold_col: folds.astype(int),
        "State": train_pivot_df["State"].to_list(),
    })
    print("\nFold x State:")
    print(pd.crosstab(tmp_pd[fold_col], tmp_pd["State"]))

    # ==================
    # save
    # ==================
    sep("Session 3: データの保存")
    pivot_csv_name = str(getattr(config, "pivot_csv_name", "df_pivot.csv"))
    train_pivot_df.write_csv(savedir / pivot_csv_name)

    sep("saved_csv")
    show_df(train_pivot_df, 3, True)
    # group_key を作って、同一groupが複数foldに出てないかチェック
    train_pivot_df = train_pivot_df.with_columns(
        (pl.col("State").cast(pl.Utf8) + "_" + pl.col("Sampling_Date").cast(pl.Utf8)).alias("group_key")
    )

    leak = (
        train_pivot_df
        .group_by("group_key")
        .agg(pl.col("Fold").n_unique().alias("n_folds"))
        .filter(pl.col("n_folds") > 1)
    )

    print("Leak groups (should be empty):")
    print(leak)

if __name__ == "__main__":
    main()
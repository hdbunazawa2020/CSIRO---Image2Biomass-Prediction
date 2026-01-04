# /mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction/src/scripts/200_wandb_sweep/200_wandb_sweep.py
# -*- coding: utf-8 -*-
"""
W&B sweep runner（v3）
- alpha_raw_total を sweep 対象に追加
- 高Dry_Totalを oversample する WeightedRandomSampler を sweep 対象に追加
- OOF 上で postprocess mode ("delta"|"none"|"sum_fix") を比較してログ
  - Clover/Dead の 0落とし閾値も OOF から grid search で最適化して mode 比較する

注意:
- このスクリプトは「高速スイープ（fold0のみ等）」を想定
- postprocess の grid search は val set が小さければ十分軽いが、
  folds を増やす・grid_n を増やすと重くなるので要注意
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from omegaconf import DictConfig, OmegaConf

# =========================================================
# Path 設定（src を import するため）
# =========================================================
SRC_DIR = Path(__file__).resolve().parents[2]  # .../src
sys.path.append(str(SRC_DIR))

from utils.data import set_seed
from utils.losses import WeightedMSELoss, MixedLogRawLoss
from utils.train_utils import build_optimizer, build_scheduler
from utils.metric import global_weighted_r2_score  # 公式実装を使う（valid_one_epochと同じ）

from datasets.dataset import CsiroDataset
from datasets.transforms import build_transforms
from training.train import train_one_epoch, valid_one_epoch

from models.convnext_regressor import ConvNeXtRegressor

# ★追加：oversample（WeightedRandomSampler）
from utils.sampling import make_total_oversample_weights, build_weighted_sampler

# ★追加：OOF後処理（Clover/Deadの0落とし + mode補正）
from utils.zero_threshold import apply_zero_thresholds


# =========================================================
# EMA helper（ema_decay==0 を「無効」として扱う）
# =========================================================
def unwrap_model(model: nn.Module) -> nn.Module:
    """DDP / DataParallel だった場合に中身を取り出す。"""
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, nn.DataParallel)):
        return model.module
    return model


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """EMA更新（指数移動平均）。

    注意:
        state_dict には BatchNorm の num_batches_tracked (Long) など、
        float ではない buffer も含まれます。
        それらに対して mul_/add_ を行うと dtype 変換エラーになります。

    対応:
        - float系（fp16/fp32/bf16/fp64）のみ EMA 更新
        - それ以外（Long/Bool/Int等）は「そのままコピー」
    """
    msd = unwrap_model(model).state_dict()
    esd = ema_model.state_dict()

    for k, v_ema in esd.items():
        if k not in msd:
            continue

        v_src = msd[k]

        # EMAは float 系だけに適用
        if torch.is_floating_point(v_ema):
            # dtype が違う可能性に備えて合わせる（安全）
            esd[k].mul_(decay).add_(v_src.to(dtype=v_ema.dtype), alpha=(1.0 - decay))
        else:
            # Long / Bool などは EMA できないので、そのまま追従させる
            esd[k].copy_(v_src)

    ema_model.load_state_dict(esd, strict=True)


# =========================================================
# postprocess: OOFから閾値をgrid searchして mode を比較
# =========================================================
def _build_threshold_grid_from_true_zero(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx: int,
    *,
    grid_n: int = 20,
    q_start: float = 0.50,
    q_end: float = 0.995,
) -> np.ndarray:
    """true==0 の予測分布から threshold 候補gridを作る（quantileベース）。

    Args:
        y_true: (N, K) raw
        y_pred: (N, K) raw
        idx: 対象列index
        grid_n: quantile点数（大きいほど探索増）
        q_start/q_end: quantile範囲

    Returns:
        grid: 1D float array（0.0含む、昇順、重複除去済み）
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask0 = (y_true[:, idx] == 0.0)
    pred0 = y_pred[mask0, idx]

    # true==0 が存在しない場合は 0.0 だけ
    if pred0.size == 0:
        return np.array([0.0], dtype=np.float64)

    qs = np.linspace(float(q_start), float(q_end), int(grid_n))
    cand = np.quantile(pred0, qs)

    # 小数を丸めて重複除去（gridが増え過ぎるのを防ぐ）
    cand = np.unique(np.round(cand.astype(np.float64), 6))

    # 0.0は必ず候補に入れる
    cand = np.unique(np.concatenate([np.array([0.0], dtype=np.float64), cand]))

    # 念のため非負だけ残す
    cand = cand[cand >= 0.0]
    return cand


def fit_zero_thresholds_grid_metric(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    *,
    target_cols: List[str],
    weights: List[float],
    mode: str,
    grid_n: int = 20,
    targets: Tuple[str, str] = ("Dry_Clover_g", "Dry_Dead_g"),
) -> Tuple[float, Dict[str, float]]:
    """OOF上で Clover/Dead の0落とし閾値を2D grid searchして最適化する。

    重要:
      - スコア計算は utils.metric.global_weighted_r2_score を使用（本番と揃える）
      - mode によって GDM/Total の扱いが変わるため、modeごとに最適閾値が変わり得る

    Args:
        y_true_raw: (N,K) raw正解
        y_pred_raw: (N,K) raw予測
        target_cols: 列順
        weights: metric weights
        mode: "none" | "delta" | "sum_fix"
        grid_n: 各閾値の候補数（目安: 10〜30）
        targets: 閾値探索する2列名（Clover/Dead）

    Returns:
        best_score: float
        best_thr: dict 例 {"Dry_Clover_g": 0.12, "Dry_Dead_g": 0.30}
    """
    cols = list(target_cols)
    w = np.asarray(weights, dtype=np.float64)

    if targets[0] not in cols or targets[1] not in cols:
        raise KeyError(f"targets {targets} must exist in target_cols={cols}")

    i1 = cols.index(targets[0])
    i2 = cols.index(targets[1])

    g1 = _build_threshold_grid_from_true_zero(y_true_raw, y_pred_raw, i1, grid_n=grid_n)
    g2 = _build_threshold_grid_from_true_zero(y_true_raw, y_pred_raw, i2, grid_n=grid_n)

    # ベース（閾値0）も含めて探索
    best_score = -np.inf
    best_thr = {targets[0]: 0.0, targets[1]: 0.0}

    for t1 in g1:
        for t2 in g2:
            thr = {targets[0]: float(t1), targets[1]: float(t2)}
            pred_pp = apply_zero_thresholds(
                preds_raw=y_pred_raw,
                target_cols=cols,
                thresholds=thr,
                mode=str(mode),
                clip_nonneg=True,
            )
            score = float(global_weighted_r2_score(y_true_raw, pred_pp, w))
            if score > best_score:
                best_score = score
                best_thr = thr

    return float(best_score), best_thr


def eval_postprocess_modes_on_oof(
    oof_true: np.ndarray,
    oof_pred: np.ndarray,
    *,
    target_cols: List[str],
    weights: List[float],
    grid_n: int,
    modes: List[str],
    targets: Tuple[str, str] = ("Dry_Clover_g", "Dry_Dead_g"),
) -> Dict[str, Any]:
    """OOFに対して postprocess mode を比較する。

    - modeごとに Clover/Dead の閾値を grid search 最適化
    - modeごとの best score を比較し、best mode を返す

    Args:
        oof_true: (N,K) raw
        oof_pred: (N,K) raw
        target_cols: 列順
        weights: metric weights
        grid_n: threshold gridの粗さ
        modes: ["none","delta","sum_fix"] など
        targets: 閾値最適化対象

    Returns:
        result: dict
            {
              "scores": {"none":..., "delta":..., "sum_fix":...},
              "thresholds": {"none": {...}, ...},
              "best_mode": "...",
              "best_score": float,
            }
    """
    scores: Dict[str, float] = {}
    thrs: Dict[str, Dict[str, float]] = {}

    for m in modes:
        s, t = fit_zero_thresholds_grid_metric(
            y_true_raw=oof_true,
            y_pred_raw=oof_pred,
            target_cols=target_cols,
            weights=weights,
            mode=m,
            grid_n=grid_n,
            targets=targets,
        )
        scores[str(m)] = float(s)
        thrs[str(m)] = {k: float(v) for k, v in t.items()}

    best_mode = max(scores.keys(), key=lambda k: scores[k])
    best_score = float(scores[best_mode])

    return {
        "scores": scores,
        "thresholds": thrs,
        "best_mode": best_mode,
        "best_score": best_score,
    }


# =========================================================
# Sweep config（bayes / maximize best/weighted_r2）
# =========================================================
def build_sweep_config(project: str) -> Dict[str, Any]:
    """W&B sweep の設定辞書を生成する。

    今回の追加パラメータ:
      - alpha_raw_total: Total専用 raw MSE boost
      - oversample_*: Dry_Total高値をoversampleする設定
    """
    return {
        "name": f"{project}-sweep-v3",
        "method": "bayes",
        "metric": {"name": "best/weighted_r2", "goal": "maximize"},
        "parameters": {
            # -------------------------
            # model
            # -------------------------
            "backbone": {
                "values": [
                    "convnext_small",
                    "convnext_base",
                    "swin_tiny_patch4_window7_224",
                    "tf_efficientnetv2_s",
                ]
            },
            "img_size": {"values": [224, 288, 320]},
            "head_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},

            # -------------------------
            # optimizer
            # -------------------------
            "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-4},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 5e-2},

            # -------------------------
            # EMA
            # -------------------------
            "ema_decay": {"values": [0.0, 0.95, 0.97, 0.99, 0.995]},

            # -------------------------
            # augmentation（必要なら増やしてOK）
            # -------------------------
            "hflip_p": {"values": [0.0, 0.25, 0.5]},
            "shift_scale_rotate_p": {"values": [0.0, 0.2, 0.5]},
            "rotate_limit": {"values": [0, 10, 20]},
            "color_jitter_p": {"values": [0.0, 0.2, 0.4]},

            # -------------------------
            # ★追加：loss（Total専用 raw_mse boost）
            # -------------------------
            "alpha_raw_total": {"distribution": "uniform", "min": 0.0, "max": 0.8},

            # -------------------------
            # ★追加：oversample（Dry_Total高値を多く見る）
            # -------------------------
            "oversample_enabled": {"values": [0, 1]},
            "oversample_strategy": {"values": ["ramp", "inv_freq"]},
            "oversample_n_bins": {"values": [4, 6, 8, 10]},
            "oversample_max_mult": {"distribution": "uniform", "min": 2.0, "max": 8.0},

            # -------------------------
            # 参考：MixUp/CutMix（※今は触らないなら削ってOK）
            # -------------------------
            "mix_prob": {"distribution": "uniform", "min": 0.0, "max": 0.2},
            "mix_mode": {"values": ["none", "mixup", "cutmix", "both"]},
            "mix_alpha": {"values": [0.2, 0.4, 1.0]},
        },
    }


# =========================================================
# config 読み込み & sweep param を反映
# =========================================================
def load_base_cfg(base_cfg_path: str) -> DictConfig:
    """base yaml を読み込む。"""
    base_cfg = OmegaConf.load(base_cfg_path)
    return base_cfg


def apply_wandb_overrides(cfg: DictConfig, wb: wandb.sdk.wandb_config.Config) -> DictConfig:
    """wandb.config の値で cfg を上書きする（安全に必要分だけ）。"""
    cfg = copy.deepcopy(cfg)

    # ---- model ----
    cfg.model.backbone = str(wb.backbone)
    cfg.img_size = int(wb.img_size)
    cfg.model.head_dropout = float(wb.head_dropout)

    # ---- optimizer ----
    cfg.optimizer.base_lr = float(wb.lr)
    cfg.optimizer.weight_decay = float(wb.weight_decay)

    # scheduler を「一定LR」に寄せる（必要なら）
    if "scheduler" in cfg and "base_lr" in cfg.scheduler:
        cfg.scheduler.base_lr = float(wb.lr)
        cfg.scheduler.max_lr = float(wb.lr)
        cfg.scheduler.min_lr = float(wb.lr)

    # ---- EMA ----
    cfg.ema.decay = float(wb.ema_decay)
    cfg.ema.enabled = bool(cfg.ema.decay > 0.0)

    # ---- augmentation ----
    cfg.augment.train.hflip_p = float(wb.hflip_p)
    cfg.augment.train.shift_scale_rotate_p = float(wb.shift_scale_rotate_p)
    cfg.augment.train.rotate_limit = int(wb.rotate_limit)
    cfg.augment.train.color_jitter_p = float(wb.color_jitter_p)

    # ---- MixUp / CutMix（現状キーがtrain.pyと一致してないなら別途調整してください）----
    # ---- MixUp / CutMix ----
    if "mixing" not in cfg:
        cfg.mixing = OmegaConf.create({})

    mix_prob = float(wb.mix_prob)
    mix_mode = str(wb.mix_mode).lower()
    mix_alpha = float(wb.mix_alpha)

    # "none" のときは無効
    if mix_mode in ("none", "off", "false", "0") or mix_prob <= 0.0:
        cfg.mixing.enabled = False
        cfg.mixing.p = 0.0
        cfg.mixing.mode = "mixup_cutmix"
        cfg.mixing.mixup_alpha = 0.4
        cfg.mixing.cutmix_alpha = 1.0
        cfg.mixing.switch_prob = 0.5
    else:
        cfg.mixing.enabled = True
        cfg.mixing.p = mix_prob

        # mode を train.py の期待に合わせる
        if mix_mode == "mixup":
            cfg.mixing.mode = "mixup"
            cfg.mixing.mixup_alpha = mix_alpha
            cfg.mixing.cutmix_alpha = 1.0
            cfg.mixing.switch_prob = 0.0
        elif mix_mode == "cutmix":
            cfg.mixing.mode = "cutmix"
            cfg.mixing.mixup_alpha = 0.4
            cfg.mixing.cutmix_alpha = mix_alpha
            cfg.mixing.switch_prob = 1.0
        elif mix_mode in ("both", "mixup_cutmix", "mixup+cutmix"):
            cfg.mixing.mode = "mixup_cutmix"
            cfg.mixing.mixup_alpha = mix_alpha
            cfg.mixing.cutmix_alpha = mix_alpha
            cfg.mixing.switch_prob = 0.5
        else:
            raise ValueError(f"Unknown mix_mode: {mix_mode}")

    # ---- ★追加：alpha_raw_total ----
    if "loss" not in cfg:
        cfg.loss = OmegaConf.create({})
    cfg.loss.alpha_raw_total = float(wb.alpha_raw_total)
    # Total index は target_cols から自動推定（なければ-1）
    try:
        cfg.loss.total_index = int(list(cfg.target_cols).index("Dry_Total_g"))
    except Exception:
        cfg.loss.total_index = -1

    # ---- ★追加：oversample ----
    if "oversample" not in cfg:
        cfg.oversample = OmegaConf.create({})
    cfg.oversample.enabled = bool(int(wb.oversample_enabled) == 1)
    cfg.oversample.strategy = str(wb.oversample_strategy)
    cfg.oversample.n_bins = int(wb.oversample_n_bins)
    cfg.oversample.max_mult = float(wb.oversample_max_mult)
    cfg.oversample.min_mult = float(getattr(cfg.oversample, "min_mult", 1.0))
    cfg.oversample.total_col = str(getattr(cfg.oversample, "total_col", "Dry_Total_g"))

    return cfg


# =========================================================
# 1 run = 1 trial
# =========================================================
def run_one_trial(base_cfg: DictConfig) -> None:
    """wandb agent から呼ばれる 1 trial。"""
    run = wandb.init()
    wb = wandb.config

    # ---- cfg を反映 ----
    cfg = apply_wandb_overrides(base_cfg, wb)

    # run名（見やすく）
    os_tag = "osOFF"
    if bool(getattr(cfg, "oversample", {}).get("enabled", False)):
        os_tag = f"os{cfg.oversample.strategy}-b{cfg.oversample.n_bins}-m{cfg.oversample.max_mult:.1f}"

    run_name = (
        f"{cfg.model.backbone}__img{cfg.img_size}__"
        f"{os_tag}__"
        f"arT{float(getattr(cfg.loss, 'alpha_raw_total', 0.0)):.2f}__"
        f"ema{cfg.ema.decay:.3f}__lr{cfg.optimizer.base_lr:.1e}"
    )
    try:
        wandb.run.name = run_name
    except Exception:
        pass

    # ---- seed ----
    set_seed(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- df_pivot ----
    pp_dir = Path(str(cfg.pp_dir)) / str(cfg.preprocess_ver)
    pivot_path = pp_dir / str(cfg.pivot_csv_name)
    df = pd.read_csv(pivot_path)

    # ---- transforms ----
    train_tfm = build_transforms(cfg, is_train=True)
    valid_tfm = build_transforms(cfg, is_train=False)

    # ---- loss ----
    if str(cfg.loss.name).lower() == "weighted_mse":
        loss_fn: nn.Module = WeightedMSELoss(list(cfg.loss.weights)).to(device)
    else:
        # mixed_log_raw（あなたの既存想定）
        loss_fn = MixedLogRawLoss(
            weights=list(cfg.loss.weights),
            alpha_raw=float(cfg.loss.alpha_raw),
            raw_loss=str(cfg.loss.raw_loss),
            raw_huber_beta=float(cfg.loss.raw_huber_beta),
            log_clip_min=float(cfg.loss.log_clip_min),
            log_clip_max=float(cfg.loss.log_clip_max),
            warmup_epochs=int(cfg.loss.alpha_warmup_epochs),
            # ★追加
            alpha_raw_total=float(getattr(cfg.loss, "alpha_raw_total", 0.0)),
            total_index=int(getattr(cfg.loss, "total_index", -1)),
        ).to(device)

    # ---- folds ----
    folds = list(cfg.folds)
    fold_col = str(cfg.fold_col)
    target_cols = list(cfg.target_cols)
    metric_weights = list(cfg.metric.weights)

    # postprocess設定（なければデフォルト）
    post_cfg = getattr(cfg, "postprocess", None)
    post_enabled = bool(getattr(post_cfg, "enabled", True)) if post_cfg is not None else True
    post_grid_n = int(getattr(post_cfg, "grid_n", 20)) if post_cfg is not None else 20
    post_modes = list(getattr(post_cfg, "modes", ["none", "delta", "sum_fix"])) if post_cfg is not None else ["none", "delta", "sum_fix"]
    post_targets = tuple(getattr(post_cfg, "targets", ["Dry_Clover_g", "Dry_Dead_g"])) if post_cfg is not None else ("Dry_Clover_g", "Dry_Dead_g")
    post_targets = (str(post_targets[0]), str(post_targets[1]))

    best_overall = -np.inf
    best_overall_epoch = -1
    best_overall_mode = None
    best_overall_thr = None

    for fold in folds:
        trn_df = df[df[fold_col] != fold].reset_index(drop=True)
        val_df = df[df[fold_col] == fold].reset_index(drop=True)

        if bool(cfg.debug):
            trn_df = trn_df.head(128).reset_index(drop=True)
            val_df = val_df.head(128).reset_index(drop=True)

        train_ds = CsiroDataset(
            df=trn_df,
            image_root=str(cfg.input_dir),
            target_cols=target_cols,
            transform=train_tfm,
            use_log1p_target=bool(cfg.use_log1p_target),
            return_target=True,
        )
        valid_ds = CsiroDataset(
            df=val_df,
            image_root=str(cfg.input_dir),
            target_cols=target_cols,
            transform=valid_tfm,
            use_log1p_target=bool(cfg.use_log1p_target),
            return_target=True,
        )

        # --------------------------
        # ★追加：oversample sampler
        # --------------------------
        use_oversample = bool(getattr(cfg, "oversample", {}).get("enabled", False))
        if use_oversample:
            weights_tensor = make_total_oversample_weights(
                trn_df,
                total_col=str(getattr(cfg.oversample, "total_col", "Dry_Total_g")),
                n_bins=int(getattr(cfg.oversample, "n_bins", 8)),
                strategy=str(getattr(cfg.oversample, "strategy", "ramp")),
                min_mult=float(getattr(cfg.oversample, "min_mult", 1.0)),
                max_mult=float(getattr(cfg.oversample, "max_mult", 5.0)),
            )
            sampler = build_weighted_sampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)

            train_loader = DataLoader(
                train_ds,
                batch_size=int(cfg.train.batch_size),
                sampler=sampler,
                shuffle=False,  # sampler使用時はFalse
                num_workers=int(cfg.num_workers),
                pin_memory=bool(cfg.pin_memory),
                persistent_workers=bool(cfg.persistent_workers),
                drop_last=False,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=int(cfg.train.batch_size),
                shuffle=True,
                num_workers=int(cfg.num_workers),
                pin_memory=bool(cfg.pin_memory),
                persistent_workers=bool(cfg.persistent_workers),
                drop_last=False,
            )

        valid_loader = DataLoader(
            valid_ds,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=bool(cfg.persistent_workers),
            drop_last=False,
        )

        # 画像サイズは transform と一致させる（あなたは W=H*2）
        img_h = int(cfg.img_size)
        img_w = int(getattr(cfg, "img_w", img_h * 2))
        # ---- model ----
        model = ConvNeXtRegressor(
            backbone=str(cfg.model.backbone),
            pretrained=bool(cfg.model.pretrained),
            num_targets=len(target_cols),
            in_chans=int(cfg.model.in_chans),
            drop_rate=float(cfg.model.drop_rate),
            drop_path_rate=float(cfg.model.drop_path_rate),
            head_dropout=float(getattr(cfg.model, "head_dropout", 0.0)),
            img_size=(img_h, img_w),  # ✅ 追加：Swinが落ちない
        ).to(device)

        optimizer = build_optimizer(cfg, model)

        total_steps = int(cfg.train.epochs) * len(train_loader) // int(cfg.train.grad_accum_steps)
        scheduler = build_scheduler(cfg, optimizer, total_steps=total_steps)

        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.use_amp))

        # ---- EMA（ema_decay==0 なら無効）----
        ema_model = None
        ema_decay = float(cfg.ema.decay)
        ema_enabled = bool(ema_decay > 0.0)
        if ema_enabled:
            ema_model = copy.deepcopy(unwrap_model(model)).to(device)
            ema_model.eval()
            for p in ema_model.parameters():
                p.requires_grad_(False)

        best_metric = -np.inf
        best_epoch = -1
        best_mode = None
        best_thr = None
        no_improve = 0

        patience = int(cfg.early_stopping.patience) if bool(cfg.early_stopping.enabled) else 0
        min_delta = float(cfg.early_stopping.min_delta) if bool(cfg.early_stopping.enabled) else 0.0

        global_step = 0

        for epoch in range(1, int(cfg.train.epochs) + 1):
            _, global_step = train_one_epoch(
                cfg=cfg,
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                device=device,
                scaler=scaler,
                epoch=epoch,
                use_amp=bool(cfg.use_amp),
                max_norm=float(cfg.train.max_norm),
                grad_accum_steps=int(cfg.train.grad_accum_steps),
                log_interval=int(cfg.train.log_interval),
                is_main_process=True,
                wandb_run=run,
                global_step=global_step,
            )

            do_val = (epoch % int(cfg.train.val_interval) == 0)
            if do_val:
                eval_model = ema_model if ema_model is not None else unwrap_model(model)

                # ★OOFが欲しいので return_oof=True
                val_loss, val_metric_base, r2_scores, oof = valid_one_epoch(
                    cfg=cfg,
                    model=eval_model,
                    loader=valid_loader,
                    loss_fn=loss_fn,
                    device=device,
                    epoch=epoch,
                    use_amp=bool(cfg.use_amp),
                    use_log1p_target=bool(cfg.use_log1p_target),
                    is_main_process=True,
                    wandb_run=run,
                    global_step=global_step,
                    target_names=target_cols,
                    return_oof=True,
                )

                # --------------------------
                # ★追加：postprocess mode比較（OOF）
                # --------------------------
                score_for_update = float(val_metric_base)
                if post_enabled and (oof is not None):
                    oof_true = oof["targets"]  # raw
                    oof_pred = oof["preds"]    # raw

                    pp_res = eval_postprocess_modes_on_oof(
                        oof_true=oof_true,
                        oof_pred=oof_pred,
                        target_cols=target_cols,
                        weights=metric_weights,
                        grid_n=post_grid_n,
                        modes=post_modes,
                        targets=post_targets,
                    )

                    # modeごとのスコアを wandb にログ
                    log_pp = {
                        "valid_pp/grid_n": int(post_grid_n),
                        "valid_pp/best_score": float(pp_res["best_score"]),
                    }
                    for m in post_modes:
                        m = str(m)
                        log_pp[f"valid_pp/weighted_r2_{m}"] = float(pp_res["scores"][m])
                        # 閾値も数値でログ（Clover/Dead）
                        thr_m = pp_res["thresholds"][m]
                        log_pp[f"valid_pp/thr_{m}_clover"] = float(thr_m.get("Dry_Clover_g", 0.0))
                        log_pp[f"valid_pp/thr_{m}_dead"] = float(thr_m.get("Dry_Dead_g", 0.0))

                    run.log(log_pp, step=global_step)

                    # sweep最適化/更新には「best_modeのスコア」を使う
                    score_for_update = float(pp_res["best_score"])

                    # 文字列best_modeはsummaryに入れる（ログでも良いがwandbで扱いづらい場合がある）
                    run.summary["valid_pp/best_mode_latest"] = str(pp_res["best_mode"])

                # --------------------------
                # best更新（ここが sweep metric になる）
                # --------------------------
                improved = (score_for_update > best_metric + min_delta)
                if improved:
                    best_metric = float(score_for_update)
                    best_epoch = int(epoch)
                    no_improve = 0

                    # run log（sweepが見る値）
                    run.log(
                        {
                            "best/weighted_r2": float(best_metric),
                            "best/epoch": int(best_epoch),
                            "best/weighted_r2_base": float(val_metric_base),
                        },
                        step=global_step,
                    )

                    # postprocessのbest情報（直近のpp_resを保持しておく）
                    if post_enabled and (oof is not None):
                        # run.summaryに入れると後で見返しやすい
                        # ※ここでは「最新のpp_res」がbestとは限らないが、improved時なので概ね一致する想定
                        run.summary["best_pp/mode"] = str(run.summary.get("valid_pp/best_mode_latest", "unknown"))

                else:
                    no_improve += 1

                if (patience > 0) and (no_improve >= patience):
                    break

            # EMA update（epoch単位）
            if ema_model is not None:
                update_ema(ema_model, model, decay=ema_decay)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # fold best を summary に載せる
        run.log({f"fold{fold}/best_weighted_r2": float(best_metric)})

        # run 全体 best
        if best_metric > best_overall:
            best_overall = float(best_metric)
            best_overall_epoch = int(best_epoch)

    # sweep の最適化対象
    run.summary["best/weighted_r2"] = float(best_overall)
    run.summary["best/epoch"] = int(best_overall_epoch)

    run.finish()


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W&B sweep runner (v3).")
    parser.add_argument("--action", choices=["create", "agent"], required=True, help="create: sweep作成 / agent: 実行")
    parser.add_argument("--base_cfg", type=str, required=True, help="base yamlへのパス")
    parser.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "Csiro-Image2BiomassPrediction"))
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument("--sweep_id", type=str, default=None, help="agent実行時の sweep_id")
    parser.add_argument("--count", type=int, default=30, help="agent の実行回数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_base_cfg(args.base_cfg)

    if args.action == "create":
        sweep_cfg = build_sweep_config(project=args.project)
        sweep_id = wandb.sweep(sweep=sweep_cfg, project=args.project, entity=args.entity)

        print("\n========== SWEEP CREATED ==========")
        print(f"project: {args.project}")
        print(f"entity : {args.entity}")
        print(f"sweep_id: {sweep_id}")
        print(sweep_id)

    elif args.action == "agent":
        if args.sweep_id is None:
            raise ValueError("--sweep_id is required for --action agent")

        def _fn():
            run_one_trial(base_cfg)

        wandb.agent(args.sweep_id, function=_fn, count=int(args.count))


if __name__ == "__main__":
    main()
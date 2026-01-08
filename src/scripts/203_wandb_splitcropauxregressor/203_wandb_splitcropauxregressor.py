# =========================================================
# 203_wandb_splitcropauxregressor.py
#   - W&B Sweep runner for ConvNeXtSplitCropAuxRegressor
#   - convnext_small 固定
#   - lrは scheduler無効（constant LR）
#   - OOM時は run を fail 扱いにして次 trial へ進む
# =========================================================

import argparse
import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import wandb

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler

from omegaconf import OmegaConf, DictConfig

# =========================
# import from src
# =========================
import sys
SRC_DIR = Path(__file__).resolve().parents[2]  # .../src
sys.path.append(str(SRC_DIR))

from utils.data import set_seed
from utils.train_utils import build_optimizer
from utils.losses import MixedLogRawLoss

from datasets.crop_dataset import CsiroDataset
from datasets.transforms_crop import build_transforms
from models.convnext_splitcrop_aux_regressor import ConvNeXtSplitCropAuxRegressor
from training.train import train_one_epoch, valid_one_epoch


# =========================================================
# DDP helpers
# =========================================================
def init_distributed(cfg_train: DictConfig) -> Tuple[bool, int, int, int, torch.device, bool]:
    """DDP初期化（sweepでは基本OFF推奨）。"""
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    ddp_enabled = bool(getattr(cfg_train.ddp, "enabled", False)) if hasattr(cfg_train, "ddp") else False
    use_ddp = (env_world_size > 1) and ddp_enabled

    if use_ddp:
        backend = str(getattr(cfg_train.ddp, "backend", "nccl"))
        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = env_world_size
        is_main = (rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        local_rank = 0
        is_main = True

    return use_ddp, rank, local_rank, world_size, device, is_main


def cleanup_distributed(use_ddp: bool) -> None:
    """DDP後処理."""
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """DDP/DataParallelのラップを外す."""
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """rank0のオブジェクトを全rankへ配布（wandb config共有用）。"""
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    obj_list = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# =========================================================
# YAML loader
# =========================================================
def load_base_cfg(base_cfg_path: Path) -> DictConfig:
    cfg = OmegaConf.load(str(base_cfg_path))
    if not isinstance(cfg, DictConfig):
        raise TypeError("base_cfg を DictConfig として読み込めませんでした。")
    return cfg


# def default_sweep_config() -> Dict[str, Any]:
#     """このスクリプト内蔵の sweep config（外部yaml不要版）。

#     Notes:
#         ここを編集すると探索範囲を簡単に変えられます。
#     """
#     return {
#         "method": "random",
#         "metric": {"name": "best/weighted_r2", "goal": "maximize"},
#         "parameters": {
#             # --- core ---
#             "epochs": {"values": [100, 200]},
#             "batch_size": {"values": [8, 12, 16, 20]},
#             "img_size": {"values": [224, 256, 288, 320]},

#             # --- optimizer ---
#             "lr": {"distribution": "log_uniform_values", "min": 5e-6, "max": 5e-4},
#             "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 5e-3},

#             # --- loss ---
#             "alpha_raw": {"values": [0.0, 0.02, 0.05, 0.10]},
#             "raw_loss": {"values": ["mse", "huber"]},
#             "raw_huber_beta": {"values": [5.0, 10.0, 20.0]},
#             "lambda_consistency": {"values": [0.0, 0.2, 0.5, 1.0]},
#             "consistency_loss": {"values": ["huber", "mse"]},
#             "consistency_beta": {"values": [5.0, 10.0, 20.0]},

#             # --- augmentation ---
#             "hflip_p": {"values": [0.0, 0.5]},
#             "vflip_p": {"values": [0.0, 0.5]},
#             "rotate90_p": {"values": [0.0, 0.5]},
#             "shift_scale_rotate_p": {"values": [0.0, 0.10, 0.15]},
#             "color_jitter_p": {"values": [0.0, 0.20, 0.30]},
#             "coarse_dropout_p": {"values": [0.0, 0.10, 0.15]},
#             "tile_shuffle_p": {"values": [0.0, 0.05, 0.10]},
#             "blur_noise_p": {"values": [0.0, 0.05, 0.10]},

#             # --- aux weights（必要なら探索）---
#             "aux_species_w": {"values": [0.0, 0.03, 0.05, 0.08]},
#             "aux_ndvi_w": {"values": [0.0, 0.01, 0.02, 0.05]},
#             "aux_height_w": {"values": [0.0, 0.01, 0.02, 0.05]},
#             "aux_warmup_epochs": {"values": [0, 5, 10, 20]},
#         },
#     }

def default_sweep_config() -> Dict[str, Any]:
    """このスクリプト内蔵の sweep config（外部yaml不要版）。

    Notes:
        ここを編集すると探索範囲を簡単に変えられます。
    """
    return {
        "method": "random",
        "metric": {"name": "best/weighted_r2", "goal": "maximize"},
        "parameters": {
            # --- core ---
            "epochs": {"values": [200]},
            "batch_size": {"values": [8, 16, 20]},
            "img_size": {"values": [224, 256, 288, 320]},

            # --- optimizer ---
            "lr": {"distribution": "log_uniform_values", "min": 5e-6, "max": 5e-4},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 5e-3},

            # --- loss ---
            "alpha_raw": {"values": [0.0, 0.02, 0.05, 0.10]},
            "raw_loss": {"values": ["mse", "huber"]},
            "raw_huber_beta": {"values": [5.0, 10.0, 20.0]},
            "lambda_consistency": {"values": [0.0, 0.2, 0.5, 1.0]},
            "consistency_loss": {"values": ["huber", "mse"]},
            "consistency_beta": {"values": [5.0, 10.0, 20.0]},

            # --- augmentation ---
            "hflip_p": {"values": [0.0, 0.25, 0.5]},
            "vflip_p": {"values": [0.0, 0.25, 0.5]},
            "rotate90_p": {"values": [0.0, 0.25, 0.5]},
            "shift_scale_rotate_p": {"values": [0.0, 0.10, 0.15]},
            "color_jitter_p": {"values": [0.0, 0.20, 0.30]},
            "coarse_dropout_p": {"values": [0.0, 0.10, 0.15]},
            "tile_shuffle_p": {"values": [0.0, 0.05, 0.10]},
            "blur_noise_p": {"values": [0.0, 0.05, 0.10]},

            # --- aux weights（必要なら探索）---
            "aux_species_w": {"values": [0.0, 0.03, 0.05, 0.08]},
            "aux_ndvi_w": {"values": [0.0, 0.01, 0.02, 0.05]},
            "aux_height_w": {"values": [0.0, 0.01, 0.02, 0.05]},
            "aux_warmup_epochs": {"values": [0, 5, 10, 20]},
        },
    }


# =========================================================
# wandb overrides
# =========================================================
# def build_wandb_default_config(cfg_train: DictConfig) -> Dict[str, Any]:
#     """wandbに渡すデフォルト（フラット）を作る。"""
#     defaults = {
#         "fold": int(cfg_train.folds[0]) if len(cfg_train.folds) > 0 else 0,
#         "epochs": int(cfg_train.train.epochs),
#         "batch_size": int(cfg_train.train.batch_size),

#         "img_size": int(cfg_train.img_h),

#         "lr": float(cfg_train.optimizer.base_lr),
#         "weight_decay": float(cfg_train.optimizer.weight_decay),

#         "alpha_raw": float(cfg_train.loss.alpha_raw),
#         "raw_loss": str(cfg_train.loss.raw_loss),
#         "raw_huber_beta": float(cfg_train.loss.raw_huber_beta),
#         "lambda_consistency": float(getattr(cfg_train.loss, "lambda_consistency", 0.0)),
#         "consistency_loss": str(getattr(cfg_train.loss, "consistency_loss", "huber")),
#         "consistency_beta": float(getattr(cfg_train.loss, "consistency_beta", 10.0)),

#         "hflip_p": float(cfg_train.augment.train.hflip_p),
#         "vflip_p": float(getattr(cfg_train.augment.train, "vflip_p", 0.0)),
#         "rotate90_p": float(getattr(cfg_train.augment.train, "rotate90_p", 0.0)),
#         "shift_scale_rotate_p": float(getattr(cfg_train.augment.train, "shift_scale_rotate_p", 0.0)),
#         "color_jitter_p": float(getattr(cfg_train.augment.train, "color_jitter_p", 0.0)),
#         "coarse_dropout_p": float(getattr(cfg_train.augment.train, "coarse_dropout_p", 0.0)),
#         "tile_shuffle_p": float(getattr(cfg_train.augment.train, "tile_shuffle_p", 0.0)),
#         "blur_noise_p": float(getattr(cfg_train.augment.train, "blur_noise_p", 0.0)),

#         "aux_species_w": float(getattr(cfg_train.aux.species, "weight", 0.0)) if hasattr(cfg_train, "aux") else 0.0,
#         "aux_ndvi_w": float(getattr(cfg_train.aux.ndvi, "weight", 0.0)) if hasattr(cfg_train, "aux") else 0.0,
#         "aux_height_w": float(getattr(cfg_train.aux.height, "weight", 0.0)) if hasattr(cfg_train, "aux") else 0.0,
#         "aux_warmup_epochs": int(getattr(cfg_train.aux, "warmup_epochs", 0)) if hasattr(cfg_train, "aux") else 0,
#     }
#     return defaults
def build_wandb_default_config(cfg_train: DictConfig) -> Dict[str, Any]:
    """wandbに渡すデフォルト（フラット）を作る。"""
    defaults = {
        "fold": int(cfg_train.folds[0]) if len(cfg_train.folds) > 0 else 0,
        "epochs": int(cfg_train.train.epochs),
        "batch_size": int(cfg_train.train.batch_size),

        "img_size": int(cfg_train.img_h),

        "lr": float(cfg_train.optimizer.base_lr),
        "weight_decay": float(cfg_train.optimizer.weight_decay),

        "alpha_raw": float(cfg_train.loss.alpha_raw),
        "raw_loss": str(cfg_train.loss.raw_loss),
        "raw_huber_beta": float(cfg_train.loss.raw_huber_beta),
        "lambda_consistency": float(getattr(cfg_train.loss, "lambda_consistency", 0.0)),
        "consistency_loss": str(getattr(cfg_train.loss, "consistency_loss", "huber")),
        "consistency_beta": float(getattr(cfg_train.loss, "consistency_beta", 10.0)),

        "hflip_p": float(cfg_train.augment.train.hflip_p),
        "vflip_p": float(getattr(cfg_train.augment.train, "vflip_p", 0.0)),
        "rotate90_p": float(getattr(cfg_train.augment.train, "rotate90_p", 0.0)),
        "shift_scale_rotate_p": float(getattr(cfg_train.augment.train, "shift_scale_rotate_p", 0.0)),
        "color_jitter_p": float(getattr(cfg_train.augment.train, "color_jitter_p", 0.0)),
        "coarse_dropout_p": float(getattr(cfg_train.augment.train, "coarse_dropout_p", 0.0)),
        "tile_shuffle_p": float(getattr(cfg_train.augment.train, "tile_shuffle_p", 0.0)),
        "blur_noise_p": float(getattr(cfg_train.augment.train, "blur_noise_p", 0.0)),

        "aux_species_w": float(getattr(cfg_train.aux.species, "weight", 0.0)) if hasattr(cfg_train, "aux") else 0.0,
        "aux_ndvi_w": float(getattr(cfg_train.aux.ndvi, "weight", 0.0)) if hasattr(cfg_train, "aux") else 0.0,
        "aux_height_w": float(getattr(cfg_train.aux.height, "weight", 0.0)) if hasattr(cfg_train, "aux") else 0.0,
        "aux_warmup_epochs": int(getattr(cfg_train.aux, "warmup_epochs", 0)) if hasattr(cfg_train, "aux") else 0,
    }
    return defaults



def apply_wandb_overrides(cfg_train: DictConfig, wcfg: Dict[str, Any]) -> None:
    """wandb.config を cfg_train に反映する。"""
    OmegaConf.set_struct(cfg_train, False)

    # fold / epochs / batch
    if "fold" in wcfg:
        cfg_train.folds = [int(wcfg["fold"])]
    if "epochs" in wcfg:
        cfg_train.train.epochs = int(wcfg["epochs"])
    if "batch_size" in wcfg:
        cfg_train.train.batch_size = int(wcfg["batch_size"])

    # img_size -> img_h/img_w
    if "img_size" in wcfg:
        s = int(wcfg["img_size"])
        cfg_train.img_h = s
        cfg_train.img_w = s

    # optimizer
    if "lr" in wcfg:
        cfg_train.optimizer.base_lr = float(wcfg["lr"])
    if "weight_decay" in wcfg:
        cfg_train.optimizer.weight_decay = float(wcfg["weight_decay"])

    # loss
    if "alpha_raw" in wcfg:
        cfg_train.loss.alpha_raw = float(wcfg["alpha_raw"])
    if "raw_loss" in wcfg:
        cfg_train.loss.raw_loss = str(wcfg["raw_loss"])
    if "raw_huber_beta" in wcfg:
        cfg_train.loss.raw_huber_beta = float(wcfg["raw_huber_beta"])

    if "lambda_consistency" in wcfg:
        cfg_train.loss.lambda_consistency = float(wcfg["lambda_consistency"])
    if "consistency_loss" in wcfg:
        cfg_train.loss.consistency_loss = str(wcfg["consistency_loss"])
    if "consistency_beta" in wcfg:
        cfg_train.loss.consistency_beta = float(wcfg["consistency_beta"])

    # augmentation
    if "hflip_p" in wcfg:
        cfg_train.augment.train.hflip_p = float(wcfg["hflip_p"])
    if "vflip_p" in wcfg:
        cfg_train.augment.train.vflip_p = float(wcfg["vflip_p"])
    if "rotate90_p" in wcfg:
        cfg_train.augment.train.rotate90_p = float(wcfg["rotate90_p"])
    if "shift_scale_rotate_p" in wcfg:
        cfg_train.augment.train.shift_scale_rotate_p = float(wcfg["shift_scale_rotate_p"])
    if "color_jitter_p" in wcfg:
        cfg_train.augment.train.color_jitter_p = float(wcfg["color_jitter_p"])
    if "coarse_dropout_p" in wcfg:
        cfg_train.augment.train.coarse_dropout_p = float(wcfg["coarse_dropout_p"])
    if "tile_shuffle_p" in wcfg:
        cfg_train.augment.train.tile_shuffle_p = float(wcfg["tile_shuffle_p"])
    if "blur_noise_p" in wcfg:
        cfg_train.augment.train.blur_noise_p = float(wcfg["blur_noise_p"])

    # aux weights
    if hasattr(cfg_train, "aux"):
        if "aux_species_w" in wcfg:
            cfg_train.aux.species.weight = float(wcfg["aux_species_w"])
        if "aux_ndvi_w" in wcfg:
            cfg_train.aux.ndvi.weight = float(wcfg["aux_ndvi_w"])
        if "aux_height_w" in wcfg:
            cfg_train.aux.height.weight = float(wcfg["aux_height_w"])
        if "aux_warmup_epochs" in wcfg:
            cfg_train.aux.warmup_epochs = int(wcfg["aux_warmup_epochs"])

    # backbone固定（安全のため強制）
    cfg_train.model.backbone = "convnext_small"
    cfg_train.ddp.enabled = False
    OmegaConf.set_struct(cfg_train, True)


def make_run_name(cfg_train: DictConfig, fold: int, wcfg: Dict[str, Any]) -> str:
    """run名（長すぎ防止）。"""
    img = int(cfg_train.img_h)
    bs = int(cfg_train.train.batch_size)
    lr = float(cfg_train.optimizer.base_lr)
    wd = float(cfg_train.optimizer.weight_decay)
    rl = str(cfg_train.loss.raw_loss)
    a = float(cfg_train.loss.alpha_raw)
    lam = float(getattr(cfg_train.loss, "lambda_consistency", 0.0))
    return f"{cfg_train.exp}_f{fold}_img{img}_bs{bs}_lr{lr:.1e}_wd{wd:.1e}_{rl}_a{a:.2f}_lc{lam:.2f}"


# =========================================================
# split-crop transform に resize を挿入（img_h/img_w探索を効かせる）
# =========================================================
def build_transforms_with_optional_resize(cfg_train: DictConfig, is_train: bool):
    """既存 build_transforms に Resize を差し込む。

    Notes:
        datasets/transforms_crop.py の split-crop 分岐は Resize をしない実装なので、
        Sweepで img_h/img_w を探索するために、ここで Resize を挿入する。
    """
    tfm = build_transforms(cfg_train, is_train=is_train)

    # split-crop のときだけ Resize を追加（img_h/img_wへ）
    if bool(getattr(cfg_train, "use_split_crop", False)):
        import albumentations as A

        h = int(getattr(cfg_train, "img_h", 0))
        w = int(getattr(cfg_train, "img_w", 0))
        if h > 0 and w > 0:
            # Normalize の直前に入れる（最後2つが Normalize/ToTensorV2 の想定）
            if hasattr(tfm, "transforms") and len(tfm.transforms) >= 2:
                resize = A.Resize(height=h, width=w)
                tfm.transforms = tfm.transforms[:-2] + [resize] + tfm.transforms[-2:]
    return tfm


# =========================================================
# 1 trial
# =========================================================
def run_one_trial(base_cfg_path: Path) -> None:
    cfg_train = load_base_cfg(base_cfg_path)

    # DDP init
    use_ddp, rank, local_rank, world_size, device, is_main = init_distributed(cfg_train)

    # seed
    set_seed(int(cfg_train.seed) + int(rank))

    # load pivot
    pp_dir = Path(str(cfg_train.pp_dir)) / str(cfg_train.preprocess_ver)
    pivot_path = pp_dir / str(cfg_train.pivot_csv_name)
    df = pd.read_csv(pivot_path)

    # -----------------------------------------------------
    # aux 前処理（学習コードと同じ：species_to_index, std）
    # -----------------------------------------------------
    aux_cfg = getattr(cfg_train, "aux", None)
    aux_enabled = bool(getattr(aux_cfg, "enabled", False)) if aux_cfg is not None else False

    sp_col = "Species"
    ndvi_col = "Pre_GSHH_NDVI"
    height_col = "Height_Ave_cm"

    if aux_cfg is not None:
        sp_cfg = getattr(aux_cfg, "species", None)
        ndvi_cfg = getattr(aux_cfg, "ndvi", None)
        h_cfg = getattr(aux_cfg, "height", None)
        if sp_cfg is not None:
            sp_col = str(getattr(sp_cfg, "col", sp_col))
        if ndvi_cfg is not None:
            ndvi_col = str(getattr(ndvi_cfg, "col", ndvi_col))
        if h_cfg is not None:
            height_col = str(getattr(h_cfg, "col", height_col))

    species_to_index = {}
    num_species = 0
    ndvi_std = 1.0
    height_std = 1.0

    if aux_enabled:
        if sp_col in df.columns:
            species_list = sorted(df[sp_col].dropna().astype(str).unique().tolist())
            species_to_index = {s: i for i, s in enumerate(species_list)}
            num_species = len(species_list)

        if ndvi_col in df.columns:
            ndvi_s = pd.to_numeric(df[ndvi_col], errors="coerce").dropna()
            if len(ndvi_s) > 0:
                ndvi_std = float(ndvi_s.std())
                if (not np.isfinite(ndvi_std)) or (ndvi_std <= 0.0):
                    ndvi_std = 1.0

        if height_col in df.columns:
            height_s = pd.to_numeric(df[height_col], errors="coerce").dropna()
            if len(height_s) > 0:
                height_std = float(height_s.std())
                if (not np.isfinite(height_std)) or (height_std <= 0.0):
                    height_std = 1.0

    # -----------------------------------------------------
    # wandb init
    # -----------------------------------------------------
    run = None
    wcfg: Dict[str, Any] = {}

    if bool(cfg_train.use_wandb) and is_main:
        default_cfg = build_wandb_default_config(cfg_train)
        # sweep安全デフォルト
        default_cfg.setdefault("fold", int(cfg_train.folds[0]) if len(cfg_train.folds) > 0 else 0)
        default_cfg.setdefault("epochs", min(int(cfg_train.train.epochs), 60))
        default_cfg.setdefault("batch_size", int(cfg_train.train.batch_size))

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", str(cfg_train.competition)),
            entity=os.environ.get("WANDB_ENTITY", str(cfg_train.author)),
            name=None,
            config=default_cfg,
        )
        wcfg = dict(wandb.config)

    wcfg = broadcast_object(wcfg, src=0)

    # apply overrides
    apply_wandb_overrides(cfg_train, wcfg)

    fold = int(cfg_train.folds[0]) if len(cfg_train.folds) > 0 else 0
    if run is not None:
        run.name = make_run_name(cfg_train, fold, wcfg)

    # -----------------------------------------------------
    # transforms（Resize差し込み版）
    # -----------------------------------------------------
    # train_tfm = build_transforms_with_optional_resize(cfg_train, is_train=True)
    # valid_tfm = build_transforms_with_optional_resize(cfg_train, is_train=False)
    train_tfm = build_transforms(cfg_train, is_train=True)
    valid_tfm = build_transforms(cfg_train, is_train=False)

    # -----------------------------------------------------
    # loss（aux有効なら wrapper を使う）
    # -----------------------------------------------------
    main_loss = MixedLogRawLoss(
        weights=list(cfg_train.loss.weights),
        alpha_raw=float(cfg_train.loss.alpha_raw),
        raw_loss=str(cfg_train.loss.raw_loss),
        raw_huber_beta=float(cfg_train.loss.raw_huber_beta),
        log_clip_min=float(cfg_train.loss.log_clip_min),
        log_clip_max=float(cfg_train.loss.log_clip_max),
        warmup_epochs=int(cfg_train.loss.alpha_warmup_epochs),
        lambda_consistency=float(getattr(cfg_train.loss, "lambda_consistency", 0.0)),
        consistency_loss=str(getattr(cfg_train.loss, "consistency_loss", "huber")),
        consistency_beta=float(getattr(cfg_train.loss, "consistency_beta", 10.0)),
        consistency_warmup_epochs=getattr(cfg_train.loss, "consistency_warmup_epochs", None),
        target_cols=list(cfg_train.target_cols),
    ).to(device)

    loss_fn: nn.Module = main_loss
    if aux_enabled:
        from utils.losses import BiomassAuxLossWrapper
        loss_fn = BiomassAuxLossWrapper(
            main_loss=main_loss,
            aux_cfg=aux_cfg,
            ndvi_std=float(ndvi_std),
            height_std=float(height_std),
        ).to(device)

    # -----------------------------------------------------
    # split
    # -----------------------------------------------------
    fold_col = str(cfg_train.fold_col)
    trn_df = df[df[fold_col] != fold].reset_index(drop=True)
    val_df = df[df[fold_col] == fold].reset_index(drop=True)

    if bool(cfg_train.debug):
        trn_df = trn_df.head(256).reset_index(drop=True)
        val_df = val_df.head(256).reset_index(drop=True)

    # Dataset
    train_ds = CsiroDataset(
        df=trn_df,
        image_root=str(cfg_train.input_dir),
        target_cols=list(cfg_train.target_cols),
        transform=train_tfm,
        use_log1p_target=bool(cfg_train.use_log1p_target),
        return_target=True,
        aux_cfg=aux_cfg,
        species_to_index=species_to_index,
        aux_cols=None,
        use_split_crop=True,
        crop_size=int(cfg_train.crop_size),
        assume_size=(1000, 2000),
        crop_mode="random",
    )
    valid_ds = CsiroDataset(
        df=val_df,
        image_root=str(cfg_train.input_dir),
        target_cols=list(cfg_train.target_cols),
        transform=valid_tfm,
        use_log1p_target=bool(cfg_train.use_log1p_target),
        return_target=True,
        aux_cfg=aux_cfg,
        species_to_index=species_to_index,
        aux_cols=None,
        use_split_crop=True,
        crop_size=int(cfg_train.crop_size),
        assume_size=(1000, 2000),
        crop_mode="center",
    )

    # Loader
    # train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    true_total = trn_df["Dry_Total_g"].values.astype(float)
    thr = np.quantile(true_total, 0.85)          # 既にseed_searchでも使ってる閾値
    w = np.ones_like(true_total, dtype=np.float32)
    w[true_total >= thr] *= 3.0                  # ここを2〜5で調整
    train_sampler = WeightedRandomSampler(
        weights=w,
        num_samples=len(w),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg_train.train.batch_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(cfg_train.num_workers),
        pin_memory=bool(cfg_train.pin_memory),
        persistent_workers=bool(cfg_train.persistent_workers),
        drop_last=False,
    )

    valid_loader = None
    if is_main:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=int(cfg_train.train.batch_size),
            shuffle=False,
            num_workers=int(cfg_train.num_workers),
            pin_memory=bool(cfg_train.pin_memory),
            persistent_workers=bool(cfg_train.persistent_workers),
            drop_last=False,
        )

    # -----------------------------------------------------
    # model
    # -----------------------------------------------------
    model = ConvNeXtSplitCropAuxRegressor(
        backbone="convnext_small",  # 固定
        pretrained=bool(cfg_train.model.pretrained),
        num_targets=len(cfg_train.target_cols),
        in_chans=int(cfg_train.model.in_chans),
        drop_rate=float(cfg_train.model.drop_rate),
        drop_path_rate=float(cfg_train.model.drop_path_rate),
        head_dropout=float(getattr(cfg_train.model, "head_dropout", 0.0)),
        fuse=str(getattr(cfg_train.model, "fuse", "concat")),
        aux_cfg=aux_cfg,
        num_species=int(num_species),
        aux_hidden_dim=int(getattr(cfg_train.model, "aux_hidden_dim", 256)),
        aux_dropout=float(getattr(cfg_train.model, "aux_dropout", 0.1)),
    ).to(device)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(getattr(cfg_train.ddp, "find_unused_parameters", False)),
        )

    optimizer = build_optimizer(cfg_train, model)

    # scheduler無効（constant LR）
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg_train.use_amp))

    # early stop
    patience = int(cfg_train.early_stopping.patience) if bool(cfg_train.early_stopping.enabled) else 0
    min_delta = float(cfg_train.early_stopping.min_delta) if bool(cfg_train.early_stopping.enabled) else 0.0

    best_metric = -np.inf
    best_epoch = -1
    no_improve = 0
    global_step = 0

    oom_happened = False

    try:
        for epoch in range(1, int(cfg_train.train.epochs) + 1):
            if use_ddp and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            # ---- train ----
            _, global_step = train_one_epoch(
                cfg=cfg_train,
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,   # constant LR
                loss_fn=loss_fn,
                device=device,
                scaler=scaler,
                epoch=epoch,
                use_amp=bool(cfg_train.use_amp),
                max_norm=float(cfg_train.train.max_norm),
                grad_accum_steps=int(cfg_train.train.grad_accum_steps),
                log_interval=int(cfg_train.train.log_interval),
                is_main_process=is_main,
                wandb_run=run,
                global_step=global_step,
            )

            stop_flag = 0
            if is_main:
                eval_model = unwrap_model(model)

                _, val_metric, _, _ = valid_one_epoch(
                    cfg=cfg_train,
                    model=eval_model,
                    loader=valid_loader,
                    loss_fn=loss_fn,
                    device=device,
                    epoch=epoch,
                    use_amp=bool(cfg_train.use_amp),
                    use_log1p_target=bool(cfg_train.use_log1p_target),
                    is_main_process=is_main,
                    wandb_run=run,
                    global_step=global_step,
                    target_names=list(cfg_train.target_cols),
                    return_oof=False,  # sweepは軽量化
                )

                improved = (float(val_metric) > float(best_metric) + float(min_delta))
                if improved:
                    best_metric = float(val_metric)
                    best_epoch = int(epoch)
                    no_improve = 0
                    if run is not None:
                        run.summary["best/weighted_r2"] = float(best_metric)
                        run.summary["best/epoch"] = int(best_epoch)
                else:
                    no_improve += 1

                if patience > 0 and no_improve >= patience:
                    stop_flag = 1

            if use_ddp:
                t = torch.tensor([stop_flag], device=device)
                dist.broadcast(t, src=0)
                stop_flag = int(t.item())

            if stop_flag == 1:
                break

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    except RuntimeError as e:
        # ---- OOM handling ----
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            oom_happened = True
            if is_main and run is not None:
                run.summary["failed/oom"] = 1
                run.summary["best/weighted_r2"] = float(best_metric) if np.isfinite(best_metric) else -1e9
                run.summary["best/epoch"] = int(best_epoch) if best_epoch >= 0 else -1
            # 例外は飲み込んで次 trial へ
        else:
            raise

    finally:
        if is_main and run is not None:
            run.summary["fold"] = int(fold)
            run.summary["failed/oom"] = int(oom_happened)
            run.finish()

        cleanup_distributed(use_ddp)

        # 次trialへ向けて掃除
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# =========================================================
# CLI: create / agent
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="203 SplitCropAuxRegressor W&B sweep runner")
    parser.add_argument("--action", type=str, required=True, choices=["create", "agent"])
    parser.add_argument("--base_cfg", type=str, required=True)

    # create
    parser.add_argument("--project", type=str, default="")
    parser.add_argument("--entity", type=str, default="")

    # agent
    parser.add_argument("--sweep_id", type=str, default="")
    parser.add_argument("--count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg_path = Path(args.base_cfg)
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"base_cfg not found: {base_cfg_path}")

    if args.action == "create":
        if not args.project or not args.entity:
            raise ValueError("--project / --entity は create のとき必須です。")
        sweep_id = wandb.sweep(default_sweep_config(), project=args.project, entity=args.entity)
        print(str(sweep_id))  # 末尾1行を sweep_id にする
        return

    # agent
    if not args.sweep_id:
        raise ValueError("--sweep_id は agent のとき必須です。")

    wandb.agent(
        args.sweep_id,
        function=lambda: run_one_trial(base_cfg_path),
        count=int(args.count),
    )


if __name__ == "__main__":
    main()
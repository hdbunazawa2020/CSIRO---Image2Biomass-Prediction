# -*- coding: utf-8 -*-
"""
WandB Sweep 実行スクリプト（CSIRO Image2Biomass）。

目的:
    - augmentation / loss / lr / model(backbone) などを bayes で探索
    - 目的指標は official に合わせた global weighted R²（maximize）

設計方針:
    - Sweep は 1 run = 1 GPU のシングルプロセスで回す（DDPは使わない）
      → 複数GPUを使う場合は、GPUごとに wandb agent を複数立ち上げるのが簡単。
    - Sweep で見つけた良さそうな設定を、最後に “全fold” で学習して確定させる。

使い方（例）:
    # 1) sweep 作成だけ（sweep_id を出力）
    python ./200_wandb_sweep/200_wandb_sweep.py --create_only

    # 2) 既存 sweep_id に対して agent を回す
    python ./200_wandb_sweep/200_wandb_sweep.py --agent --sweep_id XXXXX --count 30

    # 3) create + agent をまとめて（簡単）
    python ./200_wandb_sweep/200_wandb_sweep.py --count 30
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

# ====== add src to sys.path ======
import sys
SRC_DIR = Path(__file__).resolve().parents[2]  # .../src
sys.path.append(str(SRC_DIR))

from utils.data import set_seed
from utils.train_utils import build_optimizer, build_scheduler, get_scaler
from utils.losses import WeightedMSELoss, MixedLogRawLoss
from datasets.dataset import CsiroDataset
from datasets.transforms import build_transforms
from models.convnext_regressor import ConvNeXtRegressor
from training.train import train_one_epoch, valid_one_epoch


# =========================================================
# Sweep config
# =========================================================
def build_sweep_config() -> Dict[str, Any]:
    """WandB sweep 設定（bayes）。"""
    return {
        "method": "bayes",
        "metric": {"name": "best/weighted_r2", "goal": "maximize"},
        "parameters": {
            # ----- model -----
            "backbone": {
                "values": ["convnext_tiny", "convnext_small", "convnext_base"]
            },
            "img_size": {"values": [224, 288, 320]},
            "head_dropout": {"values": [0.0, 0.1, 0.2, 0.3]},

            # ----- optimizer -----
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 5e-4,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-1,
            },

            # ----- augmentation -----
            "hflip_p": {"values": [0.0, 0.25, 0.5, 0.75]},
            "rotate_limit": {"values": [0, 10, 20, 30]},
            "shift_scale_rotate_p": {"values": [0.0, 0.2, 0.5]},
            "color_jitter_p": {"values": [0.0, 0.2, 0.4]},

            # ----- loss -----
            "loss_type": {"values": ["weighted_mse", "mixed_log_raw"]},
            "alpha_raw": {"values": [0.0, 0.01, 0.03, 0.05, 0.1]},
            "alpha_warmup_epochs": {"values": [0, 5, 10]},
            "raw_loss": {"values": ["mse", "huber", "l1"]},
            "raw_huber_beta": {"values": [5.0, 10.0, 20.0]},

            # ----- EMA -----
            "ema_enabled": {"values": [False, True]},
            "ema_decay": {"values": [0.95, 0.97, 0.99]},
        },
    }


def apply_sweep_params(cfg_train, wb_cfg: Dict[str, Any]) -> None:
    """wandb.config の値を cfg_train に反映する（in-place 更新）。

    Args:
        cfg_train: OmegaConf（100_train_model_default.yaml を読み込んだもの）
        wb_cfg: wandb.config を dict 化したもの
    """
    # model
    cfg_train.model.backbone = str(wb_cfg["backbone"])
    cfg_train.img_size = int(wb_cfg["img_size"])
    cfg_train.model.head_dropout = float(wb_cfg["head_dropout"])

    # optimizer / scheduler（基本は “固定lr” で探す）
    lr = float(wb_cfg["lr"])
    wd = float(wb_cfg["weight_decay"])
    cfg_train.optimizer.base_lr = lr
    cfg_train.optimizer.weight_decay = wd

    # scheduler も “固定” にするなら max=min=base に合わせる
    cfg_train.scheduler.base_lr = lr
    cfg_train.scheduler.max_lr = lr
    cfg_train.scheduler.min_lr = lr

    # augmentation
    cfg_train.augment.train.hflip_p = float(wb_cfg["hflip_p"])
    cfg_train.augment.train.rotate_limit = int(wb_cfg["rotate_limit"])
    cfg_train.augment.train.shift_scale_rotate_p = float(wb_cfg["shift_scale_rotate_p"])
    cfg_train.augment.train.color_jitter_p = float(wb_cfg["color_jitter_p"])

    # loss
    cfg_train.loss.name = str(wb_cfg["loss_type"])
    cfg_train.loss.alpha_raw = float(wb_cfg["alpha_raw"])
    cfg_train.loss.alpha_warmup_epochs = int(wb_cfg["alpha_warmup_epochs"])
    cfg_train.loss.raw_loss = str(wb_cfg["raw_loss"])
    cfg_train.loss.raw_huber_beta = float(wb_cfg["raw_huber_beta"])

    # EMA
    cfg_train.ema.enabled = bool(wb_cfg["ema_enabled"])
    cfg_train.ema.decay = float(wb_cfg["ema_decay"])


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """EMA を更新する（single process 用）。"""
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k in esd.keys():
        if k in msd:
            esd[k].mul_(decay).add_(msd[k], alpha=(1.0 - decay))
    ema_model.load_state_dict(esd, strict=True)


def build_loss(cfg_train, device: torch.device) -> nn.Module:
    """cfg_train.loss に従って loss_fn を構築する。"""
    weights = list(cfg_train.loss.weights)

    name = str(cfg_train.loss.name).lower()
    if name == "weighted_mse":
        return WeightedMSELoss(weights).to(device)

    if name == "mixed_log_raw":
        return MixedLogRawLoss(
            weights=weights,
            alpha_raw=float(cfg_train.loss.alpha_raw),
            raw_loss=str(cfg_train.loss.raw_loss),
            raw_huber_beta=float(cfg_train.loss.raw_huber_beta),
            log_clip_min=float(cfg_train.loss.log_clip_min),
            log_clip_max=float(cfg_train.loss.log_clip_max),
            warmup_epochs=int(cfg_train.loss.alpha_warmup_epochs),
        ).to(device)

    raise ValueError(f"Unknown loss.name: {cfg_train.loss.name}")


def run_training_once(cfg_train, device: torch.device, run: wandb.sdk.wandb_run.Run) -> float:
    """1 run 分の学習（fold 1つ）を実行して best score を返す。

    Notes:
        - Sweep はまず “fold0 だけ” で回すのが楽＆高速。
          ベスト候補を見つけたら、別途 100_train_model で全fold学習して確認する。
    """
    # 再現性（runごとに seed は固定でOK：差分はハイパラだけにする）
    set_seed(int(cfg_train.seed))

    # ===== load df_pivot =====
    pp_dir = Path(str(cfg_train.pp_dir)) / str(cfg_train.preprocess_ver)
    pivot_path = pp_dir / str(cfg_train.pivot_csv_name)
    df = pd.read_csv(pivot_path)

    fold = int(list(cfg_train.folds)[0])
    fold_col = str(cfg_train.fold_col)

    trn_df = df[df[fold_col] != fold].reset_index(drop=True)
    val_df = df[df[fold_col] == fold].reset_index(drop=True)

    # ===== transforms =====
    train_tfm = build_transforms(cfg_train, is_train=True)
    valid_tfm = build_transforms(cfg_train, is_train=False)

    # ===== dataset / loader =====
    train_ds = CsiroDataset(
        df=trn_df,
        image_root=str(cfg_train.input_dir),
        target_cols=cfg_train.target_cols,
        transform=train_tfm,
        use_log1p_target=bool(cfg_train.use_log1p_target),
        return_target=True,
    )
    valid_ds = CsiroDataset(
        df=val_df,
        image_root=str(cfg_train.input_dir),
        target_cols=cfg_train.target_cols,
        transform=valid_tfm,
        use_log1p_target=bool(cfg_train.use_log1p_target),
        return_target=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg_train.train.batch_size),
        shuffle=True,
        num_workers=int(cfg_train.num_workers),
        pin_memory=bool(cfg_train.pin_memory),
        persistent_workers=bool(cfg_train.persistent_workers),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg_train.train.batch_size),
        shuffle=False,
        num_workers=int(cfg_train.num_workers),
        pin_memory=bool(cfg_train.pin_memory),
        persistent_workers=bool(cfg_train.persistent_workers),
        drop_last=False,
    )

    # ===== model =====
    model = ConvNeXtRegressor(
        backbone=str(cfg_train.model.backbone),
        pretrained=bool(cfg_train.model.pretrained),
        num_targets=len(cfg_train.target_cols),
        in_chans=int(cfg_train.model.in_chans),
        drop_rate=float(cfg_train.model.drop_rate),
        drop_path_rate=float(cfg_train.model.drop_path_rate),
        head_dropout=float(getattr(cfg_train.model, "head_dropout", 0.0)),
    ).to(device)

    # ===== optimizer / scheduler / loss / scaler =====
    loss_fn = build_loss(cfg_train, device=device)
    optimizer = build_optimizer(cfg_train, model)

    total_steps = int(cfg_train.train.epochs) * len(train_loader) // int(cfg_train.train.grad_accum_steps)
    scheduler = build_scheduler(cfg_train, optimizer, total_steps=total_steps)

    scaler = get_scaler(cfg_train)  # AMP scaler

    # EMA
    ema_model = None
    if bool(cfg_train.ema.enabled):
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # ===== training loop =====
    best_metric = -np.inf
    best_epoch = -1
    no_improve = 0

    patience = int(cfg_train.early_stopping.patience) if bool(cfg_train.early_stopping.enabled) else 0
    min_delta = float(cfg_train.early_stopping.min_delta) if bool(cfg_train.early_stopping.enabled) else 0.0

    global_step = 0

    for epoch in range(1, int(cfg_train.train.epochs) + 1):
        train_loss, global_step = train_one_epoch(
            cfg=cfg_train,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            epoch=epoch,
            use_amp=bool(cfg_train.use_amp),
            max_norm=float(cfg_train.train.max_norm),
            grad_accum_steps=int(cfg_train.train.grad_accum_steps),
            log_interval=int(cfg_train.train.log_interval),
            is_main_process=True,
            wandb_run=run,
            global_step=global_step,
        )

        # valid
        val_loss, val_metric, r2_scores, _ = valid_one_epoch(
            cfg=cfg_train,
            model=(ema_model if ema_model is not None else model),
            loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            use_amp=bool(cfg_train.use_amp),
            use_log1p_target=bool(cfg_train.use_log1p_target),
            is_main_process=True,
            wandb_run=run,
            global_step=global_step,
            target_names=list(cfg_train.target_cols),
            return_oof=False,
        )

        # best update
        if val_metric > best_metric + min_delta:
            best_metric = float(val_metric)
            best_epoch = int(epoch)
            no_improve = 0

            # sweep の目的メトリック（最後に最大値を残す）
            run.log({"best/weighted_r2": best_metric, "best/epoch": best_epoch}, step=global_step)
        else:
            no_improve += 1

        # EMA update
        if ema_model is not None:
            update_ema(ema_model, model, decay=float(cfg_train.ema.decay))

        # early stopping
        if patience > 0 and no_improve >= patience:
            break

    # summary にも入れておく（sweep が見やすい）
    run.summary["best/weighted_r2"] = best_metric
    run.summary["best_epoch"] = best_epoch
    run.summary["fold"] = int(list(cfg_train.folds)[0])

    return best_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # base config
    default_cfg = Path(__file__).resolve().parents[1] / "conf" / "100_train_model" / "100_train_model_default.yaml"
    parser.add_argument("--config", type=str, default=str(default_cfg), help="base yaml path")

    # sweep mode
    parser.add_argument("--create_only", action="store_true", help="create sweep only and print sweep_id")
    parser.add_argument("--agent", action="store_true", help="run as wandb agent")
    parser.add_argument("--sweep_id", type=str, default="", help="existing sweep id")
    parser.add_argument("--count", type=int, default=20, help="number of runs for agent")

    # evaluation folds/epochs (optional override)
    parser.add_argument("--fold", type=int, default=0, help="which fold to evaluate in sweep (default=0)")
    parser.add_argument("--epochs", type=int, default=-1, help="override train.epochs if >0")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load base cfg (OmegaConf)
    cfg_train = OmegaConf.load(args.config)

    # sweep は fold0 だけでまず回す（必要なら後で fold を変える）
    cfg_train.folds = [int(args.fold)]

    if args.epochs > 0:
        cfg_train.train.epochs = int(args.epochs)

    # project / entity は env を優先（なければ yaml）
    project = os.environ.get("WANDB_PROJECT", str(cfg_train.competition))
    entity = os.environ.get("WANDB_ENTITY", str(cfg_train.author))

    sweep_cfg = build_sweep_config()

    # 1) sweep 作成だけ
    if args.create_only:
        sweep_id = wandb.sweep(sweep_cfg, project=project, entity=entity)
        print(sweep_id)
        return

    # sweep_id を決める（指定がなければ作成）
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep_cfg, project=project, entity=entity)

    # agent 実行
    def _run() -> None:
        # ここで init（agent が run を管理する）
        run = wandb.init(project=project, entity=entity)

        # base cfg を deep copy して run ごとに独立させる
        cfg_run = copy.deepcopy(cfg_train)

        # wandb の提案パラメータを反映
        apply_sweep_params(cfg_run, dict(wandb.config))

        # 実験名に run_id を混ぜる（見やすくする）
        cfg_run.exp = f"{cfg_run.exp}_sweep_{run.id}"

        # 学習実行
        best = run_training_once(cfg_run, device=device, run=run)

        # 最後に best を確実に log
        run.log({"best/weighted_r2": best})
        run.finish()

    # count 回だけ回す
    wandb.agent(sweep_id, function=_run, count=int(args.count))


if __name__ == "__main__":
    main()
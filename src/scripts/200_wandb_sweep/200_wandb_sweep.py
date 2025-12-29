# /mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction/src/scripts/200_wandb_sweep/200_wandb_sweep.py
from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from utils.data import set_seed, sep, show_df
from utils.losses import WeightedMSELoss, MixedLogRawLoss
from utils.train_utils import build_optimizer, build_scheduler
from datasets.dataset import CsiroDataset
from datasets.transforms import build_transforms
from training.train import train_one_epoch, valid_one_epoch

# モデルは既存の ConvNeXtRegressor を流用（timm backbone を差し替える想定）
from models.convnext_regressor import ConvNeXtRegressor


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
    """EMA更新（指数移動平均）。"""
    msd = unwrap_model(model).state_dict()
    esd = ema_model.state_dict()
    for k in esd.keys():
        if k in msd:
            esd[k].mul_(decay).add_(msd[k], alpha=(1.0 - decay))
    ema_model.load_state_dict(esd, strict=True)


# =========================================================
# Sweep config（bayes / maximize best/weighted_r2）
# =========================================================
def build_sweep_config(project: str) -> Dict[str, Any]:
    """W&B sweep の設定辞書を生成する。"""
    return {
        "name": f"{project}-sweep-v2",
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
            # EMA（ここがポイント）
            # ema_decay==0.0 のときは EMA 無効扱い
            # -------------------------
            "ema_decay": {"values": [0.0, 0.95, 0.97, 0.99, 0.995]},

            # -------------------------
            # augmentation（例：必要なものだけ）
            # -------------------------
            "hflip_p": {"values": [0.0, 0.25, 0.5]},
            "shift_scale_rotate_p": {"values": [0.0, 0.2, 0.5]},
            "rotate_limit": {"values": [0, 10, 20]},
            "color_jitter_p": {"values": [0.0, 0.2, 0.4]},

            # -------------------------
            # MixUp / CutMix（label mixing）
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
    """100_train_model_default.yaml を読み込む。"""
    base_cfg = OmegaConf.load(base_cfg_path)
    return base_cfg


def apply_wandb_overrides(cfg: DictConfig, wb: wandb.sdk.wandb_config.Config) -> DictConfig:
    """wandb.config の値で cfg を上書きする。

    ここは「sweepで触るパラメータだけ」に限定して安全に上書きする。
    """
    cfg = copy.deepcopy(cfg)

    # ---- model ----
    cfg.model.backbone = str(wb.backbone)
    cfg.img_size = int(wb.img_size)
    cfg.model.head_dropout = float(wb.head_dropout)

    # ---- optimizer ----
    cfg.optimizer.base_lr = float(wb.lr)
    cfg.optimizer.weight_decay = float(wb.weight_decay)

    # scheduler は「一定LR」にしたい場合は base=max=min=lr に揃えるのが簡単
    if "scheduler" in cfg and "base_lr" in cfg.scheduler:
        cfg.scheduler.base_lr = float(wb.lr)
        cfg.scheduler.max_lr = float(wb.lr)
        cfg.scheduler.min_lr = float(wb.lr)

    # ---- EMA: ema_decay だけで制御（0なら無効）----
    cfg.ema.decay = float(wb.ema_decay)
    cfg.ema.enabled = bool(cfg.ema.decay > 0.0)

    # ---- augmentation ----
    cfg.augment.train.hflip_p = float(wb.hflip_p)
    cfg.augment.train.shift_scale_rotate_p = float(wb.shift_scale_rotate_p)
    cfg.augment.train.rotate_limit = int(wb.rotate_limit)
    cfg.augment.train.color_jitter_p = float(wb.color_jitter_p)

    # ---- MixUp / CutMix ----
    # train_one_epoch は cfg.mixing を見て動く（train.py を改造済み前提）
    if "mixing" not in cfg:
        cfg.mixing = OmegaConf.create({})
    cfg.mixing.prob = float(wb.mix_prob)
    cfg.mixing.mode = str(wb.mix_mode)
    cfg.mixing.alpha = float(wb.mix_alpha)

    return cfg


# =========================================================
# 1 run = 1 trial
# =========================================================
def run_one_trial(base_cfg: DictConfig) -> None:
    """wandb agent から呼ばれる 1 trial。

    注意:
      - ここで wandb.init() を呼ぶ（100_train_model.py 内で wandb.init() は呼ばない）
      - fold は base_cfg.folds を使う（高速化したいなら base_cfg.folds=[0] 推奨）
    """
    run = wandb.init()
    wb = wandb.config

    # ---- cfg を反映 ----
    cfg = apply_wandb_overrides(base_cfg, wb)

    # 🔥 run 名を見やすく（任意）
    # 例: convnext_base__img288__mix0.10__ema0.99
    run_name = (
        f"{cfg.model.backbone}__img{cfg.img_size}__"
        f"mix{cfg.mixing.prob:.2f}-{cfg.mixing.mode}__"
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
    # 今は「base_cfgに書いてある loss 設定」を使う前提（必要ならここもsweep対象に）
    if str(cfg.loss.name).lower() == "weighted_mse":
        loss_fn: nn.Module = WeightedMSELoss(list(cfg.loss.weights)).to(device)
    else:
        # mixed_log_raw（あなたが今使っている前提）
        loss_fn = MixedLogRawLoss(
            weights=list(cfg.loss.weights),
            alpha_raw=float(cfg.loss.alpha_raw),
            raw_loss=str(cfg.loss.raw_loss),
            raw_huber_beta=float(cfg.loss.raw_huber_beta),
            log_clip_min=float(cfg.loss.log_clip_min),
            log_clip_max=float(cfg.loss.log_clip_max),
            warmup_epochs=int(cfg.loss.alpha_warmup_epochs),
        ).to(device)

    # ---- folds ----
    folds = list(cfg.folds)
    fold_col = str(cfg.fold_col)
    target_cols = list(cfg.target_cols)

    best_overall = -np.inf
    best_overall_epoch = -1

    # 👍 sweepの速度優先なら folds=[0] がおすすめ
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

        # ---- model ----
        model = ConvNeXtRegressor(
            backbone=str(cfg.model.backbone),
            pretrained=bool(cfg.model.pretrained),
            num_targets=len(target_cols),
            in_chans=int(cfg.model.in_chans),
            drop_rate=float(cfg.model.drop_rate),
            drop_path_rate=float(cfg.model.drop_path_rate),
            head_dropout=float(getattr(cfg.model, "head_dropout", 0.0)),
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
        no_improve = 0

        patience = int(cfg.early_stopping.patience) if bool(cfg.early_stopping.enabled) else 0
        min_delta = float(cfg.early_stopping.min_delta) if bool(cfg.early_stopping.enabled) else 0.0

        global_step = 0

        for epoch in range(1, int(cfg.train.epochs) + 1):
            train_loss, global_step = train_one_epoch(
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

            # valid
            do_val = (epoch % int(cfg.train.val_interval) == 0)
            if do_val:
                eval_model = ema_model if ema_model is not None else unwrap_model(model)
                val_loss, val_metric, r2_scores, _ = valid_one_epoch(
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
                    return_oof=False,
                )

                improved = (val_metric > best_metric + min_delta)
                if improved:
                    best_metric = float(val_metric)
                    best_epoch = epoch
                    no_improve = 0
                    # sweep最適化用に best を逐次ログ
                    run.log({"best/weighted_r2": best_metric, "best/epoch": best_epoch}, step=global_step)
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

        # fold best を summary に載せる（見やすく）
        run.log({f"fold{fold}/best_weighted_r2": best_metric})

        # run 全体 best（複数foldなら max を取る。meanにしたければ変えてOK）
        if best_metric > best_overall:
            best_overall = best_metric
            best_overall_epoch = best_epoch

    # sweep の最適化対象（ここが一番大事）
    run.summary["best/weighted_r2"] = float(best_overall)
    run.summary["best/epoch"] = int(best_overall_epoch)

    run.finish()


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W&B sweep runner (v2).")
    parser.add_argument("--action", choices=["create", "agent"], required=True, help="create: sweep作成 / agent: 実行")
    parser.add_argument("--base_cfg", type=str, required=True, help="100_train_model_default.yaml へのパス")
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
        # ↓シェルから拾いやすいように、最後の1行は sweep_id のみ
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
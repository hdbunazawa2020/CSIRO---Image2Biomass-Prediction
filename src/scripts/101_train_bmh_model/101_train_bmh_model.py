import os
import gc
import math
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ===== original utils =====
import sys
SRC_DIR = Path(__file__).resolve().parents[2]  # .../src
sys.path.append(str(SRC_DIR))

from utils.data import set_seed

# ===== import from src  ===== 
import sys
SRC_DIR = Path(__file__).resolve().parents[2]  # .../src
sys.path.append(str(SRC_DIR))

from utils.data import set_seed, sep, show_df
from utils.wandb_utils import set_wandb
from utils.losses import WeightedMSELoss
from utils.losses import MixedLogRawLoss
from utils.losses import HurdleMixedLogRawLoss
from utils.train_utils import get_criterion, build_optimizer, build_scheduler, get_scaler
from datasets.dataset import CsiroDataset
from datasets.transforms import build_transforms
from models.convnext_regressor import ConvNeXtRegressor
from models.biomass_mil_hurdle import BiomassConvNeXtMILHurdle
from training.train import train_one_epoch, valid_one_epoch

# ===== optional deps =====
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================================================
# Distributed helpers
# =========================================================
def init_distributed(cfg) -> Tuple[bool, int, int, int, torch.device, bool]:
    """
    Returns:
      use_ddp, rank, local_rank, world_size, device, is_main_process
    """
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    ddp_enabled = bool(getattr(cfg.ddp, "enabled", False)) if hasattr(cfg, "ddp") else False
    use_ddp = (env_world_size > 1) and ddp_enabled

    if use_ddp:
        backend = getattr(cfg.ddp, "backend", "nccl")
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
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    msd = unwrap_model(model).state_dict()
    esd = ema_model.state_dict()
    for k in esd.keys():
        if k in msd:
            esd[k].mul_(decay).add_(msd[k], alpha=(1.0 - decay))
    ema_model.load_state_dict(esd, strict=True)



# =========================================================
# main
# =========================================================
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """エントリーポイント（fold loop込み）。"""
    cfg_train = cfg["101_train_bmh_model"]

    # --- DDP init ---
    use_ddp, rank, local_rank, world_size, device, is_main = init_distributed(cfg_train)

    # seed（DDPでは rank を足してずらすのが定石）
    set_seed(int(cfg_train.seed) + int(rank))

    # debug 時は exp 名を変える
    exp_name = str(cfg_train.exp)
    if bool(cfg_train.debug):
        exp_name = f"{exp_name}_debug"

    # 出力先ディレクトリ
    savedir = Path(str(cfg_train.output_dir)) / exp_name
    fold_col = str(cfg_train.fold_col)

    if is_main:
        savedir.mkdir(parents=True, exist_ok=True)
        (savedir / "yaml").mkdir(exist_ok=True)
        (savedir / "model").mkdir(exist_ok=True)
        (savedir / "oof").mkdir(exist_ok=True)
        (savedir / "history").mkdir(exist_ok=True)

        # config 保存（再現性用）
        OmegaConf.save(cfg_train, savedir / "yaml" / "config.yaml")

        print(f"[INFO] exp_dir : {savedir}")
        print(f"[INFO] device  : {device}")
        print(f"[INFO] use_ddp : {use_ddp} world_size={world_size}")

    # -----------------------------
    # df_pivot.csv 読み込み
    # -----------------------------
    pp_dir = Path(str(cfg_train.pp_dir)) / str(cfg_train.preprocess_ver)
    pivot_path = pp_dir / str(cfg_train.pivot_csv_name)
    df = pd.read_csv(pivot_path)

    if is_main:
        sep("Load df_pivot")
        print(f"[INFO] pivot_path: {pivot_path}")
        show_df(df, 3, True)

    # transforms
    train_tfm = build_transforms(cfg_train, is_train=True)
    valid_tfm = build_transforms(cfg_train, is_train=False)

    # ---- loss ----
    # loss_fn = WeightedMSELoss(list(cfg_train.loss.weights)).to(device)
    # loss_fn = MixedLogRawLoss(
    #     weights=list(cfg_train.loss.weights),
    #     alpha_raw=float(cfg_train.loss.alpha_raw),
    #     raw_loss=str(cfg_train.loss.raw_loss),
    #     raw_huber_beta=float(cfg_train.loss.raw_huber_beta),
    #     log_clip_min=float(cfg_train.loss.log_clip_min),
    #     log_clip_max=float(cfg_train.loss.log_clip_max),
    #     warmup_epochs=int(cfg_train.loss.alpha_warmup_epochs),
    # ).to(device)
    loss_fn = HurdleMixedLogRawLoss(
        weights5=list(cfg_train.loss.weights),     # 5 targets
        weights3=list(cfg_train.loss.weights3),    # 3 components
        # reg (mixed log/raw)
        reg_alpha_raw=float(cfg_train.loss.alpha_raw),
        reg_raw_loss=str(cfg_train.loss.raw_loss),
        reg_raw_huber_beta=float(cfg_train.loss.raw_huber_beta),
        log_clip_min=float(cfg_train.loss.log_clip_min),
        log_clip_max=float(cfg_train.loss.log_clip_max),
        warmup_epochs=int(cfg_train.loss.alpha_warmup_epochs),
        # hurdle aux
        lambda_presence=float(cfg_train.loss.lambda_presence),
        lambda_amount=float(cfg_train.loss.lambda_amount),
        lambda_amount_neg=float(cfg_train.loss.lambda_amount_neg),
        presence_threshold_g=float(cfg_train.loss.presence_threshold_g),
        presence_pos_weight=(list(cfg_train.loss.presence_pos_weight) if cfg_train.loss.presence_pos_weight is not None else None),
        amount_loss=str(cfg_train.loss.amount_loss),
        amount_huber_beta=float(cfg_train.loss.amount_huber_beta),
        amount_on_log=bool(cfg_train.loss.amount_on_log),
    ).to(device)

    # folds
    folds = list(cfg_train.folds)

    # -----------------------------
    # fold loop
    # -----------------------------
    for fold in folds:
        if is_main:
            sep(f"Fold {fold}")

        trn_df = df[df[fold_col] != fold].reset_index(drop=True)
        val_df = df[df[fold_col] == fold].reset_index(drop=True)

        # debug: サイズを小さくして動作確認
        if bool(cfg_train.debug):
            trn_df = trn_df.head(128).reset_index(drop=True)
            val_df = val_df.head(128).reset_index(drop=True)

        # Dataset（★tabular_cols を渡さない）
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

        # Sampler / DataLoader
        if use_ddp:
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            train_sampler = None

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

        # -----------------------------
        # Model
        # -----------------------------
        # model = ConvNeXtRegressor(
        #     backbone=str(cfg_train.model.backbone),
        #     pretrained=bool(cfg_train.model.pretrained),
        #     num_targets=len(cfg_train.target_cols),
        #     in_chans=int(cfg_train.model.in_chans),
        #     drop_rate=float(cfg_train.model.drop_rate),
        #     drop_path_rate=float(cfg_train.model.drop_path_rate),
        #     head_dropout=float(getattr(cfg_train.model, "head_dropout", 0.0)),
        # ).to(device)
        model = BiomassConvNeXtMILHurdle(
            backbone_name=str(cfg_train.model.backbone),
            pretrained=bool(cfg_train.model.pretrained),
            in_chans=int(cfg_train.model.in_chans),
            pool_dropout=float(getattr(cfg_train.model, "pool_dropout", 0.0)),
            pool_temperature=float(getattr(cfg_train.model, "pool_temperature", 1.0)),
            mil_mode=str(getattr(cfg_train.model, "mil_mode", "gated_attn")),
            mil_attn_dim=int(getattr(cfg_train.model, "mil_attn_dim", 256)),
            mil_dropout=float(getattr(cfg_train.model, "mil_dropout", 0.0)),
            head_hidden_dim=int(getattr(cfg_train.model, "head_hidden_dim", 512)),
            head_dropout=float(getattr(cfg_train.model, "head_dropout", 0.2)),
            return_attention=bool(getattr(cfg_train.model, "return_attention", False)),
        ).to(device)

        if use_ddp:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=bool(getattr(cfg_train.ddp, "find_unused_parameters", False)),
            )

        optimizer = build_optimizer(cfg_train, model)

        # scheduler は iteration step で進める
        total_steps = int(cfg_train.train.epochs) * len(train_loader) // int(cfg_train.train.grad_accum_steps)
        scheduler = build_scheduler(cfg_train, optimizer, total_steps=total_steps)

        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg_train.use_amp))

        # EMA（rank0のみ保持）
        ema_model = None
        if bool(cfg_train.ema.enabled) and is_main:
            ema_model = copy.deepcopy(unwrap_model(model)).to(device)
            ema_model.eval()
            for p in ema_model.parameters():
                p.requires_grad_(False)

        # wandb（rank0のみ）
        run = None
        run_name = f"{cfg_train.exp}_fold{fold}"
        if cfg_train.use_wandb and is_main:
            run = wandb.init(
                project=cfg_train.competition,
                entity=cfg_train.author,
                name=run_name,
                config=OmegaConf.to_container(cfg["101_train_bmh_model"], resolve=True),
            )
            print(f"[INFO] WandB logging enabled. run_name={run_name}")

        # early stopping
        patience = int(cfg_train.early_stopping.patience) if bool(cfg_train.early_stopping.enabled) else 0
        min_delta = float(cfg_train.early_stopping.min_delta) if bool(cfg_train.early_stopping.enabled) else 0.0

        best_metric = -np.inf
        best_epoch = -1
        no_improve = 0
        global_step = 0

        history = []  # 後で fold ごとの学習推移を csv に保存

        # -----------------------------
        # epoch loop
        # -----------------------------
        for epoch in range(1, int(cfg_train.train.epochs) + 1):
            if use_ddp:
                train_sampler.set_epoch(epoch)

            # ---- train ----
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
                is_main_process=is_main,
                wandb_run=run,
                global_step=global_step,
            )

            # ---- valid（rank0のみ）----
            do_val = (epoch % int(cfg_train.train.val_interval) == 0)
            if is_main and do_val:
                eval_model = ema_model if ema_model is not None else unwrap_model(model)

                val_loss, val_metric, r2_scores, oof = valid_one_epoch(
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
                    target_names=cfg_train.target_cols,
                    return_oof=True,
                )

                # console
                print(
                    f"[Fold {fold}][Epoch {epoch}] "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"weighted_r2={val_metric:.5f} r2={np.round(r2_scores, 4)}"
                )

                history.append(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "weighted_r2": val_metric,
                        **{f"r2_{t}": float(r) for t, r in zip(cfg_train.target_cols, r2_scores)},
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )

                # ---- best 更新 ----
                improved = (val_metric > best_metric + min_delta)
                if improved:
                    best_metric = val_metric
                    best_epoch = epoch
                    no_improve = 0

                    # model save（EMAを使っているなら EMA の重みを保存）
                    ckpt_path = savedir / "model" / f"best_fold{fold}.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": eval_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_metric": float(best_metric),
                            "target_cols": cfg_train.target_cols,
                            "use_log1p_target": bool(cfg_train.use_log1p_target),
                        },
                        ckpt_path,
                    )
                    print(f"🔥 Best updated: {best_metric:.5f} (epoch={best_epoch}) -> {ckpt_path}")

                    # OOF save（解析しやすいように raw/log を保存）
                    oof_path = savedir / "oof" / f"oof_fold{fold}.npz"
                    np.savez_compressed(oof_path, **oof)
                    print(f"[INFO] Saved OOF: {oof_path}")

                else:
                    no_improve += 1

                # ---- early stopping ----
                stop_flag = 0
                if patience > 0 and no_improve >= patience:
                    print(f"[INFO] Early stopping: no_improve={no_improve} patience={patience}")
                    stop_flag = 1
            else:
                stop_flag = 0

            # ---- EMA update（rank0のみ）----
            if ema_model is not None and is_main:
                update_ema(ema_model, model, decay=float(cfg_train.ema.decay))

            # ---- DDP: stop_flag を全rankで共有 ----
            if use_ddp:
                t = torch.tensor([stop_flag], device=device)
                dist.broadcast(t, src=0)
                stop_flag = int(t.item())

            if stop_flag == 1:
                break

            gc.collect()
            torch.cuda.empty_cache()

        # history 保存（foldごと）
        if is_main:
            hist_df = pd.DataFrame(history)
            hist_path = savedir / "history" / f"history_fold{fold}.csv"
            hist_df.to_csv(hist_path, index=False)
            print(f"[INFO] Saved history: {hist_path}")
            print(f"[Fold {fold}] best_metric={best_metric:.5f} best_epoch={best_epoch}")

        if run is not None:
            run.finish()

        if use_ddp:
            dist.barrier()

    cleanup_distributed(use_ddp)


if __name__ == "__main__":
    main()
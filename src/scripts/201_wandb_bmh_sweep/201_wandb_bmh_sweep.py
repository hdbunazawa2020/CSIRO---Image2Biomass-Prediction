# =========================================================
# 201_wandb_bmh_sweep.py
#   - W&B Sweep: create / agent を1ファイルで実行するためのスクリプト
#   - BiomassConvNeXtMILHurdle（BMH）用
# =========================================================
from __future__ import annotations

import argparse
import gc
import os
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import yaml
import numpy as np
import pandas as pd

import wandb

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf, DictConfig

# =========================
# import from src
# =========================
import sys
SRC_DIR = Path(__file__).resolve().parents[2]  # .../src
sys.path.append(str(SRC_DIR))

from utils.data import set_seed
from utils.losses import HurdleMixedLogRawLoss
from utils.train_utils import build_optimizer, build_scheduler
from datasets.dataset import CsiroDataset
from datasets.transforms import build_transforms
from models.biomass_mil_hurdle import BiomassConvNeXtMILHurdle
from training.train import train_one_epoch, valid_one_epoch


# =========================================================
# DDP helpers
# =========================================================
def init_distributed(cfg_train: DictConfig) -> Tuple[bool, int, int, int, torch.device, bool]:
    """DDP初期化.

    Notes:
        Sweepは通常「単一GPUで高速探索」を推奨です。
        torchrun で多プロセス起動すると "agentが複数立つ" 問題が起きやすいので注意⚠️
    """
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
    """rank0のオブジェクトを全rankへ配布（DDP時のwandb config共有用）."""
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    obj_list = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# =========================================================
# YAML / Sweep config loader
# =========================================================
def load_yaml_as_dict(path: Path) -> Dict[str, Any]:
    """YAMLをdictとして読み込む."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_base_cfg(base_cfg_path: Path) -> DictConfig:
    """base_cfg（あなたの201 default yaml）をOmegaConfで読み込む."""
    cfg = OmegaConf.load(str(base_cfg_path))
    if not isinstance(cfg, DictConfig):
        raise TypeError("base_cfg を DictConfig として読み込めませんでした。")
    return cfg


def default_sweep_config_path() -> Path:
    """sweep_config.yaml のデフォルトパス（このスクリプト配置に追従）."""
    # .../src/scripts/201_wandb_bmh_sweep/201_wandb_bmh_sweep.py
    # .../src/scripts/conf/201_wandb_bmh_sweep/sweep_config.yaml
    scripts_dir = Path(__file__).resolve().parents[1]
    return scripts_dir / "conf" / "201_wandb_bmh_sweep" / "sweep_config.yaml"


# =========================================================
# Sweep override helpers
# =========================================================
def build_wandb_default_config(cfg_train: DictConfig) -> Dict[str, Any]:
    """wandb側に見せる“デフォルト設定”（フラット）を作る.

    Notes:
        - sweep yaml で指定されない項目はこの値が使われます
        - ここに載せるキー名 = sweep_config.yamlのparameters名 になります
    """
    mix_prob = float(cfg_train.mixing.p) if bool(getattr(cfg_train.mixing, "enabled", False)) else 0.0
    ema_decay = float(cfg_train.ema.decay) if bool(getattr(cfg_train.ema, "enabled", False)) else 0.0

    defaults = {
        # ---- core ----
        "fold": int(cfg_train.folds[0]) if len(cfg_train.folds) > 0 else 0,
        "epochs": int(cfg_train.train.epochs),
        "batch_size": int(cfg_train.train.batch_size),

        # ---- model ----
        "backbone": str(cfg_train.model.backbone),
        "img_size": int(cfg_train.img_size),
        "head_dropout": float(getattr(cfg_train.model, "head_dropout", 0.0)),
        "pool_temperature": float(getattr(cfg_train.model, "pool_temperature", 1.0)),
        "pool_dropout": float(getattr(cfg_train.model, "pool_dropout", 0.0)),

        # ---- optimizer ----
        "lr": float(cfg_train.optimizer.base_lr),
        "weight_decay": float(cfg_train.optimizer.weight_decay),

        # ---- loss (hurdle) ----
        "alpha_raw": float(cfg_train.loss.alpha_raw),
        "raw_loss": str(cfg_train.loss.raw_loss),
        "raw_huber_beta": float(cfg_train.loss.raw_huber_beta),

        "lambda_presence": float(cfg_train.loss.lambda_presence),
        "lambda_amount": float(cfg_train.loss.lambda_amount),
        "lambda_amount_neg": float(cfg_train.loss.lambda_amount_neg),

        "presence_threshold_g": float(cfg_train.loss.presence_threshold_g),
        "amount_loss": str(cfg_train.loss.amount_loss),
        "amount_huber_beta": float(cfg_train.loss.amount_huber_beta),
        "amount_on_log": bool(cfg_train.loss.amount_on_log),

        # ✅ 追加：weights3（3成分の重み）
        "weights3": list(cfg_train.loss.weights3),

        # ---- augment ----
        "hflip_p": float(cfg_train.augment.train.hflip_p),
        "vflip_p": float(getattr(cfg_train.augment.train, "vflip_p", 0.0)),
        "rotate_limit": int(cfg_train.augment.train.rotate_limit),
        "shift_scale_rotate_p": float(cfg_train.augment.train.shift_scale_rotate_p),
        "color_jitter_p": float(cfg_train.augment.train.color_jitter_p),

        # ---- mix / ema ----
        "mix_prob": float(mix_prob),
        "mix_mode": str(cfg_train.mixing.mode),
        "mix_alpha": float(getattr(cfg_train.mixing, "mixup_alpha", 1.0)),
        "ema_decay": float(ema_decay),

        # ---- early stop ----
        "patience": int(cfg_train.early_stopping.patience) if bool(cfg_train.early_stopping.enabled) else 0,
    }
    return defaults


def apply_wandb_overrides(cfg_train: DictConfig, wcfg: Dict[str, Any]) -> None:
    """wandb.config で cfg_train を上書きする（型もここで吸収）."""
    OmegaConf.set_struct(cfg_train, False)

    # ---- fold / epochs / batch ----
    if "fold" in wcfg:
        cfg_train.folds = [int(wcfg["fold"])]
    if "epochs" in wcfg:
        cfg_train.train.epochs = int(wcfg["epochs"])
    if "batch_size" in wcfg:
        cfg_train.train.batch_size = int(wcfg["batch_size"])

    # ---- model ----
    if "backbone" in wcfg:
        cfg_train.model.backbone = str(wcfg["backbone"])
    if "img_size" in wcfg:
        cfg_train.img_size = int(wcfg["img_size"])
    if "head_dropout" in wcfg:
        cfg_train.model.head_dropout = float(wcfg["head_dropout"])
    if "pool_temperature" in wcfg:
        cfg_train.model.pool_temperature = float(wcfg["pool_temperature"])
    if "pool_dropout" in wcfg:
        cfg_train.model.pool_dropout = float(wcfg["pool_dropout"])

    # ---- optimizer / scheduler ----
    if "lr" in wcfg:
        lr = float(wcfg["lr"])
        cfg_train.optimizer.base_lr = lr
        # warmup_cosine前提：schedulerも同じlrへ揃える（ズレ事故防止）
        cfg_train.scheduler.base_lr = lr
        cfg_train.scheduler.max_lr = lr
        cfg_train.scheduler.min_lr = lr

    if "weight_decay" in wcfg:
        cfg_train.optimizer.weight_decay = float(wcfg["weight_decay"])

    # ---- loss ----
    if "alpha_raw" in wcfg:
        cfg_train.loss.alpha_raw = float(wcfg["alpha_raw"])
    if "raw_loss" in wcfg:
        cfg_train.loss.raw_loss = str(wcfg["raw_loss"])
    if "raw_huber_beta" in wcfg:
        cfg_train.loss.raw_huber_beta = float(wcfg["raw_huber_beta"])

    if "lambda_presence" in wcfg:
        cfg_train.loss.lambda_presence = float(wcfg["lambda_presence"])
    if "lambda_amount" in wcfg:
        cfg_train.loss.lambda_amount = float(wcfg["lambda_amount"])
    if "lambda_amount_neg" in wcfg:
        cfg_train.loss.lambda_amount_neg = float(wcfg["lambda_amount_neg"])

    if "presence_threshold_g" in wcfg:
        cfg_train.loss.presence_threshold_g = float(wcfg["presence_threshold_g"])

    if "amount_loss" in wcfg:
        cfg_train.loss.amount_loss = str(wcfg["amount_loss"])
    if "amount_huber_beta" in wcfg:
        cfg_train.loss.amount_huber_beta = float(wcfg["amount_huber_beta"])
    if "amount_on_log" in wcfg:
        cfg_train.loss.amount_on_log = bool(wcfg["amount_on_log"])

    # ✅ 追加：weights3
    if "weights3" in wcfg:
        w3 = wcfg["weights3"]
        if not isinstance(w3, (list, tuple)) or len(w3) != 3:
            raise ValueError(f"weights3 は長さ3のlist/tupleを期待します。got={w3}")
        cfg_train.loss.weights3 = [float(v) for v in w3]

    # ---- augmentation ----
    if "hflip_p" in wcfg:
        cfg_train.augment.train.hflip_p = float(wcfg["hflip_p"])
    if "vflip_p" in wcfg:
        cfg_train.augment.train.vflip_p = float(wcfg["vflip_p"])
    if "rotate_limit" in wcfg:
        cfg_train.augment.train.rotate_limit = int(wcfg["rotate_limit"])
    if "shift_scale_rotate_p" in wcfg:
        cfg_train.augment.train.shift_scale_rotate_p = float(wcfg["shift_scale_rotate_p"])
    if "color_jitter_p" in wcfg:
        cfg_train.augment.train.color_jitter_p = float(wcfg["color_jitter_p"])

    # ---- mixing ----
    if "mix_prob" in wcfg:
        p = float(wcfg["mix_prob"])
        cfg_train.mixing.enabled = p > 0.0
        cfg_train.mixing.p = p
    if "mix_mode" in wcfg:
        cfg_train.mixing.mode = str(wcfg["mix_mode"])
    if "mix_alpha" in wcfg:
        alpha = float(wcfg["mix_alpha"])
        cfg_train.mixing.mixup_alpha = alpha
        cfg_train.mixing.cutmix_alpha = alpha

    # ---- EMA ----
    if "ema_decay" in wcfg:
        d = float(wcfg["ema_decay"])
        cfg_train.ema.enabled = d > 0.0
        cfg_train.ema.decay = d

    # ---- Early stopping ----
    if "patience" in wcfg:
        p = int(wcfg["patience"])
        cfg_train.early_stopping.enabled = p > 0
        cfg_train.early_stopping.patience = p

    OmegaConf.set_struct(cfg_train, True)


def make_run_name(cfg_train: DictConfig, fold: int) -> str:
    """wandb用のrun名（長くしすぎない）."""
    backbone = str(cfg_train.model.backbone)
    img = int(cfg_train.img_size)
    lr = float(cfg_train.optimizer.base_lr)
    lp = float(cfg_train.loss.lambda_presence)
    an = float(cfg_train.loss.lambda_amount_neg)
    temp = float(getattr(cfg_train.model, "pool_temperature", 1.0))
    rl = str(cfg_train.loss.raw_loss)
    w3 = list(cfg_train.loss.weights3)
    return f"{cfg_train.exp}_sweep_f{fold}__{backbone}__img{img}__lr{lr:.1e}__lp{lp:.2f}__an{an:.2f}__T{temp:.2f}__raw{rl}__w3{w3}"


# =========================================================
# One trial (called by wandb.agent)
# =========================================================
def run_one_trial(base_cfg_path: Path) -> None:
    """wandb.agent から呼ばれる「1 trial」の処理."""
    # ---- base cfg を読み直す（agentは1プロセスで複数trial回すので、毎回freshにする）----
    cfg_train = load_base_cfg(base_cfg_path)

    # --- DDP init ---
    use_ddp, rank, local_rank, world_size, device, is_main = init_distributed(cfg_train)

    # seed（DDPはrankでずらす）
    set_seed(int(cfg_train.seed) + int(rank))

    # -----------------------------
    # df_pivot.csv 読み込み
    # -----------------------------
    pp_dir = Path(str(cfg_train.pp_dir)) / str(cfg_train.preprocess_ver)
    pivot_path = pp_dir / str(cfg_train.pivot_csv_name)
    df = pd.read_csv(pivot_path)

    # -----------------------------
    # wandb init（rank0のみ）
    # -----------------------------
    run = None
    wcfg: Dict[str, Any] = {}

    if bool(cfg_train.use_wandb) and is_main:
        default_cfg = build_wandb_default_config(cfg_train)

        # 🧯 Sweep用の安全デフォルト（長すぎ/重すぎ防止）
        default_cfg.setdefault("epochs", min(int(cfg_train.train.epochs), 100))
        default_cfg.setdefault("patience", 30)
        default_cfg.setdefault("fold", int(cfg_train.folds[0]) if len(cfg_train.folds) > 0 else 0)

        run = wandb.init(
            project=str(cfg_train.competition),
            entity=str(cfg_train.author),
            name=None,  # 後でrun_nameをセット
            config=default_cfg,
        )
        wcfg = dict(wandb.config)

    # DDP時：rank0 config を共有
    wcfg = broadcast_object(wcfg, src=0)

    # config反映
    apply_wandb_overrides(cfg_train, wcfg)

    # fold（基本1つ）
    fold = int(cfg_train.folds[0]) if len(cfg_train.folds) > 0 else 0

    # run name（反映後
    if run is not None:
        run.name = make_run_name(cfg_train, fold)
        # wandb==0.23.1 では run.save(glob_str=...) が必須
        # sweepでは必須ではないので呼ばない（エラー回避）
    # -----------------------------
    # transforms
    # -----------------------------
    train_tfm = build_transforms(cfg_train, is_train=True)
    valid_tfm = build_transforms(cfg_train, is_train=False)

    # -----------------------------
    # loss
    # -----------------------------
    loss_fn = HurdleMixedLogRawLoss(
        weights5=list(cfg_train.loss.weights),
        weights3=list(cfg_train.loss.weights3),
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

    # -----------------------------
    # split
    # -----------------------------
    fold_col = str(cfg_train.fold_col)
    trn_df = df[df[fold_col] != fold].reset_index(drop=True)
    val_df = df[df[fold_col] == fold].reset_index(drop=True)

    # debug時は小さく
    if bool(cfg_train.debug):
        trn_df = trn_df.head(256).reset_index(drop=True)
        val_df = val_df.head(256).reset_index(drop=True)

    # Dataset
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

    # Sampler / Loader
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None

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
    # model
    # -----------------------------
    model = BiomassConvNeXtMILHurdle(
        backbone_name=str(cfg_train.model.backbone),
        pretrained=bool(cfg_train.model.pretrained),
        in_chans=int(cfg_train.model.in_chans),
        pool_dropout=float(getattr(cfg_train.model, "pool_dropout", 0.0)),
        pool_temperature=float(getattr(cfg_train.model, "pool_temperature", 1.0)),
        mil_mode=str(getattr(cfg_train.model, "mil_mode", "mean")),  # 4D入力なら実質未使用
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
            find_unused_parameters=bool(getattr(cfg_train.ddp, "find_unused_parameters", True)),
        )

    optimizer = build_optimizer(cfg_train, model)

    total_steps = int(cfg_train.train.epochs) * len(train_loader) // int(cfg_train.train.grad_accum_steps)
    scheduler = build_scheduler(cfg_train, optimizer, total_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg_train.use_amp))

    # early stopping
    patience = int(cfg_train.early_stopping.patience) if bool(cfg_train.early_stopping.enabled) else 0
    min_delta = float(cfg_train.early_stopping.min_delta) if bool(cfg_train.early_stopping.enabled) else 0.0

    best_metric = -np.inf
    best_epoch = -1
    no_improve = 0
    global_step = 0

    try:
        # -----------------------------
        # epoch loop
        # -----------------------------
        for epoch in range(1, int(cfg_train.train.epochs) + 1):
            if use_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # ---- train ----
            _, global_step = train_one_epoch(
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
                    target_names=cfg_train.target_cols,
                    return_oof=False,  # sweepでは保存しない（軽量化）
                )

                improved = (val_metric > best_metric + min_delta)
                if improved:
                    best_metric = float(val_metric)
                    best_epoch = int(epoch)
                    no_improve = 0
                    if run is not None:
                        # ✅ sweepの最適化指標（sweep_config.yamlと合わせる）
                        run.summary["best/weighted_r2"] = float(best_metric)
                        run.summary["best/epoch"] = int(best_epoch)
                else:
                    no_improve += 1

                if patience > 0 and no_improve >= patience:
                    stop_flag = 1

            # ---- DDP: stop_flag broadcast ----
            if use_ddp:
                t = torch.tensor([stop_flag], device=device)
                dist.broadcast(t, src=0)
                stop_flag = int(t.item())

            if stop_flag == 1:
                break

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        if is_main and run is not None:
            run.summary["best/weighted_r2"] = float(best_metric)
            run.summary["best/epoch"] = int(best_epoch)
            run.summary["fold"] = int(fold)
            run.finish()

        cleanup_distributed(use_ddp)

        # 次trialへ向けてメモリ掃除🧹
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =========================================================
# CLI: create / agent
# =========================================================
def create_sweep(sweep_cfg_path: Path, project: str, entity: str) -> str:
    """Sweepを作成して sweep_id を返す."""
    sweep_cfg = load_yaml_as_dict(sweep_cfg_path)
    sweep_id = wandb.sweep(sweep_cfg, project=project, entity=entity)
    return str(sweep_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="201 BMH W&B sweep runner")
    parser.add_argument("--action", type=str, required=True, choices=["create", "agent"])
    parser.add_argument("--base_cfg", type=str, required=True, help="base yaml path")
    parser.add_argument("--sweep_cfg", type=str, default="", help="sweep_config.yaml path (optional)")

    # create用
    parser.add_argument("--project", type=str, default="")
    parser.add_argument("--entity", type=str, default="")

    # agent用
    parser.add_argument("--sweep_id", type=str, default="")
    parser.add_argument("--count", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg_path = Path(args.base_cfg)
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"base_cfg not found: {base_cfg_path}")

    sweep_cfg_path = Path(args.sweep_cfg) if args.sweep_cfg else default_sweep_config_path()
    if args.action == "create":
        if not args.project or not args.entity:
            raise ValueError("--project / --entity は create のとき必須です。")
        if not sweep_cfg_path.exists():
            raise FileNotFoundError(f"sweep_cfg not found: {sweep_cfg_path}")

        sweep_id = create_sweep(sweep_cfg_path, project=args.project, entity=args.entity)

        # run_wandb_sweep.sh は tail -n 1 で拾うので、最後の1行を sweep_id にする
        print(sweep_id)
        return

    # agent
    if not args.sweep_id:
        raise ValueError("--sweep_id は agent のとき必須です。")

    # wandb.agent はこのプロセス内で count 回 run_one_trial を実行します
    wandb.agent(
        args.sweep_id,
        function=lambda: run_one_trial(base_cfg_path),
        count=int(args.count),
    )


if __name__ == "__main__":
    main()
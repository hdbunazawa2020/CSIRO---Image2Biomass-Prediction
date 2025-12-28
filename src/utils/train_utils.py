from __future__ import annotations

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, ReduceLROnPlateau, StepLR

# =========================================================
# Loss
# =========================================================
def get_criterion(criterion: str) -> nn.Module:
    """損失関数（criterion）を返す。

    Args:
        criterion: 損失名。
            - "l1" / "mae" : L1Loss
            - "mse"        : MSELoss
            - "huber"      : HuberLoss (delta=1.0)

    Returns:
        torch.nn.Module: 損失関数モジュール。

    Raises:
        ValueError: 未対応の損失名が指定された場合。
    """
    name = str(criterion).lower()

    if name in ["l1", "mae"]:
        return nn.L1Loss()

    if name in ["mse", "l2"]:
        return nn.MSELoss()

    if name in ["huber", "smoothl1"]:
        # PyTorch 2.x では nn.HuberLoss が利用可能
        return nn.HuberLoss(delta=1.0)

    raise ValueError(
        f"Unsupported loss function: '{criterion}'. "
        f"Supported: l1/mae, mse/l2, huber/smoothl1"
    )


# =========================================================
# Optimizer
# =========================================================
def build_optimizer(cfg, model: nn.Module) -> Optimizer:
    """cfg.optimizer から optimizer を生成する。

    想定する cfg 例（YAML）:
        optimizer:
          name: adamw
          lr: 1e-4
          weight_decay: 1e-5
          betas: [0.9, 0.999]   # optional
          momentum: 0.9         # SGD用 optional
          nesterov: true        # SGD用 optional

    Args:
        cfg: OmegaConf / SimpleNamespace など。
        model: 学習対象のモデル。

    Returns:
        torch.optim.Optimizer: optimizer。

    Raises:
        ValueError: 未対応 optimizer 名の場合。
    """
    name = str(cfg.optimizer.name).lower()
    lr = float(cfg.optimizer.base_lr)
    wd = float(getattr(cfg.optimizer, "weight_decay", 0.0))
    betas = tuple(getattr(cfg.optimizer, "betas", (0.9, 0.999)))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    if name == "sgd":
        momentum = float(getattr(cfg.optimizer, "momentum", 0.9))
        nesterov = bool(getattr(cfg.optimizer, "nesterov", False))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=momentum,
            nesterov=nesterov,
        )

    raise ValueError(
        f"Unknown optimizer: '{name}'. Supported: adamw, adam, sgd"
    )


# =========================================================
# Scheduler
# =========================================================
def build_scheduler(cfg, optimizer: Optimizer, total_steps: int) -> Optional[LRScheduler]:
    """cfg.scheduler から scheduler を生成する。

    warmup_cosine（推奨）:
      - iteration step（= batchごと）に scheduler.step() する前提
      - base_lr, max_lr, min_lr, warmup_ratio(or warmup_steps) を使う

    plateau:
      - epochごとに scheduler.step(metric) が必要
      - 学習コード側で valid metric を渡して呼び出してください

    step:
      - epochごとに scheduler.step() が必要

    Args:
        cfg: OmegaConf / SimpleNamespace など。
        optimizer: optimizer。
        total_steps: 1epochのsteps×epochsなど（iteration scheduler 用）

    Returns:
        LRScheduler or None

    Raises:
        ValueError: scheduler 名が未対応の場合。
    """
    name = str(cfg.scheduler.name).lower()

    if name in ["none", "null", "off"]:
        return None

    # -------------------------
    # warmup + cosine decay
    # -------------------------
    if name == "warmup_cosine":
        base_lr = float(cfg.scheduler.base_lr)
        max_lr = float(cfg.scheduler.max_lr)
        min_lr = float(cfg.scheduler.min_lr)

        # warmup_steps が指定されていなければ warmup_ratio から作る
        warmup_steps = getattr(cfg.scheduler, "warmup_steps", None)
        if warmup_steps is None:
            warmup_ratio = float(getattr(cfg.scheduler, "warmup_ratio", 0.05))
            warmup_steps = int(total_steps * warmup_ratio)
        warmup_steps = int(warmup_steps)

        # total_steps を明示したい場合（上書き）
        if getattr(cfg.scheduler, "total_steps", None) is not None:
            total_steps = int(cfg.scheduler.total_steps)

        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if base_lr <= 0:
            raise ValueError(f"base_lr must be > 0, got {base_lr}")

        # LambdaLR は「倍率」を返す必要がある
        def lr_lambda(step: int) -> float:
            # step: 0..total_steps-1 を想定
            step = int(step)

            # total_steps=1 のような極端ケースの安全策
            if total_steps <= 1:
                return 1.0

            # warmup が total_steps を超えるケースの安全策
            w_steps = min(max(0, warmup_steps), total_steps)

            if w_steps > 0 and step < w_steps:
                # linear warmup: base_lr -> max_lr
                t = step / float(max(1, w_steps))
                lr = base_lr + (max_lr - base_lr) * t
            else:
                # cosine decay: max_lr -> min_lr
                # t は 0..1 に正規化
                denom = float(max(1, total_steps - w_steps))
                t = (step - w_steps) / denom
                t = min(max(t, 0.0), 1.0)

                cosine = 0.5 * (1.0 + math.cos(math.pi * t))
                lr = min_lr + (max_lr - min_lr) * cosine

            return lr / base_lr

        # optimizer の lr を base_lr に揃える（lambdaの前提が崩れるのを防ぐ）
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------------------------
    # ReduceLROnPlateau
    # -------------------------
    if name == "plateau":
        mode = str(getattr(cfg.scheduler, "mode", "min"))
        factor = float(getattr(cfg.scheduler, "factor", 0.5))
        patience = int(getattr(cfg.scheduler, "patience", 5))
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )

    # -------------------------
    # StepLR
    # -------------------------
    if name == "step":
        step_size = int(getattr(cfg.scheduler, "step_size", 5))
        gamma = float(getattr(cfg.scheduler, "gamma", 0.5))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(
        f"Unknown scheduler: '{name}'. Supported: warmup_cosine, plateau, step, none"
    )


# =========================================================
# AMP scaler
# =========================================================
def get_scaler(cfg) -> "torch.amp.GradScaler":
    """AMP用の GradScaler を返す。

    Args:
        cfg: config。以下の属性を参照します。
            - cfg.use_amp: bool
            - cfg.device: "cuda", "cuda:0", "cpu" など（あれば）

    Returns:
        torch.amp.GradScaler:
            GPUなら device="cuda"、CPUなら device="cpu" を自動選択した GradScaler。

    Notes:
        - torch.amp.GradScaler は PyTorch 2.x 系で推奨されるAPIです。
        - device に "cuda:0" のような指定が来ても、GradScaler は "cuda" / "cpu" を期待するため、
          ここで正規化しています。
    """
    enabled = bool(getattr(cfg, "use_amp", True))

    # cfg.device が "cuda:0" のような文字列でも、GradScaler には "cuda" / "cpu" を渡す
    device_str = str(getattr(cfg, "device", "cuda"))
    if ("cuda" in device_str) and torch.cuda.is_available():
        scaler_device = "cuda"
    else:
        scaler_device = "cpu"

    # torch.amp.GradScaler（device引数あり）
    try:
        from torch.amp import GradScaler
        return GradScaler(device=scaler_device, enabled=enabled)
    except Exception:
        # 古い環境向けフォールバック（cuda専用）
        from torch.cuda.amp import GradScaler
        return GradScaler(enabled=enabled)
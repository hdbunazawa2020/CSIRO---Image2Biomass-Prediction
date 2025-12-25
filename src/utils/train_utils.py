import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR

import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast

def get_criterion(criterion):
    if criterion == "l1":
        return nn.L1Loss()
    elif criterion == "mse":
        return nn.MSELoss()
    elif criterion == "huber":
        return nn.HuberLoss(delta=1.0)  # PyTorch 2.0+￥
    else:
        raise ValueError(f"Unsupported loss function: {criterion}")

def get_optimizer(config, model):
    """オプティマイザの取得"""
    if config.optimizer.name == "adam":
        return Adam(model.parameters(), lr=config.optimizer.lr)
    elif config.optimizer.name == "adamw":
        return AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, fused=True)
    elif config.optimizer.name == "sgd":
        return SGD(model.parameters(), lr=config.optimizer.lr, momentum=config.optimizer.momentum, weight_decay=config.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer.name}")

class ConstantCosineLR(_LRScheduler):
    """
    前半は一定、後半はCosineAnnealingする学習率スケジューラ。
    """
    def __init__(self, optimizer, total_steps, pct_cosine=0.5, last_epoch=-1):
        self.total_steps = total_steps
        self.milestone = int(total_steps * (1 - pct_cosine))
        self.cosine_steps = max(self.total_steps - self.milestone, 1)
        self.min_lr = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.milestone:
            factor = 1.0
        else:
            s = step - self.milestone
            factor = 0.5 * (1 + math.cos(math.pi * s / self.cosine_steps))
        return [base_lr * factor for base_lr in self.base_lrs]

import math
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineLR(LambdaLR):
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        base_lr: float = 1e-5,
        max_lr: float = 1e-3,
        min_lr: float = 0.0,
        hold_min_steps: int = 0,      # ★ 追加: min_lr でホールドするステップ数（0なら無制限にホールド）
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps is None:
            warmup_steps = int(total_steps * 0.05)
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps({warmup_steps}) must be < total_steps({total_steps})")
        if base_lr <= 0:
            raise ValueError(f"base_lr must be positive, got {base_lr}")

        # コサイン減衰に使えるステップ数（warmup と hold を除いた部分）
        cosine_steps = max(1, total_steps - warmup_steps - max(0, hold_min_steps))

        def lr_lambda(current_step: int):
            # ① Warmup フェーズ
            if current_step < warmup_steps:
                # 線形ウォームアップ: base_lr → max_lr
                # 比率として返すので base_lr で割る
                lr = base_lr + (max_lr - base_lr) * (current_step / max(1, warmup_steps))
                return lr / base_lr

            # ② min_lr でホールドするフェーズ
            #    total_steps - hold_min_steps 以降はずっと min_lr
            if current_step >= total_steps - max(0, hold_min_steps):
                return min_lr / base_lr

            # ③ コサイン減衰フェーズ
            #    warmup_steps ～ (total_steps - hold_min_steps) の間で cos 減衰
            step_in_cos = current_step - warmup_steps
            progress = step_in_cos / float(cosine_steps)
            # 念のため [0, 1] にクランプ
            progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 → 0 へ
            target_lr = min_lr + (max_lr - min_lr) * cosine      # max_lr → min_lr
            return target_lr / base_lr

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)

import math
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau, LambdaLR
)
from utils.train_utils import WarmupCosineLR, ConstantCosineLR 

def get_scheduler(config, optimizer, train_loader_len=None):
    name = config.scheduler.name.lower()

    epochs = getattr(config, "epochs", None)
    grad_accum = getattr(config, "grad_accum_steps", 1)
    if train_loader_len is None:
        raise ValueError("train_loader_len (len(train_dl)) を get_scheduler に渡してください。")

    # --- total_steps の決定 ---
    if getattr(config.scheduler, "total_steps", None) is not None:
        total_steps = config.scheduler.total_steps   # ★ yamlで指定した 8050 をそのまま使う
    else:
        total_steps = math.ceil(train_loader_len / grad_accum) * epochs
        config.scheduler.total_steps = total_steps   # 自動で保存（ログ用）

    if name == "none":
        return None

    elif name == "cosine":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler.T_0,
            T_mult=config.scheduler.T_mult,
            eta_min=config.scheduler.eta_min,
        )

    elif name == "constantcosine":
        return ConstantCosineLR(
            optimizer,
            total_steps=total_steps,
            pct_cosine=config.scheduler.pct_cosine,
        )

    elif name == "warmup_cosine":
        return WarmupCosineLR(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=config.scheduler.warmup_steps,
            base_lr=config.scheduler.base_lr,
            max_lr=config.scheduler.max_lr,
            min_lr=config.scheduler.min_lr,
            hold_min_steps=getattr(config.scheduler, "hold_min_steps", 0),  # ★ 追加
            last_epoch=getattr(config.scheduler, "last_epoch", -1),
        )

    elif name == "step":
        return StepLR(
            optimizer,
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.gamma,
        )

    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=config.scheduler.mode,
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
        )

    else:
        raise ValueError(f"Unsupported scheduler: {name}")

from torch.amp import GradScaler
def get_scaler(config):
    """AMP用GradScalerの取得（configに応じて切り替え可能）"""
    # return GradScaler(enabled=getattr(config, "use_amp", True))
    return GradScaler(device=config.device, enabled=getattr(config, "use_amp", True))
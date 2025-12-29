# -*- coding: utf-8 -*-
"""
損失関数ユーティリティ（CSIRO Image2Biomass）。

このコンペは目的変数が heavy-tail（0〜200g程度）になりやすく、
log1p 空間で学習すると安定します。

ただし公式メトリックは「元スケール（g）」での誤差が効くため、
log損失だけでは高値域を詰めきれず、過小推定に寄りがちです。

そこで、
- log空間の損失（学習安定 / 相対誤差寄り）
- raw空間の損失（高値域を強く詰める / 公式スコアに寄せる）
を混ぜる MixedLogRawLoss を用意します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_1d_float_tensor(x: Union[List[float], torch.Tensor], device: torch.device) -> torch.Tensor:
    """weights のような 1D 配列を torch.Tensor に変換する。"""
    if isinstance(x, torch.Tensor):
        t = x.detach().float().to(device)
    else:
        t = torch.tensor(list(x), dtype=torch.float32, device=device)
    return t


class WeightedMSELoss(nn.Module):
    """ターゲットごとに重みを持つ MSE Loss。

    y_pred, y_true: (B, K)
    weights: (K,)  ※ 公式 weight（例: [0.1,0.1,0.1,0.2,0.5]）を想定

    Notes:
        - 「global weighted R²」は SSE を最小化するのと同値（分母は定数）なので、
          raw 空間で weighted SSE を下げる方向はスコアに直結します。
    """

    def __init__(self, weights: List[float]) -> None:
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32).view(1, -1)  # (1, K)
        self.register_buffer("weights", w)

        # 分母でスケールを合わせる（weights の和が 1 でないケースも安全に）
        self.register_buffer("w_sum", torch.tensor(float(w.sum().item()), dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """損失を計算する。

        Args:
            y_pred: (B, K)
            y_true: (B, K)

        Returns:
            loss: scalar tensor
        """
        err2 = (y_pred - y_true) ** 2  # (B, K)
        loss = (err2 * self.weights).sum(dim=1) / (self.w_sum + 1e-12)  # (B,)
        return loss.mean()


class WeightedMAELoss(nn.Module):
    """ターゲットごとに重みを持つ L1 Loss。"""

    def __init__(self, weights: List[float]) -> None:
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32).view(1, -1)
        self.register_buffer("weights", w)
        self.register_buffer("w_sum", torch.tensor(float(w.sum().item()), dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        err = torch.abs(y_pred - y_true)
        loss = (err * self.weights).sum(dim=1) / (self.w_sum + 1e-12)
        return loss.mean()


class WeightedHuberLoss(nn.Module):
    """ターゲットごとに重みを持つ Huber(SmoothL1) Loss。

    PyTorch の SmoothL1Loss(beta=...) を reduction='none' で使います。
    """

    def __init__(self, weights: List[float], beta: float = 5.0) -> None:
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32).view(1, -1)
        self.register_buffer("weights", w)
        self.register_buffer("w_sum", torch.tensor(float(w.sum().item()), dtype=torch.float32))
        self.beta = float(beta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        err = F.smooth_l1_loss(y_pred, y_true, beta=self.beta, reduction="none")  # (B,K)
        loss = (err * self.weights).sum(dim=1) / (self.w_sum + 1e-12)
        return loss.mean()


class MixedLogRawLoss(nn.Module):
    """log空間 + raw空間 の混合損失。

    目的:
        - log空間: 学習を安定させつつ、小さい値やゼロ近傍を学びやすくする
        - raw空間: 高値域の絶対誤差を詰め、公式スコア（raw基準）に寄せる

    使い方:
        train_one_epoch / valid_one_epoch の中で loss_fn(pred_log, y_log) を呼ぶだけ。
        warmup_epochs を使う場合は、train_one_epoch 側が epoch を渡して set_epoch(epoch) を呼べばOK。

    Notes:
        - pred/target は log1p 空間で入ってくる前提（Dataset側で log1p している）
        - raw へ戻す時に expm1 を使うため、オーバーフロー防止で clip する
    """

    def __init__(
        self,
        weights: List[float],
        alpha_raw: float = 0.05,
        raw_loss: str = "mse",  # "mse" | "l1" | "huber"
        raw_huber_beta: float = 5.0,
        log_clip_min: float = -20.0,
        log_clip_max: float = 20.0,
        warmup_epochs: int = 10,
    ) -> None:
        super().__init__()

        self.alpha_raw = float(alpha_raw)
        self.raw_loss = str(raw_loss).lower()
        self.raw_huber_beta = float(raw_huber_beta)

        self.log_clip_min = float(log_clip_min)
        self.log_clip_max = float(log_clip_max)
        self.warmup_epochs = int(warmup_epochs)

        # log側は基本 MSE（安定＆スムーズ）
        self.log_loss_fn = WeightedMSELoss(weights)

        # raw側は選択式
        if self.raw_loss == "mse":
            self.raw_loss_fn = WeightedMSELoss(weights)
        elif self.raw_loss == "l1":
            self.raw_loss_fn = WeightedMAELoss(weights)
        elif self.raw_loss == "huber":
            self.raw_loss_fn = WeightedHuberLoss(weights, beta=self.raw_huber_beta)
        else:
            raise ValueError(f"Unknown raw_loss: {raw_loss}")

        # epoch に応じて変化する α を内部に保持
        self._alpha_current = 0.0
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """現在 epoch を設定し、alpha の warmup を更新する。

        warmup の考え方:
            epoch=1 では raw を効かせない（0.0）
            warmup_epochs 経過後に alpha_raw へ到達

        Args:
            epoch: 1-indexed epoch
        """
        epoch = int(epoch)
        self._current_epoch = epoch

        if self.warmup_epochs <= 0:
            self._alpha_current = self.alpha_raw
            return

        # epoch=1 -> 0.0, epoch=warmup_epochs+1 -> alpha_raw
        t = (epoch - 1) / float(self.warmup_epochs)
        t = max(0.0, min(1.0, t))
        self._alpha_current = self.alpha_raw * t

    def _log_to_raw(self, x_log: torch.Tensor) -> torch.Tensor:
        """log1p 空間 -> raw 空間へ変換する（expm1）。"""
        x_log = torch.clamp(x_log, self.log_clip_min, self.log_clip_max)
        return torch.expm1(x_log)

    def forward(self, pred_log: torch.Tensor, target_log: torch.Tensor) -> Dict[str, torch.Tensor]:
        """損失を計算する。

        Args:
            pred_log: (B, K) 予測（log1p空間）
            target_log: (B, K) 教師（log1p空間）

        Returns:
            dict:
                loss:        total loss（backwardに使う）
                loss_log:    log空間の loss
                loss_raw:    raw空間の loss
                alpha_raw:   現在の alpha（tensor）
        """
        # log loss（学習空間）
        loss_log = self.log_loss_fn(pred_log, target_log)

        # raw loss（公式スコアに寄せる用）
        pred_raw = self._log_to_raw(pred_log)
        tgt_raw = self._log_to_raw(target_log)
        loss_raw = self.raw_loss_fn(pred_raw, tgt_raw)

        # total
        alpha = float(self._alpha_current)
        loss = loss_log + alpha * loss_raw

        return {
            "loss": loss,
            "loss_log": loss_log.detach(),   # ログ用（detach）
            "loss_raw": loss_raw.detach(),   # ログ用（detach）
            "alpha_raw": torch.tensor(alpha, device=pred_log.device),
        }
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



class WeightedBCEWithLogitsLoss(nn.Module):
    """ターゲットごとに重みを持つ BCEWithLogitsLoss（多ラベル想定）。

    Args:
        weights: (K,) 各ターゲットの重み（例: 3成分なら [0.1,0.1,0.1] など）
        pos_weight: (K,) 正例を重くする係数。クラス不均衡対策（任意）。
            - BCEWithLogits の pos_weight は「正例の損失を何倍するか」
            - 目安: pos_weight = (num_neg / num_pos) くらい（ただし極端に大きいと不安定）

    Notes:
        - sigmoid + BCE なので「確率の合計=1」にはなりません（各成分は独立）。
        - まさに「独立に0〜1の確率で存在を推定」したい用途に合います。
    """

    def __init__(self, weights: List[float], pos_weight: Optional[List[float]] = None) -> None:
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32).view(1, -1)  # (1, K)
        self.register_buffer("weights", w)
        self.register_buffer("w_sum", torch.tensor(float(w.sum().item()), dtype=torch.float32))

        if pos_weight is None:
            self.pos_weight = None
        else:
            # BCEWithLogits は pos_weight を (K,) でも受ける
            pw = torch.tensor(pos_weight, dtype=torch.float32).view(-1)  # (K,)
            self.register_buffer("pos_weight", pw)

    def forward(self, logits: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
        """損失を計算する。

        Args:
            logits: (B, K)
            target01: (B, K) 0/1

        Returns:
            loss: scalar
        """
        # (B,K) 要素ごとの BCE
        loss_elem = F.binary_cross_entropy_with_logits(
            logits,
            target01,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        # ターゲットごとの重み付け平均
        loss = (loss_elem * self.weights).sum(dim=1) / (self.w_sum + 1e-12)  # (B,)
        return loss.mean()


def _safe_expm1(x_log: torch.Tensor, log_clip_min: float = -20.0, log_clip_max: float = 20.0) -> torch.Tensor:
    """log1p -> raw を安全に戻す（overflow対策で clamp）。"""
    x_log = torch.clamp(x_log, log_clip_min, log_clip_max)
    return torch.expm1(x_log)


class HurdleMixedLogRawLoss(nn.Module):
    """Hurdle（presence + amount）モデル用の総合 loss。

    モデル出力（dict）に以下がある想定:
        - pred_log1p: (B,5) 最終予測（log1p）
        - presence_logits: (B,3)
        - amount_pos: (B,3)  >=0
        - (任意) pred_components: (B,3) expected

    入力ターゲットは Dataset で log1p 化済みの (B,5) を想定。

    Total loss:
        L = L_reg(5) + λ_pres*L_pres(3) + λ_amt*L_amt_pos(3) + λ_amt_neg*L_amt_neg(3)

    Args:
        weights5: 5ターゲットの重み（公式のやつ）
        weights3: 3成分の重み（例: [0.1,0.1,0.1]）
        reg_alpha_raw/reg_*: MixedLogRawLoss に渡す
        lambda_presence: presence BCE の係数
        lambda_amount: 正例 amount 回帰の係数
        lambda_amount_neg: 負例 amount 抑制の係数（0なら無効）
        presence_threshold_g: raw(g) で「存在」とみなす閾値（例: 0.0 or 0.5）
        presence_pos_weight: BCE の pos_weight（任意）
        amount_loss: "mse" / "huber"（正例回帰）
        amount_huber_beta: huber の beta
        amount_on_log: Trueなら log1p(amount_pos) で回帰（推奨）
        log_clip_min/max: expm1 の安全クリップ
        warmup_epochs: reg_loss の alpha warmup（MixedLogRawLoss に委譲）
    """

    def __init__(
        self,
        *,
        weights5: List[float],
        weights3: List[float],
        # reg loss (MixedLogRawLoss)
        reg_alpha_raw: float = 0.05,
        reg_raw_loss: str = "mse",
        reg_raw_huber_beta: float = 5.0,
        log_clip_min: float = -20.0,
        log_clip_max: float = 20.0,
        warmup_epochs: int = 0,
        # hurdle aux losses
        lambda_presence: float = 0.2,
        lambda_amount: float = 0.1,
        lambda_amount_neg: float = 0.0,
        presence_threshold_g: float = 0.0,
        presence_pos_weight: Optional[List[float]] = None,
        amount_loss: str = "mse",   # "mse" | "huber"
        amount_huber_beta: float = 5.0,
        amount_on_log: bool = True,
    ) -> None:
        super().__init__()

        self.log_clip_min = float(log_clip_min)
        self.log_clip_max = float(log_clip_max)

        # --- main regression loss (5 targets) ---
        self.reg_loss = MixedLogRawLoss(
            weights=weights5,
            alpha_raw=reg_alpha_raw,
            raw_loss=reg_raw_loss,
            raw_huber_beta=reg_raw_huber_beta,
            log_clip_min=log_clip_min,
            log_clip_max=log_clip_max,
            warmup_epochs=warmup_epochs,
        )

        # --- presence loss (3 targets) ---
        self.presence_loss = WeightedBCEWithLogitsLoss(
            weights=weights3,
            pos_weight=presence_pos_weight,
        )

        # --- amount loss settings ---
        self.lambda_presence = float(lambda_presence)
        self.lambda_amount = float(lambda_amount)
        self.lambda_amount_neg = float(lambda_amount_neg)
        self.presence_threshold_g = float(presence_threshold_g)

        self.amount_loss = str(amount_loss).lower()
        self.amount_huber_beta = float(amount_huber_beta)
        self.amount_on_log = bool(amount_on_log)

        w3 = torch.tensor(weights3, dtype=torch.float32).view(1, -1)  # (1,3)
        self.register_buffer("weights3", w3)
        self.register_buffer("w3_sum", torch.tensor(float(w3.sum().item()), dtype=torch.float32))

        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """epoch を設定（内部の reg_loss warmup を更新）。"""
        self._current_epoch = int(epoch)
        if hasattr(self.reg_loss, "set_epoch"):
            self.reg_loss.set_epoch(epoch)

    def _amount_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """amount 回帰の要素損失（B,3）を返す（reductionなし）。"""
        if self.amount_loss == "mse":
            return (pred - target) ** 2
        if self.amount_loss == "huber":
            return F.smooth_l1_loss(pred, target, beta=self.amount_huber_beta, reduction="none")
        raise ValueError(f"Unknown amount_loss: {self.amount_loss}")

    def forward(self, model_out: Union[Dict[str, torch.Tensor], torch.Tensor], target_log5: torch.Tensor) -> Dict[str, torch.Tensor]:
        """損失計算。

        Args:
            model_out: dict 出力を想定（hurdle model）。tensor なら reg_loss のみ扱うことも可能。
            target_log5: (B,5) log1p ターゲット

        Returns:
            dict:
                loss: backward 用
                loss_reg, loss_log, loss_raw, alpha_raw: reg 内訳（ログ用）
                loss_presence: presence BCE
                loss_amount_pos: 正例 amount 回帰
                loss_amount_neg: 負例 amount 抑制（任意）
        """
        # -------------------------
        # 1) main regression (5)
        # -------------------------
        if isinstance(model_out, dict):
            pred_log5 = model_out["pred_log1p"]  # (B,5)
        else:
            # 互換用：従来の tensor 出力モデルにも対応
            pred_log5 = model_out

        reg_out = self.reg_loss(pred_log5, target_log5)  # dict

        # tensor 出力モデルのときは reg のみ返す
        if not isinstance(model_out, dict):
            return {
                "loss": reg_out["loss"],
                "loss_reg": reg_out["loss"].detach(),
                "loss_log": reg_out["loss_log"],
                "loss_raw": reg_out["loss_raw"],
                "alpha_raw": reg_out["alpha_raw"],
            }

        # hurdle 用の追加出力
        presence_logits = model_out["presence_logits"]  # (B,3)
        amount_pos = model_out["amount_pos"]            # (B,3) >=0

        # -------------------------
        # 2) presence target 作成
        # -------------------------
        target_comp_log3 = target_log5[:, :3]  # (B,3) log1p
        target_comp_raw3 = _safe_expm1(target_comp_log3, self.log_clip_min, self.log_clip_max)  # (B,3)

        # 閾値で 0/1 化（0.0 なら「>0 で存在」）
        presence_target = (target_comp_raw3 > self.presence_threshold_g).float()  # (B,3)

        # presence loss（多ラベル BCE）
        loss_presence = self.presence_loss(presence_logits, presence_target)

        # -------------------------
        # 3) amount loss（正例のみ）
        # -------------------------
        # pred/target を log 空間で合わせる（推奨）
        if self.amount_on_log:
            amount_pred = torch.log1p(amount_pos.clamp_min(0.0))  # (B,3)
            amount_tgt = target_comp_log3                         # (B,3) log1p
        else:
            amount_pred = amount_pos
            amount_tgt = target_comp_raw3

        elem_err = self._amount_error(amount_pred, amount_tgt)  # (B,3)

        # 正例mask（B,3）
        pos_mask = presence_target  # 0/1
        # 重み付きで「正例がある要素だけ」平均
        num = (elem_err * pos_mask * self.weights3).sum()
        den = (pos_mask * self.weights3).sum() + 1e-12
        loss_amount_pos = num / den

        # -------------------------
        # 4) 負例 amount 抑制（任意）
        # -------------------------
        loss_amount_neg = torch.tensor(0.0, device=target_log5.device)
        if self.lambda_amount_neg > 0.0:
            neg_mask = (1.0 - pos_mask)
            # 罰則は「amount_pred が大きいほど損」にする（L1的）
            # ※log空間なら 0 に近いほど小さい
            neg_pen = torch.abs(amount_pred)  # (B,3)
            num_n = (neg_pen * neg_mask * self.weights3).sum()
            den_n = (neg_mask * self.weights3).sum() + 1e-12
            loss_amount_neg = num_n / den_n

        # -------------------------
        # 5) total
        # -------------------------
        loss = (
            reg_out["loss"]
            + self.lambda_presence * loss_presence
            + self.lambda_amount * loss_amount_pos
            + self.lambda_amount_neg * loss_amount_neg
        )

        return {
            "loss": loss,
            # --- logging用（detach推奨） ---
            "loss_reg": reg_out["loss"].detach(),
            "loss_log": reg_out["loss_log"],
            "loss_raw": reg_out["loss_raw"],
            "alpha_raw": reg_out["alpha_raw"],
            "loss_presence": loss_presence.detach(),
            "loss_amount_pos": loss_amount_pos.detach(),
            "loss_amount_neg": loss_amount_neg.detach(),
        }
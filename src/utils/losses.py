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
from typing import Any, Dict, List, Optional, Union
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


# class MixedLogRawLoss(nn.Module):
#     """log空間 + raw空間 の混合損失。

#     目的:
#         - log空間: 学習を安定させつつ、小さい値やゼロ近傍を学びやすくする
#         - raw空間: 高値域の絶対誤差を詰め、公式スコア（raw基準）に寄せる

#     使い方:
#         train_one_epoch / valid_one_epoch の中で loss_fn(pred_log, y_log) を呼ぶだけ。
#         warmup_epochs を使う場合は、train_one_epoch 側が epoch を渡して set_epoch(epoch) を呼べばOK。

#     Notes:
#         - pred/target は log1p 空間で入ってくる前提（Dataset側で log1p している）
#         - raw へ戻す時に expm1 を使うため、オーバーフロー防止で clip する
#     """

#     def __init__(
#         self,
#         weights: List[float],
#         alpha_raw: float = 0.05,
#         raw_loss: str = "mse",  # "mse" | "l1" | "huber"
#         raw_huber_beta: float = 5.0,
#         log_clip_min: float = -20.0,
#         log_clip_max: float = 20.0,
#         warmup_epochs: int = 10,
#     ) -> None:
#         super().__init__()

#         self.alpha_raw = float(alpha_raw)
#         self.raw_loss = str(raw_loss).lower()
#         self.raw_huber_beta = float(raw_huber_beta)

#         self.log_clip_min = float(log_clip_min)
#         self.log_clip_max = float(log_clip_max)
#         self.warmup_epochs = int(warmup_epochs)

#         # log側は基本 MSE（安定＆スムーズ）
#         self.log_loss_fn = WeightedMSELoss(weights)

#         # raw側は選択式
#         if self.raw_loss == "mse":
#             self.raw_loss_fn = WeightedMSELoss(weights)
#         elif self.raw_loss == "l1":
#             self.raw_loss_fn = WeightedMAELoss(weights)
#         elif self.raw_loss == "huber":
#             self.raw_loss_fn = WeightedHuberLoss(weights, beta=self.raw_huber_beta)
#         else:
#             raise ValueError(f"Unknown raw_loss: {raw_loss}")

#         # epoch に応じて変化する α を内部に保持
#         self._alpha_current = 0.0
#         self._current_epoch = 0

#     def set_epoch(self, epoch: int) -> None:
#         """現在 epoch を設定し、alpha の warmup を更新する。

#         warmup の考え方:
#             epoch=1 では raw を効かせない（0.0）
#             warmup_epochs 経過後に alpha_raw へ到達

#         Args:
#             epoch: 1-indexed epoch
#         """
#         epoch = int(epoch)
#         self._current_epoch = epoch

#         if self.warmup_epochs <= 0:
#             self._alpha_current = self.alpha_raw
#             return

#         # epoch=1 -> 0.0, epoch=warmup_epochs+1 -> alpha_raw
#         t = (epoch - 1) / float(self.warmup_epochs)
#         t = max(0.0, min(1.0, t))
#         self._alpha_current = self.alpha_raw * t

#     def _log_to_raw(self, x_log: torch.Tensor) -> torch.Tensor:
#         """log1p 空間 -> raw 空間へ変換する（expm1）。"""
#         x_log = torch.clamp(x_log, self.log_clip_min, self.log_clip_max)
#         return torch.expm1(x_log)

#     def forward(self, pred_log: torch.Tensor, target_log: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """損失を計算する。

#         Args:
#             pred_log: (B, K) 予測（log1p空間）
#             target_log: (B, K) 教師（log1p空間）

#         Returns:
#             dict:
#                 loss:        total loss（backwardに使う）
#                 loss_log:    log空間の loss
#                 loss_raw:    raw空間の loss
#                 alpha_raw:   現在の alpha（tensor）
#         """
#         # log loss（学習空間）
#         loss_log = self.log_loss_fn(pred_log, target_log)

#         # raw loss（公式スコアに寄せる用）
#         pred_raw = self._log_to_raw(pred_log)
#         tgt_raw = self._log_to_raw(target_log)
#         loss_raw = self.raw_loss_fn(pred_raw, tgt_raw)

#         # total
#         alpha = float(self._alpha_current)
#         loss = loss_log + alpha * loss_raw

#         return {
#             "loss": loss,
#             "loss_log": loss_log.detach(),   # ログ用（detach）
#             "loss_raw": loss_raw.detach(),   # ログ用（detach）
#             "alpha_raw": torch.tensor(alpha, device=pred_log.device),
#         }
"""
MixedLogRawLoss:
    - log1p空間のloss（安定）
    - raw(g)空間のloss（高Total対策）
を混ぜるloss。

今回の拡張:
    - Dry_Total_g の raw MSE を「追加で」強くする項目（alpha_raw_total）を追加
    - 既存の alpha_raw / raw_loss は維持しつつ、Totalだけ別途押し込める
"""
def _extract_pred_log1p(model_out: Any) -> torch.Tensor:
    """model_out から log1p予測 (B,K) を取り出す。

    Args:
        model_out: Tensor または dict（BiomassConvNeXtMILHurdleの出力など）

    Returns:
        pred_log1p: (B, K)

    Raises:
        KeyError: dictに pred_log1p がない場合
    """
    if isinstance(model_out, dict):
        if "pred_log1p" not in model_out:
            raise KeyError(f"model_out dict missing 'pred_log1p'. keys={list(model_out.keys())}")
        return model_out["pred_log1p"]
    return model_out


def _huber_per_elem(x: torch.Tensor, beta: float) -> torch.Tensor:
    """SmoothL1(Huber) を要素ごとに計算（beta版）。

    Args:
        x: 任意shape
        beta: しきい値（小さいほどL1寄り、大きいほどMSE寄り）

    Returns:
        huber(x): xと同shape
    """
    beta = float(beta)
    absx = x.abs()
    # smooth_l1: 0.5*x^2/beta (|x|<beta) else |x|-0.5*beta
    return torch.where(absx < beta, 0.5 * (x * x) / beta, absx - 0.5 * beta)


class MixedLogRawLoss(nn.Module):
    """log1p loss + raw loss の混合に、Total専用の raw MSE boost を追加したloss。

    想定:
        - target は log1p 空間（CsiroDataset(use_log1p_target=True)）
        - pred も log1p 空間（TimmRegressorの出力 / BiomassConvNeXtMILHurdleの pred_log1p）

    Args:
        weights: 各ターゲットの重み（K要素）。例: [0.1,0.1,0.1,0.2,0.5]
        alpha_raw: raw全体lossの混合係数
        raw_loss: raw全体lossの種類（"l1" | "mse" | "huber"）
        raw_huber_beta: raw_loss="huber" のときだけ使用
        log_clip_min/max: expm1の前に log1p値をクリップ（数値安定化）
        warmup_epochs: alpha_raw / alpha_raw_total のwarmup（0なら無効）
        alpha_raw_total: ★追加★ Total専用 raw MSE の係数（0なら無効）
        total_index: ★追加★ Totalの列index（通常 target_cols の最後=4、-1でもOK）
    """

    def __init__(
        self,
        weights: List[float],
        alpha_raw: float = 0.0,
        raw_loss: str = "l1",
        raw_huber_beta: float = 10.0,
        log_clip_min: float = -20.0,
        log_clip_max: float = 20.0,
        warmup_epochs: int = 0,
        # --- ここから追加 ---
        alpha_raw_total: float = 0.0,
        total_index: int = -1,
    ) -> None:
        super().__init__()

        w = torch.tensor(weights, dtype=torch.float32)
        # 念のため正規化（sum=1の前提でもOKだが安全に）
        w = w / w.sum().clamp_min(1e-12)
        self.register_buffer("w", w)

        self.alpha_raw = float(alpha_raw)
        self.raw_loss = str(raw_loss).lower()
        self.raw_huber_beta = float(raw_huber_beta)

        self.log_clip_min = float(log_clip_min)
        self.log_clip_max = float(log_clip_max)
        self.warmup_epochs = int(warmup_epochs)

        self.alpha_raw_total = float(alpha_raw_total)
        self.total_index = int(total_index)

        self._epoch = 1  # warmup用に保持

    def set_epoch(self, epoch: int) -> None:
        """warmupのために現在epochをセットする。"""
        self._epoch = int(epoch)

    def _alpha_scale(self) -> float:
        """warmup係数（0〜1）"""
        if self.warmup_epochs <= 0:
            return 1.0
        # epochは1始まり想定
        t = (self._epoch - 1) / float(self.warmup_epochs)
        t = max(0.0, min(1.0, t))
        return float(t)

    def forward(self, model_out: Any, target_log1p: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward.

        Args:
            model_out: model(x) の返り値（Tensor or dict）
            target_log1p: (B, K) log1pターゲット

        Returns:
            Dict[str, Tensor]:
                - loss: 総損失
                - log_loss: log1p空間のloss
                - raw_loss: raw全体のloss（alpha_rawが0なら0）
                - raw_total_mse: Total専用のraw MSE（alpha_raw_totalが0なら0）
                - alpha_raw_eff / alpha_raw_total_eff: warmup適用後の係数
        """
        pred_log1p = _extract_pred_log1p(model_out)  # (B,K)

        # -------------------------
        # log1p loss（基本はMSE）
        # -------------------------
        diff_log = pred_log1p - target_log1p
        log_loss = (diff_log * diff_log * self.w).sum(dim=1).mean()

        # -------------------------
        # raw loss（必要なら）
        # -------------------------
        scale = self._alpha_scale()
        alpha_raw_eff = self.alpha_raw * scale
        alpha_raw_total_eff = self.alpha_raw_total * scale

        raw_loss_val = pred_log1p.new_tensor(0.0)
        raw_total_mse = pred_log1p.new_tensor(0.0)

        if (alpha_raw_eff > 0.0) or (alpha_raw_total_eff > 0.0):
            pred_log_safe = pred_log1p.clamp(self.log_clip_min, self.log_clip_max)
            targ_log_safe = target_log1p.clamp(self.log_clip_min, self.log_clip_max)

            pred_raw = torch.expm1(pred_log_safe).clamp_min(0.0)
            targ_raw = torch.expm1(targ_log_safe).clamp_min(0.0)

            err_raw = pred_raw - targ_raw  # (B,K)

            # --- raw全体loss ---
            if alpha_raw_eff > 0.0:
                if self.raw_loss == "mse":
                    per = err_raw * err_raw
                elif self.raw_loss == "l1":
                    per = err_raw.abs()
                elif self.raw_loss == "huber":
                    per = _huber_per_elem(err_raw, beta=self.raw_huber_beta)
                else:
                    raise ValueError(f"Unknown raw_loss='{self.raw_loss}' (use 'l1'|'mse'|'huber')")

                raw_loss_val = (per * self.w).sum(dim=1).mean()

            # --- Total専用 raw MSE boost（★ここが追加） ---
            if alpha_raw_total_eff > 0.0:
                k = int(pred_raw.size(1))
                idx = self.total_index if self.total_index >= 0 else (k + self.total_index)
                idx = max(0, min(idx, k - 1))

                err_total = pred_raw[:, idx] - targ_raw[:, idx]
                raw_total_mse = (err_total * err_total).mean()

        # -------------------------
        # total
        # -------------------------
        total_loss = log_loss + alpha_raw_eff * raw_loss_val + alpha_raw_total_eff * raw_total_mse

        # -------------------------
        # 返却dict（キー互換をここで保証）
        # -------------------------
        out = {
            # backward に使う本体
            "loss": total_loss,

            # ログ用（detach推奨）
            "loss_log": log_loss.detach(),
            "loss_raw": raw_loss_val.detach(),
            "raw_total_mse": raw_total_mse.detach(),

            # warmup込みの係数（wandbに出したいなら保持）
            "alpha_raw_eff": pred_log1p.new_tensor(alpha_raw_eff),
            "alpha_raw_total_eff": pred_log1p.new_tensor(alpha_raw_total_eff),

            # --- 過去互換（古いコードが log_loss/raw_loss を参照しても落ちない） ---
            "log_loss": log_loss.detach(),
            "raw_loss": raw_loss_val.detach(),
            "alpha_raw": pred_log1p.new_tensor(alpha_raw_eff),
            "alpha_raw_total": pred_log1p.new_tensor(alpha_raw_total_eff),
        }
        return out



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


# =========================================================
# soft整合用の小物ユーティリティ
# =========================================================
def _infer_target_indices(target_cols: List[str]) -> Dict[str, int]:
    """target_cols（列名の並び）から、各ターゲットの index を推定する。

    注意:
        - ここでの推定に失敗すると「誤った整合」をかけて学習を壊す可能性があるため、
          見つからない場合は例外で止めます（安全第一）。

    Args:
        target_cols: 例
            ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
            ※順序は自由。ただし5ターゲットが含まれていること。

    Returns:
        indices: dict
            {
              "green": idx,
              "clover": idx,
              "dead": idx,
              "gdm": idx,
              "total": idx
            }

    Raises:
        ValueError: 必要な列名が見つからない場合
    """
    cols = list(map(str, target_cols))

    def find_one(cands: List[str]) -> Optional[int]:
        for c in cands:
            if c in cols:
                return int(cols.index(c))
        return None

    idx_green = find_one(["Dry_Green_g"])
    idx_clover = find_one(["Dry_Clover_g"])
    idx_dead = find_one(["Dry_Dead_g"])
    idx_gdm = find_one(["GDM_g"])
    idx_total = find_one(["Dry_Total_g"])

    missing = []
    if idx_green is None: missing.append("Dry_Green_g")
    if idx_clover is None: missing.append("Dry_Clover_g")
    if idx_dead is None: missing.append("Dry_Dead_g")
    if idx_gdm is None: missing.append("GDM_g")
    if idx_total is None: missing.append("Dry_Total_g")

    if len(missing) > 0:
        raise ValueError(
            f"[MixedLogRawLoss] target_cols から必要列を推定できません: missing={missing}\n"
            f"target_cols={cols}\n"
            f"※ cfg_train.target_cols を loss に渡しているか確認してください。"
        )

    return {
        "green": idx_green,
        "clover": idx_clover,
        "dead": idx_dead,
        "gdm": idx_gdm,
        "total": idx_total,
    }


def _extract_pred_log1p(model_out: Any) -> torch.Tensor:
    """model_out から log1p 予測 (B,K) を取り出す。

    Args:
        model_out: Tensor または dict（BiomassConvNeXtMILHurdleの出力など）

    Returns:
        pred_log1p: (B, K)

    Raises:
        KeyError: dictに pred_log1p がない場合
    """
    if isinstance(model_out, dict):
        if "pred_log1p" not in model_out:
            raise KeyError(f"model_out dict missing 'pred_log1p'. keys={list(model_out.keys())}")
        return model_out["pred_log1p"]
    return model_out


def _huber_per_elem(x: torch.Tensor, beta: float) -> torch.Tensor:
    """SmoothL1(Huber) を要素ごとに計算する（beta版）。

    Args:
        x: 任意shape
        beta: しきい値（小さいほどL1寄り、大きいほどMSE寄り）

    Returns:
        huber(x): xと同shape
    """
    beta = float(beta)
    absx = x.abs()
    return torch.where(absx < beta, 0.5 * (x * x) / beta, absx - 0.5 * beta)


class MixedLogRawLoss(nn.Module):
    """log1p loss + raw loss の混合 + Total専用boost + soft整合（任意）をまとめたloss。

    概要:
        - log1p空間の weighted MSE:
            学習を安定させる（ゼロ付近・相対誤差寄り）
        - raw(g)空間の weighted loss:
            高Totalの過小推定を抑える（公式スコアに寄せる）
        - raw Total専用 MSE boost:
            Total の tail をさらに押し込む（任意）
        - soft整合（今回追加）:
            予測が以下の構造を満たすように “ゆるく” 罰則を入れる
              GDM ≒ Green + Clover
              Total ≒ GDM + Dead

    Args:
        weights: 各ターゲットの重み（K要素）。例: [0.1,0.1,0.1,0.2,0.5]
        alpha_raw: raw全体lossの混合係数（0なら無効）
        raw_loss: raw全体lossの種類（"l1" | "mse" | "huber"）
        raw_huber_beta: raw_loss="huber" のときだけ使用
        log_clip_min/max: expm1前のlog1p値クリップ（数値安定化）
        warmup_epochs: alpha_raw / alpha_raw_total / lambda_consistency の warmup（0なら無効）

        alpha_raw_total: Total専用 raw MSE の係数（0なら無効）
        total_index: Total列のindex（target_colsが渡されるなら自動推定を優先）
                    ※旧互換のため残置。基本は target_cols を渡す方が安全。

        lambda_consistency: ★今回追加★ soft整合の係数（0なら無効）
        consistency_loss: ★今回追加★ "huber" | "mse" | "l1"
        consistency_beta: ★今回追加★ huber の beta
        target_cols: ★今回追加★ 列名の並び。これを渡すと整合に必要なindexを安全に推定できる
        consistency_warmup_epochs: ★今回追加★ 整合だけ別warmupにしたい場合（Noneなら warmup_epochs を使う）
    """

    def __init__(
        self,
        weights: List[float],
        alpha_raw: float = 0.0,
        raw_loss: str = "l1",
        raw_huber_beta: float = 10.0,
        log_clip_min: float = -20.0,
        log_clip_max: float = 20.0,
        warmup_epochs: int = 0,
        # --- Total専用 boost ---
        alpha_raw_total: float = 0.0,
        total_index: int = -1,
        # --- soft整合（今回追加） ---
        lambda_consistency: float = 0.0,
        consistency_loss: str = "huber",
        consistency_beta: float = 10.0,
        target_cols: Optional[List[str]] = None,
        consistency_warmup_epochs: Optional[int] = None,
    ) -> None:
        super().__init__()

        # -------- 重み（念のため正規化）--------
        w = torch.tensor(list(weights), dtype=torch.float32)
        w = w / w.sum().clamp_min(1e-12)
        self.register_buffer("w", w)

        # -------- log/raw 混合 --------
        self.alpha_raw = float(alpha_raw)
        self.raw_loss = str(raw_loss).lower()
        self.raw_huber_beta = float(raw_huber_beta)

        self.log_clip_min = float(log_clip_min)
        self.log_clip_max = float(log_clip_max)
        self.warmup_epochs = int(warmup_epochs)

        # -------- Total boost --------
        self.alpha_raw_total = float(alpha_raw_total)
        self.total_index = int(total_index)

        # -------- soft整合（今回追加）--------
        self.lambda_consistency = float(lambda_consistency)
        self.consistency_loss = str(consistency_loss).lower()
        self.consistency_beta = float(consistency_beta)
        self.consistency_warmup_epochs = consistency_warmup_epochs

        # warmup用
        self._epoch = 1

        # 整合用のindex（target_colsが渡されるなら自動推定）
        self._idx_map: Optional[Dict[str, int]] = None
        if target_cols is not None:
            self._idx_map = _infer_target_indices(list(target_cols))

    def set_epoch(self, epoch: int) -> None:
        """warmupのために現在epochをセットする。"""
        self._epoch = int(epoch)

    def _warmup_scale(self, warmup_epochs: int) -> float:
        """epochに応じた warmup 係数（0〜1）を返す。"""
        if int(warmup_epochs) <= 0:
            return 1.0
        t = (self._epoch - 1) / float(warmup_epochs)
        t = max(0.0, min(1.0, t))
        return float(t)

    def _to_raw(self, x_log1p: torch.Tensor) -> torch.Tensor:
        """log1p -> raw へ安全に戻す（expm1 + clamp）。"""
        x = x_log1p.clamp(self.log_clip_min, self.log_clip_max)
        return torch.expm1(x).clamp_min(0.0)

    def _compute_raw_loss(self, err_raw: torch.Tensor) -> torch.Tensor:
        """raw誤差から、選択した raw loss を要素ごとに計算する（reductionなし）。"""
        if self.raw_loss == "mse":
            return err_raw * err_raw
        if self.raw_loss == "l1":
            return err_raw.abs()
        if self.raw_loss == "huber":
            return _huber_per_elem(err_raw, beta=self.raw_huber_beta)
        raise ValueError(f"Unknown raw_loss='{self.raw_loss}' (use 'l1'|'mse'|'huber')")

    def _compute_consistency_loss(
        self,
        pred_raw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """soft整合のペナルティを計算する。

        Args:
            pred_raw: (B,K) raw予測

        Returns:
            dict:
              loss_consistency: scalar
              loss_cons_gdm: scalar
              loss_cons_total: scalar

        Raises:
            ValueError: target_cols未指定で index 推定ができない場合
        """
        if self._idx_map is None:
            raise ValueError(
                "[MixedLogRawLoss] lambda_consistency>0 なのに target_cols が未指定です。\n"
                "→ loss_fn の生成時に target_cols=list(cfg_train.target_cols) を渡してください。"
            )

        i = self._idx_map
        green = pred_raw[:, i["green"]]
        clover = pred_raw[:, i["clover"]]
        dead = pred_raw[:, i["dead"]]
        gdm = pred_raw[:, i["gdm"]]
        total = pred_raw[:, i["total"]]

        # 期待される構造
        gdm_from_comp = green + clover
        total_from_comp = gdm + dead  # (= green+clover+dead)

        diff_gdm = gdm - gdm_from_comp
        diff_total = total - total_from_comp

        # ペナルティ種類
        if self.consistency_loss == "mse":
            loss_gdm = (diff_gdm * diff_gdm).mean()
            loss_total = (diff_total * diff_total).mean()
        elif self.consistency_loss == "l1":
            loss_gdm = diff_gdm.abs().mean()
            loss_total = diff_total.abs().mean()
        elif self.consistency_loss == "huber":
            loss_gdm = F.smooth_l1_loss(gdm, gdm_from_comp, beta=self.consistency_beta, reduction="mean")
            loss_total = F.smooth_l1_loss(total, total_from_comp, beta=self.consistency_beta, reduction="mean")
        else:
            raise ValueError(f"Unknown consistency_loss='{self.consistency_loss}' (use 'l1'|'mse'|'huber')")

        # 競技重みで軽く加重（Totalの重要度が高いので）
        w_gdm = self.w[i["gdm"]]
        w_total = self.w[i["total"]]
        loss_cons = w_gdm * loss_gdm + w_total * loss_total

        return {
            "loss_consistency": loss_cons,
            "loss_cons_gdm": loss_gdm,
            "loss_cons_total": loss_total,
        }

    def forward(self, model_out: Any, target_log1p: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward.

        Args:
            model_out: model(x) の返り値（Tensor or dict）
            target_log1p: (B, K) log1pターゲット

        Returns:
            Dict[str, Tensor]:
                - loss: 総損失（backwardに使用）
                - loss_log: log1p空間の損失
                - loss_raw: raw空間の損失（alpha_raw=0なら0）
                - raw_total_mse: Total専用boost項（alpha_raw_total=0なら0）
                - loss_consistency: soft整合項（lambda_consistency=0なら0）
                - alpha_raw_eff / alpha_raw_total_eff / lambda_consistency_eff: warmup後係数
        """
        pred_log1p = _extract_pred_log1p(model_out)  # (B,K)

        # -------------------------
        # 1) log1p loss（基本は weighted MSE）
        # -------------------------
        diff_log = pred_log1p - target_log1p
        loss_log = (diff_log * diff_log * self.w).sum(dim=1).mean()

        # -------------------------
        # 2) warmup係数（共通/整合専用）
        # -------------------------
        scale_main = self._warmup_scale(self.warmup_epochs)

        # 整合だけ別warmupにしたい場合
        if self.consistency_warmup_epochs is None:
            scale_cons = scale_main
        else:
            scale_cons = self._warmup_scale(int(self.consistency_warmup_epochs))

        alpha_raw_eff = self.alpha_raw * scale_main
        alpha_raw_total_eff = self.alpha_raw_total * scale_main
        lambda_cons_eff = self.lambda_consistency * scale_cons

        # -------------------------
        # 3) raw系（必要なら raw を作る）
        # -------------------------
        # raw_loss / raw_total_mse / consistency のいずれかが有効なら raw を計算
        need_raw = (alpha_raw_eff > 0.0) or (alpha_raw_total_eff > 0.0) or (lambda_cons_eff > 0.0)

        loss_raw = pred_log1p.new_tensor(0.0)
        raw_total_mse = pred_log1p.new_tensor(0.0)
        loss_consistency = pred_log1p.new_tensor(0.0)
        loss_cons_gdm = pred_log1p.new_tensor(0.0)
        loss_cons_total = pred_log1p.new_tensor(0.0)

        if need_raw:
            pred_raw = self._to_raw(pred_log1p)
            targ_raw = self._to_raw(target_log1p)
            err_raw = pred_raw - targ_raw  # (B,K)

            # --- raw全体loss ---
            if alpha_raw_eff > 0.0:
                per = self._compute_raw_loss(err_raw)  # (B,K)
                loss_raw = (per * self.w).sum(dim=1).mean()

            # --- Total専用 raw MSE boost ---
            if alpha_raw_total_eff > 0.0:
                # target_colsがあるなら totalのindexは推定を優先
                if self._idx_map is not None:
                    idx_total = int(self._idx_map["total"])
                else:
                    # 旧互換: total_index で指定（-1等を許容）
                    k = int(pred_raw.size(1))
                    idx_total = self.total_index if self.total_index >= 0 else (k + self.total_index)
                    idx_total = max(0, min(idx_total, k - 1))

                e = pred_raw[:, idx_total] - targ_raw[:, idx_total]
                raw_total_mse = (e * e).mean()

            # --- soft整合（今回追加） ---
            if lambda_cons_eff > 0.0:
                cons = self._compute_consistency_loss(pred_raw)
                loss_consistency = cons["loss_consistency"]
                loss_cons_gdm = cons["loss_cons_gdm"]
                loss_cons_total = cons["loss_cons_total"]

        # -------------------------
        # 4) total loss
        # -------------------------
        total_loss = (
            loss_log
            + alpha_raw_eff * loss_raw
            + alpha_raw_total_eff * raw_total_mse
            + lambda_cons_eff * loss_consistency
        )

        # -------------------------
        # 5) 返却dict（train.pyがそのままログできる形）
        # -------------------------
        out = {
            "loss": total_loss,

            # logging用（detach）
            "loss_log": loss_log.detach(),
            "loss_raw": loss_raw.detach(),
            "raw_total_mse": raw_total_mse.detach(),
            "loss_consistency": loss_consistency.detach(),
            "loss_cons_gdm": loss_cons_gdm.detach(),
            "loss_cons_total": loss_cons_total.detach(),

            # warmup後係数（wandbで確認用）
            "alpha_raw_eff": pred_log1p.new_tensor(alpha_raw_eff),
            "alpha_raw_total_eff": pred_log1p.new_tensor(alpha_raw_total_eff),
            "lambda_consistency_eff": pred_log1p.new_tensor(lambda_cons_eff),

            # --- 過去互換キー（古い参照があっても落ちない） ---
            "log_loss": loss_log.detach(),
            "raw_loss": loss_raw.detach(),
            "alpha_raw": pred_log1p.new_tensor(alpha_raw_eff),
            "alpha_raw_total": pred_log1p.new_tensor(alpha_raw_total_eff),
        }
        return out


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """OmegaConf / dict / 任意object から安全に値を取る."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _warmup_scale(epoch: int, warmup_epochs: int) -> float:
    """warmup係数(0→1)を返す。epochは1-indexed想定。"""
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    t = (int(epoch) - 1) / float(warmup_epochs)
    return float(max(0.0, min(1.0, t)))


def _huber_elem(err: torch.Tensor, beta: float) -> torch.Tensor:
    """Huber要素損失（reductionなし）."""
    beta = float(beta)
    absx = err.abs()
    return torch.where(absx < beta, 0.5 * (err * err) / beta, absx - 0.5 * beta)


class BiomassAuxLossWrapper(nn.Module):
    """Biomassの主loss + auxタスクlossを合成するラッパ。

    想定する model_out:
        - Tensor出力: (B,5) pred_log1p（従来モデル）
        - dict出力  : {"pred_log1p": (B,5), "aux": {...}}（aux対応モデル）

    想定する batch:
        batch["aux_target"] が dict で入っていること
        例:
          aux_target = {
            "species": LongTensor(B,),            # クラスID、未知は -1
            "ndvi": FloatTensor(B,), "ndvi_mask": FloatTensor(B,),
            "height": FloatTensor(B,), "height_mask": FloatTensor(B,)
          }

    重要:
        aux は “補助” なので、最初は弱く（weight小さめ + warmup）を推奨。
    """

    def __init__(
        self,
        main_loss: nn.Module,
        aux_cfg: Any,
        *,
        ndvi_std: float = 1.0,
        height_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.main_loss = main_loss

        self.aux_cfg = aux_cfg
        self.aux_enabled = bool(_cfg_get(aux_cfg, "enabled", False))
        self.aux_warmup_epochs = int(_cfg_get(aux_cfg, "warmup_epochs", 0))

        # ---- Species（分類） ----
        sp_cfg = _cfg_get(aux_cfg, "species", None)
        self.use_species = bool(_cfg_get(sp_cfg, "enabled", False))
        self.species_weight = float(_cfg_get(sp_cfg, "weight", 0.0))
        self.species_label_smoothing = float(_cfg_get(sp_cfg, "label_smoothing", 0.0))
        self.species_ignore_index = int(_cfg_get(sp_cfg, "ignore_index", -1))

        # ---- NDVI（回帰） ----
        ndvi_cfg = _cfg_get(aux_cfg, "ndvi", None)
        self.use_ndvi = bool(_cfg_get(ndvi_cfg, "enabled", False))
        self.ndvi_weight = float(_cfg_get(ndvi_cfg, "weight", 0.0))
        self.ndvi_loss = str(_cfg_get(ndvi_cfg, "loss", "huber")).lower()
        self.ndvi_beta = float(_cfg_get(ndvi_cfg, "beta", 0.1))
        self.ndvi_std = float(ndvi_std)

        # ---- Height（回帰） ----
        h_cfg = _cfg_get(aux_cfg, "height", None)
        self.use_height = bool(_cfg_get(h_cfg, "enabled", False))
        self.height_weight = float(_cfg_get(h_cfg, "weight", 0.0))
        self.height_loss = str(_cfg_get(h_cfg, "loss", "huber")).lower()
        self.height_beta = float(_cfg_get(h_cfg, "beta", 5.0))
        self.height_std = float(height_std)

        self._epoch = 1

    def set_epoch(self, epoch: int) -> None:
        """epochをセット（warmup用）。main_loss にも伝播する。"""
        self._epoch = int(epoch)
        if hasattr(self.main_loss, "set_epoch"):
            self.main_loss.set_epoch(epoch)

    def _reg_loss(self, pred: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor], loss_type: str, beta: float, std: float) -> torch.Tensor:
        """回帰auxのloss（mask対応、stdでスケール合わせ）."""
        pred = pred.float()
        tgt = tgt.float()

        # スケール調整：値の大きいHeightが支配しないようにする
        err = (pred - tgt) / (float(std) + 1e-12)

        if loss_type == "mse":
            per = err * err
        elif loss_type == "l1":
            per = err.abs()
        elif loss_type == "huber":
            per = _huber_elem(err, beta=beta)
        else:
            raise ValueError(f"Unknown aux reg loss: {loss_type}")

        if mask is None:
            return per.mean()

        mask = mask.float()
        return (per * mask).sum() / (mask.sum() + 1e-12)

    def forward(self, model_out: Any, target_log1p: torch.Tensor, *, batch: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Forward.

        Args:
            model_out: model(x) の出力（Tensor or dict）
            target_log1p: (B,5) log1pターゲット
            batch: Datasetが返すbatch（aux_targetなどを含む）

        Returns:
            dict: train.py がそのままログできるように dict で返す
        """
        # -------------------------
        # 1) 主loss（既存のまま）
        # -------------------------
        main_out = self.main_loss(model_out, target_log1p)
        if isinstance(main_out, torch.Tensor):
            main_out = {"loss": main_out}

        total_loss = main_out["loss"]
        out: Dict[str, torch.Tensor] = {k: v for k, v in main_out.items() if k != "loss"}  # ログ用

        # -------------------------
        # 2) aux が無ければ終了
        # -------------------------
        if (not self.aux_enabled) or (batch is None) or (not isinstance(model_out, dict)):
            out["loss"] = total_loss
            return out

        aux_pred = model_out.get("aux", None)
        aux_tgt = batch.get("aux_target", None)

        if (aux_pred is None) or (aux_tgt is None):
            out["loss"] = total_loss
            return out

        # warmup（auxは最初弱く）
        scale = _warmup_scale(self._epoch, self.aux_warmup_epochs)
        out["lambda_aux_eff"] = target_log1p.new_tensor(scale)

        loss_aux = target_log1p.new_tensor(0.0)

        # -------------------------
        # 3) Species分類（NaN対策版）
        # -------------------------
        if self.use_species and (self.species_weight > 0.0):
            logits = aux_pred["species_logits"]  # (B, C)
            y = aux_tgt["species"].to(device=logits.device, dtype=torch.long)

            valid = (y != self.species_ignore_index)
            valid_count = int(valid.sum().item())

            if valid_count == 0:
                ce = logits.new_tensor(0.0)
            else:
                # reduction='sum' で計算し、valid件数で割って mean 相当にする
                try:
                    ce_sum = F.cross_entropy(
                        logits,
                        y,
                        ignore_index=self.species_ignore_index,
                        label_smoothing=self.species_label_smoothing,
                        reduction="sum",
                    )
                except TypeError:
                    ce_sum = F.cross_entropy(
                        logits,
                        y,
                        ignore_index=self.species_ignore_index,
                        reduction="sum",
                    )

                ce = ce_sum / (valid.sum().clamp_min(1).float())

            out["loss_aux_species"] = ce.detach()
            out["aux_species_valid_rate"] = logits.new_tensor(valid.float().mean().item())
            loss_aux = loss_aux + float(self.species_weight) * ce

        # -------------------------
        # 4) NDVI回帰
        # -------------------------
        if self.use_ndvi and (self.ndvi_weight > 0.0):
            p = aux_pred["ndvi"].squeeze(-1)  # (B,)
            t = aux_tgt["ndvi"].to(device=p.device, dtype=torch.float32)
            m = aux_tgt.get("ndvi_mask", None)
            if m is not None:
                m = m.to(device=p.device, dtype=torch.float32)

            ln = self._reg_loss(p, t, m, self.ndvi_loss, self.ndvi_beta, std=self.ndvi_std)
            out["loss_aux_ndvi"] = ln.detach()
            loss_aux = loss_aux + float(self.ndvi_weight) * ln

        # -------------------------
        # 5) Height回帰
        # -------------------------
        if self.use_height and (self.height_weight > 0.0):
            p = aux_pred["height"].squeeze(-1)  # (B,)
            t = aux_tgt["height"].to(device=p.device, dtype=torch.float32)
            m = aux_tgt.get("height_mask", None)
            if m is not None:
                m = m.to(device=p.device, dtype=torch.float32)

            lh = self._reg_loss(p, t, m, self.height_loss, self.height_beta, std=self.height_std)
            out["loss_aux_height"] = lh.detach()
            loss_aux = loss_aux + float(self.height_weight) * lh

        out["loss_aux"] = loss_aux.detach()

        # 合成（warmup scale を掛ける）
        total_loss = total_loss + scale * loss_aux
        out["loss"] = total_loss
        return out
# -*- coding: utf-8 -*-
"""Training utilities (train/valid) for CSIRO Image2Biomass.

このファイルは以下に対応します:
- 旧モデル: model(x) -> Tensor (B, K)  ※多くは log1p 予測
- 新モデル: model(x) -> Dict[str, Tensor]
    例: BiomassConvNeXtMILHurdle の出力
        - pred         : (B, 5) raw(g)  [green, clover, dead, gdm, total]
        - pred_log1p   : (B, 5) log1p(pred)
        - pred_components : (B, 3) raw(g) [green, clover, dead]
        - presence_logits : (B, 3)
        - presence_prob   : (B, 3)
        - amount_pos      : (B, 3)
        - (optional) spatial_attn / tile_attn

loss_fn は以下を想定:
- 旧: loss_fn(pred_tensor, target_tensor) -> Tensor or Dict
- 新: loss_fn(model_out_dict, target_tensor) -> Dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import contextlib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.metric import global_weighted_r2_score, r2_per_target


# =========================================================
# 小物ユーティリティ
# =========================================================
def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """optimizer から現在の学習率を取得する。"""
    return float(optimizer.param_groups[0]["lr"])


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """torch.Tensor -> numpy.ndarray（CPUへ移してfloat化）"""
    return x.detach().float().cpu().numpy()


def _as_float(v: Any) -> float:
    """wandbログ用に、Tensor/数値をpython floatへ変換する。

    Args:
        v: Tensor(0-dim) or float/int

    Returns:
        float
    """
    if isinstance(v, torch.Tensor):
        return float(v.detach().float().item())
    return float(v)


def _autocast_ctx(device: torch.device, enabled: bool):
    """deviceに応じてautocastのcontextを返す。

    PyTorchのdeprecated警告を避けるため torch.amp.autocast を優先。
    """
    # CPUでも一応使えるが、基本はCUDAを想定
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def _extract_pred_log1p(model_out: Any) -> torch.Tensor:
    """モデル出力から log1p 予測 (B,5) を取り出す。

    旧モデル(Tensor)の場合:
        model_out がそのまま pred_log1p とみなす（従来互換）

    新モデル(dict)の場合:
        model_out["pred_log1p"] を使用する

    Args:
        model_out: Tensor または Dict[str, Tensor]

    Returns:
        pred_log1p: (B, K)
    """
    if isinstance(model_out, dict):
        if "pred_log1p" not in model_out:
            raise KeyError(
                "model_out is dict but missing key 'pred_log1p'. "
                "Please ensure your model returns 'pred_log1p' for metric/OOF."
            )
        return model_out["pred_log1p"]
    return model_out


def _call_loss_fn(loss_fn: nn.Module, model_out: Any, target: torch.Tensor) -> Any:
    """loss_fnを呼び出す（dictモデル出力/旧モデル両対応）。

    新モデル向けlossは model_out(dict) を受け取るのが自然だが、
    旧loss(WeightedMSEなど)は Tensor しか受けないことがある。
    その場合は pred_log1p を渡してフォールバックする。

    Args:
        loss_fn: 損失関数
        model_out: モデル出力（Tensor or dict）
        target: (B, K) target（多くはlog1p）

    Returns:
        loss_out: Tensor or dict
    """
    try:
        return loss_fn(model_out, target)
    except TypeError:
        # 旧loss互換: pred tensor を渡す
        pred_log = _extract_pred_log1p(model_out)
        return loss_fn(pred_log, target)


# =========================================================
# MixUp / CutMix
# =========================================================
def _sample_beta(alpha: float) -> float:
    """Beta(alpha, alpha) から lam をサンプルする。

    Args:
        alpha: Beta分布のパラメータ。0以下なら lam=1.0（混合しない）

    Returns:
        lam: 0〜1
    """
    if alpha is None or float(alpha) <= 0.0:
        return 1.0
    return float(np.random.beta(alpha, alpha))


def _rand_bbox(h: int, w: int, lam: float) -> Tuple[int, int, int, int]:
    """CutMix 用の矩形領域をサンプリングする。

    Args:
        h: image height
        w: image width
        lam: Betaサンプル（0〜1）

    Returns:
        x1, y1, x2, y2: 切り貼りする矩形
    """
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    return int(x1), int(y1), int(x2), int(y2)


def _mix_targets_raw_then_log(
    y: torch.Tensor,
    y_perm: torch.Tensor,
    lam: float,
    use_log1p_target: bool,
    log_clip_min: float = -20.0,
    log_clip_max: float = 20.0,
) -> torch.Tensor:
    """ターゲットを混合する（rawで混合→必要ならlog1pへ戻す）。

    Args:
        y: (B, K) ターゲット（log1p空間 or raw空間）
        y_perm: (B, K) perm後のターゲット
        lam: 元画像の寄与率（0〜1）
        use_log1p_target: y が log1p かどうか
        log_clip_min/max: expm1前の安全クリップ

    Returns:
        y_mixed: (B, K) 混合後ターゲット（元の空間に合わせる）
    """
    if use_log1p_target:
        y_raw = torch.expm1(torch.clamp(y, min=log_clip_min, max=log_clip_max))
        y_perm_raw = torch.expm1(torch.clamp(y_perm, min=log_clip_min, max=log_clip_max))

        y_mix_raw = y_raw * lam + y_perm_raw * (1.0 - lam)
        y_mix_raw = torch.clamp(y_mix_raw, min=0.0)

        return torch.log1p(y_mix_raw)

    return y * lam + y_perm * (1.0 - lam)


def apply_mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mixing_cfg: Any,
    use_log1p_target: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """batch に MixUp/CutMix を適用する。

    注意:
        - 画像が 4D (B,C,H,W) のときのみ適用推奨。
        - タイルMILの 5D (B,M,C,H,W) に対しては、まずは混合を切るのが安全。

    Args:
        x: (B, C, H, W)
        y: (B, K) ※use_log1p_target=Trueならlog1pターゲット
        mixing_cfg: cfg.mixing
        use_log1p_target: yがlog1pかどうか

    Returns:
        x_mixed, y_mixed, info
    """
    b = int(x.size(0))
    if b < 2:
        return x, y, {"mixing/used": 0.0}

    enabled = bool(getattr(mixing_cfg, "enabled", False))
    if not enabled:
        return x, y, {"mixing/used": 0.0}

    p = float(getattr(mixing_cfg, "p", 0.0))
    if p <= 0.0 or np.random.rand() > p:
        return x, y, {"mixing/used": 0.0}

    mode = str(getattr(mixing_cfg, "mode", "mixup_cutmix")).lower()
    mixup_alpha = float(getattr(mixing_cfg, "mixup_alpha", 0.4))
    cutmix_alpha = float(getattr(mixing_cfg, "cutmix_alpha", 1.0))
    switch_prob = float(getattr(mixing_cfg, "switch_prob", 0.5))

    perm = torch.randperm(b, device=x.device)
    x_perm = x[perm]
    y_perm = y[perm]

    if mode == "mixup":
        use_cutmix = False
    elif mode == "cutmix":
        use_cutmix = True
    else:
        use_cutmix = (np.random.rand() < switch_prob)

    if not use_cutmix:
        lam = _sample_beta(mixup_alpha)
        x_mixed = x * lam + x_perm * (1.0 - lam)
        y_mixed = _mix_targets_raw_then_log(y, y_perm, lam, use_log1p_target=use_log1p_target)

        return x_mixed, y_mixed, {
            "mixing/used": 1.0,
            "mixing/is_cutmix": 0.0,
            "mixing/lam": float(lam),
        }

    _, _, h, w = x.size()
    lam = _sample_beta(cutmix_alpha)
    x1, y1, x2, y2 = _rand_bbox(h, w, lam)

    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x_perm[:, :, y1:y2, x1:x2]

    area = (x2 - x1) * (y2 - y1)
    lam_adj = 1.0 - float(area) / float(h * w)

    y_mixed = _mix_targets_raw_then_log(y, y_perm, lam_adj, use_log1p_target=use_log1p_target)

    return x_mixed, y_mixed, {
        "mixing/used": 1.0,
        "mixing/is_cutmix": 1.0,
        "mixing/lam": float(lam_adj),
    }


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    cfg: Any,
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Any,
    epoch: int,
    use_amp: bool = True,
    max_norm: float = 0.0,
    grad_accum_steps: int = 1,
    log_interval: int = 50,
    is_main_process: bool = True,
    wandb_run: Optional[Any] = None,
    global_step: int = 0,
) -> Tuple[float, int]:
    """1epoch分の学習を実行する（dictモデル出力対応）。

    Args:
        cfg: hydra cfg（cfg.mixing/use_log1p_target等を参照）
        model: 学習モデル（DDPでwrapされていてもOK）
        loader: train DataLoader
        optimizer: optimizer
        scheduler: scheduler（iteration stepで進める想定）
        loss_fn: 損失関数（Tensor返し or dict返し）
        device: torch.device
        scaler: GradScaler
        epoch: 1-indexed epoch
        use_amp: AMPを使うか
        max_norm: 勾配clipのmax_norm（0なら無効）
        grad_accum_steps: 勾配蓄積ステップ数
        log_interval: tqdm表示更新間隔（batch単位）
        is_main_process: rank0かどうか
        wandb_run: wandbのrun（rank0のみ）
        global_step: optimizer step数（wandb stepとして使用）

    Returns:
        avg_loss: float（epoch平均）
        global_step: 更新後のglobal_step
    """
    model.train()

    # loss_fnがepoch依存パラメータを持つ場合（例: alpha warmup）
    if hasattr(loss_fn, "set_epoch") and callable(getattr(loss_fn, "set_epoch")):
        loss_fn.set_epoch(epoch)

    running_loss = 0.0
    n_batches = 0
    running_extra: Dict[str, float] = {}

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, disable=not is_main_process)
    for idx, batch in enumerate(pbar):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        # -------------------------------------------------
        # MixUp / CutMix（trainのみ）
        # - タイルMIL(5D)の場合はまず切るのが安全なので、4Dのときだけ適用
        # -------------------------------------------------
        mixing_cfg = getattr(cfg, "mixing", None)
        if (
            mixing_cfg is not None
            and bool(getattr(mixing_cfg, "enabled", False))
            and x.dim() == 4
        ):
            x, y, mix_info = apply_mixup_cutmix(
                x,
                y,
                mixing_cfg=mixing_cfg,
                use_log1p_target=bool(getattr(cfg, "use_log1p_target", True)),
            )
        else:
            mix_info = {"mixing/used": 0.0, "mixing/lam": 1.0, "mixing/is_cutmix": 0.0}

        with _autocast_ctx(device=device, enabled=use_amp):
            # -------------------------
            # Forward
            # -------------------------
            # 新モデル: dict出力（pred_log1p / presence_logits / amount_pos などを含む）
            # 旧モデル: tensor出力（pred_log1p相当）
            model_out = model(x)

            # -------------------------
            # Loss
            # -------------------------
            # 新lossは dict を受ける想定。旧loss互換もフォールバックで対応。
            loss_out = _call_loss_fn(loss_fn, model_out, y)

            if isinstance(loss_out, dict):
                loss = loss_out["loss"]
                extra = {k: _as_float(v) for k, v in loss_out.items() if k != "loss"}
            else:
                loss = loss_out
                extra = {}

            # 勾配蓄積
            loss = loss / float(grad_accum_steps)

        scaler.scale(loss).backward()

        do_step = ((idx + 1) % int(grad_accum_steps) == 0)
        if do_step:
            if max_norm is not None and float(max_norm) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # schedulerは「optimizer stepに同期して進める」
            if scheduler is not None:
                scheduler.step()

            global_step += 1

        # -------------------------------------------------
        # ログ集計
        # -------------------------------------------------
        running_loss += float(loss.detach().item()) * float(grad_accum_steps)
        n_batches += 1

        for k, v in extra.items():
            running_extra[k] = running_extra.get(k, 0.0) + float(v)

        # tqdm表示
        if is_main_process and (idx % int(log_interval) == 0):
            lr = get_lr(optimizer)
            pbar.set_description(f"epoch{epoch} loss{running_loss/max(1,n_batches):.4f} lr{lr:.2e}")

        # wandb logging（optimizer step単位）
        if is_main_process and (wandb_run is not None) and do_step:
            log_every = int(getattr(cfg, "wandb_log_interval_steps", 100))
            if log_every > 0 and (global_step % log_every == 0):
                log_dict = {
                    "epoch": int(epoch),
                    "train/loss": running_loss / max(1, n_batches),
                    "train/lr": get_lr(optimizer),
                    # mixing診断
                    "train/mixing_used": float(mix_info.get("mixing/used", 0.0)),
                    "train/mixing_lam": float(mix_info.get("mixing/lam", 1.0)),
                    "train/mixing_is_cutmix": float(mix_info.get("mixing/is_cutmix", 0.0)),
                }

                # loss_fnの内訳（dict返しのとき）
                for k, v_sum in running_extra.items():
                    log_dict[f"train/{k}"] = float(v_sum / max(1, n_batches))

                wandb_run.log(log_dict, step=int(global_step))

    avg_loss = running_loss / max(1, n_batches)
    return float(avg_loss), int(global_step)


# =========================================================
# Valid
# =========================================================
@torch.no_grad()
def valid_one_epoch(
    cfg: Any,
    model: nn.Module,
    loader: Any,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    use_log1p_target: bool = True,
    is_main_process: bool = True,
    wandb_run: Optional[Any] = None,
    global_step: int = 0,
    target_names: Optional[List[str]] = None,
    return_oof: bool = True,
) -> Tuple[float, float, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """1epoch分の検証を実行する（dictモデル出力対応）。

    Args:
        cfg: hydra cfg（cfg.metric.weights等を参照）
        model: 評価モデル
        loader: valid DataLoader
        loss_fn: 損失関数（Tensor返し or dict返し）
        device: torch.device
        epoch: 1-indexed epoch
        use_amp: AMPを使うか
        use_log1p_target: targetがlog1pかどうか
        is_main_process: rank0かどうか
        wandb_run: wandbのrun（rank0のみ）
        global_step: wandb step（train側のoptimizer step数）
        target_names: r2ログの名前
        return_oof: OOFを返すか

    Returns:
        val_loss: float
        global_r2: float
        r2_scores: np.ndarray shape (K,)
        oof: dict or None
    """
    model.eval()

    if hasattr(loss_fn, "set_epoch") and callable(getattr(loss_fn, "set_epoch")):
        loss_fn.set_epoch(epoch)

    loss_sum = 0.0
    n_samples = 0
    extra_sum: Dict[str, float] = {}

    ids_all: List[str] = []
    preds_all: List[np.ndarray] = []
    targs_all: List[np.ndarray] = []

    pbar = tqdm(loader, disable=not is_main_process)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        bs = int(x.size(0))

        with _autocast_ctx(device=device, enabled=use_amp):
            model_out = model(x)

            # OOF/metric用に pred_log1p を取り出す
            pred_log_tensor = _extract_pred_log1p(model_out)

            # loss
            loss_out = _call_loss_fn(loss_fn, model_out, y)
            if isinstance(loss_out, dict):
                loss = loss_out["loss"]
                extra = {k: _as_float(v) for k, v in loss_out.items() if k != "loss"}
            else:
                loss = loss_out
                extra = {}

        # ---- loss集計（サンプル数で加重平均）----
        loss_sum += float(loss.detach().item()) * bs
        n_samples += bs

        for k, v in extra.items():
            extra_sum[k] = extra_sum.get(k, 0.0) + float(v) * bs

        # ---- ids ----
        ids = batch.get("id", None)
        if ids is None:
            # Datasetによってキー名が違う場合の保険
            ids = batch.get("image_id", None)

        if isinstance(ids, (list, tuple, np.ndarray)):
            ids_all.extend([str(v) for v in ids])
        elif ids is not None:
            ids_all.append(str(ids))
        else:
            # idが無い場合もOOFは作れるが、後解析が困るので空文字で埋める
            ids_all.extend([""] * bs)

        # ---- preds/targets ----
        preds_all.append(_to_numpy(pred_log_tensor))
        targs_all.append(_to_numpy(y))

    preds_log = np.concatenate(preds_all, axis=0)
    targs_log = np.concatenate(targs_all, axis=0)

    # metricはrawスケールで計算
    if use_log1p_target:
        preds_log_safe = np.clip(preds_log, -20.0, 20.0)
        targs_log_safe = np.clip(targs_log, -20.0, 20.0)
        preds = np.expm1(preds_log_safe)
        targs = np.expm1(targs_log_safe)
    else:
        preds = preds_log
        targs = targs_log

    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    targs = np.nan_to_num(targs, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.clip(preds, 0.0, None)

    weights = np.asarray(cfg.metric.weights, dtype=np.float64)
    global_r2 = global_weighted_r2_score(targs, preds, weights)
    r2_scores = r2_per_target(targs, preds)

    val_loss = loss_sum / max(1, n_samples)

    # ---- wandb ----
    if is_main_process and (wandb_run is not None):
        if target_names is None:
            target_names = [f"t{i}" for i in range(len(r2_scores))]

        log_dict = {
            "epoch": int(epoch),
            "valid/loss": float(val_loss),
            "valid/weighted_r2": float(global_r2),
        }
        for name, r2 in zip(target_names, r2_scores):
            log_dict[f"valid/r2_{name}"] = float(r2)

        for k, v_sum in extra_sum.items():
            log_dict[f"valid/{k}"] = float(v_sum / max(1, n_samples))

        wandb_run.log(log_dict, step=int(global_step))

    oof = None
    if return_oof:
        oof = {
            "ids": np.array(ids_all),
            "preds": preds,
            "targets": targs,
            "preds_log": preds_log,
            "targets_log": targs_log,
        }

    return float(val_loss), float(global_r2), r2_scores, oof
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.metric import global_weighted_r2_score, r2_per_target


# =========================================================
# 小物ユーティリティ
# =========================================================
def get_lr(optimizer) -> float:
    """optimizer から現在の学習率を取得する。"""
    return float(optimizer.param_groups[0]["lr"])


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """torch.Tensor -> numpy.ndarray（CPUへ移してfloat化）"""
    return x.detach().float().cpu().numpy()


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

    lam は「元画像を残す比率」に近い値だが、実際は矩形の面積から再計算する。

    Args:
        h: image height
        w: image width
        lam: Betaサンプル（0〜1）

    Returns:
        x1, y1, x2, y2: 切り貼りする矩形（x方向が[ x1:x2 ], y方向が[ y1:y2 ]）
    """
    # 切り抜く面積比 = 1 - lam
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    # 中心座標をランダムに
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
    """ターゲットを混合する（重要：raw空間で混合→必要ならlog1pへ戻す）。

    背景:
        - MixUp/CutMix は「入力が線形結合（or 面積比で合成）」なので、
          ターゲットも raw（g）空間で線形結合するのが自然。
        - use_log1p_target=True のとき、y は log1p(target) で渡ってくるため、
          いったん expm1 で raw に戻してから混合し、最後に log1p に戻す。

    Args:
        y: (B, K) ターゲット（log1p空間 or raw空間）
        y_perm: (B, K) perm後のターゲット
        lam: 元画像の寄与率（0〜1）
        use_log1p_target: y が log1p かどうか
        log_clip_min/log_clip_max: expm1 の安全クリップ（発散防止）

    Returns:
        y_mixed: (B, K) 混合後ターゲット（元の空間に合わせる）
    """
    if use_log1p_target:
        # expm1 の前にクリップ（極端に大きいpred/targetでinfになるのを防ぐ）
        y_raw = torch.expm1(torch.clamp(y, min=log_clip_min, max=log_clip_max))
        y_perm_raw = torch.expm1(torch.clamp(y_perm, min=log_clip_min, max=log_clip_max))

        y_mix_raw = y_raw * lam + y_perm_raw * (1.0 - lam)
        y_mix_raw = torch.clamp(y_mix_raw, min=0.0)  # 質量は非負

        y_mix_log = torch.log1p(y_mix_raw)
        return y_mix_log

    # raw 空間ならそのまま線形結合
    return y * lam + y_perm * (1.0 - lam)


def apply_mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mixing_cfg: Any,
    use_log1p_target: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """batch に MixUp/CutMix を適用する。

    Args:
        x: (B, C, H, W)
        y: (B, K)  ※use_log1p_target=True なら log1pターゲット
        mixing_cfg: cfg.mixing（OmegaConf / SimpleNamespace / dict風）
        use_log1p_target: y が log1p かどうか

    Returns:
        x_mixed, y_mixed, info:
            info は wandb ログ等で診断できるように lam/mode を返す
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

    # どのサンプルと混ぜるか（batch内で perm）
    perm = torch.randperm(b, device=x.device)
    x_perm = x[perm]
    y_perm = y[perm]

    # mode 決定
    if mode == "mixup":
        use_cutmix = False
    elif mode == "cutmix":
        use_cutmix = True
    else:
        # mixup_cutmix
        use_cutmix = (np.random.rand() < switch_prob)

    if not use_cutmix:
        # -------------------------
        # MixUp（線形結合）
        # -------------------------
        lam = _sample_beta(mixup_alpha)
        x_mixed = x * lam + x_perm * (1.0 - lam)
        y_mixed = _mix_targets_raw_then_log(y, y_perm, lam, use_log1p_target=use_log1p_target)

        info = {
            "mixing/used": 1.0,
            "mixing/is_cutmix": 0.0,
            "mixing/lam": float(lam),
        }
        return x_mixed, y_mixed, info

    # -------------------------
    # CutMix（矩形貼り替え）
    # -------------------------
    _, _, h, w = x.size()
    lam = _sample_beta(cutmix_alpha)

    x1, y1, x2, y2 = _rand_bbox(h, w, lam)

    # 画像貼り替え（in-place気味なので clone して安全に）
    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x_perm[:, :, y1:y2, x1:x2]

    # 実際の貼り替え面積から lam を再計算（これが重要）
    area = (x2 - x1) * (y2 - y1)
    lam_adj = 1.0 - float(area) / float(h * w)

    y_mixed = _mix_targets_raw_then_log(y, y_perm, lam_adj, use_log1p_target=use_log1p_target)

    info = {
        "mixing/used": 1.0,
        "mixing/is_cutmix": 1.0,
        "mixing/lam": float(lam_adj),
    }
    return x_mixed, y_mixed, info


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    cfg,
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    use_amp: bool = True,
    max_norm: float = 0.0,
    grad_accum_steps: int = 1,
    log_interval: int = 50,
    is_main_process: bool = True,
    wandb_run=None,
    global_step: int = 0,
) -> Tuple[float, int]:
    """1epoch分の学習を実行する（画像のみ入力 + MixUp/CutMix対応）。"""

    model.train()

    # loss_fn 側に epoch を渡したい場合（例: alpha warmup）
    if hasattr(loss_fn, "set_epoch") and callable(getattr(loss_fn, "set_epoch")):
        loss_fn.set_epoch(epoch)

    running_loss = 0.0
    n_steps = 0

    # 追加ログ（MixedLossなど dict を返す場合）
    running_extra: Dict[str, float] = {}

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, disable=not is_main_process)
    for idx, batch in enumerate(pbar):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        # -------------------------------------------------
        # MixUp / CutMix（trainのみ）
        # -------------------------------------------------
        mixing_cfg = getattr(cfg, "mixing", None)
        if mixing_cfg is not None and bool(getattr(mixing_cfg, "enabled", False)):
            x, y, mix_info = apply_mixup_cutmix(
                x, y,
                mixing_cfg=mixing_cfg,
                use_log1p_target=bool(getattr(cfg, "use_log1p_target", True)),
            )
        else:
            mix_info = {"mixing/used": 0.0}

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)

            # loss が dict を返すケースも許容（将来拡張用）
            loss_out = loss_fn(pred, y)
            if isinstance(loss_out, dict):
                loss = loss_out["loss"]
                extra = {k: float(v) for k, v in loss_out.items() if k != "loss"}
            else:
                loss = loss_out
                extra = {}

            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        do_step = ((idx + 1) % grad_accum_steps == 0)
        if do_step:
            if max_norm is not None and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            global_step += 1

        # console
        running_loss += float(loss.item()) * grad_accum_steps
        n_steps += 1
        for k, v in extra.items():
            running_extra[k] = running_extra.get(k, 0.0) + float(v)

        if is_main_process and (idx % log_interval == 0):
            lr = get_lr(optimizer)
            pbar.set_description(f"epoch{epoch} loss{running_loss/max(1,n_steps):.4f} lr{lr:.2e}")

        # wandb logging（optimizer step単位）
        if is_main_process and (wandb_run is not None) and do_step:
            log_every = int(getattr(cfg, "wandb_log_interval_steps", 100))
            if log_every > 0 and (global_step % log_every == 0):
                log_dict = {
                    "train/loss": running_loss / max(1, n_steps),
                    "train/lr": get_lr(optimizer),
                    "epoch": epoch,
                    # mixing diagnostic
                    "train/mixing_used": float(mix_info.get("mixing/used", 0.0)),
                    "train/mixing_lam": float(mix_info.get("mixing/lam", 1.0)),
                    "train/mixing_is_cutmix": float(mix_info.get("mixing/is_cutmix", 0.0)),
                }
                # extra loss logs（あれば）
                for k, v in running_extra.items():
                    # 例: loss_log / loss_raw / alpha_raw など
                    log_dict[f"train/{k}"] = v / max(1, n_steps)

                wandb_run.log(log_dict, step=global_step)

    avg_loss = running_loss / max(1, n_steps)
    return float(avg_loss), int(global_step)


# =========================================================
# Valid
# =========================================================
@torch.no_grad()
def valid_one_epoch(
    cfg,
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    use_log1p_target: bool = True,
    is_main_process: bool = True,
    wandb_run=None,
    global_step: int = 0,
    target_names: Optional[List[str]] = None,
    return_oof: bool = True,
) -> Tuple[float, float, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """1epoch分の検証を実行する（画像のみ入力 / MixUp,CutMixなし）。"""

    model.eval()

    if hasattr(loss_fn, "set_epoch") and callable(getattr(loss_fn, "set_epoch")):
        loss_fn.set_epoch(epoch)

    # loss はサンプル数で加重平均にする
    loss_sum = 0.0
    n_samples = 0

    # extra loss（dict返しのため）
    extra_sum: Dict[str, float] = {}

    ids_all: List[str] = []
    preds_all: List[np.ndarray] = []
    targs_all: List[np.ndarray] = []

    pbar = tqdm(loader, disable=not is_main_process)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        bs = int(x.size(0))

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            loss_out = loss_fn(pred, y)
            if isinstance(loss_out, dict):
                loss = loss_out["loss"]
                extra = {k: float(v) for k, v in loss_out.items() if k != "loss"}
            else:
                loss = loss_out
                extra = {}

        loss_sum += float(loss.item()) * bs
        n_samples += bs

        for k, v in extra.items():
            extra_sum[k] = extra_sum.get(k, 0.0) + float(v) * bs

        ids = batch["id"]
        if isinstance(ids, (list, tuple, np.ndarray)):
            ids_all.extend([str(v) for v in ids])
        else:
            ids_all.append(str(ids))

        preds_all.append(_to_numpy(pred))
        targs_all.append(_to_numpy(y))

    preds_log = np.concatenate(preds_all, axis=0)
    targs_log = np.concatenate(targs_all, axis=0)

    # metric は raw scale で
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

    # wandb
    if is_main_process and (wandb_run is not None):
        if target_names is None:
            target_names = [f"t{i}" for i in range(len(r2_scores))]

        log_dict = {
            "valid/loss": float(val_loss),
            "valid/weighted_r2": float(global_r2),
            "epoch": epoch,
        }

        for name, r2 in zip(target_names, r2_scores):
            log_dict[f"valid/r2_{name}"] = float(r2)

        # extra loss logs
        for k, v in extra_sum.items():
            log_dict[f"valid/{k}"] = float(v / max(1, n_samples))

        wandb_run.log(log_dict, step=global_step)

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
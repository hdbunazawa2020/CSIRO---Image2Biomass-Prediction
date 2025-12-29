from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.metric import global_weighted_r2_score, r2_per_target


def get_lr(optimizer) -> float:
    """optimizer から現在の学習率を取得する。"""
    return float(optimizer.param_groups[0]["lr"])


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """torch.Tensor -> numpy.ndarray（CPUへ移してfloat化）"""
    return x.detach().float().cpu().numpy()


def _parse_loss_output(
    loss_out: Union[torch.Tensor, Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """loss_fn の戻り値を共通フォーマットに変換する。

    Args:
        loss_out:
            - torch.Tensor: 通常の損失（scalar）
            - dict: MixedLogRawLoss のように成分を返すケース
                {"loss":..., "loss_log":..., "loss_raw":..., "alpha_raw":...}

    Returns:
        loss:     backward に使う scalar tensor
        loss_log: optional
        loss_raw: optional
        alpha:    optional
    """
    if isinstance(loss_out, torch.Tensor):
        return loss_out, None, None, None

    if isinstance(loss_out, dict):
        loss = loss_out["loss"]
        loss_log = loss_out.get("loss_log", None)
        loss_raw = loss_out.get("loss_raw", None)
        alpha = loss_out.get("alpha_raw", None)
        return loss, loss_log, loss_raw, alpha

    raise TypeError(f"Unsupported loss output type: {type(loss_out)}")


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
    """1epoch分の学習を実行する。

    Args:
        cfg: config（wandb_log_interval_steps 等を参照）
        model: 学習対象モデル（forward(x) -> (B, K)）
        loader: train dataloader
        optimizer: optimizer
        scheduler: scheduler（iteration step で進める想定。None可）
        loss_fn: 損失関数（pred, target -> loss）
        device: torch.device
        scaler: AMP用 GradScaler
        epoch: 現在epoch（1-indexed）
        use_amp: AMPを使うか
        max_norm: gradient clipping の最大ノルム（0以下なら無効）
        grad_accum_steps: 勾配蓄積ステップ数
        log_interval: tqdm 表示更新間隔
        is_main_process: rank0かどうか
        wandb_run: wandb run（Noneならログしない）
        global_step: optimizer step の累積カウント（wandb step 用）

    Returns:
        avg_loss: epoch平均の train loss（total）
        global_step: 更新後 global_step
    """
    model.train()

    # MixedLogRawLoss など epoch 依存の loss を想定
    if hasattr(loss_fn, "set_epoch"):
        loss_fn.set_epoch(epoch)

    running_loss = 0.0
    running_log = 0.0
    running_raw = 0.0
    n_steps = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, disable=not is_main_process)
    for idx, batch in enumerate(pbar):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)

            loss_out = loss_fn(pred, y)
            loss, loss_log, loss_raw, alpha = _parse_loss_output(loss_out)

            # backward は total loss のみ
            loss_to_backward = loss / float(grad_accum_steps)

        scaler.scale(loss_to_backward).backward()

        # optimizer step
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

        # logging（平均は “step平均” でOK。細かくやるならサンプル数重み付けでも可）
        running_loss += float(loss.item())
        if loss_log is not None:
            running_log += float(loss_log.item())
        if loss_raw is not None:
            running_raw += float(loss_raw.item())
        n_steps += 1

        # console
        if is_main_process and (idx % log_interval == 0):
            lr = get_lr(optimizer)
            pbar.set_description(f"epoch{epoch} loss{running_loss/max(1,n_steps):.4f} lr{lr:.2e}")

        # wandb（optimizer step 単位）
        if is_main_process and (wandb_run is not None) and do_step:
            log_every = int(getattr(cfg, "wandb_log_interval_steps", 100))
            if log_every > 0 and (global_step % log_every == 0):
                log_dict = {
                    "train/loss": running_loss / max(1, n_steps),
                    "train/lr": get_lr(optimizer),
                    "epoch": epoch,
                }
                if loss_log is not None:
                    log_dict["train/loss_log"] = running_log / max(1, n_steps)
                if loss_raw is not None:
                    log_dict["train/loss_raw"] = running_raw / max(1, n_steps)
                if alpha is not None:
                    log_dict["train/alpha_raw"] = float(alpha.item())

                wandb_run.log(log_dict, step=global_step)

    avg_loss = running_loss / max(1, n_steps)
    return avg_loss, global_step


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
    """1epoch分の検証を実行する（画像のみ入力）。

    方針:
        - loss は学習空間（例: log1p）で計算
        - metric は元スケールへ戻して計算（公式: global weighted R²）

    Returns:
        val_loss: サンプル平均の valid loss（total）
        global_r2: global weighted R²（元スケール）
        r2_scores: ターゲット別R²（補助）
        oof: dict（return_oof=Trueのみ）
    """
    model.eval()

    # epoch 依存 loss の場合に合わせる
    if hasattr(loss_fn, "set_epoch"):
        loss_fn.set_epoch(epoch)

    # loss は「サンプル数」で重み付けして平均化（batch size が一定でない可能性に備える）
    loss_sum = 0.0
    loss_log_sum = 0.0
    loss_raw_sum = 0.0
    n_samples = 0

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
            loss, loss_log, loss_raw, alpha = _parse_loss_output(loss_out)

        loss_sum += float(loss.item()) * bs
        if loss_log is not None:
            loss_log_sum += float(loss_log.item()) * bs
        if loss_raw is not None:
            loss_raw_sum += float(loss_raw.item()) * bs
        n_samples += bs

        ids = batch["id"]
        if isinstance(ids, (list, tuple, np.ndarray)):
            ids_all.extend([str(v) for v in ids])
        else:
            ids_all.append(str(ids))

        preds_all.append(_to_numpy(pred))
        targs_all.append(_to_numpy(y))

    preds_log = np.concatenate(preds_all, axis=0)
    targs_log = np.concatenate(targs_all, axis=0)

    # metric 用に raw へ戻す
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
        # loss 成分（ある場合のみ）
        if n_samples > 0:
            if loss_log_sum > 0:
                log_dict["valid/loss_log"] = float(loss_log_sum / n_samples)
            if loss_raw_sum > 0:
                log_dict["valid/loss_raw"] = float(loss_raw_sum / n_samples)

        # alpha（Mixed のときだけ）
        if hasattr(loss_fn, "_alpha_current"):
            log_dict["valid/alpha_raw"] = float(getattr(loss_fn, "_alpha_current"))

        for name, r2 in zip(target_names, r2_scores):
            log_dict[f"valid/r2_{name}"] = float(r2)

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
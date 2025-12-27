# -*- coding: utf-8 -*-
"""
学習ループ（train / valid）をまとめたモジュール。

本コンペの現状方針:
- 入力: 画像のみ
- 出力: 5ターゲット回帰
- 評価: weighted_r2（utils.metric.weighted_r2_score）

このモジュールでは以下を実装します。
- train_one_epoch: 1epoch分の学習
- valid_one_epoch: 1epoch分の検証 + OOF生成

Notes:
- Dataset 側で target を log1p している場合は、
  valid で expm1 して元スケールに戻してから weighted_r2 を計算します。
- wandb_run が渡された場合のみ wandb にログを送ります（rank0想定）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.metric import weighted_r2_score


def get_lr(optimizer) -> float:
    """optimizer から現在の学習率を取得する。"""
    return float(optimizer.param_groups[0]["lr"])


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """torch.Tensor -> numpy.ndarray（CPUへ移してfloat化）"""
    return x.detach().float().cpu().numpy()


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
        cfg: Hydra config / OmegaConf（wandb_log_interval_steps 等を参照）
        model: 学習対象モデル（forward(x) -> (B, K)）
        loader: train dataloader
        optimizer: optimizer
        scheduler: scheduler（iteration stepで進める想定。None可）
        loss_fn: 損失関数（pred, target -> loss）
        device: torch.device
        scaler: AMP用 GradScaler
        epoch: 現在epoch
        use_amp: AMPを使うか
        max_norm: gradient clipping の最大ノルム（0以下なら無効）
        grad_accum_steps: 勾配蓄積ステップ数
        log_interval: tqdm の表示更新間隔
        is_main_process: rank0かどうか
        wandb_run: wandb の run（Noneならログしない）
        global_step: optimizer step の累積カウント（wandb step 用）

    Returns:
        avg_loss: epoch平均の学習損失
        global_step: 更新された global_step
    """
    model.train()

    running_loss = 0.0
    n_steps = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, disable=not is_main_process)
    for it, batch in enumerate(pbar):
        # --------------------
        # batch 取り出し
        # --------------------
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        # --------------------
        # forward + loss
        # --------------------
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)          # ★ 画像のみ
            loss = loss_fn(pred, y)
            loss = loss / grad_accum_steps

        # backward
        scaler.scale(loss).backward()

        # --------------------
        # optimizer step
        # --------------------
        do_step = ((it + 1) % grad_accum_steps == 0)
        if do_step:
            # grad clip
            if max_norm is not None and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            global_step += 1

        # --------------------
        # logging（console）
        # --------------------
        running_loss += float(loss.item()) * grad_accum_steps
        n_steps += 1

        if is_main_process and (it % log_interval == 0):
            lr = get_lr(optimizer)
            pbar.set_description(f"epoch{epoch} loss{running_loss/max(1,n_steps):.4f} lr{lr:.2e}")

        # --------------------
        # logging（wandb）
        # --------------------
        # 「optimizer step単位」でログしたいので do_step のときのみ
        if is_main_process and (wandb_run is not None) and do_step:
            log_every = int(getattr(cfg, "wandb_log_interval_steps", 100))
            if log_every > 0 and (global_step % log_every == 0):
                wandb_run.log(
                    {
                        "train/loss": running_loss / max(1, n_steps),
                        "train/lr": get_lr(optimizer),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

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
    """1epoch分の検証を実行する。

    Args:
        cfg: config
        model: 評価対象モデル（forward(x) -> (B, K)）
        loader: valid dataloader
        loss_fn: 損失関数（log空間で学習しているなら、その空間でのloss）
        device: torch.device
        epoch: 現在epoch
        use_amp: AMPを使うか（validでも使うと高速）
        use_log1p_target: Datasetでtargetをlog1pしているか
        is_main_process: rank0かどうか
        wandb_run: wandb run
        global_step: wandb step
        target_names: wandbに出す per-target 名（例: target_cols）
        return_oof: OOF（pred/target/ids）を返すか

    Returns:
        val_loss: valid loss の平均（log空間での loss）
        weighted_r2: 元スケールで計算した weighted_r2
        r2_scores: ターゲットごとの r2
        oof: OOF dict（return_oof=FalseならNone）
    """
    model.eval()

    running_loss = 0.0
    n_steps = 0

    ids_all: List[str] = []
    preds_all: List[np.ndarray] = []
    targs_all: List[np.ndarray] = []

    pbar = tqdm(loader, disable=not is_main_process)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)  # ★ 画像のみ
            loss = loss_fn(pred, y)

        running_loss += float(loss.item())
        n_steps += 1

        ids_all.extend(list(batch["id"]))
        preds_all.append(_to_numpy(pred))
        targs_all.append(_to_numpy(y))

    preds_log = np.concatenate(preds_all, axis=0)
    targs_log = np.concatenate(targs_all, axis=0)

    # --------------------
    # metric は「元スケール」で計算
    # --------------------
    if use_log1p_target:
        preds = np.expm1(preds_log)
        targs = np.expm1(targs_log)
    else:
        preds = preds_log
        targs = targs_log

    weighted_r2, r2_scores = weighted_r2_score(targs, preds)
    val_loss = running_loss / max(1, n_steps)

    # --------------------
    # wandb logging
    # --------------------
    if is_main_process and (wandb_run is not None):
        wandb_run.log(
            {
                "valid/loss": float(val_loss),
                "valid/weighted_r2": float(weighted_r2),
                "epoch": epoch,
            },
            step=global_step,
        )

        if target_names is None:
            target_names = [f"t{i}" for i in range(len(r2_scores))]

        for name, r2 in zip(target_names, r2_scores):
            wandb_run.log(
                {f"valid/r2_{name}": float(r2), "epoch": epoch},
                step=global_step,
            )

    # --------------------
    # OOF（解析用に raw / log の両方を保持）
    # --------------------
    oof = None
    if return_oof:
        oof = {
            "ids": np.array(ids_all),
            # raw-scale（推奨：後で分析しやすい）
            "preds": preds,
            "targets": targs,
            # log-scale（loss解析やデバッグ用）
            "preds_log": preds_log,
            "targets_log": targs_log,
        }

    return float(val_loss), float(weighted_r2), r2_scores, oof
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

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
        grad_accum_steps: 勾配蓄積ステップ数. (default: 1)
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
    for idx, batch in enumerate(pbar):
        # --------------------
        # batch 取り出し
        # --------------------
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        # --------------------
        # forward + loss
        # --------------------
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)          # ★ 画像のみを入力
            loss = loss_fn(pred, y)
            loss = loss / grad_accum_steps
        # backward
        scaler.scale(loss).backward()

        # --------------------
        # optimizer step
        # --------------------
        do_step = ((idx + 1) % grad_accum_steps == 0)
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

        if is_main_process and (idx % log_interval == 0):
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
    """1epoch分の検証を実行する（画像のみ入力）。

    評価方針:
    - loss は「学習で使っている空間」で計算（例: log1p空間）
    - metric は「元スケール」に戻してから計算（expm1 など）
    - 公式スコアに合わせ、global weighted R² を採用する

    Args:
        cfg: config（cfg.metric.weights を参照）
        model: 評価対象モデル（forward(x) -> (B, K)）
        loader: valid dataloader
        loss_fn: 損失関数（log空間で学習しているなら log空間のloss）
        device: torch.device
        epoch: 現在epoch
        use_amp: validでもAMPを使うか（速度優先）
        use_log1p_target: Datasetでtargetをlog1pしているか
        is_main_process: rank0かどうか（WandBログなど）
        wandb_run: wandb run（Noneならログしない）
        global_step: wandb step
        target_names: ターゲット名リスト（wandbログ用）
        return_oof: OOFを返すか

    Returns:
        val_loss: サンプル平均の valid loss（log空間のloss）
        global_r2: 公式に合わせた global weighted R²（元スケール）
        r2_scores: ターゲット別R²（補助指標）
        oof: {"ids","preds","targets","preds_log","targets_log"}（return_oof=Trueのみ）
    """
    model.eval()

    # --------
    # loss集計（サンプル平均にする）
    # --------
    loss_sum = 0.0
    n_samples = 0

    # --------
    # OOF用バッファ
    # --------
    ids_all: List[str] = []
    preds_all: List[np.ndarray] = []
    targs_all: List[np.ndarray] = []

    pbar = tqdm(loader, disable=not is_main_process)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        bs = int(x.size(0))

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)   # ★ 画像のみ入力
            loss = loss_fn(pred, y)

        # loss は「サンプル数」で重み付けして加算 → 最後に全サンプルで割る
        loss_sum += float(loss.item()) * bs
        n_samples += bs

        # id の取り出し（バッチが list[str] で来る想定。保険付き）
        ids = batch["id"]
        if isinstance(ids, (list, tuple, np.ndarray)):
            ids_all.extend([str(v) for v in ids])
        else:
            ids_all.append(str(ids))

        preds_all.append(_to_numpy(pred))
        targs_all.append(_to_numpy(y))

    # ----
    # ログ空間（学習空間）の配列
    # ----
    preds_log = np.concatenate(preds_all, axis=0)
    targs_log = np.concatenate(targs_all, axis=0)

    # ----
    # 元スケールへ戻す（metric計算用）
    # ----
    if use_log1p_target:
        # expm1のオーバーフロー保険（極端な出力でinf/NaNを防ぐ）
        preds_log_safe = np.clip(preds_log, -20.0, 20.0)
        targs_log_safe = np.clip(targs_log, -20.0, 20.0)

        preds = np.expm1(preds_log_safe)
        targs = np.expm1(targs_log_safe)
    else:
        preds = preds_log
        targs = targs_log

    # NaN/inf 保険（念のため）
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    targs = np.nan_to_num(targs, nan=0.0, posinf=0.0, neginf=0.0)

    # ★ 0未満をクリップ（質量は非負）
    preds = np.clip(preds, 0.0, None)

    # ----
    # 公式メトリック：global weighted R²
    # ----
    weights = np.asarray(cfg.metric.weights, dtype=np.float64)
    global_r2 = global_weighted_r2_score(targs, preds, weights)

    # 補助：ターゲット別R²（分析用）
    r2_scores = r2_per_target(targs, preds)

    # loss（サンプル平均）
    val_loss = loss_sum / max(1, n_samples)

    # ----
    # wandb logging（まとめて1回）
    # ----
    if is_main_process and (wandb_run is not None):
        if target_names is None:
            target_names = [f"t{i}" for i in range(len(r2_scores))]

        log_dict = {
            "valid/loss": float(val_loss),
            "valid/weighted_r2": float(global_r2),  # ※ここは global_r2 の意味
            "epoch": epoch,
        }
        for name, r2 in zip(target_names, r2_scores):
            log_dict[f"valid/r2_{name}"] = float(r2)

        wandb_run.log(log_dict, step=global_step)

    # ----
    # OOF（解析用）
    # ----
    oof = None
    if return_oof:
        oof = {
            "ids": np.array(ids_all),
            "preds": preds,          # raw-scale
            "targets": targs,        # raw-scale
            "preds_log": preds_log,  # log-scale
            "targets_log": targs_log,
        }

    return float(val_loss), float(global_r2), r2_scores, oof
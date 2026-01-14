import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from timm.utils import ModelEmaV2


def weighted_r2_score(config, y_true: np.ndarray, y_pred: np.ndarray):
    """
    Metric
    y_true, y_pred: shape (N, 5)
    """
    weights = config.r2_weights
    r2_scores = []
    
    for i in range(y_true.shape[1]):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
        
    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted_r2, r2_scores

def weighted_r2_score_global(config, y_true: np.ndarray, y_pred: np.ndarray):
    weights = config.r2_weights

    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    w = np.tile(weights, y_true.shape[0])

    mean_w = np.sum(w * flat_true) / np.sum(w)
    ss_res = np.sum(w * (flat_true - flat_pred) ** 2)
    ss_tot = np.sum(w * (flat_true - mean_w) ** 2)
    global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    avg_r2, per_r2 = weighted_r2_score(config, y_true, y_pred)
    return global_r2, avg_r2, per_r2

def biomass_loss(outputs, labels, w=None):
    total, gdm, green, clover, dead = outputs
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=5.0) # Huber loss for robust regression (beta=5.0 as recommended)
    
    l_green  = huber(green.squeeze(),  labels[:,0])
    l_dead   = huber(dead.squeeze(), labels[:,1]) # Use Huber loss for Dead
    l_clover = huber(clover.squeeze(), labels[:,2])
    l_gdm    = huber(gdm.squeeze(),    labels[:,3])
    l_total  = huber(total.squeeze(),  labels[:,4])

    # Stack per-target losses in the SAME order as CFG.ALL_TARGET_COLS
    losses = torch.stack([l_green, l_dead, l_clover, l_gdm, l_total])
    # losses = torch.stack([l_green, l_dead, l_clover])

    # Use provided weights, or default to CFG.R2_WEIGHTS
    if w is None:
        return losses.mean()
    w = torch.as_tensor(w, device=losses.device, dtype=losses.dtype)
    w = w / w.sum()
    return (losses * w).sum()

from contextlib import nullcontext
USE_BF16 = True
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

scaler = torch.amp.GradScaler(
    'cuda',
    enabled=(torch.cuda.is_available() and AMP_DTYPE == torch.float16)
)

def autocast_ctx():
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE)

def _as_col(x: torch.Tensor, bs: int) -> torch.Tensor:
    """Force predictions to (B,1) regardless of model head returning (B,) or (B,1)."""
    if x.ndim == 1:
        return x.view(bs, 1)
    if x.ndim == 2:
        return x
    return x.view(bs, 1)

def _ensure_2d_lab(lab: torch.Tensor, bs: int) -> torch.Tensor:
    """Force labels to (B,T)."""
    if lab.ndim == 1:
        return lab.view(bs, -1)
    return lab

def _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs: int) -> torch.Tensor:
    """
    Pack predictions in the exact order expected by your metric:
    CFG.ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    """
    pg = _as_col(p_green,  bs)
    pd = _as_col(p_dead,   bs)
    pc = _as_col(p_clover, bs)
    pgdm = _as_col(p_gdm,  bs)
    pt = _as_col(p_total,  bs)
    return torch.cat([pg, pd, pc, pgdm, pt], dim=1)  # (B,5)




# =========================================================
# Train
# =========================================================
# Training and Validation Loops 
def train_epoch(config, model, loader, optimizer, scheduler, device, ema: ModelEmaV2 | None = None):
    model.train()
    running = 0.0

    optimizer.zero_grad(set_to_none=True)
    itera = tqdm(loader, desc='train', leave=False) if config.use_tqdm else loader

    for i, (l, r, lab) in enumerate(itera):
        bs = l.size(0)
        l   = l.to(device, non_blocking=True)
        r   = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        with autocast_ctx():
            p_total, p_gdm, p_green, p_clover, p_dead = model(l, r)
            loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead),
                                lab, w=config.loss_weights)
            loss = loss / config.grad_acc

        if not torch.isfinite(loss):
            print("Non-finite loss detected; skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += loss.detach().float().item() * bs * config.grad_acc

        do_step = ((i + 1) % config.grad_acc == 0) or ((i + 1) == len(loader))
        if do_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if ema is not None:
                ema.update(model.module if hasattr(model, "module") else model)

            optimizer.zero_grad(set_to_none=True)

    scheduler.step()
    return running / len(loader.dataset)

# =========================================================
# Valid
# =========================================================
@torch.inference_mode()
def valid_epoch(config, eval_model, loader, device):
    eval_model.eval()

    n = len(loader.dataset)
    n_targets = len(config.all_target_cols)
    preds_cpu  = torch.empty((n, n_targets), dtype=torch.float32)
    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)

    total_loss = 0.0
    offset = 0

    for l, r, lab in loader:
        bs = l.size(0)
        l   = l.to(device, non_blocking=True)
        r   = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        with autocast_ctx():
            p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
            loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead),
                                lab, w=config.loss_weights)

        total_loss += loss.detach().float().item() * bs

        batch_pred = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs).float().cpu()
        batch_lab  = _ensure_2d_lab(lab, bs).float().cpu()

        # Safety: ensure dims match (avoids silent 4/5 bugs)
        if batch_pred.shape[1] != n_targets or batch_lab.shape[1] != n_targets:
            raise RuntimeError(
                f"Target dim mismatch: pred={batch_pred.shape}, lab={batch_lab.shape}, "
                f"CFG.ALL_TARGET_COLS={n_targets}"
            )

        preds_cpu[offset:offset+bs]  = batch_pred
        labels_cpu[offset:offset+bs] = batch_lab
        offset += bs

    pred_all    = preds_cpu.numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(config, true_labels, pred_all)

    return total_loss / n, global_r2, avg_r2, per_r2, pred_all, true_labels


@torch.inference_mode()
def valid_epoch_tta(config, eval_model, loaders, device):
    """
    Fixed: always accumulate (N,5). Works even if a head outputs (B,1).
    Assumes each loader iterates in identical order (shuffle=False).
    """
    eval_model.eval()
    assert len(loaders) > 0

    n = len(loaders[0].dataset)
    n_targets = len(config.all_target_cols)

    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)
    preds_sum  = torch.zeros((n, n_targets), dtype=torch.float32)

    total_loss = 0.0

    for tta_i, loader in enumerate(loaders):
        offset = 0
        tta_loss = 0.0

        for l, r, lab in loader:
            bs = l.size(0)
            l   = l.to(device, non_blocking=True)
            r   = r.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            with autocast_ctx():
                p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
                loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead),
                                    lab, w=config.loss_weights)

            tta_loss += loss.detach().float().item() * bs

            batch_pred = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs).float().cpu()
            preds_sum[offset:offset+bs] += batch_pred

            if tta_i == 0:
                batch_lab = _ensure_2d_lab(lab, bs).float().cpu()
                if batch_lab.shape[1] != n_targets:
                    raise RuntimeError(f"Label dim mismatch: {batch_lab.shape} vs {n_targets}")
                labels_cpu[offset:offset+bs] = batch_lab

            offset += bs

        total_loss += tta_loss / n

    avg_preds   = (preds_sum / len(loaders)).numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(config, true_labels, avg_preds)

    return total_loss / len(loaders), global_r2, avg_r2, per_r2, avg_preds, true_labels



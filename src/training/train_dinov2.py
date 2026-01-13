import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn


def weighted_r2_score(config, y_true: np.ndarray, y_pred: np.ndarray):
    """
    Metric
    y_true, y_pred: shape (N, 5)
    """
    weights = config.weights
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

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def step(self, score):
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self):
        return self.counter >= self.patience

def expand_predictions_torch(preds_3, targets):
    """
    [torch] Expand the 3 NN predictions (N, 3) to 5 predictions (N, 5).
    """
    if targets == ["Dry_Green_g", "Dry_Total_g", "GDM_g"]:
        # clip
        P_Green = torch.clamp(preds_3[:, 0], min=0)
        P_Total = torch.clamp(preds_3[:, 1], min=0)
        P_GDM = torch.clamp(preds_3[:, 2], min=0)
        
        # Compute derived targets based on constraints.
        P_Clover = torch.clamp(P_GDM - P_Green, min=0)
        P_Dead = torch.clamp(P_Total - P_GDM, min=0)
    
    elif targets == ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]:
        P_Clover = torch.clamp(preds_3[:, 0], min=0)
        P_Dead  = torch.clamp(preds_3[:, 1], min=0)
        P_Green = torch.clamp(preds_3[:, 2], min=0)

        # Compute derived targets based on constraints.
        P_GDM = torch.clamp(P_Green + P_Clover, min=0)
        P_Total = torch.clamp(P_GDM + P_Dead, min=0)
    elif targets == ["Dry_Clover_g", "Dry_Dead_g", "GDM_g"]:
        P_Clover = torch.clamp(preds_3[:, 0], min=0)
        P_Dead =torch.clamp(preds_3[:, 1], min=0)
        P_GDM = torch.clamp(preds_3[:, 2], min=0)

        # Compute derived targets based on constraints.
        P_Green = torch.clamp(P_GDM - P_Clover, min=0)
        P_Total = torch.clamp(P_GDM + P_Dead, min=0)

    
    preds_5 = torch.stack(
        [
            P_Clover, # Index 0
            P_Dead,   # Index 1
            P_Green,  # Index 2
            P_Total,  # Index 3
            P_GDM     # Index 4
        ],
        dim=1
    )
    return preds_5


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    cfg,
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Train", total=len(loader), ncols=200)
    for idx, (img_left, img_right, labels) in enumerate(pbar):
        img_left, img_right, labels = img_left.to(device), img_right.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(img_left, img_right)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img_left.size(0)

    return running_loss / len(loader.dataset)

# =========================================================
# Valid
# =========================================================
def val_fn(
    config,
    model, 
    loader, 
    criterion, 
    device, 
    targets
    ):
    """
    Validation / evaluation for 1 epoch.
    Assumes the dataset returns only 3 independent targets per sample.
    """
    model.eval()
    total_loss = 0
    all_targets_5_np = []
    all_preds_5_np = []

    pbar = tqdm(loader, desc="Validate", total=len(loader), ncols=200)
    with torch.no_grad():
        for i, (img_left, img_right, targets_3) in enumerate(pbar):
            img_left, img_right, targets_3 = img_left.to(device), img_right.to(device), targets_3.to(device)  # shape (B, 3)

            # Model forward
            outputs_3 = model(img_left, img_right)  # (B, 3)

            # Compute loss directly on independent targets
            loss = criterion(outputs_3, targets_3)
            total_loss += loss.item() * img_left.size(0)  # scale by batch size

            # Expand predictions to 5 for metrics
            preds_5 = expand_predictions_torch(outputs_3, targets)
            all_preds_5_np.append(preds_5.cpu().numpy())

            # Expand targets to 5 as well
            targets_5 = expand_predictions_torch(targets_3, targets)
            all_targets_5_np.append(targets_5.cpu().numpy())

    # Average loss over dataset
    val_loss = total_loss / len(loader.dataset)

    # Concatenate all predictions / targets
    y_true_5 = np.concatenate(all_targets_5_np, axis=0)
    y_pred_5 = np.concatenate(all_preds_5_np, axis=0)

    # Compute weighted R² metric on full 5 targets
    weighted_r2, _ = weighted_r2_score(config, y_true_5, y_pred_5)

    return val_loss, weighted_r2
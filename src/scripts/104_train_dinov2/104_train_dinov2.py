# =================
# Import libraries
# =================
import os, gc, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

import warnings
warnings.filterwarnings('ignore')

# original
import sys
sys.path.append(r"..")
from utils.data import sep, show_df, save_config_yaml, dict_to_namespace
# from utils.wandb_utils import set_wandb
from datasets.biomasstwostream_dataset import BiomassTwoStreamDataset
from datasets.transforms_dino import get_transforms
from models.dino_regressor import DINOv2TwoStreamRegressor
from training.train_dinov2 import train_one_epoch, val_fn, EarlyStopping

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")


# ===================================
# utils
# ===================================
def expand_predictions_np(preds_3, targets):
    """
    [Numpy] Expand the three NN predictions (N, 3) to five predictions (N, 5).
    """

    if targets == ["Dry_Green_g", "Dry_Total_g", "GDM_g"]:
        # clip
        P_Green = np.clip(preds_3[:, 0], a_min=0, a_max=None)
        P_Total = np.clip(preds_3[:, 1], a_min=0, a_max=None)
        P_GDM   = np.clip(preds_3[:, 2], a_min=0, a_max=None)        
        # Compute derived targets based on constraints.
        P_Clover = np.clip(P_GDM - P_Green, a_min=0, a_max=None)
        P_Dead   = np.clip(P_Total - P_GDM, a_min=0, a_max=None)
    
    elif targets == ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]:
        P_Clover = np.clip(preds_3[:, 0], a_min=0, a_max=None)
        P_Dead   = np.clip(preds_3[:, 1], a_min=0, a_max=None)
        P_Green  = np.clip(preds_3[:, 2], a_min=0, a_max=None)
        # Compute derived targets based on constraints.
        P_GDM   = np.clip(P_Green + P_Clover, a_min=0, a_max=None)
        P_Total = np.clip(P_GDM + P_Dead, a_min=0, a_max=None)

    elif targets == ["Dry_Clover_g", "Dry_Dead_g", "GDM_g"]:
        P_Clover = np.clip(preds_3[:, 0], a_min=0, a_max=None)
        P_Dead   = np.clip(preds_3[:, 1], a_min=0, a_max=None)
        P_GDM    = np.clip(preds_3[:, 2], a_min=0, a_max=None)
        # Compute derived targets based on constraints.
        P_Green = np.clip(P_GDM - P_Clover, a_min=0, a_max=None)
        P_Total = np.clip(P_GDM + P_Dead, a_min=0, a_max=None)
    
    preds_5 = np.stack(
        [P_Clover, P_Dead, P_Green, P_Total, P_GDM],
        axis=1
    )
    return preds_5


# ===================================
# main
# ===================================
# TODO: config_pathをこのスクリプトからの相対パスにする
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """description
    Args:
        cfg (DictConf): config
    """
    # set config
    config_dict = OmegaConf.to_container(cfg["104_train_dinov2"], resolve=True)
    config = dict_to_namespace(config_dict)
    # when debug
    if config.debug:
        config.exp = "104_debug" # TODO: ファイルの連番を入れる
    # # set WandB
    # if config.use_wandb:
    #     set_wandb(config)
    # make savedir
    savedir = Path(config.output_dir) / config.exp
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir / "oof", exist_ok=True)
    os.makedirs(savedir / "yaml", exist_ok=True)
    os.makedirs(savedir / "model", exist_ok=True)
    # YAMLとして保存
    output_path = Path(savedir/"yaml"/"config.yaml")
    save_config_yaml(config, output_path)
    print(f"Config saved to {output_path.resolve()}")

    # ==================
    # preprocess df
    # ==================
    # load train_df
    train_df = pd.read_csv(Path(config.input_dir) / "train.csv")
    # unique id
    train_df["image_id"] = train_df["image_path"].apply(lambda x: x.split('/')[-1].split('.')[0])
    # pivot
    train_pivot = train_df.pivot(
        index   = "image_id", 
        columns = "target_name", 
        values  = "target"
    ).reset_index()
    # meta data
    meta_df = train_df.drop_duplicates(subset="image_id").drop(
        columns=["sample_id", "target_name", "target"]
    )
    # merge
    train_processed_df = meta_df.merge(train_pivot, on="image_id", how="left")
    # fold列の追加（StratifiedKFold）
    skf = StratifiedKFold(n_splits=len(config.folds), shuffle=True, random_state=config.seed)
    train_processed_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed_df, train_processed_df["State"])):
        train_processed_df.loc[val_idx, "fold"] = fold
    sep("train_processed_df"); show_df(train_processed_df); print()
    train_df = train_processed_df.copy()

    # =================
    # training
    # =================
    sep("Training start")
    oof_predictions = np.zeros((len(train_df), len(config.target_cols)))

    for fold in config.folds:
        print(f"\n======== FOLD {fold+1}/{len(config.folds)} ========")
        targets = config.targets_configs # ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]

        # -----------------------------
        # Split fold 
        # -----------------------------
        train_fold_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        val_fold_df_with_idx = train_df[train_df["fold"] == fold]
        val_indices = val_fold_df_with_idx.index
        val_fold_df = val_fold_df_with_idx.reset_index(drop=True)

        # -----------------------------
        # Datasets & loaders
        # -----------------------------
        # datasets
        train_dataset = BiomassTwoStreamDataset(
            config=config,
            df=train_fold_df,
            transforms=get_transforms(config=config, is_train=True),
            is_test=False,
            input_res=224,
            targets=config.targets_configs
        )
        val_dataset = BiomassTwoStreamDataset(
            config=config,
            df=val_fold_df,
            transforms=get_transforms(config=config, is_train=False),
            is_test=False,
            input_res=224,
            targets=config.targets_configs
        )
        # dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size * 2,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

        # -----------------------------
        # Model
        # -----------------------------
        # model
        model = DINOv2TwoStreamRegressor(
            n_targets  = len(targets), 
            model_path = config.model.backbone,
            freeze_backbone = True, 
            hidden_dim = config.model.hidden_dim,
            dropout = config.model.dropout
        ).to(config.device)
        # optimizer
        criterion = nn.SmoothL1Loss(beta=0.5)
        best_val = -np.inf
        model_path = savedir / "model" / f"model_fold_{fold}.pth"

        # ======================================================
        # STAGE 1 — Train head only
        # ======================================================
        print("Training head only...")
        # Early stopping, optimizer, scheduler
        early_stop = EarlyStopping(patience=7)
        optimizer = torch.optim.AdamW(
            model.regressor.parameters(),
            lr=config.train.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=config.train.eta_min,
        )

        # --- training loop ---
        for epoch in range(config.train.epochs):
            # train & val
            train_loss = train_one_epoch(
                cfg = config,
                model = model,
                loader = train_loader,
                optimizer = optimizer,
                scheduler = scheduler,
                criterion = criterion,
                device = config.device,
            )
            val_loss, weighted_r2 = val_fn(
                config=config,
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=config.device,
                targets=targets
            )
            print(
                f"[HEAD] Epoch {epoch+1}/{config.train.epochs} "
                f"- Train: {train_loss:.4f} | Val: {val_loss:.4f} |  Val R2: {weighted_r2:.4f}"
            )
            scheduler.step()
            # ---- Save best model ----
            if weighted_r2 > best_val:
                print(f"🔥 Best updated: {best_val:.5f} (epoch={epoch}) -> {model_path}")
                best_val = weighted_r2
                torch.save(model.state_dict(), model_path)
            # ---- Early stopping logic ----
            early_stop.step(val_loss)
            if early_stop.should_stop():
                print("Early stopping (head)")
                break

        # backbone fine-tuningをする場合
        if config.train_backbone:
            # ======================================================
            # STAGE 2 — LayerNorm-only fine-tuning
            # ======================================================
            print("Fine-tuning LayerNorm only...")      
            # Freeze entire backbone
            for param in model.backbone.parameters():
                param.requires_grad = False        
            # Unfreeze only LayerNorms
            for name, param in model.backbone.named_parameters():
                if "norm" in name.lower():
                    param.requires_grad = True
            # Early stopping, optimizer, scheduler
            early_stop = EarlyStopping(patience=4)
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": model.regressor.parameters(),
                        "lr": config.train.lr,
                        "weight_decay": 1e-3,
                    },
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            model.backbone.parameters(),
                        ),
                        "lr": config.train.lr * 0.01,
                        "weight_decay": 1e-5,
                    },
                ]
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.3,
                patience=2,
            )
            # --- training loop (半分のepochs) ---
            for epoch in range(config.train.epochs // 2):
                # train & val
                train_loss = train_one_epoch(
                    cfg = config,
                    model = model,
                    loader = train_loader,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    criterion = criterion,
                    device = config.device,
                )        
                val_loss, weighted_r2 = val_fn(
                    config=config,
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=config.device,
                    targets=targets
                )
                print(
                    f"[FT] Epoch {epoch+1}/{config.train.epochs//2} "
                    f"- Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val R2: {weighted_r2:.4f} "
                )
                scheduler.step(val_loss)
                # ---- Save best model ----
                if weighted_r2 > best_val:
                    print(f"🔥 Best updated -FT-: {best_val:.5f} (epoch={epoch}) -> {model_path}")
                    best_val = weighted_r2
                    torch.save(model.state_dict(), model_path)
                # ---- Early stopping logic ----
                early_stop.step(val_loss)
                if early_stop.should_stop():
                    print("Early stopping (FT)")
                    break

        # -----------------------------
        # OOF predictions
        # -----------------------------
        print()
        sep("OOF predictions")
        print(f"Loading best model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        fold_preds_3 = []

        with torch.no_grad():
            for img_left, img_right, _ in val_loader:
                img_left, img_right = img_left.to(config.device), img_right.to(config.device)
                preds = model(img_left, img_right)
                fold_preds_3.append(preds.cpu().numpy())

        fold_preds_3 = np.concatenate(fold_preds_3, axis=0)

        # Expand 3 → 5
        fold_preds_5 = expand_predictions_np(fold_preds_3, targets)
        oof_predictions[val_indices] = fold_preds_5

        # -----------------------------
        # Cleanup
        # -----------------------------
        del model, train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    oof_predictions = np.array(oof_predictions)
    oof_df = train_df[["image_id"]].copy()
    for i, col in enumerate(config.target_cols):
        oof_df[col] = oof_predictions[:, i]
    oof_path = savedir / "oof" / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path.resolve()}")

if __name__ == "__main__":
    main()

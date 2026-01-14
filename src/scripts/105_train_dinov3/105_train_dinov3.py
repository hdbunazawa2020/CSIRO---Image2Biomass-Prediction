# =================
# Import libraries
# =================
import os, gc, warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import math
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from timm.utils import ModelEmaV2

import hydra
from omegaconf import DictConfig, OmegaConf

import warnings
warnings.filterwarnings('ignore')

# original
import sys
sys.path.append(r"..")
from utils.data import sep, show_df, save_config_yaml, dict_to_namespace
# from utils.wandb_utils import set_wandb
from datasets.biomass_dataset_dinov3 import BiomassDataset
from datasets.transforms_dinov3 import get_train_transforms, get_tta_transforms
from models.dinov3_regressor import BiomassModel
from training.train_dinov3 import train_epoch, valid_epoch_tta, weighted_r2_score_global

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")


# ===================================
# utils
# ===================================
# Helper for accurate GPU timings
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Utility Functions
def set_backbone_requires_grad(model: BiomassModel, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


def build_optimizer(config, model: BiomassModel):
    # 1. Get backbone parameter IDs for exclusion
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    
    # 2. Separate params into backbone vs. everything else (heads, fusion, etc.)
    backbone_params = []
    rest_params = []
    
    for p in model.parameters():
        if p.requires_grad:
            if id(p) in backbone_ids:
                backbone_params.append(p)
            else:
                rest_params.append(p)
    
    return optim.AdamW([
        {'params': backbone_params, 'lr': config.train.lr_backbone, 'weight_decay': config.optimizer.weight_decay},
        {'params': rest_params,     'lr': config.train.lr_rest,     'weight_decay': config.optimizer.weight_decay},
])

def build_scheduler(config, optimizer):
    def lr_lambda(epoch):
        e = max(0, epoch - 1)
        if e < config.train.warmup_epochs:
            return float(e + 1) / float(max(1, config.train.warmup_epochs))
        progress = (e - config.train.warmup_epochs) / float(max(1, config.train.epochs - config.train.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


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
    config_dict = OmegaConf.to_container(cfg["105_train_dinov3"], resolve=True)
    config = dict_to_namespace(config_dict)
    # when debug
    if config.debug:
        config.exp = "105_debug" # TODO: ファイルの連番を入れる
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
    # load df_long
    print('Loading data...')
    df_long = pd.read_csv(Path(config.input_dir) / "train.csv")
    # pivot
    df_wide = (
        df_long.pivot(index='image_path', columns='target_name', values='target')
        .reset_index()
    )
    # merge
    print('Merging metadata for stratification...')
    meta_df = df_long[['image_path', 'Sampling_Date', 'State']].drop_duplicates()
    df_wide = df_wide.merge(meta_df, on='image_path', how='left')
    # Keep necessary columns
    df_wide = df_wide[['image_path', 'Sampling_Date', 'State'] + config.all_target_cols]
    print(f'{len(df_wide)} training images')
    sep("DataFrame preview:"); show_df(df_wide, 3, True); print()
    df_wide.to_csv(savedir / "df_folds.csv", index=False)

    # fold列の追加 -- StratifiedGroupKFold -- 
    sgkf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    oof_true, oof_pred, fold_summary, val_idxs = [], [], [], []

    # Split based on groups (Sampling_Date) and stratification target (State)
    groups = df_wide['Sampling_Date']
    y_stratify = df_wide['State']

    # One place for loader kwargs (fast)
    DL_KW = dict(
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(config.num_workers > 0),
        prefetch_factor=4 if config.num_workers > 0 else None,
    )

    # =================
    # training
    # =================
    sep("Training start")
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df_wide, y_stratify, groups=groups)):
        if fold not in config.folds:
            print(f'Skipping fold {fold} as per configuration.')
            continue
        sep(f'FOLD {fold+1}/{config.n_folds} | {len(tr_idx)} train / {len(val_idx)} val')
        
        # NOTE: avoid empty_cache/gc inside epoch loop; only between folds if you really want
        _sync(); torch.cuda.empty_cache(); gc.collect()
        tr_df  = df_wide.iloc[tr_idx].reset_index(drop=True)
        val_df = df_wide.iloc[val_idx].reset_index(drop=True)

        # ======================
        # dataset & loader
        # ======================
        # train
        tr_set = BiomassDataset(
            config = config,
            df = tr_df, 
            transform = get_train_transforms(config=config), 
            img_dir = Path(config.input_dir) / "train"
            )
        tr_loader = DataLoader(
            tr_set,
            batch_size=config.train.batch_size,
            shuffle=True,
            drop_last=True,
            **{k: v for k, v in DL_KW.items() if v is not None},
        )
        # valid -- create TTA loaders (keep TTAs as requested) -- 
        val_loaders = []
        for mode in range(config.val_tta_times):  # 0: orig, 1: hflip, 2: vflip, 3: rot90
            val_set_tta = BiomassDataset(
                config = config,
                df = val_df, 
                transform = get_tta_transforms(config=config, mode=mode), 
                img_dir = Path(config.input_dir) / "train"
            )
            val_loader_tta = DataLoader(
                val_set_tta,
                batch_size=config.train.batch_size,
                shuffle=False,
                drop_last=False,
                **{k: v for k, v in DL_KW.items() if v is not None},
            )
            val_loaders.append(val_loader_tta)


        # ========================
        # Load model
        # ========================
        print('Building model...')
        model = BiomassModel(
            config = config,
            model_name = config.model.backbone, 
            pretrained = config.model.pretrained, 
            backbone_path = None
            ).to(config.device)
        # Load pretrained fold weights if available (for resuming or fine-tuning)
        if getattr(config, 'pretrained_dir', None) and os.path.isdir(config.pretrained_dir):
            pretrained_path = os.path.join(config.pretrained_dir, f'best_model_fold{fold}.pth')
            if os.path.exists(pretrained_path):
                try:
                    state = torch.load(pretrained_path, map_location='cpu')
                    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
                        key = 'model_state_dict' if 'model_state_dict' in state else 'state_dict'
                        sd = state[key]
                    else:
                        sd = state
                    model.load_state_dict(sd, strict=False)
                    model.to(config.device)
                    print(f'  ✓ Loaded pretrained weights for fold {fold} from {pretrained_path}')
                except Exception as e:
                    print(f'  ✗ Failed to load pretrained fold {fold}: {e}')
            else:
                print(f'  (No pretrained file for fold {fold} at {pretrained_path})')
        else:
            print('  (No PRETRAINED_DIR configured or directory missing)')

        # Single GPU: DO NOT wrap in DataParallel
        # model = nn.DataParallel(model)  # <-- removed

        # Freeze/unfreeze backbone
        set_backbone_requires_grad(model=model, requires_grad=False)
        # optimizer & scheduler
        optimizer = build_optimizer(config=config, model=model)
        scheduler = build_scheduler(config=config, optimizer=optimizer)
        # EMA on the real model
        ema = ModelEmaV2(model, decay=config.ema_decay)


        best_global_r2 = -np.inf; best_avg_r2 = -np.inf
        patience = 0
        best_fold_preds = None
        best_fold_true = None

        # set save_path
        save_path = savedir / "model" / f'best_model_fold{fold}.pth'

        for epoch in range(1, config.train.epochs + 1):
            if epoch == config.train.freeze_epochs + 1:
                patience = 0
                set_backbone_requires_grad(model=model, requires_grad=True)
                print(f'Epoch {epoch}: backbone unfrozen')

            # ---- Train timing ----
            _sync()
            t0 = time.perf_counter()
            tr_loss = train_epoch(
                config = config, 
                model = model, 
                loader = tr_loader, 
                optimizer = optimizer, 
                scheduler = scheduler, 
                device = config.device, 
                ema = ema,
            )
            _sync()
            t1 = time.perf_counter()

            # choose eval model (EMA weights)
            eval_model = ema.module if ema is not None else model

            # ---- Val timing (TTA) ----
            _sync()
            t2 = time.perf_counter()
            val_loss, global_r2, avg_r2, per_r2, preds_fold, true_fold = valid_epoch_tta(
                config = config, 
                eval_model = eval_model, 
                loaders = val_loaders, 
                device = config.device
                )
            _sync()
            t3 = time.perf_counter()

            time_tr  = t1 - t0
            time_val = t3 - t2
            time_ep  = t3 - t0

            per_r2_str = ' | '.join([f'{config.all_target_cols[i][:5]}: {r2:.3f}' for i, r2 in enumerate(per_r2)])
            lrs = [pg['lr'] for pg in optimizer.param_groups]
            lr_str = ' '.join([f'lr{i}={lr:.3e}' for i, lr in enumerate(lrs)])

            print(
                f'Fold {fold} | Epoch {epoch:02d} | '
                f'TLoss {tr_loss:.5f} | VLoss {val_loss:.5f} | '
                f'avgR2 {avg_r2:.4f} | GlobalR² {global_r2:.4f} '
                f'{"[BEST]" if global_r2 > best_global_r2 else ""} | '
                f'{lr_str} | time_tr={time_tr:.1f}s time_val={time_val:.1f}s time_ep={time_ep:.1f}s'
            )
            print(f'  → {per_r2_str}')

            if global_r2 > best_global_r2:
                print(f"🔥 Best updated: {best_global_r2:.3f} (epoch={epoch}) -> {global_r2}")
                best_global_r2 = global_r2
                best_avg_r2 = avg_r2

                # Save EMA weights directly (no CPU clone) for speed
                # NOTE: safest is to save state_dict tensors as-is; this is typically fine on DGX.
                torch.save(eval_model.state_dict(), save_path)
                print(f'  → SAVED EMA weights to {save_path} (GlobalR²: {best_global_r2:.4f})')

                patience = 0
                best_fold_preds = preds_fold
                best_fold_true = true_fold
            else:
                patience += 1
                if patience >= config.patience:
                    print(f'  → EARLY STOP (no improvement in {config.patience} epochs)')
                    break

            # keep memory tidy but avoid heavy cache/gc churn
            del preds_fold, true_fold

        if best_fold_preds is not None:
            oof_true.append(best_fold_true)
            oof_pred.append(best_fold_preds)
            fold_summary.append({'fold': fold, 'global_r2': best_global_r2, 'avg_r2': best_avg_r2})
        val_idxs.extend(val_idx)
        # Cleanup for this fold
        del model, tr_loader, val_loaders, optimizer, scheduler, ema, eval_model
        _sync()
        torch.cuda.empty_cache()
        gc.collect()

    if oof_true:
        oof_true_arr = np.concatenate(oof_true, axis=0)
        oof_pred_arr = np.concatenate(oof_pred, axis=0)
        oof_global_r2, oof_avg_r2, oof_per_r2 = weighted_r2_score_global(config, oof_true_arr, oof_pred_arr)

        print('\nTraining complete! Models saved in:', savedir / "model")
        print('Fold summary:')
        for fs in fold_summary:
            print(f"  Fold {fs['fold']}: Global R² = {fs['global_r2']:.4f}, Avg R² = {fs.get('avg_r2', float('nan')):.4f}")
        print(f'OOF Global Weighted R²: {oof_global_r2:.4f} | OOF Avg Target R²: {oof_avg_r2:.4f}')
        print('OOF Per-target:', dict(zip(config.all_target_cols, [f"{r:.4f}" for r in oof_per_r2])))

        oof_df = pd.DataFrame(
            oof_pred_arr, 
            columns=[f'oof_{col}' for col in config.all_target_cols]
        )
        oof_df['image_path'] = df_wide.iloc[val_idxs]['image_path'].values
        oof_df = oof_df[['image_path'] + [f'oof_{col}' for col in config.all_target_cols]]
        oof_path = savedir / "oof" / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        print(f'OOF predictions saved to {oof_path.resolve()}')

    else:
        print('No OOF predictions collected.')


if __name__ == "__main__":
    main()

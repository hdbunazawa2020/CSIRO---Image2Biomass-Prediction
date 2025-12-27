from __future__ import annotations

from typing import Sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(cfg, is_train: bool):
    """
    Args:
        cfg: OmegaConf / simple namespace
            required keys (examples):
              - cfg.img_size: int
              - cfg.normalize.mean: [float,float,float]
              - cfg.normalize.std:  [float,float,float]
              - cfg.augment.train.* (only if is_train)
        is_train: bool

    Returns:
        albumentations.Compose
    """
    img_size = int(getattr(cfg, "img_size", 224))

    mean: Sequence[float] = list(cfg.normalize.mean)
    std: Sequence[float] = list(cfg.normalize.std)

    # --- basic resize/pad (keep aspect ratio) ---
    base = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,   # cv2.BORDER_CONSTANT
            value=0,
        ),
    ]

    if is_train:
        # ---- read train aug params (with safe defaults) ----
        aug = cfg.augment.train
        hflip_p = float(getattr(aug, "hflip_p", 0.5))
        vflip_p = float(getattr(aug, "vflip_p", 0.0))
        rotate_limit = int(getattr(aug, "rotate_limit", 10))
        ssr_p = float(getattr(aug, "shift_scale_rotate_p", 0.2))
        cj_p = float(getattr(aug, "color_jitter_p", 0.2))

        train_aug = [
            A.HorizontalFlip(p=hflip_p),
            A.VerticalFlip(p=vflip_p),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=rotate_limit,
                border_mode=0,
                value=0,
                p=ssr_p,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=cj_p,
            ),
        ]

        tfm = A.Compose(
            base
            + train_aug
            + [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        tfm = A.Compose(
            base
            + [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    return tfm
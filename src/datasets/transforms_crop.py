from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TileShuffle(A.ImageOnlyTransform):
    """画像を grid に分割してタイルをシャッフルする（弱め推奨）。

    - H, W が grid で割り切れない端は切り捨てず、そのまま残す（中心ではなく先頭側）
    - 回帰なので強すぎは禁物。p低め、grid小さめ推奨。
    """
    def __init__(self, grid: Tuple[int, int] = (2, 2), always_apply: bool = False, p: float = 0.1):
        super().__init__(always_apply=always_apply, p=p)
        self.gh, self.gw = int(grid[0]), int(grid[1]) # default=(2,2)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w, c = img.shape
        gh, gw = self.gh, self.gw # 2, 2
        th = h // gh
        tw = w // gw
        if th <= 0 or tw <= 0:
            return img

        h2 = th * gh
        w2 = tw * gw

        core = img[:h2, :w2].copy()
        # Tileの分割
        tiles = []
        for i in range(gh):
            for j in range(gw):
                tiles.append(core[i*th:(i+1)*th, j*tw:(j+1)*tw, :])
        # Tileのシャッフル＆再配置
        perm = np.random.permutation(len(tiles))
        out = np.zeros_like(core)
        k = 0
        for i in range(gh):
            for j in range(gw):
                out[i*th:(i+1)*th, j*tw:(j+1)*tw, :] = tiles[perm[k]]
                k += 1

        img2 = img.copy()
        img2[:h2, :w2] = out
        return img2


def build_transforms(cfg, is_train: bool):
    """
    cfg 例（split-cropを使う場合）
      - cfg.use_split_crop: True
      - cfg.normalize.mean/std
      - cfg.augment.train.* (is_train=True のとき)

    split-crop の場合：
      - Dataset側で (768,768) crop 済み前提なので、ここでは Resize/Crop を基本しない
    """
    mean: Sequence[float] = list(cfg.normalize.mean)
    std: Sequence[float] = list(cfg.normalize.std)

    use_split_crop = bool(getattr(cfg, "use_split_crop", False))

    if use_split_crop:
        if is_train:
            aug = cfg.augment.train

            hflip_p = float(getattr(aug, "hflip_p", 0.5))
            vflip_p = float(getattr(aug, "vflip_p", 0.5))
            r90_p = float(getattr(aug, "rotate90_p", 0.5))
            ssr_p = float(getattr(aug, "shift_scale_rotate_p", 0.15))
            cj_p = float(getattr(aug, "color_jitter_p", 0.30))

            # CoarseDropout（弱め推奨）
            cd_p = float(getattr(aug, "coarse_dropout_p", 0.15))
            cd_max_holes = int(getattr(aug, "coarse_dropout_max_holes", 16))
            cd_min_h = int(getattr(aug, "coarse_dropout_min_h", 16))
            cd_min_w = int(getattr(aug, "coarse_dropout_min_w", 16))
            cd_max_h = int(getattr(aug, "coarse_dropout_max_h", 64))
            cd_max_w = int(getattr(aug, "coarse_dropout_max_w", 64))

            # TileShuffle（弱め推奨）
            tile_p = float(getattr(aug, "tile_shuffle_p", 0.10))
            tile_gh = int(getattr(aug, "tile_shuffle_gh", 2))
            tile_gw = int(getattr(aug, "tile_shuffle_gw", 2))

            # Blur/Noise（軽め）
            blur_noise_p = float(getattr(aug, "blur_noise_p", 0.10))

            tfm = A.Compose([
                A.HorizontalFlip(p=hflip_p),
                A.VerticalFlip(p=vflip_p),
                A.RandomRotate90(p=r90_p),

                # 軽い幾何（連続回転は無し or 極小）
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.08,
                    rotate_limit=0,
                    border_mode=0,
                    fill=0,
                    p=ssr_p,
                ),

                # 色（強すぎ注意：NDVI/植生信号を壊しやすい）
                A.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.05,
                    p=cj_p,
                ),

                # ★「たまにだけ」破壊系を入れる（どれか1つ）
                A.OneOf([
                    A.CoarseDropout(
                        max_holes=cd_max_holes,
                        min_holes=0,
                        min_height=cd_min_h,
                        min_width=cd_min_w,
                        max_height=cd_max_h,
                        max_width=cd_max_w,
                        fill_value=0,
                        p=1.0,
                    ),
                    TileShuffle(grid=(tile_gh, tile_gw), p=1.0),
                    A.OneOf([
                        A.GaussianBlur(blur_limit=3, p=1.0),
                        A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
                    ], p=1.0),
                ], p=max(cd_p, tile_p, blur_noise_p)),

                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            tfm = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        return tfm

    # ---- 従来互換：長方形に整形したい場合（いまのコードを維持）----
    img_h = int(getattr(cfg, "img_h", 224))
    img_w = int(getattr(cfg, "img_w", 224))

    base = [
        A.LongestMaxSize(max_size=max(img_h, img_w)), # 長辺を max(img_h, img_w) に合わせてアスペクト比維持で縮放
        A.PadIfNeeded( # その後、必要なら pad して最終サイズへ
            min_height=img_h,
            min_width=img_w,
            border_mode=0,
            fill=0,
        ),
        A.CenterCrop(height=img_h, width=img_w), # 余計に大きくなるケースを完全に潰すなら最後にCrop
    ]

    if is_train:
        aug = cfg.augment.train
        hflip_p = float(getattr(aug, "hflip_p", 0.5))
        vflip_p = float(getattr(aug, "vflip_p", 0.0))
        rotate_limit = int(getattr(aug, "rotate_limit", 10))
        ssr_p = float(getattr(aug, "shift_scale_rotate_p", 0.2))
        cj_p = float(getattr(aug, "color_jitter_p", 0.2))

        tfm = A.Compose(
            base # 基本の整形
            + [
                A.HorizontalFlip(p=hflip_p),
                A.VerticalFlip(p=vflip_p),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=rotate_limit,
                    border_mode=0,
                    fill=0,
                    p=ssr_p,
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=cj_p,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        tfm = A.Compose(base + [A.Normalize(mean=mean, std=std), ToTensorV2()])

    return tfm
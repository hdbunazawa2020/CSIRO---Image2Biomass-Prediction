# src/datasets/transforms_crop.py
from __future__ import annotations

from typing import Sequence, Tuple, Optional, Any, List

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TileShuffle(A.ImageOnlyTransform):
    """画像を grid に分割してタイルをシャッフルする（弱め推奨）。

    - H, W が grid で割り切れない端は切り捨てず、そのまま残す（先頭側）
    - 回帰なので強すぎは禁物。p低め、grid小さめ推奨。
    """
    def __init__(self, grid: Tuple[int, int] = (2, 2), always_apply: bool = False, p: float = 0.1):
        super().__init__(always_apply=always_apply, p=p)
        self.gh, self.gw = int(grid[0]), int(grid[1])

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w, c = img.shape
        gh, gw = self.gh, self.gw
        th = h // gh
        tw = w // gw
        if th <= 0 or tw <= 0:
            return img

        h2 = th * gh
        w2 = tw * gw

        core = img[:h2, :w2].copy()
        tiles: List[np.ndarray] = []
        for i in range(gh):
            for j in range(gw):
                tiles.append(core[i * th:(i + 1) * th, j * tw:(j + 1) * tw, :])

        perm = np.random.permutation(len(tiles))
        out = np.zeros_like(core)
        k = 0
        for i in range(gh):
            for j in range(gw):
                out[i * th:(i + 1) * th, j * tw:(j + 1) * tw, :] = tiles[perm[k]]
                k += 1

        img2 = img.copy()
        img2[:h2, :w2] = out
        return img2


# =========================================================
# small compat helpers (albumentations 1.x / 2.x)
# =========================================================
def _pad_if_needed(*, min_height: int, min_width: int, border_mode: int = 0, fill: int = 0) -> A.BasicTransform:
    """PadIfNeeded の fill/value 差を吸収."""
    try:
        return A.PadIfNeeded(min_height=min_height, min_width=min_width, border_mode=border_mode, fill=fill)
    except TypeError:
        # older albumentations
        return A.PadIfNeeded(min_height=min_height, min_width=min_width, border_mode=border_mode, value=fill)


def _shift_scale_rotate(*, shift_limit: float, scale_limit: float, rotate_limit: int,
                        border_mode: int = 0, fill: int = 0, p: float = 0.0) -> A.BasicTransform:
    """ShiftScaleRotate の fill/value 差を吸収."""
    try:
        return A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=border_mode,
            fill=fill,
            p=p,
        )
    except TypeError:
        return A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=border_mode,
            value=fill,
            p=p,
        )


def _coarse_dropout(
    *,
    max_holes: int,
    min_height: int,
    min_width: int,
    max_height: int,
    max_width: int,
    fill_value: int = 0,
    p: float = 0.0,
) -> A.BasicTransform:
    """CoarseDropout のAPI差を吸収（albumentations 1.x/2.x）."""
    # まず 1.x 系の引数
    try:
        return A.CoarseDropout(
            max_holes=max_holes,
            min_holes=0,
            min_height=min_height,
            min_width=min_width,
            max_height=max_height,
            max_width=max_width,
            fill_value=fill_value,
            p=p,
        )
    except TypeError:
        # 2.x 系の可能性（環境差があるため最小限の互換）
        # ※ 実環境で必要ならここを調整してください
        return A.CoarseDropout(
            num_holes_range=(0, max_holes),
            hole_height_range=(min_height, max_height),
            hole_width_range=(min_width, max_width),
            fill=fill_value,
            p=p,
        )


def _get_img_hw(cfg: Any) -> Tuple[int, int]:
    """cfg から (img_h, img_w) を取得（img_size fallback あり）。"""
    h = int(getattr(cfg, "img_h", 0) or 0)
    w = int(getattr(cfg, "img_w", 0) or 0)

    if h <= 0 or w <= 0:
        s = int(getattr(cfg, "img_size", 0) or 0)
        if s > 0:
            h, w = s, s

    return h, w


def build_transforms(cfg: Any, is_train: bool):
    """
    - split-crop の場合：
        Dataset側で crop 済み (例: 768x768) を受け取り、
        ここで Aug -> Resize(img_h,img_w) -> Normalize -> ToTensor
        とする（img_h/img_w 探索がそのまま効く）

    - 従来互換（split-crop無効）の場合：
        LongestMaxSize -> PadIfNeeded -> CenterCrop で整形後に Aug -> Normalize -> ToTensor
    """
    mean: Sequence[float] = list(cfg.normalize.mean)
    std: Sequence[float] = list(cfg.normalize.std)

    use_split_crop = bool(getattr(cfg, "use_split_crop", False))

    # =========================================================
    # split-crop branch
    # =========================================================
    if use_split_crop:
        img_h, img_w = _get_img_hw(cfg)  # sweepで変わる想定（0ならResizeしない）

        if is_train:
            aug = cfg.augment.train

            hflip_p = float(getattr(aug, "hflip_p", 0.5))
            vflip_p = float(getattr(aug, "vflip_p", 0.5))
            r90_p = float(getattr(aug, "rotate90_p", 0.5))
            ssr_p = float(getattr(aug, "shift_scale_rotate_p", 0.15))
            cj_p = float(getattr(aug, "color_jitter_p", 0.30))

            # CoarseDropout params
            cd_p = float(getattr(aug, "coarse_dropout_p", 0.0))
            cd_max_holes = int(getattr(aug, "coarse_dropout_max_holes", 16))
            cd_min_h = int(getattr(aug, "coarse_dropout_min_h", 16))
            cd_min_w = int(getattr(aug, "coarse_dropout_min_w", 16))
            cd_max_h = int(getattr(aug, "coarse_dropout_max_h", 64))
            cd_max_w = int(getattr(aug, "coarse_dropout_max_w", 64))

            # TileShuffle
            tile_p = float(getattr(aug, "tile_shuffle_p", 0.0))
            tile_gh = int(getattr(aug, "tile_shuffle_gh", 2))
            tile_gw = int(getattr(aug, "tile_shuffle_gw", 2))

            # Blur/Noise
            blur_p = float(getattr(aug, "blur_noise_p", 0.0))

            tfms: List[A.BasicTransform] = [
                A.HorizontalFlip(p=hflip_p),
                A.VerticalFlip(p=vflip_p),
                A.RandomRotate90(p=r90_p),

                # 軽い幾何（連続回転は無し）
                _shift_scale_rotate(
                    shift_limit=0.03,
                    scale_limit=0.08,
                    rotate_limit=0,
                    border_mode=0,
                    fill=0,
                    p=ssr_p,
                ),

                # 色（強すぎ注意）
                A.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.05,
                    p=cj_p,
                ),
            ]

            # ---- 「どれか1つ」破壊系：pがそのまま発生確率になるようにする ----
            destructive_candidates: List[A.BasicTransform] = []
            p_total = 0.0

            if cd_p > 0:
                destructive_candidates.append(
                    _coarse_dropout(
                        max_holes=cd_max_holes,
                        min_height=cd_min_h,
                        min_width=cd_min_w,
                        max_height=cd_max_h,
                        max_width=cd_max_w,
                        fill_value=0,
                        p=cd_p,  # OneOf内では重みとして扱われる（外側pと合わせて確率がcd_pになる）
                    )
                )
                p_total += cd_p

            if tile_p > 0:
                destructive_candidates.append(TileShuffle(grid=(tile_gh, tile_gw), p=tile_p))
                p_total += tile_p

            if blur_p > 0:
                destructive_candidates.append(
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=3, p=1.0),
                            A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
                        ],
                        p=blur_p,  # これも重み扱い
                    )
                )
                p_total += blur_p

            if destructive_candidates and p_total > 0:
                # p_total <= 1 の範囲なら、各pがそのまま発生確率になる
                tfms.append(A.OneOf(destructive_candidates, p=min(1.0, p_total)))

            # ---- Resize（split-cropでも img_h/img_w を効かせる）----
            if img_h > 0 and img_w > 0:
                tfms.append(A.Resize(height=img_h, width=img_w))

            tfms += [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
            return A.Compose(tfms)

        # valid
        tfms: List[A.BasicTransform] = []
        if img_h > 0 and img_w > 0:
            tfms.append(A.Resize(height=img_h, width=img_w))
        tfms += [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
        return A.Compose(tfms)

    # =========================================================
    # non split-crop branch (legacy)
    # =========================================================
    img_h, img_w = _get_img_hw(cfg)
    if img_h <= 0:
        img_h = 224
    if img_w <= 0:
        img_w = 224

    base: List[A.BasicTransform] = [
        A.LongestMaxSize(max_size=max(img_h, img_w)),
        _pad_if_needed(min_height=img_h, min_width=img_w, border_mode=0, fill=0),
        A.CenterCrop(height=img_h, width=img_w),
    ]

    if is_train:
        aug = cfg.augment.train
        hflip_p = float(getattr(aug, "hflip_p", 0.5))
        vflip_p = float(getattr(aug, "vflip_p", 0.0))
        rotate_limit = int(getattr(aug, "rotate_limit", 10))
        ssr_p = float(getattr(aug, "shift_scale_rotate_p", 0.2))
        cj_p = float(getattr(aug, "color_jitter_p", 0.2))

        tfm = A.Compose(
            base
            + [
                A.HorizontalFlip(p=hflip_p),
                A.VerticalFlip(p=vflip_p),
                _shift_scale_rotate(
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
        return tfm

    # valid
    return A.Compose(base + [A.Normalize(mean=mean, std=std), ToTensorV2()])
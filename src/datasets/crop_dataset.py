# -*- coding: utf-8 -*-
"""Aux対応 CsiroDataset (+ split & random crop).

- 入力画像(1000x2000)を左右に分割(1000x1000)x2
- 左右それぞれからランダムcrop(768x768)を作り、(2,C,768,768)で返す
- 画像 + 5ターゲット（log1p optional）
- 追加で aux_target を返せる
    - species: クラス分類（欠損/未知は -1）
    - ndvi/height: 回帰（欠損 mask 付き）

Notes:
    - DataLoader の default collate で nested dict もそのままバッチ化されます。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset


class CsiroDataset(Dataset):
    """CSIRO Image2Biomass 用 Dataset（aux対応 + split/crop対応）。

    Args:
        df: df_pivot.csv 相当（pandas.DataFrame）
        image_root: 画像ルートディレクトリ
        target_cols: 目的変数の列名（例: 5ターゲット）
        transform: albumentations transform（image -> Tensor(C,H,W)）
            - transform は「crop後の画像」に適用されます
        use_log1p_target: Trueなら target を log1p して返す
        return_target: Trueなら target を返す（学習/検証）。Falseなら返さない（推論）
        aux_cfg: cfg.aux（OmegaConf / dict / object 何でも）
        species_to_index: species文字列 -> class id の辞書
        aux_cols: aux列名の上書き（Noneなら aux_cfg から推定）
        use_split_crop: Trueなら左右split + random crop を有効化
        crop_size: cropサイズ（デフォルト768）
        assume_size: 元画像サイズ想定 (H,W) = (1000,2000)（違っても動くように安全に処理）
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # 壊れ気味画像で落ちにくくする（任意）

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        target_cols: Optional[List[str]] = None,
        transform=None,
        use_log1p_target: bool = True,
        return_target: bool = True,
        # aux
        aux_cfg: Any = None,
        species_to_index: Optional[Dict[str, int]] = None,
        aux_cols: Optional[Dict[str, str]] = None,
        # split/crop
        use_split_crop: bool = True,
        crop_size: int = 768,
        assume_size: tuple[int, int] = (1000, 2000),
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        # 設定読み取り
        self.use_log1p_target = bool(use_log1p_target)
        self.return_target = bool(return_target)
        self.target_cols = list(target_cols) if target_cols is not None else []
        # split/crop
        self.use_split_crop = bool(use_split_crop)
        self.crop_size = int(crop_size)
        self.assume_h, self.assume_w = int(assume_size[0]), int(assume_size[1]) # 想定サイズ, データは1000x2000一定なので設定しなくてもOK

        # ---- id / path を配列化（getitem高速化）----
        self.ids = self.df["image_id"].astype(str).values
        self.image_paths = self.df["image_path"].astype(str).values

        # ---- target（学習用のみ）----
        self.targets: Optional[np.ndarray] # np.ndarray か None のどちらかを想定と型を宣言
        if self.return_target:
            y = self.df[self.target_cols].values.astype(np.float32)
            if self.use_log1p_target:
                y = np.log1p(np.clip(y, 0.0, None))
            self.targets = y
        else:
            self.targets = None # test時はNone

        # ---- aux 設定 ----
        self.aux_cfg = aux_cfg
        self.aux_enabled = bool(getattr(aux_cfg, "enabled", False)) if aux_cfg is not None else False
        self.species_to_index = species_to_index or {}

        # aux列名（aux_colsが無いなら aux_cfg から推定）
        if aux_cols is None:
            sp_col = "Species"
            ndvi_col = "Pre_GSHH_NDVI"
            h_col = "Height_Ave_cm"

            if aux_cfg is not None:
                sp_col = str(getattr(getattr(aux_cfg, "species", None), "col", sp_col))
                ndvi_col = str(getattr(getattr(aux_cfg, "ndvi", None), "col", ndvi_col))
                h_col = str(getattr(getattr(aux_cfg, "height", None), "col", h_col))

            self.aux_cols = {"species": sp_col, "ndvi": ndvi_col, "height": h_col}
        else:
            self.aux_cols = dict(aux_cols)

        # ---- aux を前計算（df.iloc を避ける）----
        n = len(self.df)
        self._species_id = np.full(n, -1, dtype=np.int64)
        self._ndvi = np.zeros(n, dtype=np.float32)
        self._ndvi_mask = np.zeros(n, dtype=np.float32)
        self._height = np.zeros(n, dtype=np.float32)
        self._height_mask = np.zeros(n, dtype=np.float32)

        if self.aux_enabled:
            # Species
            sp_col = self.aux_cols["species"]
            if sp_col in self.df.columns:
                sp_vals = self.df[sp_col].values
                for i, v in enumerate(sp_vals):
                    if pd.isna(v):
                        continue
                    key = str(v)
                    self._species_id[i] = int(self.species_to_index.get(key, -1))
            # NDVI （植生の「元気さ（活性度）」を示す指標）
            ndvi_col = self.aux_cols["ndvi"]
            if ndvi_col in self.df.columns:
                nd = pd.to_numeric(self.df[ndvi_col], errors="coerce").values.astype(np.float32)
                m = ~np.isnan(nd)
                self._ndvi[m] = nd[m]
                self._ndvi_mask[m] = 1.0
            # Height
            h_col = self.aux_cols["height"]
            if h_col in self.df.columns:
                hh = pd.to_numeric(self.df[h_col], errors="coerce").values.astype(np.float32) # height
                m = ~np.isnan(hh) # mask
                self._height[m] = hh[m] # height
                self._height_mask[m] = 1.0 # mask

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, img_path: Path) -> np.ndarray:
        """画像を RGB で読み込み、HWC uint8 numpy を返す。"""
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            return np.asarray(img, dtype=np.uint8)

    def _split_lr(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """HWC画像を左右に分割して返す（left, right）。"""
        h, w, c = img.shape
        mid = w // 2
        left = img[:, :mid, :]
        right = img[:, mid:, :]
        return left, right

    def _random_crop_hwc(self, img: np.ndarray, crop_size: int) -> np.ndarray:
        """HWC uint8 からランダムcrop(HWC)を返す。"""
        h, w, _ = img.shape
        if h < crop_size or w < crop_size:
            # ここは「落とす」より「安全に中心crop（縮小なし）」に寄せた方が学習が止まらない
            top = max((h - crop_size) // 2, 0)
            left = max((w - crop_size) // 2, 0)
            return img[top:top + crop_size, left:left + crop_size, :]

        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        return img[top:top + crop_size, left:left + crop_size, :]

    def _to_tensor_chw(self, img: np.ndarray) -> torch.Tensor:
        """HWC uint8 -> Tensor(C,H,W) float32 [0,1]."""
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1サンプルを返す。
        image_id = self.ids[idx]
        img_path = self.image_root / self.image_paths[idx]
        img = self._load_image(img_path)  # HWC uint8

        # ===============================
        # 画像処理: split/crop + transform 適用
        # ===============================
        if self.use_split_crop:
            left, right = self._split_lr(img)

            left_crop = self._random_crop_hwc(left, self.crop_size)
            right_crop = self._random_crop_hwc(right, self.crop_size)

            # transform 適用
            if self.transform is not None:
                # albumentations は HWC を受ける想定
                t0 = self.transform(image=left_crop)["image"]   # Tensor(C,H,W)
                t1 = self.transform(image=right_crop)["image"]  # Tensor(C,H,W)
            else:
                t0 = self._to_tensor_chw(left_crop)
                t1 = self._to_tensor_chw(right_crop)
            # (2,C,H,W)
            image_tensor = torch.stack([t0, t1], dim=0)  # (2,C,768,768)

        else:
            # 従来互換: (C,H,W)
            if self.transform is not None:
                image_tensor = self.transform(image=img)["image"]
            else:
                image_tensor = self._to_tensor_chw(img)
        
        # ===============================
        # out: Dict[str, Any] を組み立てる
        # ===============================
        # ID, image をoutにセット
        out: Dict[str, Any] = {"id": image_id, "image": image_tensor}
        # ターゲット
        if self.return_target:
            assert self.targets is not None
            out["target"] = torch.from_numpy(self.targets[idx])  # (K,)
        # auxターゲット
        if self.aux_enabled:
            out["aux_target"] = {
                "species": torch.tensor(self._species_id[idx], dtype=torch.long),
                "ndvi": torch.tensor(self._ndvi[idx], dtype=torch.float32),
                "ndvi_mask": torch.tensor(self._ndvi_mask[idx], dtype=torch.float32),
                "height": torch.tensor(self._height[idx], dtype=torch.float32),
                "height_mask": torch.tensor(self._height_mask[idx], dtype=torch.float32),
            }

        return out
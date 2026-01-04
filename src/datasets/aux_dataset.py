# -*- coding: utf-8 -*-
"""Aux対応 CsiroDataset.

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
from PIL import Image

import torch
from torch.utils.data import Dataset


class CsiroDataset(Dataset):
    """CSIRO Image2Biomass 用 Dataset（aux対応）。

    Args:
        df: df_pivot.csv 相当（pandas.DataFrame）
        image_root: 画像ルートディレクトリ
        target_cols: 目的変数の列名（例: 5ターゲット）
        transform: albumentations transform（image -> Tensor(C,H,W)）
        use_log1p_target: Trueなら target を log1p して返す
        return_target: Trueなら target を返す（学習/検証）。Falseなら返さない（推論）
        aux_cfg: cfg.aux（OmegaConf / dict / object 何でも）
        species_to_index: species文字列 -> class id の辞書
        aux_cols: aux列名の上書き（Noneなら aux_cfg から推定）
    """

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
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

        self.use_log1p_target = bool(use_log1p_target)
        self.return_target = bool(return_target)
        self.target_cols = list(target_cols) if target_cols is not None else []

        # ---- id / path を配列化（getitem高速化）----
        self.ids = self.df["image_id"].astype(str).values
        self.image_paths = self.df["image_path"].astype(str).values

        # ---- target（学習用のみ）----
        self.targets: Optional[np.ndarray]
        if self.return_target:
            y = self.df[self.target_cols].values.astype(np.float32)
            if self.use_log1p_target:
                y = np.log1p(np.clip(y, 0.0, None))
            self.targets = y
        else:
            self.targets = None

        # ---- aux 設定 ----
        self.aux_cfg = aux_cfg
        self.aux_enabled = bool(getattr(aux_cfg, "enabled", False)) if aux_cfg is not None else False
        self.species_to_index = species_to_index or {}

        # aux列名（aux_colsが無いなら aux_cfg から推定）
        if aux_cols is None:
            # aux_cfg が無くても落ちないデフォルト
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

            # NDVI
            ndvi_col = self.aux_cols["ndvi"]
            if ndvi_col in self.df.columns:
                nd = pd.to_numeric(self.df[ndvi_col], errors="coerce").values.astype(np.float32)
                m = ~np.isnan(nd)
                self._ndvi[m] = nd[m]
                self._ndvi_mask[m] = 1.0

            # Height
            h_col = self.aux_cols["height"]
            if h_col in self.df.columns:
                hh = pd.to_numeric(self.df[h_col], errors="coerce").values.astype(np.float32)
                m = ~np.isnan(hh)
                self._height[m] = hh[m]
                self._height_mask[m] = 1.0

    def __len__(self) -> int:
        return len(self.ids)

    # def _load_image(self, img_path: Path) -> np.ndarray:
    #     """画像を RGB で読み込み、HWC の uint8 numpy を返す。"""
    #     img = Image.open(img_path).convert("RGB")
    #     return np.asarray(img)
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # 壊れ気味画像で落ちにくくする（任意）

    def _load_image(self, img_path: Path) -> np.ndarray:
        """画像を RGB で読み込み、HWC uint8 numpy を返す。"""
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            return np.asarray(img, dtype=np.uint8)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """1サンプルを返す。"""
        image_id = self.ids[idx]
        img_path = self.image_root / self.image_paths[idx]

        img = self._load_image(img_path)

        if self.transform is not None:
            image_tensor = self.transform(image=img)["image"]
        else:
            image_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        out: Dict[str, Any] = {
            "id": image_id,
            "image": image_tensor,
        }

        if self.return_target:
            assert self.targets is not None
            out["target"] = torch.from_numpy(self.targets[idx])  # (K,)

        if self.aux_enabled:
            out["aux_target"] = {
                "species": torch.tensor(self._species_id[idx], dtype=torch.long),
                "ndvi": torch.tensor(self._ndvi[idx], dtype=torch.float32),
                "ndvi_mask": torch.tensor(self._ndvi_mask[idx], dtype=torch.float32),
                "height": torch.tensor(self._height[idx], dtype=torch.float32),
                "height_mask": torch.tensor(self._height_mask[idx], dtype=torch.float32),
            }

        return out
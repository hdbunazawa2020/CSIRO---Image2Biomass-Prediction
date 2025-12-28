from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

class CsiroDataset(Dataset):
    """CSIRO Image2Biomass 用 Dataset（画像のみ）。

    Args:
        df:
            df_pivot.csv (pandas想定.)
        image_root:
            画像ファイルのルートディレクトリ。(e.g. "/mnt/.../data/raw")
        target_cols:
            e.g.: ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
        transform:
            albumentations の transform を想定。
            transform(image=np.ndarray)["image"] が torch.Tensor (C,H,W) を返す想定。
        use_log1p_target:
            True の場合、ターゲットを log1p 変換して返します。(評価時は expm1 で元スケールに戻すのが前提。)
        return_target:
            True: 学習/検証用（target を返す）, False: 推論用（target を返さない）
    """
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        target_cols: Optional[List[str]] = None,
        transform=None,
        use_log1p_target: bool = True,
        return_target: bool = True,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True) # DataFrame は index の状態で事故りやすいので必ず reset
        self.image_root = Path(image_root)
        self.transform = transform
        self.use_log1p_target = bool(use_log1p_target)
        self.return_target = bool(return_target)
        self.target_cols = list(target_cols) if target_cols is not None else []
        # -- id / image_path を先に配列化（get_itemを軽くする）---
        self.ids = self.df["image_id"].astype(str).values
        self.image_paths = self.df["image_path"].astype(str).values

        # --- target を作成（学習用のみ）---
        self.targets: Optional[np.ndarray]
        if self.return_target:
            y = self.df[self.target_cols].values.astype(np.float32)
            # log1p変換
            if self.use_log1p_target:
                y = np.log1p(y)
            self.targets = y
        else:
            self.targets = None

    def __len__(self) -> int:
        """データ数（画像枚数）を返す。"""
        return len(self.df)

    def _load_image(self, img_path: Path) -> np.ndarray:
        """画像を読み込んで numpy.ndarray(H,W,C) で返す。

        Args:
            img_path: 画像のフルパス

        Returns:
            np.ndarray: shape=(H,W,C), dtype=uint8
        """
        # PIL -> RGB固定（png/jpg混在でも統一）
        img = Image.open(img_path).convert("RGB")
        return np.asarray(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """idx番目のサンプルを返す。

        Args:
            idx: サンプルindex

        Returns:
            dict:
                学習時:
                  {"id": str, "image": Tensor(C,H,W), "target": Tensor(K,)}
                推論時:
                  {"id": str, "image": Tensor(C,H,W)}
        """
        # 画像パスの組み立て（dfが相対パスを持つ想定）
        img_path = self.image_root / self.image_paths[idx]
        # _load_image: 画像読み込み（H,W,C / uint8）
        img = self._load_image(img_path)
        # transform（albumentations想定）
        if self.transform is not None:
            # transform は dict を返し、image が torch.Tensor (C,H,W) の想定
            img_tensor = self.transform(image=img)["image"]
        else:
            # transform が無い場合の最低限フォールバック（学習には推奨しない）
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        out: Dict[str, torch.Tensor] = {
            "id": self.ids[idx],
            "image": img_tensor,
        }
        # 学習/検証時のみ target を返す
        if self.return_target:
            assert self.targets is not None
            y = torch.from_numpy(self.targets[idx])  # (K,)
            out["target"] = y
        return out
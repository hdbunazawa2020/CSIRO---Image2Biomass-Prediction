# -*- coding: utf-8 -*-
"""
dataset.py

CSIRO Image2Biomass コンペ用 Dataset（画像のみ入力）

この Dataset は、preprocess で作成した df_pivot.csv（画像単位のテーブル）を入力として、
画像ファイルを読み込み、学習/推論に必要なデータを返します。

本コンペの前提（重要）
- test データには tabular 特徴量が存在しない
  → 推論で tabular を入力に使えないため、本 Dataset も「画像のみ」を返す構成にします。
- ターゲットは 5つの回帰値（質量）で、非負が前提。
- 学習安定化のため、target を log1p して学習し、評価時に expm1 で戻すのが一般的です。

返り値（学習時）
    {
        "id": str,
        "image": torch.Tensor,  # (C, H, W)
        "target": torch.Tensor, # (K,)  ※ return_target=True のときのみ
    }

返り値（推論時）
    {
        "id": str,
        "image": torch.Tensor,  # (C, H, W)
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetKeys:
    """Datasetが返すdictのキーを一箇所で管理するための定数。"""
    ID: str = "id"
    IMAGE: str = "image"
    TARGET: str = "target"


class CsiroDataset(Dataset):
    """CSIRO Image2Biomass 用 Dataset（画像のみ）。

    Args:
        df:
            df_pivot.csv を pandas で読み込んだ DataFrame。
            1行 = 1画像を想定。
        image_root:
            画像ファイルのルートディレクトリ。
            例: "/mnt/.../data/raw"
            ※ df["image_path"] が "train/IDxxx.jpg" のような相対パスの場合、
               image_root と結合して参照します。
        id_col:
            画像ID列名。例: "image_id"
        image_col:
            画像パス列名。例: "image_path"
        target_cols:
            目的変数列名（順序が重要）。
            例: ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
            return_target=True の場合は必須。
        transform:
            albumentations の transform を想定。
            transform(image=np.ndarray)["image"] が torch.Tensor (C,H,W) を返す想定。
        use_log1p_target:
            True の場合、ターゲットを log1p 変換して返します。
            ※ 評価時は expm1 で元スケールに戻すのが前提。
        return_target:
            True: 学習/検証用（target を返す）
            False: 推論用（target を返さない）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        id_col: str,
        image_col: str,
        target_cols: Optional[List[str]] = None,
        transform=None,
        use_log1p_target: bool = True,
        return_target: bool = True,
    ) -> None:
        super().__init__()

        # DataFrame は index の状態で事故りやすいので必ず reset
        self.df = df.reset_index(drop=True)

        self.image_root = Path(image_root)
        self.id_col = str(id_col)
        self.image_col = str(image_col)

        self.transform = transform
        self.use_log1p_target = bool(use_log1p_target)
        self.return_target = bool(return_target)

        self.target_cols = list(target_cols) if target_cols is not None else []

        # -----------------------------
        # 必須カラムチェック
        # -----------------------------
        if self.id_col not in self.df.columns:
            raise KeyError(f"id_col='{self.id_col}' not found in df columns.")
        if self.image_col not in self.df.columns:
            raise KeyError(f"image_col='{self.image_col}' not found in df columns.")

        # -----------------------------
        # id / image_path を先に配列化（getitemを軽くする）
        # -----------------------------
        self.ids = self.df[self.id_col].astype(str).values
        self.image_paths = self.df[self.image_col].astype(str).values

        # -----------------------------
        # target を作成（学習用のみ）
        # -----------------------------
        self.targets: Optional[np.ndarray]
        if self.return_target:
            if len(self.target_cols) == 0:
                raise ValueError("return_target=True requires target_cols.")
            missing = [c for c in self.target_cols if c not in self.df.columns]
            if len(missing) > 0:
                raise KeyError(f"Missing target columns in df: {missing}")

            y = self.df[self.target_cols].values.astype(np.float32)

            # 万が一の NaN/inf 混入に備えて潰しておく（通常はpreprocessが正しければ不要）
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            # コンペのターゲットは非負前提。log1pのため負値があると壊れるので念のためクリップ。
            # （もし負値があり得る設計なら、ここは外す/変更する）
            y = np.clip(y, 0.0, None)

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

        # 画像読み込み（H,W,C / uint8）
        img = self._load_image(img_path)

        # transform（albumentations想定）
        if self.transform is not None:
            # transform は dict を返し、image が torch.Tensor (C,H,W) の想定
            img_tensor = self.transform(image=img)["image"]
        else:
            # transform が無い場合の最低限フォールバック（学習には推奨しない）
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        out: Dict[str, torch.Tensor] = {
            DatasetKeys.ID: self.ids[idx],
            DatasetKeys.IMAGE: img_tensor,
        }

        # 学習/検証時のみ target を返す
        if self.return_target:
            assert self.targets is not None
            y = torch.from_numpy(self.targets[idx])  # (K,)
            out[DatasetKeys.TARGET] = y

        return out
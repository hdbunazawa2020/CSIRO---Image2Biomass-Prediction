# -*- coding: utf-8 -*-
"""
timm backbone regressor (画像のみ入力).

このクラスは名前は ConvNeXtRegressor ですが、実態は
「timm の任意バックボーンを回帰に使う」ための汎用モデルです。

✅ 対応例:
- convnext_small / convnext_base
- swin_tiny_patch4_window7_224 など（Swin）
- tf_efficientnetv2_s など（EfficientNetV2）

注意:
- timm のモデルによって create_model の引数対応が微妙に違うので、
  TypeError が出たら「引数を減らして再try」するフォールバックを入れています。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import timm


_BACKBONE_ALIAS: Dict[str, str] = {
    # 使いやすいショート名 → timmの正式名
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnetv2_s": "tf_efficientnetv2_s",
}


class ConvNeXtRegressor(nn.Module):
    """timmバックボーン + 線形ヘッドの回帰モデル。

    Args:
        backbone: timm のモデル名（例: convnext_small, swin_tiny_patch4_window7_224）
        pretrained: 事前学習重みを使うか
        num_targets: 出力ターゲット数（CSIROは 5）
        in_chans: 入力チャンネル（RGB=3）
        drop_rate: backbone 内の dropout（対応モデルのみ）
        drop_path_rate: backbone 内の stochastic depth（対応モデルのみ）
        head_dropout: head の dropout（最後の線形の前）
        global_pool: 特徴の集約方法（対応モデルのみ。通常 "avg"）
    """

    def __init__(
        self,
        backbone: str = "convnext_base",
        pretrained: bool = True,
        num_targets: int = 5,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        head_dropout: float = 0.0,
        global_pool: str = "avg",
    ) -> None:
        super().__init__()

        # alias 対応（swin_tiny など）
        backbone = _BACKBONE_ALIAS.get(backbone, backbone)
        self.backbone_name = backbone

        # -------------------------
        # backbone（特徴抽出器）
        # num_classes=0 で「分類headなし」の特徴抽出モード
        # -------------------------
        kwargs = dict(
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
        )

        # timm のモデルによっては global_pool / drop_* を受け取れないことがあるので安全にフォールバック
        try:
            self.backbone = timm.create_model(backbone, **kwargs)
        except TypeError:
            # global_pool を外す
            kwargs2 = dict(kwargs)
            kwargs2.pop("global_pool", None)
            try:
                self.backbone = timm.create_model(backbone, **kwargs2)
            except TypeError:
                # drop系も外す
                kwargs3 = dict(kwargs2)
                kwargs3.pop("drop_rate", None)
                kwargs3.pop("drop_path_rate", None)
                self.backbone = timm.create_model(backbone, **kwargs3)

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            raise RuntimeError(
                f"[ERROR] timm model '{backbone}' does not have 'num_features'. "
                "Please check timm version / model name."
            )

        self.head_dropout = nn.Dropout(p=float(head_dropout)) if head_dropout and head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(int(feat_dim), int(num_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: (B, C, H, W)

        Returns:
            y: (B, num_targets)
        """
        feat = self.backbone(x)

        # 念のため、モデルによって 3D/4D で返るケースを吸収
        # 例:
        # - CNN系: (B, C) or (B, C, H, W)
        # - Transformer系: (B, N, C)
        if feat.ndim == 4:
            feat = feat.mean(dim=(2, 3))
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)

        feat = self.head_dropout(feat)
        out = self.head(feat)
        return out
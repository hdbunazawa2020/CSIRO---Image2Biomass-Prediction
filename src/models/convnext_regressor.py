# /mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction/src/models/convnext_regressor.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import timm


_BACKBONE_ALIAS: Dict[str, str] = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnetv2_s": "tf_efficientnetv2_s",
}


class ConvNeXtRegressor(nn.Module):
    """timmバックボーン + 線形ヘッドの回帰モデル。

    重要:
        あなたの前処理は (H,W)=(img_size, img_size*2) の 1:2 入力です。
        Swin など一部モデルは forward 内で「入力サイズが初期化時の img_size と一致するか」を assert します。
        そのため、Swin を使う場合は timm.create_model に img_size=(H,W) を渡す必要があります。

    Args:
        backbone: timm のモデル名（例: convnext_small, swin_tiny_patch4_window7_224）
        pretrained: 事前学習重みを使うか
        num_targets: 出力ターゲット数（CSIROは 5）
        in_chans: 入力チャンネル（RGB=3）
        drop_rate: backbone 内の dropout（対応モデルのみ）
        drop_path_rate: backbone 内の stochastic depth（対応モデルのみ）
        head_dropout: head の dropout（最後の線形の前）
        global_pool: 特徴の集約方法（対応モデルのみ。通常 "avg"）
        img_size: (H, W) を明示する（Swin等の assert 回避に必須）
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
        img_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        backbone = _BACKBONE_ALIAS.get(backbone, backbone)
        self.backbone_name = backbone

        # -------------------------
        # backbone（特徴抽出器）
        # -------------------------
        kwargs = dict(
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
        )

        # ✅ Swin 等で必要：img_size=(H,W)
        if img_size is not None:
            kwargs["img_size"] = (int(img_size[0]), int(img_size[1]))

        # timm はモデルによって受け取れる引数が違うので、段階的に削ってフォールバック
        try:
            self.backbone = timm.create_model(backbone, **kwargs)
        except TypeError:
            # 1) global_pool を外す
            kwargs2 = dict(kwargs)
            kwargs2.pop("global_pool", None)
            try:
                self.backbone = timm.create_model(backbone, **kwargs2)
            except TypeError:
                # 2) img_size を外す（ConvNeXt等では不要/未対応なことがある）
                kwargs3 = dict(kwargs2)
                kwargs3.pop("img_size", None)
                try:
                    self.backbone = timm.create_model(backbone, **kwargs3)
                except TypeError:
                    # 3) drop系も外す
                    kwargs4 = dict(kwargs3)
                    kwargs4.pop("drop_rate", None)
                    kwargs4.pop("drop_path_rate", None)
                    self.backbone = timm.create_model(backbone, **kwargs4)

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

        # モデルによって 3D/4D で返るケースを吸収
        if feat.ndim == 4:
            feat = feat.mean(dim=(2, 3))
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)

        feat = self.head_dropout(feat)
        out = self.head(feat)
        return out
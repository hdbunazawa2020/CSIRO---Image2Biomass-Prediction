"""
ConvNeXtRegressor（画像のみ入力の回帰モデル）

本コンペでは、test データで利用できる情報が
- sample_id
- image_path
- target_name
などの最小限であり、推論時にタブラー特徴量（df_pivot.csv由来）を
入力として使えません。

そのため、本モデルは以下を前提にしています。

- 入力: 画像テンソルのみ (B, C, H, W)
- 出力: 5ターゲットの質量推定 (B, 5)
- Backbone: timm の ConvNeXt 系モデル
- Pooling: Global Average Pooling（timm側で `global_pool="avg"`）

将来的に aux loss（例：高さ/NDVIの推定、Species分類など）を入れたい場合は、
`forward_features()` で画像特徴量 (B, feat_dim) を取り出し、
別 head を追加して multi-task 化できます（このファイルではまだ実装しない）。

Notes:
    - num_targets の順序は、学習・評価・推論で必ず統一してください。
      例: [Dry_Green_g, Dry_Clover_g, Dry_Dead_g, GDM_g, Dry_Total_g]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import timm


class ConvNeXtRegressor(nn.Module):
    """ConvNeXt を用いた画像回帰モデル。

    Args:
        backbone (str):
            timm で利用可能なバックボーン名。
            例: "convnext_tiny", "convnext_small", "convnext_base"
        pretrained (bool):
            ImageNet などの事前学習重みを使用するかどうか。
        num_targets (int):
            回帰ターゲット数。コンペでは 5 を想定。
        in_chans (int):
            入力チャンネル数。RGBなら 3。
        drop_rate (float):
            timm backbone の dropout（モデルによって効き方が異なる）。
        drop_path_rate (float):
            timm backbone の stochastic depth（モデルによって効き方が異なる）。
        head_dropout (float):
            回帰 head 直前に入れる dropout。過学習が強い場合に有効なことがある。

    Returns:
        torch.Tensor:
            (B, num_targets) の回帰出力。
            ※ 学習時に log1p 変換を使う場合でも、モデルは「その空間の値」を出力するだけ。
              変換/逆変換は dataset / metric 側で統一して管理するのがおすすめ。
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
    ) -> None:
        super().__init__()

        # -----------------------------
        # 1) 画像エンコーダ（特徴抽出器）
        # -----------------------------
        # num_classes=0 -> 分類 head を持たず、特徴量のみを返す
        # global_pool="avg" -> (B, feat_dim) のベクトルにする
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # timm のモデルは通常 num_features を持つ
        feat_dim = getattr(self.encoder, "num_features", None)
        if feat_dim is None:
            raise ValueError(f"timm model '{backbone}' does not have attribute 'num_features'")

        self.feat_dim = int(feat_dim)
        self.num_targets = int(num_targets)

        # -----------------------------
        # 2) 回帰 head（最小構成）
        # -----------------------------
        # ここはまずはシンプルに:
        #  - Dropout（任意）
        #  - Linear
        if head_dropout and head_dropout > 0:
            self.head = nn.Sequential(
                nn.Dropout(p=float(head_dropout)),
                nn.Linear(self.feat_dim, self.num_targets),
            )
        else:
            self.head = nn.Linear(self.feat_dim, self.num_targets)

        # -----------------------------
        # 3) aux head (将来拡張用の置き場所)
        # -----------------------------
        # 現時点では実装しない（推論で使えない特徴量を入力にしないため）
        # 例:
        #   self.aux_head = nn.Linear(self.feat_dim, aux_dim)
        #   forward() で {"pred": pred, "aux": aux} を返す等
        self.aux_head: Optional[nn.Module] = None

        # 初期化（headのみ）
        self._init_head()

    def _init_head(self) -> None:
        """回帰 head の初期化。

        timm の backbone 側は事前学習重みを使うので触らず、
        head だけを軽く初期化します。
        """
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """画像から特徴量ベクトルを抽出する。

        Args:
            x (torch.Tensor):
                画像テンソル (B, C, H, W)

        Returns:
            torch.Tensor:
                特徴量 (B, feat_dim)
        """
        # エンコーダは (B, feat_dim) を返す（global_pool="avg"）
        feat = self.encoder(x)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播（回帰）。

        Args:
            x (torch.Tensor):
                画像テンソル (B, C, H, W)

        Returns:
            torch.Tensor:
                回帰出力 (B, num_targets)
        """
        feat = self.forward_features(x)  # (B, feat_dim)
        out = self.head(feat)            # (B, num_targets)
        return out
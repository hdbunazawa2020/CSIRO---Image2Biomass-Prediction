# root/src/models/biomass_mil_hurdle.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionPool2d(nn.Module):
    """空間方向(H, W)に対するシンプルな Attention Pooling.

    特徴マップ x: (B, C, H, W) から、
    - pooled: (B, C)  ... 空間を重み付き平均して1ベクトルへ
    - attn_map: (B, 1, H, W) ... 空間上の注意分布
    を返す。

    Attentionのスコアは 1x1 Conv で (B, 1, H, W) を作り、
    softmax により H*W 次元上で確率分布にする。

    Args:
        in_channels: 入力特徴マップのチャネル数 C.
        dropout: pooled ベクトルに掛ける dropout.
        temperature: softmax の温度パラメータ。小さいほど尖る/大きいほど平坦。

    Returns:
        forward() は (pooled, attn_map) を返す:
            pooled: (B, C)
            attn_map: (B, 1, H, W)
    """

    def __init__(self, in_channels: int, dropout: float = 0.0, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature は 0 より大きい必要があります。")

        # 各空間位置に対するスコア(=logits)を作る 1x1 Conv
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        # pooled ベクトルに対してだけ dropout を適用（注意マップには掛けない）
        self.dropout = nn.Dropout(dropout)
        # softmax(logits / temperature) に使う温度
        self.temperature = float(temperature)
        """
        	•	temperature が小さい（例: 0.3）
                •	logits / T が大きくなり、softmax が より尖る（ほぼ one-hot に近づく）
                •	つまり **「特定の場所だけを強く見る」**挙動になりやすい
                •	長所: 重要領域にフォーカスしやすい
                •	短所: 学習初期に不安定になったり、外れ領域にロックしてしまうリスクもある
            •	temperature が大きい（例: 2.0）
                •	logits / T が小さくなり、softmax が 平坦（均一に近い）
                •	つまり 平均プーリング寄りの挙動
                •	長所: 安定しやすい、過度に一点集中しにくい
                •	短所: 注意のメリット（「ここを見る」）が弱まる
        """
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            x: 特徴マップ (B, C, H, W)

        Returns:
            pooled: (B, C)
            attn_map: (B, 1, H, W)
        """
        b, c, h, w = x.shape

        # (B, 1, H, W) -> (B, 1, HW)
        logits = self.attn_conv(x).reshape(b, 1, h * w)

        # 温度付き softmax：空間(HW)方向に確率分布を作る
        weights = torch.softmax(logits / self.temperature, dim=-1)  # (B, 1, HW)

        # 特徴も (B, C, HW) に畳んで重み付き和を取る
        x_flat = x.reshape(b, c, h * w)  # (B, C, HW)
        pooled = torch.sum(x_flat * weights, dim=-1)  # (B, C)  ※weightsはbroadcastされる

        pooled = self.dropout(pooled)
        attn_map = weights.reshape(b, 1, h, w)
        return pooled, attn_map


class MILAggregator(nn.Module):
    """タイル（インスタンス）埋め込みを Bag 埋め込みに集約する MIL Aggregator.

    入力:
        h: (B, M, D)
            B: バッチ
            M: タイル枚数（インスタンス数）
            D: タイル埋め込み次元

    出力:
        bag: (B, D)  ... Bag 表現
        weights: (B, M)  ... attention mode のときのみ（それ以外は None）

    mode:
        - "mean": 平均
        - "max":  最大
        - "gated_attn": Ilse et al. (Attention-based MIL) 風の gated attention

    Args:
        dim: 入力埋め込みの次元 D.
        mode: "mean" / "max" / "gated_attn"
        attn_dim: gated_attn で使う中間次元
        dropout: 入力 h への dropout
    """

    def __init__(self, dim: int, mode: str = "gated_attn", attn_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.mode = mode
        self.dropout = nn.Dropout(dropout)

        if mode == "gated_attn":
            # a_i = w( tanh(V h_i) ⊙ sigmoid(U h_i) )
            self.v = nn.Linear(dim, attn_dim)
            self.u = nn.Linear(dim, attn_dim)
            self.w = nn.Linear(attn_dim, 1)
        elif mode in ("mean", "max"):
            # これらはパラメータ無し
            self.v = None
            self.u = None
            self.w = None
        else:
            raise ValueError(f"Unknown MIL mode: {mode}")

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward.

        Args:
            h: (B, M, D)

        Returns:
            bag: (B, D)
            weights: (B, M) if gated_attn else None
        """
        b, m, d = h.shape
        h = self.dropout(h)

        # タイルが1枚なら集約の意味がないので、そのまま返す
        if m == 1:
            bag = h[:, 0]  # (B, D)
            weights = torch.ones((b, 1), device=h.device, dtype=h.dtype)
            return bag, weights

        if self.mode == "mean":
            return h.mean(dim=1), None

        if self.mode == "max":
            return h.max(dim=1).values, None

        # --- gated attention (Ilse et al. style) ---
        # (B, M, attn_dim)
        v = torch.tanh(self.v(h))
        u = torch.sigmoid(self.u(h))

        # 要素積でゲーティング
        a = v * u  # (B, M, attn_dim)

        # 各タイルのスコアへ (B, M)
        scores = self.w(a).squeeze(-1)

        # タイル方向に softmax して重みを確率分布にする（各Bagで重みの合計は1）
        weights = torch.softmax(scores, dim=1)  # (B, M)

        # 重み付き和で Bag 埋め込み
        bag = torch.sum(h * weights.unsqueeze(-1), dim=1)  # (B, D)
        return bag, weights


class HurdleHead(nn.Module):
    """Hurdle（ゼロ過剰）を想定した「存在 + 正の量」ヘッド。

    発想:
        - まず「その成分が存在するか」を推定する（presence）
        - 存在するときの「量（>=0）」を推定する（amount）
        - 最終的な期待値 expected = presence_prob * amount_pos とする

    この expected を回帰ターゲットとして扱うと、
    ゼロ（非存在）のケースを presence 側で抑制しやすくなる。

    Args:
        in_dim: 入力埋め込み次元 D.
        out_dim: 出力成分数。ここでは 3（Green/Clover/Dead）。
        hidden_dim: amount 側MLPの中間次元
        dropout: amount 側に適用
    """

    def __init__(self, in_dim: int, out_dim: int = 3, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()

        # 各成分の「存在」を独立に推定（multi-label想定）
        self.presence = nn.Linear(in_dim, out_dim)

        # 各成分の「量」を推定（最後は softplus で非負にする）
        self.amount = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward.

        Args:
            x: (B, D)

        Returns:
            Dict[str, Tensor]:
                presence_logits: (B, 3)  ... 存在のlogit
                presence_prob:   (B, 3)  ... sigmoidで(0,1)へ
                amount_pos:      (B, 3)  ... softplusで >=0
                expected:        (B, 3)  ... presence_prob * amount_pos
        """
        # 存在有無の推定
        presence_logits = self.presence(x)          # (B, 3)
        presence_prob = torch.sigmoid(presence_logits)  # (B, 3) 各成分は独立に(0,1)

        # 量の推定
        amount_raw = self.amount(x)                # (B, 3)
        amount_pos = F.softplus(amount_raw)        # (B, 3) >= 0

        # 最終的期待値
        expected = presence_prob * amount_pos      # (B, 3)
        return {
            "presence_logits": presence_logits,
            "presence_prob": presence_prob,
            "amount_pos": amount_pos,
            "expected": expected,
        }


class BiomassConvNeXtMILHurdle(nn.Module):
    """ConvNeXt + (Spatial Attention Pooling) + (MIL) + HurdleHead.

    構成:
        1) timm の backbone (ConvNeXt など) で特徴マップを抽出
        2) AttentionPool2d で 空間(H,W) を (B, D) にプール
        3) タイル入力の場合は MILAggregator で タイル方向(M) を集約
        4) HurdleHead で 3成分 (Green/Clover/Dead) を推定
        5) GDM と Total は和で作る（物理整合を担保）

    出力順:
        [Dry_Green_g, Dry_Clover_g, Dry_Dead_g, GDM_g, Dry_Total_g]

    Args:
        backbone_name: timm のモデル名
        pretrained: ImageNet pretrained を使うか
        in_chans: 入力チャンネル数
        pool_dropout: 空間プール後の dropout
        pool_temperature: 空間 attention の温度
        mil_mode: "mean" / "max" / "gated_attn"
        mil_attn_dim: gated_attn の中間次元
        mil_dropout: MIL入力への dropout
        head_hidden_dim: hurdleの amount 側隠れ次元
        head_dropout: hurdleの amount 側 dropout
        return_attention: attention map を out に含めるか
    """

    def __init__(
        self,
        backbone_name: str = "convnext_base",
        pretrained: bool = True,
        in_chans: int = 3,
        pool_dropout: float = 0.0,
        pool_temperature: float = 1.0,
        mil_mode: str = "gated_attn",
        mil_attn_dim: int = 256,
        mil_dropout: float = 0.0,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.return_attention = return_attention

        # classifier head を外して空間特徴マップを得るため global_pool="" を指定
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
        )

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            raise RuntimeError("backbone の num_features を取得できません。timm model を確認してください。")

        self.spatial_pool = AttentionPool2d(
            in_channels=feat_dim,
            dropout=pool_dropout,
            temperature=pool_temperature,
        )

        self.mil = MILAggregator(
            dim=feat_dim,
            mode=mil_mode,
            attn_dim=mil_attn_dim,
            dropout=mil_dropout,
        )

        self.head = HurdleHead(
            in_dim=feat_dim,
            out_dim=3,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backboneから特徴マップを取り出す共通処理.

        Args:
            x: (B, C, H, W)

        Returns:
            feat: (B, D, H', W') を期待
        """
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone(x)

        # timmモデルによっては list/tuple を返すことがあるので最後を取る
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        return feat

    @staticmethod
    def _compose_outputs(comp3: torch.Tensor) -> torch.Tensor:
        """3成分 (B,3) から 5出力 (B,5) を組み立てる.

        Args:
            comp3: (B, 3) = [Green, Clover, Dead]

        Returns:
            pred5: (B, 5) = [Green, Clover, Dead, GDM, Total]
        """
        green = comp3[:, 0]
        clover = comp3[:, 1]
        dead = comp3[:, 2]
        gdm = green + clover
        total = gdm + dead
        pred5 = torch.stack([green, clover, dead, gdm, total], dim=1)
        return pred5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward.

        入力形式:
            - (B, 3, H, W)        : 単一画像（single view）
            - (B, M, 3, H, W)     : タイル入力（MIL）

        Returns:
            out: Dict[str, Tensor]
                pred: (B, 5)
                pred_components: (B, 3)
                presence_logits/prob/amount_pos
                pred_log1p: (B, 5)
                return_attention=True の場合:
                    spatial_attn:
                        single: (B, 1, H', W')
                        tiles : (B, M, 1, H', W')
                    tile_attn: (B, M)  ※MILがgated_attnのとき
        """
        spatial_attn: Optional[torch.Tensor] = None
        tile_attn: Optional[torch.Tensor] = None

        # --- タイル入力 (B, M, C, H, W) ---
        if x.dim() == 5:
            b, m, c, h, w = x.shape

            # backbone は 4D 入力想定なので (B*M, C, H, W) に潰す
            x_flat = x.reshape(b * m, c, h, w)

            feat = self._forward_features(x_flat)               # (B*M, D, H', W')
            tile_emb, tile_sp_attn = self.spatial_pool(feat)    # (B*M, D), (B*M,1,H',W')

            # (B, M, D) に戻して MIL
            tile_emb = tile_emb.reshape(b, m, -1)
            bag_emb, tile_attn = self.mil(tile_emb)             # (B, D), (B, M) or None

            if self.return_attention:
                _, _, hh, ww = tile_sp_attn.shape
                spatial_attn = tile_sp_attn.reshape(b, m, 1, hh, ww)

        # --- 単一画像入力 (B, C, H, W) ---
        elif x.dim() == 4:
            feat = self._forward_features(x)                    # (B, D, H', W')
            bag_emb, sp_attn = self.spatial_pool(feat)          # (B, D), (B,1,H',W')
            if self.return_attention:
                spatial_attn = sp_attn

        else:
            raise ValueError(f"Unexpected input dim: {x.dim()} (expected 4 or 5)")

        # Hurdle head で 3成分を推定（expected = presence_prob * amount_pos）
        head_out = self.head(bag_emb)
        comp3 = head_out["expected"]  # (B, 3)

        # 5出力へ変換（GDM/Totalは和）
        pred5 = self._compose_outputs(comp3)

        out: Dict[str, torch.Tensor] = {
            "pred": pred5,
            "pred_components": comp3,
            "presence_logits": head_out["presence_logits"],
            "presence_prob": head_out["presence_prob"],
            "amount_pos": head_out["amount_pos"],
            # log1p 学習/評価のための補助出力（負を避ける）
            "pred_log1p": torch.log1p(pred5.clamp_min(0.0)),
        }

        # attention を返すオプション
        if self.return_attention:
            if spatial_attn is not None:
                out["spatial_attn"] = spatial_attn
            if tile_attn is not None:
                out["tile_attn"] = tile_attn

        return out
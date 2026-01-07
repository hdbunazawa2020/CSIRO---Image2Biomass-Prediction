from __future__ import annotations

import torch
import torch.nn as nn
import timm

from typing import Any, Dict, Optional, Union, Literal

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ConvNeXtSplitCropAuxRegressor(nn.Module):
    """
    入力:
      - x: (B, 2, C, H, W)  ※左右crop2枚
        または互換のため (B, C, H, W) も可（その場合 view=1 として扱う）

    処理:
      - 2枚を同一backbone(重み共有)に通す
      - view方向に特徴を融合 (concat / mean / max)
      - main(回帰K) + aux(height, ndvi, species) を出力
    """

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        num_targets: int,          # 5質量
        in_chans: int,
        drop_rate: float,
        drop_path_rate: float,
        head_dropout: float = 0.0,

        # view fusion
        # 左右2枚のcrop画像から得た特徴ベクトルを、どうやって1つにまとめるか（融合するか）
        fuse: Literal["concat", "mean", "max"] = "concat",

        # aux
        aux_cfg: Optional[Any] = None,
        num_species: int = 0,
        aux_hidden_dim: int = 256,
        aux_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.fuse = fuse 

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool="avg",
        )
        feat_dim = int(getattr(self.backbone, "num_features"))

        # --- fusion後の次元 ---
        if fuse == "concat":
            fused_dim = feat_dim * 2
        elif fuse in ("mean", "max"):
            fused_dim = feat_dim
        else:
            raise ValueError(f"Unknown fuse={fuse}. choose from concat/mean/max")

        # main head
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(fused_dim, int(num_targets))

        # aux heads (weight>0のみ)
        self.aux_cfg = aux_cfg
        aux_enabled_flag = bool(_cfg_get(aux_cfg, "enabled", False))
        self.aux_heads = nn.ModuleDict()

        if aux_enabled_flag:
            # Species
            sp_cfg = _cfg_get(aux_cfg, "species", None)
            sp_on = bool(_cfg_get(sp_cfg, "enabled", False)) and float(_cfg_get(sp_cfg, "weight", 0.0)) > 0.0
            if sp_on:
                if int(num_species) <= 0:
                    raise ValueError(
                        "[ConvNeXtSplitCropAuxRegressor] aux.species.enabled=True & weight>0 ですが num_species<=0 です。"
                    )
                self.aux_heads["species_logits"] = nn.Sequential(
                    nn.Linear(fused_dim, aux_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(aux_dropout),
                    nn.Linear(aux_hidden_dim, int(num_species)),
                )

            # NDVI
            ndvi_cfg = _cfg_get(aux_cfg, "ndvi", None)
            ndvi_on = bool(_cfg_get(ndvi_cfg, "enabled", False)) and float(_cfg_get(ndvi_cfg, "weight", 0.0)) > 0.0
            if ndvi_on:
                self.aux_heads["ndvi"] = nn.Sequential(
                    nn.Linear(fused_dim, aux_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(aux_dropout),
                    nn.Linear(aux_hidden_dim, 1),
                )

            # Height
            h_cfg = _cfg_get(aux_cfg, "height", None)
            h_on = bool(_cfg_get(h_cfg, "enabled", False)) and float(_cfg_get(h_cfg, "weight", 0.0)) > 0.0
            if h_on:
                self.aux_heads["height"] = nn.Sequential(
                    nn.Linear(fused_dim, aux_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(aux_dropout),
                    nn.Linear(aux_hidden_dim, 1),
                )

        self.aux_enabled = (len(self.aux_heads) > 0)

    def _fuse_feats(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, V, D) where V=2
        """
        if self.fuse == "concat":
            return torch.cat([feats[:, 0], feats[:, 1]], dim=1)  # (B, 2D)
        if self.fuse == "mean":
            return feats.mean(dim=1)  # (B, D)
        if self.fuse == "max":
            return feats.max(dim=1).values  # (B, D)
        raise RuntimeError("unreachable")

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        x:
          - (B,2,C,H,W) 推奨
          - (B,C,H,W) も可（V=1扱い→mean/maxはそのまま、concatはエラーにするのが安全）
        """
        # 入力チェック & 次元合わせ
        if x.ndim == 4:
            # (B,C,H,W) -> (B,1,C,H,W)
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"x must be 4D or 5D, got shape={tuple(x.shape)}")
        B, V, C, H, W = x.shape
        if V == 1 and self.fuse == "concat":
            raise ValueError("fuse=concat requires V=2 views, but got V=1")

        # ===========================================
        # Backbone forward
        # ===========================================
        x_ = x.reshape(B * V, C, H, W)              # (B*V,C,H,W)
        feat_ = self.backbone(x_)                   # (B*V,D)
        D = feat_.shape[1]                          # D=特徴次元数
        feats = feat_.reshape(B, V, D)              # (B,V,D), 元の形状に戻す

        # ===========================================
        # View fusion (左右crop2枚の特徴ベクトル融合)
        # ===========================================
        fused = self._fuse_feats(feats)             # (B,fused_dim)

        # ===========================================
        # Heads forward
        # ===========================================
        # --- main head ---
        pred_log1p = self.head(self.head_dropout(fused))
        # --- aux heads ---
        # aux無効ならmainのみ返す
        if not self.aux_enabled:
            return pred_log1p
        # aux有効ならauxも返す
        aux_out: Dict[str, torch.Tensor] = {}
        for name, head in self.aux_heads.items():
            aux_out[name] = head(fused)

        return {"pred_log1p": pred_log1p, "aux": aux_out}
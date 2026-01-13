from __future__ import annotations

from typing import Any, Dict, Optional, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


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

    main出力（変更点）:
      - まず 3種類（Green / Clover / Dead）を予測する（rawで非負になるよう Softplus）
      - それを足し算で 5種類に拡張する
          GDM   = Green + Clover
          Total = GDM + Dead
      - 返却は既存互換のため pred_log1p: (B,5) とする
          [Green, Clover, Dead, GDM, Total] の順

    aux出力:
      - aux(height, ndvi, species) を追加で返す（従来通り）
    """

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        num_targets: int,          # 期待は 5（[Green, Clover, Dead, GDM, Total]）
        in_chans: int,
        drop_rate: float,
        drop_path_rate: float,
        head_dropout: float = 0.0,

        # view fusion
        fuse: Literal["concat", "mean", "max"] = "concat",

        # aux
        aux_cfg: Optional[Any] = None,
        num_species: int = 0,
        num_states: int = 0,              # ★追加
        aux_hidden_dim: int = 256,
        aux_dropout: float = 0.1,

        # main head settings
        components_mode: bool = True,   # True: 3出力→5生成（今回の変更）
        eps: float = 1e-6,              # log1p/安定用（ほぼ不要だが念のため）
    ) -> None:
        super().__init__()

        self.fuse = fuse
        self.components_mode = bool(components_mode)
        self.eps = float(eps)

        if self.components_mode:
            if int(num_targets) != 5:
                raise ValueError(
                    f"[ConvNeXtSplitCropAuxRegressor] components_mode=True では num_targets=5 が必須です。"
                    f" got num_targets={num_targets}"
                )

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

        if self.components_mode:
            # ★ 3出力：Green / Clover / Dead
            self.head = nn.Linear(fused_dim, 3)
        else:
            # 従来互換（必要なら）
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
            # ★追加: State
            st_cfg = _cfg_get(aux_cfg, "state", None)
            st_on = bool(_cfg_get(st_cfg, "enabled", False)) and float(_cfg_get(st_cfg, "weight", 0.0)) > 0.0
            if st_on:
                if int(num_states) <= 0:
                    raise ValueError("aux.state enabled but num_states<=0")
                self.aux_heads["state_logits"] = nn.Sequential(
                    nn.Linear(fused_dim, aux_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(aux_dropout),
                    nn.Linear(aux_hidden_dim, int(num_states)),
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

    def _components3_to_pred_log1p5(self, comp_logits: torch.Tensor) -> torch.Tensor:
        """
        comp_logits: (B,3)  -> raw非負 -> 5ターゲット(raw) -> log1pで返す

        出力順:
          [Green, Clover, Dead, GDM, Total]
        """
        # rawの非負化（biomassは負にならない）
        comp_raw = F.softplus(comp_logits)  # (B,3), >=0

        green = comp_raw[:, 0:1]
        clover = comp_raw[:, 1:2]
        dead = comp_raw[:, 2:3]

        gdm = green + clover
        total = gdm + dead

        raw5 = torch.cat([green, clover, dead, gdm, total], dim=1)  # (B,5)

        # 数値保険（理論上softplusで>=0だが、念のため）
        raw5 = torch.clamp(raw5, min=0.0)

        pred_log1p = torch.log1p(raw5 + self.eps)  # (B,5)
        return pred_log1p

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        x:
          - (B,2,C,H,W) 推奨
          - (B,C,H,W) も可（V=1扱い→mean/maxはそのまま、concatはエラー）
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
        x_ = x.reshape(B * V, C, H, W)       # (B*V,C,H,W)
        feat_ = self.backbone(x_)            # (B*V,D)
        D = feat_.shape[1]
        feats = feat_.reshape(B, V, D)       # (B,V,D)

        # ===========================================
        # View fusion
        # ===========================================
        fused = self._fuse_feats(feats)      # (B,fused_dim)

        # ===========================================
        # Heads forward
        # ===========================================
        main_out = self.head(self.head_dropout(fused))

        if self.components_mode:
            # ★ 3出力→5出力へ
            pred_log1p = self._components3_to_pred_log1p5(main_out)
        else:
            pred_log1p = main_out  # 従来互換

        # aux無効ならmainのみ返す
        if not self.aux_enabled:
            return pred_log1p

        # aux有効ならauxも返す
        aux_out: Dict[str, torch.Tensor] = {}
        for name, head in self.aux_heads.items():
            aux_out[name] = head(fused)

        return {"pred_log1p": pred_log1p, "aux": aux_out}
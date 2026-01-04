from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import timm


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """OmegaConf / dict / object から安全に値を取得する。

    Args:
        cfg: 設定オブジェクト
        key: 属性名 / キー名
        default: 見つからない時の返り値

    Returns:
        取得値 or default
    """
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ConvNeXtAuxRegressor(nn.Module):
    """ConvNeXt backbone + 回帰head + (任意) aux head.

    Notes:
        - aux head が 0個なら従来互換で Tensor(B, K) を返す
        - aux head が 1個以上あるなら dict を返す
    """

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        num_targets: int,
        in_chans: int,
        drop_rate: float,
        drop_path_rate: float,
        head_dropout: float = 0.0,
        # --- aux ---
        aux_cfg: Optional[Any] = None,
        num_species: int = 0,
        aux_hidden_dim: int = 256,
        aux_dropout: float = 0.1,
    ) -> None:
        super().__init__()

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
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(feat_dim, int(num_targets))

        # ----------------------------
        # aux の有効判定（weight>0 のものだけ head を作る）
        # ----------------------------
        self.aux_cfg = aux_cfg
        aux_enabled_flag = bool(_cfg_get(aux_cfg, "enabled", False))

        self.aux_heads = nn.ModuleDict()

        if aux_enabled_flag:
            # ---- Species ----
            sp_cfg = _cfg_get(aux_cfg, "species", None)
            sp_on = bool(_cfg_get(sp_cfg, "enabled", False)) and float(_cfg_get(sp_cfg, "weight", 0.0)) > 0.0
            if sp_on:
                if int(num_species) <= 0:
                    raise ValueError(
                        "[ConvNeXtAuxRegressor] aux.species.enabled=True & weight>0 ですが num_species<=0 です。\n"
                        "→ 100_train_model.py 側で df から species_to_index を作り、num_species を渡してください。"
                    )
                self.aux_heads["species_logits"] = nn.Sequential(
                    nn.Linear(feat_dim, aux_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(aux_dropout),
                    nn.Linear(aux_hidden_dim, int(num_species)),
                )

            # ---- NDVI ----
            ndvi_cfg = _cfg_get(aux_cfg, "ndvi", None)
            ndvi_on = bool(_cfg_get(ndvi_cfg, "enabled", False)) and float(_cfg_get(ndvi_cfg, "weight", 0.0)) > 0.0
            if ndvi_on:
                self.aux_heads["ndvi"] = nn.Sequential(
                    nn.Linear(feat_dim, aux_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(aux_dropout),
                    nn.Linear(aux_hidden_dim, 1),
                )

            # ---- Height ----
            h_cfg = _cfg_get(aux_cfg, "height", None)
            h_on = bool(_cfg_get(h_cfg, "enabled", False)) and float(_cfg_get(h_cfg, "weight", 0.0)) > 0.0
            if h_on:
                self.aux_heads["height"] = nn.Sequential(
                    nn.Linear(feat_dim, aux_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(aux_dropout),
                    nn.Linear(aux_hidden_dim, 1),
                )

        # ★実際に head があるときだけ aux を有効化
        self.aux_enabled = (len(self.aux_heads) > 0)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Forward.

        Args:
            x: 画像テンソル (B,C,H,W)

        Returns:
            aux無し: Tensor(B, K)
            aux有り: {"pred_log1p": Tensor(B,K), "aux": {...}}
        """
        feat = self.backbone(x)              # (B, feat_dim)
        pred_log1p = self.head(self.head_dropout(feat))  # main は head_dropout を掛ける

        if not self.aux_enabled:
            return pred_log1p

        # aux は feat そのまま（各 head 内の aux_dropout に任せる）
        aux_out: Dict[str, torch.Tensor] = {}
        for name, head in self.aux_heads.items():
            aux_out[name] = head(feat)

        return {"pred_log1p": pred_log1p, "aux": aux_out}
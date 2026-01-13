import torch
import torch.nn as nn
from transformers import AutoModel

class DINOv2TwoStreamRegressor(nn.Module):
    def __init__(
        self, 
        n_targets=3, 
        model_path="facebook/dinov2-large",
        freeze_backbone=True, 
        hidden_dim=512, 
        dropout=0.1
        ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_path)
        embed_dim = self.backbone.config.hidden_size

        pooled_dim = embed_dim * 3  # CLS + avg + max

        self.regressor = nn.Sequential(
            nn.Linear(pooled_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets)
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _pool_tokens(self, outputs):
        last_hidden = outputs.last_hidden_state # (B, N, D), 最終層の隠れ状態
        cls = last_hidden[:, 0]       # (B, D), CLSトークン 
        patches = last_hidden[:, 1:]  # (B, N-1, D), パッチトークン

        avg_pool = patches.mean(dim=1)   # (B, D), 平均プーリング
        max_pool = patches.max(dim=1)[0] # (B, D), 最大プーリング

        return torch.cat([cls, avg_pool, max_pool], dim=1) # (B, D*3)

    def forward(self, x_left, x_right):
        # 左右それぞれの画像をバックボーンに通す. (B, 3, H, W) -> (B, D)
        out_l = self.backbone(pixel_values=x_left)
        out_r = self.backbone(pixel_values=x_right)
        # それぞれの特徴量をプーリングして結合
        emb_l = self._pool_tokens(out_l)
        emb_r = self._pool_tokens(out_r)

        combined = torch.cat([emb_l, emb_r], dim=1)
        return self.regressor(combined)
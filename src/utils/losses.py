import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class WeightedMSELoss(nn.Module):
    def __init__(self, weights: List[float]) -> None:
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("w", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B, K)
        loss = (pred - target) ** 2
        loss = loss * self.w[None, :]
        return loss.mean()

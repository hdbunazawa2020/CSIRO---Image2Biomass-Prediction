# -*- coding: utf-8 -*-
"""
サンプリング関連ユーティリティ。

このファイルでは「Dry_Total_g が大きいサンプルを学習で多めに見せる」ための
WeightedRandomSampler（およびDDP用の簡易分散版）を提供します。

主な使い方（単一GPU/非DDP）:
    weights = make_total_oversample_weights(trn_df, total_col="Dry_Total_g", n_bins=8)
    sampler = build_weighted_sampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=...,
        sampler=sampler,
        shuffle=False,  # samplerを使うときはshuffle=False
        ...
    )
"""

from __future__ import annotations

import math
from typing import Optional, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

try:
    import torch.distributed as dist
except Exception:
    dist = None


def make_total_oversample_weights(
    df: pd.DataFrame,
    *,
    total_col: str = "Dry_Total_g",
    n_bins: int = 8,
    strategy: Literal["ramp", "inv_freq"] = "ramp",
    min_mult: float = 1.0,
    max_mult: float = 5.0,
    inv_freq_power: float = 1.0,
    clip_min: float = 0.1,
    clip_max: float = 20.0,
) -> torch.DoubleTensor:
    """Dry_Total_g を元に oversample 用の sample weight を作る。

    方針:
      - qcut（分位）で bin を作る → bin id を使って重みを作る
      - "ramp": binが上がるほど weight を線形に増やす（穏やかでおすすめ）
      - "inv_freq": bin頻度の逆数（希少binを強く持ち上げる。やや攻め）

    Args:
        df: 学習用DataFrame（rawの Dry_Total_g 列がある前提）
        total_col: 参照する総量列名（通常 "Dry_Total_g"）
        n_bins: 分位bin数（多いほど細かく制御できるが不安定にもなる）
        strategy: "ramp" or "inv_freq"
        min_mult: 最小倍率（rampの最低値）
        max_mult: 最大倍率（rampの最大値、inv_freqでも最終clipに使う）
        inv_freq_power: inv_freqの強さ（1.0=そのまま、>1.0でより強く）
        clip_min/clip_max: 異常に大きい/小さい重みのクリップ

    Returns:
        weights: shape (N,) の torch.double Tensor（WeightedRandomSamplerに渡す用）
    """
    if total_col not in df.columns:
        raise KeyError(f"total_col='{total_col}' not in df.columns: {list(df.columns)}")

    totals = df[total_col].astype(float).to_numpy()
    if len(totals) == 0:
        raise ValueError("df is empty.")

    # qcutで分位bin（値が重複してbinが作れない場合はduplicates='drop'で落とす）
    bin_id = pd.qcut(totals, q=int(n_bins), labels=False, duplicates="drop")
    bin_id = np.asarray(bin_id, dtype=np.int64)
    n_bins_eff = int(bin_id.max()) + 1 if bin_id.size > 0 else 1
    denom = max(1, n_bins_eff - 1)

    if strategy == "ramp":
        # binが高いほど大きく（穏やか）
        mult = min_mult + (bin_id / float(denom)) * (max_mult - min_mult)

    elif strategy == "inv_freq":
        # bin頻度の逆数（希少binを強く持ち上げる）
        counts = np.bincount(bin_id, minlength=n_bins_eff).astype(np.float64)
        inv = 1.0 / np.maximum(counts, 1.0)
        inv = inv ** float(inv_freq_power)
        mult = inv[bin_id]

        # 平均が1付近になるよう正規化（学習率の見え方を安定させる）
        mult = mult / max(1e-12, float(mult.mean()))
        # その上でmax_multで上限を持つ（過激になり過ぎないように）
        mult = np.clip(mult, clip_min, max_mult)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    mult = np.clip(mult, clip_min, clip_max)
    weights = torch.as_tensor(mult, dtype=torch.double)
    return weights


def build_weighted_sampler(
    weights: torch.DoubleTensor,
    *,
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """WeightedRandomSampler を作る。

    Args:
        weights: shape (N,) の重み
        num_samples: 1epochで引くサンプル数。Noneなら N（=len(weights)）
        replacement: True推奨（oversample目的なら基本True）

    Returns:
        sampler: WeightedRandomSampler
    """
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape={tuple(weights.shape)}")

    n = int(weights.numel())
    if num_samples is None:
        num_samples = n

    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples),
        replacement=bool(replacement),
    )


class DistributedWeightedSampler(Sampler[int]):
    """DDP環境でも使える簡易 Weighted Sampler。

    注意:
        - torch.distributed が初期化されている前提（dist.is_initialized()）
        - 各epochで sampler.set_epoch(epoch) を呼ぶのが推奨（順序を変えるため）

    使い方:
        weights = make_total_oversample_weights(trn_df, ...)
        sampler = DistributedWeightedSampler(weights, num_samples=len(weights), replacement=True, seed=cfg.seed)

        train_loader = DataLoader(..., sampler=sampler, shuffle=False)

        for epoch in ...:
            sampler.set_epoch(epoch)
            ...
    """

    def __init__(
        self,
        weights: torch.DoubleTensor,
        *,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if dist is None or (not dist.is_available()) or (not dist.is_initialized()):
            raise RuntimeError("DistributedWeightedSampler requires torch.distributed initialized.")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

        n = int(self.weights.numel())
        if num_samples is None:
            num_samples = n

        # 全体で引くサンプル数（epochあたり）
        self.total_num_samples = int(num_samples)

        # 各rankが受け取るサンプル数
        if self.drop_last:
            self.num_samples_per_replica = self.total_num_samples // self.num_replicas
            self.total_size = self.num_samples_per_replica * self.num_replicas
        else:
            self.num_samples_per_replica = int(math.ceil(self.total_num_samples / self.num_replicas))
            self.total_size = self.num_samples_per_replica * self.num_replicas

        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """epochをセットして乱数系列を変える。"""
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # multinomialで全体分サンプル → rankごとに間引く
        indices = torch.multinomial(
            self.weights,
            num_samples=int(self.total_size),
            replacement=self.replacement,
            generator=g,
        ).tolist()

        # rankに対応する部分を取り出す
        indices_rank = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices_rank)

    def __len__(self) -> int:
        return int(self.num_samples_per_replica)
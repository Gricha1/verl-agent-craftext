"""Batch sampler with equal samples per reward horizon (h=1..H) in each batch."""
from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler


def align_horizon_batch_size(
    batch_size: int,
    num_horizons: int,
    micro_batch_size: Optional[int] = None,
) -> int:
    """Round down so batch divides evenly by horizon count and micro-batch size."""
    n = max(1, int(num_horizons))
    units = n
    if micro_batch_size is not None:
        m = max(1, int(micro_batch_size))
        units = math.lcm(n, m)
    aligned = (int(batch_size) // units) * units
    if aligned <= 0:
        raise ValueError(
            f"batch_size={batch_size} too small for num_horizons={n}"
            + (f" and micro_batch_size={micro_batch_size}" if micro_batch_size else "")
        )
    return aligned


class HorizonBalancedBatchSampler(Sampler[List[int]]):
    """
    Each batch contains ``batch_size // num_horizons`` samples from every horizon.

    Requires ``dataset.horizons`` — list of int horizon labels aligned with dataset indices.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_horizons: int,
        *,
        micro_batch_size: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1,
        drop_last: bool = True,
    ):
        horizons: Sequence[int] = getattr(dataset, "horizons", None)
        if horizons is None:
            raise ValueError("HorizonBalancedBatchSampler requires dataset.horizons")

        self.num_horizons = max(1, int(num_horizons))
        self.requested_batch_size = int(batch_size)
        aligned = align_horizon_batch_size(
            self.requested_batch_size, self.num_horizons, micro_batch_size
        )
        self.per_horizon = aligned // self.num_horizons
        self.batch_size = self.per_horizon * self.num_horizons

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        by_h: Dict[int, List[int]] = {h: [] for h in range(1, self.num_horizons + 1)}
        for idx, h in enumerate(horizons):
            hh = int(h)
            if 1 <= hh <= self.num_horizons:
                by_h[hh].append(int(idx))
        for h in range(1, self.num_horizons + 1):
            if not by_h[h]:
                raise ValueError(
                    f"No dataset indices with horizon={h}. "
                    f"Recollect parquet with REWARD_HORIZON={self.num_horizons} "
                    f"and train with the same REWARD_HORIZON."
                )
        self._by_h = by_h
        min_pool = min(len(by_h[h]) for h in range(1, self.num_horizons + 1))
        self._num_full_batches = min_pool // self.per_horizon if min_pool >= self.per_horizon else 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self._num_full_batches <= 0:
            return 0
        return math.ceil(self._num_full_batches / self.world_size)

    def __iter__(self) -> Iterator[List[int]]:
        if self._num_full_batches <= 0:
            return iter(())

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        pools = {}
        for h in range(1, self.num_horizons + 1):
            idxs = list(self._by_h[h])
            perm = torch.randperm(len(idxs), generator=g).tolist()
            pools[h] = [idxs[i] for i in perm]

        all_batches: List[List[int]] = []
        for b in range(self._num_full_batches):
            batch: List[int] = []
            for h in range(1, self.num_horizons + 1):
                start = b * self.per_horizon
                batch.extend(pools[h][start : start + self.per_horizon])
            # Shuffle within batch so horizons are mixed in order.
            batch_perm = torch.randperm(len(batch), generator=g).tolist()
            all_batches.append([batch[i] for i in batch_perm])

        # Shard batches across DP ranks.
        rank_batches = all_batches[self.rank :: self.world_size]
        return iter(rank_batches)

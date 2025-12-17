# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn

from wt_asbs.utils.composition import PeriodicTable


class AtomEmbedding(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_elements: int | None = None,
        periodic_table: PeriodicTable | None = None,
    ):
        if not ((num_elements is None) ^ (periodic_table is None)):
            raise ValueError(
                "Exactly one of num_elements or periodic_table must be provided."
            )
        super().__init__()
        self.num_features = num_features
        num_elements = num_elements or len(periodic_table)
        self.embedding = nn.Embedding(num_elements, num_features)
        self.periodic_table = periodic_table
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embedding.weight, a=-math.sqrt(3), b=math.sqrt(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.periodic_table is not None:
            x = self.periodic_table(x)
        return self.embedding(x)


class TimeEmbedding(nn.Module):
    def __init__(self, num_basis: int):
        super().__init__()
        assert num_basis % 2 == 0
        self.num_basis = num_basis
        freqs = torch.randn(num_basis // 2) * 2 * math.pi
        self.register_buffer("freqs", freqs)
        self.sqrt_2 = math.sqrt(2)

    def forward(self, x: torch.Tensor):
        args = self.freqs * x[..., None]
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        return emb * self.sqrt_2

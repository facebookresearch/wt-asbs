# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class ScaledSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard deviation for SiLU(Normal) is 0.6, so we scale the output by 1/0.6
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale_factor * self._activation(x)

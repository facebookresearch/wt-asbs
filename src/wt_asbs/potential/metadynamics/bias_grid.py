# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BiasGrid(nn.Module):
    def __init__(
        self,
        grid_min: list[float],
        grid_max: list[float],
        grid_bin: list[int],
        periodic: bool = False,
    ):
        super().__init__()
        if not (len(grid_min) == len(grid_max) == len(grid_bin)):
            raise ValueError("All grid parameters must have the same length.")

        # Create grid
        grid_edges = [
            torch.linspace(min_val, max_val, bins, dtype=torch.float)
            for min_val, max_val, bins in zip(grid_min, grid_max, grid_bin)
        ]
        grid_spacing = [
            (max_val - min_val) / (bins - 1)
            for min_val, max_val, bins in zip(grid_min, grid_max, grid_bin)
        ]
        meshgrid = torch.stack(
            torch.meshgrid(*grid_edges, indexing="ij"), dim=0
        ).unsqueeze(0)
        self.register_buffer("meshgrid", meshgrid)  # [1, D, n1, n2, ..., nD]
        self.register_buffer("bias_values", torch.zeros(*grid_bin, dtype=torch.float))
        self.register_buffer(
            "grid_min", torch.as_tensor(grid_min, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "grid_max", torch.as_tensor(grid_max, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "grid_spacing",
            torch.as_tensor(grid_spacing, dtype=torch.float),
            persistent=False,
        )

        # Precompute helper tensors
        self.register_buffer(
            "strides", torch.as_tensor(self.bias_values.stride(), dtype=torch.long)
        )
        dim = len(grid_min)
        corners = torch.stack(torch.meshgrid(*[[torch.tensor([0, 1])] * dim]), dim=-1)
        corners = corners.reshape(-1, dim).unsqueeze(1)  # (2^dim, 1, dim)
        self.register_buffer("corners", corners)
        self.periodic = periodic

    def forward(self, cvs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the bias potential for given CVs.
        Args:
            cvs (torch.Tensor): Input tensor of shape (batch_size, n_cv)
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - bias (torch.Tensor): Bias potential for each CV (batch_size,)
                - grad (torch.Tensor): Gradient of the bias potential (batch_size, n_cv)
        """
        # CV range check
        if (cvs < self.grid_min).any() or (cvs > self.grid_max).any():
            # raise ValueError("CV values must be within the grid range.")
            logger.warning(
                "CV values are outside the grid range. Clamping to grid limits."
            )
            # Clamp to grid limits to avoid errors
            cvs = cvs.clamp(min=self.grid_min, max=self.grid_max)

        # Compute grid coordinates
        t = (cvs - self.grid_min) / self.grid_spacing

        # Compute base index
        i0 = t.floor().long()

        # Clamp indices to ensure that i0 and i0 + 1 are within bounds
        grid_dims = torch.tensor(
            self.bias_values.shape, device=cvs.device, dtype=torch.long
        )
        i0 = i0.clamp(min=torch.zeros_like(grid_dims), max=grid_dims - 2)
        i0 = i0.unsqueeze(0)
        f = t - i0.float()  # [1, batch, dim]
        w_dim = self.corners * f + (1 - self.corners) * (1 - f)
        w = torch.prod(w_dim, dim=-1)  # [2^dim, batch]

        # Flatten indices and take values
        idx_flat = ((i0 + self.corners) * self.strides).sum(dim=-1)  # [2^dim, batch]
        vals = self.bias_values.view(-1).take(idx_flat)  # [2^dim, batch]

        # Compute bias and gradient
        bias = (w * vals).sum(dim=0)  # [batch,]
        sign = 2 * self.corners - 1
        dw_df = w.unsqueeze(-1) * sign / w_dim
        grad = (vals.unsqueeze(-1) * dw_df).sum(dim=0) / self.grid_spacing
        return bias, grad

    def add_hills(self, cvs: torch.Tensor, heights: torch.Tensor, widths: torch.Tensor):
        """
        Add hills to the bias grid.
        Args:
            cvs (torch.Tensor): CVs where hills are added [batch, n_cv]
            heights (torch.Tensor): Heights of the hills [batch,]
            widths (torch.Tensor): Widths of the hills [batch, n_cv]
        """
        if cvs.shape[1] != len(self.grid_min):
            raise ValueError("CVs must match the grid dimensions.")
        dim = self.bias_values.dim()
        cvs = cvs.view(-1, dim, *([1] * dim))  # [batch, dim, 1, ..., 1]
        widths = widths.view(-1, dim, *([1] * dim))
        heights = heights.view(-1, *([1] * dim))

        # Compute the Gaussian hills
        diff = cvs - self.meshgrid
        if self.periodic:  # Minimal image convention
            grid_range = (self.grid_max - self.grid_min).view(-1, dim, *([1] * dim))
            diff -= torch.round(diff / grid_range) * grid_range
        gauss = heights * torch.exp(-0.5 * (diff / widths).pow(2).sum(dim=1))

        # Add to the bias grid
        self.bias_values += gauss.sum(dim=0)

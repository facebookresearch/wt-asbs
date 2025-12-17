# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ase.units as units
import torch

from wt_asbs.data.atomic_data import ThermoAtomicData
from wt_asbs.potential.base import BasePotential
from wt_asbs.potential.metadynamics.bias_grid import BiasGrid
from wt_asbs.potential.metadynamics.collective_variable import BaseCV


class WellTemperedMetadynamicsBias(BasePotential):
    def __init__(
        self,
        cv: BaseCV,
        height: float,
        sigma: list[float],
        grid_min: list[float],
        grid_max: list[float],
        grid_bin: list[int],
        bias_factor: float = 10.0,
        temperature: float = 300.0,
        add_every_epoch: int = 1,
        skip_initial_epochs: int = 0,
    ):
        """
        Initialize the Well-Tempered Metadynamics bias potential.
        Args:
            cv (BaseCV): Collective variable instance.
            height (float): Height of the Gaussian hills in eV.
            sigma (list[float]): Widths of the Gaussian hills for each CV in Angstrom.
            grid_min (list[float]): Minimum values for the bias grid.
            grid_max (list[float]): Maximum values for the bias grid.
            grid_bin (list[int]): Number of bins for each dimension of the bias grid.
            bias_factor (float): Bias factor for well-tempered metadynamics.
            temperature (float): Simulation temperature in Kelvin.
            add_every_epoch (int): Frequency of adding hills (every N epochs).
            skip_initial_epochs (int): Number of initial epochs to skip.
        """
        if isinstance(cv, list):
            raise NotImplementedError("Multiple CV instances are not supported yet.")
        super().__init__()
        self.cv = cv
        if not (
            len(sigma) == len(grid_min) == len(grid_max) == len(grid_bin) == cv.dim
        ):
            raise ValueError("All grid parameters must have the same length.")

        self.bias_grid = BiasGrid(
            grid_min=grid_min,
            grid_max=grid_max,
            grid_bin=grid_bin,
            periodic=cv.periodic,
        )
        self.register_buffer(
            "height", torch.as_tensor(height, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "sigma", torch.as_tensor(sigma, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "bias_factor",
            torch.as_tensor(bias_factor, dtype=torch.float),
            persistent=False,
        )
        delta_temperature = temperature * (bias_factor - 1)
        self.register_buffer(
            "delta_temperature",
            torch.as_tensor(delta_temperature, dtype=torch.float),
            persistent=False,
        )

        # Pace and step counter (saved in buffer for checkpointing)
        self.add_every_epoch = add_every_epoch
        self.skip_initial_epochs = skip_initial_epochs
        self.register_buffer("epoch", torch.tensor(0, dtype=torch.int))

    def forward(self, data: ThermoAtomicData) -> dict[str, torch.Tensor]:
        """
        Compute the bias potential and forces for the input data.
        Args:
            data (ThermoAtomicData): Input data containing positions.
        Returns:
            dict[str, torch.Tensor]: Dictionary containing energy and forces.
            - "energy": Bias energy for each system in the batch.
            - "forces": Forces acting on each atom in the batch.
        """
        # NOTE: Currently we assume that metadynamics is only applied to a setting where
        # all systems in a batch are the same (to define the CV).
        if self.epoch < self.skip_initial_epochs:
            return {
                "energy": torch.zeros(data.num_graphs).to(data.pos),
                "forces": torch.zeros_like(data.pos),
            }
        x = data.pos.reshape(data.num_graphs, -1, 3)
        cvs = self.cv(x)  # [batch, n_cv]
        bias, grad = self.bias_grid(cvs)  # [batch,], [batch, n_cv]
        forces = -self.cv.vjp(x, grad)  # -∂U/∂x = -∂U/∂s * ∂s/∂x [batch, n_atoms, 3]
        return {
            "energy": bias,
            "forces": forces.reshape(-1, 3),  # Flatten to [batch * n_atoms, 3]
        }

    def compute_cv(self, data: ThermoAtomicData) -> torch.Tensor:
        """
        Compute the collective variable (CV) for the input data.
        Args:
            data (ThermoAtomicData): Input data containing positions.
        Returns:
            torch.Tensor: Computed CV values.
        """
        x = data.pos.reshape(data.num_graphs, -1, 3)
        return self.cv(x)  # [batch, n_cv]

    def add_hills(self, cvs: torch.Tensor):
        """
        Add hills to the bias grid based on the current CVs.
        Args:
            cvs: torch.Tensor: Collective variable values for the current batch, with
                shape [batch, n_cv].
        """
        self.epoch += 1
        if self.epoch <= self.skip_initial_epochs:
            return
        if (self.epoch - self.skip_initial_epochs) % self.add_every_epoch != 0:
            return  # Only add hills every `add_every_epoch` steps
        bias, _ = self.bias_grid(cvs)  # [batch,]
        heights = self.height * torch.exp(-bias / (units.kB * self.delta_temperature))
        widths = self.sigma * torch.ones_like(cvs)  # [batch, n_cv]
        self.bias_grid.add_hills(cvs, heights, widths)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from wt_asbs.data.atomic_data import ThermoAtomicData


class BasePotential(nn.Module, ABC):
    """Abstract base class for atomic potentials."""

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data: ThermoAtomicData) -> dict[str, torch.Tensor]:
        """Compute the potential energy and forces for a batch of atomic structures.

        Args:
            data (ThermoAtomicData): Input atomic data.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing computed physical properties,
            such as energy, forces, and stress.
        """
        pass


class SumPotential(BasePotential):
    """A potential that sums multiple base potentials."""

    def __init__(self, potentials: list[BasePotential]):
        super().__init__()
        self.potentials = nn.ModuleList(potentials)

    def forward(self, data: ThermoAtomicData) -> dict[str, torch.Tensor]:
        """Compute the sum of energies and forces from multiple potentials."""
        results = [potential(data) for potential in self.potentials]
        # Get common keys across all results
        keys = set.intersection(*(set(result.keys()) for result in results))
        return {key: sum(result[key] for result in results) for key in keys}


class SwitchMixturePotential(BasePotential):
    """A potential that combines multiple potentials with a switching function."""

    def __init__(self, switch_potential: BasePotential, potential: BasePotential):
        super().__init__()
        self.switch_potential = switch_potential
        self.potential = potential

    def forward(self, data: ThermoAtomicData) -> dict[str, torch.Tensor]:
        """Compute the potential energy and forces with a switching function."""
        switch_results = self.switch_potential(data, return_switch=True)
        potential_results = self.potential(data)

        # Get switch value
        try:
            switch = switch_results["switch"]
        except KeyError:
            raise ValueError("Switch results must contain 'switch' key.")

        # Combine results
        results = {}
        if "energy" in switch_results and "energy" in potential_results:
            results["energy"] = (
                switch * switch_results["energy"]
                + (1 - switch) * potential_results["energy"]
            )
        if "forces" in switch_results and "forces" in potential_results:
            switch_forces = switch[data.batch, None]
            results["forces"] = (
                switch_forces * switch_results["forces"]
                + (1 - switch_forces) * potential_results["forces"]
            )
        results["switch"] = switch
        return results

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import ase.units as units
import torch
import torch.nn as nn

from wt_asbs.data.atomic_data import ThermoAtomicData
from wt_asbs.potential.base import BasePotential
from wt_asbs.process.sde import BaseSDE


class BaseTerminalCost(nn.Module, ABC):
    """Base class for terminal cost functions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    @torch.no_grad()
    def forward(self, data_1: ThermoAtomicData) -> torch.Tensor:
        """Compute the terminal cost for a given atomic data instance (at time 1)."""
        pass


class PotentialGradientMixin:
    """Mixin class to provide potential gradient functionality."""

    def __init__(self, potential: BasePotential, grad_clip_val: float | None = None):
        self.potential = potential
        self.potential.eval()
        self.grad_clip_val = grad_clip_val

    @torch.no_grad()
    def potential_grad(
        self,
        data: ThermoAtomicData,
        eps: float = 1e-10,
        return_grad_norm: float = False,
    ) -> dict[str, torch.Tensor]:
        """Compute the gradient of the potential energy (negative forces) divided by
        the temperature, and return the original potential energy.
        NOTE: The gradients are ∇U/k_B T in units of Angstrom^-1, and the potential
        energy is in eV."""
        pot_result = self.potential(data)
        grad = -pot_result["forces"] / (units.kB * data.temperature[data.batch, None])
        grad_norm = torch.sqrt(grad.pow(2).sum(dim=-1, keepdim=True) + eps)
        if self.grad_clip_val is not None:
            clip_coefficient = torch.clamp(self.grad_clip_val / grad_norm, max=1.0)
        else:
            clip_coefficient = torch.ones_like(grad)
        result = {
            "energy": pot_result["energy"],
            "grad": grad * clip_coefficient,
        }
        if return_grad_norm:
            result["grad_norm"] = grad_norm.squeeze(-1)
        return result


class ASTerminalCost(BaseTerminalCost, PotentialGradientMixin):
    """Terminal cost function for Adjoint Sampling."""

    def __init__(
        self,
        base_sde: BaseSDE,
        potential: BasePotential,
        grad_clip_val: float | None = None,
    ):
        raise NotImplementedError("ASTerminalCost is not implemented yet.")
        BaseTerminalCost.__init__(self)
        PotentialGradientMixin.__init__(self, potential, grad_clip_val)
        self.base_sde = base_sde

    @torch.no_grad()
    def forward(
        self,
        data_1: ThermoAtomicData,
        return_energy: bool = False,
        return_grad_norm: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        raise NotImplementedError("ASTerminalCost is not implemented yet.")
        potential_grad, energy = self.potential_grad(data_1)
        time_1 = torch.ones(data_1.num_graphs).to(data_1.pos)
        score = self.base_sde.marginal_score(time_1, data_1)
        terminal_cost = potential_grad + score
        return (terminal_cost, energy) if return_energy else terminal_cost


class ASBSTerminalCost(BaseTerminalCost, PotentialGradientMixin):
    """Terminal cost function for ASBS."""

    def __init__(
        self,
        potential: BasePotential,
        corrector: nn.Module,
        grad_clip_val: float | None = None,
    ):
        BaseTerminalCost.__init__(self)
        PotentialGradientMixin.__init__(self, potential, grad_clip_val)
        self.corrector = corrector

    @torch.no_grad()
    def forward(
        self,
        data_1: ThermoAtomicData,
        is_init_stage: bool = False,
        return_energy: bool = False,
        return_grad_norm: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute the terminal cost for ASBS."""
        grad_result = self.potential_grad(data_1, return_grad_norm=return_grad_norm)
        time_1 = torch.ones(data_1.num_graphs).to(data_1.pos)
        if is_init_stage:  # for initial stage, corrector is not trained yet
            result = {"terminal_cost": grad_result["grad"]}
        else:
            corrector = self.corrector(time_1, data_1)
            result = {"terminal_cost": grad_result["grad"] + corrector}
        if return_energy:
            result["energy"] = grad_result["energy"]
        if return_grad_norm:
            result["grad_norm"] = grad_result["grad_norm"]
        return result

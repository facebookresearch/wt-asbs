# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

import torch

from wt_asbs.data.atomic_data import ThermoAtomicData
from wt_asbs.potential.base import BasePotential
from wt_asbs.potential.metadynamics.collective_variable import BaseCV, TorsionCV


class HarmonicRestraint(BasePotential):
    def __init__(
        self,
        cv: BaseCV,
        location: list[float],
        force_constant: list[float],
        direction: Literal["upper", "lower", "both"] = "upper",
    ):
        """
        Harmonic restraint U(x) = 0.5 * kappa * (x - x0)^2.
        Args:
            cv (BaseCV): Collective variable instance.
            location (list[float]): Location of the walls for each CV (Å).
            force_constant (list[float]): Force constant (kappa) for each CV (eV/Å^2).
            direction (Literal["upper", "lower", "both"]): Direction of the restraint.
                - "upper": Restraint only in the upper direction (x >= x0).
                - "lower": Restraint only in the lower direction (x <= x0).
                - "both": Restraint in both directions.

        """
        super().__init__()
        if not (len(location) == len(force_constant) == cv.dim):
            raise ValueError(
                "Location and force_constant must have the same length as cv dim."
            )
        self.cv = cv
        self.register_buffer(
            "location", torch.as_tensor(location, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "force_constant",
            torch.as_tensor(force_constant, dtype=torch.float),
            persistent=False,
        )
        if direction == "upper":
            self.clamp_args = {"min": 0, "max": None}
        elif direction == "lower":
            self.clamp_args = {"min": None, "max": 0}
        elif direction == "both":
            self.clamp_args = {"min": None, "max": None}
        else:
            raise ValueError("Direction must be one of 'upper', 'lower', or 'both'.")

    def forward(self, data: ThermoAtomicData) -> dict[str, torch.Tensor]:
        x = data.pos.reshape(data.num_graphs, -1, 3)
        cvs = self.cv(x)  # [batch, n_cv]
        disp = (cvs - self.location).clamp(**self.clamp_args)
        energy = torch.sum(0.5 * self.force_constant * disp.pow(2), dim=-1)  # [batch,]
        forces = -self.cv.vjp(x, self.force_constant * disp)  # [batch, n_atoms, 3]
        return {"energy": energy, "forces": forces.reshape(-1, 3)}


class ChiralityRestraint(BasePotential):
    def __init__(
        self,
        indices: list[list[int]],
        location: float = -0.6154797086703873,  # -35 degrees
        force_constant: float = 25.0,  # eV/radian^2
        tolerance: float = 0.4363323129985824,  # 25 degrees
    ):
        """
        Chirality restraint for a set of dihedral angles.
        Args:
            indices (list[list[int]]): List of atom indices defining the dihedral angles.
            location (float): Target value for the dihedral angle in radians.
            force_constant (float): Force constant for the restraint in eV/rad^2.
            tolerance (float): Tolerance for the dihedral angle in radians.
        """
        super().__init__()
        self.cv = TorsionCV(indices)
        self.location = location
        self.force_constant = force_constant
        self.tolerance = tolerance

    def forward(self, data: ThermoAtomicData) -> dict[str, torch.Tensor]:
        x = data.pos.reshape(data.num_graphs, -1, 3)
        cvs = self.cv(x)  # [batch, n_cv]
        disp = torch.atan2(
            torch.sin(cvs - self.location), torch.cos(cvs - self.location)
        )
        excess = torch.clamp(disp.abs() - self.tolerance, min=0.0)
        energy = torch.sum(0.5 * self.force_constant * excess.pow(2), dim=-1)
        forces = -self.cv.vjp(x, self.force_constant * excess * disp.sign())
        return {"energy": energy, "forces": forces.reshape(-1, 3)}

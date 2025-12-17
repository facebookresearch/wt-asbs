# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

import numpy as np
import openmm
import openmm.unit as unit
import torch
from ase.calculators.calculator import Calculator, all_changes
from openmmtools.testsystems import TestSystem

from wt_asbs.data.atomic_data import ThermoAtomicData
from wt_asbs.potential.base import BasePotential

KJ_MOL_TO_EV = 0.010364269574711572
KJ_MOL_NM_TO_EV_A = 0.001036426957471157
A_TO_NM = 0.1


class OpenMMPotential(BasePotential):
    def __init__(self, system: TestSystem, device: Literal["cuda", "cpu"] = "cuda"):
        super().__init__()
        self.system = system
        if device.lower() == "cuda":
            platform = openmm.Platform.getPlatformByName("CUDA")
            # NOTE: If we call this function in torchrun/fabric, it will set the
            # CUDA_VISIBLE_DEVICES environment variable, so we need to put "0" here.
            properties = {
                "DeviceIndex": "0",
                "Precision": "mixed",
                "UseCpuPme": "false",
            }
        elif device.lower() == "cpu":
            platform = openmm.Platform.getPlatformByName("CPU")
            properties = {}
        else:
            raise ValueError(f"Unsupported device: {device}. Use 'cuda' or 'cpu'.")
        # NOTE: Temperature etc. are not used in energy calculation, but we need to
        # create an integrator to create a Simulation object.
        integrator = openmm.LangevinIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.femtosecond
        )
        self.simulation = openmm.app.Simulation(
            topology=system.topology,
            system=system.system,
            integrator=integrator,
            platform=platform,
            platformProperties=properties,
        )

    def forward(self, data: ThermoAtomicData) -> dict[str, torch.Tensor]:
        if data.pos.isnan().any():
            raise ValueError("Input positions contain NaN values.")
        all_energies = []
        all_forces = []
        for _data in data.batch_to_atomicdata_list():
            pos = _data.pos.cpu().numpy() * A_TO_NM
            self.simulation.context.setPositions(pos)
            state = self.simulation.context.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            all_energies.append(energy * KJ_MOL_TO_EV)
            all_forces.append(forces * KJ_MOL_NM_TO_EV_A)
        return {
            "energy": torch.as_tensor(all_energies).to(data.pos),
            "forces": torch.as_tensor(np.concatenate(all_forces, axis=0)).to(data.pos),
        }


class OpenMMCalculator(Calculator):
    def __init__(
        self,
        system: TestSystem,
        device: Literal["cuda", "cpu"] = "cuda",
    ):
        super().__init__()
        self.results = {}
        self.implemented_properties = ["energy", "forces"]
        self.energy_model = OpenMMPotential(system=system, device=device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        data = ThermoAtomicData.from_ase(atoms)
        output_dict = self.energy_model(data)
        self.results["energy"] = output_dict["energy"][0].item()
        self.results["forces"] = output_dict["forces"].cpu().numpy()

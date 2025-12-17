# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import ase.io
import torch
import torch.nn as nn

from wt_asbs.data.atomic_data import ThermoAtomicData
from wt_asbs.utils.composition import composition_to_atomic_numbers
from wt_asbs.utils.geometry import is_mean_free_data
from wt_asbs.utils.topology import pdb_to_bond_indices


class BaseSource(nn.Module, ABC):
    """Base class for sampling random atomic data."""

    def __init__(
        self,
        pbc: list[bool] | torch.Tensor = [False, False, False],  # [3,] array
        cell: list[list[float]] | torch.Tensor | None = None,  # [3, 3] array
        composition: str | None = None,
        num_atoms: int | None = None,
        temperature: float = 1.0,
        charge: int | None = 0,
        spin: int | None = 1,
        center: bool = True,
    ):
        if not ((num_atoms is None) ^ (composition is None)):
            raise ValueError(
                "Exactly one of num_particles or composition must be provided."
            )

        super().__init__()
        self.register_buffer("pbc", torch.as_tensor(pbc, dtype=torch.bool))
        if cell is None:
            cell = torch.zeros(3, 3, dtype=torch.float)
        self.register_buffer("cell", torch.as_tensor(cell, dtype=torch.float))
        if composition is not None:
            atomic_numbers = composition_to_atomic_numbers(composition)
            num_atoms = len(atomic_numbers)
        else:
            atomic_numbers = [0] * num_atoms  # Default to zero for unspecified
        self.register_buffer(
            "atomic_numbers", torch.as_tensor(atomic_numbers, dtype=torch.long)
        )
        self.num_atoms = num_atoms
        self.register_buffer(
            "temperature", torch.as_tensor(temperature, dtype=torch.float)
        )
        if charge is None:
            charge = torch.zeros(num_atoms, dtype=torch.long)
        self.register_buffer("charge", torch.as_tensor(charge, dtype=torch.long))
        if spin is None:
            spin = torch.ones(num_atoms, dtype=torch.long)
        self.register_buffer("spin", torch.as_tensor(spin, dtype=torch.long))
        self.center = center

    @abstractmethod
    def sample_single_data(self) -> ThermoAtomicData:
        """Sample a single atomic data instance."""
        pass

    def sample(self, shape: int | tuple[int]) -> ThermoAtomicData:
        """Sample a batch of random atomic data."""
        if isinstance(shape, tuple):
            if len(shape) != 1:
                raise ValueError("Shape must be a single integer for batch size.")
            shape = shape[0]
        data_list = [self.sample_single_data() for _ in range(shape)]
        data = ThermoAtomicData.from_data_list(data_list)
        if self.center and not is_mean_free_data(data):
            raise ValueError("Sampled data is not mean-free.")
        return data


class PeriodicUniformSource(BaseSource):
    def sample_single_data(self) -> ThermoAtomicData:
        if not self.pbc.all():
            raise ValueError("PeriodicUniformSource requires full PBCs.")
        pos = torch.rand(self.num_atoms, 3, dtype=torch.float, device=self.cell.device)
        pos = pos @ self.cell
        if self.center:
            pos -= pos.mean(dim=0, keepdim=True)
        return ThermoAtomicData.from_geometry(
            pos=pos,
            atomic_numbers=self.atomic_numbers,
            cell=self.cell.unsqueeze(0),
            pbc=self.pbc.unsqueeze(0),
            temperature=self.temperature.unsqueeze(0),
            charge=self.charge.unsqueeze(0),
            spin=self.spin.unsqueeze(0),
        ).to(self.cell.device)


class GaussianSource(BaseSource):
    def __init__(
        self,
        pbc: list[bool] | torch.Tensor = [False, False, False],  # [3,] array
        cell: list[list[float]] | torch.Tensor | None = None,  # [3, 3] array
        composition: str | None = None,
        num_atoms: int | None = None,
        temperature: float = 1.0,
        charge: int | None = 0,
        spin: int | None = 1,
        scale: float | None = 1.0,
        center: bool = True,
    ):
        super().__init__(
            pbc=pbc,
            cell=cell,
            composition=composition,
            num_atoms=num_atoms,
            temperature=temperature,
            charge=charge,
            spin=spin,
            center=center,
        )
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float))

    def sample_single_data(self) -> ThermoAtomicData:
        pos = torch.randn(self.num_atoms, 3, dtype=torch.float, device=self.cell.device)
        pos = pos * self.scale
        if self.center:
            pos -= pos.mean(dim=0, keepdim=True)
        return ThermoAtomicData.from_geometry(
            pos=pos,
            atomic_numbers=self.atomic_numbers,
            cell=self.cell.unsqueeze(0),
            pbc=self.pbc.unsqueeze(0),
            temperature=self.temperature.unsqueeze(0),
            charge=self.charge.unsqueeze(0),
            spin=self.spin.unsqueeze(0),
        ).to(self.cell.device)


class HarmonicSource(BaseSource):
    def __init__(
        self,
        pbc: list[bool] | torch.Tensor = [False, False, False],  # [3,] array
        cell: list[list[float]] | torch.Tensor | None = None,  # [3, 3] array
        topology_pdb_file: str | None = None,
        num_atoms: int | None = None,
        temperature: float = 1.0,
        charge: int | None = 0,
        spin: int | None = 1,
        scale: float | None = 1.0,
        center: bool = True,
    ):
        composition = (
            str(ase.io.read(topology_pdb_file).symbols)
            if topology_pdb_file is not None
            else None
        )
        super().__init__(
            pbc=pbc,
            cell=cell,
            composition=composition,
            num_atoms=num_atoms,
            temperature=temperature,
            charge=charge,
            spin=spin,
            center=center,
        )

        # Construct Laplacian
        if topology_pdb_file is not None:
            L = torch.zeros((self.num_atoms, self.num_atoms), dtype=torch.float)
            bond_indices = pdb_to_bond_indices(topology_pdb_file)
            L[bond_indices[0], bond_indices[0]] += 1
            L[bond_indices[1], bond_indices[1]] += 1
            L[bond_indices[0], bond_indices[1]] = -1
            L[bond_indices[1], bond_indices[0]] = -1
        else:  # Assume fully connected graph
            L = torch.eye(self.num_atoms, dtype=torch.float) * self.num_atoms
            L = L - torch.ones((self.num_atoms, self.num_atoms), dtype=torch.float)
        L = torch.kron(L, torch.eye(3, dtype=torch.float))  # Expand to 3D
        D, P = torch.linalg.eigh(L)
        D_inv = 1 / D
        D_inv[D < 1e-6] = 0.0  # Ignore small eigenvalues
        self.register_buffer("transform", P @ torch.diag(torch.sqrt(D_inv)))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float))

    def sample_single_data(self) -> ThermoAtomicData:
        pos = (
            self.transform
            @ torch.randn(
                self.num_atoms * 3, dtype=torch.float, device=self.cell.device
            )
        ).reshape(self.num_atoms, 3)
        pos = pos * self.scale
        if self.center:
            pos -= pos.mean(dim=0, keepdim=True)
        return ThermoAtomicData.from_geometry(
            pos=pos,
            atomic_numbers=self.atomic_numbers,
            cell=self.cell.unsqueeze(0),
            pbc=self.pbc.unsqueeze(0),
            temperature=self.temperature.unsqueeze(0),
            charge=self.charge.unsqueeze(0),
            spin=self.spin.unsqueeze(0),
        ).to(self.cell.device)


class PreGeneratedAtomsSource(BaseSource):
    """Source distribution defined by pre-generated atomic configurations."""

    def __init__(
        self,
        atoms_file: str,
        wrap: bool = False,
        center: bool = True,
        temperature: float = 1.0,
        charge: int | None = 0,
        spin: int | None = 1,
    ):
        nn.Module.__init__(self)
        self.atoms = ase.io.read(atoms_file, index=":")
        # Fix for empty last frame with PDB files
        if atoms_file.endswith(".pdb") and len(self.atoms[-1]) == 0:
            self.atoms = self.atoms[:-1]
        self.center = center
        self.temperature = temperature
        # Wrap and center positions if required
        for atom in self.atoms:
            positions = atom.get_positions()
            # cell = atom.get_cell().array
            # if wrap:
            #     positions = wrap_positions(positions, cell)
            if center:
                positions -= positions.mean(axis=0)
            atom.set_positions(positions)
            if "charge" not in atom.info:
                atom.info["charge"] = charge
            if "spin" not in atom.info:
                atom.info["spin"] = spin
        self.data_list = [
            ThermoAtomicData.from_ase(atoms, temperature) for atoms in self.atoms
        ]
        # Buffer for device management
        self.register_buffer("device_buffer", torch.tensor(0, dtype=torch.long))

    def sample_single_data(self) -> ThermoAtomicData:
        """Sample a single atomic data instance from the pre-generated list."""
        index = torch.randint(0, len(self.data_list), size=(1,)).item()
        batch = self.data_list[index]
        return batch.to(self.device_buffer.device)

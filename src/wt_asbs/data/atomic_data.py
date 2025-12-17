# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Sequence

import ase
import numpy as np
import torch
from fairchem.core.datasets.atomic_data import AtomicData


class ThermoAtomicData(AtomicData):
    """A class for atomic data with thermal properties."""

    def __init__(self, *, temperature: torch.Tensor | None = None, **kwargs):
        """
        Initialize the ThermoAtomicData class.

        Args:
            temperature (torch.Tensor | None): Temperature tensor of shape (num_graph,).
            **kwargs: Additional keyword arguments for AtomicData.
        """
        super().__init__(**kwargs)
        if temperature is not None:
            self.temperature = temperature

    def validate(self):
        if hasattr(self, "temperature"):
            assert self.temperature.dim() == 1
            assert self.temperature.shape[0] == self.num_graphs
            assert self.temperature.dtype == torch.float
        super().validate()

    def clone(self):
        data_dict = {}
        for key in self.__keys__:
            if torch.is_tensor(self[key]):
                data_dict[key] = self[key].clone()
            else:
                data_dict[key] = copy.deepcopy(self[key])
        data_dict["batch"] = self.batch.clone()
        batch_stats = copy.deepcopy(self.get_batch_stats())
        cloned = ThermoAtomicData.from_dict(data_dict)
        cloned.assign_batch_stats(*batch_stats)
        return cloned

    def get_example(self, idx: int) -> "ThermoAtomicData":
        """Get a single example from the dataset."""
        if self.num_graphs == 1 and (idx == 0 or idx == -1):
            return self

        data_dict = {}
        idx = self.num_graphs + idx if idx < 0 else idx
        for key in self.__slices__:
            item = self[key]
            if self.__cat_dims__[key] is None:
                # The item was concatenated along a new batch dimension,
                # so just index in that dimension:
                item = item[idx]
            else:
                # Narrow the item based on the values in `__slices__`.
                if isinstance(item, torch.Tensor):
                    dim = self.__cat_dims__[key]
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item.narrow(dim, start, end - start)
                else:
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item[start:end]
                    item = item[0] if len(item) == 1 else item

            # Decrease its value by `cumsum` value:
            cum = self.__cumsum__[key][idx]
            if isinstance(item, torch.Tensor):
                if not isinstance(cum, int) or cum != 0:
                    item = item - cum
            elif isinstance(item, (int, float)):
                item = item - cum

            data_dict[key] = item

        data_dict["batch"] = torch.zeros_like(data_dict["atomic_numbers"])
        data_dict["sid"] = [self.sid[idx]]

        if hasattr(self, "dataset"):
            data_dict["dataset"] = self.dataset[idx]

        return ThermoAtomicData.from_dict(data_dict)

    @classmethod
    def from_ase(
        cls,
        input_atoms: ase.Atoms | list[ase.Atoms],
        temperature: float | None = None,
        **kwargs,
    ):
        # NOTE: The super method wraps positions, but we do not want to wrap here since
        # we might deal with unwrapped trajectories. Hence, CPU graph generation is
        # not supported.
        if kwargs.get("r_edges", False):
            raise ValueError("r_edges=True is not supported for ThermoAtomicData.")

        # If input_atoms is a list of ASE Atoms, convert each to ThermoAtomicData
        if isinstance(input_atoms, list) and isinstance(input_atoms[0], ase.Atoms):
            return cls.from_data_list(
                [
                    cls.from_ase(atoms, temperature=temperature, **kwargs)
                    for atoms in input_atoms
                ]
            )

        data = super().from_ase(input_atoms, **kwargs).to_dict()
        pos = np.array(input_atoms.get_positions(), copy=True)
        data["pos"] = torch.from_numpy(pos).float()  # switch to unwrapped positions
        if "temperature" in input_atoms.info:
            temperature = input_atoms.info["temperature"]
        if temperature is not None:
            data["temperature"] = torch.tensor([temperature], dtype=torch.float)
        return cls(**data)

    def to_ase_single(self) -> ase.Atoms:
        atoms = super().to_ase_single()
        if hasattr(self, "temperature"):
            atoms.info["temperature"] = self.temperature.item()
        return atoms

    @classmethod
    def from_geometry(
        cls,
        pos: torch.Tensor,  # [num_nodes, 3]
        atomic_numbers: torch.Tensor,  # [num_nodes,]
        cell: torch.Tensor,  # [num_graphs, 3, 3]
        pbc: torch.Tensor,  # [num_graphs, 3]
        temperature: torch.Tensor | None = None,  # [num_graphs,]
        charge: torch.Tensor | None = None,  # [num_graphs,]
        spin: torch.Tensor | None = None,  # [num_graphs,]
    ):
        pos = torch.as_tensor(pos, dtype=torch.float)
        atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long)
        cell = torch.as_tensor(cell, dtype=torch.float)
        pbc = torch.as_tensor(pbc, dtype=torch.bool)
        natoms = torch.tensor([pos.shape[0]], dtype=torch.long)

        # Empty graph initialization
        edge_index = torch.empty((2, 0), dtype=torch.long)
        cell_offsets = torch.empty((0, 3), dtype=torch.float)
        nedges = torch.tensor([0], dtype=torch.long)

        # Node-level features
        tags = torch.zeros_like(atomic_numbers, dtype=torch.long)
        fixed = torch.zeros_like(atomic_numbers, dtype=torch.long)

        # Graph-level features
        if charge is not None:
            charge = torch.as_tensor(charge, dtype=torch.long)
        else:
            charge = torch.tensor([0], dtype=torch.long)
        if spin is not None:
            spin = torch.as_tensor(spin, dtype=torch.long)
        else:
            spin = torch.tensor([1], dtype=torch.long)
        if temperature is not None:
            temperature = torch.as_tensor(temperature, dtype=torch.float)
        else:
            temperature = torch.tensor([0], dtype=torch.float)

        return cls(
            pos=pos,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=charge,
            spin=spin,
            fixed=fixed,
            tags=tags,
            temperature=temperature,
        )

    @classmethod
    def from_data_list(
        cls, data_list: Sequence["ThermoAtomicData"]
    ) -> "ThermoAtomicData":
        """Convert a list of ThermoAtomicData instances into a single batch."""
        keys = list(set(data_list[0].keys()))

        batched_data_dict = {k: [] for k in keys}
        batch = []

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        natoms_list, sid_list = [], []

        for i, data in enumerate(data_list):
            assert data.num_graphs == 1, (
                "data list must only contain single-graph AtomicData objects."
            )

            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, torch.Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                # 0-dimensional torch.Tensors have no dimension along which to
                # concatenate, so we set `cat_dim` to `None`.
                if isinstance(item, torch.Tensor) and item.dim() == 0:
                    cat_dim = None
                cat_dims[key] = cat_dim

                # Add a batch dimension to items whose `cat_dim` is `None`:
                if isinstance(item, torch.Tensor) and cat_dim is None:
                    cat_dim = 0  # Concatenate along this new batch dimension.
                    item = item.unsqueeze(0)
                    device = item.device
                elif isinstance(item, torch.Tensor):
                    size = item.size(cat_dim)
                    device = item.device

                batched_data_dict[key].append(item)  # Append item to the attribute list

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

            natoms_list.append(data.natoms.item())
            sid_list.extend(data.sid)
            item = torch.full((data.natoms,), i, dtype=torch.long, device=device)
            batch.append(item)

        ref_data = data_list[0]
        for key in keys:
            items = batched_data_dict[key]
            item = items[0]
            cat_dim = ref_data.__cat_dim__(key, item)
            cat_dim = 0 if cat_dim is None else cat_dim
            if torch.is_tensor(item):
                batched_data_dict[key] = torch.cat(items, cat_dim)
            else:
                batched_data_dict[key] = items

        batched_data_dict["batch"] = torch.cat(batch, dim=-1)
        batched_data_dict["sid"] = sid_list
        atomic_data_batch = ThermoAtomicData.from_dict(batched_data_dict)
        atomic_data_batch.assign_batch_stats(slices, cumsum, cat_dims, natoms_list)

        return atomic_data_batch.contiguous()

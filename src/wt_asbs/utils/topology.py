# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mdtraj as md
import numpy as np
from fairchem.core.datasets.atomic_data import AtomicData


def pdb_to_bond_indices(pdb_file: str) -> np.ndarray:
    """Extract bond indices from a PDB file."""
    pdb = md.load(pdb_file)
    bond_indices = []
    for bond in pdb.topology.bonds:
        bond_indices.append([bond[0].index, bond[1].index])
    return np.array(bond_indices).T


def save_data_to_pdb(
    data: AtomicData | list[AtomicData], topology_file: str, output_file: str
):
    """Save positions to a PDB file, using the topology from a reference PDB file."""
    topology = md.load(topology_file).topology
    if isinstance(data, AtomicData):
        pos = data.pos.unsqueeze(0).cpu().numpy()
    else:  # list of AtomicData
        pos = np.concatenate([d.pos.unsqueeze(0).cpu().numpy() for d in data])
    pos = pos.reshape(-1, topology.n_atoms, 3)
    traj = md.Trajectory(pos / 10.0, topology=topology)  # A to nm
    traj.save(output_file)

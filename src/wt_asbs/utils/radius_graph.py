# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.compute import get_pbc_distances
from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2


@torch.compiler.disable()
def wrap_and_generate_graph(
    data: AtomicData,
    cutoff: float,
    max_neighbors: int = 300,
    enforce_max_neighbors_strictly: bool = False,
) -> tuple[AtomicData, dict[str, torch.Tensor]]:
    """Wrap the positions of atoms in the data and generate a graph representation
    of the molecular structures based on the provided cutoff distance and maximum
    number of neighbors.
    The differences from the original `generate_graph` in fairchem are:
    - It uses `radius_graph_pbc_v2` for batched processing.
    - It also returns the wrapped positions of the atoms.

    Args:
        data (AtomicData): An AtomicData object containing a batch of atomic structures.
        cutoff (float): The maximum distance between atoms to consider as neighbors.
        max_neighbors (int): The maximum number of neighbors to consider for each atom.
        enforce_max_neighbors_strictly (bool): Whether to strictly enforce the maximum
            number of neighbors.

    Returns:
        AtomicData: An AtomicData object with the wrapped positions of atoms.
        dict: A dictionary containing the generated graph with the following keys:
            - 'edge_index' (torch.Tensor): Indices of the edges in the graph.
            - 'edge_distance' (torch.Tensor): Distances between the atoms connected by
                the edges.
            - 'edge_distance_vec' (torch.Tensor): Displacement vectors between the atoms
                connected by the edges.
            - 'cell_offsets' (torch.Tensor): Offsets of the cell vectors for each edge.
            - 'offset_distances' (torch.Tensor): Offset distances for each edge.
            - 'neighbors' (torch.Tensor): Number of neighbors for each atom.
    """
    # Wrap the positions of atoms in the data
    if data.pbc.all():  # full PBC
        cell_per_atom = data.cell[data.batch]
        frac_pos = torch.einsum("ni,nij->nj", data.pos, torch.linalg.inv(cell_per_atom))
        shifts = torch.floor(frac_pos)
        wrapped_pos = data.pos - torch.einsum("ni,nij->nj", shifts, cell_per_atom)
    elif not data.pbc.any():  # no PBC
        wrapped_pos = data.pos.clone()
    else:  # mixed PBC
        raise NotImplementedError("Mixed PBC is not implemented yet.")
    wrapped_data = data.clone()
    wrapped_data.pos = wrapped_pos

    # Generate the graph
    edge_index, cell_offsets, neighbors = radius_graph_pbc_v2(
        data=wrapped_data,
        radius=cutoff,
        max_num_neighbors_threshold=max_neighbors,
        enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
    )

    # Compute the graph distances and offsets
    out = get_pbc_distances(
        wrapped_data.pos,
        edge_index,
        data.cell,
        cell_offsets,
        neighbors,
        return_offsets=True,
        return_distance_vec=True,
    )
    graph_data = {
        "edge_index": out["edge_index"],
        "edge_distance": out["distances"],
        "edge_distance_vec": out["distance_vec"],
        "cell_offsets": cell_offsets,
        "offset_distances": out["offsets"],
        "neighbors": neighbors,
    }

    return wrapped_data, graph_data

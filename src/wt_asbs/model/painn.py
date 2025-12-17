# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from wt_asbs.data.atomic_data import ThermoAtomicData
from wt_asbs.model.layer.embedding_block import AtomEmbedding, TimeEmbedding
from wt_asbs.model.layer.radial_basis import RadialBasis
from wt_asbs.utils.composition import PeriodicTable
from wt_asbs.utils.radius_graph import wrap_and_generate_graph


class MessageBlock(nn.Module):
    def __init__(self, num_features: int, num_radial_basis: int):
        super().__init__()
        self.num_features = num_features

        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 4),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 4)

    def forward(
        self,
        s: torch.Tensor,  # [n_nodes, 1, n_feats]
        v: torch.Tensor,  # [n_nodes, 3, n_feats]
        radial_embeddings: torch.Tensor,  # [n_edges, 1, num_radial_basis]
        envelope: torch.Tensor,  # [n_edges, 1]
        unit_vectors: torch.Tensor,  # [n_edges, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
    ):
        idx_i, idx_j = edge_index[0], edge_index[1]
        phi = self.mlp_phi(s)
        W = self.linear_W(radial_embeddings) * envelope[..., None]
        x = phi[idx_j] * W
        x_s, x_vv, x_vs, x_vc = torch.split(x, self.num_features, dim=-1)
        ds = torch.zeros_like(s).index_add_(dim=0, index=idx_i, source=x_s)
        x_v = (
            v[idx_j] * x_vv
            + x_vs * unit_vectors[..., None]
            + x_vc * torch.cross(v[idx_j], unit_vectors[..., None], dim=1)
        )
        dv = torch.zeros_like(v).index_add_(dim=0, index=idx_i, source=x_v)
        return s + ds, v + dv


class UpdateBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.mlp_a = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        self.linear_UV = nn.Linear(num_features, num_features * 2, bias=False)

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        U_v, V_v = torch.split(self.linear_UV(v), self.num_features, dim=-1)
        V_v_norm = torch.sqrt(torch.sum(V_v**2, dim=-2, keepdim=True) + 1e-6)
        a = self.mlp_a(torch.cat((s, V_v_norm), dim=-1))
        a_vv, a_sv, a_ss = torch.split(a, self.num_features, dim=-1)
        dv = a_vv * U_v
        UV_dot = torch.sum(U_v * V_v, dim=-2, keepdim=True)
        ds = a_ss + a_sv * UV_dot
        return s + ds, v + dv


class EquivariantRMSNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.affine_s_weight = nn.Parameter(torch.ones(num_features))
            self.affine_s_bias = nn.Parameter(torch.zeros(num_features))
            self.affine_v_weight = nn.Parameter(torch.ones(num_features))

    def forward(
        self,
        s: torch.Tensor,  # [n_nodes, 1, n_feats]
        v: torch.Tensor,  # [n_nodes, 3, n_feats]
    ):
        s = s - s.mean(dim=-1, keepdim=True)
        s = s / (s.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        v = v / (v.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps).sqrt()
        if self.affine:
            s = s * self.affine_s_weight + self.affine_s_bias
            v = v * self.affine_v_weight
        return s, v


class GatedEquivariantBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_output_features: int,
        init_scale: float | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_output_features = num_output_features
        self.init_scale = init_scale
        self.linear_v1 = nn.Linear(num_features, num_features, bias=False)
        self.linear_v2 = nn.Linear(num_features, num_output_features, bias=False)
        self.mlp_s = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_output_features * 2),
        )
        self.act = nn.SiLU()

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        W_v1_norm = torch.sqrt(
            torch.sum(self.linear_v1(v) ** 2, dim=-2, keepdim=True) + 1e-6
        )
        W_v2 = self.linear_v2(v)
        s_out, v_scale = torch.split(
            self.mlp_s(torch.cat((s, W_v1_norm), dim=-1)),
            self.num_output_features,
            dim=-1,
        )
        s_out = self.act(s_out)
        v_out = W_v2 * v_scale
        return s_out, v_out


class PotentialHead(nn.Module):
    def __init__(self, num_features: int, init_scale: float | None = None):
        super().__init__()
        self.init_scale = init_scale
        self.out_energy = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.SiLU(),
            nn.Linear(num_features // 2, 1, bias=False),
        )

    def forward(self, s: torch.Tensor, v: torch.Tensor = None):
        return self.out_energy(s).squeeze(-1)  # [n_nodes, 1]


class VectorHead(nn.Module):
    def __init__(self, num_features: int, init_scale: float | None = None):
        super().__init__()
        self.out_vector = nn.ModuleList(
            [
                GatedEquivariantBlock(num_features, num_features // 2),
                GatedEquivariantBlock(num_features // 2, 1, init_scale=init_scale),
            ]
        )

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        for block in self.out_vector:
            s, v = block(s, v)
        return v.squeeze(-1)  # [n_nodes, 3]


class PaiNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_radial_basis: int,
        num_layers: int,
        num_elements: int | None = None,
        r_max: float = 6.0,
        r_offset: float = 0.0,
        time_init_mode: Literal["node", "edge", "none"] = "node",
        conservative: bool = False,
        periodic_table: PeriodicTable | None = None,
        unique_atom_indices: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_radial_basis = num_radial_basis
        self.num_layers = num_layers
        self.r_max = r_max
        self.r_offset = r_offset
        self.r_cutoff = r_max + r_offset  # internal cutoff for feature computation
        self.time_init_mode = time_init_mode
        self.conservative = conservative
        self.unique_atom_indices = unique_atom_indices

        self.atom_embedding = AtomEmbedding(num_features, num_elements, periodic_table)
        if time_init_mode == "edge":
            self.time_embedding = TimeEmbedding(num_radial_basis)
        elif time_init_mode == "node":
            self.time_embedding = TimeEmbedding(num_features)
        elif time_init_mode == "none":
            self.time_embedding = None
        else:
            raise ValueError(f"Invalid time embedding mode: {time_init_mode}")

        self.radial_embedding = RadialBasis(
            num_radial=num_radial_basis,
            cutoff=self.r_cutoff,
            rbf={"name": "gaussian"},
            envelope={"name": "polynomial", "exponent": 5},
        )

        messages, updates = [], []
        for _ in range(num_layers):
            messages.append(MessageBlock(num_features, num_radial_basis))
            updates.append(UpdateBlock(num_features))
        self.messages = nn.ModuleList(messages)
        self.updates = nn.ModuleList(updates)

        norms_1, norms_2 = [], []
        for _ in range(num_layers):
            norms_1.append(EquivariantRMSNorm(num_features, affine=True))
            norms_2.append(EquivariantRMSNorm(num_features, affine=True))
        self.norms_1 = nn.ModuleList(norms_1)
        self.norms_2 = nn.ModuleList(norms_2)

        if conservative:
            self.output_block = PotentialHead(num_features)
        else:  # direct
            self.output_block = VectorHead(num_features)

    def forward(
        self,
        time: torch.Tensor,  # [n_graphs,]
        data: ThermoAtomicData,
        return_potential: bool = False,
    ) -> (
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ):  # [n_nodes, 3] or [n_nodes, 3], [n_graphs,]
        with torch.enable_grad() if self.conservative else nullcontext():
            if self.conservative:
                data.pos.requires_grad_(True)

            # Compute the radius graph and distances
            data, graph_dict = wrap_and_generate_graph(data, self.r_cutoff)
            unit_vec = graph_dict["edge_distance_vec"] / graph_dict[
                "edge_distance"
            ].clamp(min=1e-6).unsqueeze(-1)  # [n_edges, 3]

            # Compute radial basis functions
            r = graph_dict["edge_distance"]
            r_shifted = (r + self.r_offset).clamp(max=self.r_cutoff).squeeze(-1)
            radial_emb, env = self.radial_embedding(r_shifted, return_envelope=True)
            if self.time_init_mode == "edge":
                edge_graph_index = data.batch[graph_dict["edge_index"][0]]
                time_per_edge = time[edge_graph_index]
                radial_emb = radial_emb + self.time_embedding(time_per_edge)
            radial_emb = radial_emb[:, None, :]  # [n_edges, 1, num_radial_basis]
            env = env[:, None]  # [n_edges, 1]

            # Compute initial scalar and vector features
            if self.unique_atom_indices:
                # Batch index to atom index (in graph) mapping
                atom_index = torch.arange(data.batch.numel(), device=data.batch.device)
                mask = F.pad(
                    (data.batch[1:] != data.batch[:-1]).long(), pad=(1, 0), value=1
                )
                atom_index = atom_index - torch.cummax(atom_index * mask, 0)[0]
                s = self.atom_embedding(atom_index)
            else:
                s = self.atom_embedding(data.atomic_numbers)
            if self.time_init_mode == "node":
                time_emb = self.time_embedding(time[data.batch])
                s = s + time_emb
            s = s[:, None, :]  # [n_nodes, 1, n_feats]
            v = torch.zeros_like(s).repeat(1, 3, 1)  # [n_nodes, 3, n_feats]

            # Apply message passing and update blocks
            for message, update, norm_1, norm_2 in zip(
                self.messages, self.updates, self.norms_1, self.norms_2
            ):
                s, v = message(
                    s, v, radial_emb, env, unit_vec, graph_dict["edge_index"]
                )
                s, v = norm_1(s, v)
                s, v = update(s, v)
                s, v = norm_2(s, v)

            # Apply output block
            output = self.output_block(s, v)  # [n_nodes, 3] or [n_nodes, 1]

            # Direct vector output
            if not self.conservative:
                return output

            # Conservative potential output
            potential_per_node = output.squeeze(-1)  # [n_nodes,]
            potential_per_graph = torch.zeros_like(time).index_add_(
                dim=0, index=data.batch, source=potential_per_node
            )
            field = torch.autograd.grad(
                outputs=potential_per_graph,
                inputs=data.pos,
                grad_outputs=torch.ones_like(potential_per_graph),
                create_graph=self.training,
                retain_graph=self.training,
                allow_unused=True,
            )[0]
            if field is None:
                field = torch.zeros_like(data.pos)
            if return_potential:
                return field, potential_per_graph.detach()  # [n_nodes, 3], [n_graphs,]
            return field  # [n_nodes, 3]

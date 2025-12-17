# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AminoAcidChirality(nn.Module):
    def __init__(self, index_N: int, index_CA: int, index_CB: int, index_C: int):
        super().__init__()
        self.index_N = index_N
        self.index_CA = index_CA
        self.index_CB = index_CB
        self.index_C = index_C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the chirality of an amino acid based on the positions of N, CA, CB, and
        C atoms. Returns +1 for D-configuration, -1 for L-configuration.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_atoms, 3]
        Returns:
            torch.Tensor: Chirality sign for each amino acid in the batch.
        """
        x_N = x[:, self.index_N]
        x_CA = x[:, self.index_CA]
        x_CB = x[:, self.index_CB]
        x_C = x[:, self.index_C]
        cross_product = torch.cross(x_C - x_CA, x_CB - x_CA, dim=-1)
        chirality = torch.sign(torch.sum((x_N - x_CA) * cross_product, dim=-1))
        return chirality


class BaseCV(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        The number of collective variables (CVs) computed by this CV.
        Returns:
            int: Number of CVs.
        """
        pass

    @property
    @abstractmethod
    def periodic(self) -> bool:
        """
        Whether the CV is periodic.
        Returns:
            bool: True if the CV is periodic, False otherwise.
        """
        pass

    @torch.enable_grad()
    def vjp(self, x: torch.Tensor, grad_cv: torch.Tensor) -> torch.Tensor:
        """
        Compute the vector-Jacobian product (VJP) for the CV.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, ...]
            grad_cv (torch.Tensor): Gradient of the potential with respect to the CVs
                of shape [batch_size, n_cv]
        Returns:
            torch.Tensor: Gradient of the potential with respect to the input, with
                shape [batch_size, ...] (same shape as x).
        """

        def closure(x_):
            return self.forward(x_)

        _, pullback = torch.func.vjp(closure, x)
        return pullback(grad_cv)[0]


class ListCV(BaseCV):
    def __init__(self, cv_list: list[BaseCV]):
        super().__init__()
        if [cv.periodic for cv in cv_list] != [cv_list[0].periodic] * len(cv_list):
            raise ValueError("All CVs must have the same periodicity.")
        self.cv_list = nn.ModuleList(cv_list)

    @property
    def dim(self) -> int:
        return sum(cv.dim for cv in self.cv_list)

    @property
    def periodic(self) -> bool:
        return self.cv_list[0].periodic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = [cv(x) for cv in self.cv_list]
        return torch.cat(results, dim=-1)


class LinearCombinationCV(BaseCV):
    def __init__(self, cv: BaseCV, weights: list[float]):
        super().__init__()
        self.cv = cv
        self.register_buffer(
            "weights", torch.tensor(weights, dtype=torch.float), persistent=False
        )

    @property
    def dim(self) -> int:
        return 1

    @property
    def periodic(self) -> bool:
        return self.cv.periodic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.cv(x) * self.weights).sum(dim=-1, keepdim=True)


class IdentityCV(BaseCV):
    def __init__(self, dim: int):
        super().__init__()
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def periodic(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DistanceCV(BaseCV):
    def __init__(self, indices: list[list[int]]):
        super().__init__()
        self.register_buffer(
            "indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )

    @property
    def dim(self) -> int:
        return len(self.indices)

    @property
    def periodic(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p0, p1 = x[:, self.indices].unbind(dim=2)
        distances = ((p0 - p1).pow(2).sum(dim=-1) + 1e-10).sqrt()
        return distances


class CoordinationCV(BaseCV):
    """Coordination number CV based on a switching function.
    s_ij = (1 - ((r_ij - d_0) / r_0) ** n)) / (1 - ((r_ij - d_0) / r_0) ** m)"""

    def __init__(
        self,
        indices: list[list[int]],
        r_0: float,
        d_0: float = 0.0,
        n: int = 6,
        m: int = 8,
    ):
        super().__init__()
        self.register_buffer(
            "indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.r_0 = r_0
        self.d_0 = d_0
        self.n = n
        self.m = m

    @property
    def dim(self) -> int:
        return len(self.indices)

    @property
    def periodic(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p0, p1 = x[:, self.indices].unbind(dim=2)
        distances = ((p0 - p1).pow(2).sum(dim=-1) + 1e-10).sqrt()
        scaled_distances = (distances - self.d_0) / self.r_0
        coordination = (1 - scaled_distances**self.n) / (1 - scaled_distances**self.m)
        coordination = torch.where(  # rare case of division by zero
            torch.isclose(scaled_distances, torch.ones_like(scaled_distances)),
            torch.full_like(scaled_distances, self.n / self.m),
            coordination,
        )
        return coordination


class TorsionCV(BaseCV):
    def __init__(
        self,
        indices: list[list[int]],
        chirality_indicator: AminoAcidChirality | None = None,
    ):
        super().__init__()
        self.register_buffer(
            "indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.chirality_indicator = chirality_indicator

    @property
    def dim(self) -> int:
        return len(self.indices)

    @property
    def periodic(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p0, p1, p2, p3 = x[:, self.indices].unbind(dim=2)
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1 = b1 / (b1.pow(2).sum(dim=-1, keepdim=True) + 1e-10).sqrt()
        v = b0 - b1 * (b0 * b1).sum(dim=-1, keepdim=True)
        w = b2 - b1 * (b2 * b1).sum(dim=-1, keepdim=True)
        torsions = torch.arctan2(
            (torch.cross(b1, v, dim=-1) * w).sum(dim=-1), (v * w).sum(dim=-1)
        )
        if self.chirality_indicator is not None:
            torsions = torsions * self.chirality_indicator(x)[:, None]
        return torsions


class PlanarDistanceCV(BaseCV):
    """Distance between two planes defined by two sets of three points each, defined
    as the distance of plane 2 center from plane 1 in the direction normal to plane 1,
    i.e., s = normal_1 * (center_2 - center_1)."""

    def __init__(
        self,
        indices_1: list[int],
        indices_2: list[int],
    ):
        super().__init__()
        self.register_buffer(
            "indices_1", torch.tensor(indices_1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "indices_2", torch.tensor(indices_2, dtype=torch.long), persistent=False
        )

    @property
    def dim(self) -> int:
        return 1

    @property
    def periodic(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1_0, p1_1, p1_2 = x[:, self.indices_1, :].unbind(dim=1)
        normal_1 = torch.cross(p1_1 - p1_0, p1_2 - p1_0, dim=-1)
        normal_1 = normal_1 / torch.sqrt(
            normal_1.pow(2).sum(dim=-1, keepdim=True) + 1e-6
        )
        center_1 = (p1_0 + p1_1 + p1_2) / 3
        center_2 = x[:, self.indices_2, :].mean(dim=1)
        s = torch.sum(normal_1 * (center_2 - center_1), dim=-1, keepdim=True)
        return s


class MLCV(BaseCV):
    """Machine learning based collective variable (MLCV) using a pretrained model.
    We assume that input features are the pairwise distances between selected atoms."""

    def __init__(self, checkpoint_path: str, atom_indices: list[int]):
        super().__init__()
        self.model = torch.jit.load(checkpoint_path)
        self.atom_indices = atom_indices

        # Check output size
        self.input_size = len(atom_indices) * (len(atom_indices) - 1) // 2
        device = next(self.model.parameters()).device
        dummy_input = torch.zeros(
            (1, self.input_size), dtype=torch.float, device=device
        )
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
        self.output_size = dummy_output.shape[1]

    @property
    def dim(self) -> int:
        return self.output_size

    @property
    def periodic(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if next(self.model.parameters()).device != x.device:
            self.model.to(x.device)

        x = x[:, self.atom_indices, :]
        dist = torch.cdist(x, x)
        triu_idx = torch.triu_indices(dist.size(1), dist.size(1), offset=1)
        dist_features = dist[:, triu_idx[0], triu_idx[1]]
        return self.model(dist_features)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from fairchem.core.datasets.atomic_data import AtomicData

from wt_asbs.utils.geometry import is_mean_free_data, subtract_mean_batch


class BaseSDE(nn.Module, ABC):
    """Base class for SDEs, dX_t = f(t, X_t) dt + g(t) dW_t."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def drift(self, time: torch.Tensor, data: AtomicData) -> torch.Tensor:
        """Drift term of the SDE."""
        pass

    @abstractmethod
    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        """Diffusion term of the SDE."""
        pass

    def randn_like(self, data: AtomicData) -> torch.Tensor:
        """Generate random noise with the same shape as positions."""
        return torch.randn_like(data.pos)


class MeanFreeZeroDriftSDE(BaseSDE, ABC):
    """Base class for mean-free SDEs with zero drift term, dX_t = g(t) A dW_t, where
    A is a projection matrix that ensures the mean-free condition."""

    def __init__(self):
        super().__init__()

    def drift(self, time: torch.Tensor, data: AtomicData) -> torch.Tensor:
        """Return zero drift term."""
        return torch.zeros_like(data.pos)

    @abstractmethod
    def _diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        """Compute the integral of the square of the diffusion term from 0 to t."""
        pass

    @property
    def total_variance(self) -> torch.DoubleTensor:
        """Total variance of the SDE."""
        return self._diffsquare_integral(1.0)

    def reparametrize_time(self, time: torch.Tensor) -> torch.DoubleTensor:
        """Reparametrize time according to the variance schedule."""
        return self._diffsquare_integral(time) / self.total_variance

    def randn_like(self, data: AtomicData) -> torch.Tensor:
        """Generate mean-free random noise with the same shape as positions."""
        return subtract_mean_batch(super().randn_like(data), data.batch)

    @torch.no_grad()
    def sample_posterior(
        self,
        time: torch.Tensor,  # [num_graphs,]
        data_0: AtomicData,  # state at t=0
        data_1: AtomicData,  # state at t=1
        eps: float = 1e-10,
    ) -> AtomicData:
        """Sample X_t ~ p_{t|0, 1} (X_t | X_0, X_1)."""
        if (data_0.batch != data_1.batch).any():
            raise ValueError("data_0 and data_1 must have the same batch index.")
        if not is_mean_free_data(data_0) or not is_mean_free_data(data_1):
            raise ValueError(
                "data_0 and data_1 must be mean-free data (i.e., have zero mean)."
            )
        t_reparam = self.reparametrize_time(time[data_0.batch, None])  # [num_atoms, 1]
        var = t_reparam * (1 - t_reparam) * self.total_variance
        std = torch.sqrt(var + eps).to(data_0.pos)
        t_reparam = t_reparam.to(data_0.pos)
        mean = (1 - t_reparam) * data_0.pos + t_reparam * data_1.pos
        pos_t = mean + std * self.randn_like(data_0)
        data_t = data_0.clone()
        data_t.pos = pos_t
        if not is_mean_free_data(data_t):
            raise ValueError("Sampled data_t is not mean-free")
        return data_t

    @torch.no_grad()
    def cond_score_t0(
        self,
        time: torch.Tensor,  # [num_graphs,]
        data_0: AtomicData,  # state at t=0
        data_t: AtomicData,  # state at t=t
    ) -> torch.Tensor:
        """Compute the conditional score ∇_{X_t} log p_{t|0} (X_t | X_0)."""
        if (data_0.batch != data_t.batch).any():
            raise ValueError("data_0 and data_t must have the same batch index.")
        if not is_mean_free_data(data_0) or not is_mean_free_data(data_t):
            raise ValueError(
                "data_0 and data_t must be mean-free data (i.e., have zero mean)."
            )
        var = self._diffsquare_integral(time[data_0.batch, None])
        return (data_0.pos - data_t.pos) / var.to(data_0.pos)

    @torch.no_grad()
    def cond_score_1t(
        self,
        time: torch.Tensor,  # [num_graphs,]
        data_1: AtomicData,  # state at t=1
        data_t: AtomicData,  # state at t=t
    ) -> torch.Tensor:
        """Compute the conditional score ∇_{X_t} log p_{1|t} (X_1 | X_t)."""
        if (data_1.batch != data_t.batch).any():
            raise ValueError("data_1 and data_t must have the same batch index.")
        if not is_mean_free_data(data_1) or not is_mean_free_data(data_t):
            raise ValueError(
                "data_1 and data_t must be mean-free data (i.e., have zero mean)."
            )
        var = self.total_variance - self._diffsquare_integral(time[data_1.batch, None])
        return (data_1.pos - data_t.pos) / var.to(data_1.pos)


class VESDE(MeanFreeZeroDriftSDE):
    """Variance exploding SDE."""

    def __init__(self, sigma_min: float = 0.001, sigma_max: float = 1.0):
        super().__init__()
        self.register_buffer(
            "sigma_min", torch.as_tensor(sigma_min, dtype=torch.double)
        )
        self.register_buffer(
            "sigma_max", torch.as_tensor(sigma_max, dtype=torch.double)
        )

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        sigma_ratio = self.sigma_max / self.sigma_min
        diffusion = (
            self.sigma_min
            * (sigma_ratio ** (1 - time))
            * torch.sqrt(2 * torch.log(sigma_ratio))
        )
        return diffusion.to(time)

    def _diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        time = torch.as_tensor(time, dtype=torch.double, device=self.sigma_max.device)
        sigma_ratio = self.sigma_max / self.sigma_min
        return (self.sigma_max**2) * (1 - sigma_ratio ** (-2 * time))


class EDMSDE(MeanFreeZeroDriftSDE):
    """EDM schedule from [Karras et al., 2022]."""

    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0,
        rho: float = 7.0,
    ):
        super().__init__()
        self.register_buffer(
            "sigma_min", torch.as_tensor(sigma_min, dtype=torch.double)
        )
        self.register_buffer(
            "sigma_max", torch.as_tensor(sigma_max, dtype=torch.double)
        )
        self.register_buffer("rho", torch.as_tensor(rho, dtype=torch.double))

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        return (
            (1 - time) * self.sigma_max ** (1 / self.rho)
            + time * self.sigma_min ** (1 / self.rho)
        ) ** self.rho

    def _diffsquare_integral(self, time: torch.Tensor | float) -> torch.DoubleTensor:
        time = torch.as_tensor(time, dtype=torch.double, device=self.sigma_max.device)
        return (
            self.sigma_max ** (2 + 1 / self.rho)
            - self.diffusion(time) ** (2 + 1 / self.rho)
        ) / (
            (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
            * (2 * self.rho + 1)
        )


class ControlledSDE(BaseSDE):
    """Controlled SDE with additional control term to the drift.
    dX_t = (f(t, X_t) + g(t)^2 u(t, X_t)) dt + g(t) dW_t"""

    def __init__(self, base_sde: BaseSDE, controller: nn.Module):
        super().__init__()
        self.base_sde = base_sde
        self.controller = controller

    def drift(
        self, time: torch.Tensor, data: AtomicData, return_control: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        diff = self.diffusion(time)
        control = diff * self.controller(time, data)
        drift = self.base_sde.drift(time, data) + diff * control
        if return_control:
            return drift, control
        return drift

    def diffusion(self, time: torch.Tensor) -> torch.Tensor:
        return self.base_sde.diffusion(time)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from fairchem.core.datasets.atomic_data import AtomicData
from tqdm import tqdm

from wt_asbs.process.sde import BaseSDE
from wt_asbs.utils.geometry import subtract_mean_batch


def get_timesteps(
    t0: torch.Tensor | float,
    t1: torch.Tensor | float,
    dt: torch.Tensor | float | None = None,
    steps: int | None = None,
    rescale_t: str | None = None,
) -> torch.Tensor:
    if not (steps is None) ^ (dt is None):
        raise ValueError("Exactly one of dt and steps should be defined.")
    if steps is None:
        steps = int(math.ceil((t1 - t0) / dt))
    else:
        steps = steps + 1

    if rescale_t is None:
        return torch.linspace(t0, t1, steps=steps, dtype=torch.float)
    elif rescale_t == "quad":
        return torch.sqrt(
            torch.linspace(t0, t1.square(), steps=steps, dtype=torch.float)
        ).clip(max=t1)
    elif rescale_t == "cosine":
        s = 0.008
        pre_phase = torch.linspace(t0, t1, steps=steps, dtype=torch.float) / t1
        phase = ((pre_phase + s) / (1 + s)) * torch.pi * 0.5
        dts = torch.cos(phase) ** 4
        dts = (dts / dts.sum()) * t1
        dts_out = torch.concat(
            (torch.tensor([t0], dtype=torch.float), torch.cumsum(dts, -1))
        )
        return dts_out
    raise ValueError("Unknown timestep rescaling method.")


class BaseIntegrator(nn.Module, ABC):
    """Base class for SDE integrators."""

    def __init__(self, sde: BaseSDE, timesteps: torch.Tensor | None = None):
        super().__init__()
        self.sde = sde
        if timesteps is not None:
            self.register_buffer("timesteps", timesteps)

    @abstractmethod
    @torch.no_grad()
    def step(
        self,
        data: AtomicData,
        time: torch.Tensor,  # scalar
        dt: torch.Tensor,  # scalar
        zero_noise: bool = False,
    ) -> AtomicData:
        """Perform a single step of the SDE integration."""
        pass

    @torch.no_grad()
    def run(
        self,
        initial_data: AtomicData,
        timesteps: torch.Tensor | None = None,  # [num_steps,]
        center_every_step: bool = True,
        zero_last_step_noise: bool = False,
        return_trajectory: bool = True,
        progress_bar: bool = True,
    ) -> AtomicData | list[AtomicData]:
        """Integrate the SDE over a sequence of timesteps."""
        if timesteps is None:
            if not hasattr(self, "timesteps"):
                raise ValueError("No timesteps provided and no default timesteps set.")
            timesteps = self.timesteps

        data = initial_data.clone()
        if return_trajectory:
            trajectory = [data.clone()]

        if progress_bar:
            index_range = tqdm(range(len(timesteps) - 1), desc="sdeint", leave=False)
        else:
            index_range = range(len(timesteps) - 1)
        for i in index_range:
            data = self.step(
                data,
                time=timesteps[i],
                dt=(timesteps[i + 1] - timesteps[i]),
                zero_noise=(zero_last_step_noise and i == len(timesteps) - 2),
            )
            if center_every_step:
                data.pos = subtract_mean_batch(data.pos, data.batch)
            if return_trajectory:
                trajectory.append(data.clone())

        if return_trajectory:
            return trajectory
        return data


class EulerMaruyamaIntegrator(BaseIntegrator):
    """Euler-Maruyama integrator for SDEs."""

    @torch.no_grad()
    def step(
        self,
        data: AtomicData,
        time: torch.Tensor,  # scalar
        dt: torch.Tensor,  # scalar
        zero_noise: bool = False,
    ) -> AtomicData:
        """Perform a single step of the Euler-Maruyama integration."""
        drift = self.sde.drift(time, data) * dt
        if zero_noise:
            data.pos = data.pos + drift
            return data
        diffusion = self.sde.diffusion(time) * dt.sqrt() * self.sde.randn_like(data)
        data.pos = data.pos + drift + diffusion
        return data

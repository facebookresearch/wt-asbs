# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from wt_asbs.data.atomic_data import ThermoAtomicData


class TemperatureAnnealer:
    """Annealer that adjusts the temperature of the system."""

    def __init__(self, epochs: list[int], temperatures: list[float]):
        """
        Args:
            epochs (list[int]): List of epochs at which the temperature changes.
            temperatures (list[float]): List of temperatures for the epochs.
        """
        super().__init__()
        if len(epochs) != len(temperatures):
            raise ValueError("epochs and temperatures must have the same length.")
        self.epochs = np.array(epochs, dtype=int)
        self.temperatures = np.array(temperatures, dtype=float)

    def __call__(self, data: ThermoAtomicData, epoch: int) -> ThermoAtomicData:
        temperature = np.interp(epoch, self.epochs, self.temperatures)
        data.temperature = torch.full_like(data.temperature, temperature)
        return data

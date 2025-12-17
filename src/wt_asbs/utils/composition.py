# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ase
import torch
import torch.nn as nn
from ase.formula import Formula


def composition_to_atomic_numbers(composition: str) -> list[int]:
    """Convert a chemical composition string to a list of atomic numbers."""
    return [ase.data.atomic_numbers[element] for element in Formula(composition)]


class PeriodicTable(nn.Module):
    """A simple periodic table for converting element symbols to indices in a subset of
    the periodic table."""

    def __init__(self, elements: list[str]):
        super().__init__()
        self.symbols = elements
        numbers = torch.tensor(
            [ase.data.atomic_numbers[element] for element in elements], dtype=torch.long
        )
        sorted_numbers, indices = torch.sort(numbers)
        self.register_buffer("numbers", numbers)
        self.register_buffer("sorted_numbers", sorted_numbers)
        self.register_buffer("indices", indices)

    def __len__(self) -> int:
        """Return the number of elements in the periodic table."""
        return len(self.symbols)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Convert atomic numbers to indices in the periodic table."""
        pos = torch.searchsorted(self.sorted_numbers, atomic_numbers)
        if not torch.all(self.sorted_numbers[pos] == atomic_numbers):
            raise ValueError("Elements not found in the periodic table.")
        return self.indices[pos]

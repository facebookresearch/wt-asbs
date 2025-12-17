# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from wt_asbs.data.atomic_data import ThermoAtomicData


class ThermoAtomicDataBuffer(Dataset):
    """A buffer to store ThermoAtomicData and tensors for training. Currently
    intended to be instantiated for every rank in a distributed setting."""

    def __init__(self, max_size: int, filter_nan: bool = True):
        self.max_size = max_size
        self.storage = defaultdict(list)
        self.filter_nan = filter_nan

    def extend(self, **kwargs: ThermoAtomicData | torch.Tensor) -> None:
        """Extend the buffer with a dictionary of ThermoAtomicData and tensors."""
        # Extract the batch index
        for value in kwargs.values():
            if isinstance(value, ThermoAtomicData):
                num_graphs, batch_index = value.num_graphs, value.batch
                num_atoms = batch_index.shape[0]
                batch_index = batch_index.cpu()
                break
        else:
            raise ValueError(
                "No ThermoAtomicData found in the input dictionary, cannot determine "
                "batch index."
            )

        # Add data to the buffer
        for key, value in kwargs.items():
            if not isinstance(value, (ThermoAtomicData, torch.Tensor)):
                raise TypeError(
                    f"Expected ThermoAtomicData or torch.Tensor, got {type(value)} for "
                    f"key '{key}'"
                )
            # Always move to cpu for storage
            value = value.to("cpu")
            if isinstance(value, ThermoAtomicData):
                value_list = value.batch_to_atomicdata_list()
            else:  # torch.Tensor
                if value.shape[0] == num_atoms:  # per-atom property
                    value_list = [value[batch_index == i] for i in range(num_graphs)]
                elif value.shape[0] == num_graphs:  # per-graph property
                    value_list = [value[i].unsqueeze(0) for i in range(num_graphs)]
                else:
                    raise ValueError(
                        f"Tensor shape {value.shape} does not match expected "
                        f"per-atom ({num_atoms},) or per-graph ({num_graphs},) shape."
                    )
            self.storage[key].extend(value_list)

        # If nan values are present in tensors, remove them
        if self.filter_nan:
            nan_indices = []
            for key, value in self.storage.items():
                if isinstance(value[0], torch.Tensor):
                    for i, tensor in enumerate(value):
                        if torch.isnan(tensor).any():
                            nan_indices.append(i)
            if nan_indices:
                for key, value in self.storage.items():
                    self.storage[key] = [
                        value[i] for i in range(len(value)) if i not in nan_indices
                    ]

        # Ensure the buffer does not exceed the maximum size
        if len(self) > self.max_size:
            for key in self.storage:
                self.storage[key] = self.storage[key][-self.max_size :]

    def __len__(self) -> int:
        try:
            key = next(iter(self.storage.keys()))
            return len(self.storage[key])
        except StopIteration:
            return 0

    def __getitem__(self, idx) -> dict[str, ThermoAtomicData | torch.Tensor]:
        return {key: value[idx] for key, value in self.storage.items()}

    @staticmethod
    def collate_fn(batch: Sequence[Any]) -> Any:
        """Collate function to combine a batch of data into a single dictionary."""
        element = batch[0]
        if isinstance(element, ThermoAtomicData):
            return ThermoAtomicData.from_data_list(batch)
        elif isinstance(element, torch.Tensor):
            return torch.cat(batch, dim=0)
        elif isinstance(element, float):
            return torch.as_tensor(batch, dtype=torch.float)
        elif isinstance(element, int):
            return torch.as_tensor(batch, dtype=torch.long)
        elif isinstance(element, str):
            return batch
        elif isinstance(element, Mapping):
            return {
                key: ThermoAtomicDataBuffer.collate_fn([item[key] for item in batch])
                for key in element.keys()
            }
        elif isinstance(element, Sequence):  # already not str
            return [ThermoAtomicDataBuffer.collate_fn(item) for item in zip(*batch)]
        else:
            raise TypeError(f"Unsupported type {type(element)} for collate_fn.")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

import torch
from fairchem.core import pretrained_mlip

from wt_asbs.data.atomic_data import ThermoAtomicData
from wt_asbs.potential.base import BasePotential


class UMAPotential(BasePotential):
    def __init__(
        self,
        model_name: str = "uma-s-1p1",
        task_name: str = "omol",
        device: Literal["cuda", "cpu"] = "cuda",
    ):
        super().__init__()
        self.predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
        self.predictor.move_to_device()
        self.task_name = task_name

    def forward(self, data: ThermoAtomicData) -> dict[str, torch.Tensor]:
        if data.pos.isnan().any():
            raise ValueError("Input positions contain NaN values.")

        # Wrap positions
        if data.pbc.all():  # full PBC
            cell_per_atom = data.cell[data.batch]
            frac_pos = torch.einsum(
                "ni,nij->nj", data.pos, torch.linalg.inv(cell_per_atom)
            )
            shifts = torch.floor(frac_pos)
            data.pos = data.pos - torch.einsum("ni,nij->nj", shifts, cell_per_atom)
        elif not data.pbc.any():  # no PBC
            pass  # no wrapping needed
        else:  # mixed PBC
            raise NotImplementedError("Mixed PBC is not implemented yet.")

        # Setup task and properties
        data.task_name = [self.task_name] * data.num_graphs

        # Predict
        pred = self.predictor.predict(data)
        energy = pred["energy"].to(data.pos)
        forces = pred["forces"].to(data.pos)

        return {"energy": energy, "forces": forces}

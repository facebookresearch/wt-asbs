# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from pathlib import Path

import hydra
import torch
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from wt_asbs.potential.base import SumPotential
from wt_asbs.potential.metadynamics import WellTemperedMetadynamicsBias
from wt_asbs.process.sde import ControlledSDE
from wt_asbs.utils.training import EMA


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(inference_cfg: DictConfig) -> None:
    # NOTE: changing matmul precision may result in mean drift
    torch.set_float32_matmul_precision("highest")

    ############################################################################
    # Initialization
    ############################################################################
    run_directory = Path(inference_cfg.checkpoint_directory)
    cfg = OmegaConf.load(run_directory / ".hydra/config.yaml")

    # Lightning fabric setup
    fabric = hydra.utils.instantiate(inference_cfg.fabric)
    fabric.launch()
    rank_seed = inference_cfg.seed ^ fabric.global_rank
    seed_everything(rank_seed, workers=False)

    with fabric.init_module():
        # Models
        controller = hydra.utils.instantiate(cfg.controller)
        if cfg.ema_decay is not None:
            controller = EMA(controller, cfg.ema_decay, cfg.ema_start_step)

        # Process and integration
        source = hydra.utils.instantiate(cfg.source)
        base_sde = hydra.utils.instantiate(cfg.base_sde)
        sde = ControlledSDE(base_sde=base_sde, controller=controller)
        integrator = hydra.utils.instantiate(cfg.integrator, sde=sde)

        # Potential and metadynamics
        potential = hydra.utils.instantiate(cfg.potential)
        use_metadynamics = False
        if isinstance(potential, SumPotential):
            for pot in potential.potentials:
                if isinstance(pot, WellTemperedMetadynamicsBias):
                    use_metadynamics = True
                    metadynamics_bias = pot

    ############################################################################
    # Checkpoint loading
    ############################################################################

    state = {"controller": controller}
    if use_metadynamics:
        state["metadynamics_bias"] = metadynamics_bias

    checkpoint_directory = run_directory / "checkpoints"
    if not checkpoint_directory.exists():
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_directory} does not exist."
        )
    checkpoint_files = list(checkpoint_directory.glob("*.pt"))
    if checkpoint_files:
        checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        fabric.load(checkpoint_path, state)

    if cfg.compile:
        controller = torch.compile(controller)
    controller.eval()

    ############################################################################
    # Buffer and dataloader management
    ############################################################################

    with torch.no_grad():
        # Determine the number of cycles
        samples_per_cycle = inference_cfg.batch_size * fabric.world_size
        num_cycles = math.ceil(inference_cfg.num_samples / samples_per_cycle)

        # Populate the buffer with data from the controlled SDE integration
        if fabric.global_rank == 0:
            if use_metadynamics:
                save_data = {"pos": [], "cv": [], "bias": []}
            else:
                save_data = {"pos": []}
        for _ in tqdm(
            range(num_cycles),
            desc="Generation",
            disable=fabric.global_rank != 0 or not inference_cfg.progress_bar,
        ):
            data_0 = source.sample((inference_cfg.batch_size,))
            data_1 = integrator.run(
                initial_data=data_0,
                center_every_step=cfg.mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=fabric.global_rank == 0,
            )
            if use_metadynamics:
                cvs = metadynamics_bias.compute_cv(data_1)
                bias = metadynamics_bias(data_1)["energy"]

            fabric.barrier()
            pos = fabric.all_gather(data_1.pos).reshape(-1, cfg.num_atoms, 3)
            if use_metadynamics:
                cvs = fabric.all_gather(cvs).reshape(-1, cvs.shape[-1])
                bias = fabric.all_gather(bias).reshape(-1)
            if fabric.global_rank == 0:
                save_data["pos"].append(pos.cpu())
                if use_metadynamics:
                    save_data["cv"].append(cvs.cpu())
                    save_data["bias"].append(bias.cpu())

        # Save samples
        if fabric.global_rank == 0:
            for key in save_data:
                save_data[key] = torch.cat(save_data[key], dim=0)[
                    : inference_cfg.num_samples
                ]
            torch.save(save_data, run_directory / "samples.pt")


if __name__ == "__main__":
    main()

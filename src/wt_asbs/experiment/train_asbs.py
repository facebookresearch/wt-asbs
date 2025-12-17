# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import ase.io
import hydra
import torch
from lightning.fabric import seed_everything
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from wt_asbs.potential.base import SumPotential
from wt_asbs.potential.metadynamics import WellTemperedMetadynamicsBias
from wt_asbs.process.sde import ControlledSDE
from wt_asbs.utils.topology import save_data_to_pdb
from wt_asbs.utils.training import EMA, TrainingCurriculum


class ASBSModule:
    """Module for ASBS training."""

    TASKS: list[str] = ["adjoint_matching", "corrector_matching"]

    ############################################################################
    # Initialization
    ############################################################################

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Lightning fabric setup
        self.fabric = hydra.utils.instantiate(cfg.fabric)
        self.fabric.launch()
        self.rank_seed = cfg.seed ^ self.fabric.global_rank
        seed_everything(self.rank_seed, workers=False)

        with self.fabric.init_module():
            # Models
            controller = hydra.utils.instantiate(cfg.controller)
            corrector = hydra.utils.instantiate(cfg.corrector)
            if cfg.ema_decay is not None:
                self.controller = EMA(controller, cfg.ema_decay, cfg.ema_start_step)
                self.corrector = EMA(corrector, cfg.ema_decay, cfg.ema_start_step)
            else:
                self.controller = controller
                self.corrector = corrector
            if cfg.compile:
                self.controller = torch.compile(self.controller)
                self.corrector = torch.compile(self.corrector)

            # Buffers
            self.adjoint_buffer = hydra.utils.instantiate(cfg.adjoint_buffer)
            self.corrector_buffer = hydra.utils.instantiate(cfg.corrector_buffer)

            # Process and integration
            self.source = hydra.utils.instantiate(cfg.source)
            self.base_sde = hydra.utils.instantiate(cfg.base_sde)
            self.sde = ControlledSDE(base_sde=self.base_sde, controller=self.controller)
            self.integrator = hydra.utils.instantiate(cfg.integrator, sde=self.sde)

            # Potential and cost function
            self.potential = hydra.utils.instantiate(cfg.potential)
            self.terminal_cost = hydra.utils.instantiate(
                cfg.terminal_cost, potential=self.potential, corrector=self.corrector
            )
            self.annealer = hydra.utils.instantiate(cfg.annealer)

        # Process the training curriculum
        self.curriculum = TrainingCurriculum(
            curriculum=cfg.curriculum, valid_tasks=self.TASKS
        )

        # Optimizers and schedulers
        self.controller_optimizer = hydra.utils.instantiate(
            cfg.controller_optimizer,
            params=self.controller.parameters(),
        )
        self.corrector_optimizer = hydra.utils.instantiate(
            cfg.corrector_optimizer,
            params=self.corrector.parameters(),
        )
        self.controller, self.controller_optimizer = self.fabric.setup(
            self.controller, self.controller_optimizer
        )
        self.corrector, self.corrector_optimizer = self.fabric.setup(
            self.corrector, self.corrector_optimizer
        )
        if hasattr(cfg, "controller_scheduler"):
            self.controller_scheduler = hydra.utils.instantiate(
                cfg.controller_scheduler,
                optimizer=self.controller_optimizer,
            )
        if hasattr(cfg, "corrector_scheduler"):
            self.corrector_scheduler = hydra.utils.instantiate(
                cfg.corrector_scheduler,
                optimizer=self.corrector_optimizer,
            )

        self.use_metadynamics = False
        if isinstance(self.potential, SumPotential):
            for pot in self.potential.potentials:
                if isinstance(pot, WellTemperedMetadynamicsBias):
                    self.use_metadynamics = True
                    self.metadynamics_bias = pot

    ############################################################################
    # Training loop structure
    ############################################################################

    def fit(self):
        """Main training loop for the ASBS."""
        self.global_step = 0
        self.current_epoch = 0
        if (
            hasattr(self.cfg, "pretrained_controller_checkpoint")
            and self.cfg.pretrained_controller_checkpoint
        ):
            self.load_pretrained_controller_checkpoint()
        self.load_checkpoint()
        while self.current_epoch < len(self.curriculum):
            current_task = self.curriculum[self.current_epoch]
            self.populate_buffer(current_task)
            dataloader = self.create_dataloader(current_task)
            for batch in tqdm(
                dataloader,
                desc=f"Epoch {self.current_epoch + 1}/{len(self.curriculum)}",
                leave=False,
                disable=self.fabric.global_rank != 0 or not self.cfg.progress_bar,
            ):
                self.training_step(batch, current_task)
                self.global_step += 1
            self.current_epoch += 1
            if self.current_epoch % self.cfg.save_checkpoint_interval_epoch == 0:
                self.save_checkpoint()

    ############################################################################
    # Buffer and dataloader management
    ############################################################################

    @torch.no_grad()
    def populate_buffer(self, current_task):
        """Populate the buffer with data from the controlled SDE integration and log the
        buffer size if the next task is adjoint matching."""
        if current_task.name == "adjoint_matching":
            buffer = self.adjoint_buffer
        elif current_task.name == "corrector_matching":
            buffer = self.corrector_buffer

        # Determine the number of samples to populate the buffer
        # NOTE: Currently we don't save buffer to checkpoint, so restarting the training
        # will repopulate the buffer from scratch with initial_buffer_samples
        if len(buffer) == 0:
            num_samples = self.cfg.initial_buffer_samples
        else:
            num_samples = self.cfg.buffer_samples_per_epoch
        if not num_samples % self.cfg.inference_batch_size == 0:
            rank_zero_warn(
                f"Number of samples ({num_samples}) is not divisible by "
                f"inference batch size ({self.cfg.inference_batch_size}). "
                "Adjusting to the nearest lower multiple."
            )
        num_extend_cycles = num_samples // self.cfg.inference_batch_size

        # Put the models into evaluation mode
        controller_training_state = self.controller.training
        corrector_training_state = self.corrector.training
        self.controller.eval()
        self.corrector.eval()

        # Populate the buffer with data from the controlled SDE integration
        for _ in range(num_extend_cycles):
            data_0 = self.source.sample((self.cfg.inference_batch_size,))
            data_0 = self.annealer(data_0, self.current_epoch)
            data_1 = self.integrator.run(
                initial_data=data_0,
                center_every_step=self.cfg.mean_free,
                zero_last_step_noise=False,
                return_trajectory=False,
                progress_bar=self.fabric.global_rank == 0,
            )
            populate_data = {"data_0": data_0, "data_1": data_1}
            # Add terminal cost if the next task is adjoint matching
            if current_task.name == "adjoint_matching":
                results = self.terminal_cost(
                    data_1=data_1,
                    is_init_stage=current_task.is_init_stage,
                    return_energy=self.cfg.log_buffer_energy,
                    return_grad_norm=self.cfg.log_buffer_grad_norm,
                )
                populate_data["terminal_cost"] = results["terminal_cost"]
                if self.cfg.log_buffer_energy:
                    populate_data["energy"] = results["energy"]
                if self.cfg.log_buffer_grad_norm:
                    populate_data["grad_norm"] = results["grad_norm"]
                # Update hills if using metadynamics and not in the final stage
                if self.use_metadynamics and not current_task.is_final_stage:
                    cvs = self.metadynamics_bias.compute_cv(data_1)
                    # Broadcast the CVs to all processes and add hills
                    cvs = self.fabric.all_gather(cvs).reshape(-1, cvs.shape[-1])
                    self.metadynamics_bias.add_hills(cvs)
            buffer.extend(**populate_data)

        # Log the buffer statistics
        if current_task.name == "adjoint_matching":
            # Log buffer median energy
            if self.cfg.log_buffer_energy:
                median_energy = torch.as_tensor(buffer.storage["energy"]).median()
                median_energy = self.fabric.all_reduce(
                    self.fabric.to_device(median_energy), reduce_op="mean"
                )
                self.fabric.log(
                    "train/adjoint_matching_buffer_median_energy",
                    median_energy,
                    step=self.global_step,
                )
                self.fabric.log(
                    "train/temperature", data_1.temperature[0], step=self.global_step
                )
                # Save samples
                if self.fabric.global_rank == 0:
                    if self.current_epoch % self.cfg.save_sample_interval_epoch == 0:
                        self.save_samples(data_1)
            # Log buffer max grad norm
            if self.cfg.log_buffer_grad_norm:
                median_grad_norm = torch.cat(
                    buffer.storage["grad_norm"], dim=0
                ).median()
                median_grad_norm = self.fabric.all_reduce(
                    self.fabric.to_device(median_grad_norm), reduce_op="max"
                )
                self.fabric.log(
                    "train/adjoint_matching_buffer_median_grad_norm",
                    median_grad_norm,
                    step=self.global_step,
                )
        self.fabric.log(
            f"train/{current_task.name}_buffer_size", len(buffer), step=self.global_step
        )

        # Restore the training state of the models
        self.fabric.barrier()
        self.controller.train(controller_training_state)
        self.corrector.train(corrector_training_state)

    def save_samples(self, data):
        """Save the data samples to an xyz file."""
        if self.fabric.global_rank == 0:
            sample_directory = Path(self.cfg.sample_directory)
            if not sample_directory.exists():
                sample_directory.mkdir(parents=True, exist_ok=True)
            if hasattr(self.cfg, "topology_pdb_file"):  # Save as PDB instead
                filename = sample_directory / f"epoch_{self.current_epoch:04d}.pdb"
                save_data_to_pdb(
                    data=data,
                    topology_file=self.cfg.topology_pdb_file,
                    output_file=str(filename),
                )
            else:  # default to XYZ format
                filename = sample_directory / f"epoch_{self.current_epoch:04d}.xyz"
                ase.io.write(str(filename), data.to_ase())

    def create_dataloader(self, current_task) -> DataLoader:
        # Select the appropriate dataset (buffer) based on the current task
        if current_task.name == "adjoint_matching":
            dataset = self.adjoint_buffer
        elif current_task.name == "corrector_matching":
            dataset = self.corrector_buffer

        # Create a sampler for the dataset
        # NOTE: Not using distributed sampler for now, since we have a separate buffer
        # for each process
        num_samples = self.cfg.train_batch_size * self.cfg.steps_per_epoch
        epoch_seed = self.rank_seed ^ (self.current_epoch << 16)
        generator = torch.Generator().manual_seed(epoch_seed)
        sampler = RandomSampler(
            dataset, replacement=True, num_samples=num_samples, generator=generator
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train_batch_size,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)

    ############################################################################
    # Training steps: adjoint and corrector matching
    ############################################################################

    def training_step(self, batch, current_task) -> None:
        # Get the appropriate step function based on the current task
        if current_task.name == "adjoint_matching":
            loss = self.adjoint_matching_step(batch)
            model = self.controller
            optimizer = self.controller_optimizer
            scheduler_step = hasattr(self, "controller_scheduler")
            if scheduler_step:
                scheduler = self.controller_scheduler
        elif current_task.name == "corrector_matching":
            loss = self.corrector_matching_step(batch)
            model = self.corrector
            optimizer = self.corrector_optimizer
            scheduler_step = hasattr(self, "corrector_scheduler")
            if scheduler_step:
                scheduler = self.corrector_scheduler

        # Perform the optimizer step
        optimizer.zero_grad()
        self.fabric.backward(loss)
        self.fabric.clip_gradients(
            model, optimizer, max_norm=self.cfg.gradient_clip_val
        )
        optimizer.step()

        # Update the EMA
        if self.cfg.ema_decay is not None:
            model.update_ema(self.global_step)

        # Update the scheduler
        if scheduler_step:
            if (
                not hasattr(self.cfg, "skip_scheduler_steps")
                or self.global_step >= self.cfg.skip_scheduler_steps
            ):
                scheduler.step()

        # Log the lr and loss for the current task
        if self.global_step % self.cfg.log_loss_interval_step == 0:
            self.fabric.log(
                f"train/{current_task.name}_lr",
                optimizer.param_groups[0]["lr"],
                step=self.global_step,
            )
            loss = self.fabric.all_reduce(loss.detach(), reduce_op="mean")
            self.fabric.log(
                f"train/{current_task.name}_loss", loss, step=self.global_step
            )

    def adjoint_matching_step(self, batch) -> torch.Tensor:
        """Perform a single step of the adjoint matching and return the loss.
        Equation 14 in the ASBS paper."""
        # Unpack the batch: X_0, X_1 ~ p_{0,1}^u and terminal cost (∇E + h^(k-1))(X_1)
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]
        terminal_cost = batch["terminal_cost"]

        # Sample data at random time: t ~ U[0, 1], X_t ~ p_{t|0,1}^base
        time = self.fabric.to_device(torch.rand(data_0.num_graphs, dtype=torch.float))
        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        # Predict control at time t: u_t(X_t)/σ_t
        control = self.controller(time, data_t)

        # Compute the MSE loss, averaged over all atoms and dimensions
        # NOTE: This is a scaled version: ||u_t(X_t)/σ_t + (∇E + h^(k-1))||^2
        loss = (control + terminal_cost).pow(2).mean()
        return loss

    def corrector_matching_step(self, batch) -> torch.Tensor:
        """Perform a single step of the corrector matching and return the loss.
        Equation 15 in the ASBS paper."""
        # Unpack the batch: X_0, X_1 ~ p_{0,1}^{u^(k)}
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]

        # Predict correction at time t=1: h(X_1)
        time_1 = self.fabric.to_device(torch.ones(data_0.num_graphs, dtype=torch.float))
        correction = self.corrector(time_1, data_1)

        # Compute the target score: ∇_{X_1} log p^base(X_1 | X_0)
        score = self.base_sde.cond_score_t0(time=time_1, data_0=data_0, data_t=data_1)

        # Compute the MSE loss, averaged over all atoms and dimensions
        loss = (correction - score).pow(2).mean()
        return loss

    ############################################################################
    # Checkpoint management
    ############################################################################

    def load_pretrained_controller_checkpoint(self):
        """Load a pretrained controller checkpoint if specified in the config."""
        checkpoint_path = Path(self.cfg.pretrained_controller_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained controller checkpoint not found: {checkpoint_path}"
            )
        state = {"controller": self.controller}
        self.fabric.load(checkpoint_path, state)

    def load_checkpoint(self):
        """Load the latest checkpoint if it exists."""
        state = {
            "controller": self.controller,
            "corrector": self.corrector,
            "controller_optimizer": self.controller_optimizer,
            "corrector_optimizer": self.corrector_optimizer,
        }
        if hasattr(self, "controller_scheduler"):
            state["controller_scheduler"] = self.controller_scheduler
        if hasattr(self, "corrector_scheduler"):
            state["corrector_scheduler"] = self.corrector_scheduler
        if self.use_metadynamics:
            state["metadynamics_bias"] = self.metadynamics_bias
        checkpoint_directory = Path(self.cfg.checkpoint_directory)
        if not checkpoint_directory.exists():
            return
        checkpoint_files = list(checkpoint_directory.glob("*.pt"))
        if checkpoint_files:
            checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            remainder = self.fabric.load(checkpoint_path, state)
            self.global_step = remainder.pop("global_step")
            self.current_epoch = remainder.pop("current_epoch")

    def save_checkpoint(self):
        """Save the current state of the model and optimizer to a checkpoint."""
        checkpoint_directory = Path(self.cfg.checkpoint_directory)
        if not checkpoint_directory.exists():
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_directory / f"epoch_{self.current_epoch:04d}.pt"
        state = {
            "controller": self.controller,
            "corrector": self.corrector,
            "controller_optimizer": self.controller_optimizer,
            "corrector_optimizer": self.corrector_optimizer,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }
        if hasattr(self, "controller_scheduler"):
            state["controller_scheduler"] = self.controller_scheduler
        if hasattr(self, "corrector_scheduler"):
            state["corrector_scheduler"] = self.corrector_scheduler
        if self.use_metadynamics:
            state["metadynamics_bias"] = self.metadynamics_bias
        self.fabric.save(checkpoint_path, state)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    # NOTE: changing matmul precision may result in mean drift
    torch.set_float32_matmul_precision("highest")
    trainer = ASBSModule(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()

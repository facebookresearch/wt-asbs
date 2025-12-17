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

from wt_asbs.process.sde import ControlledSDE
from wt_asbs.utils.topology import save_data_to_pdb
from wt_asbs.utils.training import EMA, TrainingCurriculum


class PretrainModule:
    """Module for bridge matching pretraining."""

    TASKS: list[str] = ["bridge_matching"]

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
            if cfg.ema_decay is not None:
                self.controller = EMA(controller, cfg.ema_decay, cfg.ema_start_step)
            else:
                self.controller = controller
            if cfg.compile:
                self.controller = torch.compile(self.controller)

            # Buffers
            # NOTE: For pretraining, buffers are not necessary, but we use them
            # for implementation simplicity, following the adjoint buffer settings.
            self.buffer = hydra.utils.instantiate(cfg.adjoint_buffer)

            # Process and integration
            self.source = hydra.utils.instantiate(cfg.source)
            self.target = hydra.utils.instantiate(cfg.pretrain_target)
            self.base_sde = hydra.utils.instantiate(cfg.base_sde)
            self.sde = ControlledSDE(base_sde=self.base_sde, controller=self.controller)
            self.integrator = hydra.utils.instantiate(cfg.integrator, sde=self.sde)

        # Process the training curriculum
        self.curriculum = TrainingCurriculum(
            curriculum=cfg.pretrain_curriculum, valid_tasks=self.TASKS
        )

        # Optimizers
        self.optimizer = hydra.utils.instantiate(
            cfg.pretrain_optimizer, params=self.controller.parameters()
        )
        self.controller, self.optimizer = self.fabric.setup(
            self.controller, self.optimizer
        )

        if hasattr(cfg, "pretrain_scheduler"):
            self.pretrain_scheduler = hydra.utils.instantiate(
                cfg.pretrain_scheduler, optimizer=self.optimizer
            )

    ############################################################################
    # Training loop structure
    ############################################################################

    def fit(self):
        """Main training loop for the ASBS."""
        self.global_step = 0
        self.current_epoch = 0
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
                self.save_samples()
                self.save_checkpoint()

    ############################################################################
    # Buffer and dataloader management
    ############################################################################

    @torch.no_grad()
    def populate_buffer(self, current_task):
        """Populate the buffer with data from the source and target distributions."""
        if current_task.name == "bridge_matching":
            buffer = self.buffer

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

        # Populate the buffer with data from the controlled SDE integration
        for _ in range(num_extend_cycles):
            data_0 = self.source.sample((self.cfg.inference_batch_size,))
            data_1 = self.target.sample((self.cfg.inference_batch_size,))
            populate_data = {"data_0": data_0, "data_1": data_1}
            buffer.extend(**populate_data)

        self.fabric.log(
            f"train/{current_task.name}_buffer_size", len(buffer), step=self.global_step
        )

    def save_samples(self):
        """Save the data samples to an xyz file."""

        # Put the models into evaluation mode
        controller_training_state = self.controller.training
        self.controller.eval()

        # Controlled SDE integration
        data_0 = self.source.sample((self.cfg.inference_batch_size,))
        data_1 = self.integrator.run(
            initial_data=data_0,
            center_every_step=self.cfg.mean_free,
            zero_last_step_noise=False,
            return_trajectory=False,
            progress_bar=self.fabric.global_rank == 0,
        )

        # Save the sampled data
        if self.fabric.global_rank == 0:
            sample_directory = Path(self.cfg.sample_directory)
            if not sample_directory.exists():
                sample_directory.mkdir(parents=True, exist_ok=True)
            if hasattr(self.cfg, "topology_pdb_file"):  # Save as PDB instead
                filename = sample_directory / f"epoch_{self.current_epoch:04d}.pdb"
                save_data_to_pdb(
                    data=data_1,
                    topology_file=self.cfg.topology_pdb_file,
                    output_file=str(filename),
                )
            else:  # default to XYZ format
                filename = sample_directory / f"epoch_{self.current_epoch:04d}.xyz"
                ase.io.write(str(filename), data_1.to_ase())

        # Restore the training state of the models
        self.controller.train(controller_training_state)

    def create_dataloader(self, current_task) -> DataLoader:
        # Select the appropriate dataset (buffer) based on the current task
        if current_task.name == "bridge_matching":
            dataset = self.buffer

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
    # Training steps: bridge matching
    ############################################################################

    def training_step(self, batch, current_task) -> None:
        # Get the appropriate step function based on the current task
        if current_task.name == "bridge_matching":
            loss = self.bridge_matching_step(batch)
            model = self.controller
            optimizer = self.optimizer
            scheduler_step = hasattr(self, "pretrain_scheduler")
            if scheduler_step:
                scheduler = self.pretrain_scheduler

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

        # Log the loss for the current task
        if self.global_step % self.cfg.log_loss_interval_step == 0:
            loss = self.fabric.all_reduce(loss.detach(), reduce_op="mean")
            self.fabric.log(
                f"train/{current_task.name}_loss", loss, step=self.global_step
            )

    def bridge_matching_step(self, batch) -> torch.Tensor:
        """Perform a single step of the bridge matching and return the loss.
        Equation 66 in the ASBS paper."""
        # Unpack the batch
        data_0 = batch["data_0"]
        data_1 = batch["data_1"]

        # Sample data at random time: t ~ U[0, 1], X_t ~ p_{t|0,1}^base
        time = self.fabric.to_device(torch.rand(data_0.num_graphs, dtype=torch.float))
        time = time * self.cfg.bridge_matching_max_time
        data_t = self.base_sde.sample_posterior(time=time, data_0=data_0, data_1=data_1)

        # Predict control at time t: u_t(X_t)/σ_t
        control = self.controller(time, data_t)

        # Compute the target score: ∇_{X_t} log p^base(X_1 | X_t)
        score = self.base_sde.cond_score_1t(time=time, data_1=data_1, data_t=data_t)

        # Compute the MSE loss, averaged over all atoms and dimensions
        # Scale the loss by the variance of the base SDE (kappa_{1|t})
        var = self.base_sde.total_variance - self.base_sde._diffsquare_integral(
            time[data_0.batch, None]
        ).to(torch.float)
        loss = torch.mean(var * (control - score).pow(2))
        return loss

    ############################################################################
    # Checkpoint management
    ############################################################################

    def load_checkpoint(self):
        """Load the latest checkpoint if it exists."""
        state = {"controller": self.controller, "optimizer": self.optimizer}
        if hasattr(self, "pretrain_scheduler"):
            state["pretrain_scheduler"] = self.pretrain_scheduler
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
            "optimizer": self.optimizer,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }
        if hasattr(self, "pretrain_scheduler"):
            state["pretrain_scheduler"] = self.pretrain_scheduler
        self.fabric.save(checkpoint_path, state)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    # NOTE: changing matmul precision may result in mean drift
    torch.set_float32_matmul_precision("highest")
    trainer = PretrainModule(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()

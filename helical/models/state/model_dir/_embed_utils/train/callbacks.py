import time
import logging
import os
import shutil

import torch
import lightning as L
from omegaconf import OmegaConf


class LogLR(L.Callback):
    def __init__(self, interval=10):
        super().__init__()
        self.interval = interval

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *args,
    ) -> None:
        if trainer.global_rank == 0:
            if trainer.global_step % self.interval == 0 and trainer.logger is not None:
                trainer.logger.log_metrics(
                    {"trainer/learning_rate": pl_module.lr_schedulers().get_last_lr()[0]},
                    step=trainer.global_step,
                )


class PerfProfilerCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.batch_start_time = None
        self.batch_times = []
        self.iterations_count = 0
        self.last_ipm_time = None
        self.ipm_history = []

    def on_train_batch_start(self, trainer: L.Trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()

        # Calculate batch time
        if self.batch_start_time:
            batch_time = current_time - self.batch_start_time
            self.batch_times.append(batch_time)

        # Track iterations per minute
        self.iterations_count += 1
        if self.last_ipm_time is None:
            self.last_ipm_time = current_time

        time_diff = current_time - self.last_ipm_time
        if time_diff >= 60:
            ipm = (self.iterations_count / time_diff) * 60
            self.ipm_history.append(ipm)
            trainer.logger.log_metrics({"perf/ipm": ipm}, step=trainer.global_step)
            # Reset counters
            self.iterations_count = 0
            self.last_ipm_time = current_time


class ProfilerCallback(L.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.batch_start_time = None
        self.batch_times = []
        self.iterations_count = 0
        self.last_ipm_time = None
        self.ipm_history = []
        self.cfg = cfg

        self.profile_steps = cfg.experiment.profile.profile_steps

    def on_train_batch_start(self, trainer: L.Trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        if batch_idx == self.profile_steps[0]:
            logging.info(f"Starting NSys profiling at step {batch_idx}")
            torch.cuda.nvtx.range_push("VCIProfiledSection")

    def on_train_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()

        # Calculate batch time
        if self.batch_start_time:
            batch_time = current_time - self.batch_start_time
            self.batch_times.append(batch_time)

        # Track iterations per minute
        self.iterations_count += 1
        if self.last_ipm_time is None:
            self.last_ipm_time = current_time

        time_diff = current_time - self.last_ipm_time
        if time_diff >= 60:
            ipm = (self.iterations_count / time_diff) * 60
            self.ipm_history.append(ipm)
            trainer.logger.log_metrics({"perf/ipm": ipm}, step=trainer.global_step)
            # Reset counters
            self.iterations_count = 0
            self.last_ipm_time = current_time

        if batch_idx == self.profile_steps[1]:
            logging.info(f"Stopping NSys profiling at step {batch_idx}")
            torch.cuda.nvtx.range_pop()


class ResumeCallback(L.Callback):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

    def on_train_start(self, trainer, pl_module):
        if self._cfg.optimizer.get("reset_lr_on_restart", False):
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    original_lr = param_group.get("lr", None)
                    param_group["lr"] = self._cfg.optimizer.max_lr
                    logging.info(f"Reset learning rate from {original_lr} to {param_group['lr']}")


class EMACallback(L.Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.beta = decay
        self.velocity = {}

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer):
        # Check if EMA is enabled via the config flag.
        if pl_module.cfg.model.get("ema", False):
            with torch.no_grad():
                for param in pl_module.parameters():
                    if param.grad is None:
                        continue

                    param_id = id(param)
                    if param_id not in self.velocity:
                        self.velocity[param_id] = torch.zeros_like(param.grad)

                    self.velocity[param_id] = self.beta * self.velocity[param_id] + (1 - self.beta) * param.grad
                    param.grad = self.velocity[param_id].clone()

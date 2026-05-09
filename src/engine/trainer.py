from pathlib import Path
import math

import torch
from torch import nn
from tqdm import tqdm

from data.mnist import InfiniteBatchIterator
from visualization import save_loss_curve, save_training_comparison


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        objective,
        dataloader,
        device: torch.device,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        checkpoint_every: int,
        log_every: int,
        image_log_every: int,
        loss_curve_every: int,
        compare_batch_size: int,
        run_dir: Path,
    ):
        self.model = model
        self.objective = objective
        self.device = device
        self.warmup_steps = warmup_steps
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.image_log_every = image_log_every
        self.loss_curve_every = loss_curve_every
        self.compare_batch_size = compare_batch_size
        self.run_dir = run_dir
        self.batch_iterator = InfiniteBatchIterator(dataloader)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.base_lr = lr
        self.max_grad_norm = 1.0
        self._num_train_steps: int = 0  # set at the start of train()
        self.loss_steps: list[int] = []
        self.loss_values: list[float] = []

    def _set_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warm-up
            lr = self.base_lr * float(step + 1) / float(self.warmup_steps)
        elif self._num_train_steps > 0:
            # Cosine annealing after warm-up
            decay_steps = self._num_train_steps - self.warmup_steps
            progress = (step - self.warmup_steps) / max(1, decay_steps)
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr = self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            ckpt_dir / f"step_{step:06d}_model.pt",
        )
        torch.save(
            self.optimizer.state_dict(),
            ckpt_dir / f"step_{step:06d}_optimizer.pt",
        )

    @torch.no_grad()
    def _save_training_comparison(
        self,
        step: int,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        batch_size = min(self.compare_batch_size, images.shape[0])
        target = images[:batch_size]
        target_labels = labels[:batch_size] if labels is not None else None

        t = self.objective.sample_time(batch_size=batch_size, device=images.device)
        x_t = self.objective.probability_path.sample_path(target, t)
        velocity = self.model(x_t, t, target_labels)
        t_view = t.view(-1, *([1] * (x_t.ndim - 1)))
        predicted_target = torch.clamp(x_t + (1.0 - t_view) * velocity, -1.0, 1.0)

        log_path = self.run_dir / "train_logs" / "comparisons" / f"step_{step:06d}.png"
        save_training_comparison(
            targets=target,
            noisy_inputs=x_t,
            predictions=predicted_target,
            output_path=log_path,
            nrow=max(1, min(8, batch_size)),
        )

    def train(self, num_steps: int) -> None:
        self.model.train()
        self._num_train_steps = num_steps
        progress = tqdm(range(num_steps), desc="train")
        for step in progress:
            lr = self._set_lr(step)
            images, labels = self.batch_iterator.next()
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self.objective.compute_loss(self.model, images, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.loss_steps.append(step)
            self.loss_values.append(float(loss.item()))

            if step % self.log_every == 0:
                progress.set_description(f"step={step} lr={lr:.2e} loss={loss.item():.4f}")
            if self.loss_curve_every > 0 and step % self.loss_curve_every == 0:
                save_loss_curve(
                    steps=self.loss_steps,
                    losses=self.loss_values,
                    output_path=self.run_dir / "train_logs" / "loss_curve.png",
                )
            if self.image_log_every > 0 and step % self.image_log_every == 0:
                self._save_training_comparison(step=step, images=images, labels=labels)
            if (self.checkpoint_every > 0 and step > 0 and step % self.checkpoint_every == 0) or step == num_steps - 1:
                self._save_checkpoint(step)

        save_loss_curve(
            steps=self.loss_steps,
            losses=self.loss_values,
            output_path=self.run_dir / "train_logs" / "loss_curve.png",
        )

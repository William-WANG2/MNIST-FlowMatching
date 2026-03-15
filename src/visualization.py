from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


@torch.no_grad()
def save_image_grid(samples: torch.Tensor, output_path: Path, nrow: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = samples.detach().cpu()
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    figure, axis = plt.subplots(figsize=(10, 10))
    axis.imshow(grid.permute(1, 2, 0), cmap="gray")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


@torch.no_grad()
def save_training_comparison(
    targets: torch.Tensor,
    noisy_inputs: torch.Tensor,
    predictions: torch.Tensor,
    output_path: Path,
    nrow: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    targets = targets.detach().cpu()
    noisy_inputs = noisy_inputs.detach().cpu()
    predictions = predictions.detach().cpu()

    rows = [
        ("target", targets),
        ("input (x_t)", noisy_inputs),
        ("predicted target", predictions),
    ]

    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
    for axis, (title, images) in zip(axes, rows):
        grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
        axis.imshow(grid.permute(1, 2, 0), cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_loss_curve(
    steps: list[int],
    losses: list[float],
    output_path: Path,
) -> None:
    if not steps:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(steps, losses, linewidth=1.5)
    axis.set_xlabel("step")
    axis.set_ylabel("train loss")
    axis.set_title("Training Loss")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


import torch

from engine.trainer import Trainer
from visualization import save_vae_reconstruction_comparison


class VAETrainer(Trainer):
    @torch.no_grad()
    def _save_training_comparison(
        self,
        step: int,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        del labels
        batch_size = min(self.compare_batch_size, images.shape[0])
        targets = images[:batch_size]
        _, _, reconstructions, _ = self.model(targets)

        log_path = self.run_dir / "train_logs" / "comparisons" / f"step_{step:06d}.png"
        save_vae_reconstruction_comparison(
            targets=targets,
            reconstructions=reconstructions,
            output_path=log_path,
            nrow=max(1, min(8, batch_size)),
        )

    def train(self, num_steps: int) -> None:
        super().train(num_steps)
        self._print_latent_statistics()

    @torch.no_grad()
    def _print_latent_statistics(self) -> None:
        """Compute and print latent scaling factor after VAE training."""
        self.model.eval()
        sum_val = torch.zeros(1, device=self.device)
        sum_sq = torch.zeros(1, device=self.device)
        count = 0

        for images, _ in self.batch_iterator.dataloader:
            images = images.to(self.device)
            z_mean, _ = self.model.encode(images)
            sum_val += z_mean.sum()
            sum_sq += z_mean.pow(2).sum()
            count += z_mean.numel()

        mean = (sum_val / count).item()
        var = (sum_sq / count).item() - mean ** 2
        std = var ** 0.5
        scaling_factor = 1.0 / std

        print("\n" + "=" * 60)
        print("Latent statistics (z_mean over training set):")
        print(f"  mean = {mean:.6f}")
        print(f"  std  = {std:.6f}")
        print(f"  latent_scaling_factor (1/std) = {scaling_factor:.6f}")
        print("\nSet in configs/representation/latent_vae.yaml:")
        print(f"  latent_scaling_factor: {scaling_factor:.4f}")
        print("=" * 60 + "\n")

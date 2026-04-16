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

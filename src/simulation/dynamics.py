from typing import Optional

import torch
from torch import nn

from simulation.solvers import ODE, SDE


class GuidedVectorFieldODE(ODE):
    def __init__(
        self,
        model: nn.Module,
        conditioning_enabled: bool,
        null_label: Optional[int],
        guidance_scale: float,
    ):
        self.model = model
        self.conditioning_enabled = conditioning_enabled
        self.null_label = null_label
        self.guidance_scale = guidance_scale

    def drift_coefficient(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.conditioning_enabled:
            return self.model(xt, t, None)
        if y is None or self.null_label is None:
            raise ValueError("Conditional sampling requires labels and a configured null label.")
        guided = self.model(xt, t, y)
        unguided = self.model(xt, t, torch.full_like(y, self.null_label))
        return (1.0 - self.guidance_scale) * unguided + self.guidance_scale * guided


class ScoreFromVectorField(nn.Module):
    '''Note that in principle the score field should be learned separately with a separate model and objective, but in practice when the path is Gaussian, the same backbone, i.e., the vector field model, can be reused with a clever rearrangement of the Gaussian-path identities.'''
    def __init__(self, model: nn.Module, probability_path):
        super().__init__()
        self.model = model
        self.probability_path = probability_path

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        view_shape = (t.shape[0],) + (1,) * (x.ndim - 1)
        alpha_t = self.probability_path.alpha(t).view(view_shape)
        beta_t = self.probability_path.beta(t).view(view_shape)
        dt_alpha_t = self.probability_path.alpha.dt(t).view(view_shape)
        dt_beta_t = self.probability_path.beta.dt(t).view(view_shape)
        vector_field = self.model(x, t, y)
        return (alpha_t * vector_field - dt_alpha_t * x) / (
            beta_t * (beta_t * dt_alpha_t - alpha_t * dt_beta_t)
        )


class VectorFieldSDE(SDE):
    def __init__(self, drift_model: GuidedVectorFieldODE, score_model: ScoreFromVectorField, diffusion_scale: float):
        self.drift_model = drift_model
        self.score_model = score_model
        self.diffusion_scale = diffusion_scale

    def drift_coefficient(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        drift = self.drift_model.drift_coefficient(xt, t, y=y)
        score = self.score_model(xt, t, y=y)
        return drift + 0.5 * (self.diffusion_scale**2) * score

    def diffusion_coefficient(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.full_like(xt, self.diffusion_scale)

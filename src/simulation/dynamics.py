from typing import Optional

import torch
from torch import nn

from simulation.solvers import ODE, SDE


def _broadcast_time_like(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    view_shape = (values.shape[0],) + (1,) * (reference.ndim - 1)
    return values.view(view_shape)


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
    def __init__(self, vector_field_model: GuidedVectorFieldODE, probability_path):
        super().__init__()
        self.vector_field_model = vector_field_model
        self.probability_path = probability_path

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha_t = _broadcast_time_like(self.probability_path.alpha(t), x)
        beta_t = _broadcast_time_like(self.probability_path.beta(t), x)
        dt_alpha_t = _broadcast_time_like(self.probability_path.alpha.dt(t), x)
        dt_beta_t = _broadcast_time_like(self.probability_path.beta.dt(t), x)
        vector_field = self.vector_field_model.drift_coefficient(x, t, y=y)
        return (alpha_t * vector_field - dt_alpha_t * x) / (
            beta_t * (beta_t * dt_alpha_t - alpha_t * dt_beta_t)
        )


class VectorFieldSDE(SDE):
    def __init__(
        self,
        drift_model: GuidedVectorFieldODE,
        score_model: ScoreFromVectorField,
        probability_path,
        diffusion_scale: float,
    ):
        self.drift_model = drift_model
        self.score_model = score_model
        self.probability_path = probability_path
        self.diffusion_scale = diffusion_scale

    def drift_coefficient(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        drift = self.drift_model.drift_coefficient(xt, t, y=y)
        score = self.score_model(xt, t, y=y)
        diffusion = self.diffusion_coefficient(xt, t, y=y)
        return drift + 0.5 * diffusion.pow(2) * score

    def diffusion_coefficient(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        beta_t = _broadcast_time_like(self.probability_path.beta(t), xt)
        return self.diffusion_scale * beta_t

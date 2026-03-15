from typing import Optional

import torch
from torch import nn

from simulation.dynamics import GuidedVectorFieldODE, ScoreFromVectorField, VectorFieldSDE
from simulation.solvers import EulerMaruyamaSolver, EulerSolver, build_time_grid


class FlowSampler:
    def __init__(
        self,
        probability_path,
        simulator_name: str,
        t_start: float,
        t_end: float,
        num_steps: int,
        conditioning_enabled: bool,
        null_label: Optional[int],
        diffusion_scale: float,
    ):
        self.probability_path = probability_path
        self.simulator_name = simulator_name
        self.t_start = t_start
        self.t_end = t_end
        self.num_steps = num_steps
        self.conditioning_enabled = conditioning_enabled
        self.null_label = null_label
        self.diffusion_scale = diffusion_scale

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        labels: Optional[torch.Tensor],
        guidance_scale: float,
        use_tqdm: bool,
    ) -> torch.Tensor:
        x0 = self.probability_path.sample_source(batch_size=batch_size, device=device)
        ts = build_time_grid(
            batch_size=batch_size,
            num_steps=self.num_steps,
            t_start=self.t_start,
            t_end=self.t_end,
            device=device,
        )
        drift = GuidedVectorFieldODE(
            model=model,
            conditioning_enabled=self.conditioning_enabled,
            null_label=self.null_label,
            guidance_scale=guidance_scale,
        )

        if self.simulator_name == "ode":
            solver = EulerSolver(drift)
        elif self.simulator_name == "sde":
            score = ScoreFromVectorField(
                vector_field_model=drift,
                probability_path=self.probability_path,
            )
            solver = EulerMaruyamaSolver(
                VectorFieldSDE(
                    drift_model=drift,
                    score_model=score,
                    probability_path=self.probability_path,
                    diffusion_scale=self.diffusion_scale,
                )
            )
        else:
            raise ValueError(f"Unsupported simulator: {self.simulator_name}")

        return solver.simulate(x0, ts, use_tqdm=use_tqdm, y=labels)

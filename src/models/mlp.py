from typing import Optional

import torch
from torch import nn

from models.base import VectorFieldModel
from models.embeddings import FourierEncoder


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.condition_adapter = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2),
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        nn.init.zeros_(self.feed_forward[-1].weight)
        nn.init.zeros_(self.feed_forward[-1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, bias = self.condition_adapter(conditioning).chunk(2, dim=-1)
        hidden = self.norm(x)
        hidden = hidden * (1.0 + scale) + bias
        hidden = self.feed_forward(hidden)
        return x + hidden


class MLPVectorField(VectorFieldModel):
    def __init__(
        self,
        image_shape,
        hidden_dims,
        time_embed_dim: int,
        class_embed_dim: int,
        num_classes: Optional[int],
    ):
        super().__init__(image_shape=image_shape, num_classes=num_classes)
        if not hidden_dims:
            raise ValueError("MLPVectorField expects at least one hidden dimension.")

        flat_dim = int(torch.tensor(image_shape).prod().item())
        conditioning_dim = int(hidden_dims[0])

        self.flat_dim = flat_dim
        self.null_label = num_classes - 1 if num_classes is not None else None
        self.time_embedder = FourierEncoder(time_embed_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_embed_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.class_embedding = (
            nn.Embedding(num_classes, class_embed_dim) if num_classes is not None else None
        )
        self.class_projection = (
            nn.Linear(class_embed_dim, conditioning_dim) if num_classes is not None else None
        )
        self.input_projection = nn.Linear(flat_dim, int(hidden_dims[0]))
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(dim=int(dim), cond_dim=conditioning_dim)
                for dim in hidden_dims
            ]
        )
        self.transitions = nn.ModuleList(
            [
                nn.Identity()
                if int(hidden_dims[index]) == int(hidden_dims[index + 1])
                else nn.Linear(int(hidden_dims[index]), int(hidden_dims[index + 1]))
                for index in range(len(hidden_dims) - 1)
            ]
        )
        self.output_norm = nn.LayerNorm(int(hidden_dims[-1]), elementwise_affine=False)
        self.output_projection = nn.Linear(int(hidden_dims[-1]), flat_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        conditioning = self.time_projection(self.time_embedder(t))
        if self.class_embedding is not None:
            if y is None:
                raise ValueError("Conditional MLPVectorField requires labels.")
            conditioning = conditioning + self.class_projection(self.class_embedding(y))

        hidden = self.input_projection(x.reshape(batch_size, self.flat_dim))
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, conditioning)
            if index < len(self.transitions):
                hidden = self.transitions[index](hidden)

        output = self.output_projection(self.output_norm(hidden))
        return output.reshape(batch_size, *self.image_shape)

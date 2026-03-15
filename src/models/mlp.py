from typing import Optional

import torch
from torch import nn

from models.base import VectorFieldModel
from models.embeddings import FourierEncoder


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


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
        flat_dim = int(torch.tensor(image_shape).prod().item())
        input_dim = flat_dim + time_embed_dim + (class_embed_dim if num_classes is not None else 0)

        self.flat_dim = flat_dim
        self.null_label = num_classes - 1 if num_classes is not None else None
        self.time_embedder = FourierEncoder(time_embed_dim)
        self.class_embedding = (
            nn.Embedding(num_classes, class_embed_dim) if num_classes is not None else None
        )
        self.backbone = MLP([input_dim, *hidden_dims, flat_dim])

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        features = [x.reshape(batch_size, self.flat_dim), self.time_embedder(t)]
        if self.class_embedding is not None:
            if y is None:
                raise ValueError("Conditional MLPVectorField requires labels.")
            features.append(self.class_embedding(y))
        return self.backbone(torch.cat(features, dim=-1)).reshape(batch_size, *self.image_shape)

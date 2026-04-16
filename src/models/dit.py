from typing import Optional

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from models.base import VectorFieldModel
from models.embeddings import FourierEncoder


class Patchifier(nn.Module):
    def __init__(self, image_shape, patch_size: int, dim: int):
        super().__init__()
        channels, height, width = image_shape
        if height != width:
            raise ValueError("DiTVectorField expects square images.")
        if height % patch_size != 0:
            raise ValueError("Patch size must divide the image size.")

        self.image_shape = tuple(image_shape)
        self.patch_size = patch_size
        self.dim = dim
        self.channels = channels
        self.projection = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        # Reshape from (B, dim, H//patch, W//patch) to (B, n_tokens, dim).
        return rearrange(projected, "b d h w -> b (h w) d")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Attention dimension must be divisible by the number of heads.")

        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.fold_heads = Rearrange('b n (h d) -> (b h) n d', h=heads) # rearrange for multi-head attention: fold the head dimension into the batch dimension for efficient computation
        self.unfold_heads = Rearrange('(b h) n d -> b n (h d)', h=heads)
        self.output_projection = nn.Linear(dim, dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1) # b n (h d)
        q, k, v = map(self.fold_heads, (q, k, v)) # (b h) n d

        # Compute attention
        qk = torch.einsum('bid,bjd->bij', q, k) * self.scale # (b h) n n
        attn = torch.softmax(qk, dim=-1) # (b h) n n

        # Compute value aggregation
        out = torch.einsum('bij,bjd->bid', attn, v) # (b h) n d
        out = self.unfold_heads(out) # b n (h d)
        return self.output_projection(out) # b n dim

def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    - scale: b n d
    - bias: b n d
    Returns:
    - x: b n d
    """
    # 1 + scale is a common trick allowing the model to learn to preserve the original normalized activations when the initial scale is near zero, while still allowing for rescaling when beneficial.
    return x * (1 + scale) + bias

class DiffusionTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn = MultiHeadSelfAttention(dim=dim, heads=heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        # efficient than LayerNorm where it only divides by the standard deviation without subtracting the meanand does not learn per-channel scale and shift parameters (elementwise_affine=False) since it will be dynamically learned by the conditioned values.
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, dim * 6) # alpha, beta, gamma for each attention and MLP layer
        )
        # Initialize conditioning to zero - stabilizes residual connection!
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = rearrange(self.ada_ln(c), 'b c -> b 1 c')
        att_scale, att_bias, att_gate, mlp_scale, mlp_bias, mlp_gate = c.chunk(6, dim=-1)
        # Attention block
        x = x + att_gate * self.attn(modulate(self.norm1(x), att_scale, att_bias))
        # MLP block
        x = x + mlp_gate * self.mlp(modulate(self.norm2(x), mlp_scale, mlp_bias))
        return x



class DiffusionTransformer(nn.Module):
    def __init__(self, depth: int, n_tokens: int, dim: int, heads: int):
        super().__init__()
        self.depth = depth
        self.n_tokens = n_tokens
        self.dim = dim
        self.heads = heads
        self.layers = nn.ModuleList(
            [DiffusionTransformerLayer(dim=dim, heads=heads) for _ in range(depth)]
        )
        self.position_embedding = nn.Parameter(torch.randn(1, n_tokens, dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x + self.position_embedding
        for layer in self.layers:
            x = layer(x, c)
        return x


class Depatchifier(nn.Module):
    def __init__(self, image_shape, patch_size: int, dim: int):
        super().__init__()
        channels, height, width = image_shape
        if height != width:
            raise ValueError("DiTVectorField expects square images.")

        self.image_shape = tuple(image_shape)
        self.patch_size = patch_size
        self.dim = dim
        self.channels = channels
        conv_hidden_channels = max(channels, min(dim, 64))
        self.net = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, patch_size * patch_size * conv_hidden_channels),
            Rearrange(
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=height // patch_size,
                w=width // patch_size,
                p1=patch_size,
                p2=patch_size,
                c=conv_hidden_channels,
            ),
            nn.Conv2d(conv_hidden_channels, conv_hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(conv_hidden_channels, channels, kernel_size=3, padding=1),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTVectorField(VectorFieldModel):
    def __init__(
        self,
        image_shape,
        patch_size: int,
        depth: int,
        dim: int,
        heads: int,
        time_embed_dim: int,
        class_embed_dim: int,
        num_classes: Optional[int],
    ):
        super().__init__(image_shape=image_shape, num_classes=num_classes)
        side = image_shape[-1]
        n_tokens = (side // patch_size) ** 2

        self.null_label = num_classes - 1 if num_classes is not None else None
        self.time_embedder = FourierEncoder(time_embed_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.class_embedding = (
            nn.Embedding(num_classes, class_embed_dim) if num_classes is not None else None
        )
        self.class_projection = (
            nn.Linear(class_embed_dim, dim) if num_classes is not None else None
        )
        self.patchifier = Patchifier(image_shape=image_shape, patch_size=patch_size, dim=dim)
        self.transformer = DiffusionTransformer(
            depth=depth,
            n_tokens=n_tokens,
            dim=dim,
            heads=heads,
        )
        self.depatchifier = Depatchifier(
            image_shape=image_shape,
            patch_size=patch_size,
            dim=dim,
        )

    def encode_conditioning(
        self,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the conditioning vector used by the DiT stack.

        In unconditional mode this is time-only; in CFG mode it combines time and label
        information in the shared transformer conditioning space.
        """
        time_condition = self.time_projection(self.time_embedder(t))
        if self.class_embedding is None:
            return time_condition
        if y is None:
            raise ValueError("Conditional DiT requires labels when class conditioning is enabled.")
        return time_condition + self.class_projection(self.class_embedding(y))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.encode_conditioning(t, y)
        x = self.patchifier(x)
        x = self.transformer(x, c)
        x = self.depatchifier(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.spectral_norm as spectral_norm

from .attention import SelfAttention
import math


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, with_attention=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = spectral_norm(nn.Conv2d(2*in_ch, out_ch, 3, padding=1))
            self.transform = spectral_norm(nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1))
        else:
            self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            self.transform = spectral_norm(nn.Conv2d(out_ch, out_ch, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.selu  = nn.SiLU()
        self.with_attention = with_attention
        if self.with_attention:
            self.attention = SelfAttention(out_ch)

    def forward(self, x, t):
        """
        Forward pass for the block.
        Args:
            x: Input tensor.
            t: Time embedding tensor.
        Returns:
            Transformed tensor after applying convolution, time embedding, and optional attention.
        """
        # First Conv
        h = self.relu(self.bnorm1(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Apply attention if required
        if self.with_attention:
            h = self.attention(h)
        # Second Conv
        h = self.relu(self.bnorm2(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Generate sinusoidal position embeddings for a given time step.
        Args:
            time: Input tensor representing the time step.
        Returns:
            Tensor containing sinusoidal position embeddings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUnet(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        image_channels = 1
        cond_channels = 1
        total_channels = image_channels + cond_channels
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 256

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim)
        )

        self.conv0 = nn.Conv2d(in_channels=total_channels, out_channels=down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1],
                                          time_emb_dim, with_attention=(i % 2 == 0))
                                    for i in range(len(down_channels) - 1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1],
                                        time_emb_dim, up=True, with_attention=True)
                                  for i in range(len(up_channels) - 1)])
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, cond, timestep, y=None):
        t = self.time_mlp(timestep)
        if y is not None:
            t += self.label_emb(y)

        # Ensure x and cond have the same dimensions
        assert x.shape[2:] == cond.shape[2:], "Input and conditioning images must have the same spatial dimensions."

        x = torch.cat((x, cond), dim=1)
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

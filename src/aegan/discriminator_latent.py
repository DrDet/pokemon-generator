import torch
from torch import nn


class DiscriminatorLatent(nn.Module):
    """A discriminator for discerning real from generated vectors.

    Input shape: (?, latent_dim)
    Output shape: (?, 1)
    """

    def __init__(self, latent_dim=8, device="cpu"):
        """Initialize the Discriminator."""
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self._init_modules()

    def _init_modules(self, depth=7, width=8):
        """Initialize the modules."""
        self.pyramid = nn.ModuleList()
        for i in range(depth):
            self.pyramid.append(
                nn.Sequential(
                    nn.Linear(
                        self.latent_dim + width * i,
                        width,
                        bias=True,
                    ),
                    nn.BatchNorm1d(width),
                    nn.LeakyReLU(),
                )
            )

        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(depth * width + self.latent_dim, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        last = input_tensor
        for module in self.pyramid:
            projection = module(last)
            rv = torch.randn(projection.size(), device=self.device) * 0.02 + 1
            projection *= rv
            last = torch.cat((last, projection), -1)
        for module in self.classifier:
            last = module(last)
        return last

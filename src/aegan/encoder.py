import torch
from torch import nn


class Encoder(nn.Module):
    """An Encoder for encoding images as latent vectors.

    Input shape: (?, 3, 96, 96)
    Output shape: (?, latent_dim)
    """

    def __init__(self, device: str = "cpu", latent_dim: int = 8):
        """Initialize encoder.

        Args:
            device: chich GPU or CPU to use.
            latent_dim: output dimension
        """
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        down_channels = [3, 64, 128, 256, 512]
        self.down = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=down_channels[i],
                        out_channels=down_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.BatchNorm2d(down_channels[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.reducer = nn.Sequential(
            nn.Conv2d(
                in_channels=down_channels[-1],
                out_channels=down_channels[-2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(down_channels[-2]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )

        up_channels = [256, 128, 64, 64, 64]
        scale_factors = [2, 2, 2, 1]
        self.up = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            self.up.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=up_channels[i] + down_channels[-2 - i],
                        out_channels=up_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.BatchNorm2d(up_channels[i + 1]),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor=scale_factors[i]),
                )
            )

        down_again_channels = [64 + 3, 64, 64, 64, 64]
        self.down_again = nn.ModuleList()
        for i in range(len(down_again_channels) - 1):
            self.down_again.append(
                nn.Conv2d(
                    in_channels=down_again_channels[i],
                    out_channels=down_again_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                )
            )
            self.down_again.append(nn.BatchNorm2d(down_again_channels[i + 1]))
            self.down_again.append(nn.LeakyReLU())

        self.projection = nn.Sequential(
            nn.Linear(
                512 * 6 * 6 + 64 * 6 * 6,
                256,
                bias=True,
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(
                256,
                128,
                bias=True,
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(
                128,
                self.latent_dim,
                bias=True,
            ),
        )

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        augmented_input = input_tensor + rv
        intermediate = augmented_input
        intermediates = [augmented_input]
        for module in self.down:
            intermediate = module(intermediate)
            intermediates.append(intermediate)
        intermediates = intermediates[:-1][::-1]

        down = intermediate.view(-1, 6 * 6 * 512)

        intermediate = self.reducer(intermediate)

        for index, module in enumerate(self.up):
            intermediate = torch.cat((intermediate, intermediates[index]), 1)
            intermediate = module(intermediate)

        intermediate = torch.cat((intermediate, input_tensor), 1)

        for module in self.down_again:
            intermediate = module(intermediate)

        intermediate = intermediate.view(-1, 6 * 6 * 64)
        intermediate = torch.cat((down, intermediate), -1)

        projected = self.projection(intermediate)

        return projected

import torch
from torch import nn


class DiscriminatorImage(nn.Module):
    """A discriminator for discerning real from generated images.

    Input shape: (?, 3, 96, 96)
    Output shape: (?, 1)
    """

    def __init__(self, device="cpu"):
        """Initialize the discriminator."""
        super().__init__()
        self.device = device
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        down_channels = [3, 64, 128, 256, 512]
        self.down = nn.ModuleList()
        leaky_relu = nn.LeakyReLU()
        for i in range(4):
            self.down.append(
                nn.Conv2d(
                    in_channels=down_channels[i],
                    out_channels=down_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                )
            )
            self.down.append(nn.BatchNorm2d(down_channels[i + 1]))
            self.down.append(leaky_relu)

        self.classifier = nn.ModuleList()
        self.width = down_channels[-1] * 6 ** 2
        self.classifier.append(nn.Linear(self.width, 1))
        self.classifier.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        rv = torch.randn(input_tensor.size(), device=self.device) * 0.02
        intermediate = input_tensor + rv
        for module in self.down:
            intermediate = module(intermediate)
            rv = torch.randn(intermediate.size(), device=self.device) * 0.02 + 1
            intermediate *= rv

        intermediate = intermediate.view(-1, self.width)

        for module in self.classifier:
            intermediate = module(intermediate)

        return intermediate

import torch
from torch import nn


class Generator(nn.Module):
    """A generator for mapping a latent space to a sample space.

    Input shape: (?, latent_dim)
    Output shape: (?, 3, 96, 96)
    """

    def __init__(self, latent_dim: int = 8):
        """Initialize generator.

        Args:
            latent_dim (int): latent dimension ("noise vector")
        """
        super().__init__()
        self.latent_dim = latent_dim
        self._init_modules()

    def build_colourspace(self, input_dim: int, output_dim: int):
        """Build a small module for selecting colours."""
        colourspace = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
                bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(
                128,
                64,
                bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(
                64,
                output_dim,
                bias=True),
            nn.Tanh(),
        )
        return colourspace

    def _init_modules(self):
        """Initialize the modules."""
        projection_widths = [8, 8, 8, 8, 8, 8, 8]
        self.projection_dim = sum(projection_widths) + self.latent_dim
        self.projection = nn.ModuleList()
        for index, i in enumerate(projection_widths):
            self.projection.append(
                nn.Sequential(
                    nn.Linear(
                        self.latent_dim + sum(projection_widths[:index]),
                        i,
                        bias=True,
                    ),
                    nn.BatchNorm1d(8),
                    nn.LeakyReLU(),
                )
            )
        self.projection_upscaler = nn.Upsample(scale_factor=3)

        self.colourspace_r = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_g = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_b = self.build_colourspace(self.projection_dim, 16)
        self.colourspace_upscaler = nn.Upsample(scale_factor=96)

        self.seed = nn.Sequential(
            nn.Linear(
                self.projection_dim,
                512 * 3 * 3,
                bias=True),
            nn.BatchNorm1d(512 * 3 * 3),
            nn.LeakyReLU(),
        )

        self.upscaling = nn.ModuleList()
        self.conv = nn.ModuleList()

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=(512) // 4,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(512 + self.projection_dim) // 4,
                out_channels=256,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(256 + self.projection_dim) // 4,
                out_channels=256,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        ))

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(256 + self.projection_dim) // 4,
                out_channels=256,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )),

        self.upscaling.append(nn.Upsample(scale_factor=2))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=(256 + self.projection_dim) // 4,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        ))

        self.upscaling.append(nn.Upsample(scale_factor=1))
        self.conv.append(nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(
                in_channels=64,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Softmax(dim=1),
        ))

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        last = input_tensor
        for module in self.projection:
            projection = module(last)
            last = torch.cat((last, projection), -1)
        projection = last

        intermediate = self.seed(projection)
        intermediate = intermediate.view((-1, 512, 3, 3))

        projection_2d = projection.view((-1, self.projection_dim, 1, 1))
        projection_2d = self.projection_upscaler(projection_2d)

        for i, (conv, upscaling) in enumerate(zip(self.conv, self.upscaling)):
            if i + 1 != len(self.upscaling):
                if i > 0:
                    intermediate = torch.cat((intermediate, projection_2d), 1)
                intermediate = torch.nn.functional.pixel_shuffle(intermediate, 2)
            intermediate = conv(intermediate)
            projection_2d = upscaling(projection_2d)

        r_space = self.colourspace_r(projection)
        r_space = r_space.view((-1, 16, 1, 1))
        r_space = self.colourspace_upscaler(r_space)
        r_space = intermediate * r_space
        r_space = torch.sum(r_space, dim=1, keepdim=True)

        g_space = self.colourspace_g(projection)
        g_space = g_space.view((-1, 16, 1, 1))
        g_space = self.colourspace_upscaler(g_space)
        g_space = intermediate * g_space
        g_space = torch.sum(g_space, dim=1, keepdim=True)

        b_space = self.colourspace_b(projection)
        b_space = b_space.view((-1, 16, 1, 1))
        b_space = self.colourspace_upscaler(b_space)
        b_space = intermediate * b_space
        b_space = torch.sum(b_space, dim=1, keepdim=True)

        output = torch.cat((r_space, g_space, b_space), dim=1)

        return output

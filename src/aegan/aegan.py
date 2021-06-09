import os
import torch
from torch import nn
from torch import optim

from .generator import Generator
from .encoder import Encoder
from .discriminator_image import DiscriminatorImage
from .discriminator_latent import DiscriminatorLatent

EPS = 1e-6
ALPHA_RECONSTRUCT_IMAGE = 1
ALPHA_RECONSTRUCT_LATENT = 0.5
ALPHA_DISCRIMINATE_IMAGE = 0.005
ALPHA_DISCRIMINATE_LATENT = 0.1


class AEGAN():
    """An Autoencoder Generative Adversarial Network for making pokemon."""

    def __init__(self, latent_dim, noise_fn, dataloader,
                 batch_size=32, device='cpu'):
        """Initialize the AEGAN.

        Args:
            latent_dim: latent-space dimension. Must be divisible by 4.
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            dataloader: a pytorch dataloader for loading images
            batch_size: training batch size. Must match that of dataloader
            device: cpu or CUDA
        """
        assert latent_dim % 4 == 0
        self.latent_dim = latent_dim
        self.device = device
        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size

        self.criterion_gen = nn.BCELoss()
        self.criterion_recon_image = nn.L1Loss()
        self.criterion_recon_latent = nn.MSELoss()
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)
        self._init_generator()
        self._init_encoder()
        self._init_dx()
        self._init_dz()

    def _init_generator(self):
        self.generator = Generator(latent_dim=self.latent_dim)
        self.generator = self.generator.to(self.device)
        self.optim_g = optim.Adam(self.generator.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999),
                                  weight_decay=1e-8)

    def _init_encoder(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, device=self.device)
        self.encoder = self.encoder.to(self.device)
        self.optim_e = optim.Adam(self.encoder.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999),
                                  weight_decay=1e-8)

    def _init_dx(self):
        self.discriminator_image = DiscriminatorImage(device=self.device).to(self.device)
        self.optim_di = optim.Adam(self.discriminator_image.parameters(),
                                   lr=1e-4, betas=(0.5, 0.999),
                                   weight_decay=1e-8)

    def _init_dz(self):
        self.discriminator_latent = DiscriminatorLatent(
            latent_dim=self.latent_dim,
            device=self.device,
        ).to(self.device)
        self.optim_dl = optim.Adam(self.discriminator_latent.parameters(),
                                   lr=1e-4, betas=(0.5, 0.999),
                                   weight_decay=1e-8)

    def _get_nets(self):
        nets = {
            'generator': self.generator,
            'encoder': self.encoder,
            'discriminator_image': self.discriminator_image,
            'discriminator_latent': self.discriminator_latent,
        }
        return nets

    def save_state(self, dir):
        for name, net in self._get_nets().items():
            torch.save(net.state_dict(), os.path.join(dir, name + ".pt"))

    def load_state(self, dir):
        for name, net in self._get_nets().items():
            net.load_state_dict(torch.load(os.path.join(dir, name + ".pt"), map_location=self.device))

    def generate_samples(self, latent_vec=None, num=None):
        """Sample images from the generator.

        Images are returned as a 4D tensor of values between -1 and 1.
        Dimensions are (number, channels, height, width). Returns the tensor
        on cpu.

        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None

        If latent_vec and num are None then use self.batch_size
        random latent vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        samples = samples.cpu()  # move images to cpu
        return samples

    def train_step_generators(self, X):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()
        self.encoder.zero_grad()

        Z = self.noise_fn(self.batch_size)

        X_hat = self.generator(Z)
        Z_hat = self.encoder(X)
        X_tilde = self.generator(Z_hat)
        Z_tilde = self.encoder(X_hat)

        X_hat_confidence = self.discriminator_image(X_hat)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)

        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_ones)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_ones)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_ones)

        X_recon_loss = self.criterion_recon_image(X_tilde, X) * ALPHA_RECONSTRUCT_IMAGE
        Z_recon_loss = self.criterion_recon_latent(Z_tilde, Z) * ALPHA_RECONSTRUCT_LATENT

        X_loss = (X_hat_loss + X_tilde_loss) / 2 * ALPHA_DISCRIMINATE_IMAGE
        Z_loss = (Z_hat_loss + Z_tilde_loss) / 2 * ALPHA_DISCRIMINATE_LATENT
        loss = X_loss + Z_loss + X_recon_loss + Z_recon_loss

        loss.backward()
        self.optim_e.step()
        self.optim_g.step()

        return X_loss.item(), Z_loss.item(), X_recon_loss.item(), Z_recon_loss.item()

    def train_step_discriminators(self, X):
        """Train the discriminator one step and return the losses."""
        self.discriminator_image.zero_grad()
        self.discriminator_latent.zero_grad()

        Z = self.noise_fn(self.batch_size)

        with torch.no_grad():
            X_hat = self.generator(Z)
            Z_hat = self.encoder(X)
            X_tilde = self.generator(Z_hat)
            Z_tilde = self.encoder(X_hat)

        X_confidence = self.discriminator_image(X)
        X_hat_confidence = self.discriminator_image(X_hat)
        X_tilde_confidence = self.discriminator_image(X_tilde)
        Z_confidence = self.discriminator_latent(Z)
        Z_hat_confidence = self.discriminator_latent(Z_hat)
        Z_tilde_confidence = self.discriminator_latent(Z_tilde)

        X_loss = 2 * self.criterion_gen(X_confidence, self.target_ones)
        X_hat_loss = self.criterion_gen(X_hat_confidence, self.target_zeros)
        X_tilde_loss = self.criterion_gen(X_tilde_confidence, self.target_zeros)
        Z_loss = 2 * self.criterion_gen(Z_confidence, self.target_ones)
        Z_hat_loss = self.criterion_gen(Z_hat_confidence, self.target_zeros)
        Z_tilde_loss = self.criterion_gen(Z_tilde_confidence, self.target_zeros)

        loss_images = (X_loss + X_hat_loss + X_tilde_loss) / 4
        loss_latent = (Z_loss + Z_hat_loss + Z_tilde_loss) / 4
        loss = loss_images + loss_latent

        loss.backward()
        self.optim_di.step()
        self.optim_dl.step()

        return loss_images.item(), loss_latent.item()

    def train_epoch(self, print_frequency=1, max_steps=0):
        """Train both networks for one epoch and return the losses.

        Args:
            print_frequency (int): print stats every `print_frequency` steps.
            max_steps (int): End epoch after `max_steps` steps, or set to 0
                             to do the full epoch.
        """
        ldx, ldz, lgx, lgz, lrx, lrz = 0, 0, 0, 0, 0, 0
        eps = 1e-9
        for batch, (real_samples, _) in enumerate(self.dataloader):
            real_samples = real_samples.to(self.device)
            ldx_, ldz_ = self.train_step_discriminators(real_samples)
            ldx += ldx_
            ldz += ldz_
            lgx_, lgz_, lrx_, lrz_ = self.train_step_generators(real_samples)
            lgx += lgx_
            lgz += lgz_
            lrx += lrx_
            lrz += lrz_
            if print_frequency and (batch + 1) % print_frequency == 0:
                print(f"{batch + 1}/{len(self.dataloader)}:"
                      f" G={lgx / (eps + (batch + 1) * ALPHA_DISCRIMINATE_IMAGE):.3f},"
                      f" E={lgz / (eps + (batch + 1) * ALPHA_DISCRIMINATE_LATENT):.3f},"
                      f" Dx={ldx / (eps + (batch + 1)):.3f},"
                      f" Dz={ldz / (eps + (batch + 1)):.3f}",
                      f" Rx={lrx / (eps + (batch + 1) * ALPHA_RECONSTRUCT_IMAGE):.3f}",
                      f" Rz={lrz / (eps + (batch + 1) * ALPHA_RECONSTRUCT_LATENT):.3f}",
                      end='\r',
                      flush=True)
            if max_steps and batch == max_steps:
                break
        if print_frequency:
            print()
        lgx /= batch
        lgz /= batch
        ldx /= batch
        ldz /= batch
        lrx /= batch
        lrz /= batch
        return lgx, lgz, ldx, ldz, lrx, lrz

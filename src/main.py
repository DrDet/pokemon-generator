from argparse import ArgumentParser

from aegan.aegan import AEGAN

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

import attr
import json
import os
import pathlib
import time



def save_images(GAN, vec, filename):
    images = GAN.generate_samples(vec)
    ims = tv.utils.make_grid(images[:36], normalize=True, nrow=6, )
    ims = ims.numpy().transpose((1, 2, 0))
    ims = np.array(ims * 255, dtype=np.uint8)
    image = Image.fromarray(ims)
    image.save(filename)


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint-epoch", type=int, help='Checkpoint epoch number to continue training')
    parser.add_argument("--test-generator", default=None, type=pathlib.Path,
                        help='Path to saved generator state. Also it disables training')

    parser.add_argument("--batch-size", default=32, type=int, help="samples per batch")
    parser.add_argument("--latent-dim", default=16, type=int, help="latent space dimension size")

    parser.add_argument("-s", "--start-epoch", default=0, type=int, help="start epochs")
    parser.add_argument("-n", "--num-epochs", default=20000, type=int, help="number of epochs")

    parser.add_argument("-l", "--log", default=None, type=pathlib.Path,
                        help="Path to the log file")
    parser.add_argument("-i", "--input", default="data/", type=pathlib.Path, help="path to the root directory with images")
    args = parser.parse_args()

    os.makedirs("results/generated", exist_ok=True)
    os.makedirs("results/reconstructed", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/losses", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = tv.transforms.Compose([
        tv.transforms.Resize((96, 96)),
        tv.transforms.RandomAffine(0, translate=(5 / 96, 5 / 96), fillcolor=(255, 255, 255)),
        tv.transforms.ColorJitter(hue=0.5),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
    ])

    dataset = ImageFolder(
        root=args.input,
        transform=transform
    )
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=8,
                            drop_last=True
                            )
    X = iter(dataloader)

    test_ims1, _ = next(X)
    test_ims2, _ = next(X)
    test_ims = torch.cat((test_ims1, test_ims2), 0)
    test_ims_show = tv.utils.make_grid(test_ims[:36], normalize=True, nrow=6, )
    test_ims_show = test_ims_show.numpy().transpose((1, 2, 0))
    test_ims_show = np.array(test_ims_show * 255, dtype=np.uint8)
    image = Image.fromarray(test_ims_show)
    image.save("results/reconstructed/test_images.png")

    noise_fn = lambda x: torch.randn((x, args.latent_dim), device=device)
    test_noise = noise_fn(36)
    gan = AEGAN(
        args.latent_dim,
        noise_fn,
        dataloader,
        device=device,
        batch_size=args.batch_size,
    )

    if args.test_generator is None:
        if args.checkpoint_epoch is not None:
            print(f'Loading state from epoch {args.checkpoint_epoch}')
            gan.load_state(f"results/checkpoints/epoch_{args.checkpoint_epoch:05d}")

        start = time.time()

        losses = {'G': [],
                  'E': [],
                  'Dx': [],
                  'Dz': [],
                  'Rx': [],
                  'Rz': []}
        for name in losses:
            with open(os.path.join('results', 'losses', f'{name}.txt'), 'w') as _: pass   # reset files with losses

        for i in range(args.start_epoch, args.num_epochs):
            while True:
                try:
                    with open("pause.json") as f:
                        pause = json.load(f)
                    if pause['pause'] == 0:
                        break
                    print(f"Pausing for {pause['pause']} seconds")
                    time.sleep(pause["pause"])
                except (KeyError, json.decoder.JSONDecodeError, FileNotFoundError):
                    break
            elapsed = int(time.time() - start)
            elapsed = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
            print(f"Epoch {i + 1}; Elapsed time = {elapsed}s")
            lgx, lgz, ldx, ldz, lrx, lrz = gan.train_epoch(max_steps=100)

            losses['G'].append(str(lgx))
            losses['E'].append(str(lgz))
            losses['Dx'].append(str(ldx))
            losses['Dz'].append(str(ldz))
            losses['Rx'].append(str(lrx))
            losses['Rz'].append(str(lrz))

            if (i + 1) % 50 == 0:
                state_dir = f"results/checkpoints/epoch_{i:05d}"
                os.makedirs(state_dir, exist_ok=True)
                for name, vals in losses.items():
                    with open(os.path.join('results', 'losses', f'{name}.txt'), 'a') as file:
                        file.write(" ".join(vals) + " ")
                    losses[name] = []
                gan.save_state(state_dir)

            if (i + 1) % 10 == 0:
                save_images(gan, test_noise, os.path.join("results", "generated", f"gen.{i:04d}.png"))

                with torch.no_grad():
                    reconstructed = gan.generator(gan.encoder(test_ims.to(device=device))).to(device=device)
                reconstructed = tv.utils.make_grid(reconstructed[:36], normalize=True, nrow=6, )
                reconstructed = reconstructed.numpy().transpose((1, 2, 0))
                reconstructed = np.array(reconstructed * 255, dtype=np.uint8)
                reconstructed = Image.fromarray(reconstructed)
                reconstructed.save(os.path.join("results", "reconstructed", f"gen.{i:04d}.png"))

    else:
        gan.generator.load_state_dict(torch.load(args.test_generator, map_location=device))

    images = gan.generate_samples()
    ims = tv.utils.make_grid(images, normalize=True)
    plt.imshow(ims.numpy().transpose((1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()

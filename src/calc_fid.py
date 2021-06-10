import argparse
import csv
import math
import os
import pathlib
import subprocess

import torch
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

import aegan.aegan as aegan


def revert(im):
    im -= im.min()
    im /= im.max()
    im = im.numpy().transpose((1, 2, 0))
    im = np.array(im * 255, dtype=np.uint8)
    return im


def main(args):
    noise_fn = lambda x: torch.randn((x, 16), device='cpu')

    with open("fid.csv", ("wa" if args.append else "w")) as f:
        csv_writer = csv.DictWriter(f, fieldnames=['iternum', 'FID'])
        if not args.append:
            csv_writer.writeheader()

        paths = sorted(os.listdir(args.checkpoints_dir))
        paths = paths[args.from_n:]
        for path in tqdm(paths):
            print(path)
            path = pathlib.Path(path)
            gan = aegan.AEGAN(16, noise_fn, None, device="cuda")

            gan.generator.load_state_dict(torch.load(args.checkpoints_dir / path / "generator.pt", map_location="cpu"))
            gan.generator.train(False)
            generator = gan.generator

            latent_vec = noise_fn(args.num).cuda()
            batch_size = 50
            samples = []
            with torch.no_grad():
                for i in range(0, math.ceil(args.num / batch_size)):
                    l = i * batch_size
                    r = min(args.num, i * batch_size + batch_size)
                    samples.append(generator(latent_vec[l:r]))
            samples = torch.cat(samples)
            samples = samples.cpu()

            data_path = args.checkpoints_dir / path / "gen_dataset"
            if data_path.exists():
                shutil.rmtree(data_path, ignore_errors=True)
            os.makedirs(data_path, exist_ok=False)

            for i in range(args.num):
                im = Image.fromarray(revert(samples[i]))
                im.save(args.checkpoints_dir / path / "gen_dataset" / f"{i:04d}.png")

            result = subprocess.run(["pytorch-fid",
                                     "--batch-size", "16",
                                     "--device", "cuda",
                                     args.original_data, data_path], stdout=subprocess.PIPE)
            fid = result.stdout.decode().split()[-1]
            csv_writer.writerow({"iternum": int(str(path)[-5:]), "FID": fid})
            f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoints_dir", type=pathlib.Path)
    parser.add_argument("original_data", type=pathlib.Path)
    parser.add_argument("-n", "--num", type=int, required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--from_n", type=int, required=True)

    main(parser.parse_args())

import attr
import csv
import pathlib
import sys

@attr.s(auto_attribs=True)
class TrainInfo:
    iteration: int
    generator_loss: float
    encoder_loss: float
    discrem_x_loss: float
    discrem_z_loss: float
    reconstr_x_loss: float
    reconstr_z_loss: float


class TrainInfoDumper:
    def __init__(self, log_path: pathlib.Path, write_header=True):
        self.csvfile = log_path.open("w" if write_header else "a") if log_path is not None else sys.stdout
        self.csvwriter = csv.DictWriter(self.csvfile, fieldnames=['iternum', 'G', 'E', 'Dx', 'Dz', 'Rx', 'Rz'])

        if write_header:
            self.csvwriter.writeheader()

    def close(self):
        self.csvfile.close()

    def append(self, info: TrainInfo):
        self.csvwriter.writerow(
            {
                "iternum": info.iteration,
                 "G": info.generator_loss,
                 "E": info.encoder_loss,
                 "Dx": info.discrem_x_loss,
                 "Dz": info.discrem_z_loss,
                 "Rx": info.reconstr_x_loss,
                 "Rz": info.reconstr_z_loss
            }
        )
        self.csvfile.flush()
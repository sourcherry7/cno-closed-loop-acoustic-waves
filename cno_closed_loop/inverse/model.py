"""Inverse CNO: estimates the velocity field c(x, y) from two wavefield snapshots.

The network is the time-conditioned CNO (``CNO_time``) from
camlab-ethz/ConvolutionalNeuralOperator, used unchanged. Only the data pipeline
differs: the train/validation loaders yield (u(t1), u(t2), dt) -> c samples from
``InverseWaveGaussians``. The two loader methods mirror the single-dataset branch
of ``CNO_time.train_dataloader`` / ``val_dataloader`` (MIT License).
"""
from torch.utils.data import DataLoader

from CNO_timeModule_CIN import CNO_time
from cno_closed_loop.inverse.config import _load_inverse_dataset


class InverseCNO(CNO_time):

    def _inverse_dataset(self, which_loader):
        which = self.loader_dictionary["which"]
        mask = None
        if self.loader_dictionary.get("is_masked") is not None:
            mask = [1.0, 1.0, 1.0, 0.0] if which[:2] == "ns" else [1.0, 1.0, 1.0, 1.0]
        return _load_inverse_dataset(dic=self.loader_dictionary,
                                     which=which,
                                     which_loader=which_loader,
                                     in_dim=self.in_dim,
                                     out_dim=self.out_dim,
                                     masked_input=mask)

    def train_dataloader(self):
        return DataLoader(self._inverse_dataset("train"), batch_size=self.batch_size,
                          shuffle=True, num_workers=6)

    def val_dataloader(self):
        val_dataset = self._inverse_dataset("val")
        self.val_labels = [self.loader_dictionary["which"] + "_"]
        self.num_validation_loaders = 1
        self.num_out_loaders = 0
        return [DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)]

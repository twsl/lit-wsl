from collections.abc import Callable, Iterable, Iterator
import copy
import logging
from pathlib import Path
from typing import Any, cast

from lightning import pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import Sampler

from lit_wsl.training.stages import Stage


class DummyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        get_dataset: Callable[[Stage], Dataset],
        batch_size: int = 2,
        drop_last: bool = False,
        pin_memory: bool = False,
        shuffle: bool | None = True,
        num_workers: int | None = None,
        collate_fn: Callable | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

        if num_workers is None:
            # shm size is currently a problem
            num_workers = 0  # int(os.cpu_count() / 2)
        self.num_workers = num_workers

        self._train_ds = get_dataset(Stage.Training)
        self._val_ds = get_dataset(Stage.Validating)
        self._test_ds = get_dataset(Stage.Testing)
        self._predict_ds = get_dataset(Stage.Predicting)

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self) -> None:
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None) -> None:
        pass
        # # transforms
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # # split dataset
        # if stage in (None, "fit"):
        #     mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
        #     self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        #     self.dims = self.mnist_train[0][0].shape
        # if stage == (None, "test"):
        #     self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)
        #     self.dims = getattr(self, "dims", self.mnist_test[0][0].shape)

    @property
    def train_dataset(self) -> Dataset | None:
        """This property returns the train dataset."""
        return self._train_ds

    @property
    def val_dataset(self) -> Dataset | None:
        """This property returns the validation dataset."""
        return self._val_ds

    @property
    def test_dataset(self) -> Dataset | None:
        """This property returns the test dataset."""
        return self._test_ds

    @property
    def predict_dataset(self) -> Dataset | None:
        """This property returns the predict dataset."""
        return self._predict_ds

    def get_dataloader(
        self,
        dataset: Dataset | Subset,
        sampler: Sampler | Iterator | None = None,
    ) -> DataLoader:
        """Get training loader from the dataset.

        Args:
            dataset: Dataset to create dataloader from.
            sampler: Sampler to use for the dataset.
            collate_fn: Collate function to overwrite dataset collate.

        Returns:
            The configured dataloader.
        """
        sampler = sampler if sampler is not None else None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=sampler,
            # batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self.get_dataloader(self._train_ds)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self._val_ds)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self._test_ds)

    def predict_dataloader(self) -> DataLoader:
        return self.get_dataloader(self._predict_ds)

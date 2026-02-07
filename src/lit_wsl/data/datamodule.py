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
            num_workers = 0
        self.num_workers = num_workers

        self._train_ds = get_dataset(Stage.Training)
        self._val_ds = get_dataset(Stage.Validating)
        self._test_ds = get_dataset(Stage.Testing)
        self._predict_ds = get_dataset(Stage.Predicting)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        pass

    @property
    def train_dataset(self) -> Dataset | None:
        return self._train_ds

    @property
    def val_dataset(self) -> Dataset | None:
        return self._val_ds

    @property
    def test_dataset(self) -> Dataset | None:
        return self._test_ds

    @property
    def predict_dataset(self) -> Dataset | None:
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

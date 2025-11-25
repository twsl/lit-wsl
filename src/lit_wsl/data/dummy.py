from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from lit_wsl.training.stages import Stage

LightningModuleType = TypeVar("LightningModuleType", bound=pl.LightningModule)

InputType = TypeVar("InputType", bound=Tensor)


class DummyDataset(Dataset):
    def __init__(self, stage: Stage, *shapes: tuple[int, ...], num_samples: int = 1000) -> None:
        """Initialize a new dummy dataset.

        Args:
            stage: The stage for which the dataset is being created.
            shapes: The shapes of the tensors to generate for each sample.
            num_samples: The number of samples to generate.

        Example:
            >>> ds = DummyDataset((1, 28, 28), (1,))
            >>> dl = DataLoader(ds, batch_size=7)
            >>> # get first batch
            >>> batch = next(iter(dl))
            >>> x, y = batch
            >>> x.size()
            torch.Size([7, 1, 28, 28])
            >>> y.size()
            torch.Size([7, 1])

        Raises:
            ValueError: If `num_samples` is less than 1.
        """
        super().__init__()
        self.stage = stage
        self.shapes = shapes

        if num_samples < 1:
            raise ValueError("Provide an argument greater than 0 for `num_samples`")

        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        sample = []
        for shape in self.shapes:
            if len(shape) > 0:
                spl = torch.rand(*shape)
                sample.append(spl)
        return tuple(sample)

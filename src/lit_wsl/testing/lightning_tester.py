from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, _utils

from lit_wsl.data.datamodule import DummyDataModule
from lit_wsl.data.dummy import DummyDataset
from lit_wsl.training.stages import Stage

LightningModuleType = TypeVar("LightningModuleType", bound=pl.LightningModule)

InputType = TypeVar("InputType", bound=Tensor | Any)


class ModelTester(Generic[LightningModuleType, InputType]):
    @property
    def batch_size(self) -> int:
        return 2

    @property
    def num_samples(self) -> int:
        return 100

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (3, 64, 64)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1,)

    def compile_options_params(self) -> list[dict[str, Any]]:
        # Instance method that returns params
        return [
            {"mode": mode, "dynamic": dynamic}
            for mode in ["default", "reduce-overhead", "max-autotune"]
            for dynamic in [True, False]
        ]

    def pytest_generate_tests(self, metafunc) -> None:
        # Dynamically parametrize the fixture using the instance method
        if "compile_options" in metafunc.fixturenames:
            params = self.compile_options_params()
            metafunc.parametrize("compile_options", params)

    def compile_options(self, request: Any) -> dict[str, Any]:
        """Fixture to provide compile options for the model."""
        # The param value is set by pytest_generate_tests
        return request.param

    def get_input_sample(self, *args, **kwargs) -> InputType:
        dm = self.get_datamodule()
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        x, _y = batch
        return x

    @abstractmethod
    def lightning_model(self, *args, **kwargs) -> LightningModuleType:
        pass

    def test_lightning_init(self, lightning_model: LightningModuleType) -> None:
        assert lightning_model is not None  # noqa: S101  # nosec B101

    def test_lightning_compile(self, lightning_model: LightningModuleType, compile_options: dict[str, Any]) -> None:
        # lightning_model.compile(**compile_options)
        # assert lightning_model is not None  # noqa: S101
        model = torch.compile(lightning_model, **compile_options)
        assert model is not None  # noqa: S101  # nosec B101

    def test_lightning_save_load(self, lightning_model: LightningModuleType, tmp_path: Path) -> None:
        path = tmp_path / lightning_model.__class__.__name__
        torch.save(lightning_model, path)
        model = torch.load(path, weights_only=False)  # nosec B614
        model.eval()

    def get_dataset(self, stage: Stage, *args, **kwargs) -> Dataset:
        return DummyDataset(stage, self.input_shape, self.output_shape, num_samples=self.num_samples)

    def get_datamodule(self, *args, **kwargs) -> pl.LightningDataModule:
        return DummyDataModule(
            get_dataset=self.get_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch_data: list[InputType]) -> Any:
        value = _utils.collate.default_collate(batch_data)  # pyright: ignore[reportReturnType]
        return value

    def test_boring(self, lightning_model: LightningModuleType, *args, **kwargs) -> None:
        dm = self.get_datamodule()

        self.trainer.fit(lightning_model, datamodule=dm)
        self.trainer.test(lightning_model, datamodule=dm)

        results = self.trainer.predict(lightning_model, datamodule=dm)
        assert results is not None  # noqa: S101  # nosec B101

    @property
    def trainer(self) -> pl.Trainer:
        if not hasattr(self, "_trainer") or self._trainer is None:
            self._trainer = pl.Trainer(
                default_root_dir=Path.cwd(),
                limit_train_batches=1,
                limit_val_batches=1,
                limit_test_batches=1,
                num_sanity_val_steps=0,
                max_epochs=1,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
        return self._trainer

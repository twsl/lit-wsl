from typing import Any

from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import pytest
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.sgd import SGD

from lit_wsl.testing.lightning_tester import ModelTester


class BoringModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def loss(self, batch: Tensor, prediction: Tensor) -> Tensor:
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict:
        feat, targets = batch
        output = self.forward(feat)
        loss = self.loss(targets, output)
        return {"loss": loss}

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def training_step_end(self, training_step_outputs: dict) -> dict:
        return training_step_outputs

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict:
        feat, targets = batch
        output = self.forward(feat)
        loss = self.loss(targets, output)
        return {"x": loss}

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict:
        feat, targets = batch
        output = self.forward(feat)
        loss = self.loss(targets, output)
        self.log("fake_test_acc", loss)
        return {"y": loss}

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        feat, targets = batch
        output = self.forward(feat)
        return output

    def on_predict_batch_end(self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        return super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def configure_optimizers(self) -> tuple[list[SGD], list[CosineAnnealingLR]]:
        optimizer = SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=1024)
        return [optimizer], [lr_scheduler]


class TestBoringModel(ModelTester[BoringModel, torch.Tensor]):
    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 32)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, 2)

    @pytest.fixture()
    def lightning_model(self, *args, **kwargs) -> BoringModel:  # pyright: ignore[reportIncompatibleMethodOverride]
        return BoringModel()

    @pytest.fixture()
    def compile_options(self, request: pytest.FixtureRequest) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # return {"mode": "default", "dynamic": False}
        return request.param

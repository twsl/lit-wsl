import pytest
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """A custom convolutional block sub-module."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class MLPBlock(nn.Module):
    """A custom MLP block sub-module."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class SimpleModel(nn.Module):
    """Simple test model with configurable nested structure."""

    def __init__(self, num_conv_layers: int = 2, num_fc_layers: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = ConvBlock(3, 64, num_layers=num_conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = MLPBlock(64, hidden_dim, 10, num_layers=num_fc_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


class RenamedModel(nn.Module):
    """Same architecture with renamed layers and different structure."""

    def __init__(self, num_conv_layers: int = 2, num_fc_layers: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_extractor = ConvBlock(3, 64, num_layers=num_conv_layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = MLPBlock(64, hidden_dim, 10, num_layers=num_fc_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.head(x)


@pytest.fixture
def simple_model_class() -> type[SimpleModel]:
    """Fixture that returns the SimpleModel class."""
    return SimpleModel


@pytest.fixture
def renamed_model_class() -> type[RenamedModel]:
    """Fixture that returns the RenamedModel class."""
    return RenamedModel


@pytest.fixture
def simple_model() -> SimpleModel:
    """Fixture that returns an instance of SimpleModel."""
    return SimpleModel()


@pytest.fixture
def renamed_model() -> RenamedModel:
    """Fixture that returns an instance of RenamedModel."""
    return RenamedModel()

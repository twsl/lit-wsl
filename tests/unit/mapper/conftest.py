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


class ComplexModel(nn.Module):
    """Complex model with multiple levels of nested modules."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Multi-stage backbone
        self.backbone = nn.Module()
        self.backbone.stage1 = ConvBlock(3, 64, num_layers=2)
        self.backbone.stage2 = ConvBlock(64, 128, num_layers=2)
        self.backbone.stage3 = ConvBlock(128, 256, num_layers=3)

        # Feature pyramid neck
        self.neck = nn.Module()
        self.neck.fpn_layer1 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.neck.fpn_layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Classification head
        self.head = nn.Module()
        self.head.pool = nn.AdaptiveAvgPool2d(1)
        self.head.classifier = MLPBlock(64, 128, num_classes, num_layers=3)

    def forward(self, x: Tensor) -> Tensor:
        # Backbone forward
        x = self.backbone.stage1(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)

        # Neck forward
        x = self.neck.fpn_layer1(x)
        x = self.neck.fpn_layer2(x)

        # Head forward
        x = self.head.pool(x)
        x = x.flatten(1)
        return self.head.classifier(x)


class ComplexModelReimplemented(nn.Module):
    """Reimplementation with modules reorganized into different nested structures."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Feature pyramid with reorganized modules
        self.feature_pyramid = nn.Module()

        # Early features (combines original stage1 and stage2)
        self.feature_pyramid.early_features = nn.Module()
        self.feature_pyramid.early_features.conv_block_1 = ConvBlock(3, 64, num_layers=2)
        self.feature_pyramid.early_features.conv_block_2 = ConvBlock(64, 128, num_layers=2)

        # Late features (combines original stage3 and fpn_layer1)
        self.feature_pyramid.late_features = nn.Module()
        self.feature_pyramid.late_features.conv_block_3 = ConvBlock(128, 256, num_layers=3)
        self.feature_pyramid.late_features.reduction = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Fusion layer (original fpn_layer2)
        self.feature_pyramid.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Prediction head (reorganized head components)
        self.prediction_head = nn.Module()
        self.prediction_head.pool = nn.AdaptiveAvgPool2d(1)
        self.prediction_head.classifier = MLPBlock(64, 128, num_classes, num_layers=3)

    def forward(self, x: Tensor) -> Tensor:
        # Early features
        x = self.feature_pyramid.early_features.conv_block_1(x)
        x = self.feature_pyramid.early_features.conv_block_2(x)

        # Late features
        x = self.feature_pyramid.late_features.conv_block_3(x)
        x = self.feature_pyramid.late_features.reduction(x)

        # Fusion
        x = self.feature_pyramid.fusion(x)

        # Prediction
        x = self.prediction_head.pool(x)
        x = x.flatten(1)
        return self.prediction_head.classifier(x)


@pytest.fixture
def complex_model_class() -> type[ComplexModel]:
    """Fixture that returns the ComplexModel class."""
    return ComplexModel


@pytest.fixture
def complex_model_reimplemented_class() -> type[ComplexModelReimplemented]:
    """Fixture that returns the ComplexModelReimplemented class."""
    return ComplexModelReimplemented


@pytest.fixture
def complex_model() -> ComplexModel:
    """Fixture that returns an instance of ComplexModel."""
    return ComplexModel()


@pytest.fixture
def complex_model_reimplemented() -> ComplexModelReimplemented:
    """Fixture that returns an instance of ComplexModelReimplemented."""
    return ComplexModelReimplemented()

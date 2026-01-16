from pathlib import Path

import torch
from torch import nn

from lit_wsl.models.weight_mapper import ParameterInfo, WeightMapper


def test_parameter_info() -> None:
    """Test ParameterInfo creation."""
    tensor = torch.randn(10, 20)
    info = ParameterInfo("model.layer.weight", tensor)

    assert info.name == "model.layer.weight"
    assert info.shape == (10, 20)
    assert info.depth == 3
    assert info.param_name == "weight"
    assert info.module_path == "model.layer"
    assert "weight" in info.tokens
    assert "layer" in info.tokens


def test_weight_mapper_initialization(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test WeightMapper initialization."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)

    # Each model has: 2 conv layers (weight+bias+bn_weight+bn_bias each) + 2 fc layers (weight+bias each)
    # = 8 conv params + 4 fc params = 12 total
    assert len(mapper.source_params) == 12
    assert len(mapper.target_params) == 12
    assert len(mapper.target_by_shape) > 0


def test_suggest_mapping_best_match(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test basic mapping suggestion."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)
    mapping = mapper.suggest_mapping(threshold=0.5)

    # Should find matches for most parameters
    assert len(mapping) >= 10  # At least most params should match

    # Check some specific nested mappings exist
    assert any("encoder.layers" in key for key in mapping)
    assert any("classifier.layers" in key for key in mapping)


def test_suggest_mapping_shape_only(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test shape-only mapping strategy."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)
    mapping = mapper.suggest_mapping(strategy="shape_only")

    # Should match all parameters by shape since architectures are identical
    assert len(mapping) == 12


def test_suggest_mapping_conservative(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test conservative mapping strategy."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)
    mapping = mapper.suggest_mapping(strategy="conservative")

    # Conservative strategy may match fewer parameters due to high threshold
    assert len(mapping) <= 12
    assert len(mapping) > 0  # Should match at least some


def test_get_unmatched(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test unmatched parameter detection."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)
    mapper.suggest_mapping(threshold=0.99)  # Very high threshold

    unmatched = mapper.get_unmatched()

    assert "source" in unmatched
    assert "target" in unmatched
    assert isinstance(unmatched["source"], list)
    assert isinstance(unmatched["target"], list)


def test_get_mapping_dict(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test getting mapping dictionary."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)
    mapping = mapper.suggest_mapping()

    mapping_dict = mapper.get_mapping_dict()

    assert isinstance(mapping_dict, dict)
    assert len(mapping_dict) == len(mapping)


def test_get_mapping_with_scores(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test getting mapping with scores."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)
    mapper.suggest_mapping()

    mappings = mapper.get_mapping_with_scores()

    assert isinstance(mappings, list)
    assert all(len(item) == 3 for item in mappings)  # (source, target, score)
    assert all(0.0 <= item[2] <= 1.0 for item in mappings)  # Scores in valid range


def test_shape_matching(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test shape matching logic."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)

    # Get info for first conv weights (should have same shape)
    source_conv = mapper.source_params["encoder.layers.0.weight"]
    target_conv = mapper.target_params["feature_extractor.layers.0.weight"]

    score = mapper._compute_shape_score(source_conv, target_conv)
    assert score == 1.0  # Exact match


def test_name_similarity(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test name similarity computation."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)

    # Similar names should have higher scores
    source_fc = mapper.source_params["classifier.layers.0.weight"]
    target_fc = mapper.target_params["head.layers.0.weight"]

    score = mapper._compute_name_similarity(source_fc, target_fc)
    assert 0.0 <= score <= 1.0
    assert score > 0.0  # Should have some similarity due to "weight" and "layers" tokens


def test_custom_weights(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test custom scoring weights."""
    source = simple_model
    target = renamed_model

    mapper = WeightMapper(source, target)

    custom_weights = {"shape": 0.7, "name": 0.2, "hierarchy": 0.1}
    mapping = mapper.suggest_mapping(weights=custom_weights)

    # Should still find matches
    assert len(mapping) > 0


def test_exact_match_names(simple_model_class: nn.Module) -> None:
    """Test that exact name matches get high scores."""
    source = simple_model_class()
    # Create target with same names
    target = simple_model_class()

    mapper = WeightMapper(source, target)
    mapping = mapper.suggest_mapping()

    # All parameters should match with their exact counterparts
    assert len(mapping) == 12
    for source_name in mapping:
        assert mapping[source_name] == source_name  # Exact name match


def test_from_state_dict(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test creating WeightMapper from state dict."""
    # Create a model and extract its state dict
    source = simple_model
    state_dict = source.state_dict()

    # Create target model
    target = renamed_model

    # Create mapper from state dict
    mapper = WeightMapper.from_state_dict(state_dict, target)

    # Source from state dict includes BatchNorm buffers (18 total)
    # Target from module has only parameters (12 total)
    assert len(mapper.source_params) == 18
    assert len(mapper.target_params) == 12

    # Should be able to create mapping
    mapping = mapper.suggest_mapping()
    assert len(mapping) > 0

    # Source module should be None when created from state dict
    assert mapper.source_module is None
    assert mapper.target_module is target


def test_from_state_dict_nested(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test from_state_dict with nested checkpoint structure."""
    source = simple_model
    # Simulate Lightning checkpoint structure
    checkpoint = {
        "state_dict": source.state_dict(),
        "epoch": 10,
        "global_step": 100,
    }

    target = renamed_model

    # Extract state_dict manually
    mapper = WeightMapper.from_state_dict(checkpoint["state_dict"], target)

    # Source from state dict includes BatchNorm buffers (18)
    # Target from module has only parameters (12)
    assert len(mapper.source_params) == 18
    assert len(mapper.target_params) == 12

    mapping = mapper.suggest_mapping()
    assert len(mapping) > 0


def test_from_checkpoint(tmp_path: Path, simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test creating WeightMapper from checkpoint file using from_checkpoint method."""
    # Create a model
    source = simple_model

    # Save checkpoint to temporary file
    checkpoint_path = tmp_path / "checkpoint.pth"

    # Save as typical PyTorch checkpoint
    torch.save(
        {
            "state_dict": source.state_dict(),
            "epoch": 5,
            "optimizer_state": {},
        },
        checkpoint_path,
    )

    # Create target model
    target = renamed_model

    # Create mapper from checkpoint file
    mapper = WeightMapper.from_checkpoint(checkpoint_path, target)

    # Source from state dict includes BatchNorm buffers (18)
    # Target from module has only parameters (12)
    assert len(mapper.source_params) == 18
    assert len(mapper.target_params) == 12

    # Should be able to create mapping
    mapping = mapper.suggest_mapping()
    assert len(mapping) > 0

    # Source module should be None when created from checkpoint
    assert mapper.source_module is None
    assert mapper.target_module is target


def test_from_checkpoint_simple_state_dict(tmp_path: Path, simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test from_checkpoint with simple state dict (no nested structure)."""
    source = simple_model

    # Save checkpoint to temporary file
    checkpoint_path = tmp_path / "simple_checkpoint.pth"

    # Save as simple state dict (no wrapper)
    torch.save(source.state_dict(), checkpoint_path)

    # Create target model
    target = renamed_model

    # Create mapper from checkpoint file
    mapper = WeightMapper.from_checkpoint(checkpoint_path, target)

    # Source from state dict includes BatchNorm buffers (18)
    # Target from module has only parameters (12)
    assert len(mapper.source_params) == 18
    assert len(mapper.target_params) == 12

    # Should be able to create mapping
    mapping = mapper.suggest_mapping()
    assert len(mapping) > 0


def test_different_model_configurations(simple_model_class: nn.Module, renamed_model_class: nn.Module) -> None:
    """Test mapping between models with different constructor parameters."""
    # Source model with 3 conv layers and 3 fc layers
    source = simple_model_class(num_conv_layers=3, num_fc_layers=3, hidden_dim=256)

    # Target model with 2 conv layers and 2 fc layers (different configuration)
    target = renamed_model_class(num_conv_layers=2, num_fc_layers=2, hidden_dim=128)

    mapper = WeightMapper(source, target)

    # Source has more parameters due to extra layers
    assert len(mapper.source_params) > len(mapper.target_params)

    # Should still find some matches for compatible layers
    mapping = mapper.suggest_mapping(threshold=0.5)
    assert len(mapping) > 0

    # Verify unmatched parameters are detected
    unmatched = mapper.get_unmatched()
    assert len(unmatched["source"]) > 0  # Source has extra layers
    assert len(unmatched["target"]) >= 0  # Target may have unmatched params

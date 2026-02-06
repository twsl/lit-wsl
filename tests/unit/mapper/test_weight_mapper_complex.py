"""Tests for weight mapping with complex nested models where modules are reorganized."""

from pathlib import Path

import pytest
import torch
from torch import nn

from lit_wsl.mapper.weight_mapper import WeightMapper


def test_complex_model_initialization(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test WeightMapper initialization with complex nested models."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)

    # Verify both models have the same number of parameters and buffers
    assert len(mapper.source_params) == len(mapper.target_params)
    # Count should include all Conv2d, BatchNorm2d, and Linear parameters and buffers
    assert len(mapper.source_params) > 0
    assert len(mapper.target_params) > 0


def test_complex_model_shape_compatibility(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test that complex models have compatible parameter shapes."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)

    # Verify shape distribution is identical between models
    source_shapes = sorted([info.shape for info in mapper.source_params.values()])
    target_shapes = sorted([info.shape for info in mapper.target_params.values()])

    assert source_shapes == target_shapes, "Models should have matching parameter shapes"


def test_complex_model_suggest_mapping(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test basic mapping suggestion for complex nested models."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)
    result = mapper.suggest_mapping(threshold=0.5)

    mapping = result.get_mapping()

    unmatched = result.get_unmatched()

    # Should find matches for most parameters despite reorganization
    assert len(mapping) > 0
    assert isinstance(mapping, dict)
    assert isinstance(unmatched, dict)

    # With the same architecture, most if not all parameters should map
    total_params = len(mapper.source_params)
    assert len(mapping) >= total_params * 0.8, "At least 80% of parameters should map"


def test_complex_model_exact_mapping_coverage(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test that mapping covers all parameters without duplicates."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)
    result = mapper.suggest_mapping(threshold=0.5)

    mapping = result.get_mapping()

    unmatched = result.get_unmatched()

    # Check no duplicate targets
    target_params = list(mapping.values())
    assert len(target_params) == len(set(target_params)), "No duplicate target mappings"

    # Verify unmatched parameters are disjoint from mapped ones
    assert all(param not in mapping for param in unmatched)


def test_complex_model_nested_module_mapping(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test that parameters from deeply nested modules are correctly mapped."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)
    result = mapper.suggest_mapping(threshold=0.5)

    mapping = result.get_mapping()

    # Check that backbone.stage1 maps to feature_pyramid.early_features.conv_block_1
    stage1_params = [k for k in mapping.keys() if "backbone.stage1" in k]
    assert len(stage1_params) > 0, "Should find stage1 parameters"

    for param in stage1_params:
        target_param = mapping[param]
        assert "feature_pyramid.early_features.conv_block_1" in target_param, (
            f"Backbone.stage1 param {param} should map to early_features.conv_block_1, got {target_param}"
        )

    # Check that backbone.stage2 maps to feature_pyramid.early_features.conv_block_2
    stage2_params = [k for k in mapping.keys() if "backbone.stage2" in k]
    assert len(stage2_params) > 0, "Should find stage2 parameters"

    for param in stage2_params:
        target_param = mapping[param]
        assert "feature_pyramid.early_features.conv_block_2" in target_param


def test_complex_model_cross_module_mapping(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test mapping of modules that moved from one parent to another."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)
    result = mapper.suggest_mapping(threshold=0.5)

    mapping = result.get_mapping()

    # backbone.stage3 -> feature_pyramid.late_features.conv_block_3
    stage3_params = [k for k in mapping if "backbone.stage3" in k]
    assert len(stage3_params) > 0, "Should find stage3 parameters"

    for param in stage3_params:
        target_param = mapping[param]
        assert "feature_pyramid.late_features.conv_block_3" in target_param

    # neck.fpn_layer1 -> feature_pyramid.late_features.reduction
    fpn1_params = [k for k in mapping if "neck.fpn_layer1" in k]
    assert len(fpn1_params) > 0, "Should find fpn_layer1 parameters"

    for param in fpn1_params:
        target_param = mapping[param]
        assert "feature_pyramid.late_features.reduction" in target_param

    # neck.fpn_layer2 -> feature_pyramid.fusion
    fpn2_params = [k for k in mapping if "neck.fpn_layer2" in k]
    assert len(fpn2_params) > 0, "Should find fpn_layer2 parameters"

    for param in fpn2_params:
        target_param = mapping[param]
        assert "feature_pyramid.fusion" in target_param


def test_complex_model_head_mapping(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test that head components are correctly mapped despite renaming."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)
    result = mapper.suggest_mapping(threshold=0.5)

    mapping = result.get_mapping()

    # head.pool -> prediction_head.pool
    pool_params = [k for k in mapping.keys() if "head.pool" in k]
    # AdaptiveAvgPool2d may not have parameters, only buffers

    # head.classifier -> prediction_head.classifier
    classifier_params = [k for k in mapping.keys() if "head.classifier" in k]
    assert len(classifier_params) > 0, "Should find classifier parameters"

    for param in classifier_params:
        target_param = mapping[param]
        assert "prediction_head.classifier" in target_param


def test_complex_model_mapping_with_scores(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test mapping with similarity scores for complex nested models."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)
    result = mapper.suggest_mapping(threshold=0.5)

    mappings = result.get_mapping_with_scores()

    unmatched = result.get_unmatched()

    assert isinstance(mappings, dict)
    assert isinstance(unmatched, dict)

    # Each mapping should include a score
    for src, value in mappings.items():
        assert isinstance(value, tuple)
        assert len(value) == 2
        target_name, score = value
        assert isinstance(target_name, str)
        assert 0.0 <= score <= 1.0
        # For architecturally identical models, scores should be reasonably high
        assert score >= 0.5, f"Score for {src} -> {target_name} should be >= 0.5, got {score:.4f}"


def test_complex_model_high_threshold(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test mapping with high threshold on complex models."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)
    result = mapper.suggest_mapping(threshold=0.95)

    mapping = result.get_mapping()

    unmatched = result.get_unmatched()

    # High threshold may be strict, but shape matches should still work
    assert isinstance(mapping, dict)
    assert isinstance(unmatched, dict)

    # Verify all mapped parameters have matching shapes
    for source_name, target_name in mapping.items():
        source_info = mapper.source_params[source_name]
        # target_name is just a string here (not a tuple) since return_scores=False by default
        target_info = mapper.target_params[target_name]
        assert source_info.shape == target_info.shape


def test_complex_model_weight_transfer(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test actual weight transfer between complex nested models."""
    source = complex_model
    target = complex_model_reimplemented

    # Initialize with different values
    with torch.no_grad():
        for param in source.parameters():
            param.fill_(1.0)
        for param in target.parameters():
            param.fill_(0.0)

    # Create mapping
    mapper = WeightMapper(source, target)
    result = mapper.suggest_mapping(threshold=0.5)

    mapping = result.get_mapping()

    # Transfer weights manually using the mapping
    source_state = source.state_dict()
    target_state = target.state_dict()

    # mapping values are strings (not tuples) when return_scores=False
    for source_name, target_name in mapping.items():
        if source_name in source_state and target_name in target_state:
            target_state[target_name] = source_state[source_name].clone()

    target.load_state_dict(target_state, strict=False)

    # Verify weights were transferred for mapped parameters
    for source_name, target_name in mapping.items():
        if source_name in source_state and target_name in target_state:
            source_tensor = source_state[source_name]
            target_tensor = target_state[target_name]
            assert torch.allclose(source_tensor, target_tensor), (
                f"Weight transfer failed for {source_name} -> {target_name}"
            )


def test_complex_model_from_state_dict(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test creating WeightMapper from state dict with complex models."""
    source = complex_model
    state_dict = source.state_dict()

    target = complex_model_reimplemented

    # Create mapper from state dict
    mapper = WeightMapper.from_state_dict(state_dict, target)

    assert len(mapper.source_params) == len(state_dict)
    assert len(mapper.target_params) > 0
    assert mapper.source_module is None
    assert mapper.target_module is target

    # Should still be able to create mappings
    result = mapper.suggest_mapping(threshold=0.5)

    mapping = result.get_mapping()

    unmatched = result.get_unmatched()
    assert len(mapping) > 0


def test_complex_model_from_checkpoint(
    tmp_path: Path, complex_model: nn.Module, complex_model_reimplemented: nn.Module
) -> None:
    """Test creating WeightMapper from checkpoint file with complex models."""
    source = complex_model

    # Save checkpoint
    checkpoint_path = tmp_path / "complex_checkpoint.pth"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "epoch": 10,
            "optimizer_state": {},
        },
        checkpoint_path,
    )

    target = complex_model_reimplemented

    # Load and create mapper
    checkpoint = torch.load(checkpoint_path)  # nosec B614
    mapper = WeightMapper.from_state_dict(checkpoint["state_dict"], target)

    assert len(mapper.source_params) > 0
    assert len(mapper.target_params) > 0

    result = mapper.suggest_mapping(threshold=0.5)

    mapping = result.get_mapping()
    assert len(mapping) > 0


def test_complex_model_unmatched_detection(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test unmatched parameter detection with very high threshold."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)
    mapper.suggest_mapping(threshold=0.99)  # Very strict threshold

    unmatched = mapper.get_unmatched()

    assert "source" in unmatched
    assert "target" in unmatched
    assert isinstance(unmatched["source"], list)
    assert isinstance(unmatched["target"], list)

    # With very high threshold, there should be some unmatched (unless perfect name match)
    # The exact count depends on the similarity scoring


def test_complex_model_custom_weights(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test custom scoring weights with complex nested models."""
    source = complex_model
    target = complex_model_reimplemented

    mapper = WeightMapper(source, target)

    # Prioritize shape matching heavily
    custom_weights = {"shape": 0.9, "name": 0.05, "hierarchy": 0.05}
    result = mapper.suggest_mapping(weights=custom_weights, threshold=0.5)

    mapping = result.get_mapping()

    # Should still find matches based on shape
    assert len(mapping) > 0

    # Verify all matches have identical shapes
    for source_name, target_name in mapping.items():
        source_info = mapper.source_params[source_name]
        # target_name is a string here (not a tuple) since return_scores=False by default
        target_info = mapper.target_params[target_name]
        assert source_info.shape == target_info.shape


def test_complex_model_forward_equivalence(complex_model: nn.Module, complex_model_reimplemented: nn.Module) -> None:
    """Test that mapped weights produce equivalent forward passes."""
    source = complex_model
    target = complex_model_reimplemented

    # Set to eval mode to disable dropout, etc.
    source.eval()
    target.eval()

    # Initialize source with random weights
    torch.manual_seed(42)
    for param in source.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)

    # Create mapping and transfer weights
    mapper = WeightMapper(source, target)
    result = mapper.suggest_mapping(threshold=0.5)

    mapping = result.get_mapping()

    source_state = source.state_dict()
    target_state = target.state_dict()

    for source_name, target_name in mapping.items():
        if source_name in source_state and target_name in target_state:
            target_state[target_name] = source_state[source_name].clone()

    target.load_state_dict(target_state, strict=False)

    # Test forward pass
    torch.manual_seed(123)
    test_input = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        source_output = source(test_input)
        target_output = target(test_input)

    # Outputs should be very close if mapping is correct
    assert source_output.shape == target_output.shape
    assert torch.allclose(source_output, target_output, atol=1e-5), (
        "Forward passes should produce equivalent outputs after weight transfer"
    )

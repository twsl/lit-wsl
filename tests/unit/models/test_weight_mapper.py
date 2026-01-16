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


def test_execution_order_tracking(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test that execution order tracking works when dummy_input is provided."""
    # Without dummy_input - no execution order
    mapper_no_order = WeightMapper(simple_model, renamed_model)
    assert all(p.execution_order is None for p in mapper_no_order.source_params.values())
    assert all(p.execution_order is None for p in mapper_no_order.target_params.values())

    # With dummy_input - execution order tracked
    dummy_input = torch.randn(1, 3, 32, 32)
    mapper_with_order = WeightMapper(simple_model, renamed_model, dummy_input=dummy_input)

    # Should have execution order for most parameters (those in modules that get executed)
    source_with_order = sum(1 for p in mapper_with_order.source_params.values() if p.execution_order is not None)
    target_with_order = sum(1 for p in mapper_with_order.target_params.values() if p.execution_order is not None)

    assert source_with_order > 0, "Should track execution order for at least some source parameters"
    assert target_with_order > 0, "Should track execution order for at least some target parameters"


def test_execution_order_improves_matching_scores(simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test that execution order tracking can improve matching scores."""
    dummy_input = torch.randn(1, 3, 32, 32)

    # Create two mappers: one with execution order, one without
    mapper_no_order = WeightMapper(simple_model, renamed_model)
    mapper_with_order = WeightMapper(simple_model, renamed_model, dummy_input=dummy_input)

    # Generate mappings
    mapping_no_order = mapper_no_order.suggest_mapping(threshold=0.5)
    mapping_with_order = mapper_with_order.suggest_mapping(threshold=0.5)

    # Both should find the same number of mappings (same architectures)
    assert len(mapping_no_order) == len(mapping_with_order)

    # Get scores for comparison
    scores_no_order = mapper_no_order.get_mapping_with_scores()
    scores_with_order = mapper_with_order.get_mapping_with_scores()

    # Convert to dicts for easier comparison
    score_dict_no_order = {src: score for src, _, score in scores_no_order}
    score_dict_with_order = {src: score for src, _, score in scores_with_order}

    # Count how many scores improved, stayed the same, or got worse
    improved = 0
    same = 0
    worse = 0

    for src in score_dict_no_order:
        if src in score_dict_with_order:
            diff = score_dict_with_order[src] - score_dict_no_order[src]
            if abs(diff) < 1e-6:  # Essentially the same
                same += 1
            elif diff > 0:
                improved += 1
            else:
                worse += 1

    print(f"\nExecution order impact: {improved} improved, {same} same, {worse} worse")

    # For models with renamed but structurally similar architectures,
    # execution order should help or at least not hurt
    assert improved + same >= worse, "Execution order should improve or maintain scores"

    # Average scores should be at least as good
    avg_score_no_order = sum(score_dict_no_order.values()) / len(score_dict_no_order)
    avg_score_with_order = sum(score_dict_with_order.values()) / len(score_dict_with_order)

    print(f"Average score without order: {avg_score_no_order:.4f}")
    print(f"Average score with order: {avg_score_with_order:.4f}")

    # With execution order, average score should be at least as good
    assert avg_score_with_order >= avg_score_no_order - 1e-6


def test_execution_order_from_checkpoint(tmp_path: Path, simple_model: nn.Module, renamed_model: nn.Module) -> None:
    """Test execution order tracking with from_checkpoint method."""
    # Save source model
    checkpoint_path = tmp_path / "model.pth"
    torch.save({"state_dict": simple_model.state_dict()}, checkpoint_path)

    # Without dummy_input
    mapper_no_order = WeightMapper.from_checkpoint(checkpoint_path, renamed_model)
    assert all(p.execution_order is None for p in mapper_no_order.target_params.values())

    # With dummy_input
    dummy_input = torch.randn(1, 3, 32, 32)
    mapper_with_order = WeightMapper.from_checkpoint(checkpoint_path, renamed_model, dummy_input=dummy_input)

    target_with_order = sum(1 for p in mapper_with_order.target_params.values() if p.execution_order is not None)
    assert target_with_order > 0, "Should track execution order when dummy_input provided"


def test_execution_order_consistency(simple_model: nn.Module) -> None:
    """Test that execution order is consistent across multiple runs."""
    dummy_input = torch.randn(1, 3, 32, 32)

    # Create mapper twice with same input
    mapper1 = WeightMapper(simple_model, simple_model, dummy_input=dummy_input)
    mapper2 = WeightMapper(simple_model, simple_model, dummy_input=dummy_input)

    # Execution order should be the same
    for param_name in mapper1.source_params:
        order1 = mapper1.source_params[param_name].execution_order
        order2 = mapper2.source_params[param_name].execution_order
        assert order1 == order2, f"Execution order should be consistent for {param_name}"


def test_execution_order_with_complex_architectures(
    simple_model_class: nn.Module, renamed_model_class: nn.Module
) -> None:
    """Test execution order with more complex model configurations."""
    # Create larger models
    source = simple_model_class(num_conv_layers=3, num_fc_layers=3, hidden_dim=256)
    target = renamed_model_class(num_conv_layers=3, num_fc_layers=3, hidden_dim=256)

    dummy_input = torch.randn(1, 3, 32, 32)

    # With execution order tracking
    mapper = WeightMapper(source, target, dummy_input=dummy_input)
    mapping = mapper.suggest_mapping(threshold=0.5)

    # Should find good matches even with larger architectures
    assert len(mapping) > 15, "Should find many matches in larger identical architectures"

    # Get scores to verify quality
    scores = mapper.get_mapping_with_scores()
    avg_score = sum(s for _, _, s in scores) / len(scores)

    print(f"\nComplex architecture average score: {avg_score:.4f}")
    assert avg_score > 0.6, "Should maintain good matching quality with execution order"

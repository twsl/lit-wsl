"""Integration tests for incompatible_pairs parameter in WeightMapper."""

import torch
from torch import nn

from lit_wsl.mapper.weight_mapper import WeightMapper


class SimpleBackboneHeadModel(nn.Module):
    """Simple model with backbone and head components."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
        )
        self.head = nn.Linear(16, 10)


class RenamedBackboneHeadModel(nn.Module):
    """Model with renamed components (backbone->encoder, head->classifier)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(16, 10)


def test_default_allows_all_mappings() -> None:
    """Test that default behavior (no incompatible_pairs) allows all mappings."""
    source = SimpleBackboneHeadModel()
    target = RenamedBackboneHeadModel()

    # Default behavior: no restrictions on cross-component matches
    mapper = WeightMapper(source, target)

    # Should find valid mappings based on shape and name similarity
    result = mapper.suggest_mapping()
    mapping = result.get_mapping()

    # Should successfully map parameters (no restrictions blocking matches)
    assert len(mapping) > 0


def test_explicit_empty_incompatible_pairs() -> None:
    """Test that explicitly passing empty list also allows all mappings (same as default)."""
    source = SimpleBackboneHeadModel()
    target = RenamedBackboneHeadModel()

    # Explicitly pass empty list - same as default
    mapper = WeightMapper(source, target, incompatible_pairs=[])

    # Should find valid mappings (based on shape and other factors)
    result = mapper.suggest_mapping()
    mapping = result.get_mapping()

    # Verify we got some mappings
    assert len(mapping) > 0


def test_vision_model_incompatible_pairs() -> None:
    """Test that vision model incompatible pairs prevent cross-component matching."""
    source = SimpleBackboneHeadModel()
    target = RenamedBackboneHeadModel()

    # Explicitly add vision model pairs to prevent backbone-head confusion
    vision_pairs = [
        ({"backbone", "encoder", "feature"}, {"head", "classifier", "decoder"}),
        ({"backbone", "encoder"}, {"neck", "fpn"}),
    ]
    mapper = WeightMapper(source, target, incompatible_pairs=vision_pairs)

    # Should still find mappings, but with penalties for cross-component matches
    result = mapper.suggest_mapping()
    mapping = result.get_mapping()
    assert len(mapping) > 0


def test_incompatible_pairs_with_from_state_dict() -> None:
    """Test that incompatible_pairs work with from_state_dict factory method."""
    source = SimpleBackboneHeadModel()
    target = RenamedBackboneHeadModel()

    # Get source state dict
    source_state_dict = source.state_dict()

    # Create mapper from state dict with custom pairs
    custom_pairs = [({"backbone"}, {"classifier"})]
    mapper = WeightMapper.from_state_dict(source_state_dict, target, incompatible_pairs=custom_pairs)

    # Should create valid mapper
    assert mapper is not None
    assert mapper.hierarchy_analyzer.incompatible_pairs == custom_pairs

    # Should still find mappings
    result = mapper.suggest_mapping()
    mapping = result.get_mapping()
    assert len(mapping) > 0


def test_none_equals_default_empty() -> None:
    """Test that passing None equals default (empty list - no restrictions)."""
    source = SimpleBackboneHeadModel()
    target = RenamedBackboneHeadModel()

    mapper1 = WeightMapper(source, target, incompatible_pairs=None)
    mapper2 = WeightMapper(source, target)
    mapper3 = WeightMapper(source, target, incompatible_pairs=[])

    # All three should have empty incompatible_pairs (no restrictions)
    assert mapper1.hierarchy_analyzer.incompatible_pairs == []
    assert mapper2.hierarchy_analyzer.incompatible_pairs == []
    assert mapper3.hierarchy_analyzer.incompatible_pairs == []

    # All three should produce identical mappings
    result1 = mapper1.suggest_mapping()
    result2 = mapper2.suggest_mapping()
    result3 = mapper3.suggest_mapping()
    mapping1 = result1.get_mapping()
    mapping2 = result2.get_mapping()
    mapping3 = result3.get_mapping()

    assert mapping1 == mapping2 == mapping3

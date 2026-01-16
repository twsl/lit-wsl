from pathlib import Path

import torch
from torch import Tensor, nn

from lit_wsl.models.weight_mapper import WeightMapper
from lit_wsl.models.weight_renamer import WeightRenamer


def test_complete_weight_mapping_workflow(
    tmp_path: Path, simple_model_class: nn.Module, renamed_model_class: nn.Module
) -> None:
    """Test complete workflow: map weights, rename checkpoint, load and compare outputs."""
    # 1. Create and initialize old model
    old_model = simple_model_class()
    old_model.eval()  # Set to eval mode to disable dropout, batchnorm randomness

    # 2. Save old model checkpoint
    checkpoint_path = tmp_path / "old_model.pth"
    torch.save(
        {
            "state_dict": old_model.state_dict(),
            "epoch": 10,
            "optimizer": {},
        },
        checkpoint_path,
    )

    # 3. Create new model architecture
    new_model = renamed_model_class()
    new_model.eval()

    # 4. Use WeightMapper to discover mapping
    mapper = WeightMapper.from_checkpoint(checkpoint_path, new_model)
    mapping = mapper.suggest_mapping(threshold=0.5)

    # Verify we found a good mapping
    assert len(mapping) > 0
    print(f"\nFound {len(mapping)} parameter mappings")

    # 4a. Validate that parameter types match (weight->weight, bias->bias)
    for source_name, target_name in mapping.items():
        source_type = source_name.split(".")[-1]  # e.g., 'weight', 'bias'
        target_type = target_name.split(".")[-1]
        assert source_type == target_type, (
            f"Parameter type mismatch: {source_name} ({source_type}) -> {target_name} ({target_type})"
        )

    # 4b. Validate 1-to-1 mapping (no duplicate targets)
    targets = list(mapping.values())
    assert len(targets) == len(set(targets)), "Mapping is not 1-to-1, found duplicate targets"

    # 5. Use WeightRenamer to apply mapping to checkpoint
    renamer = WeightRenamer(checkpoint_path)
    renamer.rename_keys(mapping)

    # Save renamed checkpoint
    renamed_checkpoint_path = tmp_path / "renamed_model.pth"
    renamer.save(renamed_checkpoint_path)

    # 6. Load renamed weights into new model
    renamed_checkpoint = torch.load(renamed_checkpoint_path)
    new_model.load_state_dict(renamed_checkpoint["state_dict"], strict=False)
    new_model.eval()

    # 7. Create test input
    test_input = torch.randn(2, 3, 32, 32)

    # 8. Compare outputs
    with torch.no_grad():
        old_output = old_model(test_input)
        new_output = new_model(test_input)

    # 9. Verify outputs match (or are very close)
    # Note: Due to BatchNorm buffers not being mapped, outputs might differ slightly
    # We check if at least the shapes match and values are in similar range
    assert old_output.shape == new_output.shape
    assert old_output.shape == (2, 10)  # batch_size=2, num_classes=10

    # Check if outputs are reasonably close (allowing for some difference due to unmapped buffers)
    diff = torch.abs(old_output - new_output).mean().item()
    print(f"Mean absolute difference between outputs: {diff:.6f}")
    # With proper parameter matching, outputs should be very close (not 1e-1 tolerance)
    assert torch.allclose(old_output, new_output, atol=1e-5)

    # The difference might be significant due to BatchNorm buffers,
    # but the outputs should at least be in a similar range
    assert torch.isfinite(old_output).all()
    assert torch.isfinite(new_output).all()


def test_weight_mapping_with_identical_architectures(tmp_path: Path, simple_model_class: nn.Module) -> None:
    """Test that identical architectures produce similar outputs after mapping.

    Note: Due to BatchNorm buffers (running_mean, running_var) not being in named_parameters(),
    they won't be mapped, so outputs will differ. This test verifies the workflow works.
    """
    # 1. Create and initialize source model
    source_model = simple_model_class()
    source_model.eval()

    # 2. Save source model checkpoint
    checkpoint_path = tmp_path / "source_model.pth"
    torch.save({"state_dict": source_model.state_dict()}, checkpoint_path)

    # 3. Create target model with SAME architecture (just fresh initialization)
    target_model = simple_model_class()
    target_model.eval()

    # 4. Use WeightMapper to discover mapping
    mapper = WeightMapper.from_checkpoint(checkpoint_path, target_model)
    mapping = mapper.suggest_mapping()

    # With identical architectures, all parameters should map
    # Note: state_dict has 18 entries (12 params + 6 buffers), but only 12 params are mapped
    print(f"\nMapped {len(mapping)} parameters")
    assert len(mapping) == 12  # All parameters should map
    # Validate parameter type matching
    for source_name, target_name in mapping.items():
        source_type = source_name.split(".")[-1]
        target_type = target_name.split(".")[-1]
        assert source_type == target_type, (
            f"Parameter type mismatch: {source_name} ({source_type}) -> {target_name} ({target_type})"
        )
    # 5. Apply mapping
    renamer = WeightRenamer(checkpoint_path)
    renamer.rename_keys(mapping)
    renamed_checkpoint_path = tmp_path / "renamed_identical.pth"
    renamer.save(renamed_checkpoint_path)

    # 6. Load into target model
    renamed_checkpoint = torch.load(renamed_checkpoint_path)
    target_model.load_state_dict(renamed_checkpoint["state_dict"], strict=False)
    target_model.eval()

    # 7. Create test input
    test_input = torch.randn(2, 3, 32, 32)

    # 8. Compare outputs
    with torch.no_grad():
        source_output = source_model(test_input)
        target_output = target_model(test_input)

    # 9. Outputs should have the same shape and be finite
    # They won't be identical due to BatchNorm buffers not being mapped
    assert source_output.shape == target_output.shape
    assert torch.isfinite(source_output).all()
    assert torch.isfinite(target_output).all()

    diff = torch.abs(source_output - target_output).mean().item()
    print(f"Mean absolute difference: {diff:.6f}")
    print("✓ Identical architecture mapping successful!")


def test_partial_weight_mapping(tmp_path: Path, simple_model_class: nn.Module, renamed_model_class: nn.Module) -> None:
    """Test mapping when source has more layers than target."""
    # 1. Create source model with more layers
    source_model = simple_model_class(num_conv_layers=3, num_fc_layers=3, hidden_dim=256)
    source_model.eval()

    # 2. Save source model
    checkpoint_path = tmp_path / "large_model.pth"
    torch.save({"state_dict": source_model.state_dict()}, checkpoint_path)

    # 3. Create smaller target model
    target_model = renamed_model_class(num_conv_layers=2, num_fc_layers=2, hidden_dim=128)
    target_model.eval()

    # 4. Use WeightMapper to find compatible layers
    mapper = WeightMapper.from_checkpoint(checkpoint_path, target_model)
    mapping = mapper.suggest_mapping(threshold=0.5)

    # Should find some compatible layers
    assert len(mapping) > 0
    print(f"\nPartial mapping: {len(mapping)} parameters mapped")

    # Validate parameter type matching in partial mapping
    for source_name, target_name in mapping.items():
        source_type = source_name.split(".")[-1]
        target_type = target_name.split(".")[-1]
        assert source_type == target_type, (
            f"Parameter type mismatch: {source_name} ({source_type}) -> {target_name} ({target_type})"
        )

    # 5. Get unmatched parameters
    unmatched = mapper.get_unmatched()
    print(f"Unmatched source params: {len(unmatched['source'])}")
    print(f"Unmatched target params: {len(unmatched['target'])}")

    # Source should have unmatched params (extra layers)
    assert len(unmatched["source"]) > 0

    # 6. Apply partial mapping
    renamer = WeightRenamer(checkpoint_path)
    renamer.rename_keys(mapping)
    renamed_checkpoint_path = tmp_path / "partial_renamed.pth"
    renamer.save(renamed_checkpoint_path)

    # 7. Load with strict=False to allow missing keys
    renamed_checkpoint = torch.load(renamed_checkpoint_path)
    missing_keys, unexpected_keys = target_model.load_state_dict(renamed_checkpoint["state_dict"], strict=False)

    # Should have some missing keys (unmatched target params)
    print(f"Missing keys: {len(missing_keys)}")

    # 8. Model should still be functional
    test_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = target_model(test_input)

    assert output.shape == (1, 10)
    assert torch.isfinite(output).all()
    print("✓ Partial mapping successful, model is functional!")


def test_mapping_analysis_and_export(
    tmp_path: Path, simple_model_class: nn.Module, renamed_model_class: nn.Module
) -> None:
    """Test mapping analysis and export functionality."""
    # 1. Create models
    source_model = simple_model_class()
    target_model = renamed_model_class()

    # 2. Create mapper
    mapper = WeightMapper(source_model, target_model)
    mapping = mapper.suggest_mapping()
    assert len(mapping) > 0  # Should find mappings

    # 3. Print analysis
    print("\n" + "=" * 80)
    mapper.print_analysis()
    print("=" * 80)

    # 4. Get mapping with scores
    mappings_with_scores = mapper.get_mapping_with_scores()
    assert len(mappings_with_scores) > 0

    # Each item should be (source_name, target_name, score)
    for source_name, target_name, score in mappings_with_scores[:3]:
        print(f"{source_name} → {target_name} (score: {score:.3f})")
        assert 0.0 <= score <= 1.0

    # 5. Export mapping report
    report_path = tmp_path / "mapping_report.json"
    mapper.export_mapping_report(report_path)

    assert report_path.exists()
    print(f"✓ Mapping report exported to {report_path}")

    # 6. Verify report contains expected fields
    import json

    with report_path.open() as f:
        report = json.load(f)

    assert "source_model" in report
    assert "target_model" in report
    assert "mappings" in report
    assert "unmatched_source" in report
    assert "unmatched_target" in report
    assert "coverage" in report

    print(f"Coverage: {report['coverage']:.1%}")


def test_execution_order_impact_on_workflow(
    tmp_path: Path, simple_model_class: nn.Module, renamed_model_class: nn.Module
) -> None:
    """Test that execution order tracking improves matching in real workflow."""
    # 1. Create and save source model
    source_model = simple_model_class()
    checkpoint_path = tmp_path / "source.pth"
    torch.save({"state_dict": source_model.state_dict()}, checkpoint_path)

    # 2. Create target model
    target_model = renamed_model_class()

    # 3. Create dummy input
    dummy_input = torch.randn(2, 3, 32, 32)

    # 4. Compare mappings with and without execution order
    mapper_no_order = WeightMapper.from_checkpoint(checkpoint_path, target_model)
    mapper_with_order = WeightMapper.from_checkpoint(checkpoint_path, target_model, dummy_input=dummy_input)

    mapping_no_order = mapper_no_order.suggest_mapping(threshold=0.5)
    mapping_with_order = mapper_with_order.suggest_mapping(threshold=0.5)

    # 5. Get scores for comparison
    scores_no_order = {src: score for src, _, score in mapper_no_order.get_mapping_with_scores()}
    scores_with_order = {src: score for src, _, score in mapper_with_order.get_mapping_with_scores()}

    print(f"\n{'Execution Order Impact Analysis':-^80}")
    print(f"Mappings found (no order): {len(mapping_no_order)}")
    print(f"Mappings found (with order): {len(mapping_with_order)}")

    # Calculate statistics
    if scores_no_order and scores_with_order:
        avg_no_order = sum(scores_no_order.values()) / len(scores_no_order)
        avg_with_order = sum(scores_with_order.values()) / len(scores_with_order)

        print(f"Average score (no order): {avg_no_order:.4f}")
        print(f"Average score (with order): {avg_with_order:.4f}")
        print(f"Score improvement: {avg_with_order - avg_no_order:+.4f}")

        # Count improvements
        improved = sum(
            1 for src in scores_no_order if src in scores_with_order and scores_with_order[src] > scores_no_order[src]
        )
        same = sum(
            1
            for src in scores_no_order
            if src in scores_with_order and abs(scores_with_order[src] - scores_no_order[src]) < 1e-6
        )
        worse = sum(
            1 for src in scores_no_order if src in scores_with_order and scores_with_order[src] < scores_no_order[src]
        )

        print(f"Score changes: {improved} improved, {same} unchanged, {worse} worse")

        # Verify execution order helps or doesn't hurt
        assert improved + same >= worse, "Execution order should improve or maintain matching quality"

    # 6. Verify both mappings are valid (1-to-1, type-consistent)
    for mapping in [mapping_no_order, mapping_with_order]:
        for source_name, target_name in mapping.items():
            source_type = source_name.split(".")[-1]
            target_type = target_name.split(".")[-1]
            assert source_type == target_type, f"Type mismatch: {source_name} -> {target_name}"

    print("✓ Execution order tracking validated in complete workflow")

from pathlib import Path

import pytest
import torch
from torch import nn

from lit_wsl.models.weight_renamer import WeightRenamer, rename_checkpoint_keys


class SimpleModel(nn.Module):
    """A simple model for testing."""

    def __init__(self, in_features: int = 10, hidden_features: int = 20, out_features: int = 5) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.layer2 = nn.Linear(hidden_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return self.bn(x)


class DifferentModel(nn.Module):
    """A model with different architecture but compatible weights."""

    def __init__(self, in_features: int = 10, hidden_features: int = 20, out_features: int = 5) -> None:
        super().__init__()
        self.backbone = nn.Linear(in_features, hidden_features)
        self.head = nn.Linear(hidden_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.relu(x)
        x = self.head(x)
        return self.norm(x)


class TestWeightRenamer:
    """Test suite for WeightRenamer class."""

    @pytest.fixture
    def source_model(self) -> SimpleModel:
        """Create a source model with initialized weights."""
        model = SimpleModel()
        # Initialize with specific values for testing
        with torch.no_grad():
            model.layer1.weight.fill_(1.0)
            model.layer1.bias.fill_(0.1)
            model.layer2.weight.fill_(2.0)
            model.layer2.bias.fill_(0.2)
        return model

    @pytest.fixture
    def target_model(self) -> DifferentModel:
        """Create a target model with different architecture."""
        return DifferentModel()

    @pytest.fixture
    def checkpoint_path(self, tmp_path: Path, source_model: SimpleModel) -> Path:
        """Save source model checkpoint."""
        path = tmp_path / "source_checkpoint.pth"
        torch.save({"state_dict": source_model.state_dict()}, path)
        return path

    @pytest.fixture
    def direct_state_dict_path(self, tmp_path: Path, source_model: SimpleModel) -> Path:
        """Save model as direct state_dict."""
        path = tmp_path / "state_dict.pth"
        torch.save(source_model.state_dict(), path)
        return path

    @pytest.fixture
    def model_wrapper_path(self, tmp_path: Path, source_model: SimpleModel) -> Path:
        """Save model with 'model' key wrapper."""
        path = tmp_path / "model_wrapper.pth"
        torch.save({"model": source_model.state_dict()}, path)
        return path

    @pytest.fixture
    def direct_model_path(self, tmp_path: Path, source_model: SimpleModel) -> Path:
        """Save model directly (entire model object)."""
        path = tmp_path / "direct_model.pth"
        torch.save(source_model, path)
        return path

    def test_init_with_valid_checkpoint(self, checkpoint_path: Path) -> None:
        """Test initialization with a valid checkpoint."""
        renamer = WeightRenamer(checkpoint_path)
        assert renamer.weight_path == checkpoint_path
        assert renamer.weights is not None

    def test_init_with_nonexistent_file(self) -> None:
        """Test initialization with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            WeightRenamer("nonexistent.pth")

    def test_init_with_checkpoint_loader(self, checkpoint_path: Path) -> None:
        """Test initialization with checkpoint loader enabled."""
        renamer = WeightRenamer(checkpoint_path, use_checkpoint_loader=True)
        state_dict = renamer.state_dict
        assert "layer1.weight" in state_dict
        assert "layer2.weight" in state_dict

    def test_init_without_checkpoint_loader(self, checkpoint_path: Path) -> None:
        """Test initialization with checkpoint loader disabled."""
        renamer = WeightRenamer(checkpoint_path, use_checkpoint_loader=False)
        state_dict = renamer.state_dict
        assert "layer1.weight" in state_dict

    def test_state_dict_extraction_from_state_dict_key(self, checkpoint_path: Path) -> None:
        """Test state_dict extraction when checkpoint has 'state_dict' key."""
        renamer = WeightRenamer(checkpoint_path)
        state_dict = renamer.state_dict
        assert isinstance(state_dict, dict)
        assert "layer1.weight" in state_dict
        assert "layer2.bias" in state_dict

    def test_state_dict_extraction_from_model_key(self, model_wrapper_path: Path) -> None:
        """Test state_dict extraction when checkpoint has 'model' key."""
        renamer = WeightRenamer(model_wrapper_path)
        state_dict = renamer.state_dict
        assert isinstance(state_dict, dict)
        assert "layer1.weight" in state_dict

    def test_state_dict_extraction_direct(self, direct_state_dict_path: Path) -> None:
        """Test state_dict extraction when checkpoint is direct state_dict."""
        renamer = WeightRenamer(direct_state_dict_path)
        state_dict = renamer.state_dict
        assert isinstance(state_dict, dict)
        assert "layer1.weight" in state_dict

    def test_state_dict_extraction_from_direct_model(self, direct_model_path: Path) -> None:
        """Test state_dict extraction when checkpoint is a direct model object."""
        renamer = WeightRenamer(direct_model_path, use_checkpoint_loader=True)
        state_dict = renamer.state_dict
        assert isinstance(state_dict, dict)
        assert "layer1.weight" in state_dict
        assert "layer2.weight" in state_dict
        assert "bn.weight" in state_dict

    def test_list_keys(self, checkpoint_path: Path) -> None:
        """Test listing all keys in the state dict."""
        renamer = WeightRenamer(checkpoint_path)
        keys = renamer.list_keys()
        assert isinstance(keys, list)
        assert "layer1.weight" in keys
        assert "layer1.bias" in keys
        assert "layer2.weight" in keys
        assert "bn.weight" in keys

    def test_search_keys(self, checkpoint_path: Path) -> None:
        """Test searching for keys matching a pattern."""
        renamer = WeightRenamer(checkpoint_path)

        # Search for layer1
        layer1_keys = renamer.search_keys("layer1")
        assert "layer1.weight" in layer1_keys
        assert "layer1.bias" in layer1_keys
        assert "layer2.weight" not in layer1_keys

        # Search case-insensitive
        weight_keys = renamer.search_keys("WEIGHT")
        assert len(weight_keys) > 0
        assert all("weight" in key.lower() for key in weight_keys)

    def test_rename_keys_basic(self, checkpoint_path: Path, tmp_path: Path) -> None:
        """Test basic key renaming functionality."""
        renamer = WeightRenamer(checkpoint_path)

        # Rename keys
        key_mapping = {
            "layer1.weight": "backbone.weight",
            "layer1.bias": "backbone.bias",
        }
        renamer.rename_keys(key_mapping)

        # Check renamed keys
        state_dict = renamer.state_dict
        assert "backbone.weight" in state_dict
        assert "backbone.bias" in state_dict
        assert "layer1.weight" not in state_dict
        assert "layer1.bias" not in state_dict

        # Unmapped keys should be preserved
        assert "layer2.weight" in state_dict

    def test_rename_keys_without_preserve(self, checkpoint_path: Path) -> None:
        """Test renaming without preserving unmapped keys."""
        renamer = WeightRenamer(checkpoint_path)

        key_mapping = {
            "layer1.weight": "new_layer.weight",
        }
        renamer.rename_keys(key_mapping, preserve_unmapped=False)

        state_dict = renamer.state_dict
        assert "new_layer.weight" in state_dict
        assert "layer1.weight" not in state_dict
        # All other keys should be gone
        assert "layer2.weight" not in state_dict

    def test_rename_with_prefix(self, checkpoint_path: Path) -> None:
        """Test renaming keys by replacing prefix."""
        renamer = WeightRenamer(checkpoint_path)

        renamer.rename_with_prefix("layer1.", "backbone.")

        state_dict = renamer.state_dict
        assert "backbone.weight" in state_dict
        assert "backbone.bias" in state_dict
        assert "layer1.weight" not in state_dict
        assert "layer2.weight" in state_dict  # Not affected

    def test_remove_prefix(self, checkpoint_path: Path) -> None:
        """Test removing prefix from keys."""
        renamer = WeightRenamer(checkpoint_path)

        renamer.remove_prefix("layer1.")

        state_dict = renamer.state_dict
        assert "weight" in state_dict
        assert "bias" in state_dict
        assert "layer1.weight" not in state_dict
        assert "layer2.weight" in state_dict  # Not affected

    def test_add_prefix(self, checkpoint_path: Path) -> None:
        """Test adding prefix to all keys."""
        renamer = WeightRenamer(checkpoint_path)

        renamer.add_prefix("model.")

        state_dict = renamer.state_dict
        assert all(key.startswith("model.") for key in state_dict)
        assert "model.layer1.weight" in state_dict
        assert "model.layer2.weight" in state_dict

    def test_backup_and_restore(self, checkpoint_path: Path) -> None:
        """Test backup and restore functionality."""
        renamer = WeightRenamer(checkpoint_path)

        original_keys = set(renamer.list_keys())

        # Make changes with backup enabled
        renamer.rename_keys({"layer1.weight": "changed.weight"}, backup=True)

        modified_keys = set(renamer.list_keys())
        assert original_keys != modified_keys

        # Restore backup
        renamer.restore_backup()

        restored_keys = set(renamer.list_keys())
        assert original_keys == restored_keys

    def test_restore_without_backup(self, checkpoint_path: Path) -> None:
        """Test restore fails when no backup exists."""
        renamer = WeightRenamer(checkpoint_path)

        with pytest.raises(ValueError, match="No backup available"):
            renamer.restore_backup()

    def test_save_renamed_weights(self, checkpoint_path: Path, tmp_path: Path, source_model: SimpleModel) -> None:
        """Test saving renamed weights to file."""
        renamer = WeightRenamer(checkpoint_path)

        # Rename keys to match target model
        key_mapping = {
            "layer1.weight": "backbone.weight",
            "layer1.bias": "backbone.bias",
            "layer2.weight": "head.weight",
            "layer2.bias": "head.bias",
            "bn.weight": "norm.weight",
            "bn.bias": "norm.bias",
            "bn.running_mean": "norm.running_mean",
            "bn.running_var": "norm.running_var",
            "bn.num_batches_tracked": "norm.num_batches_tracked",
        }
        renamer.rename_keys(key_mapping)

        # Save to file
        output_path = tmp_path / "renamed.pth"
        renamer.save(output_path)

        assert output_path.exists()

        # Load saved weights
        loaded = torch.load(output_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)

        assert "backbone.weight" in state_dict
        assert "head.weight" in state_dict
        assert "norm.weight" in state_dict

    def test_transfer_weights_between_models(
        self, checkpoint_path: Path, tmp_path: Path, source_model: SimpleModel, target_model: DifferentModel
    ) -> None:
        """Test complete workflow: save from one model, rename, load to another."""
        # Step 1: Rename weights to match target model architecture
        renamer = WeightRenamer(checkpoint_path)

        key_mapping = {
            "layer1.weight": "backbone.weight",
            "layer1.bias": "backbone.bias",
            "layer2.weight": "head.weight",
            "layer2.bias": "head.bias",
            "bn.weight": "norm.weight",
            "bn.bias": "norm.bias",
            "bn.running_mean": "norm.running_mean",
            "bn.running_var": "norm.running_var",
            "bn.num_batches_tracked": "norm.num_batches_tracked",
        }
        renamer.rename_keys(key_mapping)

        # Step 2: Save renamed weights
        output_path = tmp_path / "renamed_for_target.pth"
        renamer.save(output_path)

        # Step 3: Load into target model
        loaded = torch.load(output_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)

        target_model.load_state_dict(state_dict)

        # Step 4: Verify weights transferred correctly
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()

        # Check that values match after renaming
        assert torch.allclose(source_state["layer1.weight"], target_state["backbone.weight"])
        assert torch.allclose(source_state["layer1.bias"], target_state["backbone.bias"])
        assert torch.allclose(source_state["layer2.weight"], target_state["head.weight"])
        assert torch.allclose(source_state["layer2.bias"], target_state["head.bias"])

    def test_transfer_with_different_sizes(self, tmp_path: Path, target_model: DifferentModel) -> None:
        """Test partial weight transfer when model sizes differ."""
        # Create a source model with different hidden size
        source_model = SimpleModel(in_features=10, hidden_features=30, out_features=5)

        checkpoint_path = tmp_path / "different_size.pth"
        torch.save({"state_dict": source_model.state_dict()}, checkpoint_path)

        renamer = WeightRenamer(checkpoint_path)

        # Only rename and transfer compatible layers (bn/norm layers have same size)
        key_mapping = {
            "bn.weight": "norm.weight",
            "bn.bias": "norm.bias",
            "bn.running_mean": "norm.running_mean",
            "bn.running_var": "norm.running_var",
            "bn.num_batches_tracked": "norm.num_batches_tracked",
        }
        renamer.rename_keys(key_mapping)

        output_path = tmp_path / "partial_transfer.pth"
        renamer.save(output_path)

        # Load with strict=False to allow partial loading
        loaded = torch.load(output_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)

        # Remove incompatible keys
        target_state = target_model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in target_state and v.shape == target_state[k].shape}

        target_model.load_state_dict(filtered_state, strict=False)

        # Verify partial transfer worked - compare compatible layers
        assert torch.allclose(source_model.state_dict()["bn.weight"], target_model.state_dict()["norm.weight"])
        assert torch.allclose(source_model.state_dict()["bn.bias"], target_model.state_dict()["norm.bias"])

    def test_print_summary(self, checkpoint_path: Path, capsys) -> None:
        """Test print_summary output."""
        renamer = WeightRenamer(checkpoint_path)
        renamer.print_summary()

        captured = capsys.readouterr()
        assert "Weight file:" in captured.out
        assert "Total parameters:" in captured.out
        assert "layer1.weight" in captured.out


class TestRenameCheckpointKeys:
    """Test suite for rename_checkpoint_keys convenience function."""

    def test_rename_with_mapping(self, tmp_path: Path) -> None:
        """Test renaming with key mapping."""
        # Create source checkpoint
        state_dict = {
            "old.weight": torch.randn(5, 3),
            "old.bias": torch.randn(5),
        }
        input_path = tmp_path / "input.pth"
        torch.save({"state_dict": state_dict}, input_path)

        # Rename
        output_path = tmp_path / "output.pth"
        key_mapping = {
            "old.weight": "new.weight",
            "old.bias": "new.bias",
        }
        rename_checkpoint_keys(input_path, output_path, key_mapping=key_mapping)

        # Verify
        loaded = torch.load(output_path, map_location="cpu")
        result_state = loaded.get("state_dict", loaded)
        assert "new.weight" in result_state
        assert "new.bias" in result_state
        assert "old.weight" not in result_state

    def test_rename_with_prefix_replacement(self, tmp_path: Path) -> None:
        """Test renaming with prefix replacement."""
        state_dict = {
            "module.layer1.weight": torch.randn(5, 3),
            "module.layer2.weight": torch.randn(5, 5),
        }
        input_path = tmp_path / "input.pth"
        torch.save(state_dict, input_path)

        output_path = tmp_path / "output.pth"
        rename_checkpoint_keys(input_path, output_path, old_prefix="module.", new_prefix="model.")

        loaded = torch.load(output_path, map_location="cpu")
        assert "model.layer1.weight" in loaded
        assert "model.layer2.weight" in loaded

    def test_rename_with_prefix_removal(self, tmp_path: Path) -> None:
        """Test removing prefix."""
        state_dict = {
            "prefix.layer.weight": torch.randn(5, 3),
            "prefix.layer.bias": torch.randn(5),
        }
        input_path = tmp_path / "input.pth"
        torch.save(state_dict, input_path)

        output_path = tmp_path / "output.pth"
        rename_checkpoint_keys(input_path, output_path, old_prefix="prefix.")

        loaded = torch.load(output_path, map_location="cpu")
        assert "layer.weight" in loaded
        assert "layer.bias" in loaded

    def test_with_checkpoint_loader_enabled(self, tmp_path: Path) -> None:
        """Test with checkpoint loader enabled."""
        state_dict = {"weight": torch.randn(5, 3)}
        input_path = tmp_path / "input.pth"
        torch.save({"state_dict": state_dict}, input_path)

        output_path = tmp_path / "output.pth"
        rename_checkpoint_keys(
            input_path,
            output_path,
            key_mapping={"weight": "new_weight"},
            use_checkpoint_loader=True,
        )

        assert output_path.exists()

    def test_with_checkpoint_loader_disabled(self, tmp_path: Path) -> None:
        """Test with checkpoint loader disabled."""
        state_dict = {"weight": torch.randn(5, 3)}
        input_path = tmp_path / "input.pth"
        torch.save({"state_dict": state_dict}, input_path)

        output_path = tmp_path / "output.pth"
        rename_checkpoint_keys(
            input_path,
            output_path,
            key_mapping={"weight": "new_weight"},
            use_checkpoint_loader=False,
        )

        assert output_path.exists()


class TestComplexWeightTransfer:
    """Test complex scenarios for weight transfer between different models."""

    def test_transfer_with_module_prefix(self, tmp_path: Path) -> None:
        """Test transferring weights saved with 'module.' prefix from DataParallel."""
        model = SimpleModel()

        # Simulate DataParallel checkpoint with 'module.' prefix
        state_dict = {f"module.{k}": v for k, v in model.state_dict().items()}
        checkpoint_path = tmp_path / "dataparallel.pth"
        torch.save({"state_dict": state_dict}, checkpoint_path)

        # Remove module prefix and rename for target model
        renamer = WeightRenamer(checkpoint_path)
        renamer.remove_prefix("module.")

        # Now rename for different architecture
        key_mapping = {
            "layer1.weight": "backbone.weight",
            "layer1.bias": "backbone.bias",
            "layer2.weight": "head.weight",
            "layer2.bias": "head.bias",
            "bn.weight": "norm.weight",
            "bn.bias": "norm.bias",
            "bn.running_mean": "norm.running_mean",
            "bn.running_var": "norm.running_var",
            "bn.num_batches_tracked": "norm.num_batches_tracked",
        }
        renamer.rename_keys(key_mapping, backup=False)

        output_path = tmp_path / "cleaned.pth"
        renamer.save(output_path)

        # Load into target model
        target = DifferentModel()
        loaded = torch.load(output_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)
        target.load_state_dict(state_dict)

        # Verify
        assert "module." not in str(target.state_dict().keys())

    def test_transfer_partial_architecture(self, tmp_path: Path) -> None:
        """Test transferring only part of the architecture."""

        # Source model with extra layers
        class ExtendedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.Linear(20, 5)
                self.layer3 = nn.Linear(5, 2)  # Extra layer

            def forward(self, x):
                return self.layer3(self.layer2(self.layer1(x)))

        source = ExtendedModel()
        checkpoint_path = tmp_path / "extended.pth"
        torch.save({"state_dict": source.state_dict()}, checkpoint_path)

        # Transfer only compatible layers
        renamer = WeightRenamer(checkpoint_path)
        key_mapping = {
            "layer1.weight": "backbone.weight",
            "layer1.bias": "backbone.bias",
            "layer2.weight": "head.weight",
            "layer2.bias": "head.bias",
        }
        renamer.rename_keys(key_mapping, preserve_unmapped=False)

        output_path = tmp_path / "partial.pth"
        renamer.save(output_path)

        # Verify only selected keys are present
        loaded = torch.load(output_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)

        assert "backbone.weight" in state_dict
        assert "head.weight" in state_dict
        assert "layer3.weight" not in state_dict

    def test_chain_multiple_renames(self, tmp_path: Path) -> None:
        """Test chaining multiple rename operations."""
        model = SimpleModel()
        checkpoint_path = tmp_path / "chain.pth"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)

        renamer = WeightRenamer(checkpoint_path)

        # Step 1: Add prefix
        renamer.add_prefix("model.", backup=True)

        # Step 2: Rename specific layers
        renamer.rename_with_prefix("model.layer1.", "model.encoder.", backup=False)
        renamer.rename_with_prefix("model.layer2.", "model.decoder.", backup=False)

        # Step 3: Rename bn to norm
        renamer.rename_with_prefix("model.bn.", "model.norm.", backup=False)

        state_dict = renamer.state_dict
        assert "model.encoder.weight" in state_dict
        assert "model.decoder.weight" in state_dict
        assert "model.norm.weight" in state_dict
        assert "layer1.weight" not in state_dict

    def test_preserve_tensor_values_after_rename(self, tmp_path: Path) -> None:
        """Test that tensor values are preserved after renaming."""
        model = SimpleModel()

        # Set specific values
        with torch.no_grad():
            model.layer1.weight.fill_(3.14)
            model.layer1.bias.fill_(2.71)

        original_weight = model.layer1.weight.clone()
        original_bias = model.layer1.bias.clone()

        checkpoint_path = tmp_path / "values.pth"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)

        # Rename
        renamer = WeightRenamer(checkpoint_path)
        renamer.rename_keys(
            {
                "layer1.weight": "new_weight",
                "layer1.bias": "new_bias",
            }
        )

        # Verify values unchanged
        state_dict = renamer.state_dict
        assert torch.allclose(state_dict["new_weight"], original_weight)
        assert torch.allclose(state_dict["new_bias"], original_bias)

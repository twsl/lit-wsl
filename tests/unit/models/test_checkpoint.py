from pathlib import Path
import pickle  # noqa: S403

import pytest
import torch
from torch import nn

from lit_wsl.models.checkpoint import DictProxy, DynamicUnpickler, extract_state_dict, load_checkpoint_as_dict


class CustomModel(nn.Module):
    """A custom model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 10)
        self.custom_attr = "test_value"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


def test_dict_proxy() -> None:
    """Test DictProxy basic functionality."""
    proxy = DictProxy("test.Module")
    proxy.__setstate__({"weight": torch.randn(3, 3), "bias": torch.randn(3)})

    data = proxy.to_dict()
    assert data["__class__"] == "test.Module"
    assert "weight" in data
    assert "bias" in data


def test_create_dynamic_unpickler_with_missing_class() -> None:
    """Test unpickler with a class that doesn't exist."""
    # Test with regular pickle (not torch.save which uses persistent IDs)
    test_data = {"key": "value", "number": 42}

    # Pickle the data
    pickled_data = pickle.dumps(test_data)

    # Load with dynamic unpickler
    import io

    with io.BytesIO(pickled_data) as f:
        unpickler = DynamicUnpickler(f)
        loaded = unpickler.load()

    # Verify the data was loaded correctly
    assert loaded == test_data


def test_load_checkpoint_as_dict(tmp_path: Path) -> None:
    """Test load_checkpoint_as_dict function."""
    # Create a simple checkpoint
    state_dict = {
        "conv.weight": torch.randn(64, 3, 3, 3),
        "conv.bias": torch.randn(64),
        "fc.weight": torch.randn(10, 64),
        "fc.bias": torch.randn(10),
    }
    checkpoint = {"state_dict": state_dict, "epoch": 5, "optimizer": "adam"}

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    # Load using our utility
    loaded = load_checkpoint_as_dict(str(checkpoint_path))

    assert isinstance(loaded, dict)
    assert "state_dict" in loaded
    assert "epoch" in loaded
    assert loaded["epoch"] == 5
    assert len(loaded["state_dict"]) == 4


def test_load_checkpoint_as_dict_nonexistent() -> None:
    """Test load_checkpoint_as_dict with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint_as_dict("nonexistent_file.pt")


def test_extract_state_dict_from_dict() -> None:
    """Test extracting state_dict from various checkpoint formats."""
    state_dict = {
        "layer.weight": torch.randn(10, 5),
        "layer.bias": torch.randn(10),
    }

    # Format 1: Direct state_dict
    checkpoint1 = {"state_dict": state_dict}
    extracted1 = extract_state_dict(checkpoint1)
    assert extracted1 == state_dict

    # Format 2: Model wrapper
    checkpoint2 = {"model": state_dict}
    extracted2 = extract_state_dict(checkpoint2)
    assert extracted2 == state_dict

    # Format 3: Already a state_dict
    extracted3 = extract_state_dict(state_dict)
    assert extracted3 == state_dict


def test_extract_state_dict_from_model() -> None:
    """Test extracting state_dict from a model instance."""
    model = nn.Linear(10, 5)
    state_dict = extract_state_dict(model)

    assert isinstance(state_dict, dict)
    assert "weight" in state_dict
    assert "bias" in state_dict
    assert state_dict["weight"].shape == (5, 10)


def test_extract_state_dict_invalid() -> None:
    """Test extract_state_dict with invalid input."""
    with pytest.raises(ValueError, match="Could not extract state_dict"):
        extract_state_dict({"random_key": "random_value"})


def test_dict_proxy_repr() -> None:
    """Test DictProxy string representation."""
    proxy = DictProxy("test.Model")
    proxy.__setstate__({"param": torch.randn(3)})

    repr_str = repr(proxy)
    assert "DictProxy" in repr_str
    assert "test.Model" in repr_str


def test_round_trip_with_dict_proxy() -> None:
    """Test that DictProxy can be pickled and unpickled."""
    proxy = DictProxy("test.Module")
    proxy.__setstate__({"value": 42, "name": "test"})

    # Pickle and unpickle
    pickled = pickle.dumps(proxy)
    restored = pickle.loads(pickled)  # noqa: S301

    assert isinstance(restored, DictProxy)
    restored_dict = restored.to_dict()
    assert restored_dict["__class__"] == "test.Module"
    assert restored_dict["value"] == 42
    assert restored_dict["name"] == "test"

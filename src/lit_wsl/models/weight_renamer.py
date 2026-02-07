from pathlib import Path
from typing import Any

import torch

from lit_wsl.models.checkpoint import load_checkpoint_as_dict


class WeightRenamer:
    """Utility class for renaming keys in PyTorch weight files.

    This class provides functionality to load PyTorch checkpoint files,
    rename keys according to a mapping dictionary, and save the modified
    weights back to disk.

    Example:
        >>> from yolo.utils.weight_renamer import WeightRenamer
        >>>
        >>> # Load weights with robust checkpoint loader (default)
        >>> renamer = WeightRenamer("model.pth")
        >>>
        >>> # Or load with standard torch.load
        >>> renamer = WeightRenamer("model.pth", use_checkpoint_loader=False)
        >>>
        >>> # List all keys
        >>> renamer.print_summary()
        >>>
        >>> # Search for specific keys
        >>> matching = renamer.search_keys("backbone")
        >>>
        >>> # Rename specific keys
        >>> renamer.rename_keys({
        >>>     "backbone.conv1": "backbone.layer1.conv",
        >>>     "head.fc": "head.classifier"
        >>> })
        >>>
        >>> # Replace a prefix
        >>> renamer.rename_with_prefix("model.", "backbone.")
        >>>
        >>> # Remove a prefix
        >>> renamer.remove_prefix("module.")
        >>>
        >>> # Save modified weights
        >>> renamer.save("renamed_model.pth")
    """

    def __init__(self, weight_path: str | Path, use_checkpoint_loader: bool = True) -> None:
        """Initialize the WeightRenamer with a weight file.

        Args:
            weight_path: Path to the PyTorch weight file (.pth, .pt, or .ckpt)
            use_checkpoint_loader: Whether to use load_checkpoint_as_dict for robust loading
        """
        self.weight_path = Path(weight_path)
        if not self.weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        if use_checkpoint_loader:
            self.weights = load_checkpoint_as_dict(self.weight_path)
        else:
            self.weights = torch.load(self.weight_path, map_location="cpu")  # nosec B614
        self._original_weights = None

    @property
    def state_dict(self) -> dict[str, Any]:
        """Get the state dict from the loaded weights.

        Returns:
            The state dictionary containing model parameters
        """
        if isinstance(self.weights, dict):
            # Handle different checkpoint formats
            if "state_dict" in self.weights:
                return self.weights["state_dict"]
            elif "model" in self.weights:
                return self.weights["model"]
            else:
                return self.weights
        else:
            raise TypeError("Unsupported weight file format.")

    def list_keys(self) -> list[str]:
        """List all keys in the state dict.

        Returns:
            List of all parameter keys
        """
        return list(self.state_dict.keys())

    def search_keys(self, pattern: str) -> list[str]:
        """Search for keys matching a pattern.

        Args:
            pattern: String pattern to search for (case-insensitive)

        Returns:
            List of keys containing the pattern
        """
        pattern_lower = pattern.lower()
        return [key for key in self.list_keys() if pattern_lower in key.lower()]

    def rename_keys(
        self,
        key_mapping: dict[str, str],
        preserve_unmapped: bool = True,
        backup: bool = True,
    ) -> None:
        """Rename keys in the state dict according to a mapping.

        Args:
            key_mapping: Dictionary mapping old key names to new key names
            preserve_unmapped: Whether to keep keys not in the mapping
            backup: Whether to backup the original weights before renaming
        """
        if backup and self._original_weights is None:
            self._original_weights = self.weights.copy()

        state_dict = self.state_dict
        new_state_dict = {}

        for old_key, value in state_dict.items():
            if old_key in key_mapping:
                new_key = key_mapping[old_key]
                new_state_dict[new_key] = value
                print(f"Renamed: {old_key} -> {new_key}")
            elif preserve_unmapped:
                new_state_dict[old_key] = value

        # Update the weights with the new state dict
        if isinstance(self.weights, dict):
            if "state_dict" in self.weights:
                self.weights["state_dict"] = new_state_dict
            elif "model" in self.weights:
                self.weights["model"] = new_state_dict
            else:
                self.weights = new_state_dict
        else:
            self.weights = new_state_dict

    def rename_with_prefix(
        self,
        old_prefix: str,
        new_prefix: str,
        backup: bool = True,
    ) -> None:
        """Rename all keys by replacing a prefix.

        Args:
            old_prefix: Prefix to replace
            new_prefix: New prefix to use
            backup: Whether to backup the original weights before renaming
        """
        if backup and self._original_weights is None:
            self._original_weights = self.weights.copy()

        state_dict = self.state_dict
        key_mapping = {}

        for key in state_dict:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                key_mapping[key] = new_key

        if key_mapping:
            self.rename_keys(key_mapping, preserve_unmapped=True, backup=False)
        else:
            print(f"No keys found with prefix: {old_prefix}")

    def remove_prefix(self, prefix: str, backup: bool = True) -> None:
        """Remove a prefix from all matching keys.

        Args:
            prefix: Prefix to remove
            backup: Whether to backup the original weights before renaming
        """
        self.rename_with_prefix(prefix, "", backup=backup)

    def add_prefix(self, prefix: str, backup: bool = True) -> None:
        """Add a prefix to all keys.

        Args:
            prefix: Prefix to add
            backup: Whether to backup the original weights before renaming
        """
        if backup and self._original_weights is None:
            self._original_weights = self.weights.copy()

        state_dict = self.state_dict
        key_mapping = {key: f"{prefix}{key}" for key in state_dict}
        self.rename_keys(key_mapping, preserve_unmapped=False, backup=False)

    def restore_backup(self) -> None:
        """Restore the original weights from backup."""
        if self._original_weights is None:
            raise ValueError("No backup available. Enable backup when renaming.")
        self.weights = self._original_weights.copy()
        print("Restored original weights from backup")

    def save(self, output_path: str | Path) -> None:
        """Save the modified weights to a file.

        Args:
            output_path: Path where the modified weights should be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.weights, output_path)
        print(f"Saved modified weights to: {output_path}")

    def print_summary(self) -> None:
        """Print a summary of the loaded weights."""
        state_dict = self.state_dict
        print(f"Weight file: {self.weight_path}")
        print(f"Total parameters: {len(state_dict)}")
        print(f"Total elements: {sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel'))}")
        print("\nFirst 10 keys:")
        for key in list(state_dict.keys())[:10]:
            shape = state_dict[key].shape if hasattr(state_dict[key], "shape") else "N/A"
            print(f"  {key}: {shape}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more")


def rename_checkpoint_keys(
    input_path: str | Path,
    output_path: str | Path,
    key_mapping: dict[str, str] | None = None,
    old_prefix: str | None = None,
    new_prefix: str | None = None,
    use_checkpoint_loader: bool = True,
) -> None:
    """Convenience function to rename keys in a checkpoint file.

    Args:
        input_path: Path to the input checkpoint file
        output_path: Path to save the modified checkpoint
        key_mapping: Dictionary mapping old key names to new key names
        old_prefix: Prefix to replace (used with new_prefix)
        new_prefix: New prefix to use (used with old_prefix)
        use_checkpoint_loader: Whether to use load_checkpoint_as_dict for robust loading
    """
    renamer = WeightRenamer(input_path, use_checkpoint_loader=use_checkpoint_loader)

    if key_mapping:
        renamer.rename_keys(key_mapping)

    if old_prefix is not None and new_prefix is not None:
        renamer.rename_with_prefix(old_prefix, new_prefix, backup=False)
    elif old_prefix is not None:
        renamer.remove_prefix(old_prefix, backup=False)

    renamer.save(output_path)

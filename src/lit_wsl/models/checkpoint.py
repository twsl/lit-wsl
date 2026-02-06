from pathlib import Path
import pickle  # nosec B403
from typing import Any

import torch
from torch import Tensor, nn


class DictProxy:
    """A proxy class that captures object state as a dictionary."""

    def __init__(self, class_name: str = "") -> None:
        self._class_name = class_name
        self._attributes = {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Capture the object's state when unpickling."""
        # Ensure _class_name exists even if not initialized
        if not hasattr(self, "_class_name"):
            self._class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._attributes = state

    def __getstate__(self) -> dict[str, Any]:
        """Return the captured state."""
        return self._attributes

    def __reduce__(self) -> tuple[type, tuple[str], dict[str, Any]]:
        """Support pickling of DictProxy objects."""
        return (self.__class__, (self._class_name,), self.__getstate__())

    def to_dict(self) -> dict:
        """Convert the proxy to a plain dictionary."""
        # Ensure _class_name exists (fallback for edge cases)
        if not hasattr(self, "_class_name"):
            self._class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return {
            "__class__": self._class_name,
            **self._attributes,
        }

    def __repr__(self) -> str:
        class_name = getattr(self, "_class_name", f"{self.__class__.__module__}.{self.__class__.__name__}")
        return f"DictProxy({class_name}, {getattr(self, '_attributes', {})})"


class DynamicUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        original_class_path = f"{module}.{name}"
        if name == "_METADATA_":
            return super().find_class(module, name)
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            print(f"Class '{original_class_path}' not found. Loading as dictionary.")

            # Create a dynamic class that inherits from DictProxy
            # This ensures it's a proper type, not a function, which avoids
            # "NEWOBJ class argument must be a type" errors
            class DynamicProxy(DictProxy):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    super().__init__(original_class_path)

                def __setstate__(self, state: dict[str, Any]) -> None:
                    """Ensure _class_name is set before calling parent's __setstate__."""
                    self._class_name = original_class_path
                    super().__setstate__(state)

            DynamicProxy.__name__ = name
            DynamicProxy.__module__ = module

            return DynamicProxy


def load_checkpoint_as_dict(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint without requiring original class definitions.

    This function can load any PyTorch checkpoint (.pt, .pth, .ckpt) and returns
    the content as a dictionary, even if the original model classes are not available.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        A dictionary containing the checkpoint data. If the checkpoint contains
        model instances, they will be converted to DictProxy objects that can
        be inspected as dictionaries.

    Example:
        >>> checkpoint = load_checkpoint_as_dict("model.pt")
        >>> print(checkpoint.keys())
        >>> if "state_dict" in checkpoint:
        ...     print("State dict keys:", checkpoint["state_dict"].keys())
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Monkey-patch pickle.Unpickler temporarily
    original_unpickler = pickle.Unpickler

    try:
        # Temporarily replace the Unpickler
        pickle.Unpickler = DynamicUnpickler
        # Use torch.load which will use our modified Unpickler
        loaded_object = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        # Restore original Unpickler
        pickle.Unpickler = original_unpickler

    # Convert DictProxy objects to plain dictionaries recursively
    def convert_to_dict(obj: Any) -> Any:
        # Check for PyTorch modules first (before DictProxy check)
        # because DynamicProxy instances might also match isinstance(obj, DictProxy)
        if isinstance(obj, nn.Module):
            # Handle PyTorch modules that were successfully loaded
            try:
                return {
                    "__class__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                    "state_dict": obj.state_dict(),
                }
            except Exception:  # noqa: S110
                # If state_dict() fails, treat as DictProxy or regular object
                pass  # nosec B110

        if isinstance(obj, DictProxy):
            data = obj.to_dict()
            # Recursively convert nested DictProxy objects
            return {k: convert_to_dict(v) for k, v in data.items()}
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_to_dict(item) for item in obj)
        else:
            return obj

    result = convert_to_dict(loaded_object)

    # Ensure we return a dictionary
    if not isinstance(result, dict):
        result = {"checkpoint": result}

    return result


def extract_state_dict(checkpoint: dict | Any) -> dict[str, Tensor]:
    """Extract the state_dict from a loaded checkpoint.

    Handles various checkpoint formats:
    - Direct state_dict: {"layer.weight": tensor, ...}
    - Model wrapper: {"model": state_dict, ...}
    - Lightning checkpoint: {"state_dict": state_dict, ...}
    - Model instance: model.state_dict()
    - DictProxy-based structures from unpickling

    Args:
        checkpoint: The loaded checkpoint object.

    Returns:
        A dictionary mapping parameter names to tensors.

    Raises:
        ValueError: If no state_dict can be extracted.
    """

    def extract_from_module(module: nn.Module | DictProxy | dict, prefix: str = "") -> dict[str, Tensor]:
        """Recursively extract state_dict from a module (real or DictProxy)."""
        state_dict: dict[str, Tensor] = {}

        # Handle DictProxy objects (check for _attributes to catch subclasses)
        if isinstance(module, DictProxy) or (hasattr(module, "_attributes") and hasattr(module, "_class_name")):
            attrs = getattr(module, "_attributes", {})

            for name, param in attrs.get("_parameters", {}).items():
                if param is not None:
                    key = f"{prefix}.{name}" if prefix else name
                    state_dict[key] = param

            for name, buffer in attrs.get("_buffers", {}).items():
                if buffer is not None:
                    key = f"{prefix}.{name}" if prefix else name
                    state_dict[key] = buffer

            for name, submodule in attrs.get("_modules", {}).items():
                if submodule is not None:
                    subprefix = f"{prefix}.{name}" if prefix else name
                    state_dict.update(extract_from_module(submodule, subprefix))

        # Handle regular PyTorch modules
        elif hasattr(module, "_parameters"):
            for name, param in getattr(module, "_parameters", {}).items():
                if param is not None:
                    key = f"{prefix}.{name}" if prefix else name
                    state_dict[key] = param

            if hasattr(module, "_buffers"):
                for name, buffer in getattr(module, "_buffers", {}).items():
                    if buffer is not None:
                        key = f"{prefix}.{name}" if prefix else name
                        state_dict[key] = buffer

            if hasattr(module, "_modules"):
                for name, submodule in getattr(module, "_modules", {}).items():
                    if submodule is not None:
                        subprefix = f"{prefix}.{name}" if prefix else name
                        state_dict.update(extract_from_module(submodule, subprefix))

        return state_dict

    if isinstance(checkpoint, dict):
        # Check common keys
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            model_obj = checkpoint["model"]
            if isinstance(model_obj, dict):
                # Check if it's a module-like dict with _modules, _parameters, _buffers
                if "_modules" in model_obj and "_parameters" in model_obj and "_buffers" in model_obj:
                    # Extract from the nested module structure
                    # First try to extract from the main module
                    if "_modules" in model_obj and "model" in model_obj["_modules"]:
                        # This is the common case for YOLO checkpoints
                        result = extract_from_module(model_obj["_modules"]["model"], prefix="model")
                    else:
                        # Extract from the top-level module
                        result = extract_from_module(model_obj)
                    if result:
                        return result
                # Otherwise check if it's already a state_dict
                values = list(model_obj.values())
                if values and all(isinstance(v, Tensor) for k, v in model_obj.items() if not k.startswith("_")):
                    return model_obj
            elif hasattr(model_obj, "state_dict"):
                try:
                    return model_obj.state_dict()  # type: ignore[union-attr]
                except AttributeError:
                    # state_dict() failed, try manual extraction
                    result = extract_from_module(model_obj)
                    if result:
                        return result
        else:
            # Check if this is already a state_dict (contains tensors)
            values = list(checkpoint.values())
            if values and all(isinstance(v, Tensor) for k, v in checkpoint.items() if not k.startswith("_")):
                return checkpoint

    # Try to call state_dict() if available
    if hasattr(checkpoint, "state_dict"):
        try:
            return checkpoint.state_dict()  # type: ignore[union-attr]
        except AttributeError:
            # state_dict() failed, try manual extraction
            result = extract_from_module(checkpoint)
            if result:
                return result

    raise ValueError("Could not extract state_dict from checkpoint")

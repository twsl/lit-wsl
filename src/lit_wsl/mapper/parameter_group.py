from typing import Self

from lit_wsl.mapper.parameter_info import ParameterInfo


class ParameterGroup:
    """Represents a group of logically connected parameters (e.g., weight+bias for a layer)."""

    def __init__(self, module_path: str, params: dict[str, ParameterInfo]) -> None:
        """Initialize a parameter group.

        Args:
            module_path: The module path that these parameters belong to (e.g., 'layer1.conv')
            params: Dictionary mapping parameter type (weight, bias, etc.) to ParameterInfo
        """
        self.module_path = module_path
        self.params = params
        self.param_types = set(params.keys())

        # Extract common metadata
        if params:
            first_param = next(iter(params.values()))
            self.depth = first_param.depth - 1  # Depth of the module, not the parameter
            self.module_parts = module_path.split(".") if module_path else []
            self.execution_order = first_param.execution_order
        else:
            self.depth = 0
            self.module_parts = []
            self.execution_order = None

    def has_param_type(self, param_type: str) -> bool:
        """Check if this group has a specific parameter type."""
        return param_type in self.param_types

    def get_param(self, param_type: str) -> ParameterInfo | None:
        """Get parameter info for a specific type."""
        return self.params.get(param_type)

    def is_compatible_with(self, other: Self) -> bool:
        """Check if this group is compatible with another group for mapping.

        Compatible groups have:
        1. At least some common parameter types (subset matching allowed)
        2. Matching shapes for all common parameter types

        This allows partial matching where one group may have additional parameters
        (e.g., BatchNorm buffers) that the other doesn't have.
        """
        # Must have at least some common parameter types
        common_types = self.param_types & other.param_types
        if not common_types:
            return False

        # All common parameter types must have the same shape
        for param_type in common_types:
            self_param = self.params[param_type]
            other_param = other.params[param_type]
            if self_param.shape != other_param.shape:
                return False

        return True

    def is_compatible_with_excluding_buffers(self, other: Self) -> bool:
        """Check compatibility excluding statistical buffers.

        This allows matching when only buffers differ, focusing on trainable parameters.

        Returns:
            True if trainable parameters are compatible, regardless of buffer differences
        """
        # Get non-buffer parameter types from both groups
        self_core_types = {
            param_type for param_type, param_info in self.params.items() if not param_info.is_statistical_buffer
        }
        other_core_types = {
            param_type for param_type, param_info in other.params.items() if not param_info.is_statistical_buffer
        }

        # Must have at least some common core parameter types
        common_core_types = self_core_types & other_core_types
        if not common_core_types:
            return False

        # All common core types must have matching shapes
        for param_type in common_core_types:
            self_param = self.params[param_type]
            other_param = other.params[param_type]
            if self_param.shape != other_param.shape:
                return False

        return True

    def is_compatible_with_mode(self, other: Self, buffer_mode: str = "lenient") -> bool:
        """Check compatibility with configurable buffer handling.

        Args:
            other: Other parameter group to compare
            buffer_mode: How to handle buffers
                - 'strict': Buffers must match (original behavior)
                - 'lenient': Exclude statistical buffers from compatibility check (DEFAULT)
                - 'exclude': Exclude all buffers from compatibility check

        Returns:
            True if groups are compatible under the specified mode
        """
        if buffer_mode == "strict":
            return self.is_compatible_with(other)
        elif buffer_mode == "lenient":
            return self.is_compatible_with_excluding_buffers(other)
        elif buffer_mode == "exclude":
            # Even stricter: exclude ALL buffers
            self_trainable = {
                ptype
                for ptype, pinfo in self.params.items()
                if not getattr(pinfo, "is_buffer", False)  # Handle backward compat
            }
            other_trainable = {
                ptype
                for ptype, pinfo in other.params.items()
                if not getattr(pinfo, "is_buffer", False)  # Handle backward compat
            }

            common = self_trainable & other_trainable
            if not common:
                return False

            for param_type in common:
                if self.params[param_type].shape != other.params[param_type].shape:
                    return False

            return True
        else:
            raise ValueError(f"Unknown buffer_mode: {buffer_mode}")

    def has_exact_match_with(self, other: Self) -> bool:
        """Check if this group has exact parameter types and shapes match.

        Args:
            other: Other parameter group to compare with

        Returns:
            True if param_types are identical and all shapes match exactly
        """
        # Parameter types must be identical (not just overlapping)
        if self.param_types != other.param_types:
            return False

        # All parameter types must have the same shape
        for param_type in self.param_types:
            self_param = self.params[param_type]
            other_param = other.params[param_type]
            if self_param.shape != other_param.shape:
                return False

        return True

    def __repr__(self) -> str:
        param_types = ", ".join(sorted(self.param_types))
        return f"ParameterGroup(path='{self.module_path}', types=[{param_types}])"

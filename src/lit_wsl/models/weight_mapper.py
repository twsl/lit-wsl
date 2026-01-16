from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lit_wsl.utils.logger import get_logger


class ModuleNode:
    """Represents a node in the hierarchical module structure."""

    def __init__(self, name: str, full_path: str, depth: int) -> None:
        """Initialize a module node.

        Args:
            name: The module name (e.g., 'conv1')
            full_path: The full module path (e.g., 'encoder.layer1.conv1')
            depth: Depth in the hierarchy (0 for root)
        """
        self.name = name
        self.full_path = full_path
        self.depth = depth
        self.children: dict[str, ModuleNode] = {}
        self.parent: ModuleNode | None = None
        self.parameter_group: ParameterGroup | None = None

    def add_child(self, child: ModuleNode) -> None:
        """Add a child node."""
        self.children[child.name] = child
        child.parent = self

    def get_descendant_groups(self) -> list[ParameterGroup]:
        """Get all parameter groups in this subtree."""
        groups = []
        if self.parameter_group is not None:
            groups.append(self.parameter_group)
        for child in self.children.values():
            groups.extend(child.get_descendant_groups())
        return groups

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (has no children)."""
        return len(self.children) == 0

    def __repr__(self) -> str:
        param_info = f", params={len(self.parameter_group.param_types)}" if self.parameter_group else ""
        return f"ModuleNode('{self.full_path}', depth={self.depth}{param_info})"


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

    def is_compatible_with(self, other: ParameterGroup) -> bool:
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

    def __repr__(self) -> str:
        param_types = ", ".join(sorted(self.param_types))
        return f"ParameterGroup(path='{self.module_path}', types=[{param_types}])"


class ParameterInfo:
    """Metadata container for a model parameter."""

    def __init__(self, name: str, tensor: torch.Tensor, execution_order: int | None = None) -> None:
        """Initialize parameter metadata.

        Args:
            name: Full parameter name (e.g., 'backbone.layer1.conv.weight')
            tensor: The parameter tensor
            execution_order: Order in which the parent module was executed (for call order tracking)
        """
        self.name = name
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.numel = tensor.numel()
        self.execution_order = execution_order

        # Parse hierarchical structure
        self.parts = name.split(".")
        self.depth = len(self.parts)
        self.param_name = self.parts[-1]  # e.g., 'weight', 'bias'
        self.module_path = ".".join(self.parts[:-1]) if self.depth > 1 else ""

        # Extract tokens for name matching
        self.tokens = self._extract_tokens(name)

    def _extract_tokens(self, name: str) -> set[str]:
        """Extract meaningful tokens from parameter name.

        Args:
            name: Parameter name

        Returns:
            Set of tokens
        """
        # Split by common separators and extract meaningful parts
        import re

        # Split on dots, underscores, and camelCase boundaries
        tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+", name.replace(".", "_"))
        return {t.lower() for t in tokens if len(t) > 1}

    def __repr__(self) -> str:
        return f"ParameterInfo(name='{self.name}', shape={self.shape})"


class WeightMapper:
    """Analyze two nn.Module instances and suggest weight mapping between them.

    This class compares the parameters of a source model and a target model,
    computing similarity scores based on tensor shapes, parameter names, and
    hierarchical structure. It then suggests a mapping dictionary that can be
    used with WeightRenamer to adapt weights from the source model to the target.

    The matching algorithm prioritizes:
    1. Exact shape matching (required)
    2. Parameter type matching (weight->weight, bias->bias, etc.)
    3. Name similarity (edit distance, token overlap)
    4. Hierarchical position similarity
    5. Execution order similarity (when dummy_input is provided)

    Example:
        >>> from lit_wsl.models.weight_mapper import WeightMapper
        >>> from lit_wsl.models.weight_renamer import WeightRenamer
        >>> import torch
        >>>
        >>> # Basic usage - works great without dummy_input
        >>> source_model = OldModel()
        >>> target_model = NewModel()
        >>> mapper = WeightMapper(source_model, target_model)
        >>> mapping = mapper.suggest_mapping(threshold=0.6)
        >>>
        >>> # Optional: Provide dummy_input for better matching (~2% score improvement)
        >>> dummy_input = torch.randn(1, 3, 224, 224)
        >>> mapper = WeightMapper(source_model, target_model, dummy_input=dummy_input)
        >>> mapping = mapper.suggest_mapping(threshold=0.6)
        >>>
        >>> # From checkpoint (most common case)
        >>> from lit_wsl.models.checkpoint import load_checkpoint_as_dict
        >>> target_model = NewModel()
        >>> checkpoint = load_checkpoint_as_dict("old_weights.pth")
        >>> mapper = WeightMapper.from_state_dict(
        ...     source_state_dict=checkpoint.get("state_dict", checkpoint), target_module=target_model
        ... )
        >>>
        >>> # Analyze and apply mapping
        >>> mapping = mapper.suggest_mapping(threshold=0.6)
        >>> mapper.print_analysis()
        >>> renamer = WeightRenamer("old_weights.pth")
        >>> renamer.rename_keys(mapping)
        >>> renamer.save("adapted_weights.pth")
    """

    def __init__(
        self,
        source_module: nn.Module | None = None,
        target_module: nn.Module | None = None,
        shape_tolerance: float = 0.0,
        *,
        source_params: dict[str, ParameterInfo] | None = None,
        target_params: dict[str, ParameterInfo] | None = None,
        dummy_input: torch.Tensor | None = None,
    ):
        """Initialize the WeightMapper.

        Args:
            source_module: The source model (with weights to adapt from)
            target_module: The target model (to adapt weights to)
            shape_tolerance: Relative tolerance for shape matching (0.0 = exact match only)
            source_params: Pre-extracted source parameters (internal use)
            target_params: Pre-extracted target parameters (internal use)
            dummy_input: Optional dummy input tensor for execution order tracking.
                        Works perfectly fine without it. When provided, runs a forward pass
                        to track layer execution order, which can improve matching scores
                        by ~2% on average. Recommended for complex architectures with
                        significant structural changes.
        """
        self.logger: Logger = get_logger(self.__class__.__name__)
        self.source_module = source_module
        self.target_module = target_module
        self.shape_tolerance = shape_tolerance
        self.dummy_input = dummy_input

        # Extract parameter information
        if source_params is not None:
            self.source_params = source_params
        elif source_module is not None:
            self.source_params = self._extract_parameters(source_module)
        else:
            raise ValueError("Either source_module or source_params must be provided")

        if target_params is not None:
            self.target_params = target_params
        elif target_module is not None:
            self.target_params = self._extract_parameters(target_module)
        else:
            raise ValueError("Either target_module or target_params must be provided")

        # Build shape index for fast lookups
        self.target_by_shape = self._build_shape_index(self.target_params)

        # Extract parameter groups
        self.source_groups = self._extract_parameter_groups(self.source_params)
        self.target_groups = self._extract_parameter_groups(self.target_params)

        # Build hierarchical structure
        self.source_hierarchy = self._build_hierarchy(self.source_groups)
        self.target_hierarchy = self._build_hierarchy(self.target_groups)

        # Build group index by parameter types for fast lookups
        self.target_groups_by_types = self._build_group_index(self.target_groups)

        # Storage for mapping results
        self._mapping: dict[str, str] | None = None
        self._scores: dict[str, float] | None = None
        self._group_mapping: dict[str, str] | None = None  # Maps module paths
        self._group_scores: dict[str, float] | None = None
        self._hierarchy_context: dict[str, float] | None = None  # Hierarchical bonus scores
        self._transformations: dict[str, dict[str, Any]] | None = None  # Transformation metadata

    @classmethod
    def from_state_dict(
        cls,
        source_state_dict: dict[str, torch.Tensor],
        target_module: nn.Module,
        shape_tolerance: float = 0.0,
        dummy_input: torch.Tensor | None = None,
    ) -> WeightMapper:
        """Create a WeightMapper from a source state dictionary and target module.

        This is useful when you only have a checkpoint file but not the original model.

        Args:
            source_state_dict: State dictionary from the source model (e.g., loaded checkpoint)
            target_module: The target model to adapt weights to
            shape_tolerance: Relative tolerance for shape matching (0.0 = exact match only)
            dummy_input: (Optional) Dummy input tensor for execution order tracking.
                        Not required - the mapper works well without it. Provides
                        ~2% average score improvement when included.

        Returns:
            WeightMapper instance

        Example:
            >>> import torch
            >>> from lit_wsl.models.weight_mapper import WeightMapper
            >>>
            >>> # Load old weights
            >>> old_weights = torch.load("old_model.pth")
            >>> if "state_dict" in old_weights:
            ...     old_weights = old_weights["state_dict"]
            >>>
            >>> # Basic usage - works great as-is
            >>> new_model = NewModel()
            >>> mapper = WeightMapper.from_state_dict(old_weights, new_model)
            >>> mapping = mapper.suggest_mapping()
            >>>
            >>> # Optional: Add dummy_input for slightly better matching
            >>> dummy_input = torch.randn(1, 3, 224, 224)
            >>> mapper = WeightMapper.from_state_dict(old_weights, new_model, dummy_input=dummy_input)
            >>> mapping = mapper.suggest_mapping()
        """
        # Extract parameters from state dict
        source_params = cls._extract_parameters_from_state_dict(source_state_dict)
        target_params = cls._extract_parameters_from_module(target_module, dummy_input=dummy_input)

        return cls(
            source_module=None,
            target_module=target_module,
            shape_tolerance=shape_tolerance,
            source_params=source_params,
            target_params=target_params,
            dummy_input=dummy_input,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        target_module: nn.Module,
        shape_tolerance: float = 0.0,
        dummy_input: torch.Tensor | None = None,
    ) -> WeightMapper:
        """Create a WeightMapper from a checkpoint file and target module.

        Convenience method that loads the checkpoint and extracts the state dict.

        Args:
            checkpoint_path: Path to the checkpoint file
            target_module: The target model to adapt weights to
            shape_tolerance: Relative tolerance for shape matching (0.0 = exact match only)
            dummy_input: (Optional) Dummy input tensor for execution order tracking.
                        Not required - works perfectly without it.

        Returns:
            WeightMapper instance

        Example:
            >>> from lit_wsl.models.weight_mapper import WeightMapper
            >>> import torch
            >>>
            >>> # Simple usage
            >>> new_model = NewModel()
            >>> mapper = WeightMapper.from_checkpoint("old_model.pth", new_model)
            >>> mapping = mapper.suggest_mapping()
            >>>
            >>> # With optional dummy_input for better scores
            >>> dummy_input = torch.randn(1, 3, 224, 224)
            >>> mapper = WeightMapper.from_checkpoint("old_model.pth", new_model, dummy_input=dummy_input)
            >>> mapping = mapper.suggest_mapping()
        """
        from lit_wsl.models.checkpoint import load_checkpoint_as_dict

        # Load checkpoint
        checkpoint = load_checkpoint_as_dict(checkpoint_path)

        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        return cls.from_state_dict(state_dict, target_module, shape_tolerance, dummy_input=dummy_input)

    @staticmethod
    def _extract_parameters_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, ParameterInfo]:
        """Extract parameter information from a state dictionary.

        Args:
            state_dict: Dictionary mapping parameter names to tensors

        Returns:
            Dictionary mapping parameter names to ParameterInfo objects
        """
        params = {}
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                params[name] = ParameterInfo(name, tensor)
        return params

    @staticmethod
    def _extract_parameters_from_module(
        module: nn.Module, dummy_input: torch.Tensor | None = None
    ) -> dict[str, ParameterInfo]:
        """Extract parameter information from a module.

        Args:
            module: PyTorch module to extract from
            dummy_input: (Optional) Dummy input tensor for execution order tracking.
                        When None (default), no execution order tracking is performed.
                        When provided, runs a forward pass to track layer execution order.

        Returns:
            Dictionary mapping parameter names to ParameterInfo objects
        """
        execution_order_map = {}

        if dummy_input is not None:
            import warnings

            try:
                execution_order_map = WeightMapper._get_execution_order(module, dummy_input)
            except Exception as e:
                warnings.warn(
                    f"Execution order tracking failed: {e}. Proceeding without execution order information.",
                    stacklevel=2,
                )

        params = {}
        # Extract parameters
        for name, param in module.named_parameters():
            # Get module path from parameter name (remove last component which is the param type)
            module_path = ".".join(name.split(".")[:-1])
            execution_order = execution_order_map.get(module_path, None)
            params[name] = ParameterInfo(name, param, execution_order)

        # Also extract buffers (running_mean, running_var, num_batches_tracked, etc.)
        # This ensures consistency with state_dict which includes both parameters and buffers
        for name, buffer in module.named_buffers():
            module_path = ".".join(name.split(".")[:-1])
            execution_order = execution_order_map.get(module_path, None)
            params[name] = ParameterInfo(name, buffer, execution_order)

        return params

    @staticmethod
    def _get_execution_order(module: nn.Module, dummy_input: torch.Tensor) -> dict[str, int]:
        """Get execution order of modules by running a forward pass.

        Args:
            module: PyTorch module to analyze
            dummy_input: Dummy input tensor to use for forward pass

        Returns:
            Dictionary mapping module paths to execution order indices

        Raises:
            RuntimeError: If forward pass fails with the provided dummy_input
        """
        import torch

        # Validate that dummy_input is compatible with module
        try:
            module.eval()
            with torch.no_grad():
                test_output = module(dummy_input)
            del test_output  # Free memory
        except Exception as e:
            raise RuntimeError(
                f"Failed to run forward pass with provided dummy_input. "
                f"Please ensure the input shape and type are compatible with the model. "
                f"Error: {e}"
            ) from e

        execution_order = {}
        order_counter = [0]  # Use list to allow modification in closure

        # Register hooks on all named modules
        hooks = []
        for name, submodule in module.named_modules():
            if name == "":  # Skip the root module
                continue

            def hook(module, input, output, module_name=name):
                if module_name not in execution_order:
                    execution_order[module_name] = order_counter[0]
                    order_counter[0] += 1

            hooks.append(submodule.register_forward_hook(hook))

        # Run forward pass with provided dummy input
        module.eval()
        with torch.no_grad():
            module(dummy_input)

        # Remove all hooks
        for h in hooks:
            h.remove()

        return execution_order

    def _extract_parameters(self, module: nn.Module) -> dict[str, ParameterInfo]:
        """Extract parameter information from a module.

        Args:
            module: PyTorch module to extract from

        Returns:
            Dictionary mapping parameter names to ParameterInfo objects
        """
        return self._extract_parameters_from_module(module, dummy_input=self.dummy_input)

    def _build_shape_index(self, params: dict[str, ParameterInfo]) -> dict[tuple, list[str]]:
        """Build an index of parameters by shape for fast lookup.

        Args:
            params: Dictionary of parameter information

        Returns:
            Dictionary mapping shapes to list of parameter names
        """
        index = {}
        for param_name, info in params.items():
            if info.shape not in index:
                index[info.shape] = []
            index[info.shape].append(param_name)
        return index

    def _extract_parameter_groups(self, params: dict[str, ParameterInfo]) -> dict[str, ParameterGroup]:
        """Extract parameter groups from parameters.

        Groups parameters by their module path (e.g., all weight, bias for a layer).

        Args:
            params: Dictionary of parameter information

        Returns:
            Dictionary mapping module paths to ParameterGroup objects
        """
        groups_dict: dict[str, dict[str, ParameterInfo]] = {}

        for _name, info in params.items():
            module_path = info.module_path
            param_type = info.param_name

            if module_path not in groups_dict:
                groups_dict[module_path] = {}

            groups_dict[module_path][param_type] = info

        # Convert to ParameterGroup objects
        groups = {}
        for module_path, param_dict in groups_dict.items():
            groups[module_path] = ParameterGroup(module_path, param_dict)

        return groups

    def _build_group_index(self, groups: dict[str, ParameterGroup]) -> dict[frozenset, list[str]]:
        """Build an index of groups by their parameter types.

        Args:
            groups: Dictionary of ParameterGroup objects

        Returns:
            Dictionary mapping frozensets of parameter types to list of module paths
        """
        index: dict[frozenset, list[str]] = {}
        for module_path, group in groups.items():
            param_types = frozenset(group.param_types)
            if param_types not in index:
                index[param_types] = []
            index[param_types].append(module_path)
        return index

    def _build_hierarchy(self, groups: dict[str, ParameterGroup]) -> ModuleNode:
        """Build a hierarchical tree structure from parameter groups.

        Args:
            groups: Dictionary mapping module paths to ParameterGroup objects

        Returns:
            Root ModuleNode of the hierarchy tree
        """
        root = ModuleNode("", "", 0)
        nodes: dict[str, ModuleNode] = {"": root}

        # Sort paths to ensure parents are created before children
        sorted_paths = sorted(groups.keys(), key=lambda x: (x.count("."), x))

        for module_path in sorted_paths:
            if not module_path:  # Skip empty path
                continue

            parts = module_path.split(".")

            # Create all intermediate nodes if they don't exist
            for i in range(1, len(parts) + 1):
                partial_path = ".".join(parts[:i])
                if partial_path not in nodes:
                    parent_path = ".".join(parts[: i - 1]) if i > 1 else ""
                    parent_node = nodes[parent_path]
                    new_node = ModuleNode(parts[i - 1], partial_path, i)
                    parent_node.add_child(new_node)
                    nodes[partial_path] = new_node

            # Attach parameter group to the leaf node
            if module_path in groups:
                nodes[module_path].parameter_group = groups[module_path]

        return root

    def _compute_hierarchy_context_score(
        self,
        source_path: str,
        target_path: str,
        group_mapping: dict[str, str],
    ) -> float:
        """Compute hierarchical context score based on parent/sibling mappings.

        Args:
            source_path: Source module path
            target_path: Target module path
            group_mapping: Current group mapping (parent modules may be mapped)

        Returns:
            Context bonus score between 0.0 and 1.0
        """
        if not source_path or not target_path:
            return 0.5  # Neutral for root

        source_parts = source_path.split(".")
        target_parts = target_path.split(".")

        # Check if parent modules are mapped
        parent_match_bonus = 0.0
        for i in range(1, min(len(source_parts), len(target_parts))):
            source_parent = ".".join(source_parts[:i])
            target_parent = ".".join(target_parts[:i])

            if source_parent in group_mapping and group_mapping[source_parent] == target_parent:
                # Parent is mapped correctly - strong bonus
                parent_match_bonus += 0.3 / i  # Closer parents get higher weight

        # Check structural similarity (same depth, similar position)
        depth_match = 1.0 if len(source_parts) == len(target_parts) else 0.5

        # Combine scores
        return min(1.0, 0.5 * depth_match + 0.5 * min(1.0, parent_match_bonus))

    def _compute_shape_score_with_transform(
        self, source_info: ParameterInfo, target_info: ParameterInfo
    ) -> tuple[float, dict[str, Any] | None]:
        """Compute shape similarity score and determine if transformation is needed.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info

        Returns:
            Tuple of (score, transformation_info) where transformation_info is None
            if no transformation needed, or a dict with transformation details
        """
        if source_info.shape == target_info.shape:
            return 1.0, None

        # Check if shapes are transposed (e.g., for different Conv implementations)
        if len(source_info.shape) == len(target_info.shape) and sorted(source_info.shape) == sorted(target_info.shape):
            # Find the permutation needed
            # This is a simple case - for full implementation, would need more sophisticated matching
            transform_info = {
                "type": "transpose",
                "note": "Shapes are permutations of each other",
                "source_shape": source_info.shape,
                "target_shape": target_info.shape,
            }
            return 0.7, transform_info

        # Check relative size similarity if tolerance is set
        if self.shape_tolerance > 0:
            size_ratio = min(source_info.numel, target_info.numel) / max(source_info.numel, target_info.numel)
            if size_ratio >= (1.0 - self.shape_tolerance):
                transform_info = {
                    "type": "reshape",
                    "note": "Shapes have similar total elements",
                    "source_shape": source_info.shape,
                    "target_shape": target_info.shape,
                }
                return 0.5 * size_ratio, transform_info

        return 0.0, None

    def _compute_shape_score(self, source_info: ParameterInfo, target_info: ParameterInfo) -> float:
        """Compute shape similarity score.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info

        Returns:
            Score between 0.0 and 1.0
        """
        score, _ = self._compute_shape_score_with_transform(source_info, target_info)
        return score

    def _compute_name_similarity(self, source_info: ParameterInfo, target_info: ParameterInfo) -> float:
        """Compute name similarity score using multiple metrics.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info

        Returns:
            Score between 0.0 and 1.0
        """
        # CRITICAL: Parameter types must match (weight->weight, bias->bias, etc.)
        # This prevents weight being mapped to bias and vice versa
        if source_info.param_name != target_info.param_name:
            return 0.0

        # Exact match
        if source_info.name == target_info.name:
            return 1.0

        # Token overlap (Jaccard similarity)
        intersection = len(source_info.tokens & target_info.tokens)
        union = len(source_info.tokens | target_info.tokens)
        token_score = intersection / union if union > 0 else 0.0

        # Edit distance (normalized)
        edit_distance = self._levenshtein_distance(source_info.name, target_info.name)
        max_len = max(len(source_info.name), len(target_info.name))
        edit_score = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0

        # Longest common substring
        lcs_len = self._longest_common_substring_length(source_info.name, target_info.name)
        lcs_score = lcs_len / max_len if max_len > 0 else 0.0

        # Since param_name already matches, give high weight to other factors
        # Weighted combination
        return 0.4 * token_score + 0.3 * edit_score + 0.3 * lcs_score

    def _compute_hierarchy_similarity(self, source_info: ParameterInfo, target_info: ParameterInfo) -> float:
        """Compute hierarchical position similarity.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info

        Returns:
            Score between 0.0 and 1.0
        """
        # Depth similarity
        max_depth = max(source_info.depth, target_info.depth)
        depth_score = 1.0 - abs(source_info.depth - target_info.depth) / max_depth if max_depth > 0 else 1.0

        # Module path similarity
        if source_info.module_path and target_info.module_path:
            source_parts = source_info.parts[:-1]
            target_parts = target_info.parts[:-1]

            # Count matching parts from the beginning
            matching_levels = 0
            for s, t in zip(source_parts, target_parts, strict=False):
                if s == t:
                    matching_levels += 1
                else:
                    break

            max_levels = max(len(source_parts), len(target_parts))
            path_score = matching_levels / max_levels if max_levels > 0 else 0.0
        else:
            path_score = 0.5 if source_info.module_path == target_info.module_path else 0.0

        # Execution order similarity (if available)
        order_score = 0.5  # default neutral score
        if source_info.execution_order is not None and target_info.execution_order is not None:
            # Normalize by the max order to get a score between 0 and 1
            max_order = max(source_info.execution_order, target_info.execution_order)
            if max_order > 0:
                order_diff = abs(source_info.execution_order - target_info.execution_order)
                order_score = 1.0 - (order_diff / max_order)

        return 0.4 * depth_score + 0.4 * path_score + 0.2 * order_score

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _longest_common_substring_length(self, s1: str, s2: str) -> int:
        """Find the length of the longest common substring.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Length of longest common substring
        """
        m, n = len(s1), len(s2)
        max_len = 0

        # Create a table to store lengths of longest common suffixes
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_len = max(max_len, dp[i][j])

        return max_len

    def _compute_composite_score(
        self,
        source_info: ParameterInfo,
        target_info: ParameterInfo,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute composite similarity score.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info
            weights: Custom weights for scoring components
                     Default: {'shape': 0.5, 'name': 0.3, 'hierarchy': 0.2}

        Returns:
            Composite score between 0.0 and 1.0
        """
        if weights is None:
            weights = {"shape": 0.5, "name": 0.3, "hierarchy": 0.2}

        shape_score = self._compute_shape_score(source_info, target_info)

        # Shape match is required - if shapes don't match, return 0
        if shape_score == 0.0:
            return 0.0

        name_score = self._compute_name_similarity(source_info, target_info)
        hierarchy_score = self._compute_hierarchy_similarity(source_info, target_info)

        return weights["shape"] * shape_score + weights["name"] * name_score + weights["hierarchy"] * hierarchy_score

    def _compute_group_similarity(
        self,
        source_group: ParameterGroup,
        target_group: ParameterGroup,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute similarity score between two parameter groups.

        Args:
            source_group: Source parameter group
            target_group: Target parameter group
            weights: Custom weights for scoring components

        Returns:
            Composite score between 0.0 and 1.0
        """
        # Groups must be compatible (same param types and shapes)
        if not source_group.is_compatible_with(target_group):
            return 0.0

        # Compute average score across all parameters in the group
        total_score = 0.0
        num_params = len(source_group.param_types)

        for param_type in source_group.param_types:
            source_info = source_group.params[param_type]
            target_info = target_group.params[param_type]
            total_score += self._compute_composite_score(source_info, target_info, weights)

        return total_score / num_params if num_params > 0 else 0.0

    def suggest_mapping(
        self,
        threshold: float = 0.6,
        strategy: str = "best_match",
        weights: dict[str, float] | None = None,
    ) -> dict[str, str]:
        """Generate suggested parameter name mapping using group-based matching.

        This method groups logically connected parameters (e.g., weight+bias) and
        assigns them together, ensuring all connected elements are mapped as a unit.

        Args:
            threshold: Minimum score threshold for suggesting a match
            strategy: Matching strategy:
                - 'best_match': Select highest scoring match for each source
                - 'conservative': Only suggest high-confidence matches
                - 'shape_only': Only match parameters with identical shapes
            weights: Custom weights for scoring components

        Returns:
            Dictionary mapping source parameter names to target parameter names
            This ensures 1-to-1 mapping with parameter type matching and
            groups of connected parameters assigned together
        """
        # Adjust threshold based on strategy
        # Note: With hierarchical context, final scores are: 0.8*base + 0.2*context
        # So we adjust thresholds to account for this
        if strategy == "conservative":
            threshold = max(threshold, 0.65)
        elif strategy == "shape_only":
            weights = {"shape": 1.0, "name": 0.0, "hierarchy": 0.0}

        # First, perform group-based matching
        group_mapping, group_scores = self._suggest_group_mapping(threshold, weights)

        # Convert group mapping to individual parameter mapping
        mapping = {}
        scores = {}

        for source_module_path, target_module_path in group_mapping.items():
            source_group = self.source_groups[source_module_path]
            target_group = self.target_groups[target_module_path]
            group_score = group_scores[source_module_path]

            # Map only common parameters (allows partial group matching)
            # This handles cases where source has buffers that target doesn't have
            common_param_types = source_group.param_types & target_group.param_types
            for param_type in common_param_types:
                source_param = source_group.params[param_type]
                target_param = target_group.params[param_type]

                mapping[source_param.name] = target_param.name
                scores[source_param.name] = group_score

        # Fallback: Try to match remaining unmapped parameters individually
        # This helps when groups are incompatible but individual parameters could still match
        unmapped_source = set(self.source_params.keys()) - set(mapping.keys())
        unmapped_target = set(self.target_params.keys()) - set(mapping.values())

        if unmapped_source and unmapped_target:
            individual_mapping, individual_scores = self._suggest_individual_mapping(
                unmapped_source, unmapped_target, threshold, weights
            )
            mapping.update(individual_mapping)
            scores.update(individual_scores)

        # Validate mapping
        self._validate_mapping(mapping)

        # Store results
        self._mapping = mapping
        self._scores = scores
        self._group_mapping = group_mapping
        self._group_scores = group_scores

        return mapping

    def _suggest_group_mapping(
        self,
        threshold: float,
        weights: dict[str, float] | None = None,
    ) -> tuple[dict[str, str], dict[str, float]]:
        """Suggest mapping at the group level using hierarchical structure.

        This method matches modules in a top-down manner, leveraging the hierarchical
        structure to improve matching. Parent module mappings provide context for
        matching child modules.

        Args:
            threshold: Minimum score threshold
            weights: Custom weights for scoring

        Returns:
            Tuple of (group_mapping, group_scores) where:
            - group_mapping: dict mapping source module paths to target module paths
            - group_scores: dict mapping source module paths to their match scores
        """
        group_mapping = {}
        group_scores = {}
        hierarchy_context = {}
        used_targets = set()

        # Sort source groups by depth (shallow first) and then by path
        # This ensures parent modules are matched before children
        sorted_source_paths = sorted(self.source_groups.keys(), key=lambda x: (x.count("."), x))

        for source_path in sorted_source_paths:
            source_group = self.source_groups[source_path]

            # Find candidate target groups
            # With subset matching, we need to find groups that have at least some common types
            # and matching shapes for those types
            candidates = []
            for target_path, target_group in self.target_groups.items():
                if target_path in used_targets:
                    continue

                # Quick compatibility check (now allows subset matching)
                if not source_group.is_compatible_with(target_group):
                    continue

                # Base similarity score
                base_score = self._compute_group_similarity(source_group, target_group, weights)

                if base_score < threshold:
                    continue

                # Hierarchical context score (used as tiebreaker, not blended into score)
                context_score = self._compute_hierarchy_context_score(source_path, target_path, group_mapping)

                # Store both scores for sorting
                candidates.append((target_path, base_score, context_score))

            # Select best match
            if candidates:
                # Sort by base score first, then context score as tiebreaker
                # This prevents weak matches from being boosted by context
                candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                best_target, best_score, context = candidates[0]

                group_mapping[source_path] = best_target
                group_scores[source_path] = best_score
                hierarchy_context[source_path] = context
                used_targets.add(best_target)

        # Store hierarchy context for analysis
        self._hierarchy_context = hierarchy_context

        return group_mapping, group_scores

    def _suggest_individual_mapping(
        self,
        unmapped_source: set[str],
        unmapped_target: set[str],
        threshold: float,
        weights: dict[str, float] | None = None,
    ) -> tuple[dict[str, str], dict[str, float]]:
        """Suggest mapping for individual parameters that weren't matched at group level.

        This fallback method tries to match remaining unmapped parameters individually,
        which is useful when groups are incompatible but individual parameters could still match.

        Args:
            unmapped_source: Set of source parameter names not yet mapped
            unmapped_target: Set of target parameter names not yet mapped
            threshold: Minimum score threshold
            weights: Custom weights for scoring

        Returns:
            Tuple of (mapping, scores) dictionaries
        """
        mapping = {}
        scores = {}
        used_targets = set()

        # Sort source parameters by name for consistent ordering
        sorted_source = sorted(unmapped_source)

        for source_name in sorted_source:
            source_info = self.source_params[source_name]

            # Find candidates with matching shape
            candidates = []
            for target_name in unmapped_target:
                if target_name in used_targets:
                    continue

                target_info = self.target_params[target_name]

                # Compute individual parameter score
                score = self._compute_composite_score(source_info, target_info, weights)

                if score >= threshold:
                    candidates.append((target_name, score))

            # Select best match
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_target, best_score = candidates[0]

                mapping[source_name] = best_target
                scores[source_name] = best_score
                used_targets.add(best_target)

        return mapping, scores

    def _validate_mapping(self, mapping: dict[str, str]) -> None:
        """Validate that the mapping is 1-to-1 and parameter types match.

        Args:
            mapping: The mapping dictionary to validate

        Raises:
            ValueError: If mapping is not 1-to-1 or has parameter type mismatches
        """
        # Check for duplicate targets (should not happen with used_targets logic)
        target_counts = {}
        for target in mapping.values():
            target_counts[target] = target_counts.get(target, 0) + 1

        duplicates = {t: c for t, c in target_counts.items() if c > 1}
        if duplicates:
            raise ValueError(f"Mapping is not 1-to-1. Duplicate targets: {duplicates}")

        # Validate parameter type matching (should always be true due to _compute_name_similarity)
        # This is a sanity check to catch Any logic errors
        for source, target in mapping.items():
            source_type = self.source_params[source].param_name
            target_type = self.target_params[target].param_name
            if source_type != target_type:
                msg = (
                    f"INTERNAL ERROR: Parameter type mismatch in mapping. "
                    f"This should not happen as _compute_name_similarity returns 0.0 for mismatched types. "
                    f"Found: {source} ({source_type}) -> {target} ({target_type})"
                )
                raise RuntimeError(msg)

    def get_unmatched(self) -> dict[str, list[str]]:
        """Get parameters that couldn't be matched.

        Returns:
            Dictionary with 'source' and 'target' lists of unmatched parameter names
        """
        if self._mapping is None:
            self.suggest_mapping()

        if self._mapping is None:
            msg = "Mapping should not be None after suggest_mapping()"
            raise RuntimeError(msg)
        matched_sources = set(self._mapping.keys())
        matched_targets = set(self._mapping.values())

        unmatched_sources = [name for name in self.source_params if name not in matched_sources]
        unmatched_targets = [name for name in self.target_params if name not in matched_targets]

        return {
            "source": unmatched_sources,
            "target": unmatched_targets,
        }

    def print_analysis(self, top_n: int = 10, show_unmatched: bool = True) -> None:
        """Print detailed analysis of the mapping.

        Args:
            top_n: Number of top mappings to display
            show_unmatched: Whether to show unmatched parameters
        """
        if self._mapping is None:
            self.suggest_mapping()

        if self._mapping is None or self._scores is None:
            msg = "Mapping and scores should not be None after suggest_mapping()"
            raise RuntimeError(msg)

        self.logger.info("=" * 80)
        self.logger.info("Weight Mapping Analysis")
        self.logger.info("=" * 80)
        source_name = self.source_module.__class__.__name__ if self.source_module else "State Dict"
        self.logger.info(f"\nSource: {source_name}")
        self.logger.info(f"  Total parameters: {len(self.source_params)}")

        target_name = self.target_module.__class__.__name__ if self.target_module else "State Dict"
        self.logger.info(f"\nTarget: {target_name}")
        self.logger.info(f"  Total parameters: {len(self.target_params)}")

        self.logger.info("\nMatching results:")
        self.logger.info(f"  Matched: {len(self._mapping)}")
        self.logger.info(f"  Coverage: {len(self._mapping) / len(self.source_params) * 100:.1f}%")

        if self._mapping:
            self.logger.info(f"\n{'Top suggested mappings:':-^80}")
            self.logger.info(f"{'Source':<40} → {'Target':<30} {'Score':>8}")
            self.logger.info("-" * 80)

            # Sort by score
            sorted_mappings = sorted(
                self._mapping.items(),
                key=lambda x: self._scores[x[0]],  # type: ignore[index]
                reverse=True,
            )

            for source_name, target_name in sorted_mappings[:top_n]:
                score = self._scores[source_name]
                source_shape = self.source_params[source_name].shape

                # Truncate names if too long
                source_display = source_name if len(source_name) <= 38 else source_name[:35] + "..."
                target_display = target_name if len(target_name) <= 28 else target_name[:25] + "..."

                self.logger.info(f"{source_display:<40} → {target_display:<30} {score:>7.3f}")
                self.logger.info(f"  Shape: {source_shape}")

            if len(sorted_mappings) > top_n:
                self.logger.info(f"  ... and {len(sorted_mappings) - top_n} more matches")

        if show_unmatched:
            unmatched = self.get_unmatched()

            if unmatched["source"]:
                self.logger.info(f"\n{'Unmatched source parameters:':-^80}")
                for name in unmatched["source"][:10]:
                    shape = self.source_params[name].shape
                    self.logger.info(f"  {name:<60} {shape}")
                if len(unmatched["source"]) > 10:
                    self.logger.info(f"  ... and {len(unmatched['source']) - 10} more")

            if unmatched["target"]:
                self.logger.info(f"\n{'Unmatched target parameters:':-^80}")
                for name in unmatched["target"][:10]:
                    shape = self.target_params[name].shape
                    self.logger.info(f"  {name:<60} {shape}")
                if len(unmatched["target"]) > 10:
                    self.logger.info(f"  ... and {len(unmatched['target']) - 10} more")

        self.logger.info("=" * 80)

    def get_mapping_dict(self) -> dict[str, str]:
        """Get the current mapping dictionary.

        Returns:
            Dictionary mapping source parameter names to target parameter names
        """
        if self._mapping is None:
            self.suggest_mapping()
        if self._mapping is None:
            msg = "Mapping should not be None after suggest_mapping()"
            raise RuntimeError(msg)
        return self._mapping.copy()

    def get_mapping_with_scores(self) -> list[tuple[str, str, float]]:
        """Get mapping with confidence scores.

        Returns:
            List of tuples (source_name, target_name, score)
        """
        if self._mapping is None:
            self.suggest_mapping()

        if self._mapping is None or self._scores is None:
            msg = "Mapping and scores should not be None after suggest_mapping()"
            raise RuntimeError(msg)

        return [(source, target, self._scores[source]) for source, target in self._mapping.items()]

    def get_mapping_with_transformations(self) -> list[tuple[str, str, float, dict[str, Any] | None]]:
        """Get mapping with confidence scores and transformation information.

        Returns:
            List of tuples (source_name, target_name, score, transform_info) where
            transform_info is None if no transformation needed, or a dict with:
            - type: Type of transformation needed ('transpose', 'reshape', etc.)
            - note: Human-readable description
            - source_shape: Original shape
            - target_shape: Target shape
        """
        if self._mapping is None:
            self.suggest_mapping()

        if self._mapping is None or self._scores is None:
            msg = "Mapping and scores should not be None after suggest_mapping()"
            raise RuntimeError(msg)

        # Compute transformations if not already cached
        if self._transformations is None:
            self._transformations = {}
            for source_name, target_name in self._mapping.items():
                source_info = self.source_params[source_name]
                target_info = self.target_params[target_name]
                _, transform_info = self._compute_shape_score_with_transform(source_info, target_info)
                if transform_info is not None:
                    self._transformations[source_name] = transform_info

        result = []
        for source, target in self._mapping.items():
            score = self._scores[source]
            transform = self._transformations.get(source, None)
            result.append((source, target, score, transform))

        return result

    def export_mapping_report(self, output_path: str | Path) -> None:
        """Export detailed mapping report to a file.

        Args:
            output_path: Path to save the report
        """
        import json

        if self._mapping is None:
            self.suggest_mapping()

        if self._mapping is None or self._scores is None:
            msg = "Mapping and scores should not be None after suggest_mapping()"
            raise RuntimeError(msg)

        unmatched = self.get_unmatched()

        report = {
            "source_model": self.source_module.__class__.__name__ if self.source_module else "StateDict",
            "target_model": self.target_module.__class__.__name__ if self.target_module else "StateDict",
            "source_params_count": len(self.source_params),
            "target_params_count": len(self.target_params),
            "matched_count": len(self._mapping),
            "coverage": (len(self._mapping) / len(self.source_params) if self.source_params else 0),
            "mappings": [
                {
                    "source": source,
                    "target": target,
                    "score": self._scores[source],
                    "shape": list(self.source_params[source].shape),
                }
                for source, target in self._mapping.items()
            ],
            "unmatched_source": unmatched["source"],
            "unmatched_target": unmatched["target"],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Mapping report saved to: {output_path}")

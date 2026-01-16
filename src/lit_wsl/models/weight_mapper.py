from pathlib import Path

import torch
import torch.nn as nn


class ParameterInfo:
    """Metadata container for a model parameter."""

    def __init__(self, name: str, tensor: torch.Tensor) -> None:
        """Initialize parameter metadata.

        Args:
            name: Full parameter name (e.g., 'backbone.layer1.conv.weight')
            tensor: The parameter tensor
        """
        self.name = name
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.numel = tensor.numel()

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
    2. Name similarity (edit distance, token overlap)
    3. Hierarchical position similarity

    Example:
        >>> from lit_wsl.models.weight_mapper import WeightMapper
        >>> from lit_wsl.models.weight_renamer import WeightRenamer
        >>>
        >>> # Option 1: From models (when you have both architectures)
        >>> source_model = OldModel()
        >>> target_model = NewModel()
        >>> mapper = WeightMapper(source_model, target_model)
        >>>
        >>> # Option 2: From state_dict (common case - you only have weights)
        >>> from lit_wsl.models.checkpoint import load_checkpoint_as_dict
        >>> target_model = NewModel()
        >>> checkpoint = load_checkpoint_as_dict("old_weights.pth")
        >>> mapper = WeightMapper.from_state_dict(
        ...     source_state_dict=checkpoint.get("state_dict", checkpoint), target_module=target_model
        ... )
        >>>
        >>> # Analyze and get mapping
        >>> mapping = mapper.suggest_mapping(threshold=0.6)
        >>> mapper.print_analysis()
        >>>
        >>> # Use the mapping with WeightRenamer
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
    ):
        """Initialize the WeightMapper.

        Args:
            source_module: The source model (with weights to adapt from)
            target_module: The target model (to adapt weights to)
            shape_tolerance: Relative tolerance for shape matching (0.0 = exact match only)
            source_params: Pre-extracted source parameters (internal use)
            target_params: Pre-extracted target parameters (internal use)
        """
        self.source_module = source_module
        self.target_module = target_module
        self.shape_tolerance = shape_tolerance

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

        # Storage for mapping results
        self._mapping: dict[str, str] | None = None
        self._scores: dict[str, float] | None = None

    @classmethod
    def from_state_dict(
        cls,
        source_state_dict: dict[str, torch.Tensor],
        target_module: nn.Module,
        shape_tolerance: float = 0.0,
    ) -> "WeightMapper":
        """Create a WeightMapper from a source state dictionary and target module.

        This is useful when you only have a checkpoint file but not the original model.

        Args:
            source_state_dict: State dictionary from the source model (e.g., loaded checkpoint)
            target_module: The target model to adapt weights to
            shape_tolerance: Relative tolerance for shape matching (0.0 = exact match only)

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
            >>> # Create mapper with new model
            >>> new_model = NewModel()
            >>> mapper = WeightMapper.from_state_dict(old_weights, new_model)
            >>> mapping = mapper.suggest_mapping()
        """
        # Extract parameters from state dict
        source_params = cls._extract_parameters_from_state_dict(source_state_dict)
        target_params = cls._extract_parameters_from_module(target_module)

        return cls(
            source_module=None,
            target_module=target_module,
            shape_tolerance=shape_tolerance,
            source_params=source_params,
            target_params=target_params,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        target_module: nn.Module,
        shape_tolerance: float = 0.0,
    ) -> "WeightMapper":
        """Create a WeightMapper from a checkpoint file and target module.

        Convenience method that loads the checkpoint and extracts the state dict.

        Args:
            checkpoint_path: Path to the checkpoint file
            target_module: The target model to adapt weights to
            shape_tolerance: Relative tolerance for shape matching (0.0 = exact match only)

        Returns:
            WeightMapper instance

        Example:
            >>> from lit_wsl.models.weight_mapper import WeightMapper
            >>>
            >>> new_model = NewModel()
            >>> mapper = WeightMapper.from_checkpoint("old_model.pth", new_model)
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

        return cls.from_state_dict(state_dict, target_module, shape_tolerance)

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
    def _extract_parameters_from_module(module: nn.Module) -> dict[str, ParameterInfo]:
        """Extract parameter information from a module.

        Args:
            module: PyTorch module to extract from

        Returns:
            Dictionary mapping parameter names to ParameterInfo objects
        """
        params = {}
        for name, param in module.named_parameters():
            params[name] = ParameterInfo(name, param)
        return params

    def _extract_parameters(self, module: nn.Module) -> dict[str, ParameterInfo]:
        """Extract parameter information from a module.

        Args:
            module: PyTorch module to extract from

        Returns:
            Dictionary mapping parameter names to ParameterInfo objects
        """
        return self._extract_parameters_from_module(module)

    def _build_shape_index(self, params: dict[str, ParameterInfo]) -> dict[tuple, list[str]]:
        """Build an index of parameters by shape for fast lookup.

        Args:
            params: Dictionary of parameter information

        Returns:
            Dictionary mapping shapes to list of parameter names
        """
        index = {}
        for name, info in params.items():
            if info.shape not in index:
                index[info.shape] = []
            index[info.shape].append(name)
        return index

    def _compute_shape_score(self, source_info: ParameterInfo, target_info: ParameterInfo) -> float:
        """Compute shape similarity score.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info

        Returns:
            Score between 0.0 and 1.0
        """
        if source_info.shape == target_info.shape:
            return 1.0

        # Check if shapes are transposed (e.g., for different Conv implementations)
        if len(source_info.shape) == len(target_info.shape) and sorted(source_info.shape) == sorted(target_info.shape):
            return 0.7

        # Check relative size similarity if tolerance is set
        if self.shape_tolerance > 0:
            size_ratio = min(source_info.numel, target_info.numel) / max(source_info.numel, target_info.numel)
            if size_ratio >= (1.0 - self.shape_tolerance):
                return 0.5 * size_ratio

        return 0.0

    def _compute_name_similarity(self, source_info: ParameterInfo, target_info: ParameterInfo) -> float:
        """Compute name similarity score using multiple metrics.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info

        Returns:
            Score between 0.0 and 1.0
        """
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

        # Parameter name match (weight, bias, etc.)
        param_name_match = 1.0 if source_info.param_name == target_info.param_name else 0.0

        # Weighted combination
        return 0.3 * token_score + 0.2 * edit_score + 0.2 * lcs_score + 0.3 * param_name_match

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

        return 0.5 * depth_score + 0.5 * path_score

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

    def suggest_mapping(
        self,
        threshold: float = 0.6,
        strategy: str = "best_match",
        weights: dict[str, float] | None = None,
    ) -> dict[str, str]:
        """Generate suggested parameter name mapping.

        Args:
            threshold: Minimum score threshold for suggesting a match
            strategy: Matching strategy:
                - 'best_match': Select highest scoring match for each source
                - 'conservative': Only suggest high-confidence matches
                - 'shape_only': Only match parameters with identical shapes
            weights: Custom weights for scoring components

        Returns:
            Dictionary mapping source parameter names to target parameter names
        """
        mapping = {}
        scores = {}
        used_targets = set()

        # Adjust threshold based on strategy
        if strategy == "conservative":
            threshold = max(threshold, 0.8)
        elif strategy == "shape_only":
            weights = {"shape": 1.0, "name": 0.0, "hierarchy": 0.0}

        # Sort source parameters for consistent ordering
        source_names = sorted(self.source_params.keys())

        for source_name in source_names:
            source_info = self.source_params[source_name]

            # Find candidate targets with matching shapes
            candidates = []

            # Get targets with exact shape match
            if source_info.shape in self.target_by_shape:
                candidate_names = self.target_by_shape[source_info.shape]

                for target_name in candidate_names:
                    if target_name in used_targets:
                        continue

                    target_info = self.target_params[target_name]
                    score = self._compute_composite_score(source_info, target_info, weights)

                    if score >= threshold:
                        candidates.append((target_name, score))

            # Select best match
            if candidates:
                # Sort by score (descending)
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_target, best_score = candidates[0]

                mapping[source_name] = best_target
                scores[source_name] = best_score
                used_targets.add(best_target)

        # Store results
        self._mapping = mapping
        self._scores = scores

        return mapping

    def get_unmatched(self) -> dict[str, list[str]]:
        """Get parameters that couldn't be matched.

        Returns:
            Dictionary with 'source' and 'target' lists of unmatched parameter names
        """
        if self._mapping is None:
            self.suggest_mapping()

        assert self._mapping is not None  # Type narrowing for type checker
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

        assert self._mapping is not None  # Type narrowing for type checker
        assert self._scores is not None  # Type narrowing for type checker

        print("=" * 80)
        print("Weight Mapping Analysis")
        print("=" * 80)
        source_name = self.source_module.__class__.__name__ if self.source_module else "State Dict"
        print(f"\nSource: {source_name}")
        print(f"  Total parameters: {len(self.source_params)}")

        target_name = self.target_module.__class__.__name__ if self.target_module else "State Dict"
        print(f"\nTarget: {target_name}")
        print(f"  Total parameters: {len(self.target_params)}")

        print("\nMatching results:")
        print(f"  Matched: {len(self._mapping)}")
        print(f"  Coverage: {len(self._mapping) / len(self.source_params) * 100:.1f}%")

        if self._mapping:
            print(f"\n{'Top suggested mappings:':-^80}")
            print(f"{'Source':<40} → {'Target':<30} {'Score':>8}")
            print("-" * 80)

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

                print(f"{source_display:<40} → {target_display:<30} {score:>7.3f}")
                print(f"  Shape: {source_shape}")

            if len(sorted_mappings) > top_n:
                print(f"  ... and {len(sorted_mappings) - top_n} more matches")

        if show_unmatched:
            unmatched = self.get_unmatched()

            if unmatched["source"]:
                print(f"\n{'Unmatched source parameters:':-^80}")
                for name in unmatched["source"][:10]:
                    shape = self.source_params[name].shape
                    print(f"  {name:<60} {shape}")
                if len(unmatched["source"]) > 10:
                    print(f"  ... and {len(unmatched['source']) - 10} more")

            if unmatched["target"]:
                print(f"\n{'Unmatched target parameters:':-^80}")
                for name in unmatched["target"][:10]:
                    shape = self.target_params[name].shape
                    print(f"  {name:<60} {shape}")
                if len(unmatched["target"]) > 10:
                    print(f"  ... and {len(unmatched['target']) - 10} more")

        print("=" * 80)

    def get_mapping_dict(self) -> dict[str, str]:
        """Get the current mapping dictionary.

        Returns:
            Dictionary mapping source parameter names to target parameter names
        """
        if self._mapping is None:
            self.suggest_mapping()
        assert self._mapping is not None  # Type narrowing for type checker
        return self._mapping.copy()

    def get_mapping_with_scores(self) -> list[tuple[str, str, float]]:
        """Get mapping with confidence scores.

        Returns:
            List of tuples (source_name, target_name, score)
        """
        if self._mapping is None:
            self.suggest_mapping()

        assert self._mapping is not None  # Type narrowing for type checker
        assert self._scores is not None  # Type narrowing for type checker

        return [(source, target, self._scores[source]) for source, target in self._mapping.items()]

    def export_mapping_report(self, output_path: str | Path) -> None:
        """Export detailed mapping report to a file.

        Args:
            output_path: Path to save the report
        """
        import json

        if self._mapping is None:
            self.suggest_mapping()

        assert self._mapping is not None  # Type narrowing for type checker
        assert self._scores is not None  # Type narrowing for type checker

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

        print(f"Mapping report saved to: {output_path}")

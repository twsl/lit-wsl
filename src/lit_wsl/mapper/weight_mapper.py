from logging import Logger
from pathlib import Path
from typing import Any, Self

import torch
from torch import nn

from lit_wsl.mapper.hierarchy_analyzer import HierarchyAnalyzer
from lit_wsl.mapper.mapping_strategy import MappingStrategy
from lit_wsl.mapper.module_node import ModuleNode
from lit_wsl.mapper.parameter_extractor import ParameterExtractor
from lit_wsl.mapper.parameter_group import ParameterGroup
from lit_wsl.mapper.parameter_info import ParameterInfo
from lit_wsl.mapper.result_types import (
    GroupMatchResult,
    MappingResult,
    ParameterMatchResult,
    ScoreBreakdown,
    TransformationInfo,
)
from lit_wsl.mapper.similarity_scorer import SimilarityScorer
from lit_wsl.utils.logger import get_logger


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
        >>> from lit_wsl.mapper.weight_mapper import WeightMapper
        >>> from lit_wsl.models.weight_renamer import WeightRenamer
        >>> import torch
        >>>
        >>> # Basic usage - automatic threshold selection
        >>> source_model = OldModel()
        >>> target_model = NewModel()
        >>> mapper = WeightMapper(source_model, target_model)
        >>> mapping, unmatched = mapper.suggest_mapping()  # Returns all matches
        >>>
        >>> # With confidence scores
        >>> mapping_with_scores, unmatched = mapper.suggest_mapping(return_scores=True)
        >>> for src, (tgt, score) in mapping_with_scores.items():
        ...     print(f"{src} -> {tgt} (confidence: {score:.3f})")
        >>> # Check unmatched sources
        >>> for src in unmatched:
        ...     print(f"Unmatched: {src}")
        >>>
        >>> # With threshold filtering
        >>> mapping, unmatched = mapper.suggest_mapping(threshold=0.6)
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
        >>> mapping, unmatched = mapper.suggest_mapping(threshold=0.6)
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
        buffer_matching_mode: str = "lenient",
        *,
        source_params: dict[str, ParameterInfo] | None = None,
        target_params: dict[str, ParameterInfo] | None = None,
        dummy_input: torch.Tensor | None = None,
        incompatible_pairs: list[tuple[set[str], set[str]]] | None = None,
    ):
        """Initialize the WeightMapper.

        Args:
            source_module: The source model (with weights to adapt from)
            target_module: The target model (to adapt weights to)
            shape_tolerance: Relative tolerance for shape matching (0.0 = exact match only)
            buffer_matching_mode: How to handle buffer matching (default: 'lenient')
                - 'strict': Buffers must match shapes exactly (safest, original behavior)
                - 'lenient': Buffer shape mismatches get soft penalty, not hard reject
                - 'exclude': Ignore all buffers in matching (trainable params only)
            source_params: Pre-extracted source parameters (internal use)
            target_params: Pre-extracted target parameters (internal use)
            dummy_input: Optional dummy input tensor for execution order tracking.
                        Works perfectly fine without it. When provided, runs a forward pass
                        to track layer execution order, which can improve matching scores
                        by ~2% on average. Recommended for complex architectures with
                        significant structural changes.
            incompatible_pairs: Optional list of incompatible component pairs for semantic matching.
                        Each pair is a tuple of two sets of semantic chunks. Passed to
                        HierarchyAnalyzer. If None (default), no cross-component restrictions
                        are applied. Provide explicit pairs to prevent specific combinations
                        (e.g., [({{"backbone"}}, {{"head"}})] to prevent backbone-head matches).
        """
        self.logger: Logger = get_logger(self.__class__.__name__)
        self.source_module = source_module
        self.target_module = target_module
        self.shape_tolerance = shape_tolerance
        self.buffer_matching_mode = buffer_matching_mode
        self.dummy_input = dummy_input

        # Initialize modular components
        self.extractor = ParameterExtractor()
        self.scorer = SimilarityScorer(shape_tolerance=shape_tolerance, buffer_matching_mode=buffer_matching_mode)
        self.hierarchy_analyzer = HierarchyAnalyzer(incompatible_pairs=incompatible_pairs)

        # Extract parameter information
        if source_params is not None:
            self.source_params = source_params
        elif source_module is not None:
            self.source_params = self.extractor.extract_from_module(source_module, dummy_input=dummy_input)
        else:
            raise ValueError("Either source_module or source_params must be provided")

        if target_params is not None:
            self.target_params = target_params
        elif target_module is not None:
            self.target_params = self.extractor.extract_from_module(target_module, dummy_input=dummy_input)
        else:
            raise ValueError("Either target_module or target_params must be provided")

        # Build shape index for fast lookups
        self.target_by_shape = self.extractor.build_shape_index(self.target_params)

        # Extract parameter groups
        self.source_groups = self.extractor.extract_parameter_groups(self.source_params)
        self.target_groups = self.extractor.extract_parameter_groups(self.target_params)

        # Build hierarchical structure
        self.source_hierarchy = self.hierarchy_analyzer.build_hierarchy(self.source_groups)
        self.target_hierarchy = self.hierarchy_analyzer.build_hierarchy(self.target_groups)

        # Build group index by parameter types for fast lookups
        self.target_groups_by_types = self.extractor.build_group_index(self.target_groups)

        # Storage for mapping results
        self._result: MappingResult | None = None

    @classmethod
    def from_state_dict(
        cls,
        source_state_dict: dict[str, torch.Tensor],
        target_module: nn.Module,
        shape_tolerance: float = 0.0,
        dummy_input: torch.Tensor | None = None,
        incompatible_pairs: list[tuple[set[str], set[str]]] | None = None,
    ) -> Self:
        """Create a WeightMapper from a source state dictionary and target module.

        This is useful when you only have a checkpoint file but not the original model.

        Args:
            source_state_dict: State dictionary from the source model (e.g., loaded checkpoint)
            target_module: The target model to adapt weights to
            shape_tolerance: Relative tolerance for shape matching (0.0 = exact match only)
            dummy_input: (Optional) Dummy input tensor for execution order tracking.
                        Provides ~2% average score improvement when included.
            incompatible_pairs: List of incompatible component pairs for semantic matching.
                        See __init__ for details.

        Returns:
            WeightMapper instance

        Example:
            >>> import torch
            >>> from lit_wsl.mapper.weight_mapper import WeightMapper
            >>>
            >>> # Load old weights
            >>> old_weights = torch.load("old_model.pth")
            >>> if "state_dict" in old_weights:
            ...     old_weights = old_weights["state_dict"]
            >>>
            >>> # Create mapper and get mappings with scores
            >>> new_model = NewModel()
            >>> mapper = WeightMapper.from_state_dict(old_weights, new_model)
            >>> mapping_with_scores, unmatched = mapper.suggest_mapping(return_scores=True)
            >>>
            >>> # Filter by confidence threshold
            >>> high_confidence = {src: tgt for src, (tgt, score) in mapping_with_scores.items() if score > 0.7}
        """
        # Extract parameters from state dict
        extractor = ParameterExtractor()
        source_params = extractor.extract_from_state_dict(source_state_dict)
        target_params = extractor.extract_from_module(target_module, dummy_input=dummy_input)

        return cls(
            source_module=None,
            target_module=target_module,
            shape_tolerance=shape_tolerance,
            source_params=source_params,
            target_params=target_params,
            dummy_input=dummy_input,
            incompatible_pairs=incompatible_pairs,
        )

    def suggest_mapping(
        self,
        threshold: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> MappingResult:
        """Generate suggested parameter name mapping with full transparency.

        This method uses a hierarchical matching strategy: first groups modules together,
        then handles remaining parameters individually. Returns complete transparency into
        all scoring components and match decisions.

        Args:
            threshold: Minimum score threshold for suggesting a match.
                      If None (default), uses 0.0 (returns all possible matches).
                      Recommended values: 0.5-0.7 for typical use cases.
            weights: Custom weights for scoring components
                     Default: {'shape': 0.4, 'name': 0.1, 'hierarchy': 0.5}

        Returns:
            MappingResult with complete information about all matches, scores,
            and transparency data

        Example:
            >>> mapper = WeightMapper(source_model, target_model)
            >>> result = mapper.suggest_mapping(threshold=0.6)
            >>> # Access simple mapping
            >>> mapping_dict = result.get_mapping()
            >>> # Inspect low confidence matches
            >>> low_conf = result.get_low_confidence_matches(0.7)
            >>> for match in low_conf:
            ...     print(f"{match.source_name} -> {match.target_name}")
            ...     print(
            ...         f"  Score breakdown: shape={match.score_breakdown.shape_score:.2f}, "
            ...         f"name={match.score_breakdown.name_score:.2f}"
            ...     )
            >>> # Access all scored details
            >>> for param_name, match_result in result.parameter_matches.items():
            ...     if match_result.matched:
            ...         print(f"{param_name}: {match_result.final_score:.3f}")
        """
        # Set default threshold
        if threshold is None:
            threshold = 0.0

        if weights is None:
            weights = {"shape": 0.4, "name": 0.1, "hierarchy": 0.5}

        # Log warning if no execution order available
        has_execution_order = any(p.execution_order is not None for p in self.source_params.values()) and any(
            p.execution_order is not None for p in self.target_params.values()
        )

        if not has_execution_order and self.dummy_input is None:
            self.logger.warning(
                "No execution order available (dummy_input not provided). "
                "Cross-component mappings will rely on structural heuristics only. "
                "Consider providing dummy_input for more accurate matching."
            )

        # Create mapping strategy instance
        mapping_strategy = MappingStrategy(
            source_params=self.source_params,
            target_params=self.target_params,
            source_groups=self.source_groups,
            target_groups=self.target_groups,
            scorer=self.scorer,
            hierarchy_analyzer=self.hierarchy_analyzer,
            buffer_matching_mode=self.buffer_matching_mode,
        )

        # Phase 1: Group-based matching
        group_results = mapping_strategy.suggest_group_mapping(threshold, weights)

        # Build group mapping dict for tracking
        group_mapping = {gr.source_path: gr.target_path for gr in group_results if gr.matched}

        # Phase 2: Convert group matches to parameter matches
        parameter_results = []
        group_matched_params = set()

        for group_result in group_results:
            if not group_result.matched:
                continue

            source_group = self.source_groups[group_result.source_path]
            target_group = self.target_groups[group_result.target_path]  # ty:ignore[invalid-argument-type]

            for param_type in group_result.param_types_matched:
                source_param = source_group.params[param_type]
                target_param = target_group.params[param_type]

                # Compute detailed score for this specific parameter
                score_breakdown = self.scorer.compute_composite_score(
                    source_param,
                    target_param,
                    weights,
                    group_mapping,  # ty:ignore[invalid-argument-type]
                    self.hierarchy_analyzer,
                )

                # Get transformation info
                _, transform_info = self.scorer.compute_shape_score_with_transform(source_param, target_param)

                param_result = ParameterMatchResult(
                    source_name=source_param.name,
                    target_name=target_param.name,
                    score_breakdown=score_breakdown,
                    final_score=score_breakdown.composite_score,
                    matched=True,
                    unmatch_reason=None,
                    match_type="group",
                    transformation=transform_info,
                    source_module_path=source_param.module_path,
                    target_module_path=target_param.module_path,
                )

                parameter_results.append(param_result)
                group_matched_params.add(source_param.name)

        # Phase 3: Individual fallback matching
        unmapped_source = set(self.source_params.keys()) - group_matched_params
        unmapped_target = set(self.target_params.keys()) - {r.target_name for r in parameter_results if r.target_name}

        if unmapped_source and unmapped_target:
            individual_results = mapping_strategy.suggest_individual_mapping(
                unmapped_source, unmapped_target, threshold, weights, group_results
            )
            parameter_results.extend(individual_results)

        # Phase 4: Build MappingResult
        # Create a lookup dict for results
        results_by_source = {r.source_name: r for r in parameter_results}

        # Build parameter_matches in source_params order to preserve ordering
        parameter_matches = {
            source_name: results_by_source[source_name]
            for source_name in self.source_params
            if source_name in results_by_source
        }
        group_matches = {gr.source_path: gr for gr in group_results}

        matched_params = [r for r in parameter_results if r.matched]
        unmatched_params = [r for r in parameter_results if not r.matched]

        # Find unmatched target parameters
        all_matched_targets = {r.target_name for r in matched_params if r.target_name}
        unmatched_targets = [name for name in self.target_params if name not in all_matched_targets]

        coverage = len(matched_params) / len(self.source_params) if self.source_params else 0.0

        result = MappingResult(
            parameter_matches=parameter_matches,
            group_matches=group_matches,
            matched_params=matched_params,
            unmatched_params=unmatched_params,
            unmatched_targets=unmatched_targets,
            coverage=coverage,
            threshold=threshold,
            weights=weights,
        )

        # Store result
        self._result = result

        return result

    def get_unmatched(self) -> dict[str, list[str]]:
        """Get parameters that couldn't be matched.

        Returns:
            Dictionary with 'source' and 'target' lists of unmatched parameter names
        """
        if self._result is None:
            self._result = self.suggest_mapping()

        return {
            "source": [r.source_name for r in self._result.unmatched_params],
            "target": self._result.unmatched_targets,
        }

    def print_analysis(self, top_n: int = 10, show_unmatched: bool = True) -> None:
        """Print detailed analysis of the mapping.

        Args:
            top_n: Number of top mappings to display
            show_unmatched: Whether to show unmatched parameters
        """
        if self._result is None:
            self._result = self.suggest_mapping()

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
        self.logger.info(f"  Matched: {len(self._result.matched_params)}")
        self.logger.info(f"  Coverage: {self._result.coverage * 100:.1f}%")

        if self._result.matched_params:
            self.logger.info(f"\n{'Top suggested mappings:':-^80}")
            self.logger.info(f"{'Source':<40} → {'Target':<30} {'Score':>8}")
            self.logger.info("-" * 80)

            # Sort by score
            sorted_matches = sorted(
                self._result.matched_params,
                key=lambda x: x.final_score,
                reverse=True,
            )

            for match in sorted_matches[:top_n]:
                source_shape = self.source_params[match.source_name].shape

                # Truncate names if too long
                source_display = match.source_name if len(match.source_name) <= 38 else match.source_name[:35] + "..."
                target_display = (
                    match.target_name
                    if match.target_name and len(match.target_name) <= 28
                    else (match.target_name[:25] + "..." if match.target_name else "")
                )

                self.logger.info(f"{source_display:<40} → {target_display:<30} {match.final_score:>7.3f}")
                self.logger.info(f"  Shape: {source_shape}")

            if len(sorted_matches) > top_n:
                self.logger.info(f"  ... and {len(sorted_matches) - top_n} more matches")

        if show_unmatched:
            if self._result.unmatched_params:
                self.logger.info(f"\n{'Unmatched source parameters:':-^80}")
                for match in self._result.unmatched_params[:10]:
                    shape = self.source_params[match.source_name].shape
                    reason = match.unmatch_reason or "unknown"
                    self.logger.info(f"  {match.source_name:<50} {shape} ({reason})")
                if len(self._result.unmatched_params) > 10:
                    self.logger.info(f"  ... and {len(self._result.unmatched_params) - 10} more")

            if self._result.unmatched_targets:
                self.logger.info(f"\n{'Unmatched target parameters:':-^80}")
                for name in self._result.unmatched_targets[:10]:
                    shape = self.target_params[name].shape
                    self.logger.info(f"  {name:<60} {shape}")
                if len(self._result.unmatched_targets) > 10:
                    self.logger.info(f"  ... and {len(self._result.unmatched_targets) - 10} more")

        self.logger.info("=" * 80)

    def get_mapping_dict(self) -> dict[str, str]:
        """Get the current mapping dictionary.

        Returns:
            Dictionary mapping source parameter names to target parameter names
        """
        if self._result is None:
            self._result = self.suggest_mapping()
        return self._result.get_mapping()

    def export_mapping_report(self, output_path: str | Path) -> None:
        """Export detailed mapping report to a file.

        Args:
            output_path: Path to save the report
        """
        import json

        if self._result is None:
            self._result = self.suggest_mapping()

        report = {
            "source_model": self.source_module.__class__.__name__ if self.source_module else "StateDict",
            "target_model": self.target_module.__class__.__name__ if self.target_module else "StateDict",
            "source_params_count": len(self.source_params),
            "target_params_count": len(self.target_params),
            "matched_count": len(self._result.matched_params),
            "coverage": self._result.coverage,
            "mappings": [
                {
                    "source": match.source_name,
                    "target": match.target_name,
                    "score": match.final_score,
                    "shape": list(self.source_params[match.source_name].shape),
                    "match_type": match.match_type,
                }
                for match in self._result.matched_params
            ],
            "unmatched_source": [
                {"name": match.source_name, "reason": match.unmatch_reason} for match in self._result.unmatched_params
            ],
            "unmatched_target": self._result.unmatched_targets,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Mapping report saved to: {output_path}")

    def get_score_breakdown(self, source_param: str) -> ScoreBreakdown:
        """Get detailed score breakdown for a specific parameter.

        Args:
            source_param: Source parameter name

        Returns:
            ScoreBreakdown with all component scores

        Raises:
            KeyError: If parameter not found in mapping results
        """
        if self._result is None:
            self._result = self.suggest_mapping()

        if source_param not in self._result.parameter_matches:
            raise KeyError(f"Parameter '{source_param}' not found in mapping results")

        return self._result.parameter_matches[source_param].score_breakdown

    def get_parameter_details(self, source_param: str) -> ParameterMatchResult:
        """Get full match details for a specific parameter.

        Args:
            source_param: Source parameter name

        Returns:
            ParameterMatchResult with complete match information

        Raises:
            KeyError: If parameter not found in mapping results
        """
        if self._result is None:
            self._result = self.suggest_mapping()

        if source_param not in self._result.parameter_matches:
            raise KeyError(f"Parameter '{source_param}' not found in mapping results")

        return self._result.parameter_matches[source_param]

    def get_group_details(self, source_path: str) -> GroupMatchResult:
        """Get full match details for a specific module group.

        Args:
            source_path: Source module path

        Returns:
            GroupMatchResult with complete group match information

        Raises:
            KeyError: If group not found in mapping results
        """
        if self._result is None:
            self._result = self.suggest_mapping()

        if source_path not in self._result.group_matches:
            raise KeyError(f"Group '{source_path}' not found in mapping results")

        return self._result.group_matches[source_path]

    def get_compatible_groups(
        self,
        source_path: str | None = None,
        threshold: float = 0.0,
        max_candidates: int | None = None,
    ) -> dict[str, list[GroupMatchResult]]:
        """Get all compatible group mappings with scores (not just best match).

        This reveals all scored candidates for exploring alternative mappings.

        Args:
            source_path: Specific source group to analyze (None = all groups)
            threshold: Minimum score threshold for candidates
            max_candidates: Maximum candidates to return per source (None = all)

        Returns:
            Dictionary mapping source paths to lists of compatible GroupMatchResult candidates

        Example:
            >>> mapper = WeightMapper(source_model, target_model)
            >>> # Get all compatible targets for a specific module
            >>> candidates = mapper.get_compatible_groups("backbone.layer1")
            >>> for result in candidates["backbone.layer1"]:
            ...     print(f"{result.target_path}: {result.combined_score:.3f}")
        """
        mapping_strategy = MappingStrategy(
            source_params=self.source_params,
            target_params=self.target_params,
            source_groups=self.source_groups,
            target_groups=self.target_groups,
            scorer=self.scorer,
            hierarchy_analyzer=self.hierarchy_analyzer,
            buffer_matching_mode=self.buffer_matching_mode,
        )

        return mapping_strategy.suggest_compatible_groups(source_path, threshold, None, max_candidates)

    def visualize_mapping(
        self,
        result: MappingResult | None = None,
        *,
        show_unmatched: bool = True,
        max_matches: int = 20,
        max_unmatched: int = 10,
    ) -> None:
        """Visualize mapping results using Rich for beautiful console output.

        Displays a summary panel, matched parameters table with color-coded scores,
        transformation indicators, and optionally unmatched parameters.

        Args:
            result: MappingResult to visualize (uses cached result if None)
            show_unmatched: Whether to display unmatched parameters
            max_matches: Maximum number of matches to display
            max_unmatched: Maximum number of unmatched items to display

        Example:
            >>> mapper = WeightMapper(source_model, target_model)
            >>> result = mapper.suggest_mapping(threshold=0.6)
            >>> mapper.visualize_mapping(result, max_matches=30)
        """
        from rich.console import Console

        from lit_wsl.mapper.rich_visualizer import print_mapping_analysis

        if result is None:
            if self._result is None:
                self._result = self.suggest_mapping()
            result = self._result

        console = Console()
        print_mapping_analysis(
            result=result,
            source_count=len(self.source_params),
            target_count=len(self.target_params),
            console=console,
            show_unmatched=show_unmatched,
            max_matches=max_matches,
            max_unmatched=max_unmatched,
        )

    def visualize_hierarchies(
        self,
        *,
        show_matches: bool = False,
        max_depth: int | None = None,
    ) -> None:
        """Visualize source and target hierarchies side by side using Rich trees.

        Displays the module hierarchy of both models with optional highlighting
        of matched modules for visual comparison.

        Args:
            show_matches: Whether to highlight matched modules (requires suggest_mapping() first)
            max_depth: Maximum tree depth to display (None for unlimited)

        Example:
            >>> mapper = WeightMapper(source_model, target_model)
            >>> # Show basic hierarchies
            >>> mapper.visualize_hierarchies(max_depth=3)
            >>>
            >>> # Show with matches highlighted
            >>> mapper.suggest_mapping()
            >>> mapper.visualize_hierarchies(show_matches=True, max_depth=5)
        """
        from rich.console import Console

        from lit_wsl.mapper.rich_visualizer import print_side_by_side_hierarchies

        console = Console()

        # Get matched paths if requested
        matched_source_paths = None
        matched_target_paths = None

        if show_matches:
            if self._result is None:
                self._result = self.suggest_mapping()

            # Extract matched module paths from group matches
            matched_source_paths = {path for path, match in self._result.group_matches.items() if match.matched}
            matched_target_paths = {
                match.target_path
                for match in self._result.group_matches.values()
                if match.matched and match.target_path
            }

        print_side_by_side_hierarchies(
            source_root=self.source_hierarchy,
            target_root=self.target_hierarchy,
            matched_source_paths=matched_source_paths,
            matched_target_paths=matched_target_paths,
            max_depth=max_depth,
            console=console,
        )

    def visualize_score_breakdown(self, source_param: str) -> None:
        """Visualize detailed score breakdown for a specific parameter.

        Shows all scoring components (shape, name, hierarchy) with weights,
        contributions, and visual progress bars using Rich tables.

        Args:
            source_param: Source parameter name to analyze

        Raises:
            KeyError: If parameter not found in mapping results

        Example:
            >>> mapper = WeightMapper(source_model, target_model)
            >>> mapper.suggest_mapping()
            >>> mapper.visualize_score_breakdown("backbone.conv1.weight")
        """
        from rich.console import Console

        from lit_wsl.mapper.rich_visualizer import render_score_breakdown

        match = self.get_parameter_details(source_param)

        console = Console()
        console.print()
        console.print(render_score_breakdown(match.score_breakdown, source_param))
        console.print()

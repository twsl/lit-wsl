from logging import Logger
from typing import TYPE_CHECKING, Any

from lit_wsl.mapper.hierarchy_analyzer import HierarchyAnalyzer
from lit_wsl.mapper.parameter_group import ParameterGroup
from lit_wsl.mapper.parameter_info import ParameterInfo
from lit_wsl.mapper.result_types import GroupMatchResult, ParameterMatchResult, ScoreBreakdown, TransformationInfo
from lit_wsl.mapper.similarity_scorer import SimilarityScorer
from lit_wsl.utils.logger import get_logger


class MappingStrategy:
    """Strategy for suggesting parameter mappings between source and target models.

    This class implements the core mapping algorithm:
    - Group-based matching with exact parameter type and shape requirements
    - Hierarchical structure scoring
    - Individual parameter fallback matching
    - Validation of 1-to-1 mapping constraints
    """

    def __init__(
        self,
        source_params: dict[str, ParameterInfo],
        target_params: dict[str, ParameterInfo],
        source_groups: dict[str, ParameterGroup],
        target_groups: dict[str, ParameterGroup],
        scorer: SimilarityScorer,
        hierarchy_analyzer: HierarchyAnalyzer,
        buffer_matching_mode: str = "lenient",
    ) -> None:
        """Initialize the MappingStrategy.

        Args:
            source_params: Dictionary of source parameter information
            target_params: Dictionary of target parameter information
            source_groups: Dictionary of source parameter groups
            target_groups: Dictionary of target parameter groups
            scorer: SimilarityScorer instance for computing scores
            hierarchy_analyzer: HierarchyAnalyzer instance for hierarchy checks
            buffer_matching_mode: How to handle buffer matching (default: 'lenient')
                - 'strict': Require exact buffer matches
                - 'lenient': Allow buffer shape mismatches with soft penalty
                - 'exclude': Ignore buffers entirely in matching
        """
        self.logger: Logger = get_logger(self.__class__.__name__)
        self.source_params = source_params
        self.target_params = target_params
        self.source_groups = source_groups
        self.target_groups = target_groups
        self.scorer = scorer
        self.hierarchy_analyzer = hierarchy_analyzer
        self.buffer_matching_mode = buffer_matching_mode

    def suggest_group_mapping(
        self,
        threshold: float,
        weights: dict[str, float] | None = None,
    ) -> list[GroupMatchResult]:
        """Suggest mapping at the group level using hierarchical structure.

        This method uses a two-phase approach:
        1. COMPATIBLE MATCH: First filter candidates by compatible param_types and shapes
        2. STRUCTURAL SCORING: Then score by hierarchy, step numbers, and chunks

        Args:
            threshold: Minimum score threshold
            weights: Custom weights for scoring

        Returns:
            List of GroupMatchResult objects for all source groups (matched and unmatched)
        """
        results = []
        group_mapping = {}  # Track for context scoring
        used_targets = set()

        # Sort source groups by depth (shallow first) and then by path
        # This ensures parent modules are matched before children
        sorted_source_paths = sorted(self.source_groups.keys(), key=lambda x: (x.count("."), x))

        for source_path in sorted_source_paths:
            source_group = self.source_groups[source_path]
            source_metadata = self.hierarchy_analyzer.extract_hierarchy_metadata(source_path)

            # PHASE 1: Filter by compatible param_types and shapes match
            scored_candidates = []
            for target_path, target_group in self.target_groups.items():
                if target_path in used_targets:
                    continue

                if not source_group.is_compatible_with_mode(target_group, self.buffer_matching_mode):
                    continue

                # Additional verification: Check that each common param_type maps correctly
                common_types = source_group.param_types & target_group.param_types
                verification_failed = False
                for param_type in common_types:
                    source_param = source_group.params[param_type]
                    target_param = target_group.params[param_type]
                    if source_param.param_name != target_param.param_name:
                        self.logger.warning(
                            f"VERIFICATION FAILED: param_name mismatch for type '{param_type}' "
                            f"in groups {source_path} -> {target_path}: "
                            f"{source_param.param_name} != {target_param.param_name}"
                        )
                        verification_failed = True
                        break

                if verification_failed:
                    continue

                # PHASE 2: Compute scores
                target_metadata = self.hierarchy_analyzer.extract_hierarchy_metadata(target_path)
                structure_score = self.hierarchy_analyzer.compute_hierarchy_structure_score(
                    source_metadata, target_metadata
                )
                base_score = self.scorer.compute_group_similarity(
                    source_group, target_group, weights, group_mapping, self.hierarchy_analyzer
                )

                if base_score < threshold:
                    continue

                combined_score = 0.6 * structure_score + 0.4 * base_score
                context_score = self.hierarchy_analyzer.compute_hierarchy_context_score(
                    source_path, target_path, group_mapping
                )

                # Build param type shapes mapping
                param_type_shapes = {}
                for param_type in common_types:
                    source_param = source_group.params[param_type]
                    target_param = target_group.params[param_type]
                    param_type_shapes[param_type] = (source_param.shape, target_param.shape)

                scored_candidates.append(
                    (
                        target_path,
                        target_metadata,
                        combined_score,
                        structure_score,
                        base_score,
                        context_score,
                        common_types,
                        param_type_shapes,
                    )
                )

            # PHASE 3: Select best match and create result
            if scored_candidates:
                scored_candidates.sort(key=lambda x: (x[2], x[5]), reverse=True)
                (
                    best_target,
                    target_metadata,
                    combined_score,
                    structure_score,
                    base_score,
                    context_score,
                    common_types,
                    param_type_shapes,
                ) = scored_candidates[0]

                result = GroupMatchResult(
                    source_path=source_path,
                    target_path=best_target,
                    combined_score=combined_score,
                    structure_score=structure_score,
                    base_score=base_score,
                    context_score=context_score,
                    source_metadata=source_metadata,
                    target_metadata=target_metadata,
                    matched=True,
                    unmatch_reason=None,
                    param_types_matched=common_types,
                    param_type_shapes=param_type_shapes,
                )

                group_mapping[source_path] = best_target
                used_targets.add(best_target)
            else:
                # Create unmatched result
                unmatch_reason = (
                    "no_compatible_target_group"
                    if not any(source_group.is_compatible_with(tg) for tg in self.target_groups.values())
                    else f"score_below_threshold_{threshold}"
                )

                result = GroupMatchResult(
                    source_path=source_path,
                    target_path=None,
                    combined_score=0.0,
                    structure_score=0.0,
                    base_score=0.0,
                    context_score=0.0,
                    source_metadata=source_metadata,
                    target_metadata=None,
                    matched=False,
                    unmatch_reason=unmatch_reason,
                    param_types_matched=set(),
                    param_type_shapes={},
                )

            results.append(result)

        return results

    def suggest_individual_mapping(
        self,
        unmapped_source: set[str],
        unmapped_target: set[str],
        threshold: float,
        weights: dict[str, float] | None = None,
        group_results: list[GroupMatchResult] | None = None,
    ) -> list[ParameterMatchResult]:
        """Suggest mapping for individual parameters that weren't matched at group level.

        This fallback method tries to match remaining unmapped parameters individually,
        which is useful when groups are incompatible but individual parameters could still match.

        Note: Hierarchical constraints are now SOFT (penalties) not hard rejections.

        Args:
            unmapped_source: Set of source parameter names not yet mapped
            unmapped_target: Set of target parameter names not yet mapped
            threshold: Minimum score threshold
            weights: Custom weights for scoring
            group_results: Optional group match results for additional context

        Returns:
            List of ParameterMatchResult objects for all unmapped parameters
        """
        results = []
        used_targets = set()

        # Preserve source_params order
        sorted_source = [name for name in self.source_params if name in unmapped_source]

        for source_name in sorted_source:
            source_info = self.source_params[source_name]

            # Find candidates
            candidates = []
            for target_name in unmapped_target:
                if target_name in used_targets:
                    continue

                target_info = self.target_params[target_name]

                # Compute individual parameter score (no hard hierarchical rejection)
                score_breakdown = self.scorer.compute_composite_score(
                    source_info, target_info, weights, None, self.hierarchy_analyzer
                )

                if score_breakdown.composite_score >= threshold:
                    # Get transformation info
                    _, transform_info = self.scorer.compute_shape_score_with_transform(source_info, target_info)
                    candidates.append((target_name, score_breakdown, transform_info))

            # Select best match
            if candidates:
                candidates.sort(key=lambda x: x[1].composite_score, reverse=True)
                best_target, best_score_breakdown, transform_info = candidates[0]

                result = ParameterMatchResult(
                    source_name=source_name,
                    target_name=best_target,
                    score_breakdown=best_score_breakdown,
                    final_score=best_score_breakdown.composite_score,
                    matched=True,
                    unmatch_reason=None,
                    match_type="individual",
                    transformation=transform_info,
                    source_module_path=source_info.module_path,
                    target_module_path=self.target_params[best_target].module_path,
                )
                used_targets.add(best_target)
            else:
                # Create unmatched result
                if not any(
                    source_info.shape == self.target_params[t].shape for t in unmapped_target if t not in used_targets
                ):
                    unmatch_reason = f"no_matching_shape_{source_info.shape}"
                else:
                    unmatch_reason = f"score_below_threshold_{threshold}"

                result = ParameterMatchResult(
                    source_name=source_name,
                    target_name=None,
                    score_breakdown=ScoreBreakdown(
                        shape_score=0.0,
                        name_score=0.0,
                        hierarchy_score=0.0,
                        composite_score=0.0,
                        weights_used=weights or {},
                    ),
                    final_score=0.0,
                    matched=False,
                    unmatch_reason=unmatch_reason,
                    match_type="individual",
                    transformation=None,
                    source_module_path=source_info.module_path,
                    target_module_path=None,
                )

            results.append(result)

        return results

    def validate_mapping(self, mapping: dict[str, str]) -> None:
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

    def compute_transformations(self, mapping: dict[str, str]) -> dict[str, TransformationInfo]:
        """Compute transformation information for each mapped parameter.

        Args:
            mapping: Dictionary mapping source parameter names to target parameter names

        Returns:
            Dictionary mapping source parameter names to TransformationInfo dataclass
            (only includes parameters that need transformation)
        """
        transformations = {}
        for source_name, target_name in mapping.items():
            source_info = self.source_params[source_name]
            target_info = self.target_params[target_name]
            _, transform_info = self.scorer.compute_shape_score_with_transform(source_info, target_info)
            if transform_info is not None:
                transformations[source_name] = transform_info
        return transformations

    def suggest_compatible_groups(
        self,
        source_path: str | None = None,
        threshold: float = 0.0,
        weights: dict[str, float] | None = None,
        max_candidates: int | None = None,
    ) -> dict[str, list[GroupMatchResult]]:
        """Get all compatible group mappings with scores (not just best match).

        This reveals the full compatibility landscape, showing all scored candidates
        that were considered during matching. Useful for exploring alternative mappings
        and understanding matching decisions.

        Args:
            source_path: Specific source group to analyze (None = all groups)
            threshold: Minimum score threshold for candidates
            weights: Custom weights for scoring
            max_candidates: Maximum candidates to return per source (None = all)

        Returns:
            Dictionary mapping source paths to lists of GroupMatchResult candidates,
            sorted by score descending
        """
        results = {}
        source_paths = (
            [source_path] if source_path else sorted(self.source_groups.keys(), key=lambda x: (x.count("."), x))
        )

        for src_path in source_paths:
            source_group = self.source_groups[src_path]
            source_metadata = self.hierarchy_analyzer.extract_hierarchy_metadata(src_path)
            candidates = []

            # Score all compatible targets
            for target_path, target_group in self.target_groups.items():
                if not source_group.is_compatible_with(target_group):
                    continue

                # Verify param names match
                common_types = source_group.param_types & target_group.param_types
                verification_failed = False
                for param_type in common_types:
                    if source_group.params[param_type].param_name != target_group.params[param_type].param_name:
                        verification_failed = True
                        break

                if verification_failed:
                    continue

                # Compute scores
                target_metadata = self.hierarchy_analyzer.extract_hierarchy_metadata(target_path)
                structure_score = self.hierarchy_analyzer.compute_hierarchy_structure_score(
                    source_metadata, target_metadata
                )
                base_score = self.scorer.compute_group_similarity(
                    source_group, target_group, weights, {}, self.hierarchy_analyzer
                )

                combined_score = 0.6 * structure_score + 0.4 * base_score

                if combined_score < threshold:
                    continue

                context_score = self.hierarchy_analyzer.compute_hierarchy_context_score(src_path, target_path, {})

                # Build param type shapes
                param_type_shapes = {}
                for param_type in common_types:
                    source_param = source_group.params[param_type]
                    target_param = target_group.params[param_type]
                    param_type_shapes[param_type] = (source_param.shape, target_param.shape)

                candidate = GroupMatchResult(
                    source_path=src_path,
                    target_path=target_path,
                    combined_score=combined_score,
                    structure_score=structure_score,
                    base_score=base_score,
                    context_score=context_score,
                    source_metadata=source_metadata,
                    target_metadata=target_metadata,
                    matched=True,
                    unmatch_reason=None,
                    param_types_matched=common_types,
                    param_type_shapes=param_type_shapes,
                )
                candidates.append(candidate)

            # Sort by combined score
            candidates.sort(key=lambda x: x.combined_score, reverse=True)

            # Apply max_candidates limit
            if max_candidates is not None:
                candidates = candidates[:max_candidates]

            results[src_path] = candidates

        return results

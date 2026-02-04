"""Structured result types for weight mapping with full scoring transparency.

This module defines dataclasses that replace the old tuple/dict return types,
providing comprehensive visibility into all scoring components and metadata.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoreBreakdown:
    """Detailed breakdown of similarity scoring components.

    Provides complete transparency into how the composite score was calculated,
    exposing all individual components that were previously hidden.

    Attributes:
        shape_score: Shape similarity score (0.0-1.0)
        name_score: Parameter name similarity score (0.0-1.0)
        hierarchy_score: Hierarchical structure similarity score (0.0-1.0)
        composite_score: Final weighted composite score (0.0-1.0)
        weights_used: Weights applied to each component
        token_score: Token overlap (Jaccard) score (0.0-1.0), None if not computed
        edit_score: Edit distance (Levenshtein) score (0.0-1.0), None if not computed
        lcs_score: Longest common substring score (0.0-1.0), None if not computed
        depth_score: Module depth similarity score (0.0-1.0), None if not computed
        path_score: Module path similarity score (0.0-1.0), None if not computed
        order_score: Execution order similarity score (0.0-1.0), None if not computed
    """

    shape_score: float
    name_score: float
    hierarchy_score: float
    composite_score: float
    weights_used: dict[str, float]
    token_score: float | None = None
    edit_score: float | None = None
    lcs_score: float | None = None
    depth_score: float | None = None
    path_score: float | None = None
    order_score: float | None = None

    def __post_init__(self) -> None:
        """Validate score ranges."""
        for score_name in [
            "shape_score",
            "name_score",
            "hierarchy_score",
            "composite_score",
            "token_score",
            "edit_score",
            "lcs_score",
            "depth_score",
            "path_score",
            "order_score",
        ]:
            score = getattr(self, score_name)
            if score is not None and not 0.0 <= score <= 1.0:
                msg = f"{score_name} must be between 0.0 and 1.0, got {score}"
                raise ValueError(msg)


@dataclass(frozen=True)
class HierarchyMetadata:
    """Hierarchical structure information for a module path.

    Provides semantic and structural analysis of a module's position
    in the model hierarchy.

    Attributes:
        depth: Module depth (number of path components)
        chunks: Set of semantic tokens extracted from path
        numeric_indices: List of numeric indices found in path (e.g., [0, 1] for "layer.0.block.1")
        stages: Detailed breakdown of path stages with indices and chunks
    """

    depth: int
    chunks: set[str]
    numeric_indices: list[int | None]
    stages: list[dict[str, Any]]  # [{name, index, chunks}, ...]


@dataclass(frozen=True)
class TransformationInfo:
    """Information about required tensor transformations.

    Describes how a source parameter's shape needs to be transformed
    to match the target parameter's shape.

    Attributes:
        type: Transformation type ("transpose", "reshape", or "none")
        note: Human-readable description of the transformation
        source_shape: Original shape of source parameter
        target_shape: Expected shape of target parameter
    """

    type: str
    note: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate transformation type."""
        if self.type not in ("transpose", "reshape", "none"):
            msg = f"Invalid transformation type: {self.type}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ParameterMatchResult:
    """Result of matching a single parameter with full transparency.

    Provides complete information about how a source parameter was matched
    (or why it wasn't matched) including detailed score breakdowns and metadata.

    Attributes:
        source_name: Full parameter name in source model
        target_name: Full parameter name in target model (None if unmatched)
        score_breakdown: Detailed breakdown of all scoring components
        final_score: Overall match score (composite_score from breakdown)
        matched: Whether this parameter was successfully matched
        unmatch_reason: Reason for no match (None if matched)
        match_type: How the match was made ("group" or "individual")
        transformation: Required transformation (None if shapes match exactly)
        source_module_path: Parent module path in source model
        target_module_path: Parent module path in target model (None if unmatched)
    """

    source_name: str
    target_name: str | None
    score_breakdown: ScoreBreakdown
    final_score: float
    matched: bool
    unmatch_reason: str | None
    match_type: str  # "group" or "individual"
    transformation: TransformationInfo | None
    source_module_path: str
    target_module_path: str | None

    def __post_init__(self) -> None:
        """Validate consistency."""
        if self.matched and self.target_name is None:
            msg = "Matched parameters must have a target_name"
            raise ValueError(msg)
        if not self.matched and self.unmatch_reason is None:
            msg = "Unmatched parameters must have an unmatch_reason"
            raise ValueError(msg)
        if self.match_type not in ("group", "individual"):
            msg = f"Invalid match_type: {self.match_type}"
            raise ValueError(msg)


@dataclass(frozen=True)
class GroupMatchResult:
    """Result of matching a parameter group (module) with full transparency.

    Provides detailed information about module-level matching including
    structural scoring, hierarchy metadata, and parameter type compatibility.

    Attributes:
        source_path: Module path in source model
        target_path: Module path in target model (None if unmatched)
        combined_score: Overall match score (structure + base weighted)
        structure_score: Hierarchical structure similarity score
        base_score: Average parameter similarity score
        context_score: Hierarchical context bonus (parent/sibling matches)
        source_metadata: Hierarchical metadata for source module
        target_metadata: Hierarchical metadata for target module (None if unmatched)
        matched: Whether this group was successfully matched
        unmatch_reason: Reason for no match (None if matched)
        param_types_matched: Set of parameter types that matched (e.g., {"weight", "bias"})
        param_type_shapes: Mapping of param types to (source_shape, target_shape) tuples
    """

    source_path: str
    target_path: str | None
    combined_score: float
    structure_score: float
    base_score: float
    context_score: float
    source_metadata: HierarchyMetadata
    target_metadata: HierarchyMetadata | None
    matched: bool
    unmatch_reason: str | None
    param_types_matched: set[str]
    param_type_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]

    def __post_init__(self) -> None:
        """Validate consistency."""
        if self.matched and self.target_path is None:
            msg = "Matched groups must have a target_path"
            raise ValueError(msg)
        if not self.matched and self.unmatch_reason is None:
            msg = "Unmatched groups must have an unmatch_reason"
            raise ValueError(msg)


@dataclass(frozen=True)
class MappingResult:
    """Complete mapping result with full transparency and convenience access.

    The primary return type from WeightMapper.suggest_mapping(), providing
    comprehensive visibility into all aspects of the mapping process.

    Attributes:
        parameter_matches: All parameter match results keyed by source name
        group_matches: All group match results keyed by source path
        matched_params: List of successfully matched ParameterMatchResult objects
        unmatched_params: List of unmatched ParameterMatchResult objects
        unmatched_targets: List of target parameter names that weren't matched
        coverage: Fraction of source parameters successfully matched (0.0-1.0)
        threshold: Score threshold used for matching
        weights: Score component weights used for matching
    """

    parameter_matches: dict[str, ParameterMatchResult]
    group_matches: dict[str, GroupMatchResult]
    matched_params: list[ParameterMatchResult] = field(default_factory=list)
    unmatched_params: list[ParameterMatchResult] = field(default_factory=list)
    unmatched_targets: list[str] = field(default_factory=list)
    coverage: float = 0.0
    threshold: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)

    def get_mapping(self) -> dict[str, str]:
        """Get simple source -> target mapping dict (backward compatibility helper).

        Returns:
            Dictionary mapping source parameter names to target parameter names
            (only includes successfully matched parameters), preserving source order
        """
        # Build mapping from parameter_matches to preserve insertion order
        return {
            source_name: match.target_name
            for source_name, match in self.parameter_matches.items()
            if match.matched and match.target_name is not None
        }

    def get_mapping_with_scores(self) -> dict[str, tuple[str, float]]:
        """Get mapping with scores: source -> (target, score).

        Returns:
            Dictionary mapping source names to (target_name, score) tuples, preserving source order
        """
        return {
            source_name: (match.target_name, match.final_score)
            for source_name, match in self.parameter_matches.items()
            if match.matched and match.target_name is not None
        }

    def get_unmatched(self) -> dict[str, str]:
        """Get unmatched sources as dict: source -> reason (backward compatibility helper).

        Returns:
            Dictionary mapping unmatched source names to unmatch reasons
        """
        return {m.source_name: m.unmatch_reason or "unknown" for m in self.unmatched_params}

    def filter_by_score(self, min_score: float) -> "MappingResult":
        """Filter matches by minimum score threshold.

        Args:
            min_score: Minimum score to include

        Returns:
            New MappingResult with only matches above threshold
        """
        filtered_matched = [m for m in self.matched_params if m.final_score >= min_score]
        below_threshold = [m for m in self.matched_params if m.final_score < min_score]

        # Convert below-threshold matches to unmatched
        unmatched_below = [
            ParameterMatchResult(
                source_name=m.source_name,
                target_name=None,
                score_breakdown=m.score_breakdown,
                final_score=m.final_score,
                matched=False,
                unmatch_reason=f"score_below_threshold_{min_score}",
                match_type=m.match_type,
                transformation=m.transformation,
                source_module_path=m.source_module_path,
                target_module_path=None,
            )
            for m in below_threshold
        ]

        new_unmatched = self.unmatched_params + unmatched_below
        new_parameter_matches = {
            **{m.source_name: m for m in filtered_matched},
            **{m.source_name: m for m in new_unmatched},
        }

        new_coverage = len(filtered_matched) / len(self.parameter_matches) if self.parameter_matches else 0.0

        return MappingResult(
            parameter_matches=new_parameter_matches,
            group_matches=self.group_matches,
            matched_params=filtered_matched,
            unmatched_params=new_unmatched,
            unmatched_targets=self.unmatched_targets,
            coverage=new_coverage,
            threshold=min_score,
            weights=self.weights,
        )

    def get_low_confidence_matches(self, threshold: float = 0.7) -> list[ParameterMatchResult]:
        """Get matches with scores below a confidence threshold.

        Useful for identifying matches that may need manual review.

        Args:
            threshold: Confidence threshold (default 0.7)

        Returns:
            List of matched parameters with scores below threshold
        """
        return [m for m in self.matched_params if m.final_score < threshold]

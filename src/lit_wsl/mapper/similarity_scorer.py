from typing import Any, NamedTuple

from lit_wsl.mapper.parameter_group import ParameterGroup
from lit_wsl.mapper.parameter_info import ParameterInfo
from lit_wsl.mapper.result_types import ScoreBreakdown, TransformationInfo


class NameScoreComponents(NamedTuple):
    """Components of name similarity score."""

    token_score: float
    edit_score: float
    lcs_score: float
    combined: float


class HierarchyScoreComponents(NamedTuple):
    """Components of hierarchy similarity score."""

    depth_score: float
    path_score: float
    order_score: float
    combined: float


class SimilarityScorer:
    """Computes similarity scores between parameters for weight mapping.

    This class implements multiple scoring strategies:
    - Shape similarity (exact match, transpose, reshape with tolerance)
    - Name similarity (edit distance, token overlap, common substrings)
    - Hierarchical position similarity (depth, module path, execution order)
    - Composite scoring with customizable weights
    """

    def __init__(self, shape_tolerance: float = 0.0, buffer_matching_mode: str = "lenient") -> None:
        """Initialize the SimilarityScorer.

        Args:
            shape_tolerance: Relative tolerance for shape matching (0.0 = exact match only)
            buffer_matching_mode: How to handle buffer mismatches
                - 'strict': Buffer shape mismatches cause complete failure
                - 'lenient': Buffer shape mismatches get soft penalty (DEFAULT)
                - 'exclude': Buffers are not scored at all
        """
        self.shape_tolerance = shape_tolerance
        self.buffer_matching_mode = buffer_matching_mode

    def compute_shape_score_with_transform(
        self, source_info: ParameterInfo, target_info: ParameterInfo
    ) -> tuple[float, TransformationInfo | None]:
        """Compute shape similarity score and determine if transformation is needed.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info

        Returns:
            Tuple of (score, transformation_info) where transformation_info is None
            if no transformation needed, or a TransformationInfo dataclass
        """
        if source_info.shape == target_info.shape:
            return 1.0, None

        # Check if shapes are transposed (e.g., for different Conv implementations)
        if len(source_info.shape) == len(target_info.shape) and sorted(source_info.shape) == sorted(target_info.shape):
            # Find the permutation needed
            # This is a simple case - for full implementation, would need more sophisticated matching
            transform_info = TransformationInfo(
                type="transpose",
                note="Shapes are permutations of each other",
                source_shape=source_info.shape,
                target_shape=target_info.shape,
            )
            return 0.7, transform_info

        # Check relative size similarity if tolerance is set
        if self.shape_tolerance > 0:
            size_ratio = min(source_info.numel, target_info.numel) / max(source_info.numel, target_info.numel)
            if size_ratio >= (1.0 - self.shape_tolerance):
                transform_info = TransformationInfo(
                    type="reshape",
                    note="Shapes have similar total elements",
                    source_shape=source_info.shape,
                    target_shape=target_info.shape,
                )
                return 0.5 * size_ratio, transform_info

        return 0.0, None

    def compute_shape_score(self, source_info: ParameterInfo, target_info: ParameterInfo) -> float:
        """Compute shape similarity score.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info

        Returns:
            Score between 0.0 and 1.0
        """
        score, _ = self.compute_shape_score_with_transform(source_info, target_info)
        return score

    def _compute_name_similarity(self, source_info: ParameterInfo, target_info: ParameterInfo) -> NameScoreComponents:
        """Compute name similarity score using multiple metrics.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info

        Returns:
            NameScoreComponents with individual and combined scores
        """
        # CRITICAL: Parameter types must match (weight->weight, bias->bias, etc.)
        # This prevents weight being mapped to bias and vice versa
        if source_info.param_name != target_info.param_name:
            return NameScoreComponents(0.0, 0.0, 0.0, 0.0)

        # Exact match
        if source_info.name == target_info.name:
            return NameScoreComponents(1.0, 1.0, 1.0, 1.0)

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
        combined = 0.4 * token_score + 0.3 * edit_score + 0.3 * lcs_score
        return NameScoreComponents(token_score, edit_score, lcs_score, combined)

    def compute_hierarchy_similarity(
        self,
        source_info: ParameterInfo,
        target_info: ParameterInfo,
        group_mapping: dict[str, str] | None = None,
        hierarchy_analyzer: Any = None,
    ) -> HierarchyScoreComponents:
        """Compute hierarchical position similarity.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info
            group_mapping: Existing group mappings for ordering validation
            hierarchy_analyzer: HierarchyAnalyzer instance for validation checks

        Returns:
            HierarchyScoreComponents with individual and combined scores
        """
        # Depth similarity
        max_depth = max(source_info.depth, target_info.depth)
        depth_score = 1.0 - abs(source_info.depth - target_info.depth) / max_depth if max_depth > 0 else 1.0

        # Module path similarity
        if source_info.module_path and target_info.module_path:
            source_parts = source_info.parts[:-1]
            target_parts = target_info.parts[:-1]

            # Check numeric index compatibility
            # This ensures sequence ordering is preserved (stages.0 before stages.1)
            if hierarchy_analyzer is not None and not hierarchy_analyzer.check_numeric_index_compatibility(
                source_info.module_path, target_info.module_path, group_mapping
            ):
                # Numeric ordering violation - return very low scores
                return HierarchyScoreComponents(0.0, 0.0, 0.0, 0.0)

            # Top-level module matching with semantic equivalence
            # This is now a SOFT constraint - affects scoring but doesn't hard reject
            semantic_penalty = 1.0
            if len(source_parts) > 0 and len(target_parts) > 0:
                # Check first level (e.g., 'backbone', 'neck', 'yolo_head' vs 'head')
                if hierarchy_analyzer is not None and not hierarchy_analyzer.are_modules_semantically_equivalent(
                    source_parts[0], target_parts[0]
                ):
                    # Cross-component mapping - apply penalty but don't hard reject
                    semantic_penalty = 0.5

                # Check second level for important structural components
                if len(source_parts) > 1 and len(target_parts) > 1:
                    if source_parts[1] != target_parts[1]:
                        # Penalty for second-level mismatch
                        path_score = 0.1
                    else:
                        # Count matching parts from the beginning
                        matching_levels = 0
                        for s, t in zip(source_parts, target_parts, strict=False):
                            if s == t:
                                matching_levels += 1
                            else:
                                break

                        max_levels = max(len(source_parts), len(target_parts))
                        # Give high score for deep matches
                        path_score = (matching_levels / max_levels) ** 0.5 if max_levels > 0 else 0.0
                else:
                    # Only one level deep
                    matching_levels = 1
                    max_levels = max(len(source_parts), len(target_parts))
                    path_score = matching_levels / max_levels if max_levels > 0 else 0.0
            else:
                # Count matching parts from the beginning
                matching_levels = 0
                for s, t in zip(source_parts, target_parts, strict=False):
                    if s == t:
                        matching_levels += 1
                    else:
                        break

                max_levels = max(len(source_parts), len(target_parts))
                path_score = matching_levels / max_levels if max_levels > 0 else 0.0

            # Apply semantic penalty to path score
            path_score *= semantic_penalty
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

        # Weighted combination with increased weight on execution order
        combined = 0.2 * depth_score + 0.4 * path_score + 0.4 * order_score
        return HierarchyScoreComponents(depth_score, path_score, order_score, combined)

    def compute_composite_score(
        self,
        source_info: ParameterInfo,
        target_info: ParameterInfo,
        weights: dict[str, float] | None = None,
        group_mapping: dict[str, str] | None = None,
        hierarchy_analyzer: Any = None,
    ) -> ScoreBreakdown:
        """Compute composite similarity score with full breakdown.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info
            weights: Custom weights for scoring components
                     Default: {'shape': 0.4, 'name': 0.1, 'hierarchy': 0.5}
            group_mapping: Existing group mappings for ordering validation
            hierarchy_analyzer: HierarchyAnalyzer instance for validation checks

        Returns:
            ScoreBreakdown with all component scores and composite score
        """
        if weights is None:
            weights = {"shape": 0.4, "name": 0.1, "hierarchy": 0.5}

        shape_score = self.compute_shape_score(source_info, target_info)

        # Handle shape mismatch based on buffer mode
        if shape_score == 0.0:
            # Check if this is a buffer-only mismatch that can be lenient
            if self.buffer_matching_mode == "lenient":
                # Give statistical buffers a soft penalty instead of hard reject
                if source_info.is_statistical_buffer and target_info.is_statistical_buffer:
                    # Buffers with mismatched shapes get low but non-zero score
                    # This allows name/hierarchy to still contribute
                    shape_score = 0.2
                else:
                    # Non-buffer shape mismatch is still a hard fail
                    return ScoreBreakdown(
                        shape_score=0.0,
                        name_score=0.0,
                        hierarchy_score=0.0,
                        composite_score=0.0,
                        weights_used=weights,
                        token_score=0.0,
                        edit_score=0.0,
                        lcs_score=0.0,
                        depth_score=0.0,
                        path_score=0.0,
                        order_score=0.0,
                    )
            elif self.buffer_matching_mode == "exclude":
                # If excluding buffers, skip buffer-to-buffer comparisons entirely
                source_is_buf = getattr(source_info, "is_buffer", False)
                target_is_buf = getattr(target_info, "is_buffer", False)
                if source_is_buf or target_is_buf:
                    return ScoreBreakdown(
                        shape_score=0.0,
                        name_score=0.0,
                        hierarchy_score=0.0,
                        composite_score=0.0,
                        weights_used=weights,
                        token_score=0.0,
                        edit_score=0.0,
                        lcs_score=0.0,
                        depth_score=0.0,
                        path_score=0.0,
                        order_score=0.0,
                    )
            else:  # strict mode
                # Strict mode: hard fail on any shape mismatch
                return ScoreBreakdown(
                    shape_score=0.0,
                    name_score=0.0,
                    hierarchy_score=0.0,
                    composite_score=0.0,
                    weights_used=weights,
                    token_score=0.0,
                    edit_score=0.0,
                    lcs_score=0.0,
                    depth_score=0.0,
                    path_score=0.0,
                    order_score=0.0,
                )

        # Continue with normal scoring if shape_score > 0
        name_components = self._compute_name_similarity(source_info, target_info)
        hierarchy_components = self.compute_hierarchy_similarity(
            source_info, target_info, group_mapping, hierarchy_analyzer
        )

        composite_score = (
            weights["shape"] * shape_score
            + weights["name"] * name_components.combined
            + weights["hierarchy"] * hierarchy_components.combined
        )

        return ScoreBreakdown(
            shape_score=shape_score,
            name_score=name_components.combined,
            hierarchy_score=hierarchy_components.combined,
            composite_score=composite_score,
            weights_used=weights,
            token_score=name_components.token_score,
            edit_score=name_components.edit_score,
            lcs_score=name_components.lcs_score,
            depth_score=hierarchy_components.depth_score,
            path_score=hierarchy_components.path_score,
            order_score=hierarchy_components.order_score,
        )

    def compute_group_similarity(
        self,
        source_group: ParameterGroup,
        target_group: ParameterGroup,
        weights: dict[str, float] | None = None,
        group_mapping: dict[str, str] | None = None,
        hierarchy_analyzer: Any = None,
    ) -> float:
        """Compute similarity score between two parameter groups.

        Args:
            source_group: Source parameter group
            target_group: Target parameter group
            weights: Custom weights for scoring components
            group_mapping: Existing group mappings for ordering validation
            hierarchy_analyzer: HierarchyAnalyzer instance for validation checks

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
            source_info = source_group.params.get(param_type)
            target_info = target_group.params.get(param_type)
            if source_info is not None and target_info is not None:
                score_breakdown = self.compute_composite_score(
                    source_info, target_info, weights, group_mapping, hierarchy_analyzer
                )
                total_score += score_breakdown.composite_score

        return total_score / num_params if num_params > 0 else 0.0

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return SimilarityScorer._levenshtein_distance(s2, s1)

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

    @staticmethod
    def _longest_common_substring_length(s1: str, s2: str) -> int:
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

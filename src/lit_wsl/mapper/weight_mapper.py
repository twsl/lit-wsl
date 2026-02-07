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
    ValidationMetrics,
)
from lit_wsl.mapper.similarity_scorer import SimilarityScorer
from lit_wsl.models.checkpoint import extract_state_dict
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
        >>> # From checkpoint (most common case) - automatically extracts model state
        >>> from lit_wsl.models.checkpoint import load_checkpoint_as_dict
        >>> target_model = NewModel()
        >>> checkpoint = load_checkpoint_as_dict("old_weights.pth")
        >>> mapper = WeightMapper.from_state_dict(source_state_dict=checkpoint, target_module=target_model)
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
        source_params: dict[str, ParameterInfo] | dict[str, Any] | None = None,
        target_params: dict[str, ParameterInfo] | dict[str, Any] | None = None,
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
            source_params: Pre-extracted source parameters or raw checkpoint/state dict.
                          Can be dict[str, ParameterInfo] (pre-extracted), dict[str, Tensor] (state dict),
                          or a checkpoint dict with keys like 'model', 'state_dict', etc.
                          The model state will be extracted automatically if needed.
            target_params: Pre-extracted target parameters or raw checkpoint/state dict.
                          Same flexibility as source_params.
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

        self.extractor = ParameterExtractor()
        self.scorer = SimilarityScorer(shape_tolerance=shape_tolerance, buffer_matching_mode=buffer_matching_mode)
        self.hierarchy_analyzer = HierarchyAnalyzer(incompatible_pairs=incompatible_pairs)
        if source_params is not None:
            self.source_params = self._prepare_params(source_params)
        elif source_module is not None:
            self.source_params = self.extractor.extract_from_module(source_module, dummy_input=dummy_input)
        else:
            raise ValueError("Either source_module or source_params must be provided")

        if target_params is not None:
            self.target_params = self._prepare_params(target_params)
        elif target_module is not None:
            self.target_params = self.extractor.extract_from_module(target_module, dummy_input=dummy_input)
        else:
            raise ValueError("Either target_module or target_params must be provided")

        self.target_by_shape = self.extractor.build_shape_index(self.target_params)

        self.source_groups = self.extractor.extract_parameter_groups(self.source_params)
        self.target_groups = self.extractor.extract_parameter_groups(self.target_params)

        self.source_hierarchy = self.hierarchy_analyzer.build_hierarchy(self.source_groups)
        self.target_hierarchy = self.hierarchy_analyzer.build_hierarchy(self.target_groups)

        self.target_groups_by_types = self.extractor.build_group_index(self.target_groups)

        self._result: MappingResult | None = None

    def _prepare_params(self, params: dict[str, ParameterInfo] | dict[str, Any]) -> dict[str, ParameterInfo]:
        """Prepare parameters by extracting from checkpoint/state dict if needed.

        Args:
            params: Either pre-extracted dict[str, ParameterInfo], a state dict,
                   or a checkpoint dict with keys like 'model', 'state_dict', etc.

        Returns:
            dict[str, ParameterInfo]: Extracted parameter information
        """
        if params and isinstance(next(iter(params.values())), ParameterInfo):
            return params

        try:
            actual_state_dict = extract_state_dict(params)
        except ValueError:
            actual_state_dict = params

        state_dict = self.extractor.extract_from_state_dict(actual_state_dict)
        return state_dict

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
            source_state_dict: State dictionary from the source model (e.g., loaded checkpoint).
                              Can be a raw checkpoint dict with keys like 'model', 'state_dict',
                              'optimizer', etc. The actual model parameters will be extracted automatically.
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
            >>> # Load old weights (can contain 'model', 'optimizer', etc.)
            >>> old_weights = torch.load("old_model.pth")
            >>>
            >>> # Create mapper - automatically extracts model state dict
            >>> new_model = NewModel()
            >>> mapper = WeightMapper.from_state_dict(old_weights, new_model)
            >>> mapping_with_scores, unmatched = mapper.suggest_mapping(return_scores=True)
            >>>
            >>> # Filter by confidence threshold
            >>> high_confidence = {src: tgt for src, (tgt, score) in mapping_with_scores.items() if score > 0.7}
        """
        # Just pass the raw source_state_dict to __init__, _prepare_params will handle extraction
        return cls(
            source_module=None,
            target_module=target_module,
            shape_tolerance=shape_tolerance,
            source_params=source_state_dict,
            target_params=None,
            dummy_input=dummy_input,
            incompatible_pairs=incompatible_pairs,
        )

    def _infer_input_shape_from_weights(self, module: nn.Module) -> torch.Tensor | None:
        """Infer input shape from the first layer's weights.

        Args:
            module: Module to analyze

        Returns:
            Dummy tensor with inferred shape, or None if inference fails
        """
        try:
            # Find the first genuine input layer (skip containers)
            def find_first_layer(mod: nn.Module, prefix: str = "") -> tuple[str, nn.Module] | None:
                for name, child in mod.named_children():
                    full_name = f"{prefix}.{name}" if prefix else name

                    # Skip container modules
                    if isinstance(child, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                        result = find_first_layer(child, full_name)
                        if result:
                            return result
                    # Check if this is an input layer
                    elif hasattr(child, "weight") and child.weight is not None:
                        return full_name, child
                    # Recursively check children
                    else:
                        result = find_first_layer(child, full_name)
                        if result:
                            return result
                return None

            first_layer_info = find_first_layer(module)
            if not first_layer_info:
                return None

            layer_name, first_layer = first_layer_info
            weight_shape = first_layer.weight.shape

            # Infer based on layer type
            if isinstance(first_layer, nn.Conv2d):
                # Conv2d weight shape: (out_channels, in_channels, kernel_h, kernel_w)
                in_channels = weight_shape[1]
                spatial_size = 224  # Default image size
                inferred_shape = (1, in_channels, spatial_size, spatial_size)
                self.logger.info(
                    f"Inferred input shape {inferred_shape} from Conv2d layer '{layer_name}' "
                    f"with weight shape {weight_shape}"
                )
                return torch.randn(inferred_shape)

            elif isinstance(first_layer, nn.Conv1d):
                # Conv1d weight shape: (out_channels, in_channels, kernel_size)
                in_channels = weight_shape[1]
                sequence_length = 32  # Default sequence length
                inferred_shape = (1, in_channels, sequence_length)
                self.logger.info(
                    f"Inferred input shape {inferred_shape} from Conv1d layer '{layer_name}' "
                    f"with weight shape {weight_shape}"
                )
                return torch.randn(inferred_shape)

            elif isinstance(first_layer, nn.Conv3d):
                # Conv3d weight shape: (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
                in_channels = weight_shape[1]
                spatial_size = 16  # Default 3D spatial size
                inferred_shape = (1, in_channels, spatial_size, spatial_size, spatial_size)
                self.logger.info(
                    f"Inferred input shape {inferred_shape} from Conv3d layer '{layer_name}' "
                    f"with weight shape {weight_shape}"
                )
                return torch.randn(inferred_shape)

            elif isinstance(first_layer, nn.Linear):
                # Linear weight shape: (out_features, in_features)
                in_features = weight_shape[1]
                inferred_shape = (1, in_features)
                self.logger.info(
                    f"Inferred input shape {inferred_shape} from Linear layer '{layer_name}' "
                    f"with weight shape {weight_shape}"
                )
                return torch.randn(inferred_shape)

            elif isinstance(first_layer, nn.Embedding):
                # Embedding layer - use default sequence length
                sequence_length = 32
                inferred_shape = (1, sequence_length)
                self.logger.info(f"Inferred input shape {inferred_shape} from Embedding layer '{layer_name}'")
                return torch.randint(0, weight_shape[0], inferred_shape)

            else:
                self.logger.warning(f"Unable to infer input shape from layer type {type(first_layer).__name__}")
                return None

        except Exception as e:
            self.logger.warning(f"Failed to infer input shape: {e}")
            return None

    def _validate_outputs(
        self,
        source_module: nn.Module,
        target_module: nn.Module,
        dummy_input: torch.Tensor,
        mapping: dict[str, str],
    ) -> ValidationMetrics:
        """Validate that mapped weights produce similar outputs.

        Args:
            source_module: Source model
            target_module: Target model
            dummy_input: Input tensor for forward pass
            mapping: Suggested parameter mapping

        Returns:
            ValidationMetrics with comparison results
        """
        try:
            source_module.eval()
            target_module.eval()

            with torch.no_grad():
                source_output = source_module(dummy_input)

            import copy

            target_copy = copy.deepcopy(target_module)
            source_state = source_module.state_dict()
            target_state = target_copy.state_dict()

            for source_name, target_name in mapping.items():
                if source_name in source_state and target_name in target_state:
                    target_state[target_name] = source_state[source_name]

            target_copy.load_state_dict(target_state, strict=False)

            with torch.no_grad():
                target_output = target_copy(dummy_input)

            if isinstance(source_output, tuple):
                source_output = source_output[0]
            if isinstance(target_output, tuple):
                target_output = target_output[0]

            source_flat = source_output.flatten()
            target_flat = target_output.flatten()

            cosine_sim = float(
                torch.nn.functional.cosine_similarity(source_flat.unsqueeze(0), target_flat.unsqueeze(0)).item()
            )
            mse = float(torch.nn.functional.mse_loss(source_flat, target_flat).item())
            max_diff = float(torch.max(torch.abs(source_flat - target_flat)).item())

            validation_passed = cosine_sim >= 0.95

            return ValidationMetrics(
                output_cosine_sim=cosine_sim,
                output_mse=mse,
                output_max_diff=max_diff,
                validation_passed=validation_passed,
                error_message=None,
            )

        except Exception as e:
            self.logger.warning(f"Output validation failed: {e}")
            return ValidationMetrics(
                output_cosine_sim=None,
                output_mse=None,
                output_max_diff=None,
                validation_passed=False,
                error_message=str(e),
            )

    def _generate_transformation_code(
        self, source_info: ParameterInfo, target_info: ParameterInfo, transformation: TransformationInfo | None
    ) -> str | None:
        """Generate executable Python code for tensor transformation.

        Args:
            source_info: Source parameter info
            target_info: Target parameter info
            transformation: Transformation info (if any)

        Returns:
            Python code string or None if no transformation needed
        """
        if transformation is None:
            return None

        source_shape = source_info.shape
        target_shape = target_info.shape

        if transformation.type == "transpose":
            if len(source_shape) == 2:
                return "target_weight = source_weight.transpose(0, 1)"
            elif len(source_shape) == 4:
                if source_shape[0] == target_shape[0] and source_shape[1] == target_shape[1]:
                    # Spatial dimensions transposed
                    return "target_weight = source_weight.transpose(2, 3)"
                elif source_shape[0] == target_shape[1] and source_shape[1] == target_shape[0]:
                    # Channel dimensions transposed
                    return "target_weight = source_weight.transpose(0, 1)"
                else:
                    # Complex permutation - generate generic code
                    return f"# Manual permutation needed: {source_shape} -> {target_shape}"
            else:
                return f"# Transpose needed: {source_shape} -> {target_shape}"

        elif transformation.type == "reshape":
            # Generate reshape code
            target_shape_str = ", ".join(str(dim) for dim in target_shape)
            return f"target_weight = source_weight.reshape({target_shape_str})"

        return None

    def _compute_confidence_and_alternatives(
        self,
        parameter_results: list[ParameterMatchResult],
        weights: dict[str, float],
    ) -> list[ParameterMatchResult]:
        """Compute confidence scores and alternative matches for each parameter.

        Confidence is based on the score gap between the best match and second-best match.
        A large gap indicates high confidence, while a small gap suggests ambiguity.

        Args:
            parameter_results: List of parameter matches
            weights: Scoring weights used

        Returns:
            Updated list of parameter matches with confidence and alternatives
        """
        updated_results = []

        for param_result in parameter_results:
            if not param_result.matched:
                # Unmatched parameters have no alternatives
                updated_results.append(param_result)
                continue

            source_name = param_result.source_name
            source_info = self.source_params[source_name]
            matched_target = param_result.target_name

            # Find all compatible targets and score them
            scored_candidates = []
            for target_name, target_info in self.target_params.items():
                # Skip if shapes don't match (required for compatibility)
                if source_info.shape != target_info.shape:
                    # Check for transposed shapes
                    if len(source_info.shape) == 2 and source_info.shape[::-1] == target_info.shape:
                        pass  # Allow transposed
                    else:
                        continue

                # Skip if param types don't match
                if source_info.param_name != target_info.param_name:
                    continue

                # Compute score
                score_breakdown = self.scorer.compute_composite_score(
                    source_info, target_info, weights, None, self.hierarchy_analyzer
                )

                scored_candidates.append((target_name, score_breakdown.composite_score))

            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Compute confidence
            if len(scored_candidates) >= 2:
                best_score = scored_candidates[0][1]
                second_best_score = scored_candidates[1][1]
                score_gap = best_score - second_best_score

                # Normalize confidence: gap of 0.3 or more = full confidence
                confidence = min(1.0, score_gap / 0.3)
            else:
                # Only one candidate - high confidence
                confidence = 1.0

            # Get top 3-5 alternatives (excluding the matched one)
            alternatives = [(target, score) for target, score in scored_candidates if target != matched_target][:5]

            # Generate transformation code if needed
            transformation_code = None
            if matched_target is not None:
                transformation_code = self._generate_transformation_code(
                    source_info, self.target_params[matched_target], param_result.transformation
                )

            # Create updated result with confidence and alternatives
            updated_result = ParameterMatchResult(
                source_name=param_result.source_name,
                target_name=param_result.target_name,
                score_breakdown=param_result.score_breakdown,
                final_score=param_result.final_score,
                matched=param_result.matched,
                unmatch_reason=param_result.unmatch_reason,
                match_type=param_result.match_type,
                transformation=param_result.transformation,
                source_module_path=param_result.source_module_path,
                target_module_path=param_result.target_module_path,
                confidence_score=confidence,
                alternative_matches=alternatives,
                transformation_code=transformation_code,
            )

            updated_results.append(updated_result)

        return updated_results

    def suggest_mapping(
        self,
        threshold: float | None = None,
        weights: dict[str, float] | None = None,
        validate_outputs: bool = True,
        strategy: str = "greedy",
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
            validate_outputs: Whether to validate mapped weights by comparing outputs (default: True).
                            Requires both source and target modules. If dummy_input is not provided,
                            will attempt to infer input shape from source model weights.
            strategy: Matching strategy to use. Options:
                     - 'greedy': Fast greedy selection (current default for backward compatibility)
                     - 'optimal': Global optimization using Hungarian algorithm (recommended)

        Returns:
            MappingResult with complete information about all matches, scores,
            transparency data, and optional output validation metrics

        Example:
            >>> mapper = WeightMapper(source_model, target_model)
            >>> result = mapper.suggest_mapping(threshold=0.6)
            >>> # Access simple mapping
            >>> mapping_dict = result.get_mapping()
            >>> # Check validation results
            >>> if result.output_validation:
            ...     print(f"Validation passed: {result.output_validation.validation_passed}")
            ...     print(f"Cosine similarity: {result.output_validation.output_cosine_sim:.4f}")
            >>> # Inspect low confidence matches
            >>> low_conf = result.get_low_confidence_matches(0.7)
            >>> for match in low_conf:
            ...     print(f"{match.source_name} -> {match.target_name}")
            ...     print(
            ...         f"  Score breakdown: shape={match.score_breakdown.shape_score:.2f}, "
            ...         f"name={match.score_breakdown.name_score:.2f}"
            ...     )
        """
        # Set default threshold
        if threshold is None:
            threshold = 0.0

        if weights is None:
            weights = {"shape": 0.4, "name": 0.1, "hierarchy": 0.5}

        # Try to infer dummy_input if not provided and validation is requested
        dummy_input_for_validation = self.dummy_input
        if validate_outputs and self.source_module and self.target_module and dummy_input_for_validation is None:
            self.logger.info("Attempting to infer input shape from source model weights...")
            dummy_input_for_validation = self._infer_input_shape_from_weights(self.source_module)
            if dummy_input_for_validation is None:
                self.logger.warning(
                    "Could not infer input shape. Output validation will be skipped. "
                    "Provide dummy_input explicitly for validation."
                )
                validate_outputs = False

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
        group_results = mapping_strategy.suggest_group_mapping(threshold, weights, strategy=strategy)

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

        # Phase 3.5: Compute confidence scores and alternatives
        parameter_results = self._compute_confidence_and_alternatives(parameter_results, weights)

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

        # Phase 5: Output validation (if requested and possible)
        validation_metrics = None
        if validate_outputs and self.source_module and self.target_module and dummy_input_for_validation is not None:
            self.logger.info("Validating mapped weights by comparing model outputs...")
            mapping_dict = {r.source_name: r.target_name for r in matched_params if r.target_name}
            validation_metrics = self._validate_outputs(
                self.source_module, self.target_module, dummy_input_for_validation, mapping_dict
            )

            if validation_metrics.validation_passed:
                self.logger.info(
                    f"✓ Output validation PASSED (cosine similarity: {validation_metrics.output_cosine_sim:.4f})"
                )
            else:
                if validation_metrics.error_message:
                    self.logger.warning(f"✗ Output validation FAILED: {validation_metrics.error_message}")
                else:
                    self.logger.warning(
                        f"✗ Output validation FAILED (cosine similarity: {validation_metrics.output_cosine_sim:.4f}, "
                        f"threshold: 0.95). This may indicate incorrect parameter mapping."
                    )

        result = MappingResult(
            parameter_matches=parameter_matches,
            group_matches=group_matches,
            matched_params=matched_params,
            unmatched_params=unmatched_params,
            unmatched_targets=unmatched_targets,
            coverage=coverage,
            threshold=threshold,
            weights=weights,
            output_validation=validation_metrics,
            strategy=strategy,
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

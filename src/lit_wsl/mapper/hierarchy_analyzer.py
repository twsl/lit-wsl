import re
from typing import Any

from lit_wsl.mapper.module_node import ModuleNode
from lit_wsl.mapper.parameter_group import ParameterGroup
from lit_wsl.mapper.result_types import HierarchyMetadata


class HierarchyAnalyzer:
    """Analyzes hierarchical structure of modules for better weight mapping.

    This class handles:
    - Building hierarchical tree structures from parameter groups
    - Extracting and comparing hierarchy metadata
    - Checking numeric index compatibility and ordering
    - Semantic module name equivalence checking
    - Computing hierarchical context scores
    """

    def __init__(self, incompatible_pairs: list[tuple[set[str], set[str]]] | None = None) -> None:
        """Initialize the HierarchyAnalyzer.

        Args:
            incompatible_pairs: Optional list of incompatible component pairs that should never match.
                Each pair is a tuple of two sets of semantic chunks representing different
                architectural components. For example: ({{"backbone"}}, {{"head"}}) prevents
                backbone modules from matching with head modules. If None (default), no
                cross-component restrictions are applied - all matches are allowed based on
                other scoring factors. Provide explicit pairs to prevent specific combinations.
        """
        # By default, allow all matches - users can provide explicit restrictions if needed
        self.incompatible_pairs = incompatible_pairs if incompatible_pairs is not None else []

        # Semantic equivalents: components with similar meanings
        self.semantic_equivalents = {
            "encoder": {"backbone", "feature_extractor", "stem", "features"},
            "backbone": {"encoder", "feature_extractor", "stem", "features"},
            "decoder": {"head", "classifier", "predictor", "output"},
            "head": {"decoder", "classifier", "predictor", "output"},
            "feature_extractor": {"encoder", "backbone", "stem", "features"},
            "classifier": {"head", "decoder", "predictor", "output"},
            "attention": {"attn", "self_attention", "msa"},
            "attn": {"attention", "self_attention", "msa"},
            "feedforward": {"ff", "ffn", "mlp"},
            "ff": {"feedforward", "ffn", "mlp"},
            "ffn": {"feedforward", "ff", "mlp"},
            "mlp": {"feedforward", "ff", "ffn"},
            "norm": {"normalization", "ln", "bn", "layernorm", "batchnorm"},
            "bn": {"batchnorm", "batch_norm", "norm"},
            "ln": {"layernorm", "layer_norm", "norm"},
        }

    def build_hierarchy(self, groups: dict[str, ParameterGroup]) -> ModuleNode:
        """Build a hierarchical tree structure from parameter groups.

        Args:
            groups: Dictionary mapping module paths to ParameterGroup objects

        Returns:
            Root ModuleNode of the hierarchy tree
        """
        root = ModuleNode("", "", 0)
        nodes: dict[str, ModuleNode] = {"": root}

        sorted_paths = sorted(groups.keys(), key=lambda x: (x.count("."), x))

        for module_path in sorted_paths:
            if not module_path:
                continue

            parts = module_path.split(".")

            for i in range(1, len(parts) + 1):
                partial_path = ".".join(parts[:i])
                if partial_path not in nodes:
                    parent_path = ".".join(parts[: i - 1]) if i > 1 else ""
                    parent_node = nodes[parent_path]
                    new_node = ModuleNode(parts[i - 1], partial_path, i)
                    parent_node.add_child(new_node)
                    nodes[partial_path] = new_node

            if module_path in groups:
                nodes[module_path].parameter_group = groups[module_path]

        return root

    def compute_hierarchy_context_score(
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
                parent_match_bonus += 0.3 / i
        depth_match = 1.0 if len(source_parts) == len(target_parts) else 0.5

        # Combine scores
        return min(1.0, 0.5 * depth_match + 0.5 * min(1.0, parent_match_bonus))

    @staticmethod
    def split_module_name_into_chunks(name: str) -> set[str]:
        """Split a module name into semantic chunks.

        Splits by:
        - Underscores: 'yolo_head' -> ['yolo', 'head']
        - CamelCase: 'YoloHead' -> ['yolo', 'head']
        - Numbers: 'stage1' -> ['stage', '1']

        Args:
            name: Module name to split

        Returns:
            Set of lowercase chunks
        """
        parts = name.split("_")

        chunks = []
        for part in parts:
            split_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+", part)
            chunks.extend(split_parts)

        return {chunk.lower() for chunk in chunks if chunk}

    def are_modules_semantically_equivalent(self, module1: str, module2: str) -> bool:
        """Check if two module names are semantically equivalent.

        Uses chunk-based matching to handle common naming variations like:
        - 'yolo_head' <-> 'head' (head chunk matches)
        - 'YoloHead' <-> 'detection_head' (head chunk matches)
        - 'backbone' <-> 'Backbone' (exact match, case-insensitive)
        - 'fpn' <-> 'feature_pyramid_network' (fpn chunk matches)

        Now also considers semantic equivalents (e.g., backbone <-> encoder).

        Args:
            module1: First module name
            module2: Second module name

        Returns:
            True if modules are semantically equivalent, with boosted score for semantic matches
        """
        if module1 == module2:
            return True

        chunks1 = self.split_module_name_into_chunks(module1)
        chunks2 = self.split_module_name_into_chunks(module2)
        for group1, group2 in self.incompatible_pairs:
            has_group1_in_first = bool(chunks1 & group1)
            has_group2_in_first = bool(chunks1 & group2)
            has_group1_in_second = bool(chunks2 & group1)
            has_group2_in_second = bool(chunks2 & group2)

            if (has_group1_in_first and has_group2_in_second) or (has_group2_in_first and has_group1_in_second):
                return False

        return True

    def compute_semantic_similarity_boost(self, chunks1: set[str], chunks2: set[str]) -> float:
        """Compute a similarity boost for semantically equivalent components.

        Args:
            chunks1: Chunks from first module path
            chunks2: Chunks from second module path

        Returns:
            Boost factor (1.0 = no boost, 1.2 = 20% boost)
        """
        for chunk1 in chunks1:
            if chunk1 in self.semantic_equivalents:
                equivalent_chunks = self.semantic_equivalents[chunk1]
                if chunks2 & equivalent_chunks:
                    return 1.2

        return 1.0

    def check_numeric_sequence_consistency(self, mapping_dict: dict[str, str]) -> dict[str, float]:
        """Check if numeric indices in mappings maintain sequential ordering.

        For example, if stages.0 maps to blocks.0 and stages.1 maps to blocks.1,
        the sequence is consistent. But if stages.0 maps to blocks.5 and stages.1
        maps to blocks.3, it violates expected ordering.

        Args:
            mapping_dict: Dictionary of source_path -> target_path mappings

        Returns:
            Dictionary of source_path -> penalty (1.0 = no penalty, 0.3 = strong penalty)
        """
        penalties = {}

        # Group mappings by common prefix
        prefix_groups = {}
        for source_path, target_path in mapping_dict.items():
            # Extract prefix (everything before the last numeric index)
            source_parts = source_path.split(".")
            target_parts = target_path.split(".")

            # Find the common prefix structure
            source_prefix = ".".join(source_parts[:-1]) if len(source_parts) > 1 else ""
            target_prefix = ".".join(target_parts[:-1]) if len(target_parts) > 1 else ""

            key = (source_prefix, target_prefix)
            if key not in prefix_groups:
                prefix_groups[key] = []

            # Extract numeric indices
            source_indices = self.extract_numeric_indices(source_path)
            target_indices = self.extract_numeric_indices(target_path)

            if source_indices and target_indices:
                prefix_groups[key].append((source_path, target_path, source_indices[-1], target_indices[-1]))

        # Check ordering consistency within each group
        for group_key, mappings in prefix_groups.items():
            if len(mappings) < 2:
                continue

            # Sort by source index
            sorted_mappings = sorted(mappings, key=lambda x: x[2] if x[2] is not None else -1)

            # Check if target indices are also in order
            prev_target_idx = None
            ordering_violations = []

            for source_path, target_path, source_idx, target_idx in sorted_mappings:
                if source_idx is None or target_idx is None:
                    continue

                if prev_target_idx is not None:
                    if target_idx < prev_target_idx:
                        # Ordering violation!
                        ordering_violations.append(source_path)
                        penalties[source_path] = 0.3  # Strong penalty

                prev_target_idx = target_idx

        return penalties

    @staticmethod
    def extract_numeric_indices(path: str) -> list[int | None]:
        """Extract numeric indices from a module path.

        Args:
            path: Module path (e.g., 'backbone.stages.0.blocks.1')

        Returns:
            List of numeric indices in order (e.g., [0, 1])
        """
        parts = path.split(".")
        indices = []
        for part in parts:
            # Check if part is numeric or contains numeric index
            if part.isdigit():
                indices.append(int(part))
            else:
                # Check for patterns like 'stage1', 'conv2', etc.
                match = re.search(r"(\d+)$", part)
                if match:
                    indices.append(int(match.group(1)))
                else:
                    indices.append(None)
        return indices

    def extract_hierarchy_metadata(self, path: str) -> HierarchyMetadata:
        """Extract structured hierarchy metadata from a module path.

        Args:
            path: Module path (e.g., 'backbone.stages.0.blocks.1.conv')

        Returns:
            HierarchyMetadata dataclass with structured information
        """
        parts = path.split(".")
        stages = []
        all_chunks = []
        numeric_indices = []

        for part in parts:
            # Extract numeric index if present
            if part.isdigit():
                numeric_indices.append(int(part))
                stages.append({"name": part, "index": int(part), "chunks": set()})
            else:
                # Check for patterns like 'stage1', 'conv2', etc.
                match = re.search(r"(\d+)$", part)
                if match:
                    idx = int(match.group(1))
                    name_part = part[: -len(match.group(1))]
                    numeric_indices.append(idx)
                    chunks = self.split_module_name_into_chunks(name_part)
                    stages.append({"name": name_part, "index": idx, "chunks": chunks})
                    all_chunks.extend(chunks)
                else:
                    chunks = self.split_module_name_into_chunks(part)
                    stages.append({"name": part, "index": None, "chunks": chunks})
                    all_chunks.extend(chunks)

        return HierarchyMetadata(
            depth=len(parts),
            chunks=set(all_chunks),
            numeric_indices=numeric_indices,
            stages=stages,
        )

    def compute_hierarchy_structure_score(
        self, source_metadata: HierarchyMetadata, target_metadata: HierarchyMetadata
    ) -> float:
        """Compute structural similarity score based on hierarchy metadata.

        Args:
            source_metadata: HierarchyMetadata for source module
            target_metadata: HierarchyMetadata for target module

        Returns:
            Score between 0.0 and 1.0
        """
        # Depth similarity
        depth_diff = abs(source_metadata.depth - target_metadata.depth)
        max_depth = max(source_metadata.depth, target_metadata.depth)
        depth_score = 1.0 - (depth_diff / max_depth) if max_depth > 0 else 1.0

        # Chunk overlap (semantic similarity)
        source_chunks = source_metadata.chunks
        target_chunks = target_metadata.chunks
        if source_chunks and target_chunks:
            intersection = len(source_chunks & target_chunks)
            union = len(source_chunks | target_chunks)
            chunk_score = intersection / union if union > 0 else 0.0
        else:
            chunk_score = 0.0

        # Stage-by-stage comparison
        source_stages = source_metadata.stages
        target_stages = target_metadata.stages
        stage_score = 0.0

        min_stages = min(len(source_stages), len(target_stages))
        if min_stages > 0:
            matching_stages = 0
            for src_stage, tgt_stage in zip(source_stages, target_stages, strict=False):
                # Check if stage names match or have common chunks
                if src_stage["name"] == tgt_stage["name"]:
                    matching_stages += 1
                elif src_stage["chunks"] & tgt_stage["chunks"]:
                    matching_stages += 0.5

            stage_score = matching_stages / max(len(source_stages), len(target_stages))

        # Combine scores
        return 0.3 * depth_score + 0.3 * chunk_score + 0.4 * stage_score

    def check_numeric_index_compatibility(
        self, source_path: str, target_path: str, group_mapping: dict[str, str] | None = None
    ) -> bool:
        """Check if numeric indices in paths are compatible.

        Allows different starting indices but enforces ordering constraints:
        - stages.0 can map to stage1 (different starting points)
        - But if stages.2 → stage2, then stages.3 cannot map to stage1
          (violates ordering: 3 > 2 but target 1 < 2)

        Args:
            source_path: Source module path
            target_path: Target module path
            group_mapping: Existing group mappings to check ordering constraints

        Returns:
            True if indices are compatible (preserves relative ordering)
        """
        source_indices = self.extract_numeric_indices(source_path)
        target_indices = self.extract_numeric_indices(target_path)

        # Extract only the non-None indices
        source_nums = [i for i in source_indices if i is not None]
        target_nums = [i for i in target_indices if i is not None]

        # If either has no numeric indices, consider compatible
        if not source_nums or not target_nums:
            return True

        # If no existing mappings, allow any match
        if not group_mapping:
            return True

        # Check ordering constraints with existing mappings
        # Find existing mappings with similar path structure (same non-numeric prefix)
        source_parts = source_path.split(".")
        target_parts = target_path.split(".")

        # Get non-numeric prefix for comparison (e.g., "backbone.stages" from "backbone.stages.0")
        source_prefix_parts = []
        target_prefix_parts = []

        for i, part in enumerate(source_parts):
            if i < len(source_indices) and source_indices[i] is not None:
                break
            source_prefix_parts.append(part)

        for i, part in enumerate(target_parts):
            if i < len(target_indices) and target_indices[i] is not None:
                break
            target_prefix_parts.append(part)

        source_prefix = ".".join(source_prefix_parts)
        target_prefix = ".".join(target_prefix_parts)

        # Check all existing mappings with the same prefix structure
        for existing_source, existing_target in group_mapping.items():
            existing_source_indices = self.extract_numeric_indices(existing_source)
            existing_target_indices = self.extract_numeric_indices(existing_target)

            existing_source_nums = [i for i in existing_source_indices if i is not None]
            existing_target_nums = [i for i in existing_target_indices if i is not None]

            if not existing_source_nums or not existing_target_nums:
                continue

            # Check if this existing mapping shares the same prefix structure
            existing_source_parts = existing_source.split(".")
            existing_source_prefix_parts = []
            for i, part in enumerate(existing_source_parts):
                if i < len(existing_source_indices) and existing_source_indices[i] is not None:
                    break
                existing_source_prefix_parts.append(part)
            existing_source_prefix = ".".join(existing_source_prefix_parts)

            existing_target_parts = existing_target.split(".")
            existing_target_prefix_parts = []
            for i, part in enumerate(existing_target_parts):
                if i < len(existing_target_indices) and existing_target_indices[i] is not None:
                    break
                existing_target_prefix_parts.append(part)
            existing_target_prefix = ".".join(existing_target_prefix_parts)

            # Only compare if prefixes match (comparing elements from same sequence)
            if existing_source_prefix != source_prefix or existing_target_prefix != target_prefix:
                continue

            # Check ordering constraint: if source[i] < source[j], then target[i] <= target[j]
            # Compare first differing index
            for src_idx, existing_src_idx, tgt_idx, existing_tgt_idx in zip(
                source_nums, existing_source_nums, target_nums, existing_target_nums, strict=False
            ):
                if src_idx != existing_src_idx:
                    # Found first differing index - check ordering
                    if src_idx < existing_src_idx and tgt_idx > existing_tgt_idx:
                        return False  # Violates ordering
                    if src_idx > existing_src_idx and tgt_idx < existing_tgt_idx:
                        return False  # Violates ordering
                    break  # Only check first differing index

        return True

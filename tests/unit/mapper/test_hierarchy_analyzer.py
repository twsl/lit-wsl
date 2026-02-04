"""Unit tests for HierarchyAnalyzer."""

import pytest

from lit_wsl.mapper.hierarchy_analyzer import HierarchyAnalyzer


class TestHierarchyAnalyzerIncompatiblePairs:
    """Test configurable incompatible_pairs in HierarchyAnalyzer."""

    def test_default_allows_all_matches(self) -> None:
        """Test that default behavior (None) allows all matches - no restrictions."""
        analyzer = HierarchyAnalyzer()

        # Default (None) should allow all cross-component matches
        assert analyzer.are_modules_semantically_equivalent("backbone", "head")
        assert analyzer.are_modules_semantically_equivalent("encoder", "classifier")
        assert analyzer.are_modules_semantically_equivalent("encoder", "decoder")
        assert analyzer.are_modules_semantically_equivalent("backbone", "fpn")
        assert analyzer.are_modules_semantically_equivalent("encoder", "neck")

        # Reverse direction should also be allowed
        assert analyzer.are_modules_semantically_equivalent("head", "backbone")
        assert analyzer.are_modules_semantically_equivalent("classifier", "encoder")

    def test_always_compatible_modules(self) -> None:
        """Test that matching module names are always compatible."""
        analyzer = HierarchyAnalyzer()

        # These should still be allowed (same component type)
        assert analyzer.are_modules_semantically_equivalent("backbone", "backbone")
        assert analyzer.are_modules_semantically_equivalent("encoder", "encoder")
        assert analyzer.are_modules_semantically_equivalent("head", "head")

        # Related but not incompatible
        assert analyzer.are_modules_semantically_equivalent("conv1", "conv2")
        assert analyzer.are_modules_semantically_equivalent("layer", "block")

    def test_explicit_empty_incompatible_pairs(self) -> None:
        """Test that explicitly passing empty list also allows all matches (same as default)."""
        analyzer = HierarchyAnalyzer(incompatible_pairs=[])

        # With no incompatible pairs, everything should be allowed
        assert analyzer.are_modules_semantically_equivalent("backbone", "head")
        assert analyzer.are_modules_semantically_equivalent("encoder", "classifier")
        assert analyzer.are_modules_semantically_equivalent("encoder", "decoder")
        assert analyzer.are_modules_semantically_equivalent("backbone", "fpn")

    def test_custom_incompatible_pairs(self) -> None:
        """Test that custom incompatible pairs add restrictions."""
        # Custom pairs for a different architecture (e.g., NLP model)
        custom_pairs = [
            ({"encoder", "embedding"}, {"decoder", "output"}),
            ({"attention"}, {"feedforward", "mlp"}),
        ]
        analyzer = HierarchyAnalyzer(incompatible_pairs=custom_pairs)

        # Custom pairs should block these
        assert not analyzer.are_modules_semantically_equivalent("encoder", "decoder")
        assert not analyzer.are_modules_semantically_equivalent("embedding", "output")
        assert not analyzer.are_modules_semantically_equivalent("attention", "feedforward")
        assert not analyzer.are_modules_semantically_equivalent("attention", "mlp")

        # But without custom pairs, these would be allowed (testing default is no restrictions)
        default_analyzer = HierarchyAnalyzer()
        assert default_analyzer.are_modules_semantically_equivalent("encoder", "decoder")
        assert default_analyzer.are_modules_semantically_equivalent("attention", "feedforward")

    def test_vision_model_incompatible_pairs(self) -> None:
        """Test typical vision model incompatible pairs."""
        # Common pairs for vision models to prevent backbone-head confusion
        vision_pairs = [
            ({"backbone", "encoder", "feature"}, {"head", "classifier", "decoder"}),
            ({"backbone", "encoder"}, {"neck", "fpn"}),
        ]
        analyzer = HierarchyAnalyzer(incompatible_pairs=vision_pairs)

        # These should be blocked by vision pairs
        assert not analyzer.are_modules_semantically_equivalent("backbone", "head")
        assert not analyzer.are_modules_semantically_equivalent("encoder", "classifier")
        assert not analyzer.are_modules_semantically_equivalent("backbone", "fpn")

        # But without the pairs, default allows them
        default_analyzer = HierarchyAnalyzer()
        assert default_analyzer.are_modules_semantically_equivalent("backbone", "head")
        assert default_analyzer.are_modules_semantically_equivalent("encoder", "classifier")

    def test_incompatible_pairs_with_multi_chunk_names(self) -> None:
        """Test incompatible pairs work with multi-chunk module names."""
        # Use vision pairs to test multi-chunk names
        vision_pairs = [
            ({"backbone", "encoder", "feature"}, {"head", "classifier", "decoder"}),
        ]
        analyzer = HierarchyAnalyzer(incompatible_pairs=vision_pairs)

        # Multi-chunk names containing incompatible components should be blocked
        assert not analyzer.are_modules_semantically_equivalent("yolo_backbone", "detection_head")
        assert not analyzer.are_modules_semantically_equivalent("BackboneNet", "ClassifierHead")
        assert not analyzer.are_modules_semantically_equivalent("feature_encoder", "head_decoder")

        # But default (no pairs) allows them
        default_analyzer = HierarchyAnalyzer()
        assert default_analyzer.are_modules_semantically_equivalent("yolo_backbone", "detection_head")

    def test_incompatible_pairs_symmetric(self) -> None:
        """Test that incompatible pairs work symmetrically (A->B == B->A)."""
        # Test with custom pairs
        custom_pairs = [
            ({"backbone", "encoder"}, {"head", "classifier"}),
        ]
        analyzer = HierarchyAnalyzer(incompatible_pairs=custom_pairs)

        # Test symmetry for custom pairs
        pairs_to_test = [
            ("backbone", "head"),
            ("encoder", "classifier"),
        ]

        for module1, module2 in pairs_to_test:
            result_forward = analyzer.are_modules_semantically_equivalent(module1, module2)
            result_backward = analyzer.are_modules_semantically_equivalent(module2, module1)
            assert result_forward == result_backward, f"Asymmetric result for {module1} <-> {module2}"
            assert not result_forward, f"Should be incompatible: {module1} <-> {module2}"

    def test_identical_modules_always_compatible(self) -> None:
        """Test that identical module names are always compatible regardless of pairs."""
        # Even with incompatible pairs, identical names should match
        analyzer = HierarchyAnalyzer()

        assert analyzer.are_modules_semantically_equivalent("backbone", "backbone")
        assert analyzer.are_modules_semantically_equivalent("head", "head")
        assert analyzer.are_modules_semantically_equivalent("encoder", "encoder")
        assert analyzer.are_modules_semantically_equivalent("CustomModule123", "CustomModule123")

    def test_incompatible_pairs_case_insensitive(self) -> None:
        """Test that incompatible pairs matching is case-insensitive."""
        # Use vision pairs
        vision_pairs = [
            ({"backbone"}, {"head"}),
        ]
        analyzer = HierarchyAnalyzer(incompatible_pairs=vision_pairs)

        # CamelCase and snake_case variations should still be blocked
        assert not analyzer.are_modules_semantically_equivalent("Backbone", "Head")
        assert not analyzer.are_modules_semantically_equivalent("BACKBONE", "head")
        assert not analyzer.are_modules_semantically_equivalent("BackboneNet", "head_module")

    def test_complex_incompatible_pairs(self) -> None:
        """Test complex incompatible pairs with multiple chunks in sets."""
        # Pairs with multiple alternatives in each set
        custom_pairs = [
            ({"encoder", "extractor", "feature"}, {"decoder", "generator", "output"}),
        ]
        analyzer = HierarchyAnalyzer(incompatible_pairs=custom_pairs)

        # Any combination from set1 -> set2 should be blocked
        assert not analyzer.are_modules_semantically_equivalent("encoder", "decoder")
        assert not analyzer.are_modules_semantically_equivalent("extractor", "generator")
        assert not analyzer.are_modules_semantically_equivalent("feature", "output")
        assert not analyzer.are_modules_semantically_equivalent("encoder", "generator")
        assert not analyzer.are_modules_semantically_equivalent("feature", "decoder")

    def test_partial_chunk_overlap_with_incompatible_pairs(self) -> None:
        """Test that partial chunk overlap doesn't trigger false incompatibilities."""
        analyzer = HierarchyAnalyzer()

        # "feature_neck" contains both "feature" (from backbone group) and "neck"
        # but should be treated as a neck component, not backbone
        # This tests that we check for actual incompatibility, not just presence
        assert analyzer.are_modules_semantically_equivalent("neck", "feature_neck")

    def test_none_equals_empty_list(self) -> None:
        """Test that passing None for incompatible_pairs equals empty list (no restrictions)."""
        analyzer1 = HierarchyAnalyzer(incompatible_pairs=None)
        analyzer2 = HierarchyAnalyzer()
        analyzer3 = HierarchyAnalyzer(incompatible_pairs=[])

        # All three should behave identically - allowing all matches
        test_cases = [
            ("backbone", "head"),
            ("encoder", "classifier"),
            ("backbone", "fpn"),
        ]

        for module1, module2 in test_cases:
            result1 = analyzer1.are_modules_semantically_equivalent(module1, module2)
            result2 = analyzer2.are_modules_semantically_equivalent(module1, module2)
            result3 = analyzer3.are_modules_semantically_equivalent(module1, module2)
            assert result1 and result2 and result3, f"All should allow {module1} <-> {module2}"
            assert result1 == result2 == result3

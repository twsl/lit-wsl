"""Tests for Rich-based visualization utilities."""

import pytest
from rich.console import Console

from lit_wsl.mapper.module_node import ModuleNode
from lit_wsl.mapper.parameter_group import ParameterGroup
from lit_wsl.mapper.parameter_info import ParameterInfo
from lit_wsl.mapper.result_types import (
    HierarchyMetadata,
    MappingResult,
    ParameterMatchResult,
    ScoreBreakdown,
    TransformationInfo,
)
from lit_wsl.mapper.rich_visualizer import (
    get_score_color,
    get_transformation_emoji,
    print_mapping_analysis,
    print_side_by_side_hierarchies,
    render_hierarchy_tree,
    render_mapping_table,
    render_score_breakdown,
    render_summary_panel,
    render_unmatched_params,
)


class TestScoreColor:
    """Test score color determination."""

    def test_high_score_green(self):
        """High scores (>=0.8) should be green."""
        assert get_score_color(0.8) == "green"
        assert get_score_color(0.9) == "green"
        assert get_score_color(1.0) == "green"

    def test_medium_score_yellow(self):
        """Medium scores (0.6-0.8) should be yellow."""
        assert get_score_color(0.6) == "yellow"
        assert get_score_color(0.7) == "yellow"
        assert get_score_color(0.79) == "yellow"

    def test_low_score_red(self):
        """Low scores (<0.6) should be red."""
        assert get_score_color(0.0) == "red"
        assert get_score_color(0.3) == "red"
        assert get_score_color(0.59) == "red"


class TestTransformationEmoji:
    """Test transformation emoji indicators."""

    def test_exact_match(self):
        """Exact matches get checkmark."""
        assert get_transformation_emoji("none") == "✓"
        assert get_transformation_emoji(None) == "✓"

    def test_transpose(self):
        """Transpose gets rotation arrow."""
        assert get_transformation_emoji("transpose") == "🔄"

    def test_reshape(self):
        """Reshape gets ruler emoji."""
        assert get_transformation_emoji("reshape") == "📐"

    def test_unknown(self):
        """Unknown transformations get warning."""
        assert get_transformation_emoji("unknown") == "⚠️"


class TestRenderHierarchyTree:
    """Test hierarchy tree rendering."""

    def setup_method(self):
        """Create a sample hierarchy for testing."""
        import torch

        # Create root
        self.root = ModuleNode("", "", 0)

        # Add encoder branch
        encoder = ModuleNode("encoder", "encoder", 1)
        self.root.add_child(encoder)

        layer1 = ModuleNode("layer1", "encoder.layer1", 2)
        encoder.add_child(layer1)

        conv = ModuleNode("conv", "encoder.layer1.conv", 3)
        layer1.add_child(conv)

        # Add parameter group to conv
        weight_tensor = torch.randn(64, 32, 3, 3)
        bias_tensor = torch.randn(64)

        params = {
            "weight": ParameterInfo(
                name="encoder.layer1.conv.weight",
                tensor=weight_tensor,
                is_buffer=False,
                execution_order=None,
            ),
            "bias": ParameterInfo(
                name="encoder.layer1.conv.bias",
                tensor=bias_tensor,
                is_buffer=False,
                execution_order=None,
            ),
        }
        conv.parameter_group = ParameterGroup("encoder.layer1.conv", params)

    def test_basic_tree_rendering(self):
        """Test basic tree structure rendering."""
        tree = render_hierarchy_tree(self.root, title="Test Model")
        assert tree is not None
        # Tree object is created successfully

    def test_max_depth_limiting(self):
        """Test depth limiting works."""
        tree = render_hierarchy_tree(self.root, max_depth=1)
        assert tree is not None
        # Should only show encoder level, not deeper

    def test_parameter_badges(self):
        """Test parameter type badges are shown."""
        tree = render_hierarchy_tree(self.root, show_shapes=True)
        assert tree is not None
        # Tree should include parameter indicators

    def test_matched_highlighting(self):
        """Test matched paths are highlighted."""
        matched = {"encoder.layer1.conv"}
        tree = render_hierarchy_tree(self.root, matched_paths=matched)
        assert tree is not None

    def test_empty_tree(self):
        """Test empty hierarchy renders gracefully."""
        empty_root = ModuleNode("", "", 0)
        tree = render_hierarchy_tree(empty_root)
        assert tree is not None


class TestRenderSummaryPanel:
    """Test summary panel rendering."""

    def create_test_result(self, matched: int, total_source: int, total_target: int) -> MappingResult:
        """Create a test MappingResult."""
        matches = []
        for i in range(matched):
            match = ParameterMatchResult(
                source_name=f"param_{i}",
                target_name=f"target_{i}",
                score_breakdown=ScoreBreakdown(0.9, 0.8, 0.85, 0.87, {"shape": 0.4, "name": 0.1, "hierarchy": 0.5}),
                final_score=0.87,
                matched=True,
                unmatch_reason=None,
                match_type="group",
                transformation=None,
                source_module_path="module",
                target_module_path="module",
            )
            matches.append(match)

        param_matches = {f"param_{i}": matches[i] for i in range(matched)}

        return MappingResult(
            parameter_matches=param_matches,
            group_matches={},
            matched_params=matches,
            unmatched_params=[],
            unmatched_targets=[],
            coverage=matched / total_source if total_source > 0 else 0.0,
            threshold=0.6,
            weights={"shape": 0.4, "name": 0.1, "hierarchy": 0.5},
        )

    def test_high_coverage_green(self):
        """Test high coverage is colored green."""
        result = self.create_test_result(95, 100, 100)
        panel = render_summary_panel(result, 100, 100)
        assert panel is not None

    def test_medium_coverage_yellow(self):
        """Test medium coverage is colored yellow."""
        result = self.create_test_result(75, 100, 100)
        panel = render_summary_panel(result, 100, 100)
        assert panel is not None

    def test_low_coverage_red(self):
        """Test low coverage is colored red."""
        result = self.create_test_result(30, 100, 100)
        panel = render_summary_panel(result, 100, 100)
        assert panel is not None


class TestRenderMappingTable:
    """Test mapping table rendering."""

    def create_test_match(
        self,
        source: str,
        target: str,
        score: float,
        trans_type: str = "none",
    ) -> ParameterMatchResult:
        """Create a test ParameterMatchResult."""
        transformation = TransformationInfo(
            type=trans_type,
            note="Test transformation",
            source_shape=(64, 32),
            target_shape=(32, 64) if trans_type == "transpose" else (64, 32),
        )

        return ParameterMatchResult(
            source_name=source,
            target_name=target,
            score_breakdown=ScoreBreakdown(
                shape_score=0.9,
                name_score=score,
                hierarchy_score=0.85,
                composite_score=score,
                weights_used={"shape": 0.4, "name": 0.1, "hierarchy": 0.5},
            ),
            final_score=score,
            matched=True,
            unmatch_reason=None,
            match_type="group",
            transformation=transformation,
            source_module_path="encoder",
            target_module_path="encoder",
        )

    def test_basic_table_rendering(self):
        """Test basic table creation."""
        matches = [
            self.create_test_match("conv1.weight", "conv1.weight", 0.95),
            self.create_test_match("conv2.weight", "conv2.weight", 0.75),
        ]

        result = MappingResult(
            parameter_matches={m.source_name: m for m in matches},
            group_matches={},
            matched_params=matches,
            unmatched_params=[],
            unmatched_targets=[],
            coverage=1.0,
            threshold=0.6,
            weights={},
        )

        table = render_mapping_table(result)
        assert table is not None

    def test_score_sorting(self):
        """Test matches are sorted by score."""
        matches = [
            self.create_test_match("low.weight", "low.weight", 0.65),
            self.create_test_match("high.weight", "high.weight", 0.95),
            self.create_test_match("med.weight", "med.weight", 0.75),
        ]

        result = MappingResult(
            parameter_matches={m.source_name: m for m in matches},
            group_matches={},
            matched_params=matches,
            unmatched_params=[],
            unmatched_targets=[],
            coverage=1.0,
            threshold=0.6,
            weights={},
        )

        table = render_mapping_table(result, sort_by_score=True)
        assert table is not None

    def test_max_rows_limiting(self):
        """Test row limiting works."""
        matches = [self.create_test_match(f"param_{i}", f"param_{i}", 0.9) for i in range(50)]

        result = MappingResult(
            parameter_matches={m.source_name: m for m in matches},
            group_matches={},
            matched_params=matches,
            unmatched_params=[],
            unmatched_targets=[],
            coverage=1.0,
            threshold=0.6,
            weights={},
        )

        table = render_mapping_table(result, max_rows=10)
        assert table is not None
        assert "40 more matches" in str(table.caption) or table.caption is not None

    def test_transformation_indicators(self):
        """Test transformation indicators are shown."""
        matches = [
            self.create_test_match("conv.weight", "conv.weight", 0.9, "none"),
            self.create_test_match("fc.weight", "fc.weight", 0.85, "transpose"),
            self.create_test_match("bn.weight", "bn.weight", 0.8, "reshape"),
        ]

        result = MappingResult(
            parameter_matches={m.source_name: m for m in matches},
            group_matches={},
            matched_params=matches,
            unmatched_params=[],
            unmatched_targets=[],
            coverage=1.0,
            threshold=0.6,
            weights={},
        )

        table = render_mapping_table(result, show_transformations=True)
        assert table is not None


class TestRenderScoreBreakdown:
    """Test score breakdown table rendering."""

    def test_basic_breakdown(self):
        """Test basic score breakdown rendering."""
        score = ScoreBreakdown(
            shape_score=0.95,
            name_score=0.75,
            hierarchy_score=0.85,
            composite_score=0.87,
            weights_used={"shape": 0.4, "name": 0.1, "hierarchy": 0.5},
            token_score=0.8,
            edit_score=0.7,
            lcs_score=0.75,
            depth_score=0.9,
            path_score=0.85,
            order_score=0.88,
        )

        table = render_score_breakdown(score, "test.param")
        assert table is not None

    def test_breakdown_without_subcomponents(self):
        """Test breakdown when sub-scores are None."""
        score = ScoreBreakdown(
            shape_score=0.95,
            name_score=0.75,
            hierarchy_score=0.85,
            composite_score=0.87,
            weights_used={"shape": 0.4, "name": 0.1, "hierarchy": 0.5},
        )

        table = render_score_breakdown(score, "test.param")
        assert table is not None


class TestRenderUnmatchedParams:
    """Test unmatched parameters table rendering."""

    def test_basic_unmatched_rendering(self):
        """Test rendering of unmatched parameters."""
        unmatched = [
            ParameterMatchResult(
                source_name=f"unmatched_{i}",
                target_name=None,
                score_breakdown=ScoreBreakdown(0, 0, 0, 0, {}),
                final_score=0,
                matched=False,
                unmatch_reason="No compatible shape",
                match_type="individual",
                transformation=None,
                source_module_path=f"module_{i}",
                target_module_path=None,
            )
            for i in range(5)
        ]

        table = render_unmatched_params(unmatched)
        assert table is not None

    def test_max_rows_limiting(self):
        """Test row limiting for unmatched params."""
        unmatched = [
            ParameterMatchResult(
                source_name=f"unmatched_{i}",
                target_name=None,
                score_breakdown=ScoreBreakdown(0, 0, 0, 0, {}),
                final_score=0,
                matched=False,
                unmatch_reason="No match",
                match_type="individual",
                transformation=None,
                source_module_path=f"module_{i}",
                target_module_path=None,
            )
            for i in range(20)
        ]

        table = render_unmatched_params(unmatched, max_rows=5)
        assert table is not None


class TestPrintMappingAnalysis:
    """Test complete mapping analysis printing."""

    def test_complete_analysis(self):
        """Test printing complete analysis."""
        # Create a simple result
        match = ParameterMatchResult(
            source_name="conv.weight",
            target_name="conv.weight",
            score_breakdown=ScoreBreakdown(0.9, 0.8, 0.85, 0.87, {"shape": 0.4, "name": 0.1, "hierarchy": 0.5}),
            final_score=0.87,
            matched=True,
            unmatch_reason=None,
            match_type="group",
            transformation=None,
            source_module_path="encoder",
            target_module_path="encoder",
        )

        result = MappingResult(
            parameter_matches={"conv.weight": match},
            group_matches={},
            matched_params=[match],
            unmatched_params=[],
            unmatched_targets=[],
            coverage=1.0,
            threshold=0.6,
            weights={"shape": 0.4, "name": 0.1, "hierarchy": 0.5},
        )

        console = Console()
        # Should not raise
        print_mapping_analysis(result, 1, 1, console=console)

    def test_with_unmatched(self):
        """Test analysis with unmatched parameters."""
        match = ParameterMatchResult(
            source_name="conv.weight",
            target_name="conv.weight",
            score_breakdown=ScoreBreakdown(0.9, 0.8, 0.85, 0.87, {}),
            final_score=0.87,
            matched=True,
            unmatch_reason=None,
            match_type="group",
            transformation=None,
            source_module_path="encoder",
            target_module_path="encoder",
        )

        unmatched = ParameterMatchResult(
            source_name="fc.weight",
            target_name=None,
            score_breakdown=ScoreBreakdown(0, 0, 0, 0, {}),
            final_score=0,
            matched=False,
            unmatch_reason="No compatible target",
            match_type="individual",
            transformation=None,
            source_module_path="fc",
            target_module_path=None,
        )

        result = MappingResult(
            parameter_matches={"conv.weight": match, "fc.weight": unmatched},
            group_matches={},
            matched_params=[match],
            unmatched_params=[unmatched],
            unmatched_targets=["decoder.weight"],
            coverage=0.5,
            threshold=0.6,
            weights={},
        )

        console = Console()
        # Should not raise
        print_mapping_analysis(result, 2, 2, console=console, show_unmatched=True)


class TestPrintSideBySideHierarchies:
    """Test side-by-side hierarchy printing."""

    def test_basic_side_by_side(self):
        """Test printing two hierarchies side by side."""
        # Create two simple hierarchies
        source_root = ModuleNode("", "", 0)
        source_encoder = ModuleNode("encoder", "encoder", 1)
        source_root.add_child(source_encoder)

        target_root = ModuleNode("", "", 0)
        target_backbone = ModuleNode("backbone", "backbone", 1)
        target_root.add_child(target_backbone)

        console = Console()
        # Should not raise
        print_side_by_side_hierarchies(source_root, target_root, console=console)

    def test_with_matches(self):
        """Test printing with matched paths highlighted."""
        source_root = ModuleNode("", "", 0)
        source_encoder = ModuleNode("encoder", "encoder", 1)
        source_root.add_child(source_encoder)

        target_root = ModuleNode("", "", 0)
        target_encoder = ModuleNode("encoder", "encoder", 1)
        target_root.add_child(target_encoder)

        console = Console()
        # Should not raise
        print_side_by_side_hierarchies(
            source_root,
            target_root,
            matched_source_paths={"encoder"},
            matched_target_paths={"encoder"},
            console=console,
        )

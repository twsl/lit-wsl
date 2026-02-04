"""Rich-based visualization utilities for weight mapping.

Provides beautiful, colorful console output for hierarchical trees,
mapping tables, score breakdowns, and summary panels using the Rich library.
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from lit_wsl.mapper.module_node import ModuleNode
from lit_wsl.mapper.result_types import MappingResult, ParameterMatchResult, ScoreBreakdown


def get_score_color(score: float) -> str:
    """Get color for a score value based on confidence thresholds.

    Args:
        score: Score value between 0.0 and 1.0

    Returns:
        Color name for Rich styling
    """
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "yellow"
    else:
        return "red"


def get_transformation_emoji(transformation_type: str | None) -> str:
    """Get emoji indicator for transformation type.

    Args:
        transformation_type: Type of transformation ("transpose", "reshape", "none", or None)

    Returns:
        Emoji string
    """
    if transformation_type is None or transformation_type == "none":
        return "✓"
    elif transformation_type == "transpose":
        return "🔄"
    elif transformation_type == "reshape":
        return "📐"
    else:
        return "⚠️"


def render_hierarchy_tree(
    root: ModuleNode,
    title: str = "Model Hierarchy",
    max_depth: int | None = None,
    show_shapes: bool = True,
    matched_paths: set[str] | None = None,
) -> Tree:
    """Render a hierarchical tree of module structure using Rich.

    Args:
        root: Root ModuleNode of the hierarchy
        title: Title for the tree
        max_depth: Maximum depth to display (None for unlimited)
        show_shapes: Whether to show parameter shapes
        matched_paths: Set of module paths that are matched (for highlighting)

    Returns:
        Rich Tree object ready for rendering
    """
    tree = Tree(f"[bold cyan]{title}[/bold cyan]")
    matched_paths = matched_paths or set()

    def add_node_to_tree(node: ModuleNode, parent_tree: Tree, current_depth: int = 0) -> None:
        """Recursively add nodes to the tree."""
        if max_depth is not None and current_depth > max_depth:
            return

        # Build node label
        node_name = node.name or "ROOT"
        is_matched = node.full_path in matched_paths

        # Add parameter type badges
        param_info = ""
        if node.parameter_group:
            param_types = sorted(node.parameter_group.param_types)
            badges = []
            for ptype in param_types:
                if ptype == "weight":
                    badges.append("[cyan]W[/cyan]")
                elif ptype == "bias":
                    badges.append("[yellow]B[/yellow]")
                else:
                    badges.append(f"[dim]{ptype[0].upper()}[/dim]")
            param_info = f" [{''.join(badges)}]"

            # Add shape information if requested
            if show_shapes and node.parameter_group.params:
                shapes = []
                for ptype in param_types:
                    if ptype in node.parameter_group.params:
                        param = node.parameter_group.params[ptype]
                        shapes.append(f"{ptype}:{param.shape}")
                if shapes:
                    param_info += f" [dim]({', '.join(shapes)})[/dim]"

        # Style based on match status
        if is_matched:
            label = f"[bold green]✓ {node_name}[/bold green]{param_info}"
        elif node.parameter_group:
            label = f"[bold]{node_name}[/bold]{param_info}"
        else:
            label = f"[dim]{node_name}[/dim]{param_info}"

        # Add to tree
        branch = parent_tree.add(label)

        # Recursively add children (sorted by name for consistency)
        for child_name in sorted(node.children.keys()):
            child = node.children[child_name]
            add_node_to_tree(child, branch, current_depth + 1)

    # Build tree starting from root's children
    if root.children:
        for child_name in sorted(root.children.keys()):
            child = root.children[child_name]
            add_node_to_tree(child, tree, 0)
    else:
        tree.add("[dim]No modules[/dim]")

    return tree


def render_summary_panel(result: MappingResult, source_count: int, target_count: int) -> Panel:
    """Render a summary panel with mapping statistics.

    Args:
        result: MappingResult containing mapping statistics
        source_count: Total number of source parameters
        target_count: Total number of target parameters

    Returns:
        Rich Panel with summary information
    """
    # Create summary text
    matched = len(result.matched_params)
    coverage_pct = result.coverage * 100

    # Color-code coverage
    if coverage_pct >= 90:
        coverage_color = "green"
    elif coverage_pct >= 70:
        coverage_color = "yellow"
    else:
        coverage_color = "red"

    summary_text = Text()
    summary_text.append("Source Parameters: ", style="bold")
    summary_text.append(f"{source_count}\n", style="cyan")
    summary_text.append("Target Parameters: ", style="bold")
    summary_text.append(f"{target_count}\n", style="cyan")
    summary_text.append("Matched: ", style="bold")
    summary_text.append(f"{matched} ", style="green")
    summary_text.append(f"({coverage_pct:.1f}%)\n", style=coverage_color)
    summary_text.append("Unmatched Source: ", style="bold")
    summary_text.append(f"{len(result.unmatched_params)}\n", style="red")
    summary_text.append("Unmatched Target: ", style="bold")
    summary_text.append(f"{len(result.unmatched_targets)}\n", style="red")
    summary_text.append("Threshold: ", style="bold")
    summary_text.append(f"{result.threshold:.2f}\n", style="dim")

    # Add score component weights
    if result.weights:
        summary_text.append("\nWeights: ", style="bold")
        weight_parts = [f"{k}={v:.2f}" for k, v in result.weights.items()]
        summary_text.append(", ".join(weight_parts), style="dim")

    return Panel(summary_text, title="[bold]Mapping Summary[/bold]", border_style="cyan")


def render_mapping_table(
    result: MappingResult,
    max_rows: int | None = None,
    show_transformations: bool = True,
    show_scores: bool = True,
    sort_by_score: bool = True,
) -> Table:
    """Render a table of parameter mappings.

    Args:
        result: MappingResult containing matches
        max_rows: Maximum number of rows to display (None for all)
        show_transformations: Whether to show transformation indicators
        show_scores: Whether to show score column
        sort_by_score: Whether to sort by score (descending)

    Returns:
        Rich Table with mapping information
    """
    table = Table(title="[bold]Parameter Mappings[/bold]", show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column("Source", style="cyan", no_wrap=False, max_width=40)
    table.add_column("Target", style="cyan", no_wrap=False, max_width=40)
    if show_scores:
        table.add_column("Score", justify="right", style="bold", width=8)
    table.add_column("Shape", style="dim", width=20)
    if show_transformations:
        table.add_column("Transform", justify="center", width=10)
    table.add_column("Type", justify="center", width=8)

    # Get matches and optionally sort
    matches = result.matched_params
    if sort_by_score:
        matches = sorted(matches, key=lambda x: x.final_score, reverse=True)

    # Apply row limit
    if max_rows is not None:
        matches = matches[:max_rows]
        rows_hidden = len(result.matched_params) - max_rows
    else:
        rows_hidden = 0

    # Add rows
    for match in matches:
        source_name = match.source_name
        target_name = match.target_name or "N/A"
        score = match.final_score
        score_color = get_score_color(score)

        # Get shape info
        if match.transformation and match.transformation.type != "none":
            shape_info = f"{match.transformation.source_shape} → {match.transformation.target_shape}"
        else:
            shape_info = str(match.transformation.source_shape if match.transformation else "")

        # Build row
        row = [source_name, target_name]
        if show_scores:
            row.append(f"[{score_color}]{score:.3f}[/{score_color}]")
        row.append(shape_info)
        if show_transformations:
            trans_type = match.transformation.type if match.transformation else "none"
            emoji = get_transformation_emoji(trans_type)
            row.append(emoji)

        # Match type badge
        if match.match_type == "group":
            row.append("[green]GROUP[/green]")
        else:
            row.append("[yellow]INDIV[/yellow]")

        table.add_row(*row)

    if rows_hidden > 0:
        table.caption = f"[dim]... and {rows_hidden} more matches[/dim]"

    return table


def render_score_breakdown(score: ScoreBreakdown, param_name: str) -> Table:
    """Render a detailed score breakdown table.

    Args:
        score: ScoreBreakdown with detailed scoring components
        param_name: Parameter name for the table title

    Returns:
        Rich Table with score breakdown
    """
    table = Table(
        title=f"[bold]Score Breakdown: {param_name}[/bold]",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Component", style="bold")
    table.add_column("Score", justify="right", width=10)
    table.add_column("Weight", justify="right", width=10)
    table.add_column("Contribution", justify="right", width=12)
    table.add_column("Bar", width=20)

    # Main components
    components = [
        ("Shape", score.shape_score, score.weights_used.get("shape", 0.0)),
        ("Name", score.name_score, score.weights_used.get("name", 0.0)),
        ("Hierarchy", score.hierarchy_score, score.weights_used.get("hierarchy", 0.0)),
    ]

    for name, value, weight in components:
        contribution = value * weight
        color = get_score_color(value)

        # Create progress bar
        bar_filled = int(value * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)

        table.add_row(
            name,
            f"[{color}]{value:.3f}[/{color}]",
            f"{weight:.2f}",
            f"{contribution:.3f}",
            f"[{color}]{bar}[/{color}]",
        )

    # Add separator and composite
    table.add_row("", "", "", "", "")
    composite_color = get_score_color(score.composite_score)
    table.add_row(
        "[bold]Composite[/bold]",
        f"[bold {composite_color}]{score.composite_score:.3f}[/bold {composite_color}]",
        "",
        "",
        "",
    )

    # Sub-components (if available)
    sub_components = []
    if score.token_score is not None:
        sub_components.append(("  Token", score.token_score))
    if score.edit_score is not None:
        sub_components.append(("  Edit Distance", score.edit_score))
    if score.lcs_score is not None:
        sub_components.append(("  LCS", score.lcs_score))
    if score.depth_score is not None:
        sub_components.append(("  Depth", score.depth_score))
    if score.path_score is not None:
        sub_components.append(("  Path", score.path_score))
    if score.order_score is not None:
        sub_components.append(("  Order", score.order_score))

    if sub_components:
        table.add_row("", "", "", "", "")
        for name, value in sub_components:
            color = get_score_color(value)
            bar_filled = int(value * 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            table.add_row(
                f"[dim]{name}[/dim]",
                f"[{color}]{value:.3f}[/{color}]",
                "[dim]—[/dim]",
                "[dim]—[/dim]",
                f"[dim]{bar}[/dim]",
            )

    return table


def render_unmatched_params(
    unmatched: list[ParameterMatchResult],
    title: str = "Unmatched Parameters",
    max_rows: int = 10,
) -> Table:
    """Render a table of unmatched parameters.

    Args:
        unmatched: List of unmatched ParameterMatchResult objects
        title: Title for the table
        max_rows: Maximum number of rows to show

    Returns:
        Rich Table with unmatched parameters
    """
    table = Table(title=f"[bold red]{title}[/bold red]", show_header=True, header_style="bold magenta")

    table.add_column("Parameter", style="dim", no_wrap=False, max_width=50)
    table.add_column("Reason", style="yellow", no_wrap=False, max_width=30)
    table.add_column("Module Path", style="dim", no_wrap=False, max_width=40)

    rows_to_show = unmatched[:max_rows]
    for match in rows_to_show:
        table.add_row(
            match.source_name,
            match.unmatch_reason or "Unknown",
            match.source_module_path,
        )

    if len(unmatched) > max_rows:
        table.caption = f"[dim]... and {len(unmatched) - max_rows} more[/dim]"

    return table


def print_mapping_analysis(
    result: MappingResult,
    source_count: int,
    target_count: int,
    console: Console | None = None,
    show_unmatched: bool = True,
    max_matches: int = 20,
    max_unmatched: int = 10,
) -> None:
    """Print a complete mapping analysis using Rich visualizations.

    Args:
        result: MappingResult to visualize
        source_count: Total number of source parameters
        target_count: Total number of target parameters
        console: Rich Console to use (creates new one if None)
        show_unmatched: Whether to show unmatched parameters
        max_matches: Maximum number of matches to display
        max_unmatched: Maximum number of unmatched items to display
    """
    if console is None:
        console = Console()

    # Print summary panel
    console.print()
    console.print(render_summary_panel(result, source_count, target_count))
    console.print()

    # Print mapping table
    if result.matched_params:
        console.print(render_mapping_table(result, max_rows=max_matches))
        console.print()
    else:
        console.print("[bold red]No matches found![/bold red]")
        console.print()

    # Print unmatched parameters
    if show_unmatched:
        if result.unmatched_params:
            console.print(
                render_unmatched_params(result.unmatched_params, "Unmatched Source Parameters", max_unmatched)
            )
            console.print()

        if result.unmatched_targets:
            table = Table(title="[bold red]Unmatched Target Parameters[/bold red]", show_header=True)
            table.add_column("Parameter", style="dim", no_wrap=False)

            for name in result.unmatched_targets[:max_unmatched]:
                table.add_row(name)

            if len(result.unmatched_targets) > max_unmatched:
                table.caption = f"[dim]... and {len(result.unmatched_targets) - max_unmatched} more[/dim]"

            console.print(table)
            console.print()


def print_side_by_side_hierarchies(
    source_root: ModuleNode,
    target_root: ModuleNode,
    matched_source_paths: set[str] | None = None,
    matched_target_paths: set[str] | None = None,
    max_depth: int | None = None,
    console: Console | None = None,
) -> None:
    """Print source and target hierarchies side by side.

    Args:
        source_root: Root node of source hierarchy
        target_root: Root node of target hierarchy
        matched_source_paths: Set of matched source module paths (for highlighting)
        matched_target_paths: Set of matched target module paths (for highlighting)
        max_depth: Maximum depth to display
        console: Rich Console to use (creates new one if None)
    """
    if console is None:
        console = Console()

    from rich.columns import Columns

    source_tree = render_hierarchy_tree(
        source_root,
        title="Source Model",
        max_depth=max_depth,
        matched_paths=matched_source_paths,
    )

    target_tree = render_hierarchy_tree(
        target_root,
        title="Target Model",
        max_depth=max_depth,
        matched_paths=matched_target_paths,
    )

    console.print()
    console.print(Columns([source_tree, target_tree], equal=True, expand=True))
    console.print()

"""Command-line interface for lit-wsl weight mapping visualization."""

import argparse
from pathlib import Path
import sys

import torch
from torch import nn

from lit_wsl.__about__ import __version__


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load a checkpoint file and extract state dict.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        State dictionary from checkpoint

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint can't be loaded
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec B614
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_model_class(model_spec: str) -> type[nn.Module]:
    """Load a model class from a module path.

    Args:
        model_spec: Module path like 'package.module:ClassName'

    Returns:
        Model class

    Raises:
        ValueError: If model_spec format is invalid
        ImportError: If module can't be imported
        AttributeError: If class not found in module
    """
    if ":" not in model_spec:
        raise ValueError(f"Invalid model spec '{model_spec}'. Expected format: 'module.path:ClassName'")

    module_path, class_name = model_spec.rsplit(":", 1)

    try:
        import importlib

        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}") from e

    try:
        model_class = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'") from e

    if not issubclass(model_class, nn.Module):
        raise TypeError(f"{class_name} is not a torch.nn.Module")

    return model_class


def cmd_visualize_hierarchy(args: argparse.Namespace) -> int:
    """Command: visualize model hierarchy tree.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from lit_wsl.mapper.weight_mapper import WeightMapper

    try:
        state_dict = load_checkpoint(args.checkpoint)

        if args.model:
            model_class = load_model_class(args.model)
            target_model = model_class()
        else:
            target_model = None

        mapper = WeightMapper.from_state_dict(
            source_state_dict=state_dict,
            target_module=target_model,
        )

        if args.side == "both" or args.side == "source":
            from rich.console import Console

            from lit_wsl.mapper.rich_visualizer import render_hierarchy_tree

            console = Console()
            console.print()
            tree = render_hierarchy_tree(
                mapper.source_hierarchy,
                title="Source Model Hierarchy",
                max_depth=args.max_depth,
                show_shapes=args.show_shapes,
            )
            console.print(tree)

        if args.side == "both" and target_model:
            print()

        if args.side == "both" or args.side == "target":
            if target_model is None:
                print("Error: --model required to show target hierarchy", file=sys.stderr)
                return 1

            from rich.console import Console

            from lit_wsl.mapper.rich_visualizer import render_hierarchy_tree

            console = Console()
            console.print()
            tree = render_hierarchy_tree(
                mapper.target_hierarchy,
                title="Target Model Hierarchy",
                max_depth=args.max_depth,
                show_shapes=args.show_shapes,
            )
            console.print(tree)

        if args.output:
            console = Console(record=True)
            tree = render_hierarchy_tree(
                mapper.source_hierarchy if args.side == "source" else mapper.target_hierarchy,
                title=f"{args.side.title()} Model Hierarchy",
                max_depth=args.max_depth,
                show_shapes=args.show_shapes,
            )
            console.print(tree)

            if args.format == "html":
                console.save_html(args.output)
                print(f"Saved to {args.output}")
            else:
                console.save_text(args.output)
                print(f"Saved to {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_visualize_mapping(args: argparse.Namespace) -> int:
    """Command: visualize weight mapping results.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from lit_wsl.mapper.weight_mapper import WeightMapper

    try:
        state_dict = load_checkpoint(args.checkpoint)

        model_class = load_model_class(args.model)
        target_model = model_class()

        mapper = WeightMapper.from_state_dict(
            source_state_dict=state_dict,
            target_module=target_model,
        )

        result = mapper.suggest_mapping(threshold=args.threshold)
        mapper.visualize_mapping(
            result=result,
            show_unmatched=args.show_unmatched,
            max_matches=args.max_matches,
            max_unmatched=args.max_unmatched,
        )

        if args.output:
            from rich.console import Console

            console = Console(record=True)
            from lit_wsl.mapper.rich_visualizer import print_mapping_analysis

            print_mapping_analysis(
                result=result,
                source_count=len(mapper.source_params),
                target_count=len(mapper.target_params),
                console=console,
                show_unmatched=args.show_unmatched,
                max_matches=args.max_matches,
                max_unmatched=args.max_unmatched,
            )

            if args.format == "html":
                console.save_html(args.output)
                print(f"\nSaved to {args.output}")
            else:
                console.save_text(args.output)
                print(f"\nSaved to {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        if args.verbose:
            traceback.print_exc()
        return 1


def cmd_analyze_mapping(args: argparse.Namespace) -> int:
    """Command: comprehensive mapping analysis with hierarchies and scores.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from lit_wsl.mapper.weight_mapper import WeightMapper

    try:
        state_dict = load_checkpoint(args.checkpoint)

        model_class = load_model_class(args.model)
        target_model = model_class()

        mapper = WeightMapper.from_state_dict(
            source_state_dict=state_dict,
            target_module=target_model,
        )
        if args.show_hierarchy:
            print("\n" + "=" * 80)
            print("MODEL HIERARCHIES")
            print("=" * 80)
            mapper.visualize_hierarchies(
                show_matches=False,
                max_depth=args.max_depth,
            )

        result = mapper.suggest_mapping(threshold=args.threshold)

        print("\n" + "=" * 80)
        print("MAPPING ANALYSIS")
        print("=" * 80)
        mapper.visualize_mapping(
            result=result,
            show_unmatched=args.show_unmatched,
            max_matches=args.max_matches,
            max_unmatched=args.max_unmatched,
        )

        if args.show_hierarchy:
            print("\n" + "=" * 80)
            print("MATCHED HIERARCHIES")
            print("=" * 80)
            mapper.visualize_hierarchies(
                show_matches=True,
                max_depth=args.max_depth,
            )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        if args.verbose:
            traceback.print_exc()
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Command: print version information.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (always 0)
    """
    print(f"lit-wsl v{__version__}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="lit-wsl",
        description="Weight mapping and visualization tools for PyTorch models",
    )
    parser.add_argument("--version", action="version", version=f"lit-wsl {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # visualize-hierarchy command
    hierarchy_parser = subparsers.add_parser(
        "visualize-hierarchy",
        help="Visualize model hierarchy tree structure",
    )
    hierarchy_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file",
    )
    hierarchy_parser.add_argument(
        "--model",
        type=str,
        help="Target model class (format: 'module.path:ClassName'). Required for target/both views.",
    )
    hierarchy_parser.add_argument(
        "--side",
        type=str,
        choices=["source", "target", "both"],
        default="source",
        help="Which hierarchy to show (default: source)",
    )
    hierarchy_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth to display (default: unlimited)",
    )
    hierarchy_parser.add_argument(
        "--show-shapes",
        action="store_true",
        default=True,
        help="Show parameter shapes (default: true)",
    )
    hierarchy_parser.add_argument(
        "--output",
        type=str,
        help="Save output to file",
    )
    hierarchy_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "html"],
        default="html",
        help="Output format (default: html)",
    )
    hierarchy_parser.set_defaults(func=cmd_visualize_hierarchy)

    # visualize-mapping command
    mapping_parser = subparsers.add_parser(
        "visualize-mapping",
        help="Visualize weight mapping results",
    )
    mapping_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to source checkpoint file",
    )
    mapping_parser.add_argument(
        "model",
        type=str,
        help="Target model class (format: 'module.path:ClassName')",
    )
    mapping_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum score threshold for matching (default: 0.0)",
    )
    mapping_parser.add_argument(
        "--show-unmatched",
        action="store_true",
        default=True,
        help="Show unmatched parameters (default: true)",
    )
    mapping_parser.add_argument(
        "--max-matches",
        type=int,
        default=20,
        help="Maximum number of matches to display (default: 20)",
    )
    mapping_parser.add_argument(
        "--max-unmatched",
        type=int,
        default=10,
        help="Maximum number of unmatched items to display (default: 10)",
    )
    mapping_parser.add_argument(
        "--output",
        type=str,
        help="Save output to file",
    )
    mapping_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "html"],
        default="html",
        help="Output format (default: html)",
    )
    mapping_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed error messages",
    )
    mapping_parser.set_defaults(func=cmd_visualize_mapping)

    # analyze-mapping command
    analyze_parser = subparsers.add_parser(
        "analyze-mapping",
        help="Comprehensive mapping analysis with hierarchies and detailed scores",
    )
    analyze_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to source checkpoint file",
    )
    analyze_parser.add_argument(
        "model",
        type=str,
        help="Target model class (format: 'module.path:ClassName')",
    )
    analyze_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum score threshold for matching (default: 0.0)",
    )
    analyze_parser.add_argument(
        "--show-hierarchy",
        action="store_true",
        default=True,
        help="Show hierarchical tree structures (default: true)",
    )
    analyze_parser.add_argument(
        "--show-unmatched",
        action="store_true",
        default=True,
        help="Show unmatched parameters (default: true)",
    )
    analyze_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth for hierarchy display (default: unlimited)",
    )
    analyze_parser.add_argument(
        "--max-matches",
        type=int,
        default=20,
        help="Maximum number of matches to display (default: 20)",
    )
    analyze_parser.add_argument(
        "--max-unmatched",
        type=int,
        default=10,
        help="Maximum number of unmatched items to display (default: 10)",
    )
    analyze_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed error messages",
    )
    analyze_parser.set_defaults(func=cmd_analyze_mapping)

    # version command (also handled by --version flag)
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )
    version_parser.set_defaults(func=cmd_version)

    return parser


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

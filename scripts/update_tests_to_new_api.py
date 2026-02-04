#!/usr/bin/env python3
"""Script to update test files from old tuple-based API to new MappingResult dataclass API.

This script updates all test files that use WeightMapper.suggest_mapping() to use the new
dataclass-based return type instead of tuple unpacking.

Changes:
- OLD: mapping, unmatched = mapper.suggest_mapping(...)
- NEW: result = mapper.suggest_mapping(...)
       mapping = result.get_mapping()
       unmatched = result.get_unmatched()

- OLD: mappings, unmatched = mapper.suggest_mapping(return_scores=True, ...)
- NEW: result = mapper.suggest_mapping(...)
       mappings = result.get_mapping_with_scores()
       unmatched = result.get_unmatched()
"""

from pathlib import Path
import re


def update_suggest_mapping_calls(content: str) -> str:
    """Update suggest_mapping calls to use new dataclass API."""
    # Pattern 1: mapping, unmatched = mapper.suggest_mapping(...)
    # Matches with or without return_scores
    pattern1 = re.compile(
        r"(\s+)(mapping|mappings),\s+(unmatched|_)\s*=\s*mapper\.suggest_mapping\(((?:.*?))\)", re.MULTILINE
    )

    def replace_pattern1(match):
        indent = match.group(1)
        mapping_var = match.group(2)  # 'mapping' or 'mappings'
        unmatched_var = match.group(3)  # 'unmatched' or '_'
        args = match.group(4)

        # Check if return_scores=True is in args
        has_return_scores = "return_scores=True" in args

        # Remove return_scores from args if present
        args_cleaned = re.sub(r",?\s*return_scores=True", "", args)
        args_cleaned = re.sub(r"return_scores=True,?\s*", "", args_cleaned)
        args_cleaned = args_cleaned.strip()

        # Build the replacement
        lines = []
        lines.append(f"{indent}result = mapper.suggest_mapping({args_cleaned})")

        if has_return_scores:
            lines.append(f"{indent}{mapping_var} = result.get_mapping_with_scores()")
        else:
            lines.append(f"{indent}{mapping_var} = result.get_mapping()")

        if unmatched_var != "_":
            lines.append(f"{indent}{unmatched_var} = result.get_unmatched()")

        return "\n".join(lines)

    content = pattern1.sub(replace_pattern1, content)

    # Pattern 2: Just running suggest_mapping without assignment (like mapper.suggest_mapping(threshold=0.99))
    # This is fine as-is, no change needed

    return content


def update_file(file_path: Path) -> bool:
    """Update a single file. Returns True if changes were made."""
    print(f"Processing {file_path}...")

    try:
        content = file_path.read_text()
        original_content = content

        # Apply transformations
        content = update_suggest_mapping_calls(content)

        # Check if any changes were made
        if content != original_content:
            file_path.write_text(content)
            print(f"  ✓ Updated {file_path}")
            return True
        else:
            print(f"  - No changes needed for {file_path}")
            return False
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False


def main():
    """Main entry point."""
    # Find all test files
    workspace_root = Path(__file__).parent.parent
    test_files = []

    # Unit tests
    unit_mapper_tests = workspace_root / "tests" / "unit" / "mapper"
    if unit_mapper_tests.exists():
        test_files.extend(unit_mapper_tests.glob("test_*.py"))

    # Integration tests
    integration_tests = workspace_root / "tests" / "integration"
    if integration_tests.exists():
        test_files.extend(integration_tests.glob("test_*.py"))

    print(f"Found {len(test_files)} test files to process\n")

    updated_count = 0
    for test_file in sorted(test_files):
        if update_file(test_file):
            updated_count += 1

    print(f"\n{'=' * 60}")
    print(f"Summary: Updated {updated_count} out of {len(test_files)} files")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

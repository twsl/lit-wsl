#!/usr/bin/env python3
"""Script to update test files from old tuple-based API to new MappingResult dataclass API."""

from pathlib import Path
import re


def update_test_file(filepath: Path) -> tuple[int, list[str]]:
    """Update a single test file to use the new API.

    Returns:
        Tuple of (number of changes, list of change descriptions)
    """
    content = filepath.read_text()
    original_content = content
    changes = []

    # Pattern 1: mapping, unmatched = mapper.suggest_mapping(...)
    # Replace with: result = mapper.suggest_mapping(...); mapping = result.get_mapping(); unmatched = result.get_unmatched()
    pattern1 = r"(\s+)mapping,\s*unmatched\s*=\s*(mapper\.suggest_mapping\([^)]*\))"
    replacement1 = r"\1result = \2\n\1mapping = result.get_mapping()\n\1unmatched = result.get_unmatched()"
    new_content, count1 = re.subn(pattern1, replacement1, content)
    if count1 > 0:
        changes.append(f"Updated {count1} basic mapping assignments")
        content = new_content

    # Pattern 2: mapping, _ = mapper.suggest_mapping(...)
    # Replace with: result = mapper.suggest_mapping(...); mapping = result.get_mapping()
    pattern2 = r"(\s+)mapping,\s*_\s*=\s*(mapper\.suggest_mapping\([^)]*\))"
    replacement2 = r"\1result = \2\n\1mapping = result.get_mapping()"
    new_content, count2 = re.subn(pattern2, replacement2, content)
    if count2 > 0:
        changes.append(f"Updated {count2} mapping assignments (ignoring unmatched)")
        content = new_content

    # Pattern 3: mappings, unmatched = mapper.suggest_mapping(return_scores=True, ...)
    # Replace with: result = mapper.suggest_mapping(...); mappings = result.get_mapping_with_scores(); unmatched = result.get_unmatched()
    # First remove return_scores=True parameter
    pattern3a = r"(\s+)mappings,\s*unmatched\s*=\s*mapper\.suggest_mapping\(return_scores=True,\s*([^)]*)\)"
    replacement3a = r"\1result = mapper.suggest_mapping(\2)\n\1mappings = result.get_mapping_with_scores()\n\1unmatched = result.get_unmatched()"
    new_content, count3a = re.subn(pattern3a, replacement3a, content)
    if count3a > 0:
        changes.append(f"Updated {count3a} score-based mappings (return_scores=True, ...)")
        content = new_content

    # Pattern 3b: mappings, unmatched = mapper.suggest_mapping(..., return_scores=True)
    pattern3b = r"(\s+)mappings,\s*unmatched\s*=\s*mapper\.suggest_mapping\(([^)]*),\s*return_scores=True\)"
    replacement3b = r"\1result = mapper.suggest_mapping(\2)\n\1mappings = result.get_mapping_with_scores()\n\1unmatched = result.get_unmatched()"
    new_content, count3b = re.subn(pattern3b, replacement3b, content)
    if count3b > 0:
        changes.append(f"Updated {count3b} score-based mappings (..., return_scores=True)")
        content = new_content

    # Pattern 4: mapping_result, unmatched = mapper.suggest_mapping(...)
    # (for test_weight_mapping_workflow.py which uses mapping_result variable)
    pattern4 = r"(\s+)mapping_result,\s*unmatched\s*=\s*(mapper\.suggest_mapping\([^)]*\))"
    replacement4 = r"\1result = \2\n\1mapping_result = result.get_mapping()\n\1unmatched = result.get_unmatched()"
    new_content, count4 = re.subn(pattern4, replacement4, content)
    if count4 > 0:
        changes.append(f"Updated {count4} mapping_result assignments")
        content = new_content

    # Pattern 5: mappings_with_scores, unmatched_scores = mapper.suggest_mapping(return_scores=True)
    pattern5a = r"(\s+)mappings_with_scores,\s*unmatched_scores\s*=\s*mapper\.suggest_mapping\(return_scores=True\)"
    replacement5a = r"\1result = mapper.suggest_mapping()\n\1mappings_with_scores = result.get_mapping_with_scores()\n\1unmatched_scores = result.get_unmatched()"
    new_content, count5a = re.subn(pattern5a, replacement5a, content)
    if count5a > 0:
        changes.append(f"Updated {count5a} mappings_with_scores assignments")
        content = new_content

    # Pattern 6: scores_no_order, unmatched_scores_no = mapper_no_order.suggest_mapping(return_scores=True)
    pattern6 = r"(\s+)scores_(\w+),\s*unmatched_scores_(\w+)\s*=\s*(mapper_\w+)\.suggest_mapping\(return_scores=True\)"
    replacement6 = r"\1result_\2 = \4.suggest_mapping()\n\1scores_\2 = result_\2.get_mapping_with_scores()\n\1unmatched_scores_\3 = result_\2.get_unmatched()"
    new_content, count6 = re.subn(pattern6, replacement6, content)
    if count6 > 0:
        changes.append(f"Updated {count6} scores_* assignments")
        content = new_content

    # Pattern 7: mapping_no_order, unmatched_no_order = mapper_no_order.suggest_mapping(...)
    pattern7 = r"(\s+)mapping_(\w+),\s*unmatched_(\w+)\s*=\s*(mapper_\w+)\.suggest_mapping\(([^)]*)\)"
    replacement7 = r"\1result_\2 = \4.suggest_mapping(\5)\n\1mapping_\2 = result_\2.get_mapping()\n\1unmatched_\3 = result_\2.get_unmatched()"
    new_content, count7 = re.subn(pattern7, replacement7, content)
    if count7 > 0:
        changes.append(f"Updated {count7} mapping_* assignments")
        content = new_content

    total_changes = count1 + count2 + count3a + count3b + count4 + count5a + count6 + count7

    if content != original_content:
        filepath.write_text(content)
        print(f"\n✓ Updated {filepath}")
        for change in changes:
            print(f"  - {change}")
        return total_changes, changes
    else:
        print(f"\n  No changes needed for {filepath}")
        return 0, []


def main():
    """Update all test files."""
    test_files = [
        Path("tests/unit/mapper/test_weight_mapper_complex.py"),
        Path("tests/unit/mapper/test_weight_mapper.py"),
        Path("tests/unit/mapper/test_weight_mapping_workflow.py"),
        Path("tests/integration/test_real_models.py"),
    ]

    total_changes = 0
    for filepath in test_files:
        if filepath.exists():
            changes, _ = update_test_file(filepath)
            total_changes += changes
        else:
            print(f"\n⚠ File not found: {filepath}")

    print(f"\n{'=' * 60}")
    print(f"Total changes made: {total_changes}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

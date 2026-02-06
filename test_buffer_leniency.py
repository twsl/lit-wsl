#!/usr/bin/env python3
"""Quick test to demonstrate lenient buffer matching feature."""

from pathlib import Path
import pickle  # nosec B403

from lit_wsl.mapper.weight_mapper import WeightMapper


def test_buffer_modes():
    """Compare mapping results with different buffer modes."""
    with Path("data/PPYOLOE/source_params.pkl").open("rb") as f:
        source_params = pickle.load(f)  # nosec B301  # noqa: S301

    with Path("data/PPYOLOE/target_params.pkl").open("rb") as f:
        target_params = pickle.load(f)  # nosec B301  # noqa: S301

    modes = ["strict", "lenient", "exclude"]
    results = {}

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"Testing buffer_matching_mode='{mode}'")
        print(f"{'=' * 60}")

        mapper = WeightMapper(
            source_params=source_params,
            target_params=target_params,
            buffer_matching_mode=mode,
        )

        result = mapper.suggest_mapping(threshold=0.3)
        mapping = result.get_mapping()
        unmatched = result.get_unmatched()

        results[mode] = {
            "matched": len(mapping),
            "unmatched_source": len(unmatched.get("source", [])),
            "unmatched_target": len(unmatched.get("target", [])),
        }

        print(f"  Matched parameters: {results[mode]['matched']}")
        print(f"  Unmatched source: {results[mode]['unmatched_source']}")
        print(f"  Unmatched target: {results[mode]['unmatched_target']}")

        # Show a few sample matches
        print("\n  Sample matches (first 5):")
        for i, (src, tgt) in enumerate(list(mapping.items())[:5]):
            print(f"    {src} -> {tgt}")

    print(f"\n{'=' * 60}")
    print("Comparison Summary")
    print(f"{'=' * 60}")
    for mode in modes:
        print(
            f"{mode:10s}: {results[mode]['matched']:4d} matched, "
            f"{results[mode]['unmatched_source']:4d} unmatched source"
        )

    # Verify lenient mode >= strict mode in matches
    assert results["lenient"]["matched"] >= results["strict"]["matched"], (
        f"Lenient mode should match >= strict mode, "
        f"but got {results['lenient']['matched']} vs {results['strict']['matched']}"
    )

    print("\n✅ Lenient buffer matching is working!")
    print(
        f"   Lenient mode matched {results['lenient']['matched'] - results['strict']['matched']} "
        f"additional parameters compared to strict mode."
    )


if __name__ == "__main__":
    test_buffer_modes()

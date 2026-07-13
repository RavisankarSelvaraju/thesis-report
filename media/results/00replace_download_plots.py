#!/usr/bin/env python3

"""
Replace newly downloaded result files while preserving the existing
Trajectory_With_Heading plots.

Behaviour
---------
1. Recursively processes every dataset directory.
2. Converts spaces in downloaded PNG filenames to underscores.
3. Removes browser duplicate suffixes such as " (1)", " (2)", etc.
4. Replaces existing PNG files with the newly downloaded versions.
5. Preserves the existing Trajectory_With_Heading plots.
6. Replaces existing latex_metrics_table files with downloaded
   versions such as "latex_metrics_table_DATA_02_03 (1).txt".
7. Runs in dry-run mode unless --apply is provided.
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path


PROTECTED_PLOT_PREFIXES = (
    "trajectory_with_heading_",
)

METRICS_TABLE_PREFIX = "latex_metrics_table_"


def normalize_downloaded_filename(filename: str) -> str:
    """
    Convert a downloaded filename to the filename used by LaTeX.

    Examples
    --------
    APE rotational comparison DATA 02 03.png
        -> APE_rotational_comparison_DATA_02_03.png

    latex_metrics_table_DATA_02_03 (1).txt
        -> latex_metrics_table_DATA_02_03.txt

    yaw error comparison DATA 02 03 (2).png
        -> yaw_error_comparison_DATA_02_03.png
    """
    path = Path(filename)
    suffix = path.suffix
    stem = path.stem

    # Remove browser duplicate suffixes: " (1)", " (2)", etc.
    stem = re.sub(r"\s+\(\d+\)$", "", stem)

    # Replace one or more spaces with an underscore.
    stem = re.sub(r"\s+", "_", stem.strip())

    return f"{stem}{suffix}"


def is_protected_trajectory(filename: str) -> bool:
    """Check whether the file is a trajectory plot that must be preserved."""
    normalized_name = normalize_downloaded_filename(filename).lower()

    return any(
        normalized_name.startswith(prefix)
        for prefix in PROTECTED_PLOT_PREFIXES
    )


def collect_files(
    root: Path,
) -> tuple[dict[Path, list[Path]], dict[Path, list[Path]]]:
    """
    Collect downloaded PNG files and duplicate metrics-table files.

    Files are grouped according to their final normalized target path.
    """
    image_groups: dict[Path, list[Path]] = defaultdict(list)
    metrics_groups: dict[Path, list[Path]] = defaultdict(list)

    for source in root.rglob("*"):
        if not source.is_file():
            continue

        normalized_name = normalize_downloaded_filename(source.name)

        # Already correctly named files are existing target files.
        if normalized_name == source.name:
            continue

        target = source.with_name(normalized_name)
        suffix = source.suffix.lower()

        if suffix == ".png":
            image_groups[target].append(source)

        elif (
            suffix == ".txt"
            and normalized_name.lower().startswith(METRICS_TABLE_PREFIX)
        ):
            metrics_groups[target].append(source)

    return image_groups, metrics_groups


def newest_file(files: list[Path]) -> Path:
    """Return the most recently modified file."""
    return max(files, key=lambda path: path.stat().st_mtime_ns)


def process_image_files(
    image_groups: dict[Path, list[Path]],
    apply_changes: bool,
    counters: dict[str, int],
) -> None:
    """Replace image files while preserving trajectory plots."""

    for target, downloaded_files in sorted(
        image_groups.items(),
        key=lambda item: str(item[0]),
    ):
        if is_protected_trajectory(target.name):
            if target.exists():
                print("\n[KEEP EXISTING TRAJECTORY]")
                print(f"  Kept: {target}")

                for downloaded in downloaded_files:
                    print(f"  Remove downloaded copy: {downloaded}")

                    if apply_changes and downloaded.exists():
                        downloaded.unlink()

                    counters["trajectory_downloads_removed"] += 1

                counters["trajectories_kept"] += 1

            else:
                print("\n[WARNING: TRAJECTORY TARGET NOT FOUND]")
                print(f"  Expected existing file: {target}")
                print("  Downloaded trajectory file was left unchanged.")

                for downloaded in downloaded_files:
                    print(f"  Downloaded file: {downloaded}")

                counters["warnings"] += 1

            continue

        # When multiple downloads exist, use the newest one.
        selected_file = newest_file(downloaded_files)

        if target.exists():
            print("\n[REPLACE IMAGE]")
            print(f"  Old: {target}")
            print(f"  New: {selected_file}")
            print(f"  Final filename: {target.name}")

            counters["images_replaced"] += 1
        else:
            print("\n[RENAME IMAGE]")
            print(f"  From: {selected_file}")
            print(f"  To:   {target}")

            counters["images_renamed"] += 1

        if apply_changes:
            # Replaces the old file or renames the new file when no target exists.
            os.replace(selected_file, target)

        # Remove additional downloads such as (1), (2), etc.
        for extra_file in downloaded_files:
            if extra_file == selected_file:
                continue

            print(f"  Remove extra duplicate: {extra_file}")

            if apply_changes and extra_file.exists():
                extra_file.unlink()

            counters["image_duplicates_removed"] += 1


def process_metrics_tables(
    metrics_groups: dict[Path, list[Path]],
    apply_changes: bool,
    counters: dict[str, int],
) -> None:
    """Replace existing metrics tables with newly downloaded versions."""

    for target, downloaded_files in sorted(
        metrics_groups.items(),
        key=lambda item: str(item[0]),
    ):
        selected_file = newest_file(downloaded_files)

        if target.exists():
            print("\n[REPLACE METRICS TABLE]")
            print(f"  Old: {target}")
            print(f"  New: {selected_file}")
            print(f"  Final filename: {target.name}")

            counters["metrics_replaced"] += 1
        else:
            print("\n[RENAME METRICS TABLE]")
            print(f"  From: {selected_file}")
            print(f"  To:   {target}")

            counters["metrics_renamed"] += 1

        if apply_changes:
            # Removes/replaces the old table and gives the new table its name.
            os.replace(selected_file, target)

        # Remove additional copies such as (2), (3), etc.
        for extra_file in downloaded_files:
            if extra_file == selected_file:
                continue

            print(f"  Remove extra duplicate: {extra_file}")

            if apply_changes and extra_file.exists():
                extra_file.unlink()

            counters["metrics_duplicates_removed"] += 1


def print_summary(
    root: Path,
    apply_changes: bool,
    counters: dict[str, int],
) -> None:
    """Print the final operation summary."""
    print("\n" + "=" * 70)

    if apply_changes:
        print("CHANGES APPLIED")
    else:
        print("DRY RUN: NO FILES WERE CHANGED")

    print(f"Root directory:                 {root}")
    print(f"Images replaced:               {counters['images_replaced']}")
    print(f"Images renamed:                {counters['images_renamed']}")
    print(
        "Extra image duplicates removed: "
        f"{counters['image_duplicates_removed']}"
    )
    print(f"Existing trajectories kept:    {counters['trajectories_kept']}")
    print(
        "Downloaded trajectories removed: "
        f"{counters['trajectory_downloads_removed']}"
    )
    print(f"Metrics tables replaced:       {counters['metrics_replaced']}")
    print(f"Metrics tables renamed:        {counters['metrics_renamed']}")
    print(
        "Extra metrics duplicates removed: "
        f"{counters['metrics_duplicates_removed']}"
    )
    print(f"Warnings:                      {counters['warnings']}")

    if not apply_changes:
        print("\nRun again with --apply after checking the output.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace downloaded result plots and metrics tables while "
            "preserving existing trajectory plots."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this option, perform only a dry run.",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Root results directory. By default, the directory containing "
            "this script is used."
        ),
    )

    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    counters = {
        "images_replaced": 0,
        "images_renamed": 0,
        "image_duplicates_removed": 0,
        "trajectories_kept": 0,
        "trajectory_downloads_removed": 0,
        "metrics_replaced": 0,
        "metrics_renamed": 0,
        "metrics_duplicates_removed": 0,
        "warnings": 0,
    }

    image_groups, metrics_groups = collect_files(root)

    process_image_files(
        image_groups=image_groups,
        apply_changes=args.apply,
        counters=counters,
    )

    process_metrics_tables(
        metrics_groups=metrics_groups,
        apply_changes=args.apply,
        counters=counters,
    )

    print_summary(
        root=root,
        apply_changes=args.apply,
        counters=counters,
    )


if __name__ == "__main__":
    main()

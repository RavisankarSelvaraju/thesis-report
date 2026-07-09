#!/usr/bin/env python3

from pathlib import Path

ROOT_DIR = Path(".")   # change this if needed, e.g. Path("/path/to/results")
DRY_RUN = False        # set True to preview without renaming

for png_file in ROOT_DIR.rglob("*.png"):
    if " " not in png_file.name:
        continue

    new_name = png_file.name.replace(" ", "_")
    new_path = png_file.with_name(new_name)

    if new_path.exists():
        print(f"[SKIP] Target already exists: {new_path}")
        continue

    print(f"[RENAME] {png_file} -> {new_path}")

    if not DRY_RUN:
        png_file.rename(new_path)

print("Done.")

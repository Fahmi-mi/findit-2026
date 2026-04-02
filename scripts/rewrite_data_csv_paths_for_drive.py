from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "data"
DST_DIR = PROJECT_ROOT / "data_drive"

# Target path used in Colab notebook
DRIVE_ROOT = "/content/drive/MyDrive/ML/Findit-2026"
# Local root used when converting back from Drive paths
LOCAL_ROOT = r"c:\Users\Axioo\Documents\Fahmi\ai\ml\findit-2026"

# Matches local absolute paths that contain ...\findit-2026\...
LOCAL_PATH_RE = re.compile(r"(?i)^[a-z]:\\.*?\\findit-2026\\(.+)$")
DRIVE_PATH_RE = re.compile(r"^/content/drive/MyDrive/ML/Findit-2026/(.+)$")


def convert_to_drive_value(value: object) -> object:
    if not isinstance(value, str):
        return value

    s = value.strip()
    m = LOCAL_PATH_RE.match(s)
    if not m:
        return value

    suffix = m.group(1).replace("\\", "/")
    return f"{DRIVE_ROOT}/{suffix}"


def convert_to_local_value(value: object) -> object:
    if not isinstance(value, str):
        return value

    s = value.strip()
    m = DRIVE_PATH_RE.match(s)
    if not m:
        return value

    suffix = m.group(1).replace("/", "\\")
    return f"{LOCAL_ROOT}\\{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite CSV file_path values between local Windows and Colab Drive paths."
    )
    parser.add_argument(
        "--mode",
        choices=["to_drive", "to_local"],
        default="to_drive",
        help="to_drive: local -> drive, to_local: drive -> local",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "to_drive":
        converter = convert_to_drive_value
    else:
        converter = convert_to_local_value

    csv_files = sorted(SRC_DIR.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found under data/")
        return

    converted_files = 0
    converted_cells = 0

    for src_path in csv_files:
        rel = src_path.relative_to(SRC_DIR)
        dst_path = DST_DIR / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(src_path)
        before = df.astype(str)
        df = df.apply(lambda col: col.map(converter))
        after = df.astype(str)

        changed = (before != after).sum().sum()
        if changed > 0:
            converted_files += 1
            converted_cells += int(changed)

        df.to_csv(dst_path, index=False)

    print(f"Mode: {args.mode}")
    print(f"Scanned CSV files: {len(csv_files)}")
    print(f"Files with converted paths: {converted_files}")
    print(f"Converted cells: {converted_cells}")
    print(f"Output folder: {DST_DIR}")


if __name__ == "__main__":
    main()

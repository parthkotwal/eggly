"""CLI to generate synthetic candidate profiles for the matcher."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

if __package__:
    from .service import FEATURE_COLUMNS
    from .synthetic import generate_users
else:  # pragma: no cover - direct execution
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from ml.service import FEATURE_COLUMNS  # type: ignore[import-not-found]
    from ml.synthetic import generate_users  # type: ignore[import-not-found]

DEFAULT_COUNT = 200
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "_artifacts" / "candidates.pkl"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic candidate dataset.")
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of synthetic candidates to generate (default: {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write pickle dataset (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9,
        help="Random seed for reproducibility (default: 9).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    df = generate_users(args.count, seed=args.seed)

    # Ensure all expected feature columns exist; fill missing with zeros
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[["user_id", "age", "vegan", "region"] + FEATURE_COLUMNS]

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)

    print(f"Wrote {len(df)} synthetic candidates to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

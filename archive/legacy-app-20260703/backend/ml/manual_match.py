"""Utility script to exercise the matcher against the seeded candidate set.

Run either as a module (`python -m ml.manual_match`) or directly
(`python backend/ml/manual_match.py`). You can supply a JSON profile and/or
override the candidate dataset path via CLI flags.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_DATASET = PACKAGE_DIR / "_artifacts" / "candidates.pkl"
DEFAULT_PROFILE: Dict[str, Any] = {
    "vegan": True,
    "region": "SouthAsia",
    "age": 42,
    "cuisines": {"Indian": 0.8, "Japanese": 0.2},
    "flavors": {"spicy": 0.9, "umami": 0.7},
    "textures": {"tender": 0.7, "crispy": 0.3},
}


def _bootstrap_import() -> None:
    """Ensure package imports work when script is executed directly."""
    if __package__:
        return
    sys.path.append(str(PROJECT_ROOT))


def _resolve_dataset_path(explicit: Optional[Path]) -> Path:
    """Pick a candidate dataset path, favoring CLI then env then default."""
    if explicit:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Candidate dataset not found: {path}")
        os.environ["EGGLY_CANDIDATE_DATA"] = str(path)
        return path

    env_path = os.getenv("EGGLY_CANDIDATE_DATA")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path

    if DEFAULT_DATASET.exists():
        os.environ.setdefault("EGGLY_CANDIDATE_DATA", str(DEFAULT_DATASET))
        return DEFAULT_DATASET

    raise FileNotFoundError(
        "Candidate dataset missing. Supply --data or set EGGLY_CANDIDATE_DATA."
    )


def _load_profile(path: Optional[Path]) -> Dict[str, Any]:
    """Load a profile from disk or return the default example."""
    if not path:
        return DEFAULT_PROFILE.copy()
    with path.expanduser().open() as fh:
        return json.load(fh)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual matcher smoke test.")
    parser.add_argument(
        "--profile",
        type=Path,
        help="Path to a JSON file describing the query profile.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to a pickle candidate dataset. Overrides EGGLY_CANDIDATE_DATA.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of matches to return (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    _bootstrap_import()
    if __package__:
        from .service import match  # type: ignore[attr-defined]
    else:  # pragma: no cover - runtime fallback
        from service import match  # type: ignore[import-not-found]

    args = parse_args(argv)
    dataset_path = _resolve_dataset_path(args.data)
    profile = _load_profile(args.profile)

    # Allow profile JSON to include top_k override.
    top_k = args.top_k or profile.get("top_k", 5)
    profile.pop("top_k", None)

    matches = match(profile, top_k=top_k)
    payload = {
        "dataset": str(dataset_path),
        "profile": profile,
        "top_k": top_k,
        "matches": matches,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

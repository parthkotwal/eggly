import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .config import CUISINES, FLAVORS, PRIORS, TEXTURES
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import CUISINES, FLAVORS, PRIORS, TEXTURES


FEATURE_PREFIX_WEIGHTS: Dict[str, float] = {
    "cui_": 1.0,
    "flv_": 0.7,
    "tex_": 0.4,
}

FEATURE_COLUMNS: List[str] = (
    [f"cui_{name}" for name in CUISINES]
    + [f"flv_{name}" for name in FLAVORS]
    + [f"tex_{name}" for name in TEXTURES]
)

#-----PRIVATE METHODS-----
def _build_feature_weights() -> np.ndarray:
    weights: List[float] = []
    for col in FEATURE_COLUMNS:
        for prefix, wt in FEATURE_PREFIX_WEIGHTS.items():
            if col.startswith(prefix):
                weights.append(wt)
                break
        else:
            weights.append(1.0)
    return np.array(weights, dtype=float)


FEATURE_WEIGHTS = _build_feature_weights()


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _ensure_map(keys: Sequence[str], provided: Any) -> Dict[str, float]:
    if provided is None:
        provided = {}
    data: Dict[str, float] = {}
    if isinstance(provided, dict):
        for key in keys:
            data[key] = float(provided.get(key, 0.0))
    elif isinstance(provided, Iterable) and not isinstance(provided, (str, bytes)):
        provided_set = {str(item) for item in provided}
        for key in keys:
            data[key] = 1.0 if key in provided_set else 0.0
    else:
        for key in keys:
            data[key] = float(provided if provided else 0.0)
    return data


def _apply_additive(base: Dict[str, float], adjustments: Dict[str, float], scale: float = 1.0) -> None:
    for key, delta in adjustments.items():
        if key in base:
            base[key] += scale * float(delta)


def _soft_merge(base: Dict[str, float], blend: Dict[str, float], alpha: float = 0.3) -> Dict[str, float]:
    merged = base.copy()
    for key, value in blend.items():
        merged[key] = merged.get(key, 0.0) * (1 - alpha) + float(value) * alpha
    return merged


def _clip01(values: Dict[str, float]) -> Dict[str, float]:
    return {key: float(np.clip(val, 0.0, 1.0)) for key, val in values.items()}


def _normalize(values: Dict[str, float]) -> Dict[str, float]:
    arr = np.array(list(values.values()), dtype=float)
    if arr.size == 0:
        return {}
    max_val = arr.max()
    min_val = arr.min()
    if max_val <= 1.0 and min_val >= 0.0:
        return values
    if max_val - min_val < 1e-9:
        return {k: float(np.clip(v, 0.0, 1.0)) for k, v in values.items()}
    return {k: (float(v) - min_val) / (max_val - min_val) for k, v in values.items()}


def _normalize_if_needed(values: Dict[str, float]) -> Dict[str, float]:
    return _clip01(_normalize(values))


def _feature_vector_from_blocks(cuisine: Dict[str, float], flavor: Dict[str, float], texture: Dict[str, float]) -> np.ndarray:
    feat_map = {
        **{f"cui_{k}": cuisine.get(k, 0.0) for k in CUISINES},
        **{f"flv_{k}": flavor.get(k, 0.0) for k in FLAVORS},
        **{f"tex_{k}": texture.get(k, 0.0) for k in TEXTURES},
    }
    return np.array([feat_map[col] for col in FEATURE_COLUMNS], dtype=float)


def _load_candidate_frame() -> pd.DataFrame:
    env_path = os.getenv("EGGLY_CANDIDATE_DATA")
    candidate_paths: List[Path] = []
    if env_path:
        candidate_paths.append(Path(env_path))
    base_dir = Path(__file__).resolve().parent
    candidate_paths.append(base_dir / "candidates.pkl")
    for path in candidate_paths:
        if not path.exists():
            continue
        if path.suffix.lower() not in {".pkl", ".pickle"}:
            continue
        return pd.read_pickle(path)
    raise FileNotFoundError(
        "Candidate data not found. Provide a pickle dataset via EGGLY_CANDIDATE_DATA "
        "or place a candidates.pkl file next to service.py."
    )

#-----PUBLIC METHODS-----

def build_feature_vector(profile: dict) -> np.ndarray:
    cuisine = _ensure_map(CUISINES, profile.get("cuisine_preferences") or profile.get("cuisines"))
    flavor = _ensure_map(FLAVORS, profile.get("flavor_preferences") or profile.get("flavors"))
    texture = _ensure_map(TEXTURES, profile.get("texture_preferences") or profile.get("textures"))

    region = profile.get("region")
    if region in PRIORS["region_bias"]:
        _apply_additive(cuisine, PRIORS["region_bias"][region])

    vegan = profile.get("vegan")
    if vegan is not None and _as_bool(vegan):
        _apply_additive(cuisine, PRIORS["vegan_cuisine_boost"])
        _apply_additive(flavor, PRIORS["vegan_flavor_adjust"])

    dietary = profile.get("dietary_preferences") or profile.get("dietary_focus") or []
    if isinstance(dietary, str):
        dietary = [dietary]
    dietary_lower = {str(item).lower() for item in dietary}
    if profile.get("health_focus") or "health" in dietary_lower or "healthy" in dietary_lower:
        _apply_additive(flavor, PRIORS["health_focus_flavor_adjust"])
    if profile.get("low_carb") or "low_carb" in dietary_lower or "lowcarb" in dietary_lower:
        _apply_additive(cuisine, PRIORS["low_carb_cuisine_boost"])

    age = profile.get("age")
    if isinstance(age, (int, float)):
        if age > 50 and "age>50_texture_adjust" in PRIORS:
            _apply_additive(texture, PRIORS["age>50_texture_adjust"])
            _apply_additive(flavor, PRIORS["age>50_flavor_adjust"])
        elif age <= 25 and "age<25_flavor_adjust" in PRIORS:
            _apply_additive(flavor, PRIORS["age<25_flavor_adjust"])

    implied_flavors: Dict[str, float] = {}
    for cuisine_name, weight in cuisine.items():
        profile_map = PRIORS["cuisine_flavor_profile"].get(cuisine_name)
        if not profile_map:
            continue
        scaled = {k: v * weight for k, v in profile_map.items()}
        implied_flavors = _soft_merge(implied_flavors, scaled, alpha=0.5)
    flavor = _soft_merge(flavor, implied_flavors, alpha=0.35)

    implied_textures: Dict[str, float] = {}
    for cuisine_name, weight in cuisine.items():
        profile_map = PRIORS["cuisine_texture_profile"].get(cuisine_name)
        if not profile_map:
            continue
        scaled = {k: v * weight for k, v in profile_map.items()}
        implied_textures = _soft_merge(implied_textures, scaled, alpha=0.5)
    texture = _soft_merge(texture, implied_textures, alpha=0.30)

    for base_flavor, correlated in PRIORS["flavor_cooccurrence"].items():
        base_val = flavor.get(base_flavor, 0.0)
        if base_val <= 0.5:
            continue
        strength = (base_val - 0.5) * 2.0
        for target, boost in correlated.items():
            if target in flavor:
                flavor[target] += boost * strength * 0.3

    for base_texture, correlated in PRIORS["texture_cooccurrence"].items():
        base_val = texture.get(base_texture, 0.0)
        if base_val <= 0.5:
            continue
        strength = (base_val - 0.5) * 2.0
        for target, boost in correlated.items():
            if target in texture:
                texture[target] += boost * strength * 0.3

    for base_flavor, correlated in PRIORS["flavor_cooccurrence"].items():
        base_val = flavor.get(base_flavor, 0.0)
        if base_val <= 0.5:
            continue
        strength = (base_val - 0.5) * 2.0
        for target, boost in correlated.items():
            if target in texture:
                texture[target] += boost * strength * 0.25

    cuisine = _normalize_if_needed(cuisine)
    flavor = _normalize_if_needed(flavor)
    texture = _normalize_if_needed(texture)

    return _feature_vector_from_blocks(cuisine, flavor, texture)


def load_candidate_matrix() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    path = os.getenv("EGGLY_CANDIDATE_DATA")
    df = _load_candidate_frame().copy()
    feature_cols = FEATURE_COLUMNS
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    X = df[feature_cols].to_numpy(dtype=float)

    return X, FEATURE_WEIGHTS.copy(), df


def similarity_matrix(X_query: np.ndarray, X_pool: np.ndarray) -> np.ndarray:
    if X_pool.size == 0:
        return np.zeros((1, 0), dtype=float)
    query = np.atleast_2d(X_query.astype(float))
    weighted_query = query * FEATURE_WEIGHTS
    weighted_pool = X_pool * FEATURE_WEIGHTS
    sims = cosine_similarity(weighted_query, weighted_pool)
    return np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)


def apply_constraints(df: pd.DataFrame, sims: np.ndarray, query: dict) -> np.ndarray:
    scores = np.array(sims, dtype=float).reshape(-1)
    if scores.size != len(df):
        raise ValueError("Similarity scores and candidate dataframe length mismatch.")

    penalty = np.ones_like(scores)

    vegan_pref = query.get("vegan")
    if vegan_pref is not None and "vegan" in df.columns:
        vegan_bool = df["vegan"].astype(bool).to_numpy()
        if _as_bool(vegan_pref):
            penalty = np.where(vegan_bool, penalty, penalty * 0.35)
        else:
            penalty = np.where(vegan_bool, penalty * 0.85, penalty)

    region_pref = query.get("region")
    if region_pref and "region" in df.columns:
        penalty = np.where(df["region"] == region_pref, penalty, penalty * 0.9)

    exclude_ids = query.get("exclude_ids") or []
    if exclude_ids:
        exclude_set = {str(item) for item in exclude_ids}
        id_col = next((col for col in ("id", "candidate_id", "profile_id", "user_id") if col in df.columns), None)
        if id_col:
            id_values = df[id_col].astype(str)
            mask = id_values.isin(exclude_set)
            penalty = np.where(mask, 0.0, penalty)

    return scores * penalty


def explain_top_features(query_vec: np.ndarray, cand_vec: np.ndarray, feature_names: Sequence[str], top_n: int = 5) -> List[str]:
    diffs = np.square(query_vec - cand_vec)
    order = np.argsort(diffs)
    msgs = []
    for idx in order:
        q = float(query_vec[idx]); c = float(cand_vec[idx])
        if q < 0.35 and c < 0.35:  # shared meh: skip
            continue
        feat = feature_names[idx]; prefix, label = feat[:4], feat[4:].replace("_", " ").title()
        if prefix == "cui_": kind = f"{label} cuisine"
        elif prefix == "flv_": kind = f"{label.lower()} flavor"
        elif prefix == "tex_": kind = f"{label.lower()} texture"
        else: kind = label

        if q >= 0.6 and c >= 0.6:
            msg = f"Both strongly prefer {kind} ({q:.2f} & {c:.2f})."
        elif q <= 0.2 and c <= 0.2:
            msg = f"Both avoid {kind} ({q:.2f} & {c:.2f})."
        else:
            msg = f"Close on {kind} ({q:.2f} vs {c:.2f})."
        msgs.append(msg)
        if len(msgs) >= top_n: break

    if msgs: return msgs
    # fallback
    top_idx = np.argsort(-cand_vec)[:top_n]
    return [f"Candidate is strong on {feature_names[i][4:].replace('_',' ').title()} ({cand_vec[i]:.2f})." for i in top_idx]


def _extract_metadata(row: pd.Series) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key, value in row.items():
        if key in FEATURE_COLUMNS:
            continue
        if isinstance(value, (np.generic,)):
            metadata[key] = value.item()
        else:
            metadata[key] = value
    return metadata


def match(profile: dict, top_k: int = 5) -> List[dict]:
    X_pool, _, candidates = load_candidate_matrix()
    if candidates.empty:
        return []

    query_vec = build_feature_vector(profile)
    sims = similarity_matrix(query_vec, X_pool)
    scores = apply_constraints(candidates, sims, profile)
    if not np.any(scores):
        return []

    top_indices = np.argsort(-scores)[:top_k]
    feature_names = FEATURE_COLUMNS
    results: List[dict] = []
    for idx in top_indices:
        candidate_row = candidates.iloc[idx]
        explanation = explain_top_features(query_vec, X_pool[idx], feature_names)
        result = {
            "score": float(scores[idx]),
            "explanation": explanation,
            "metadata": _extract_metadata(candidate_row),
        }
        id_col = next((col for col in ("id", "candidate_id", "profile_id", "user_id") if col in candidate_row.index), None)
        if id_col:
            candidate_id = candidate_row[id_col]
            if hasattr(candidate_id, "item"):
                candidate_id = candidate_id.item()
            result["id"] = candidate_id
        results.append(result)
    return results


__all__ = [
    "build_feature_vector",
    "load_candidate_matrix",
    "similarity_matrix",
    "apply_constraints",
    "explain_top_features",
    "match",
]

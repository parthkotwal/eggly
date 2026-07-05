import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import CUISINES, FLAVORS, PRIORS, REGIONS, TEXTURES


RANDOM_NOISE = 0.12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clip01(d: Dict[str, float]) -> Dict[str, float]:
    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in d.items()}


def init_uniform(keys: List[str]) -> Dict[str, float]:
    return {k: np.random.rand() for k in keys}


def apply_additive(d: Dict[str, float], adj: Dict[str, float], scale: float = 1.0) -> None:
    for k, delta in adj.items():
        if k in d:
            d[k] += scale * delta


def normalize_dict(d: Dict[str, float]) -> Dict[str, float]:
    values = np.array(list(d.values()))
    if values.size == 0:
        return {}
    if np.ptp(values) < 1e-9:
        return d
    mn, mx = values.min(), values.max()
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


def jitter_dict(d: Dict[str, float], std: float = RANDOM_NOISE) -> Dict[str, float]:
    return {k: v + np.random.normal(0, std) for k, v in d.items()}


def soft_merge(base: Dict[str, float], blend: Dict[str, float], alpha: float = 0.3) -> Dict[str, float]:
    result = base.copy()
    for k, v in blend.items():
        result[k] = result.get(k, 0.0) * (1 - alpha) + v * alpha
    return result


@dataclass
class SyntheticUser:
    user_id: int
    age: int
    vegan: int
    region: str
    cuisine_prefs: Dict[str, float]
    flavor_prefs: Dict[str, float]
    texture_prefs: Dict[str, float]


def generate_user(user_id: int) -> SyntheticUser:
    age = int(np.random.randint(18, 71))
    vegan = np.random.rand() < 0.18
    region = random.choice(REGIONS)

    cuisine = init_uniform(CUISINES)
    flavor = init_uniform(FLAVORS)
    texture = init_uniform(TEXTURES)

    apply_additive(cuisine, PRIORS["region_bias"].get(region, {}))

    if vegan:
        apply_additive(cuisine, PRIORS["vegan_cuisine_boost"])
        apply_additive(flavor, PRIORS["vegan_flavor_adjust"])

    if np.random.rand() < 0.20:
        apply_additive(flavor, PRIORS["health_focus_flavor_adjust"])

    if np.random.rand() < 0.15:
        apply_additive(cuisine, PRIORS["low_carb_cuisine_boost"])

    if age > 50:
        apply_additive(texture, PRIORS["age>50_texture_adjust"])
        apply_additive(flavor, PRIORS["age>50_flavor_adjust"])
    elif age <= 25:
        apply_additive(flavor, PRIORS["age<25_flavor_adjust"])

    implied_flavors: Dict[str, float] = {}
    for cuisine_name, weight in cuisine.items():
        profile = PRIORS["cuisine_flavor_profile"].get(cuisine_name)
        if not profile:
            continue
        scaled = {k: v * weight for k, v in profile.items()}
        implied_flavors = soft_merge(implied_flavors, scaled, alpha=0.5)
    flavor = soft_merge(flavor, implied_flavors, alpha=0.35)

    implied_textures: Dict[str, float] = {}
    for cuisine_name, weight in cuisine.items():
        profile = PRIORS["cuisine_texture_profile"].get(cuisine_name)
        if not profile:
            continue
        scaled = {k: v * weight for k, v in profile.items()}
        implied_textures = soft_merge(implied_textures, scaled, alpha=0.5)
    texture = soft_merge(texture, implied_textures, alpha=0.30)

    for base_flavor, correlated in PRIORS["flavor_cooccurrence"].items():
        base_val = flavor.get(base_flavor, 0.0)
        if base_val <= 0.5:
            continue
        strength = (base_val - 0.5) * 2
        for target, boost in correlated.items():
            if target in flavor:
                flavor[target] += boost * strength * 0.3
            if target in texture:
                texture[target] += boost * strength * 0.25

    for base_texture, correlated in PRIORS["texture_cooccurrence"].items():
        base_val = texture.get(base_texture, 0.0)
        if base_val <= 0.5:
            continue
        strength = (base_val - 0.5) * 2
        for target, boost in correlated.items():
            if target in texture:
                texture[target] += boost * strength * 0.3

    cuisine = normalize_dict(clip01(jitter_dict(cuisine)))
    flavor = normalize_dict(clip01(jitter_dict(flavor)))
    texture = normalize_dict(clip01(jitter_dict(texture)))

    return SyntheticUser(
        user_id=user_id,
        age=age,
        vegan=int(vegan),
        region=region,
        cuisine_prefs=cuisine,
        flavor_prefs=flavor,
        texture_prefs=texture,
    )


def generate_users(n: int, seed: int = 9) -> pd.DataFrame:
    set_seed(seed)
    users = [generate_user(i) for i in range(n)]
    records = []
    for user in users:
        row = {
            "user_id": user.user_id,
            "age": user.age,
            "vegan": user.vegan,
            "region": user.region,
        }
        row.update({f"cui_{k}": v for k, v in user.cuisine_prefs.items()})
        row.update({f"flv_{k}": v for k, v in user.flavor_prefs.items()})
        row.update({f"tex_{k}": v for k, v in user.texture_prefs.items()})
        records.append(row)
    return pd.DataFrame(records)


__all__ = ["generate_users", "generate_user", "set_seed"]

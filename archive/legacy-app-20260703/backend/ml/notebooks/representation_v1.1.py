import math, random, json

from dataclasses import dataclass

from typing import Dict, List, Tuple

import numpy as np

import pandas as pd



from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import PCA



from umap import UMAP

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import plotly.express as px

import plotly.graph_objects as go



np.set_printoptions(suppress=True, precision=3)

pd.set_option("display.max_columns", 200)

# domain vocab + priors



CUISINES = [

    "Italian","Mexican","Japanese","Indian","Greek","American","Thai","Chinese",

    "Mediterranean","Korean","French","Vietnamese","Lebanese","Turkish",

    "Spanish","Moroccan","Ethiopian","Brazilian","Nordic","Caribbean"

]

FLAVORS = ["sweet","spicy","savory","sour","bitter","umami","salty","fresh","acidic","smoky","herbal","earthy"]

TEXTURES = [

    "crispy","chewy","creamy","crunchy","silky","tender","flaky",

    "juicy","firm","smooth","gooey","fatty","fibrous","brittle"

]



REGIONS = ["USA","Europe","SouthAsia","EastAsia","MiddleEast","LatAm"]



# Priors encode correlations you believe are plausible

PRIORS = {



    # -------------------------------

    # 1️⃣ Lifestyle / dietary correlations

    # -------------------------------

    "vegan_cuisine_boost": {

        "Indian": 0.25, "Mediterranean": 0.25, "Thai": 0.2,

        "Vietnamese": 0.2, "Ethiopian": 0.20, "Nordic": 0.10,

        "Lebanese": 0.25, "Turkish": 0.15

    },



    "vegan_flavor_adjust": {

        "umami": -0.10, "savory": +0.05, "spicy": +0.05,

        "fresh": +0.10, "earthy": +0.10

    },



    "low_carb_cuisine_boost": {

        "Mediterranean": 0.25, "Nordic": 0.20, "Japanese": 0.15, "American": 0.10

    },



    "health_focus_flavor_adjust": {

        "sweet": -0.10, "fresh": +0.15, "herbal": +0.10, "acidic": +0.05

    },



    # -------------------------------

    # 2️⃣ Age-based perceptual shifts

    # -------------------------------

    "age>50_texture_adjust": {

        "crispy": -0.15, "tender": +0.10, "creamy": +0.05, "smooth": +0.05, "chewy": -0.10

    },



    "age>50_flavor_adjust": {

        "savory": +0.10, "sweet": -0.05, "spicy": -0.05, "umami": +0.05, "bitter": -0.05

    },



    "age<25_flavor_adjust": {

        "spicy": +0.10, "sweet": +0.10, "bitter": -0.05, "earthy": -0.05, "salty": +0.10

    },



    # -------------------------------

    # 3️⃣ Region preference biases

    # -------------------------------

    "region_bias": {

        "USA": {"American": +0.20, "Mexican": +0.15, "Italian": +0.10, "Japanese": +0.05, "Chinese": +0.10},

        "Europe": {"Italian": +0.20, "French": +0.15, "Mediterranean": +0.15, "Nordic": +0.10, "Greek": +0.10, "Turkish": +0.05},

        "SouthAsia": {"Indian": +0.30, "Thai": +0.10, "Chinese": +0.05},

        "EastAsia": {"Japanese": +0.15, "Korean": +0.15, "Chinese": +0.20, "Vietnamese": +0.10},

        "MiddleEast": {"Lebanese": +0.25, "Turkish": +0.25, "Moroccan": +0.15, "Mediterranean": +0.10},

        "LatAm": {"Mexican": +0.30, "Brazilian": +0.20, "Caribbean": +0.15}

    },



    # -------------------------------

    # 4️⃣ Cuisines' internal flavor priors

    # -------------------------------

    "cuisine_flavor_profile": {

        "Indian": {"spicy": .85, "savory": .75, "sweet": .30, "umami": .50, "earthy": .45},

        "Mediterranean": {"savory": .75, "fresh": .55, "umami": .45, "sour": .35, "herbal": .55},

        "Italian": {"savory": .80, "umami": .65, "sweet": .45, "acidic": .35},

        "Mexican": {"spicy": .75, "savory": .70, "sour": .40, "smoky": .40, "sweet": .35},

        "Japanese": {"umami": .85, "savory": .70, "sweet": .30, "fresh": .45, "acidic": .25},

        "Thai": {"spicy": .80, "sweet": .60, "sour": .60, "savory": .60, "fresh": .40},

        "Chinese": {"umami": .70, "savory": .70, "sweet": .40, "spicy": .45},

        "Korean": {"spicy": .65, "umami": .65, "savory": .70, "sweet": .35},

        "French": {"savory": .75, "umami": .50, "sweet": .50, "buttery": .35},

        "Vietnamese": {"sour": .55, "savory": .65, "sweet": .45, "fresh": .60, "herbal": .55},

        "American": {"savory": .60, "sweet": .55, "smoky": .45, "salty": .55},

        "Lebanese": {"savory": .65, "fresh": .55, "herbal": .60, "acidic": .40, "spicy": .30},

        "Turkish": {"savory": .70, "sweet": .45, "spicy": .45, "umami": .55},

        "Moroccan": {"spicy": .55, "sweet": .45, "earthy": .60, "savory": .65, "herbal": .35},

        "Ethiopian": {"spicy": .80, "earthy": .60, "savory": .70},

        "Brazilian": {"savory": .65, "smoky": .55, "sweet": .40, "spicy": .35},

        "Nordic": {"fresh": .65, "savory": .50, "acidic": .45, "earthy": .35},

        "Caribbean": {"spicy": .70, "sweet": .60, "smoky": .50, "fresh": .40},

        "Greek": {"savory": .70, "fresh": .60, "herbal": .55, "acidic": .45, "sweet": .35},

        "Spanish": {"savory": .70, "smoky": .50, "sweet": .40, "fresh": .45, "spicy": .35}

    },



    

    # -------------------------------

    # 5️⃣ Cuisine → texture tendencies

    # -------------------------------

    "cuisine_texture_profile": {

        "Japanese": {"tender": .55, "silky": .50, "smooth": .45},

        "Indian": {"creamy": .55, "tender": .50, "juicy": .45},

        "French": {"creamy": .70, "silky": .60, "tender": .55},

        "American": {"crispy": .60, "chewy": .50, "juicy": .45, "smooth": .40},

        "Mexican": {"crispy": .55, "chewy": .55, "juicy": .45, "tender": .40},

        "Thai": {"tender": .55, "juicy": .50, "crunchy": .45},

        "Mediterranean": {"smooth": .50, "tender": .50, "flaky": .40},

        "Korean": {"chewy": .60, "crispy": .55, "juicy": .45},

        "Chinese": {"chewy": .55, "smooth": .50, "juicy": .45},

        "Nordic": {"firm": .55, "tender": .50, "smooth": .45},

        "Greek": {"flaky": .50, "tender": .50, "creamy": .45},

        "Spanish": {"crispy": .55, "tender": .50, "juicy": .45},

        "Lebanese": {"tender": .50, "flaky": .45, "creamy": .40},

        "Turkish": {"tender": .50, "flaky": .45, "smooth": .40},

        "Moroccan": {"tender": .50, "juicy": .45, "fibrous": .35},

        "Ethiopian": {"smooth": .50, "creamy": .45, "tender": .45},

        "Brazilian": {"juicy": .55, "chewy": .50, "crispy": .45},

        "Caribbean": {"juicy": .55, "crispy": .50, "tender": .45}

    },





    # -------------------------------

    # 6️⃣ Cross-flavor and texture correlations

    # -------------------------------

    "flavor_cooccurrence": {

        "spicy": {"savory": +0.35, "umami": +0.25, "sweet": +0.10},

        "sweet": {"creamy": +0.25, "smooth": +0.15, "spicy": +0.10, "smoky": +0.15},

        "umami": {"creamy": +0.20, "savory": +0.40},

        "fresh": {"herbal": +0.40, "acidic": +0.30, "sweet": +0.05},

        "smoky": {"savory": +0.35, "juicy": +0.25, "sweet": +0.15}

    },



    "texture_cooccurrence": {

        "crispy": {"crunchy": +0.40, "chewy": -0.20},

        "creamy": {"smooth": +0.35, "fatty": +0.20},

        "tender": {"juicy": +0.30, "flaky": +0.20},

        "chewy": {"firm": +0.25, "crispy": -0.15}

    }

}





RANDOM_NOISE = 0.12  # how “messy” individuals are around the priors (0–0.3 is a nice range)
# Helpers



def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    

def clip01(d):

    return {k: float(np.clip(v, 0.0, 1.0)) for k,v in d.items()}



def init_uniform(keys):

    return {k: np.random.rand() for k in keys}



def apply_additive(d, adj: Dict[str, float], scale=1.0):

    for k,delta in adj.items():

        if k in d:

            d[k] += scale * delta



def normalize_dict(d):

    # min-max per dict; if flat, leave as-is

    vals = np.array(list(d.values()))

    if np.ptp(vals) < 1e-9: return d

    mn, mx = vals.min(), vals.max()

    return {k: (v - mn) / (mx - mn) for k,v in d.items()}



def jitter_dict(d, std=RANDOM_NOISE):

    return {k: v + np.random.normal(0, std) for k,v in d.items()}



def soft_merge(base: Dict[str,float], blend: Dict[str,float], alpha=0.3):

    out = base.copy()

    for k,v in blend.items():

        out[k] = out.get(k, 0.0) * (1-alpha) + v * alpha

    return out

    
# User generation

@dataclass

class User:

    user_id: int

    age: int

    vegan: int

    region: str

    cuisine_prefs: Dict[str, float]

    flavor_prefs: Dict[str, float]

    texture_prefs: Dict[str, float]



def generate_user(user_id: int) -> User:

    # Demographics

    age = np.random.randint(18, 71)

    vegan = np.random.rand() < 0.18

    region = random.choice(REGIONS)

    

    # Initialize preferences

    cuisine = init_uniform(CUISINES)

    flavor = init_uniform(FLAVORS)

    texture = init_uniform(TEXTURES)

    

    # ============================================

    # 1️⃣ Regional cuisine bias

    # ============================================

    apply_additive(cuisine, PRIORS["region_bias"].get(region, {}), scale=1.0)

    

    # ============================================

    # 2️⃣ Lifestyle / dietary correlations

    # ============================================

    if vegan:

        apply_additive(cuisine, PRIORS["vegan_cuisine_boost"], scale=1.0)

        apply_additive(flavor, PRIORS["vegan_flavor_adjust"], scale=1.0)

    

    # Health-conscious behavior (20% of users)

    health_focused = np.random.rand() < 0.20

    if health_focused:

        apply_additive(flavor, PRIORS["health_focus_flavor_adjust"], scale=1.0)

    

    # Low-carb diet (15% of users)

    low_carb = np.random.rand() < 0.15

    if low_carb:

        apply_additive(cuisine, PRIORS["low_carb_cuisine_boost"], scale=1.0)

    

    # ============================================

    # 3️⃣ Age-based perceptual shifts

    # ============================================

    if age > 50:

        apply_additive(texture, PRIORS["age>50_texture_adjust"], scale=1.0)

        apply_additive(flavor, PRIORS["age>50_flavor_adjust"], scale=1.0)

    elif age <= 25:

        apply_additive(flavor, PRIORS["age<25_flavor_adjust"], scale=1.0)

    

    # ============================================

    # 4️⃣ Cuisine → flavor tendencies

    # ============================================

    implied_flavors = {}

    for c, w in cuisine.items():

        if c in PRIORS["cuisine_flavor_profile"]:

            profile = PRIORS["cuisine_flavor_profile"][c]

            implied_flavors = soft_merge(

                implied_flavors, 

                {k: v * w for k, v in profile.items()}, 

                alpha=0.5

            )

    flavor = soft_merge(flavor, implied_flavors, alpha=0.35)

    

    # ============================================

    # 5️⃣ Cuisine → texture tendencies

    # ============================================

    implied_textures = {}

    for c, w in cuisine.items():

        if c in PRIORS["cuisine_texture_profile"]:

            profile = PRIORS["cuisine_texture_profile"][c]

            implied_textures = soft_merge(

                implied_textures,

                {k: v * w for k, v in profile.items()},

                alpha=0.5

            )

    texture = soft_merge(texture, implied_textures, alpha=0.30)

    

    # ============================================

    # 6️⃣ Cross-flavor correlations

    # ============================================

    for base_flavor, correlated in PRIORS["flavor_cooccurrence"].items():

        if base_flavor in flavor and flavor[base_flavor] > 0.5:

            # Strong preference for base flavor influences correlated attributes

            strength = (flavor[base_flavor] - 0.5) * 2  # Scale 0.5-1.0 to 0.0-1.0

            for target, boost in correlated.items():

                if target in flavor:

                    flavor[target] += boost * strength * 0.3  # Moderate influence

    

    # ============================================

    # 7️⃣ Cross-texture correlations

    # ============================================

    for base_texture, correlated in PRIORS["texture_cooccurrence"].items():

        if base_texture in texture and texture[base_texture] > 0.5:

            strength = (texture[base_texture] - 0.5) * 2

            for target, boost in correlated.items():

                if target in texture:

                    texture[target] += boost * strength * 0.3

    

    # ============================================

    # 8️⃣ Flavor → texture bridge (sweet pairs with creamy/smooth)

    # ============================================

    for base_flavor, correlated in PRIORS["flavor_cooccurrence"].items():

        if base_flavor in flavor and flavor[base_flavor] > 0.5:

            strength = (flavor[base_flavor] - 0.5) * 2

            for target, boost in correlated.items():

                # Check if target is a texture attribute

                if target in texture:

                    texture[target] += boost * strength * 0.25

    

    # ============================================

    # Finalization: noise, clamp, normalize

    # ============================================

    cuisine = clip01(jitter_dict(cuisine))

    flavor = clip01(jitter_dict(flavor))

    texture = clip01(jitter_dict(texture))

    

    # Normalize within each category for comparability

    cuisine = normalize_dict(cuisine)

    flavor = normalize_dict(flavor)

    texture = normalize_dict(texture)

    

    return User(

        user_id=user_id, 

        age=age, 

        vegan=int(vegan), 

        region=region,

        cuisine_prefs=cuisine, 

        flavor_prefs=flavor,  # Fixed: was flavor_profile

        texture_prefs=texture

    )



def generate_users(n=120) -> pd.DataFrame:

    users = [generate_user(i) for i in range(n)]

    rows = []

    for u in users:

        row = {

            "user_id": u.user_id, 

            "age": u.age, 

            "vegan": int(u.vegan), 

            "region": u.region,

        }

        row.update({f"cui_{k}": v for k, v in u.cuisine_prefs.items()})

        row.update({f"flv_{k}": v for k, v in u.flavor_prefs.items()})  # Fixed: was flavor_profile

        row.update({f"tex_{k}": v for k, v in u.texture_prefs.items()})

        rows.append(row)

    return pd.DataFrame(rows)
# Generate + peak

set_seed(9)

df = generate_users(1000)

df.head()
# Vectorization - weighted



cuisine_cols = [c for c in df.columns if c.startswith("cui_")]

flavor_cols = [c for c in df.columns if c.startswith("flv_")]

texture_cols = [c for c in df.columns if c.startswith("tex_")]



META_COLS = ["user_id","age","vegan","region","age_bin"]

FEATURE_COLS = cuisine_cols + flavor_cols + texture_cols



FEATURE_WEIGHTS = {

    "cui_": 1.0,   # cuisines dominate

    "flv_": 0.7,   # flavors moderate importance

    "tex_": 0.4,   # textures lower importance

}



def apply_feature_weights(df, feature_cols, weights_dict):

    w = np.ones(len(feature_cols))

    for i, col in enumerate(feature_cols):

        for prefix, wt in weights_dict.items():

            if col.startswith(prefix):

                w[i] = wt

                break

    X = df[feature_cols].values.astype(float)

    X_weighted = X * w

    return X_weighted, w



Xw, wvec = apply_feature_weights(df, FEATURE_COLS, FEATURE_WEIGHTS)

X = df[FEATURE_COLS].values.astype(float)

ids = df["user_id"].values
# Similarity + top k neighbors

SIM = cosine_similarity(X)



def constrained_similarity(df, sim_matrix, weight=0.4):

    vegan = df["vegan"].values.astype(int)

    same_vegan = (vegan[:, None] == vegan[None, :]).astype(float)

    penalty = 1 - weight * (1 - same_vegan)

    return sim_matrix * penalty



def top_k_similar(user_id: int, k=5, exclude_self=True) -> List[Tuple[int, float]]:

    idx = int(np.where(ids == user_id)[0][0])

    sims = SIM[idx]

    order = np.argsort(-sims)

    result = []

    for j in order:

        if exclude_self and j == idx: 

            continue

        result.append((int(ids[j]), float(sims[j])))

        if len(result) >= k: break

    return result



def explain_pair(u1: int, u2: int, top_n=5):

    i = int(np.where(ids == u1)[0][0])

    j = int(np.where(ids == u2)[0][0])

    v1, v2 = X[i], X[j]

    delta = (v1 - v2) ** 2

    # map deltas back to feature names

    contrib = sorted(

        [(FEATURE_COLS[t], float(delta[t])) for t in range(len(FEATURE_COLS))],

        key=lambda x: x[1]

    )

    # smallest deltas = most similar features

    return contrib[:top_n], contrib[-top_n:]
SIM_weighted = cosine_similarity(Xw)

SIM_final = constrained_similarity(df, SIM_weighted, weight=0.5)



def groupwise_similarity(df, sim_matrix, group_col):

    """

    Computes average pairwise similarity within each group.

    """

    groups = df[group_col].unique()

    results = []

    for g in groups:

        idx = df.index[df[group_col] == g].tolist()

        if len(idx) < 2:

            continue

        sub = sim_matrix[np.ix_(idx, idx)]

        mean_sim = sub[np.triu_indices_from(sub, 1)].mean()

        results.append((g, mean_sim))

    return pd.DataFrame(results, columns=[group_col, "mean_intragroup_similarity"]).sort_values("mean_intragroup_similarity", ascending=False)



print("=== Vegan grouping ===")

display(groupwise_similarity(df, SIM, "vegan"))

display(groupwise_similarity(df, SIM_final, "vegan"))



print("=== Regional grouping ===")

display(groupwise_similarity(df, SIM, "region"))

display(groupwise_similarity(df, SIM_final, "region"))
def topk_attribute_overlap(df, sim_matrix, k=5, attr="vegan"):

    ids = df["user_id"].values

    same = []

    for i, uid in enumerate(ids):

        sims = sim_matrix[i]

        order = np.argsort(-sims)

        topk = [ids[j] for j in order if j != uid][:k]

        matches = df.loc[df["user_id"].isin(topk), attr].values

        user_attr = df.loc[i, attr]

        same.append((matches == user_attr).mean())

    return np.mean(same)



print("Average Top-5 same vegan (before):", topk_attribute_overlap(df, SIM, k=5, attr="vegan"))

print("Average Top-5 same vegan (after):", topk_attribute_overlap(df, SIM_final, k=5, attr="vegan"))

print("Average Top-5 same region (before):", topk_attribute_overlap(df, SIM, k=5, attr="region"))

print("Average Top-5 same region (after):", topk_attribute_overlap(df, SIM_final, k=5, attr="region"))

from sklearn.metrics import silhouette_score



# Compute silhouette score using vegan labels

Xw_norm = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-9)

score_before = silhouette_score(Xw_norm, df["vegan"])

score_after  = silhouette_score(Xw_norm, df["vegan"], metric="cosine", sample_size=100)



print("Silhouette wrt vegan (before):", round(score_before, 3))

print("Silhouette wrt vegan (after weighting):", round(score_after, 3))

# Example usage for user 3

u = 3

print("Meta:", df.loc[df.user_id==u, ["age","vegan","region"]].to_dict(orient="records")[0])

print("Top-5 similar:", top_k_similar(u, k=5))



best = top_k_similar(u, k=1)[0][0]

same, different = explain_pair(u, best, top_n=6)

print("\nClosest shared features:", same)

print("\nLargest differences:", different)
pca = PCA(n_components=3, random_state=9)

Z = pca.fit_transform(Xw)



vis_pca = pd.DataFrame({

    "PC1": Z[:, 0],

    "PC2": Z[:, 1],

    "PC3": Z[:, 2],

    "vegan": df["vegan"].map({0: "non-vegan", 1: "vegan"}),

    "region": pd.Categorical(df["region"], categories=REGIONS),

    "age": df["age"]

})



# --- Helper to create color-specific trace list ---

def make_pca_traces(color_attr):

    if color_attr == "age":

        # Continuous variable

        return [

            go.Scatter3d(

                x=vis_pca["PC1"],

                y=vis_pca["PC2"],

                z=vis_pca["PC3"],

                mode="markers",

                marker=dict(

                    size=5,

                    color=vis_pca["age"],

                    colorscale="Viridis",

                    colorbar=dict(title="Age"),

                    opacity=0.8

                ),

                text=vis_pca.apply(lambda r: f"Region: {r['region']}<br>Age: {r['age']}<br>Vegan: {r['vegan']}", axis=1),

                hoverinfo="text",

                name="age"

            )

        ]

    else:

        # Categorical variable (vegan or region)

        traces = []

        palette = px.colors.qualitative.Safe

        categories = vis_pca[color_attr].unique()

        for i, cat in enumerate(categories):

            subset = vis_pca[vis_pca[color_attr] == cat]

            traces.append(

                go.Scatter3d(

                    x=subset["PC1"], y=subset["PC2"], z=subset["PC3"],

                    mode="markers",

                    marker=dict(size=5, color=palette[i % len(palette)], opacity=0.8),

                    name=str(cat),

                    text=subset.apply(lambda r: f"Region: {r['region']}<br>Age: {r['age']}<br>Vegan: {r['vegan']}", axis=1),

                    hoverinfo="text"

                )

            )

        return traces



fig_pca = go.Figure()



# Store all traces for all modes

trace_groups = {}

attributes = ["vegan", "region", "age"]



for attr in attributes:

    traces = make_pca_traces(attr)

    trace_groups[attr] = traces

    for i, trace in enumerate(traces):

        trace.visible = (attr == "vegan")  # only vegan visible at start

        fig_pca.add_trace(trace)



# Build visibility masks

dropdown_buttons = []

total_traces = sum(len(v) for v in trace_groups.values())



for attr in attributes:

    visibility = []

    for a in attributes:

        visibility.extend([a == attr] * len(trace_groups[a]))

    

    dropdown_buttons.append(

        dict(

            label=attr.capitalize(),

            method="update",

            args=[

                {"visible": visibility},

                {"title": f"Taste Space (PCA 3D Interactive) - Colored by {attr.capitalize()}"}

            ]

        )

    )



fig_pca.update_layout(

    title="Taste Space (PCA 3D Interactive) - Colored by Vegan",

    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),

    updatemenus=[{

        "buttons": dropdown_buttons,

        "direction": "down",

        "x": 1.15,

        "y": 0.9,

        "xanchor": "left",

        "showactive": True

    }],

    margin=dict(l=0, r=0, b=0, t=50)

)



fig_pca.show()
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=9)

Z_umap = umap.fit_transform(Xw)



vis_umap = pd.DataFrame({

    "UMAP1": Z_umap[:, 0],

    "UMAP2": Z_umap[:, 1],

    "vegan": df["vegan"].map({0: "non-vegan", 1: "vegan"}),

    "region": pd.Categorical(df["region"], categories=REGIONS),

    "age": df["age"]

})



def make_umap_traces(color_attr):

    if color_attr == "age":

        return [

            go.Scatter(

                x=vis_umap["UMAP1"],

                y=vis_umap["UMAP2"],

                mode="markers",

                marker=dict(

                    size=6,

                    color=vis_umap["age"],

                    colorscale="Viridis",

                    colorbar=dict(title="Age"),

                    opacity=0.8

                ),

                text=vis_umap.apply(lambda r: f"Region: {r['region']}<br>Age: {r['age']}<br>Vegan: {r['vegan']}", axis=1),

                hoverinfo="text",

                name="age"

            )

        ]

    else:

        traces = []

        palette = px.colors.qualitative.Safe

        categories = vis_umap[color_attr].unique()

        for i, cat in enumerate(categories):

            subset = vis_umap[vis_umap[color_attr] == cat]

            traces.append(

                go.Scatter(

                    x=subset["UMAP1"], y=subset["UMAP2"],

                    mode="markers",

                    marker=dict(size=6, color=palette[i % len(palette)], opacity=0.8),

                    name=str(cat),

                    text=subset.apply(lambda r: f"Region: {r['region']}<br>Age: {r['age']}<br>Vegan: {r['vegan']}", axis=1),

                    hoverinfo="text"

                )

            )

        return traces



fig_umap = go.Figure()



# Store all traces for all modes

trace_groups = {}

attributes = ["vegan", "region", "age"]



for attr in attributes:

    traces = make_umap_traces(attr)

    trace_groups[attr] = traces

    for i, trace in enumerate(traces):

        trace.visible = (attr == "vegan")  # Only "vegan" visible at the start

        fig_umap.add_trace(trace)



# Build visibility masks for the dropdown buttons

dropdown_buttons = []

total_traces = sum(len(v) for v in trace_groups.values())



for attr in attributes:

    visibility = []

    for a in attributes:

        visibility.extend([a == attr] * len(trace_groups[a]))



    dropdown_buttons.append(

        dict(

            label=attr.capitalize(),

            method="update",

            args=[

                {"visible": visibility},

                {"title": f"Taste Space (UMAP 2D Interactive) - Colored by {attr.capitalize()}"}

            ]

        )

    )



fig_umap.update_layout(

    title="Taste Space 1.1 (UMAP 2D Interactive) - Colored by Vegan",

    xaxis_title="UMAP1",

    yaxis_title="UMAP2",

    updatemenus=[{

        "buttons": dropdown_buttons,

        "direction": "down",

        "x": 1.1,

        "y": 0.9,

        "xanchor": "left",

        "showactive": True

    }],

    margin=dict(l=0, r=0, b=0, t=50)

)



fig_umap.show()
# Ablations

def regen_with_noise(noise):

    global RANDOM_NOISE

    RANDOM_NOISE_OLD = 0.12

    RANDOM_NOISE = noise

    df2 = generate_users(150)

    RANDOM_NOISE = RANDOM_NOISE_OLD

    return df2



df_low_noise  = regen_with_noise(0.05)

df_high_noise = regen_with_noise(0.25)



print("Mean Indian (vegan=1 vs 0) — low noise")

display(df_low_noise.groupby("vegan")[["cui_Indian"]].mean().round(3))

print("Mean Indian (vegan=1 vs 0) — high noise")

display(df_high_noise.groupby("vegan")[["cui_Indian"]].mean().round(3))


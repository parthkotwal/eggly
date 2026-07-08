# eggly Architecture

This document describes what eggly is trying to be, how it works technically, what stage the project is at, and what's deferred. It is meant to be enough context for a fresh coding agent (or a fresh version of the author returning after months away) to understand the project without re-deriving it.

Update this doc as decisions change. Prefer editing to accumulating stale notes. It has gone stale once already (it claimed "no code written" while two notebooks had completed runs); treat "current stage" drift as a bug.

Companion docs: `AGENTS.md` (how to work here), `EVAL.md` (evaluation protocol, results ledger, and the interpretation rules every experiment must follow).

## What eggly is (and isn't, right now)

Eggly, in full, is a taste-based social platform: food recommendations, user-to-user taste matching, communities around cuisines or ingredients, IRL and virtual events. That full vision is not what's being built right now.

The current focus is the **Yolk** — the underlying representation, matching, and adaptation system that would power all of the above. The reasoning: the ML core is the interesting-to-learn part and the thing that gates everything else. A social layer built on a shallow taste model is worse than no social layer. A solid taste model can have a social layer bolted on in a weekend of agent-assisted work later.

So: this repo is a recommender system with just-enough API scaffolding around it, not an app. And importantly, it is a recommender for a *social platform about taste* — not a recipe recommender. Recipes (Food.com) are just the instance data we happen to have. Design decisions should be judged against the platform vision (cold start, interpretability, user-user matching), not against "predict the next recipe click" alone.

## What the Yolk does

The Yolk answers three questions, all of which reduce to nearest-neighbor lookup in a shared embedding space:

1. Given a user, which foods should they like? (user → food)
2. Given a user, which other users share their taste? (user → user)
3. Given a food, who would like it? (food → user)

To answer these, it does two things: **represent** users and foods as vectors in a shared space where "close = compatible," and **update** those representations as new information comes in.

## Entities and how they relate

The world eggly models has two kinds of things.

**Concepts** are abstract taste primitives users have opinions about directly: cuisines (Italian, Thai), dishes (carbonara, butter chicken), flavors (umami, spicy), diets (vegan, keto), textures (crispy, creamy), ingredients-as-taste-objects (mango, cilantro). A user can like "mango" without any specific mango recipe being involved. The concept universe is small — hundreds to low thousands.

**Instances** are concrete things in the world that can be experienced and logged: a specific recipe, a specific restaurant, a specific dish at a specific restaurant, a specific event. The instance universe is large — millions.

Instances are tagged with concepts. A butter chicken recipe carries tags like `cuisine:indian`, `dish:butter_chicken`, `flavor:creamy`, `ingredient:chicken`, `diet:non-vegetarian`. Some entities live at both levels — "mango" is a concept and can also be an instance you eat and log. That overlap is real and the data model needs to handle it, but it doesn't require special-casing; a mango-the-instance is just an instance whose primary tag is mango-the-concept.

Users can have edges to either type. Interactions with instances (cooked this recipe, rated this restaurant) are the main behavioral signal. Interactions with concepts (followed the "mango" tag at onboarding, joined a "spicy food" group) are how cold-start and stated preferences enter the system.

This is a heterogeneous graph. It doesn't need graph neural networks in V0 — it just needs a data model that respects the structure so we don't have to rewrite it later.

## The embedding space

All three node types — concepts, instances, users — live in the same vector space. This is what makes the whole system work: cosine similarity between a user and a recipe, between two users, or between a user and a raw concept like "mango" is all the same operation.

- **Concept embeddings** can be bootstrapped from sentence-transformers on the concept name + a short definition.
- **Instance embeddings** come from the instance's own text (title, ingredients, description) combined with the embeddings of its tagged concepts.
- **User embeddings** come from aggregating the embeddings of things they've interacted with, weighted by action strength and recency. In V0 this is a weighted mean; later it becomes a learned aggregation, eventually a two-tower model.

**A hard-won caveat on this space** (see "What we've learned so far"): a content-derived embedding space encodes *what things are like*, not *what people will do next*. It is the right substrate for cold start, interpretability, and user-user matching. It is structurally incapable of beating behavioral signal (popularity, collaborative filtering) at exact-next-interaction prediction, and it should never be asked to. The plan for making content and behavior meet is the V2 two-tower model, which *trains* the mapping from content space into interaction space.

## The pipeline

There are six logical stages. Not all are needed in V0.

1. **Food embedding.** Offline batch job. Recipe metadata → vector. Store in pgvector or FAISS.
2. **User embedding.** Interaction history + static profile + priors → vector. Recomputed on some cadence (or on the fly in V0).
3. **Retrieval.** Given a user vector, nearest-neighbor search across food vectors returns top ~200 candidates. Must be fast.
4. **Ranking.** A more expensive model rescores those 200 candidates using richer features. Skip in V0.
5. **Serving + logging.** API returns recommendations; every impression, click, like, skip is logged. The logs are tomorrow's training data.
6. **Adaptation.** New interactions update user embeddings (and eventually food embeddings and the models themselves). Runs offline on a schedule.

## Datasets

**Primary interaction dataset: `shuyangli94/food-com-recipes-and-user-interactions` (Kaggle).** The standard academic Food.com dataset: 231,637 recipes (`RAW_recipes.csv`), 1,132,367 interactions with ratings, dates, and review text (`RAW_interactions.csv`), plus official pre-existing splits (`interactions_train/validation/test.csv`) and preprocessed token files (`PP_*.csv`).

**Split status:** the V0/V1 notebooks built their own ad-hoc global temporal 80/20 cut instead of using the official splits — a docs/code divergence that stood for a while. The plan is to move to the official splits (protocol v2 in `EVAL.md`) so numbers are comparable to published baselines. Until then, every reported number states which protocol produced it.

**Recipe corpus expansion (V2+): RecipeNLG (Kaggle, `saldenisov/recipenlg`).** 2.2M recipes, a superset of Recipe1M+, cleanly parsed ingredient arrays. Use this when we want to embed a much larger recipe pool than Food.com covers. Not needed now.

Not using: the other Food.com variants (irkaal, realalexanderwei, AkashPS11). They overlap with shuyangli without adding meaningful signal, and pulling in more datasets means more schema-wrangling for the same learning outcome.

Restaurants, events, and any real user data are future concerns.

## Cold start

Every new eggly user is cold — this is one of the reasons the project is interesting. The plan:

- At onboarding, the user picks concepts (cuisines they like, diets they follow, flavors that appeal, ingredients they love or hate).
- Their initial embedding is a weighted mean of those concept embeddings, plus prior-based adjustments (the hand-authored regional/age/dietary priors from the earlier version of this project, in `archive_ml/config.py`, which are being kept and reframed as principled cold-start bootstrapping).
- Once real interactions start flowing, the adaptation layer takes over and the priors decay in influence.

New instances (a newly-added recipe with no interactions) are cold too, but easier — their embedding comes from content alone. Cold start is where the content/concept representation earns its keep — this, not beating collaborative baselines at retrieval, is its primary job.

## Evaluation

The full protocol, the results ledger, and the interpretation rules live in **`EVAL.md`**. Read it before touching any eval code. The rules that exist because they were each learned the hard way:

- No metric is reportable without random + popularity (+ collaborative, once built) reference rows on the same split.
- Convert rates to user counts before claiming one method beats another; sub-2-standard-error gaps are ties.
- Full-catalog retrieval and sampled-negative ranking answer different questions (behavior prediction vs. representation quality); don't conflate them.
- Temporal splits only. Train/test leakage is the classic recsys footgun.
- Offline metrics measure "did we predict what the user did," not "did we recommend something they'd love." For a taste platform those diverge sharply — this gap is a core learning topic of the project, not noise to average away.

User-to-user matching is meaningfully harder to evaluate because there's no natural label. Deferring evaluation strategy for it until user-food is solid.

## What we've learned so far

Recorded so future work doesn't re-derive (or re-litigate) it. Full numbers in `EVAL.md`.

1. **V0 works end-to-end and carries real signal.** MiniLM content embeddings + weighted-mean user vector retrieves at ~25× random on a 231K-item catalog. Every line understood. V0's goal was met.

2. **Mean collapse is real but eval-dependent.** On full-catalog Recall@10, max-sim (the complete cure for collapse) moved only 0.0009 → 0.0012 — a tie in noise. On sampled-negative ranking, max-sim pulled ahead clearly: AUC 0.619 vs mean's 0.571 (~0.05 gap). The collapse hypothesis was tested on the wrong eval first. Lesson: mean collapse costs ranking quality but the effect drowns in the 231K-way retrieval lottery. The aspect-decomposed representation (V1 notebook) landed between mean and max-sim on both evals, as expected for a partial de-averaging.

3. **Content similarity is not preference prediction.** The qualitative check made it vivid: a user's retrieval was semantically coherent with their history (casseroles, flatbreads) while their actual test positives were a taco seasoning mix and a cheese ball. Next-interaction prediction is dominated by exposure, popularity, and collaborative effects that content embeddings cannot see. This is the single most important recsys lesson in the project so far.

4. **The reference frame now exists.** Random (0.00004) → Content (~0.001, 23–53 hits) → Implicit ALS (0.014, 499 hits) → Popularity (0.020, 608 hits). The non-personalized popularity baseline beats every personalized content method by 22× and even beats personalized CF by ~20%. This is the most important empirical fact in the project: on Food.com's sparse interaction data (~0.0006% density), "recommend what's popular" is extremely hard to beat. CF trails popularity because there isn't enough co-occurrence signal to personalize better than the global mode.

5. **Content-based methods are in their own tier on full-catalog retrieval — all tied.** Mean, max-sim, and aspect-decomposed all land between 23 and 53 hits out of 10,000 users (within noise). Iterating on content aggregation is a dead end for full-catalog interaction-prediction metrics.

6. **Content is the only method that works on cold items — and it works well.** On the official LOO split (100% cold test items), popularity and ALS get zero hits and AUC *below* random (0.17 and 0.45 respectively — they anti-predict). Content max-sim holds steady at AUC 0.62 across both warm and cold settings. Content mean is ~0.05 AUC behind due to mean collapse, confirmed on both protocols.

7. **The case for a hybrid is now empirical, not hypothetical.** Content handles cold items (AUC 0.62), behavior handles warm items (AUC 0.62), and each fails where the other succeeds. The V2 two-tower model is the right architecture to blend them — it trains a mapping from content space into interaction space, keeping content's cold-item capability while learning behavioral patterns.

Consequence for the roadmap: the content representation is empirically validated for its primary jobs — cold-item scoring, user-user matching, interpretability. It is not a failed retrieval model. The retrieval-metric path runs through the V2 two-tower, which blends content and behavioral signal.

## Roadmap and current stage

**V0 — foundation, in a notebook. ✅ Done.** Recipes as instances only, Food.com tags as concept vocabulary. Recipe embeddings from title + ingredients + tags via sentence-transformers (all-MiniLM-L6-v2, 384-dim, cached at `data/recipe_embeddings_v0.npy`). Weighted-mean user embedding, cosine retrieval, temporal holdout. Also ran the max-sim ablation that falsified the collapse hypothesis. Lives in `notebooks/v0_pipeline.ipynb`.

**V0.5 — the reference frame. ✅ Done.** All baselines (random, popularity, implicit ALS), both evals (full-catalog retrieval, sampled-negative ranking), both protocols (v1 temporal, v2 official LOO cold-item). Key finding: content max-sim is the only method with consistent ~0.62 AUC across warm and cold items; behavioral methods anti-predict on cold items (AUC < 0.50). See `EVAL.md` cross-protocol summary. Lives in `notebooks/v0_pipeline.ipynb` and `notebooks/v05_official_splits.ipynb`.

**V1 — real data model.** Promote concepts to first-class nodes with their own embeddings; introduce the shared embedding space explicitly; cold-start via concept onboarding + priors; move out of notebooks into a package; start logging what would be interactions. Justified by platform needs (cold start, interpretability, user-user matching) — explicitly *not* expected to move interaction-prediction metrics, per lesson 3. A partial spike of this exists in `notebooks/v1_aspect_embeddings.ipynb` (the tag→aspect taxonomy is reusable; the retrieval fusion is not the point).

**V2 — modeling upgrades.** Two-tower model trained contrastively on interactions — this is where content features and behavioral signal finally meet, and the expected first big metric jump. Optional: expand food corpus with RecipeNLG. Add a ranking stage on top of retrieval.

**V3 — the app.** FastAPI backend exposing the recommender. Minimal frontend (agent-assisted, because this part is not the point). Deploy somewhere with a URL. Enough of a UX to actually generate interaction logs.

**V4+ — the interesting part, if energy remains.** Real adaptation layer with proper feedback loops. User-to-user matching. Restaurants as another instance type. Exploration strategies (bandits, epsilon-greedy). The social layer starts to make sense here.

## Open questions

- How exactly to weight action types in user embedding aggregation (view vs like vs cook vs rate). Start with something hand-tuned, revisit with data. (First data point: rating-weighting {4:1.0, 5:2.0} changed nothing on Food.com.)
- How to represent negative signals (skips, dislikes) — as negative-weight edges? A separate "avoid" vector? Defer until we have any real interaction data.
- Whether to unify "user interacted with concept" and "user interacted with instance" into one edge type with a target-type field, or keep them separate. Probably one edge type, decide when implementing.
- How to blend behavioral (CF) and content signal before V2's two-tower exists — e.g., popularity-aware re-ranking of content retrieval, or CF with content fallback for cold items. Becomes concrete once V0.5 baselines exist.
- Restaurant data source when we get there. Yelp Open Dataset is the obvious option.

## What's deliberately deferred

Social features (groups, meetups, events, chat, sharing), authentication, real frontend, mobile, notifications, real user data collection, anything on-device, anything about images. These are all real parts of the full eggly vision. They're not gone — they're just not V0–V3.

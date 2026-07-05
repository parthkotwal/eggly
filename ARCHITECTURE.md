# eggly Architecture

This document describes what eggly is trying to be, how it works technically, what stage the project is at, and what's deferred. It is meant to be enough context for a fresh coding agent (or a fresh version of the author returning after months away) to understand the project without re-deriving it.

Update this doc as decisions change. Prefer editing to accumulating stale notes.

## What eggly is (and isn't, right now)

Eggly, in full, is a taste-based social platform: food recommendations, user-to-user taste matching, communities around cuisines or ingredients, IRL and virtual events. That full vision is not what's being built right now.

The current focus is the **Yolk** — the underlying representation, matching, and adaptation system that would power all of the above. The reasoning: the ML core is the interesting-to-learn part and the thing that gates everything else. A social layer built on a shallow taste model is worse than no social layer. A solid taste model can have a social layer bolted on in a weekend of agent-assisted work later.

So: this repo is a recommender system with just-enough API scaffolding around it, not an app.

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

## The pipeline

There are six logical stages. Not all are needed in V0.

1. **Food embedding.** Offline batch job. Recipe metadata → vector. Store in pgvector or FAISS.
2. **User embedding.** Interaction history + static profile + priors → vector. Recomputed on some cadence (or on the fly in V0).
3. **Retrieval.** Given a user vector, nearest-neighbor search across food vectors returns top ~200 candidates. Must be fast.
4. **Ranking.** A more expensive model rescores those 200 candidates using richer features. Skip in V0.
5. **Serving + logging.** API returns recommendations; every impression, click, like, skip is logged. The logs are tomorrow's training data.
6. **Adaptation.** New interactions update user embeddings (and eventually food embeddings and the models themselves). Runs offline on a schedule.

## Datasets

**Primary interaction dataset: `shuyangli94/food-com-recipes-and-user-interactions` (Kaggle).** This is the standard academic Food.com dataset — ~180K recipes, ~700K user interactions with ratings, pre-existing train/val/test splits, tags already parsed. This is what the recsys is trained and evaluated on.

**Recipe corpus expansion (V2+): RecipeNLG (Kaggle, `saldenisov/recipenlg`).** 2.2M recipes, a superset of Recipe1M+, cleanly parsed ingredient arrays. Use this when we want to embed a much larger recipe pool than Food.com covers. Not needed for V0.

Not using: the other Food.com variants (irkaal, realalexanderwei, AkashPS11). They overlap with shuyangli without adding meaningful signal, and pulling in more datasets means more schema-wrangling for the same learning outcome.

Restaurants, events, and any real user data are future concerns.

## Cold start

Every new eggly user is cold — this is one of the reasons the project is interesting. The plan:

- At onboarding, the user picks concepts (cuisines they like, diets they follow, flavors that appeal, ingredients they love or hate).
- Their initial embedding is a weighted mean of those concept embeddings, plus prior-based adjustments (the hand-authored regional/age/dietary priors from the earlier version of this project, which are being kept and reframed as principled cold-start bootstrapping).
- Once real interactions start flowing, the adaptation layer takes over and the priors decay in influence.

New instances (a newly-added recipe with no interactions) are cold too, but easier — their embedding comes from content alone.

## Evaluation

Offline metrics on held-out Food.com interactions: recall@k, nDCG@k, hit rate. These are the standard, they're honest signals for retrieval quality, and they let this project's numbers be compared against published baselines on the same dataset.

Two things to watch:
- Offline metrics on interaction data measure "did we predict what the user actually did," not "did we recommend something they'd love." Those are related but not the same. This gap is a real learning topic for the project, not something to solve away.
- Train/test leakage is the classic recsys footgun. Splits need to be temporal (train on earlier interactions, evaluate on later ones), not random, or the metrics will lie.

User-to-user matching is meaningfully harder to evaluate because there's no natural label. Deferring evaluation strategy for it until user-food is solid.

## Roadmap and current stage

**V0 — foundation, in a notebook.** Recipes as instances only, no separate concept nodes. Use Food.com's built-in tags as the concept vocabulary. Recipe embeddings from title + ingredients + tags via sentence-transformers. User embedding as weighted mean of liked recipes. Cosine similarity retrieval. Recall@10 on a temporal holdout. Target: a clean notebook where the whole pipeline runs end-to-end and every line is understood.

**V1 — real data model.** Promote concepts to first-class nodes with their own embeddings. Introduce the shared embedding space explicitly. Cold-start via concept onboarding + priors. Move out of a notebook into a proper Python package. Start logging what would be interactions if there were a frontend.

**V2 — modeling upgrades.** Two-tower model trained contrastively on interactions. Optional: expand food corpus with RecipeNLG. Add a ranking stage on top of retrieval. Real evaluation loop with multiple metrics.

**V3 — the app.** FastAPI backend exposing the recommender. Minimal frontend (Next.js or plain React, agent-assisted, because this part is not the point). Deploy somewhere with a URL. Enough of a UX to actually generate interaction logs.

**V4+ — the interesting part, if energy remains.** Real adaptation layer with proper feedback loops. User-to-user matching. Restaurants as another instance type. Exploration strategies (bandits, epsilon-greedy). The social layer starts to make sense here.

**Current stage:** pre-V0. Design settled, no code written in the new repo yet. Some prior work exists from an earlier attempt (feature priors, synthetic candidate generator, weighted cosine matching, a Django scaffold) — the prior work on representation and priors is worth cannibalizing; the Django scaffold and Next.js starter are not.

## Open questions

- How exactly to weight action types in user embedding aggregation (view vs like vs cook vs rate). Start with something hand-tuned, revisit with data.
- How to represent negative signals (skips, dislikes) — as negative-weight edges? A separate "avoid" vector? Defer until we have any real interaction data.
- Whether to unify "user interacted with concept" and "user interacted with instance" into one edge type with a target-type field, or keep them separate. Probably one edge type, decide when implementing.
- Restaurant data source when we get there. Yelp Open Dataset is the obvious option.

## What's deliberately deferred

Social features (groups, meetups, events, chat, sharing), authentication, real frontend, mobile, notifications, real user data collection, anything on-device, anything about images. These are all real parts of the full eggly vision. They're not gone — they're just not V0-V3.
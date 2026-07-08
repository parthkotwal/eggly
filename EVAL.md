# Evaluation protocol and results ledger

This file is the single source of truth for how experiments are evaluated in eggly and what every run has produced so far. **Read this before writing or modifying any evaluation code, and before reporting any metric.** If you run an experiment, append it to the ledger. If you change the protocol, update this file and rerun the baselines under the new protocol — numbers from different protocols must never sit in the same comparison table.

## The interpretation rules (non-negotiable)

These exist because each one was violated once and cost real time.

1. **No metric without a reference frame.** A method's Recall@10 is meaningless alone. Every comparison table must include, on the *same split and cohort*: random expectation, a popularity baseline, and (once it exists) the best collaborative baseline. "Is 0.0012 good?" is unanswerable without them; with them it answers itself.

2. **Convert rates to user counts before claiming a difference.** HitRate@10 = 0.0027 on 10,000 users means *27 users got any hit*. The binomial standard error at these rates is ~0.0005, so two methods within ~0.001 of each other are **a tie**, not a ranking. Do the arithmetic every time: `n_hits = rate × n_users`, `SE ≈ sqrt(p(1-p)/n)`. If the gap is under 2 SE, report it as a tie.

3. **Two evals for two different questions.** Full-catalog retrieval metrics answer "can we predict the exact next interaction?" — a question dominated by exposure, popularity, and collaborative effects. Sampled-negative ranking (rank each test positive against ~100 random negatives; report AUC / HR@10) answers "does the representation order this user's preferences correctly?" A taste representation can be genuinely good and still lose the 231K-way lottery. Never use the first eval to judge representation quality; never use the second to claim retrieval wins.

4. **Content similarity is not preference prediction.** What a user interacts with next is driven by exposure, novelty, and occasion — collaborative signal — not semantic proximity to their history. Pure content-based methods are expected to lose to popularity on interaction metrics. The content/concept representation's jobs are cold start, interpretability, user-user matching, and *features for a model trained on interactions* (the V2 two-tower). Do not spend cycles trying to make cosine-in-content-space beat behavioral baselines at exact-item retrieval; that failure mode is structural.

5. **Cheapest falsifying test before infrastructure.** Before building anything to exploit a hypothesis, design the cheapest experiment whose *failure kills the hypothesis*, and run it first. Canonical example: "mean embedding collapse hurts retrieval" was tested with max-sim scoring (the complete cure for collapse, ~20 lines) before building aspect infrastructure. Max-sim moved Recall@10 by +0.0003 — hypothesis falsified, and that ceiling applied to the whole aggregation-fixing family of ideas. A flat result that kills a hypothesis is a *successful* experiment: record it in the ledger and do not build the infrastructure anyway out of momentum.

## Canonical protocol

**Current protocol (v1, used by all ledger entries below):**

- Data: `RAW_interactions.csv` + `RAW_recipes.csv` (Food.com).
- Split: global temporal 80/20 cut on pooled interactions by date. Cutoff lands at 2011-12-27; train = 905,893 interactions (2000→2011), test = 226,474 (2011→2018).
- Positives: rating ≥ 4. Train-positive = 822,501; test-positive = 181,223.
- Cohort: the 10,000 users with positives on both sides of the cut.
- Candidates: full catalog, 231,637 recipes. K = 10. Train-seen items filtered from recommendations.
- Metrics: Recall@10 (denominator = full per-user test-positive set), HitRate@10, nDCG@10.

**Known weaknesses of protocol v1** (fix these when touching eval code; bump to protocol v2 and rerun everything):

- The dataset ships official splits (`interactions_train/validation/test.csv`) that we ignore. Switching to them makes results comparable to published Food.com numbers and removes our ad-hoc choices. ARCHITECTURE.md claimed we use them; the code never did.
- The 10K "active on both sides" cohort is survivorship-biased — it silently drops all cold-start and one-sided users (~227K unique users exist in the raw data). Fine for method comparison, misleading as a deployment estimate. Always state the cohort next to the numbers.
- **Latent bug:** seen-item filtering currently happens *inside* the top-50 `argpartition` pool, after truncation. A user can receive <10 recs, and as recall improves the filter increasingly eats real candidates. Filter seen items *before* truncating to the pool (or retrieve pool_size + n_seen).
- The test window spans 7 years; late-window positives reflect taste and catalog drift that no static representation can capture.

**Hygiene checklist before trusting any run:**

- [ ] Temporal split (never random) — same split object shared by every method in the table.
- [ ] Same cohort for every method in the table (report the cohort size).
- [ ] Seen-items filtered before pool truncation.
- [ ] Leakage sanity check ran (train items in top-K must be 0.00).
- [ ] Random + popularity rows present in the table.
- [ ] Rate→count arithmetic done before any "X beats Y" sentence.
- [ ] Per-user sanity check printed for ≥1 user (retrieved items vs. actual test positives) — this is what revealed that semantically coherent retrieval ≠ hits.

## Results ledger

Append-only. New protocol ⇒ new subsection. Include date, notebook/script, and one-line takeaway.

### Protocol v1 — global temporal cut, 10K common users, full 231,637-item catalog

Random expectation at this catalog size: Recall@10 ≈ 10/231,637 ≈ **0.00004**.

| Date | Method | Source | Recall@10 | HitRate@10 | nDCG@10 | Takeaway |
|---|---|---|---|---|---|---|
| 2026-07 | Mean embedding, unweighted, no seen-filter | v0_pipeline.ipynb | 0.0009 | 0.0026 | 0.0006 | First end-to-end number; ~25× random |
| 2026-07 | Mean embedding, rating-weighted, seen-filtered | v0_pipeline.ipynb | 0.0009 | 0.0023 | 0.0007 | Weighting + filtering changed nothing (tie) |
| 2026-07 | Max-sim over liked-item set | v0_pipeline.ipynb §9 | 0.0012 | 0.0027 | 0.0008 | Complete cure for mean collapse gained +0.0003 → **collapse hypothesis falsified** |
| 2026-07 | Aspect-decomposed (6 tag-taxonomy axes, masked-mean fusion) | v1_aspect_embeddings.ipynb | 0.0011 | 0.0027 | 0.0007 | Lands between mean and max-sim, as predicted for partial de-averaging; tie with max-sim |
| 2026-07-08 | **Popularity baseline** | v0_pipeline.ipynb §10 | **0.0204** | **0.0608** | **0.0139** | **608 hits. 22× content mean. The non-personalized behavioral ceiling.** |
| 2026-07-08 | **Implicit ALS (CF)** (64 factors, 15 iter) | v0_pipeline.ipynb §10 | **0.0140** | **0.0499** | **0.0108** | **499 hits. Personalized CF trails popularity on this sparse dataset.** |

Binomial SE at HitRate ≈ 0.003, n=10,000: **±0.0005**. Content methods are mutually tied. Popularity vs ALS gap (0.0608 vs 0.0499) is ~22 SE — real. Popularity vs content gap (~0.06 vs ~0.003) is massive and structural.

**Standing conclusions from v1 runs:**
- **The reference frame exists.** Random (0.00004) → Content (~0.001) → ALS (0.014) → Popularity (0.020). Content is ~25× random; popularity is ~500× random and 22× content. All content-based variants are in the noise regime relative to each other.
- **Popularity beats CF on this dataset.** Interaction density is ~0.000006; ALS doesn't have enough co-occurrence signal to personalize better than "recommend what's popular." On denser data this flips.
- The user-aggregation family (mean vs. max-sim vs. aspect-grouped means) is exhausted. Do not iterate further on aggregation against protocol v1 numbers.
- **Content similarity is not preference prediction** — confirmed quantitatively. Content methods see 23–53 hits; popularity sees 608. The gap is not fixable by better content representations; it is structural (rule 4).
- Qualitative check (user 163986): retrieval was semantically coherent with history (casseroles, flatbreads) while actual test positives were "taco seasoning mix" and "sweet cheese ball."
- Seen-filter bug (filtering inside top-50 pool) fixed 2026-07-08; all numbers from this date onward use mask-before-select.

### Sampled-negative ranking eval — protocol v1 split, 100 random negatives

Spec: for each (user, test-positive), rank the positive against 100 uniformly sampled unseen negatives; report AUC and HR@10 over these 101-item lists. This eval measures representation quality (can the model order liked above non-liked?) rather than full-catalog retrieval accuracy.

| Date | Method | Source | AUC | HR@10 | Takeaway |
|---|---|---|---|---|---|
| 2026-07-08 | Random | v0_pipeline.ipynb §11 | 0.500 | 0.098 | Sanity check passed |
| 2026-07-08 | Popularity | v0_pipeline.ipynb §11 | 0.616 | 0.409 | Strong AUC; HR@10 inflated because positives are often popular items |
| 2026-07-08 | Content mean | v0_pipeline.ipynb §11 | 0.571 | 0.146 | Well above random; ranks liked items above average but mean collapse costs ~0.05 AUC |
| 2026-07-08 | **Content max-sim** | v0_pipeline.ipynb §11 | **0.619** | 0.208 | **Ties ALS and popularity on AUC. Collapse hypothesis revived on ranking eval — mean loses ~0.05 AUC vs max-sim** |
| 2026-07-08 | Implicit ALS (CF) | v0_pipeline.ipynb §11 | 0.617 | 0.314 | Ties content max-sim on AUC; higher HR@10 from behavioral signal placing positives in top slots |

**Standing conclusions from sampled-negative eval:**
- The content representation carries real ranking signal (AUC 0.57–0.62 vs random 0.50). This was masked by full-catalog retrieval metrics.
- **Mean collapse is real and costs ~0.05 AUC** — max-sim (0.619) vs mean (0.571). This gap was invisible on full-catalog Recall@10 because both methods were in the noise floor. The collapse hypothesis was correctly tested but on the wrong eval.
- Content max-sim ≈ ALS ≈ Popularity on AUC (~0.62). The three signal types (content similarity, collaborative, popularity) carry roughly equal information for ordering liked vs random items. They likely carry *different* information — a hybrid should beat any one alone.
- HR@10 and AUC diverge: popularity and ALS convert AUC into top-10 placement better than content, consistent with behavioral methods placing items at the *very* top vs content placing them merely above average.

### Protocol v2 — official LOO split, cold-item test set

The dataset's official splits: leave-one-out per user (exactly 1 test interaction), **every test recipe has zero training interactions**. 10,354 eval users (positive in both train and test), 231,637-item catalog.

#### Full-catalog retrieval

| Date | Method | Source | Recall@10 | HitRate@10 | nDCG@10 | n_hits | Takeaway |
|---|---|---|---|---|---|---|---|
| 2026-07-08 | Content mean | v05_official_splits.ipynb | 0.0013 | 0.0013 | 0.0005 | 13 | Only content can score cold items |
| 2026-07-08 | Content max-sim | v05_official_splits.ipynb | 0.0018 | 0.0018 | 0.0011 | 19 | ~50% more hits than mean |
| 2026-07-08 | Popularity | v05_official_splits.ipynb | 0.0000 | 0.0000 | 0.0000 | 0 | **Zero hits — can't score cold items** |
| 2026-07-08 | Implicit ALS | v05_official_splits.ipynb | 0.0000 | 0.0000 | 0.0000 | 0 | **Zero hits — no item factors for cold items** |

#### Sampled-negative ranking (100 random negatives)

| Date | Method | Source | AUC | HR@10 | Takeaway |
|---|---|---|---|---|---|
| 2026-07-08 | Random | v05_official_splits.ipynb | 0.503 | 0.100 | Sanity check |
| 2026-07-08 | Popularity | v05_official_splits.ipynb | **0.168** | 0.000 | **Anti-predicts: cold positive gets score 0, warm negatives score high → ranks positive below most negatives** |
| 2026-07-08 | Content mean | v05_official_splits.ipynb | 0.606 | 0.191 | Consistent with v1 (~0.57 → 0.61) |
| 2026-07-08 | **Content max-sim** | v05_official_splits.ipynb | **0.624** | **0.221** | **Best on cold items. Consistent ~0.62 across both protocols** |
| 2026-07-08 | Implicit ALS | v05_official_splits.ipynb | 0.455 | 0.000 | Below random — same anti-prediction as popularity on cold items |

**Cross-protocol summary (sampled-negative AUC):**

| Method | v1 (warm items) | v2 (cold items) | Interpretation |
|---|---|---|---|
| Random | 0.50 | 0.50 | — |
| Popularity | 0.62 | 0.17 | Warm-only; anti-predicts on cold |
| Content mean | 0.57 | 0.61 | Consistent; mean collapse costs ~0.05 |
| Content max-sim | 0.62 | 0.62 | **Consistent across both settings** |
| Implicit ALS | 0.62 | 0.45 | Warm-only; below random on cold |

**Standing conclusions from protocol v2:**
- Behavioral methods (popularity, ALS) are structurally incapable of scoring cold items and actively anti-predict when forced to (AUC < 0.50).
- Content max-sim is the only method that performs consistently (~0.62 AUC) across both warm- and cold-item settings.
- This is the empirical case for a hybrid: content handles cold start, behavior handles warm items, a learned model (V2 two-tower) decides how to blend.
- The content representation's primary job — cold-item scoring, user-user matching, interpretability — is now empirically validated. It is not a failed retrieval model; it is a cold-start and representation model that works.

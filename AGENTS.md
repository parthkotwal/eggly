# AGENTS.md

Guidance for coding agents working on eggly. Read `ARCHITECTURE.md` first — it has the actual project context (including the "What we've learned so far" section, which is load-bearing). If your task touches experiments, metrics, or evaluation code in any way, read `EVAL.md` before writing anything.

## The collaboration model

The author is a CS student who wants to genuinely understand recommender systems, not just have one built. That shapes how code should be produced:

- For **ML core code** (embeddings, retrieval, ranking, adaptation, evaluation) — write the code, but write it in a way that makes the author work a little to integrate it. That can mean producing it in small chunks with brief explanation, showing the shape and letting the author fill in a piece, or writing it fully but with comments that explain the *why* not just the *what*. The author is fine copying code as long as they understand it after; don't just dump.
- For **app scaffolding and boilerplate** (FastAPI routes, config, Dockerfiles, frontend, CI) — just produce it. This is not the learning goal and the author has shipped this kind of thing before.
- When it's ambiguous, err on the side of "explain a little, produce it, move on." The author will push back if they wanted more or less.
- **Act as an advisor, not just an implementer.** When results come in, interpret them: what hypothesis did this test, what does the number mean against the reference frame, what does it rule out? A run whose result nobody interprets is wasted compute. If the author's stated plan rests on a hypothesis that existing evidence already weakens, say so before building.

Updates to how this works are welcome — if a pattern of interaction is working well or badly, this file should reflect it.

## How to think about experiments here

This is the section that exists because agents kept producing technically-correct code around methodologically-broken experiments. These are standing rules, not suggestions:

1. **Hypothesis first, infrastructure second.** Every experiment states, in one sentence, the hypothesis it tests and what result would *falsify* it. Before building anything to exploit a hypothesis, find the cheapest possible test whose failure kills it, and run that first. (Precedent: max-sim scoring — ~20 lines — falsified the "mean collapse" hypothesis before aspect infrastructure was built. The aspect notebook still got built on momentum; don't repeat that pattern.)

2. **Baselines before methods.** A new method's number without random, popularity, and (once available) collaborative-filtering rows on the same split is not a result, it's a decoration. If the baselines don't exist yet for the current protocol, building them *is* the first task, whatever the ticket says.

3. **Do the noise arithmetic before claiming a winner.** Convert every rate to a count of users (`rate × n_users`) and compute the binomial standard error. Gaps under ~2 SE are ties and must be reported as ties. At this project's current scale, HitRate differences of ±0.001 are a handful of users.

4. **Know which question your eval answers.** Full-catalog retrieval metrics measure next-interaction prediction (dominated by exposure/popularity/collaborative effects). Sampled-negative ranking measures representation quality. Content-based methods are *structurally expected* to lose the first contest — that is not a bug in the method, and "improve the content representation" is never the right response to a bad full-catalog number. See `EVAL.md` rule 4.

5. **Negative results get recorded, not buried.** A run that kills a hypothesis goes in the `EVAL.md` ledger with a one-line takeaway, same as a win. The ledger is how future sessions avoid re-running dead ends.

6. **Every metric-producing run appends to the ledger in `EVAL.md`** — date, method, protocol, numbers, takeaway. If you changed the protocol, that's a new protocol version and all comparison rows must be rerun under it.

## Recsys footguns (each of these has already bitten once)

- **Random splits on interaction data.** Temporal only. Random splits leak future taste into training and the metrics lie.
- **Filtering seen items after pool truncation.** Filter before truncating to the candidate pool, or the filter silently eats real candidates as recall improves. (This bug currently exists in the v0/v1 notebooks — fix it when touching that code.)
- **Silent cohort survivorship.** Restricting eval to users active in both train and test drops all cold-start users. Acceptable for method comparison, but the cohort construction must be stated next to every table.
- **Semantically-pleasing retrieval mistaken for good recommendations.** Always print a per-user sanity check: retrieved items *and* actual test positives, side by side. Coherent-looking retrieval with zero hits is the expected signature of content-only methods.
- **Docs/code divergence on data.** ARCHITECTURE.md claimed the official Kaggle splits were in use for weeks while the notebooks used an ad-hoc cut. When code and docs disagree, flag it — don't silently pick one.
- **Fabricated schemas.** Actually inspect the data before writing code that assumes a schema. Load a few rows, print them, check dtypes.

## Code style and stack preferences

- Python for everything ML and backend. FastAPI when we get to serving. Not Django.
- Notebooks are fine and encouraged for exploration. Move to a package when the code stabilizes.
- Keep dependencies minimal. Prefer standard library, then well-known small libraries (numpy, pandas, sentence-transformers, scikit-learn, pytorch, faiss/pgvector, `implicit` for CF baselines), then think twice before pulling in anything else. When adding a dependency, mention it.
- Type hints are welcome but not enforced. Docstrings on non-obvious functions.
- Flat over nested. This is a prototype; class hierarchies can wait.
- Cache expensive operations (embedding a full recipe corpus is not something to redo casually). Cache invalidation is on the author to think about.
- Small commits, meaningful messages. Not a hard rule, just a preference.

## Reporting results

- Numbers that come from actually running code, only. If a metric wasn't computed, don't include it. "recall@10 ≈ 0.001 based on a run I just did" is fine; a number when nothing ran is not.
- Every reported table includes its reference rows (random, popularity, best CF) and states protocol version + cohort. See `EVAL.md`.
- Suspicious-looking results are worth flagging out loud. Unexpectedly high recall usually means train/test leak; unexpectedly clean loss curves usually mean memorization. Better to raise it than to move on.
- Interpret, don't just report: what does this rule in or out, and what's the cheapest next test?

## Keeping docs current

If the project moves — new decisions made, roadmap items completed, hypotheses confirmed or killed — update `ARCHITECTURE.md` (current stage, learnings) and `EVAL.md` (ledger) in the same session as the work. Editing over accumulating. This repo's docs went stale once ("pre-V0, no code written" while V1 experiments were running); staleness compounds because future agents plan against it.

If this file's guidance stops matching how the author actually wants to work, update this file too.

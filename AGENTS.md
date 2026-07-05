# AGENTS.md

Guidance for coding agents working on eggly. Read `ARCHITECTURE.md` first — it has the actual project context. This file is about how to write code here.

## The collaboration model

The author is a CS student who wants to genuinely understand recommender systems, not just have one built. That shapes how code should be produced:

- For **ML core code** (embeddings, retrieval, ranking, adaptation, evaluation) — write the code, but write it in a way that makes the author work a little to integrate it. That can mean producing it in small chunks with brief explanation, showing the shape and letting the author fill in a piece, or writing it fully but with comments that explain the *why* not just the *what*. The author is fine copying code as long as they understand it after; don't just dump.
- For **app scaffolding and boilerplate** (FastAPI routes, config, Dockerfiles, frontend, CI) — just produce it. This is not the learning goal and the author has shipped this kind of thing before.
- When it's ambiguous, err on the side of "explain a little, produce it, move on." The author will push back if they wanted more or less.

Updates to how this works are welcome — if a pattern of interaction is working well or badly, this file should reflect it.

## Code style and stack preferences

- Python for everything ML and backend. FastAPI when we get to serving. Not Django.
- Notebooks are fine and encouraged for V0. Move to a package when the code stabilizes.
- Keep dependencies minimal. Prefer standard library, then well-known small libraries (numpy, pandas, sentence-transformers, scikit-learn, pytorch, faiss/pgvector), then think twice before pulling in anything else. When adding a dependency, mention it.
- Type hints are welcome but not enforced. Docstrings on non-obvious functions.
- Flat over nested. This is a prototype; class hierarchies can wait.
- Small commits, meaningful messages. Not a hard rule, just a preference.

## Working with data

- Actually inspect the data before writing code that assumes a schema. Load a few rows, print them, check dtypes. Fabricated schemas cause real time loss.
- Temporal splits for evaluation on interaction data, not random splits. This is easy to get wrong and hard to notice.
- Cache expensive operations (embedding a full recipe corpus is not something to redo casually). Cache invalidation is on the author to think about.

## Reporting results

- Numbers that come from actually running code, only. If a metric wasn't computed, don't include it. "recall@10 ≈ 0.3 based on a run I just did" is fine; "recall@10 = 0.34" when nothing ran is not.
- Suspicious-looking results are worth flagging out loud. Unexpectedly high recall usually means train/test leak; unexpectedly clean loss curves usually mean something is being memorized. Better to raise it than to move on.

## Keeping docs current

If the project moves — new decisions made, roadmap items completed, questions answered — `ARCHITECTURE.md` should be updated to reflect that. Editing over accumulating. If you're not sure whether an update is warranted, mention it and let the author decide.

If this file's guidance stops matching how the author actually wants to work, update this file too.
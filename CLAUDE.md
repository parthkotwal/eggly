# eggly — agent entry point

Read in this order:

1. `AGENTS.md` — how to work here: collaboration model (the author is learning recsys — advise and explain, don't just build), experimental method, footguns, style.
2. `ARCHITECTURE.md` — what eggly is (a social platform for taste, **not** a recipe recommender), the Concepts/Instances model, roadmap, and the "What we've learned so far" section (falsified hypotheses live there — don't re-derive or re-litigate them).
3. `EVAL.md` — **mandatory before touching any experiment, metric, or eval code.** Canonical protocol, results ledger (append every run), and the interpretation rules (baselines-first, noise arithmetic, which eval answers which question).

The `run-experiment` skill (`.claude/skills/run-experiment/`) enforces the experiment workflow — it should trigger automatically on any experiment/eval task.

Current stage: **V0.5 complete → V1 next** (reference frame built, both evals running, both protocols done). See ARCHITECTURE.md roadmap.

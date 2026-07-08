---
name: run-experiment
description: Use whenever designing, implementing, running, or reporting a recommender experiment or evaluation in eggly — adding a method or baseline, computing Recall/HitRate/nDCG/AUC, comparing approaches, modifying eval or split code, or interpreting metric changes. Enforces the repo's experimental method (hypothesis → cheapest falsifying test → baselines → noise check → ledger).
---

# Running an experiment in eggly

Read `EVAL.md` (protocol, ledger, interpretation rules) before writing any code. This skill is the workflow; EVAL.md is the law.

## Before writing code

1. **State the hypothesis in one sentence, plus the result that would falsify it.** If you can't write the falsification condition, the experiment isn't designed yet — say so to the author instead of coding.
2. **Find the cheapest test.** Ask: is there a ≤50-line version of this that would kill the hypothesis if it fails? Run that before building any infrastructure. Precedent: max-sim (20 lines) falsified "mean collapse" before aspect infrastructure was justified — the infrastructure got built anyway and taught nothing extra. Don't repeat that.
3. **Check the ledger first.** If EVAL.md already contains a run that answers (or bounds) this question, report that instead of rerunning. The aggregation family (mean / max-sim / aspect-grouped means) is already known to be exhausted under protocol v1.
4. **Confirm the reference frame exists.** Random + popularity (+ best CF once built) rows must exist for the current protocol. If they don't, building them is step zero, regardless of the requested task.

## While implementing

- Reuse the canonical split/protocol from EVAL.md — never invent a new split inside a notebook. A protocol change is a deliberate decision: bump the protocol version and rerun all comparison rows.
- Temporal splits only. Same split object and same cohort for every method in a comparison.
- Filter seen items **before** truncating to the candidate pool.
- Include the leakage sanity check (train items in top-K == 0.00) in the run output.
- Include a per-user qualitative check: for at least one user, print retrieved items next to their actual test positives. Semantically coherent retrieval with zero hits is expected for content-only methods — show it rather than hiding it.

## When reporting

1. **One table, reference rows included**, protocol version and cohort size stated above it.
2. **Do the noise arithmetic in the report itself:** rate × n_users = hit count; binomial SE; any gap under ~2 SE is written as a tie. Never write "X beats Y" across a sub-2-SE gap.
3. **Name the question the eval answered:** full-catalog retrieval = next-interaction prediction (behavioral); sampled-negative ranking = representation quality. Don't judge a representation by the first or claim retrieval wins from the second.
4. **Append the run to the EVAL.md ledger** — date, method, source notebook/script, numbers, one-line takeaway — including (especially) negative results.
5. **Interpret:** state what is now ruled in/out and the cheapest next test. If the result kills the hypothesis, say plainly that the follow-on infrastructure is no longer justified.

## Red flags to raise immediately

- Recall/nDCG that jumps an order of magnitude → suspect leakage before celebrating.
- A method comparison where all numbers are within noise → the eval lacks sensitivity; propose sampled-negative eval rather than more method variants.
- A request to refine the content representation to improve full-catalog interaction metrics → structurally unpromising (EVAL.md rule 4); push back with the reasoning and offer the behavioral-baseline path.

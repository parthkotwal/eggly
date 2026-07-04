# Eggly ML Restart

This repo has been reset around the ML and learning work.

## Current workspace

- `ml/config.py`: domain vocabulary and hand-authored priors.
- `ml/service.py`: feature-vector construction, similarity scoring, constraints, and match explanations.
- `ml/synthetic.py`: synthetic profile generation from the priors.
- `ml/generate_candidates.py`: CLI for generating local candidate datasets.
- `ml/manual_match.py`: CLI smoke test for matching a sample profile.
- `ml/notebooks/`: prior notebook explorations retained for reference.

## Archived app shell

The old Django, Next.js, Docker, editor, and environment scaffolding was moved to:

`archive/legacy-app-20260703/`

That archive is intentionally kept out of the active project shape so the ML work can restart without the old app structure driving decisions.

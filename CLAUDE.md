# Ilmastoneuvottelupeli — Project Notes for Claude

## What this is

A Streamlit classroom app for grades 7–9 climate negotiation simulation.
Run with `streamlit run app.py`. Conda env in `environment.yml` (Python 3.11, FaIR 2.2).

## Publishing goal

**The intended end state is a publicly accessible website hosted on Streamlit Community Cloud
(streamlit.io/cloud), with the source code released on a public GitHub repository.**

This means: when developing new features, prefer approaches that are compatible with
cloud deployment on Streamlit Community Cloud. Specifically:
- Avoid local file paths that would break in a cloud environment
- Keep dependencies on PyPI (pip-installable), not just conda-forge
- If adding precomputed data files, they must be committable to git or generatable on startup
- No secrets or credentials hardcoded — use `python-dotenv` / environment variables
- The app should start cleanly with `streamlit run app.py` from a fresh clone

The app is not yet ready for release. The plan for release is saved at
`.claude/plans/what-are-my-options-serene-origami.md`.

## Pipeline

- `precompute_configs.py` — screens FaIR ensemble members, generates config data files
- `fair_runner.py` — FaIR v2.2 wrapper used by the app
- `app.py` — Streamlit UI (main entry point)
- `make_presentation_figure.py` — standalone figure generation utility

## Scientific conventions

- Temperature anomalies relative to 1850–1900 baseline
- CO₂ emissions in GtCO₂/yr
- FaIR posterior from fair-calibrate v1.4, all-2022 constraint set
- 2024 warming target: within ±0.10 °C of IGCC 2024 observed value

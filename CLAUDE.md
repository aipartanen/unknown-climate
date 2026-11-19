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

- `precompute_configs.py` — screens FaIR ensemble members, generates `data/screened_configs.csv`
  (rule and rationale are documented in the script header)
- `fair_runner.py` — FaIR v2.2 wrapper used by the app (also used by the screening, so the
  seeded stochastic runs shown in the game are the ones that were screened)
- `app.py` — Streamlit UI (main entry point)
- `make_presentation_figure.py` — standalone figure generation utility

## Scientific conventions

- Temperature anomalies relative to 1850–1900 baseline
- CO₂ emissions in GtCO₂/yr
- FaIR 2.2.4 with the fair-calibrate v1.6.0 constrained posterior (841 members), vendored in
  `data/fair_calibrate/` (copied from ghgbudgets)
- FaIR temperature is on timebounds (1 Jan); IGCC observations are annual means at mid-year.
  Annual mean of year Y = mean of timebounds Y and Y+1.
- Observations: IGCC-2025a annual GMST 1850–2025 in `data/IGCC_GMST_1850-2025.csv` (Zenodo
  10.5281/zenodo.21494229). The calibration used the earlier 1850–2024 vintage (differs by ~0.01 °C).
- Players set only fossil CO₂ (CO₂ FFI). Land-use CO₂ (AFOLU) and all other species follow SSP2-4.5.
  CO₂ FFI: SSP2-4.5 to 2022, observed global totals for 2023–24 (GCB via OWID), regional baselines
  for 2025 (`REGION_EMISSIONS_BASE_GTCO2`, GCB 2025 projection), pledge ramp 2025→2040.
- Screening (continuity with observations after 2025): keep members whose seeded stochastic 2026
  annual mean is within the 90th percentile of the historical misfit (≈0.25 °C) of the observed
  2023–25 mean (IGCC). No ECS trim.

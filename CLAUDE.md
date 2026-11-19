# Ilmastoneuvottelupeli — Project Notes for Claude

## What this is

A Streamlit classroom app for grades 7–9 climate negotiation simulation (also usable with adults).
Run with `streamlit run app.py`; instructor slider mode with `streamlit run app.py -- --sliders`.
Conda env in `environment.yml` (Python 3.11, FaIR 2.2). UI text is Finnish.

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
- `precompute_estimates.py` — fits a per-member linear emulator (T = a + b·cumulative CO₂ FFI) of
  the annual-mean temperature for every year 2031–2045 and 2100, writes `data/estimate_emulator.csv`.
  Gives the app ensemble estimates without running FaIR. Re-run (~5 min) after any change to the
  emissions setup, screening or targets; run `precompute_configs.py` first.
- `app.py` — Streamlit UI (main entry point)
- `make_presentation_figure.py` — standalone figure generation utility

## App structure

Eight screens in order (`SCREENS` in `app.py`): pledges → prior → draw → simulation 2040 →
observation 2040 → research → net-zero years (Phase 2) → reveal. The impacts page opens on top of
any screen. Pledges lock once a member is drawn (the observation depends on them).

The game is about uncertainty: one screened member is drawn uniformly and its ECS stays hidden until
the reveal. Estimates are the ensemble reweighted by Bayes (`posterior_weights`); the scientific
choices are commented by the constants in `app.py` (`OBS_WINDOW`, `OBS_2040_SD`, `ECS_CLUE_SD`,
`CENTRED_WINDOW`). In short:
- Update 1: observed 2031–2040 mean, noise SD 0.05 °C. Member internal variability is not added
  (it is part of the member's identity, fixed seed). Usually narrows ECS only ~20 %.
- Update 2: research estimate of ECS, noise SD 0.5 °C. Weights of the two updates multiply.
- Both noise draws happen once per game (at the first simulation), so revisiting screens cannot
  average them away.
- Paris verdict at 2040 uses the decade centred on 2040 (2035–2044): known years 2035–2040 plus
  2041–2044 projected with the update-1 weights (not the prior, which is less accurate).
- Exceedance probabilities (1.5/2/3/4 °C in 2100) are sums of the current weights.

## Scientific conventions

- Temperature anomalies relative to 1850–1900 baseline
- CO₂ emissions in GtCO₂/yr
- FaIR 2.2.4 with the fair-calibrate v1.6.0 constrained posterior (841 members), vendored in
  `data/fair_calibrate/` (copied from ghgbudgets)
- FaIR temperature is on timebounds (1 Jan); IGCC observations are annual means at mid-year.
  Annual mean of year Y = mean of timebounds Y and Y+1. Everything the game shows or uses is an
  annual mean (`annual_series()` / `annual_mean()` in `fair_runner.py`), plotted at mid-year
  (Y + 0.5). `cumulative_co2` sums emissions through year Y, since timebound Y+1 depends on them.
- Observations: IGCC-2025a annual GMST 1850–2025 in `data/IGCC_GMST_1850-2025.csv` (Zenodo
  10.5281/zenodo.21494229). The calibration used the earlier 1850–2024 vintage (differs by ~0.01 °C).
- Players set only fossil CO₂ (CO₂ FFI). Land-use CO₂ (AFOLU) and all other species follow SSP2-4.5.
  CO₂ FFI: SSP2-4.5 to 2022, observed global totals for 2023–24 (GCB via OWID), regional baselines
  for 2025 (`REGION_EMISSIONS_BASE_GTCO2`, GCB 2025 projection), pledge ramp 2025→2040.
- Screening (continuity with observations after 2025): keep members whose seeded stochastic 2026
  annual mean is within the 90th percentile of the historical misfit (≈0.25 °C) of the observed
  2023–25 mean (IGCC). No ECS trim.

## UI conventions

- Light theme pinned in `.streamlit/config.toml` (projector readability). Matplotlib charts use
  `CHART_BG`/`CHART_FG`; level colours get darker shades for text via `TEXT_SHADE`.
- Finnish decimal commas in everything displayed: `_fi()` for text, `COMMA_TICKS` for matplotlib,
  a `labelExpr` for Altair axes.
- The temperature chart is Altair (`_temperature_chart`), animated by redrawing it in a placeholder
  once per result. Do not reintroduce `st.components.v1.html` (deprecated).
- Slider mode shows the 2025→2040 change in emissions (same sign as the radio labels), range
  −100…+80 %; internally the cut fraction is still used.

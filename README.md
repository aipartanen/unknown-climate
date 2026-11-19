# Ilmastoneuvottelupeli

Interactive classroom demo for grades 7–9 on climate negotiations and the Paris Agreement.
Built with [Streamlit](https://streamlit.io/) and the [FaIR v2.2](https://docs.fairmodel.net/) climate model.

## What it does

Students are divided into five groups, each representing a major emitting region:
**China, USA, EU, India, Rest of World**.

The game runs as a sequence of screens. Its second theme is uncertainty: the climate
sensitivity of the simulated Earth is unknown to the players and has to be learned.

1. **Pledges** — each group sets a fossil CO₂ cut for 2040, relative to 2025 emissions.
2. **Prior estimate** — histograms of ECS and 2100 warming over the whole model ensemble,
   with the probability of exceeding 1.5, 2, 3 and 4 °C.
3. **Draw** — one ensemble member ("one possible Earth") is drawn at random. Its ECS stays
   hidden, and the pledges are locked.
4. **Simulation to 2040** — FaIR run of the drawn member. The single year 2040 is shown next to
   the mean of the decade centred on 2040 (2035–2044; the years after 2040 are projected).
   The Paris thresholds are judged on the decade mean.
5. **Observation 2040** — the observed 2031–2040 mean updates the estimates (Bayesian
   reweighting of the ensemble). Usually this narrows them only a little.
6. **Research advances** — a noisy research estimate of ECS narrows them clearly.
7. **Net-zero years** — each group decides when their region's CO₂ emissions reach net zero,
   using the updated estimates. A second FaIR run shows the trajectory to peak warming.
8. **Reveal** — the drawn Earth's true ECS and 2100 warming against the estimates.

An impacts page (what 1.5–4 °C means globally and in Finland, with the probability of each
level under the players' decisions) can be opened from every screen. All temperatures are
annual means relative to 1850–1900.

## Climate model

Each game session draws one configuration at random from a pre-screened shortlist of
FaIR posterior ensemble members (fair-calibrate v1.6.0, FaIR 2.2.4; 841 members).
Screening is only for visual continuity with the observations: a member is kept if its 2026
annual-mean temperature is within about 0.25 °C of the mean of the observed (IGCC-2025a) 2023–2025
temperatures. The rule and its rationale are documented in `precompute_configs.py`.
Only fossil (fuel and industry) CO₂ is set by the players. Observed global totals are used for
2023–2024 and regional baselines for 2025 (Global Carbon Budget, via Our World in Data).
Land-use CO₂ and all non-CO₂ forcing follow the SSP2-4.5 scenario throughout.
Runs are stochastic with fixed per-member seeds, so a member always gives the same curve.

## Setup

### Prerequisites

- [Mamba](https://mamba.readthedocs.io/) or Conda
- fair-calibrate v1.6.0 data, included in `data/fair_calibrate/` (see the README there)

### Install

```bash
mamba env create -f environment.yml
mamba activate climate-game
```

### Pre-compute screened configs (once)

```bash
python precompute_configs.py
```

This writes `data/screened_configs.csv` (736 configs) and `figures/screening_diagnostics.png`.
Re-run if you change the screening criteria. The CSV is committed so the app starts from a fresh clone.

### Run

```bash
streamlit run app.py
```

Open the printed URL in a browser (or project it for the class).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (Finnish text) |
| `fair_runner.py` | FaIR wrapper: emissions builder, single-config run, estimate helpers |
| `precompute_estimates.py` | Offline emulator fit → `data/estimate_emulator.csv` (estimates without FaIR) |
| `precompute_configs.py` | Offline screening script → `data/screened_configs.csv` |
| `environment.yml` | Conda environment definition |
| `data/screened_configs.csv` | Screened member shortlist; regenerate with `precompute_configs.py` |
| `data/fair_calibrate/` | fair-calibrate v1.6.0 files used by the model (copied from ghgbudgets) |

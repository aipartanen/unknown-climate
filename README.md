# Ilmastoneuvottelupeli

Interactive classroom demo for grades 7–9 on climate negotiations and the Paris Agreement.
Built with [Streamlit](https://streamlit.io/) and the [FaIR v2.2](https://docs.fairmodel.net/) climate model.

## What it does

Students are divided into five groups, each representing a major emitting region:
**China, USA, EU, India, Rest of World**.

The game runs in two phases:

**Phase 1 — Emission pledges to 2040**
Each group sets a CO₂ reduction target for their region (0–100 %, in 5 % steps).
After all groups agree, the teacher clicks *Simuloi!* to run a FaIR simulation.
The app reveals the current modelled warming (2024) and the projected temperature in 2040,
and shows which Paris Agreement threshold the combined pledges hit.
At this point the climate model's Equilibrium Climate Sensitivity (ECS) is also revealed,
giving students context for Phase 2.

**Phase 2 — Net-zero years**
Each group decides when their region's CO₂ emissions reach net zero (2040–2100).
A second FaIR run shows the full temperature trajectory to peak warming,
including how it compares to the Phase 1 scenario.

## Climate model

Each game session draws one configuration at random from a pre-screened shortlist of
FaIR posterior ensemble members (fair-calibrate v1.4, all-2022 constraint set).
Screening criteria: modelled 2024 warming within ±0.10 °C of the IGCC 2024 observed value
(1.36 °C above 1850–1900) and ECS within the IPCC AR6 very likely range (2.0–5.0 °C).
Non-CO₂ forcing follows the SSP2-4.5 scenario throughout.
All runs are deterministic (`stochastic_run=False`).

## Setup

### Prerequisites

- [Mamba](https://mamba.readthedocs.io/) or Conda
- fair-calibrate data at the path configured in `fair_runner.py` (`FAIR_CAL_DIR`)

### Install

```bash
mamba env create -f environment.yml
mamba activate climate-game
```

### Pre-compute screened configs (once)

```bash
python precompute_configs.py
```

This writes `data/screened_configs.csv` (~434 configs). Re-run if you change the screening criteria.

### Run

```bash
streamlit run app.py
```

Open the printed URL in a browser (or project it for the class).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (Finnish text) |
| `fair_runner.py` | FaIR wrapper: emissions builder, single-config run |
| `precompute_configs.py` | Offline screening script → `data/screened_configs.csv` |
| `environment.yml` | Conda environment definition |
| `data/screened_configs.csv` | Gitignored; regenerate with `precompute_configs.py` |

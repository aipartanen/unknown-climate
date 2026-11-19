# Observational uncertainty: options for later

Notes from 2026-11-19. Status: **option A is implemented** (commit 626f827). Option B is open.

## Current state (option A: band around the true line)

- The chart shows the drawn climate's **true** annual line and a 95 % HadCRUT5 band around it
  and around the IGCC record (`precompute_obs_error.py` → `data/obs_uncertainty.csv`,
  `_load_obs_band()` and the band layer in `_temperature_chart` in `app.py`).
- Band = HadCRUT5 200-member ensemble plus coverage uncertainty, **rebaselined to 1850–1900**.
  Half-width is ~0.08 °C for recent years (1σ ≈ 0.043), mostly baseline error. From 2025 onwards
  the 2015–2024 mean (0.084 °C) is used, because HadCRUT 5.0.2.0's 2025 looks provisional (its
  coverage σ is 0.068 vs ~0.004 in the neighbouring years).
- Update 1 (Bayes) still uses `observed = true 2031–2040 mean + N(0, OBS_2040_SD = 0.05)`. The
  noisy value appears only in the update-1 text, not on the chart.
- Design rule so far: noisy observations are not plotted (worry: a fake 1.5/2 °C crossing).

Known inconsistency: the chart shows the true line, so the exact 2031–2040 mean can in
principle be read off it, while the text gives a different "observed" mean. The two differ by up
to ~0.1 °C at σ = 0.05.

## Option B: show only observations; the same series feeds the ECS inference

Students see one observed series, update 1 uses exactly that series, and the truth is hidden
until the reveal.

### Key scientific point: which error is relevant

FaIR members are calibrated against the observed record, baseline included. In the game the
"true" member temperature is therefore already on the observational scale, and the 1850–1900
baseline error largely cancels. What matters is the error in **recent change**.

HadCRUT5 errors for 2015–2024, coverage uncertainty included (computed 2026-11-19):

| Errors relative to | annual σ (°C) | decade-mean σ (°C) |
|---|---|---|
| 1850–1900 (the current band) | 0.043 | 0.040 |
| 1961–1990 | 0.018 | 0.012 |
| 1995–2024 | 0.014 | 0.005 |

The year-to-year part is about 0.013 °C in every case. The rest is a persistent offset that
depends on the baseline.

Consequences:
- `OBS_2040_SD = 0.05` is **larger than the data support** for "error relative to what the model
  was fitted to" (about 0.005–0.012 for a decade mean). Keep 0.05 as a deliberate teaching
  choice, or derive about 0.01–0.02 from data. Either way, say which in the `app.py` comment.
- With σ about 0.01, update 1 nearly pins the drawn climate's decade mean. ECS still narrows only
  modestly, because members with different ECS give the same 2031–2040 mean.
- The ±0.08 °C band is fair for "how well do we know warming since preindustrial". It
  overstates "how far are the points from the model truth".

### How to implement

1. **Error model** (extend `precompute_obs_error.py`): fit an AR(1) to HadCRUT errors
   rebaselined to a recent period (annual σ ≈ 0.014 plus its lag-1 autocorrelation). Save the
   parameters and the implied decade-mean σ. AR(1) works for any year, after 2040 too, unlike
   copying HadCRUT errors from earlier years.
2. **`_simulate_phase1`**: draw one error series per game (with the other noise draws) and store
   `obs_phase1 = true + error` for 2026–2040. Update-1 observation = mean of `obs_phase1` over
   `OBS_WINDOW`. Likelihood σ = decade-mean σ from step 1 (replaces the hand-set constant).
3. **Chart**: up to 2040, plot `obs_phase1` and draw the band around it (meaning: "the truth is
   in here"). Do not draw the true line.
4. **Paris decade**: known years 2035–2040 from `obs_phase1`, 2041–2044 projected from the
   weighted members. Add the observation error to the interval in quadrature (tiny).
5. **Single-year metric** (`s1_year`): the observed 2040 value.
6. **Reveal**: overlay the true line and the observed points, plus one sentence on the size of
   this game's observation error.
7. **Texts** (FI/EN) for the new legend entries and the reveal sentence.

### Complications / decisions

- **After 2040 (Phase 2 and reveal charts):** either keep "observing" (continue the AR(1)
  error), which is consistent but hides the true line until the reveal, or switch to the true
  line after 2040, which is simpler but mixes two kinds of series in one chart.
- **Threshold verdicts** (Paris texts at 2040) are judged on observed values. With σ ≈ 0.014 per
  year, a false crossing needs the truth within ~0.03 °C of the threshold. That is rare, and the
  reveal can explain it. With σ = 0.05 it would be a real risk (the original worry).
- **Join at 2025:** the IGCC record ends 2025 and the game's observations start in 2026. Because
  the error is relative to a recent baseline, there is no artificial step.
- **Re-checks:** re-run the offline checks quoted in the `app.py` comments for the new σ: the
  ECS narrowing (~80 % of the prior with update 1, 400 games) and the Paris-decade error SD
  (0.038 °C with both updates, 1200 games). Both are emulator-only runs.
- **The band's meaning:** if option B is adopted, consider making the band itself use the
  recent-baseline error, so that the band and the inference agree.

### Effort

Moderate, about half a day: precompute extension, ~40–60 lines in `app.py`, a few texts.
Decide first: (1) error size, 0.05 vs about 0.01–0.02; (2) what the charts show after 2040.

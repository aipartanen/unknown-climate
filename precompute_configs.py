#!/usr/bin/env python3
"""
Screen the 841-member FaIR posterior (fair-calibrate v1.6.0) so that the simulated
temperature continues seemingly consistently with the observations at the point where the
observed record ends.

Run once offline:
    python precompute_configs.py

Produces: data/screened_configs.csv  (config_id, T_2026, dev_from_ref, ECS)
          figures/screening_diagnostics.png

PURPOSE
-------
The game shows one posterior member's temperature curve next to the observed GMST. The
screening exists only so that this curve does not visibly jump away from the observations
right after the observed record ends (2025). It is NOT meant to constrain the projections:
the posterior already is constrained (RMSE against observed GMST, ECS, TCR, ocean heat, ...).
Nothing before 2026 is screened on, and ECS is deliberately not trimmed.

RULE
----
Keep member i if   |T_2026,i - ref| <= limit,   where
  T_2026,i  member's 2026 annual mean (seeded stochastic run, 1850-1900 baseline)
  ref       mean of the observed annual GMST 2023-2025 (IGCC-2025a)
  limit     90th percentile of the historical misfit |T_y,i - mean(obs of the 3 preceding
            years)| pooled over all members and years y = 1977...2025

WHY THESE CHOICES
-----------------
* Time convention. FaIR temperature sits on timebounds (1 January); IGCC values are annual
  means stamped at mid-year. The annual mean of year Y is the average of timebounds Y and
  Y+1 (the calibration likewise compares observation Y.5 with the timebound after it). The
  2026 annual mean therefore needs timebounds 2026 and 2027.
* Reference = 3-year mean, not the last year alone. A single year is boosted or suppressed by
  El Nino / La Nina; in the earlier screening (reference 2022-24, 1.370 degC) screening on 2024
  alone (1.523 degC) selected members whose noise happened to be high and shifted the passing
  members' median ECS from 2.99 to 3.16-3.28 degC, whereas the 3-year mean left it unchanged.
  The window is now 2023-25 (1.447 degC). It contains the El Nino-boosted 2023-24, so it sits
  ~0.08 degC above the IGCC human-induced warming of 2025 (1.37 degC). CHECK the printed ECS
  quantiles of the passing members against those of all members for a shift.
* Limit from the historical model-minus-observed misfit, not from observed one-year steps.
  FaIR has LESS year-to-year variability than the observations (std of one-year steps
  0.071 vs 0.126 degC, 1975-2023) but MORE persistent variability around the trend (detrended
  std 0.112 vs 0.097 degC). Members therefore wander further from the observed level than the
  observed one-year steps suggest, and the observed steps would be too strict a yardstick.
  The misfit distribution is what the calibration itself accepted (RMSE < 0.19 degC).
* The historical comparison mirrors the 2026 comparison (member year vs mean of the 3
  preceding observed years), so the forced-trend lag is included consistently.
* 90th percentile: removes clear outliers but keeps most of the ensemble.
* Emissions are the game's: observed CO2 FFI totals for 2023-24 and the 2025 baseline held flat
  (no cuts; fair_runner.build_co2_fossil_trajectory). The game differs only in 2026 (the
  pledge ramp has started; at most ~2.6 GtCO2 below flat for a 100 % cut by 2040), which changes
  the 2026 annual mean by ~0.001 degC. Other species: SSP2-4.5.

Results of the earlier screening (reference 2022-24, screening year 2025, SSP2-4.5 emissions):
ref 1.370, limit ~0.253 degC, 736 of 841 members pass, passing ECS 5/50/95% = 2.0 / 2.99 / 4.9.
Results of the current rule: printed by this script (compare ECS shift as noted above).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fair_runner import (
    FIRST_OVERRIDE_YEAR,
    SCREENED_CONFIGS_PATH,
    annual_mean,
    build_co2_fossil_trajectory,
    load_observed_gmst,
    load_posterior,
    rebase_to_1850_1900,
    setup_fair,
)

# Last timebound simulated. The 2026 annual mean needs timebounds 2026 and 2027; the run is
# extended to 2037 only so the diagnostic figure can show the lines up to 2036.
YEAR_END = 2037
SCREEN_YEAR = 2026               # first year after the observed record (IGCC ends with 2025)
REF_YEARS = (2023, 2024, 2025)   # observed years averaged to form the reference
N_PRECEDING = len(REF_YEARS)     # window length in the historical misfit statistic
HIST_FIRST_YEAR = 1977           # first year of the historical misfit pool (~ last 50 years,
                                 # when forced warming dominates the record)
LIMIT_PERCENTILE = 90
FIG_PATH = Path('figures') / 'screening_diagnostics.png'


def main():
    FIG_PATH.parent.mkdir(exist_ok=True)

    df_post = load_posterior()
    config_ids = list(df_post.index)
    print(f'{len(config_ids)} posterior members; running seeded stochastic FaIR to {YEAR_END} ...')

    f = setup_fair(config_ids, YEAR_END)
    # Same CO2 FFI as the game, with no pledged cuts (see header): observed totals 2023-24, then flat.
    trajectory = build_co2_fossil_trajectory({}, None, YEAR_END)
    co2_idx = list(f.emissions.specie.values).index('CO2 FFI')
    f.emissions.values[FIRST_OVERRIDE_YEAR - 1750:, 0, :, co2_idx] = trajectory[FIRST_OVERRIDE_YEAR - 1750:, None]
    f.run()
    temp = rebase_to_1850_1900(f.temperature.sel(layer=0, scenario='game'))   # (timebounds, config)

    # ECS from FaIR's analytical EBM property (scenario independent)
    ecs = f.ebms['ecs'].to_series()
    ecs.index = config_ids

    df_obs = load_observed_gmst()
    obs = dict(zip(np.floor(df_obs['year'].values).astype(int), df_obs['gmst'].values))

    # --- reference: mean observed 2023-25 ---
    ref = float(np.mean([obs[y] for y in REF_YEARS]))

    # --- limit: pooled historical misfit ---
    misfit = []
    for year in range(HIST_FIRST_YEAR, SCREEN_YEAR):
        obs_ref = np.mean([obs[y] for y in range(year - N_PRECEDING, year)])
        misfit.append(np.abs(annual_mean(temp, year).values - obs_ref))
    misfit = np.concatenate(misfit)
    limit = float(np.percentile(misfit, LIMIT_PERCENTILE))

    # --- apply to the screening year ---
    t_screen = annual_mean(temp, SCREEN_YEAR).values
    dev = t_screen - ref
    passed = np.abs(dev) <= limit

    # Order-of-magnitude checks against the study that defined the rule
    print(f'\nObserved reference (mean 2023-25): {ref:.3f} degC')
    print(f'Historical misfit pool: n={len(misfit)}, {LIMIT_PERCENTILE}th percentile limit = {limit:.3f} degC')
    print(f'Member T_{SCREEN_YEAR} min/median/max: {t_screen.min():.3f} / {np.median(t_screen):.3f} / {t_screen.max():.3f} degC')
    print(f'Members passing: {passed.sum()} / {len(passed)}')
    print('ECS 5/50/95% all:     ' + ' / '.join(f'{v:.2f}' for v in np.percentile(ecs, [5, 50, 95])))
    print('ECS 5/50/95% passing: ' + ' / '.join(f'{v:.2f}' for v in np.percentile(ecs[passed], [5, 50, 95])))
    assert abs(ref - 1.447) < 0.001, 'observed 2023-25 reference changed'
    assert 0.20 < limit < 0.30, 'misfit limit outside the expected ~0.25 degC'
    assert 600 < passed.sum() < 841, 'unexpected number of passing members'

    screened = pd.DataFrame({
        f'T_{SCREEN_YEAR}': t_screen[passed],
        'dev_from_ref': dev[passed],
        'ECS': ecs.values[passed],
    }, index=pd.Index(np.array(config_ids)[passed], name='config_id'))
    SCREENED_CONFIGS_PATH.parent.mkdir(exist_ok=True)
    screened.to_csv(SCREENED_CONFIGS_PATH)
    print(f'\nSaved {len(screened)} configs to {SCREENED_CONFIGS_PATH}')

    # --- diagnostic figure ---
    idx_pass = np.where(passed)[0]
    i_hi = idx_pass[np.argmax(dev[idx_pass])]
    i_lo = idx_pass[np.argmin(dev[idx_pass])]

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    ax = axs[0, 0]
    ax.hist(misfit, bins=40, color='#5A84E2', alpha=0.7)
    ax.axvline(limit, color='k', linestyle='--', label=f'{LIMIT_PERCENTILE}th percentile = {limit:.2f} °C')
    ax.set_xlabel('|member annual mean − mean of 3 preceding observed years| (°C)')
    ax.set_ylabel('Member-years')
    ax.set_title(f'Historical misfit {HIST_FIRST_YEAR}–{SCREEN_YEAR - 1} (defines the limit)')
    ax.legend()

    ax = axs[0, 1]
    ax.hist(dev, bins=40, color='#5A84E2', alpha=0.7)
    ax.axvline(-limit, color='k', linestyle='--')
    ax.axvline(limit, color='k', linestyle='--', label=f'± {limit:.2f} °C')
    ax.set_xlabel(f'T_{SCREEN_YEAR} − observed 2023–25 mean ({ref:.2f} °C)')
    ax.set_ylabel('Members')
    ax.set_title(f'{SCREEN_YEAR} deviation: {passed.sum()} of {len(passed)} members pass')
    ax.legend()

    ax = axs[1, 0]
    bins = np.linspace(1.2, 8, 35)
    ax.hist(ecs, bins=bins, color='lightgrey', density=True, label=f'all {len(ecs)} (median {ecs.median():.2f})')
    ax.hist(ecs[passed], bins=bins, histtype='step', linewidth=2, color='#E0792B', density=True,
            label=f'passing {passed.sum()} (median {np.median(ecs[passed]):.2f})')
    ax.set_xlabel('ECS (°C)')
    ax.set_ylabel('Density')
    ax.set_title('ECS before and after screening (no ECS trim)')
    ax.legend()

    ax = axs[1, 1]
    years = np.arange(1950, YEAR_END)
    series = np.array([annual_mean(temp, y).values for y in years])   # (years, members)
    ax.fill_between(years + 0.5, np.percentile(series[:, passed], 5, axis=1),
                    np.percentile(series[:, passed], 95, axis=1), color='#5A84E2', alpha=0.2,
                    label='passing members 5–95%')
    ax.plot(years + 0.5, series[:, i_hi], color='#D1495B',
            label=f'highest passing ({config_ids[i_hi]}, ECS {ecs.values[i_hi]:.2f})')
    ax.plot(years + 0.5, series[:, i_lo], color='#2E86AB',
            label=f'lowest passing ({config_ids[i_lo]}, ECS {ecs.values[i_lo]:.2f})')
    obs_years = np.array(sorted(obs))
    mask = obs_years >= 1950
    ax.plot(obs_years[mask] + 0.5, [obs[y] for y in obs_years[mask]], color='k', marker='o',
            markersize=3, label='IGCC observed')
    ax.errorbar([SCREEN_YEAR + 0.5], [ref], yerr=[[limit], [limit]], color='k', capsize=5, linewidth=1.5,
                label='reference ± limit')
    ax.axvline(SCREEN_YEAR, color='grey', linestyle=':', linewidth=1)
    ax.set_xlabel('Year (annual means at mid-year)')
    ax.set_ylabel('Temperature anomaly vs 1850–1900 (°C)')
    ax.set_title('Passing members; game emissions (observed 2023-24, then flat)')
    ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f'Saved {FIG_PATH}')


if __name__ == '__main__':
    main()

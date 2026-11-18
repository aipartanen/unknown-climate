#!/usr/bin/env python3
"""
Screen the 841-member FaIR posterior (fair-calibrate v1.6.0) so that the simulated
temperature continues seemingly consistently with the observations at the point where the
observed record ends.

Run once offline:
    python precompute_configs.py

Produces: data/screened_configs.csv  (config_id, T_2025, dev_from_ref, ECS)
          figures/screening_diagnostics.png

PURPOSE
-------
The game shows one posterior member's temperature curve next to the observed GMST. The
screening exists only so that this curve does not visibly jump away from the observations
right after the observed record ends (2024). It is NOT meant to constrain the projections:
the posterior already is constrained (RMSE against observed GMST, ECS, TCR, ocean heat, ...).
Nothing before 2025 is screened on, and ECS is deliberately not trimmed.

RULE
----
Keep member i if   |T_2025,i - ref| <= limit,   where
  T_2025,i  member's 2025 annual mean (seeded stochastic run, 1850-1900 baseline)
  ref       mean of the observed annual GMST 2022-2024 (IGCC)
  limit     90th percentile of the historical misfit |T_y,i - mean(obs of the 3 preceding
            years)| pooled over all members and years y = 1977...2024

WHY THESE CHOICES
-----------------
* Time convention. FaIR temperature sits on timebounds (1 January); IGCC values are annual
  means stamped at mid-year. The annual mean of year Y is the average of timebounds Y and
  Y+1 (the calibration likewise compares observation Y.5 with the timebound after it). The
  2025 annual mean therefore needs timebounds 2025 and 2026.
* Reference = 2022-24 mean (1.370 degC), not 2024 alone (1.523 degC). 2024 was boosted by
  El Nino; screening on it selects members whose noise happened to be high and shifts the
  passing members' median ECS from 2.99 to 3.16-3.28 degC. The 3-year mean leaves the ECS
  distribution essentially unchanged (median 2.99).
* Limit from the historical model-minus-observed misfit, not from observed one-year steps.
  FaIR has LESS year-to-year variability than the observations (std of one-year steps
  0.071 vs 0.126 degC, 1975-2023) but MORE persistent variability around the trend (detrended
  std 0.112 vs 0.097 degC). Members therefore wander further from the observed level than the
  observed one-year steps suggest, and the observed steps would be too strict a yardstick.
  The misfit distribution is what the calibration itself accepted (RMSE < 0.19 degC).
* The historical comparison mirrors the 2025 comparison (member year vs mean of the 3
  preceding observed years), so the forced-trend lag is included consistently.
* 90th percentile: removes clear outliers but keeps most of the ensemble.
* Screening on SSP2-4.5 emissions, whereas the game overrides CO2 FFI from 2023. The
  difference over 2023-2025 is a few tenths of GtCO2 per year and is negligible for 2025.

Expected results (from the study that led to this rule): ref 1.370, limit ~0.253 degC,
736 of 841 members pass, passing ECS 5/50/95% = 2.0 / 2.99 / 4.9 degC.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fair_runner import (
    SCREENED_CONFIGS_PATH,
    annual_mean,
    load_observed_gmst,
    load_posterior,
    rebase_to_1850_1900,
    setup_fair,
)

# Last timebound simulated. 2025 annual mean needs timebounds 2025 and 2026; the run is
# extended to 2036 only so the diagnostic figure can show the lines up to 2035.
YEAR_END = 2036
REF_YEARS = (2022, 2023, 2024)   # observed years averaged to form the reference
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
    f.run()
    temp = rebase_to_1850_1900(f.temperature.sel(layer=0, scenario='game'))   # (timebounds, config)

    # ECS from FaIR's analytical EBM property (scenario independent)
    ecs = f.ebms['ecs'].to_series()
    ecs.index = config_ids

    df_obs = load_observed_gmst()
    obs = dict(zip(np.floor(df_obs['year'].values).astype(int), df_obs['gmst'].values))

    # --- reference: mean observed 2022-24 ---
    ref = float(np.mean([obs[y] for y in REF_YEARS]))

    # --- limit: pooled historical misfit ---
    misfit = []
    for year in range(HIST_FIRST_YEAR, 2025):
        obs_ref = np.mean([obs[y] for y in range(year - N_PRECEDING, year)])
        misfit.append(np.abs(annual_mean(temp, year).values - obs_ref))
    misfit = np.concatenate(misfit)
    limit = float(np.percentile(misfit, LIMIT_PERCENTILE))

    # --- apply to 2025 ---
    t_2025 = annual_mean(temp, 2025).values
    dev = t_2025 - ref
    passed = np.abs(dev) <= limit

    # Order-of-magnitude checks against the study that defined the rule
    print(f'\nObserved reference (mean 2022-24): {ref:.3f} degC')
    print(f'Historical misfit pool: n={len(misfit)}, {LIMIT_PERCENTILE}th percentile limit = {limit:.3f} degC')
    print(f'Member T_2025 min/median/max: {t_2025.min():.3f} / {np.median(t_2025):.3f} / {t_2025.max():.3f} degC')
    print(f'Members passing: {passed.sum()} / {len(passed)}')
    print('ECS 5/50/95% all:     ' + ' / '.join(f'{v:.2f}' for v in np.percentile(ecs, [5, 50, 95])))
    print('ECS 5/50/95% passing: ' + ' / '.join(f'{v:.2f}' for v in np.percentile(ecs[passed], [5, 50, 95])))
    assert abs(ref - 1.370) < 0.001, 'observed 2022-24 reference changed'
    assert 0.20 < limit < 0.30, 'misfit limit outside the expected ~0.25 degC'
    assert 600 < passed.sum() < 841, 'unexpected number of passing members'

    screened = pd.DataFrame({
        'T_2025': t_2025[passed],
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
    ax.set_title(f'Historical misfit {HIST_FIRST_YEAR}–2024 (defines the limit)')
    ax.legend()

    ax = axs[0, 1]
    ax.hist(dev, bins=40, color='#5A84E2', alpha=0.7)
    ax.axvline(-limit, color='k', linestyle='--')
    ax.axvline(limit, color='k', linestyle='--', label=f'± {limit:.2f} °C')
    ax.set_xlabel(f'T_2025 − observed 2022–24 mean ({ref:.2f} °C)')
    ax.set_ylabel('Members')
    ax.set_title(f'2025 deviation: {passed.sum()} of {len(passed)} members pass')
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
    years = np.arange(1950, 2036)
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
    ax.errorbar([2025.5], [ref], yerr=[[limit], [limit]], color='k', capsize=5, linewidth=1.5,
                label='reference ± limit')
    ax.axvline(2025, color='grey', linestyle=':', linewidth=1)
    ax.set_xlabel('Year (annual means at mid-year)')
    ax.set_ylabel('Temperature anomaly vs 1850–1900 (°C)')
    ax.set_title('Passing members; SSP2-4.5 after 2024')
    ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f'Saved {FIG_PATH}')


if __name__ == '__main__':
    main()

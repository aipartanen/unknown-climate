#!/usr/bin/env python3
"""
Observational uncertainty of annual GMST relative to 1850-1900, from the HadCRUT5 ensemble.
Used only to draw a 95 % uncertainty band on the game's temperature chart (around the observed
record and around the drawn climate's line after it). It does NOT enter the Bayes updates.

Run once offline:
    python precompute_obs_error.py

Input:    data/raw/HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.csv  (not committed)
          https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/
          (Morice et al. 2021, JGR Atmospheres, doi:10.1029/2019JD032361)
Produces: data/obs_uncertainty.csv  (year, half_width_95)
          figures/obs_uncertainty.png

METHOD AND WHY
--------------
* The 200 realisations sample HadCRUT5's correlated errors (bias adjustments, measurement,
  sampling). They are relative to 1961-1990; the game's anomalies are relative to 1850-1900,
  whose mean is itself uncertain (sparse early coverage). Rebaselining every realisation to its
  own 1850-1900 mean carries that baseline error into every year, exactly and with the right
  correlations. This is why the ensemble is used and not the summary file's 95 % range.
* The coverage uncertainty (1 sigma, a separate column) is NOT in the realisations. It is drawn
  here as independent annual N(0, sigma) errors, added before rebaselining (independence between
  years is an assumption; coverage errors are partly persistent, so this slightly understates
  the baseline part). N_COVERAGE_DRAWS copies of each realisation keep the quantiles smooth.
* 2025 in HadCRUT 5.0.2.0 has a coverage sigma of 0.068 degC vs ~0.004 in 2015-2024 (half-width
  0.16 vs 0.08 degC), probably a provisional year. The CSV keeps it; the app uses the 2015-2024
  mean half-width from 2025 onwards (OBS_BAND_RECENT in app.py).
* Result: half-width ~0.08 degC for recent years, mostly the 1850-1900 baseline error (vs
  1961-1990 it is ~0.03). Being a common offset, it barely averages down over a decade, so the
  game's OBS_2040_SD = 0.05 degC (decade mean, 1 sigma) is consistent with ~0.043 from here.
* The band is HadCRUT's own uncertainty. It is drawn around the IGCC series, a multi-dataset
  mean, as a proxy; the spread between datasets (structural uncertainty) is not included.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_PATH = Path('data') / 'raw' / 'HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.csv'
OUT_PATH = Path('data') / 'obs_uncertainty.csv'
FIG_PATH = Path('figures') / 'obs_uncertainty.png'
BASELINE = (1850, 1900)
N_COVERAGE_DRAWS = 50
SEED = 2021


def main():
    FIG_PATH.parent.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    years = raw['Time'].values.astype(int)
    realisations = raw[[c for c in raw.columns if c.startswith('Realization')]].values   # (years, 200)
    coverage_sd = raw['Coverage uncertainty (1 sigma)'].values
    print(f'{realisations.shape[1]} realisations, years {years[0]}-{years[-1]}')

    # Each realisation repeated N_COVERAGE_DRAWS times, each copy with its own coverage errors
    rng = np.random.default_rng(SEED)
    n_years, n_real = realisations.shape
    samples = np.repeat(realisations, N_COVERAGE_DRAWS, axis=1)
    samples = samples + rng.normal(0.0, 1.0, samples.shape) * coverage_sd[:, None]

    # Rebaseline every sample to its own 1850-1900 mean (carries the baseline error, see header)
    in_base = (years >= BASELINE[0]) & (years <= BASELINE[1])
    samples = samples - samples[in_base].mean(axis=0)

    lo = np.percentile(samples, 2.5, axis=1)
    hi = np.percentile(samples, 97.5, axis=1)
    half_width = (hi - lo) / 2

    # For comparison: the same without the baseline error (relative to 1961-1990, as published)
    in_6190 = (years >= 1961) & (years <= 1990)
    samples_6190 = samples - samples[in_6190].mean(axis=0)
    half_width_6190 = (np.percentile(samples_6190, 97.5, axis=1)
                       - np.percentile(samples_6190, 2.5, axis=1)) / 2
    baseline_mean_sd = float(np.std(realisations[in_base].mean(axis=0)
                                    - realisations[in_6190].mean(axis=0)))

    print('95 % half-width (degC), vs 1850-1900 | vs 1961-1990 | coverage sigma')
    for year in (1850, 1875, 1900, 1950, 2000, 2016, 2020, 2024, 2025):
        i = int(np.where(years == year)[0][0])
        print(f'  {year}: {half_width[i]:.3f} | {half_width_6190[i]:.3f} | {coverage_sd[i]:.4f}')
    recent = (years >= 2015) & (years <= 2024)
    print(f'mean half-width 2015-2024: {half_width[recent].mean():.3f} degC '
          f'(1 sigma ~ {half_width[recent].mean() / 1.96:.3f}; game OBS_2040_SD = 0.05 for a decade mean)')
    print(f'SD of the 1850-1900 mean vs 1961-1990 across realisations: {baseline_mean_sd:.3f} degC')
    # Plausibility: AR6 gives ~0.1 degC (5-95 %) for recent warming vs 1850-1900 across datasets
    assert 0.02 < half_width[recent].mean() < 0.2, 'recent half-width outside expected range'

    pd.DataFrame({'year': years, 'half_width_95': np.round(half_width, 4)}).to_csv(OUT_PATH, index=False)
    print(f'Saved {OUT_PATH}')

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axs[0]
    ax.plot(years, half_width, color='k', label='vs 1850–1900 (used)')
    ax.plot(years, half_width_6190, color='#5A84E2', label='vs 1961–1990 (published baseline)')
    ax.plot(years, 1.96 * coverage_sd, color='#E0792B', linestyle=':', label='1.96 × coverage σ')
    ax.set_xlabel('Year')
    ax.set_ylabel('95 % half-width (°C)')
    ax.set_title('HadCRUT5 observational uncertainty')
    ax.legend()
    ax = axs[1]
    for k in range(5):
        ax.plot(years, samples[:, k * N_COVERAGE_DRAWS] - samples.mean(axis=1), linewidth=0.8)
    ax.fill_between(years, lo - samples.mean(axis=1), hi - samples.mean(axis=1), color='grey', alpha=0.3,
                    label='95 % range')
    ax.set_xlabel('Year')
    ax.set_ylabel('Error vs ensemble mean (°C)')
    ax.set_title('Example error series (rebaselined to 1850–1900)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=120)
    print(f'Saved {FIG_PATH}')


if __name__ == '__main__':
    main()

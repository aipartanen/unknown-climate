#!/usr/bin/env python3
"""
FaIR v2.2 wrapper for the climate negotiations classroom demo.

Adapted from ghgbudgets/fair_runs.py (createConstrainedRuns).
Uses the fair-calibrate v1.6.0 constrained posterior (841 members, FaIR 2.2.4),
vendored in data/fair_calibrate/ (see the README there).

A single config is drawn randomly from a pre-screened shortlist
(data/screened_configs.csv, produced by precompute_configs.py).
CO2 FFI emissions 2023+ are replaced by the user-defined game trajectory.
All other species follow SSP2-4.5 from the harmonized emissions NetCDF.

Runs are stochastic with the seeds stored in the posterior, so a given config always
reproduces the same internal variability. This is what makes the screening in
precompute_configs.py valid for the runs shown in the game.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

_HERE = Path(__file__).parent
FAIR_CAL_DIR = _HERE / 'data' / 'fair_calibrate'
POSTERIOR_PATH = FAIR_CAL_DIR / 'output' / 'posteriors' / 'calibrated_constrained_parameters.csv'
SPECIES_PROPERTIES_PATH = FAIR_CAL_DIR / 'output' / 'posteriors' / 'species_configs_properties.csv'
SCREENED_CONFIGS_PATH = _HERE / 'data' / 'screened_configs.csv'

# Base SSP for non-CO2 gases and historical CO2.
# SSP2-4.5 is "middle of the road" — reasonable as current-policy backdrop.
BASE_SSP = 'ssp245'

# Phase 1 pledge year: CO2 cuts are specified relative to this year.
# Must match CHECKPOINT in app.py.
CHECKPOINT_YEAR = 2040

# Approximate 2023 fossil CO2 emissions by region (GtCO2/yr), from GCP 2023.
# Total ~40 GtCO2/yr, consistent with recent GCP estimates.
REGION_EMISSIONS_2023_GTCO2 = {
    'Kiina 🇨🇳':      12.0,
    'USA 🇺🇸':         4.9,
    'EU 🇪🇺':          3.4,
    'Intia 🇮🇳':       3.0,
    'Muu maailma 🌍': 16.7,
}


@lru_cache(maxsize=1)
def load_posterior() -> pd.DataFrame:
    """The 841 calibrated, constrained posterior members (rows) and their parameters."""
    return pd.read_csv(POSTERIOR_PATH, index_col=0)


def load_screened_configs() -> pd.DataFrame:
    """Load the pre-screened config shortlist produced by precompute_configs.py."""
    if not SCREENED_CONFIGS_PATH.exists():
        raise FileNotFoundError(
            f'{SCREENED_CONFIGS_PATH} not found. Run precompute_configs.py first.'
        )
    return pd.read_csv(SCREENED_CONFIGS_PATH, index_col='config_id')


def load_observed_gmst() -> pd.DataFrame:
    """
    Load the IGCC observed annual GMST series, 1850–2024 (the series the calibration
    was constrained against), relative to ~1850–1900 as in IPCC AR6.

    Returns a DataFrame with columns 'year' (float, mid-year: 1850.5 ... 2024.5) and 'gmst' (°C).
    """
    df = pd.read_csv(FAIR_CAL_DIR / 'data' / 'forcing' / 'IGCC_GMST_1850-2024.csv',
                     encoding='utf-8-sig')
    return df[['year', 'gmst']]


def rebase_to_1850_1900(temp: xr.DataArray) -> xr.DataArray:
    """
    Express temperature relative to the 1850–1900 mean, exactly as the calibration does.

    FaIR temperature sits on timebounds (1 January of each year). The mean of the annual
    means 1850–1900 equals the mean of timebounds 1850…1901 with half weight on the two
    end points.
    """
    base = temp.sel(timebounds=slice(1850, 1901))
    weights = np.ones(base.sizes['timebounds'])
    weights[0] = 0.5
    weights[-1] = 0.5
    weights = xr.DataArray(weights, coords={'timebounds': base.timebounds})
    baseline = base.weighted(weights).mean(dim='timebounds')
    return temp - baseline


def annual_mean(temp: xr.DataArray, year: int) -> xr.DataArray:
    """
    Annual mean temperature of `year`: the average of timebounds `year` and `year + 1`.

    Observations are annual means stamped at mid-year, so the observation for year Y
    corresponds to this quantity, not to the timebound `Y` (1 January of Y).
    """
    return 0.5 * (temp.sel(timebounds=year) + temp.sel(timebounds=year + 1))


def build_co2_fossil_trajectory(cuts: dict, netzero_years: dict | None, year_end: int) -> np.ndarray:
    """
    Build a global annual CO2 FFI emission time series (GtCO₂/yr) from 1750 to year_end.

    1750–2022: zeros here — caller leaves the SSP2-4.5 historical values intact.
    2023–CHECKPOINT_YEAR: each region ramps linearly from its 2023 baseline to (1−cut)×baseline.
    CHECKPOINT_YEAR–netzero_year: each region ramps linearly from its checkpoint level to zero.
    After netzero_year (or if no netzero given): held at the checkpoint level.

    Parameters
    ----------
    cuts : dict
        {region: fraction_cut} where fraction_cut ∈ [0, 1].
    netzero_years : dict or None
        {region: year} for Phase 2; None keeps each region at its checkpoint level.
    year_end : int
        Last year of the FaIR run.

    Returns
    -------
    trajectory : ndarray, shape (year_end − 1750,)
        Timepoints-aligned global CO2 FFI in GtCO₂/yr — same unit as the
        harmonized SSP NetCDF that FaIR's parameters were calibrated against.
        Indices 0…(2022−1750) are zero and will be overwritten by SSP data.
    """
    years = np.arange(1750, year_end)
    trajectory = np.zeros(len(years))

    for region, base_gtco2 in REGION_EMISSIONS_2023_GTCO2.items():
        cut     = cuts.get(region, 0.0)
        em_2023 = base_gtco2                    # GtCO₂/yr
        em_chk  = base_gtco2 * (1.0 - cut)     # GtCO₂/yr at CHECKPOINT_YEAR

        # 2023 → CHECKPOINT_YEAR: linear ramp
        mask1 = (years >= 2023) & (years < CHECKPOINT_YEAR)
        t1 = (years[mask1] - 2023) / float(CHECKPOINT_YEAR - 2023)
        trajectory[mask1] += em_2023 + t1 * (em_chk - em_2023)

        nz = (netzero_years or {}).get(region)
        if nz and nz > CHECKPOINT_YEAR:
            # CHECKPOINT_YEAR → netzero_year: linear decline to zero
            mask2 = (years >= CHECKPOINT_YEAR) & (years < nz)
            t2 = (years[mask2] - CHECKPOINT_YEAR) / float(nz - CHECKPOINT_YEAR)
            trajectory[mask2] += em_chk * (1.0 - t2)
            # after netzero_year: zero (no CDR)
        else:
            trajectory[years >= CHECKPOINT_YEAR] += em_chk

    return trajectory


def setup_fair(config_ids, year_end: int, base_ssp: str = BASE_SSP) -> FAIR:
    """
    Configure a FaIR instance for the given posterior members (not yet run).

    Follows ghgbudgets/fair_runs.py:createConstrainedRuns (fair-calibrate v1.6.0). The single
    scenario is named 'game'; its emissions are those of base_ssp and the caller may overwrite
    CO2 FFI. Runs are stochastic with each member's stored seed.

    Parameters
    ----------
    config_ids : sequence of int
        Index values of data/fair_calibrate/.../calibrated_constrained_parameters.csv.
    year_end : int
        Last timebound of the run (timebounds are 1 January of 1750 … year_end).
    """
    config_ids = list(config_ids)
    df_configs = load_posterior().loc[config_ids]
    nyears = year_end - 1750 + 1   # number of timebounds

    # Solar ERF is relative to 1850–2019 and volcanic ERF to 1850–2021, matching the
    # baselines used when the posterior forcing scales were calibrated.
    df_solar = pd.read_csv(
        FAIR_CAL_DIR / 'output' / 'forcing' / 'solar_forcing_timebounds_cmip7.csv', index_col=0)
    df_volcanic = pd.read_csv(
        FAIR_CAL_DIR / 'data' / 'forcing' / 'volcanic_forcing_timebounds_cmip7.csv', index_col=0)
    solar_forcing = df_solar['solar_erf_rel_1850-2019'].loc[1750:year_end].values
    volcanic_forcing = df_volcanic['volcanic_erf_rel_1850-2021'].loc[1750:year_end].values

    f = FAIR(ch4_method='Thornhill2021')
    f.define_time(1750, year_end, 1)
    f.define_scenarios(['game'])
    f.define_configs(config_ids)

    species, properties = read_properties(SPECIES_PROPERTIES_PATH)
    species.remove('Irrigation')
    # Land use forcing is derived from cumulative CO2 AFOLU emissions
    properties['Land use']['input_mode'] = 'calculated'
    f.define_species(species, properties)
    f.allocate()

    # Harmonized emissions of base_ssp as the base for all species
    da_emissions = xr.load_dataarray(
        FAIR_CAL_DIR / 'output' / 'emissions' / 'ssps_harmonized_1750-2499.nc')
    da = da_emissions.loc[
        dict(config='unspecified', scenario=[base_ssp], specie=species)][:nyears - 1, ...]
    fe = da.expand_dims(dim=['config'], axis=(2))
    f.emissions = fe.drop_vars('config') * np.ones((1, 1, len(config_ids), 1))
    f.emissions.coords['config'] = f.configs
    f.emissions = f.emissions.assign_coords(scenario=['game'])

    fill(f.forcing,
         volcanic_forcing[:, None, None] * df_configs['forcing_scale[Volcanic]'].values.squeeze(),
         specie='Volcanic')
    fill(f.forcing,
         solar_forcing[:, None, None] * df_configs['forcing_scale[Solar]'].values.squeeze(),
         specie='Solar')

    # Species defaults first, then the full posterior parameter set (climate response, carbon
    # cycle, aerosols, ozone, forcing scales, seeds, stochastic_run=True ...) on top.
    f.fill_species_configs(SPECIES_PROPERTIES_PATH)
    f.override_defaults(POSTERIOR_PATH)

    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    return f


def run_fair_game(cuts: dict, netzero_years: dict | None, config_id: int) -> xr.DataArray:
    """
    Set up and run a single-config seeded stochastic FaIR run for the classroom game.

    Parameters
    ----------
    cuts : dict
        {region: fraction_cut} where fraction_cut ∈ [0, 1].
    netzero_years : dict or None
        {region: year} for Phase 2 net-zero dates; None holds checkpoint levels flat.
    config_id : int
        Index value from data/screened_configs.csv / the calibrated posterior.

    Returns
    -------
    temp : xr.DataArray
        Surface temperature anomaly (°C re. 1850–1900 mean) on timebounds, dim (timebounds,).
    """
    # Give the model enough runway to capture peak temperature after net-zero.
    # High-ECS configs take 80–120 years post-net-zero to reach their maximum.
    # The cap keeps the run length (and the app's response time) bounded.
    _YEAR_END_DEFAULT = 2150
    _YEAR_END_MAX = 2300
    _POST_NETZERO_BUFFER = 150
    if netzero_years:
        latest_nz = max(netzero_years.values())
        year_end = min(max(_YEAR_END_DEFAULT, latest_nz + _POST_NETZERO_BUFFER), _YEAR_END_MAX)
    else:
        year_end = _YEAR_END_DEFAULT

    f = setup_fair([config_id], year_end, base_ssp=BASE_SSP)

    # Overwrite CO2 FFI from 2023 onwards with the game trajectory.
    # dims of f.emissions: (timepoints, scenario, config, specie)
    game_co2 = build_co2_fossil_trajectory(cuts, netzero_years, year_end)
    co2_idx  = list(f.emissions.specie.values).index('CO2 FFI')
    f.emissions.values[2023 - 1750:, 0, 0, co2_idx] = game_co2[2023 - 1750:]

    f.run()

    temp = f.temperature.sel(layer=0, scenario='game', config=config_id)
    return rebase_to_1850_1900(temp)


def run_fair_ssp(ssp_name: str, config_id: int, year_end: int = 2100) -> xr.DataArray:
    """
    Run a seeded stochastic FaIR simulation using a pre-defined SSP scenario without
    any custom CO2 override (all emissions taken directly from the SSP NetCDF).

    Parameters
    ----------
    ssp_name : str
        SSP scenario key as it appears in the harmonized emissions file,
        e.g. 'ssp119', 'ssp245', 'ssp370', 'ssp585'.
    config_id : int
        Index value from data/screened_configs.csv / the calibrated posterior.
    year_end : int
        Last year of the FaIR run (default 2100).

    Returns
    -------
    temp : xr.DataArray
        Surface temperature anomaly (°C re. 1850–1900 mean), dim (timebounds,).
    """
    f = setup_fair([config_id], year_end, base_ssp=ssp_name)
    f.run()

    temp = f.temperature.sel(layer=0, scenario='game', config=config_id)
    return rebase_to_1850_1900(temp)

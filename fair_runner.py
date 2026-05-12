#!/usr/bin/env python3
"""
FaIR v2.2 wrapper for the climate negotiations classroom demo.

Adapted from ghgbudgets/fair_runs.py (createConstrainedRuns).
Uses the fair-calibrate v1.4 / all-2022 constrained posterior.

A single config is drawn randomly from a pre-screened shortlist
(data/screened_configs.csv, produced by precompute_configs.py).
CO2 FFI emissions 2023+ are replaced by the user-defined game trajectory.
All other species follow SSP2-4.5 from the harmonized emissions NetCDF.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

FAIR_CAL_DIR = Path('/Users/partanen/version-controlled/ghgbudgets/data/fair_calibrate')
FAIR_V = '2.1.3'
CAL_V  = '1.4'
CONSTRAINT_SET = 'all-2022'

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

# Atomic mass ratio C/CO2: 12.011 / (12.011 + 2×15.999)
# Used to convert GtCO2 → GtC for FaIR's internal CO2 FFI emission unit.
CO2_TO_C = 12.011 / 44.009   # ≈ 0.2729


def load_screened_configs() -> pd.DataFrame:
    """Load the pre-screened config shortlist produced by precompute_configs.py."""
    path = Path('data/screened_configs.csv')
    if not path.exists():
        raise FileNotFoundError(
            'data/screened_configs.csv not found. '
            'Run precompute_configs.py first.'
        )
    return pd.read_csv(path, index_col='config_id')


def load_observed_gmst() -> pd.DataFrame:
    """
    Load the IGCC observed GMST time series (1850–2022) and append
    the 2023 and 2024 annual means from the Indicators of Global
    Climate Change 2024 report.

    The IGCC series uses a ~1850–1900 pre-industrial baseline,
    consistent with IPCC AR6.

    Returns a DataFrame with columns 'year' (float, mid-year) and 'gmst' (°C).
    """
    igcc_path = FAIR_CAL_DIR / 'data' / 'forcing' / 'IGCC_GMST_1850-2022.csv'
    df = pd.read_csv(igcc_path, names=['year', 'gmst'], header=0)

    # Extend with IGCC 2024 values: 2023 El Niño peak, 2024 annual mean
    extension = pd.DataFrame({
        'year': [2023.5, 2024.5],
        'gmst': [1.45, 1.36],
    })
    df = pd.concat([df, extension], ignore_index=True)
    return df


def build_co2_fossil_trajectory(cuts: dict, netzero_years: dict | None, year_end: int) -> np.ndarray:
    """
    Build a global annual CO2 FFI emission time series (GtC/yr) from 1750 to year_end.

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
        Timepoints-aligned global CO2 FFI in GtC/yr.
        Indices 0…(2022−1750) are zero and will be overwritten by SSP data.
    """
    years = np.arange(1750, year_end)
    trajectory = np.zeros(len(years))

    for region, base_gtco2 in REGION_EMISSIONS_2023_GTCO2.items():
        cut    = cuts.get(region, 0.0)
        em_2023 = base_gtco2 * CO2_TO_C                    # GtC/yr
        em_chk  = base_gtco2 * (1.0 - cut) * CO2_TO_C     # pledged level at CHECKPOINT_YEAR

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


def _setup_fair(config_id: int, df_configs_row: pd.DataFrame,
                df_methane: pd.DataFrame, df_landuse: pd.DataFrame,
                df_lapsi: pd.DataFrame, year_end: int,
                base_ssp: str = 'ssp245') -> FAIR:
    """
    Configure a single-config, deterministic FaIR instance.
    Emissions are filled from base_ssp; CO2 FFI may be overwritten by caller.
    """
    forcing_dir = FAIR_CAL_DIR / 'data' / 'forcing'
    df_solar    = pd.read_csv(forcing_dir / 'solar_erf_timebounds.csv', index_col='year')
    df_volcanic = pd.read_csv(forcing_dir / 'volcanic_ERF_1750-2101_timebounds.csv',
                               index_col='timebounds')

    nyears = year_end - 1750 + 1
    solar_forcing    = df_solar['erf'].loc[1750:year_end].values
    volcanic_forcing = np.zeros(nyears)
    year_end_volcanic = min(2101, year_end)
    volcanic_forcing[:year_end_volcanic - 1750 + 1] = (
        df_volcanic['erf'].loc[1750:year_end_volcanic].values
    )

    trend_shape = np.ones(nyears)
    trend_shape[:271] = np.linspace(0, 1, 271)

    valid_all = df_configs_row.index   # single config_id

    f = FAIR(ch4_method='Thornhill2021')
    f.define_time(1750, year_end, 1)
    f.define_scenarios(['game'])
    f.define_configs(valid_all)

    species, properties = read_properties()
    species.remove('Halon-1202')
    species.remove('NOx aviation')
    species.remove('Contrails')
    f.define_species(species, properties)
    f.allocate()

    # Load SSP2-4.5 harmonized emissions as the base for all species
    da_emissions = xr.load_dataarray(
        FAIR_CAL_DIR / 'output' / f'fair-{FAIR_V}' / f'v{CAL_V}' / CONSTRAINT_SET
        / 'emissions' / 'ssps_harmonized_1750-2499.nc'
    )
    da = da_emissions.loc[
        dict(config='unspecified', scenario=[base_ssp], specie=species)
    ][:nyears - 1, ...]
    fe = da.expand_dims(dim=['config'], axis=(2))
    f.emissions = fe.drop_vars('config') * np.ones((1, 1, 1, 1))
    f.emissions.coords['config'] = valid_all
    f.emissions = f.emissions.assign_coords(scenario=['game'])

    fill(f.forcing,
         volcanic_forcing[:, None, None] * df_configs_row['fscale_Volcanic'].values.squeeze(),
         specie='Volcanic')
    fill(f.forcing,
         solar_forcing[:, None, None] * df_configs_row['fscale_solar_amplitude'].values.squeeze()
         + trend_shape[:, None, None] * df_configs_row['fscale_solar_trend'].values.squeeze(),
         specie='Solar')

    fill(f.climate_configs['ocean_heat_capacity'],   df_configs_row.loc[:, 'clim_c1':'clim_c3'].values)
    fill(f.climate_configs['ocean_heat_transfer'],   df_configs_row.loc[:, 'clim_kappa1':'clim_kappa3'].values)
    fill(f.climate_configs['deep_ocean_efficacy'],   df_configs_row['clim_epsilon'].values.squeeze())
    fill(f.climate_configs['gamma_autocorrelation'], df_configs_row['clim_gamma'].values.squeeze())
    fill(f.climate_configs['sigma_eta'],             df_configs_row['clim_sigma_eta'].values.squeeze())
    fill(f.climate_configs['sigma_xi'],              df_configs_row['clim_sigma_xi'].values.squeeze())
    fill(f.climate_configs['seed'],                  df_configs_row['seed'])
    # Deterministic run: stochastic noise disabled for a clean classroom trajectory
    fill(f.climate_configs['stochastic_run'],        False)
    fill(f.climate_configs['use_seed'],              True)
    fill(f.climate_configs['forcing_4co2'],          df_configs_row['clim_F_4xCO2'])

    f.fill_species_configs()

    fill(f.species_configs['iirf_0'],           df_configs_row['cc_r0'].values.squeeze(),  specie='CO2')
    fill(f.species_configs['iirf_airborne'],    df_configs_row['cc_rA'].values.squeeze(),  specie='CO2')
    fill(f.species_configs['iirf_uptake'],      df_configs_row['cc_rU'].values.squeeze(),  specie='CO2')
    fill(f.species_configs['iirf_temperature'], df_configs_row['cc_rT'].values.squeeze(),  specie='CO2')
    fill(f.species_configs['land_use_cumulative_emissions_to_forcing'],
         df_landuse.loc['historical_best', 'CO2_AFOLU'], specie='CO2 AFOLU')

    fill(f.species_configs['aci_scale'],  df_configs_row['aci_beta'].values.squeeze())
    fill(f.species_configs['aci_shape'],  df_configs_row['aci_shape_so2'].values.squeeze(), specie='Sulfur')
    fill(f.species_configs['aci_shape'],  df_configs_row['aci_shape_bc'].values.squeeze(),  specie='BC')
    fill(f.species_configs['aci_shape'],  df_configs_row['aci_shape_oc'].values.squeeze(),  specie='OC')

    fill(f.species_configs['unperturbed_lifetime'],
         df_methane.loc['historical_best', 'base'], specie='CH4')
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'CH4'], specie='CH4')
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'N2O'], specie='N2O')
    fill(f.species_configs['lifetime_temperature_sensitivity'],
         df_methane.loc['historical_best', 'temp'])
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'VOC'], specie='VOC')
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'NOx'], specie='NOx')
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'HC'],
         specie='Equivalent effective stratospheric chlorine')
    fill(f.species_configs['lapsi_radiative_efficiency'],
         df_lapsi.loc['historical_best', 'BC'], specie='BC')

    fill(f.species_configs['baseline_emissions'], 19.41683292,    specie='NOx')
    fill(f.species_configs['baseline_emissions'], 2.293964929,    specie='Sulfur')
    fill(f.species_configs['baseline_emissions'], 348.4549732,    specie='CO')
    fill(f.species_configs['baseline_emissions'], 60.62284009,    specie='VOC')
    fill(f.species_configs['baseline_emissions'], 2.096765609,    specie='BC')
    fill(f.species_configs['baseline_emissions'], 15.44571911,    specie='OC')
    fill(f.species_configs['baseline_emissions'], 6.656462698,    specie='NH3')
    fill(f.species_configs['baseline_emissions'], 38.246272,      specie='CH4')
    fill(f.species_configs['baseline_emissions'], 0.92661989,     specie='N2O')
    fill(f.species_configs['baseline_emissions'], 0.02129917,     specie='CCl4')
    fill(f.species_configs['baseline_emissions'], 202.7251231,    specie='CHCl3')
    fill(f.species_configs['baseline_emissions'], 211.0095537,    specie='CH2Cl2')
    fill(f.species_configs['baseline_emissions'], 4544.519056,    specie='CH3Cl')
    fill(f.species_configs['baseline_emissions'], 111.4920237,    specie='CH3Br')
    fill(f.species_configs['baseline_emissions'], 0.008146006,    specie='Halon-1211')
    fill(f.species_configs['baseline_emissions'], 0.000010554155, specie='SO2F2')
    fill(f.species_configs['baseline_emissions'], 0,              specie='CF4')

    for specie in ['BC', 'CH4', 'N2O', 'NH3', 'NOx', 'OC', 'Sulfur', 'VOC',
                   'Equivalent effective stratospheric chlorine']:
        fill(f.species_configs['erfari_radiative_efficiency'],
             df_configs_row[f'ari_{specie}'], specie=specie)

    fill(f.species_configs['forcing_scale'],
         df_configs_row['fscale_CO2'].values.squeeze(), specie='CO2')
    for specie in ['CH4', 'N2O']:
        fill(f.species_configs['forcing_scale'],
             df_configs_row[f'fscale_{specie}'].values.squeeze(), specie=specie)
    for specie in ['Stratospheric water vapour', 'Light absorbing particles on snow and ice']:
        fill(f.species_configs['forcing_scale'],
             df_configs_row[f'fscale_{specie}'].values.squeeze(), specie=specie)
    fill(f.species_configs['forcing_scale'],
         df_configs_row['fscale_Land use'].values.squeeze(), specie='Land use')

    minor_ghgs = [
        'CFC-11', 'CFC-12', 'CFC-113', 'CFC-114', 'CFC-115', 'HCFC-22',
        'HCFC-141b', 'HCFC-142b', 'CCl4', 'CHCl3', 'CH2Cl2', 'CH3Cl',
        'CH3CCl3', 'CH3Br', 'Halon-1211', 'Halon-1301', 'Halon-2402',
        'CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'C5F12', 'C6F14',
        'C7F16', 'C8F18', 'NF3', 'SF6', 'SO2F2', 'HFC-125', 'HFC-134a',
        'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23', 'HFC-236fa',
        'HFC-245fa', 'HFC-32', 'HFC-365mfc', 'HFC-4310mee',
    ]
    for specie in minor_ghgs:
        fill(f.species_configs['forcing_scale'],
             df_configs_row['fscale_minorGHG'].values.squeeze(), specie=specie)

    for specie in ['Equivalent effective stratospheric chlorine', 'CO', 'VOC', 'NOx', 'CH4', 'N2O']:
        fill(f.species_configs['ozone_radiative_efficiency'],
             df_configs_row[f'o3_{specie}'], specie=specie)

    fill(f.species_configs['forcing_efficacy'], 0.6, specie='Volcanic')
    fill(f.species_configs['baseline_concentration'],
         df_configs_row['cc_co2_concentration_1750'].values.squeeze(), specie='CO2')

    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    return f


def run_fair_game(cuts: dict, netzero_years: dict | None, config_id: int) -> xr.DataArray:
    """
    Set up and run a single-config deterministic FaIR run for the classroom game.

    Parameters
    ----------
    cuts : dict
        {region: fraction_cut} where fraction_cut ∈ [0, 1].
    netzero_years : dict or None
        {region: year} for Phase 2 net-zero dates; None holds checkpoint levels flat.
    config_id : int
        Row index from data/screened_configs.csv / the calibrated posterior.

    Returns
    -------
    temp : xr.DataArray
        Surface temperature anomaly (°C re. 1850–1900 mean), dim (timebounds,).
    """
    year_end = 2150   # long enough to capture peak temperature after any net-zero year

    cal_out = FAIR_CAL_DIR / 'output' / f'fair-{FAIR_V}' / f'v{CAL_V}' / CONSTRAINT_SET
    df_configs_all = pd.read_csv(
        cal_out / 'posteriors' / 'calibrated_constrained_parameters.csv', index_col=0
    )
    df_methane = pd.read_csv(cal_out / 'calibrations' / 'CH4_lifetime.csv',      index_col=0)
    df_landuse = pd.read_csv(cal_out / 'calibrations' / 'landuse_scale_factor.csv', index_col=0)
    df_lapsi   = pd.read_csv(cal_out / 'calibrations' / 'lapsi_scale_factor.csv',   index_col=0)

    df_row = df_configs_all.loc[[config_id]]

    f = _setup_fair(config_id, df_row, df_methane, df_landuse, df_lapsi, year_end, base_ssp='ssp245')

    # Overwrite CO2 FFI from 2023 onwards with the game trajectory.
    # dims of f.emissions: (timepoints, scenario, config, specie)
    game_co2 = build_co2_fossil_trajectory(cuts, netzero_years, year_end)
    co2_idx  = list(f.emissions.specie.values).index('CO2 FFI')
    f.emissions.values[2023 - 1750:, 0, 0, co2_idx] = game_co2[2023 - 1750:]

    f.run()

    # Surface temperature (layer 0), anomaly relative to 1850–1900 pre-industrial mean.
    # 1850–1900 baseline is consistent with IPCC AR6 observed warming reference.
    temp = f.temperature.sel(layer=0, scenario='game', config=config_id)
    baseline = temp.sel(timebounds=slice(1850, 1900)).mean(dim='timebounds')
    return temp - baseline


def run_fair_ssp(ssp_name: str, config_id: int, year_end: int = 2100) -> xr.DataArray:
    """
    Run a deterministic FaIR simulation using a pre-defined SSP scenario without
    any custom CO2 override (all emissions taken directly from the SSP NetCDF).

    Parameters
    ----------
    ssp_name : str
        SSP scenario key as it appears in the harmonized emissions file,
        e.g. 'ssp119', 'ssp245', 'ssp370', 'ssp585'.
    config_id : int
        Row index from data/screened_configs.csv / the calibrated posterior.
    year_end : int
        Last year of the FaIR run (default 2100).

    Returns
    -------
    temp : xr.DataArray
        Surface temperature anomaly (°C re. 1850–1900 mean), dim (timebounds,).
    """
    cal_out = FAIR_CAL_DIR / 'output' / f'fair-{FAIR_V}' / f'v{CAL_V}' / CONSTRAINT_SET
    df_configs_all = pd.read_csv(
        cal_out / 'posteriors' / 'calibrated_constrained_parameters.csv', index_col=0
    )
    df_methane = pd.read_csv(cal_out / 'calibrations' / 'CH4_lifetime.csv',       index_col=0)
    df_landuse = pd.read_csv(cal_out / 'calibrations' / 'landuse_scale_factor.csv', index_col=0)
    df_lapsi   = pd.read_csv(cal_out / 'calibrations' / 'lapsi_scale_factor.csv',   index_col=0)

    df_row = df_configs_all.loc[[config_id]]

    f = _setup_fair(config_id, df_row, df_methane, df_landuse, df_lapsi, year_end,
                    base_ssp=ssp_name)
    f.run()

    temp = f.temperature.sel(layer=0, scenario='game', config=config_id)
    baseline = temp.sel(timebounds=slice(1850, 1900)).mean(dim='timebounds')
    return temp - baseline

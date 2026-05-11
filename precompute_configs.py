#!/usr/bin/env python3
"""
Pre-screen the full 841-member FaIR posterior ensemble and save a shortlist
of configs that reproduce present-day warming and span a plausible ECS range.

Run once offline:
    python precompute_configs.py

Produces: data/screened_configs.csv  (config_id, T_2024, ECS)
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
BASE_SSP = 'ssp245'

# Observed 2024 annual mean warming from Indicators of Global Climate Change 2024
T_2024_OBS = 1.36   # °C above 1850–1900

# Accept configs within this window around observed
T_TOLERANCE = 0.10  # °C

# IPCC AR6 "very likely" ECS range (2.0–5.0 °C) — filter out extreme outliers
ECS_MIN = 2.0
ECS_MAX = 5.0

OUTPUT_DIR = Path('data')


def _load_calibration_data():
    cal_out = FAIR_CAL_DIR / 'output' / f'fair-{FAIR_V}' / f'v{CAL_V}' / CONSTRAINT_SET
    df_configs  = pd.read_csv(cal_out / 'posteriors' / 'calibrated_constrained_parameters.csv', index_col=0)
    df_methane  = pd.read_csv(cal_out / 'calibrations' / 'CH4_lifetime.csv', index_col=0)
    df_landuse  = pd.read_csv(cal_out / 'calibrations' / 'landuse_scale_factor.csv', index_col=0)
    df_lapsi    = pd.read_csv(cal_out / 'calibrations' / 'lapsi_scale_factor.csv', index_col=0)
    return df_configs, df_methane, df_landuse, df_lapsi


def _load_forcing_data(year_end: int):
    forcing_dir = FAIR_CAL_DIR / 'data' / 'forcing'
    df_solar    = pd.read_csv(forcing_dir / 'solar_erf_timebounds.csv', index_col='year')
    df_volcanic = pd.read_csv(forcing_dir / 'volcanic_ERF_1750-2101_timebounds.csv', index_col='timebounds')

    nyears = year_end - 1750 + 1
    solar_forcing    = df_solar['erf'].loc[1750:year_end].values
    volcanic_forcing = np.zeros(nyears)
    year_end_volcanic = min(2101, year_end)
    volcanic_forcing[:year_end_volcanic - 1750 + 1] = df_volcanic['erf'].loc[1750:year_end_volcanic].values
    return solar_forcing, volcanic_forcing


def run_historical(df_configs, df_methane, df_landuse, df_lapsi, year_end: int) -> xr.DataArray:
    """
    Run FaIR 1750–year_end with all configs in df_configs using ssp245 emissions.
    Returns surface temperature anomaly DataArray (timebounds × config),
    rebased to 1850–1900.
    """
    n_configs = len(df_configs)
    valid_all  = df_configs.index
    nyears     = year_end - 1750 + 1

    solar_forcing, volcanic_forcing = _load_forcing_data(year_end)

    trend_shape = np.ones(nyears)
    trend_shape[:271] = np.linspace(0, 1, 271)

    f = FAIR(ch4_method='Thornhill2021')
    f.define_time(1750, year_end, 1)
    f.define_scenarios([BASE_SSP])
    f.define_configs(valid_all)

    species, properties = read_properties()
    species.remove('Halon-1202')
    species.remove('NOx aviation')
    species.remove('Contrails')
    f.define_species(species, properties)
    f.allocate()

    da_emissions = xr.load_dataarray(
        FAIR_CAL_DIR / 'output' / f'fair-{FAIR_V}' / f'v{CAL_V}' / CONSTRAINT_SET
        / 'emissions' / 'ssps_harmonized_1750-2499.nc'
    )
    da = da_emissions.loc[dict(config='unspecified', scenario=[BASE_SSP], specie=species)][:nyears - 1, ...]
    fe = da.expand_dims(dim=['config'], axis=(2))
    f.emissions = fe.drop_vars('config') * np.ones((1, 1, n_configs, 1))
    f.emissions.coords['config'] = valid_all

    fill(f.forcing,
         volcanic_forcing[:, None, None] * df_configs['fscale_Volcanic'].values.squeeze(),
         specie='Volcanic')
    fill(f.forcing,
         solar_forcing[:, None, None] * df_configs['fscale_solar_amplitude'].values.squeeze()
         + trend_shape[:, None, None] * df_configs['fscale_solar_trend'].values.squeeze(),
         specie='Solar')

    fill(f.climate_configs['ocean_heat_capacity'],    df_configs.loc[:, 'clim_c1':'clim_c3'].values)
    fill(f.climate_configs['ocean_heat_transfer'],    df_configs.loc[:, 'clim_kappa1':'clim_kappa3'].values)
    fill(f.climate_configs['deep_ocean_efficacy'],    df_configs['clim_epsilon'].values.squeeze())
    fill(f.climate_configs['gamma_autocorrelation'],  df_configs['clim_gamma'].values.squeeze())
    fill(f.climate_configs['sigma_eta'],              df_configs['clim_sigma_eta'].values.squeeze())
    fill(f.climate_configs['sigma_xi'],               df_configs['clim_sigma_xi'].values.squeeze())
    fill(f.climate_configs['seed'],                   df_configs['seed'])
    # Deterministic run to match game mode; stochastic noise would create
    # per-seed T_2024 offsets that don't reflect the forced-response screening target.
    fill(f.climate_configs['stochastic_run'],         False)
    fill(f.climate_configs['use_seed'],               True)
    fill(f.climate_configs['forcing_4co2'],           df_configs['clim_F_4xCO2'])

    f.fill_species_configs()

    fill(f.species_configs['iirf_0'],           df_configs['cc_r0'].values.squeeze(),  specie='CO2')
    fill(f.species_configs['iirf_airborne'],    df_configs['cc_rA'].values.squeeze(),  specie='CO2')
    fill(f.species_configs['iirf_uptake'],      df_configs['cc_rU'].values.squeeze(),  specie='CO2')
    fill(f.species_configs['iirf_temperature'], df_configs['cc_rT'].values.squeeze(),  specie='CO2')
    fill(f.species_configs['land_use_cumulative_emissions_to_forcing'],
         df_landuse.loc['historical_best', 'CO2_AFOLU'], specie='CO2 AFOLU')
    fill(f.species_configs['aci_scale'],  df_configs['aci_beta'].values.squeeze())
    fill(f.species_configs['aci_shape'],  df_configs['aci_shape_so2'].values.squeeze(), specie='Sulfur')
    fill(f.species_configs['aci_shape'],  df_configs['aci_shape_bc'].values.squeeze(),  specie='BC')
    fill(f.species_configs['aci_shape'],  df_configs['aci_shape_oc'].values.squeeze(),  specie='OC')
    fill(f.species_configs['unperturbed_lifetime'],
         df_methane.loc['historical_best', 'base'], specie='CH4')
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'CH4'],  specie='CH4')
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'N2O'],  specie='N2O')
    fill(f.species_configs['lifetime_temperature_sensitivity'],
         df_methane.loc['historical_best', 'temp'])
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'VOC'],  specie='VOC')
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'NOx'],  specie='NOx')
    fill(f.species_configs['ch4_lifetime_chemical_sensitivity'],
         df_methane.loc['historical_best', 'HC'],
         specie='Equivalent effective stratospheric chlorine')
    fill(f.species_configs['lapsi_radiative_efficiency'],
         df_lapsi.loc['historical_best', 'BC'], specie='BC')

    fill(f.species_configs['baseline_emissions'], 19.41683292,     specie='NOx')
    fill(f.species_configs['baseline_emissions'], 2.293964929,     specie='Sulfur')
    fill(f.species_configs['baseline_emissions'], 348.4549732,     specie='CO')
    fill(f.species_configs['baseline_emissions'], 60.62284009,     specie='VOC')
    fill(f.species_configs['baseline_emissions'], 2.096765609,     specie='BC')
    fill(f.species_configs['baseline_emissions'], 15.44571911,     specie='OC')
    fill(f.species_configs['baseline_emissions'], 6.656462698,     specie='NH3')
    fill(f.species_configs['baseline_emissions'], 38.246272,       specie='CH4')
    fill(f.species_configs['baseline_emissions'], 0.92661989,      specie='N2O')
    fill(f.species_configs['baseline_emissions'], 0.02129917,      specie='CCl4')
    fill(f.species_configs['baseline_emissions'], 202.7251231,     specie='CHCl3')
    fill(f.species_configs['baseline_emissions'], 211.0095537,     specie='CH2Cl2')
    fill(f.species_configs['baseline_emissions'], 4544.519056,     specie='CH3Cl')
    fill(f.species_configs['baseline_emissions'], 111.4920237,     specie='CH3Br')
    fill(f.species_configs['baseline_emissions'], 0.008146006,     specie='Halon-1211')
    fill(f.species_configs['baseline_emissions'], 0.000010554155,  specie='SO2F2')
    fill(f.species_configs['baseline_emissions'], 0,               specie='CF4')

    for specie in ['BC', 'CH4', 'N2O', 'NH3', 'NOx', 'OC', 'Sulfur', 'VOC',
                   'Equivalent effective stratospheric chlorine']:
        fill(f.species_configs['erfari_radiative_efficiency'],
             df_configs[f'ari_{specie}'], specie=specie)

    fill(f.species_configs['forcing_scale'],
         df_configs['fscale_CO2'].values.squeeze(), specie='CO2')
    for specie in ['CH4', 'N2O']:
        fill(f.species_configs['forcing_scale'],
             df_configs[f'fscale_{specie}'].values.squeeze(), specie=specie)
    for specie in ['Stratospheric water vapour', 'Light absorbing particles on snow and ice']:
        fill(f.species_configs['forcing_scale'],
             df_configs[f'fscale_{specie}'].values.squeeze(), specie=specie)
    fill(f.species_configs['forcing_scale'],
         df_configs['fscale_Land use'].values.squeeze(), specie='Land use')

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
             df_configs['fscale_minorGHG'].values.squeeze(), specie=specie)

    for specie in ['Equivalent effective stratospheric chlorine', 'CO', 'VOC', 'NOx', 'CH4', 'N2O']:
        fill(f.species_configs['ozone_radiative_efficiency'],
             df_configs[f'o3_{specie}'], specie=specie)

    fill(f.species_configs['forcing_efficacy'], 0.6, specie='Volcanic')
    fill(f.species_configs['baseline_concentration'],
         df_configs['cc_co2_concentration_1750'].values.squeeze(), specie='CO2')

    initialise(f.concentration, f.species_configs['baseline_concentration'])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    f.run()

    # ECS per config from FaIR's built-in analytical EBM property (no diagnostic runs).
    # f.ebms['ecs'] has only a 'config' dimension — ECS is scenario-independent.
    ecs_series = f.ebms['ecs'].to_series()

    # Surface temperature, rebased to 1850–1900 mean (IPCC AR6 pre-industrial baseline)
    temp = f.temperature.sel(layer=0, scenario=BASE_SSP)
    baseline = temp.sel(timebounds=slice(1850, 1900)).mean(dim='timebounds')
    temp_rebased = temp - baseline

    return temp_rebased, ecs_series


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print('Loading calibration data...')
    df_configs, df_methane, df_landuse, df_lapsi = _load_calibration_data()
    print(f'  {len(df_configs)} posterior configs loaded.')

    print('Running FaIR 1750–2025 for all configs...')
    temp_all, ecs_all = run_historical(df_configs, df_methane, df_landuse, df_lapsi, year_end=2025)

    # T_2024: use the timebounds value at 2024 (mid-year point)
    t_2024 = temp_all.sel(timebounds=2024).to_series()

    print(f'\nT_2024 range: {t_2024.min():.3f} – {t_2024.max():.3f} °C')
    print(f'T_2024 median: {t_2024.median():.3f} °C  (target {T_2024_OBS} ± {T_TOLERANCE})')

    # Step 1: filter by T_2024 proximity to observed
    mask_t = (t_2024 - T_2024_OBS).abs() <= T_TOLERANCE
    print(f'\nStep 1 — T_2024 filter: {mask_t.sum()} / {len(t_2024)} configs pass.')

    # Step 2: filter by ECS plausibility (IPCC AR6 very likely range 2.0–5.0 °C)
    mask_ecs = (ecs_all >= ECS_MIN) & (ecs_all <= ECS_MAX)
    mask_both = mask_t & mask_ecs
    print(f'Step 2 — ECS filter ({ECS_MIN}–{ECS_MAX} °C): {mask_both.sum()} configs pass.')

    screened = pd.DataFrame({
        'T_2024': t_2024[mask_both],
        'ECS':    ecs_all[mask_both],
    })
    screened.index.name = 'config_id'

    ecs_q = screened['ECS'].quantile([0.05, 0.50, 0.95])
    print(f'\nScreened ECS percentiles — 5th: {ecs_q[0.05]:.2f}  '
          f'50th: {ecs_q[0.50]:.2f}  95th: {ecs_q[0.95]:.2f} °C')

    out_path = OUTPUT_DIR / 'screened_configs.csv'
    screened.to_csv(out_path)
    print(f'\nSaved {len(screened)} configs to {out_path}')


if __name__ == '__main__':
    main()

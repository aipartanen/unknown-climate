# Credits, data sources and licences

## What the MIT licence covers

The code in this repository and the data files derived by its scripts are © 2026 Finnish
Meteorological Institute under the [MIT licence](LICENSE). The derived files are:

- `data/screened_configs.csv` (`precompute_configs.py`)
- `data/estimate_emulator.csv` (`precompute_estimates.py`)
- `data/obs_uncertainty.csv` (`precompute_obs_error.py`, computed from HadCRUT5; see the
  HadCRUT5 section for its attribution)

Third-party files in `data/` keep their own licences, listed below. Please cite the original
sources if you reuse them.

## Climate model

**FaIR v2.2.4** (Finite amplitude Impulse Response simple climate model), installed as a Python
package and not included in this repository. Apache License 2.0,
<https://github.com/OMS-NetZero/FAIR>.

> Leach, N. J., Jenkins, S., Nicholls, Z., Smith, C. J., Lynch, J., Cain, M., Walsh, T., Wu, B.,
> Tsutsui, J., and Allen, M. R. (2021). FaIRv2.0.0: a generalized impulse response model for
> climate uncertainty and future scenario exploration. *Geosci. Model Dev.*, 14, 3007–3036.
> <https://doi.org/10.5194/gmd-14-3007-2021>

## Calibration and forcing data (`data/fair_calibrate/`)

These are output and input files of **fair-calibrate v1.6.0**, the calibrated and constrained
parameter set for FaIR 2.2.4 (841 posterior members). The calibration data are licensed CC BY 4.0
(Zenodo, <https://doi.org/10.5281/zenodo.7112539>). The fair-calibrate code is MIT,
<https://github.com/chrisroadmap/fair-calibrate>.

> Smith, C., Cummins, D. P., Fredriksen, H.-B., Nicholls, Z., Meinshausen, M., Allen, M.,
> Jenkins, S., Leach, N., Mathison, C., and Partanen, A.-I. (2024). fair-calibrate v1.4.1:
> calibration, constraining, and validation of the FaIR simple climate model for reliable
> future climate projections. *Geosci. Model Dev.*, 17, 8569–8592.
> <https://doi.org/10.5194/gmd-17-8569-2024>

Some of the files originate from further datasets:

- **`output/emissions/ssps_harmonized_1750-2499.nc`** contains SSP emissions harmonized by
  fair-calibrate. The scenarios come from the **SSP Database** (© IIASA and the contributing
  modelling teams), which is distributed under IIASA's licence adapted from CC BY 4.0,
  <https://ssp.apps.ece.iiasa.ac.at/license>. This repository includes only two scenarios,
  SSP1-1.9 and SSP2-4.5, which also carry the 1750–2022 history. Their values are unchanged.
- **`output/forcing/solar_forcing_timebounds_cmip7.csv`** is effective radiative forcing computed
  by fair-calibrate from the CMIP7 SOLARIS-HEPPA solar forcing (input4MIPs, v4.6). The SOLARIS-HEPPA
  data are CC BY 4.0, <https://www.solarisheppa.kit.edu/75.php>.
  > Funke, B., et al. (2024). Towards the definition of a solar forcing dataset for CMIP7.
  > *Geosci. Model Dev.*, 17, 1217. <https://doi.org/10.5194/gmd-17-1217-2024>
- **`data/forcing/IGCC_GMST_1850-2024.csv`**: the observed temperatures the calibration was
  constrained against. The game no longer reads this file (see the next section).

## Observations

**Global mean surface temperature** (`data/IGCC_GMST_1850-2025.csv`) is from the Indicators of
Global Climate Change 2025 data, release IGCC-2025a, licensed CC BY 4.0,
<https://doi.org/10.5281/zenodo.21494229>. The corresponding paper was submitted to *Earth System
Science Data*. The most recent published edition is:

> Forster, P. M., Smith, C., Walsh, T., Lamb, W. F., Lamboll, R., et al. (2025). Indicators of
> Global Climate Change 2024: annual update of key indicators of the state of the climate system
> and human influence. *Earth Syst. Sci. Data*, 17, 2641–2680.
> <https://doi.org/10.5194/essd-17-2641-2025>

**Observational uncertainty band** (`data/obs_uncertainty.csv`) is derived from HadCRUT5.
HadCRUT.5.0.2.0 analysis, global annual ensemble series, downloaded 2026-11-19 from
<https://www.metoffice.gov.uk/hadobs/hadcrut5/>. © British Crown Copyright, Met Office 2021.
Provided under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

> Morice, C. P., Kennedy, J. J., Rayner, N. A., et al. (2021). An updated assessment of
> near-surface temperature change from 1850: the HadCRUT5 data set. *J. Geophys. Res. Atmos.*,
> 126, e2019JD032361. <https://doi.org/10.1029/2019JD032361>

## Emissions 2023–2025

The fossil CO₂ emissions for 2023–2025 (`HISTORICAL_CO2_FFI_GTCO2` and
`REGION_EMISSIONS_BASE_GTCO2` in `fair_runner.py`) are from the **Global Carbon Budget**, 2025
data release, as redistributed by **Our World in Data** in `owid-co2-data.csv`,
<https://github.com/owid/co2-data>. Our World in Data publishes its data under CC BY 4.0;
the Global Carbon Budget data keep their own terms.

- Global Carbon Budget 2025. The data are distributed through the ICOS Carbon Portal under its
  data licence, CC BY 4.0 (<https://www.icos-cp.eu/data-services/data-portal/data-licence>). The
  Global Carbon Project asks that the original data sources be cited:
  > Friedlingstein, P., O'Sullivan, M., Jones, M. W., et al. (2026). Global Carbon Budget 2025.
  > *Earth Syst. Sci. Data*, 18, 3211–3288. <https://doi.org/10.5194/essd-18-3211-2026>

  > Global Carbon Project (2026). Supplemental data of Global Carbon Budget 2025 (Version 1.0)
  > [Data set]. <https://doi.org/10.18160/gcp-2025>
- Our World in Data, CO₂ and Greenhouse Gas Emissions,
  <https://ourworldindata.org/co2-and-greenhouse-gas-emissions>.

## Climate impacts page

The global figures on the impacts page are from:

- IPCC, 2021: *Climate Change 2021: The Physical Science Basis.* Contribution of Working Group I
  to the Sixth Assessment Report. <https://doi.org/10.1017/9781009157896>
- IPCC, 2018: *Global Warming of 1.5 °C.* Special Report.
  <https://doi.org/10.1017/9781009157940>

The figures for Finland are from the Finnish Meteorological Institute.
**TODO: name the exact publication(s).**

## Software

The app is built with these open-source Python packages. They are installed as dependencies and
not included in this repository:

| Package | Licence |
|---|---|
| [Streamlit](https://streamlit.io/) | Apache 2.0 |
| [NumPy](https://numpy.org/) | BSD 3-Clause |
| [pandas](https://pandas.pydata.org/) | BSD 3-Clause |
| [xarray](https://xarray.dev/) | Apache 2.0 |
| [Matplotlib](https://matplotlib.org/) | Matplotlib licence (PSF-based) |
| [Vega-Altair](https://altair-viz.github.io/) | BSD 3-Clause |
| [netCDF4](https://unidata.github.io/netcdf4-python/) | MIT |

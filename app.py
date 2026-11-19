#!/usr/bin/env python3
"""
Ilmastoneuvottelupeli — FaIR-pohjainen luokkahuonedemosovellus.

Screen flow (SCREENS):
  pledges   Phase 1: each group sets its 2040 fossil CO2 cut (CHECKPOINT)
  prior     what is known before anything is drawn: ECS and 2100 warming over the whole ensemble
  draw      one ensemble member ("one possible Earth") is drawn; its ECS stays hidden
  sim1      FaIR run of the drawn member to 2040; single year 2040 vs the decade centred on 2040
  obs2040   update 1: the observed 2031-2040 mean narrows the estimates (only a little)
  research  update 2: a noisy research estimate of ECS narrows them clearly
  phase2    Phase 2: each group sets its net-zero year; FaIR run to the peak
  reveal    the drawn member's true ECS and 2100 warming against the estimates
The impacts page can be opened from any screen.

All temperatures are annual means (year Y = mean of FaIR timebounds Y and Y+1).
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
from matplotlib.transforms import blended_transform_factory
import sys
import streamlit as st
import streamlit.components.v1 as components

# Pass --sliders on the command line to enable free continuous sliders
# instead of the default radio buttons.  Useful for instructor demos.
USE_SLIDERS: bool = '--sliders' in sys.argv

from fair_runner import (
    BASE_YEAR,
    REGION_EMISSIONS_BASE_GTCO2,
    load_screened_configs,
    load_observed_gmst,
    run_fair_game,
    annual_series,
    regional_emissions_in_year,
    emulate_temperature,
    posterior_weights,
    weighted_mean_interval,
)

TOTAL_BASE   = sum(REGION_EMISSIONS_BASE_GTCO2.values())
PARIS_15     = 1.5
PARIS_20     = 2.0
CHECKPOINT      = 2040   # Phase 1 pledge year
ANIM_START      = 2026   # first year of the simulated line, animated year by year. The observed
                         # record (IGCC) ends with 2025, so the model is drawn only after it.
ANIM_INTERVAL   = 80     # ms per animation frame (increase to slow down)
ESTIMATE_YEAR   = 2100   # year of the warming estimate shown to the players

# HOW THE ESTIMATES ARE UPDATED
# Prior: every screened member is equally likely (the member is drawn uniformly).
# Update 1 ("Havainto 2040"): the players observe the drawn member's mean temperature over
# OBS_WINDOW, with N(0, OBS_2040_SD) error. A decadal mean, not the single year 2040, because a
# single year is dominated by internal variability. OBS_2040_SD is observational uncertainty of a
# decadal GMST mean; the member's own internal variability is NOT added, because in the game it is
# part of the member's identity (fixed seed), so the emulated window mean already contains it.
# Checked offline (400 simulated games): the ECS 95 % interval narrows only to ~80 % of the prior,
# because 2040 warming mostly reflects the transient response and aerosol forcing, not ECS.
# Update 2 ("Tutkimus etenee"): a research estimate of ECS = true ECS + N(0, ECS_CLUE_SD), standing
# for paleoclimate and process studies. ECS_CLUE_SD = 0.5 °C is tighter than AR6 (likely range
# 2.5-4 °C, i.e. sigma ~0.75 °C): research is assumed to progress by 2040. Narrows the ECS interval
# to ~45 % of the prior but 2100 warming only to ~70 % (it also depends on TCR, aerosols, carbon cycle).
# The two error draws are independent, so the posterior weights multiply. Both are drawn once per
# game, at the first simulation, so revisiting screens cannot average the noise away.
OBS_WINDOW      = (2031, 2040)   # annual years observed by the end of 2040 (trailing decade)
OBS_2040_SD     = 0.05           # °C
ECS_CLUE_SD     = 0.5            # °C
# Decade centred on 2040: annual years 2035-2044 (mid-points 2035.5…2044.5 centre on 2040.0).
# By 2040 only 2035-2040 are known; 2041-2044 are projected with the members weighted by update 1
# (checked offline: error sd 0.04 °C, vs 0.06 °C for the single year 2040 and 0.08 °C if the
# unweighted prior were used for the projection). Paris thresholds are judged on this mean,
# like the WMO/Copernicus "current warming level" (observations + constrained projections).
CENTRED_WINDOW  = (2035, 2044)
EXCEEDANCE_LEVELS = (1.5, 2.0, 3.0, 4.0)   # °C, the warming levels of the impacts page

# Emission reduction options per region (4 discrete choices, ordered least to most ambitious).
# Based on emission_reduction_options.md.
# Each entry: (Finnish label, cut_fraction) where cut_fraction > 0 = reduction,
# < 0 = emissions increase (Passiivinen / Varovainen for some regions).
EMISSION_OPTIONS = {
    'Kiina 🇨🇳': [
        ('Passiivinen (+15 %)',   -0.15),   # SSP3-7.0 / SSP5-8.5 BAU growth
        ('Varovainen (−10 %)',    0.10),
        ('Vahva (−40 %)',         0.40),
        ('Maksimaalinen (−65 %)', 0.65),
    ],
    'USA 🇺🇸': [
        ('Passiivinen (0 %)',      0.00),   # No additional policy beyond baseline
        ('Varovainen (−20 %)',    0.20),
        ('Vahva (−55 %)',         0.55),
        ('Maksimaalinen (−80 %)', 0.80),
    ],
    'EU 🇪🇺': [
        ('Passiivinen (−20 %)',   0.20),   # Below current NDC, minimal action
        ('Varovainen (−55 %)',    0.55),
        ('Vahva (−75 %)',         0.75),
        ('Maksimaalinen (−90 %)', 0.90),
    ],
    'Intia 🇮🇳': [
        ('Passiivinen (+80 %)',  -0.80),   # Strong BAU growth, SSP5-like
        ('Varovainen (+35 %)',   -0.35),   # NDC intensity target → absolute rise
        ('Vahva (−20 %)',         0.20),
        ('Maksimaalinen (−50 %)', 0.50),
    ],
    'Muu maailma 🌍': [
        ('Passiivinen (+20 %)',  -0.20),   # Developing-world BAU growth
        ('Varovainen (−15 %)',    0.15),
        ('Vahva (−35 %)',         0.35),
        ('Maksimaalinen (−60 %)', 0.60),
    ],
}

# Maximum post-CHECKPOINT decarbonisation rate: 3 % of each region's BASE_YEAR baseline
# per year (in absolute GtCO2 terms, not fraction of remaining stock).
# Basis: AR6 WGIII 1.5 °C pathways; fastest observed national transitions.
# Sets a physically plausible floor on the net-zero year given Phase 1 ambition.
MAX_DECARB_RATE_PER_YEAR = 0.03

# Net-zero timing options for Phase 2 (offsets in years from the per-region minimum).
# Minimum = CHECKPOINT + ceil(em_remaining / MAX_DECARB_RATE_PER_YEAR).
NETZERO_OFFSETS = [
    ('Nopein mahdollinen',     0),
    ('Maltillinen (+25 v)',   25),
    ('Hidas (+75 v)',         75),
    ('Ei sitouduta (+150 v)', 150),  # net-zero year well past 2100; emissions stay high throughout simulation
]

# (temp °C, color, label shown right of y-axis)
# Two lines separated by 
TEMP_ANNOTATIONS = [
    (1.5, '#65C06C', '70–90 % koralleista häviää'),
    (2.0, '#FFA73E', 'korallit katoavat\nItämeri ei enää jäädy'),
    (3.0, '#E2824A', 'Jäämeri jäätön useimpina kesinä\nLapin porotalous uhattuna'),
    (4.0, '#E24444', 'helleaallot 9,4×\netelä-Suomi käytännössä lumeton'),
]

REGION_COLORS = {
    'Kiina 🇨🇳':      '#E26D91',
    'USA 🇺🇸':         '#5A84E2',
    'EU 🇪🇺':          '#65C06C',
    'Intia 🇮🇳':       '#FFA73E',
    'Muu maailma 🌍': '#8760C1',
}

# Impact data at four warming levels (vs. 1850–1900 pre-industrial baseline).
# Each entry: (icon, label, value_string, severity_0_to_100)
# Global figures from IPCC AR6 WG1 (2021) and SR1.5 (2018).
# Finland figures from Finnish Meteorological Institute projections
# (Finland warms ~1.5–2× the global average due to Arctic amplification).
IMPACTS = {
    1.5: {
        'color': '#65C06C',
        'global': [
            ('🌊', 'Merenpinnannousu (2100)',         '~40 cm',                             25),
            ('🪸', 'Koralliriutat',                   '70–90 % häviää',                     35),
            ('🧊', 'Arktis: jäätön kesä',             'kerran sadassa vuodessa',             15),
            ('🦋', 'Lajien elinympäristö puolittuu',  '~6 % hyönteisistä',                  20),
            ('☀️', 'Äärimmäiset helleaallot',         '4× yleisempiä kuin 1800-luvulla',    20),
        ],
        'finland': [
            ('🌡️', 'Lämpötilan nousu Suomessa',       '+2–3 °C',                            25),
            ('🏙️', 'Helsinki talvella',               'kuin Tukholma tänään',               25),
            ('❄️', 'Lumiset talvet Etelä-Suomessa',  'harvinaisempia',                      25),
            ('⛷️', 'Hiihtosesonki etelässä',          'lyhenee 2–3 viikkoa',                20),
            ('🦌', 'Porotalous Lapissa',              'jäätymisriski kasvaa',                20),
        ],
    },
    2.0: {
        'color': '#FFA73E',
        'global': [
            ('🌊', 'Merenpinnannousu (2100)',         '~50 cm',                              45),
            ('🪸', 'Koralliriutat',                   '>99 % häviää — käytännössä kaikki',  75),
            ('🧊', 'Arktis: jäätön kesä',             'kerran kymmenessä vuodessa',          40),
            ('🦋', 'Lajien elinympäristö puolittuu',  '~18 % hyönteisistä',                 45),
            ('☀️', 'Äärimmäiset helleaallot',         '5,6× yleisempiä',                    45),
        ],
        'finland': [
            ('🌡️', 'Lämpötilan nousu Suomessa',       '+3–4 °C',                            45),
            ('🏙️', 'Helsinki talvella',               'kuin Varsova tänään',                45),
            ('❄️', 'Lumiset talvet Etelä-Suomessa',  'epäluotettavia',                      50),
            ('⛷️', 'Hiihtosesonki etelässä',          'lyhenee 1–2 kuukautta',              50),
            ('🦌', 'Porotalous Lapissa',              'merkittävä uhka',                     45),
        ],
    },
    3.0: {
        'color': '#E2824A',
        'global': [
            ('🌊', 'Merenpinnannousu (2100)',         '~80 cm',                              65),
            ('🪸', 'Koralliriutat',                   'käytännössä hävinneet',               90),
            ('🧊', 'Arktis: jäätön kesä',             'useimpina vuosina',                   70),
            ('🦋', 'Lajien elinympäristö puolittuu',  '~49 % hyönteisistä',                 70),
            ('☀️', 'Äärimmäiset helleaallot',         '~8× yleisempiä',                     70),
        ],
        'finland': [
            ('🌡️', 'Lämpötilan nousu Suomessa',       '+4,5–6 °C',                          65),
            ('🏙️', 'Helsinki talvella',               'kuin Praha tänään',                  65),
            ('❄️', 'Lumiset talvet Etelä-Suomessa',  'hyvin harvinaisia',                   75),
            ('⛷️', 'Hiihtosesonki etelässä',          'siirtynyt pohjoiseen',               75),
            ('🦌', 'Porotalous Lapissa',              'vakavasti uhattuna',                   75),
        ],
    },
    4.0: {
        'color': '#E24444',
        'global': [
            ('🌊', 'Merenpinnannousu (2100)',         '~100 cm+',                            85),
            ('🪸', 'Koralliriutat',                   'hävinneet',                          100),
            ('🧊', 'Arktis: jäätön kesä',             'joka vuosi',                          95),
            ('🦋', 'Lajien elinympäristö puolittuu',  '>35 % lajeista',                     85),
            ('☀️', 'Äärimmäiset helleaallot',         '9,4× yleisempiä',                    90),
        ],
        'finland': [
            ('🌡️', 'Lämpötilan nousu Suomessa',       '+6–8 °C',                            85),
            ('🏙️', 'Helsinki talvella',               'kuin Wien tänään',                   85),
            ('❄️', 'Lumiset talvet Etelä-Suomessa',  'kadonnut muisto',                     95),
            ('⛷️', 'Hiihtosesonki etelässä',          'ei enää taloudellisesti kannattavaa', 90),
            ('🦌', 'Porotalous Lapissa',              'romahtamassa',                        90),
        ],
    },
}


def _plain_name(region: str) -> str:
    """Return the text part of a region name, dropping the trailing flag emoji."""
    return region.rsplit(' ', 1)[0]


def _min_netzero_year(cut_fraction: float) -> int:
    """
    Earliest physically plausible net-zero year given a Phase 1 cut fraction.

    em_remaining = 1 − cut_fraction is the fraction of BASE_YEAR emissions still
    present at CHECKPOINT. For India Varovainen (cut = −0.35), em_remaining = 1.35.
    Dividing by MAX_DECARB_RATE_PER_YEAR gives the minimum years needed from CHECKPOINT.
    """
    em_remaining = 1.0 - cut_fraction
    years_needed = int(np.ceil(em_remaining / MAX_DECARB_RATE_PER_YEAR))
    return CHECKPOINT + max(years_needed, 1)


@st.cache_data
def _load_screened():
    return load_screened_configs()


@st.cache_data
def _load_gmst():
    return load_observed_gmst()


# Screens in game order (see the module docstring). Players move forward with the primary
# buttons and can step back; the impacts page opens on top of any screen.
SCREENS = ['pledges', 'prior', 'draw', 'sim1', 'obs2040', 'research', 'phase2', 'reveal']
SCREEN_TITLES = {
    'pledges':  'Lupaukset',
    'prior':    'Alkuarvio',
    'draw':     'Arvonta',
    'sim1':     'Simulaatio 2040',
    'obs2040':  'Havainto 2040',
    'research': 'Tutkimus etenee',
    'phase2':   'Nollapäästöt',
    'reveal':   'Paljastus',
}
# Estimate stages: (Finnish label, colour). A stage is shown from its own screen onwards.
STAGE_STYLE = {
    'prior':    ('Alkuarvio', '#9AA0A6'),
    'obs2040':  ('Havainnon 2040 jälkeen', '#5A84E2'),
    'research': ('Tutkimuksen jälkeen', '#E26D91'),
}
CHART_BG = '#0e1117'


def _init_state():
    # Default Phase 1: index 0 (Passiivinen, no new policy) for all regions
    default_cuts = {r: EMISSION_OPTIONS[r][0][1] for r in REGION_EMISSIONS_BASE_GTCO2}
    defaults = {
        'screen':          'pledges',
        'furthest':        0,        # index in SCREENS of the furthest screen reached
        'show_impacts':    False,
        'option_indices':  {r: 0 for r in REGION_EMISSIONS_BASE_GTCO2},
        'cuts':            default_cuts,
        # Default Phase 2: Maltillinen (index 1) for all regions
        'netzero_indices': {r: 1 for r in REGION_EMISSIONS_BASE_GTCO2},
        'netzero_years':   {
            r: _min_netzero_year(default_cuts[r]) + NETZERO_OFFSETS[1][1]
            for r in REGION_EMISSIONS_BASE_GTCO2
        },
        # Set at the draw: the member (hidden from the players until the reveal)
        'config_id':       None,
        'ecs':             None,
        'grid_position':   None,     # cell highlighted in the lottery grid (anonymous)
        # Set at the first simulation (annual-mean series, dim 'year')
        'temp_phase1':     None,
        'obs_window_mean': None,     # observed OBS_WINDOW mean, with observation error
        'ecs_clue':        None,     # research estimate of ECS, with error
        'weights_obs':     None,     # member weights after update 1 (pd.Series by config_id)
        'weights_research': None,    # member weights after updates 1 and 2
        'centred_decade':  None,     # (mean, lo, hi) of the CENTRED_WINDOW mean, °C
        # Set at the Phase 2 simulation
        'temp_phase2':     None,
        'phase2_netzero_used': None, # net-zero years of temp_phase2, to detect stale results
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _goto(screen):
    st.session_state['screen'] = screen
    st.session_state['show_impacts'] = False
    st.session_state['furthest'] = max(st.session_state['furthest'], SCREENS.index(screen))
    st.rerun()


def _window_mean_members(cuts, window):
    """Emulated mean of the annual temperatures window[0]..window[1] for every member (Phase 1 pathway)."""
    first, last = window
    total = sum(emulate_temperature(cuts, None, year) for year in range(first, last + 1))
    return total / (last - first + 1)


def _simulate_phase1(cuts):
    """
    Run FaIR for the drawn member to 2040 and draw both noisy pieces of information of the game.
    Both are drawn here, once, so that revisiting the screens cannot average the noise away.
    """
    config_id = st.session_state['config_id']
    temp = annual_series(run_fair_game(cuts, netzero_years=None, config_id=config_id))
    st.session_state['temp_phase1'] = temp

    # Update 1: the drawn member's OBS_WINDOW mean (from the FaIR run) + observation error.
    # The emulator reproduces the FaIR window mean to ~0.001 °C, so the reference is consistent.
    true_window = float(temp.sel(year=slice(*OBS_WINDOW)).mean())
    observed = true_window + float(np.random.normal(0.0, OBS_2040_SD))
    reference = _window_mean_members(cuts, OBS_WINDOW)
    weights_obs = posterior_weights(reference, observed, OBS_2040_SD)

    # Update 2: research estimate of ECS. Independent error, so the weights multiply.
    ecs = _load_screened()['ECS'].loc[weights_obs.index]
    clue = st.session_state['ecs'] + float(np.random.normal(0.0, ECS_CLUE_SD))
    weights_research = weights_obs * posterior_weights(ecs, clue, ECS_CLUE_SD)
    weights_research = weights_research / weights_research.sum()

    # Mean of the decade centred on 2040: the drawn member's known years up to 2040, the rest
    # projected per member. Its spread across the update-1-weighted members is the uncertainty
    # that comes from not yet knowing 2041-2044.
    known_sum = float(temp.sel(year=slice(CENTRED_WINDOW[0], OBS_WINDOW[1])).sum())
    projected_sum = sum(emulate_temperature(cuts, None, year)
                        for year in range(OBS_WINDOW[1] + 1, CENTRED_WINDOW[1] + 1))
    n_years = CENTRED_WINDOW[1] - CENTRED_WINDOW[0] + 1
    decade_members = (known_sum + projected_sum) / n_years
    st.session_state['centred_decade'] = weighted_mean_interval(decade_members, weights_obs)

    st.session_state['obs_window_mean'] = observed
    st.session_state['ecs_clue'] = clue
    st.session_state['weights_obs'] = weights_obs
    st.session_state['weights_research'] = weights_research
    # Phase 2 starts from Maltillinen (index 1), anchored to these Phase 1 cuts
    st.session_state['netzero_indices'] = {r: 1 for r in REGION_EMISSIONS_BASE_GTCO2}
    st.session_state['netzero_years'] = {
        r: _min_netzero_year(cuts[r]) + NETZERO_OFFSETS[1][1] for r in REGION_EMISSIONS_BASE_GTCO2
    }
    st.session_state['temp_phase2'] = None
    st.session_state['phase2_netzero_used'] = None


def _stage_weights(upto_screen):
    """{stage: member weights} of the estimate stages reached by `upto_screen`, oldest first."""
    screened = _load_screened()
    stages = {'prior': pd.Series(1.0 / len(screened), index=screened.index)}
    position = SCREENS.index(upto_screen)
    if position >= SCREENS.index('obs2040') and st.session_state['weights_obs'] is not None:
        stages['obs2040'] = st.session_state['weights_obs']
    if position >= SCREENS.index('research') and st.session_state['weights_research'] is not None:
        stages['research'] = st.session_state['weights_research']
    return stages


def _exceedance(temperature, weights):
    """Probability (0-1) that the temperature exceeds each of EXCEEDANCE_LEVELS."""
    values = temperature.loc[weights.index].values
    return {level: float(weights.values[values > level].sum()) for level in EXCEEDANCE_LEVELS}


def _estimate_figure(temperature, stages, truth=None):
    """
    Weighted histograms of ECS (left) and ESTIMATE_YEAR warming (right), one per estimate stage
    (newest stage solid, older ones faint), with each stage's mean and 95 % interval as a bar
    underneath. truth = (ecs, temperature) marks the drawn member's real values (reveal).
    """
    ecs = _load_screened()['ECS']
    fig, axes = plt.subplots(2, 2, figsize=(11, 4.4), sharex='col',
                             gridspec_kw={'height_ratios': [3, 1.3], 'hspace': 0.08})
    fig.patch.set_facecolor(CHART_BG)
    panels = [
        (ecs, 'Ilmastoherkkyys (ECS)', 'ECS (°C)', None if truth is None else truth[0]),
        (temperature, f'Lämpeneminen vuonna {ESTIMATE_YEAR}', 'Lämpötila-anomalia (°C)',
         None if truth is None else truth[1]),
    ]
    n_stages = len(stages)
    for col, (values, title, xlabel, true_value) in enumerate(panels):
        ax_hist, ax_bar = axes[0, col], axes[1, col]
        for ax in (ax_hist, ax_bar):
            ax.set_facecolor(CHART_BG)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')
        bins = np.linspace(values.min(), values.max(), 36)
        bar_transform = blended_transform_factory(ax_bar.transAxes, ax_bar.transData)
        for row, (stage, weights) in enumerate(stages.items()):
            label, color = STAGE_STYLE[stage]
            newest = row == n_stages - 1
            ax_hist.hist(values.loc[weights.index].values, bins=bins, weights=weights.values,
                         density=True, histtype='stepfilled', color=color,
                         alpha=0.85 if newest else 0.30, label=label)
            mean, lo, hi = weighted_mean_interval(values, weights)
            ax_bar.plot([lo, hi], [row, row], color=color, linewidth=7, solid_capstyle='butt',
                        alpha=1.0 if newest else 0.6)
            ax_bar.plot([mean], [row], 'o', color='white', markersize=5)
            ax_bar.text(1.01, row, f'{mean:.1f} ({lo:.1f}–{hi:.1f})', transform=bar_transform,
                        color=color, fontsize=9, va='center', clip_on=False)
        if col == 1:
            ax_hist.axvline(PARIS_15, color='#65C06C', linewidth=1.2, linestyle='--')
            ax_hist.axvline(PARIS_20, color='#FFA73E', linewidth=1.2, linestyle='--')
        if true_value is not None:
            for ax in (ax_hist, ax_bar):
                ax.axvline(true_value, color='white', linewidth=2.0, linestyle=':')
            ax_hist.text(true_value, ax_hist.get_ylim()[1] * 0.95, f' oikea arvo {true_value:.2f}',
                         color='white', fontsize=9, va='top')
        ax_hist.set_title(title, color='white', fontsize=11)
        ax_hist.set_yticks([])
        ax_bar.set_yticks([])
        ax_bar.set_ylim(n_stages - 0.4, -0.6)    # oldest stage on top
        ax_bar.set_xlabel(xlabel, color='white')
    # Legend above the panels, so it never hides the histograms or the true-value label
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
               labelcolor='white', fontsize=9, bbox_to_anchor=(0.45, 1.0))
    fig.subplots_adjust(left=0.03, right=0.88, wspace=0.32, bottom=0.14, top=0.86)
    return fig


def _show_estimates(cuts, netzero_years, upto_screen, truth=None):
    """Histograms of the estimate stages and the exceedance probabilities of the newest stage."""
    temperature = emulate_temperature(cuts, netzero_years, ESTIMATE_YEAR)
    stages = _stage_weights(upto_screen)
    st.pyplot(_estimate_figure(temperature, stages, truth))
    st.caption(
        'Pylväät: kuinka todennäköinen kukin arvo on. Alla: paras arvio (piste) ja 95 %:n väli – '
        'oikea arvo osuu tälle välille 19 kertaa 20:stä.'
    )

    newest = list(stages)[-1]
    probabilities = _exceedance(temperature, stages[newest])
    prior_probabilities = _exceedance(temperature, stages['prior'])
    st.markdown(f'**Todennäköisyys, että lämpeneminen vuonna {ESTIMATE_YEAR} ylittää tason** '
                f'({STAGE_STYLE[newest][0].lower()}):')
    cols = st.columns(len(EXCEEDANCE_LEVELS))
    for col, level in zip(cols, EXCEEDANCE_LEVELS):
        delta = None
        change = (probabilities[level] - prior_probabilities[level]) * 100
        if newest != 'prior' and abs(change) >= 0.5:   # no "-0" arrows for rounding noise
            delta = f'{change:+.0f} %-yks. alkuarviosta'
        col.metric(f'yli {level:g} °C'.replace('.', ','), f'{probabilities[level] * 100:.0f} %',
                   delta=delta, delta_color='inverse')
    return temperature, stages


def _interval_width_change(values, stages, old, new):
    """Widths (°C) of the 95 % interval of values at two stages."""
    _, lo_old, hi_old = weighted_mean_interval(values, stages[old])
    _, lo_new, hi_new = weighted_mean_interval(values, stages[new])
    return hi_old - lo_old, hi_new - lo_new


def _lottery_html(n_cells, highlight):
    """Grid of n_cells anonymous 'Earths' with one highlighted (positions carry no information)."""
    cells = []
    for i in range(n_cells):
        if i == highlight:
            style = 'background:#FFD54A;box-shadow:0 0 10px 3px #FFD54A;'
        else:
            style = 'background:#2f3b52;'
        cells.append(f'<div style="width:12px;height:12px;border-radius:50%;{style}"></div>')
    return ('<div style="display:flex;flex-wrap:wrap;gap:4px;max-width:960px;'
            'padding:12px 0">' + ''.join(cells) + '</div>')


def _impacts_page():
    st.title('🌡️ Mitä lämpeneminen tarkoittaa?')
    st.markdown(
        'Valitse lämpenemistaso nähdäksesi vaikutukset maailmalla ja Suomessa.  \n'
        '*(Verrattuna esiteolliseen aikaan 1850–1900)*'
    )
    if st.button('⬅ Takaisin peliin'):
        st.session_state['show_impacts'] = False
        st.rerun()
    st.markdown('---')

    selected = st.radio(
        'Lämpenemistaso',
        options=[1.5, 2.0, 3.0, 4.0],
        format_func=lambda x: f'+{x} °C',
        horizontal=True,
        key='impacts_temp',
        label_visibility='collapsed',
    )

    data   = IMPACTS[selected]
    color  = data['color']

    # Large temperature badge
    st.markdown(
        f'<div style="text-align:center;padding:16px 0 24px 0">'
        f'<span style="font-size:3.5em;font-weight:900;color:{color}">'
        f'+{selected} °C</span></div>',
        unsafe_allow_html=True,
    )

    def _bar(pct):
        return (
            f'<div style="background:#2a2a3a;border-radius:4px;height:8px;margin:4px 0 14px 0">'
            f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>'
            f'</div>'
        )

    col_global, col_finland = st.columns(2)

    with col_global:
        st.subheader('🌍 Maailmanlaajuisesti')
        for icon, label, value, sev in data['global']:
            st.markdown(f'**{icon} {label}**')
            st.markdown(
                f'<span style="color:{color};font-size:1.05em">{value}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(_bar(sev), unsafe_allow_html=True)

    with col_finland:
        st.subheader('🇫🇮 Suomessa')
        for icon, label, value, sev in data['finland']:
            st.markdown(f'**{icon} {label}**')
            st.markdown(
                f'<span style="color:{color};font-size:1.05em">{value}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(_bar(sev), unsafe_allow_html=True)

    # Link the chosen level to the players' own decisions: the probability of exceeding it in
    # ESTIMATE_YEAR with the current knowledge (newest estimate stage reached so far).
    st.markdown('---')
    screen = st.session_state['screen']
    netzero_years = st.session_state['netzero_years'] if screen in ('phase2', 'reveal') else None
    temperature = emulate_temperature(st.session_state['cuts'], netzero_years, ESTIMATE_YEAR)
    stages = _stage_weights(screen)
    newest = list(stages)[-1]
    probability = _exceedance(temperature, stages[newest])[selected]
    decisions = 'lupauksillanne ja nollapäästövuosillanne' if netzero_years else 'lupauksillanne'
    message = (f'📊 **Teidän {decisions}** lämpeneminen ylittää **+{selected} °C** vuonna '
               f'{ESTIMATE_YEAR} todennäköisyydellä **{probability * 100:.0f} %** '
               f'({STAGE_STYLE[newest][0].lower()}).')
    if st.session_state['centred_decade'] is not None:
        decade_mean = st.session_state['centred_decade'][0]
        message += (f'  \nSimulaationne vuosikymmenen {CENTRED_WINDOW[0]}–{CENTRED_WINDOW[1]} '
                    f'keskilämpötila: **+{decade_mean:.2f} °C**.')
    st.info(message)

    st.markdown(
        '<p style="color:#4A5568;font-size:0.8em;margin-top:24px">'
        'Lähteet: IPCC AR6 WG1 (2021), SR1.5 (2018) · '
        'Suomen luvut: Ilmatieteen laitos</p>',
        unsafe_allow_html=True,
    )


def _nav_bar():
    """Step indicator, back/forward between reached screens, and the impacts page."""
    screen = st.session_state['screen']
    position = SCREENS.index(screen)
    furthest = st.session_state['furthest']
    steps = []
    for i, name in enumerate(SCREENS):
        title = f'{i + 1} {SCREEN_TITLES[name]}'
        # Colours via opacity/inherit so the indicator reads on both light and dark themes
        if i == position:
            steps.append(f'<b style="color:#E2824A">{title}</b>')
        elif i <= furthest:
            steps.append(f'<span>{title}</span>')
        else:
            steps.append(f'<span style="opacity:0.35">{title}</span>')
    st.markdown('<div style="font-size:0.9em">' + ' &nbsp;›&nbsp; '.join(steps) + '</div>',
                unsafe_allow_html=True)

    col_back, col_forward, col_impacts = st.columns([1, 1, 1])
    with col_back:
        if position > 0 and st.button('⬅ Edellinen'):
            st.session_state['screen'] = SCREENS[position - 1]
            st.rerun()
    with col_forward:
        if position < furthest and st.button('Seuraava ➡'):
            st.session_state['screen'] = SCREENS[position + 1]
            st.rerun()
    with col_impacts:
        if st.button('🌡️ Vaikutukset'):
            st.session_state['show_impacts'] = True
            st.rerun()
    st.markdown('---')


def _make_temperature_animation(temp_da, year_start, year_end,
                                 highlight_year=None, show_peak=False,
                                 phase1_da=None, inset_metrics=None, decade=None):
    """
    Temperature trajectory: observations (static, up to 2025) followed by the simulated
    line from ANIM_START onward, drawn year-by-year. The simulation is not shown over the
    observed period. Returns an HTML string for components.html().

    Both series are annual means plotted at mid-year (Y + 0.5), as the IGCC file stamps them.

    temp_da        : xr.DataArray dim 'year' (annual means, see annual_series), single config
    phase1_da      : if given, draw Phase 1 trajectory as grey dashed line (static)
    highlight_year : dotted vertical line (Phase 1: CHECKPOINT; Phase 2: peak year)
    show_peak      : annotate peak with arrow on final animation frame
    inset_metrics  : list of (label, value_str) shown as text box on final frame
    decade         : (mean, lo, hi) of the CENTRED_WINDOW mean, drawn on the final frame as a
                     bar over the decade: solid for the known years, dashed and shaded for the
                     projected ones. Extends the x axis to the end of the decade.
    """
    gmst = _load_gmst()

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    years  = temp_da.year.values
    mask   = (years >= year_start) & (years <= year_end)
    yr     = years[mask]
    t_vals = temp_da.isel(year=mask).values

    # Only the part after the observed record is shown
    fut_mask = yr >= ANIM_START
    yr_fut   = yr[fut_mask]
    t_fut    = t_vals[fut_mask]
    x_fut    = yr_fut + 0.5    # mid-year, like the observations

    obs_mask = (gmst['year'] >= year_start) & (gmst['year'] <= min(ANIM_START, year_end))

    # Fix axis limits before drawing so text placements are stable
    y_min = min(float(gmst.loc[obs_mask, 'gmst'].min()), 0.0) - 0.15
    y_max = max(float(t_fut.max()), float(gmst.loc[obs_mask, 'gmst'].max())) + 0.45
    x_end = year_end + 1
    if decade is not None:
        y_max = max(y_max, decade[2] + 0.3)
        x_end = max(x_end, CENTRED_WINDOW[1] + 2)
    ax.set_xlim(year_start, x_end)
    ax.set_ylim(y_min, y_max)

    # --- static elements ---
    if phase1_da is not None:
        yr1 = phase1_da.year.values
        m1  = (yr1 >= ANIM_START) & (yr1 <= CHECKPOINT)
        ax.plot(yr1[m1] + 0.5, phase1_da.isel(year=m1).values,
                color='#888888', linewidth=2.5, linestyle='--', label='Vaihe 1')

    ax.plot(gmst.loc[obs_mask, 'year'], gmst.loc[obs_mask, 'gmst'],
            color='white', linewidth=1.5, label='Havaittu')

    ax.axhline(PARIS_15, color='#65C06C', linewidth=1.5, linestyle='--')
    ax.axhline(PARIS_20, color='#FFA73E', linewidth=1.5, linestyle='--')
    ax.text(year_start + 1, PARIS_15 + 0.04, '1,5 °C (Pariisin tavoite)',
            color='#65C06C', fontsize=9)
    ax.text(year_start + 1, PARIS_20 + 0.04, '2,0 °C (Pariisin raja)',
            color='#FFA73E', fontsize=9)

    # Right-side impact annotations: one label per warming threshold.
    # Uses a blended transform so x is in axes fraction (outside right edge)
    # and y is in data coordinates (matches the temperature scale).
    fig.subplots_adjust(right=0.72)
    _rtrans = blended_transform_factory(ax.transAxes, ax.transData)
    for t_ann, col_ann, lbl_ann in TEMP_ANNOTATIONS:
        if t_ann > y_max + 0.15:
            continue                          # off the top of the chart
        if t_ann > PARIS_20:                  # 1.5 and 2.0 already have axhlines above
            ax.axhline(t_ann, color=col_ann, linewidth=1.0,
                       linestyle='--', alpha=0.50)
        ax.text(1.03, t_ann, lbl_ann,
                transform=_rtrans,
                color=col_ann, fontsize=7, va='center',
                linespacing=1.55, clip_on=False)

    if highlight_year is not None:
        ax.axvline(highlight_year + 0.5, color='white', linewidth=1.2, linestyle=':')
        if not show_peak:
            ax.text(highlight_year + 1.5, y_min + 0.05, str(highlight_year),
                    color='white', fontsize=9)

    ax.set_xlabel('Vuosi', color='white')
    ax.set_ylabel('Lämpötila-anomalia (°C)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    # Animated future line (empty at start)
    model_line, = ax.plot([], [], color='#5A84E2', linewidth=2.5, label='Ilmastomalli')
    ax.legend(facecolor='#1e2130', labelcolor='white', fontsize=9)

    # The step from the last observed point (2025) to the first modelled point is drawn only
    # on the final frame, after the rest of the simulated line. Plot only: the FaIR results
    # themselves are not modified.
    jump_line, = ax.plot([], [], color='#5A84E2', linewidth=2.5)
    jump_x = [gmst.loc[obs_mask, 'year'].iloc[-1], x_fut[0]]
    jump_y = [gmst.loc[obs_mask, 'gmst'].iloc[-1], t_fut[0]]

    n_frames = len(yr_fut)

    def update(frame):
        model_line.set_data(x_fut[:frame + 1], t_fut[:frame + 1])
        # On the final frame, add annotations that summarise the result
        if frame == n_frames - 1:
            jump_line.set_data(jump_x, jump_y)
            if decade is not None:
                # The decade covers annual years CENTRED_WINDOW, i.e. x from its first year to
                # one past its last; the years after OBS_WINDOW[1] are not yet known.
                mean, lo, hi = decade
                x0, x_known, x1 = CENTRED_WINDOW[0], OBS_WINDOW[1] + 1, CENTRED_WINDOW[1] + 1
                ax.fill_between([x_known, x1], lo, hi, color='#FFD54A', alpha=0.20, linewidth=0)
                ax.plot([x0, x_known], [mean, mean], color='#FFD54A', linewidth=3)
                ax.plot([x_known, x1], [mean, mean], color='#FFD54A', linewidth=3, linestyle='--')
                ax.text(x0, mean + 0.08,
                        f'{CENTRED_WINDOW[0]}–{CENTRED_WINDOW[1]} keskiarvo {mean:.2f} °C',
                        color='#FFD54A', fontsize=9, ha='left')
            if show_peak:
                peak_temp = float(t_vals.max())
                peak_yr   = int(yr[t_vals.argmax()])
                ax.annotate(
                    f'Huippu: {peak_temp:.2f} °C ({peak_yr})',
                    xy=(peak_yr + 0.5, peak_temp),
                    xytext=(peak_yr + 5.5, peak_temp + 0.1),
                    color='white', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                )
            if inset_metrics:
                lines = '\n'.join(f'{label}:  {val}' for label, val in inset_metrics)
                ax.text(0.02, 0.02, lines,
                        transform=ax.transAxes,
                        color='white', fontsize=10, va='bottom', ha='left',
                        bbox=dict(facecolor='#1e2130', edgecolor='#555555',
                                  boxstyle='round,pad=0.5', alpha=0.9))

    anim = mpl_anim.FuncAnimation(
        fig, update, frames=n_frames,
        interval=ANIM_INTERVAL, blit=False, repeat=False,
    )
    html = anim.to_jshtml()
    plt.close(fig)
    return html


def _emissions_bar_chart(cuts):
    """Bar chart: BASE_YEAR baseline vs. pledged CHECKPOINT emissions per region."""
    regions = list(REGION_EMISSIONS_BASE_GTCO2.keys())
    base    = [REGION_EMISSIONS_BASE_GTCO2[r] for r in regions]
    target  = [REGION_EMISSIONS_BASE_GTCO2[r] * (1 - cuts[r]) for r in regions]
    em_2100 = regional_emissions_in_year(cuts, ESTIMATE_YEAR)
    later   = [em_2100[r] for r in regions]
    colors  = [REGION_COLORS[r] for r in regions]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    x     = np.arange(len(regions))
    width = 0.27
    ax.bar(x - width, base,   width, color=colors, alpha=0.30, label=f'Päästöt {BASE_YEAR}')
    ax.bar(x,         target, width, color=colors, alpha=0.90, label=f'Päästöt {CHECKPOINT} (lupaus)')
    ax.bar(x + width, later,  width, color=colors, alpha=0.60, hatch='//',
           label=f'Päästöt {ESTIMATE_YEAR} (sama vuosimuutos jatkuu)')

    ax.set_xticks(x)
    ax.set_xticklabels([_plain_name(r) for r in regions], color='white', fontsize=9)
    ax.set_ylabel('GtCO₂/vuosi', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e2130', labelcolor='white', fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    fig.tight_layout()
    return fig


def _emissions_pathway_chart(cuts, netzero_years):
    """Stacked area chart: per-region CO2 pathway BASE_YEAR → each region's net-zero year."""
    max_nz   = max(netzero_years.values())
    plot_end = min(max_nz + 10, 2110)
    years    = np.arange(BASE_YEAR, plot_end + 1)
    regions  = list(REGION_EMISSIONS_BASE_GTCO2.keys())

    region_em = {}
    for r, base in REGION_EMISSIONS_BASE_GTCO2.items():
        cut    = cuts.get(r, 0.0)
        em_base = base
        em_chk = base * (1 - cut)
        nz     = netzero_years.get(r, 2070)

        em = np.zeros(len(years))
        for i, y in enumerate(years):
            if y < CHECKPOINT:
                em[i] = em_base + (em_chk - em_base) * (y - BASE_YEAR) / (CHECKPOINT - BASE_YEAR)
            elif y < nz:
                em[i] = em_chk * (1 - (y - CHECKPOINT) / (nz - CHECKPOINT))
            else:
                em[i] = 0.0
        region_em[r] = em

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    stack_bottom = np.zeros(len(years))
    for r in regions:
        ax.fill_between(years, stack_bottom, stack_bottom + region_em[r],
                        color=REGION_COLORS[r], alpha=0.75, label=_plain_name(r))
        stack_bottom += region_em[r]

    ax.axvline(CHECKPOINT, color='white', linewidth=1, linestyle=':')
    ax.set_ylabel('GtCO₂/vuosi', color='white')
    ax.set_xlabel('Vuosi', color='white')
    ax.set_xlim(BASE_YEAR, plot_end)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.legend(facecolor='#1e2130', labelcolor='white', fontsize=8,
              loc='upper right', ncol=2)
    ax.set_title('Päästöpolku alueittain', color='white', fontsize=10)
    fig.tight_layout()
    return fig


def _cut_input(region: str, option_indices: dict, disabled: bool = False) -> float:
    """
    Return the chosen cut fraction for a region.
    In slider mode (--sliders flag): free slider, returns fraction converted from %.
    In radio mode (default): 4-option radio from EMISSION_OPTIONS.
    Updates option_indices[region] in place. disabled locks the input (after the draw).
    """
    if USE_SLIDERS:
        # Initialise slider from whichever radio option is currently selected
        stored_idx  = option_indices.get(region, 0)
        current_pct = int(EMISSION_OPTIONS[region][stored_idx][1] * 100)
        pct = st.slider(
            region, min_value=-50, max_value=100,
            value=current_pct, step=5, format='%d%%',
            key=f'slider_p1_{region}', disabled=disabled,
        )
        return pct / 100.0
    options = EMISSION_OPTIONS[region]
    idx = st.radio(
        region,
        options=range(len(options)),
        format_func=lambda i, opts=options: opts[i][0],
        index=option_indices.get(region, 0),
        key=f'radio_p1_{region}', disabled=disabled,
    )
    option_indices[region] = idx
    return options[idx][1]


def _nz_input(region: str, min_nz: int, netzero_indices: dict) -> int:
    """
    Return the chosen net-zero year for a region.
    In slider mode: free slider starting at min_nz.
    In radio mode: 3 options anchored to min_nz via NETZERO_OFFSETS.
    Updates netzero_indices[region] in place.
    """
    if USE_SLIDERS:
        current = netzero_indices.get(region, 1)
        nz_opts = [(lbl, min_nz + off) for lbl, off in NETZERO_OFFSETS]
        default_yr = nz_opts[current][1]
        max_nz = max(yr for _, yr in nz_opts)
        return st.slider(
            region, min_value=min_nz, max_value=max_nz,
            value=max(default_yr, min_nz), step=5,
            key=f'slider_nz_{region}',
        )
    nz_opts = [(lbl, min_nz + off) for lbl, off in NETZERO_OFFSETS]
    idx = st.radio(
        region,
        options=range(len(nz_opts)),
        format_func=lambda i, o=nz_opts: f'{o[i][0]} ({o[i][1]})',
        index=netzero_indices.get(region, 1),
        key=f'radio_nz_{region}',
    )
    netzero_indices[region] = idx
    return nz_opts[idx][1]


def _screen_pledges():
    st.title('🌍 Ilmastoneuvottelut — Mitä lupaatte?')
    st.markdown(
        'Jokainen ryhmä edustaa yhtä maailman suurta päästöaluetta. '
        f'Sopikaa, kuinka paljon leikkaatte **fossiilisia CO₂-päästöjä** vuoteen **{CHECKPOINT}** mennessä.'
    )
    st.caption(
        'Fossiiliset CO₂-päästöt tarkoittavat fossiilisten polttoaineiden ja teollisuuden päästöjä. '
        'Maankäytön päästöt (esim. metsäkato) eivät muutu pelissä: ne noudattavat SSP2-4.5-skenaariota.'
    )

    _nav_bar()

    # Once a member is drawn the 2040 observation depends on these pledges, so they are locked.
    locked = st.session_state['config_id'] is not None
    if locked:
        st.warning('🔒 Lupaukset on lukittu, koska maapallo on jo arvottu. '
                   'Jos haluatte muuttaa niitä, aloittakaa alusta.')

    cuts          = st.session_state['cuts']
    option_indices = st.session_state['option_indices']

    st.subheader('Aseta jokaisen alueen lupaus')
    cols = st.columns(len(REGION_EMISSIONS_BASE_GTCO2))
    for col, (region, base_em) in zip(cols, REGION_EMISSIONS_BASE_GTCO2.items()):
        with col:
            cut    = _cut_input(region, option_indices, disabled=locked)
            cuts[region] = cut
            em_chk = base_em * (1.0 - cut)
            st.caption(
                f'{BASE_YEAR}: **{base_em:.1f}** GtCO₂/v  \n'
                f'{CHECKPOINT}: **{em_chk:.1f}** GtCO₂/v'
            )
    st.session_state['cuts']          = cuts
    st.session_state['option_indices'] = option_indices

    total_chk     = sum(REGION_EMISSIONS_BASE_GTCO2[r] * (1 - cuts[r]) for r in cuts)
    total_cut_pct = (1 - total_chk / TOTAL_BASE) * 100
    change_str = (f'{total_cut_pct:.0f} % leikkaus' if total_cut_pct >= 0
                  else f'{-total_cut_pct:.0f} % kasvu')
    st.info(
        f'**Yhteensä {CHECKPOINT}:** {total_chk:.1f} GtCO₂/vuosi  '
        f'({change_str} vuoden {BASE_YEAR} tasosta)'
    )
    st.pyplot(_emissions_bar_chart(cuts))

    # Prior estimate (whole screened ensemble, equal weights): mean and 95 % interval.
    est_40  = emulate_temperature(cuts, None, CHECKPOINT)
    est_end = emulate_temperature(cuts, None, ESTIMATE_YEAR)
    st.info(
        '**Alkuarvio lämpötilasta** (keskiarvo ja 95 %:n väli – emme vielä tiedä, '
        'kuinka herkkä maapallomme on):  \n'
        f'{CHECKPOINT}: **{est_40.mean():.1f} °C** '
        f'({est_40.quantile(0.025):.1f}–{est_40.quantile(0.975):.1f} °C)  \n'
        f'{ESTIMATE_YEAR}: **{est_end.mean():.1f} °C** '
        f'({est_end.quantile(0.025):.1f}–{est_end.quantile(0.975):.1f} °C)'
    )
    st.caption(
        f'Oletus {CHECKPOINT} jälkeen: kunkin alueen päästöt muuttuvat vuosittain yhtä paljon kuin '
        f'vuosina {BASE_YEAR}–{CHECKPOINT} (laskevat korkeintaan nollaan; kasvavat pysyvät ennallaan).'
    )

    if st.button('Katsotaan alkuarvio tarkemmin →', type='primary'):
        _goto('prior')


def _screen_prior():
    st.title('🤔 Mitä tiedämme ennen kuin mitään on tapahtunut?')
    _nav_bar()
    st.markdown(
        'Tutkijat eivät tiedä tarkalleen, **kuinka herkästi maapallo reagoi** hiilidioksidiin. '
        f'Siksi ilmastomallissa on {len(_load_screened())} versiota – '
        '**mahdollista maapalloa** – jotka kaikki sopivat yhteen tähänastisten havaintojen kanssa.'
    )
    st.markdown(
        '**Ilmastoherkkyys (ECS)** kertoo, kuinka paljon maapallo lopulta lämpenisi, jos ilman '
        'CO₂-pitoisuus kaksinkertaistuisi pysyvästi. Mitä suurempi ECS, sitä enemmän samat päästöt '
        'lämmittävät.'
    )
    _show_estimates(st.session_state['cuts'], None, 'prior')
    st.info('👆 Tämä on **alkuarvio**. Harmaa jakauma pysyy näkyvissä koko pelin ajan, '
            'jotta näette, miten arvio tarkentuu.')
    if st.button('Arvotaan meidän maapallomme →', type='primary'):
        _goto('draw')


def _screen_draw():
    st.title('🎲 Arvonta: mikä maapalloista on meidän?')
    _nav_bar()
    screened = _load_screened()
    n_members = len(screened)
    st.markdown(
        f'Arvotaan yksi {n_members} mahdollisesta maapallosta. Se on pelin "todellinen" maapallo: '
        '**sen ilmastoherkkyys on salaisuus**, ja teidän on pääteltävä sitä havainnoista, '
        'kuten tutkijat tekevät oikeassakin maailmassa.'
    )

    placeholder = st.empty()
    if st.session_state['config_id'] is None:
        placeholder.markdown(_lottery_html(n_members, None), unsafe_allow_html=True)
        if st.button('🎲 Arvo maapallo', type='primary'):
            # Flicker through random cells, slowing down. The cell is only a visual: the member is
            # drawn uniformly and independently, so the grid position carries no information.
            for delay in np.linspace(0.04, 0.28, 18):
                placeholder.markdown(_lottery_html(n_members, np.random.randint(n_members)),
                                     unsafe_allow_html=True)
                time.sleep(delay)
            config_id = int(screened.index[np.random.randint(n_members)])
            st.session_state['config_id'] = config_id
            st.session_state['ecs'] = float(screened.loc[config_id, 'ECS'])
            st.session_state['grid_position'] = int(np.random.randint(n_members))
            st.rerun()
        return

    placeholder.markdown(_lottery_html(n_members, st.session_state['grid_position']),
                         unsafe_allow_html=True)
    st.success(f'✨ Maapallo nro {st.session_state["grid_position"] + 1} arvottu! '
               'Sen ilmastoherkkyys paljastetaan vasta pelin lopussa.')
    if st.session_state['temp_phase1'] is None:
        if st.button(f'🔬 Simuloi {ANIM_START}–{CHECKPOINT} →', type='primary'):
            with st.spinner('Lasketaan FaIR-simulaatio...'):
                _simulate_phase1(st.session_state['cuts'])
            _goto('sim1')


def _screen_sim1():
    st.title(f'📊 Mitä tapahtui vuoteen {CHECKPOINT} mennessä?')
    _nav_bar()
    temp = st.session_state['temp_phase1']
    t_single = float(temp.sel(year=CHECKPOINT))
    decade_mean, decade_lo, decade_hi = st.session_state['centred_decade']
    gmst = _load_gmst()
    obs_last = float(gmst.loc[np.floor(gmst['year']) == ANIM_START - 1, 'gmst'].iloc[0])

    components.html(
        _make_temperature_animation(
            temp, year_start=1900, year_end=CHECKPOINT,
            highlight_year=CHECKPOINT,
            inset_metrics=[
                (f'Havaittu {ANIM_START - 1}', f'+{obs_last:.2f} °C'),
                (f'Vuosi {CHECKPOINT}', f'+{t_single:.2f} °C'),
                (f'Keskiarvo {CENTRED_WINDOW[0]}–{CENTRED_WINDOW[1]}',
                 f'+{decade_mean:.2f} °C ({decade_lo:.2f}–{decade_hi:.2f})'),
            ],
            decade=st.session_state['centred_decade'],
        ),
        height=520,
    )

    # Paris thresholds refer to long-term warming, so the verdict uses the decade centred on 2040.
    if decade_mean < PARIS_15:
        st.success(f'✅ Vuosikymmenen {CENTRED_WINDOW[0]}–{CENTRED_WINDOW[1]} keskilämpötila '
                   'on alle 1,5 °C:n tavoitteen!')
    elif decade_mean < PARIS_20:
        st.warning(f'⚠️ Vuosikymmenen {CENTRED_WINDOW[0]}–{CENTRED_WINDOW[1]} keskilämpötila on '
                   'yli 1,5 °C mutta alle 2 °C:n rajan.')
    else:
        st.error(f'❌ Vuosikymmenen {CENTRED_WINDOW[0]}–{CENTRED_WINDOW[1]} keskilämpötila ylittää '
                 '2 °C:n Pariisin rajan.')

    single_above = t_single >= PARIS_15
    decade_above = decade_mean >= PARIS_15
    st.markdown(
        f'**Miksi keskiarvo eikä vain vuosi {CHECKPOINT}?** Yksittäinen vuosi heittelee luonnollisen '
        'vaihtelun (esim. El Niño) takia. Pariisin sopimuksen rajat tarkoittavat pitkän ajan '
        f'lämpenemistä, joten sitä arvioidaan vuosikymmenen keskiarvona. Vuoden {CHECKPOINT} lopussa '
        f'vuodet {OBS_WINDOW[1] + 1}–{CENTRED_WINDOW[1]} ovat vielä tulevaisuutta, joten ne on '
        'arvioitu (katkoviiva ja varjostus).'
    )
    if single_above != decade_above:
        st.info(f'💡 Huomatkaa: vuosi {CHECKPOINT} ({t_single:.2f} °C) ja vuosikymmenen keskiarvo '
                f'({decade_mean:.2f} °C) ovat eri puolilla 1,5 °C:n rajaa. Yksi vuosi ei vielä kerro, '
                'onko raja ylitetty.')
    st.caption('Oikeassa maailmassa vuosi 2024 oli ensimmäinen yli 1,5 °C:n vuosi (1,51 °C), mutta '
               'vuosien 2016–2025 keskiarvo oli 1,26 °C (IGCC-2025a).')

    if st.button('Mitä opimme havainnoista? →', type='primary'):
        _goto('obs2040')


def _screen_obs2040():
    st.title(f'🔭 Havainto {CHECKPOINT}: mitä lämpeneminen kertoo maapallostamme?')
    _nav_bar()
    observed = st.session_state['obs_window_mean']
    st.markdown(
        f'Vuoden {CHECKPOINT} lopussa tiedämme, kuinka paljon meidän maapallomme lämpeni. '
        f'Mitattu keskilämpötila vuosina {OBS_WINDOW[0]}–{OBS_WINDOW[1]}: '
        f'**{observed:.2f} °C** (mittausepävarmuus ±{1.96 * OBS_2040_SD:.1f} °C).  \n'
        'Maapallot, joiden lämpeneminen sopii tähän mittaukseen, muuttuvat todennäköisemmiksi.'
    )
    temperature, stages = _show_estimates(st.session_state['cuts'], None, 'obs2040')
    ecs = _load_screened()['ECS']
    ecs_old, ecs_new = _interval_width_change(ecs, stages, 'prior', 'obs2040')
    t_old, t_new = _interval_width_change(temperature, stages, 'prior', 'obs2040')
    ecs_narrowing = 1 - ecs_new / ecs_old
    message = (
        f'Ilmastoherkkyyden 95 %:n väli kapeni **{ecs_old:.1f} → {ecs_new:.1f} °C** '
        f'({ecs_narrowing * 100:.0f} %), vuoden {ESTIMATE_YEAR} lämpenemisen '
        f'**{t_old:.1f} → {t_new:.1f} °C** ({(1 - t_new / t_old) * 100:.0f} %). '
    )
    # Typically (median over simulated games) the ECS interval narrows by only ~20 %; it narrows
    # much more when the drawn member is in a tail of the ensemble. Describe the actual case.
    if ecs_narrowing < 0.3:
        message += 'Arvio tarkentui – mutta vähemmän kuin voisi odottaa!'
    else:
        prior_mean = _window_mean_members(st.session_state['cuts'], OBS_WINDOW).mean()
        direction = 'vähemmän' if observed < prior_mean else 'enemmän'
        message += (f'Tällä kertaa havainto kertoi paljon, koska maapallomme lämpeni selvästi '
                    f'{direction} kuin maapallot keskimäärin ({prior_mean:.2f} °C). Yleensä '
                    'arvio tarkentuu tässä vaiheessa vain vähän.')
    st.info(message)
    with st.expander('Miksi 15 vuoden lämpeneminen kertoo ilmastoherkkyydestä yleensä niin vähän?'):
        st.markdown(
            '- Lähivuosikymmenten lämpeneminen riippuu ennen kaikkea **lyhyen aikavälin vasteesta** '
            '(TCR): meret lämpenevät hitaasti, eikä ilmasto ole ehtinyt tasapainoon. ECS kuvaa '
            'lopullista tasapainoa.\n'
            '- **Pienhiukkaset (aerosolit)** viilentävät ilmastoa, ja niiden vaikutus tunnetaan '
            'huonosti. Herkkä maapallo, jolla on voimakas aerosolien viilennys, lämpenee lähes '
            'yhtä paljon kuin vähemmän herkkä, jolla viilennys on heikko.\n'
            '- Sama pätee oikeaan maailmaan: havaittu lämpeneminen rajaa ECS:ää vain löyhästi.'
        )
    if st.button('Tutkimus etenee →', type='primary'):
        _goto('research')


def _screen_research():
    st.title('🧪 Tutkimus etenee')
    _nav_bar()
    clue = st.session_state['ecs_clue']
    st.markdown(
        f'Vuoteen {CHECKPOINT} mennessä tutkijat ovat oppineet lisää esimerkiksi muinaisilmastoista '
        'ja pilvien käyttäytymisestä. Uusi tutkimus arvioi meidän maapallomme ilmastoherkkyydeksi '
        f'**{clue:.1f} °C ± {1.96 * ECS_CLUE_SD:.1f} °C** (95 %:n väli).'
    )
    temperature, stages = _show_estimates(st.session_state['cuts'], None, 'research')
    ecs = _load_screened()['ECS']
    ecs_old, ecs_new = _interval_width_change(ecs, stages, 'prior', 'research')
    t_old, t_new = _interval_width_change(temperature, stages, 'prior', 'research')
    st.info(
        f'Alkuarvioon verrattuna ilmastoherkkyyden 95 %:n väli kapeni **{ecs_old:.1f} → '
        f'{ecs_new:.1f} °C** ({(1 - ecs_new / ecs_old) * 100:.0f} %), vuoden {ESTIMATE_YEAR} '
        f'lämpenemisen **{t_old:.1f} → {t_new:.1f} °C** ({(1 - t_new / t_old) * 100:.0f} %).'
    )
    with st.expander('Miksi lämpenemisen arvio ei tarkentunut yhtä paljon kuin ECS?'):
        st.markdown(
            f'Vuoden {ESTIMATE_YEAR} lämpeneminen ei riipu vain ilmastoherkkyydestä. Siihen '
            'vaikuttavat myös lyhyen aikavälin vaste, aerosolien viilennys ja se, kuinka paljon '
            'meret ja kasvillisuus sitovat hiilidioksidia. Vaikka ECS tunnettaisiin tarkasti, '
            'lämpenemisessä olisi silti epävarmuutta.'
        )
    if st.button('Vaihe 2: Nollapäästövuodet →', type='primary'):
        _goto('phase2')


def _screen_phase2():
    st.title('🎯 Milloin kukin alue saavuttaa nollapäästöt?')
    st.markdown(
        'Jokainen ryhmä päättää, milloin **oman alueensa** fossiiliset CO₂-päästöt laskevat nollaan. '
        'Metaani, maankäytön päästöt ja muut kasvihuonekaasut seuraavat nykyistä kehitystä (SSP2-4.5). '
        'Käyttäkää päätöksessä sitä, mitä nyt tiedätte maapallostanne.'
    )

    _nav_bar()

    cuts            = st.session_state['cuts']
    netzero_years   = st.session_state['netzero_years']
    netzero_indices = st.session_state['netzero_indices']

    st.subheader('Valitse nollapäästövuosi alueittain')
    cols = st.columns(len(REGION_EMISSIONS_BASE_GTCO2))
    for col, region in zip(cols, REGION_EMISSIONS_BASE_GTCO2):
        with col:
            min_nz = _min_netzero_year(cuts.get(region, 0.0))
            nz     = _nz_input(region, min_nz, netzero_indices)
            netzero_years[region] = nz
            em_chk = REGION_EMISSIONS_BASE_GTCO2[region] * (1 - cuts.get(region, 0.0))
            st.caption(f'{CHECKPOINT}: {em_chk:.1f} GtCO₂/v → 0 vuoteen **{nz}**')
    st.session_state['netzero_years']   = netzero_years
    st.session_state['netzero_indices'] = netzero_indices

    st.pyplot(_emissions_pathway_chart(cuts, netzero_years))

    st.subheader('Arvio näillä päätöksillä')
    _show_estimates(cuts, netzero_years, 'phase2')

    if st.button('🔬 Laske huipputulämpötila', type='primary'):
        with st.spinner('Lasketaan FaIR-simulaatio...'):
            temp = run_fair_game(
                cuts,
                netzero_years=netzero_years,
                config_id=st.session_state['config_id'],
            )
            st.session_state['temp_phase2'] = annual_series(temp)
            st.session_state['phase2_netzero_used'] = dict(netzero_years)
        st.rerun()

    if st.session_state['temp_phase2'] is None:
        return
    if st.session_state['phase2_netzero_used'] != netzero_years:
        st.info('Valinnat muuttuivat – laskekaa simulaatio uudelleen nähdäksenne tuloksen.')
        return

    temp2 = st.session_state['temp_phase2']
    temp1 = st.session_state['temp_phase1']

    after_chk = temp2.sel(year=temp2.year > CHECKPOINT)
    peak_temp = float(after_chk.max())
    peak_yr   = int(after_chk.year.values[after_chk.values.argmax()])

    sim_end = int(temp2.year.max())
    display_end = min(peak_yr + 30, sim_end)

    st.subheader('📊 Simulaatiotulokset')
    components.html(
        _make_temperature_animation(
            temp2, year_start=1900, year_end=display_end,
            highlight_year=peak_yr,
            show_peak=True,
            phase1_da=temp1,
        ),
        height=520,
    )

    if peak_temp < PARIS_15:
        st.success('✅ Pysytään 1,5 °C:n tavoitteessa!')
    elif peak_temp < PARIS_20:
        st.warning('⚠️ Alle 2 °C, mutta yli 1,5 °C')
    else:
        st.error('❌ Ylitetään 2 °C:n Pariisin raja')

    if st.button('🔎 Paljasta maapallon salaisuus →', type='primary'):
        _goto('reveal')


def _screen_reveal():
    st.title('🔎 Paljastus: millainen maapallomme oikeasti oli?')
    _nav_bar()
    cuts = st.session_state['cuts']
    netzero_years = st.session_state['phase2_netzero_used']
    if netzero_years is None or st.session_state['temp_phase2'] is None:
        st.info('Laskekaa ensin vaiheen 2 simulaatio.')
        return

    true_ecs = st.session_state['ecs']
    true_temp = float(st.session_state['temp_phase2'].sel(year=ESTIMATE_YEAR))
    st.markdown(
        f'Meidän maapallomme ilmastoherkkyys oli **{true_ecs:.2f} °C**, ja teidän päätöksillänne '
        f'se lämpeni vuoteen {ESTIMATE_YEAR} mennessä **{true_temp:.2f} °C**.'
    )
    temperature, stages = _show_estimates(cuts, netzero_years, 'reveal',
                                          truth=(true_ecs, true_temp))

    final = stages['research']
    ecs = _load_screened()['ECS']
    checks = [
        ('Ilmastoherkkyys', ecs, true_ecs),
        (f'Lämpeneminen {ESTIMATE_YEAR}', temperature, true_temp),
    ]
    for name, values, true_value in checks:
        mean, lo, hi = weighted_mean_interval(values, final)
        if lo <= true_value <= hi:
            st.success(f'🎯 {name}: oikea arvo {true_value:.2f} °C osui arvion 95 %:n välille '
                       f'({lo:.2f}–{hi:.2f} °C, paras arvio {mean:.2f} °C).')
        else:
            st.warning(f'😮 {name}: oikea arvo {true_value:.2f} °C jäi arvion 95 %:n välin '
                       f'({lo:.2f}–{hi:.2f} °C) ulkopuolelle. Näin käy noin kerran 20 pelistä – '
                       'se kuuluu epävarmuuteen.')
    st.markdown(
        '**Mitä tästä opimme?** Emme voi tietää etukäteen, kuinka herkkä maapallo on. Päätökset on '
        'silti tehtävä nyt – ja mitä varovaisempia ne ovat, sitä pienempi on riski päätyä '
        'vaarallisen lämpenemisen puolelle.'
    )


def main():
    st.set_page_config(
        page_title='Ilmastoneuvottelupeli',
        page_icon='🌍',
        layout='wide',
    )
    _init_state()

    if st.session_state['show_impacts']:
        _impacts_page()
    else:
        screen_functions = {
            'pledges':  _screen_pledges,
            'prior':    _screen_prior,
            'draw':     _screen_draw,
            'sim1':     _screen_sim1,
            'obs2040':  _screen_obs2040,
            'research': _screen_research,
            'phase2':   _screen_phase2,
            'reveal':   _screen_reveal,
        }
        screen_functions[st.session_state['screen']]()

    st.markdown('---')
    if st.button('🔄 Aloita alusta (uusi maapallo)'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == '__main__':
    main()

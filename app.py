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
The language (Finnish/English) is chosen before the first screen and can be switched at any time;
all user-visible text is in texts.py.

All temperatures are annual means (year Y = mean of FaIR timebounds Y and Y+1).
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import sys
import streamlit as st
import altair as alt

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
from texts import LANGUAGES, lang, t, tr

TOTAL_BASE   = sum(REGION_EMISSIONS_BASE_GTCO2.values())
PARIS_15     = 1.5
PARIS_20     = 2.0
CHECKPOINT      = 2040   # Phase 1 pledge year
ANIM_START      = 2026   # first year of the simulated line, animated year by year. The observed
                         # record (IGCC) ends with 2025, so the model is drawn only after it.
ANIM_SECONDS    = 5.0    # duration of the year-by-year line animation (increase to slow down)
ANIM_MAX_FRAMES = 60     # chart updates sent to the browser; years are grouped beyond this
OBS_COLOR       = '#555B66'   # observed GMST: dark grey on the light theme
MODEL_COLOR     = '#5A84E2'
DECADE_COLOR    = '#D9961A'   # darker than the old yellow so it reads on white too
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
# By 2040 only 2035-2040 are known; 2041-2044 are projected with the members weighted by updates 1
# and 2, since in the story both the observation and the new research exist at the end of 2040.
# Checked offline (1200 simulated games, emulator): error sd of the decade mean 0.038 °C with
# both updates, 0.040 °C with update 1 only, 0.064 °C with update 2 only and 0.079 °C with the
# unweighted prior (vs 0.06 °C for the single year 2040). The research estimate adds little here:
# near-term warming is set by the transient response and aerosol forcing, which the observed
# 2031-2040 mean constrains, not by ECS. Paris thresholds are judged on this mean,
# like the WMO/Copernicus "current warming level" (observations + constrained projections).
CENTRED_WINDOW  = (2035, 2044)
EXCEEDANCE_LEVELS = (1.5, 2.0, 3.0, 4.0)   # °C, the warming levels of the impacts page

# Emission reduction options per region (4 discrete choices, ordered least to most ambitious).
# Based on emission_reduction_options.md.
# Each entry: (level, cut_fraction) where cut_fraction > 0 = reduction,
# < 0 = emissions increase (Passiivinen / Varovainen for some regions).
# The radio label is the level name (OPTION_LEVELS) plus the % change implied by cut_fraction.
OPTION_LEVELS = [
    {'fi': 'Passiivinen',   'en': 'Passive'},
    {'fi': 'Varovainen',    'en': 'Cautious'},
    {'fi': 'Vahva',         'en': 'Strong'},
    {'fi': 'Maksimaalinen', 'en': 'Maximal'},
]
EMISSION_OPTIONS = {
    'Kiina 🇨🇳': [
        (0, -0.15),   # SSP3-7.0 / SSP5-8.5 BAU growth
        (1,  0.10),
        (2,  0.40),
        (3,  0.65),
    ],
    'USA 🇺🇸': [
        (0,  0.00),   # No additional policy beyond baseline
        (1,  0.20),
        (2,  0.55),
        (3,  0.80),
    ],
    'EU 🇪🇺': [
        (0,  0.20),   # Below current NDC, minimal action
        (1,  0.55),
        (2,  0.75),
        (3,  0.90),
    ],
    'Intia 🇮🇳': [
        (0, -0.80),   # Strong BAU growth, SSP5-like
        (1, -0.35),   # NDC intensity target → absolute rise
        (2,  0.20),
        (3,  0.50),
    ],
    'Muu maailma 🌍': [
        (0, -0.20),   # Developing-world BAU growth
        (1,  0.15),
        (2,  0.35),
        (3,  0.60),
    ],
}

# The Finnish region names are the internal IDs (keys of REGION_EMISSIONS_BASE_GTCO2 in
# fair_runner.py and of the session state); this maps them to the displayed names.
REGION_NAMES = {
    'Kiina 🇨🇳':      {'fi': 'Kiina 🇨🇳',      'en': 'China 🇨🇳'},
    'USA 🇺🇸':         {'fi': 'USA 🇺🇸',         'en': 'USA 🇺🇸'},
    'EU 🇪🇺':          {'fi': 'EU 🇪🇺',          'en': 'EU 🇪🇺'},
    'Intia 🇮🇳':       {'fi': 'Intia 🇮🇳',       'en': 'India 🇮🇳'},
    'Muu maailma 🌍': {'fi': 'Muu maailma 🌍', 'en': 'Rest of the world 🌍'},
}

# Maximum post-CHECKPOINT decarbonisation rate: 3 % of each region's BASE_YEAR baseline
# per year (in absolute GtCO2 terms, not fraction of remaining stock).
# Basis: AR6 WGIII 1.5 °C pathways; fastest observed national transitions.
# Sets a physically plausible floor on the net-zero year given Phase 1 ambition.
MAX_DECARB_RATE_PER_YEAR = 0.03

# Net-zero timing options for Phase 2 (offsets in years from the per-region minimum).
# Minimum = CHECKPOINT + ceil(em_remaining / MAX_DECARB_RATE_PER_YEAR).
NETZERO_OFFSETS = [
    ({'fi': 'Nopein mahdollinen',    'en': 'Fastest possible'},       0),
    ({'fi': 'Maltillinen (+25 v)',   'en': 'Moderate (+25 yr)'},      25),
    ({'fi': 'Hidas (+75 v)',         'en': 'Slow (+75 yr)'},          75),
    ({'fi': 'Ei sitouduta (+150 v)', 'en': 'No commitment (+150 yr)'}, 150),  # net-zero year well past 2100; emissions stay high throughout simulation
]

# (temp °C, color, label shown right of y-axis); '\n' splits a label into two lines
TEMP_ANNOTATIONS = [
    (1.5, '#65C06C', {'fi': '70–90 % koralleista häviää',
                      'en': '70–90% of corals lost'}),
    (2.0, '#FFA73E', {'fi': 'korallit katoavat\nItämeri ei enää jäädy',
                      'en': 'corals disappear\nBaltic Sea no longer freezes'}),
    (3.0, '#E2824A', {'fi': 'Jäämeri jäätön useimpina kesinä\nLapin porotalous uhattuna',
                      'en': 'Arctic Ocean ice-free most summers\nLapland reindeer herding threatened'}),
    (4.0, '#E24444', {'fi': 'helleaallot 9,4×\netelä-Suomi käytännössä lumeton',
                      'en': 'heatwaves 9.4×\nsouthern Finland practically snowless'}),
]

REGION_COLORS = {
    'Kiina 🇨🇳':      '#E26D91',
    'USA 🇺🇸':         '#5A84E2',
    'EU 🇪🇺':          '#65C06C',
    'Intia 🇮🇳':       '#FFA73E',
    'Muu maailma 🌍': '#8760C1',
}

# Impact data at four warming levels (vs. 1850–1900 pre-industrial baseline).
# Each entry: (icon, label key in IMPACT_LABELS, value text, severity_0_to_100)
# Global figures from IPCC AR6 WG1 (2021) and SR1.5 (2018).
# Finland figures from Finnish Meteorological Institute projections
# (Finland warms ~1.5–2× the global average due to Arctic amplification).
IMPACT_LABELS = {
    'sea_level':  {'fi': 'Merenpinnannousu (2100)',        'en': 'Sea level rise (2100)'},
    'corals':     {'fi': 'Koralliriutat',                  'en': 'Coral reefs'},
    'arctic':     {'fi': 'Arktis: jäätön kesä',            'en': 'Arctic: ice-free summer'},
    'habitat':    {'fi': 'Lajien elinympäristö puolittuu', 'en': 'Species losing half their habitat'},
    'heatwaves':  {'fi': 'Äärimmäiset helleaallot',        'en': 'Extreme heatwaves'},
    'fi_warming': {'fi': 'Lämpötilan nousu Suomessa',      'en': 'Warming in Finland'},
    'helsinki':   {'fi': 'Helsinki talvella',              'en': 'Helsinki in winter'},
    'snow':       {'fi': 'Lumiset talvet Etelä-Suomessa',  'en': 'Snowy winters in southern Finland'},
    'ski':        {'fi': 'Hiihtosesonki etelässä',         'en': 'Ski season in the south'},
    'reindeer':   {'fi': 'Porotalous Lapissa',             'en': 'Reindeer herding in Lapland'},
}
IMPACTS = {
    1.5: {
        'color': '#65C06C',
        'global': [
            ('🌊', 'sea_level', {'fi': '~40 cm', 'en': '~40 cm'}, 25),
            ('🪸', 'corals',    {'fi': '70–90 % häviää', 'en': '70–90% lost'}, 35),
            ('🧊', 'arctic',    {'fi': 'kerran sadassa vuodessa', 'en': 'once in a hundred years'}, 15),
            ('🦋', 'habitat',   {'fi': '~6 % hyönteisistä', 'en': '~6% of insects'}, 20),
            ('☀️', 'heatwaves', {'fi': '4× yleisempiä kuin 1800-luvulla',
                                 'en': '4× more frequent than in the 1800s'}, 20),
        ],
        'finland': [
            ('🌡️', 'fi_warming', {'fi': '+2–3 °C', 'en': '+2–3 °C'}, 25),
            ('🏙️', 'helsinki',   {'fi': 'kuin Tukholma tänään', 'en': 'like Stockholm today'}, 25),
            ('❄️', 'snow',       {'fi': 'harvinaisempia', 'en': 'less common'}, 25),
            ('⛷️', 'ski',        {'fi': 'lyhenee 2–3 viikkoa', 'en': 'shorter by 2–3 weeks'}, 20),
            ('🦌', 'reindeer',   {'fi': 'jäätymisriski kasvaa', 'en': 'growing risk of icy pastures'}, 20),
        ],
    },
    2.0: {
        'color': '#FFA73E',
        'global': [
            ('🌊', 'sea_level', {'fi': '~50 cm', 'en': '~50 cm'}, 45),
            ('🪸', 'corals',    {'fi': '>99 % häviää — käytännössä kaikki',
                                 'en': '>99% lost — practically all'}, 75),
            ('🧊', 'arctic',    {'fi': 'kerran kymmenessä vuodessa', 'en': 'once in ten years'}, 40),
            ('🦋', 'habitat',   {'fi': '~18 % hyönteisistä', 'en': '~18% of insects'}, 45),
            ('☀️', 'heatwaves', {'fi': '5,6× yleisempiä', 'en': '5.6× more frequent'}, 45),
        ],
        'finland': [
            ('🌡️', 'fi_warming', {'fi': '+3–4 °C', 'en': '+3–4 °C'}, 45),
            ('🏙️', 'helsinki',   {'fi': 'kuin Varsova tänään', 'en': 'like Warsaw today'}, 45),
            ('❄️', 'snow',       {'fi': 'epäluotettavia', 'en': 'unreliable'}, 50),
            ('⛷️', 'ski',        {'fi': 'lyhenee 1–2 kuukautta', 'en': 'shorter by 1–2 months'}, 50),
            ('🦌', 'reindeer',   {'fi': 'merkittävä uhka', 'en': 'a significant threat'}, 45),
        ],
    },
    3.0: {
        'color': '#E2824A',
        'global': [
            ('🌊', 'sea_level', {'fi': '~80 cm', 'en': '~80 cm'}, 65),
            ('🪸', 'corals',    {'fi': 'käytännössä hävinneet', 'en': 'practically gone'}, 90),
            ('🧊', 'arctic',    {'fi': 'useimpina vuosina', 'en': 'in most years'}, 70),
            ('🦋', 'habitat',   {'fi': '~49 % hyönteisistä', 'en': '~49% of insects'}, 70),
            ('☀️', 'heatwaves', {'fi': '~8× yleisempiä', 'en': '~8× more frequent'}, 70),
        ],
        'finland': [
            ('🌡️', 'fi_warming', {'fi': '+4,5–6 °C', 'en': '+4.5–6 °C'}, 65),
            ('🏙️', 'helsinki',   {'fi': 'kuin Praha tänään', 'en': 'like Prague today'}, 65),
            ('❄️', 'snow',       {'fi': 'hyvin harvinaisia', 'en': 'very rare'}, 75),
            ('⛷️', 'ski',        {'fi': 'siirtynyt pohjoiseen', 'en': 'moved north'}, 75),
            ('🦌', 'reindeer',   {'fi': 'vakavasti uhattuna', 'en': 'seriously threatened'}, 75),
        ],
    },
    4.0: {
        'color': '#E24444',
        'global': [
            ('🌊', 'sea_level', {'fi': '~100 cm+', 'en': '~100 cm+'}, 85),
            ('🪸', 'corals',    {'fi': 'hävinneet', 'en': 'gone'}, 100),
            ('🧊', 'arctic',    {'fi': 'joka vuosi', 'en': 'every year'}, 95),
            ('🦋', 'habitat',   {'fi': '>35 % lajeista', 'en': '>35% of species'}, 85),
            ('☀️', 'heatwaves', {'fi': '9,4× yleisempiä', 'en': '9.4× more frequent'}, 90),
        ],
        'finland': [
            ('🌡️', 'fi_warming', {'fi': '+6–8 °C', 'en': '+6–8 °C'}, 85),
            ('🏙️', 'helsinki',   {'fi': 'kuin Wien tänään', 'en': 'like Vienna today'}, 85),
            ('❄️', 'snow',       {'fi': 'kadonnut muisto', 'en': 'a thing of the past'}, 95),
            ('⛷️', 'ski',        {'fi': 'ei enää taloudellisesti kannattavaa',
                                  'en': 'no longer economically viable'}, 90),
            ('🦌', 'reindeer',   {'fi': 'romahtamassa', 'en': 'collapsing'}, 90),
        ],
    },
}


def _num(value, decimals):
    """Number in the current language: decimal comma in Finnish, e.g. _num(1.234, 2) -> '1,23'."""
    text = f'{value:.{decimals}f}'
    if lang() == 'fi':
        text = text.replace('.', ',')
    return text


def _level(level):
    """A warming level without a trailing zero: 1.5 -> '1,5' / '1.5', 2.0 -> '2'."""
    text = f'{level:g}'
    if lang() == 'fi':
        text = text.replace('.', ',')
    return text


def _pct(value):
    """A percentage rounded to an integer, '15 %' in Finnish and '15%' in English."""
    return t('pct', value=f'{value:.0f}')


# Matplotlib tick labels in the current language (rounded first, so 0.30000000000000004 -> '0,3').
# The language is read when the figure is drawn, i.e. in the same script run that creates it.
NUM_TICKS = mticker.FuncFormatter(lambda value, pos: _level(round(value, 6)))


def _region_name(region: str) -> str:
    """Displayed name of a region (internal ID = Finnish name), with its flag emoji."""
    return tr(REGION_NAMES[region])


def _plain_name(region: str) -> str:
    """Displayed name of a region without the trailing flag emoji."""
    return _region_name(region).rsplit(' ', 1)[0]


def _option_label(region: str, index: int) -> str:
    """Radio label of a Phase 1 option: level name and the 2025->2040 change, e.g. 'Vahva (−40 %)'."""
    level, cut = EMISSION_OPTIONS[region][index]
    change = round(-cut * 100)
    if change > 0:
        signed = f'+{change}'
    elif change < 0:
        signed = f'−{-change}'   # typographic minus
    else:
        signed = '0'
    return f'{tr(OPTION_LEVELS[level])} ({t("pct", value=signed)})'


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
    'pledges':  {'fi': 'Lupaukset',       'en': 'Pledges'},
    'prior':    {'fi': 'Alkuarvio',       'en': 'Initial estimate'},
    'draw':     {'fi': 'Arvonta',         'en': 'Draw'},
    'sim1':     {'fi': 'Simulaatio 2040', 'en': 'Simulation 2040'},
    'obs2040':  {'fi': 'Havainnot 2040',  'en': 'Observations 2040'},
    'research': {'fi': 'Tutkimus etenee', 'en': 'Research advances'},
    'phase2':   {'fi': 'Nollapäästöt',    'en': 'Net-zero emissions'},
    'reveal':   {'fi': 'Paljastus',       'en': 'Reveal'},
}
# Estimate stages: (label, colour). A stage is shown from its own screen onwards.
STAGE_STYLE = {
    'prior':    ({'fi': 'Alkuarvio', 'en': 'Initial estimate'}, '#8A9099'),
    'obs2040':  ({'fi': '2025–2040 havaintojen jälkeen', 'en': 'After observations 2025–2040'}, '#5A84E2'),
    'research': ({'fi': 'Tutkimuksen jälkeen', 'en': 'After research'}, '#E26D91'),
}


def _stage_label(stage):
    return tr(STAGE_STYLE[stage][0])
# Light chart style (the app theme is pinned to light in .streamlit/config.toml)
CHART_BG     = 'white'
CHART_FG     = '#262730'   # Streamlit's light-theme text colour
CHART_SPINE  = '#C8CCD2'
# The level colours are fine for lines and bars but too pale for text on white:
# text uses a darker shade of the same hue.
TEXT_SHADE = {
    '#65C06C': '#2E7D32',
    '#FFA73E': '#B85C00',
    '#E2824A': '#A84A12',
    '#E24444': '#B71C1C',
}


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
        'animations_played': set(),  # chart keys whose animation has been shown once
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
    # projected per member. Its spread across the members weighted by both updates (all that is
    # known at the end of 2040) is the uncertainty that comes from not yet knowing 2041-2044.
    known_sum = float(temp.sel(year=slice(CENTRED_WINDOW[0], OBS_WINDOW[1])).sum())
    projected_sum = sum(emulate_temperature(cuts, None, year)
                        for year in range(OBS_WINDOW[1] + 1, CENTRED_WINDOW[1] + 1))
    n_years = CENTRED_WINDOW[1] - CENTRED_WINDOW[0] + 1
    decade_members = (known_sum + projected_sum) / n_years
    st.session_state['centred_decade'] = weighted_mean_interval(decade_members, weights_research)

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
        (ecs, t('est_ecs_title'), 'ECS (°C)', None if truth is None else truth[0]),
        (temperature, t('est_temp_title', year=ESTIMATE_YEAR), t('axis_temp'),
         None if truth is None else truth[1]),
    ]
    n_stages = len(stages)
    for col, (values, title, xlabel, true_value) in enumerate(panels):
        ax_hist, ax_bar = axes[0, col], axes[1, col]
        for ax in (ax_hist, ax_bar):
            ax.set_facecolor(CHART_BG)
            ax.tick_params(colors=CHART_FG)
            for spine in ax.spines.values():
                spine.set_edgecolor(CHART_SPINE)
        bins = np.linspace(values.min(), values.max(), 36)
        bar_transform = blended_transform_factory(ax_bar.transAxes, ax_bar.transData)
        for row, (stage, weights) in enumerate(stages.items()):
            label, color = _stage_label(stage), STAGE_STYLE[stage][1]
            newest = row == n_stages - 1
            ax_hist.hist(values.loc[weights.index].values, bins=bins, weights=weights.values,
                         density=True, histtype='stepfilled', color=color,
                         alpha=0.85 if newest else 0.30, label=label)
            mean, lo, hi = weighted_mean_interval(values, weights)
            ax_bar.plot([lo, hi], [row, row], color=color, linewidth=7, solid_capstyle='butt',
                        alpha=1.0 if newest else 0.6)
            ax_bar.plot([mean], [row], 'o', color='white', markeredgecolor=CHART_FG, markersize=6)
            ax_bar.text(1.01, row, f'{_num(mean, 1)} ({_num(lo, 1)}–{_num(hi, 1)})', transform=bar_transform,
                        color=color, fontsize=9, va='center', clip_on=False)
        if col == 1:
            ax_hist.axvline(PARIS_15, color='#65C06C', linewidth=1.2, linestyle='--')
            ax_hist.axvline(PARIS_20, color='#FFA73E', linewidth=1.2, linestyle='--')
        if true_value is not None:
            for ax in (ax_hist, ax_bar):
                ax.axvline(true_value, color=CHART_FG, linewidth=2.0, linestyle=':')
            ax_hist.text(true_value, ax_hist.get_ylim()[1] * 0.95, t('est_true_value', value=_num(true_value, 2)),
                         color=CHART_FG, fontsize=9, va='top')
        ax_hist.set_title(title, color=CHART_FG, fontsize=11)
        ax_hist.set_yticks([])
        ax_bar.set_yticks([])
        ax_bar.set_ylim(n_stages - 0.4, -0.6)    # oldest stage on top
        ax_bar.set_xlabel(xlabel, color=CHART_FG)
        ax_bar.xaxis.set_major_formatter(NUM_TICKS)
    # Legend above the panels, so it never hides the histograms or the true-value label
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
               labelcolor=CHART_FG, fontsize=9, bbox_to_anchor=(0.45, 1.0))
    fig.subplots_adjust(left=0.03, right=0.88, wspace=0.32, bottom=0.14, top=0.86)
    return fig


def _show_estimates(cuts, netzero_years, upto_screen, truth=None):
    """Histograms of the estimate stages and the exceedance probabilities of the newest stage."""
    temperature = emulate_temperature(cuts, netzero_years, ESTIMATE_YEAR)
    stages = _stage_weights(upto_screen)
    st.pyplot(_estimate_figure(temperature, stages, truth))
    st.caption(t('est_caption'))

    newest = list(stages)[-1]
    probabilities = _exceedance(temperature, stages[newest])
    prior_probabilities = _exceedance(temperature, stages['prior'])
    st.markdown(t('exceed_heading', year=ESTIMATE_YEAR, stage=_stage_label(newest).lower()))
    cols = st.columns(len(EXCEEDANCE_LEVELS))
    for col, level in zip(cols, EXCEEDANCE_LEVELS):
        delta = None
        change = (probabilities[level] - prior_probabilities[level]) * 100
        if newest != 'prior' and abs(change) >= 0.5:   # no "-0" arrows for rounding noise
            delta = t('exceed_delta', change=f'{change:+.0f}')
        col.metric(t('exceed_label', level=_level(level)), _pct(probabilities[level] * 100),
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
            style = 'background:#E2824A;box-shadow:0 0 0 3px #FFD9BF,0 0 12px 4px #E2824A;'
        else:
            style = 'background:#C3CBD6;'
        cells.append(f'<div style="width:12px;height:12px;border-radius:50%;{style}"></div>')
    return ('<div style="display:flex;flex-wrap:wrap;gap:4px;max-width:960px;'
            'padding:12px 0">' + ''.join(cells) + '</div>')


def _impacts_page():
    st.title(t('imp_title'))
    st.markdown(t('imp_intro'))
    if st.button(t('imp_back')):
        st.session_state['show_impacts'] = False
        st.rerun()
    st.markdown('---')

    selected = st.radio(
        t('imp_radio'),
        options=[1.5, 2.0, 3.0, 4.0],
        format_func=lambda x: f'+{_num(x, 1)} °C',
        horizontal=True,
        key='impacts_temp',
        label_visibility='collapsed',
    )

    data   = IMPACTS[selected]
    color  = data['color']

    # Large temperature badge
    st.markdown(
        f'<div style="text-align:center;padding:16px 0 24px 0">'
        f'<span style="font-size:3.5em;font-weight:900;color:{TEXT_SHADE.get(color, color)}">'
        f'+{_num(selected, 1)} °C</span></div>',
        unsafe_allow_html=True,
    )

    def _bar(pct):
        return (
            f'<div style="background:#E6E8EC;border-radius:4px;height:8px;margin:4px 0 14px 0">'
            f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>'
            f'</div>'
        )

    col_global, col_finland = st.columns(2)

    with col_global:
        st.subheader(t('imp_global'))
        for icon, label_key, value, sev in data['global']:
            st.markdown(f'**{icon} {tr(IMPACT_LABELS[label_key])}**')
            st.markdown(
                f'<span style="color:{TEXT_SHADE.get(color, color)};font-size:1.05em">{tr(value)}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(_bar(sev), unsafe_allow_html=True)

    with col_finland:
        st.subheader(t('imp_finland'))
        for icon, label_key, value, sev in data['finland']:
            st.markdown(f'**{icon} {tr(IMPACT_LABELS[label_key])}**')
            st.markdown(
                f'<span style="color:{TEXT_SHADE.get(color, color)};font-size:1.05em">{tr(value)}</span>',
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
    decisions = t('imp_decisions_both') if netzero_years else t('imp_decisions_pledges')
    message = t('imp_message', decisions=decisions, level=_num(selected, 1), year=ESTIMATE_YEAR,
                prob=_pct(probability * 100), stage=_stage_label(newest).lower())
    if st.session_state['centred_decade'] is not None:
        decade_mean = st.session_state['centred_decade'][0]
        message += t('imp_decade', first=CENTRED_WINDOW[0], last=CENTRED_WINDOW[1],
                     value=_num(decade_mean, 2))
    st.info(message)

    st.markdown(
        f'<p style="color:#4A5568;font-size:0.8em;margin-top:24px">{t("imp_sources")}</p>',
        unsafe_allow_html=True,
    )


def _nav_bar():
    """Step indicator, back/forward between reached screens, and the impacts page."""
    screen = st.session_state['screen']
    position = SCREENS.index(screen)
    furthest = st.session_state['furthest']
    steps = []
    for i, name in enumerate(SCREENS):
        title = f'{i + 1} {tr(SCREEN_TITLES[name])}'
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
        if position > 0 and st.button(t('nav_back')):
            st.session_state['screen'] = SCREENS[position - 1]
            st.rerun()
    with col_forward:
        if position < furthest and st.button(t('nav_forward')):
            st.session_state['screen'] = SCREENS[position + 1]
            st.rerun()
    with col_impacts:
        if st.button(t('nav_impacts')):
            st.session_state['show_impacts'] = True
            st.rerun()
    st.markdown('---')


def _temperature_chart(temp_da, year_start, year_end, upto_year, final,
                       phase1_da=None, highlight_year=None, show_peak=False, decade=None):
    """
    Altair chart of the temperature trajectory: observations (up to 2025) and the simulated
    annual means from ANIM_START up to upto_year (the animation reveals the line year by year).
    Both are annual means plotted at mid-year (Y + 0.5), as the IGCC file stamps them.
    The axes are fixed from the full series, so they do not jump between animation frames.
    Colours are chosen to read on both the light and the dark Streamlit theme.

    temp_da        : xr.DataArray dim 'year' (annual means, see annual_series), single config
    phase1_da      : if given, the Phase 1 trajectory as a grey dashed line
    highlight_year : dotted vertical line (Phase 1: CHECKPOINT)
    final          : add the summary marks: the step from the last observation to the first
                     model year, the peak (show_peak) and the decade bar (decade)
    decade         : (mean, lo, hi) of the CENTRED_WINDOW mean: solid over the known years,
                     dashed and shaded over the projected ones. Extends the x axis.
    """
    gmst = _load_gmst()
    obs = gmst[(gmst['year'] >= year_start) & (gmst['year'] <= ANIM_START)]
    years = temp_da.year.values
    future = (years >= ANIM_START) & (years <= year_end)
    yr_fut = years[future]
    t_fut = temp_da.values[future]

    x_end = year_end + 1
    y_min = min(float(obs['gmst'].min()), 0.0) - 0.15
    y_max = max(float(t_fut.max()), float(obs['gmst'].max())) + 0.45
    if decade is not None:
        x_end = max(x_end, CENTRED_WINDOW[1] + 2)
        y_max = max(y_max, decade[2] + 0.3)
    x_scale = alt.Scale(domain=[year_start, x_end], nice=False, zero=False)
    y_scale = alt.Scale(domain=[y_min, y_max], nice=False, zero=False)
    x_enc = alt.X('x:Q', scale=x_scale, axis=alt.Axis(title=t('axis_year'), format='d'))
    # Vega has no Finnish locale here, so the decimal comma is put in with a label expression
    label_expr = "format(datum.value, '.1f')"
    if lang() == 'fi':
        label_expr = f"replace({label_expr}, '.', ',')"
    y_enc = alt.Y('T:Q', scale=y_scale, axis=alt.Axis(title=t('axis_temp'), labelExpr=label_expr))

    # --- lines: observations, (Phase 1), model ---
    shown = yr_fut <= upto_year
    model = pd.DataFrame({'x': yr_fut[shown] + 0.5, 'T': t_fut[shown]})
    if final:
        # The step from the last observed year to the first model year is drawn only on the
        # final frame, after the rest of the line (plot only, the results are not modified).
        first = pd.DataFrame({'x': [float(obs['year'].iloc[-1])], 'T': [float(obs['gmst'].iloc[-1])]})
        model = pd.concat([first, model], ignore_index=True)
    # Series names are the legend entries, so they are in the current language
    model['sarja'] = t('series_model')
    observed = pd.DataFrame({'x': obs['year'].values, 'T': obs['gmst'].values, 'sarja': t('series_obs')})
    parts = [observed, model]
    series, colors, dashes = [t('series_obs'), t('series_model')], [OBS_COLOR, MODEL_COLOR], [[1, 0], [1, 0]]
    if phase1_da is not None:
        years1 = phase1_da.year.values
        keep1 = (years1 >= ANIM_START) & (years1 <= CHECKPOINT)
        parts.append(pd.DataFrame({'x': years1[keep1] + 0.5, 'T': phase1_da.values[keep1],
                                   'sarja': t('series_phase1')}))
        series.append(t('series_phase1'))
        colors.append('#9AA0A6')
        dashes.append([6, 4])
    lines = pd.concat(parts, ignore_index=True)
    lines['vuosi'] = np.floor(lines['x']).astype(int)    # calendar year for the tooltip
    lines['lämpötila'] = [_num(value, 2) + ' °C' for value in lines['T']]
    layers = [
        alt.Chart(lines).mark_line(strokeWidth=2.2, clip=True).encode(
            x=x_enc, y=y_enc, detail='sarja:N', order='x:Q',
            color=alt.Color('sarja:N', scale=alt.Scale(domain=series, range=colors),
                            legend=alt.Legend(title=None, orient='bottom-right')),
            strokeDash=alt.StrokeDash('sarja:N', scale=alt.Scale(domain=series, range=dashes),
                                      legend=None),
            tooltip=[alt.Tooltip('sarja:N', title=t('tt_series')),
                     alt.Tooltip('vuosi:Q', title=t('axis_year'), format='d'),
                     alt.Tooltip('lämpötila:N', title=t('tt_temp'))],
        )
    ]

    # --- warming levels: Paris lines labelled on the left, impacts on the right ---
    levels = pd.DataFrame([{'T': level, 'väri': color, 'tekstiväri': TEXT_SHADE.get(color, color),
                            'vaikutus': tr(label)}
                           for level, color, label in TEMP_ANNOTATIONS if level <= y_max])
    paris = pd.DataFrame([{'T': PARIS_15, 'tekstiväri': TEXT_SHADE['#65C06C'],
                           'nimi': t('paris_15')},
                          {'T': PARIS_20, 'tekstiväri': TEXT_SHADE['#FFA73E'],
                           'nimi': t('paris_20')}])
    paris = paris[paris['T'] <= y_max]    # text marks are not clipped: drop labels above the plot
    layers += [
        alt.Chart(levels).mark_rule(strokeDash=[5, 4], strokeWidth=1.3).encode(
            y=alt.Y('T:Q', scale=y_scale), color=alt.Color('väri:N', scale=None)),
        # Impact labels in the right margin (outside the plot, see the padding below)
        alt.Chart(levels).mark_text(align='left', baseline='middle', dx=8, fontSize=11,
                                    lineBreak='\n').encode(
            x=alt.datum(x_end), y=alt.Y('T:Q', scale=y_scale),
            text='vaikutus:N', color=alt.Color('tekstiväri:N', scale=None)),
        alt.Chart(paris).mark_text(align='left', baseline='bottom', dx=4, dy=-3, fontSize=11,
                                   fontWeight='bold').encode(
            x=alt.datum(year_start), y=alt.Y('T:Q', scale=y_scale),
            text='nimi:N', color=alt.Color('tekstiväri:N', scale=None)),
    ]

    if highlight_year is not None:
        mark = pd.DataFrame({'x': [highlight_year + 0.5], 'nimi': [str(highlight_year)]})
        layers += [
            alt.Chart(mark).mark_rule(strokeDash=[2, 3], color=OBS_COLOR).encode(x=x_enc),
            alt.Chart(mark).mark_text(align='left', baseline='bottom', dx=4, color=OBS_COLOR).encode(
                x=x_enc, y=alt.datum(y_min), text='nimi:N'),
        ]

    if final and show_peak:
        after = temp_da.sel(year=temp_da.year > CHECKPOINT)
        peak_temp = float(after.max())
        peak_year = int(after.year.values[after.values.argmax()])
        peak = pd.DataFrame({'x': [peak_year + 0.5], 'T': [peak_temp],
                             'nimi': [t('peak_label', value=_num(peak_temp, 2), year=peak_year)]})
        layers += [
            alt.Chart(peak).mark_rule(strokeDash=[2, 3], color=OBS_COLOR).encode(x=x_enc),
            alt.Chart(peak).mark_point(size=90, filled=True, color=MODEL_COLOR).encode(x=x_enc, y=y_enc),
            alt.Chart(peak).mark_text(align='left', dx=8, dy=-10, fontSize=13, fontWeight='bold',
                                      color=MODEL_COLOR).encode(x=x_enc, y=y_enc, text='nimi:N'),
        ]

    if final and decade is not None:
        # The decade covers annual years CENTRED_WINDOW, i.e. x from its first year to one past
        # its last; the years after OBS_WINDOW[1] are not known yet in 2040 (projected).
        mean, lo, hi = decade
        x0, x_known, x1 = CENTRED_WINDOW[0], OBS_WINDOW[1] + 1, CENTRED_WINDOW[1] + 1
        band = pd.DataFrame({'x': [x_known], 'x2': [x1], 'T': [lo], 'T2': [hi]})
        known = pd.DataFrame({'x': [x0], 'x2': [x_known], 'T': [mean]})
        projected = pd.DataFrame({'x': [x_known], 'x2': [x1], 'T': [mean]})
        layers += [
            alt.Chart(band).mark_rect(opacity=0.25, color=DECADE_COLOR).encode(
                x=x_enc, x2='x2:Q', y=y_enc, y2='T2:Q'),
            alt.Chart(known).mark_rule(strokeWidth=4, color=DECADE_COLOR).encode(
                x=x_enc, x2='x2:Q', y=y_enc),
            alt.Chart(projected).mark_rule(strokeWidth=4, strokeDash=[5, 3], color=DECADE_COLOR).encode(
                x=x_enc, x2='x2:Q', y=y_enc),
        ]

    # Right padding holds the impact labels (the decade value is shown as a metric below)
    return (alt.layer(*layers).resolve_scale(color='independent')
            .properties(height=440, padding={'left': 5, 'top': 10, 'bottom': 5, 'right': 210}))


def _show_temperature_animation(key, **chart_kwargs):
    """
    Show the temperature chart. The first time `key` is shown the model line is drawn year by
    year (about ANIM_SECONDS; years are grouped beyond ANIM_MAX_FRAMES frames), afterwards the
    final chart is shown directly. A button replays the animation.
    """
    placeholder = st.empty()
    played = st.session_state['animations_played']
    temp_da, year_end = chart_kwargs['temp_da'], chart_kwargs['year_end']
    if key not in played:
        years = [int(y) for y in temp_da.year.values if ANIM_START <= y <= year_end]
        step = max(1, int(np.ceil(len(years) / ANIM_MAX_FRAMES)))
        frames = years[::step]
        for upto_year in frames:
            placeholder.altair_chart(_temperature_chart(upto_year=upto_year, final=False, **chart_kwargs),
                                     width='stretch')
            time.sleep(ANIM_SECONDS / len(frames))
        played.add(key)
    placeholder.altair_chart(_temperature_chart(upto_year=year_end, final=True, **chart_kwargs),
                             width='stretch')
    if st.button(t('replay'), key=f'replay_{key}'):
        played.discard(key)
        st.rerun()


def _emissions_bar_chart(cuts):
    """Bar chart: BASE_YEAR baseline vs. pledged CHECKPOINT emissions per region."""
    regions = list(REGION_EMISSIONS_BASE_GTCO2.keys())
    base    = [REGION_EMISSIONS_BASE_GTCO2[r] for r in regions]
    target  = [REGION_EMISSIONS_BASE_GTCO2[r] * (1 - cuts[r]) for r in regions]
    em_2100 = regional_emissions_in_year(cuts, ESTIMATE_YEAR)
    later   = [em_2100[r] for r in regions]
    colors  = [REGION_COLORS[r] for r in regions]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    x     = np.arange(len(regions))
    width = 0.27
    ax.bar(x - width, base,   width, color=colors, alpha=0.30, label=t('bar_base', year=BASE_YEAR))
    ax.bar(x,         target, width, color=colors, alpha=0.90, label=t('bar_pledge', year=CHECKPOINT))
    ax.bar(x + width, later,  width, color=colors, alpha=0.60, hatch='//',
           label=t('bar_later', year=ESTIMATE_YEAR))

    ax.set_xticks(x)
    ax.set_xticklabels([_plain_name(r) for r in regions], color=CHART_FG, fontsize=9)
    ax.set_ylabel(t('unit_per_year'), color=CHART_FG)
    ax.yaxis.set_major_formatter(NUM_TICKS)
    ax.tick_params(colors=CHART_FG)
    ax.legend(facecolor=CHART_BG, labelcolor=CHART_FG, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_SPINE)
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
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    stack_bottom = np.zeros(len(years))
    for r in regions:
        ax.fill_between(years, stack_bottom, stack_bottom + region_em[r],
                        color=REGION_COLORS[r], alpha=0.75, label=_plain_name(r))
        stack_bottom += region_em[r]

    ax.axvline(CHECKPOINT, color=CHART_FG, linewidth=1, linestyle=':')
    ax.set_ylabel(t('unit_per_year'), color=CHART_FG)
    ax.yaxis.set_major_formatter(NUM_TICKS)
    ax.set_xlabel(t('axis_year'), color=CHART_FG)
    ax.set_xlim(BASE_YEAR, plot_end)
    ax.tick_params(colors=CHART_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_SPINE)
    ax.legend(facecolor=CHART_BG, labelcolor=CHART_FG, fontsize=8,
              loc='upper right', ncol=2)
    ax.set_title(t('pathway_title'), color=CHART_FG, fontsize=10)
    fig.tight_layout()
    return fig


def _cut_input(region: str, option_indices: dict, disabled: bool = False) -> float:
    """
    Return the chosen cut fraction for a region.
    In slider mode (--sliders flag): free slider of the emissions change 2025→2040 in %
    (negative = reduction), the same sign as the radio labels; returns the cut fraction.
    In radio mode (default): 4-option radio from EMISSION_OPTIONS.
    Updates option_indices[region] in place. disabled locks the input (after the draw).
    """
    if USE_SLIDERS:
        # Initialise the slider from whichever radio option is currently selected.
        # Range: from net zero by 2040 (-100 %) to the largest growth among the options
        # (India Passiivinen, +80 %), so every radio option can be reached.
        stored_idx = option_indices.get(region, 0)
        change_pct = int(round(-EMISSION_OPTIONS[region][stored_idx][1] * 100))
        change_pct = st.slider(
            _region_name(region), min_value=-100, max_value=80,
            value=change_pct, step=5, format='%+d %%' if lang() == 'fi' else '%+d%%',
            key=f'slider_p1_{region}', disabled=disabled,
        )
        return -change_pct / 100.0
    options = EMISSION_OPTIONS[region]
    idx = st.radio(
        _region_name(region),
        options=range(len(options)),
        format_func=lambda i: _option_label(region, i),
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
            _region_name(region), min_value=min_nz, max_value=max_nz,
            value=max(default_yr, min_nz), step=5,
            key=f'slider_nz_{region}',
        )
    nz_opts = [(lbl, min_nz + off) for lbl, off in NETZERO_OFFSETS]
    idx = st.radio(
        _region_name(region),
        options=range(len(nz_opts)),
        format_func=lambda i, o=nz_opts: f'{tr(o[i][0])} ({o[i][1]})',
        index=netzero_indices.get(region, 1),
        key=f'radio_nz_{region}',
    )
    netzero_indices[region] = idx
    return nz_opts[idx][1]


def _screen_pledges():
    st.title(t('pl_title'))
    st.markdown(t('pl_intro', year=CHECKPOINT))
    st.caption(t('pl_caption'))

    _nav_bar()

    # Once a member is drawn the 2040 observation depends on these pledges, so they are locked.
    locked = st.session_state['config_id'] is not None
    if locked:
        st.warning(t('pl_locked'))

    cuts          = st.session_state['cuts']
    option_indices = st.session_state['option_indices']

    st.subheader(t('pl_subheader'))
    if USE_SLIDERS:
        st.caption(t('pl_slider_caption', base_year=BASE_YEAR, year=CHECKPOINT))
    cols = st.columns(len(REGION_EMISSIONS_BASE_GTCO2))
    for col, (region, base_em) in zip(cols, REGION_EMISSIONS_BASE_GTCO2.items()):
        with col:
            cut    = _cut_input(region, option_indices, disabled=locked)
            cuts[region] = cut
            em_chk = base_em * (1.0 - cut)
            st.caption(t('pl_region_caption', base_year=BASE_YEAR, base=_num(base_em, 1),
                         year=CHECKPOINT, pledged=_num(em_chk, 1)))
    st.session_state['cuts']          = cuts
    st.session_state['option_indices'] = option_indices

    total_chk     = sum(REGION_EMISSIONS_BASE_GTCO2[r] * (1 - cuts[r]) for r in cuts)
    total_cut_pct = (1 - total_chk / TOTAL_BASE) * 100
    change_str = (t('pl_cut', pct=f'{total_cut_pct:.0f}') if total_cut_pct >= 0
                  else t('pl_growth', pct=f'{-total_cut_pct:.0f}'))
    st.info(t('pl_total', year=CHECKPOINT, total=_num(total_chk, 1), change=change_str,
              base_year=BASE_YEAR))
    st.pyplot(_emissions_bar_chart(cuts))

    # Prior estimate (whole screened ensemble, equal weights): mean and 95 % interval.
    est_40  = emulate_temperature(cuts, None, CHECKPOINT)
    est_end = emulate_temperature(cuts, None, ESTIMATE_YEAR)
    st.info(t(
        'pl_prior',
        year1=CHECKPOINT, mean1=_num(est_40.mean(), 1),
        lo1=_num(est_40.quantile(0.025), 1), hi1=_num(est_40.quantile(0.975), 1),
        year2=ESTIMATE_YEAR, mean2=_num(est_end.mean(), 1),
        lo2=_num(est_end.quantile(0.025), 1), hi2=_num(est_end.quantile(0.975), 1),
    ))
    st.caption(t('pl_assumption', year=CHECKPOINT, base_year=BASE_YEAR))

    if st.button(t('pl_next'), type='primary'):
        _goto('prior')


def _screen_prior():
    st.title(t('pr_title'))
    _nav_bar()
    st.markdown(t('pr_intro', n=len(_load_screened())))
    st.markdown(t('pr_ecs'))
    _show_estimates(st.session_state['cuts'], None, 'prior')
    st.info(t('pr_info'))
    if st.button(t('pr_next'), type='primary'):
        _goto('draw')


def _screen_draw():
    st.title(t('dr_title'))
    _nav_bar()
    screened = _load_screened()
    n_members = len(screened)
    st.markdown(t('dr_intro', n=n_members))

    placeholder = st.empty()
    if st.session_state['config_id'] is None:
        placeholder.markdown(_lottery_html(n_members, None), unsafe_allow_html=True)
        if st.button(t('dr_button'), type='primary'):
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
    st.success(t('dr_success', number=st.session_state['grid_position'] + 1))
    if st.session_state['temp_phase1'] is None:
        if st.button(t('dr_simulate', start=ANIM_START, end=CHECKPOINT), type='primary'):
            with st.spinner(t('spinner_fair')):
                _simulate_phase1(st.session_state['cuts'])
            _goto('sim1')


def _screen_sim1():
    st.title(t('s1_title', year=CHECKPOINT))
    _nav_bar()
    temp = st.session_state['temp_phase1']
    t_single = float(temp.sel(year=CHECKPOINT))
    decade_mean, decade_lo, decade_hi = st.session_state['centred_decade']
    gmst = _load_gmst()
    obs_last = float(gmst.loc[np.floor(gmst['year']) == ANIM_START - 1, 'gmst'].iloc[0])

    _show_temperature_animation(
        f'sim1-{st.session_state["config_id"]}',
        # From 1960 (most of the warming, little flat early record) so that the 2035-2044
        # decade bar and the single year 2040 are large enough to compare
        temp_da=temp, year_start=1960, year_end=CHECKPOINT,
        highlight_year=CHECKPOINT, decade=st.session_state['centred_decade'],
    )
    col_obs, col_single, col_decade = st.columns(3)
    col_obs.metric(t('s1_obs', year=ANIM_START - 1), f'+{_num(obs_last, 2)} °C')
    col_single.metric(t('s1_year', year=CHECKPOINT), f'+{_num(t_single, 2)} °C')
    col_decade.metric(t('s1_decade', first=CENTRED_WINDOW[0], last=CENTRED_WINDOW[1]),
                      f'+{_num(decade_mean, 2)} °C',
                      help=t('s1_decade_help', lo=_num(decade_lo, 2), hi=_num(decade_hi, 2),
                             p0=OBS_WINDOW[1] + 1, p1=CENTRED_WINDOW[1]))

    # Paris thresholds refer to long-term warming, so the verdict uses the decade centred on 2040.
    window = dict(first=CENTRED_WINDOW[0], last=CENTRED_WINDOW[1])
    if decade_mean < PARIS_15:
        st.success(t('s1_below15', **window))
    elif decade_mean < PARIS_20:
        st.warning(t('s1_between', **window))
    else:
        st.error(t('s1_above20', **window))

    single_above = t_single >= PARIS_15
    decade_above = decade_mean >= PARIS_15
    st.markdown(t('s1_why', year=CHECKPOINT, p0=OBS_WINDOW[1] + 1, p1=CENTRED_WINDOW[1]))
    if single_above != decade_above:
        st.info(t('s1_split', year=CHECKPOINT, single=_num(t_single, 2), decade=_num(decade_mean, 2)))
    st.caption(t('s1_real'))

    if st.button(t('s1_next'), type='primary'):
        _goto('obs2040')


def _screen_obs2040():
    st.title(t('ob_title', year=CHECKPOINT))
    _nav_bar()
    observed = st.session_state['obs_window_mean']
    st.markdown(t('ob_intro', year=CHECKPOINT, first=OBS_WINDOW[0], last=OBS_WINDOW[1],
                  observed=_num(observed, 2), err=_num(1.96 * OBS_2040_SD, 1)))
    temperature, stages = _show_estimates(st.session_state['cuts'], None, 'obs2040')
    ecs = _load_screened()['ECS']
    ecs_old, ecs_new = _interval_width_change(ecs, stages, 'prior', 'obs2040')
    t_old, t_new = _interval_width_change(temperature, stages, 'prior', 'obs2040')
    ecs_narrowing = 1 - ecs_new / ecs_old
    message = t('ob_narrow', ecs_old=_num(ecs_old, 1), ecs_new=_num(ecs_new, 1),
                ecs_pct=f'{ecs_narrowing * 100:.0f}', year=ESTIMATE_YEAR,
                t_old=_num(t_old, 1), t_new=_num(t_new, 1), t_pct=f'{(1 - t_new / t_old) * 100:.0f}')
    # Typically (median over simulated games) the ECS interval narrows by only ~20 %; it narrows
    # much more when the drawn member is in a tail of the ensemble. Describe the actual case.
    if ecs_narrowing < 0.3:
        message += t('ob_small')
    else:
        prior_mean = _window_mean_members(st.session_state['cuts'], OBS_WINDOW).mean()
        direction = t('ob_less') if observed < prior_mean else t('ob_more')
        message += t('ob_large', direction=direction, prior_mean=_num(prior_mean, 2))
    st.info(message)
    with st.expander(t('ob_expander_title')):
        st.markdown(t('ob_expander_body'))
    if st.button(t('ob_next'), type='primary'):
        _goto('research')


def _screen_research():
    st.title(t('re_title'))
    _nav_bar()
    clue = st.session_state['ecs_clue']
    st.markdown(t('re_intro', year=CHECKPOINT, clue=_num(clue, 1), err=_num(1.96 * ECS_CLUE_SD, 1)))
    temperature, stages = _show_estimates(st.session_state['cuts'], None, 'research')
    ecs = _load_screened()['ECS']
    ecs_old, ecs_new = _interval_width_change(ecs, stages, 'prior', 'research')
    t_old, t_new = _interval_width_change(temperature, stages, 'prior', 'research')
    st.info(t('re_narrow', ecs_old=_num(ecs_old, 1), ecs_new=_num(ecs_new, 1),
              ecs_pct=f'{(1 - ecs_new / ecs_old) * 100:.0f}', year=ESTIMATE_YEAR,
              t_old=_num(t_old, 1), t_new=_num(t_new, 1), t_pct=f'{(1 - t_new / t_old) * 100:.0f}'))
    with st.expander(t('re_expander_title')):
        st.markdown(t('re_expander_body', year=ESTIMATE_YEAR))
    if st.button(t('re_next'), type='primary'):
        _goto('phase2')


def _screen_phase2():
    st.title(t('p2_title'))
    st.markdown(t('p2_intro'))

    _nav_bar()

    cuts            = st.session_state['cuts']
    netzero_years   = st.session_state['netzero_years']
    netzero_indices = st.session_state['netzero_indices']

    st.subheader(t('p2_subheader'))
    cols = st.columns(len(REGION_EMISSIONS_BASE_GTCO2))
    for col, region in zip(cols, REGION_EMISSIONS_BASE_GTCO2):
        with col:
            min_nz = _min_netzero_year(cuts.get(region, 0.0))
            nz     = _nz_input(region, min_nz, netzero_indices)
            netzero_years[region] = nz
            em_chk = REGION_EMISSIONS_BASE_GTCO2[region] * (1 - cuts.get(region, 0.0))
            st.caption(t('p2_region_caption', year=CHECKPOINT, em=_num(em_chk, 1), nz=nz))
    st.session_state['netzero_years']   = netzero_years
    st.session_state['netzero_indices'] = netzero_indices

    st.pyplot(_emissions_pathway_chart(cuts, netzero_years))

    st.subheader(t('p2_estimate'))
    _show_estimates(cuts, netzero_years, 'phase2')

    if st.button(t('p2_button'), type='primary'):
        with st.spinner(t('spinner_fair')):
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
        st.info(t('p2_stale'))
        return

    temp2 = st.session_state['temp_phase2']
    temp1 = st.session_state['temp_phase1']

    after_chk = temp2.sel(year=temp2.year > CHECKPOINT)
    peak_temp = float(after_chk.max())
    peak_yr   = int(after_chk.year.values[after_chk.values.argmax()])

    sim_end = int(temp2.year.max())
    display_end = min(peak_yr + 30, sim_end)

    st.subheader(t('p2_results'))
    netzero_key = '-'.join(str(netzero_years[r]) for r in REGION_EMISSIONS_BASE_GTCO2)
    _show_temperature_animation(
        f'phase2-{st.session_state["config_id"]}-{netzero_key}',
        temp_da=temp2, year_start=1900, year_end=display_end,
        show_peak=True, phase1_da=temp1,
    )

    if peak_temp < PARIS_15:
        st.success(t('p2_below15'))
    elif peak_temp < PARIS_20:
        st.warning(t('p2_between'))
    else:
        st.error(t('p2_above20'))

    if st.button(t('p2_next'), type='primary'):
        _goto('reveal')


def _screen_reveal():
    st.title(t('rv_title'))
    _nav_bar()
    cuts = st.session_state['cuts']
    netzero_years = st.session_state['phase2_netzero_used']
    if netzero_years is None or st.session_state['temp_phase2'] is None:
        st.info(t('rv_need_phase2'))
        return

    true_ecs = st.session_state['ecs']
    true_temp = float(st.session_state['temp_phase2'].sel(year=ESTIMATE_YEAR))
    st.markdown(t('rv_intro', ecs=_num(true_ecs, 2), year=ESTIMATE_YEAR, temp=_num(true_temp, 2)))
    temperature, stages = _show_estimates(cuts, netzero_years, 'reveal',
                                          truth=(true_ecs, true_temp))

    final = stages['research']
    ecs = _load_screened()['ECS']
    checks = [
        (t('rv_ecs_name'), ecs, true_ecs),
        (t('rv_temp_name', year=ESTIMATE_YEAR), temperature, true_temp),
    ]
    for name, values, true_value in checks:
        mean, lo, hi = weighted_mean_interval(values, final)
        result = dict(name=name, value=_num(true_value, 2), lo=_num(lo, 2), hi=_num(hi, 2))
        if lo <= true_value <= hi:
            st.success(t('rv_hit', mean=_num(mean, 2), **result))
        else:
            st.warning(t('rv_miss', **result))
    st.markdown(t('rv_lesson'))


def _language_picker():
    """First page of a new session: choose the language (both languages shown, nothing chosen yet)."""
    st.title('🌍 Ilmastoneuvottelupeli · Climate Negotiation Game')
    st.markdown('Valitse kieli · Choose the language')
    col_fi, col_en, _ = st.columns([1, 1, 2])
    if col_fi.button('🇫🇮 Suomi', type='primary', width='stretch'):
        st.session_state['lang'] = 'fi'
        st.rerun()
    if col_en.button('🇬🇧 English', type='primary', width='stretch'):
        st.session_state['lang'] = 'en'
        st.rerun()


def _language_switch():
    """Small FI/EN switch at the top right of every page. Choices and results are kept."""
    _, col = st.columns([6, 1])
    st.session_state['lang'] = col.radio(
        'Kieli · Language', options=LANGUAGES, format_func=str.upper,
        index=LANGUAGES.index(st.session_state['lang']), horizontal=True,
        key='lang_switch', label_visibility='collapsed',
    )


def main():
    st.set_page_config(
        page_title='Ilmastoneuvottelupeli · Climate Negotiation Game',
        page_icon='🌍',
        layout='wide',
    )
    if 'lang' not in st.session_state:
        _language_picker()
        return
    _language_switch()
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
    if st.button(t('start_over')):
        # The language is kept: whoever restarts has already chosen it
        for key in list(st.session_state.keys()):
            if key != 'lang':
                del st.session_state[key]
        st.rerun()


if __name__ == '__main__':
    main()

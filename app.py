#!/usr/bin/env python3
"""
Ilmastoneuvottelupeli — FaIR-pohjainen luokkahuonedemosovellus.

Vaihe 1: Ryhmät sopivat CO2-päästöleikkauksista vuoteen 2040 (CHECKPOINT).
Vaihe 2: Kukin ryhmä sopii omasta nollapäästövuodestaan ja näkee huipputulämpötilan.

ECS paljastetaan vasta ensimmäisen simulaation jälkeen, jotta se voidaan
huomioida vaiheen 2 päätöksissä.
"""
import numpy as np
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
    REGION_EMISSIONS_2023_GTCO2,
    load_screened_configs,
    load_observed_gmst,
    run_fair_game,
)

TOTAL_2023   = sum(REGION_EMISSIONS_2023_GTCO2.values())
PARIS_15     = 1.5
PARIS_20     = 2.0
CHECKPOINT      = 2040   # Phase 1 pledge year
ANIM_START      = 2026   # year from which temperature trajectory is animated
ANIM_INTERVAL   = 80     # ms per animation frame (increase to slow down)

# Emission reduction options per region (3 discrete choices).
# Based on emission_reduction_options.md.
# Each entry: (Finnish label, cut_fraction) where cut_fraction > 0 = reduction,
# < 0 = emissions increase (India Varovainen).
EMISSION_OPTIONS = {
    'Kiina 🇨🇳': [
        ('Varovainen (−10 %)',    0.10),
        ('Vahva (−40 %)',         0.40),
        ('Maksimaalinen (−65 %)', 0.65),
    ],
    'USA 🇺🇸': [
        ('Varovainen (−20 %)',    0.20),
        ('Vahva (−55 %)',         0.55),
        ('Maksimaalinen (−80 %)', 0.80),
    ],
    'EU 🇪🇺': [
        ('Varovainen (−55 %)',    0.55),
        ('Vahva (−75 %)',         0.75),
        ('Maksimaalinen (−90 %)', 0.90),
    ],
    'Intia 🇮🇳': [
        ('Varovainen (+35 %)',   -0.35),   # NDC intensity target → absolute rise
        ('Vahva (−20 %)',         0.20),
        ('Maksimaalinen (−50 %)', 0.50),
    ],
    'Muu maailma 🌍': [
        ('Varovainen (−15 %)',    0.15),
        ('Vahva (−35 %)',         0.35),
        ('Maksimaalinen (−60 %)', 0.60),
    ],
}

# Maximum post-CHECKPOINT decarbonisation rate: 10 % of each region's 2023 baseline
# per year (in absolute GtCO2 terms, not fraction of remaining stock).
# Basis: AR6 WGIII 1.5 °C pathways; fastest observed national transitions.
# Sets a physically plausible floor on the net-zero year given Phase 1 ambition.
MAX_DECARB_RATE_PER_YEAR = 0.10

# Net-zero timing options for Phase 2 (offsets in years from the per-region minimum).
# Minimum = CHECKPOINT + ceil(em_remaining / MAX_DECARB_RATE_PER_YEAR).
NETZERO_OFFSETS = [
    ('Nopein mahdollinen',   0),
    ('Maltillinen (+20 v)', 20),
    ('Hidas (+40 v)',       40),
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

    em_remaining = 1 − cut_fraction is the fraction of 2023 emissions still
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


def _draw_config(screened_df):
    """Randomly pick one config from the screened shortlist."""
    row = screened_df.sample(1).iloc[0]
    return int(row.name), float(row['ECS'])


def _init_state():
    # Default Phase 1: index 0 (Varovainen) for all regions
    _default_cuts = {r: EMISSION_OPTIONS[r][0][1] for r in REGION_EMISSIONS_2023_GTCO2}
    defaults = {
        'phase':           1,
        'option_indices':  {r: 0 for r in REGION_EMISSIONS_2023_GTCO2},
        'cuts':            _default_cuts,
        # Default Phase 2: Maltillinen (+20 v) for all regions
        'netzero_indices': {r: 1 for r in REGION_EMISSIONS_2023_GTCO2},
        'netzero_years':   {
            r: min(_min_netzero_year(_default_cuts[r]) + 20, 2100)
            for r in REGION_EMISSIONS_2023_GTCO2
        },
        'temp_phase1':     None,
        'temp_phase2':     None,
        # config_id and ecs are None until "Simuloi!" is first clicked
        'config_id':       None,
        'ecs':             None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _ecs_banner():
    ecs = st.session_state['ecs']
    st.info(
        f'🎲 **Tämän kierroksen ilmastomalli:** '
        f'Tasapainoherkkyys (ECS) = **{ecs:.2f} °C**  \n'
        f'*(Näin paljon Maa lämpenisi, jos CO₂-pitoisuus kaksinkertaistuisi pysyvästi)*'
    )


def _impacts_page():
    st.title('🌡️ Mitä lämpeneminen tarkoittaa?')
    st.markdown(
        'Valitse lämpenemistaso nähdäksesi vaikutukset maailmalla ja Suomessa.  \n'
        '*(Verrattuna esiteolliseen aikaan 1850–1900)*'
    )
    _nav_bar()

    selected = st.radio(
        '',
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

    # Cross-reference with Phase 1 simulation result if available
    if st.session_state.get('temp_phase1') is not None:
        temp  = st.session_state['temp_phase1']
        t_chk = float(temp.sel(timebounds=CHECKPOINT))
        closest = min(IMPACTS.keys(), key=lambda x: abs(x - t_chk))
        st.markdown('---')
        st.info(
            f'📊 **Teidän simulaationne:** lämpötila {CHECKPOINT} → **+{t_chk:.2f} °C**  \n'
            f'Tämä on lähimpänä **+{closest} °C** tasoa yllä olevassa vertailussa.'
        )

    st.markdown(
        '<p style="color:#4A5568;font-size:0.8em;margin-top:24px">'
        'Lähteet: IPCC AR6 WG1 (2021), SR1.5 (2018) · '
        'Suomen luvut: Ilmatieteen laitos</p>',
        unsafe_allow_html=True,
    )


def _nav_bar():
    """Persistent top navigation. Phase 2 only accessible after Phase 1 sim."""
    col0, col1, col2 = st.columns([1, 1, 1])
    with col0:
        if st.session_state['phase'] != 0:
            if st.button('🌡️ Vaikutukset'):
                st.session_state['phase'] = 0
                st.rerun()
    with col1:
        if st.session_state['phase'] != 1:
            if st.button('⬅ Vaihe 1: Päästövähennykset'):
                st.session_state['phase'] = 1
                st.rerun()
    with col2:
        if st.session_state['temp_phase1'] is not None and st.session_state['phase'] != 2:
            if st.button('Vaihe 2: Nollapäästövuodet ➡'):
                st.session_state['phase'] = 2
                st.rerun()
    st.markdown('---')


def _make_temperature_animation(temp_da, year_start, year_end,
                                 highlight_year=None, show_peak=False,
                                 phase1_da=None, inset_metrics=None):
    """
    Animated temperature trajectory: historical portion (up to ANIM_START) is static;
    future portion is drawn year-by-year. Returns an HTML string for components.html().

    temp_da        : xr.DataArray dim 'timebounds', single config
    phase1_da      : if given, draw Phase 1 trajectory as grey dashed line (static)
    highlight_year : dotted vertical line (Phase 1: CHECKPOINT; Phase 2: peak year)
    show_peak      : annotate peak with arrow on final animation frame
    inset_metrics  : list of (label, value_str) shown as text box on final frame
    """
    gmst = _load_gmst()

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    years  = temp_da.timebounds.values
    mask   = (years >= year_start) & (years <= year_end)
    yr     = years[mask]
    t_vals = temp_da.isel(timebounds=mask).values

    # Split at animation boundary; overlap by one point so the line is continuous
    h_mask   = yr <= ANIM_START
    fut_mask = yr >= ANIM_START
    yr_fut   = yr[fut_mask]
    t_fut    = t_vals[fut_mask]

    # Fix axis limits before drawing so text placements are stable
    y_min = min(float(t_vals.min()), 0.0) - 0.15
    y_max = float(t_vals.max()) + 0.45
    ax.set_xlim(year_start, year_end)
    ax.set_ylim(y_min, y_max)

    # --- static elements ---
    if phase1_da is not None:
        yr1 = phase1_da.timebounds.values
        m1  = (yr1 >= year_start) & (yr1 <= CHECKPOINT)
        ax.plot(yr1[m1], phase1_da.isel(timebounds=m1).values,
                color='#888888', linewidth=2.5, linestyle='--', label='Vaihe 1')

    ax.plot(yr[h_mask], t_vals[h_mask], color='#5A84E2', linewidth=2.5)

    obs_mask = (gmst['year'] >= year_start) & (gmst['year'] <= min(2025, year_end))
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
        ax.axvline(highlight_year, color='white', linewidth=1.2, linestyle=':')
        if not show_peak:
            ax.text(highlight_year + 1, y_min + 0.05, str(highlight_year),
                    color='white', fontsize=9)

    ax.set_xlabel('Vuosi', color='white')
    ax.set_ylabel('Lämpötila-anomalia (°C)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    # Animated future line (empty at start)
    model_line, = ax.plot([], [], color='#5A84E2', linewidth=2.5, label='Ilmastomalli')
    ax.legend(facecolor='#1e2130', labelcolor='white', fontsize=9)

    n_frames = len(yr_fut)

    def update(frame):
        model_line.set_data(yr_fut[:frame + 1], t_fut[:frame + 1])
        # On the final frame, add annotations that summarise the result
        if frame == n_frames - 1:
            if show_peak:
                peak_temp = float(t_vals.max())
                peak_yr   = int(yr[t_vals.argmax()])
                ax.annotate(
                    f'Huippu: {peak_temp:.2f} °C ({peak_yr})',
                    xy=(peak_yr, peak_temp),
                    xytext=(peak_yr + 5, peak_temp + 0.1),
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
    """Bar chart: 2023 baseline vs. pledged CHECKPOINT emissions per region."""
    regions = list(REGION_EMISSIONS_2023_GTCO2.keys())
    base    = [REGION_EMISSIONS_2023_GTCO2[r] for r in regions]
    target  = [REGION_EMISSIONS_2023_GTCO2[r] * (1 - cuts[r]) for r in regions]
    colors  = [REGION_COLORS[r] for r in regions]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    x     = np.arange(len(regions))
    width = 0.38
    ax.bar(x - width / 2, base,   width, color=colors, alpha=0.45, label='Päästöt 2023')
    ax.bar(x + width / 2, target, width, color=colors, alpha=0.90, label=f'Päästöt {CHECKPOINT} (lupaus)')

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
    """Stacked area chart: per-region CO2 pathway 2023 → each region's net-zero year."""
    max_nz   = max(netzero_years.values())
    plot_end = min(max_nz + 10, 2110)
    years    = np.arange(2023, plot_end + 1)
    regions  = list(REGION_EMISSIONS_2023_GTCO2.keys())

    region_em = {}
    for r, base in REGION_EMISSIONS_2023_GTCO2.items():
        cut    = cuts.get(r, 0.0)
        em2023 = base
        em_chk = base * (1 - cut)
        nz     = netzero_years.get(r, 2070)

        em = np.zeros(len(years))
        for i, y in enumerate(years):
            if y < CHECKPOINT:
                em[i] = em2023 + (em_chk - em2023) * (y - 2023) / (CHECKPOINT - 2023)
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
    ax.set_xlim(2023, plot_end)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.legend(facecolor='#1e2130', labelcolor='white', fontsize=8,
              loc='upper right', ncol=2)
    ax.set_title('Päästöpolku alueittain', color='white', fontsize=10)
    fig.tight_layout()
    return fig


def _cut_input(region: str, option_indices: dict) -> float:
    """
    Return the chosen cut fraction for a region.
    In slider mode (--sliders flag): free slider, returns fraction converted from %.
    In radio mode (default): 3-option radio from EMISSION_OPTIONS.
    Updates option_indices[region] in place.
    """
    if USE_SLIDERS:
        # Initialise slider from whichever radio option is currently selected
        stored_idx  = option_indices.get(region, 0)
        current_pct = int(EMISSION_OPTIONS[region][stored_idx][1] * 100)
        pct = st.slider(
            region, min_value=-50, max_value=100,
            value=current_pct, step=5, format='%d%%',
            key=f'slider_p1_{region}',
        )
        return pct / 100.0
    options = EMISSION_OPTIONS[region]
    idx = st.radio(
        region,
        options=range(len(options)),
        format_func=lambda i, opts=options: opts[i][0],
        index=option_indices.get(region, 0),
        key=f'radio_p1_{region}',
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
        nz_opts = [(lbl, min(min_nz + off, 2100)) for lbl, off in NETZERO_OFFSETS]
        default_yr = nz_opts[current][1]
        return st.slider(
            region, min_value=min_nz, max_value=2150,
            value=max(default_yr, min_nz), step=5,
            key=f'slider_nz_{region}',
        )
    nz_opts = [(lbl, min(min_nz + off, 2100)) for lbl, off in NETZERO_OFFSETS]
    idx = st.radio(
        region,
        options=range(len(nz_opts)),
        format_func=lambda i, o=nz_opts: f'{o[i][0]} ({o[i][1]})',
        index=netzero_indices.get(region, 1),
        key=f'radio_nz_{region}',
    )
    netzero_indices[region] = idx
    return nz_opts[idx][1]


def _phase1():
    st.title('🌍 Ilmastoneuvottelut — Mitä lupaatte?')
    st.markdown(
        'Jokainen ryhmä edustaa yhtä maailman suurta päästöaluetta. '
        f'Sopikaa, kuinka paljon leikkaatte **CO₂-päästöjä** vuoteen **{CHECKPOINT}** mennessä.'
    )

    _nav_bar()

    cuts          = st.session_state['cuts']
    option_indices = st.session_state['option_indices']

    st.subheader('Aseta jokaisen alueen lupaus')
    cols = st.columns(len(REGION_EMISSIONS_2023_GTCO2))
    for col, (region, base_em) in zip(cols, REGION_EMISSIONS_2023_GTCO2.items()):
        with col:
            cut    = _cut_input(region, option_indices)
            cuts[region] = cut
            em_chk = base_em * (1.0 - cut)
            st.caption(
                f'2023: **{base_em:.1f}** GtCO₂/v  \n'
                f'{CHECKPOINT}: **{em_chk:.1f}** GtCO₂/v'
            )
    st.session_state['cuts']          = cuts
    st.session_state['option_indices'] = option_indices

    total_chk     = sum(REGION_EMISSIONS_2023_GTCO2[r] * (1 - cuts[r]) for r in cuts)
    total_cut_pct = (1 - total_chk / TOTAL_2023) * 100
    change_str = (f'{total_cut_pct:.0f} % leikkaus' if total_cut_pct >= 0
                  else f'{-total_cut_pct:.0f} % kasvu')
    st.info(
        f'**Yhteensä {CHECKPOINT}:** {total_chk:.1f} GtCO₂/vuosi  '
        f'({change_str} vuoden 2023 tasosta)'
    )
    st.pyplot(_emissions_bar_chart(cuts))

    if st.button('🔬 Simuloi! →', type='primary'):
        screened = _load_screened()
        cfg_id, ecs = _draw_config(screened)
        st.session_state['config_id'] = cfg_id
        st.session_state['ecs']       = ecs
        # Phase 2 result is stale whenever cuts change and Phase 1 reruns
        st.session_state['temp_phase2'] = None
        # Reset Phase 2 net-zero choices to Maltillinen (+20 v) based on new Phase 1 cuts
        st.session_state['netzero_indices'] = {r: 1 for r in REGION_EMISSIONS_2023_GTCO2}
        st.session_state['netzero_years'] = {
            r: min(_min_netzero_year(cuts[r]) + 20, 2100)
            for r in REGION_EMISSIONS_2023_GTCO2
        }
        with st.spinner('Lasketaan FaIR-simulaatio...'):
            temp = run_fair_game(
                cuts,
                netzero_years=None,
                config_id=cfg_id,
            )
            st.session_state['temp_phase1'] = temp
        st.rerun()

    if st.session_state['temp_phase1'] is not None:
        temp   = st.session_state['temp_phase1']
        t_chk = float(temp.sel(timebounds=CHECKPOINT))
        t_now = float(temp.sel(timebounds=2024))

        _ecs_banner()
        st.subheader('📊 Simulaatiotulokset')
        components.html(
            _make_temperature_animation(
                temp, year_start=1900, year_end=CHECKPOINT,
                highlight_year=CHECKPOINT,
                inset_metrics=[
                    ('Malli: lämpötila nyt (2024)',        f'+{t_now:.2f} °C'),
                    (f'Malli: keskilämpötila {CHECKPOINT}', f'+{t_chk:.2f} °C'),
                ],
            ),
            height=520,
        )

        if t_chk < PARIS_15:
            st.success(f'✅ Alle 1,5 °C:n tavoitteen vuonna {CHECKPOINT}!')
        elif t_chk < PARIS_20:
            st.warning(f'⚠️ Alle 2 °C:n rajan vuonna {CHECKPOINT}, mutta yli 1,5 °C')
        else:
            st.error(f'❌ Yli 2 °C:n Pariisin rajan vuonna {CHECKPOINT}')


def _phase2():
    st.title('🎯 Milloin kukin alue saavuttaa nollapäästöt?')
    st.markdown(
        'Jokainen ryhmä päättää, milloin **oma alueensa** CO₂-päästöt laskevat nollaan. '
        'Metaani ja muut kasvihuonekaasut seuraavat nykyistä kehitystä (SSP2-4.5).'
    )

    _nav_bar()
    _ecs_banner()

    cuts            = st.session_state['cuts']
    netzero_years   = st.session_state['netzero_years']
    netzero_indices = st.session_state['netzero_indices']

    st.subheader('Valitse nollapäästövuosi alueittain')
    cols = st.columns(len(REGION_EMISSIONS_2023_GTCO2))
    for col, region in zip(cols, REGION_EMISSIONS_2023_GTCO2):
        with col:
            min_nz = _min_netzero_year(cuts.get(region, 0.0))
            nz     = _nz_input(region, min_nz, netzero_indices)
            netzero_years[region] = nz
            em_chk = REGION_EMISSIONS_2023_GTCO2[region] * (1 - cuts.get(region, 0.0))
            st.caption(f'{CHECKPOINT}: {em_chk:.1f} GtCO₂/v → 0 vuoteen **{nz}**')
    st.session_state['netzero_years']   = netzero_years
    st.session_state['netzero_indices'] = netzero_indices

    st.pyplot(_emissions_pathway_chart(cuts, netzero_years))

    if st.button('🔬 Laske huipputulämpötila', type='primary'):
        with st.spinner('Lasketaan FaIR-simulaatio...'):
            temp = run_fair_game(
                cuts,
                netzero_years=netzero_years,
                config_id=st.session_state['config_id'],
            )
            st.session_state['temp_phase2'] = temp
        st.rerun()

    if st.session_state['temp_phase2'] is not None:
        temp2 = st.session_state['temp_phase2']
        temp1 = st.session_state['temp_phase1']

        after_chk = temp2.sel(timebounds=temp2.timebounds > CHECKPOINT)
        peak_temp = float(after_chk.max())
        peak_yr   = int(after_chk.timebounds.values[after_chk.values.argmax()])

        display_end = min(peak_yr + 30, 2150)

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

    st.markdown('---')
    if st.button('🔄 Aloita alusta (uusi ilmastomalli)'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def main():
    st.set_page_config(
        page_title='Ilmastoneuvottelupeli',
        page_icon='🌍',
        layout='wide',
    )
    _init_state()

    if st.session_state['phase'] == 0:
        _impacts_page()
    elif st.session_state['phase'] == 1:
        _phase1()
    else:
        _phase2()


if __name__ == '__main__':
    main()

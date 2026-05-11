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
import streamlit as st
import streamlit.components.v1 as components

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

REGION_COLORS = {
    'Kiina 🇨🇳':      '#E26D91',
    'USA 🇺🇸':         '#5A84E2',
    'EU 🇪🇺':          '#65C06C',
    'Intia 🇮🇳':       '#FFA73E',
    'Muu maailma 🌍': '#8760C1',
}


def _plain_name(region: str) -> str:
    """Return the text part of a region name, dropping the trailing flag emoji."""
    return region.rsplit(' ', 1)[0]


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
    defaults = {
        'phase':         1,
        'cuts':          {r: 0.0 for r in REGION_EMISSIONS_2023_GTCO2},
        'netzero_years': {r: 2070 for r in REGION_EMISSIONS_2023_GTCO2},
        'temp_phase1':   None,
        'temp_phase2':   None,
        # config_id and ecs are None until "Simuloi!" is first clicked
        'config_id':     None,
        'ecs':           None,
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


def _nav_bar():
    """Persistent top navigation between phases. Phase 2 only accessible after Phase 1 sim."""
    col1, col2 = st.columns([1, 1])
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


def _phase1():
    st.title('🌍 Ilmastoneuvottelut — Mitä lupaatte?')
    st.markdown(
        'Jokainen ryhmä edustaa yhtä maailman suurta päästöaluetta. '
        f'Sopikaa, kuinka paljon leikkaatte **CO₂-päästöjä** vuoteen **{CHECKPOINT}** mennessä.'
    )

    _nav_bar()

    cuts = st.session_state['cuts']

    st.subheader('Aseta jokaisen alueen lupaus')
    cols = st.columns(len(REGION_EMISSIONS_2023_GTCO2))
    for col, (region, base_em) in zip(cols, REGION_EMISSIONS_2023_GTCO2.items()):
        with col:
            pct = st.slider(
                region,
                min_value=0, max_value=100,
                value=int(cuts[region] * 100),
                step=5, format='%d%%',
                key=f'slider_p1_{region}',
            )
            cuts[region] = pct / 100.0
            em_chk = base_em * (1 - cuts[region])
            st.caption(
                f'2023: **{base_em:.1f}** GtCO₂/v  \n'
                f'{CHECKPOINT}: **{em_chk:.1f}** GtCO₂/v'
            )
    st.session_state['cuts'] = cuts

    total_chk     = sum(REGION_EMISSIONS_2023_GTCO2[r] * (1 - cuts[r]) for r in cuts)
    total_cut_pct = (1 - total_chk / TOTAL_2023) * 100
    st.info(
        f'**Yhteensä {CHECKPOINT}:** {total_chk:.1f} GtCO₂/vuosi  '
        f'({total_cut_pct:.0f} % leikkaus vuoden 2023 tasosta)'
    )
    st.pyplot(_emissions_bar_chart(cuts))

    if st.button('🔬 Simuloi! →', type='primary'):
        screened = _load_screened()
        cfg_id, ecs = _draw_config(screened)
        st.session_state['config_id'] = cfg_id
        st.session_state['ecs']       = ecs
        # Phase 2 result is stale whenever cuts change and Phase 1 reruns
        st.session_state['temp_phase2'] = None
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

    cuts          = st.session_state['cuts']
    netzero_years = st.session_state['netzero_years']

    st.subheader('Valitse nollapäästövuosi alueittain')
    cols = st.columns(len(REGION_EMISSIONS_2023_GTCO2))
    for col, region in zip(cols, REGION_EMISSIONS_2023_GTCO2):
        with col:
            nz = st.slider(
                region,
                min_value=CHECKPOINT, max_value=2100,
                value=netzero_years[region],
                step=5,
                key=f'slider_nz_{region}',
            )
            netzero_years[region] = nz
            em_chk = REGION_EMISSIONS_2023_GTCO2[region] * (1 - cuts.get(region, 0.0))
            st.caption(f'Nollapäästöt: **{nz}**  \n{CHECKPOINT}: {em_chk:.1f} GtCO₂/v → 0')
    st.session_state['netzero_years'] = netzero_years

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

    if st.session_state['phase'] == 1:
        _phase1()
    else:
        _phase2()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Ilmastoneuvottelupeli — FaIR-pohjainen luokkahuonedemosovellus.

Vaihe 1: Ryhmät sopivat CO2-päästöleikkauksista vuoteen 2040.
Vaihe 2: Kukin ryhmä sopii omasta nollapäästövuodestaan ja näkee huipputulämpötilan.

Jokainen uusi peli arpoo yhden ilmastomallin 382 hyväksytystä konfiguraatiosta.
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from fair_runner import (
    REGION_EMISSIONS_2023_GTCO2,
    load_screened_configs,
    load_observed_gmst,
    run_fair_game,
)

TOTAL_2023 = sum(REGION_EMISSIONS_2023_GTCO2.values())
PARIS_15   = 1.5
PARIS_20   = 2.0

# Colour palette for regions (CartoColor Safe)
REGION_COLORS = {
    'Kiina 🇨🇳':      '#E26D91',
    'USA 🇺🇸':         '#5A84E2',
    'EU 🇪🇺':          '#65C06C',
    'Intia 🇮🇳':       '#FFA73E',
    'Muu maailma 🌍': '#8760C1',
}


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
    screened = _load_screened()
    defaults = {
        'phase':        1,
        'cuts':         {r: 0.0 for r in REGION_EMISSIONS_2023_GTCO2},
        'netzero_years': {r: 2070 for r in REGION_EMISSIONS_2023_GTCO2},
        'temp_phase1':  None,
        'temp_phase2':  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Draw a config on first visit
    if 'config_id' not in st.session_state:
        cfg_id, ecs = _draw_config(screened)
        st.session_state['config_id'] = cfg_id
        st.session_state['ecs']        = ecs


def _ecs_banner():
    ecs = st.session_state['ecs']
    st.info(
        f'🎲 **Tämän kierroksen ilmastomalli:** '
        f'Tasapainoherkkyys (ECS) = **{ecs:.2f} °C**  \n'
        f'*(Näin paljon Maa lämpenisi, jos CO₂-pitoisuus kaksinkertaistuisi pysyvästi)*'
    )


def _make_temperature_plot(temp_da, year_start, year_end,
                            highlight_year=None, show_peak=False,
                            phase1_da=None):
    """
    Draw a single-config temperature trajectory with observed GMST overlay.

    temp_da   : xr.DataArray with dim 'timebounds', single config (no config dim)
    phase1_da : if given, draw Phase 1 trajectory as a grey dashed prefix up to 2040
    """
    gmst = _load_gmst()

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    years  = temp_da.timebounds.values
    mask   = (years >= year_start) & (years <= year_end)
    yr     = years[mask]
    t_vals = temp_da.isel(timebounds=mask).values

    # Phase 1 trajectory shown as grey dashed prefix in Phase 2
    if phase1_da is not None:
        yr1 = phase1_da.timebounds.values
        m1  = (yr1 >= year_start) & (yr1 <= 2040)
        ax.plot(yr1[m1], phase1_da.isel(timebounds=m1).values,
                color='#888888', linewidth=2.5, linestyle='--',
                label='Vaihe 1')

    ax.plot(yr, t_vals, color='#5A84E2', linewidth=2.5, label='FaIR-malli')

    # Observed GMST (thin white line) — shows El Niño spike at 2023
    obs_mask = (gmst['year'] >= year_start) & (gmst['year'] <= min(2025, year_end))
    ax.plot(gmst.loc[obs_mask, 'year'], gmst.loc[obs_mask, 'gmst'],
            color='white', linewidth=1.5, linestyle='-',
            label='Havaittu (IGCC + WMO)')

    # Paris Agreement target lines
    ax.axhline(PARIS_15, color='#65C06C', linewidth=1.5, linestyle='--')
    ax.axhline(PARIS_20, color='#FFA73E', linewidth=1.5, linestyle='--')
    ax.text(year_start + 1, PARIS_15 + 0.04, '1,5 °C (Pariisin tavoite)',
            color='#65C06C', fontsize=9)
    ax.text(year_start + 1, PARIS_20 + 0.04, '2,0 °C (Pariisin raja)',
            color='#FFA73E', fontsize=9)

    if highlight_year is not None:
        ax.axvline(highlight_year, color='white', linewidth=1.2, linestyle=':')
        ax.text(highlight_year + 1, ax.get_ylim()[0] + 0.05,
                str(highlight_year), color='white', fontsize=9)

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

    ax.set_xlabel('Vuosi', color='white')
    ax.set_ylabel('Lämpötila-anomalia (°C)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.legend(facecolor='#1e2130', labelcolor='white', fontsize=9)
    fig.tight_layout()
    return fig


def _emissions_bar_chart(cuts):
    """Bar chart: 2023 baseline vs. pledged 2040 emissions per region."""
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
    ax.bar(x + width / 2, target, width, color=colors, alpha=0.90, label='Päästöt 2040 (lupaus)')

    ax.set_xticks(x)
    ax.set_xticklabels(regions, color='white', fontsize=9)
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

    # Build per-region emission trajectories (GtCO2/yr for display)
    region_em = {}
    for r, base in REGION_EMISSIONS_2023_GTCO2.items():
        cut    = cuts.get(r, 0.0)
        em2023 = base
        em2040 = base * (1 - cut)
        nz     = netzero_years.get(r, 2070)

        em = np.zeros(len(years))
        for i, y in enumerate(years):
            if y < 2040:
                em[i] = em2023 + (em2040 - em2023) * (y - 2023) / 17
            elif y < nz:
                em[i] = em2040 * (1 - (y - 2040) / (nz - 2040))
            else:
                em[i] = 0.0
        region_em[r] = em

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    stack_bottom = np.zeros(len(years))
    for r in regions:
        ax.fill_between(years, stack_bottom, stack_bottom + region_em[r],
                        color=REGION_COLORS[r], alpha=0.75, label=r)
        stack_bottom += region_em[r]

    ax.axvline(2040, color='white', linewidth=1, linestyle=':')
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
        'Sopikaa, kuinka paljon leikkaatte **CO₂-päästöjä** vuoteen **2040** mennessä.'
    )
    _ecs_banner()

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
            em_2040 = base_em * (1 - cuts[region])
            st.caption(
                f'2023: **{base_em:.1f}** GtCO₂/v  \n'
                f'2040: **{em_2040:.1f}** GtCO₂/v'
            )
    st.session_state['cuts'] = cuts

    total_2040      = sum(REGION_EMISSIONS_2023_GTCO2[r] * (1 - cuts[r]) for r in cuts)
    total_cut_pct   = (1 - total_2040 / TOTAL_2023) * 100
    st.info(
        f'**Yhteensä 2040:** {total_2040:.1f} GtCO₂/vuosi  '
        f'({total_cut_pct:.0f} % leikkaus vuoden 2023 tasosta)'
    )
    st.pyplot(_emissions_bar_chart(cuts))

    if st.button('🔬 Simuloi! →', type='primary'):
        with st.spinner('Lasketaan FaIR-simulaatio...'):
            temp = run_fair_game(
                cuts,
                netzero_years=None,
                config_id=st.session_state['config_id'],
            )
            st.session_state['temp_phase1'] = temp
        st.rerun()

    if st.session_state['temp_phase1'] is not None:
        temp      = st.session_state['temp_phase1']
        t_2040    = float(temp.sel(timebounds=2040))
        t_now     = float(temp.sel(timebounds=2024))

        st.subheader('📊 Simulaatiotulokset')
        st.pyplot(_make_temperature_plot(temp, year_start=1900, year_end=2060,
                                          highlight_year=2040))

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric('Malli: lämpötila nyt (2024)', f'+{t_now:.2f} °C')
            st.metric('Malli: keskilämpötila 2040',  f'+{t_2040:.2f} °C')
        with col_b:
            if t_2040 < PARIS_15:
                st.success('✅ Alle 1,5 °C:n tavoitteen!')
            elif t_2040 < PARIS_20:
                st.warning('⚠️ Alle 2 °C:n rajan, mutta yli 1,5 °C')
            else:
                st.error('❌ Yli 2 °C:n Pariisin rajan')

        st.markdown('---')
        if st.button('➡️ Vaihe 2: Päätä nollapäästövuodet', type='primary'):
            st.session_state['phase'] = 2
            st.rerun()


def _phase2():
    st.title('🎯 Milloin kukin alue saavuttaa nollapäästöt?')
    st.markdown(
        'Jokainen ryhmä päättää, milloin **oma alueensa** CO₂-päästöt laskevat nollaan. '
        'Metaani ja muut kasvihuonekaasut seuraavat nykyistä kehitystä (SSP2-4.5).'
    )
    _ecs_banner()

    cuts         = st.session_state['cuts']
    netzero_years = st.session_state['netzero_years']

    st.subheader('Valitse nollapäästövuosi alueittain')
    cols = st.columns(len(REGION_EMISSIONS_2023_GTCO2))
    for col, region in zip(cols, REGION_EMISSIONS_2023_GTCO2):
        with col:
            nz = st.slider(
                region,
                min_value=2040, max_value=2100,
                value=netzero_years[region],
                step=5,
                key=f'slider_nz_{region}',
            )
            netzero_years[region] = nz
            em_2040 = REGION_EMISSIONS_2023_GTCO2[region] * (1 - cuts.get(region, 0.0))
            st.caption(f'Nollapäästöt: **{nz}**  \n2040: {em_2040:.1f} GtCO₂/v → 0')
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

        # Peak temperature: maximum of model trajectory after 2040
        after2040   = temp2.sel(timebounds=temp2.timebounds > 2040)
        peak_temp   = float(after2040.max())
        peak_yr     = int(after2040.timebounds.values[after2040.values.argmax()])

        display_end = min(peak_yr + 30, 2150)
        latest_nz   = max(netzero_years.values())

        st.subheader('📊 Simulaatiotulokset')
        st.pyplot(
            _make_temperature_plot(
                temp2, year_start=1900, year_end=display_end,
                highlight_year=latest_nz,
                show_peak=True,
                phase1_da=temp1,
            )
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric('Huipputulämpötila', f'+{peak_temp:.2f} °C ({peak_yr})')
        with col_b:
            if peak_temp < PARIS_15:
                st.success('✅ Pysytään 1,5 °C:n tavoitteessa!')
            elif peak_temp < PARIS_20:
                st.warning('⚠️ Alle 2 °C, mutta yli 1,5 °C')
            else:
                st.error('❌ Ylitetään 2 °C:n Pariisin raja')

    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('⬅️ Takaisin vaiheeseen 1'):
            st.session_state['phase'] = 1
            st.session_state['temp_phase2'] = None
            st.rerun()
    with col2:
        if st.button('🔄 Aloita alusta (uusi ilmastomalli)'):
            screened = _load_screened()
            cfg_id, ecs = _draw_config(screened)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state['config_id'] = cfg_id
            st.session_state['ecs']        = ecs
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

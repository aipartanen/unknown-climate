#!/usr/bin/env python3
"""
Standalone script to produce three presentation-ready PNG figures showing
warming trajectories for SSP1-1.9 and SSP2-4.5 at three ECS quantiles.

Figures match the visual style of the classroom game (dark background,
Paris reference lines, Finnish impact annotations on the right).

Outputs (saved to current directory):
  presentation_fig_v1_median.png   — median ECS only (both SSPs)
  presentation_fig_v2_add_low.png  — add low ECS (0.15 quantile)
  presentation_fig_v3_all.png      — add high ECS (0.85 quantile)

Usage:
  python make_presentation_figure.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from fair_runner import load_screened_configs, load_observed_gmst, run_fair_ssp, annual_series

# ── Scenario settings ──────────────────────────────────────────────────────────
SSPS = {
    'SSP1-1.9': 'ssp119',
    'SSP2-4.5': 'ssp245',
}

# Colors for each SSP trajectory family
SSP_COLORS = {
    'SSP1-1.9': '#5A84E2',   # blue — same as the game trajectory
    'SSP2-4.5': '#E2824A',   # warm orange — clearly distinct
}

# Finnish legend labels for each SSP
SSP_LABELS = {
    'SSP1-1.9': 'Matalat päästöt',
    'SSP2-4.5': 'Nykyiset ilmastolupaukset',
}

YEAR_START = 1900
YEAR_END   = 2100

# ── ECS quantiles ──────────────────────────────────────────────────────────────
QUANTILES = {
    'q15': 0.15,   # low ECS
    'q50': 0.50,   # median ECS
    'q85': 0.85,   # high ECS
}

# Line styles for each quantile
QUANTILE_STYLES = {
    'q50': {'linestyle': '-',    'lw': 2.5, 'alpha': 1.0},
    'q15': {'linestyle': '--',   'lw': 1.8, 'alpha': 0.75},
    'q85': {'linestyle': ':',    'lw': 1.8, 'alpha': 0.75},
}

# ── Impact annotations (copied from app.py TEMP_ANNOTATIONS) ──────────────────
TEMP_ANNOTATIONS = [
    (1.5, '#65C06C', '70–90 % koralleista häviää'),
    (2.0, '#FFA73E', 'korallit katoavat\nItämeri ei enää jäädy'),
    (3.0, '#E2824A', 'Jäämeri jäätön useimpina kesinä\nLapin porotalous uhattuna'),
    (4.0, '#E24444', 'helleaallot 9,4×\netelä-Suomi käytännössä lumeton'),
]

PARIS_15 = 1.5
PARIS_20 = 2.0

BG_COLOR    = '#0e1117'
LEG_COLOR   = '#1e2130'
SPINE_COLOR = '#444444'


def _select_quantile_configs(configs, quantiles):
    """
    For each quantile level, return the screened config whose ECS is closest
    to that quantile of the ECS distribution.

    Returns a dict {label: config_id} and prints a sanity-check table.
    """
    result = {}
    print('ECS quantile selection:')
    print(f'  {"label":4s}  {"target ECS":>10s}  {"selected ECS":>12s}  config_id')
    for label, q in quantiles.items():
        target = configs['ECS'].quantile(q)
        idx = (configs['ECS'] - target).abs().argmin()
        cfg_id = configs.index[idx]
        ecs    = configs.loc[cfg_id, 'ECS']
        result[label] = cfg_id
        print(f'  {label:4s}  {target:10.3f}  {ecs:12.3f}  {cfg_id}')
    print()
    return result


def _run_all_simulations(quantile_configs):
    """
    Run FaIR for every (SSP, ECS quantile) combination.

    Returns a nested dict: results[ssp_label][q_label] -> annual-mean temperature DataArray
    (dim 'year'). Prints progress so the user can see the six runs completing.
    """
    results = {}
    for ssp_label, ssp_key in SSPS.items():
        results[ssp_label] = {}
        for q_label, cfg_id in quantile_configs.items():
            print(f'  Running {ssp_label} / {q_label} (config {cfg_id}) …', flush=True)
            # The annual mean of YEAR_END needs timebound YEAR_END + 1
            temp = run_fair_ssp(ssp_key, cfg_id, year_end=YEAR_END + 1)
            results[ssp_label][q_label] = annual_series(temp)
    print()
    return results


def _make_figure(results, obs_gmst, include_low=False, include_high=False):
    """
    Build one static matplotlib figure.

    Parameters
    ----------
    results : dict
        Nested dict results[ssp_label][q_label] -> xr.DataArray.
    obs_gmst : pd.DataFrame
        Observed GMST with columns 'year' and 'gmst'.
    include_low : bool
        If True, draw the q15 (low ECS) lines.
    include_high : bool
        If True, draw the q85 (high ECS) lines.

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Right margin reserved for impact annotation text
    fig.subplots_adjust(left=0.08, right=0.72, top=0.95, bottom=0.12)

    # ── Compute y-limits from ALL trajectories (all SSPs, all ECS quantiles)
    # so that axis size is identical across all three figure versions.
    all_temps = []
    for ssp_label in SSPS:
        for q_label in QUANTILES:
            temp = results[ssp_label][q_label]
            vals = temp.sel(year=slice(YEAR_START, YEAR_END)).values
            all_temps.extend(vals[~np.isnan(vals)])
    obs_crop = obs_gmst[obs_gmst['year'] >= YEAR_START]
    all_temps.extend(obs_crop['gmst'].values)

    y_min = min(all_temps) - 0.15
    y_max = max(all_temps) + 0.45
    ax.set_xlim(YEAR_START, YEAR_END + 1)
    ax.set_ylim(y_min, y_max)

    # ── Observed GMST ──────────────────────────────────────────────────────────
    obs_plot = obs_gmst[obs_gmst['year'] >= YEAR_START]
    ax.plot(obs_plot['year'], obs_plot['gmst'],
            color='white', lw=1.5, zorder=4, label='Havaittu lämpeneminen')

    # ── Paris reference lines ──────────────────────────────────────────────────
    ax.axhline(PARIS_15, color='#65C06C', lw=1.2, ls='--', zorder=3)
    ax.axhline(PARIS_20, color='#FFA73E', lw=1.2, ls='--', zorder=3)
    ax.text(YEAR_START + 1, PARIS_15 + 0.04, '1,5 °C', color='#65C06C', fontsize=7, va='bottom')
    ax.text(YEAR_START + 1, PARIS_20 + 0.04, '2,0 °C', color='#FFA73E', fontsize=7, va='bottom')

    # ── SSP trajectory lines ───────────────────────────────────────────────────
    # Draw in order: q15, q85, q50 (so median is on top)
    q_order = []
    if include_low:
        q_order.append('q15')
    if include_high:
        q_order.append('q85')
    q_order.append('q50')

    for ssp_label, ssp_color in SSP_COLORS.items():
        for q_label in q_order:
            temp = results[ssp_label][q_label]
            years = temp.year.values
            mask = (years >= 2026) & (years <= YEAR_END)
            style = QUANTILE_STYLES[q_label]

            # Legend entry only for median line of each SSP. Annual means at mid-year, like IGCC.
            label = SSP_LABELS[ssp_label] if q_label == 'q50' else None
            ax.plot(years[mask] + 0.5, temp.values[mask],
                    color=ssp_color, zorder=5,
                    label=label, **style)

    # ── Impact annotations (right of y-axis, blended transform) ───────────────
    # x is in axes fraction (1.03 = just outside right edge),
    # y is in data coordinates.
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for t_ann, color, text in TEMP_ANNOTATIONS:
        if t_ann > y_max - 0.1:
            continue  # skip annotations outside the y range
        # Horizontal dashed line at thresholds above 2 °C
        if t_ann > PARIS_20:
            ax.axhline(t_ann, color=color, lw=0.8, ls='--', alpha=0.5, zorder=2)
        ax.text(1.03, t_ann, text,
                transform=trans,
                color=color, fontsize=7,
                va='center', ha='left',
                linespacing=1.55,
                clip_on=False)

    # ── ECS range note in legend ───────────────────────────────────────────────
    # Add invisible proxy artists to explain line styles
    if include_low or include_high:
        from matplotlib.lines import Line2D
        proxy_handles = []
        proxy_labels  = []
        if include_low:
            proxy_handles.append(Line2D([], [], color='gray', lw=1.5, ls='--', alpha=0.75))
            proxy_labels.append('Matala ilmastoherkkyys')
        proxy_handles.append(Line2D([], [], color='gray', lw=2.0, ls='-'))
        proxy_labels.append('Keskiverto ilmastoherkkyys')
        if include_high:
            proxy_handles.append(Line2D([], [], color='gray', lw=1.5, ls=':', alpha=0.75))
            proxy_labels.append('Korkea ilmastoherkkyys')

    # ── Axis styling ───────────────────────────────────────────────────────────
    ax.set_xlabel('Vuosi', color='white', fontsize=9)
    ax.set_ylabel('Lämpötila-anomalia (°C)', color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    # ── Legend ─────────────────────────────────────────────────────────────────
    leg = ax.legend(
        loc='upper left',
        fontsize=9,
        facecolor=LEG_COLOR,
        edgecolor=SPINE_COLOR,
        labelcolor='white',
    )
    if include_low or include_high:
        # Add a second legend for line-style meaning
        leg2 = ax.legend(
            handles=proxy_handles,
            labels=proxy_labels,
            loc='lower right',
            fontsize=8,
            facecolor=LEG_COLOR,
            edgecolor=SPINE_COLOR,
            labelcolor='white',
        )
        ax.add_artist(leg)   # keep first legend visible

    return fig


def main():
    print('Loading screened configs …')
    configs = load_screened_configs()

    print(f'  {len(configs)} configs, ECS range: '
          f'{configs["ECS"].min():.2f}–{configs["ECS"].max():.2f} °C\n')

    print('Selecting ECS quantile configs …')
    quantile_configs = _select_quantile_configs(configs, QUANTILES)

    print('Running 6 FaIR simulations (2 SSPs × 3 ECS quantiles) …')
    results = _run_all_simulations(quantile_configs)

    print('Loading observed GMST …')
    obs_gmst = load_observed_gmst()

    # Sanity check: SSP1-1.9 median 2100 temperature
    t119 = results['SSP1-1.9']['q50'].sel(year=2100).item()
    t245 = results['SSP2-4.5']['q50'].sel(year=2100).item()
    print(f'Median ECS warming at 2100:  SSP1-1.9 = {t119:.2f} °C,  SSP2-4.5 = {t245:.2f} °C')
    print('  (expected roughly 1.3–1.6 °C and 2.2–2.8 °C respectively)\n')

    print('Drawing figures …')

    fig1 = _make_figure(results, obs_gmst, include_low=False, include_high=False)
    fig1.savefig('presentation_fig_v1_median.png', dpi=150, bbox_inches='tight',
                 facecolor=BG_COLOR)
    plt.close(fig1)
    print('  Saved presentation_fig_v1_median.png')

    fig2 = _make_figure(results, obs_gmst, include_low=True, include_high=False)
    fig2.savefig('presentation_fig_v2_add_low.png', dpi=150, bbox_inches='tight',
                 facecolor=BG_COLOR)
    plt.close(fig2)
    print('  Saved presentation_fig_v2_add_low.png')

    fig3 = _make_figure(results, obs_gmst, include_low=True, include_high=True)
    fig3.savefig('presentation_fig_v3_all.png', dpi=150, bbox_inches='tight',
                 facecolor=BG_COLOR)
    plt.close(fig3)
    print('  Saved presentation_fig_v3_all.png')

    print('\nDone.')


if __name__ == '__main__':
    main()

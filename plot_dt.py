"""Plot dt generalization from eval_dt_episodes.csv (written by eval_dt.py) to plot_dt.pdf.

Normalized score (transfer - random) / (trained - random) vs frametime, with 95% bootstrap CI.
1 = as good as a policy trained at that dt, 0 = no better than random.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from dt_utils import ROOT, DEFAULT_FRAMETIME

IN_CSV = os.path.join(ROOT, 'eval_dt_episodes.csv')
OUT_PDF = os.path.join(ROOT, 'plot_dt.pdf')
N_BOOT = 2000

STYLE = {  # fixed colour per policy
    'transfer': {'color': '#2a78d6'},
    'trained': {'color': '#eb6834'},
    'random': {'color': '#8c8c86'},
}
INK, MUTED, GRID = '#1a1a19', '#6b6b66', '#e4e3dc'


def bootstrap_score(df, rng):
    rows = []
    for mult, d in df.groupby('mult'):
        samples = {p: d.loc[d.policy == p, 'return_per_step'].to_numpy() for p in STYLE}
        if any(len(s) == 0 for s in samples.values()):
            continue
        boot = {p: rng.choice(s, size=(N_BOOT, len(s))).mean(axis=1) for p, s in samples.items()}
        means = {p: s.mean() for p, s in samples.items()}
        denom = means['trained'] - means['random']
        boot_scores = (boot['transfer'] - boot['random']) / (boot['trained'] - boot['random'])
        rows.append({'mult': mult, 'score': (means['transfer'] - means['random']) / denom,
                     'lo': np.percentile(boot_scores, 2.5), 'hi': np.percentile(boot_scores, 97.5),
                     # the score is meaningless when the trained policy is not clearly better than random,
                     # or when the physics blew up
                     'valid': denom > 0 and np.percentile(boot['trained'] - boot['random'], 2.5) > 0
                              and not d['unstable'].any()})
    return pd.DataFrame(rows)


def style_axis(ax):
    ax.set_xscale('log')
    ticks = [1 / 10, 1 / 5, 1 / 2, 1, 2, 5, 10]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(['1/10', '1/5', '1/2', '1', '2', '5', '10'])
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.axvline(1, color=MUTED, linestyle='--', linewidth=1)


def main():
    score = bootstrap_score(pd.read_csv(IN_CSV), np.random.default_rng(0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(1, color=STYLE['trained']['color'], linewidth=1, linestyle=':')
    ax.axhline(0, color=STYLE['random']['color'], linewidth=1, linestyle=':')
    if not score.empty:
        ok = score[score.valid]
        ax.fill_between(ok['mult'], ok['lo'], ok['hi'], color=STYLE['transfer']['color'], alpha=0.15, linewidth=0)
        ax.plot(ok['mult'], ok['score'], 'o-', color=STYLE['transfer']['color'], linewidth=2, markersize=6)
        for m in score.loc[~score.valid, 'mult']:
            ax.axvspan(m / 1.06, m * 1.06, color=GRID, linewidth=0)
    ax.text(1.0, 1, 'as good as trained at this dt ', transform=ax.get_yaxis_transform(), ha='right', va='bottom', color=MUTED, fontsize=9)
    ax.text(1.0, 0, 'random ', transform=ax.get_yaxis_transform(), ha='right', va='bottom', color=MUTED, fontsize=9)
    ax.text(1, 1.0, ' training frametime', transform=ax.get_xaxis_transform(), color=MUTED, fontsize=9, va='top')
    ax.set_title('HalfCheetah: PPO policy transferred to other frametimes', color=INK, loc='left', fontsize=12)
    ax.set_ylabel('Normalized score\n(transfer − random) / (trained − random)', color=INK)
    ax.set_xlabel(f'Frametime relative to training frametime ({DEFAULT_FRAMETIME:g} s), log scale\n'
                  'grey band: score undefined (unstable physics, or trained-at-dt policy not clearly better than random)',
                  color=INK, fontsize=9)
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print(f'saved {OUT_PDF}')


if __name__ == '__main__':
    main()

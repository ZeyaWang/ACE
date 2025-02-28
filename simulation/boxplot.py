import matplotlib as mpl
from matplotlib import gridspec
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle as pk
from matplotlib.patches import Patch
tasks = ['sim_dense_1.0','sim_dense_2.0','sim_sparse_1.0', 'sim_sparse_2.0']
tasksv = {'sim_dense_1.0': r'Example 2 ($\sigma = 1$)','sim_dense_2.0': 'Example 2 ($\sigma = 2$)',
          'sim_sparse_1.0': r'Example 1 ($\sigma = 1$)', 'sim_sparse_2.0': r'Example 1 ($\sigma = 2$)'}

strategys= ['raw', 'pair', 'pool', 'ace']
strategys_legend= ['Raw score', 'Paired score', 'Pooled score', 'ACE', 'True K']
index = ['Silhouette score (cosine distance)', 'Silhouette score (euclidean distance)', 'Davies-Bouldin index', 'Calinski-Harabasz index']
index2 = ['Silhouette (cosine)', 'Silhouette (euclidean)', 'Davies-Bouldin', 'Calinski-Harabasz']

strategy_colors = {
#'Horovod': "g",
'truth': mpl.colors.cnames["steelblue"],
'raw': '#FFB6C1',
'pair': '#90EE90',
'pool':  'lightgrey',
'ace': '#00BFFF'}
colors = [strategy_colors['raw'], strategy_colors['pair'], strategy_colors['pool'], strategy_colors['ace']]
hatch_styles = ['///', '---', '||', '\\\\\\']
flierprops = dict(marker='o', color='black', alpha=0.5, markersize=1.5)  # Smaller outlier dots
medianprops = dict(color='darkred', linewidth=1.5)  # Bold mean line
meanprops = {
    "marker": "^",  # Shape of the mean dot
    "markerfacecolor": "red",  # Fill color of the mean dot
    "markeredgecolor": "red",  # Edge color of the mean dot
    "markersize": 5
}
for ext in ['nmi', 'acc']:
    with open('box_{}.pkl'.format(ext), 'rb') as op:
        data_load = pk.load(op)
    for task in tasks:
        data_t = data_load[task]
        # Generating sample data for 16 boxplots
        for crit in ['tau', 'corr']:
            data_c = data_t[crit]
            fig, ax = plt.subplots(figsize=(12, 6))
            # Define the positions for the boxplots
            positions = []
            group_gap = 5  # Larger space between groups
            for i in range(4):
                positions.extend([i * group_gap + j for j in range(4)])
            for i, metric in enumerate(['dav', 'ch', 'cosine', 'euclidean']):
                data = data_c[metric]
                start = i * 4
                end = start + 4
                bp = ax.boxplot(data[:, ::-1], positions=positions[start:end], patch_artist=True,  boxprops=dict(linewidth=2, color="black"),
                                widths=0.75, flierprops=flierprops,  medianprops=medianprops, showmeans=True, meanprops=meanprops)  # showmeans=True,Apply mean properties
                for patch, color, hatch in zip(bp['boxes'], colors, hatch_styles):
                    patch.set_facecolor(color)
                    patch.set_hatch(hatch)
                if i != 3:
                    ax.axvline(x= positions[end-1] + 1, color='gray', linestyle='--', linewidth=1)
            #handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
            handles = [
                Patch(facecolor=color, edgecolor='black', hatch=hatch)
                for color, hatch in zip(colors, hatch_styles)
            ]
            #ax.set_ylim(data.min()-0.2, data.max())
            ax.legend(handles, strategys_legend, loc = 'lower right', ncol = 1, fontsize = 12.5)
            plt.setp(ax.get_legend().get_texts(), fontweight='bold')  # Bold the legend text
            # Customizing the plot
            current_ylim = ax.get_ylim()
            new_ylim = (current_ylim[0] - 0.1, current_ylim[1])
            ax.set_ylim(new_ylim)
            ax.set_xticks([(group_gap * i) + 1.5 for i in range(4)])  # Tick at the middle of each group
            ax.set_xticklabels([index2[i] for i in range(len(index2))], fontsize=14, fontweight='bold')
            ax.set_title(tasksv[task], fontsize=16, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_linewidth(2)  # Bold frame
                spine.set_color("black")  # Optional: change color of the frame
            #ax.set_xlabel('Groups')
            if crit == 'tau':
                if ext == 'acc':
                    ax.set_ylabel(r'Kendall rank correlation $\tau_{B}$ (vs. ACC)',fontsize = 12, fontweight='bold')
                else:
                    ax.set_ylabel(r'Kendall rank correlation $\tau_{B}$ (vs. NMI)',fontsize = 12, fontweight='bold')
            else:
                if ext == 'acc':
                    ax.set_ylabel(r"Spearman's rank correlation $r_s$ (vs. ACC)", fontsize=12, fontweight='bold')
                else:
                    ax.set_ylabel(r"Spearman's rank correlation $r_s$ (vs. NMI)",fontsize = 12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            # plt.tight_layout()
            # plt.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.85)
            plt.savefig('{}_{}_{}.pdf'.format(ext, task, crit), transparent=True, bbox_inches='tight', pad_inches=0.05)

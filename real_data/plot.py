#!/usr/bin/env python
import matplotlib as mpl
from matplotlib import gridspec
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle as pk

tasks = ['jule_num', 'DEPICTnum']
task_name = {'jule_num': 'JULE', 'DEPICTnum': 'DEPICT'}


strategys= ['raw', 'pair', 'pool', 'ace']
strategys_legend= ['Raw score', 'Paired score', 'Pooled score', 'ACE', 'True K']
truth = {"YTF": 41, "USPS": 10, "MNIST-test": 10, "FRGC": 20, "UMist": 20, "COIL-20": 20, "COIL-100": 100, "CMU-PIE": 68}
index = ['Silhouette score (cosine distance)', 'Silhouette score (euclidean distance)', 'Davies-Bouldin index', 'Calinski-Harabasz index']
index2 = ['Silhouette (cosine)', 'Silhouette (euclidean)', 'Davies-Bouldin', 'Calinski-Harabasz']

strategy_colors = {
'truth': mpl.colors.cnames["steelblue"],
'raw': '#FFB6C1',
'pair': '#90EE90',
'pool':  'lightgrey',
'ace': '#00BFFF'}

strategy_hatchs = {
'raw': '///',
'pair': '---',
'pool':  '||',
'ace': '\\\\\\'}


for task in tasks:
    if 'jule' in task:
        dataset = ["YTF",  "FRGC", "COIL-20", "COIL-100", "CMU-PIE", "UMist", "USPS", "MNIST-test"]
    else:
        dataset = ["YTF",  "FRGC", "CMU-PIE", "USPS", "MNIST-test"]

    with open('n_{}.pkl'.format(task), 'rb') as op:
        timesets = pk.load(op)
    timesets = {
        inner_key: {outer_key: inner_dict[inner_key] for outer_key, inner_dict in timesets.items()}
        for inner_key in {key for inner_dict in timesets.values() for key in inner_dict}
    }

    for key in dataset:
        timeset = timesets[key]
        ymax = max(value for inner_dict in timeset.values() for value in inner_dict.values())
        ymax2 = max(list(timeset['Calinski-Harabasz index'].values()))
        timeset = {
            inner_key: {outer_key: inner_dict[inner_key] for outer_key, inner_dict in timeset.items()}
            for inner_key in {key for inner_dict in timeset.values() for key in inner_dict}
        }

        if ymax2 == ymax:
            ymax += 50
        else:
            ymax += 30
        ymin = 0
        print(ymin, ymax, key)

        width = 0.15
        fig = plt.figure(figsize=(6.5, 3.5))
        ax = fig.subplots()
        mpl.rc('xtick', labelsize=10)
        mpl.rc('ytick', labelsize=10)
        offset = 0.1
        space = 1*width

        rects = []
        xticks_truth = np.zeros(len(index))
        for n, strategy in enumerate(strategys):
            xticks = []
            yticks = []
            for i, d in enumerate(index):
                xv = offset + i*(len(strategys)*width+space) + n*width
                xticks.append(xv)
                yv = timeset[strategy][d]
                yticks.append(yv)
            rects.append(ax.bar(xticks, yticks,
                width=width, color=strategy_colors[strategy], edgecolor='k', hatch=strategy_hatchs[strategy], zorder=2))

            if (n == 1) or (n == 2):
                xticks_truth+=np.array(xticks)/2 # get the middle of the two ticks

        xticks_truth = xticks_truth.tolist()

        xticks = []
        rects.append(ax.bar(xticks_truth, [truth[key] for _ in index], width=width*4, color='none', linewidth=2, edgecolor='red', zorder=2))
        ax.set_xticks(xticks_truth)

        ax.set_xticklabels([index2[i] for i in range(len(index2))], fontsize=8)
        ax.set_xlim(-0.1, (len(index)*len(strategys)-1)*width + (len(index)-1)*space+offset+0.2)
        ax.set_xlabel('Internal measures', fontsize = 10)

        ax.tick_params(axis=u'x', which=u'both',length=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for ytick in ax.get_yticklabels():
            ytick.set_rotation(90)

        ax.yaxis.set_ticks_position('left')

        if ymax >= 200:
            cticks = [40, 80, 120, 160, 200]
        elif ymax >= 150:
            cticks = [40, 80, 120, 160, 200]
        elif ymax >= 100:
            cticks = [20, 40, 60, 80, 100, 120, 140 ,160]
        elif ymax > 50:
            cticks = [20, 40, 60, 80]
        elif ymax > 20:
            cticks = [10, 20, 30, 40, 50]
        elif ymax > 10:
            cticks = [5, 10 ,15, 20, 25, 30]
        else:
            cticks = [5, 10 ,15, 20]



        ax.yaxis.set_ticks(cticks)
        ax.set_ylim(ymin, ymax+5)
        # # set the grid lines to dotted
        ax.grid(True,alpha=0.7, zorder=1)
        gridlines = ax.get_ygridlines() + ax.get_xgridlines()
        for line in gridlines:
            line.set_linestyle('-.')

        ax.set_ylabel('Number of clusters', fontsize = 10)
        ax.legend(rects, strategys_legend, loc = 'upper right', ncol = 1, fontsize = 8)
        ax.text(offset, 0.92*ymax, '{}:{}'.format(task_name[task], key), fontweight='bold', fontsize = 12)


        plt.savefig('{}_{}.pdf'.format(task, key), transparent = True, bbox_inches = 'tight', pad_inches = 0.05)


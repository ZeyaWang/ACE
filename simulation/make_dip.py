import h5py
import numpy as np
import os, sys
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import argparse


def get_files(directory, substring, suffix):
    fs = [file for file in os.listdir(directory) if substring in file and file.endswith(suffix)]
    return fs

subf = open('submit.py','w')
subf.write('import os\n')
root_path='.'

simtypes = ['sim_dense_1.0','sim_sparse_1.0','sim_dense_2.0','sim_sparse_2.0']

for simtype in simtypes:
    for simseed in range(50):
        eval_dir_root = os.path.join(root_path, simtype)
        eval_dir = os.path.join(eval_dir_root, str(simseed))
        modelFiles = get_files(eval_dir, 'output', 'npz')
        print(modelFiles)
        cmd = 'Rscript clusterable.R {} {}'.format(eval_dir_root, simseed)
        for m in modelFiles:
            cmd += ' {}'.format(os.path.join(eval_dir, m))
        print(cmd)
        respath = os.path.join(eval_dir_root, 'dip_{}.npz'.format(simseed))
        if os.path.exists(respath):
            continue
        job = 'dip{}_{}'.format(simtype, simseed)
        jobName = job + '.sh'
        outf = open(jobName,'w')
        outf.write('#!/bin/bash\n')
        outf.write('\n')
        outf.write('#SBATCH --partition=stats_medium\n')
        outf.write('#SBATCH --nodes=1 --mem=16G --time=24:00:00\n')
        outf.write('#SBATCH --ntasks=1\n')
        outf.write('#SBATCH --cpus-per-task=2\n')
        outf.write('#SBATCH --output=slurm-%A.%a.out\n')
        outf.write('#SBATCH --error=slurm-%A.%a.err\n')
        outf.write('#SBATCH --mail-type=ALL\n')
        outf.write('\n')
        outf.write('module load R/4.2.1\n')
        outf.write(cmd)
        outf.close()
        subf.write('os.system("sbatch %s")\n' % jobName)
subf.close()



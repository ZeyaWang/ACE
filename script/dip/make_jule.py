import h5py
import numpy as np
import os, sys
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import argparse


def get_files_with_substring_and_suffix(directory, substring, suffix):
    files = []
    all_files = os.listdir(directory)
    files = [file for file in all_files if substring in file and file.endswith(suffix)]
    return files


subf = open('submit.py','w')
subf.write('import os\n')

datasets = ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
for eval_data in datasets:
    modelFiles = get_files_with_substring_and_suffix('DEPICT_hyper', 'output'+eval_data, 'npz')
    #modelFiles = get_files_with_substring_and_suffix('DEPICT_num', 'output'+eval_data, 'npz')
    print(modelFiles)
    #modelFiles = [m[5:-4] for m in modelFiles]
    cmd = 'Rscript clusterable_DEPICT.R {}'.format(eval_data)
    for m in modelFiles:
        cmd += ' {}'.format(m)
    # print(cmd)
    # os.system(cmd)
    # while not os.path.isfile('dip_{}.npz'.format(eval_data)):
    #     time.sleep(0.1)
    job = 'dip{}'.format(eval_data)
    jobName = job + '.sh'
    outf = open(jobName,'w')
    outf.write('#!/bin/bash\n')
    outf.write('\n')
    outf.write('#SBATCH --partition=stats_medium\n') 
    outf.write('#SBATCH --nodes=1 --mem=32G --time=24:00:00\n')
    outf.write('#SBATCH --ntasks=1\n')
    outf.write('#SBATCH --cpus-per-task=2\n')
    outf.write('#SBATCH --output=slurm-%A.%a.out\n')
    outf.write('#SBATCH --error=slurm-%A.%a.err\n')
    outf.write('#SBATCH --mail-type=ALL\n')
    outf.write('\n')
    outf.write('module load R/4.1.2\n')
    outf.write(cmd)
    outf.close()
    subf.write('os.system("sbatch %s")\n' % jobName)
subf.close()



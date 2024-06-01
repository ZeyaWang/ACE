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
    # Use os.listdir to get a list of all files in the directory
    all_files = os.listdir(directory)
    
    # Use a list comprehension to filter files based on the substring and suffix
    files = [file for file in all_files if substring in file and file.endswith(suffix)]
    
    return files


subf = open('submit.py','w')
subf.write('import os\n')

cmd = 'Rscript clusterable.R dcl'

for ii in range(4, 100, 5):
    file = os.path.join('npfiles_val', 'pro_output_{}.npz'.format(ii))
    cmd += ' {}'.format(file)

job = 'dip_dcl'
jobName = job + '.sh'
outf = open(jobName,'w')
outf.write('#!/bin/bash\n')
outf.write('\n')
outf.write('#SBATCH --partition=stats_medium\n') 
outf.write('#SBATCH --nodes=1 --mem=32G --time=24:00:00\n')
#outf.write('#SBATCH --gpus-per-node=rtxa5500:2\n')
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



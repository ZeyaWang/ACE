import h5py
import numpy as np
import os, sys
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import argparse

datasets_jule = ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
datasets_depict = ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
datasets_all = {
    'JULE_hyper': datasets_jule ,
    'JULE_num': datasets_jule ,
    'DEPICT_hyper': datasets_depict,
    'DEPICT_num': datasets_depict,
}


subf = open('submit.py','w')
subf.write('import os\n')


for task in datasets_all.keys():
    datasets = datasets_all[task]
    for dataset in datasets:
        job = 'gettruth_{}_{}'.format(task, dataset)
        jobName=job + '.sh'
        outf = open(jobName,'w')
        outf.write('#!/bin/bash\n')
        outf.write('\n')
        outf.write('#SBATCH --partition=stats_short\n') 
        outf.write('#SBATCH --nodes=1 --mem=2G --time=1:00:00\n')
        outf.write('#SBATCH --ntasks=1\n')
        outf.write('#SBATCH --cpus-per-task=2\n')
        outf.write('#SBATCH --output=slurm-%A.%a.out\n')
        outf.write('#SBATCH --error=slurm-%A.%a.err\n')
        outf.write('#SBATCH --mail-type=ALL\n')
        outf.write('\n')
        outf.write('conda info --envs\n')
        outf.write('eval $(conda shell.bash hook)\n')
        outf.write('source ~/miniconda/etc/profile.d/conda.sh\n')
        outf.write('conda activate dcl\n')
        outf.write('python get_truth.py --dataset {} --task {} \n'.format(dataset, task))
        outf.close()
        subff.write('os.system("sbatch %s")\n' % jobName)
subf.close()



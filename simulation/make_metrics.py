import sys
import os
import pickle as pk


subff = open('submit.py','w')
subff.write('import os\n')
root_path='.'



simtypes = ['sim_convex_dense_1.0','sim_convex_sparse_1.0','sim_convex_dense_2.0','sim_convex_sparse_2.0']

for metric in ['dav', 'ch', 'euclidean', 'cosine']:
    for simtype in simtypes:
        for simseed in range(50):
            eval_dir_root = os.path.join(root_path, simtype)
            eval_dir = os.path.join(eval_dir_root, str(simseed))
            tput1 = os.path.join(eval_dir,'raw_metric', 'merge_{}_score.pkl'.format(metric))
            tput2 = os.path.join(eval_dir,'embedded_metric', 'merge_{}_score.pkl'.format(metric))
            if ((os.path.exists(tput1)) and (os.path.exists(tput2))):
                continue
            cmd = 'python calculate_metric.py --modelpath {} --metric {} \n'.format(eval_dir, metric)
            print(cmd)
            job = 'mergesplit_{}_{}_{}'.format(simtype, simseed, metric)
            jobName=job + '.sh'
            outf = open(jobName,'w')
            outf.write('#!/bin/bash\n')
            outf.write('\n')
            if 'dense' in simtype:
                outf.write('#SBATCH --partition=stats_short\n')
            else:
                outf.write('#SBATCH --partition=stats_medium\n')
            if 'dense' in simtype:
                outf.write('#SBATCH --nodes=1 --mem=16G --time=1:00:00\n')
            else:
                outf.write('#SBATCH --nodes=1 --mem=16G --time=24:00:00\n')
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
            outf.write(cmd)
            outf.close()
            subff.write('os.system("sbatch %s")\n' % jobName)


subff.close()

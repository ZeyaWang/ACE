import sys
import os
import pickle as pk


subff = open('submit.py','w')
subff.write('import os\n')
options = ['dense','sparse']
root_path='.'

stds = [1, 2]

for option in options:
    for std in stds:
        for seed in range(50):
            cmd = 'python sim.py --option {} --cluster_std {} --seed {}'.format(option, std, seed)
            print(cmd)
            job = 'sim_{}_{}_{}'.format(option, std, seed)
            jobName=job + '.sh'
            outf = open(jobName,'w')
            outf.write('#!/bin/bash\n')
            outf.write('\n')
            if 'dense' in options:
                outf.write('#SBATCH --partition=stats_long\n')
            else:
                outf.write('#SBATCH --partition=stats_medium\n')
            if 'dense' in options:
                outf.write('#SBATCH --nodes=1 --mem=16G --time=168:00:00\n')
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


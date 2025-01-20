import sys
import os
import pickle as pk

subff = open('submit.py','w')
subff.write('import os\n')

modelpath = {
    'jule': 'JULE_hyper',
    'julenum': 'JULE_num',
    'DEPICT': 'DEPICT_hyper',
    'DEPICTnum': 'DEPICT_num',
}


datasets_jule = ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
datasets_depict = ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
datasets_all = {
    'jule': datasets_jule ,
    'julenum': datasets_jule ,
    'DEPICT': datasets_depict,
    'DEPICTnum': datasets_depict,
}


step = 1

# step one
if step == 1:
    for task in datasets_all.keys():
        datasets = datasets_all[task]
        for metric in ['dav', 'ch', 'euclidean', 'cosine']:
            for dataset in datasets:
                job = 'raw_{}_{}_{}'.format(task, dataset, metric)
                jobName=job + '.sh'
                outf = open(jobName,'w')
                outf.write('#!/bin/bash\n')
                outf.write('\n')
                if dataset == 'COIL-10':
                    outf.write('#SBATCH --partition=stats_long\n') 
                else:
                    outf.write('#SBATCH --partition=stats_medium\n') 
                if dataset == 'COIL-10':
                    outf.write('#SBATCH --nodes=1 --mem=64G --time=24:00:00\n')
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
                outf.write('python3 get_raw.py --dataset {} --metric {} --task {} \n'.format(dataset, metric, task))
                outf.close()
                #if dataset in ['COIL-20', 'UMist']:
                ofile = os.path.join(task,'raw_metric', 'merge_{}_{}_score.pkl'.format(dataset, metric))
                #print(ofile)
                #if not os.path.isfile(ofile):
                subff.write('os.system("sbatch %s")\n' % jobName)
    subff.close()
elif step == 2:
    # step 2
    subf = open('submit2.py','w')
    subf.write('import os\n')
    for task in datasets_all.keys():
        #subf = open('submit{}.py'.format(task),'w')
        datasets = datasets_all[task]
        for dataset in datasets:
            with open(os.path.join('file_list', task, "{}.txt".format(dataset)), "r") as file:
                modelFiles = [line.strip() for line in file.readlines()]
            for m1 in modelFiles:
                ff = '{}/raw_tmp/rr_{}.npz'.format(task, m1)
                if not os.path.isfile(ff):
                    job = 'Rmerge_{}_{}_{}'.format(task, dataset, m1)
                    jobName=job + '.sh'
                    outf = open(jobName,'w')
                    outf.write('#!/bin/bash\n')
                    outf.write('\n')
                    outf.write('#SBATCH --partition=stats_medium\n')
                    outf.write('#SBATCH --nodes=1 --mem=32G --time=24:00:00\n')
                    outf.write('#SBATCH --ntasks=1\n')
                    outf.write('#SBATCH --cpus-per-task=1\n')
                    outf.write('#SBATCH --output=slurm-%A.%a.out\n')
                    outf.write('#SBATCH --error=slurm-%A.%a.err\n')
                    outf.write('#SBATCH --mail-type=ALL\n')
                    outf.write('\n')
                    outf.write('module load R/4.1.2\n')
                    outf.write('Rscript getraw.r {} {}\n'.format(task, m1))
                    outf.close()
                    if dataset != 'COIL-100':
                        subf.write('os.system("sbatch %s")\n' % jobName)
    subf.close()

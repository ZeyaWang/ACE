import os,sys
import time

datasets = ['USPS', 'MNIST-test', 'FRGC', 'YTF', 'UMist'] #
face = ['FRGC', 'UMist', 'YTF']

for dataset in datasets:
  if dataset == 'YTF':
    numcls = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
  elif dataset == 'USPS':
    numcls = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
  elif dataset == 'MNIST-test':
    numcls = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
  elif dataset == 'UMist':
    numcls = [10, 15, 20, 25, 30, 35, 40, 45, 50]
  elif dataset == 'FRGC':
    numcls = [10, 15, 20, 25, 30, 35, 40, 45, 50]


  for numcl in numcls:
    try:
        cmd = 'CUDA_VISIBLE_DEVICES=0 python DEPICT_num.py --dataset={} --num_clusters={}'.format(dataset, numcl)
        print(cmd)
        os.system(cmd)
        while not os.path.isfile('done.o'):
            time.sleep(0.1)
        os.system('rm done.o')
    except:
        print('continue')
        continue

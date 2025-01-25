import os,sys
import time

datasets = ['COIL-100', 'COIL-20', 'MNIST-test', 'USPS', 'CMU-PIE', 'FRGC',  'UMist', 'YTF']
face = ['FRGC', 'UMist', 'YTF']

for dataset in datasets:
  if dataset == 'YTF':
    numcls = [11, 15, 20, 25, 30, 35, 40, 45, 50] # 11 is the least K without an error
  elif (dataset == 'USPS') or (dataset == 'MNIST-test') or (dataset == 'UMist') or (dataset == 'FRGC') or (dataset == 'COIL-20'):
    numcls = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
  elif dataset == 'CMU-PIE':
    numcls = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  else:
    numcls = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

  for numcl in numcls:
    try:
        if dataset in face:
          eta = 0.2
        else:
          eta = 0.9
        cmd = 'CUDA_VISIBLE_DEVICES=0 th train_num.lua -dataset {} -eta {} -numcl {} -use_fast 0'.format(dataset, eta, numcl)
        print(cmd)
        os.system(cmd)
        while not os.path.isfile('done.o'):
            time.sleep(0.1)
        os.system('rm done.o')
    except:
        print('continue')
        continue

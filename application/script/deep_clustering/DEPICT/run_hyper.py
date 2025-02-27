import os,sys
import time

datasets = ['USPS', 'MNIST-test', 'FRGC', 'YTF', 'CMU-PIE'] 

lrs = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
rhs = [0.1, 1.0, 10.0]

for dataset in datasets:
  for lr in lrs:
    for rh in rhs:
      outputfile = "outputFRGC_20_{}_{}_1.0.npz".format(lr, rh)
      if not os.path.isfile(outputfile):
        try:
            cmd = 'python DEPICT_hyper.py --dataset={} --learning_rate={} --reconstruct_hyperparam={}  > {}_{}_{}.log'.format(dataset, lr, rh)
            print(cmd)
            os.system(cmd)
            while not os.path.isfile('done.o'):
                time.sleep(0.1)
            os.system('rm done.o')
        except:
            print('continue')
            continu
      else:
        print('pass', outputfile)

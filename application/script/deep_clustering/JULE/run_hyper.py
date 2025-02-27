import os,sys
import time



for eta in [0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]:
    for lr in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        for m in ['COIL-100', 'COIL-20', 'MNIST-test', 'USPS', 'CMU-PIE', 'FRGC',  'UMist', 'YTF']:
            cmd = 'CUDA_VISIBLE_DEVICES=0 th train_hyper.lua -dataset {} -eta {} -learningRate {}'.format(m, eta, lr)
            print(cmd)
            os.system(cmd)
            while not os.path.isfile('done.o'):
                time.sleep(0.1)
            os.system('rm done.o')

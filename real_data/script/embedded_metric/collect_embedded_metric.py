import numpy as np
import sklearn
from sklearn import metrics
import os, h5py
import pickle as pk
from collections import defaultdict
import argparse
from numpy import linalg as LA
import math


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', default='USPS')
    #parser.add_argument('--metric', default='euclidean')

    args = parser.parse_args()
    #eval_data = args.dataset

    modelpath = {
        'jule': 'JULE_hyper',
        'julenum': 'JULE_num',
        'DEPICT': 'DEPICT_hyper',
        'DEPICTnum': 'DEPICT_num',
    }
    rootpath = {
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


    def dd():
        return defaultdict(dict)


    metric_list = ['ccc','dunn','cind','db','sdbw', 'ccdbw']



    for task in datasets_all.keys():
        datasets = datasets_all[task]

        for eval_data in datasets:
            scored = defaultdict(dd)
            with open(os.path.join('file_list', task, "{}.txt".format(eval_data)), "r") as file:
                modelFiles = [line.strip() for line in file.readlines()]
            for m in modelFiles:
                for key in modelFiles:
                    
                    file = "{}/tmp/rr_{}_{}.npz".format(task, m, key)
                    data = np.load(file)
                    for metric in metric_list:
                        value = data[metric]
                        if value == True:
                            print('cv',value)
                            value = np.nan
                        if value == -2147483648:
                            print('cv',value)
                            value = np.nan
                        if math.isinf(value):
                            print('cv',value)
                            value = np.nan
                        
                        scored[metric][m][key] = value 

            for metric in metric_list:
                os.makedirs(os.path.join(task, 'embedded_metric'), exist_ok=True)
                with open('{}/embedded_metric/merge_other_{}_{}_score.pkl'.format(task, eval_data, metric), 'wb') as file:
                    pk.dump(scored[metric], file)

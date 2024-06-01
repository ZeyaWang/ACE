import numpy as np
import sklearn
from sklearn import metrics
import os, h5py
import pickle as pk
from collections import defaultdict
import argparse
from numpy import linalg as LA
import math


def get_files_with_substring_and_suffix(directory, substring, suffix):
    files = []
    # Use os.listdir to get a list of all files in the directory
    all_files = os.listdir(directory)

    # Use a list comprehension to filter files based on the substring and suffix
    files = [file for file in all_files if substring in file and file.endswith(suffix)]

    return files

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', default='USPS')
    #parser.add_argument('--metric', default='euclidean')

    args = parser.parse_args()
    #eval_data = args.dataset

    modelpath = {
        'jule': '/home/zwa281/unsupeval/jule',
        'julenum': '/home/zwa281/unsupeval_num/julenum',
        'DEPICT': '/home/zwa281/DEPICT',
        'DEPICTnum': '/home/zwa281/DEPICTnum',
    }
    rootpath = {
        'jule': '/home/zwa281/unsupeval',
        'julenum': '/home/zwa281/unsupeval_num',
        'DEPICT': '/home/zwa281/DEPICT',
        'DEPICTnum': '/home/zwa281/DEPICTnum',
    }

    datasets_jule = ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
    #datasets_jule = ['USPS',  'COIL-20',  'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']

    datasets_depict = ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
    datasets_all = {
        'jule': datasets_jule ,
        'julenum': datasets_jule ,
        'DEPICT': datasets_depict,
        'DEPICTnum': datasets_depict,
    }





    metric_list = ['ccc','dunn','cind','db','sdbw', 'ccdbw']

    #min_group = ['cind', 'cind2', 'db', 'sdbw']


    for task in datasets_all.keys():
        datasets = datasets_all[task]

        for eval_data in datasets:
            scored = defaultdict(dict)
            if 'jule' in task:
                modelFiles = get_files_with_substring_and_suffix(modelpath[task], 'feature'+eval_data, 'h5')
                modelFiles = [m[7:-3] for m in modelFiles]
            else:
                modelFiles = get_files_with_substring_and_suffix(modelpath[task], 'output'+eval_data, 'npz')


            if eval_data not in ['COIL-100', 'UMist']:
                for key in modelFiles:
                    
                    file = "{}/raw_tmp/rr_{}.npz".format(task, key)
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
                        
                        scored[metric][key] = value 
            else:
                file = "{}/raw_metric/r_{}.npz".format(task, eval_data)
                data = np.load(file,allow_pickle=True)
                #print(task,eval_data,list(data.keys()))
                models = data['models'][1:]
                for metric in metric_list:
                    if not metric in data.keys():
                        value = None
                    else:
                        if metric != 'ccdbw':
                            value = data['r{}'.format(metric)].tolist()
                        else:
                            value = data['r{}'.format('cdbw')].tolist()
                    #print(value.tolist())
                    #print(value, isinstance(value, np.ndarray))
                    #if (not isinstance(value, np.ndarray)) and value == None:
                    #print(value)
                    if value == None:
                        #print('---')
                        value = [0 for _ in range(len(models))]
                    if metric in ['cind', 'db', 'sdbw']:
                        value = [- v for v in value]
                    rv = dict(zip(models, value))
                    scored[metric] = rv

            # for metric in metric_list:
            #     with open('{}/raw_metric/merge_other_{}_{}_score.pkl'.format(task, eval_data, metric), 'wb') as file:
            #         pk.dump(scored[metric], file)

            for metric in ['dav', 'ch', 'euclidean', 'cosine']:
                with open('{}/raw_metric/merge_{}_{}_score.pkl'.format(task, eval_data, metric), 'rb') as file:
                    #pk.dump(scored[metric], file)
                    scored2 = pk.load(file)
                for key, value in scored2.items():
                    print(len(value))
                    scored[key] = value

            #print(scored)
            with open('{}/raw_metric/merge_all_{}_score.pkl'.format(task, eval_data), 'wb') as file:
                pk.dump(scored, file)

            

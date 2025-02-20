import os,sys
import pandas as pd
import math
# the script is for generating the tables in  the main paper
import numpy as np
import pickle as pk
critd = {'tau': '${}tau_B$'.format(chr(92)), 'cor': '$r_s$'}
import re
def strip_bracket(x):
    if isinstance(x, str):
        x = x.split(' ')[0]
        if x in ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']:
            return x
        else:
            return float(x)
    else:
        return x

def cut_digit(x):
    if isinstance(x, str):
        xx = x.split(' ')
        if len(xx) > 1:
            x0 = xx[0]
            x1 = xx[1]
            if x0 in ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']:
                return x
            else:
                #print(x0,x1)
                return ' '.join([str(round(float(x0), 2)), str(x1)])
    else:
        #return round(x, 2)
        return x


def convert_float_num(x):
    if isinstance(x, str):
        if x != 'N/A':
            xx = x.split(' ')
            if len(xx) > 1:
                x0 = xx[0]
                x1 = xx[1]
                if x0 in ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']:
                    return x
                else:
                    return float(x0)
            else:
                return float(xx[0])
        else:
            return np.nan
    else:
        return x

def get_cluster_num(x):
    if isinstance(x, str):
        if x != 'N/A':
            xx = x.split(' ')
            if len(xx) > 1:
                x0 = xx[0]
                x1 = xx[1]
                if x0 in ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']:
                    return x0
                else:
                    return int(x1[1:-1])
            else:
                return np.nan
        else:
            return np.nan
    else:
        return np.nan



def convert_float(x):
    if isinstance(x, str):
        if x != 'N/A':
            if x in ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']:
                return x
            else:
                return float(x)
        else:
            return np.nan
    else:
        return x

def create_subcol(colnames):
    prefix = [(c.split('=')[1], critd[c.split('=')[0]]) for c in colnames if c != 'dataset']
    #prefix = sorted(prefix,  key=lambda x: (x[0], prefix.index(x)))
    return prefix

def short_name(df, ext):
    idnames = {}
    for c in df.index.tolist():
        if ext in c:
            cc=c.split('vs {}'.format(ext))
            cc = ''.join(cc).rstrip()
        # else:
        #     cc = c
            idnames[c]=cc
    return  idnames

def transpose(df):
    transposed_df = df.T
    transposed_df.columns = transposed_df.iloc[0]
    # Drop the first row (original column headers)
    transposed_df = transposed_df.drop(transposed_df.index[0])
    return transposed_df

def row_max_indices(row):
    max_val = row.max()  # Find the max value
    return row[row == max_val].index.tolist()  # Return all column indices where the value equals the max

root_dir = '.'


maps = {'jule_hyper': '\\emph{JULE}',
        'jule_num': '\\emph{JULE}',
        'DEPICT': '\\emph{DEPICT}',
        'DEPICTnum': '\\emph{DEPICT}',
        'dav': 'Davies-Bouldin index',
        'ch': 'Calinski-Harabasz index',
        'cosine': 'Silhouette score (cosine distance)',
        'euclidean': 'Silhouette score (euclidean distance)',
        'ccc': 'Cubic clustering criterion',
        'dunn': 'Dunn index',
        'cind': 'Cindex',
        'sdbw': 'SDbw index',
        'ccdbw': 'CDbw index'}




import argparse
parser = argparse.ArgumentParser(description='Generate table',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ext', type=str, default='nmi', help='nmi or acc')
parser.add_argument('--exp', type=str, default='hyper', help='hyper or num')

args = parser.parse_args()

bracket_turn = False

tasks = ['jule_hyper','jule_num','DEPICT','DEPICTnum']
crits = ['cor', 'tau']
compared = []
compared.append('eval')

dcols = {'nmi': ['ACE vs nmi', 'pooled score vs nmi',
       'pooled score vs nmi (w/o dip)', 'paired score vs nmi',
       'raw score vs nmi'],
       'acc': ['ACE vs acc', 'pooled score vs acc',
       'pooled score vs acc (w/o dip)', 'paired score vs acc',
       'raw score vs acc']}

# exts = ['nmi', 'acc']
# ext = exts[1] # acc
# ext = exts[0] # nmi

ext = args.ext
metrics = ['ch', 'dav', 'cosine', 'euclidean']

#tasks = ['jule_hyper','jule_num','DEPICT','DEPICTnum']

if args.exp == 'hyper':
    tasks = ['jule_hyper','DEPICT']
else:
    tasks = ['jule_num','DEPICTnum']

iii = 0
outputs = []

for task in tasks:
    ncluster, ncluster1 = {}, {}
    cols = dcols[ext]
    cp_cols = [cols[0]]  
    ref_cols = [cols[1],cols[3], cols[4]]
    for metric in metrics:
        for i, ftb in enumerate(compared):
            ftb = os.path.join(root_dir, ftb)
            ftb = os.path.join(ftb, 'results_{}'.format(task))
            suffix1 = '{}_{}.csv'.format(metric, crits[0])
            suffix2 = '{}_{}.csv'.format(metric, crits[1])

            ftb1 = os.path.join(ftb, suffix1)
            ftb2 = os.path.join(ftb, suffix2)

            tb1 = pd.read_csv(ftb1)
            tb2 = pd.read_csv(ftb2)
            if 'num' in task:
                tb1 = tb1.applymap(cut_digit)
                tb2 = tb2.applymap(cut_digit)
                tb11 = tb1.applymap(strip_bracket)
                tb22 = tb2.applymap(strip_bracket)
                avg1 = ['Average']+tb11.drop(columns='dataset').mean().to_list()
                avg2 = ['Average']+tb22.drop(columns='dataset').mean().to_list()
            else:
                avg1 = ['Average']+tb1.drop(columns='dataset').mean().to_list()
                avg2 = ['Average']+tb2.drop(columns='dataset').mean().to_list()
            tb1.loc[len(tb1)] = avg1
            tb2.loc[len(tb2)] = avg2
            if 'DEPICT' in task:
                if 'num' in task:
                    addnm = ['UMist (20)', 'COIL-20 (20)', 'COIL-100 (100)']
                else:
                    addnm = ['UMist', 'COIL-20', 'COIL-100']
                for anm in addnm:
                    tbadd1 = pd.Series([anm]+['N/A'] * (len(tb1.columns)-1), index=tb1.columns)
                    tb1.loc[len(tb1)] = tbadd1
                    tbadd2 = pd.Series([anm]+['N/A'] * (len(tb2.columns)-1), index=tb2.columns)
                    tb2.loc[len(tb2)] = tbadd2
            tb1 = tb1[['dataset']+cols]
            tb2 = tb2[['dataset']+cols]

            if i == 0:
                tb1 = tb1[['dataset'] + cp_cols + ref_cols]
                tb2 = tb2[['dataset'] + cp_cols + ref_cols]
            else:
                tb1 = tb1[['dataset'] + cp_cols]
                tb2 = tb2[['dataset'] + cp_cols]
            
            tb1 = transpose(tb1)
            tb2 = transpose(tb2)

            tb1 = tb1.rename(columns={c: '{}={}'.format(crits[0], c) for c in tb1.columns})
            tb2 = tb2.rename(columns={c: '{}={}'.format(crits[1], c) for c in tb2.columns})

            tb = pd.merge(tb1, tb2, left_index=True, right_index=True)
            tb.index = tb.index.map(short_name(tb,ext))
            if i == 0:
                result_df = tb
            else:
                result_df = pd.concat([result_df, tb.rename(index=lambda x: f"{x}_{i}")], axis=0)

        result_df.index = result_df.index.str.replace('_', ' ') # for horizontal table
        subcols = create_subcol(result_df.columns.tolist())
        result_df.columns = pd.MultiIndex.from_tuples(subcols)
        result_df = result_df.sort_index(axis=1, level=0, ascending=False)

        if len(compared) == 1:
            result_df = result_df.reindex(['transformed score', 'mean score', 'embedding score', 'raw score'][::-1])
        else:
            result_df = result_df.reindex(['transformed score', 'transformed score 1', 'embedding score', 'raw score'][::-1])

        if 'num' not in task:
            desired_order = ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE', 'UMist', 'COIL-20', 'COIL-100', 'Average']
        else:
            desired_order = ['USPS (10)',  'YTF (41)', 'FRGC (20)', 'MNIST-test (10)', 'CMU-PIE (68)', 'UMist (20)', 'COIL-20 (20)', 'COIL-100 (100)', 'Average']

        result_df = result_df.reindex(columns=desired_order, level=0)
        result_df = result_df.reindex(columns=['$r_s$','$\\tau_B$'], level=1)

        print(metric)

        result_df = result_df.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        if ('num' in task) and (ext == 'nmi'):
            back_df = result_df.applymap(convert_float_num)
            num_df = result_df.applymap(get_cluster_num)
            ncluster[maps[metric]] = {}
            ncluster1[maps[metric]] = {}
            for n in num_df.columns.levels[0]:
                ncluster[maps[metric]][n.split(' ')[0]] = num_df[(n, '$r_s$')].to_numpy()
                ncluster1[maps[metric]][n.split(' ')[0]] = dict(zip(['raw', 'pair', 'pool', 'ace'], num_df[(n, '$r_s$')].to_numpy().tolist()))
            print(ncluster1)

        else:
            back_df = result_df.applymap(convert_float)
        max_indices = back_df.idxmax()

        for (level_1, level_2), value in max_indices.items():
            if not pd.isna(value):
                original_value = result_df.loc[value, (level_1, level_2)]
                tmp_s = result_df.loc[:,(level_1, level_2)]
                index_of_value = tmp_s[tmp_s == original_value].index.tolist()
                for val in index_of_value:
                    result_df.loc[val, (level_1, level_2)] = '\\textbf{' + result_df.loc[val, (level_1, level_2)] + '}'
        output = result_df.to_latex(index=True,
                           escape=False,
                           formatters={"name": str.upper},
                           float_format="{:.2f}".format)
        if iii == 0:
            start = output.split('\\midrule\n')[0] + "\\midrule\n"

        end = '\\bottomrule\n' + output.split('\\bottomrule\n')[1]
        output = output.split('\\midrule\n')[1].split('\\bottomrule\n')[0]

        outputs.append('\\hline \n \\multicolumn{{19}}{{c}}{{{}: {}}} \\\\\n \\hline \n'.format(maps[task], maps[metric]))
        outputs.append(output)
        iii = iii + 1

    if ('num' in task) and (ext == 'nmi'):
        with open('n_{}.pkl'.format(task), 'wb') as op:
            pk.dump(ncluster1, op)
        ncluster = pd.DataFrame(ncluster).T  # Transpose to get rows as the original keys
        ncluster.to_csv('n_{}.csv'.format(task), index=True)

outputs = start + ''.join(outputs) + end
outputs = outputs.replace('raw score', 'Raw score')
outputs = outputs.replace('embedding score', 'Paired score')
outputs = outputs.replace('mean score', 'Pooled score')
outputs = outputs.replace('transformed score', '\\textbf{ACE}')
outputs = outputs.replace('N/A', '')
if not bracket_turn:
    outputs = re.sub(r'\s\(.*?\)', '', outputs)
print(outputs)
print('-----------------------------------------------------------------------')
import h5py
import numpy as np
import pandas as pd
from utils import *
import pickle as pk

critd = {'tau': '${}tau_B$'.format(chr(92)), 'cor': '$r_s$'}
def transpose(df):
    transposed_df = df.T
    transposed_df.columns = transposed_df.iloc[0]
    # Drop the first row (original column headers)
    transposed_df = transposed_df.drop(transposed_df.index[0])
    return transposed_df

def create_subcol(colnames, vertical):
    if vertical:
        prefix = [(c.split('=')[1], critd[c.split('=')[0]]) for c in colnames if c != 'metric']
    else:
        prefix = [('metric','')]+[(c.split('=')[1], critd[c.split('=')[0]]) for c in colnames if c != 'metric']
    return prefix

if __name__ == '__main__':
    l = 0
    num = 20

    nmi_tau, acc_tau, nmi_cor, acc_cor = [], [], [], [] 

    for metric in ['cosine', 'euclidean', 'dav', 'ch']:
        l = l + 1
        nmv = []
        acv = []
        modelFiles = []
        pair_scores = np.zeros((num, num))
        for i in range(num):
            fname = 'saved_{}_{}.pkl'.format(i*5+4, metric)
            with open(fname, 'rb') as f:
                silds, nm, ac = pk.load(f)
            for j in range(num):
                kname = '{}'.format(j*5+4)
                pair_scores[i,j] = silds[kname]
            nmv.append(np.round(nm,3))
            acv.append(np.round(ac,3))
            modelFiles.append('model{}'.format(i))

        print(modelFiles)
        nmv = dict(zip(modelFiles, nmv))
        acv = dict(zip(modelFiles, acv))

        sv = np.diag(pair_scores)
        sv = dict(zip(modelFiles, sv))

        nmvd, _, nmv = sort_and_match(nmv, modelFiles)
        acvd, _, acv = sort_and_match(acv, modelFiles)
        svd, _, sv = sort_and_match(sv, modelFiles)

        label_score = eval_pool(modelFiles, pair_scores)

        lsd, _, label_score = sort_and_match(label_score, modelFiles)
        st_score, graph, outliers, labels, best_n, prv, labels_initial = eval_ace(modelFiles, pair_scores, modelFiles, 0.05,
                                                                        0.1, 'hdbscan', 'pr')

        print('--------------------------------------------------')

        std, _, st_score = sort_and_match(st_score, modelFiles)


        print(nmvd)
        print(acvd)
        print(svd)
        print(lsd)
        print(std)
        
        tau_label, _ = kendalltau(label_score, nmv)
        tau_st, _ = kendalltau(st_score, nmv)
        tau_sv, _ = kendalltau(sv, nmv)
        cor_label, _ = spearmanr(label_score, nmv)
        cor_st, _ = spearmanr(st_score, nmv)
        cor_sv, _ = spearmanr(sv, nmv)
        nmi_tau.append([metric, tau_st,tau_label, tau_sv])
        nmi_cor.append([metric, cor_st, cor_label, cor_sv])

        tau_label, _ = kendalltau(label_score, acv)
        tau_st, _ = kendalltau(st_score, acv)
        tau_sv, _ = kendalltau(sv, acv)
        cor_label, _ = spearmanr(label_score, acv)
        cor_st, _ = spearmanr(st_score, acv)
        cor_sv, _ = spearmanr(sv, acv)

        acc_tau.append([metric, tau_st,tau_label, tau_sv])
        acc_cor.append([metric, cor_st, cor_label, cor_sv])


    vertical = True

    
    lname = ['metric',  'proposed score', 'label score', 'space score']
    mname = ['cosine', 'euclidean', 'dav', 'ch']
    if vertical:
        cols = ['cosine', 'euclidean', 'dav', 'ch']
    else:
        cols = ['proposed score', 'label score', 'space score']

    df_nmi_tau = pd.DataFrame(columns=lname, data=nmi_tau).round(2)
    df_nmi_cor = pd.DataFrame(columns=lname, data=nmi_cor).round(2)
    df_acc_tau = pd.DataFrame(columns=lname, data=acc_tau).round(2)
    df_acc_cor = pd.DataFrame(columns=lname, data=acc_cor).round(2)

    #print(df_nmi_cor)

    if vertical:
        df_nmi_tau = transpose(df_nmi_tau)
        df_nmi_cor = transpose(df_nmi_cor)
        df_acc_tau = transpose(df_acc_tau)
        df_acc_cor = transpose(df_acc_cor)

    #print(df_nmi_cor)

    df_nmi_tau = df_nmi_tau.rename(columns={c: 'tau={}'.format(c) for c in df_nmi_tau.columns if c in cols})
    df_nmi_cor = df_nmi_cor.rename(columns={c: 'cor={}'.format(c) for c in df_nmi_cor.columns if c in cols})
    df_acc_tau = df_acc_tau.rename(columns={c: 'tau={}'.format(c) for c in df_acc_tau.columns if c in cols})
    df_acc_cor = df_acc_cor.rename(columns={c: 'cor={}'.format(c) for c in df_acc_cor.columns if c in cols})
    #print(df_nmi_cor)

    if not vertical:
        df_nmi = pd.merge(df_nmi_cor, df_nmi_tau, on='metric', how='inner')
        df_acc = pd.merge(df_acc_cor, df_acc_tau, on='metric', how='inner')
    else:
        df_nmi = pd.merge(df_nmi_cor, df_nmi_tau, left_index=True, right_index=True)
        df_acc = pd.merge(df_acc_cor, df_acc_tau, left_index=True, right_index=True)



    subcols = create_subcol(df_nmi.columns.tolist(), vertical)
    df_nmi.columns = pd.MultiIndex.from_tuples(subcols)

    subcols = create_subcol(df_acc.columns.tolist(), vertical)
    df_acc.columns = pd.MultiIndex.from_tuples(subcols)

    df_nmi = df_nmi.sort_index(axis=1, level=0, ascending=False)
    df_acc = df_acc.sort_index(axis=1, level=0, ascending=False)

    #print(df_nmi)

    if vertical:
        df_nmi = df_nmi.reindex(lname[1:][::-1])
        df_acc = df_acc.reindex(lname[1:][::-1])
        df_nmi = df_nmi.reindex(columns=mname, level=0)
        df_acc = df_acc.reindex(columns=mname, level=0)
    else:
        df_nmi = df_nmi.reindex(columns=lname, level=0)
        df_acc = df_acc.reindex(columns=lname, level=0)

    print(df_nmi)
    print(df_acc)



    if vertical:
        print(df_nmi.to_latex(index=True,
                        escape = False,
                        formatters={"name": str.upper},
                         float_format="{:.2f}".format).replace('space score', 'Paired score').replace('label score', 'Pooled score').replace('proposed score', '\\textbf{ACE}')
        )
        print(df_acc.to_latex(index=True,
                        escape = False,
                        formatters={"name": str.upper},
                         float_format="{:.2f}".format).replace('space score', 'Paired score').replace('label score', 'Pooled score').replace('proposed score', '\\textbf{ACE}')
        )
    else:
        print(df_nmi.to_latex(index=False,
                        escape = False,
                        formatters={"name": str.upper},
                         float_format="{:.2f}".format).replace('space score', 'Paired score').replace('label score', 'Pooled score').replace('proposed score', '\\textbf{ACE}')
        )
        print(df_acc.to_latex(index=False,
                        escape = False,
                        formatters={"name": str.upper},
                         float_format="{:.2f}".format).replace('space score', 'Paired score').replace('label score', 'Pooled score').replace('proposed score', '\\textbf{ACE}')
        )        
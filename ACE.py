import pandas as pd
from utils import *
import pickle as pk

tasks = { 'jule_hyper': './JULE_hyper',
          'jule_num': './JULE_num',
           'DEPICT': './DEPICT_hyper',
           'DEPICTnum': './DEPICT_num'
}



true_num = {
    'USPS':10, 'UMist':20, 'COIL-20':20, 'COIL-100':100, 'YTF':41, 'FRGC':20, 'MNIST-test':10, 'CMU-PIE':68
}

tasks_datasets = { 'jule_hyper': ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE'],
          'jule_num': ['USPS', 'UMist', 'COIL-20', 'COIL-100', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE'],
          'DEPICT': ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE'],
          'DEPICTnum': ['USPS', 'YTF', 'FRGC', 'MNIST-test', 'CMU-PIE']
}


def make_data_plot(dlabels, best_n, eval_data, task, metric, save_path):
    # get saving path (save_path: saved other results)

    spaceFiles = np.array(list(dlabels.keys()))
    labels = np.array(list(dlabels.values()))
    save_path = os.path.join(save_path, 'tnse')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, eval_data)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, metric)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    rpath = tasks[task]
    if 'jule' in task:
        rpath = os.path.join(rpath, 'jule') # root path
    # prepare path for truth label
    dpath = os.path.join(rpath, 'datasets') # data path
    dpath = os.path.join(dpath, eval_data)
    if 'jule' in task:
        dpath = os.path.join(dpath, 'data4torch.h5')
    else:
        dpath = os.path.join(dpath, 'data.h5')
    y = np.squeeze(np.array(h5py.File(dpath, 'r')['labels']))
    _, y = np.unique(y, return_inverse=True)
    labels_unique = np.unique(labels)
    best_space = None
    if best_n not in labels_unique:
        best_space = spaceFiles[best_n]

    for ul in labels_unique:
        if ul == best_n:
            uu = '{}_best'.format(ul)
        else:
            uu= str(ul)
        spath = os.path.join(save_path, uu)
        if not os.path.isdir(spath):
            os.mkdir(spath)
        spaces = spaceFiles[labels==ul].tolist()
        for sp in spaces:
            if 'jule' in task:
                fpath = os.path.join(rpath, 'feature{}.h5'.format(sp))
                X = np.array(h5py.File(fpath, 'r')['feature'])
            else:
                fpath = os.path.join(rpath, sp)
                X=np.array(np.load(fpath)['y_features'])
            fig = plot_tsne(X, y, eval_data)
            if (best_space != None) and (best_space == sp):
                wpath = os.path.join(spath,"{}_best.png".format(sp))
            else:
                wpath = os.path.join(spath,"{}.png".format(sp))

            fig.write_image(wpath)

if __name__ == '__main__':

    import argparse


    parser = argparse.ArgumentParser(description='Code for evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--merge", action='store_true', help="if take merge test")
    parser.add_argument("--tsne", action='store_true', help="plot tsne")
    parser.add_argument('--merge_threshold', type=float, default=0.5)
    parser.add_argument('--cl_method', type=str, default='hdbscan', help='dbscan or hdbscan')
    parser.add_argument('--rank_method', type=str, default='pr', help='pr or hits')
    parser.add_argument('--eps', type=float, default=0.05, help='eps require for dbscan')
    parser.add_argument('--filter_alpha', type=float, default=0.05, help='eps require for dbscan')
    parser.add_argument('--graph_alpha', type=float, default=0.05, help='eps require for dbscan')

    args = parser.parse_args()

    eval_dir = 'eval'
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)


    l = 0
    taus_all, corrs_all = [], []
    for task, root_path in tasks.items():
        save_path = os.path.join(eval_dir, 'results_{}'.format(task))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for metric in ['dav', 'ch', 'cosine', 'euclidean', 'ccc', 'dunn', 'cind', 'sdbw','ccdbw']:
            taus_again, corrs_again = [], []
            datasets = tasks_datasets[task]
            tau_avg = np.zeros(10)
            corr_avg = np.zeros(10)

            for eval_data in datasets:
                l = l + 1
                # get the file list
                if 'jule' in task:
                    modelFiles = get_files_with_substring_and_suffix(root_path, 'feature' + eval_data, 'h5')
                    modelFiles = [m[7:-3] for m in modelFiles]
                else:
                    modelFiles = get_files_with_substring_and_suffix(root_path, 'output' + eval_data, 'npz')

                # first load nmi and acc results
                tpath = 'true_{}.pkl'.format(eval_data)
                tpath = os.path.join(root_path, tpath)
                truth_nmi, truth_acc = pk.load(open(tpath, 'rb'))
                _, _, nmv = sort_and_match(truth_nmi, modelFiles)
                _, _, acv = sort_and_match(truth_acc, modelFiles)

                # get raw scores
                raw_truth = collect_raw_files(eval_data, root=os.path.join(root_path, 'raw_metric'))
                raw_results = process_raw(raw_truth[metric])
                _, _, rv = sort_and_match(raw_results, modelFiles)
                rv = np.where(np.isinf(rv) & (rv < 0), np.nan, rv)
                rv_min = np.nanmin(rv)
                rv = np.nan_to_num(rv, nan=rv_min)

                # load dip test results
                dpath = os.path.join(root_path, 'dip_{}.npz'.format(eval_data))
                files = np.load(dpath, allow_pickle=True)
                pvalues = files['pvalues1']
                models = files['models'][1:]
                pvalues = pvalues[np.isin(models, np.array(modelFiles))]
                models = models[np.isin(models, np.array(modelFiles))]
                # filter spaces based on the p-values
                keep_models = filter_out(pvalues, models, alpha=args.filter_alpha)
                keep_index = np.array([True if m in keep_models else False for m in modelFiles])
                spaceFiles = np.array([m for m in modelFiles if m in keep_models])

                all_scores = filter_merge_files_rep(eval_data, metric, root=root_path)
                scores, diag_scores = generate_scores_rep(all_scores, modelFiles)
                scores = np.where(np.isinf(scores) & (scores < 0), np.nan, scores)
                scores_min = np.nanmin(scores)
                scores = np.nan_to_num(scores, nan=scores_min)
                for k, v in diag_scores.items():
                    diag_scores[k] = np.nan_to_num(v, nan=scores_min)
                # get pair scores
                _, _, sv = sort_and_match(diag_scores, modelFiles)

                # eval mean score
                pool_score_before =eval_pool(modelFiles, scores) # before filtering based on dip test
                _, _, pool_score_before = sort_and_match(pool_score_before, modelFiles)
                scores = scores[keep_index, :] # after filtering based on dip tests
                print(task, metric, eval_data, scores.shape, len(modelFiles))

                # eval mean score
                pool_score =eval_pool(modelFiles, scores)
                _, _, pool_score = sort_and_match(pool_score, modelFiles)

                # get ACE scores
                st_score, graph, outliers, labels, best_n, prv, labels_initial = eval_ace(modelFiles, scores, spaceFiles, args.eps,
                                                                                     args.graph_alpha, args.cl_method, args.rank_method)
                _, key_st, st_score = sort_and_match(st_score, modelFiles)

                # make the graph
                l = graph_plot(l, graph, labels, labels_initial, save_path, eval_data, metric)
                if args.tsne:
                    make_data_plot(labels, best_n, eval_data, task, metric, save_path)


                # get the evaluate performance
                # NMI
                tau_pool, _ = kendalltau(pool_score, nmv)
                tau_pool_before, _ = kendalltau(pool_score_before, nmv)
                tau_st, _ = kendalltau(st_score, nmv)
                tau_sv, _ = kendalltau(sv, nmv)
                tau_rv, _ = kendalltau(rv, nmv)
                # ACC
                tau_pool1, _ = kendalltau(pool_score, acv)
                tau_pool1_before, _ = kendalltau(pool_score_before, acv)
                tau_st1, _ = kendalltau(st_score, acv)
                tau_sv1, _ = kendalltau(sv, acv)
                tau_rv1, _ = kendalltau(rv, acv)
                # NMI
                cor_pool, _ = spearmanr(pool_score, nmv)
                cor_pool_before, _ = spearmanr(pool_score_before, nmv)
                cor_st, _ = spearmanr(st_score, nmv)
                cor_sv, _ = spearmanr(sv, nmv)
                cor_rv, _ = spearmanr(rv, nmv)
                # Acc
                cor_pool1, _ = spearmanr(pool_score, acv)
                cor_pool1_before, _ = spearmanr(pool_score_before, acv)
                cor_st1, _ = spearmanr(st_score, acv)
                cor_sv1, _ = spearmanr(sv, acv)
                cor_rv1, _ = spearmanr(rv, acv)

                tau = [eval_data, tau_st, tau_pool, tau_pool_before, tau_sv, tau_rv, tau_st1, tau_pool1,  tau_pool1_before, tau_sv1, tau_rv1]
                corr = [eval_data, cor_st, cor_pool, cor_pool_before, cor_sv, cor_rv, cor_st1, cor_pool1, cor_pool1_before, cor_sv1, cor_rv1]
                tau_avg += np.array(tau[1:])/len(datasets)
                corr_avg += np.array(corr[1:])/len(datasets)
                ##
                modelFiles = sorted(modelFiles)
                if 'num' in task:
                    st_argmax = max_number(st_score, modelFiles, eval_data)
                    label_argmax = max_number(pool_score, modelFiles, eval_data)
                    label_before_argmax = max_number(pool_score_before, modelFiles, eval_data)
                    sv_argmax = max_number(sv, modelFiles, eval_data)
                    rv_argmax = max_number(rv, modelFiles, eval_data)
                    tau_st = '{} ({})'.format(tau_st, st_argmax)
                    tau_pool = '{} ({})'.format(tau_pool, label_argmax)
                    tau_pool_before = '{} ({})'.format(tau_pool_before, label_before_argmax)
                    tau_sv = '{} ({})'.format(tau_sv, sv_argmax)
                    tau_rv = '{} ({})'.format(tau_rv, rv_argmax)
                    cor_st = '{} ({})'.format(cor_st, st_argmax)
                    cor_pool = '{} ({})'.format(cor_pool, label_argmax)
                    cor_pool_before = '{} ({})'.format(cor_pool_before, label_before_argmax)
                    cor_sv = '{} ({})'.format(cor_sv, sv_argmax)
                    cor_rv = '{} ({})'.format(cor_rv, rv_argmax)
                    eval_data = '{} ({})'.format(eval_data, true_num[eval_data])
                #################
                tau = [eval_data, tau_st, tau_pool, tau_pool_before, tau_sv, tau_rv, tau_st1, tau_pool1,  tau_pool1_before, tau_sv1, tau_rv1]
                corr = [eval_data, cor_st, cor_pool, cor_pool_before, cor_sv, cor_rv, cor_st1, cor_pool1, cor_pool1_before, cor_sv1, cor_rv1]
                taus_again.append(tau)
                corrs_again.append(corr)

            taus_all.append([task, metric] + ['' for _ in range(7)])
            corrs_all.append([task, metric] + ['' for _ in range(7)])
            taus_all.extend(taus_again)
            corrs_all.extend(corrs_again)
            taus_all.append(['avg']+tau_avg.tolist())
            corrs_all.append(['avg'] + corr_avg.tolist())

            tdf = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pool score vs nmi', 'pool score vs nmi (w/o dip)',
                                        'pair score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                        'pool score vs acc', 'pool score vs acc (w/o dip)', 'pair score vs acc', 'raw score vs acc'], data=taus_again)
            cdf = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pool score vs nmi', 'pool score vs nmi (w/o dip)',
                                        'pair score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                        'pool score vs acc',  'pool score vs acc (w/o dip)', 'pair score vs acc', 'raw score vs acc'], data=corrs_again)

            tdf.to_csv(os.path.join(save_path, '{}_tau.csv'.format(metric)), index=False)
            cdf.to_csv(os.path.join(save_path, '{}_cor.csv'.format(metric)), index=False)

    tdf1 = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pool score vs nmi', 'pool score vs nmi (w/o dip)',
                                'pair score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                'pool score vs acc', 'pool score vs acc (w/o dip)', 'pair score vs acc', 'raw score vs acc'], data=taus_all)
    cdf1 = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pool score vs nmi', 'pool score vs nmi (w/o dip)',
                                'pair score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                'pool score vs acc',  'pool score vs acc (w/o dip)', 'pair score vs acc', 'raw score vs acc'], data=corrs_all)

    tdf1.to_csv(os.path.join(eval_dir, '{}_tau.csv'.format('combined')), index=False)
    cdf1.to_csv(os.path.join(eval_dir, '{}_cor.csv'.format('combined')), index=False)


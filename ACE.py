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
    import time
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

    eval_dir = 'plot_{}_{}_{}_{}_{}_{}'.format(args.cl_method, args.rank_method, args.eps,
                                                     args.filter_alpha, args.graph_alpha, args.merge)
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
                time0 = time.time()
                l = l + 1
                # get the file list
                if 'jule' in task:
                    modelFiles = get_files_with_substring_and_suffix(root_path, 'feature' + eval_data, 'h5')
                    modelFiles = [m[7:-3] for m in modelFiles]
                else:
                    modelFiles = get_files_with_substring_and_suffix(root_path, 'output' + eval_data, 'npz')
                key_model = sorted(modelFiles)

                # first load nmi and acc results
                tpath = 'true_{}.pkl'.format(eval_data)
                tpath = os.path.join(root_path, tpath)
                truth_nmi, truth_acc = pk.load(open(tpath, 'rb'))
                truth_nmi1, key_nmi1, nmv1 = sort_and_match(truth_nmi, modelFiles)
                truth_acc1, key_acc1, acv1 = sort_and_match(truth_acc, modelFiles)

                raw_truth = collect_raw_files(eval_data, root=os.path.join(root_path, 'raw_metric'))
                truth_nmi = raw_truth['nmi']
                truth_acc = raw_truth['acc']
                truth_nmi, key_nmi, nmv = sort_and_match(truth_nmi, modelFiles)
                truth_acc, key_acc, acv = sort_and_match(truth_acc, modelFiles)


                raw_results = process_raw(raw_truth[metric])
                raw_results, key_raw, rv = sort_and_match(raw_results, modelFiles)

                rv = np.where(np.isinf(rv) & (rv < 0), np.nan, rv)
                rv_min = np.nanmin(rv)
                rv = np.nan_to_num(rv, nan=rv_min)

                from utils import *
                # load dip test results
                dpath = os.path.join(root_path, 'dip_{}.npz'.format(eval_data))
                files = np.load(dpath, allow_pickle=True)
                pvalues = files['pvalues1']
                models = files['models'][1:]
                pvalues = pvalues[np.isin(models, np.array(modelFiles))]
                models = models[np.isin(models, np.array(modelFiles))]
                keep_models = filter_out(pvalues, models, alpha=args.filter_alpha)

                all_scores = filter_merge_files_rep(eval_data, metric, root=os.path.join(root_path, 'metric_result2'))

                pair_scores, diag_scores = generate_pair_scores_rep(all_scores, modelFiles)

                pair_scores = np.where(np.isinf(pair_scores) & (pair_scores < 0), np.nan, pair_scores)
                pair_scores_min = np.nanmin(pair_scores)
                pair_scores = np.nan_to_num(pair_scores, nan=pair_scores_min)
                for k, v in diag_scores.items():
                    diag_scores[k] = np.nan_to_num(v, nan=pair_scores_min)


                ###test diag_scores and sv are matched
                diag_scores, key_diag, sv = sort_and_match(diag_scores, modelFiles)

                # select the spaces we need
                keep_index = np.array([True if m in keep_models else False for m in modelFiles])
                spaceFiles = np.array([m for m in modelFiles if m in keep_models])

                label_score_before =eval_pool(modelFiles, pair_scores)
                _, _, label_score_before = sort_and_match(label_score_before, modelFiles)

                pair_scores = pair_scores[keep_index, :] # remove based on dip tests
                print(task, metric, eval_data, pair_scores.shape, len(modelFiles))

                # eval mean score
                label_score =eval_pool(modelFiles, pair_scores)
                _, _, label_score = sort_and_match(label_score, modelFiles)

                st_score, graph, outliers, labels, best_n, prv, labels_initial = eval_ace(modelFiles, pair_scores, spaceFiles, args.eps,
                                                                                     args.graph_alpha, args.cl_method, args.rank_method)
                _, key_st, st_score = sort_and_match(st_score, modelFiles)

                # make the graph
                l = graph_plot(l, graph, labels, labels_initial, save_path, eval_data, metric)
                if args.tsne:
                    make_data_plot(labels, best_n, eval_data, task, metric, save_path)


                # get the evaluate performance
                tau_label, _ = kendalltau(label_score, nmv)
                tau_label_before, _ = kendalltau(label_score_before, nmv)

                tau_st, _ = kendalltau(st_score, nmv)
                tau_sv, _ = kendalltau(sv, nmv)
                tau_rv, _ = kendalltau(rv, nmv)
                #
                tau_label1, _ = kendalltau(label_score, acv)
                tau_label1_before, _ = kendalltau(label_score_before, acv)
                tau_st1, _ = kendalltau(st_score, acv)
                tau_sv1, _ = kendalltau(sv, acv)
                tau_rv1, _ = kendalltau(rv, acv)
                #
                cor_label, _ = spearmanr(label_score, nmv)
                cor_label_before, _ = spearmanr(label_score_before, nmv)
                cor_st, _ = spearmanr(st_score, nmv)
                cor_sv, _ = spearmanr(sv, nmv)
                cor_rv, _ = spearmanr(rv, nmv)
                #
                cor_label1, _ = spearmanr(label_score, acv)
                cor_label1_before, _ = spearmanr(label_score_before, acv)
                cor_st1, _ = spearmanr(st_score, acv)
                cor_sv1, _ = spearmanr(sv, acv)
                cor_rv1, _ = spearmanr(rv, acv)

                tau = [eval_data, tau_st, tau_label, tau_label_before, tau_sv, tau_rv, tau_st1, tau_label1,  tau_label1_before, tau_sv1, tau_rv1]
                corr = [eval_data, cor_st, cor_label, cor_label_before, cor_sv, cor_rv, cor_st1, cor_label1, cor_label1_before, cor_sv1, cor_rv1]
                tau_avg += np.array(tau[1:])/len(datasets)
                corr_avg += np.array(corr[1:])/len(datasets)
                ##
                modelFiles = sorted(modelFiles)
                if 'num' in task:
                    st_argmax = max_number(st_score, modelFiles, eval_data)
                    label_argmax = max_number(label_score, modelFiles, eval_data)
                    label_before_argmax = max_number(label_score_before, modelFiles, eval_data)
                    sv_argmax = max_number(sv, modelFiles, eval_data)
                    rv_argmax = max_number(rv, modelFiles, eval_data)
                    tau_st = '{} ({})'.format(tau_st, st_argmax)
                    tau_label = '{} ({})'.format(tau_label, label_argmax)
                    tau_label_before = '{} ({})'.format(tau_label_before, label_before_argmax)
                    tau_sv = '{} ({})'.format(tau_sv, sv_argmax)
                    tau_rv = '{} ({})'.format(tau_rv, rv_argmax)
                    cor_st = '{} ({})'.format(cor_st, st_argmax)
                    cor_label = '{} ({})'.format(cor_label, label_argmax)
                    cor_label_before = '{} ({})'.format(cor_label_before, label_before_argmax)
                    cor_sv = '{} ({})'.format(cor_sv, sv_argmax)
                    cor_rv = '{} ({})'.format(cor_rv, rv_argmax)
                    eval_data = '{} ({})'.format(eval_data, true_num[eval_data])
                #################
                tau = [eval_data, tau_st, tau_label, tau_label_before, tau_sv, tau_rv, tau_st1, tau_label1,  tau_label1_before, tau_sv1, tau_rv1]
                corr = [eval_data, cor_st, cor_label, cor_label_before, cor_sv, cor_rv, cor_st1, cor_label1, cor_label1_before, cor_sv1, cor_rv1]
                taus_again.append(tau)
                corrs_again.append(corr)



            taus_all.append([task, metric] + ['' for _ in range(7)])
            corrs_all.append([task, metric] + ['' for _ in range(7)])
            taus_all.extend(taus_again)
            corrs_all.extend(corrs_again)
            taus_all.append(['avg']+tau_avg.tolist())
            corrs_all.append(['avg'] + corr_avg.tolist())

            tdf = pd.DataFrame(columns=['dataset',  'transformed_score vs nmi', 'mean_score vs nmi', 'mean_score vs nmi (before)',
                                        'embedding_score vs nmi', 'raw_score vs nmi', 'transformed_score vs acc',
                                        'mean_score vs acc', 'mean_score vs acc (before)', 'embedding_score vs acc', 'raw_score vs acc'], data=taus_again)
            cdf = pd.DataFrame(columns=['dataset',  'transformed_score vs nmi', 'mean_score vs nmi', 'mean_score vs nmi (before)',
                                        'embedding_score vs nmi', 'raw_score vs nmi', 'transformed_score vs acc',
                                        'mean_score vs acc',  'mean_score vs acc (before)', 'embedding_score vs acc', 'raw_score vs acc'], data=corrs_again)

            tdf.to_csv(os.path.join(save_path, '{}_tau.csv'.format(metric)), index=False)
            cdf.to_csv(os.path.join(save_path, '{}_cor.csv'.format(metric)), index=False)

    tdf1 = pd.DataFrame(columns=['dataset',  'transformed_score vs nmi', 'mean_score vs nmi', 'mean_score vs nmi (before)',
                                'embedding_score vs nmi', 'raw_score vs nmi', 'transformed_score vs acc',
                                'mean_score vs acc', 'mean_score vs acc (before)', 'embedding_score vs acc', 'raw_score vs acc'], data=taus_all)
    cdf1 = pd.DataFrame(columns=['dataset',  'transformed_score vs nmi', 'mean_score vs nmi', 'mean_score vs nmi (before)',
                                'embedding_score vs nmi', 'raw_score vs nmi', 'transformed_score vs acc',
                                'mean_score vs acc',  'mean_score vs acc (before)', 'embedding_score vs acc', 'raw_score vs acc'], data=corrs_all)

    tdf1.to_csv(os.path.join(eval_dir, '{}_tau.csv'.format('combined')), index=False)
    cdf1.to_csv(os.path.join(eval_dir, '{}_cor.csv'.format('combined')), index=False)


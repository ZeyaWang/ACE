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



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Code for evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tsne", action='store_true', help="plot tsne")
    parser.add_argument('--cl_method', type=str, default='hdbscan', help='dbscan or hdbscan')
    parser.add_argument('--rank_method', type=str, default='pr', help='pr or hits')
    parser.add_argument('--eps', type=float, default=0.05, help='eps require for dbscan')
    parser.add_argument('--filter_alpha', type=float, default=0.05, help='FWER for dip test')
    parser.add_argument('--graph_alpha', type=float, default=0.05, help='FWER for creating graph')

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
                # get the list of files from all the evaluation results
                if 'jule' in task:
                    modelFiles = get_files_with_substring_and_suffix(root_path, 'feature' + eval_data, 'h5')
                    modelFiles = [m[7:-3] for m in modelFiles]
                else:
                    modelFiles = get_files_with_substring_and_suffix(root_path, 'output' + eval_data, 'npz')

                # first load nmi and acc results (used as truth for evaluation)
                tpath = 'true_{}.pkl'.format(eval_data)
                tpath = os.path.join(root_path, tpath)
                truth_nmi, truth_acc = pk.load(open(tpath, 'rb'))
                # sort the results to have the same order
                _, _, nmv = sort_and_match(truth_nmi, modelFiles)
                _, _, acv = sort_and_match(truth_acc, modelFiles)

                # get raw scores
                raw_truth = collect_raw_files(eval_data, root=os.path.join(root_path, 'raw_metric'))
                raw_results = process_raw(raw_truth[metric])
                _, _, raw_score = sort_and_match(raw_results, modelFiles)
                raw_score = np.where(np.isinf(raw_score) & (raw_score < 0), np.nan, raw_score)
                raw_score_min = np.nanmin(raw_score)
                raw_score = np.nan_to_num(raw_score, nan=raw_score_min)

                # load dip test results
                dpath = os.path.join(root_path, 'dip_{}.npz'.format(eval_data))
                files = np.load(dpath, allow_pickle=True)
                pvalues = files['pvalues1']
                models = files['models'][1:]
                pvalues = pvalues[np.isin(models, np.array(modelFiles))]
                models = models[np.isin(models, np.array(modelFiles))]
                # filter spaces based on p-values
                keep_models = filter_out(pvalues, models, alpha=args.filter_alpha)
                keep_index = np.array([True if m in keep_models else False for m in modelFiles])
                spaceFiles = np.array([m for m in modelFiles if m in keep_models])

                # collect all the internal measure values given the metric
                all_scores = collect_score_files(eval_data, metric, root=root_path)
                scores, diag_scores = collect_all_scores(all_scores, modelFiles)
                scores = np.where(np.isinf(scores) & (scores < 0), np.nan, scores)
                scores_min = np.nanmin(scores)
                scores = np.nan_to_num(scores, nan=scores_min)
                for k, v in diag_scores.items():
                    diag_scores[k] = np.nan_to_num(v, nan=scores_min)

                # get paired scores
                _, _, pair_score = sort_and_match(diag_scores, modelFiles)

                # eval pooled score
                pool_score_before =eval_pool(modelFiles, scores) # before dip test
                _, _, pool_score_before = sort_and_match(pool_score_before, modelFiles)

                # eval pooled score
                scores = scores[keep_index, :] # after dip tests
                pool_score =eval_pool(modelFiles, scores)
                _, _, pool_score = sort_and_match(pool_score, modelFiles)

                # get ACE scores
                ace_score, graph, outliers, labels, selected_group, prv, labels_initial = eval_ace(modelFiles, scores, spaceFiles, args.eps,
                                                                                     args.graph_alpha, args.cl_method, args.rank_method)
                _, _, ace_score = sort_and_match(ace_score, modelFiles)

                # make the graph of spaces
                l = graph_plot(l, graph, labels, save_path, eval_data, metric)
                # make the tsne plot
                if args.tsne:
                    make_data_plot(labels, selected_group, eval_data, task, metric, save_path)


                # get the evaluate performance
                # NMI
                tau_pool, _ = kendalltau(pool_score, nmv)
                tau_pool_before, _ = kendalltau(pool_score_before, nmv)
                tau_ace_score, _ = kendalltau(ace_score, nmv)
                tau_pair_score, _ = kendalltau(pair_score, nmv)
                tau_raw_score, _ = kendalltau(raw_score, nmv)
                # ACC
                tau_pool1, _ = kendalltau(pool_score, acv)
                tau_pool1_before, _ = kendalltau(pool_score_before, acv)
                tau_ace_score1, _ = kendalltau(ace_score, acv)
                tau_pair_score1, _ = kendalltau(pair_score, acv)
                tau_raw_score1, _ = kendalltau(raw_score, acv)
                # NMI
                cor_pool, _ = spearmanr(pool_score, nmv)
                cor_pool_before, _ = spearmanr(pool_score_before, nmv)
                cor_ace_score, _ = spearmanr(ace_score, nmv)
                cor_pair_score, _ = spearmanr(pair_score, nmv)
                cor_raw_score, _ = spearmanr(raw_score, nmv)
                # Acc
                cor_pool1, _ = spearmanr(pool_score, acv)
                cor_pool1_before, _ = spearmanr(pool_score_before, acv)
                cor_ace_score1, _ = spearmanr(ace_score, acv)
                cor_pair_score1, _ = spearmanr(pair_score, acv)
                cor_raw_score1, _ = spearmanr(raw_score, acv)

                tau = [eval_data, tau_ace_score, tau_pool, tau_pool_before, tau_pair_score, tau_raw_score, tau_ace_score1, tau_pool1,  tau_pool1_before, tau_pair_score1, tau_raw_score1]
                corr = [eval_data, cor_ace_score, cor_pool, cor_pool_before, cor_pair_score, cor_raw_score, cor_ace_score1, cor_pool1, cor_pool1_before, cor_pair_score1, cor_raw_score1]
                tau_avg += np.array(tau[1:])/len(datasets)
                corr_avg += np.array(corr[1:])/len(datasets)
                ##
                modelFiles = sorted(modelFiles)
                if 'num' in task:
                    st_argmax = max_number(ace_score, modelFiles, eval_data)
                    label_argmax = max_number(pool_score, modelFiles, eval_data)
                    label_before_argmax = max_number(pool_score_before, modelFiles, eval_data)
                    pair_score_argmax = max_number(pair_score, modelFiles, eval_data)
                    raw_score_argmax = max_number(raw_score, modelFiles, eval_data)
                    tau_ace_score = '{} ({})'.format(tau_ace_score, st_argmax)
                    tau_pool = '{} ({})'.format(tau_pool, label_argmax)
                    tau_pool_before = '{} ({})'.format(tau_pool_before, label_before_argmax)
                    tau_pair_score = '{} ({})'.format(tau_pair_score, pair_score_argmax)
                    tau_raw_score = '{} ({})'.format(tau_raw_score, raw_score_argmax)
                    cor_ace_score = '{} ({})'.format(cor_ace_score, st_argmax)
                    cor_pool = '{} ({})'.format(cor_pool, label_argmax)
                    cor_pool_before = '{} ({})'.format(cor_pool_before, label_before_argmax)
                    cor_pair_score = '{} ({})'.format(cor_pair_score, pair_score_argmax)
                    cor_raw_score = '{} ({})'.format(cor_raw_score, raw_score_argmax)
                    eval_data = '{} ({})'.format(eval_data, true_num[eval_data])
                #################
                tau = [eval_data, tau_ace_score, tau_pool, tau_pool_before, tau_pair_score, tau_raw_score, tau_ace_score1, tau_pool1,  tau_pool1_before, tau_pair_score1, tau_raw_score1]
                corr = [eval_data, cor_ace_score, cor_pool, cor_pool_before, cor_pair_score, cor_raw_score, cor_ace_score1, cor_pool1, cor_pool1_before, cor_pair_score1, cor_raw_score1]
                taus_again.append(tau)
                corrs_again.append(corr)

            taus_all.append([task, metric] + ['' for _ in range(7)])
            corrs_all.append([task, metric] + ['' for _ in range(7)])
            taus_all.extend(taus_again)
            corrs_all.extend(corrs_again)
            taus_all.append(['avg']+tau_avg.tolist())
            corrs_all.append(['avg'] + corr_avg.tolist())

            tdf = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pooled score vs nmi', 'pooled score vs nmi (w/o dip)',
                                        'paired score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                        'pooled score vs acc', 'pooled score vs acc (w/o dip)', 'paired score vs acc', 'raw score vs acc'], data=taus_again)
            cdf = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pooled score vs nmi', 'pooled score vs nmi (w/o dip)',
                                        'paired score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                        'pooled score vs acc',  'pooled score vs acc (w/o dip)', 'paired score vs acc', 'raw score vs acc'], data=corrs_again)

            tdf.to_csv(os.path.join(save_path, '{}_tau.csv'.format(metric)), index=False)
            cdf.to_csv(os.path.join(save_path, '{}_cor.csv'.format(metric)), index=False)

    tdf1 = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pooled score vs nmi', 'pooled score vs nmi (w/o dip)',
                                'paired score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                'pooled score vs acc', 'pooled score vs acc (w/o dip)', 'paired score vs acc', 'raw score vs acc'], data=taus_all)
    cdf1 = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pooled score vs nmi', 'pooled score vs nmi (w/o dip)',
                                'paired score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                'pooled score vs acc',  'pooled score vs acc (w/o dip)', 'paired score vs acc', 'raw score vs acc'], data=corrs_all)

    tdf1.to_csv(os.path.join(eval_dir, '{}_tau.csv'.format('combined')), index=False)
    cdf1.to_csv(os.path.join(eval_dir, '{}_cor.csv'.format('combined')), index=False)


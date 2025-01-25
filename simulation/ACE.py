import pandas as pd
from utils import *
import pickle as pk
from collections import defaultdict


tasks = ['sim_dense_1.0','sim_sparse_1.0','sim_dense_2.0','sim_sparse_2.0']#


def convert_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: convert_to_dict(v) for k, v in d.items()}
    return d

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Code for evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tsne", action='store_true', help="plot tsne")
    parser.add_argument('--cl_method', type=str, default='hdbscan', help='dbscan or hdbscan')
    parser.add_argument('--rank_method', type=str, default='pr', help='pr or hits')
    parser.add_argument('--eps', type=float, default=0.05, help='eps require for dbscan')
    parser.add_argument('--filter_alpha', type=float, default=0.05, help='FWER for dip test')
    parser.add_argument('--graph_alpha', type=float, default=0.1, help='FWER for creating graph')

    args = parser.parse_args()

    eval_dir = 'eval'
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    l = 0
    nrep = 50
    taus_all, corrs_all = [], []
    # for task, root_path in tasks.items():
    nested_dict = lambda: defaultdict(nested_dict)
    box_nmi = nested_dict()
    box_acc = nested_dict()

    for task in tasks:
        root_path = task
        save_path = os.path.join(eval_dir, 'results_{}'.format(task))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for metric in ['dav', 'ch', 'cosine', 'euclidean']:
            taus_again, corrs_again = [], []
            tau_avg, corr_avg = [], []
            for eval_data in range(nrep):
                l = l + 1
                root_seed_path = os.path.join(root_path, str(eval_data))
                modelFiles = get_files_with_substring_and_suffix(root_seed_path, 'output', 'npz')
                modelFiles = sorted(modelFiles)
                # first load nmi and acc results (used as truth for evaluation)
                tpath = os.path.join(root_seed_path, 'true.pkl')
                truth_nmi, truth_acc = pk.load(open(tpath, 'rb'))
                # sort the results to have the same order
                _, _, nmv = sort_and_match(truth_nmi, modelFiles)
                _, _, acv = sort_and_match(truth_acc, modelFiles)

                # get raw scores
                raw_truth = collect_score_files(metric, root=os.path.join(root_seed_path, 'raw_metric'))

                raw_results = process_raw(raw_truth[metric])
                _, _, raw_score = sort_and_match(raw_results, modelFiles)
                #print(raw_score)
                raw_score = np.where(np.isinf(raw_score) & (raw_score < 0), np.nan, raw_score)
                #print(raw_score)
                raw_score_min = np.nanmin(raw_score)
                raw_score = np.nan_to_num(raw_score, nan=raw_score_min)

                # load dip test results
                dpath = os.path.join(root_path, 'dip_{}.npz'.format(eval_data))

                files = np.load(dpath, allow_pickle=True)
                pvalues = files['pvalues1']
                models = files['models'][2:]
                models = np.array([os.path.basename(m) for m in models])
                pvalues = pvalues[np.isin(models, np.array(modelFiles))]
                models = models[np.isin(models, np.array(modelFiles))]
                # filter spaces based on p-values
                keep_models = filter_out(pvalues, models, alpha=args.filter_alpha)
                if len(keep_models) < 1:
                    print(task, eval_data,'continue')
                    keep_models = list(modelFiles)
                keep_index = np.array([True if m in keep_models else False for m in modelFiles])
                spaceFiles = np.array([m for m in modelFiles if m in keep_models])

                # collect all the internal measure values given the metric
                all_scores = collect_score_files(metric, root=os.path.join(root_seed_path, 'embedded_metric'))
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
                is_constant = np.any(np.all(scores == scores[:, [0]], axis=1))
                if is_constant:
                    print(nmv, acv)
                    continue
                if len(spaceFiles) == 1: 
                    ace_score = {}
                    for i, m in enumerate(modelFiles):
                        ace_score[m] = scores[:,i]
                else:
                    ace_score, graph, outliers, labels, selected_group, prv, labels_initial = eval_ace(modelFiles, scores, spaceFiles, args.eps,
                                                                                        args.graph_alpha, args.cl_method, args.rank_method)
                _, _, ace_score = sort_and_match(ace_score, modelFiles)


                # get the evaluate performance
                # NMI
                #print(task, metric, eval_data, nmv, ace_score, pair_score, raw_score)
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
                ##
                modelFiles = sorted(modelFiles)
                tau = [eval_data, tau_ace_score, tau_pool, tau_pool_before, tau_pair_score, tau_raw_score, tau_ace_score1, tau_pool1,  tau_pool1_before, tau_pair_score1, tau_raw_score1]
                corr = [eval_data, cor_ace_score, cor_pool, cor_pool_before, cor_pair_score, cor_raw_score, cor_ace_score1, cor_pool1, cor_pool1_before, cor_pair_score1, cor_raw_score1]
                #taus_again.append(tau)
                #corrs_again.append(corr)
                tau_avg.append(tau[1:])
                corr_avg.append(corr[1:])

            tau_avg = np.stack(tau_avg, axis=0)
            corr_avg = np.stack(corr_avg, axis=0)
            
            nmi_tau_all = tau_avg[:, [0,1,3,4]]
            acc_tau_all = tau_avg[:, [5,6,8,9]]
            nmi_corr_all = corr_avg[:, [0,1,3,4]]
            acc_corr_all = corr_avg[:, [5,6,8,9]]
            box_nmi[task]['tau'][metric] = nmi_tau_all
            box_nmi[task]['corr'][metric] = nmi_corr_all
            box_acc[task]['tau'][metric] = acc_tau_all
            box_acc[task]['corr'][metric] = acc_corr_all

            tavg = np.mean(tau_avg, axis=0)
            cavg = np.mean(corr_avg, axis=0)
            tavg = np.round(tavg, 2)
            cavg = np.round(cavg, 2)
            tsd = np.std(tau_avg, axis=0)
            csd = np.std(corr_avg, axis=0)
            tsd = np.round(tsd, 2)
            csd = np.round(csd, 2)

            tau_avg = [f"{v1} ({v2})" for v1, v2 in zip(tavg, tsd)]
            corr_avg = [f"{v1} ({v2})" for v1, v2 in zip(cavg, csd)]

            taus_all.append([task, metric] + ['' for _ in range(9)])
            corrs_all.append([task, metric] + ['' for _ in range(9)])
            taus_all.append(['avg']+tau_avg)
            corrs_all.append(['avg'] + corr_avg)


            tdf = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pooled score vs nmi', 'pooled score vs nmi (w/o dip)',
                                        'paired score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                        'pooled score vs acc', 'pooled score vs acc (w/o dip)', 'paired score vs acc', 'raw score vs acc'], data=[['avg']+tau_avg])
            cdf = pd.DataFrame(columns=['dataset',  'ACE vs nmi', 'pooled score vs nmi', 'pooled score vs nmi (w/o dip)',
                                        'paired score vs nmi', 'raw score vs nmi', 'ACE vs acc',
                                        'pooled score vs acc',  'pooled score vs acc (w/o dip)', 'paired score vs acc', 'raw score vs acc'], data=[['avg']+corr_avg])

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

    box_nmi = convert_to_dict(box_nmi)
    box_acc = convert_to_dict(box_acc)

    with open(os.path.join(eval_dir, 'box_nmi.pkl'), 'wb') as op1:
        pk.dump(box_nmi, op1)
    with open(os.path.join(eval_dir, 'box_acc.pkl'), 'wb') as op2:
        pk.dump(box_acc, op2)
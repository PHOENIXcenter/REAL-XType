import os, copy, time, argparse, platform, pickle, collections, random
from os.path import join

sys = platform.system()
parser = argparse.ArgumentParser(description='Process some parameters.')
# data preprocess
parser.add_argument('--log2', type=bool, default=True)
parser.add_argument('--zscore', type=bool, default=True)
parser.add_argument('--feat_select', type=str, default='f1269')
parser.add_argument('--min_impute', type=bool, default=True)

# model params
parser.add_argument('--bias', type=bool, default=False)
parser.add_argument('--h_dim', type=int, default=16)
parser.add_argument('--activ', type=str, default='sigmoid')
parser.add_argument('--regu', type=float, default=1e-3)
parser.add_argument('--loss_su', type=float, default=1.)
parser.add_argument('--loss_nll', type=float, default=1.)
parser.add_argument('--loss_da', type=float, default=0.3)
parser.add_argument('--loss_var', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--step_num', type=int, default=500)
parser.add_argument('--b_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--n_estimators', type=float, default=100)
parser.add_argument('--max_depth', type=float, default=None)

# settings
parser.add_argument('--test_cohort', type=str, default='SH')
parser.add_argument('--para_id', type=int, default=0)
parser.add_argument('--method', default='XType')
parser.add_argument('--device', default='/cpu')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--km_viz', type=bool, default=False)
parser.add_argument('--n_jobs', type=int, default=10)

if sys == 'Linux':
    parser.add_argument('--data_path', default='/data/linhai/HCC/data')
else:
    parser.add_argument('--data_path', default=r'D:\Data\Tumor_clinic')

args = parser.parse_args()

experiment_path = join(
    args.data_path, 
    'benchmark', 
    args.method
)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

from para_space import get_paras
args, comb_dict = get_paras(args, join(experiment_path, 'param_list.csv'))
if comb_dict == None:
    quit()

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lifelines.utils import concordance_index

import data_utils as utils
from network_model.cox_cdan_model import *
from viz_utils import *
from data_regroups import *

result_path = join(
    experiment_path,
    'test-'+str(args.test_cohort),
    'para-'+str(args.para_id),
    'seed-'+str(args.seed),
)

print(args)

def real_xtype(datasets, datasets_train, datasets_valid, datasets_test):
    classifier, optimizer = make_simple_models(
        args.h_dim,
        3, 
        len(datasets_train),
        args.lr,
        args.regu,
        args.activ,
        args.bias
    )

    start_time = time.time()
    groups_unique = []
    for i, dataset in enumerate(datasets_train):
        datasets_train[i]['groups'] = np.ones_like(dataset['OS']) * len(groups_unique)
        if dataset['cohort'] not in groups_unique:
            groups_unique.append(dataset['cohort'])

    random.seed(args.seed)
    for epoch in range(args.step_num):
        r_train = []
        x1, y, x2, t1, e1, t2, e2, y2 = [], [], [], [], [], [], [], []
        n1, n2 = 0., 0.
        random.shuffle(datasets_train)
        for cohort_id, dataset in enumerate(datasets_train):
            x = dataset['data'].astype(np.float32)
            OS = dataset['OS']
            status = dataset['status']
            DFS = dataset['DFS']
            recurrence = dataset['recurrence']
            groups = dataset['groups']
            n = len(OS)
            if 'Jiang' in dataset['cohort']:
                ids = np.random.choice(np.arange(n), args.b_size, replace=False)
                x1.append(x[ids])
                y.append(dataset['subtypes'].astype(np.int32)[ids])
            ids = np.random.choice(np.arange(n), n, replace=False).astype(int)
            x2.append(x[ids])
            t1.append(OS[ids])
            e1.append(status[ids])
            t2.append(DFS[ids])
            e2.append(recurrence[ids])
            y2.append(groups[ids])
        x1 = np.concatenate(x1, axis=0).astype(np.float32)
        y = np.concatenate(y, axis=0)
        x2 = np.concatenate(x2, axis=0).astype(np.float32)
        t1 = np.concatenate(t1, axis=0).astype(np.float32)
        e1 = np.concatenate(e1, axis=0).astype(np.float32)
        t2 = np.concatenate(t2, axis=0).astype(np.float32)
        e2 = np.concatenate(e2, axis=0).astype(np.float32)
        y2 = np.concatenate(y2, axis=0).astype(np.int32)

        # train model with a sliding window minibatch
        start = 0
        loss_da = 0.
        acc_da = 0.
        while start < len(t1) - 50:
            ids = np.arange(start , min(start + args.b_size, len(t1)))
            training_start_time = time.time()
            loss_su, loss_nll, loss_da, loss_l1, logits = train_step(
                x1, y, x2[ids], t1[ids], e1[ids], t2[ids], e2[ids], y2[ids],
                classifier, optimizer, 
                [args.loss_su, args.loss_nll, args.loss_da, args.loss_var],
                args.dropout,
                args.alpha
            )
            acc_da = np.mean(np.argmax(classifier.domain_test(x2), axis=1) == y2)
            acc_train = np.mean(np.argmax(logits, axis=1) == y)
            a = classifier.predict(x2[ids])
            r_train.append((
                    utils.get_reward(t1[ids], e1[ids], a) + 
                    utils.get_reward(t2[ids], e2[ids], a)
                )[0]
            )
            start += args.b_size
        r_train = np.mean(r_train)

        # visualizing results
        viz_str = 'epoch:{:}|'.format(epoch)
        viz_str += 'T:{:.1f}|'.format((time.time() - start_time)/60.)
        viz_str += 'l_su:{:.2f}|'.format(np.mean(loss_su)) 
        viz_str += 'l_nll:{:.2f}|'.format(loss_nll)
        viz_str += 'l_l1:{:.3f}|'.format(loss_l1) 
        viz_str += 'acc:{:.2f}|'.format(acc_train) 
        viz_str += 'r:{:.2f}|'.format(r_train)
        viz_str += 'l_da:{:.2f}|'.format(loss_da) 
        viz_str += 'acc_da:{:.2f}|'.format(acc_da) 
        print(viz_str)

    # save model and results
    model_path = join(result_path, 'ckpt_fold')
    classifier.save_weights(model_path)

    for dataset in datasets_valid + datasets_test:
        assignments = classifier.predict(dataset['data'].astype(np.float32))
        probs = classifier.prob(dataset['data'].astype(np.float32))
        prob_code = 0
        for subtype in range(probs.shape[1]):
            prob_code = prob_code * 100 + np.floor(probs[:, subtype] * 100) / 100.
        prob_code = prob_code * 100

        df = pd.DataFrame(data={
                'patients': dataset['patients'],
                'OS': dataset['OS'],
                'status': dataset['status'],
                'DFS': dataset['DFS'],
                'recurrence': dataset['recurrence'],
                'subtype': dataset['subtypes'],
                'assignment': assignments,
                'prob': prob_code
            }
        )
        df.to_csv(join(result_path, '{:}.csv'.format(dataset['cohort'])), index=False)

def sklearn_classifier(datasets, datasets_train, datasets_valid, datasets_test):
    dataset = datasets_train[0]
    x = dataset['data'].astype(np.float32)
    y = dataset['subtypes'].astype(int)
    if 'LogisticRegression' in args.method:
        classifier = LogisticRegression(
            max_iter=1000, 
            C=args.alpha
        ).fit(x, y)
    elif 'RandomForest' in args.method:
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=args.seed
        ).fit(x, y)

    for dataset in datasets_valid + datasets_test:
        assignments = classifier.predict(dataset['data'].astype(np.float32))
        probs = classifier.predict_proba(dataset['data'].astype(np.float32))
        prob_code = 0
        for subtype in range(probs.shape[1]):
            prob_code = prob_code * 100 + np.floor(probs[:, subtype] * 100) / 100.
        prob_code = prob_code * 100

        df = pd.DataFrame(data={
                'patients': dataset['patients'],
                'OS': dataset['OS'],
                'status': dataset['status'],
                'DFS': dataset['DFS'],
                'recurrence': dataset['recurrence'],
                'subtype': dataset['subtypes'],
                'assignment': assignments,
                'prob': prob_code
            }
        )
        df.to_csv(join(result_path, '{:}.csv'.format(dataset['cohort'])), index=False)


def evaluation(km_viz=False):
    results_df = []
    cohorts_methods_dict = {'SH':[], 'GZ':[], 'FZ':[], 'Gao':[]}
    cohorts_methods_best = {'SH':{}, 'GZ':{}, 'FZ':{}, 'Gao':{}}
    for test_cohort in cohorts_methods_dict.keys():
        for method in ['LogisticRegression', 'RandomForest', 'XTypeSuperviseOnly', 'XTypeNoCox', 'XTypeNoCDAN', 'XType']:
            cohort_path = join(args.data_path, 'benchmark', method, 'test-'+test_cohort)
            param_ids = [int(x.split('-')[1]) for x in os.listdir(cohort_path) if '-' in x]
            for param_id in param_ids:
                param_path = join(cohort_path, 'para-{:}'.format(param_id))
                seeds = [int(x.split('-')[1]) for x in os.listdir(param_path) if '-' in x]
                Parallel(n_jobs=args.n_jobs)(delayed(evaluate_a_result)(
                    param_path,
                    param_id, 
                    seed,
                    km_viz) for seed in seeds
                )
            params_result = {'Avg': [], 'Std': []}
            for param_id in param_ids:
                param_seeds = []
                for seed in [int(x.split('-')[1]) for x in os.listdir(param_path) if '-' in x]:
                    seed_path = join(cohort_path, 'para-'+str(param_id), 'seed-'+str(seed))
                    param_df = pd.read_csv(join(seed_path, 'results.csv'), index_col=0)
                    param_seeds.append(param_df)
                param_seeds = pd.concat(param_seeds)
                params_result['Avg'].append(param_seeds.mean(axis=0))
                params_result['Std'].append(param_seeds.std(axis=0))
            avg = pd.concat(params_result['Avg'], axis=1).T
            std = pd.concat(params_result['Std'], axis=1).T
            headers = list(avg.columns.values)
            best_id = avg['avg_Valid_dRMST'].argmax()
            cohorts_methods_best[test_cohort][method] = best_id
            method_cohort_dict = {}
            for _, header in enumerate(headers):
                method_cohort_dict[header] = ['{:.3f}±{:.3f}'.format(avg[header][best_id], std[header][best_id])]
            method_cohort_df = pd.DataFrame(method_cohort_dict, index=[method])
            cohorts_methods_dict[test_cohort].append(method_cohort_df)

        results_df = pd.concat(cohorts_methods_dict[test_cohort])
        results_df = results_df[[col for col in list(results_df) if ('test_' in col) and ('dRMST' not in col) or ('Accuracy' in col)]]
        results_df.to_csv(join(args.data_path, 'benchmark', 'results_{:}.csv'.format(test_cohort)))
        cohorts_methods_dict[test_cohort] = results_df

    print('Best param_id:\n', cohorts_methods_best)

    barplot_benchmark(
        cohorts_methods_dict, 
        fig_size=(8, 4), 
        save_path=join(args.data_path, 'benchmark')
    )

    # compare the prediction accuracy on Jiang et al.'s cohort
    barplot_jiang_acc(
        results_df,
        fig_size=(8, 4), 
        save_path=join(args.data_path, 'benchmark')
    )

    # mean pair-wise similarity among random seeds
    print('Mean pair-wise prediction similarity of XType')
    for test_cohort in cohorts_methods_dict.keys():
        cohort_path = join(args.data_path, 'benchmark', 'XType', 'test-'+test_cohort)
        param_path = join(cohort_path, 'para-{:}'.format(cohorts_methods_best[test_cohort]['XType']))
        seeds = [int(x.split('-')[1]) for x in os.listdir(param_path) if '-' in x]
        seeds_subtype = []
        for seed in seeds:
            group_df = pd.read_csv(join(param_path, 'seed-'+str(seed), '{:}.csv'.format(test_cohort))) 
            seeds_subtype.append(group_df['assignment'].to_numpy())
        sim = []
        for i in range(len(seeds)):
            for j in range(i+1, len(seeds)):
                sim.append(np.mean(seeds_subtype[i] == seeds_subtype[j]))
        print('{:} cohort: {:.4f}'.format(test_cohort, np.mean(sim)))

    # comparison between XType and RandomForest, using the best param_id and compare the simlarity on corresponding random seeds
    print('Mean prediction similarity between XType and RandomForest')
    for test_cohort in cohorts_methods_dict.keys():
        methods_seeds_subtype = []
        for method in ['XType', 'RandomForest']:
            seeds_subtype = []
            cohort_path = join(args.data_path, 'benchmark', method, 'test-'+test_cohort)
            param_path = join(cohort_path, 'para-{:}'.format(cohorts_methods_best[test_cohort][method]))
            seeds = [int(x.split('-')[1]) for x in os.listdir(param_path) if '-' in x]
            for seed in seeds:
                group_df = pd.read_csv(join(param_path, 'seed-'+str(seed), '{:}.csv'.format(test_cohort))) 
                seeds_subtype.append(group_df['assignment'].to_numpy())
            methods_seeds_subtype.append(seeds_subtype)
        sim = []
        for i in range(len(seeds)):
            sim.append(np.mean(methods_seeds_subtype[0][i] == methods_seeds_subtype[1][i]))
        print('{:} cohort: {:.4f}'.format(test_cohort, np.mean(sim)))

def evaluate_a_result(param_path, param_id, seed, km_viz=False):
    curr_path = join(param_path, 'seed-'+str(seed))
    # get data frames of all groups
    group_df_dict = collections.OrderedDict()
    identifier = '.csv'
    groups = sorted([x[:-len(identifier)] for x in os.listdir(curr_path) if identifier in x and 'results' not in x and 'GSEA' not in x])
    # calculate metrics for each test cohort
    train_dict, valid_dict, test_dict = {}, {}, {}
    for group in groups:
        group_df = pd.read_csv(join(curr_path, '{:}.csv'.format(group))) 
        assignments = group_df['assignment'].to_numpy()
        prob_codes = group_df['prob'].to_numpy()
        subtype_num = 3
        prob_codes[prob_codes == 100**subtype_num] = 100**subtype_num - 1
        prob_subtypes = []
        for subtype in range(subtype_num):
            prob_subtypes.append((prob_codes % 100) / 100.)
            prob_codes = np.floor(prob_codes / 100.)
        probs = np.stack(prob_subtypes[::-1], axis=1)
        probs = probs / np.tile(np.sum(probs, axis=1, keepdims=True), [1, subtype_num])
        if 'Jiang' in group:
            subtypes = group_df['subtype'].to_numpy()
            acc = np.mean(assignments[subtypes!=-1] == subtypes[subtypes!=-1])
            valid_dict['valid_Jiang_Accuracy'] = acc

        for time_name, event_name in zip(['OS', 'DFS'], ['status', 'recurrence']):
            times = group_df[time_name].to_numpy()
            events = group_df[event_name].to_numpy()
            p = utils.multivariate_logrank_test(times, events, assignments)
            dRMST = utils.get_delta_rmst(times, events, assignments, 3)
            c_index = concordance_index(times, probs[:, 0] - probs[:, 2], events)
            if 'Jiang' in group:
                prefix = group + '_' + time_name
                test_dict[prefix + '_Log-rank score'] = -np.log10(p)
                test_dict[prefix + '_dRMST'] = np.mean(dRMST)
                test_dict[prefix + '_C-Index'] = c_index
            else:
                prefix = 'test' + '_' + group + '_' + time_name
                test_dict[prefix + '_Log-rank score'] = -np.log10(p)
                test_dict[prefix + '_dRMST'] = np.mean(dRMST)
                test_dict[prefix + '_C-Index'] = c_index

                if km_viz:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    title = group + '_' + time_name 
                    try:
                        plot_km_curve_custom(
                            times, 
                            events, 
                            assignments, 
                            3,
                            ax, 
                            title=title + '_acc{:.2f}'.format(acc) if 'Jiang' in group else title,
                            text_bias=30,
                            clip_time=60
                        )
                        fig.tight_layout()
                        plt.savefig(join(curr_path, 'km_' + title + '.png'))
                        plt.close()
                    except:
                        print('km viz failed')

    results_dict = {**train_dict, **valid_dict, **test_dict}
    for metric in ['_dRMST', '_Log-rank score']:
        results_dict['avg_Valid' + metric] = np.mean([v for k, v in results_dict.items() if 'valid_' in k and metric in k])
    results_df = pd.DataFrame(data=results_dict, index=[param_id])
    results_df.to_csv(join(curr_path, 'results.csv'))

if __name__ == '__main__':
    start_time = time.time()

    datasets = utils.load_dataset(args.data_path)
    datasets = utils.dataset_preprocess(
        datasets, 
        data_path=args.data_path,
        log2=args.log2, 
        zscore=args.zscore, 
        min_impute=args.min_impute, 
        feat_select=args.feat_select, 
        test_cohort=args.test_cohort,
        clip=60
    )

    np.random.seed(args.seed)
    datasets = k_fold_split(
        datasets, 5, 4, False
    )
    
    if args.test_cohort == 'SH':
        cohorts = 'Jiang_GZ_FZ_SH_Gao'.split('_')
    elif args.test_cohort == 'GZ':
        cohorts = 'Jiang_SH_FZ_GZ_Gao'.split('_')
    elif args.test_cohort == 'FZ':
        cohorts = 'Jiang_SH_GZ_FZ_Gao'.split('_')
    elif args.test_cohort == 'Gao':
        cohorts = 'Jiang_SH_GZ_Gao_FZ'.split('_')
    train_valid_test_groups = {
        'train':
        [
            '{:}_train'.format(cohorts[0]),
            '{:}_train'.format(cohorts[1]), 
            '{:}_train'.format(cohorts[2]),
            '{:}_train'.format(cohorts[4])
        ],

        # validation sets
        'valid':
        [
            '{:}_{:}_{:}_{:}_valid'.format(cohorts[0], cohorts[1], cohorts[2], cohorts[4])
        ],


        # test sets
        'test':
        [
            '{:}'.format(cohorts[3])
        ]
    }
    datasets_train, datasets_valid, datasets_test = datasets_regroup(datasets, train_valid_test_groups)

    if not args.evaluate:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.threading.set_intra_op_parallelism_threads(10)
        tf.config.threading.set_inter_op_parallelism_threads(10)
        tf.random.set_seed(args.seed)
        if 'XType' in args.method:
            with tf.device(args.device):
                real_xtype(datasets, datasets_train, datasets_valid, datasets_test)
                tf.keras.backend.clear_session()
        else:
            sklearn_classifier(datasets, datasets_train, datasets_valid, datasets_test)
    else:
        evaluation(args.km_viz)
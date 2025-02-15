import os, copy, time, argparse, platform, pickle, collections, random
from os.path import join

sys = platform.system()
parser = argparse.ArgumentParser(description='Process some parameters.')

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
parser.add_argument('--step_num', type=int, default=2000)
parser.add_argument('--b_size', type=int, default=128)
parser.add_argument('--alpha', type=float, default=0.1)

# settings
parser.add_argument('--para_id', type=int, default=0)
parser.add_argument('--method', default='XType')
parser.add_argument('--device', default='/cpu')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--valid_fold', type=int, default=0)
parser.add_argument('--data_path', default='data')
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--n_jobs', type=int, default=5)

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
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
from joblib import Parallel, delayed

import data_utils as utils
from network_model.cox_cdan_model import *
from viz_utils import *

result_path = join(
    experiment_path,
    'para-'+str(args.para_id),
    'seed-'+str(args.seed),
)

def fetch_data(df):
    x = df['proteome'].to_numpy().astype(np.float32)
    t1 = df[('meta', 'OS')].to_numpy().astype(np.float32)
    e1 = df[('meta', 'status')].to_numpy().astype(np.float32)
    t2 = df[('meta', 'DFS')].to_numpy().astype(np.float32)
    e2 = df[('meta', 'recurrence')].to_numpy().astype(np.float32)
    return x, t1, e1, t2, e2

def real_xtype(train_df, all_df):
    classifier, optimizer = make_simple_models(
        args.h_dim, 3, 3, args.lr, args.regu, args.activ, args.bias
    )

    start_time = time.time()
    # note that we used a 5-fold cross validation (valid_fold = 0~4)
    # valid_fold==5 means we are training the final model with all training samples
    train_fold_df = train_df[train_df[('meta', 'fold')] != args.valid_fold]
    shuffled_df = train_fold_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    for epoch in range(args.step_num):

        # supervised learning on training samples from Jiang 
        sub_df = shuffled_df[shuffled_df[('meta', 'cohort')] == 'Jiang']
        df_batch = sub_df.sample(n=args.b_size, random_state=args.seed) if args.b_size < sub_df.shape[0] else sub_df
        x1 = df_batch['proteome'].to_numpy().astype(np.float32)
        y1 = df_batch[('meta', 'subtype')].to_numpy().astype(np.int32)
        data_subtype = (x1, y1)

        # cox regression on training samples from SH and GZ
        sub_df = shuffled_df[shuffled_df[('meta', 'cohort')].isin(['SH', 'GZ'])]
        df_batch = sub_df.sample(n=args.b_size, random_state=args.seed) if args.b_size < sub_df.shape[0] else sub_df
        x21, t11, e11, t12, e12 = fetch_data(df_batch)
        data_cox = [(x21, t11, e11, t12, e12)]

        # domain adversarial learning on all training samples
        df_batch = shuffled_df.sample(n=args.b_size, random_state=args.seed) if args.b_size < shuffled_df.shape[0] else shuffled_df
        x3 = df_batch['proteome'].to_numpy().astype(np.float32)
        y3 = df_batch[('meta', 'cohort')].map({'Jiang': 0, 'SH': 1, 'GZ': 2}).to_numpy().astype(np.int32)
        data_da = (x3, y3)

        loss_su, loss_nll, loss_da, loss_l1, logits = train_step(
            data_subtype, data_cox, data_da,
            classifier, optimizer, 
            [args.loss_su, args.loss_nll, args.loss_da, args.loss_var],
            args.dropout,
            args.alpha
        )
        shuffled_df = shuffled_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # save model
    model_path = join(result_path, 'model_weights')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    classifier.save_weights(join(model_path, f'ckpt_fold-{args.valid_fold:}'))

    # save results
    x = all_df['proteome'].to_numpy().astype(np.float32)
    assignments = classifier.predict(x)
    probs = classifier.prob(x)
    prob_code = 0
    for subtype in range(probs.shape[1]):
        prob_code = prob_code * 100 + np.floor(probs[:, subtype] * 100) / 100.
    prob_code = prob_code * 100

    result_file = join(result_path, 'results.csv')
    if not os.path.exists(result_file):
        results_df = all_df.loc[:, all_df.columns.get_level_values(0) == 'meta']
        results_df.columns = results_df.columns.droplevel(0)
        train_df_ = train_df.copy()
        train_df_.columns = train_df_.columns.droplevel(0)
        results_df = results_df.merge(train_df_[['patient', 'fold']], on='patient', how='left')
        results_df['fold'] = results_df['fold'].fillna(-1)
    else:
        results_df = pd.read_csv(result_file)

    results_df[f'assignment_fold-{args.valid_fold:}'] = assignments.astype(int)
    results_df[f'prob_fold-{args.valid_fold:}'] = prob_code.astype(int)
    results_df.to_csv(result_file, index=False)

def sklearn_classifier(train_df, all_df):
    train_fold_df = train_df[train_df[('meta', 'fold')] != args.valid_fold]
    sub_df = train_fold_df[train_fold_df[('meta', 'cohort')] == 'Jiang']
    x = sub_df['proteome'].to_numpy().astype(np.float32)
    y = sub_df[('meta', 'subtype')].to_numpy().astype(np.int32)
    if 'LogisticRegression' in args.method:
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(
            max_iter=1000, 
            C=args.alpha
        ).fit(x, y)
    elif 'RandomForest' in args.method:
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=args.seed
        ).fit(x, y)
    elif 'XGBoost' in args.method:
        from xgboost.sklearn import XGBClassifier
        classifier = XGBClassifier(seed=args.seed)
        classifier.fit(x, y)

    # save results
    x = all_df['proteome'].to_numpy().astype(np.float32)
    assignments = classifier.predict(x)
    probs = classifier.predict_proba(x)
    prob_code = 0
    for subtype in range(probs.shape[1]):
        prob_code = prob_code * 100 + np.floor(probs[:, subtype] * 100) / 100.
    prob_code = prob_code * 100

    result_file = join(result_path, 'results.csv')
    if not os.path.exists(result_file):
        results_df = all_df.loc[:, all_df.columns.get_level_values(0) == 'meta']
        results_df.columns = results_df.columns.droplevel(0)
        train_df_ = train_df.copy()
        train_df_.columns = train_df_.columns.droplevel(0)
        results_df = results_df.merge(train_df_[['patient', 'fold']], on='patient', how='left')
        results_df['fold'] = results_df['fold'].fillna(-1)
    else:
        results_df = pd.read_csv(result_file)

    results_df[f'assignment_fold-{args.valid_fold:}'] = assignments.astype(int)
    results_df[f'prob_fold-{args.valid_fold:}'] = prob_code.astype(int)
    results_df.to_csv(result_file, index=False)


def evaluation():
    results_df = []
    methods = ['LogisticRegression', 'RandomForest', 'XGBoost', 'DNN', 'XTypeNoCox', 'XTypeNoCDAN', 'XType']
    metrics_methods_seeds = []
    metrics_methods_pval = np.zeros((5, 5, len(methods) - 1))

    methods_metircs = []
    methods_best = {}
    methods_best_metircs = []
    methods_best_vals = []
    for method in methods:
        method_path = join(args.data_path, 'benchmark', method)
        param_ids = [int(x.split('-')[1]) for x in os.listdir(method_path) if '-' in x]
        for param_id in param_ids:
            param_path = join(method_path, 'para-{:}'.format(param_id))
            seeds = [int(x.split('-')[1]) for x in os.listdir(param_path) if '-' in x]
            Parallel(n_jobs=args.n_jobs)(delayed(evaluate_a_result)(
                param_path,
                param_id, 
                seed) for seed in seeds
            )

        params_result = {'Avg': [], 'Std': [], 'Vals': []}
        params_seeds_metrics = []
        for param_id in param_ids:
            seeds_metics = []
            for seed in [int(x.split('-')[1]) for x in os.listdir(param_path) if '-' in x]:
                seed_path = join(method_path, 'para-'+str(param_id), 'seed-'+str(seed))
                param_df = pd.read_csv(join(seed_path, 'metrics.csv'), index_col=0)
                seeds_metics.append(param_df)
            seeds_metics = pd.concat(seeds_metics)
            params_seeds_metrics.append(seeds_metics.to_numpy())
            seeds_combined = seeds_metics.apply(lambda col: "|".join([f"{value:.3f}" for value in col]), axis=0)
            seeds_combined_df = pd.DataFrame([seeds_combined.values], columns=[f"{col}" for col in seeds_combined.index], index=[method])
            params_result['Avg'].append(seeds_metics.mean(axis=0))
            params_result['Std'].append(seeds_metics.std(axis=0))
            params_result['Vals'].append(seeds_combined_df)
        avg = pd.concat(params_result['Avg'], axis=1).T
        std = pd.concat(params_result['Std'], axis=1).T
        vals = pd.concat(params_result['Vals'], axis=1).T
        headers = list(avg.columns.values)
        best_id = avg['SH_GZ_cv-valid_DFS_Log-rank score'].argmax()
        methods_best[method] = best_id
        methods_best_vals.append(params_result['Vals'][best_id])

        method_best_dict = {}
        method_seeds_dict = {}
        for _, header in enumerate(headers):
            method_best_dict[header] = ['{:.3f}_{:.3f}'.format(avg[header][best_id], std[header][best_id])]
        methods_best_metircs.append(pd.DataFrame(method_best_dict, index=[method+'_'+str(best_id)]))

        method_dict = {}
        for param_id in param_ids:
            for _, header in enumerate(headers):
                method_dict[header] = ['{:.3f}_{:.3f}'.format(avg[header][param_id], std[header][param_id])]
            methods_metircs.append(pd.DataFrame(method_dict, index=[method+'_'+str(param_id)]))

    results_df = pd.concat(methods_metircs)
    results_df.to_csv(join(args.data_path, 'benchmark', 'results.csv'))

    best_results_df = pd.concat(methods_best_metircs)
    best_results_df.to_csv(join(args.data_path, 'benchmark', 'best_results.csv'))

    methods_best_vals = pd.concat(methods_best_vals)
    methods_best_vals.to_csv(join(args.data_path, 'benchmark', 'best_vals.csv'))


def evaluate_a_result(param_path, param_id, seed):
    curr_path = join(param_path, 'seed-'+str(seed))
    df = pd.read_csv(join(curr_path, 'results.csv'))
    df[["assignment", "prob"]] = df.apply(
        lambda row: pd.Series(utils.process_results(row, mode='retrain')), axis=1
    )

    metric_dict = {}
    for group in ['Jiang', 'SH_GZ_cv-valid', 'SH_GZ_ex-valid', 'FZ', 'Gao']:
        group_df = utils.filter_patients(df, group)
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
            metric_dict['Jiang_Accuracy'] = acc
        else:
            for time_name, event_name in zip(['OS', 'DFS'], ['status', 'recurrence']):
                times = group_df[time_name].to_numpy()
                events = group_df[event_name].to_numpy()
                p = utils.multivariate_logrank_test(times, events, assignments)
                c_index = concordance_index(times, probs[:, 0] - probs[:, 2], events)
                prefix = group + '_' + time_name
                metric_dict[prefix + '_Log-rank score'] = -np.log10(p)
                metric_dict[prefix + '_C-Index'] = c_index

    results_df = pd.DataFrame(data=metric_dict, index=[param_id])
    results_df.to_csv(join(curr_path, 'metrics.csv'))

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()

    df = pd.read_csv(join(args.data_path, 'HCC_datasets.csv'), header=[0, 1])

    # get train_df
    train_df = df[(df[('meta', 'ex_valid')] != 1) & (df[('meta', 'cohort')].isin(['Jiang', 'SH', 'GZ']))]

    # Initialize StratifiedKFold with the determined number of splits and shuffle
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # Adding a 'fold' column to the DataFrame
    train_df.loc[:, ('meta', 'fold')] = -1

    # Assigning fold numbers with stratification based on cohort
    for fold, (_, test_index) in enumerate(skf.split(train_df, train_df[('meta', 'cohort')])):
        train_df.iloc[test_index, train_df.columns.get_loc(('meta', 'fold'))] = fold

    # Getting column location for 'ex_valid' and inserting 'fold' after it
    columns = list(train_df.columns)
    fold_index = columns.index(('meta', 'ex_valid')) + 1

    # Reordering columns to place 'fold' right after 'ex_valid'
    columns.insert(fold_index, columns.pop(columns.index(('meta', 'fold'))))
    train_df = train_df[columns]
    print('training samples (Jiang + 70%SH + 70%GZ) N=', train_df.shape[0]) 

    if not args.evaluate:
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if args.method in ['DNN', 'XTypeNoCDAN', 'XTypeNoCox', 'XType']:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.threading.set_intra_op_parallelism_threads(10)
            tf.config.threading.set_inter_op_parallelism_threads(10)
            tf.random.set_seed(args.seed)
            with tf.device(args.device):
                real_xtype(train_df, df)
                tf.keras.backend.clear_session()
        elif args.method in ['LogisticRegression', 'XGBoost', 'RandomForest']:
            sklearn_classifier(train_df, df)

    elif args.evaluate:
        evaluation()
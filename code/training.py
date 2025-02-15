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
parser.add_argument('--test_cohort', type=str, default='FZ_Gao')
parser.add_argument('--para_id', type=int, default=0)
parser.add_argument('--method', default='XType')
parser.add_argument('--device', default='/cpu')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--valid_fold', type=int, default=0)
parser.add_argument('--data_path', default='data')

args = parser.parse_args()

experiment_path = join(
    args.data_path, 
    'application', 
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

import data_utils as utils
from network_model.cox_cdan_model import *
from viz_utils import *

result_path = join(
    experiment_path,
    'test-'+str(args.test_cohort),
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

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(10)
    tf.config.threading.set_inter_op_parallelism_threads(10)
    tf.random.set_seed(args.seed)
    with tf.device(args.device):
        real_xtype(train_df, df)
        tf.keras.backend.clear_session()
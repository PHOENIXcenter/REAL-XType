import os, copy, argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import progressbar
from joblib import Parallel, delayed
from os.path import join

from viz_utils import *
from para_space import get_paras
import data_utils as utils

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--test_cohort', type=str, default='FZ_Gao')
parser.add_argument('--para_id', type=int, default=0)
parser.add_argument('--method', default='XType')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_jobs', type=int, default=10)
parser.add_argument('--data_path', default='data')

args = parser.parse_args()


experiment_path = join(
    args.data_path, 
    'application', 
    args.method
)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

args, comb_dict = get_paras(args, join(experiment_path, 'param_list.csv'))

result_path = join(
    experiment_path,
    'test-'+str(args.test_cohort),
    'para-'+str(args.para_id),
    'seed-'+str(args.seed),
)

def evaluation():
    param_list_df = pd.read_csv(join(experiment_path, 'param_list.csv'), index_col=0)
    param_dfs = []
    test_cohort = 'FZ_Gao'

    metric_file = 'metrics.csv'
    if not os.path.exists(join(experiment_path, 'test-'+test_cohort, metric_file)):
        # save results for all params
        param_ids = [int(x.split('-')[1]) for x in os.listdir(join(experiment_path, 'test-'+test_cohort)) if '-' in x]
        
        Parallel(n_jobs=args.n_jobs)(delayed(evaluate_a_result)(
            experiment_path,
            test_cohort,
            param_id, 
            args.seed) for param_id in progressbar.progressbar(param_ids)
        )

        # read results for all params
        for param_id in param_ids:
            seed_path = join(experiment_path, 'test-'+test_cohort, 'para-'+str(param_id), 'seed-'+str(args.seed))
            param_df = pd.read_csv(join(seed_path, 'metrics.csv'), index_col=0)
            param_dfs.append(param_df)

        results_df = pd.concat(param_dfs)
        sub_results_df = results_df[[col for col in list(results_df) if 'Jiang_' not in col]]
        params_results_df = pd.merge(param_list_df, sub_results_df, left_index=True, right_index=True)
        params_results_df.to_csv(join(experiment_path, 'test-'+test_cohort, metric_file))
    else:
        params_results_df = pd.read_csv(join(experiment_path, 'test-'+test_cohort, metric_file), index_col=0)

    params_results_df.reset_index(inplace=True)
    params_results_df.rename(columns={'index': 'param_id'}, inplace=True)
    eval_metric = 'SH_GZ_cv-valid_DFS_Log-rank score'

    best_param_id = np.argmax(params_results_df[eval_metric])
    viz_a_result(experiment_path, test_cohort, best_param_id, args.seed)

def evaluate_a_result(experiment_path, test_cohort, param_id, seed):
    curr_path = join(experiment_path, 'test-'+test_cohort, 'para-'+str(param_id), 'seed-'+str(seed))
    df = pd.read_csv(join(curr_path, 'results.csv'))

    df[["assignment", "prob"]] = df.apply(
        lambda row: pd.Series(utils.process_results(row, mode='retrain')), axis=1
    )
    df.to_csv(join(curr_path, 'results_processed.csv'), index=False)

    metric_dict = {}
    for group in ['SH_GZ_cv-valid',]:
        group_df = utils.filter_patients(df, group)
        assignments = group_df['assignment'].to_numpy()

        for time_name, event_name in zip(['OS', 'DFS'], ['status', 'recurrence']):
            times = group_df[time_name].to_numpy()
            events = group_df[event_name].to_numpy()

            p = utils.multivariate_logrank_test(times, events, assignments)
            dRMST = utils.get_delta_rmst(times, events, assignments, 3)
            metric_dict[group + '_' + time_name + '_Log-rank score'] = -np.log10(p)
            metric_dict[group + '_' + time_name + '_dRMST-avg'] = np.mean(dRMST)

    save_path = join(experiment_path, 'test-'+test_cohort, 'para-'+str(param_id), 'seed-'+str(seed))
    results_df = pd.DataFrame(data=metric_dict, index=[param_id])
    results_df.to_csv(join(save_path, 'metrics.csv'))

def viz_a_result(experiment_path, test_cohort, param_id, seed):
    result_path = join(experiment_path, 'test-'+test_cohort, 'para-'+str(param_id), 'seed-'+str(seed))

    df = pd.read_csv(join(result_path, 'results_processed.csv'))

    save_path = join(experiment_path, 'results_viz', 'para-'+str(param_id), 'seed-'+str(seed))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_csv(join(save_path, 'results_processed.csv'), index=False)

    for group in [        
        'SH_GZ_cv-valid', 'SH_GZ_ex-valid', 'FZ', 'Gao', 'SH_GZ_FZ_low-risk'
    ]:
        group_df = utils.filter_patients(df, group)
        assignment_col = 'assignment_fold-5' if 'low-risk' in group else 'assignment'
        assignments = group_df[assignment_col].to_numpy()

        for time_name, event_name in zip(['OS', 'DFS'], ['status', 'recurrence']):

            times = group_df[time_name].to_numpy()
            events = group_df[event_name].to_numpy()
            fig, ax = plt.subplots(figsize=(4, 4))
            title = group + '_' + time_name 

            if 'low-risk' in group: 
                n_subtype = 2 
                label_dict = {0:'Non-R-III', 1: 'R-III'}
                temp_assign = copy.deepcopy(assignments)
                temp_assign[temp_assign < 2] = 0
                temp_assign[temp_assign == 2] = 1
                text_bias = 35
            else:
                n_subtype = 3 
                label_dict = {0: 'R-I', 1: 'R-II', 2: 'R-III'}
                temp_assign = copy.deepcopy(assignments)
                text_bias = 30

            try:
                plot_km_curve_custom(
                    times, 
                    events, 
                    temp_assign, 
                    n_subtype,
                    ax, 
                    title=title + '_acc{:.2f}'.format(acc) if 'Jiang' in group else title,
                    text_bias=text_bias,
                    clip_time=60,
                    label_dict=label_dict
                )
                fig.tight_layout()
                plt.savefig(join(save_path, 'km_' + title + '.png'))
                plt.close()
            except:
                print(group, time_name, 'km viz failed')

if __name__ == '__main__':
    evaluation()
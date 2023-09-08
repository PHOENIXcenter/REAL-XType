import os, time, progressbar
import numpy as np
import pandas as pd
import gsea_utils as gsea_utils
import data_utils as data_utils
import viz_utils as viz_utils
import matplotlib.pyplot as plt

from lifelines import statistics
from lifelines.utils import concordance_index
from joblib import Parallel, delayed
from os.path import join
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

def oneline_eval(datasets, classifier, eval_str, subtype_num, show_reward=False):
    for domain, dataset in enumerate(datasets):
        x = dataset['data'].astype(np.float32)
        OS = dataset['OS']
        status = dataset['status']
        DFS = dataset['DFS']
        recurrence = dataset['recurrence']
        a = classifier.predict(x)

        if 'subtypes' in dataset.keys():
            acc = int(np.mean(a==dataset['subtypes']) * 100)
            eval_str = eval_str + 'acc_D' + str(domain) + '={:d}% '.format(acc)

        r = (data_utils.get_reward(OS, status, a, subtype_num) + 
            data_utils.get_reward(DFS, recurrence, a, subtype_num)
        )/2
        r = r[0]
        if show_reward:
            eval_str = eval_str + 'r_D' + str(domain) + '={:.1f} '.format(r)

    return eval_str, r

def test_and_save_results(datasets, classifier, result_path, test_fold, all_assignments=None):
    all_patients = []
    all_OS = np.concatenate([dataset['OS'] for dataset in datasets], axis=0)
    all_status = np.concatenate([dataset['status'] for dataset in datasets], axis=0)
    all_DFS = np.concatenate([dataset['DFS'] for dataset in datasets], axis=0)
    all_recurrence = np.concatenate([dataset['recurrence'] for dataset in datasets], axis=0)
    all_cohort = np.concatenate(
        [np.ones_like(dataset['OS']) * i for i, dataset in enumerate(datasets)], axis=0
    )
    all_subtype = np.concatenate(
        [dataset['subtypes'] if 'subtypes' in dataset.keys() 
            else np.ones_like(dataset['OS']) * -1 for i, dataset in enumerate(datasets)], 
        axis=0
    )

    if all_assignments is None:
        all_assignments = np.concatenate(
            [classifier.predict(dataset['data'].astype(np.float32)) for dataset in datasets], axis=0
        ) 

        all_probs = []
        for dataset in datasets:
            prob_code = 0
            probs = classifier.predict_proba(dataset['data'].astype(np.float32))
            for subtype in range(probs.shape[1]):
                prob_code = prob_code * 100 + np.floor(probs[:, subtype] * 100) / 100.
            all_probs.append(prob_code*100)
        all_probs = np.concatenate(all_probs, axis=0)
    else:
        probs = np.zeros((all_assignments.size, all_assignments.max()+1))
        probs[np.arange(all_assignments.size), all_assignments] = 0.99
        prob_code = 0
        for subtype in range(probs.shape[1]):
            prob_code = prob_code * 100 + np.floor(probs[:, subtype] * 100) / 100.
        all_probs = prob_code * 100

    for dataset in datasets:
       all_patients += dataset['patients'] 

    df = pd.DataFrame(data={
            'patients': all_patients,
            'OS': all_OS,
            'status': all_status,
            'DFS': all_DFS,
            'recurrence': all_recurrence,
            'cohort': all_cohort,
            'assignment': all_assignments,
            'prob': all_probs,
            'label': all_subtype
        }
    )
    df.to_csv(join(result_path, 'fold-' + str(test_fold) + '.csv'), index=False)
    test_str = 'Acc = '
    for i in range(len(datasets)):
        logic = np.logical_and(all_subtype >= 0, all_cohort==i)
        test_str += '{:.3f} | '.format(np.mean(all_subtype[logic] == all_assignments[logic]))
    print(test_str)

def evaluate_results(result_path, datasets, nature_pathway2name_and_type_dict, plot, max_t, data_path, data):
    start_time = time.time()
    fold_num = len([x for x in os.listdir(result_path) if 'fold-' in x])
    df = pd.concat([pd.read_csv(join(result_path, 'fold-' + str(fold) + '.csv')) for fold in range(fold_num)])
    subtype_num = int(np.amax(datasets[0]['subtypes']) + 1)

    w, h = 4, 4
    results_dict = {} 

    # functional similarity
    # print(result_path)
    if 'Jiang' in result_path or 'Xu' in result_path:
        sim_list = gsea_utils.ssGSEA_similarity(
            datasets,
            result_path,
            nature_pathway2name_and_type_dict
        ) 
    else: 
        sim_list = None

    for dataset_id, dataset in enumerate(datasets):
        dataset_label = dataset['label']
        dataset_df = df.loc[df['cohort'] == dataset_id]
        assignments = dataset_df['assignment'].to_numpy()
        if 'prob' in list(dataset_df.columns.values):
            prob_codes = dataset_df['prob'].to_numpy()
        else:
            prob_codes = np.ones_like(assignments) * 99
        prob_codes[prob_codes == 100**subtype_num] = 100**subtype_num - 1

        # acc and functional similarity
        if dataset_id > 0 and sim_list is not None:
            results_dict[dataset_label + '_ssGSEA_Similarity'] = sim_list[dataset_id - 1]
        
        if 'subtypes' in dataset.keys():
            labels = dataset_df['label'].to_numpy()
            # ACC
            acc = np.mean(assignments == labels)
            results_dict[dataset_label + '_Accuracy'] = acc 

        life_start_time = time.time()
        for time_name, event_name in zip(['OS', 'DFS'], ['status', 'recurrence']):
            km_start_time = time.time()
            times = dataset_df[time_name].to_numpy()
            events = dataset_df[event_name].to_numpy()

            if dataset_id >= 0:
                # logrank p value
                logrank_start_time = time.time()
                logrank_results = statistics.multivariate_logrank_test(times, assignments, events)
                p = logrank_results.p_value + 1e-100
                results_dict[dataset_label + '_' + time_name + '_Log-rank score'] = -np.log10(p)
                # print('logrank:', time.time() - logrank_start_time)

    result_df = pd.DataFrame(data=results_dict,index=[0])
    result_df.to_csv(join(result_path, 'results.csv'))

def metric_table_seed(datasets, save_path, pathway2gene_dict, plot, max_t, data_path, data):
    method_list = sorted([x for x in os.listdir(save_path) if '.' not in x])
    df_list = []
    for method in method_list:
        result_path = join(save_path, method)
        # print(result_path)
        evaluate_results(result_path, datasets, pathway2gene_dict, plot, max_t, data_path, data)

        df = pd.read_csv(join(result_path, 'results.csv'))
        new_cols = []
        for col in list(df.columns.values):
            if 'Accuracy' in col:
               new_cols.append(col)
        for col in list(df.columns.values):
            if 'Similarity' in col:
               new_cols.append(col)
        for col in list(df.columns.values):
            if 'Log-rank score' in col:
               new_cols.append(col)
        df = df[new_cols]
        df_list.append(df)
    df = pd.concat(df_list)
    table_dict = {'method': method_list}
    arr = df.to_numpy()
    for j, col in enumerate(new_cols):
        table_col_list = []
        for i in range(arr.shape[0]):
            table_elem = ('{:.3f}').format(arr[i, j]) 
            table_col_list.append(table_elem)
        table_dict[col] = table_col_list
    df = pd.DataFrame(data=table_dict)
    df.to_csv(join(save_path, 'compare_table.csv'), encoding='utf-8-sig')
    return arr, method_list, new_cols

def metric_table(datasets, data_path, experiment, data, plot, max_t):
    save_path = join(data_path, experiment)
    print(save_path)
    seed_list = ['seed' + str(seed) for seed in range(5)]
    pathway2gene_dict, _, _ = gsea_utils.read_nature_pathway(join(data_path, 'r_data'))

    res = Parallel(n_jobs=len(seed_list))(delayed(metric_table_seed)(
            datasets, 
            join(save_path, 'seed' + str(seed)), 
            pathway2gene_dict, 
            plot, 
            max_t,
            data_path, 
            data
        ) for seed in range(len(seed_list))
    )
    arr_list = [item[0] for item in res]
    param_list = res[0][1]
    new_cols = res[0][2]
    
    aug_arr = np.stack(arr_list, axis=-1)
    mean_arr = np.mean(aug_arr, axis=-1)
    std_arr = np.std(aug_arr, axis=-1)
    methods = [method.split('_')[0] for method in param_list]
    table_dict = {'method': methods, 'param': param_list}
    for j, col in enumerate(new_cols):
        table_col_list = []
        for i in range(mean_arr.shape[0]):
            # table_elem = ('{:.3f}\n±{:.3f}').format(mean_arr[i, j], std_arr[i, j]) 
            table_elem = ('{:.3f}').format(mean_arr[i, j]) 
            table_col_list.append(table_elem)
        table_dict[col] = table_col_list
    df = pd.DataFrame(data=table_dict)
    df.to_csv(join(save_path, 'param_table.csv'), encoding='utf-8-sig')
    print(df)
import csv, re, copy, os, time, itertools, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
from os.path import join
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from lifelines import KaplanMeierFitter, statistics, plotting
from lifelines.utils import restricted_mean_survival_time
from scipy import stats
from data_regroups import *

def load_dataset(data_path):
    dataset_df = pd.read_csv(join(data_path, 'HCC_datasets_0815.csv'), index_col=0)
    cohorts = ['Jiang', 'SH', 'GZ', 'FZ', 'Gao']
    datasets = []
    for cohort in cohorts:
        df = dataset_df[dataset_df['cohort']==cohort]
        datasets.append({
            'data': df.iloc[:, 10:].to_numpy(),
            'protes': list(df)[10:],
            'patients': df['patient'].tolist(),
            'subtypes': df['subtype'].to_numpy(),
            'OS': df['OS'].to_numpy(),
            'status': df['status'].to_numpy(),
            'DFS': df['DFS'].to_numpy(),
            'recurrence': df['recurrence'].to_numpy(),
            'risk': df['risk'].to_numpy(),
            'bclc': df['bclc'].to_numpy(),
            'hbv': df['hbv'].to_numpy(),
            'cohort': cohort
        })
    return datasets

def dataset_preprocess(datasets, data_path, log2, zscore, min_impute, feat_select, test_cohort, clip=60):
    if feat_select == 'f1269':
        feats = read_prior_protein_list(
            join(data_path, '2017-04-04932E-Supplementary Table 11.xlsx'),
            1.
        )
    elif 'PC' in feat_select:
        feats = load_prog_consis_protes(
            join(
                data_path, 
                'realworld', 
                'data_investigation', 
                'f1097-4cohorts-continuous',
                'LOSO_consistent_{:}.csv'.format(test_cohort)),
            int(feat_select[2:])
        )
    elif 'GCP' in feat_select:
        feats = load_prog_consis_genes(
            join(data_path,
            'realworld',
            'data_investigation',
            'gene_consis_prote.csv'),
            int(feat_select[3:])
        )
    else:
        feats = datasets[0]['protes']

    for i, dataset in enumerate(datasets):
        data = dataset['data']
        if dataset['cohort'] == 'Gao':
            data[data==0] = np.nan
        if log2:
           data[~np.isnan(data)] = np.log2(data[~np.isnan(data)]) 
        if zscore:
            data = stats.zscore(data, axis=0, nan_policy='omit')
        if min_impute:
            data[np.isnan(data)] = np.nanmin(data)
        else:
            data[np.isnan(data)] = np.nanmin(np.concatenate([dataset['data'] for dataset in datasets]), axis=0)
        datasets[i]['data'] = data

        intersected_protes = get_the_intersection([dataset['protes'], feats])
        ids = [datasets[i]['protes'].index(prote) for prote in intersected_protes]
        datasets[i]['data'] = datasets[i]['data'][:, ids]
        datasets[i]['protes'] = intersected_protes

        OS = dataset['OS']
        DFS = dataset['DFS']
        datasets[i]['status'][OS>clip] = 0
        datasets[i]['OS'][OS>clip] = clip
        datasets[i]['recurrence'][DFS>clip] = 0
        datasets[i]['DFS'][DFS>clip] = clip
    print('processed feat num:', len(intersected_protes))
    return datasets

def read_prior_protein_list(data_path, logfc_thresh=1.0):
    df = pd.read_excel(
        data_path, 
        sheet_name='Signature proteins for subtypes',
        header=[0, 1],
        engine='openpyxl'
    )
    prote_list = df['ID']['Uniport'].tolist()
    logfc_21 = df['Log-fold change (logFC)']['S-II_T vs. S-I_T'].tolist()
    logfc_31 = df['Log-fold change (logFC)']['S-III_T vs. S-I_T'].tolist()
    logfc_32 = df['Log-fold change (logFC)']['S-III_T vs. S-II_T'].tolist()
    valid_protes = []
    for prote_id in range(len(prote_list)):
        if prote_id <= 57 and \
            logfc_21[prote_id] < -logfc_thresh and \
            logfc_31[prote_id] < -logfc_thresh:
            # I UP
            valid_protes.append(prote_list[prote_id])
        elif prote_id > 57 and prote_id <= 458 and \
            logfc_21[prote_id] > logfc_thresh and \
            logfc_31[prote_id] > logfc_thresh:
            # I DOWN
            valid_protes.append(prote_list[prote_id])
        elif prote_id > 458 and prote_id <= 501 and \
            logfc_21[prote_id] > logfc_thresh and \
            logfc_32[prote_id] < -logfc_thresh:
            # II UP
            valid_protes.append(prote_list[prote_id])
        elif prote_id > 501 and prote_id <= 538 and \
            logfc_21[prote_id] < -logfc_thresh and \
            logfc_32[prote_id] > logfc_thresh:
            # II DOWN
            valid_protes.append(prote_list[prote_id])
        elif prote_id > 538 and prote_id <= 1017 and \
            logfc_31[prote_id] > logfc_thresh and \
            logfc_32[prote_id] > logfc_thresh:
            # III UP
            valid_protes.append(prote_list[prote_id])
        elif prote_id > 1017 and \
            logfc_31[prote_id] < -logfc_thresh and \
            logfc_32[prote_id] < -logfc_thresh:
            # III DOWN
            valid_protes.append(prote_list[prote_id])
    print('nature prote num: ', len(valid_protes))
    return valid_protes
    
def read_protein_list(data_path, unique=False):
    prote_list = read_list(data_path)
    if unique:
        unique_list = []
        for item in prote_list:
            if item not in unique_list:
                unique_list.append(item)
        return unique_list
    else:
        return prote_list

def read_list(data_path):
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        return [row[0] for row in csv_reader]

def get_unique(X):
    unique_X = []
    for x in X:
        if x not in unique_X:
            unique_X.append(x)
    return unique_X

def gene2prote(file_path):
    df = pd.read_excel(
        file_path, 
        sheet_name='Annotation-Summary',
        header=[0],
        engine='openpyxl'
    )
    prote_list = df['uniprotAccession'].tolist()
    gene_list = df['GeneName'].tolist()
    gene2prote_dict = {}
    for gene, prote in zip(gene_list, prote_list):
        gene2prote_dict[gene] = prote
    return gene2prote_dict

def prote2gene(file_path):
    df = pd.read_excel(
        file_path, 
        sheet_name='HCC-Proteomics-Annotation',
        header=[0],
        engine='openpyxl'
    )
    prote_list = df['uniprotAccession'].tolist()
    gene_list = df['GeneName'].tolist()
    prote2gene_dict = {}
    for gene, prote in zip(gene_list, prote_list):
        prote2gene_dict[prote] = gene
    return prote2gene_dict

def gene2prote_and_prote2gene(file_path):
    df = pd.read_excel(
        file_path, 
        sheet_name='HCC-Proteomics-Annotation',
        header=[0],
        engine='openpyxl'
    )
    prote_list = df['uniprotAccession'].tolist()
    gene_list = df['GeneName'].tolist()
    gene2prote_dict = {}
    for gene, prote in zip(gene_list, prote_list):
        gene2prote_dict[gene] = prote

    prote2gene_dict = {}
    for gene, prote in zip(gene_list, prote_list):
        prote2gene_dict[prote] = gene

    return gene2prote_dict, prote2gene_dict

def read_assignments(data_path, patients=None):
    subtype_dict = {'S1': 0, 'S2': 1, 'S3': 2, 'Others': -1}
    if patients is None:
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            assignments = []
            patients = []
            for i, row in enumerate(csv_reader):
                if i > 0:
                    patients.append(row[0])
                    assignments.append(subtype_dict[row[1]])
        return patients, np.asarray(assignments)
    else:
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            assignments = np.zeros([len(patients)], dtype=np.int32)
            cnt = 0
            for i, row in enumerate(csv_reader):
                if i > 0:
                    if row[0] in patients:
                        assignments[patients.index(row[0])] = subtype_dict[row[1]]
                        cnt += 1
        if cnt < len(patients):
            print('not all patients get assignments')
        return assignments

def feat_selection(data, thresh=0.7):
    valid_ids = []
    for col_id, col in enumerate(data.T):
        if np.sum(np.isnan(col)) <= len(col) * thresh:
            valid_ids.append(col_id)
    print('nan pass prote num:', len(valid_ids))
    return np.asarray(valid_ids)

def get_the_intersection(lists):
    global_X = []
    for x in lists[0]:
        miss_flag = False
        for X in lists[1:]:
            if x not in X:
                miss_flag = True
        if not miss_flag:
            global_X.append(x)
    return global_X

def get_km_curve(times, events, clip_time=60):
    # get km curve
    unique_times = np.asarray(list(set(times)))
    sorted_unique_times = np.sort(unique_times)
    S_list = [1.]
    time_list = [0.]
    censor_list = [False]
    at_risk_list = [len(times)]
    live_at_the_start = len(times)
    S_t = 1.
    start_time = 0
    RMST = 0.
    for i in range(len(sorted_unique_times)):
        end_time = sorted_unique_times[i]
        event_num = np.sum(events[times==end_time])
        at_risk_list.append(live_at_the_start)
        live_at_the_start = np.sum(times >= end_time)
        if end_time <= clip_time:
            RMST += (S_t * (end_time - start_time))
        S_list.append(S_t)
        S_t *= (1. - event_num/live_at_the_start)
        S_list.append(S_t)
        time_list.append(end_time)
        time_list.append(end_time)
        censor_list.append(0 in events[times==end_time])
        censor_list.append(0 in events[times==end_time])
        at_risk_list.append(live_at_the_start)
        start_time = end_time
    if np.amax(times) < clip_time:
        RMST += (S_t * (60 - end_time))
    return S_list, time_list, censor_list, at_risk_list, RMST

@jit
def get_rmst_custom(times, events, clip_time=60):
    # get km curve
    unique_times = np.asarray(list(set(times)))
    sorted_unique_times = np.sort(unique_times)
    S_list = [1.]
    time_list = [0.]
    S_t = 1.
    start_time = 0
    RMST = 0.
    for i in range(len(sorted_unique_times)):
        end_time = sorted_unique_times[i]
        event_num = np.sum(events[times==end_time])
        live_at_the_start = np.sum(times >= end_time)
        if end_time <= clip_time:
            RMST += (S_t * (end_time - start_time))
        S_t *= (1. - event_num/live_at_the_start)
        S_list.append(S_t)
        time_list.append(end_time)
        start_time = end_time
    if np.amax(times) < clip_time:
        RMST += (S_t * (60 - end_time))
    return RMST

def get_rmst(time, event, label):
    kmf = KaplanMeierFitter().fit(time, event, label=label)
    return restricted_mean_survival_time(kmf, t=60)

def get_delta_rmst(time, event, assignments, k, return_RMST=False, case2=False):
    rmst_list = []
    for i in range(k):
        if np.sum(assignments==i) <= 3:
            return_list = [-10.] * (k - 1) if not return_RMST else [-10.] * (k - 1 + k)
            return return_list
        rmst = get_rmst_custom(
            time[assignments==i],
            event[assignments==i], 
        )
        rmst_list.append(rmst)
    if k == 3:
        if case2:
            return_list = [rmst_list[0] - rmst_list[2], rmst_list[1] - rmst_list[2]]
        else:
            return_list = [rmst_list[0] - rmst_list[1], rmst_list[1] - rmst_list[2]]
    elif k == 2:
        return_list = [rmst_list[0] - rmst_list[1]]
    if return_RMST:
        return_list += rmst_list
    return return_list

def get_reward(os_month, os_status, assignments):
    rmst_list = []
    for subtype in range(3):
        if np.sum(assignments==subtype) <= 3:
            return -2. * np.ones_like(assignments)
        os = os_month[assignments==subtype]
        status = os_status[assignments==subtype]
        # rmst = get_rmst(os, status, 'label: ' + str(subtype))
        rmst = get_rmst_custom(os, status)
        rmst_list.append(rmst)
    r = min(rmst_list[0] - rmst_list[1], rmst_list[1] - rmst_list[2])/10.
    return (r * np.ones_like(assignments)).astype(np.float32)

def multivariate_logrank_test(times, events, assignments):
    logrank_results = statistics.multivariate_logrank_test(times, assignments, events)
    p = logrank_results.p_value + 1e-100
    return p

def compare_assignments(file_a, file_b):
    patients_a, assignments_a = read_assignments(file_a)
    patients_b, assignments_b = read_assignments(file_b)
    same = 0
    total = 0
    for assign_a, patient in zip(assignments_a, patients_a):
        if patient in patients_b:
            assign_b = assignments_b[patients_b.index(patient)]
            if assign_a == assign_b:
                same += 1 
            total += 1
    print('same assignments: {:}/{:}, {:.3f}%'.format(
        same, 
        total, 
        float(same)/total)
    )

def write_assignments(file_path, patients, assignments, labels=None):
    with open(file_path, 'w',newline='') as f:
        writer = csv.writer(
            f, 
            delimiter=',', 
            quotechar='"', 
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(['ID', 'Subtype'])
        for patient, assignment in zip(patients, assignments):
            if labels:
                 writer.writerow([patient, labels[assignment]])
            else:
                writer.writerow([patient, 'S{:d}'.format(assignment+1)])

def write_protes(protes, write_path):
    with open(write_path, 'w') as f:
        for prote in protes:
            f.write("%s\n" % prote)

def load_prog_consis_protes(file_path, top_k):
    df = pd.read_csv(file_path, index_col=0)
    protes = df.index.values
    consis = df['consistency'].to_numpy()
    return list(protes[:top_k])

def load_prog_consis_genes(file_path, top_k):
    df = pd.read_csv(file_path, index_col=0)
    genes = df['gene'].to_list()[:top_k]
    protes = df['prote'].to_list()[:top_k]
    return protes
    
if __name__ == '__main__':
    pass
import csv, re, copy, os, time, progressbar, itertools
import numpy as np
import pandas as pd
import harmonypy as hm

from numba import jit
from os.path import join
from joblib import Parallel, delayed

from scipy import stats, spatial
from lifelines import CoxPHFitter, statistics

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from scipy.stats import t

def load_HCC_data(data_path, gene2prote_dict):
    # read hcc101 data, patients, subtypes, OS, status
    print('HCC Jiang cohort')
    datasets = []
    data, patients, protes, subtypes, OS, status, recurrence, DFS = read_hcc101_data(
        join(data_path, 'sub_data_quantile.csv'),
        join(data_path, 'Nature_101HCC_subtype2sampleID.txt'),
        join(data_path, 'HCC_101tumor_Nature_survival.txt')
    )

    datasets.append({
            'data': data, 
            'patients': patients,
            'protes': protes,
            'subtypes': subtypes,
            'OS': OS,
            'DFS': DFS,
            'status': status,
            'recurrence': recurrence,
            'label': 'Jiang'
        }
    )

    # read center123 data
    label_dict = {1: 'SH', 2: 'GZ', 3: 'FZ'}
    for data_id in range(1, 4):
        print('HCC ' + label_dict[data_id] + ' cohort')
        data, patients, protes, subtypes, OS, status, DFS, recurrence = read_center_data(
            join(data_path, 'temp{:d}_quantile.csv'.format(data_id)),
            join(data_path, 'ClinInfo_Center123_1024Patients_普适性subtype-20210330.csv'.format(data_id))
        )

        datasets.append({
                'data': data,
                'patients': patients,
                'protes': protes, 
                'subtypes': subtypes,
                'OS': OS,
                'status': status,
                'DFS': DFS,
                'recurrence': recurrence,
                'label': label_dict[data_id],
            }
        )

    # read Cell data
    print('HCC Gao cohort')
    data, patients, genes, subtypes, OS, status, DFS, recurrence = read_Cell_data(
        join(data_path, 'Fudan_Cell_HCC159_6478gene_nolog2.csv'),
        join(data_path, 'ClinicalInfo_Fudan_Cell_159sample.csv')
    )

    valid_feat_ids = []
    protes = []
    for gene_id, gene in enumerate(genes):
        if gene in gene2prote_dict.keys():
            protes.append(gene2prote_dict[gene])
            valid_feat_ids.append(gene_id)
    data = data[:, valid_feat_ids]

    cell_dataset = {
        'data': data,
        'patients': patients,
        'protes': protes, 
        'OS': OS,
        'status': status,
        'DFS': DFS,
        'recurrence': recurrence,
        'label': 'Gao'
    }
    datasets.append(cell_dataset)
    return datasets

def read_hcc101_data(x_path, y_path, z_path):
    # read protein data
    with open(x_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        protes = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                patients = row[99:]
            else:
                protes.append(row[0])
                data.append([float(elem) if elem != 'NA' else np.nan for elem in row[99:]])
    data = np.asarray(data).transpose()

    # read subtypes
    with open(y_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        subtypes = np.zeros([len(patients)])
        for i, row in enumerate(csv_reader):
            if i > 0:
                str_list = re.split(r'\t+', row[0])
                subtypes[patients.index(str_list[1])] = len(str_list[0]) - 3

    # read survival
    with open(z_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        OS = np.zeros([len(patients)])
        status = np.zeros([len(patients)])
        recurrence = np.zeros([len(patients)])
        DFS = np.zeros([len(patients)])
        for i, row in enumerate(csv_reader):
            if i > 0:
                str_list = re.split(r'\t+', row[0])
                status[patients.index(str_list[0])] = int(str_list[3])
                OS[patients.index(str_list[0])] = float(str_list[4])
                recurrence[patients.index(str_list[0])] = int(str_list[1])
                DFS[patients.index(str_list[0])] = float(str_list[2])
    return data, patients, protes, subtypes, OS, status, recurrence, DFS

def read_center_data(x_path, y_path):
    with open(x_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        protes = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                patients = row[1:]
            else:
                protes.append(row[0])
                data.append([float(elem) if elem != 'NA' else np.nan for elem in row[1:] ])
    data = np.asarray(data).transpose()

    with open(y_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        OS = np.zeros([len(patients)])
        DFS = np.zeros([len(patients)])
        status = np.zeros([len(patients)])
        recurrence = np.zeros([len(patients)])
        subtypes = np.zeros([len(patients)])
        valid = np.zeros([len(patients)], dtype=bool)
        subtype_dict = {'S-I':0, 'S-II': 1, 'S-III': 2}
        for i, row in enumerate(csv_reader):
            if i > 0:
                ID = 'T' + row[0] if 'G' not in row[0] and 'T' not in row[0] else row[0]
                if ID in patients:
                    idx = patients.index(ID)
                    OS[idx] = float(row[18])
                    status[idx] = int(row[17])
                    recurrence[idx] = int(row[15])
                    DFS[idx] = float(row[16])
                    subtypes[idx] = subtype_dict[row[21]]
                    valid[idx] = True
    data = data[valid]
    if np.sum(valid) < len(valid):
        print(len(valid) - np.sum(valid), 'patients missing when loading clinicalinfo')
    patients = [patient for is_valid, patient in zip(valid, patients) if is_valid]
    OS = OS[valid]
    status = status[valid]
    DFS = DFS[valid]
    recurrence = recurrence[valid]
    subtypes = subtypes[valid]
    return data, patients, protes, subtypes, OS, status, DFS, recurrence

def read_Cell_data(x_path, y_path):
    with open(x_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        genes = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                patients = row[1:]
            else:
                genes.append(row[0])
                data.append([float(elem) if elem != 'NA' else np.nan for elem in row[1:] ])
    data = np.asarray(data).transpose()

    with open(y_path, mode='r', encoding='UTF-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        OS = np.zeros([len(patients)])
        status = np.zeros([len(patients)])
        DFS = np.zeros([len(patients)])
        recurrence = np.zeros([len(patients)])
        subtypes = np.zeros([len(patients)])
        for i, row in enumerate(csv_reader):
            if i > 0:
                idx = patients.index(row[0])
                DFS[idx] = row[5]
                OS[idx] = row[6]
                recurrence[idx] = row[7]
                status[idx] = row[8]
                subtypes[idx] = int(row[26]) - 1
    return data, patients, genes, subtypes, OS, status, DFS, recurrence 

def HCC_data_preprocess(datasets, data_path, prote2gene_dict, clip, nan_thresh, zscore, log2, f1269, impute=False):
    for i, dataset in enumerate(datasets):
        # clinical info
        data, protes = dataset['data'], dataset['protes']
        OS, status, DFS, recurrence = dataset['OS'], dataset['status'], dataset['DFS'], dataset['recurrence']
        status[OS>clip] = 0
        OS[OS>clip] = clip
        recurrence[DFS>clip] = 0
        DFS[DFS>clip] = clip
        datasets[i]['OS'], datasets[i]['status'], datasets[i]['DFS'], datasets[i]['recurrence'] = OS, status, DFS, recurrence

        valid_feat_ids = feat_selection(data, nan_thresh)
        data = data[:, valid_feat_ids]
        protes = [protes[prote_id] for prote_id in valid_feat_ids]
        datasets[i]['protes'] = protes

        data[np.isnan(data)] = np.nanmin(data) if impute else 0
        if log2 and impute:
            data = np.log2(data)

        data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5) if zscore else data
        datasets[i]['data'] = data

    # read the nature selected protein list
    nature_prote_list = read_prior_protein_list(
        join(data_path, '2017-04-04932E-Supplementary Table 11.xlsx'),
        1
    )

    # get the instersection of proteins
    lists = [datasets[i]['protes'] for i in range(len(datasets))]
    if f1269:
        lists += [nature_prote_list]

    intersected_protes = get_the_intersection(lists)
    print('intersected feat num: ', len(intersected_protes))

    # adjust the data according to the intersected_protes
    for dataset_id in range(len(datasets)):
        ids = [datasets[dataset_id]['protes'].index(prote) for prote in intersected_protes]
        datasets[dataset_id]['data'] = datasets[dataset_id]['data'][:, ids]
        datasets[dataset_id]['protes'] = [datasets[dataset_id]['protes'][prote_id] for prote_id in ids]
        datasets[dataset_id]['genes'] = [prote2gene_dict[prote] for prote in intersected_protes]

    for dataset_id, dataset in enumerate(datasets):
        print('dataset ', dataset['label'], ' sample num: ', len(dataset['data']))
    return datasets

def read_LUAD_data(data_path):
    datasets = []

    print('LUAD Xu cohort')
    df = pd.read_excel(
        join(data_path, 'Xu', 'mmc4.xlsx'), 
        sheet_name='Table S4A',
        header=[0],
        engine='openpyxl'
    )
    data = df.to_numpy()[:, 3:3+103].astype(np.float32).transpose()
    patients = [patient.split(' ')[1][:-1] for patient in list(df.columns)[3:] if patient[-1] == 'T']
    genes, is_valid = [], [] 
    for i, gene in enumerate(df['Gene names'].tolist()):
        if isinstance(gene, float):
            is_valid.append(False)
        else:
            is_valid.append(True)
            if ';' in gene:
                genes.append(gene.split(';')[0])
            else:
                genes.append(gene) 
    data = data[:, np.asarray(is_valid)]

    df = pd.read_csv(join(data_path, 'Xu', 'ClinicalInfor.csv'), header=[0], delimiter = ',', encoding = "ISO-8859-1")
    patients_clinical = df['Sample ID'].tolist()
    ids = [patients_clinical.index(patient) for patient in patients]
    subtypes = np.asarray([len(subtype) - 1 for subtype in df['Subtype'].tolist()])[ids]

    OS = np.around(df['OS-month'].to_numpy()[ids] * 10) / 10. 
    status = df['OS'].to_numpy()[ids]
    DFS = np.around(df['DFS-month'].to_numpy()[ids] * 10) / 10.
    recurrence = df['DFS'].to_numpy()[ids]

    datasets.append({
            'label': 'Xu',
            'data': data,
            'patients': patients,
            'genes': genes,
            'subtypes': subtypes,
            'OS': OS,
            'status': status,
            'DFS': DFS,
            'recurrence': recurrence
        }
    )

    print('LUAD Gillette cohort')
    with open(join(data_path, 'Gillette', '2020-Cell-ADC-Carr-IntensityDF.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        genes = []
        for i, row in enumerate(csv_reader):
            if i == 1:
                patients = row[102:]
            elif i > 2:
                genes.append(row[0])
                data.append([np.nan if (elem == None or len(elem)==0) else float(elem) for elem in row[102:]])
    data = np.asarray(data).transpose()

    df = pd.read_csv(join(data_path, 'Gillette', '2020-Cell-ADC-Carr-ClinialDF.csv'), header=[0], delimiter = ',', encoding = "ISO-8859-1")
    patients_clinical = df['PatientID'].tolist()
    OS = df['OS'].to_numpy()
    status = df['OS Status'].to_numpy()
    DFS = df['DFS'].to_numpy()
    recurrence = df['DFS Status'].to_numpy()
    filtered_patients, ids = [], []
    is_valid = np.zeros((len(patients)), dtype=bool)
    for i, patient in enumerate(patients):
        if patient in patients_clinical:
            idx = patients_clinical.index(patient)
            if not np.isnan(OS[idx]) and not np.isnan(status[idx]) and not np.isnan(DFS[idx]) and not np.isnan(recurrence[idx]):
                filtered_patients.append(patient)
                ids.append(idx)
                is_valid[i] = True
    data = data[is_valid, :]
    patients = filtered_patients

    OS = np.around(df['OS'].to_numpy()[ids] * 10) / 10. 
    status = df['OS Status'].to_numpy()[ids]
    DFS = np.around(df['DFS'].to_numpy()[ids] * 10) / 10.
    recurrence = df['DFS Status'].to_numpy()[ids]

    datasets.append({
            'label': 'Gillette',
            'data': data,
            'patients': patients,
            'genes': genes,
            'OS': OS,
            'status': status,
            'DFS': DFS,
            'recurrence': recurrence
        }
    )
    # for i, patient in enumerate(patients):
    #     print(i, patient, OS[i], DFS[i])

    return datasets

def LUAD_data_preprocess(datasets, clip, nan_thresh, exp_list, log2_list, zscore, subtype_num, subtype_feat, data_path):
    for i, dataset in enumerate(datasets):
        OS, status, DFS, recurrence = dataset['OS'], dataset['status'], dataset['DFS'], dataset['recurrence']
        status[OS>clip] = 0
        OS[OS>clip] = clip
        recurrence[DFS>clip] = 0
        DFS[DFS>clip] = clip
        datasets[i]['OS'], datasets[i]['status'], datasets[i]['DFS'], datasets[i]['recurrence'] = OS, status, DFS, recurrence

        data = dataset['data']
        genes = dataset['genes']
        valid_feat_ids = feat_selection(data, nan_thresh)
        data = data[:, valid_feat_ids]
        genes = [genes[feat_id] for feat_id in valid_feat_ids]
        if exp_list[i]:
            data[np.logical_not(np.isnan(data))] = 2**data[np.logical_not(np.isnan(data))]
        if log2_list[i]:
            data[np.isnan(data)] = np.nanmin(data)
            data = np.log2(data)
        else:
            data[np.isnan(data)] = 0

        data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5) if zscore else data
        datasets[i]['data'] = data
        datasets[i]['genes'] = genes

        if subtype_num == 2 and 'subtypes' in dataset.keys():
            subtypes = dataset['subtypes']
            subtypes[subtypes==0] = 1
            subtypes -= 1
            datasets[i]['subtypes'] = subtypes

    # get the instersection of proteins
    lists = [datasets[i]['genes'] for i in range(len(datasets))]

    if subtype_feat:
        df = pd.read_excel(
            join(data_path, 'Xu', 'mmc4.xlsx'), 
            sheet_name='Table S4I',
            header=[0],
            engine='openpyxl'
        )
        feature_genes = df['Up-regulated proteins in S-I'].tolist(
            ) + df['Up-regulated proteins in S-II'].tolist(
            ) + df['Up-regulated proteins in S-III'].tolist()

        lists += [feature_genes]
    else:
        pass

    features = get_the_intersection(lists)
    print('intersected feat num: ', len(features))

    # adjust the data according to the features
    for dataset_id in range(len(datasets)):
        ids = [datasets[dataset_id]['genes'].index(feat) for feat in features]
        datasets[dataset_id]['data'] = datasets[dataset_id]['data'][:, ids]
        datasets[dataset_id]['genes'] = [datasets[dataset_id]['genes'][feat_id] for feat_id in ids]

    for dataset_id, dataset in enumerate(datasets):
        print('raw dataset ', dataset['label'], ' sample num: ', len(dataset['data']))

    return datasets

def cross_tumor_preprocessing(datasets, prote2gene_dict, clip, nan_thresh, exp_list, impute_list, log2_list, zscore, subtype_feat, data_path):
    for i, dataset in enumerate(datasets):

        if 'genes' not in dataset.keys():
            valid_ids = []
            genes = []
            for j, prote in enumerate(dataset['protes']):
                if prote in prote2gene_dict.keys():
                    genes.append(prote2gene_dict[prote])
                    valid_ids.append(j)
            datasets[i]['data'] = datasets[i]['data'][:, valid_ids]
            datasets[i]['genes'] = genes
            datasets[i].pop('protes', None)

        data, genes = dataset['data'], dataset['genes']
        OS, status, DFS, recurrence = dataset['OS'], dataset['status'], dataset['DFS'], dataset['recurrence']
        status[OS>clip] = 0
        OS[OS>clip] = clip
        recurrence[DFS>clip] = 0
        DFS[DFS>clip] = clip
        datasets[i]['OS'], datasets[i]['status'], datasets[i]['DFS'], datasets[i]['recurrence'] = OS, status, DFS, recurrence

        valid_feat_ids = feat_selection(data, nan_thresh)
        data = data[:, valid_feat_ids]
        genes = [genes[gene_id] for gene_id in valid_feat_ids]
        datasets[i]['genes'] = genes

        if exp_list[i]:
            data[np.logical_not(np.isnan(data))] = 2**data[np.logical_not(np.isnan(data))]
        if impute_list[i]:
            data[np.isnan(data)] = np.nanmin(data)
        else:
            data[np.isnan(data)] = 0
        if log2_list[i]:
            data = np.log2(data)

        data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5) if zscore else data
        datasets[i]['data'] = data

    if subtype_feat == 'Jiang':
        # read the nature selected protein list
        sig_protes = read_prior_protein_list(
            join(data_path, 'HCC_data', '2017-04-04932E-Supplementary Table 11.xlsx'),
            1
        )
        sig_genes = [prote2gene_dict[prote] for prote in sig_protes]
    elif subtype_feat == 'Xu':
        df = pd.read_excel(
            join(data_path, 'LUAD_data', 'Xu', 'mmc4.xlsx'), 
            sheet_name='Table S4I',
            header=[0],
            engine='openpyxl'
        )
        sig_genes = df['Up-regulated proteins in S-I'].tolist(
            ) + df['Up-regulated proteins in S-II'].tolist(
            ) + df['Up-regulated proteins in S-III'].tolist()

    # get the instersection of proteins
    lists = [datasets[i]['genes'] for i in range(len(datasets))] + [sig_genes]

    intersected_protes = get_the_intersection(lists)
    print('intersected feat num: ', len(intersected_protes))

    # adjust the data according to the intersected_protes
    for dataset_id in range(len(datasets)):
        ids = [datasets[dataset_id]['genes'].index(prote) for prote in intersected_protes]
        datasets[dataset_id]['data'] = datasets[dataset_id]['data'][:, ids]
        datasets[dataset_id]['genes'] = [datasets[dataset_id]['genes'][gene_id] for gene_id in ids]

    for dataset_id, dataset in enumerate(datasets):
        print('dataset ', dataset['label'], ' sample num: ', len(dataset['data']))
    return datasets

def univariate_cox_regression(prote_id, prote, times, events, data):
    data_dict = {}
    data_dict[prote] = data[:, prote_id]
    data_dict['event_time'] = times
    data_dict['event'] = events
    df = pd.DataFrame(data=data_dict)
    cph = CoxPHFitter()
    cph.fit(df, duration_col='event_time', event_col='event')
    return cph.params_.tolist()[0], cph.hazard_ratios_.tolist()[0], cph.summary['p'].tolist()[0]

def cox_check(datasets, save_path):
    results = {}
    protes = datasets[0]['protes'] if 'protes' in datasets[0].keys() else datasets[0]['genes']
    for dataset in datasets:
        for event_time, event_status in zip(['OS', 'DFS'], ['status', 'recurrence']):
            file_path = join(save_path, 'cox_' + dataset['label'] + '_' + event_time + '.csv')
            if os.path.isfile(file_path):
                data = np.genfromtxt(file_path, dtype=float, delimiter=',', names=True) 
                coef_list, hr_list, p_list = data['coef'], data['hr'], data['p']
            else:
                times = dataset[event_time]
                events = dataset[event_status]
                data = dataset['data']
                res = Parallel(n_jobs=10)(delayed(univariate_cox_regression)(prote_id, protes[prote_id], times, events, data) 
                    for prote_id in progressbar.progressbar(range(len(protes))))
                coef_list = np.asarray([item[0] for item in res])
                hr_list = np.asarray([item[1] for item in res])
                p_list = np.asarray([item[2]for item in res])
                np.savetxt(file_path, 
                    np.c_[coef_list, hr_list, p_list], 
                    delimiter=',', 
                    header='coef,hr,p',
                    comments=''
                )
            results[dataset['label'] + '_' + event_time] = [coef_list, hr_list, p_list]
    return results

def read_synthetic_data(data_path, batch_effect, subtype_num, domain_num, log, zscore, cell_num):
    # load synthetic RNA-seq data generated by R splatter package
    with open(join(data_path, 'data' + str(cell_num) + '_' + batch_effect + '_batch_effect_surv.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        patients = []
        subtypes = []
        domains = []
        times = []
        status = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                genes = row[1:-4]
            elif i > 0:
                patients.append(row[0])
                data.append(row[1:-4])
                subtypes.append(row[-4])
                domains.append(row[-3])
                times.append(row[-2])
                status.append(row[-1])
        data = np.asarray(data, dtype=int)
        subtypes = np.asarray(subtypes, dtype=int) - 1
        domains = np.asarray(domains, dtype=int) - 1
        times = np.asarray(times, dtype=float)
        status = np.asarray(status, dtype=int)

    data[data > 0] = np.log(data[data > 0]) if log else data[data > 0]

    ids = np.argsort(subtypes)
    data = data[ids, :]
    domains = domains[ids]
    patients = [patients[idx] for idx in ids]
    times = times[ids]
    status = status[ids]
    subtypes = subtypes[ids]

    datasets = []
    for i in range(domain_num):
        is_valid = np.logical_and(domains == i, subtypes < subtype_num)
        x = data[is_valid, :]
        datasets.append({
            'data': (x - np.mean(x, axis=0))/(np.std(x, axis=0)+1e-5) if zscore else x,
            'genes': genes,
            'patients': [patient for patient, is_true in zip(patients, is_valid) if is_true],
            'subtypes': subtypes[is_valid],
            'OS': times[is_valid],
            'status': status[is_valid],
            'DFS': times[is_valid],
            'recurrence': status[is_valid],
            'label': 'Domain' + str(i) 
        })

    return datasets

def synthesize_life_time(datasets, subtype_num):
    # generate life times for each subtypes
    survival_plateau_list = [0.9 - (0.9 - 0.) / (subtype_num - 1) * subtype for subtype in range(subtype_num)]
    for i, dataset_i in enumerate(datasets):
        OS_list, status_list = [], []
        for subtype in range(subtype_num):
            survival_data = simulate_survival_data_linear(
                np.sum(dataset_i['subtypes']==subtype), 
                survival_plateau_list[subtype], 60, 60, 0.1
            )
            OS_list += survival_data['Time'].tolist()
            status_list += survival_data['Event'].tolist()
        datasets[i]['OS'] = np.asarray(OS_list)
        datasets[i]['status'] = np.asarray(status_list)
        datasets[i]['DFS'] = np.asarray(OS_list)
        datasets[i]['recurrence'] = np.asarray(status_list)
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
        RMST += (S_t * (clip_time - end_time))
    return RMST

@jit
def get_survival_at_t(times, events, t, clip_time=60):
    if len(times) <= 1:
        return 0.
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
        if time_list[i - 1] <= t and time_list[i] > t:
            return S_t
    return S_t

def get_delta_rmst(times, events, assignments, k, clip_time=60):
    rmst_list = [
        get_rmst_custom(
            times[assignments==i], events[assignments==i], clip_time
        ) for i in np.sort(np.unique(assignments))
    ]
    return  np.diff(rmst_list[::-1])

def get_reward(times, events, assignments, k=3, clip_time=60):
    if np.amin([np.sum(assignments==s) for s in range(k)]) <= 1:
        return -2. * np.ones_like(assignments)   
    delta_rmst = get_delta_rmst(times, events, assignments, k, clip_time)
    r = np.amin(delta_rmst)/10.
    return (r * np.ones_like(assignments)).astype(np.float32)

def get_multivariate_logrank_p(time, event, assignments, k):
    unique_times = np.unique(time)
    rm_list = [[] for _ in range(k)]
    obs_list = [[] for _ in range(k)]
    for t in unique_times:
        for group in range(k):
            rm_list[group].append(np.sum((time==t)*(assignments==group)))
            obs_list[group].append(np.sum((time==t)*(event==1)*(assignments==group)))
    rm_list =  np.asarray(rm_list).transpose()
    obs_list = np.asarray(obs_list).transpose()

    n_ij = np.cumsum(rm_list, axis=0)
    n_ij = np.roll(n_ij, 1, axis=0)
    n_ij[0, :] = 0
    n_ij = np.sum(rm_list, axis=0) - n_ij
    d_i = np.sum(obs_list, 1)
    n_i = np.cumsum(np.sum(rm_list, axis=1))
    n_i = np.roll(n_i, 1, axis=0)
    n_i[0] = 0
    n_i = np.sum(rm_list) - n_i
    ev_i = n_ij * np.stack([d_i / n_i for _ in range(k)], axis=-1)

    N_j = np.sum(obs_list, axis=0)
    ev = np.sum(ev_i, axis=0)
    Z_j = N_j -  ev
    assert abs(np.sum(Z_j)) < 1e-7, "Sum is not zero." 

    with np.errstate(divide='ignore'):
        factor = (n_i - d_i) / (n_i - 1)
    factor[np.isnan(factor)] = 1
    factor[np.isinf(factor)] = 1
    factor = factor * d_i / n_i ** 2
    n_ij = np.c_[n_ij, n_i]
    V_ = np.stack([factor for _ in range(k+1)], axis=-1)
    V_[V_<0] = 0
    V_ = n_ij * np.sqrt(V_)
    V = -np.dot(V_.T, V_)
    ix = np.arange(k)
    V[ix, ix] = V[ix, ix] - V[-1, ix]
    V = V[:-1, :-1]
    U = Z_j[:-1].transpose() @ np.linalg.pinv(V[:-1, :-1]) @ Z_j[:-1]
    return stats.chi2.sf(U, k - 1)

def get_pairwise_logrank_p(time, event, assignments, k):
    p_list = []
    for a, b in list(itertools.combinations(np.arange(k), 2)):
        curr_time = time[np.logical_or(assignments==a, assignments==b)]
        curr_event = event[np.logical_or(assignments==a, assignments==b)]
        curr_assignment = assignments[np.logical_or(assignments==a, assignments==b)]
        unique_times = np.unique(curr_time)
        rm_list = [[], []]
        obs_list = [[], []]
        for t in unique_times:
            label_list = [a, b]
            for group in range(2):
                rm_list[group].append(np.sum((curr_time==t)*(curr_assignment==label_list[group])))
                obs_list[group].append(np.sum((curr_time==t)*(curr_event==1)*(curr_assignment==label_list[group])))
        rm_list =  np.asarray(rm_list).transpose()
        obs_list = np.asarray(obs_list).transpose()

        n_ij = np.cumsum(rm_list, axis=0)
        n_ij = np.roll(n_ij, 1, axis=0)
        n_ij[0, :] = 0
        n_ij = np.sum(rm_list, axis=0) - n_ij
        d_i = np.sum(obs_list, 1)
        n_i = np.cumsum(np.sum(rm_list, axis=1))
        n_i = np.roll(n_i, 1, axis=0)
        n_i[0] = 0
        n_i = np.sum(rm_list) - n_i
        ev_i = n_ij * np.c_[d_i / n_i, d_i / n_i]

        N_j = np.sum(obs_list, axis=0)
        ev = np.sum(ev_i, axis=0)
        Z_j = N_j -  ev
        assert abs(np.sum(Z_j)) < 1e-7, "Sum is not zero." 

        with np.errstate(divide='ignore'):
            factor = (n_i - d_i) / (n_i - 1)
        factor[np.isnan(factor)] = 1
        factor[np.isinf(factor)] = 1
        factor = factor * d_i / n_i ** 2
        n_ij = np.c_[n_ij, n_i]
        V_ = np.c_[factor, factor, factor]
        V_[V_<0] = 0
        V_ = n_ij * np.sqrt(V_)
        V = -np.dot(V_.T, V_)
        ix = np.arange(k)
        V[ix, ix] = V[ix, ix] - V[-1, ix]
        V = V[:-1, :-1]
        U = Z_j[:-1].transpose() @ np.linalg.pinv(V[:-1, :-1]) @ Z_j[:-1]
        p_list.append(stats.chi2.sf(U, 1))
    return p_list

def n_fold_split(datasets, n, test_fold, stratify=False):
    for i, dataset in enumerate(datasets):
        n_sample = len(dataset['patients'])
        folds = np.zeros((n_sample), dtype=int)
        ids = np.arange(n_sample)
        if not stratify:
            for k in range(n-1):
                valid_ids = ids[folds==0]
                selected_ids = np.random.choice(valid_ids, int(len(valid_ids) / (n - k)), replace=False)
                folds[selected_ids] = k + 1
        else:
            skf = StratifiedKFold(n_splits=n)
            for k, (train, test) in enumerate(skf.split(np.arange(n_sample), dataset['subtypes'])):
                # print(np.bincount(dataset['subtypes'][test]))
                folds[test] = k
        datasets[i]['folds'] = folds

        valid_fold = test_fold + 1 if test_fold < n - 1 else 0
        groups = np.zeros_like(folds) 
        groups[folds==valid_fold] = 1 # validation
        groups[folds==test_fold] = 2 # testing
        datasets[i]['groups'] = groups
    return datasets

def dataset_group_filtering(dataset, group, label):
    # create a new dataset for groups == group
    new_dataset = {'label': label}
    if 'protes' in dataset.keys():
        new_dataset['protes'] = dataset['protes']
    if 'genes' in dataset.keys():
        new_dataset['genes'] = dataset['genes']
    groups = dataset['groups']
    for key in dataset.keys():
        if isinstance(dataset[key], list) or isinstance(dataset[key], np.ndarray):
            if key in ['protes', 'genes']:
                continue
            items = [item for item, true in zip(dataset[key], groups==group) if true
            ] if isinstance(dataset[key], list) else dataset[key][groups==group]
            if key not in new_dataset.keys():
                new_dataset[key] = items
            elif isinstance(dataset[key], list):
                new_dataset[key] += items
            elif isinstance(dataset[key][groups==group], np.ndarray):
                new_dataset[key] = np.concatenate([new_dataset[key], items], axis=0)
    return new_dataset

def train_valid_test_regroup(datasets):
    group_label = ['train', 'valid', 'test']
    datasets_groups = [[] for _ in range(len(group_label))]
    for dataset in datasets:
        for group in range(len(group_label)):
            if np.sum(dataset['groups']==group) > 0:
                new_dataset = dataset_group_filtering(
                    dataset, 
                    group,
                    dataset['label'] + '_' + group_label[group]
                )
                datasets_groups[group].append(new_dataset)
    return datasets_groups

def run_harmony(datasets, subtype_num, theta, lamb, sigma, n_components=100):
    print('run_harmony')
    data = np.concatenate([dataset['data'] for dataset in datasets], axis=0).astype(float)
    data_pca = PCA(n_components=n_components).fit_transform(data)
    cohorts = np.concatenate([np.ones_like(dataset['OS'], dtype=int) * i for i, dataset in enumerate(datasets)], axis=0)
    dataset_labels = [dataset['label'] for dataset in datasets]
    meta_data = pd.DataFrame(data={'dataset_labels': [dataset_labels[cohort] for cohort in cohorts]})
    vars_use = ['dataset_labels']
    data_mat = data_pca
    data_mat[:, np.sum(data_mat, axis=0)==0] = 1.
    ho = hm.run_harmony(
        data_mat, meta_data, vars_use,
        nclust=subtype_num
    )
    # assignments = np.argmax(ho.R.T, axis=-1)
    # print(assignments)
    # assert False
    res = pd.DataFrame(ho.Z_corr)
    res.columns = ['X{}'.format(i + 1) for i in range(res.shape[1])]
    new_data = res.to_numpy().transpose()
    feat_name = 'genes' if 'genes' in datasets[0].keys() else 'protes'
    for i, dataset in enumerate(datasets):
        datasets[i]['data'] = new_data[cohorts==i]
        datasets[i][feat_name] = datasets[i][feat_name][:n_components] 
    return datasets

def _findSurvivalDistribution_np(lifetimes, deads, weights=None):
    if weights is None:
        # If weights not given use, w = 1
        weights = np.ones_like(lifetimes, dtype=np.float32)
    freq_lifetimes = np.bincount(lifetimes, weights)
    freq_lifetimesDead = np.bincount(lifetimes, weights * deads)
    nAlive = freq_lifetimes[::-1].cumsum()[::-1]

    KMLambda = freq_lifetimesDead / nAlive
    KMProd = (1 - KMLambda).cumprod(0)
    return KMProd

def findSurvivalDistribution(lifetimes, deads, weights=None):
    if type(lifetimes) == np.ndarray and type(deads) == np.ndarray:
        return _findSurvivalDistribution_np(lifetimes, deads, weights)
    elif type(lifetimes) == torch.Tensor and type(deads) == torch.Tensor:
        return _findSurvivalDistribution_torch(lifetimes, deads, weights)
    else:
        raise NotImplementedError

def _findSurvivalDistrosPerUser(lifetimes, deads, labels):
    maxT = lifetimes.max()
    survivalDistrosPerUser = np.zeros((lifetimes.shape[0], maxT + 1))
    for i in np.unique(labels):
        distro = findSurvivalDistribution(lifetimes[labels == i], deads[labels == i])
        if len(distro) == 0:
            distro = np.pad(
                distro, pad_width=(0, maxT + 1 - distro.shape[0]), mode="constant", constant_values=0.0
            )
        else:
            distro = np.pad(
                distro, pad_width=(0, maxT + 1 - distro.shape[0]), mode="minimum"
            )
        survivalDistrosPerUser[labels == i] = distro

    return survivalDistrosPerUser

def get_logrank_p(times, events, assignments):
    logrank_results = statistics.pairwise_logrank_test(times, assignments, events)
    return logrank_results.p_value

def write_protes(protes, write_path):
    with open(write_path, 'w') as f:
        for prote in protes:
            f.write("%s\n" % prote)

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

def welch_ttest(x1, x2, alternative):
    n1 = x1.size
    n2 = x2.size
    m1 = np.mean(x1)
    m2 = np.mean(x2)
    v1 = np.var(x1, ddof=1)
    v2 = np.var(x2, ddof=1)
    tstat = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2)**2 / (v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))
    if alternative == "equal":
        p = 2 * t.cdf(-abs(tstat), df)
    if alternative == "lesser":
        p = t.cdf(tstat, df)
    if alternative == "greater":
        p = 1-t.cdf(tstat, df)
    return tstat, df, p

def one_sample_ttest(x1, mean, alternative):
    n = x1.size
    m = np.mean(x1)
    v = np.var(x1, ddof=1)
    tstat = (m - mean) / np.sqrt(v / n)
    df = n - 1
    if alternative == "equal":
        p = 2 * t.cdf(-abs(tstat), df)
    if alternative == "lesser":
        p = t.cdf(tstat, df)
    if alternative == "greater":
        p = 1-t.cdf(tstat, df)
    return tstat, df, p
    
def write_protes(protes, write_path):
    with open(write_path, 'w') as f:
        for prote in protes:
            f.write("%s\n" % prote)

def read_hcc_samples(data_path):
    with open(data_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        protes = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                samples = row[1:]
            else:
                protes.append(row[0])
                data.append([float(elem) if elem != 'NA' else np.nan for elem in row[1:]])
    data = np.asarray(data).transpose()
    return data, protes, samples

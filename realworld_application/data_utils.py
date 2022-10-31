import csv, re, copy, os, time, progressbar, itertools, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
from os.path import join
from collections import Counter
from lifelines import KaplanMeierFitter, statistics, plotting
from lifelines.utils import restricted_mean_survival_time
from scipy import stats
from data_regroups import *

def load_data(args, gene2prote_dict, prote2gene_dict, 
        load_feat_prote=None, log10=False, zscore=True, f1269=True, clip=60, discrete=False, inequal_quantile=False
    ):
    data_path = args.data_path
    # read hcc101 data, patients, subtypes, OS, status
    print('nature data')
    datasets = []
    data, patients, protes, subtypes, OS, status, recurrence, DFS = read_hcc101_data(
        join(data_path, 'sub_data_quantile.csv'),
        join(data_path, 'Nature_101HCC_subtype2sampleID.txt'),
        join(data_path, 'HCC_101tumor_Nature_survival.txt')
    )
    status[OS>clip] = 0
    OS[OS>clip] = clip
    recurrence[DFS>clip] = 0
    DFS[DFS>clip] = clip
    valid_feat_ids = feat_selection(data, args.nan_thresh)
    data = data[:, valid_feat_ids]
    protes = [protes[prote_id] for prote_id in valid_feat_ids]
    if log10:
        data[np.isnan(data)] = np.nanmin(data)
        data = np.log10(data)
    else:
        data[np.isnan(data)] = 0
    data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5) if zscore else data
    datasets.append({
            'data': data, 
            'patients': patients,
            'protes': protes,
            'subtypes': subtypes,
            'OS': OS,
            'DFS': DFS,
            'status': status,
            'recurrence': recurrence,
            'label': 'Nature'
        }
    )

    # read center123 data
    for data_id in range(1, 4):
        print('center ', data_id, ' data')
        reads = read_center_data(
            join(data_path, 'temp{:d}_quantile.csv'.format(data_id)),
            join(data_path, 'ClinInfo_Center123_1024Patients_普适性subtype-20210330-Risk.csv'.format(data_id))
        )
        data, patients, protes, OS, status, bclc, recurrence, t_size, hbv, hcv, DFS, t_num, mvi, afp, diff, risk = reads
        status[OS>clip] = 0
        OS[OS>clip] = clip
        recurrence[DFS>clip] = 0
        DFS[DFS>clip] = clip
        valid_feat_ids = feat_selection(data, args.nan_thresh)
        data = data[:, valid_feat_ids]
        protes = [protes[prote_id] for prote_id in valid_feat_ids]
        if log10:
            data[np.isnan(data)] = np.nanmin(data)
            data = np.log10(data)
        else:
            data[np.isnan(data)] = 0
        feat_mean, feat_std = np.mean(data, axis=0), np.std(data, axis=0)
        data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5) if zscore else data

        datasets.append({
                'data': data,
                'patients': patients,
                'protes': protes, 
                'raw_protes': protes,
                'OS': OS,
                'status': status,
                'recurrence': recurrence,
                'bclc': bclc,
                't_size': t_size,
                'hbv': hbv,
                'hcv': hcv,
                'DFS': DFS,
                't_num': t_num,
                'mvi': mvi,
                'afp': afp,
                'diff': diff,
                'risk': risk,
                'label': 'Center' + str(data_id),
                'feat_mean': feat_mean,
                'feat_std': feat_std
            }
        )

    # read Cell data
    print('cell data')
    data, patients, genes, OS, status, DFS, recurrence, bclc = read_Cell_data(
        join(data_path, 'Fudan_Cell_HCC159_6478gene_nolog2.csv'),
        join(data_path, 'ClinicalInfo_Fudan_Cell_159sample.csv')
    )
    status[OS>clip] = 0
    OS[OS>clip] = clip
    recurrence[DFS>clip] = 0
    DFS[DFS>clip] = clip
    valid_feat_ids = []
    protes = []
    for gene_id, gene in enumerate(genes):
        if gene in gene2prote_dict.keys():
            protes.append(gene2prote_dict[gene])
            valid_feat_ids.append(gene_id)
    data = data[:, valid_feat_ids]
    valid_feat_ids = feat_selection(data, args.nan_thresh)
    data = np.c_[data[:, valid_feat_ids], np.zeros([len(data), 1])]
    protes = [protes[prote_id] for prote_id in valid_feat_ids]
    if log10:
        data[data==0] = np.amin(data[data>0])
        data = np.log10(data)
    data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5) if zscore else data
    cell_dataset = {
        'data': data,
        'patients': patients,
        'protes': protes, 
        'OS': OS,
        'status': status,
        'DFS': DFS,
        'recurrence': recurrence,
        'bclc': bclc,
        'label': 'Cell'
    }
    datasets.append(cell_dataset)

    # read the nature selected protein list
    nature_prote_list = read_prior_protein_list(
        join(data_path, '2017-04-04932E-Supplementary Table 11.xlsx'),
        args.logfc_thresh
    )

    if load_feat_prote is not None:
        selected_protes = read_protein_list(load_feat_prote, unique=True)
        temp_selected_protes = copy.deepcopy(selected_protes)
        check_protes = get_the_intersection([dataset['protes'] for dataset in datasets[:-1]])
        for prote in selected_protes:
            if prote not in check_protes or prote not in prote2gene_dict.keys():
                temp_selected_protes.remove(prote)
                continue
            if prote not in nature_prote_list:
                nature_prote_list += [prote]
            for dataset_id, dataset in enumerate(datasets):
                if prote not in dataset['protes']:
                    datasets[dataset_id]['protes'].append(prote)
                    datasets[dataset_id]['data'] = np.c_[
                        datasets[dataset_id]['data'], 
                        np.zeros([len(datasets[dataset_id]['data']), 1], dtype=np.float32)
                    ]
                    print('append gene ', prote2gene_dict[prote], ' to dataset ', dataset['label'])
        selected_protes = temp_selected_protes
        print('Selected genes: ', [prote2gene_dict[prote] for prote in selected_protes])

    # get the instersection of proteins
    lists = [datasets[i]['protes'] for i in range(len(datasets))]
    if f1269:
        lists += [nature_prote_list]

    special_list = []
    if args.blood_flag:
        blood_prote_list = read_blood_protein_list(
            join(data_path, 'HCC-Proteomics-Annotation-20210226.xlsx')
        )
        special_list += blood_prote_list
    if args.drug_flag:
        drug_target_prote_list = read_drug_target_protein_list(data_path)
        special_list += drug_target_prote_list
    if args.candidate_31_flag:
        candidate_31_prote_list = read_protein_list(join(data_path, '31protes.txt'))
        print('candidate_31_prote num:', len(candidate_31_prote_list))
        special_list += candidate_31_prote_list
    if args.blood_flag or args.drug_flag or args.candidate_31_flag:
        print('special_list length:', len(special_list))
        lists += [get_unique(special_list)]

    if args.consistency:
        consistent_prote_list = read_consistent_protein_list(
            join(data_path, 'HCC-Proteomics-Annotation-20210226.xlsx')
        )
        lists += [consistent_prote_list]
    if load_feat_prote is not None:
        lists += [selected_protes]
    intersected_protes = get_the_intersection(lists)
    print('intersected feat num: ', len(intersected_protes))

    # adjust the data according to the intersected_protes
    for dataset_id in range(len(datasets)):
        ids = [datasets[dataset_id]['protes'].index(prote) for prote in intersected_protes]
        datasets[dataset_id]['data'] = datasets[dataset_id]['data'][:, ids]
        datasets[dataset_id]['protes'] = [datasets[dataset_id]['protes'][prote_id] for prote_id in ids]

    for dataset_id, dataset in enumerate(datasets):
        print('raw dataset ', dataset_id, ' sample num: ', len(dataset['data']))

    if discrete:
        datasets = discretize_data(
            datasets, -1., [219./868., (219.+332.)/868.]
        ) if inequal_quantile else discretize_data(datasets)

    return datasets, intersected_protes

def load_blood_data(data_path, nan_thresh, log10, zscore, f1269, discrete, inequal_quantile, clip):
    print('FZ blood data')
    df = pd.read_csv(
        join(data_path, '归一化与填充结果中心3', '2原始数据及归一化后的结果', 'Quantile-normalized.txt'), 
        delimiter = '\t'
    )
    data = df.to_numpy()[:, 1:].astype(np.float32).transpose()
    protes = df.to_numpy()[:, 0].tolist()
    blood_ids = [int(blood_id.split('_')[1]) for blood_id in list(df.columns)[1:]]

    df = pd.read_excel(
        join(data_path, '归一化与填充结果中心3', '中心三 312例血清样本和组织样本编号对照表（2021年4月2日）.xlsx'),
        sheet_name='Sheet1',
        header=[0],
        engine='openpyxl'
    )
    blood_id2patient_dict = {}
    for patient, blood_id in zip(df[' 组织样本编号'].to_list(), df['术前血清编号'].to_list()):
        blood_id2patient_dict[blood_id] = patient
    patients = [blood_id2patient_dict[blood_id] for blood_id in blood_ids]
    
    df = pd.read_csv(
        join(data_path, 'ClinInfo_Center123_1024Patients_普适性subtype-20210330.csv'), 
    )
    c123_patients = df['ID'].to_list()
    selected_patients = [c123_patients.index(patient) for patient in patients]

    OS = df['OS.Month'].to_numpy()[selected_patients]
    status = df['Died.of.Recurrence'].to_numpy()[selected_patients]
    DFS = df['DFS.Month'].to_numpy()[selected_patients]
    recurrence = df['Cancer.Recurrence'].to_numpy()[selected_patients]
    bclc = df['BCLC.stage'].to_list()
    bclc_dict = {'0': 0, 'A': 1, 'B': 2, 'C': 3}
    bclc = np.asarray([bclc_dict[stage] for stage in bclc])[selected_patients]
    t_num = df['Tumor.Number'].to_numpy()[selected_patients]
    t_size = df['Diameter.of.Tumor..cm.'].to_numpy()[selected_patients]
    subtypes = df['Subtype'].to_list()
    subtype_dict = {'S-I': 0, 'S-II': 1, 'S-III': 2}
    subtypes = np.asarray([subtype_dict[subtype] for subtype in subtypes])[selected_patients]

    condition = np.logical_or(bclc >= 2, np.logical_and(t_num == 1, t_size >= 10))
    print('Data', data.shape, 'B:', np.sum(bclc >= 2), 'S&L:', np.sum(np.logical_and(t_num == 1, t_size >= 10)), 'Total:', np.sum(condition))

    status[OS>clip] = 0
    OS[OS>clip] = clip
    recurrence[DFS>clip] = 0
    DFS[DFS>clip] = clip
    valid_feat_ids = feat_selection(data, nan_thresh)
    data = data[:, valid_feat_ids]
    protes = [protes[prote_id] for prote_id in valid_feat_ids]
    if log10:
        data[np.isnan(data)] = np.nanmin(data)
        data = np.log10(data)
    else:
        data[np.isnan(data)] = 0
    data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5) if zscore else data

    dataset = {
        'data': data[condition],
        'patients': [patient for patient, is_true in zip(patients, condition) if is_true],
        'protes': protes, 
        'OS': OS[condition],
        'status': status[condition],
        'recurrence': recurrence[condition],
        'DFS': DFS[condition],
        'label': 'FZ_blood',
        'subtypes': subtypes[condition]
    }
    print('data shape:', data[condition].shape)

    datasets = [dataset]

    if discrete:
        datasets = discretize_data(
            datasets, -1., [np.sum(subtypes==0)/np.len(subtypes), np.sum(subtypes<=1)/np.len(subtypes)]
        ) if inequal_quantile else discretize_data(datasets)

    return datasets, protes

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

    # read OS
    if '.txt' in z_path:
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
    else:
        df = pd.read_excel(
            z_path, 
            sheet_name='Sheet1',
            header=[0],
            engine='openpyxl'
        )
        OS = np.zeros([len(patients)])
        status = np.zeros([len(patients)])
        DFS = np.zeros([len(patients)])
        recurrence = np.zeros([len(patients)])
        patient_list = df['No.'].tolist()
        patient_list = [patient[:-1]+'T' for patient in patient_list]
        OS_list = df['总生存时间Total follow up period(m)'].tolist()
        status_list = df['Died of recurrence(0=No;1=Yes)'].tolist()
        DFS_list = df['复发时间Disease free survival(m)'].tolist()
        recurrence_list = df['Cancer recurrence(0=No;1=Yes)'].tolist()
        valid = np.zeros([len(patients)], dtype=bool)
        for i in range(len(patient_list)):
            ID = patient_list[i]
            if ID in patients:
                idx = patients.index(ID)
                status[idx] = status_list[i]
                OS[idx] = OS_list[i]
                DFS[idx] = DFS_list[i]
                recurrence[idx] = recurrence_list[i]
                valid[idx] = True
            else:
                print('missing ', patient_list[i])
        data = data[valid]
        patients = [patient for is_valid, patient in zip(valid, patients) if is_valid]
        OS = OS[valid]
        status = status[valid]
        recurrence = recurrence[valid]
        DFS = DFS[valid]    
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
        bclc = np.zeros([len(patients)])
        bclc_dict = {'0': 0, 'A': 1, 'B': 2, 'C': 3}
        bool_dict = {'-': 0, '+': 1}
        recurrence = np.zeros([len(patients)])
        t_size = np.zeros([len(patients)])
        hbv = np.zeros([len(patients)])
        hcv = np.zeros([len(patients)])
        t_num = np.zeros([len(patients)])
        mvi = np.zeros([len(patients)])
        diff = np.zeros([len(patients)])
        diff_dict = {'Edmondson I-II': 0, 'Edmondson III-IV': 1, 'NA': -1}
        afp = np.zeros([len(patients)])
        risk = np.zeros([len(patients)])
        risk_dict = {'Low': 0, 'Middle': 1, 'High': 2}
        valid = np.zeros([len(patients)], dtype=bool)
        for i, row in enumerate(csv_reader):
            if i > 0:
                ID = 'T' + row[0] if 'G' not in row[0] and 'T' not in row[0] else row[0]
                if ID in patients:
                    idx = patients.index(ID)
                    OS[idx] = float(row[18])
                    status[idx] = int(row[17])
                    bclc[idx] = bclc_dict[row[8]]
                    recurrence[idx] = int(row[15])
                    t_size[idx] = float(row[5])
                    hbv[idx] = bool_dict[row[11]]
                    hcv[idx] = bool_dict[row[12]]
                    DFS[idx] = float(row[16])
                    t_num[idx] = int(row[4]) if '≥' not in row[4] else int(row[4][1:])
                    mvi[idx] = int(row[6])
                    risk[idx] = risk_dict[row[22]]
                    if '>' in row[10]:
                        afp[idx] = int(float(row[10][1:]) > 400)
                    elif row[10] == 'NA':
                        afp[idx] = -1
                    else:
                        afp[idx] = int(float(row[10]) > 400)
                    diff[idx] = diff_dict[row[14]]
                    valid[idx] = True
    data = data[valid]
    if np.sum(valid) < len(valid):
        print(len(valid) - np.sum(valid), 'patients missing when loading clinicalinfo')
    patients = [patient for is_valid, patient in zip(valid, patients) if is_valid]
    OS = OS[valid]
    status = status[valid]
    bclc = bclc[valid]
    recurrence = recurrence[valid]
    t_size = t_size[valid]
    hbv = hbv[valid]
    hcv = hcv[valid]
    DFS = DFS[valid]
    t_num = t_num[valid]
    mvi = mvi[valid]
    afp = afp[valid]
    diff = diff[valid]
    risk = risk[valid]
    return [data, patients, protes, OS, status, bclc, recurrence, t_size, hbv, hcv, DFS, t_num, mvi, afp, diff, risk]

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

    bclc_dict = {'0': 0, 'A': 1, 'B': 2, 'C': 3}
    with open(y_path, mode='r', encoding='UTF-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        OS = np.zeros([len(patients)])
        status = np.zeros([len(patients)])
        DFS = np.zeros([len(patients)])
        recurrence = np.zeros([len(patients)])
        bclc = np.zeros([len(patients)])
        for i, row in enumerate(csv_reader):
            if i > 0:
                patient_id = patients.index(row[0])
                DFS[patient_id] = row[5]
                OS[patient_id] = row[6]
                recurrence[patient_id] = row[7]
                status[patient_id] = row[8]
                bclc[patient_id] = bclc_dict[row[23]]
    return data, patients, genes, OS, status, DFS, recurrence, bclc 

def read_BCLC_C_data(x_path, y_path):
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

    df = pd.read_excel(
        y_path, 
        sheet_name='Sheet1',
        header=[0],
        engine='openpyxl'
    )
    OS = np.zeros([len(patients)])
    status = np.zeros([len(patients)])
    DFS = np.zeros([len(patients)])
    recurrence = np.zeros([len(patients)])
    center_id = np.zeros([len(patients)])
    patient_list = df['病例号'].tolist()
    OS_list = df['总生存期'].tolist()
    status_list = df['是否死亡(0=No;1=Yes)'].tolist()
    DFS_list = df['无病生存期'].tolist()
    recurrence_list = df['是否复发(0=No;1=Yes)'].tolist()
    center_list = df['Cohort'].tolist()
    valid = np.zeros([len(patients)], dtype=bool)
    center_dict = {'SH': 0, 'GZ': 1, 'FZ': 2}
    for i in range(len(patient_list)):
        ID = patient_list[i]
        if ID in patients:
            idx = patients.index(ID)
            status[idx] = status_list[i]
            OS[idx] = OS_list[i]
            DFS[idx] = DFS_list[i]
            recurrence[idx] = recurrence_list[i]
            center_id[idx] = center_dict[center_list[i]]
            valid[idx] = True
        else:
            print('missing ', patient_list[i])
    data = data[valid]
    patients = [patient for is_valid, patient in zip(valid, patients) if is_valid]
    OS = OS[valid]
    status = status[valid]
    recurrence = recurrence[valid]
    DFS = DFS[valid]
    center_id = center_id[valid]    

    return data, protes, patients, OS, status, DFS, recurrence, center_id

def load_BCLC_C_data(data_path, intersected_protes, datasets, prote2gene_dict):
    data, protes, patients, OS, status, DFS, recurrence, center_id = read_BCLC_C_data(
        join(data_path, 'tempC_quantile.csv'),
        join(data_path, '临床信息统计1024例+C期样本信息83例（汇总-20210304）.xlsx')
    )
    data[np.isnan(data)] = 0
    # data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5)
    for i, dataset in enumerate(datasets[1:4]):
        print(dataset['label'])
        center_feat_mean = dataset['feat_mean']
        center_feat_std = dataset['feat_std']
        center_protes = dataset['raw_protes']
        for prote, mean, std in zip(center_protes, center_feat_mean, center_feat_std):
            if prote in protes and prote in intersected_protes:
                sub_data = data[center_id==i, :]
                sub_data[:, protes.index(prote)] = (sub_data[:, protes.index(prote)] - mean) / std
                data[center_id==i] = sub_data
        
    feat_min_vals = np.amin(np.concatenate([dataset['data'] for dataset in datasets[1:4]]), axis=0)
    for prote, min_val in zip(intersected_protes, feat_min_vals.tolist()):
        if prote not in protes:
            print('append ' + prote2gene_dict[prote] + 'to data')
            protes.append(prote)
            data = np.c_[data, np.ones((len(data), 1)) * min_val]

    feat_ids = [protes.index(prote) for prote in intersected_protes]
    return data[:, feat_ids], patients, OS, status, DFS, recurrence

def read_hcc_drug_target(data_path):
    df = pd.read_excel(
        data_path, 
        sheet_name='Sheet1',
        header=[0],
        engine='openpyxl'
    )
    prote_list = df['蛋白1'].tolist()
    return prote_list

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

def read_blood_protein_list(data_path):
    prote_info_dict = read_protein_info(data_path)
    blood_protes = [prote for prote, in_blood in zip(
        prote_info_dict['uniprotAccession'], 
        prote_info_dict['血浆蛋白']
    ) if in_blood == 'T']
    print('blood prote num: ', len(blood_protes))
    return blood_protes

def read_consistent_protein_list(data_path):
    prote_info_dict = read_protein_info(data_path)
    consistent_protes = [prote for prote, OS in zip(
        prote_info_dict['uniprotAccession'], 
        prote_info_dict['T-OS-Summary']
    ) if OS not in ['No trend']]
    print('consistent prote num: ', len(consistent_protes))
    return consistent_protes

def read_prognosis_relevant_list(data_path):
    prote_info_dict = read_protein_info(data_path)
    prognosis_relevant_protes_list = []
    for col_name in [
        'OS 一期肿瘤', 'OS 中心1肿瘤', 'OS 中心2肿瘤', 'OS 中心3肿瘤', 
        'DFS 中心1肿瘤', 'DFS 中心2肿瘤', 'DFS 中心3肿瘤'
    ]:
        prognosis_relevant_protes = [prote for prote, event in zip(
            prote_info_dict['uniprotAccession'], 
            prote_info_dict[col_name]
        ) if event in ['Favor', 'Unfavor']]
        prognosis_relevant_protes_list.append(prognosis_relevant_protes)
    prognosis_relevant_protes = get_the_intersection(prognosis_relevant_protes_list)
    print('prognosis relevant prote num: ', len(prognosis_relevant_protes))
    return prognosis_relevant_protes

def read_high_TP_ratio_prote_list(data_path):
    prote_info_dict = read_protein_info(data_path)
    high_TP_ratio_protes_list = []
    for col_name in [
        '一期 N/T>1.5百分比', '中心1 N/T>1.5百分比', '中心2 N/T>1.5百分比'
    ]:
        high_TP_ratio_protes = [prote for prote, TP_ratio in zip(
            prote_info_dict['uniprotAccession'], 
            prote_info_dict[col_name]
        ) if TP_ratio > 0.5]
        high_TP_ratio_protes_list.append(high_TP_ratio_protes)
    high_TP_ratio_protes = get_the_intersection(high_TP_ratio_protes_list)
    print('high TP ratio prote num: ', len(high_TP_ratio_protes))
    return high_TP_ratio_protes

def read_nature_feat_prote_list(data_path):
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        protes = []
        feats = []
        feat_dict = {
            'S-I Up-regulation': 0, 
            'S-I Down-regulation': 1, 
            'S-II Up-regulation': 2, 
            'S-II Down-regulation': 3, 
            'S-III Up-regulation': 4, 
            'S-III Down-regulation': 5
        }
        for i, row in enumerate(csv_reader):
            if i > 0:
                protes.append(row[0])
                feats.append(feat_dict[row[2]])
    return protes, np.asarray(feats)

def read_28_proteins(data_path):
    df = pd.read_excel(
        data_path, 
        sheet_name='28protein_source',
        header=[0],
        engine='openpyxl'
    )
    prote_list = df['ID '].tolist()[:28]
    feat_subtypes = np.asarray(df['亚型'].tolist()[:28]) - 1
    return prote_list, feat_subtypes
    
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

def read_drug_target_protein_list(data_path):
    with open(join(data_path, 'pharmacologically_active.csv'), 'rb') as csvfile:
        df = pd.read_csv(csvfile, header=[0])
        prote_list = df['UniProt ID'].tolist()
    prote_list = get_unique(prote_list)
    print('drug target prote num:', len(prote_list))
    return prote_list

def read_training_records(file_path):
    df = pd.read_csv(
        file_path, 
        header=[0]
    )
    train_acc = df['acc'].to_numpy()
    test_acc = df['val_acc'].to_numpy()
    train_os = df['os_diff'].to_numpy()
    test_os = df['val_os_diff'].to_numpy()
    return np.stack([train_acc, test_acc, train_os, test_os], axis=-1)

def search_for_files(dir, file_end_with):
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(file_end_with):
                # print(os.path.join(root, file))
                file_list.append(os.path.join(root, file))
    return file_list

def find_all_protes(data_path):
    file_list = search_for_files(data_path, 'selected_protes_.txt')
    prote_list = []
    hcc_drug_target_prote_list = read_hcc_drug_target(
        os.path.join(r'G:\Data\Tumer_clinic', 'HCC药靶.xlsx')
    )
    for file in file_list:
        protes = read_list(file)
        prote_list += protes
        for drug_target_prote in hcc_drug_target_prote_list:
            if drug_target_prote in protes:
                print(
                    'find drug target prote: ', 
                    drug_target_prote, 
                    ' in ', 
                    file
                )
    print(Counter(prote_list))

    return prote_list

def count_protes(prote_list):
    return Counter(prote_list)

def retrieve_prote_info(prote_info_dict, prote_list):
    info_key_list = prote_info_dict['keys']
    sub_prote_info_dict = {}
    for info_key in info_key_list:
        info_list = prote_info_dict[info_key]
        if info_key == 'uniprotAccession':
            prote_ids = [info_list.index(prote) for prote in prote_list]
        sub_prote_info_dict[info_key] = [info_list[prote_id] for prote_id in prote_ids]
    return sub_prote_info_dict

def display_prote_info(prote_info_dict):
    chinese_str_dict = {
        '一期': 'Nature', 
        '百分比': ' percentage', 
        '中心': 'center', 
        '樊嘉': 'Cell', 
        '肿瘤': '',
        '特征基因': ' feat prote',
        '血浆蛋白': 'blood prote',
        '药靶蛋白': 'drug target prote',
        '文献数目': ' pub num'
    }
    print_str = ''
    width = 5
    for key in prote_info_dict.keys():
        line = ''
        key_str = key
        for chinese in chinese_str_dict.keys():
            if chinese in key_str:
                key_str = key_str.replace(chinese, chinese_str_dict[chinese])
        line += '{:<21}'.format(key_str)
        for info in prote_info_dict[key]:
            line += '{:>12}'.format(info)
        line += '\n'
        print_str += line
    return print_str

def read_protein_info(file_path):
    df = pd.read_excel(
        file_path, 
        sheet_name='HCC-Proteomics-Annotation',
        header=[0],
        engine='openpyxl'
    )
    info_key_list = [
        'uniprotAccession', 
        'GeneName',
        '一期 N/T>1.5百分比',
        '中心1 N/T>1.5百分比',
        '中心2 N/T>1.5百分比',
        'Cell T/N>1.5百分比',
        'OS 一期肿瘤',
        'OS 中心1肿瘤',
        'OS 中心2肿瘤',
        'OS 中心3肿瘤',
        'OS Cell肿瘤',
        'T-OS-Summary',
        'DFS 一期肿瘤',
        'DFS 中心1肿瘤',
        'DFS 中心2肿瘤',
        'DFS 中心3肿瘤',
        'DFS Cell肿瘤',
        'T-DFS-Summary',
        '血浆蛋白',
        '药靶蛋白',
        'Pubmed文献数目',
        '一期特征基因'
        ]

    protes_info_dict = {}
    for info_key in info_key_list:
        info_list = df[info_key].tolist()
        protes_info_dict[info_key] = info_list
    protes_info_dict['keys'] = info_key_list
    return protes_info_dict



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

def feature_selection(data):
    """
        Select features with large variance
    """
    nan_num = np.sum(np.isnan(data), axis=0)
    data = data[:, nan_num <= data.shape[0] * 0.7]

    var = np.nanvar(data, axis=0)
    mean = np.nanmean(data, axis=0)
    large_var_idx = (-mean/var).argsort()[:int(len(var)*0.7)]
    data = data[:, large_var_idx]

    return data

def nan_imputation(data, val):
    if val <= 0 :
        data[np.isnan(data)] = val
    elif val > 0:
        col_num = data.shape[1]
        for col_id in range(col_num):
            col = data[:, col_id]
            if np.isnan(col).all() :
                data[:, col_id] = 0.
            else:
                clean_col = col[(1 - np.isnan(col)).astype(bool)]
                sorted_col = np.sort(clean_col)
                data[np.isnan(col), col_id] = sorted_col[int(len(sorted_col)*val)]
    return data

def feature_normalization(data):
    """
        normalize data along each dimension
    """
    n_sample, dim = np.shape(data)

    mean_vals = np.nanmean(data, axis=0, keepdims=True)
    std_vals = np.nanstd(data, axis=0, keepdims=True)
    A = data - np.tile(mean_vals, [n_sample, 1])
    B = std_vals
    B[B == 0] = 1
    B = np.tile(B, [n_sample, 1])
    data = A/B
    return data

def custom_normalization(data, v_min, no_split=False, bias=1, scale=0.2):
    if not no_split:
        vect_list = []
        bot = -10000
        for vect in np.hsplit(data, data.shape[1]):
            vect = np.squeeze(vect)
            vect[vect==v_min] = bot
            if np.std(vect[vect>bot]) > 0:
                vect[vect>bot] = (vect[vect>bot] - np.mean(vect[vect>bot]))/np.std(vect[vect>bot])
            vect_list.append(vect)
        data = np.stack(vect_list, axis=1)
        data[data>bot] = data[data>bot] * scale + bias
        data[data==bot] = 0.
    else:
        vect_list = []
        for vect in np.hsplit(data, data.shape[1]):
            vect = np.squeeze(vect)
            if np.std(vect) > 0:
                vect= (vect - np.mean(vect))/np.std(vect)
            vect_list.append(vect)
        data = np.stack(vect_list, axis=1)
        data = data * scale + bias
    return data

def feature_pre_check(data):
    for col_id in range(data.shape[1]):
        col = data[:, col_id]
        clean_col = col[(1 - np.isnan(col)).astype(bool)]
        plt.hist(clean_col, bins=20)
        plt.show()

def feature_cls_viz(data, label):
    for col_id in range(data.shape[1]):
        col1 = data[label==0, col_id]
        col2 = data[label==1, col_id]
        clean_col1 = col1[(1 - np.isnan(col1)).astype(bool)]
        clean_col2 = col2[(1 - np.isnan(col2)).astype(bool)]
        plt.figure()
        plt.hist([clean_col1, clean_col2], 20, 
                density=True, 
                histtype='bar', 
                color=['tab:blue', 'tab:green']
            )
        plt.legend(['mvi=0', 'mvi=1'])
        plt.grid(True)
        plt.show()

def data_ceiling(data, val):
    for col_id in range(data.shape[1]):
        col = data[:, col_id]
        clean_col = col[(1 - np.isnan(col)).astype(bool)]
        ceil = np.sort(clean_col)[int(len(clean_col) * (1 - val))]
        col[col > ceil] = ceil
        data[:, col_id] = col
    return data

def feat_selection(data, thresh=0.7):
    valid_ids = []
    for col_id, col in enumerate(data.T):
        if np.sum(np.isnan(col)) <= len(col) * thresh:
            valid_ids.append(col_id)
    print('nan pass prote num:', len(valid_ids))
    return np.asarray(valid_ids)

def dataset_feature_consistency_selection(data_path, datasets, intersected_protes, rank_range=500):
    datasets_subtypes_genes_mean = []
    for i, dataset in enumerate(datasets):
        subtypes = read_assignments(join(
                data_path,
                'reinforced_classification',
                'reinforce_datasets140_hdim3_bias_losssu004',
                'results0',
                'test_' + dataset['label'] + '_classification_result.csv'
            ), dataset['patients']
        )
        subtypes_genes_mean = []
        for subtype in range(3):
            patients_genes = dataset['data'][subtypes==subtype, :]
            genes_mean = np.mean(patients_genes, axis=0)
            subtypes_genes_mean.append(genes_mean)
        subtypes_genes_mean = np.stack(subtypes_genes_mean, axis=0)
        datasets_subtypes_genes_mean.append(subtypes_genes_mean)
    datasets_subtypes_genes_mean = np.stack(datasets_subtypes_genes_mean, axis=0)
    genes_subtypes_datasets_mean = np.transpose(datasets_subtypes_genes_mean, [2, 1, 0])
    genes_subtypes_std = np.std(genes_subtypes_datasets_mean, axis=-1)
    genes_max_std = np.amax(genes_subtypes_std, axis=-1)
    sorted_ids = np.argsort(genes_max_std)[:rank_range]

    for dataset_id, dataset in enumerate(datasets):
        datasets[dataset_id]['data'] = dataset['data'][:, sorted_ids]
        datasets[dataset_id]['protes'] = [intersected_protes[sorted_id] for sorted_id in sorted_ids]
    intersected_protes = [intersected_protes[sorted_id] for sorted_id in sorted_ids]
    return datasets, intersected_protes

def rank_dict(data, prote_list):
    mean = np.nanmedian(data, axis=0)
    ranked_ids = np.argsort(-mean)
    prote_rank_dict = {}
    for i, ranked_id in enumerate(ranked_ids):
        prote_rank_dict[prote_list[ranked_id]] = i
    return prote_rank_dict

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

@jit
def get_stats(times, events, clip_time=60):
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
    else:
        return RMST

def get_rmst(time, event, label):
    kmf = KaplanMeierFitter().fit(time, event, label=label)
    return restricted_mean_survival_time(kmf, t=60)

def get_delta_rmst(time, event, assignments, k, return_RMST=False):
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

def get_logrank_p(time, event, assignments):
    logrank_results = statistics.pairwise_logrank_test(time, assignments, event)
    return logrank_results.p_value

def get_mutivarate_logrank_p_custom(time, event, assignments, k):
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

def get_logrank_p_custom(time, event, assignments, k):
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

def write_prote_log(save_path, prote2gene_dict, rank_dict_list, nature_prote_list, blood_prote_list):
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Protein', 'Gene', 
            'NatureRank', 'Center1Rank', 'Center2Rank', 'Center3Rank', 'CellRank',
            'LogFC', 'BloodDetectable'])
        for prote in prote2gene_dict.keys():
            if prote2gene_dict[prote] is np.nan:
                continue
            row = []
            row += [prote, prote2gene_dict[prote]]
            for i in range(5):
                rank = [rank_dict_list[i][prote]] \
                    if prote in rank_dict_list[i].keys() else [np.nan]
                row += rank
            # logfc
            row += [prote in nature_prote_list]
            # row += if_in_natire_prote_list
            # blood detectable
            row += [prote in blood_prote_list]
            writer.writerow(row)

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

def read_comb(data_path, gene2prote_dict):
    comb_list = []
    acc_list = []
    for i in range(1, 22):
        with open(join(data_path, 'result{:}.txt'.format(i))) as csv_file:
            csv_reader = csv.reader(csv_file)
            for i, row in enumerate(csv_reader):
                str_list = re.split(r'\t+', row[0])
                prote_list = [gene2prote_dict[gene] for gene in str_list[:-1]]
                comb_list.append(prote_list)
                acc_list.append(1 - float(str_list[-1]))
    acc_list = np.asarray(acc_list)
    sorted_id = np.argsort(-acc_list)[:10000]
    print(acc_list[sorted_id])
    assert False
    comb_list = [comb_list[comb_id] for comb_id in sorted_id]
    return comb_list

def write_protes(protes, write_path):
    with open(write_path, 'w') as f:
        for prote in protes:
            f.write("%s\n" % prote)

# def single_result_check(datasets, test_path):
#     check_p005_list = [
#         'Center1_0A', 'Center2_0A', 'Center3_0A', 'Center123_0A_test'
#     ]
#     check_p01_list = ['Cell']
#     check_rmst12_list = ['Center123_B_test']
#     check_p_and_rmst12_list = ['Center123_B', 'Center123_large']

#     score = 0
#     for dataset in datasets:
#         if dataset['label'] not in check_p005_list + check_p01_list + check_rmst12_list + check_p_and_rmst12_list:
#             continue
#         assignments = read_assignments(
#             join(test_path, 'test_' + dataset['label'] + '_classification_result.csv'),
#             dataset['patients']
#         )

#         if np.amin([np.sum(assignments==subtype) for subtype in range(3)]) < 3:
#             # print('invalid assignments', [np.sum(assignments==subtype) for subtype in range(3)])
#             score -= 1
#             continue

#         OS = dataset['OS']
#         status = dataset['status']
#         DFS = dataset['DFS']
#         recurrence = dataset['recurrence']
#         for times, events in zip([OS, DFS], [status, recurrence]):
#             delta_rmst_list = get_delta_rmst(times, events, assignments, 3)
#             p_list = get_logrank_p_custom(times, events, assignments, 3)
#             if dataset['label'] in check_p005_list:
#                 score = score - 1 if max(p_list[0], p_list[2]) >= 0.05 else score
#             elif dataset['label'] in check_p01_list:
#                 score = score - 1 if max(p_list[0], p_list[2]) >= 0.1 else score
#             elif dataset['label'] in check_rmst12_list:
#                 score = score - 1 if delta_rmst_list[0] < 7 else score
#             elif dataset['label'] in check_p_and_rmst12_list:
#                 score = score - 1 if delta_rmst_list[0] < 7 or p_list[0] >= 0.1 else score
#     return score

def read_gmt(pathway_gene_dict, gene_pathway_dict, path):
    with open(path) as gmt:
        for i, line in enumerate(gmt.read().splitlines()):
            items = line.split('\t')
            if items[0] not in pathway_gene_dict.keys():
                pathway_gene_dict[items[0]] = items[2:]
            for gene in items[2:]:
                if gene not in gene_pathway_dict.keys():
                    gene_pathway_dict[gene] = [items[0]]
                else:
                    gene_pathway_dict[gene].append(items[0])
    return pathway_gene_dict, gene_pathway_dict

def pathway_mask(data_path, all_protes=None, gene2prote_dict=None, prote2gene_dict=None):
    pathway2gene_dict, gene2pathway_dict = read_gmt(
        {}, 
        {} , 
        join(data_path, 'c2.all.v7.2.symbols.gmt')
    )
    pathway2gene_dict, gene2pathway_dict = read_gmt(
        pathway2gene_dict, 
        gene2pathway_dict, 
        join(data_path, 'h.all.v7.2.symbols.gmt')
    )
    gene_list = [prote2gene_dict[prote] for prote in all_protes]
    pathway_list = list(pathway2gene_dict.keys())
    mask = np.zeros((len(gene_list), len(pathway_list)), dtype=np.uint8) # gene num x pathway num
    for gene in gene_list:
        if gene in gene2pathway_dict.keys():
            for pathway in gene2pathway_dict[gene]:
                mask[gene_list.index(gene), pathway_list.index(pathway)] = 1

    # sort pathway according to the number of genes included
    sorted_pathway_ids = np.argsort(np.sum(mask, axis=0))[::-1]
    sorted_pathway_list = [pathway_list[sorted_pathway_id] for sorted_pathway_id in sorted_pathway_ids]
    sorted_mask = mask[:, sorted_pathway_ids]
    # print(np.amax(sorted_mask), np.amin(sorted_mask))

    # get the curve of the growing coverage of genes w.r.t thg growing number of pathways 
    refered_times_of_each_gene = np.cumsum(sorted_mask, axis=1)
    gene_coverage = np.sum(refered_times_of_each_gene>0, axis=0)
    # fig = plt.figure()
    # plt.plot(gene_coverage)
    # plt.grid()
    # plt.xlabel('number of pathways covered, {:} in total'.format(mask.shape[1]))
    # plt.ylabel('number of genes covered, {:} in total'.format(mask.shape[0]))
    # plt.savefig(join(data_path, 'gene_coverage.png'))

    # only use the pathway that provides increments on gene coverage
    gene_coverage_tm1 = np.roll(gene_coverage, 1)
    gene_coverage_tm1[0] = 0
    delta_converage = gene_coverage - gene_coverage_tm1
    # fig = plt.figure()
    # plt.plot(delta_converage)
    # plt.grid()
    # plt.xlabel('number of pathways covered, {:} in total'.format(mask.shape[1]))
    # plt.ylabel('delta of genes covered, {:} in total'.format(mask.shape[0]))
    # plt.savefig(join(data_path, 'delta_gene_coverage.png'))
    select_list = delta_converage > 0
    final_mask = sorted_mask[:, select_list]
    final_pathway_list = [sorted_pathway for sorted_pathway, is_selected in zip(sorted_pathway_list, select_list) if is_selected]
    fig = plt.figure(figsize=(8, 32))
    plt.imshow(final_mask)
    plt.xlabel('pathway')
    plt.ylabel('gene')
    plt.title('number of connections = {:}'.format(np.sum(final_mask)))
    plt.tight_layout()
    plt.savefig(join(data_path, 'final_mask.png'))
    return final_mask, final_pathway_list

def read_nature_pathway(data_path):
    nature_pathway2name_and_type_dict = {}
    pathway2gene_dict, gene2pathway_dict = {}, {}
    with open(join(data_path, 'Nature_Extended_Fig7a_terms_Neutrophils_Immune.csv'), 'rb') as csvfile:
        df = pd.read_csv(csvfile)
        for line_list in df.values.tolist():
            nature_pathway2name_and_type_dict[line_list[0]] = line_list[1:]

    with open(join(data_path, 'geneset.csv'), 'rb') as csvfile:
        df = pd.read_csv(csvfile, dtype=str)
        sub_df = df[list(nature_pathway2name_and_type_dict.keys())]
        for pathway in nature_pathway2name_and_type_dict.keys():
            gene_list = [item for item in sub_df[pathway].tolist() if isinstance(item, str)]
            pathway2gene_dict[pathway] = gene_list
            for gene in gene_list:
                if gene not in gene2pathway_dict.keys():
                    gene2pathway_dict[gene] = [pathway]
                else:
                    gene2pathway_dict[gene].append(pathway)
    
    return pathway2gene_dict, gene2pathway_dict, nature_pathway2name_and_type_dict

def nature_pathway_mask(data_path, all_protes=None, gene2prote_dict=None, prote2gene_dict=None):
    pathway2gene_dict, gene2pathway_dict, nature_pathway2name_and_type_dict = read_nature_pathway(data_path)
    gene_list = [prote2gene_dict[prote] for prote in all_protes]
    pathway_list = list(pathway2gene_dict.keys())
    mask = np.zeros((len(gene_list), len(pathway_list)), dtype=np.uint8) # gene num x pathway num
    for gene in gene_list:
        if gene in gene2pathway_dict.keys():
            for pathway in gene2pathway_dict[gene]:
                mask[gene_list.index(gene), pathway_list.index(pathway)] = 1

    fig = plt.figure(figsize=(8, 32))
    plt.imshow(mask)
    plt.xlabel('pathway')
    plt.ylabel('gene')
    plt.title('number of connections = {:}, gene_coverage = {:}'.format(np.sum(mask), np.sum(np.sum(mask, axis=1)>0)))
    plt.tight_layout()
    plt.savefig(join(data_path, 'nature_mask.png'))
    return mask, pathway_list

def load_pseudo_label(datasets, label_path):
    label_files = search_for_files(label_path, '_classification_result.csv')
    all_subtypes, all_patients = [], []
    for label_file in label_files:
        patients, assignments = read_assignments(label_file)
        all_patients += patients
        all_subtypes.append(assignments)
    all_subtypes = np.concatenate(all_subtypes, axis=0)

    for i, dataset in enumerate(datasets):
        ids = [all_patients.index(patient) for patient in dataset['patients']]
        datasets[i]['subtypes'] = all_subtypes[ids]
    return datasets

def read_selected_protes(data_path, gene2prote_dict, target_subtype):
    df = pd.read_excel(
        data_path, 
        sheet_name='Sheet1',
        header=[0],
        engine='openpyxl'
    )
    if target_subtype==2:
        genes = df['S3_candidates'].tolist()
    elif target_subtype==0:
        genes = df['S1_candidates'].tolist()
    protes = [gene2prote_dict[gene] for gene in genes if gene in gene2prote_dict.keys()]
    return protes

def discretize_data(datasets, valid_thresh=0., thresh=[1/3., 2/3.]):
    for dataset_id, dataset in enumerate(datasets):
        for i in range(dataset['data'].shape[1]):
            col = dataset['data'][:, i]
            low_thresh = np.quantile(col[col>valid_thresh], thresh[0])
            high_thresh = np.quantile(col[col>valid_thresh], thresh[1])
            discrete_col = copy.deepcopy(col)
            discrete_col[col <= low_thresh] = -1.
            discrete_col[np.logical_and(col > low_thresh, col <= high_thresh)] = 0.
            discrete_col[col > high_thresh] = 1.
            datasets[dataset_id]['data'][:, i] = discrete_col
        datasets[dataset_id]['data'] += 1
    return datasets

def find_subtype_feature_protes(data, protes, subtypes, target_subtype, up_thresh, down_thresh):
    up_regu_protes = []
    down_regu_protes = []
    feat_protes = []
    for prote_id, prote in enumerate(protes):
        col = data[:, prote_id]
        target_subtype_val = np.mean(col[subtypes==target_subtype])
        other_subtypes_val = np.mean(col[subtypes!=target_subtype])
        ratio = target_subtype_val / (other_subtypes_val + 1e-5)
        if ratio > up_thresh:
            up_regu_protes.append(prote)
            feat_protes.append(prote)
        elif ratio < 1. / down_thresh:
            down_regu_protes.append(prote)
            feat_protes.append(prote)
    return feat_protes, up_regu_protes, down_regu_protes

if __name__ == '__main__':
    # for c in range(1, 4):
    #     print('center ', c)
    #     compare_assignments(
    #         os.path.join(
    #             r'D:\Data\Tumer_clinic\concrete_feature_selection\archive\data_before_20201128\10_features\keras_semi_pi_linear\results39_os_97_f_MNDA',
    #             'test_center{:}_classification_result.txt'.format(c)
    #         ),
    #         os.path.join(
    #             r'D:\Data\Tumer_clinic\concrete_feature_selection\archive\data_before_20201128\10_features\keras_semi_pi_linear\results39_os_97_f_MNDA',
    #             'center123_C_83T.txt'  
    #         )
    #     )
    # read_28_proteins(r'G:\Data\Tumer_clinic\28protein_综合筛选分型标志物-姜颖-20190904.xlsx')

    # times = np.array([1,2,3,3,3,5,6,30,60])
    # events = np.array([1,1,1,1,0,1,1,0,1])
    # assignments = np.array([0,2,0,1,0,1,0,0,2])

    # start_time = time.time()
    # print(get_logrank_p(times, events, assignments), time.time() - start_time)
    # start_time = time.time()
    # print(get_logrank_p_custom(times, events, assignments, np.amax(assignments)+1), time.time() - start_time)

    # start_time = time.time()
    # rmst1, density1, time1 = get_rmst_custom(times, events)
    # print(rmst1, time.time() - start_time)
    # for t, density in zip(time1, density1):
    #     print(t, '\t', density)
    # print(' ')
    # start_time = time.time()
    # kmf = KaplanMeierFitter().fit(times, events, label='0')
    # rmst2 = restricted_mean_survival_time(kmf, t=60)
    # print(rmst2, time.time() - start_time)
    # print(kmf.survival_function_)

    # prote_list = read_protein_list(
    #     r'G:\Data\Tumer_clinic\reinforced_classification\reinforce_feature_30\reinforce_d104_concrete_d104_f30_hdim16_bias_losssu004_regu0\results19\validation\selected_protes.txt'
    #     , True
    # )
    # print(len(prote_list), prote_list)

    # read_protein_info(r'G:\Data\Tumer_clinic\HCC-Proteomics-Annotation-20210226.xlsx')
    gene2prote_dict, prote2gene_dict = gene2prote_and_prote2gene(r'G:\Data\Tumor_clinic\HCC-Proteomics-Annotation-20210226.xlsx')
    # read_blood_protein_list(r'G:\Data\Tumer_clinic\HCC-Proteomics-Annotation-20210226.xlsx')

    # pathway_mask(r'G:\Data\Tumor_clinic')
    # read_nature_pathway(r'G:\Data\Tumor_clinic')
    # read_drug_target_protein_list(r'G:\Data\Tumor_clinic')  
    read_sun_selected_protes(r'G:\Data\Tumor_clinic\feature_selection', gene2prote_dict)
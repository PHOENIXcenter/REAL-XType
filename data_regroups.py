import copy
import numpy as np

def datasets_regroup(datasets, train_valid_test_groups):
    datasets_cohort = [dataset['cohort'] for dataset in datasets]
    group_labels = []
    groups = train_valid_test_groups
    for key in groups.keys():
        group_labels += groups[key]

    print('data regroup')
    data_aug_list  = [[] for _ in range(len(group_labels))]
    patients_list  = [[] for _ in range(len(group_labels))]

    for group_id, group_label in enumerate(group_labels):
        # get datasets related to the group
        rela_datasets = []
        for cohort in datasets_cohort:
            if cohort in group_label:
                rela_datasets.append(datasets[datasets_cohort.index(cohort)])
        print(group_label)
        for dataset in rela_datasets:
            data_aug = np.c_[
                dataset['data'],
                np.expand_dims(dataset['bclc'], axis=1),
                np.expand_dims(dataset['risk'], axis=1),
                np.expand_dims(dataset['hbv'], axis=1),
                dataset['subtypes'] if 'subtypes' in dataset.keys() else -1 * np.ones_like(dataset['OS']),
                np.expand_dims(dataset['DFS'], axis=1),
                np.expand_dims(dataset['OS'], axis=1),
                np.expand_dims(dataset['status'], axis=1),
                np.expand_dims(dataset['recurrence'], axis=1)
            ]
            patients = dataset['patients']

            # retrieve the possible conditions in this dataset
            condition_dicts = {
                '_0A': (dataset['bclc']<2).astype(int) if 'bclc' in dataset.keys() else None,
                '_B': (dataset['bclc']==2).astype(int) if 'bclc' in dataset.keys() else None,
                '_high-risk': (dataset['risk']>0).astype(int) if 'risk' in dataset.keys() else None,
                '_low-risk': (dataset['risk']==0).astype(int) if 'risk' in dataset.keys() else None,
                '_HBV+': (dataset['hbv']==1).astype(int) if 'hbv' in dataset.keys() else None,
                '_HBV-': (dataset['hbv']==0).astype(int) if 'hbv' in dataset.keys() else None,
                '_train': (1-dataset['test']).astype(int) if 'test' in dataset.keys() else None,
                '_valid': (dataset['test']).astype(int) if 'test' in dataset.keys() else None
            }
            conditions = [condition_dicts[key] for key in condition_dicts.keys() if key in group_label]

            group_condition = np.ones_like(dataset['status'], dtype=int)
            for condition in conditions:
                group_condition *= condition
            group_condition = group_condition.astype(bool)

            data_aug_list[group_id].append(data_aug[group_condition])
            patients_list[group_id] += [patient for patient, valid in zip(patients, group_condition) if valid]

    for i in range(len(data_aug_list)):
        if isinstance(data_aug_list[i], list):
            data_aug_list[i] = np.concatenate(data_aug_list[i], axis=0)

    regrouped_datasets = data_aug2dataset(data_aug_list, patients_list, group_labels, datasets[0]['protes'])
    datasets_train, datasets_valid, datasets_test = [], [], []
    for i, dataset in enumerate(regrouped_datasets):
        if dataset['cohort'] in groups['train']:
            datasets_train.append(dataset)
        elif dataset['cohort'] in groups['valid']:
            datasets_valid.append(dataset)
        elif dataset['cohort'] in groups['test']:
            datasets_test.append(dataset)

        print('regrouped dataset {:<2} {:<21} sample_num: {:<3}'.format(i, dataset['cohort'], len(dataset['data'])))
    
    return datasets_train, datasets_valid, datasets_test

def data_aug2dataset(data_aug_list, patients_list, group_labels, protes):
    datasets = []
    for data_aug_id, data_aug in enumerate(data_aug_list):
        data = data_aug[:, :-8]
        bclc =  data_aug[:, -8]
        risk =  data_aug[:, -7]
        hbv =  data_aug[:, -6]
        subtypes = data_aug[:, -5]
        DFS = data_aug[:, -4]
        OS = data_aug[:, -3]
        status = data_aug[:, -2]
        recurrence = data_aug[:, -1]
        patients = patients_list[data_aug_id]
        datasets.append({
                'data': data,
                'patients': patients,
                'protes': protes, 
                'subtypes': subtypes,
                'OS': OS,
                'status': status,
                'DFS': DFS,
                'recurrence': recurrence,
                'risk': risk,
                'bclc': bclc,
                'hbv': hbv,
                'cohort': group_labels[data_aug_id]
            }
        )
    return datasets


def k_fold_split(datasets, n, test_fold, stratify=False):
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
                folds[test] = k
        datasets[i]['folds'] = folds
        is_test = np.zeros_like(folds) 
        is_test[folds==test_fold] = 1
        datasets[i]['test'] = is_test
    return datasets
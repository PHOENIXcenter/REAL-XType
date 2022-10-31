import copy
import numpy as np

def test_choice(data, test_num):
    is_for_test = np.zeros([len(data)], dtype=bool)
    is_for_test[np.random.choice(
        len(data), 
        test_num,
        replace=False
    )] =  True
    return data[is_for_test]

def test_split(datasets, test_rate=0.2):
    for dataset_id, dataset in enumerate(datasets[1:4]):
        # 1. split to 0AL+, 0AL-, 0AS+, 0AS-, B+, B-
        # 2. split to train and test sets
        bclc, t_size, hbv, t_num = dataset['bclc'], dataset['t_size'], dataset['hbv'], dataset['t_num']
        all_ids = np.arange(len(bclc), dtype=int)
        condition_list = [
            ((bclc<2) * (t_size>=10) * (t_num==1) * hbv).astype(bool), 
            ((bclc<2) * (t_size>=10) * (t_num==1) * (1-hbv)).astype(bool), 
            ((bclc<2) * ((t_size<10) + (t_num>1)) * hbv).astype(bool), 
            ((bclc<2) * ((t_size<10) + (t_num>1)) * (1-hbv)).astype(bool), 
            ((bclc==2) * hbv).astype(bool), 
            ((bclc==2) * (1-hbv)).astype(bool), 
        ]
        cum_num = 0
        test_ids = []
        for condition in condition_list:
            cum_num += np.sum(condition)
            test_num = int(cum_num * test_rate) - len(test_ids)
            test_ids += test_choice(all_ids[condition], test_num).tolist()
        test = np.zeros_like(dataset['bclc'])
        test[test_ids] = 1
        datasets[dataset_id+1]['test'] = test
    return datasets 

def test_split_random(datasets, test_rate=0.2):
    for dataset_id, dataset in enumerate(datasets[1:4]):
        all_ids = np.arange(len(dataset['bclc']), dtype=int)
        test_ids = test_choice(all_ids, int(len(all_ids)*0.2))
        test = np.zeros_like(dataset['bclc'])
        test[test_ids] = 1
        datasets[dataset_id+1]['test'] = test
    return datasets

def data_aug2dataset(data_aug_list, patients_list, datasets, data_dim, intersected_protes):
    for data_aug_id, data_aug in enumerate(data_aug_list):
        data = data_aug[:, :data_dim]
        mvi = data_aug[:, -5]
        DFS = data_aug[:, -4]
        OS = data_aug[:, -3]
        status = data_aug[:, -2]
        recurrence = data_aug[:, -1]
        patients = patients_list[data_aug_id]
        protes = intersected_protes
        datasets[data_aug_id + 1] = {
            'data': data,
            'patients': patients,
            'protes': protes, 
            'OS': OS,
            'status': status,
            'DFS': DFS,
            'recurrence': recurrence,
            'mvi': mvi
        }
    return datasets

def datasets_regroup(datasets, intersected_protes, selection_code):
    if selection_code == '7 5':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A_train', 'Center2_0A_train', 'Center3_0A_train',
            'Center123_B_train',
            'Center123_large_train',
            'Center123_HBV-_train',
            'Center123_0A_test', 
            'Center123_B_test',
            'Center123_large',
            'Center123_HBV-',
            'Cell'
        ]
    elif selection_code == '12 2':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A_train', 'Center2_0A_train', 'Center3_0A_train',
            'Center1_B_train', 'Center2_B_train', 'Center3_B_train',
            'Center1_large_train', 'Center2_large_train', 'Center3_large_train',
            'Center123_B_train',
            'Center123_large_train',

            'Center123_test', 
            'Cell_0AB'
        ]
    elif selection_code == '12 3':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A_train', 'Center2_0A_train', 'Center3_0A_train',
            'Center1_B_train', 'Center2_B_train', 'Center3_B_train',
            'Center1_large_train', 'Center2_large_train', 'Center3_large_train',
            'Center123_B_train',
            'Center123_large_train',

            'Center123_0A_test', 
            'Center123_B_test',
            'Cell'
        ]
    elif selection_code == '15 1':
        selected_dataset_labels = [
            'Nature', 
            'Center1_low-risk', 'Center2_low-risk', 'Center3_low-risk',
            'Center1_0A_train', 'Center2_0A_train', 'Center3_0A_train',
            'Center1_B_train', 'Center2_B_train', 'Center3_B_train',
            'Center1_large_train', 'Center2_large_train', 'Center3_large_train',
            'Center123_B_train',
            'Center123_large_train',

            'Center123_test'
        ]
    elif selection_code == '16 3':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A_train', 'Center2_0A_train', 'Center3_0A_train',
            'Center1_B_train', 'Center2_B_train', 'Center3_B_train',
            'Center1_large_train', 'Center2_large_train', 'Center3_large_train',
            'Center1_HBV-_train', 'Center2_HBV-_train', 'Center3_HBV-_train',
            'Center123_B_train',
            'Center123_large_train',
            'Center123_HBV-_train',

            'Center123_0A_test', 
            'Center123_B_test',
            'Cell'
        ]
    elif selection_code == '6 2':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A_train', 'Center2_0A_train', 'Center3_0A_train',
            'Center123_B_train',
            'Center123_large_train',

            'Center123_test', 
            'Cell_0AB'
        ]
    elif selection_code == '6 2 test':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A', 'Center2_0A', 'Center3_0A',
            'Center123_B',
            'Center123_large',

            'Center123_test', 
            'Cell_0AB',
        ]
    elif selection_code == '9 2 test':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A', 'Center2_0A', 'Center3_0A',
            'Center1_low-risk', 'Center2_low-risk', 'Center3_low-risk',
            'Center123_B',
            'Center123_large',

            'Center123_test', 
            'Cell',
        ]
    elif selection_code == '6 3':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A', 'Center2_0A', 'Center3_0A',
            'Center123_B',
            'Center123_large',

            'Center123_0A_test', 
            'Center123_B_test',
            'Cell'
        ]
    elif selection_code == '6 0':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A', 'Center2_0A', 'Center3_0A',
            'Center123_B',
            'Cell_0AB'
        ]
    elif selection_code == '4 2':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A_train', 'Center2_0A_train', 'Center3_0A_train',
            'Center123_0A_test',
            'Cell_0A'
        ]
    elif selection_code == '4 2 test':
        selected_dataset_labels = [
            'Nature', 
            'Center1_0A', 'Center2_0A', 'Center3_0A',
            'Center123_0A_test',
            'Cell_0A'
        ]
    elif selection_code == '2 3 0A':
        selected_dataset_labels = [
            'Nature', 'Center2_0A',
            'Center1_0A', 'Center3_0A', 'Cell_0A'
        ]
    elif selection_code == '2 3':
        selected_dataset_labels = [
            'Nature', 'Center2',
            'Center1', 'Center3', 'Cell_0A'
        ]
    elif selection_code == '1 2':
        selected_dataset_labels = [
            'Center1_low-risk', 'Center2_low-risk', 'Center3_low-risk'
        ]
    else:
        print('unknown code!')
        assert False

    dataset_labels = [
        'Nature', 

        'Center1', 'Center2', 'Center3',

        'Center1_low-risk', 'Center2_low-risk', 'Center3_low-risk',

        'Center1_0A', 'Center2_0A', 'Center3_0A',
        'Center1_0A_train', 'Center2_0A_train', 'Center3_0A_train',
        'Center1_0A_test', 'Center2_0A_test', 'Center3_0A_test',

        'Center1_B', 'Center2_B', 'Center3_B',
        'Center1_B_train', 'Center2_B_train', 'Center3_B_train',

        'Center1_large', 'Center2_large', 'Center3_large',
        'Center1_large_train', 'Center2_large_train', 'Center3_large_train',

        'Center1_HBV-', 'Center2_HBV-', 'Center3_HBV-',
        'Center1_HBV-_train', 'Center2_HBV-_train', 'Center3_HBV-_train',

        'Center123_0A_test', 

        'Center123_B',
        'Center123_B_train',
        'Center123_B_test',

        'Center123_large',
        'Center123_large_train',
        'Center123_large_test',

        'Center123_HBV-',
        'Center123_HBV-_train',
        'Center123_HBV-_test',

        'Center123_test',

        'Cell_0A',
        'Cell_0AB',
        'Cell'
    ]

    print('data regroup')
    data_aug_list  = [[] for _ in range(len(dataset_labels)-2)]
    patients_list  = [[] for _ in range(len(dataset_labels)-2)]
    data_dim = len(intersected_protes)
    for i, dataset in enumerate(datasets[1:4]):
        bclc = dataset['bclc']
        t_size = dataset['t_size']
        hbv = dataset['hbv']
        hcv = dataset['hcv']
        t_num = dataset['t_num']
        mvi = dataset['mvi']
        risk = dataset['risk']
        test = dataset['test']

        sub_condition_dict = {
            '_0A': (bclc<2).astype(int),
            '_B': (bclc==2).astype(int),
            '_large': ((t_size>=10)*(t_num==1)).astype(int),
            '_HBV-': (1-hbv).astype(int),
            '_low-risk': (risk==0).astype(int),
            '_train': (1-test).astype(int),
            '_test': (test).astype(int)
        }

        data_aug = np.c_[
            dataset['data'],
            np.expand_dims(dataset['mvi'], axis=1),
            np.expand_dims(dataset['DFS'], axis=1),
            np.expand_dims(dataset['OS'], axis=1),
            np.expand_dims(dataset['status'], axis=1),
            np.expand_dims(dataset['recurrence'], axis=1)
        ]
        patients = dataset['patients']
        for j, label in enumerate(dataset_labels[1:-2]):
            condition = np.ones_like(bclc, dtype=int)
            for sub_condition_key in sub_condition_dict.keys():
                if sub_condition_key in label:
                    condition *= sub_condition_dict[sub_condition_key]
            condition = condition.astype(bool)
            
            if '123' not in label:
                if j%3 == i:
                    data_aug_list[j] = data_aug[condition]
                    patients_list[j] = [patient for patient, valid in zip(patients, condition) if valid]
            else:
                data_aug_list[j].append(data_aug[condition])
                patients_list[j] += [patient for patient, valid in zip(patients, condition) if valid]
    # Cell_0A
    dataset = datasets[-1]
    data_aug = np.c_[
        dataset['data'],
        np.expand_dims(dataset['DFS'], axis=1),
        np.expand_dims(dataset['OS'], axis=1),
        np.expand_dims(dataset['status'], axis=1),
        np.expand_dims(dataset['recurrence'], axis=1)
    ]
    condition = dataset['bclc'] < 2
    patients = dataset['patients']
    data_aug_list[-2] = data_aug[condition]
    patients_list[-2] = [patient for patient, valid in zip(patients, condition) if valid]

    # Cell_0AB
    dataset = datasets[-1]
    data_aug = np.c_[
        dataset['data'],
        np.expand_dims(dataset['DFS'], axis=1),
        np.expand_dims(dataset['OS'], axis=1),
        np.expand_dims(dataset['status'], axis=1),
        np.expand_dims(dataset['recurrence'], axis=1)
    ]
    condition = dataset['bclc'] < 3
    patients = dataset['patients']
    data_aug_list[-1] = data_aug[condition]
    patients_list[-1] = [patient for patient, valid in zip(patients, condition) if valid]

    for i in range(len(data_aug_list)):
        if isinstance(data_aug_list[i], list):
            data_aug_list[i] = np.concatenate(data_aug_list[i], axis=0)
        # print(dataset_labels[i+1], len(data_aug_list[i]))

    datasets += [None for _ in range(len(data_aug_list) - 4)]
    datasets += [datasets[4]]
    datasets = data_aug2dataset(data_aug_list, patients_list, datasets, data_dim, intersected_protes)

    selected_datasets = []
    for i, label in enumerate(selected_dataset_labels):
        dataset_id = dataset_labels.index(label)
        dataset = datasets[dataset_id]
        dataset['label'] = label
        selected_datasets.append(dataset)
        print('regrouped dataset {:<2} {:<21} sample_num: {:<3}'.format(i, dataset['label'], len(dataset['data'])))
    
    return selected_datasets

def datasets_mix(datasets, label='Mixed'):
    mixed_dataset = {'label': label}
    for dataset in datasets:
        for key in dataset.keys():
            if isinstance(dataset[key], list) or isinstance(dataset[key], np.ndarray):
                if key in ['raw_protes', 'feat_mean', 'feat_std', 'protes']:
                    continue
                if key not in mixed_dataset.keys():
                    mixed_dataset[key] = dataset[key]
                elif isinstance(dataset[key], list):
                    mixed_dataset[key] += dataset[key]
                elif isinstance(dataset[key], np.ndarray):
                    mixed_dataset[key] = np.concatenate([mixed_dataset[key], dataset[key]], axis=0)
    mixed_dataset['protes'] = datasets[0]['protes']
    return mixed_dataset

def get_n_folds(n_sample, n_fold):
    folds = np.zeros((n_sample), dtype=int)
    ids = np.arange(n_sample)
    for k in range(n_fold-1):
        valid_ids = ids[folds==0]
        selected_ids = np.random.choice(valid_ids, int(len(valid_ids) / (n_fold - k)), replace=False)
        folds[selected_ids] = k + 1
    return folds

def dataset_divide(dataset, labels):
    datasets = []
    folds = get_n_folds(len(dataset['patients']), len(labels))
    for fold, label in enumerate(labels):
        new_dataset = {'label': label}
        for key in dataset.keys():
            if isinstance(dataset[key], list) or isinstance(dataset[key], np.ndarray):
                if key in ['protes']:
                    new_dataset[key] = dataset[key]
                elif isinstance(dataset[key], list):
                    new_dataset[key] = [item for item, is_true in zip(dataset[key], folds==fold) if is_true]
                elif isinstance(dataset[key], np.ndarray):
                    new_dataset[key] = dataset[key][folds==fold]
        datasets.append(new_dataset)
    return datasets

def n_fold_split(datasets, n, test_fold):
    for i, dataset in enumerate(datasets):
        folds = get_n_folds(len(dataset['patients']), n)
        datasets[i]['folds'] = folds

        valid_fold = test_fold + 1 if test_fold < n - 1 else 0
        groups = np.zeros_like(folds) 
        groups[folds==valid_fold] = 1 # validation
        # groups[folds==test_fold] = 2  # testing
        datasets[i]['groups'] = groups
    return datasets

def dataset_group_filtering(dataset, group, label):
    new_dataset = {'label': label}
    groups = dataset['groups']
    for key in dataset.keys():
        if isinstance(dataset[key], list) or isinstance(dataset[key], np.ndarray):
            if key in ['raw_protes', 'feat_mean', 'feat_std', 'protes']:
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
    group_label = ['train', 'test']
    datasets_groups = [[], []]
    for dataset in datasets:
        for group in range(2):
            if np.sum(dataset['groups']==group) > 0:
                datasets_groups[group].append(
                    dataset_group_filtering(
                        dataset, 
                        group, 
                        dataset['label'] + '_' + group_label[group]
                    )
                )
    return datasets_groups
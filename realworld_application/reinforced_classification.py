import os, copy, time, argparse, gc, platform, re, pickle, progressbar
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from distutils.dir_util import copy_tree

import data_utils as utils
from network_model.reinforced_model import *
from viz_utils import *
from data_regroups import *
from test import compare_prediction_difference

sys = platform.system()
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--train_num', type=int, default=15)
parser.add_argument('--test_num', type=int, default=1)
parser.add_argument('--h_dim', type=int, default=3)
parser.add_argument('--subtype_num', type=int, default=3)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--use_bias', type=bool, default=False)
parser.add_argument('--simple', type=bool, default=True)
parser.add_argument('--group', type=bool, default=False)

parser.add_argument('--epoch_num', type=int, default=7000)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--baseline_learning_rate', type=float, default=0.001)
parser.add_argument('--loss_su_rate', type=float, default=0.04)
parser.add_argument('--loss_rl_rate', type=float, default=1.)
parser.add_argument('--loss_mvi_rate', type=float, default=0.03)
parser.add_argument('--model_name', default='0505_gao_test_only')
parser.add_argument('--logfc_thresh', type=float, default=1.0)
parser.add_argument('--rank_thresh', type=int, default=9000)
parser.add_argument('--blood_flag', type=bool, default=False)
parser.add_argument('--drug_flag', type=bool, default=False)
parser.add_argument('--candidate_31_flag', type=bool, default=False)
parser.add_argument('--consistency', type=bool, default=False)
parser.add_argument('--nan_thresh', type=float, default=0.7)
parser.add_argument('--training', type=bool, default=False)
parser.add_argument('--load_feat_prote', default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--regularization', type=float, default=1e-4)
parser.add_argument('--scatter_viz', type=bool, default=False)
parser.add_argument('--weight_viz', type=bool, default=False)
parser.add_argument('--log10', type=bool, default=False)
parser.add_argument('--dropout1', type=float, default=0.8)
parser.add_argument('--dropout2', type=float, default=0.3)
parser.add_argument('--check', type=bool, default=False)
parser.add_argument('--check_thresh', type=int, default=0)
parser.add_argument('--check_path', default=None)
parser.add_argument('--save_pickle', type=bool, default=False)
parser.add_argument('--load_pseudo_label', type=bool, default=False)
parser.add_argument('--loss_su_w', type=float, default=0.5)
parser.add_argument('--os_w', type=float, default=1.0)
parser.add_argument('--dfs_w', type=float, default=1.0)
parser.add_argument('--km_viz', type=bool, default=False)

if sys == 'Linux':
    parser.add_argument('--data_path', default='/data/linhai/Tumor_clinic')
else:
    disk = os.getcwd().split(':')[0]
    parser.add_argument('--data_path', default=disk + r':\Data\Tumor_clinic')

args = parser.parse_args()
print(args)
POSTFIX = ''
if args.load_feat_prote:
    if 'concrete' in args.load_feat_prote:
        feat_select_nums = re.findall(r'\d+', args.load_feat_prote)
        feat_select_id = feat_select_nums[2]
        feat_select_code = 'concrete_d{:}_f{:}_r{:}'.format(
            feat_select_nums[0], 
            feat_select_nums[1], 
            feat_select_nums[2]
        )
    elif 'model_selected' in args.load_feat_prote or 'sun_selected' in args.load_feat_prote:
        feat_select_code = args.load_feat_prote.split(os.sep)[-1].split('.')[0]
    else:
        print('Invalid feature selection code')
        assert False
    feat_select_code = feat_select_code + '_blood' if 'blood' in args.load_feat_prote else feat_select_code
    feat_select_code = feat_select_code + '_supervised' if 'supervised' in args.load_feat_prote else feat_select_code 
else:
    feat_select_code = 'all_feat'
print('load data', feat_select_code)

global_mask, global_pathways = None, None

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def training(all_datasets, test_datasets, gene2prote_dict, prote2gene_dict):
    global POSTFIX
    save_path = join(
        args.data_path, 
        'reinforced_classification', 
        args.model_name,
        feat_select_code
    ) if args.load_feat_prote else join(
        args.data_path, 
        'reinforced_classification', 
        args.model_name
    )
    create_path(save_path)
    result_path = os.path.join(save_path, 'results' + POSTFIX)
    create_path(result_path)

    datasets = all_datasets[:args.train_num]
    valid_datasets = all_datasets[args.train_num:]
    if args.simple:
        models, optimizers = make_simple_models(
            args.h_dim,
            args.subtype_num, 
            [args.learning_rate, args.baseline_learning_rate], 
            args.regularization
        )
    elif args.group:
        mask, pathways = utils.pathway_mask(args.data_path, datasets[0]['protes'], gene2prote_dict, prote2gene_dict)
        global global_mask, global_pathways
        global_mask, global_pathways = mask, pathways
        models, optimizers = make_grouped_models(
            args.h_dim,
            args.subtype_num, 
            mask.shape[1],
            mask,
            [args.learning_rate, args.baseline_learning_rate], 
            args.regularization
        )
    else:
        models, optimizers = make_models(
            args.h_dim, 
            args.subtype_num, 
            [args.learning_rate, args.baseline_learning_rate],
            args.regularization,
            args.use_bias
        )
    classifier, baseline = models
    baseline(tf.zeros((1, datasets[0]['data'].shape[1])), tf.zeros(1))


    if args.training:
        summary_writer = tf.summary.create_file_writer(result_path)
        start_time = time.time()
        best_r = 0.5
        valid_best_r = 0.3
        valid_mean_r = 0.
        dropouts = [args.dropout1, args.dropout2]

        for epoch in range(args.epoch_num):
            # train
            rmst_time = []
            x1_list, y1_list, x2_list, a_list, r_list, r_argmax_list, w_su_list, mvi_list = [], [], [], [], [], [], [], []
            s1_mvi_list = []
            for domain, dataset in enumerate(datasets):
                x = dataset['data'].astype(np.float32) 
                OS = dataset['OS']
                status = dataset['status']
                DFS = dataset['DFS']
                recurrence = dataset['recurrence']
                mvi = dataset['mvi'] if 'mvi' in dataset.keys() else 0
                if args.os_w + args.dfs_w == 0:
                    OS_DFS_w = [1., 0.] if '_0A' in dataset['label'] else [0., 1.]
                elif args.os_w + args.dfs_w == -1:
                    OS_DFS_w = [0., 1.] if '_B' in dataset['label'] else [1., 0.]
                else:
                    OS_DFS_w = [args.os_w, args.dfs_w]
                if 'subtypes' in dataset.keys():
                    x1_list.append(x)
                    y1_list.append(dataset['subtypes'].astype(np.int32))
                    w = 1. if domain == 0 else args.loss_su_w
                    w_su_list.append(np.ones_like(dataset['subtypes']) * w)
                if domain > 0:
                    a_sample, a_argmax = classifier.sample(x)
                    rmst_start_time = time.time()
                    r = (utils.get_reward(OS, status, a_sample)*OS_DFS_w[0] + 
                        utils.get_reward(DFS, recurrence, a_sample)*OS_DFS_w[1]) / np.sum(OS_DFS_w)
                    rmst_time.append(time.time() - rmst_start_time)
                    x2_list.append(x)
                    a_list.append(a_sample)
                    r_list.append(r)
                    r_argmax = (utils.get_reward(OS, status, a_argmax)*OS_DFS_w[0] + 
                        utils.get_reward(DFS, recurrence, a_argmax)*OS_DFS_w[1]) / np.sum(OS_DFS_w)
                    r_argmax_list.append(r_argmax[0])
                    mvi_list.append(mvi)
                    s1_mvi_list.append(np.sum((a_argmax==0) * mvi) / np.sum(a_argmax==0))
            x1 = np.concatenate(x1_list, axis=0)
            y1 = np.concatenate(y1_list, axis=0)
            x2 = np.concatenate(x2_list, axis=0)
            a = np.concatenate(a_list, axis=0).astype(np.int32)
            reward = np.concatenate(r_list, axis=0).astype(np.float32)
            w_su = np.concatenate(w_su_list, axis=0).astype(np.float32)
            mvi_list = np.concatenate(mvi_list, axis=0).astype(np.float32)

            # evaluating model
            mean_r = np.mean(r_argmax_list)
            if mean_r > best_r and epoch > 1000:
                best_r = copy.deepcopy(mean_r)

            if len(valid_datasets) > 0:
                valid_r_list, valid_mvi_list = [], []
                valid_s1_mvi_list = []
                for domain, dataset in enumerate(valid_datasets):
                    x = dataset['data'].astype(np.float32)
                    OS = dataset['OS']
                    status = dataset['status']
                    DFS = dataset['DFS']
                    recurrence = dataset['recurrence']
                    mvi = dataset['mvi'] if 'mvi' in dataset.keys() else 0
                    a_valid = classifier.predict(x)
                    r_valid = (utils.get_reward(OS, status, a_valid)*OS_DFS_w[0] + 
                        utils.get_reward(DFS, recurrence, a_valid)*OS_DFS_w[1]) / np.sum(OS_DFS_w)
                    valid_r_list.append(r_valid[0])
                    valid_s1_mvi_list.append(np.sum((a_valid==0) * mvi) / np.sum(a_valid==0))
                valid_mean_r = np.mean(valid_r_list)
                if valid_mean_r > valid_best_r and epoch > 1000:
                    classifier.save_weights(join(result_path, 'ckpt'))
                    valid_best_r = copy.deepcopy(valid_mean_r)
                    print('save validation model')

            # trainig model
            training_start_time = time.time()
            loss_su, loss_rl, loss_bl, loss_mvi, loss_l1, curr_logits_list = train_step_rl(
                x1, y1, x2, a, reward, mvi_list,
                models, 
                optimizers, 
                [args.loss_su_rate, args.loss_rl_rate, args.loss_mvi_rate],
                dropouts,
                tf.convert_to_tensor(w_su),
                args.simple,
                args.group
            )

            acc = np.mean(np.argmax(curr_logits_list[0], axis=1) == y1)
            
            # visualizing results
            print('epoch:{:}|'.format(epoch) +
                'T:{:.1f}|'.format((time.time() - start_time)/60.) +
                'l_su:{:.2f}|'.format(np.mean(loss_su)) +
                'l_rl:{:.2f}|'.format(np.fabs(np.mean(loss_rl))) +
                'l_bl:{:.2f}|'.format(np.mean(loss_bl)) +
                'l_mvi:{:.2f}|'.format(loss_mvi) +
                'l_l1:{:.3f}|'.format(loss_l1) +
                'acc:{:.2f}|'.format(acc) +
                'r:{:.2f}|'.format(mean_r) +
                'best_r:{:.2f}|'.format(best_r) +
                's1_mvi:{:.2f}|'.format(np.mean(s1_mvi_list)) +  
                'v_r:{:.2f}|'.format(valid_mean_r) +
                'v_best_r:{:.2f}|'.format(valid_best_r) +
                'v_s1_mvi:{:.2f}'.format(np.mean(valid_s1_mvi_list))
            )

            with summary_writer.as_default():
                tf.summary.scalar('acc', acc, step=epoch)
                tf.summary.scalar('mean_r', mean_r, step=epoch)
                tf.summary.scalar('best_r', best_r, step=epoch)
                tf.summary.scalar('valid_mean_r', valid_mean_r, step=epoch)
                tf.summary.scalar('valid_best_r', valid_best_r, step=epoch)

    start_time = time.time()

    classifier.load_weights(join(result_path, 'ckpt'))
    testing(classifier, test_datasets, result_path)


def testing(classifier, datasets, result_path):
    r_list = []
    for i, dataset in enumerate(datasets):
        # print('test on dataset ' + dataset['label'])
        assignments = classifier.predict(dataset['data'].astype(np.float32))
        prob = classifier.prob(dataset['data'].astype(np.float32))
        encode = classifier.encode(dataset['data'].astype(np.float32))
        OS = dataset['OS']
        status = dataset['status']
        DFS = dataset['DFS']
        recurrence = dataset['recurrence']
        protes = dataset['protes']
        if 'subtypes' in dataset.keys():
            acc = np.mean(assignments==dataset['subtypes'])
            title = 'Acc {:.3f} | '.format(acc) + dataset['label']
        else:
            title = dataset['label']
            r = (utils.get_reward(OS, status, assignments) + utils.get_reward(DFS, recurrence, assignments))/2
            r_list.append(r[0])
        if args.km_viz:
            plot_survival_cruve(
                OS, 
                status, 
                assignments, 
                3, 
                title=title + ' OS ', 
                path=result_path, 
                prefix='test_' + dataset['label'] + '_OS_',
                pop_viz=False
            )
            plot_survival_cruve(
                DFS, 
                recurrence, 
                assignments, 
                3, 
                title=title + ' DFS ', 
                path=result_path, 
                prefix='test_' + dataset['label'] + '_DFS_',
                pop_viz=False
            )
        patients = dataset['patients']
        utils.write_assignments(
            join(
                result_path, 
                'test_' + dataset['label'] + '_classification_result.csv'
            ),
            patients, 
            assignments
        )
        if args.scatter_viz:
            scatter_viz(
                prob, assignments, prob, OS, status, dataset['label'], 
                join(result_path, 'test_' + dataset['label'] + '_classification_result.png')
            )
    if args.load_feat_prote is not None:
        utils.write_protes(protes, join(result_path, 'selected_protes.txt'))

    if args.weight_viz:
        w1 = classifier.masked_linear.get_masked_weight() if args.simple else classifier.dense1.get_weights()[0]
        weight_viz(w1, join(result_path, 'w1.png'))
        feat_importance = np.amax(np.fabs(w1), axis=1)
        sorted_ids = np.argsort(-feat_importance)
        sorted_protes = [datasets[0]['protes'][sorted_id] for sorted_id in sorted_ids]
        utils.write_protes(sorted_protes, join(result_path, 'sorted_protes.txt'))

    if args.group:
        w1 = classifier.masked_linear.get_masked_weight()
        w2 = classifier.dense.get_weights()[0]
        pathway_viz(w1, w2, global_mask, global_pathways, result_path)
        weight_viz(w2, join(result_path, 'w2.png'))
        utils.write_protes(global_pathways, join(result_path, 'pathways.txt'))

def single_result_check(survival_info, test_path, km_viz=False):
    score = 0
    for label in ['Center1_0A', 'Center2_0A', 'Center3_0A','Center123_B','Center123_large',
        'Center123_test','Cell_0AB', 'Center1_low-risk', 'Center2_low-risk', 'Center3_low-risk',
    ]:
        patients, assignments = utils.read_assignments(
            join(test_path, 'test_' + label + '_classification_result.csv')
        )
        if np.amin([np.sum(assignments==subtype) for subtype in range(3)]) < 3:
            # print('invalid assignments', [np.sum(assignments==subtype) for subtype in range(3)])
            score -= 1
            continue

        ids = [survival_info['patients'].index(patient) for patient in patients]
        times, events = survival_info['OS'][ids], survival_info['status'][ids]
        delta_rmst_list = utils.get_delta_rmst(times, events, assignments, 3, return_RMST=True)
        rmst_list = delta_rmst_list[-3:]

        p_list = utils.get_logrank_p_custom(times, events, assignments, 3)
        p = utils.get_mutivarate_logrank_p_custom(times, events, assignments, 3)
        if label in ['Cell_0AB', 'Center123_test']:
            score = score - 2 if max(p_list[0], p_list[2]) >= 0.3 or p >= 0.05 else score
        elif label in ['Center2_0A', 'Center3_0A']:
            score = score - 1 if max(p_list[0], p_list[2]) >= 0.2 else score
        elif label in ['Center1_0A', 'Center123_B']:
            score = score - 1 if max(p_list[0], p_list[2]) >= 0.05 else score
        elif label in ['Center123_large']:
            score = score - 2 if rmst_list[0] < 50 or max(p_list[0], p_list[2]) >= 0.05 else score

        if km_viz:
            plot_survival_cruve(
                times, 
                events, 
                assignments, 
                3, 
                title=label + ' OS ', 
                path=test_path, 
                prefix='test_' + label + '_OS_',
                pop_viz=False
            )

        times, events = survival_info['DFS'][ids], survival_info['recurrence'][ids]
        delta_rmst_list = utils.get_delta_rmst(times, events, assignments, 3, return_RMST=True)
        rmst_list = delta_rmst_list[-3:]
        p_list = utils.get_logrank_p_custom(times, events, assignments, 3)
        p = utils.get_mutivarate_logrank_p_custom(times, events, assignments, 3)
        if label in ['Cell_0AB', 'Center123_test']:
            score = score - 2 if max(p_list[0], p_list[2]) >= 0.3 or p >= 0.05 else score
        elif label in ['Center2_0A', 'Center3_0A']:
            score = score - 1 if max(p_list[0], p_list[2]) >= 0.2 else score
        elif label in ['Center1_0A', 'Center123_B']:
            score = score - 1 if max(p_list[0], p_list[2]) >= 0.05 else score
        elif label in ['Center123_large']:
            score = score - 2 if delta_rmst_list[0] < 10 or p_list[0] >= 0.05 else score

        if km_viz:
            plot_survival_cruve(
                times, 
                events, 
                assignments, 
                3, 
                title=label + ' DFS ', 
                path=test_path, 
                prefix='test_' + label + '_DFS_',
                pop_viz=False
            )

    return score

def checking_offline(raw_datasets, check_path, thresh=0):
    start_time = time.time()
    results_num = 0
    save_folder = 'top_candidates'
    root_list = []
    for root, dirs, files in os.walk(check_path):
        if save_folder in root:
            continue
        if 'test_Nature_classification_result.csv' in files:
            root_list.append(root)

    # get the survival info
    patients_list = []
    for dataset in raw_datasets[1:]:
        patients_list += dataset['patients']
    OS_list = np.concatenate([dataset['OS'] for dataset in raw_datasets[1:]])
    status_list = np.concatenate([dataset['status'] for dataset in raw_datasets[1:]])
    DFS_list = np.concatenate([dataset['DFS'] for dataset in raw_datasets[1:]])
    recurrence_list = np.concatenate([dataset['recurrence'] for dataset in raw_datasets[1:]])
    survival_info_C123Cell = {
        'patients': patients_list,
        'OS': OS_list,
        'status': status_list,
        'DFS': DFS_list,
        'recurrence': recurrence_list
    }

    score_list = []
    for root in progressbar.progressbar(root_list):
        results_num += 1
        score = single_result_check(survival_info_C123Cell, root)
        score_list.append(score)
        if score >= thresh:
            print('high score:', root)
            root_folders = root.split(os.path.sep)
            copy_path = join(
                check_path, save_folder, root_folders[-3], root_folders[-2], root_folders[-1]
            ) if root_folders[-1] == 'validation' else join(
                check_path, save_folder, root_folders[-2], root_folders[-1]
            )
            if not os.path.exists(copy_path):
                os.makedirs(copy_path)
            copy_tree(root, copy_path)
            single_result_check(survival_info_C123Cell, copy_path, True)

    print('checked {:} results, {:.1f} sec used'.format(results_num, time.time() - start_time))

    sorted_ids = np.argsort(-np.asarray(score_list))[:10]
    for rank, sorted_id in enumerate(sorted_ids):
        print('rank {:}, score {:}, '.format(rank, score_list[sorted_id]) + root_list[sorted_id])



if __name__ == '__main__':
    # test_on_BCLC_C()
    # assert False

    start_time = time.time()
    np.random.seed(395) # 395 166

    pickle_folder_path = join(args.data_path, 'temp_files')
    if not os.path.exists(pickle_folder_path):
        os.makedirs(pickle_folder_path)
    pickle_file_path = join(pickle_folder_path, feat_select_code + '.p')

    if not os.path.isfile(pickle_file_path):
        gene2prote_dict, prote2gene_dict = utils.gene2prote_and_prote2gene(
            join(args.data_path, 'HCC-Proteomics-Annotation-20210226.xlsx')
        )
        raw_datasets, intersected_protes = utils.load_data(
            args, 
            gene2prote_dict, 
            prote2gene_dict,
            load_feat_prote=args.load_feat_prote, 
            log10=args.log10, 
            zscore=True, 
            f1269=True, 
            clip=60
        )
        pickle.dump([raw_datasets, intersected_protes, gene2prote_dict, prote2gene_dict], open(pickle_file_path, 'wb'))
    else:
        raw_datasets, intersected_protes, gene2prote_dict, prote2gene_dict = pickle.load(open(pickle_file_path, 'rb'))

    splited_datasets = test_split(raw_datasets)
    all_datasets = datasets_regroup(copy.deepcopy(splited_datasets), intersected_protes, '{:} {:}'.format(args.train_num, args.test_num))
    test_datasets = datasets_regroup(copy.deepcopy(splited_datasets), intersected_protes, '9 2 test')

    print('feat num:', len(intersected_protes))

    if not args.check:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        POSTFIX = str(args.seed)
        with tf.device('/cpu:0'):
            training(all_datasets, test_datasets, gene2prote_dict, prote2gene_dict)
            tf.keras.backend.clear_session()
    else:
        checking_offline(
            raw_datasets,
            args.check_path, 
            args.check_thresh
        )
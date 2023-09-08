import os, copy, time, argparse, platform, pickle

parser = argparse.ArgumentParser(description='Process some parameters.')
# model param
parser.add_argument('--subtype_num', type=int, default=3)
parser.add_argument('--hdim', type=int, default=3)
parser.add_argument('--encoder_hdim', type=int, default=20)
parser.add_argument('--class_hdim', type=int, default=10)
parser.add_argument('--use_bias', type=bool, default=False)
parser.add_argument('--encoder_layer_num', type=int, default=1)
parser.add_argument('--class_layer_num', type=int, default=1)
parser.add_argument('--epoch_num', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--bl_lr', type=float, default=0.001)
parser.add_argument('--loss_su', type=float, default=0.04)
parser.add_argument('--loss_rl', type=float, default=1)
parser.add_argument('--loss_survival', type=float, default=0.05)
parser.add_argument('--regu', type=float, default=1e-4)
parser.add_argument('--activation', default=None)
parser.add_argument('--dropout', type=float, default=0.8)
parser.add_argument('--dropout_rl', type=float, default=0.0)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--initial_valid_best_r', type=float, default=-100)
parser.add_argument('--discrete', type=bool, default=False)
parser.add_argument('--max_depth', type=int, default=None)
parser.add_argument('--optimiser', type=str, default='ADAM')
parser.add_argument('--no_early_stop', type=bool, default=False)

parser.add_argument('--theta', type=float, default=2)
parser.add_argument('--lamb', type=float, default=1.)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--n_components', type=int, default=100)

# preprocess param
parser.add_argument('--zscore', type=bool, default=True)
parser.add_argument('--imputation', type=bool, default=False)
parser.add_argument('--log2', type=bool, default=False)
parser.add_argument('--f1269', type=bool, default=True)
parser.add_argument('--nan_thresh', type=float, default=0.7)
parser.add_argument('--max_t', type=int, default=60)
parser.add_argument('--supervised_rate', type=float, default=0.5)

parser.add_argument('--mode', default='SRPS')
parser.add_argument('--data', default='HCC')
parser.add_argument('--experiment', default='test')
parser.add_argument('--fold_num', type=int, default=5)
parser.add_argument('--test_fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--noise_rate', type=float, default=0.)
parser.add_argument('--plot', type=bool, default=False)
parser.add_argument('--batch_effect', default='lv3')
parser.add_argument('--mini_batch', type=bool, default=False)
parser.add_argument('--cell_num', type=int, default=1000)
parser.add_argument('--para_id', type=int, default=0)

# toy dataset data generation
parser.add_argument('--os_swap_i', type=int, default=0)
parser.add_argument('--batch_noise_j', type=int, default=0)
parser.add_argument('--os_swap_rate_step', type=float, default=0.02)
parser.add_argument('--batch_noise_rate_step', type=float, default=0.4)

# system settings
parser.add_argument('--data_path', default='data')

platform_sys = platform.system()
if platform_sys == 'Linux':
    parser.add_argument('--device', default='/gpu:0')
else:
    parser.add_argument('--device', default='/cpu:0')

args = parser.parse_args()

from para_space import get_paras

args, selected_paras, result_path = get_paras(args)
if selected_paras is None:
    quit()

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from evaluating import *
from survival_correlated_data import generate_survival_correlated_synthetic_data

import viz_utils as viz_utils
import data_utils as data_utils
import models.SRPSsoft as SRPSsoft
import models.SRPS as SRPS
import models.DANN as DANN

def training_toy_data(datasets_groups):
    datasets_train, datasets_valid, datasets_test = datasets_groups

    for i in range(len(datasets_train)):
        datasets_train[i]['data'] = np.concatenate([datasets_train[i]['data'], datasets_valid[i]['data'], datasets_test[i]['data']])
        datasets_train[i]['patients'] = datasets_train[i]['patients'] + datasets_valid[i]['patients'] + datasets_test[i]['patients']
        datasets_train[i]['subtypes'] = np.concatenate([datasets_train[i]['subtypes'], datasets_valid[i]['subtypes'], datasets_test[i]['subtypes']])
        datasets_train[i]['OS'] = np.concatenate([datasets_train[i]['OS'], datasets_valid[i]['OS'], datasets_test[i]['OS']])
        datasets_train[i]['status'] = np.concatenate([datasets_train[i]['status'], datasets_valid[i]['status'], datasets_test[i]['status']])
        datasets_train[i]['DFS'] = np.concatenate([datasets_train[i]['DFS'], datasets_valid[i]['DFS'], datasets_test[i]['DFS']])
        datasets_train[i]['recurrence'] = np.concatenate([datasets_train[i]['recurrence'], datasets_valid[i]['recurrence'], datasets_test[i]['recurrence']])
        datasets_train[i]['groups'] = np.concatenate([datasets_train[i]['groups'], datasets_valid[i]['groups'], datasets_test[i]['groups']])

    result_path = join(args.data_path, args.experiment, 'seed' + str(args.seed))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    acc_save_path = join(result_path, 'acc_save')
    if not os.path.exists(acc_save_path):
        os.makedirs(acc_save_path)

    print('Trarining', args.mode)

    models, optimizers = SRPS.make_models(
        [(args.hdim, True, 'relu', args.regu), (args.hdim, True, 'relu', args.regu), (args.subtype_num, True, None, args.regu)],
        [(args.hdim, True, 'sigmoid'), (args.hdim, True, 'sigmoid'), (1, True, None)],
        [args.learning_rate, args.bl_lr], 
    )
    classifier, baseline = models
    baseline(tf.zeros((1, datasets_train[0]['data'].shape[1])), tf.zeros(1))

    start_time = time.time()
    acc_list = []
    for epoch in range(args.epoch_num):
        x1_list, y1_list, x2_list, a_list, r_list, r_argmax_list = [], [], [], [], [], []
        for dataset_id, dataset in enumerate(datasets_train):
            x = dataset['data'].astype(np.float32)
            OS = dataset['OS']
            status = dataset['status']

            if dataset_id == 0:
                x1_list.append(x)
                y1_list.append(dataset['subtypes'].astype(np.int32))
            else:
                a_sample, a_argmax = classifier.sample(x)
                r = data_utils.get_reward(OS, status, a_sample, args.subtype_num)
                r_argmax = data_utils.get_reward(OS, status, a_argmax, args.subtype_num)
                r_argmax_list.append(r_argmax[0])
                x2_list.append(x)
                a_list.append(a_sample)
                r_list.append(r)
        x1 = np.concatenate(x1_list, axis=0)
        y1 = np.concatenate(y1_list, axis=0)
        x2 = np.concatenate(x2_list, axis=0)
        a = np.concatenate(a_list, axis=0).astype(np.int32)
        reward = np.concatenate(r_list, axis=0).astype(np.float32)
        mean_r = np.mean(r_argmax_list)

        # trainig model
        training_start_time = time.time()
        loss_su, loss_rl, loss_bl, loss_l1, curr_logits_list = SRPS.train_step_rl(
            x1, y1, x2, a, reward,
            models, 
            optimizers, 
            [args.loss_su, args.loss_rl],
            args.dropout,
            args.dropout_rl
        )
        train_source_acc = np.mean(np.argmax(curr_logits_list[0], axis=1) == y1)
        train_target_acc = np.mean(np.argmax(curr_logits_list[1], axis=1) == datasets_train[1]['subtypes'])

        # visualizing results
        if epoch == args.epoch_num - 1:
            print('epoch:{:}|'.format(epoch) +
                'T:{:.1f}|'.format((time.time() - start_time)/60.) +
                'tr_s_acc:{:.2f}|'.format(train_source_acc) +
                'tr_t_acc:{:.2f}|'.format(train_target_acc) + 
                'tr_r:{:.2f}|'.format(mean_r) + 
                'loss_su:{:.3f}|'.format(np.mean(loss_su)) + 
                'loss_rl:{:.3f}|'.format(np.mean(loss_rl)) + 
                'loss_bl:{:.3f}|'.format(np.mean(loss_bl))
            )

        acc_list.append(train_target_acc)
    np.savetxt(join(acc_save_path, 'acc_para{:}_i{:}_j{:}.csv'.format(args.para_id, args.os_swap_i, args.batch_noise_j)), np.asarray(acc_list))

    # save accuracy
    file_name = 'target_domain_acc_SRPS.csv' if 'SRPS' in args.mode else 'target_domain_acc_DNN.csv'
    with open(join(result_path, file_name), 'a+') as f:
        f.write('{:},{:},{:.3f}\n'.format(
                args.os_swap_i * args.os_swap_rate_step, 
                args.batch_noise_j * args.batch_noise_rate_step, 
                train_target_acc
            )
        )

def training_SRPS(datasets_groups):
    datasets_train, datasets_valid, datasets_test = datasets_groups

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, 'ckpt_fold{:}'.format(args.test_fold))

    print('Trarining', args.mode)

    models, optimizers = SRPS.make_models(
        [(args.subtype_num, False, None, args.regu)],
        [(args.hdim, True, 'sigmoid'), (args.hdim, True, 'sigmoid'), (1, True, None)],
        [args.learning_rate, args.bl_lr], 
    )
    classifier, baseline = models

    # build the baseline model
    baseline(tf.zeros((1, datasets_train[0]['data'].shape[1])), tf.zeros(1))

    start_time = time.time()
    best_r = args.initial_valid_best_r
    valid_best_r = args.initial_valid_best_r
    test_saved_r = args.initial_valid_best_r
    for epoch in range(args.epoch_num):
        x1_list, y1_list, x2_list, a_list, r_list, r_argmax_list = [], [], [], [], [], []
        for dataset_id, dataset in enumerate(datasets_train):
            x = dataset['data'].astype(np.float32) 
            OS = dataset['OS']
            status = dataset['status']
            DFS = dataset['DFS']
            recurrence = dataset['recurrence']

            if dataset_id == 0:
                x1_list.append(x)
                y1_list.append(dataset['subtypes'].astype(np.int32))
            else:
                a_sample, a_argmax = classifier.sample(x)
                r = (data_utils.get_reward(OS, status, a_sample, args.subtype_num) + 
                    data_utils.get_reward(DFS, recurrence, a_sample, args.subtype_num)
                ) / 2
                r_argmax = (data_utils.get_reward(OS, status, a_argmax, args.subtype_num) + 
                    data_utils.get_reward(DFS, recurrence, a_argmax, args.subtype_num)
                ) / 2
                r_argmax_list.append(r_argmax[0])
                x2_list.append(x)
                a_list.append(a_sample)
                r_list.append(r)
        x1 = np.concatenate(x1_list, axis=0)
        y1 = np.concatenate(y1_list, axis=0)
        x2 = np.concatenate(x2_list, axis=0)
        a = np.concatenate(a_list, axis=0).astype(np.int32)
        reward = np.concatenate(r_list, axis=0).astype(np.float32)
        mean_r = np.mean(r_argmax_list)

        valid_r_list = []
        for domain, dataset in enumerate(datasets_valid):
            valid_ids = dataset['groups'] == 1
            x = dataset['data'][valid_ids].astype(np.float32)
            OS = dataset['OS'][valid_ids]
            status = dataset['status'][valid_ids]
            DFS = dataset['DFS'][valid_ids]
            recurrence = dataset['recurrence'][valid_ids]

            valid_a = classifier.predict(x)
            if domain == 0:
                valid_acc = np.mean(valid_a==dataset['subtypes'][valid_ids])
            else:
                r_valid = (data_utils.get_reward(OS, status, valid_a, args.subtype_num) + 
                    data_utils.get_reward(DFS, recurrence, valid_a, args.subtype_num)
                )/2
                valid_r_list.append(r_valid[0])
        valid_mean_r = np.mean(valid_r_list)

        test_r_list = []
        for dataset_id, dataset in enumerate(datasets_test):
            x = dataset['data'].astype(np.float32)
            OS = dataset['OS']
            status = dataset['status']
            DFS = dataset['DFS']
            recurrence = dataset['recurrence']
            test_a = classifier.predict(x)

            if dataset_id == 0:
                test_acc = np.mean(test_a==dataset['subtypes'])
            else:
                r_test = (data_utils.get_reward(OS, status, test_a, args.subtype_num) + 
                    data_utils.get_reward(DFS, recurrence, test_a, args.subtype_num)
                )/2
                test_r_list.append(r_test[0])
        test_mean_r = np.mean(test_r_list)

        start_save_ep = 1000 if args.data == 'LUAD' else 100
        if valid_best_r < valid_mean_r and epoch > start_save_ep and not args.no_early_stop:
            classifier.save_weights(model_path)
            valid_best_r = copy.deepcopy(valid_mean_r)
            test_saved_r = copy.deepcopy(test_mean_r)
            print('save model')

        # trainig model
        training_start_time = time.time()
        loss_su, loss_rl, loss_bl, loss_l1, curr_logits_list = SRPS.train_step_rl(
            x1, y1, x2, a, reward,
            models, 
            optimizers, 
            [args.loss_su, args.loss_rl],
            args.dropout,
            args.dropout_rl
        ) if args.bl_lr > 0. else SRPS.train_step_rl_no_baseline(
            x1, y1, x2, a, reward,
            models, 
            optimizers, 
            [args.loss_su, args.loss_rl],
            args.dropout,
            args.dropout_rl
        )
        acc = np.mean(np.argmax(curr_logits_list[0], axis=1) == y1)

        # visualizing results
        if (epoch + 1) % 100 == 0:
            print('epoch:{:}|'.format(epoch) +
                'T:{:.1f}|'.format((time.time() - start_time)/60.) +
                'l_su:{:.2f}|'.format(np.mean(loss_su)) +
                'l_rl:{:.2f}|'.format(np.fabs(np.mean(loss_rl))) +
                'l_bl:{:.2f}|'.format(np.mean(loss_bl)) +
                'l_l1:{:.3f}|'.format(loss_l1) +
                'acc:{:.2f}|'.format(acc) +
                'r:{:.2f}|'.format(mean_r) +
                'v_acc:{:.2f}|'.format(valid_acc) +
                'v_r:{:.2f}|'.format(valid_mean_r) +
                'v_best_r:{:.2f}|'.format(valid_best_r) + 
                't_r: {:.2f} |'.format(test_saved_r)
            )
    if not args.no_early_stop:
        classifier.load_weights(model_path)
    else:
        classifier.save_weights(model_path)
    test_and_save_results(datasets_test, classifier, result_path, args.test_fold)

def training_soft_survival(datasets_groups):
    datasets_train, datasets_valid, datasets_test = datasets_groups

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, 'ckpt_fold{:}'.format(args.test_fold))

    args.epoch_num = 2000
    layer_params = {
        'encoder_layer_num': args.encoder_layer_num,
        'encoder_hdim': args.encoder_hdim,
        'class_layer_num': args.class_layer_num,
        'class_hdim': args.class_hdim,
        'use_bias': args.use_bias,
        'activation': args.activation,
        'regularization': args.regu,
        'subtype_num': args.subtype_num
    }

    if 'SRPS' in args.mode:
        classifier, optimizer = SRPSsoft.make_models(
            layer_params, args.learning_rate
        )

    print('Trarining', args.mode)

    start_time = time.time()
    valid_best_r = args.initial_valid_best_r
    test_saved_r = args.initial_valid_best_r
    batch_size = args.batch_size
    start_id_source = 0
    start_id_target = 0
    full_ids_source = np.arange(len(datasets_train[0]['OS']))
    full_ids_target = np.arange(len(datasets_train[1]['OS']))
    step_num = int(np.ceil(len(full_ids_source) / batch_size)) if batch_size > 0 else 1
    print('epoch_num:', args.epoch_num, ' step_num:', step_num)

    for epoch in range(args.epoch_num):
        np.random.shuffle(full_ids_source)
        for step in range(step_num):
            if batch_size > 0:
                ids_source = full_ids_source[step * batch_size : min((step + 1) * batch_size, len(full_ids_source))]
                x_source = datasets_train[0]['data'].astype(np.float32)[ids_source, :]
                y_source = datasets_train[0]['subtypes'].astype(np.int32)[ids_source]

                if start_id_target < len(full_ids_target):
                    ids_target = full_ids_target[start_id_target : min(start_id_target + batch_size, len(full_ids_target))]
                    start_id_target += batch_size
                else:
                    start_id_target = 0
                    np.random.shuffle(full_ids_target)
                x_target = datasets_train[1]['data'].astype(np.float32)[ids_target, :]
                OS = datasets_train[1]['OS'].astype(np.float32)[ids_target]
                status = datasets_train[1]['status'].astype(np.float32)[ids_target]
                DFS = datasets_train[1]['DFS'].astype(np.float32)[ids_target]
                recurrence = datasets_train[1]['recurrence'].astype(np.float32)[ids_target]
            else:
                x_source = datasets_train[0]['data'].astype(np.float32)
                y_source = datasets_train[0]['subtypes'].astype(np.int32)
                x_target = datasets_train[1]['data'].astype(np.float32)
                OS = datasets_train[1]['OS'].astype(np.float32)
                status = datasets_train[1]['status'].astype(np.float32)
                DFS = datasets_train[1]['DFS'].astype(np.float32)
                recurrence = datasets_train[1]['recurrence'].astype(np.float32)

            # optimization
            if 'SRPS' in args.mode:
                loss_su, loss_surv, loss_l1 = SRPSsoft.train_step(
                    x_source, y_source, x_target, OS, status, DFS, recurrence,
                    classifier, 
                    optimizer,
                    [args.loss_su, args.loss_survival],
                    args.dropout,
                    args.learning_rate
                )

        eval_str = 'epoch={:} T={:.1f} '.format(epoch, (time.time() - start_time)/60.)
        eval_str, r_train = oneline_eval(datasets_train, classifier, eval_str + '|Train ', args.subtype_num, False)
        eval_str, r_valid = oneline_eval(datasets_valid, classifier, eval_str + '|Valid ', args.subtype_num, False)
        eval_str, r_test = oneline_eval(datasets_test, classifier, eval_str + '|Test ', args.subtype_num, False)

        if valid_best_r < r_valid and epoch > 100 and args.data == 'HCC':
            classifier.save_weights(model_path)
            valid_best_r = copy.deepcopy(r_valid)
            test_saved_r = copy.deepcopy(r_test)
            print('save model')

        extra_str = '|best_v_r={:.1f} saved_t_r={:.1f}'.format(valid_best_r, test_saved_r)

        if (epoch + 1) % 100 == 0:
            print(eval_str + extra_str)

    # classifier.load_weights(model_path)
    test_and_save_results(datasets_test, classifier, result_path, args.test_fold)

def training_DA(datasets_groups):
    datasets_train, datasets_valid, datasets_test = datasets_groups

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, 'ckpt_fold{:}'.format(args.test_fold))

    print('Trarining', args.mode)
    # load model
    layer_params = {
        'encoder_layer_num': args.encoder_layer_num,
        'encoder_hdim': args.encoder_hdim,
        'class_layer_num': args.class_layer_num,
        'class_hdim': args.class_hdim,
        'use_bias': args.use_bias,
        'activation': args.activation,
        'regularization': args.regu,
        'subtype_num': args.subtype_num
    }
    if 'DANN' in args.mode:
        args.learning_rate = 0.01
        classifier, optimizer = DANN.make_models(args.learning_rate, layer_params)

    start_time = time.time()
    valid_best_r = args.initial_valid_best_r

    batch_size = args.batch_size
    start_id_source = 0
    start_id_target = 0
    full_ids_source = np.arange(len(datasets_train[0]['OS']))
    full_ids_target = np.arange(len(datasets_train[1]['OS']))
    step_num = int(np.ceil(len(full_ids_source) / batch_size))
    loss1_list, loss2_list, loss3_list = [], [], []
    print('epoch_num:', args.epoch_num, ' step_num:', step_num)

    nan_flag = False
    for epoch in range(args.epoch_num):
        np.random.shuffle(full_ids_source)
        for step in range(step_num):
            ids_source = full_ids_source[step * batch_size : min((step + 1) * batch_size, len(full_ids_source))]
            x_source = datasets_train[0]['data'][ids_source, :].astype(np.float32)
            y_source = datasets_train[0]['subtypes'][ids_source].astype(np.int32)

            if start_id_target < len(full_ids_target):
                ids_target = full_ids_target[start_id_target : min(start_id_target + batch_size, len(full_ids_target))]
                start_id_target += batch_size
            else:
                start_id_target = 0
                np.random.shuffle(full_ids_target)
            x_target = datasets_train[1]['data'][ids_target, :].astype(np.float32)

            p = float(epoch * step_num + step) / (args.epoch_num * step_num)
            alpha = 2. / (1. + np.exp(-10. * p)) - 1
            lr = args.learning_rate / (1. + 10 * p)**0.75

            # optimization
            if 'DANN' in args.mode:
                loss_su, loss_do = DANN.train_step(
                    x_source, y_source, x_target,
                    classifier, 
                    optimizer,
                    args.dropout, 
                    tf.cast(alpha, dtype=tf.float32),
                    tf.cast(lr, dtype=tf.float32)
                )
                if np.isnan(loss_su.numpy()) or np.isnan(loss_do.numpy()):
                    nan_flag = True
                    break

        if nan_flag:
            break

        # domain acc
        if 'DANN' in args.mode:
            x_domain = np.concatenate(
                [datasets_train[0]['data'].astype(np.float32), datasets_train[1]['data'].astype(np.float32)],
                axis=0
            )
            y_domain = np.concatenate(
                [np.zeros((len(datasets_train[0]['data']))), np.ones((len(datasets_train[1]['data'])))],
                axis=0
            )
            acc_domain_train = np.mean(np.argmax(classifier.domain_pred_test(x_domain), axis=-1) == y_domain)

            x_domain = np.concatenate(
                [datasets_valid[0]['data'].astype(np.float32), datasets_valid[1]['data'].astype(np.float32)],
                axis=0
            )
            y_domain = np.concatenate(
                [np.zeros((len(datasets_valid[0]['data']))), np.ones((len(datasets_valid[1]['data'])))],
                axis=0
            )
            acc_domain_valid = np.mean(np.argmax(classifier.domain_pred_test(x_domain), axis=-1) == y_domain)
        else: 
            acc_domain_train = acc_domain_valid = 0.

        eval_str = 'epoch={:} '.format(epoch) + \
            'T={:.1f} '.format((time.time() - start_time)/60.)

        eval_str, _ = oneline_eval(datasets_train, classifier, eval_str + '|Train ', args.subtype_num)
        eval_str, _ = oneline_eval(datasets_valid, classifier, eval_str + '|Valid ', args.subtype_num)
        # eval_str, _ = oneline_eval(datasets_test, classifier, eval_str + '|Test ', args.subtype_num)

        extra_str = ''
        if 'DANN' in args.mode:
            extra_str = '| acc_D train={:.3f}'.format(acc_domain_train) + ' valid={:.3f}'.format(acc_domain_valid)

        if (epoch + 1) % 100 == 0:
            print(eval_str  + extra_str)

    test_and_save_results(datasets_test, classifier, result_path, args.test_fold)
    classifier.save_weights(model_path)
    print('save model')

def training_sklearn_classifier(datasets_groups):
    datasets_train, datasets_valid, datasets_test = datasets_groups

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, 'ckpt_fold{:}'.format(args.test_fold))

    print('Trarining', args.mode)
    dataset = datasets_train[0]
    x = dataset['data'].astype(np.float32)
    y = dataset['subtypes'].astype(int)
    if 'LogisticRegression' in args.mode:
        classifier = LogisticRegression(max_iter=1000, C=args.alpha).fit(x, y)
    elif 'RandomForest' in args.mode:
        classifier = RandomForestClassifier(max_depth=args.max_depth, random_state=args.seed).fit(x, y)
    test_and_save_results(datasets_test, classifier, result_path, args.test_fold)

if __name__ == '__main__':
    start_time = time.time()

    # set random seed
    np.random.seed(args.seed)

    # set data path
    platform_sys = platform.system()
    if platform_sys == 'Linux':
        file_name = 'all_datasets_linux.p'
    else:
        file_name = 'all_datasets_windows.p'

    pickle_file_path = join(args.data_path, file_name)
    
    # load data
    datasets_dict, gene2prote_dict, prote2gene_dict = pickle.load(open(pickle_file_path, 'rb'))

    if args.data == 'toy':
        data_path = join(args.data_path, args.data)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        raw_datasets = generate_survival_correlated_synthetic_data(
            args.seed, 
            n_sample=200,
            censor_rate=0., 
            os_swap_rate=args.os_swap_i * args.os_swap_rate_step,
            batch_noise_rate=args.batch_noise_j * args.batch_noise_rate_step,
            viz=True, 
            save_path=data_path, 
        )
    elif args.data == 'synthetic':
        raw_datasets = datasets_dict['synthetic-' + args.batch_effect]
    elif args.data == 'HCC':
        raw_datasets = datasets_dict['HCC-Jiang2Gao']
    elif args.data == 'LUAD':
        raw_datasets = datasets_dict['LUAD-Xu2Gillette']
    elif args.data == 'HCC_LUAD':
        raw_datasets = datasets_dict['HCC_LUAD-Jiang2Xu']
    else: 
        assert False, 'unknown data name'

    if args.data in ['HCC', 'LUAD', 'HCC_LUAD', 'toy']:
        for i, dataset in enumerate(raw_datasets):
            data = raw_datasets[i]['data']
            data = (data - np.mean(data, axis=0))/(np.std(data, axis=0) + 1e-5)
            raw_datasets[i]['data'] = data

    # split into n folds
    print('n fold split')
    datasets = data_utils.n_fold_split(
        raw_datasets, 
        args.fold_num, 
        args.test_fold, 
        stratify=True if args.data in ['synthetic', 'toy'] else False
    )

    # run harmony
    if 'harmony' in args.experiment or 'Harmony' in args.mode:
        datasets = data_utils.run_harmony(
            copy.deepcopy(datasets), 
            args.subtype_num, 
            theta=args.theta, 
            lamb=args.lamb, 
            sigma=args.sigma, 
            n_components=args.n_components
        )

    # divide into training, validation and testing groups
    datasets_groups = data_utils.train_valid_test_regroup(copy.deepcopy(datasets))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    device = args.device
    tf.config.threading.set_intra_op_parallelism_threads(10)
    tf.config.threading.set_inter_op_parallelism_threads(10)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if args.data == 'toy':
        training_toy_data(datasets_groups)
    else:
        if ('SRPS(soft)' in args.mode):
            with tf.device(device):
                classifier = training_soft_survival(datasets_groups)
        elif 'SRPS' in args.mode:
            with tf.device(device):
                classifier = training_SRPS(datasets_groups)
        elif 'DANN' in args.mode:
            with tf.device(device):
                training_DA(datasets_groups)
        elif 'RandomForest' in args.mode:
            training_sklearn_classifier(datasets_groups)
        elif args.mode == 'compare':
            experiment = join(args.experiment, 'batch_effect_' + args.batch_effect) if args.data == 'synthetic' else args.experiment
            data = args.data if '_' not in args.data else args.data.split('_')[0]
            metric_table(datasets, args.data_path, experiment, data, args.plot, args.max_t)
        else:
            print('unknown mode')

    print('time used: {:d} sec'.format(int(time.time() - start_time)))

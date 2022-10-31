import re, copy, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import pandas as pd
import data_utils as utils
import seaborn as sns
import matplotlib.lines as mlines
from os.path import join
from numba import jit
from collections import Counter
from lifelines import KaplanMeierFitter, statistics, plotting
from lifelines.utils import restricted_mean_survival_time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from scipy.signal import savgol_filter

colors = ['tab:blue', 'orange', 'tab:red', 'tab:green', 'tab:purple', 
         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
lines = ['-', '--', ':']
markers = ['.', 'x', '^']

def read_and_plot():
    save_path = os.path.join(args.data_path, 'concrete_feature_selection', args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_path = os.path.join(save_path, 'results'+POSTFIX)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    datasets, train_data, test_data, all_data, raw_all_data, all_patients = load_data(args.data_path)
    indices_list = [np.arange(len(all_data))[all_data[:, -1]==i] for i in range(len(datasets))]
    input_list = [all_data[indices, :-5] for indices in indices_list]
    OS_list = [all_data[indices, -4] for indices in indices_list]
    status_list = [all_data[indices, -3] for indices in indices_list]
    for i, (indices, OS, status) in enumerate(zip(indices_list, OS_list, status_list)):
        assignments = utils.read_assignments(
            os.path.join(
                result_path, 
                'center{:}_classification_result.txt'.format(i)
            ),
            datasets[i]['patients']
        )
        utils.plot_survival_cruve(
            OS, 
            status, 
            assignments, 
            3, 
            title=None, 
            path=result_path, 
            prefix='center{:}_'.format(i),
            pop_viz=True
        )

def visualize_features(datasets):
    all_data = [
        np.c_[
            dataset['data'], 
            dataset_id * np.ones([len(dataset['data'],), 1])
        ] for dataset_id, dataset in enumerate(datasets)
    ]
    all_data = np.concatenate(all_data, axis=0)
    feat_2d = TSNE(n_components=2).fit_transform(all_data[:, :-1])
    plt.figure(figsize=(12, 12))
    for i in range(len(datasets)):
        feat = feat_2d[all_data[:, -1]==i]
        plt.scatter(feat[:, 0], feat[:, 1], c=colors[i], label='domain {:}'.format(i))
    plt.title('2D TSNE visualization for normalized features')
    plt.legend()
    plt.show()

def plot_survival_cruve(time, event, assignments, k, title='Survival Curve', 
        path=None, prefix=[], pop_viz=False, new_fig=True, at_risk=True, clip_time=60, ax=None):
    if new_fig:
        plt.figure(figsize=(5, 4))

    sc_list = []
    rmst_list = []
    kmf_list = []
    for i in range(k):
        label = 'subtype {:}, population {:}'.format(
            i, 
            np.sum(assignments==i)
        ) if pop_viz else 'subtype {:}'.format(i)
        if np.sum(assignments==i) == 0: 
            print('no class {:}, result not save'.format(i))
            return 
        kmf = KaplanMeierFitter().fit(
            time[assignments==i], 
            event[assignments==i], 
            label=label
        )  
        ax = kmf.plot(
            ci_show=False, 
            color=colors[i],
            show_censors=True
        ) if i == 0 or ax is None else kmf.plot(
            ax=ax, 
            ci_show=False, 
            color=colors[i],
            show_censors=True
        )
        sc_list.append([kmf.timeline, np.squeeze(kmf.survival_function_.to_numpy())])
        rmst = restricted_mean_survival_time(kmf, t=clip_time)
        rmst_list.append(rmst)
        kmf_list.append(kmf)
    plt.xlim((0, clip_time))
    plt.ylabel('cum survival %')
    plt.ylim((-0.1, 1.1))
    plt.grid(True)
    if at_risk:
        plotting.add_at_risk_counts(
            kmf_list[0], kmf_list[1], kmf_list[2],
            labels=['subtype 1', 'subtype 2', 'subtype 3'],
            rows_to_show=['At risk']
        )

    logrank_results = statistics.pairwise_logrank_test(time, assignments, event)
    multivariate_logrank_result = statistics.multivariate_logrank_test(
        time, 
        assignments, 
        event
    )

    plt.title(title+'| \u0394RMST: {:.1f}, {:.1f}'.format(
            rmst_list[0] - rmst_list[1], 
            rmst_list[1] - rmst_list[2],
        )
    )
    p_string = 'P(1,2): {:.5f}\nP(2,3): {:.5f}\nP: {:.5f}'.format(
        logrank_results.p_value[0],
        logrank_results.p_value[2],
        multivariate_logrank_result.p_value
    )
    plt.text(40, 0, p_string)
    
    # plt.xlabel('months')
    plt.tight_layout()
    if path is not None:
        plt.savefig(os.path.join(path, prefix+'plot_survival_cruve.png'))
    if new_fig:
        plt.close()
    return kmf, ax

def plot_km_curve_custom(times, events, assignments, k, ax, title, dRMST=True, labels=None, text_bias=33, additional_text=''):
    subtype_dict = {0: 'I', 1: 'II', 2: 'III'}
    rmst_list = []
    for i in range(k):
        S_list, time_list, censor_list, at_risk_list, RMST = utils.get_km_curve(
            times[assignments==i], 
            events[assignments==i], 
            clip_time=60
        )
        rmst_list.append((RMST))
        label = 'S-{}'.format(subtype_dict[i]) if labels is None else labels[i]
        ax.plot(time_list, S_list, c=colors[i], label=label+' N={:}'.format(np.sum(assignments==i)))
        ax.plot(
            np.asarray(time_list)[censor_list], np.asarray(S_list)[censor_list], 
            '|', c=colors[i], markeredgewidth=1, markersize=10)
    logrank_results = statistics.pairwise_logrank_test(times, assignments, events)
    multivariate_logrank_result = statistics.multivariate_logrank_test(
        times, 
        assignments, 
        events
    )
    
    title = title +' \u0394RMST: {:.1f}, {:.1f}'.format(
        rmst_list[0] - rmst_list[1], 
        rmst_list[1] - rmst_list[2],
    ) if dRMST else title
    ax.set_title(title)

    p_string = 'P(I,II): {:.5f}\nP(II,III): {:.5f}\nP: {:.5f}'.format(
        logrank_results.p_value[0],
        logrank_results.p_value[2],
        multivariate_logrank_result.p_value
    ) if k == 3 else 'P: {:.5f}'.format(multivariate_logrank_result.p_value)
    p_string += additional_text
    ax.text(text_bias, 0, p_string)
    ax.grid()
    ax.legend(loc='lower left')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, 60)
    return ax
        

def plot_2_survival_cruves(time, event, assignments, labels, title='Survival Curve', path=None, clip_time=60, ax=None):
    plt.figure(figsize=(5, 4))
    rmst_list = []
    kmf_list = []
    for i in range(2):
        label = labels[i] + ', N={:}'.format(np.sum(assignments==i))
        if np.sum(assignments==i) == 0: 
            print('no class {:}, result not save'.format(i))
            return 
        kmf = KaplanMeierFitter().fit(
            time[assignments==i], 
            event[assignments==i], 
            label=label
        )  
        ax = kmf.plot(
            ci_show=False, 
            color=colors[i],
            show_censors=True
        ) if (i == 0 or ax is None) else kmf.plot(
            ax=ax, 
            ci_show=False, 
            color=colors[i],
            show_censors=True
        )
        rmst = restricted_mean_survival_time(kmf, t=clip_time)
        rmst_list.append(rmst)
        kmf_list.append(kmf)
    plt.xlim((0, clip_time))
    plt.ylabel('cum survival %')
    plt.ylim((-0.1, 1.1))
    plt.grid(True)

    logrank_results = statistics.pairwise_logrank_test(time, assignments, event)
    multivariate_logrank_result = statistics.multivariate_logrank_test(time, assignments, event)

    plt.title(title+'_\u0394RMST:{:.1f}'.format(rmst_list[0] - rmst_list[1]))
    p_string = 'P: {:.5f}'.format(multivariate_logrank_result.p_value)
    plt.text(40, 0, p_string)
    
    plt.tight_layout()
    plt.savefig(path)
    return kmf


def plot_count_result(count_result, prote2gene_dict, save_path):
    gene_list, freq_list = [], []
    for prote in count_result.keys():
        gene_list.append(prote2gene_dict[prote])
        freq_list.append(count_result[prote])
    sorted_gene_id = np.argsort(-np.asarray(freq_list))[:10]
    sorted_freq_list = [freq_list[gene_id] for gene_id in sorted_gene_id]
    sorted_gene_list = [gene_list[gene_id] for gene_id in sorted_gene_id]
    fig, ax = plt.subplots() 
    ax.bar(
        list(range(len(sorted_freq_list))), 
        sorted_freq_list,
        edgecolor='black'
    )
    ax.set_yticks(np.arange(np.amax(sorted_freq_list)+4, step=4))
    ax.set_xticks(np.arange(len(sorted_gene_list)))
    ax.set_xticklabels(sorted_gene_list, rotation=40)
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.savefig(
        join(save_path, 'gene_count.png'), 
        format='png', 
        bbox_inches='tight'
    )
    plt.close()


def plot_learning_curvs():
    base_path = r'G:\Data\Tumer_clinic\full_feat_classification'

    path_list = [
        os.path.join(base_path, 'full_feat_hdim1'),
        # os.path.join(base_path, 'full_feat_hdim2'),
        # os.path.join(base_path, 'full_feat_hdim4')
    ]
    labels = [path[len(base_path):] for path in path_list]

    fig, axs = plt.subplots(4, figsize=(4, 8))
    for path_id, path in enumerate(path_list):
        train_log_list = []
        file_list = utils.search_for_files(path, 'csv')
        for file in file_list:
            try:
                train_log_list.append(utils.read_training_records(file))
            except Exception as e:
                continue
            
        train_log_array = np.asarray(train_log_list)
        train_log_avg = np.mean(train_log_array, axis=0)
        train_log_std = np.std(train_log_array, axis=0)
        window_size = 101
        train_acc_avg = savgol_filter(train_log_avg[:, 0], window_size, 3)
        train_acc_std = savgol_filter(train_log_std[:, 0], window_size, 3)
        test_acc_avg = savgol_filter(train_log_avg[:, 1], window_size, 3)
        test_acc_std = savgol_filter(train_log_std[:, 1], window_size, 3)  
        axs[0].plot(
            train_acc_avg, 
            label=labels[path_id],
            linestyle=lines[0],
            color=colors[int(path_id)]

        )
        # axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Training Acc')
        axs[0].set_title('Subtype prediction')
        axs[0].grid(True)

        axs[1].plot(
            test_acc_avg, 
            label=labels[path_id],
            linestyle=lines[0],
            color=colors[int(path_id)]
        )
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Testing Acc')
        axs[1].grid(True)

        train_acc_avg = savgol_filter(train_log_avg[:, 2], window_size, 3)
        train_acc_std = savgol_filter(train_log_std[:, 2], window_size, 3)
        test_acc_avg = savgol_filter(train_log_avg[:, 3], window_size, 3)
        test_acc_std = savgol_filter(train_log_std[:, 3], window_size, 3)  
        axs[2].plot(
            train_acc_avg, 
            label=labels[path_id],
            linestyle=lines[0],
            color=colors[int(path_id)]

        )
        # axs[0].set_xlabel('Epochs')
        axs[2].set_ylabel('Training Delta OS')
        axs[2].grid(True)

        axs[3].plot(
            test_acc_avg, 
            label=labels[path_id],
            linestyle=lines[0],
            color=colors[int(path_id)]
        )
        axs[3].set_xlabel('Epochs')
        axs[3].set_ylabel('Testing Delta OS')
        axs[3].grid(True)

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(join(base_path, 'learning_curvs.png'))

def plot_err_samples(assignments_list, labels, patients, save_path):
    validation_samples = ['X102T', 'X277T', 'X308T', 'X356T', 'XH03T', 'XL18T', 'X1259T', 'X366T', 'XH19T', 'XL06T', 'X12281T', 'X296T', 'X417T', 'XH06T', 'XL07T']
    clf_err_list = [1 - (assignments==labels) for assignments in assignments_list]
    clf_err_samples = np.sum(np.stack(clf_err_list, axis=0), axis=0)
    assignments_array = np.stack(assignments_list, axis=1)
    assign_class_nums = np.stack([
            np.sum(assignments_array==i, axis=1) for i in range(3)
        ], axis=1
    )
    sorted_sample_ids = np.argsort(-clf_err_samples)
    sorted_err_samples = clf_err_samples[sorted_sample_ids]
    above_zero = sorted_err_samples > 5
    sorted_sample_ids = sorted_sample_ids[above_zero]
    sorted_err_samples = sorted_err_samples[above_zero]
    sorted_labels = labels[sorted_sample_ids]
    sorted_patients = [patients[sample_id] for sample_id in sorted_sample_ids]
    sorted_assign_class_nums = assign_class_nums[sorted_sample_ids, :]
    fig, ax = plt.subplots() 
    bars = ax.bar(
        np.arange(len(sorted_err_samples)), 
        sorted_err_samples,
        color=[colors[int(label)] for label in sorted_labels],
        edgecolor='black'
    )
    ax.set_ylim([0,np.amax(sorted_err_samples)+10])
    ax.set_yticks(np.arange(np.amax(sorted_err_samples)+10, step=4))
    ax.set_xticks(np.arange(len(sorted_patients)))
    ax.set_xticklabels(sorted_patients, rotation=40)
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', label='subtype1'),
        Patch(facecolor=colors[1], edgecolor='black', label='subtype2'),
        Patch(facecolor=colors[2], edgecolor='black', label='subtype3')
    ]
    plt.legend(handles=legend_elements)
    for i, v in enumerate(sorted_err_samples):
        ax.text(i - 0.25, v + 5, str(sorted_assign_class_nums[i, 0]), color=colors[0])
        ax.text(i - 0.25, v + 3, str(sorted_assign_class_nums[i, 1]), color=colors[1])
        ax.text(i - 0.25, v + 1, str(sorted_assign_class_nums[i, 2]), color=colors[2])
        if sorted_patients[i] in validation_samples:
            ax.text(i - 0.25, v + 7, 'v')
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.savefig(
        os.path.join(save_path, 'err_patients.png'), 
        format='png', 
        bbox_inches='tight'
    )
    plt.close()

def barchat_compare_with_baselines(feat_num_list, fs_names, fnum_fs_val_mean, fnum_fs_val_std, ylabel, save_path, ylimit = None):
    fig, ax = plt.subplots()
    xticks = np.arange(len(feat_num_list))
    width = 1./(len(fs_names) + 1)
    bias = (np.arange(len(fs_names)) - len(fs_names)/2. + 0.5) * width
    for fs_id in range(len(fs_names)):
        rect = ax.bar(
            xticks + bias[fs_id], 
            fnum_fs_val_mean[:, fs_id], 
            width=width,
            yerr=fnum_fs_val_std[:, fs_id],
            label=fs_names[fs_id],
            color=colors[fs_id], 
            capsize=3
        )
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(ylimit[0], ylimit[1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(feat_num_list)
    ax.set_xlabel('Number of features')
    ax.set_title('Compare features selected by different approches')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_km_curve(time, event, assignments, title, labels, color_list, save_path):
    plt.figure(figsize=(4, 4))
    kmf_list = []
    for i in range(2):
        kmf = KaplanMeierFitter().fit(
            time[assignments==i], 
            event[assignments==i], 
            label=labels[i]
        )  
        ax = kmf.plot(
            ci_show=False, 
            color=colors[color_list[i]]
        ) if i == 0 else kmf.plot(
            ax=ax, 
            ci_show=False, 
            color=colors[color_list[i]]
        )
        kmf_list.append(kmf)
    plt.xlim((0, 60))
    plt.ylabel('cum survival %')
    plt.ylim((-0.1, 1.1))
    plt.grid(True)
    plotting.add_at_risk_counts(
        kmf_list[0], kmf_list[1],
        labels=labels,
        rows_to_show=['At risk']
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_datasets_km_curve(all_datasets):
    nature_os = all_datasets[0]['OS']
    nature_status = all_datasets[0]['status']
    nature_label = all_datasets[0]['label']

    for center_id in range(3):
        for predix in [1, 4, 7]:
            OS = all_datasets[center_id+predix]['OS']
            status = all_datasets[center_id+predix]['status']
            label = all_datasets[center_id+predix]['label']
  
            time = np.r_[nature_os, OS]
            event = np.r_[nature_status, status]
            assignments = np.r_[np.zeros_like(nature_os), np.ones_like(OS)]
            logrank_results = statistics.pairwise_logrank_test(time, assignments, event)
            p_value = logrank_results.p_value[0]
            plot_km_curve(
                time, 
                event, 
                assignments,
                nature_label + ' and ' + label + ' | P={:.4f}'.format(p_value),
                [nature_label, label],
                [0, 2],
                join(r'G:\Data\Tumer_clinic\reinforced_classification\reinforce_14datasets_20k\plot', nature_label + ' and ' + label + '.png')
            )

def viz_classification_results_2d(data_list, assignments_list, label_list, title, save_path):
    plt.figure(figsize=(8, 8))
    for j, (data, assignments, label) in enumerate(zip(data_list, assignments_list, label_list)):
        if data.shape[1] > 2:
            transformer = PCA(n_components=2, svd_solver='full').fit(data)
            data = transformer.transform(data)
        for i in range(3):
            plt.scatter(
                data[assignments == i, 0], 
                data[assignments == i, 1], 
                c=colors[i], 
                label='subtype'+str(i+1)+'_'+label,
                marker=markers[j]
            )
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def viz_classification_results_3d(prob, assignments, label_list, title, save_path):
    rgb_convert_arr = np.stack([plt_colors.to_rgb(colors[i]) for i in range(3)], axis=0).transpose()
    color_arr = np.matmul(rgb_convert_arr, np.asarray(prob).transpose()).transpose()
    color_arr[color_arr>1] = 1
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for i in range(3):
        ax.scatter(
            prob[assignments==i, 0], 
            prob[assignments==i, 1], 
            prob[assignments==i, 2],
            c=color_arr[assignments==i, :], 
            label=label_list[i],
            marker=markers[i],
            alpha=1
        )
    # cbar = plt.colorbar(sc_list[0])
    # cbar.ax.set_ylabel('months')
    plt.legend()
    ax.plot([0., 1.], [0., 0.], [0., 0.], c=plt_colors.to_rgb(colors[0]))
    ax.plot([0., 0.], [0., 1.], [0., 0.], c=plt_colors.to_rgb(colors[1]))
    ax.plot([0., 0.], [0., 0.], [0., 1.], c=plt_colors.to_rgb(colors[2]))
    ax.text(0.5, 0., 0.05, 'P(S-I)', 'x', color=colors[0])
    ax.text(0., 0.2, -0.1, 'P(S-II)', 'y', color=colors[1])
    ax.text(0.02, 0., 0.2, 'P(S-III)', 'z', color=colors[2])
    ax.text(0., 0., 0., 'O', color='black')
    plt.title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.)
    ax.set_zlim(0, 1.)
    ax.view_init(elev=30., azim=45)
    plt.tight_layout()
    plt.savefig(save_path)

def viz_classification_results_OS_3d(prob, OS, status, label_list, title, save_path):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    sc_list = []
    for i in range(2):
        sc = ax.scatter(
            prob[status==i, 0], 
            prob[status==i, 1], 
            prob[status==i, 2],
            c=OS[status==i], 
            label=label_list[i],
            marker=markers[i],
            alpha=1,
            vmin=0, vmax=60
        )
        sc_list.append(sc)
    cbar = plt.colorbar(sc_list[0])
    cbar.ax.set_ylabel('months')
    plt.legend()
    ax.plot([0., 1.], [0., 0.], [0., 0.], c=plt_colors.to_rgb(colors[0]))
    ax.plot([0., 0.], [0., 1.], [0., 0.], c=plt_colors.to_rgb(colors[1]))
    ax.plot([0., 0.], [0., 0.], [0., 1.], c=plt_colors.to_rgb(colors[2]))
    ax.text(0.6, 0., 0.05, 'P(S-I)', 'x', color=colors[0])
    ax.text(0., 0.2, -0.1, 'P(S-II)', 'y', color=colors[1])
    ax.text(0.02, 0., 0.2, 'P(S-III)', 'z', color=colors[2])
    ax.text(0., 0., 0., 'O', color='black')
    plt.title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.)
    ax.set_zlim(0, 1.)
    ax.view_init(elev=30., azim=45)
    plt.tight_layout()
    plt.savefig(save_path)

def viz_classification_results_3d_in_2d(data_list, assignments_list, label_list, title, save_path):
    fig = plt.figure(figsize=(8, 8))
    A = np.array([[0, np.cos(np.pi/6), -np.cos(np.pi/6)], 
                  [1, -np.sin(np.pi/6), -np.sin(np.pi/6)]])
    data_2d_list = [np.matmul(A, data.transpose()).transpose() for data in data_list]
    max_projected_dist = np.amax(np.fabs(np.concatenate(data_list, axis=0)))
    base_vects_3d = np.eye(3) * max_projected_dist
    base_vects_2d = np.matmul(A, base_vects_3d.transpose()).transpose()
    # plot axis
    bias_list = [[max_projected_dist/40.,0.], [0., max_projected_dist/40.], [-max_projected_dist/40., max_projected_dist/40.]]
    for base_vect_2d, axis_label, bias in zip(base_vects_2d, ['X', 'Y', 'Z'], bias_list):
        plt.plot([0, base_vect_2d[0]], [0, base_vect_2d[1]], c='k', linewidth=2, zorder=0)
        plt.text(base_vect_2d[0] + bias[0], base_vect_2d[1] + bias[1], axis_label)
    # plot points
    for j, (data, assignments, label) in enumerate(zip(data_2d_list, assignments_list, label_list)):
        for i in range(3):
            plt.scatter(
                data[assignments==i, 0], 
                data[assignments==i, 1], 
                c=colors[i], 
                label='subtype'+str(i+1)+label,
                marker=markers[j],
                zorder=1
            )
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def viz_classification_results_OS_3d_in_2d(data, OS, status, label_list, title, save_path):
    fig = plt.figure(figsize=(10, 8))
    A = np.array([[0, np.cos(np.pi/6), -np.cos(np.pi/6)], 
                  [1, -np.sin(np.pi/6), -np.sin(np.pi/6)]])
    data_2d = np.matmul(A, data.transpose()).transpose()
    max_projected_dist = np.amax(data)
    base_vects_3d = np.eye(3) * max_projected_dist
    base_vects_2d = np.matmul(A, base_vects_3d.transpose()).transpose()
    # plot axis
    bias_list = [[max_projected_dist/40.,0.], [0., max_projected_dist/40.], [-max_projected_dist/40., max_projected_dist/40.]]
    for base_vect_2d, axis_label, bias in zip(base_vects_2d, ['X', 'Y', 'Z'], bias_list):
        plt.plot([0, base_vect_2d[0]], [0, base_vect_2d[1]], c='k', linewidth=2, zorder=0)
        plt.text(base_vect_2d[0] + bias[0], base_vect_2d[1] + bias[1], axis_label)
    # plot points
    sc_list = []
    for i in range(2):
        sc = plt.scatter(
            data_2d[status==i, 0], 
            data_2d[status==i, 1], 
            c=OS[status==i]/60., 
            label=label_list[i],
            marker=markers[i]
        )
        sc_list.append(sc)
    plt.colorbar(sc_list[0])
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def scatter_assignments(data, assignments, probs, A, title):
    rgb_convert_arr = np.stack([plt_colors.to_rgb(colors[i]) for i in range(3)], axis=0).transpose()
    color_arr = np.matmul(rgb_convert_arr, np.asarray(probs).transpose()).transpose()
    color_arr[color_arr>1.] = 1.
    for i in range(3):
        plt.scatter(
            data[assignments==i, 0], 
            data[assignments==i, 1], 
            # c=probs[assignments==i, ::-1],
            c=color_arr[assignments==i],
            label='S'+str(i+1),
            marker=markers[i],
            zorder=1,
        )
    plt.xticks([np.pi/2, np.pi/2+np.pi/3*2, np.pi/2+np.pi/3*4], ['S-1', 'S-2', 'S-3'])
    plt.legend()
    plt.tight_layout()
    plt.title(title)


def scatter_OS(data, OS, status, A, title, fig):
    sc_list = []
    label_list = ['not dead', 'dead']
    for i in range(2):
        sc = plt.scatter(
            data[status==i, 0], 
            data[status==i, 1], 
            c=OS[status==i]/60., 
            label=label_list[i],
            marker=markers[i],
            vmin=0, vmax=1
        )
        sc_list.append(sc)
    plt.xticks([np.pi/2, np.pi/2+np.pi/3*2, np.pi/2+np.pi/3*4], ['S-1', 'S-2', 'S-3'])
    plt.colorbar(sc_list[0])
    plt.legend()
    plt.tight_layout()
    plt.title(title)

def scatter_mvi_prob(data, mvi, probs, A, title):
    rgb_convert_arr = np.stack([plt_colors.to_rgb(colors[i]) for i in range(3)], axis=0).transpose()
    color_arr = np.matmul(rgb_convert_arr, np.asarray(probs).transpose()).transpose()
    mvi_dict = {0: 'negative', 1:'positive'}
    for i in range(2):
        plt.scatter(
            data[mvi==i, 0], 
            data[mvi==i, 1], 
            c=color_arr,
            label='MVI '+mvi_dict[i],
            marker=markers[i],
            zorder=1,
        )
    plt.xticks([np.pi/2, np.pi/2+np.pi/3*2, np.pi/2+np.pi/3*4], ['S-1', 'S-2', 'S-3'])
    plt.legend()
    plt.tight_layout()
    plt.title(title)

def debug_viz(data_list, assignments_list, prob_list, OS_list, status_list, title_list, save_path):
    fig = plt.figure(figsize=(18, 10))
    A = np.array([[0, -np.cos(np.pi/6), np.cos(np.pi/6)], 
                  [1, -np.sin(np.pi/6), -np.sin(np.pi/6)]])
    col_num = 4
    row_num = 2
    for row, (data, assignments, prob, OS, status, title) in enumerate(
        zip(data_list, assignments_list, prob_list, OS_list, status_list, title_list)
    ):
        data_2d = np.matmul(A, data.transpose()).transpose()
        data_polar = np.stack([np.arctan2(data_2d[:, 1], data_2d[:, 0]), np.sqrt(data_2d[:, 0]**2 + data_2d[:, 1]**2)], axis=1)
        fig.add_subplot(row_num, col_num, row*col_num+1, projection='polar')
        scatter_assignments(data_polar, assignments, prob, A, title+'_probs')
        fig.add_subplot(row_num, col_num, row*col_num+2, projection='polar')
        scatter_OS(data_polar, OS, status, A, title+'_OS', fig)
        fig.add_subplot(row_num, col_num, row*col_num+3)
        plot_survival_cruve(OS, status, assignments, 3, title, pop_viz=True, new_fig=False, at_risk=False)
        if row > 0:
            above_thresh = np.amax(prob, axis=1) > 0.9
            print('pass num:', np.sum(above_thresh))
            fig.add_subplot(row_num, col_num, 4, projection='polar')
            scatter_assignments(data_polar[above_thresh], assignments[above_thresh], prob[above_thresh], A, title+'_probs_filter')
            fig.add_subplot(row_num, col_num, row*col_num+4)
            plot_survival_cruve(
                OS[above_thresh], 
                status[above_thresh], 
                assignments[above_thresh], 
                3, 
                title+'_OS_filter', 
                pop_viz=True, 
                new_fig=False,
                at_risk=False
            )

    plt.savefig(save_path)
    plt.close()

def scatter_viz(data, assignments, prob, OS, status, title, save_path):
    fig = plt.figure(figsize=(10, 5))
    A = np.array([[0, -np.cos(np.pi/6), np.cos(np.pi/6)], 
                  [1, -np.sin(np.pi/6), -np.sin(np.pi/6)]])
    data_2d = np.matmul(A, data.transpose()).transpose()
    data_polar = np.stack([np.arctan2(data_2d[:, 1], data_2d[:, 0]), np.sqrt(data_2d[:, 0]**2 + data_2d[:, 1]**2)], axis=1)
    fig.add_subplot(121, projection='polar')
    scatter_assignments(data_polar, assignments, prob, A, title+'_probs')
    fig.add_subplot(122, projection='polar')
    scatter_OS(data_polar, OS, status, A, title+'_OS', fig)

    plt.savefig(save_path)
    plt.close()

def scatter_viz_mvi(prob, mvi, title, save_path):
    fig = plt.figure(figsize=(10, 5))
    A = np.array([[0, -np.cos(np.pi/6), np.cos(np.pi/6)], 
                  [1, -np.sin(np.pi/6), -np.sin(np.pi/6)]])
    data_2d = np.matmul(A, prob.transpose()).transpose()
    data_polar = np.stack([np.arctan2(data_2d[:, 1], data_2d[:, 0]), np.sqrt(data_2d[:, 0]**2 + data_2d[:, 1]**2)], axis=1)
    fig.add_subplot(projection='polar')
    scatter_mvi_prob(data_polar, mvi, prob, A, title+'_mvi_and_prob')
    plt.savefig(save_path)
    plt.close()

def plot_data_distrib(datasets, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    row_num = 2
    col_num = 3
    for i, dataset in enumerate(datasets):
        fig.add_subplot(row_num, col_num, i+1)
        plt.hist(dataset['data'].flatten(), 50)
        plt.title(dataset['label'])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()

def weight_viz(weights, save_path=None):
    weights_num, col_num = weights.shape
    weight_list = np.hsplit(weights, col_num)
    fig = plt.figure(figsize=(10, 10))
    feat_mask = np.zeros_like(weight_list[0].flatten())
    for i, weight in enumerate(weight_list):
        weight = weight.flatten()
        fig.add_subplot(col_num, 1, i+1)
        plt.hist(weight, 50)
        feat_mask += (weight!=0)
        title = 'Dim {:}, non-zero num: {:d}'.format(i, np.sum(weight!=0).astype(int))
        if i == 2:
            title += ', totle feat num: {:d}'.format(np.sum(feat_mask>0))
        plt.title(title)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()

def distri_compare(data, label, legend_list, title, save_path):
    label_lines = []
    fig, ax = plt.subplots()
    for cls_id in range(len(legend_list)):
        ids = ((label==cls_id) * (1 - np.isnan(data))).astype(bool)
        d = {'x': data[ids], 'cls': ['{}'.format(cls_id)] * np.sum(ids)}
        sns.kdeplot(data=pd.DataFrame(d), x='x', ax=ax, color=colors[cls_id])
        label_lines.append(mlines.Line2D([], [], color=colors[cls_id], label=legend_list[cls_id]))
    plt.title(title)
    plt.legend(handles=label_lines, loc='upper right')
    plt.savefig(save_path)
    plt.close()

def pathway_viz(w1, w2, mask, pathways, save_path):
    
    rest_s_dict = {0:[1, 2], 1:[0, 2], 2:[0, 1]}
    legend_elements = [
        Patch(facecolor=colors[3], label='up-regulation'),
        Patch(facecolor=colors[2], label='down-regulation')
    ]
    for s in range(3):
        fig, ax = plt.subplots(figsize=(10, 8))
        window = 20
        s_weight = w2[:, s]
        y = np.arange(window)[::-1]

        rest_max = np.amax(w2[:, rest_s_dict[s]], axis=1)
        positive_delta = s_weight - rest_max
        sorted_ids = np.argsort(-positive_delta)[:window]
        sorted_pathways = [pathways[sorted_id] for sorted_id in sorted_ids]
        ax.barh(y, positive_delta[sorted_ids], height=0.8, color=colors[3])
        ax.set_xlabel('\u0394 weight')
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_pathways, fontsize=8)
        ax.set_title('S{:} top {:} contributing pathways'.format(s+1, window))
        ax.grid()
        ax.set_axisbelow(True)

        rest_min = np.amin(w2[:, rest_s_dict[s]], axis=1)
        negative_delta = s_weight - rest_min
        sorted_ids = np.argsort(negative_delta)[:window]
        sorted_pathways = [pathways[sorted_id] for sorted_id in sorted_ids]
        ax2 = ax.twinx() 
        ax2.barh(y[:window], negative_delta[sorted_ids], height=0.8, color=colors[2])
        ax2.set_yticks(y)
        ax2.set_yticklabels(sorted_pathways, fontsize=8)

        ax2.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        fig.savefig(join(save_path, 'S_{:}_pathway_weights.png'.format(s+1)))
        plt.close()

def heatmap_viz(data, gene_list, color_list, ax):
    data_dict = {}
    for i, gene in enumerate(gene_list):
        data_dict[gene] = data[:, i]
    df = pd.DataFrame(data=data_dict)
    ax = sns.clustermap(data, row_colors=color_list, row_cluster=False, ax=ax_list[i])
    return ax


if __name__ == '__main__':
    plot_learning_curvs()


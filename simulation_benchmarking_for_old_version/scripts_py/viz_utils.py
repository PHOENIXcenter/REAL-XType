import copy, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_utils as data_utils
import matplotlib.lines as mlines

from lifelines import statistics
from os.path import join
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
from matplotlib import cm
from scipy import stats
from sklearn.decomposition import PCA, KernelPCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from matplotlib_venn import venn2, venn3
from statsmodels.sandbox.stats.multicomp import multipletests

colors = ['tab:blue', 'orange', 'tab:red', 'tab:green', 'tab:purple', 
         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

colors2 = ['tab:blue', 'blue', 'tab:green', 'green', 'orange', 'lightcoral', 'indianred', 'tab:red', 'tab:purple', 
         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

lines = ['-', '--', ':']
markers = ['.', 'x', '^', '1', 's', '|']

def plot_km_curve_custom(times, events, assignments, k, ax, title, text_bias=33, additional_text='', clip_time=60, label_dict=None):
    if label_dict is None:
        label_dict = {}
        for i in range(k):
            label_dict[i] = 'S-'+''.join(['I']*(i+1))
    rmst_list = []
    for i in range(k):
        S_list, time_list, censor_list, at_risk_list, RMST = data_utils.get_km_curve(
            times[assignments==i], 
            events[assignments==i], 
            clip_time
        )
        rmst_list.append((RMST))
        ax.plot(time_list, S_list, c=colors[i], label=label_dict[i]+', N={:}'.format(np.sum(assignments==i)))
        ax.plot(
            np.asarray(time_list)[censor_list], np.asarray(S_list)[censor_list], 
            '|', c=colors[i], markeredgewidth=1, markersize=10
        )
    logrank_p = statistics.multivariate_logrank_test(times, assignments, events).p_value

    ax.set_title(title)
    p_string = 'Log-rank p\n= {:.5f}'.format(logrank_p) if logrank_p >= 1e-5 else 'Log-rank p\n< 0.00001'
    ax.text(text_bias, 0, p_string)
    ax.grid()
    ax.legend(loc='lower left')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, clip_time)
    return ax

def dataset_prognosis_viz(dataset, time_unit, clip_time, save_path):
    fig, ax_list = plt.subplots(2, 1, figsize=(6, 4))
    plot_km_curve_custom(
        dataset['OS'] * time_unit, dataset['status'], dataset['subtypes'], 2, ax_list[0], dataset['label'], 
        text_bias=33, additional_text='', clip_time=clip_time
    )
    plot_km_curve_custom(
        dataset['DFS'] * time_unit, dataset['recurrence'], dataset['subtypes'], 2, ax_list[1], dataset['label'], 
        text_bias=33, additional_text='', clip_time=clip_time
    )
    plt.tight_layout()
    plt.savefig(save_path)

def find_inliers(data, m=2):
    return np.prod(abs(data - np.mean(data, axis=0)) < m * np.std(data, axis=0), axis=1).astype(bool)

def data_2d_viz(x, subtypes, subtype_labels, domains, domain_label, ax):
    x_2d = PCA(n_components=2).fit_transform(x)
    is_inlier = find_inliers(x_2d, m=3)
    for domain, domain_label in enumerate(domain_label):
        for subtype, subtype_label in enumerate(subtype_labels):
            ax.scatter(
                x_2d[np.logical_and(np.logical_and(subtypes==subtype, domains==domain), is_inlier), 0], 
                x_2d[np.logical_and(np.logical_and(subtypes==subtype, domains==domain), is_inlier), 1],
                c=colors[subtype],
                marker=markers[domain],
                label=domain_label + '_' + subtype_label + ' n={:}'.format(
                    np.sum(np.logical_and(subtypes==subtype, domains==domain))
                )
            )
    ax.legend(framealpha=0.5)
    return ax

def datasets_viz(datasets, save_path, subtype_num):
    fig, ax = plt.subplots(figsize=(5, 5))
    data_2d_viz(
        np.concatenate([dataset['data'] for dataset in datasets], axis=0),
        np.concatenate([
                dataset['subtypes']
                if 'subtypes' in dataset.keys() else np.zeros_like(dataset['OS'], dtype=int) 
                for dataset in datasets
            ], 
            axis=0
        ),
        ['S-' + ''.join(['I'] * (i + 1)) for i in range(subtype_num)],
        np.concatenate([np.ones_like(dataset['OS'], dtype=int) * i for i, dataset in enumerate(datasets)], axis=0),
        [dataset['label'] for dataset in datasets],
        ax
    )
    plt.tight_layout()
    plt.savefig(save_path)

def data_3d_viz(x, subtypes, subtype_labels, domains, domain_label, ax):
    x_3d = PCA(n_components=3).fit_transform(x)
    is_inlier = find_inliers(x_3d, m=3)
    for domain, domain_label in enumerate(domain_label):
        for subtype, subtype_label in enumerate(subtype_labels):
            ax.scatter(
                x_3d[np.logical_and(np.logical_and(subtypes==subtype, domains==domain), is_inlier), 0], 
                x_3d[np.logical_and(np.logical_and(subtypes==subtype, domains==domain), is_inlier), 1],
                x_3d[np.logical_and(np.logical_and(subtypes==subtype, domains==domain), is_inlier), 2],
                c=colors[subtype],
                marker=markers[domain],
                label=domain_label + ' ' + subtype_label
                # ' n={:}'.format(
                #     np.sum(np.logical_and(subtypes==subtype, domains==domain))
                # )
            )
    ax.legend(framealpha=0.5)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    return ax

def datasets_3d_viz(datasets, save_path, subtype_num):
    fig = plt.figure(1, figsize=(5, 5))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=34, azim=156)
    data_3d_viz(
        np.concatenate([dataset['data'] for dataset in datasets], axis=0),
        np.concatenate([
                dataset['subtypes']
                if 'subtypes' in dataset.keys() else np.zeros_like(dataset['OS'], dtype=int) 
                for dataset in datasets
            ], 
            axis=0
        ),
        ['S-' + ''.join(['I'] * (i + 1)) for i in range(subtype_num)],
        np.concatenate([np.ones_like(dataset['OS'], dtype=int) * i for i, dataset in enumerate(datasets)], axis=0),
        # [dataset['label'] for dataset in datasets],
        ['Source', 'Target'],
        ax
    )
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

def data_img_viz(dataset, save_path):
    fig, ax = plt.subplots()
    ax.imshow(dataset['data'], interpolation='nearest', aspect='auto', cmap=cm.coolwarm)
    if 'subtypes' in dataset.keys():
        print(dataset['subtypes'])
    ax.set_title(dataset['label'])
    plt.tight_layout()
    plt.savefig(save_path)

def subtype_model_weight_viz(classifier, experiment_path, save_path, model_name, genes, seed_num, fold_num):
    w_list = []
    for seed in range(seed_num):
        for fold in range(fold_num):
            classifier.load_weights(join(experiment_path, 'seed'+str(seed), model_name, 'ckpt_fold'+str(fold)))
            w_list.append(classifier.layer_list[0].get_weights()[0])
    w = np.mean(np.stack(w_list, axis=-1), axis=-1)

    # plot important features
    subtype_num = w.shape[1]
    fig, ax_list = plt.subplots(subtype_num, 1, figsize=(4, 10))
    rest_s_dict = {0:[1, 2], 1:[0, 2], 2:[0, 1]} if subtype_num == 3 else {0: [1], 1: [0]}
    legend_elements = [
        Patch(facecolor=colors[3], label='positive'),
        Patch(facecolor=colors[2], label='negative')
    ]
    results = {'genes': genes}
    for s in range(subtype_num):
        window = 10
        s_w = w[:, s]
        x = np.arange(window)

        rest_max = np.amax(w[:, rest_s_dict[s]], axis=1)
        positive_delta = s_w - rest_max
        sorted_ids = np.argsort(-positive_delta)[:window]
        ax_list[s].bar(x, positive_delta[sorted_ids], width=0.8, color=colors[3])
        ax_list[s].set_ylabel('S{:} \u0394 weight'.format(s+1))
        ax_list[s].set_xticks(x)
        ax_list[s].set_xticklabels([genes[sorted_id] for sorted_id in sorted_ids], rotation=0)
        ax_list[s].grid()
        ax_list[s].set_axisbelow(True)

        rest_min = np.amin(w[:, rest_s_dict[s]], axis=1)
        negative_delta = s_w - rest_min
        sorted_ids = np.argsort(negative_delta)[:window]
        ax2 = ax_list[s].twiny() 
        ax2.bar(x[:window], negative_delta[sorted_ids], width=0.8, color=colors[2])
        ax2.set_xticks(x)
        ax2.set_xticklabels([genes[sorted_id] for sorted_id in sorted_ids], rotation=0)
        if s == subtype_num - 1:
            ax2.legend(handles=legend_elements, loc='lower right')

        rest_mean = np.mean(w[:, rest_s_dict[s]], axis=1)
        mean_delta = s_w - rest_mean
        results['S'+str(s+1)+'_posi_delta'] = positive_delta
        results['S'+str(s+1)+'_mean_delta'] = mean_delta
        results['S'+str(s+1)+'_val'] = s_w
    plt.tight_layout()
    # fig.savefig(join(save_path, 'weights.png'))
    plt.close()

    # top 30
    fig, ax = plt.subplots(figsize=(10, 3))
    s3_mean_delta = results['S3_mean_delta']
    window = 30
    sorted_ids = np.argsort(-s3_mean_delta)[:window]
    x = np.arange(window)
    # HCC_related = [
    #     'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 
    #     'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'N', 
    #     'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y'
    # ]
    HCC_related = [
        'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 
        'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 
        'Y', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y'
    ]
    HCC_related = np.asarray([False if elem == 'N' else True for elem in HCC_related])
    ax.bar(x[HCC_related], s3_mean_delta[sorted_ids][HCC_related], 
        color=colors[2], label='HCC Related'
    )
    ax.bar(x[~HCC_related], s3_mean_delta[sorted_ids][~HCC_related], 
        color=colors[0], label='Unknown'
    )
    ax.set_xticks(x)
    ax.set_xticklabels([genes[sorted_id] for sorted_id in sorted_ids], rotation=45)
    ax.set_ylabel('\u0394weight')
    ax.grid()
    ax.legend()
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(join(save_path, 'S3_weights_top30.png'))
    data_utils.write_protes([genes[sorted_id] for sorted_id in sorted_ids], join(save_path, 'S3_HWP.txt'))
    
    # top 10
    fig, ax = plt.subplots(figsize=(3, 3))
    window = 10
    sorted_ids = np.argsort(-s3_mean_delta)[:window]
    x = np.arange(window)
    HCC_related = HCC_related[:window]
    ax.bar(x[HCC_related], s3_mean_delta[sorted_ids][HCC_related], 
        color=colors[2], label='HCC Related'
    )
    ax.bar(x[~HCC_related], s3_mean_delta[sorted_ids][~HCC_related], 
        color=colors[0], label='Unknown'
    )
    ax.set_xticks(x)
    ax.set_xticklabels([genes[sorted_id] for sorted_id in sorted_ids], rotation=60)
    ax.set_ylim([0.06, 0.12])
    ax.set_ylabel('\u0394weight')
    ax.grid()
    ax.legend()
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(join(save_path, 'S3_weights_top10.png'))

    return results

def compare_correlation_between_delta_weight_and_hr(models, experiment_paths, model_names, datasets, save_path):
    # get cox results
    cox_results = data_utils.cox_check(datasets, save_path)

    # get delta_weight
    for model, model_path, model_name in zip(models, experiment_paths, model_names):
        delta_weight = get_delta_weight(model, model_path, model_name)
        for dataset in datasets:
            for event_time, event_status in zip(['OS', 'DFS'], ['status', 'recurrence']):
                times = dataset[event_time]
                status = dataset[event_status]
                [coef, hr, p] = cox_results[dataset['label'] + '_' + event_time]
                slope, intercept, r, p, stderr = stats.linregress(delta_weight, coef)
                print(dataset['label'], event_time, r, p)
                name = model_name.split('_')[0]
                viz_correlation(
                    delta_weight,
                    coef,
                    '\u0394weight',
                    'COX weight',
                    None,
                    join(save_path, 'corr_weight_COX' + name + '_' + dataset['label'] + '_' + event_time + '.png'),
                )


def get_delta_weight(model, experiment_path, model_name, seed_num=5, fold_num=5):
    w_list = []
    for seed in range(seed_num):
        for fold in range(fold_num):
            model.load_weights(join(experiment_path, 'seed'+str(seed), model_name, 'ckpt_fold'+str(fold)))
            w_list.append(model.layer_list[0].get_weights()[0])
    w = np.mean(np.stack(w_list, axis=-1), axis=-1)

    subtype_num = w.shape[1]
    rest_s_dict = {0:[1, 2], 1:[0, 2], 2:[0, 1]} if subtype_num == 3 else {0: [1], 1: [0]}
    s_w = w[:, 2]
    rest_mean = np.mean(w[:, rest_s_dict[2]], axis=1)
    mean_delta = s_w - rest_mean
    return mean_delta
    

def DEP_viz(experiment_path, save_path, model_name, datasets, genes, seed_num, fold_num):
    results = {'genes': genes}
    for dataset in datasets:
        # get assignments
        onehot_assignments = None
        all_patients = dataset['patients']
        all_data = dataset['data']
        for seed in range(seed_num):
            df = pd.concat([pd.read_csv(join(experiment_path, 'seed'+str(seed), model_name, 'fold-' + str(fold) + '.csv')) for fold in range(fold_num)])
            patients = df['patients'].to_list()
            ids = [patients.index(patient) for patient in all_patients]
            assignments = df['assignment'].to_numpy()[ids]
            onehot_assi = np.zeros((assignments.size, assignments.max()+1))
            onehot_assi[np.arange(len(onehot_assi)), assignments] = 1
            onehot_assignments = onehot_assi if onehot_assignments is None else onehot_assignments + onehot_assi
        assignments = np.argmax(onehot_assignments, axis=-1)

        for subtype in range(3):
            # fold change
            fold_change = np.mean(all_data[assignments==subtype, :], axis=0) / np.mean(all_data[assignments!=subtype, :], axis=0)
            log2_fold_change = np.log2(fold_change)
            # statistic test
            pvalue = stats.ttest_ind(
                all_data[assignments==subtype], 
                all_data[assignments!=subtype],
                nan_policy="omit",
                equal_var=False
            ).pvalue.data
            pvalue_adj = multipletests(pvalue, method='fdr_bh', is_sorted=False, returnsorted=False)[1]
            
            # plot 
            groups = feat_prote_scatter_viz(
                log2_fold_change,
                pvalue_adj,
                fc_thresh=1,
                pval_thresh=0.01,
                protes=genes,
                fig_size=(3, 3),
                save_path=join(save_path, 'DEP_'+dataset['label']+'_S'+str(subtype+1)+'.png')
            )

            results[dataset['label']+'_S'+str(subtype+1)+'_logFC'] = log2_fold_change
            results[dataset['label']+'_S'+str(subtype+1)+'_nlog_pval'] = -np.log10(pvalue_adj)
            results[dataset['label']+'_S'+str(subtype+1)+'_DEP'] = set([gene for i, gene in enumerate(genes) if groups[i] == 2])

    for subtype in range(3):
        venn_viz(
            [results[dataset['label']+'_S'+str(subtype+1)+'_DEP'] for dataset in datasets], 
            labels=[dataset['label'] for dataset in datasets], 
            save_path=join(save_path, 'DEP_S'+str(subtype+1)+'_venn.png'),
            title=None
        )

    s3_DEP = list(
        results[datasets[0]['label']+'_S3_DEP'].intersection(results[datasets[1]['label']+'_S3_DEP'])
    )
    data_utils.write_protes(s3_DEP, join(save_path, 'S3_DEP.txt'))
    return results

def feat_prote_scatter_viz(log2_fc, adj_pval, fc_thresh, pval_thresh, protes, fig_size, save_path, mark_protes=[]):
    nlog_adj_pval = -np.log10(adj_pval)
    nlog_pval_thresh = -np.log10(pval_thresh)
    xmin = int(np.around(np.amin(log2_fc))) - 1
    xmax = int(np.around(np.amax(log2_fc))) + 1
    ymin = int(np.around(np.amin(nlog_adj_pval))) - 1
    ymax = int(np.around(np.amax(nlog_adj_pval))) + 3

    data = np.c_[log2_fc, nlog_adj_pval]
    groups = np.zeros_like(log2_fc)
    groups[np.logical_and(log2_fc < -fc_thresh, nlog_adj_pval > nlog_pval_thresh)] = 1
    groups[np.logical_and(log2_fc > fc_thresh, nlog_adj_pval > nlog_pval_thresh)] = 2
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), title='')
    pt_colors = ['grey', 'tab:blue', 'tab:red']
    pt_labels = ['normal', 'down', 'up']
    for group in range(3):
        ax.scatter(
            data[groups==group, 0], 
            data[groups==group, 1], 
            s=3, 
            c=pt_colors[group],
            label=pt_labels[group])
    ax.set_ylabel('-Log10(Q value)')
    ax.set_xlabel('Log2 (fold change)')
    ax.legend()

    # # annotation
    # top_dep_ids = np.argsort(-np.fabs(log2_fc))[:10]
    # top_deps = [protes[idx] for idx in top_dep_ids]
    # for idx, prote in zip(top_dep_ids, top_deps):
    #     ax.text(data[idx, 0], data[idx, 1], prote)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plot dashed lines
    ax.vlines(-fc_thresh, ymin, ymax, color='dimgrey',linestyle='dashed', linewidth=1)
    ax.vlines(fc_thresh, ymin, ymax, color='dimgrey',linestyle='dashed', linewidth=1)
    ax.hlines(nlog_pval_thresh, xmin, xmax, color='dimgrey',linestyle='dashed', linewidth=1)

    ax.set_xticks(range(xmin, xmax, int((xmax - xmin)/5)))
    ax.set_yticks(range(ymin, ymax, int((ymax - ymin)/5)))

    plt.tight_layout()
    plt.savefig(save_path)

    return groups

def prote_viz(datasets, seed_num, fold_num, experiment_path, model_name, gene, prote, save_path):
    for i, dataset in enumerate(datasets):
        # get assignments
        onehot_assignments = None
        all_patients = dataset['patients']
        all_data = dataset['data']
        for seed in range(seed_num):
            df = pd.concat([pd.read_csv(join(
                experiment_path, 'seed'+str(seed), model_name, 'fold-' + str(fold) + '.csv')) for fold in range(fold_num)]
            )
            patients = df['patients'].to_list()
            ids = [patients.index(patient) for patient in all_patients]
            assignments = df['assignment'].to_numpy()[ids]
            onehot_assi = np.zeros((assignments.size, assignments.max()+1))
            onehot_assi[np.arange(len(onehot_assi)), assignments] = 1
            onehot_assignments = onehot_assi if onehot_assignments is None else onehot_assignments + onehot_assi
        assignments = np.argmax(onehot_assignments, axis=-1)

        sorted_ids = np.argsort(assignments)
        
        # expression viz
        protes = dataset['protes']
        data = dataset['data']
        data = data[:, protes.index(prote)]
        assi_sorted_data = data[sorted_ids]
        assi_sorted_assignments = assignments[sorted_ids]
        fig, ax = plt.subplots(figsize=(4, 2))
        N = 0
        for s in range(3):
            n = np.sum(assi_sorted_assignments==s)
            ax.bar(
                np.arange(N, N + n), 
                np.sort(assi_sorted_data[assi_sorted_assignments==s]), 
                color=colors[s],
                label='S-'+''.join(['I']*(s+1))
            )
            N += n
        ax.legend()
        plt.tight_layout()
        plt.savefig(join(save_path, 'express_viz_'+gene+'_'+dataset['label']+'.png'))

        # prognosis discrimination viz
        OS = dataset['OS']
        status = dataset['status']
        DFS = dataset['DFS']
        recurrence = dataset['recurrence']
        sorted_data = np.sort(data)
        best_OS_p = 1.
        best_DFS_p = 1.
        best_max_p = 1.
        for thresh_id in range(int(0.1 * len(data)), int(0.9 * len(data))):
        # for thresh_id in [int(0.9 * len(data))]:
            thresh = sorted_data[thresh_id]
            assignments = np.zeros_like(status)
            assignments[data > thresh] = 1

            dRMST_OS = data_utils.get_delta_rmst(OS, status, assignments, 2)[0]
            logrank_OS_p = data_utils.get_multivariate_logrank_p(OS, status, assignments, 2)
            dRMST_DFS = data_utils.get_delta_rmst(DFS, recurrence, assignments, 2)[0]
            logrank_DFS_p = data_utils.get_multivariate_logrank_p(DFS, recurrence, assignments, 2)
            if logrank_OS_p < best_OS_p:
                best_OS_thresh = thresh
                best_OS_assignments = assignments
                best_OS_p = logrank_OS_p
            if logrank_DFS_p < best_DFS_p:
                best_DFS_thresh = thresh
                best_DFS_assignments = assignments
                best_DFS_p = logrank_DFS_p
            # if max(logrank_OS_p, logrank_DFS_p) < best_max_p:
            #     best_thresh = thresh
            #     best_OS_assignments = assignments
            #     best_DFS_assignments = assignments
            #     best_max_p = max(logrank_OS_p, logrank_DFS_p)

        fig, ax = plt.subplots(figsize=(3, 3))
        label_dict = {0: 'low', 1: 'high'}
        plot_km_curve_custom(
            OS, status, best_OS_assignments, 2, ax, None, text_bias=38, label_dict=label_dict) 
        fig.tight_layout()
        plt.savefig(join(save_path, 'km_' + gene +'_OS_' + dataset['label'] + '.png'))
        plt.close()

        fig, ax = plt.subplots(figsize=(3, 3))
        label_dict = {0: 'low', 1: 'high'}
        plot_km_curve_custom(
            DFS, recurrence, best_DFS_assignments, 2, ax, None, text_bias=38, label_dict=label_dict) 
        fig.tight_layout()
        plt.savefig(join(save_path, 'km_' + gene +'_DFS_' + dataset['label'] + '.png'))
        plt.close()


def plot_results(result_path, datasets, classifier, fold_num, subtype_num, max_t):
    df = pd.concat([pd.read_csv(join(result_path, 'fold-' + str(fold) + '.csv')) for fold in range(fold_num)])

    w, h = 4, 4 
    for dataset_id, dataset in enumerate(datasets):
        dataset_label = dataset['label']
        dataset_df = df.loc[df['cohort'] == dataset_id]
        assignments = dataset_df['assignment'].to_numpy()

        # acc
        if dataset_id == 0:
            labels = dataset_df['label'].to_numpy()
            acc = np.mean(assignments == labels)

        for time_name, event_name in zip(['OS', 'DFS'], ['status', 'recurrence']):
            times = dataset_df[time_name].to_numpy()
            events = dataset_df[event_name].to_numpy()

            fig, ax = plt.subplots(figsize=(w, h))
            title = dataset_label + '_' + time_name 
            plot_km_curve_custom(
                times, 
                events, 
                assignments, 
                subtype_num,
                ax, 
                title=title + '_acc{:.2f}'.format(acc) if dataset_id == 0  else title,
                text_bias=30,
                clip_time=max_t
            )
            fig.tight_layout()
            plt.savefig(join(result_path, 'km_' + title + '.png'))
            plt.close()

        # tsne visualization
        all_data = np.concatenate([dataset['data'] for dataset in datasets])
        all_patients = []
        for dataset in datasets:
            all_patients += dataset['patients']
        patient_ids = [df['patients'].to_list().index(patient) for patient in all_patients]
        assignments = df['assignment'].to_numpy()[patient_ids]
        cohorts = df['cohort'].to_numpy()[patient_ids]

        fig, ax = plt.subplots(figsize=(w, h))
        tsne_viz(
            all_data, 
            assignments,
            ['S'+str(s+1) for s in range(subtype_num)],
            cohorts,
            [dataset['label'] for dataset in datasets],
            ax
        )
        fig.tight_layout()
        plt.savefig(join(result_path, 'tsne.png'))
        plt.close()

    # classifer weight viz
    genes = datasets[0]['genes']
    if classifier is not None:
        subtype_feature_gene_viz(classifier, result_path, genes, fold_num)

def plot_annotated_heatmap(data_arr, x_labels, y_labels, rotation, title, save_path, fig_size=(5, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    data_arr = (data_arr * 100).astype(int)/100
    im = ax.imshow(data_arr, interpolation='nearest', aspect='auto', cmap=cm.coolwarm)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, data_arr[i, j], ha="center", va="center", color="k")

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def heatmap_vectors_with_stats(data_arr, x_labels, y_labels, rotation, fig_size, save_path, show_stats=False):
    # data_arr: methods * metrics * samples
    data_mean = np.mean(data_arr, axis=-1)
    data_std = np.std(data_arr, axis=-1)
    fig, ax_list = plt.subplots(1, len(x_labels), figsize=fig_size)
    # for each metric/col
    for i, ax in enumerate(ax_list): 
        im = ax.imshow(np.expand_dims(data_mean[:, i], axis=1), interpolation='nearest', aspect='auto', cmap=cm.Blues)
        if i == 0:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels, rotation=rotation)
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel(x_labels[i], rotation=0, ha="center", va="top")

        for j in range(len(y_labels)):
            # text annotation
            normalized_vector = (data_mean[:, i] - np.amin(data_mean[:, i])) / (np.amax(data_mean[:, i]) - np.amin(data_mean[:, i]))
            text = ax.text(0, j, 
                ('{:.3f}\n±{:.3f}').format(data_mean[j, i], data_std[j, i]), 
                ha="center", 
                va="center", 
                color=np.zeros(3) if normalized_vector[j] < 0.7 else np.ones(3),
                size=10
            )
            # print(y_labels[i], x_labels[j], data_arr[j, i, :])
            # if show_stats:
            #     # p value
            #     if j < len(x_labels) - 1:
            #         try:
            #             p_values = stats.ttest_ind(data_arr[j, i, :], data_arr[-1, i, :])[1]
            #         except Exception as e:
            #             p_values = 1.

            #         line_x = np.array([j, len(x_labels) - 1])
            #         line_y = -np.ones(2) * (0.6 + 0.4 * j)
            #         ax.plot(line_x, line_y, c='k')
            #         p_str = 'P={:.4f}'.format(p_values)
            #         ax.text(
            #             np.mean(line_x), 
            #             line_y[0] - 0.15, 
            #             p_str,
            #             ha="center", 
            #             va="center", 
            #             size=8
            #         )

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_grid(ax, x_range, y_range, line_width, bias, x_on, y_on):
    if x_on:
        for x in range(x_range[0], x_range[1] + 1):
            ax.plot([x + bias, x + bias], [y_range[0] + bias,  y_range[1] + bias], '-k', linewidth=4)
    if y_on:
        for y in range(y_range[0], y_range[1] + 1):
            ax.plot([x_range[0] + bias, x_range[1] + bias], [y + bias, y + bias], '-k', linewidth=4)
    return ax

def check_feat_histogram(datasets, save_path):
    result_path = join(save_path, 'feat_hist')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    feats = datasets[0]['genes']
    raw_data_list = [dataset['data'] for dataset in datasets]
    data_list = []
    for raw_data in raw_data_list:
        data = copy.deepcopy(raw_data)
        data[np.isnan(data)] = 0
        data = (data - np.mean(data, axis=0))/(np.std(data, axis=0)+1e-5)
        data_list.append(data)

    for i, feat in enumerate(feats[:20]):
        fig, axs = plt.subplots(len(datasets), 3, figsize=(8, 6))
        for j, data_list_j in enumerate([raw_data_list, data_list]):
            axs[j, 0].hist(
                [data[:, i] for data in data_list_j],
                10, density=False, histtype='bar', color=colors[:len(data_list_j)], label=[dataset['label'] for dataset in datasets]
            )
            axs[j, 0].set_title(feats[i])
            axs[j, 0].legend(framealpha=0.5)

            axs[j, 1].hist(data_list_j[0][:, i], 10, density=False, histtype='bar', color=colors[0])
            axs[j, 1].set_title(datasets[0]['label'])

            axs[j, 2].hist(data_list_j[1][:, i], 10, density=False, histtype='bar', color=colors[1])
            axs[j, 2].set_title(datasets[1]['label'])

        fig.tight_layout()
        plt.savefig(join(result_path, feat + '.png'))

def grid_search_viz(datasets, datasets_events_p_scores, thresh_vals, gene_name, range_str, result_path):
    fig, ax_list = plt.subplots(2, figsize=(3, 6))
    event_list = ['DFS', 'OS']
    for j, ax in enumerate(ax_list):
        for i, dataset in enumerate(datasets):
            scores = datasets_events_p_scores[i, j]
            ax.plot(thresh_vals, scores, label=dataset['label'])
        if j == 0:
            ax.set_title(gene_name + range_str)
        else:
            ax.set_xlabel('Threshold') 
        ax.set_ylabel('Log-rank score ({})'.format(event_list[j])) 
        ax.set_ylim(-3, 3)          
        ax.legend()
        ax.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(result_path, gene_name + '_grid_search.png'))
    plt.close()

def prote_expression_in_subtypes_viz(datasets, prote, intersected_protes, gene_name, result_path):
    prote_id = intersected_protes.index(prote)
    fig, ax_list = plt.subplots(len(datasets), figsize=(4, len(datasets)*2))
    for i, (ax, dataset) in enumerate(zip(ax_list, datasets)):
        subtypes = dataset['subtypes']
        prote_vals = dataset['data'][:, prote_id]
        start = 0
        for subtype in range(3):
            sorted_vals = np.sort(prote_vals[subtypes == subtype])
            ax.bar(
                np.arange(start, start + len(sorted_vals)), 
                sorted_vals, 
                color=colors[subtype], 
                label='S-' + ''.join(['I'] * (subtype + 1))
            )
            start += len(sorted_vals)
        title = gene_name + '\n' + dataset['label'] if i == 0 else dataset['label']
        ax.set_title(title)
        ax.set_ylim(-3, 3) 
        ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(result_path, gene_name + '_express_in_subtypes.png'))
    plt.close()

def viz_correlation(x, y, xlabel, ylabel, title, save_path, top_prote=False, robust=False):
    fig, ax = plt.subplots(figsize=(3, 3))

    slope, intercept, r, p, stderr = stats.linregress(x, y)
    # line1 = f'y={intercept:.2f}+{slope:.2f}x\n R={r:.2f}, p={p:.4f}' 
    # line2 = f'y={intercept:.2f}+{slope:.2f}x\n R={r:.2f}, p<0.0001'
    line1 = f'R={r:.2f}, p={p:.4f}' 
    line2 = f'R={r:.2f}, p<0.0001'

    if robust:
        def outlier_rejection(vals, window=3):
            window = 2
            x_sorted = np.sort(vals)
            thresh = vals[-window]
            mask = vals < thresh
            vals = vals[mask]
            return vals, mask

        x, mask = outlier_rejection(x)
        y = y[mask]

    ax.plot(x, y, linewidth=0, marker='.')
    top_30_ids = np.argsort(-x)[:30]

    if top_prote:
        ax.plot(x[top_30_ids], y[top_30_ids], linewidth=0, marker='.', color='red', label='top 30 proteins')
    ax.plot(x, intercept + slope * x, label=line1 if p >= 0.0001 else line2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return ax

def cox_check(datasets, save_path):
    result_path = join(save_path, 'corr')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    df_dict = data_utils.cox_check(datasets, save_path)

    # visualize correlation between HR_OS and HR_DFS
    for i, dataset in enumerate(datasets):
        viz_correlation(
            df_dict[dataset['label'] + '_OS'][1], 
            df_dict[dataset['label'] + '_DFS'][1], 
            'Hazard ratio in OS', 'Hazard ratio in RFS', dataset['label'] + ' et al.',
            join(result_path, 'corr_' + dataset['label'] + '_OS_DFS.png'),
            robust=True
        )

    # if len(datasets) == 2:
    #     viz_correlation(
    #         df_dict[datasets[0]['label'] + '_HR_OS'], 
    #         df_dict[datasets[1]['label'] + '_HR_OS'], 
    #         datasets[0]['label'] + '_HR_OS', 
    #         datasets[1]['label'] + '_HR_OS', 
    #         'HR_OS',
    #         join(result_path, 'corr_OS_' + datasets[0]['label'] + '_' + datasets[1]['label'] + '.png')
    #     )
    #     viz_correlation(
    #         df_dict[datasets[0]['label'] + '_HR_DFS'], 
    #         df_dict[datasets[1]['label'] + '_HR_DFS'], 
    #         datasets[0]['label'] + '_HR_DFS', 
    #         datasets[1]['label'] + '_HR_DFS', 
    #         'HR_DFS',
    #         join(result_path, 'corr_DFS_' + datasets[0]['label'] + '_' + datasets[1]['label'] + '.png')
    #     )

def batch_effect_bar_plots(lv_method_metric_seed, methods, metric, lvs, fig_size, ylimit_list, save_path):
    lv_num, method_num, metric_num, seed_num = lv_method_metric_seed.shape
    # metric_method_lv_seed = np.transpose(lv_method_metric_seed, [2, 1, 0, 3])
    for i, lv in enumerate(lvs):
        method_metric_mean = np.mean(lv_method_metric_seed[i], axis=-1)
        method_metric_std = np.std(lv_method_metric_seed[i], axis=-1)
        fig, ax = plt.subplots(figsize=fig_size)
        width = 0.9/len(methods)
        for j, method in enumerate(methods):
            bias = (j - method_num/2. + 0.5) * width
            ax.bar(
                np.arange(metric_num) + bias,
                method_metric_mean[j],
                width,
                yerr=method_metric_std[j],
                ecolor='black', 
                capsize=3,
                label=method,
                color=colors2[j],
                zorder=3
            )
        # ax.set_xlabel('Batch effect level')
        ax.set_ylabel(metric)
        if ylimit_list[i] is not None:
            ax.set_ylim(ylimit_list[i])
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(['Source cohort', 'Target cohort'])
        ax.grid(zorder=0)
        # if 'Log-rank' in metric:
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=2)
        plt.tight_layout()
        plt.savefig(join(save_path, 'synthetic_' + metric + '_' + lv + '.png'))
        plt.close()

def mesh_plot(X, Y, data_arr):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(0, 11) * 0.2
    Y = np.arange(0, 11) * 0.05
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(
        X, Y, data_arr, cmap=cm.coolwarm,
        linewidth=0, antialiased=False
    )

    # Customize the z axis.
    ax.set_zlim(-0.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.set_ylabel('OS swap rate')
    ax.set_xlabel('batch noise variance')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def imshow(image, size, title, save_path, vmax=None, vmin=None):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(
        image, 
        interpolation='nearest', 
        aspect='auto', 
        cmap=cm.coolwarm, 
        vmax=vmax, 
        vmin=vmin
    )
    fig.colorbar(im, ax=ax)
    ax.set_ylabel('survival correlation')
    ax.set_yticks(np.arange(11))
    ax.set_yticklabels(['{:.1f}'.format(1 - num * 0.1) if num % 2 == 0 else None for num in np.arange(11)])
    ax.set_xlabel('batch effect')
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels(['{:.1f}'.format(num * 0.1) if num % 2 == 0 else None for num in np.arange(11)])
    ax.set_title(title ,fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    return ax

def venn_viz(sets, labels, save_path, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    if len(sets) == 2:
        venn2(sets, labels, ax=ax)
    elif len(sets) == 3:
        venn3(sets, labels, ax=ax)
    ax.set_title(title)
    # plt.tight_layout()
    plt.savefig(save_path)

if __name__ == '__main__':
    # test_synthetic_datasets_with_LR(r'G:\Data\SRPS\synthetic_data')
    data1 = set(['P{:}'.format(x) for x in np.concatenate([np.arange(10), np.arange(15, 20)])])
    data2 = set(['P{:}'.format(x) for x in np.concatenate([np.arange(5), np.arange(20, 25)])])
    data3 = set(['P{:}'.format(x) for x in np.concatenate([np.arange(3), np.arange(25, 40)])])

    venn_viz(
        [data1, data2, data3], 
        ['data1', 'data2', 'data3'],
        r'D:\Data\SRPS\test\venn.png',
        None
    )
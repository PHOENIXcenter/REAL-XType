import re, copy, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_utils as utils
from os.path import join
from lifelines import KaplanMeierFitter, statistics, plotting
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from sksurv.metrics import cumulative_dynamic_auc

colors = ['tab:blue', 'orange', 'tab:red', 'tab:green', 'tab:purple', 
         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
colors2 = ['tab:blue', 'blue', 'tab:brown', 'tab:purple', 'indianred', 'tab:red', ]

lines = ['-', '--', ':']
markers = ['.', 'x', '^']

def plot_km_curve_custom(
        times, 
        events, 
        assignments, 
        k, 
        ax, 
        title, 
        text_bias=33, 
        additional_text='', 
        clip_time=60, 
        label_dict=None, 
        probs=None
    ):
    if label_dict is None:
        label_dict = {}
        for i in range(k):
            label_dict[i] = 'S-'+''.join(['I']*(i+1))
    rmst_list = []
    for i in range(k):
        S_list, time_list, censor_list, at_risk_list, RMST = utils.get_km_curve(
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
    ax.set_title(title.replace('DFS', 'RFS'))
    p_string = 'Log-rank p={:.5f}'.format(logrank_p) if logrank_p >= 1e-5 else 'Log-rank p<0.00001'
    if probs is not None:
        y = np.array(
            [(bool(a), b) for a, b in zip(events, times)],
            dtype=[('status', 'bool'), ('time', 'float')]
        )
        s1_auc, mean_s1_auc = cumulative_dynamic_auc(y, y, 1-probs[:, 0], [12, 24, 36, 48])
        s3_auc, mean_s3_auc = cumulative_dynamic_auc(y, y, probs[:, 2], [12, 24, 36, 48])
        p_string += '\nAUC(S-I)={:.3f}'.format(mean_s1_auc)
        p_string += '\nAUC(S-III)={:.3f}'.format(mean_s3_auc)
    ax.text(text_bias, 0, p_string)
    ax.grid()
    ax.legend(loc='lower left')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, clip_time)
    ax.set_xticks(np.arange(0, clip_time+1, 12))
    return ax        

def barplot_benchmark(cohorts_methods_dict, fig_size, save_path):
    cohorts = list(cohorts_methods_dict.keys())
    metrics = cohorts_methods_dict[cohorts[0]].columns.values
    metrics = [' '.join(metric.split('_')[2:]) for metric in metrics]
    methods = cohorts_methods_dict[cohorts[0]].index.values
    cohorts_methods_metrics_avg = np.zeros((len(cohorts), len(methods), len(metrics)))
    cohorts_methods_metrics_std= np.zeros((len(cohorts), len(methods), len(metrics)))

    for i, cohort in enumerate(cohorts):
        methods_metrics_df = cohorts_methods_dict[cohort]
        avg_map = lambda x: float(x.split('±')[0])
        std_map = lambda x: float(x.split('±')[1])
        cohorts_methods_metrics_avg[i] = methods_metrics_df.applymap(avg_map).to_numpy()
        cohorts_methods_metrics_std[i] = methods_metrics_df.applymap(std_map).to_numpy()

    # extend for an overall cohort
    cohorts += ['Overall']
    overall_methods_metrics_avg = np.mean(cohorts_methods_metrics_avg, axis=0)
    overall_methods_metrics_std = np.sqrt(np.mean(np.square(cohorts_methods_metrics_std), axis=0))
    cohorts_methods_metrics_avg = np.concatenate([cohorts_methods_metrics_avg, np.expand_dims(overall_methods_metrics_avg, axis=0)], axis=0)
    cohorts_methods_metrics_std = np.concatenate([cohorts_methods_metrics_std, np.expand_dims(overall_methods_metrics_std, axis=0)], axis=0)
    
    # transpose
    metrics_cohorts_methods_avg = cohorts_methods_metrics_avg.transpose((2, 0, 1))
    metrics_cohorts_methods_std = cohorts_methods_metrics_std.transpose((2, 0, 1))

    ylimit_dict = {
        'Accuracy': [0., 1.],
        'Log-rank score': [0., 10.],
        'dRMST': [-4., 12.],
        'C-Index': [0.4, 0.8]
    }
    for i, metric in enumerate(metrics):
        cohorts_methods_avg = metrics_cohorts_methods_avg[i]
        cohorts_methods_std = metrics_cohorts_methods_std[i]
        fig, ax = plt.subplots(figsize=fig_size)
        width = 0.8/len(methods)
        for j, method in enumerate(methods):
            bias = (j - len(methods)/2. + 0.5) * width
            ax.bar(
                np.arange(len(cohorts)) + bias,
                cohorts_methods_avg[:, j],
                width,
                yerr=cohorts_methods_std[:, j],
                ecolor='black', 
                capsize=3,
                label=method,
                color=colors2[j],
                zorder=3
            )
        metric = metric.replace('dRMST', 'ΔRMST') if 'dRMST' in metric else metric
        metric = metric.replace('DFS', 'RFS') if 'DFS' in metric else metric
        ax.set_ylabel(metric)
        ax.set_ylim([ylimit_dict[key] for key in ylimit_dict.keys() if key in metric][0])
        ax.set_xticks(np.arange(len(cohorts)))
        ax.set_xticklabels(cohorts)
        ax.grid(zorder=0)
        ax.legend(loc="best", ncol=2)
        plt.tight_layout()
        plt.savefig(join(save_path, 'benchmark_' + metric + '.png'))
        plt.close()

def barplot_jiang_acc(df, fig_size, save_path):
    methods = df.index.values
    avg_map = lambda x: float(x.split('±')[0])
    std_map = lambda x: float(x.split('±')[1])
    acc_avg = df['valid_Jiang_Accuracy'].apply(avg_map).to_numpy()
    acc_std = df['valid_Jiang_Accuracy'].apply(std_map).to_numpy()
    fig, ax = plt.subplots(figsize=fig_size)
    ax.bar(
        np.arange(len(methods)),
        acc_avg,
        0.5,
        yerr=acc_std,
        ecolor='black', 
        capsize=3,
        color=colors2[:len(methods)],
        zorder=3
    )
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.5, 1.1])
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=15) 
    ax.grid(zorder=0)
    plt.tight_layout()
    plt.savefig(join(save_path, 'benchmark_acc.png'))
    plt.close()


if __name__ == '__main__':
    pass


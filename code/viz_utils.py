import re, copy, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_utils as utils
from os.path import join
from lifelines import statistics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy import stats

colors = ['tab:blue', 'orange', 'tab:red', 'tab:green', 'tab:purple', 
         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
colors_2subtypes = ['#919160', 'tab:red']

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
        clip_time=84, 
        label_dict=None, 
        legend_pos='lower left',
    ):
    if label_dict is None:
        label_dict = {}
        for i in range(k):
            label_dict[i] = 'R-'+''.join(['I']*(i+1))
    rmst_list = []
    if k == 3:
        color = colors
    elif k == 2:
        color = colors_2subtypes
    else:
        assert False, 'k must be 2 or 3'
    for i in range(k):
        S_list, time_list, censor_list, at_risk_list, RMST = utils.get_km_curve(
            times[assignments==i], 
            events[assignments==i], 
            clip_time
        )
        if 'Recurrence' in title or 'recurrence' in title:
             S_list = [1. - S for S in S_list]
        rmst_list.append((RMST))
        ax.plot(time_list, S_list, c=color[i], label=label_dict[i]+', N={:}'.format(np.sum(assignments==i)))
        ax.plot(
            np.asarray(time_list)[censor_list], np.asarray(S_list)[censor_list], 
            '|', c=color[i], markeredgewidth=1, markersize=10
        )
    logrank_p = statistics.multivariate_logrank_test(times, assignments, events).p_value
    ax.set_title(title.replace('DFS', 'RFS'))
    p_string = 'Log-rank p={:.3f}'.format(logrank_p) if logrank_p >= 1e-3 else 'Log-rank p<0.001'
    text_y = 0

    if 'DFS' in title:
        times_2y = copy.deepcopy(times)
        events_2y = copy.deepcopy(events)
        events_2y[times_2y > 24] = 0
        times_2y[times_2y > 24] = 24
        logrank_p = statistics.multivariate_logrank_test(times_2y, assignments, events_2y).p_value
        p_string += '\n2-year p ={:.3f}'.format(logrank_p) if logrank_p >= 1e-3 else '\n2-year p<0.001'

    ax.text(text_bias, text_y, p_string)
    ax.grid()
    ax.legend(loc=legend_pos)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, clip_time)
    ax.set_xticks(np.arange(0, clip_time+1, 12))
    return ax, rmst_list[0] - rmst_list[-1]     

def barplot_benchmark(df, fig_size, save_path, cohorts_metrics_methods_pval=None):
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
        'Log-rank score': [0., 11.],
        'dRMST': [-4., 12.],
        'C-Index': [0.5, 0.75]
    }
    for i, metric in enumerate(metrics):
        cohorts_methods_avg = metrics_cohorts_methods_avg[i]
        cohorts_methods_std = metrics_cohorts_methods_std[i]
        fig, ax = plt.subplots(figsize=fig_size)
        width = 0.9/len(methods)
        for j, method in enumerate(methods):
            bias = (j - len(methods)/2. + 0.5) * width
            x = np.arange(len(cohorts)) + bias
            ax.bar(
                x,
                cohorts_methods_avg[:, j],
                width,
                yerr=cohorts_methods_std[:, j],
                ecolor='black', 
                capsize=3,
                label=method,
                color=colors2[j],
                zorder=3
            )
            if j < len(methods) - 1:
                for c, cohort in enumerate(cohorts):
                    pval = cohorts_metrics_methods_pval[c, i, j]
                    if pval < 0.01:
                        txt = '***'
                    elif pval < 0.05:
                        txt = '**'
                    elif pval < 0.1:
                        txt = '*'
                    else:
                        txt = ' '
                    h = 0.1 if 'Log-rank' in metric else 0.01
                    # txt = '{:.3f}'.format(pval)
                    ax.text(x[c], cohorts_methods_avg[c, j] + cohorts_methods_std[c, j] + h, txt, ha='center',  color='k', fontsize=8)
        metric = metric.replace('dRMST', 'ΔRMST') if 'dRMST' in metric else metric
        metric = metric.replace('DFS', 'Recurrence') if 'DFS' in metric else metric
        ax.set_ylabel(metric)
        ax.set_ylim([ylimit_dict[key] for key in ylimit_dict.keys() if key in metric][0])
        ax.set_xticks(np.arange(len(cohorts)))
        ax.set_xticklabels(cohorts)
        ax.grid(zorder=0)
        ax.legend(loc="best", ncol=2)
        plt.tight_layout()
        plt.savefig(join(save_path, 'benchmark_' + metric + '.pdf'))
        plt.close()
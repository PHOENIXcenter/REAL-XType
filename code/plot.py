import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import scipy.stats as stats

color_scheme = ['#b7b8b8', '#c6c48d', '#dadbeb', '#b4a8dc', '#edc39a', '#d89d9d', '#64afaf']

def barplot_benchmark(data_path, plot_metric='C-Index', ymin=0., ymax=1., file_type='.png'):
    data = pd.read_csv(join(data_path, 'best_vals.csv'))
    custom_x_labels = [
        "Extension set\nOS", "Extension set\nrecurrence", "Validation set\nOS", "Validation set\nrecurrence",
        "Internal test\nset OS", "Internal test\nset recurrence", "External test\nset OS", "External test\nset recurrence",
        "Overall\nperformance"
    ]
    # Filter columns containing only plot_metric
    filtered_columns = [col for col in data.columns if plot_metric in col]
    filtered_data = data[['Unnamed: 0'] + filtered_columns]
    methods = filtered_data['Unnamed: 0']
    cleaned_methods = [method.rsplit('_', 1)[0] for method in methods]
    metrics = filtered_columns
    t = metrics
    metrics = [t[1], t[0], t[3], t[2], t[5], t[4], t[7], t[6]]
    t = custom_x_labels
    custom_x_labels = [t[1], t[0], t[3], t[2], t[5], t[4], t[7], t[6], t[8]]

    metrics_methods_seeds = np.zeros((len(metrics)+1, len(methods), 5))

    for i, metric in enumerate(metrics):
        for j, value in enumerate(filtered_data[metric]): # methods
            values = list(map(float, value.split('|')))
            metrics_methods_seeds[i, j, :] = np.array(values)

    # add a metric "Overall Performance"
    metrics_methods_seeds[-1, :, :] = np.mean(metrics_methods_seeds[:-1, :, :], axis=0)
    metrics.append("Overall\nperformance")

    metrics_methods_avg = np.mean(metrics_methods_seeds, axis=2)
    metrics_methods_std = np.std(metrics_methods_seeds, axis=2)

    # Plot grouped bar chart
    width = 0.15  # the width of the bars
    group_spacing = 0.2  # Additional spacing between groups
    x = np.arange(len(metrics)) * (len(methods) * width + group_spacing)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(methods):
        ax.bar(
            x + i * width,
            metrics_methods_avg[:, i],
            width,
            label=cleaned_methods[i],
            color=color_scheme[i % len(color_scheme)],
            yerr=metrics_methods_std[:, i],
            error_kw={'elinewidth': 1, 'capthick': 0},  # Remove horizontal caps
            capsize=5,
        )

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel(plot_metric)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(custom_x_labels, rotation=0)
    ax.set_ylim(ymin, ymax)
    ax.legend(title='Methods')

    # Prepare p-value results (NS, *, **, ***) for significance annotations
    metrics_methods_pval = np.empty((len(metrics), len(methods)), dtype='float')
    for i, metric in enumerate(metrics):
        # Perform T-tests for XType vs. other methods and store p-values
        x_type = 'XType'
        for j, method in enumerate(cleaned_methods):
            if method != x_type:
                # Extract data for XType and other methods for each group
                x_type_data = metrics_methods_seeds[i, cleaned_methods.index(x_type), :]
                method_data = metrics_methods_seeds[i, j, :]

                # Perform T-test
                t_stat, p_val = stats.ttest_ind(x_type_data, method_data)
            else:
                p_val = None # No comparison for XType vs itself
            metrics_methods_pval[i, j] = p_val

    unit_dist = (ymax-ymin)*0.03
    # Annotate the plot with significance results
    x_type_idx = len(cleaned_methods) - 1  # XType is the last method
    for i, metric in enumerate(metrics):
        for j, method in enumerate(cleaned_methods[:-1]):  # Compare XType with all other methods
            if method != 'XType':  
                method_idx = cleaned_methods.index(method)
                
                # XType and method positions in the bar chart
                x1 = x[i] + x_type_idx * width
                x2 = x[i] + method_idx * width
                
                # Plot a horizontal black line between XType and the other method
                y = np.amax(metrics_methods_avg[i, :] + metrics_methods_std[i, :]) + (len(cleaned_methods)-j-1)*unit_dist
                ax.plot([x1, x2], [y, y], color='black', lw=1, zorder=10)

                # Add significance symbol above the black line
                p_val = metrics_methods_pval[i, method_idx]

                # Assign p-value symbols based on p-value
                if p_val is None:
                    p_str = ' '  # No comparison
                elif p_val < 0.01:
                    p_str = '***'
                elif p_val < 0.05:
                    p_str = '**'
                elif p_val < 0.1:
                    p_str = '*'
                else:
                    p_str = 'ns'

                if p_str == 'ns':
                    bias = 0.4 
                    fs = 8
                else:
                    bias = 0.1
                    fs = 10
                ax.text((x1 + x2) / 2, y + unit_dist * bias,
                        p_str, ha='center', va='center', fontsize=fs, color='black')

    # Show the plot
    plt.tight_layout()
    plt.savefig(join(data_path, plot_metric+file_type))
    plt.close()

if __name__ == '__main__':
    barplot_benchmark(r'data\benchmark', plot_metric='C-Index', ymin=0.5, ymax=0.75, file_type='.pdf')
    barplot_benchmark(r'data\benchmark', plot_metric='Log-rank score', ymin=0., ymax=9., file_type='.pdf')
import argparse, os, random
from os.path import join
import data_utils as utils
from network_model.cox_cdan_model import *
from viz_utils import *
import numpy as np

parser = argparse.ArgumentParser(description='Process some parameters.')
# data preprocess
parser.add_argument('--log2', type=bool, default=True)
parser.add_argument('--zscore', type=bool, default=True)
parser.add_argument('--feat_select', type=str, default='f1269')
parser.add_argument('--min_impute', type=bool, default=True)

# model params
parser.add_argument('--bias', type=bool, default=False)
parser.add_argument('--h_dim', type=int, default=16)
parser.add_argument('--activ', type=str, default='sigmoid')
parser.add_argument('--regu', type=float, default=1e-3)
parser.add_argument('--loss_su', type=float, default=1.)
parser.add_argument('--loss_nll', type=float, default=1.)
parser.add_argument('--loss_da', type=float, default=0.3)
parser.add_argument('--loss_var', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--step_num', type=int, default=500)
parser.add_argument('--b_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=0.1)
args = parser.parse_args()

# Load Ng et al.'s cohort, using the sample postfixed with 02 among the triplicated samples
sample_group = 1
dataset = utils.load_Ng_cohort('data', sample_group=1)
classifier, _ = make_simple_models(
    args.h_dim,
    3, 
    4,
    args.lr,
    args.regu,
    args.activ,
    args.bias
)

# ensemble the best classifiers in 4 test cohorts by averaging the predicted probability
prob_list = []
for dataset_id, test_cohort in enumerate(['SH', 'GZ', 'FZ', 'Gao']):
    classifier.load_weights(join('data', 'model_weights', test_cohort, 'ckpt_fold')).expect_partial()
    prob = classifier.prob(dataset['data'])
    prob_list.append(prob)
mean_prob = np.mean(np.stack(prob_list, axis=2), axis=2) # patients * subtypes
prob = mean_prob / np.tile(np.sum(mean_prob, axis=1, keepdims=True), (1, 3))
assignments = np.argmax(prob, axis=1)
dataset['assignments'] = assignments
fig, ax = plt.subplots(figsize=(4, 4))
plot_km_curve_custom(
    dataset['OS'], 
    dataset['status'], 
    assignments, 
    3,
    ax, 
    title='Ng_OS_REAL',
    text_bias=30,
    clip_time=60
)
fig.tight_layout()
plt.savefig(join('data', 'km_Ng_OS_{:}.png'.format(sample_group)))
plt.close()
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import join

import data_utils as data_utils
import viz_utils as viz_utils

def generate_survival_correlated_synthetic_data(
		seed=0, 
		n_sample=100,
		censor_rate=0., 
		batch_noise_rate=1., 
		os_swap_rate=0., 
		viz=False, 
		save_path=None, 
	):

	np.random.seed(seed)

	# synthesize data
	unit_dim = 10

	true_covar = np.arange(n_sample)/n_sample
	posi_covars = np.tile(np.expand_dims(true_covar, axis=-1), (1, unit_dim))
	nega_covars = 1 - posi_covars
	rand_covars = np.random.rand(n_sample, unit_dim)
	raw_covars = np.concatenate([posi_covars, nega_covars], axis=-1)
	# raw_covars += np.random.normal(0., 0.5, size=raw_covars.shape)
	subtypes = np.r_[np.zeros(int(n_sample/2)), np.ones(int(n_sample/2))]
	OS = 60 - true_covar * 60.
	status = np.ones(n_sample)
	if censor_rate > 0.:
		status[np.random.choice(
				np.arange(n_sample), int(n_sample * censor_rate), replace=False
			)
		] = 0

	# batch effect
	ids_pool = np.arange(n_sample)
	batches = np.zeros(n_sample)
	n_batch = 2
	datasets = []
	covars = copy.deepcopy(raw_covars)
	for batch_id in range(n_batch):
		ids_pool.sort()
		select_num = min(int(n_sample/n_batch), len(ids_pool))
		mean = np.tile(np.expand_dims(np.random.rand(raw_covars.shape[1]), axis=0), (select_num, 1))
		noise = np.random.normal(loc=mean, scale=0.2) * batch_noise_rate

		subtypes_selected_ids = []
		for subtype in range(2):
			subtype_ids_pool = ids_pool[subtypes[ids_pool] == subtype]
			subtype_selected_ids = np.random.choice(
				subtype_ids_pool,
				int(select_num/2),
				replace=False
			)
			subtypes_selected_ids.append(subtype_selected_ids)
		selected_ids = np.concatenate(subtypes_selected_ids, axis=0)	
		selected_ids.sort()
		batches[selected_ids] = batch_id
		covars[selected_ids, :] = raw_covars[selected_ids, :]
		raw_data = raw_covars[selected_ids, :]
		data = raw_data + noise
		ids_pool = np.array(list(set(ids_pool) - set(selected_ids)))
		# data = covars[selected_ids, :]
		datasets.append(
			{	
				'raw_data': raw_data,
				'data': data,
				'OS': OS[selected_ids],
				'status': status[selected_ids],
				'DFS': OS[selected_ids],
				'recurrence': status[selected_ids],
				'subtypes': subtypes[selected_ids],
				'patients': [str(selected_id) for selected_id in selected_ids],
				'label': 'batch_' + str(batch_id)
			}
		)

	# # randomize OS
	# for i, dataset in enumerate(datasets):
	# 	np.random.shuffle(datasets[i]['OS'])

	# randomly swap OS between subtypes
	if os_swap_rate > 0:
		n = len(datasets[1]['subtypes'])
		swap_ids_s0 = np.random.choice(np.arange(n)[datasets[1]['subtypes']==0], int(n/2 * os_swap_rate), replace=False)
		swap_ids_s1 = np.random.choice(np.arange(n)[datasets[1]['subtypes']==1], int(n/2 * os_swap_rate), replace=False)
		temp_os = copy.deepcopy(datasets[1]['OS'][swap_ids_s0])
		datasets[1]['OS'][swap_ids_s0] = copy.deepcopy(datasets[1]['OS'][swap_ids_s1])
		datasets[1]['OS'][swap_ids_s1] = copy.deepcopy(temp_os)

	if viz:
		# visualisation
		fig, axs = plt.subplots(2, 3, figsize=(6, 6))
		cmap = cm.coolwarm
		for i, dataset in enumerate(datasets):
			ax = axs[i, 0].imshow(datasets[i]['raw_data'], interpolation='nearest', aspect='auto', cmap=cmap)
			axs[i, 0].set_title(dataset['label'] + ' raw data')
			axs[i, 1].imshow(datasets[i]['data'], interpolation='nearest', aspect='auto', cmap=cmap)
			# axs[i, 1].set_title(dataset['label'] + ' data')
			axs[i, 1].set_xlabel('feature')
			axs[i, 1].set_ylabel('sample')
			axs[i, 2] = viz_utils.plot_km_curve_custom(
				datasets[i]['OS'], 
				datasets[i]['status'], 
				datasets[i]['subtypes'], 
				2, 
				axs[i, 2], 
				title=dataset['label'] + ' OS'
			)
		fig.colorbar(ax, ax=axs[0, 0])
		fig.colorbar(ax, ax=axs[1, 0])
		plt.tight_layout()
		plt.savefig(save_path)

	return datasets


# 	return datasets

if __name__ == '__main__':
	for i in [0, 10]:
		datasets = generate_survival_correlated_synthetic_data(
			seed=0, 
			n_sample=1000,
			censor_rate=0., 
			batch_noise_rate=i * 0.4, 
			os_swap_rate=i * 0.02, 
			viz=True, 
			save_path=r'D:\Data\github\SRPS\data\toy-20220608-2\covar_and_OS{:}.png'.format(i), 
			# save_path=r'D:\Data\SRPS\survival_correlated_data\covar_and_OS', 
		)



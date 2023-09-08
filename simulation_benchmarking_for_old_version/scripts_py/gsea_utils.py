import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import data_utils as data_utils
import viz_utils as viz_utils
import numpy as np
import pickle, copy, os, platform, time

from numpy.linalg import norm
from os.path import join

def GSEA(datasets, result_path, fold_num, geneset_path, nature_pathway2name_and_type_dict, viz=False):
    start_time = time.time()
    pathway_list = list(nature_pathway2name_and_type_dict.keys())
    name_list = [nature_pathway2name_and_type_dict[key][0] for key in nature_pathway2name_and_type_dict.keys()]
    type_list = [nature_pathway2name_and_type_dict[key][1] for key in nature_pathway2name_and_type_dict.keys()]

    df = pd.concat([pd.read_csv(join(result_path, 'fold-' + str(fold) + '.csv')) for fold in range(fold_num)])
    subtype_num = int(np.amax(datasets[0]['subtypes']) + 1)
    assignments = df['assignment'].to_numpy()
    datasets_vects = []
    for dataset_id, dataset in enumerate([datasets[0]] + datasets): 
        profile_dict = {'NAME': dataset['genes'], 'DESCRIPTION': ['NA'] * len(dataset['genes'])}
        for sample, sample_vect in zip(dataset['patients'], dataset['data']):
            profile_dict[sample] = np.log2(sample_vect)
        profile_df = pd.DataFrame(data=profile_dict)
        patient_ids = [df['patients'].to_list().index(patient) for patient in dataset['patients']]
        dataset_vects = []
        subtypes = dataset['subtypes'] if dataset_id == 0 else assignments[patient_ids]
        label = dataset['label'] + '_source' if dataset_id == 0 else dataset['label']
        for target_subtype in range(int(np.amax(datasets[0]['subtypes']) + 1)):
            binary_subtypes = copy.deepcopy(subtypes)
            binary_subtypes[subtypes != target_subtype] = 0
            binary_subtypes[subtypes == target_subtype] = 1
            class_vector = ['S-' + ''.join(['I'] * (target_subtype + 1)) if subtype == 1 else 'others' for subtype in binary_subtypes]

            gs_res = gp.gsea(data=profile_df,
                gene_sets=geneset_path, # enrichr library names, e.g. 'KEGG_2016'
                cls=class_vector,
                # set permutation_type to phenotype if samples >=15
                permutation_type='phenotype',
                permutation_num=100, # reduce number to speed up test
                outdir=None,  # do not write output to disk
                no_plot=True, # Skip plotting
                method='signal_to_noise',
                processes=1, 
                seed=0,
                format='png'
            )
            res_df = gs_res.res2d
            res_pathways = res_df.index.to_list()
            res_nes = res_df['es'].to_numpy()
            res_fdr = res_df['fdr'].to_numpy()
            nes_vect = []
            for pathway in pathway_list:
                nes_val = 0
                if pathway in res_pathways:
                    idx = res_pathways.index(pathway)
                    if res_fdr[idx] < 0.1:
                        nes_val = res_nes[idx]
                nes_vect.append(nes_val)
            dataset_vects.append(np.asarray(nes_vect))
        if viz:
            viz_utils.plot_annotated_heatmap(
                data_arr=np.stack(dataset_vects, axis=1),
                x_labels=['S-' + ''.join(['I'] * (s + 1)) for s in range(subtype_num)], 
                y_labels=[pathway_name + '_' + pathway_type for pathway_name, pathway_type in zip(name_list, type_list)], 
                rotation=0, 
                title='GSEA_' + label + '.png', 
                save_path=join(result_path, 'GSEA_' + label + '.png')
            )
        datasets_vects.append(dataset_vects)
    datasets_similarity = []

    for i, dataset_vects in enumerate(datasets_vects[1:]):
        sim_list = []
        for a, b in zip(datasets_vects[0], dataset_vects):
            # print(a, b)
            sim_list.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-5)  
        datasets_similarity.append(np.mean(sim_list))
        print(datasets[i]['label'], sim_list)

    return datasets_similarity

def ssGSEA_similarity(datasets, result_path, nature_pathway2name_and_type_dict):
    # pathway_list = list(nature_pathway2name_and_type_dict.keys())
    # name_list = [nature_pathway2name_and_type_dict[key][0] for key in nature_pathway2name_and_type_dict.keys()]
    # type_list = [nature_pathway2name_and_type_dict[key][1] for key in nature_pathway2name_and_type_dict.keys()]
    score_df_list = [pd.read_csv(
        join(result_path, 'GSEA_' + dataset['label'] + '_subtype_score.csv'), 
        header=0, index_col=0
    ) for dataset in datasets]
    merged_score_df = score_df_list[0]
    for score_df in score_df_list[1:]:
        # merged_score_df = merged_score_df.merge(score_df, how='outer')
        merged_score_df = pd.concat([merged_score_df, score_df], axis=1)
    score_arr = merged_score_df.to_numpy().astype(float)
    score_arr[np.isnan(score_arr)] = 0.
    datasets_scores = np.split(score_arr, len(datasets), axis=1)
    sim_list = []
    for dataset_scores in datasets_scores[1:]:
        inner_prod = np.sum(dataset_scores * datasets_scores[0], axis=0)
        norm_prod = np.linalg.norm(dataset_scores, axis=0) * np.linalg.norm(datasets_scores[0], axis=0)
        similarity = np.fabs(np.mean(inner_prod / norm_prod))
        sim_list.append(similarity)
    return sim_list

def read_nature_pathway(data_path):
    nature_pathway2name_and_type_dict = {}
    pathway2gene_dict, gene2pathway_dict = {}, {}
    with open(join(data_path, 'selected_pathways.csv'), 'rb') as csvfile:
        df = pd.read_csv(csvfile)
        for line_list in df.values.tolist():
            nature_pathway2name_and_type_dict[line_list[0]] = line_list[1:]

    with open(join(data_path, 'geneset.csv'), 'rb') as csvfile:
        df = pd.read_csv(csvfile, dtype=str)
        sub_df = df[list(nature_pathway2name_and_type_dict.keys())]
        for pathway in nature_pathway2name_and_type_dict.keys():
            gene_list = [item for item in sub_df[pathway].tolist() if isinstance(item, str)]
            pathway2gene_dict[pathway] = gene_list
            for gene in gene_list:
                if gene not in gene2pathway_dict.keys():
                    gene2pathway_dict[gene] = [pathway]
                else:
                    gene2pathway_dict[gene].append(pathway)
    
    return pathway2gene_dict, gene2pathway_dict, nature_pathway2name_and_type_dict

def save_gmt(pathway2gene_dict, data_path):
    with open(join(data_path, 'genest.gmt'), 'w') as f:
        for pathway in pathway2gene_dict.keys():
            f.write('\t'.join([pathway, 'na'] + pathway2gene_dict[pathway]) + '\n')
     
if __name__ == '__main__':
    platform_sys = platform.system()
    if platform_sys == 'Linux':
        data_path = '/data/linhai/SRPS'
    else:
        disk = os.getcwd().split(':')[0]
        data_path = disk + r':\Data\SRPS'



    pathway2gene_dict, gene2pathway_dict, nature_pathway2name_and_type_dict = read_nature_pathway(join(data_path, 'r_data'))

    # for pathway in nature_pathway2name_and_type_dict.keys():
    #     print(pathway, nature_pathway2name_and_type_dict[pathway])
    # save_gmt(pathway2gene_dict, join(data_path, 'HCC_data'))
    # assert False 

    pickle_file_path = join(data_path, 'temp_files', 'HCC.p')
    if not os.path.isfile(pickle_file_path):
        gene2prote_dict, prote2gene_dict = data_utils.gene2prote_and_prote2gene(
            join(data_path, 'HCC_data', 'HCC-Proteomics-Annotation-20210226.xlsx')
        )
        raw_datasets = data_utils.load_HCC_data(
            join(data_path, 'HCC_data'), 
            gene2prote_dict
        ) 
        pickle.dump([raw_datasets, gene2prote_dict, prote2gene_dict], open(pickle_file_path, 'wb'))
    else:
        raw_datasets, gene2prote_dict, prote2gene_dict = pickle.load(open(pickle_file_path, 'rb'))
    raw_datasets = data_utils.HCC_data_preprocess(
        raw_datasets, 
        join(data_path, 'HCC_data'), 
        prote2gene_dict, 
        clip=60, 
        nan_thresh=0.7, 
        zscore=False, 
        imputation=True, 
        log2=False, 
        f1269=False
    )
    ssGSEA_similarity(
        [raw_datasets[0], raw_datasets[-1]],
        join(data_path,
            'archive',
            '20210602_HCC-Jiang2Gao_tuning',
            'seed0',
            'deepCLife_loss_su-1e-01_regu-1e-03'
        ),
        nature_pathway2name_and_type_dict
    )

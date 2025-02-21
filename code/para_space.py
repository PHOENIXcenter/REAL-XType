import copy
import itertools
import pandas as pd
import collections
from os.path import join

def get_paras(args, experiment_path):
    # Here we directly provide the best hyperparameter for a quick reproduction.
    # The full search of all hyperparameters can take more than 10 hours.
    if args.method == 'XType':
        # para_dicts = collections.OrderedDict(
        #     dropout = [0.6, 0.8],
        #     loss_var = [0.01, 0.05, 0.1],
        #     loss_da = [0.1, 0.3, 0.5],
        #     regu = [1e-4, 1e-3, 2e-3],
        # )
        para_dicts = collections.OrderedDict(
            dropout = [0.6],
            loss_var = [0.01],
            loss_da = [0.3],
            regu = [2e-3],
        )
    elif args.method == 'DNN':
        # para_dicts = collections.OrderedDict(
        #     loss_nll = [0.], 
        #     loss_da = [0.],
        #     loss_var = [0.],
        #     dropout = [0.2, 0.4]
        # )
        para_dicts = collections.OrderedDict(
            loss_nll = [0.], 
            loss_da = [0.],
            loss_var = [0.],
            dropout = [0.2]
        )
    elif args.method == 'RandomForest':
        # para_dicts = collections.OrderedDict(
        #     n_estimators=[10, 50, 100],
        #     max_depth=[None, 10, 20]
        # )
        para_dicts = collections.OrderedDict(
            n_estimators=[50],
            max_depth=[None]
        )
    elif args.method == 'LogisticRegression':
        # para_dicts = collections.OrderedDict(
        #     alpha = [0.1, 0.3, 0.6, 0.9]
        # )
        para_dicts = collections.OrderedDict(
            alpha = [0.6]
        )
    elif args.method == 'XGBoost':
        # para_dicts = collections.OrderedDict(
        #     n_estimators=[10, 50, 100],
        #     max_depth=[6, 10, 20] 
        # )
        para_dicts = collections.OrderedDict(
            n_estimators=[10],
            max_depth=[6] 
        )
    
    para_names = list(para_dicts.keys())
    para_names_vals = [para_dicts[para_name] for para_name in para_names]
    param_list = list(itertools.product(*para_names_vals))
    print('param num', len(param_list))
    if args.para_id < len(param_list):
        selected_paras = param_list[args.para_id]  
        for i, para_name in enumerate(para_names):
            setattr(args, para_name, selected_paras[i])
    else:
        return args, None

    # save all parameters
    comb_dict = {}
    for i, params in enumerate(param_list):
        for key, val in zip(para_names, params):
            if key in comb_dict.keys():
                comb_dict[key].append(val)
            else:
                comb_dict[key] = [val]

    df = pd.DataFrame(comb_dict)
    df.to_csv(experiment_path)
    return args, comb_dict, 
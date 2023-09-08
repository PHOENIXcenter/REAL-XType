import copy
import itertools
from os.path import join

def get_paras(args):
    if args.data == 'toy':
        para_dicts = {
            'SRPS': {
                'loss_su': [0.04],
                'loss_rl': [1.],
                'bl_lr': [1e-3],
                'hdim': [20],
                'dropout': [0.],
                'dropout_rl': [0.],
                'epoch_num': [10000],
                'subtype_num': [2]
            },
            'SourceOnly': {
                'loss_su': [0.04],
                'loss_rl': [0.],
                'bl_lr': [0.],
                'hdim': [20],
                'dropout': [0.],
                'dropout_rl': [0.],
                'epoch_num': [2000],
                'subtype_num': [2]
            },
        }
    elif args.data == 'synthetic':
        if args.batch_effect == 'lv3':
            para_dicts = {
                'SRPS': {
                    'hdim': [10],
                    'bl_lr': [1e-2],
                    'loss_su': [4e-2],
                    'regu': [1e-3],
                    'dropout': [0.3],
                    'no_early_stop': [False]
                },
                
                'SRPS(no_baseline)': {
                    'hdim': [10],
                    'bl_lr': [0.],
                    'loss_su': [4e-2],
                    'regu': [1e-3],
                    'dropout': [0.3],
                    'no_early_stop': [False]
                },

                'SRPS(soft)': {
                    'learning_rate': [1e-2],
                    'dropout': [0.],
                    'regu': [1e-3],
                    'loss_su': [1.],
                    'loss_survival': [1e-3]
                },
                
                'deepCLife': {
                    'learning_rate': [1e-2],
                    'dropout': [0.],
                    'regu': [1e-3],
                    'loss_su': [1.],
                    'loss_survival': [1e-3]
                },

                'SourceOnly': {
                    'encoder_layer_num': [2],
                    'encoder_hdim': [20],
                    'class_layer_num': [1],
                    'class_hdim': [10],
                    'learning_rate': [1e-2],
                    'dropout': [0.3]
                },

                'DANN': {
                    'encoder_layer_num': [2],
                    'encoder_hdim': [20],
                    'class_layer_num': [1],
                    'class_hdim': [10],
                    'learning_rate': [1e-2],
                    'dropout': [0.]
                },

                'RandomForestHarmony': {
                    'n_components': [10]
                },

                'RandomForest': {
                    'max_depth': [20]
                },
            }
        else:
            para_dicts = {
                'SRPS': {
                    'hdim': [10],
                    'bl_lr': [1e-2],
                    'loss_su': [4e-2],
                    'regu': [1e-3],
                    'dropout': [0.3],
                    'no_early_stop': [True]
                },
                
                'SRPS(no_baseline)': {
                    'hdim': [10],
                    'bl_lr': [0.],
                    'loss_su': [4e-2],
                    'regu': [1e-3],
                    'dropout': [0.3],
                    'no_early_stop': [True]
                },

                'SRPS(soft)': {
                    'learning_rate': [1e-2],
                    'dropout': [0.],
                    'regu': [1e-3],
                    'loss_su': [1.],
                    'loss_survival': [1e-3]
                },
                
                'deepCLife': {
                    'learning_rate': [1e-2],
                    'dropout': [0.],
                    'regu': [1e-3],
                    'loss_su': [1.],
                    'loss_survival': [1e-3]
                },

                'SourceOnly': {
                    'encoder_layer_num': [2],
                    'encoder_hdim': [20],
                    'class_layer_num': [1],
                    'class_hdim': [10],
                    'learning_rate': [1e-2],
                    'dropout': [0.3]
                },

                'DANN': {
                    'encoder_layer_num': [1],
                    'encoder_hdim': [20],
                    'class_layer_num': [2],
                    'class_hdim': [10],
                    'learning_rate': [1e-2],
                    'dropout': [0.]
                },

                'RandomForestHarmony': {
                    'n_components': [10]
                },

                'RandomForest': {
                    'max_depth': [20]
                },
            }
    elif args.data =='HCC':
        para_dicts = {
            'SRPS': {
                'hdim': [10],
                'bl_lr': [1e-4],
                'loss_su': [4e-2],
                'regu': [1e-4],
                'dropout_rl': [0.1],
                'dropout': [0.8],
                'no_early_stop': [False]
            },
            
            'deepCLife': {
                'learning_rate': [1e-3],
                'regu': [1e-4],
                'dropout': [0.8],
                'batch_size': [0],
                'loss_survival': [0.1],
                'encoder_layer_num': [0]
            },

            'SourceOnly': {
                'encoder_layer_num': [1],
                'encoder_hdim': [20],
                'class_layer_num': [1],
                'class_hdim': [10],
                'activation': [None],
                'learning_rate': [1e-2],
                'dropout': [0.8]
            },

            'DANN': {
                'encoder_layer_num': [2],
                'encoder_hdim': [20],
                'class_layer_num': [2],
                'class_hdim': [10],
                'activation': [None],
                'learning_rate': [1e-2],
                'dropout': [0.3]
            },

            'RandomForestHarmony': {
                'n_components': [100],
                'max_depth': [10]
            },

            'RandomForest': {
                'max_depth': [2]
            },
        }
    elif args.data =='HCC_LUAD':
        para_dicts = {
            'SRPS': {
                'hdim': [100],
                'bl_lr': [1e-4],
                'loss_su': [4e-2],
                'regu': [1e-5],
                'dropout_rl': [0.1],
                'no_early_stop': [False]
            },
            
            'deepCLife': {
                'learning_rate': [1e-2],
                'regu': [1e-4],
                'dropout': [0.8],
                'batch_size': [0],
                'loss_survival': [0.01],
                'encoder_layer_num': [0]
            },

            'SourceOnly': {
                'encoder_layer_num': [1],
                'encoder_hdim': [20],
                'class_layer_num': [1],
                'class_hdim': [10],
                'activation': [None],
                'learning_rate': [1e-2],
                'dropout': [0.8]
            },

            'DANN': {
                'encoder_layer_num': [1],
                'encoder_hdim': [20],
                'class_layer_num': [2],
                'class_hdim': [10],
                'activation': [None],
                'learning_rate': [1e-2],
                'dropout': [0.]
            },

            'RandomForestHarmony': {
                'n_components': [20]
            },

            'RandomForest': {
                'max_depth': [2]
            },
        }
    else:
        assert False, 'param error!'
        
        
    if args.mode in para_dicts.keys():
        for method in para_dicts.keys():
            if method == args.mode:
                mode_list = []
                para_names = list(para_dicts[method].keys())
                para_names_vals = [para_dicts[method][para_name] for para_name in para_names]
                paras_list = list(itertools.product(*para_names_vals))
                if args.para_id < len(paras_list):
                    selected_paras = paras_list[args.para_id]  
                    for i, para_name in enumerate(para_names):
                        setattr(args, para_name, selected_paras[i])
                        para_str = '-{:.0e}'.format(selected_paras[i]) if type(
                            selected_paras[i]) is float else '-' + str(selected_paras[i])
                        args.mode = args.mode + '_' + para_name + para_str
                else:
                    selected_paras = None
                break
    elif args.mode == 'compare':
        args.zscore = False
        args.imputation = True
        args.log2 = True
        args.f1269 = False
        selected_paras = 0
    else:
        selected_paras = 0

    if args.data == 'synthetic':
        result_path = join(
            args.data_path, 
            args.experiment, 
            'batch_effect_' + args.batch_effect,
            'seed'+str(args.seed),
            args.mode
        ) 
    else:
        result_path = join(
            args.data_path, 
            args.experiment, 
            'seed'+str(args.seed),
            args.mode
        ) 

    # print(args)

    return args, selected_paras, result_path
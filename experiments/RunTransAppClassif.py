#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia
# @description : appliance detection experiments on CER dataset
# @component: src/utils/
# @file : RunTransAppClassif.py
#
#################################################################################################################

import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

root = Path(os.getcwd()).resolve().parents[0]
sys.path.append(str(root))
from experiments.data_utils import *
from src.Models.TransApp import *
from src.AD_Framework.Framework import *

def launch_training(model, 
                    save_path, m, win,
                    datas_tuple,
                    dict_params):
    """
    Launch model training

    Input :
    - model : model instance
    - save_path : path to save model / case
    - m : number of variable of the MTS
    - win : window size of subsequences
    - datas_tuple : [X_train, y_train, ... X_test_voter , y_test_voter]
    - dict_params : dictionary of parameters
    """

    # Scliced data
    X_train = datas_tuple[0]
    y_train = datas_tuple[1]
    X_valid = datas_tuple[2]
    y_valid = datas_tuple[3]
    X_test  = datas_tuple[4]
    y_test  = datas_tuple[5]

    # Entire curves data
    X_train_voter = datas_tuple[6]
    y_train_voter = datas_tuple[7]
    X_valid_voter = datas_tuple[8]
    y_valid_voter = datas_tuple[9]
    X_test_voter  = datas_tuple[10]
    y_test_voter  = datas_tuple[11]

    # Dataset
    train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
    valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
    test_dataset  = TSDataset(X_test, y_test,   scaler=True, scale_dim=[0])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dict_params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True)

    model_trainer = AD_Framework(model,
                                 train_loader=train_loader, valid_loader=valid_loader,
                                 learning_rate=dict_params['lr'], weight_decay=dict_params['wd'],
                                 criterion=nn.CrossEntropyLoss(),
                                 patience_es=dict_params['p_es'], patience_rlr=dict_params['p_rlr'],
                                 f_metrics=getmetrics(),
                                 n_warmup_epochs=dict_params['n_warmup_epochs'],
                                 scale_by_subseq_in_voter=True, scale_dim=[0],
                                 verbose=True, plotloss=False, 
                                 save_fig=False, path_fig=None,
                                 device="cuda", all_gpu=True,
                                 save_checkpoint=True, path_checkpoint=save_path)

    model_trainer.train(dict_params['epochs'])

    #============ eval last model ============#
    model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1), mask='test_metrics_lastmodel')

    #============ restore best weight and evaluate ============#    
    model_trainer.restore_best_weights()
    model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1))

    #============ find best quantile on valid dataset ============#
    model_trainer.ADFFindBestQuantile(TSDataset(X_valid_voter, y_valid_voter), m=m, win=win)
    quant_metric = model_trainer.ADFvoter_proba(TSDataset(X_test_voter, y_test_voter), m=m, win=win)
    print(quant_metric)

    return


def get_model_inst(m, win, dim_model, path_select_core=None):

    if path_select_core is not None:
        n_enc_layers = 5 if 'Large' in path_select_core else 3
    else:
        n_enc_layers = 3

    TApp = TransApp(max_len=win, c_in=m,
                    mode="classif",
                    n_embed_blocks=1, 
                    encoding_type='noencoding',
                    n_encoder_layers=n_enc_layers,
                    kernel_size=5,
                    d_model=dim_model, pffn_ratio=2, n_head=4,
                    prenorm=True, norm="LayerNorm",
                    activation='gelu',
                    store_att=False, attn_dp_rate=0.2, head_dp_rate=0., dp_rate=0.2,
                    att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
                    c_reconstruct=1, apply_gap=True, nb_class=2)

    if path_select_core is not None:
        TApp.load_state_dict(torch.load(path_select_core)['model_state_dict'])

    return TApp


if __name__ == "__main__":
    print("TransApp experiments on CER data detection cases.")

    path_results = str(root) + '/results/TransAppResults/'
    path_pretrained_core = '/results/TransAppPretrained/'

    case_name  = str(sys.argv[1])
    model_name = str(sys.argv[2])
    dim_model  = int(sys.argv[3])
    frac       = str(sys.argv[4])

    win = 1024

    # List of possible Embedding : Univariate, Time Embedding 
    list_exo_variable = [['hours_cos', 'hours_sin', 'days_cos', 'days_sin']]
    name_exo_variables = ['Embed']

    for i, l in enumerate(list_exo_variable):
        if model_name=='TransApp':
            path_core = None
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 15,
                           'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
        elif model_name=='TransAppPT':
            path_core = path_pretrained_core + str(name_exo_variables[i]) + '/TransApp' + str(dim_model) + '.pt'
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 15,
                           'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
        elif model_name=='TransAppLPT':
            path_core = path_pretrained_core + str(name_exo_variables[i]) + '/TransAppL' + str(dim_model) + '_' + frac + '.pt'
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 15,
                           'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
        elif model_name=='TransAppLargePT':
            path_core = path_pretrained_core + str(name_exo_variables[i]) + '/TransAppLarge' + str(dim_model) + '_' + frac + '.pt'
            dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 15,
                           'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}
        else:
            raise ValueError('Model Name unknown.')

        _ = create_dir(path_results + name_exo_variables[i] + '/')
        path = create_dir(path_results + name_exo_variables[i] + '/' + case_name + '/')

        m = len(l) + 1 # MTS Number of variables

        for rd_state in range(0, 3):
            if model_name=='TransAppLPT' or model_name=='TransAppLargePT':
                save_path = path + model_name + str(dim_model) + '_' + frac + '_' + str(rd_state)
            else:
                save_path = path + model_name + str(dim_model) + '_' + str(rd_state)

            model = get_model_inst(m=m, win=win, dim_model=dim_model, path_select_core=path_core)

            datas_tuple = CER_get_data_case(case_name, seed=rd_state, exo_variable=l,
                                            win=win, ratio_resample=0.8, group='residential')
            launch_training(model, 
                            save_path, m, win, 
                            datas_tuple,  
                            dict_params)

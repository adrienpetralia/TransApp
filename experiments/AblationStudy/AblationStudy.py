import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

root = Path(os.getcwd()).resolve().parents[1]
sys.path.append(str(root))

from experiments.data_utils import *
from src.AD_Framework.Framework import *
from src.TransAppModel.TransApp import *

from sklearn.preprocessing import StandardScaler


def launch_training(model, 
                    save_path, m, win,
                    datas_tuple,
                    dict_params,
                    voter_df=False):

    # Scliced data
    X_train = datas_tuple[0]
    y_train = datas_tuple[1]
    X_valid = datas_tuple[2]
    y_valid = datas_tuple[3]
    X_test  = datas_tuple[4]
    y_test  = datas_tuple[5]

    # Voter data
    if voter_df:
        df_train_voter = datas_tuple[6]
        df_valid_voter = datas_tuple[7]
        df_test_voter  = datas_tuple[8]
    else:
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

    model_trainer = DAP_Framework(model,
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

    #============ find quantile on valid ============#
    if voter_df:
        model_trainer.DAPFindBestQuantile(df_valid_voter, m=m, win=win)
        quant_metric = model_trainer.DAPvoter_proba(df_test_voter, m=m, win=win)
    else:
        model_trainer.DAPFindBestQuantile(TSDataset(X_valid_voter, y_valid_voter), m=m, win=win)
        quant_metric = model_trainer.DAPvoter_proba(TSDataset(X_test_voter, y_test_voter), m=m, win=win)
    print(quant_metric)

    return


def get_model_inst(m, win, dim_model, encoding_type="noencoding", att_mask_diag=True):

    TApp = TransApp(max_len=win, c_in=m,
                    mode="classif",
                    n_embed_blocks=1, 
                    encoding_type=encoding_type,
                    n_encoder_layers=3,
                    kernel_size=5,
                    d_model=dim_model, pffn_ratio=2, n_head=4,
                    prenorm=True, norm="LayerNorm",
                    activation='gelu',
                    store_att=False, attn_dp_rate=0.2, head_dp_rate=0.1, dp_rate=0.2,
                    att_param={'attenc_mask_diag': att_mask_diag, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
                    c_reconstruct=1, apply_gap=True, nb_class=2)

    return TApp


if __name__ == "__main__":
    print("TransApp Ablation Study")

    path_results = str(root) + '/results/AblationStudy/'

    ablation = str(sys.argv[1])
    dataset  = str(sys.argv[2])
    dim_model = 96
    model_name = 'TransApp'

    if ablation=='att':
        path_results = path_results + 'DiagAttImpact/'
    else:
        path_results = path_results + 'EncodingImpact/'

    win = 1024
    m = 5
    dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 15,
                   'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}


    list_case_CER = ['cooker_case', 'dishwasher_case', 'desktopcomputer_case', 
                     'ecs_case', 'pluginheater_case', 'tumbledryer_case', 
                     'tv_greater21inch_case', 'tv_less21inch_case', 'laptopcomputer_case']

    list_case_CERBis = ["clim", "convpac", "dishwasher", "electricvehicle", 
                        "heater", "tumbledryer", "waterheater"]

    if dataset=='CER':
        path_results = path_results + '/CER/'
        for case_name in list_case_CER:
            path = create_dir(path_results + '/' + case_name + '/')
            for rd_state in range(0, 3):
                if ablation=='att':
                    save_path = path + model_name + str(dim_model) + '_' + str(rd_state)
                    model = get_model_inst(m=m, win=win, dim_model=dim_model, att_mask_diag=False)
                else:
                    save_path = path + model_name + str(dim_model) + '_' + ablation + '_' + str(rd_state)
                    model = get_model_inst(m=m, win=win, dim_model=dim_model, encoding_type=ablation) 

                datas_tuple = CER_get_data_case(case_name, seed=rd_state, exo_variable=['hours_cos', 'hours_sin', 'days_cos', 'days_sin'],
                                                win=win, ratio_resample=0.8, group='residential')
                launch_training(model, 
                                save_path, m, win, 
                                datas_tuple,  
                                dict_params, 
                                voter_df=False)
    else:
        return
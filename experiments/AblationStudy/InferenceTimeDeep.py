import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

root = Path(os.getcwd()).resolve().parents[1]
sys.path.append(str(root))

from experiments.data_utils import *
from src.AD_Framework.Framework import *
from src.Models.InceptionTime import *
from src.Models.ResNetAtt import *
from src.Models.ResNet import *
from src.Models.FCN import * 


def launch_training(model, 
                    len_i,
                    in_framework,
                    n_indiv,
                    save_path, m, win,
                    datas_tuple,
                    dict_params):

    # Scliced data
    X_train = datas_tuple[0]
    y_train = datas_tuple[1]
    X_valid = datas_tuple[2]
    y_valid = datas_tuple[3]
    X_test  = datas_tuple[4]
    y_test  = datas_tuple[5]

    # Voter data
    X_test_voter = datas_tuple[6]
    y_test_voter = datas_tuple[7]

    # Dataset
    train_dataset = TSDataset(X_train, y_train, scaler=False, scale_dim=[0])
    valid_dataset = TSDataset(X_valid, y_valid, scaler=False, scale_dim=[0])
    test_dataset  = TSDataset(X_test,  y_test,  scaler=False, scale_dim=[0])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True)

    X_test_voter = X_test_voter[:n_indiv].reshape(n_indiv, m, X_test_voter[:n_indiv].shape[-1]//m)[:, :, :len_i]
    y_test_voter = y_test_voter[:n_indiv]
    if in_framework:
        X_test_voter = X_test_voter.reshape(n_indiv, X_test_voter.shape[1] * X_test_voter.shape[2])
    else:
        test_dataset  = TSDataset(X_test_voter[:, :1, :], y_test_voter.ravel(), scaler=False, scale_dim=[0])
        train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    model_trainer = DAP_Framework(model,
                                  train_loader=train_loader, valid_loader=valid_loader,
                                  learning_rate=dict_params['lr'], weight_decay=dict_params['wd'],
                                  criterion=nn.CrossEntropyLoss(),
                                  patience_es=dict_params['p_es'], patience_rlr=dict_params['p_rlr'],
                                  f_metrics=accuracymetrics(),
                                  n_warmup_epochs=dict_params['n_warmup_epochs'],
                                  scale_by_subseq_in_voter=True, scale_dim=[0],
                                  verbose=True, plotloss=False, 
                                  save_fig=False, path_fig=None,
                                  device="cuda", all_gpu=True,
                                  save_checkpoint=True, path_checkpoint=save_path, 
                                  batch_size_voter=32)

    if not in_framework:
        #============ model time inference ============#
        print("model time inference")
        model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1))
        print(model_trainer.log['eval_time'])
    else:
        #============ framework time inference ============#
        print("framework time inference")
        #model_trainer.DAPvoter_proba(TSDataset(X_test_voter, y_test_voter), m=m, win=win)
        model_trainer.DAPvoter(TSDataset(X_test_voter, y_test_voter), m=m, win=win, mask_time="test_probavoter_time")
        print(model_trainer.log['test_probavoter_time'])

    return


def get_model_inst(m, win, dim_model=96):

    TApp = TransApp(max_len=win, c_in=m,
                    mode="classif",
                    n_embed_blocks=1, 
                    encoding_type="noencoding",
                    n_encoder_layers=3,
                    kernel_size=5,
                    d_model=dim_model, pffn_ratio=2, n_head=4,
                    prenorm=True, norm="LayerNorm",
                    activation='gelu',
                    store_att=False, attn_dp_rate=0.2, head_dp_rate=0.1, dp_rate=0.2,
                    att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
                    c_reconstruct=1, apply_gap=True, nb_class=2)

    return TApp


if __name__ == "__main__":
    print("Inference Time Study")

    model_name = str(sys.argv[1])
    in_framework = bool(int(sys.argv[2]))
    dim_model = 96

    print(in_framework)
    print(model_name)

    if in_framework:
        path_results = str(root) + '/results/AblationStudy/TestingTime/InFramework/'
    else:
        path_results = str(root) + '/results/AblationStudy/TestingTime/OutFramework/'

    models = {'ResNet': {'Instance': ResNet, 'lr': 1e-3, 'wd': 0, 'batch_size': 32,
                         'p_es': 10, 'p_rlr': 3, 'n_warmup_epochs': 0},
              'Inception': {'Instance': Inception, 'lr': 1e-3, 'wd':0, 'batch_size': 32,
                         'p_es': 10, 'p_rlr': 3, 'n_warmup_epochs': 0},
              'FCN': {'Instance': FCN, 'lr': 1e-3, 'wd': 0, 'batch_size': 32,
                      'p_es': 10, 'p_rlr': 3, 'n_warmup_epochs': 0},
              'FrameworkResNetAtt': {'Instance': ResNetAtt, 'lr': 0.0002, 'wd': 0.5, 'batch_size': 32,
                                      'p_es': 10, 'p_rlr': 3, 'n_warmup_epochs': 0}
             }

    win = 1024
    m = 5
    dict_params = {'lr': 1e-4, 'wd': 1e-3, 'batch_size': 16, 'epochs': 15,
                   'p_es': 5, 'p_rlr': 3, 'n_warmup_epochs': 0}

    datas_tuple = CER_get_data_case("cooker_case", seed=0, exo_variable=['hours_cos', 'hours_sin', 'days_cos', 'days_sin'],
                                    win=win, ratio_resample=0.8, group='residential')

    list_len_i = [1024, 2048, 4096, 8192, 16384, 25728]
    list_n_indiv = [1, 10, 100, 1000]

    
    for len_i in list_len_i:
        for n_indiv in list_n_indiv:
            if model_name=='TransApp':
                if in_framework:
                    model = get_model_inst(m=m, win=win)
                else:
                    model = get_model_inst(m=1, win=len_i)
            else:
                if in_framework:
                    model = models[model_name]['Instance'](in_channels=m)
                else:
                    model = models[model_name]['Instance'](in_channels=1)

            save_path = path_results + model_name + '_' + str(len_i) + '_' + str(n_indiv)

            launch_training(model, 
                            len_i,
                            in_framework,
                            n_indiv,
                            save_path, m, win,
                            datas_tuple,
                            dict_params)
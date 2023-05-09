#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia
# @description : Pretrained TransApp
# @component: experiments/
# @file : RunTransAppPretraining.py
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
from src.utils.losses import *

def launch_pretraining(model, 
                       save_path, m, win,
                       X_train,
                       GeomMask,
                       dict_params):

    pretraining_dataset = TSDataset(X_train, scaler=True, scale_dim=[0])
    train_loader = torch.utils.data.DataLoader(pretraining_dataset, batch_size=dict_params['batch_size'], shuffle=True)

    model_pretrainer = self_pretrainer(model,                                     
                                       train_loader, valid_loader=None,
                                       learning_rate=dict_params['lr'], weight_decay=dict_params['wd'],
                                       name_scheduler='CosineAnnealingLR',
                                       dict_params_scheduler={'T_max': dict_params['epochs'], 'eta_min': 1e-6},
                                       warmup_duration=None,
                                       criterion=MaskedMSELoss(type_loss='L1'), mask=GeomMask,
                                       device="cuda", all_gpu=True,
                                       verbose=True, plotloss=False, 
                                       save_fig=False, path_fig=None,
                                       save_only_core=False,
                                       save_checkpoint=True, path_checkpoint=save_path)

    model_pretrainer.train(dict_params['epochs'])

    return

def get_model_inst(m, win, dim_model):

    TApp = TransApp(max_len=win, c_in=m,
                    mode="pretraining",
                    n_embed_blocks=1, 
                    encoding_type='noencoding',
                    n_encoder_layers=3,
                    kernel_size=5,
                    d_model=dim_model, pffn_ratio=2, n_head=4,
                    prenorm=True, norm="LayerNorm",
                    activation='gelu',
                    store_att=False, attn_dp_rate=0.2, head_dp_rate=0., dp_rate=0.2,
                    att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
                    c_reconstruct=1, apply_gap=True, nb_class=2)

    return TApp


if __name__ == "__main__":
    print("TransApp Pretraining")

    i = int(sys.argv[1])
    dim_model = int(sys.argv[2])

    win = 1024

    list_exo_variable = [[],
                         ['hours_cos', 'hours_sin', 'days_cos', 'days_sin']]

    name_exo_variables = ['None', 'Embed']

    path_results = '/results/PretrainedModels/' + name_exo_variables[i] + '/'
    _ = create_dir(path_results)

    dict_params = {'lr': 1e-4, 'wd': 1e-4, 'batch_size': 16, 'epochs': 20}

    path = path_results + 'TransApp' + str(dim_model)

    m = len(list_exo_variable[i]) + 1

    X_train = CER_get_data_pretraining(exo_variable=list_exo_variable[i])

    model = get_model_inst(m, win, dim_model=dim_model)

    GeomMask = GeometricMask(mean_length=24, masking_ratio=0.5, type_corrupt='zero', dim_masked=0)

    launch_pretraining(model, 
                       path, m, win, 
                       X_train,
                       GeomMask,
                       dict_params)

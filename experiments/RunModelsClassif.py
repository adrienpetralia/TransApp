#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia
# @description : appliance detection experiments on CER dataset
# @component: /experiments/
# @file : RunModelsClassif.py
#
#################################################################################################################

import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

from data_utils import *

from ..src.AD_Framework.Framework import *
from ..src.Models.InceptionTime import *
from ..src.Models.ResNetAtt import *
from ..src.Models.ResNet import *
from ..src.Models.FCN import * 


def process_data(case_name, seed, exo_variable=[], win=1024, ratio_resample=0.8, group='residential', slicing=None):
    df_data_x = pd.read_csv('../data/x_'+group+'_25728.csv').set_index('id_pdl')
    case = pd.read_csv('../data/labels/'+case_name+'.csv').set_index('id_pdl')

    # All curve normalization
    data = pd.DataFrame(StandardScaler().fit_transform(df_data_x.T).T, columns=df_data_x.columns, index=df_data_x.index)

    if case_name=='pluginheater_case':
        data = data.iloc[:, 6672:10991]

    if exo_variable:
        extra = pd.read_csv('../data/ExogeneData/extra_25728.csv')
        extra['date'] = pd.to_datetime(extra['date'])
        if case_name=='pluginheater_case':
            extra = extra.iloc[6672:10991, :]

        data = Add_Exogene_CER(data, df_extra=extra, list_variable=exo_variable, reshape2D=True)
        
    X = pd.merge(data, case, on='id_pdl')

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test_pdl(X, test_size=0.2, 
                                                                                    valid_size=0.2, seed=seed)
    
    X_train_voter = X_train.copy()
    y_train_voter = y_train.copy()
    X_valid_voter = X_valid.copy()
    y_valid_voter = y_valid.copy()
    X_test_voter  = X_test.copy()
    y_test_voter  = y_test.copy()

    ratio = y_train.sum() / (len(y_train) - y_train.sum()) if (len(y_train) - y_train.sum()) > y_train.sum() else (len(y_train) - y_train.sum()) / y_train.sum()
    if ratio > ratio_resample:
        sampling_strategy='auto'
    else:
        sampling_strategy=ratio_resample

    m = 1 + len(exo_variable)

    if slicing=='oversample':
        train_slicer      = WindowOversampler(pickrandomly=False, 
                                              scaler=None, 
                                              seed=seed)
        X_train, y_train = train_slicer.fit_transform(X_train, y_train, win=win)
    else:
        train_slicer      = MTWindowSlicer(equalize_class=True, 
                                           sampling_strategy=sampling_strategy, 
                                           seed=seed)
        X_train, y_train = train_slicer.fit_transform(X_train, y_train, m=m, win=win)

    test_valid_slicer = MTWindowSlicer(equalize_class=False, seed=seed)

    X_valid, y_valid = test_valid_slicer.fit_transform(X_valid, y_valid, m=m, win=win)
    X_test, y_test   = test_valid_slicer.fit_transform(X_test, y_test, m=m, win=win)

    X_train = np.reshape(X_train, (X_train.shape[0], m, X_train.shape[-1]//m))
    X_valid = np.reshape(X_valid, (X_valid.shape[0], m, X_valid.shape[-1]//m))
    X_test  = np.reshape(X_test, (X_test.shape[0], m, X_test.shape[-1]//m))

    returned_tuple = (X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_voter, y_train_voter, X_valid_voter, y_valid_voter, X_test_voter, y_test_voter)

    return returned_tuple


def launch_training(model_inst, save_path, m, win,
                    datas_tuple,
                    dict_params,
                    epoch=20):

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

    train_dataset = TSDataset(X_train, y_train, scaler=None)
    valid_dataset = TSDataset(X_valid, y_valid, scaler=None)
    test_dataset  = TSDataset(X_test, y_test,   scaler=None)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dict_params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True)

    model_trainer = AD_Framework(model_inst,
                                 train_loader=train_loader, valid_loader=valid_loader,
                                 learning_rate=dict_params['lr'], weight_decay=dict_params['wd'],
                                 criterion=nn.CrossEntropyLoss(),
                                 patience_es=dict_params['p_es'], patience_rlr=dict_params['p_rlr'],
                                 f_metrics=getmetrics(),
                                 n_warmup_epochs=dict_params['n_warmup_epochs'],
                                 verbose=True, plotloss=False, 
                                 save_fig=False, path_fig=None,
                                 device="cuda", all_gpu=True,
                                 save_checkpoint=True, path_checkpoint=save_path)

    model_trainer.train(epoch)

    #============ eval last model ============#
    model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1), mask='test_metrics_lastmodel')

    #============ restore best weight and evaluate ============#    
    model_trainer.restore_best_weights()
    model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1))

    #============ find threshold on valid (Deng et al. voter) ============#
    model_trainer.ADFFindBestThreshold(TSDataset(X_valid_voter, y_valid_voter, scaler=None), m=m, win=win)
    th_metric = model_trainer.ADFvoter(TSDataset(X_test_voter, y_test_voter, scaler=None), m=m, win=win)
    print(th_metric)

    #============ find quantile on valid ============#
    model_trainer.ADFFindBestQuantile(TSDataset(X_valid_voter, y_valid_voter, scaler=None), m=m, win=win)
    quant_metric = model_trainer.ADFvoter_proba(TSDataset(X_test_voter, y_test_voter, scaler=None), m=m, win=win)
    print(quant_metric)

    return

if __name__ == "__main__":
    print("Deep Model Comparaison Classif") 

    models = {'ResNet': {'Instance': ResNet, 'lr': 1e-3, 'wd':0, 'batch_size': 32,
                         'p_es': 10, 'p_rlr': 3, 'n_warmup_epochs': 0},
              'Inception': {'Instance': Inception, 'lr': 1e-3, 'wd':0, 'batch_size': 32,
                         'p_es': 10, 'p_rlr': 3, 'n_warmup_epochs': 0},
              'FCN': {'Instance': FCN, 'lr': 1e-3, 'wd':0, 'batch_size': 32,
                      'p_es': 10, 'p_rlr': 3, 'n_warmup_epochs': 0},
              'ResNetAtt': {'Instance': ResNetAtt, 'lr': 0.0002, 'wd':0.5, 'batch_size': 32,
                            'p_es': 10, 'p_rlr': 3, 'n_warmup_epochs': 0},
              'FrameworkResNetAtt': {'Instance': ResNetAtt, 'lr': 0.0002, 'wd':0.5, 'batch_size': 32,
                                      'p_es': 10, 'p_rlr': 3, 'n_warmup_epochs': 0}
             }

    path_results = '../results/'

    list_exo_variable = [[],
                         ['hours_cos', 'hours_sin', 'days_cos', 'days_sin']]
    name_exo_variables = ['None', 'Embed']

    for i, l in enumerate(list_exo_variable):
        path = create_dir(path_results + name_exo_variables[i] + '/' + str(sys.argv[1]) + '/')

        m = len(l) + 1
        win = 1024

        for rd_state in range(0, 3):
            save_path = path + str(sys.argv[2])+'_' + str(rd_state)

            if str(sys.argv[2])=='FrameworkResNetAtt':
                datas_tuple = process_data(str(sys.argv[1]), seed=rd_state, exo_variable=l,
                                           win=win, ratio_resample=0.8, group='residential', 
                                           slicing='oversample')
            else:
                datas_tuple = process_data(str(sys.argv[1]), seed=rd_state, exo_variable=l,
                                           win=win, ratio_resample=0.8, group='residential')
                                       
            launch_training(models[str(sys.argv[2])]['Instance'](in_channels=m), 
                            save_path, m, win,
                            datas_tuple,
                            dict_params=models[str(sys.argv[2])])

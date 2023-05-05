import os, sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from ..src.utils import *

path_cer_data = '' # To complete

def CER_get_data_case(case_name, seed, exo_variable=[], win=1024, ratio_resample=0.8, group='residential'):
    data = pd.read_csv(path_cer_data+'data/x_'+group+'_25728.csv').set_index('id_pdl')
    case = pd.read_csv(path_cer_data+'labels/'+case_name+'.csv').set_index('id_pdl')

    if case_name=='pluginheater_case':
        data = data.iloc[:, 6672:10991]

    if exo_variable:
        extra = pd.read_csv(path_cer_data+'ExogeneData/extra_25728.csv')
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
        equalize_class=False
    else:
        equalize_class=True

    m = 1 + len(exo_variable)

    train_slicer      = MTWindowSlicer(equalize_class=equalize_class, 
                                       sampling_strategy=ratio_resample, 
                                       seed=seed)
    test_valid_slicer = MTWindowSlicer(equalize_class=False, seed=seed)

    X_train, y_train = train_slicer.fit_transform(X_train, y_train, m=m, win=win)
    X_valid, y_valid = test_valid_slicer.fit_transform(X_valid, y_valid, m=m, win=win)
    X_test, y_test   = test_valid_slicer.fit_transform(X_test, y_test, m=m, win=win)

    X_train = np.reshape(X_train, (X_train.shape[0], m, X_train.shape[-1]//m))
    X_valid = np.reshape(X_valid, (X_valid.shape[0], m, X_valid.shape[-1]//m))
    X_test  = np.reshape(X_test,  (X_test.shape[0],  m, X_test.shape[-1]//m))

    returned_tuple = (X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_voter, y_train_voter, X_valid_voter, y_valid_voter, X_test_voter, y_test_voter)

    return returned_tuple


def CER_get_data_pretraining(seed=0, win=1024, exo_variable=[], group='residential', entire_curve_normalization=True):
    data = pd.read_csv(path_cer_data+'data/x_'+group+'_25728.csv').set_index('id_pdl')
    
    if entire_curve_normalization:
        data = pd.DataFrame(StandardScaler().fit_transform(data.T).T, columns=data.columns, index=data.index)

    if exo_variable:
        extra = pd.read_csv(path_cer_data+'ExogeneData/extra_25728.csv')
        extra['date'] = pd.to_datetime(extra['date'])
        data = Add_Exogene_CER(data, df_extra=extra, list_variable=exo_variable, reshape2D=True)
        
    m = 1 + len(exo_variable)

    slicer = MTWindowSlicer(seed=seed)

    X_train = slicer.fit_transform(data.values, m=m, win=win)
    X_train = np.reshape(X_train, (X_train.shape[0], m, X_train.shape[-1]//m))

    return X_train

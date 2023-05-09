import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


root = Path(os.getcwd()).resolve().parents[1]
sys.path.append(str(root))
from experiments.data_utils import *
from src.AD_Framework.Framework import *

from sklearn.preprocessing import StandardScaler
from sktime.classification.kernel_based import Arsenal, RocketClassifier



def launch_training(clf,
                    len_i,
                    in_framework,
                    n_indiv,
                    save_path, m, win,
                    datas_tuple):

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
    X_test_voter = datas_tuple[6]
    y_test_voter = datas_tuple[7]

    X_test_voter = X_test_voter[:n_indiv].reshape(n_indiv, m, X_test_voter[:n_indiv].shape[-1]//m)[:, :, :len_i]
    y_test_voter = y_test_voter[:n_indiv]
    if in_framework:
        X_test_voter = X_test_voter.reshape(n_indiv, X_test_voter.shape[1] * X_test_voter.shape[2])
        X_train = X_train[:2]
        y_train = np.array([0,1])
    else:
        X_train = np.squeeze(X_train_voter[:2].reshape(2, m, X_train_voter[:2].shape[-1]//m)[:, :1, :len_i])
        y_train = np.array([0,1])

        X_test = np.squeeze(X_test_voter[:, :1, :], axis=1) 
        y_test = y_test_voter.ravel()

    dapframework = DAP_Framework_Sktime(clf,
                                        f_metrics=accuracymetrics(),
                                        verbose=True, save_model=False,
                                        save_checkpoint=True, path_checkpoint=save_path)

    dapframework.train(X_train=X_train, y_train=y_train)

    if not in_framework:
        #============ model time inference ============#
        dapframework.evaluate(X_test, y_test)
        print("Model time inference :", dapframework.log['test_time'])
    else:
        #============ framework time inference ============#
        dapframework.DAPvoter_proba(TSDataset(X_test_voter, y_test_voter), m=m, win=win)
        print("Framework time inference :", dapframework.log['test_probavoter_time'])

    return

if __name__ == "__main__":
    print("Inference Time Study Sktime")

    model_name = str(sys.argv[1])
    in_framework = bool(int(sys.argv[2]))

    print("In Framework :", in_framework)
    print("Model Name :", model_name)

    if in_framework:
        path_results = str(root) + '/results/AblationStudy/TestingTime/InFramework/'
    else:
        path_results = str(root) + '/results/AblationStudy/TestingTime/OutFramework/'

    win = 1024
    m = 5

    datas_tuple = CER_get_data_case("cooker_case", seed=0, exo_variable=['hours_cos', 'hours_sin', 'days_cos', 'days_sin'],
                                    win=win, ratio_resample=0.8, group='residential')

    list_len_i = [1024, 2048, 4096, 8192, 16384, 25728]

    for n_indiv in [1, 10, 100, 1000]:
        for len_i in list_len_i:
            if model_name=='Arsenal':
                clf = Arsenal(rocket_transform='rocket', n_jobs=-1)
            else:
                clf = RocketClassifier(num_kernels=10000, rocket_transform='rocket', n_jobs=-1)

            save_path = path_results + model_name + '_' + str(len_i) + '_' + str(n_indiv)

            launch_training(clf, 
                            len_i,
                            in_framework,
                            n_indiv,
                            save_path, m, win,
                            datas_tuple)
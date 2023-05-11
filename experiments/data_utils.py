import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

from src.utils import *

path_data = str(Path(os.getcwd()).resolve().parents[0]) + '/data/'

def CER_get_data_case(case_name, seed, exo_variable=[], win=1024, ratio_resample=0.8):
    data = pd.read_csv(path_data+'Inputs/x_residential_25728.csv').set_index('id_pdl')
    case = pd.read_csv(path_data+'Labels/'+case_name+'.csv').set_index('id_pdl')

    if case_name=='pluginheater_case':
        data = data.iloc[:, 6672:10991]

    if exo_variable:
        extra = pd.read_csv(path_data+'ExogeneData/extra_25728.csv')
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


def CER_get_data_pretraining(seed=0, win=1024, exo_variable=[], entire_curve_normalization=True):
    data = pd.read_csv(path_data+'Inputs/x_residential_25728.csv').set_index('id_pdl')
    
    if entire_curve_normalization:
        data = pd.DataFrame(StandardScaler().fit_transform(data.T).T, columns=data.columns, index=data.index)

    if exo_variable:
        extra = pd.read_csv(path_data+'ExogeneData/extra_25728.csv')
        extra['date'] = pd.to_datetime(extra['date'])
        data = Add_Exogene_CER(data, df_extra=extra, list_variable=exo_variable, reshape2D=True)
        
    m = 1 + len(exo_variable)

    slicer = MTWindowSlicer(seed=seed)

    X_train = slicer.fit_transform(data.values, m=m, win=win)
    X_train = np.reshape(X_train, (X_train.shape[0], m, X_train.shape[-1]//m))

    return X_train


def create_dir(path):
    try:
        os.mkdir(path)
    except:
        pass
    return path

def check_file_exist(path):
    return os.path.isfile(path)


def Add_Exogene_CER(df_data, df_extra, list_variable, reshape2D=True):
    """
        Add Exogene data to CER load curves
        
        Return : 3D np.ndarray of size [N, m, win] if reshape2D = False
                 or
                 2D pd.core.frame.DataFrame instance [N, m * win] if reshape2D = True
    """
    
    m = len(list_variable) + 1
    tmp_extra = df_extra[list_variable].values.T
    tmp = np.zeros((df_data.values.shape[0], m, df_data.values.shape[1]))
    
    for i in range(len(tmp)):
        tmp[i, 0, :] = df_data.values[i, :]
        for j in range(len(list_variable)):
            tmp[i, j+1, :] = tmp_extra[j, :]
    
    if reshape2D:
        tmp = tmp.reshape(tmp.shape[0], -1)
        all_data = pd.DataFrame(data=tmp, index=df_data.index)
        return all_data
    else:
        return tmp

    
def split_train_valid_test_pdl(df_data, test_size=0.2, valid_size=0, nb_label_col=1, seed=0, return_df=False):
    """
    Split DataFrame based on index ID (ID PDL for example)
    
    - Input : df_data -> DataFrame
              test_size -> Percentage data for test
              valid_size -> Percentage data for valid
              nb_label_col -> Number of columns of label
              seed -> Set seed
              return_df -> Return DataFrame instances, or Numpy Instances
    - Output:
            np.arrays or DataFrame Instances
    """

    np.random.seed(seed)
    list_pdl = np.array(df_data.index.unique())
    np.random.shuffle(list_pdl)
    pdl_train_valid = list_pdl[:int(len(list_pdl) * (1-test_size))]
    pdl_test = list_pdl[int(len(list_pdl) * (1-test_size)):]
    np.random.shuffle(pdl_train_valid)
    pdl_train = pdl_train_valid[:int(len(pdl_train_valid) * (1-valid_size))]
    
    df_train = df_data.loc[pdl_train, :].copy()
    df_test = df_data.loc[pdl_test, :].copy()
    

    df_train = df_train.sample(frac=1, random_state=seed)
    df_test = df_test.sample(frac=1, random_state=seed)
    
    if valid_size != 0:
        pdl_valid = pdl_train_valid[int(len(pdl_train_valid) * (1-valid_size)):]
        df_valid = df_data.loc[pdl_valid, :].copy()
        df_valid = df_valid.sample(frac=1, random_state=seed)
            
    if return_df:
        if valid_size != 0:
            return df_train, df_valid, df_test
        else:
            return df_train, df_test
    else:
        X_train = df_train.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y_train = df_train.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)
        X_test  = df_test.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y_test  = df_test.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)

        if valid_size != 0:
            X_valid = df_valid.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
            y_valid = df_valid.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            return X_train, y_train, X_test, y_test    
    

def RandomUnderSampler_(X, y=None, sampling_strategy='auto', seed=0, nb_label=1):
    np.random.seed(seed)
    
    if isinstance(X, pd.core.frame.DataFrame):
        col = X.columns
        y = X.values[:, -nb_label].astype(int)
        X = X.values[:, :-nb_label]
        X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
        Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
        Mat = pd.DataFrame(data=Mat, columns=col)
        Mat = Mat.sample(frac=1, random_state=seed)
        
        return Mat
    else:
        assert y is not None, f"For np.array, please provide an y vector."
        X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
        Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
        np.random.shuffle(Mat)
        Mat = Mat.astype(np.float32)
        
        return Mat[:, :-1], Mat[:, -1]
    
    
    
class MTWindowSlicer():
    def __init__(self, equalize_class=False, sampling_strategy='auto', nb_window_per_obs=None, seed=0):
        self.equalize_class=equalize_class
        self.sampling_strategy = sampling_strategy
        self.nb_window_per_obs=nb_window_per_obs
        self.seed=seed
        
        self.undersampler = RandomUnderSampler(random_state=self.seed, sampling_strategy=sampling_strategy)
    
    def fit_transform(self, X_input, y_input=None, m=1, win=100):
        if isinstance(X_input, pd.core.frame.DataFrame):
            X_input = X_input.values
            
        if y_input is not None:

            if isinstance(y_input, pd.core.frame.DataFrame):
                y_input = y_input.values

            if len(y_input.shape)==1:
                y_input = np.expand_dims(y_input, axis=-1)
            
        return self._non_random(X_input, y_input, m, win=win)
    
    def _non_random(self, X_input, y_input, m, win):
        np.random.seed(self.seed)
        
        X_input = np.reshape(X_input, (X_input.shape[0], m, int(X_input.shape[1]/m)))

        if y_input is not None:
            self.nb_window_per_obs = X_input.shape[-1] // win
            
            ind_0, _ = np.where(y_input==0)
            ind_1, _ = np.where(y_input==1)
            
            X_0, X_1 = X_input[ind_0, :, :self.nb_window_per_obs*win], X_input[ind_1, :, :self.nb_window_per_obs*win]
            X_0, X_1 = self._MTSlicew(X_0, self.nb_window_per_obs, m, win), self._MTSlicew(X_1, self.nb_window_per_obs, m, win)
            y_0, y_1 = np.repeat(0, X_0.shape[0]), np.repeat(1, X_1.shape[0])
            X_0, X_1 = np.concatenate((X_0, np.reshape(y_0, (y_0.shape[0],  1))), axis=1), np.concatenate((X_1, np.reshape(y_1, (y_1.shape[0],  1))), axis=1)

            Mat = np.concatenate((X_0, X_1))
            np.random.shuffle(Mat)
            Mat = Mat.astype(np.float32)

            if self.equalize_class:
                return self._undersample(Mat[:, :-1], Mat[:, -1])
            else:
                return Mat[:, :-1], Mat[:, -1]

        else:
            if self.nb_window_per_obs is None:
                self.nb_window_per_obs = X_input.shape[-1] // win
                Mat = X_input[:, :, :self.nb_window_per_obs*win]
                np.random.shuffle(Mat)
                Mat = self._MTSlicew(Mat, self.nb_window_per_obs, m, win)
                
            else:
                Mat = np.empty((X_input.shape[0] * self.nb_window_per_obs, m, win))
                overlap_win = (X_input.shape[-1]-win) // self.nb_window_per_obs
                cpt=0
                for k in range(m):
                    for i in range(X_input.shape[0]):
                        for j in range(self.nb_window_per_obs):
                            Mat[cpt, m, :] = X_input[i, m, j*overlap_win:j*overlap_win + win]
                            cpt += 1
            return Mat
    
    def _undersample(self, X, y):
        np.random.seed(self.seed)
        X_, y_ = self.undersampler.fit_resample(X, y)
        Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
        np.random.shuffle(Mat)
        Mat = Mat.astype(np.float32)
        
        return Mat[:, :-1], Mat[:, -1]
    
    def _MTSlicew(self, X, nb_window_per_obs, m, win):
        output = np.zeros((nb_window_per_obs * X.shape[0], m, win))

        for i in range(m):
            output[:, i, :] = np.reshape(X[:, i, :], (nb_window_per_obs * X.shape[0], win))

        return np.reshape(output, (output.shape[0], -1))
    
    
    
class WindowOversampler():
    def __init__(self, pickrandomly=True, scaler=None, seed=0):
        self.pickrandomly = pickrandomly
        self.scaler = scaler
        self.seed = seed
        
    def fit_transform(self, X_input, y_input, win=100, overlap_win=None):
        np.random.seed(self.seed)
        
        if self.scaler is not None:
            X_input = self.scaler.fit_transform(X_input.T).T

        if len(np.unique(y_input)) != 2:
            raise ValueError('Oversampling with sliding window only implemented for binary case, please provide y with 0 and 1 only. Found {} differents labels values.'
                              .format(len(np.unique(y_input))))

        ratio = np.count_nonzero(y_input == 0) / (np.count_nonzero(y_input == 1) + np.count_nonzero(y_input == 0))

        
        if ratio > 0.95 or ratio < 0.05:
            raise ValueError('Calculated unbalance ratio : {}. Unbalance ratio too high, please undersample majority class before.'
                             .format(ratio))
        #elif ratio > 0.43 and ratio < 0.57:
        #    return ValueError('Use slicing.')
        else:
            if np.count_nonzero(y_input == 0) > np.count_nonzero(y_input == 1):
                label_minor = 1
                label_major = 0
            else:
                label_minor = 0
                label_major = 1  

            ind_minor, _ = np.where(y_input == label_minor)
            ind_major, _ = np.where(y_input == label_major)
            X_major = X_input[ind_major, :]
            X_minor = X_input[ind_minor, :]
            
            nb_window_per_obs = X_input.shape[1] // win
            
            if self.pickrandomly:
                
                X_tmp_major = np.empty((X_major.shape[0] * nb_window_per_obs, win+1))
                cpt=0
                for i in range(X_major.shape[0]):
                    for j in range(nb_window_per_obs):
                        ids = np.random.randint(X_major.shape[1]-win, size=1)[0]
                        X_tmp_major[cpt, :-1] = X_major[i, ids:ids+win]
                        X_tmp_major[cpt, -1] = label_major
                        cpt+=1

                nb_window_per_obs_min = (X_major.shape[0] // X_minor.shape[0]) * nb_window_per_obs
                X_tmp_minor = np.empty((X_minor.shape[0] * nb_window_per_obs_min, win+1))
                cpt=0
                for i in range(X_minor.shape[0]):
                    for j in range(nb_window_per_obs_min):
                        ids = np.random.randint(X_minor.shape[1]-win, size=1)[0]
                        X_tmp_minor[cpt, :-1] = X_minor[i, ids:ids+win]
                        X_tmp_minor[cpt, -1] = label_minor
                        cpt+=1
                        
                Mat = np.concatenate((X_tmp_major, X_tmp_minor))
                np.random.shuffle(Mat)
                Mat = Mat.astype(np.float32)
                
                return Mat[:, :-1], Mat[:, -1]
                        
            else:
                n_ma = X_major.shape[0]
                n_mi = X_minor.shape[0]
                X_major = X_major[:, :nb_window_per_obs*win]
                X_major = np.reshape(X_major, (nb_window_per_obs * n_ma, win))

                if overlap_win is None:
                    overlap_win = int((n_mi / n_ma) * win)
                n_overlap_win = ((X_input.shape[1] - win) // overlap_win) + 1

                cpt = 0
                X_tmp = np.empty((n_mi*n_overlap_win, win))
                for i in range(n_mi):
                    for j in range(n_overlap_win):
                        X_tmp[cpt, :] = X_minor[i, j*overlap_win:j*overlap_win + win]
                        cpt += 1

                y_minor = np.repeat(label_minor, X_tmp.shape[0])
                y_major = np.repeat(label_major, X_major.shape[0])
                X_minor = np.concatenate((X_tmp, np.reshape(y_minor, (y_minor.shape[0],  1))), axis=1)
                X_major = np.concatenate((X_major, np.reshape(y_major, (y_major.shape[0],  1))), axis=1)

                Mat = np.concatenate((X_major, X_minor))
                np.random.shuffle(Mat)
                Mat = Mat.astype(np.float32)
                
                return Mat[:, :-1], Mat[:, -1]

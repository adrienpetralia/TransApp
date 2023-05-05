#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia
# @description : Appliance Detection Framework implementation
# @component: src/AD_Framework/
# @file : Framework.py
#
#################################################################################################################

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score


# ====================================== Pytorch TS Dataset Class ====================================== #

class TSDataset(torch.utils.data.Dataset):
    """
    MAP-Style PyTorch Time series Dataset with possibility of scaling
    
    - X matrix of TS input, can be 2D or 3D, Dataframe instance or Numpy array instance.
    - Labels : y labels associated to time series for classification. Possible to be None.
    - scaler : provided type of scaler (sklearn StandardScaler, MinMaxScaler instance for example).
    - scale_dim : list of dimensions to be scaled in case of multivariate TS.
    """
    def __init__(self, X, labels=None, scaler=False, scale_dim=None):
        
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(labels, pd.core.frame.DataFrame):
            labels = labels.values
        
        if scaler:
            # ==== Multivariate case ==== #
            if len(X.shape)==3:
                self.scaler_list = []
                self.samples = X
                if scale_dim is None:                    
                    for i in range(X.shape[1]):
                        self.scaler_list.append(StandardScaler())
                        self.samples[:,i,:] = self.scaler_list[i].fit_transform(X[:,i,:].T).T.astype(np.float32)
                else:
                    for idsc, i in enumerate(scale_dim):
                        self.scaler_list.append(StandardScaler())
                        self.samples[:,i,:] = self.scaler_list[idsc].fit_transform(X[:,i,:].T).T.astype(np.float32)
                        
            # ==== Univariate case ==== #
            else:
                self.scaler_list = [StandardScaler()]
                self.samples = self.scaler_list[0].fit_transform(X.T).T.astype(np.float32)
        else:
            self.samples = X
            
        if len(self.samples.shape)==2:
            self.samples = np.expand_dims(self.samples, axis=1)
        
        if labels is not None:
            self.labels = labels.ravel()
            assert len(self.samples)==len(self.labels), f"Number of X sample {len(self.samples)} doesn't match number of y sample {len(self.labels)}."
        else:
            self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        if self.labels is None:
            return self.samples[idx]
        else:
            return self.samples[idx], self.labels[idx]
        
        
        

# ====================================== Imbalance Metric Class ====================================== #        
class getmetrics():
    """
    Basics metrics class for imbalance classification
    """
    def __init__(self, minority_class=None):
        self.minority_class = minority_class
        
    def __call__(self, y, y_hat, y_hat_prob=None):
        metrics = {}

        if self.minority_class is not None:
            minority_class = self.minority_class                
        else:
            y_label = np.unique(y)

            if np.count_nonzero(y==y_label[0]) > np.count_nonzero(y==y_label[1]):
                minority_class = y_label[1]
            else :
                minority_class = y_label[0]

        metrics['ACCURACY'] = accuracy_score(y, y_hat)
        
        metrics['PRECISION'] = precision_score(y, y_hat, pos_label=minority_class, average='binary')
        metrics['RECALL'] = recall_score(y, y_hat, pos_label=minority_class, average='binary')
        metrics['PRECISION_MACRO'] = precision_score(y, y_hat, average='macro')
        metrics['RECALL_MACRO'] = recall_score(y, y_hat, average='macro')
        
        metrics['F1_SCORE'] = f1_score(y, y_hat, pos_label=minority_class, average='binary')
        metrics['F1_SCORE_MACRO'] = f1_score(y, y_hat, average='macro')
        metrics['F1_SCORE_WEIGHTED'] = f1_score(y, y_hat, average='weighted')
        
        metrics['CONFUSION_MATRIX'] = confusion_matrix(y, y_hat)
        
        if y_hat_prob is not None:
            metrics['ROC_AUC_SCORE'] = roc_auc_score(y, y_hat_prob)
            metrics['ROC_AUC_SCORE_MACRO'] = roc_auc_score(y, y_hat_prob, average='macro')
            metrics['ROC_AUC_SCORE_WEIGHTED'] = roc_auc_score(y, y_hat_prob, average='weighted')

        return metrics
          
          
          
# ====================================== Pytorch Pretrainer Class ====================================== #
class self_pretrainer(object):
    def __init__(self,
                 model,
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=0,
                 name_scheduler=None,
                 dict_params_scheduler=None,
                 warmup_duration=None,
                 criterion=nn.MSELoss(), mask=None, loss_in_model=False,
                 device="cuda", all_gpu=False,
                 verbose=True, plotloss=True, 
                 save_fig=False, path_fig=None,
                 save_only_core=False,
                 save_checkpoint=False, path_checkpoint=None):

        # =======================class variables======================= #
        self.device = device
        self.all_gpu = all_gpu
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.mask = mask
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.save_only_core = save_only_core
        self.loss_in_model = loss_in_model
        self.name_scheduler = name_scheduler
        
        if name_scheduler is None:
            self.scheduler = None
        else:
            assert isinstance(dict_params_scheduler, dict)
            
            if name_scheduler=='MultiStepLR':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=dict_params_scheduler['milestones'], gamma=dict_params_scheduler['gamma'], verbose=self.verbose)

            elif name_scheduler=='CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=dict_params_scheduler['T_max'], eta_min=dict_params_scheduler['eta_min'], verbose=self.verbose)

            elif name_scheduler=='CosineAnnealingWarmRestarts':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=dict_params_scheduler['T_0'], T_mult=dict_params_scheduler['T_mult'], eta_min=dict_params_scheduler['eta_min'], verbose=self.verbose)

            elif name_scheduler=='ExponentialLR':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=dict_params_scheduler['gamma'], verbose=self.verbose)

            else:
                raise ValueError('Type of scheduler {} unknown, only "MultiStepLR", "ExponentialLR", "CosineAnnealingLR" or "CosineAnnealingWarmRestarts".'.format(encoding_type))
        
        #if warmup_duration is not None:
        #    self.scheduler = create_lr_scheduler_with_warmup(scheduler,
        #                                                     warmup_start_value=1e-7,
        #                                                     warmup_end_value=learning_rate,
        #                                                     warmup_duration=warmup_duration)
        #else:
        #    self.scheduler = scheduler
            
        if self.all_gpu:
            # ===========dummy forward to intialize Lazy Module=========== #
            for ts in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # ===========data Parrallel Module call=========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model'

        self.log = {}
        self.train_time = 0
        self.passed_epochs = 0
        self.loss_train_history = []
        self.loss_valid_history = []
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        t = time.time()
        for epoch in range(n_epochs):
            # =======================one epoch===================== #
            train_loss = self.__train(epoch)
            self.loss_train_history.append(train_loss)
            
            if self.valid_loader is not None:
                valid_loss = self.__evaluate()
                self.loss_valid_history.append(valid_loss)

            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
                print('    Train loss : {:.6f}'.format(train_loss))
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.6f}'.format(valid_loss))
            
            if epoch%5==0 or epoch==n_epochs-1:
                # =========================log========================= #
                if self.save_only_core:
                    self.log = {'model_state_dict': self.model.module.core.state_dict() if self.device=="cuda" and self.all_gpu else self.model.core.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss_train_history': self.loss_train_history,
                                'loss_valid_history': self.loss_valid_history,
                                'time': (time.time() - t)
                               }
                else:
                    self.log = {'model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss_train_history': self.loss_train_history,
                                'loss_valid_history': self.loss_valid_history,
                                'time': (time.time() - t)
                               }
                    
                if self.save_checkpoint:
                    self.save()
                    
            if self.scheduler is not None: 
                if self.name_scheduler!='CosineAnnealingWarmRestarts':   
                    self.scheduler.step()
                
            self.passed_epochs+=1
            
        self.train_time = round((time.time() - t), 3)
        
        if self.save_checkpoint:
            self.save()

        if self.plotloss:
            self.plot_history()
            
        return
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    
    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.save_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr
        return

    def __train(self, epoch):
        """
        Private function : model training loop over data loader
        """
        loss_train = 0
        iters = len(self.train_loader)
        
        for i, ts in enumerate(self.train_loader):
            self.model.train()
            # ===================variables=================== #
            ts = Variable(ts.float())
            if self.mask is not None:
                mask_loss, ts_masked = self.mask(ts)
            # ===================forward===================== #
            self.optimizer.zero_grad()
            if self.mask is not None:
                outputs = self.model(ts_masked.to(self.device))
                loss    = self.criterion(outputs, ts.to(self.device), mask_loss.to(self.device))
            else:
                if self.loss_in_model:
                    outputs, loss = self.model(ts.to(self.device))
                else:
                    outputs = self.model(ts.to(self.device))
                    loss    = self.criterion(outputs, ts.to(self.device))
            # ===================backward==================== #              
            loss.backward()
            self.optimizer.step()
            loss_train += loss.item()
            
            if self.name_scheduler=='CosineAnnealingWarmRestarts':
                self.scheduler.step(epoch + i / iters)

        loss_train = loss_train / len(self.train_loader)
        return loss_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        loss_valid = 0
        with torch.no_grad():
            for ts in self.valid_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = Variable(ts.float())
                if self.mask is not None:
                    mask_loss, ts_masked = self.mask(ts)
                # ===================forward===================== #
                if self.mask is not None:
                    outputs = self.model(ts_masked.to(self.device))
                    loss    = self.criterion(outputs, ts.to(self.device), mask_loss.to(self.device))
                else:
                    if self.loss_in_model:
                        outputs, loss = self.model(ts.to(self.device))
                    else:
                        outputs = self.model(ts.to(self.device))
                        loss    = self.criterion(outputs, ts.to(self.device))
                loss_valid += loss.item()

        loss_valid = loss_valid / len(self.valid_loader)
        return loss_valid

      
      
      
# ====================================== Pytorch Framework Implementation ====================================== #
class BasedClassifTrainer(object):
    def __init__(self,
                 model, 
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=1e-2, 
                 criterion=nn.CrossEntropyLoss(),
                 patience_es=None, patience_rlr=None,
                 device="cuda", all_gpu=False,
                 valid_criterion=None,
                 n_warmup_epochs=0,
                 f_metrics=getmetrics(),
                 verbose=True, plotloss=True, 
                 save_fig=False, path_fig=None,
                 save_checkpoint=False, path_checkpoint=None):
        """
        PyTorch Parent Class : Model Trainer for classification case
        """

        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.f_metrics = f_metrics
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        self.scheduler = None
        
        self.train_criterion = criterion
        if valid_criterion is None:
            self.valid_criterion = criterion
        else:
            self.valid_criterion = valid_criterion
        
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model' 
            
        if patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                        patience=patience_rlr, 
                                                                        verbose=self.verbose,
                                                                        eps=1e-7)
            
        #if n_warmup_epochs > 0 and self.scheduler is not None:
        #    self.scheduler = create_lr_scheduler_with_warmup(self.scheduler,
        #                                                     warmup_start_value=1e-6,
        #                                                     warmup_end_value=learning_rate,
        #                                                     warmup_duration=n_warmup_epochs)

        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.voter_time = 0
        self.passed_epochs = 0
        self.best_loss = np.Inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []
               
        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module if all GPU used =========== #
            self.model.to("cpu")
            for ts, _ in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        
        #flag_es = 0
        tmp_time = time.time()
        
        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            train_loss, train_accuracy = self.__train()
            self.loss_train_history.append(train_loss)
            self.accuracy_train_history.append(train_accuracy)
            if self.valid_loader is not None:
                valid_loss, valid_accuracy = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
                self.accuracy_valid_history.append(valid_accuracy)
            else:
                valid_loss = train_loss
                
            # =======================reduce lr======================= #
            if self.scheduler:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if self.passed_epochs > self.n_warmup_epochs: # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        #flag_es  = 1
                        es_epoch = epoch+1
                        self.passed_epochs+=1
                        if self.verbose:
                            print('Early stopping after {} epochs !'.format(epoch+1))
                        break
        
            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch+1, n_epochs))
                print('    Train loss : {:.4f}, Train acc : {:.2f}%'
                          .format(train_loss, train_accuracy*100))
                
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.4f}, Valid  acc : {:.2f}%'
                              .format(valid_loss, valid_accuracy*100))

            # =======================save log======================= #
            if valid_loss <= self.best_loss and self.passed_epochs>=self.n_warmup_epochs:
                self.best_loss = valid_loss
                self.log = {'valid_metrics': valid_accuracy if self.valid_loader is not None else train_accuracy,
                            'model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss_train_history': self.loss_train_history,
                            'loss_valid_history': self.loss_valid_history,
                            'accuracy_train_history': self.accuracy_train_history,
                            'accuracy_valid_history': self.accuracy_valid_history,
                            'value_best_loss': self.best_loss,
                            'epoch_best_loss': self.passed_epochs,
                            'time_best_loss': round((time.time() - tmp_time), 3),
                            }
                if self.save_checkpoint:
                    self.save()
                
            self.passed_epochs+=1
                    
        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()
            
        if self.save_checkpoint:
            self.log['best_model_state_dict'] = torch.load(self.path_checkpoint+'.pt')['model_state_dict']
        
        # =======================update log======================= #
        self.log['training_time'] = self.train_time
        self.log['loss_train_history'] = self.loss_train_history
        self.log['loss_valid_history'] = self.loss_valid_history
        self.log['accuracy_train_history'] = self.accuracy_train_history
        self.log['accuracy_valid_history'] = self.accuracy_valid_history
        
        #if flag_es != 0:
        #    self.log['final_epoch'] = es_epoch
        #else:
        #    self.log['final_epoch'] = n_epochs
        
        if self.save_checkpoint:
            self.save()
        return
    
    def evaluate(self, test_loader, mask='test_metrics', return_output=False):
        """
        Public function : model evaluation on test dataset
        """
        tmp_time = time.time()
        mean_loss_eval = []
        y = np.array([])
        y_hat = np.array([])
        with torch.no_grad():
            for ts, labels in test_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = Variable(ts.float()).to(self.device)
                labels = Variable(labels.float()).to(self.device)
                # ===================forward===================== #
                logits = self.model(ts)
                loss = self.valid_criterion(logits.float(), labels.long())
                # =================concatenate=================== #
                _, predicted = torch.max(logits, 1)
                mean_loss_eval.append(loss.item())
                y_hat = np.concatenate((y_hat, predicted.detach().cpu().numpy())) if y_hat.size else predicted.detach().cpu().numpy()
                y = np.concatenate((y, torch.flatten(labels).detach().cpu().numpy())) if y.size else torch.flatten(labels).detach().cpu().numpy()
                
        metrics = self._apply_metrics(y, y_hat)
        self.eval_time = round((time.time() - tmp_time), 3)
        self.log['eval_time'] = self.eval_time
        self.log[mask] = metrics
        
        if self.save_checkpoint:
            self.save()
        
        if return_output:
            return np.mean(mean_loss_eval), metrics, y, y_hat
        else:
            return np.mean(mean_loss_eval), metrics
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    
    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g['lr'] = new_lr
        return
            
    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log['best_model_state_dict'])
            else:
                self.model.load_state_dict(self.log['best_model_state_dict'])
            print('Restored best model met during training.')
        except KeyError:
            print('Error during loading log checkpoint state dict : no update.')
        return
    
    def __train(self):
        """
        Private function : model training loop over data loader
        """
        total_sample_train = 0
        mean_loss_train = []
        mean_accuracy_train = []
        
        for ts, labels in self.train_loader:
            self.model.train()
            # ===================variables=================== #
            ts = Variable(ts.float()).to(self.device)
            labels = Variable(labels.float()).to(self.device)
            # ===================forward===================== #
            self.optimizer.zero_grad()
            logits = self.model(ts)
            # ===================backward==================== #
            loss_train = self.train_criterion(logits.float(), labels.long())
            loss_train.backward()
            self.optimizer.step()
            # ================eval on train================== #
            total_sample_train += labels.size(0)
            _, predicted_train = torch.max(logits, 1)
            correct_train = (predicted_train.to(self.device) == labels.to(self.device)).sum().item()
            mean_loss_train.append(loss_train.item())
            mean_accuracy_train.append(correct_train)
            
        return np.mean(mean_loss_train), np.sum(mean_accuracy_train)/total_sample_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        total_sample_valid = 0
        mean_loss_valid = []
        mean_accuracy_valid = []
        
        with torch.no_grad():
            for ts, labels in self.valid_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = Variable(ts.float()).to(self.device)
                labels = Variable(labels.float()).to(self.device)
                logits = self.model(ts)
                loss_valid = self.valid_criterion(logits.float(), labels.long())
                # ================eval on test=================== #
                total_sample_valid += labels.size(0)
                _, predicted = torch.max(logits, 1)
                correct = (predicted.to(self.device) == labels.to(self.device)).sum().item()
                mean_loss_valid.append(loss_valid.item())
                mean_accuracy_valid.append(correct)

        return np.mean(mean_loss_valid), np.sum(mean_accuracy_valid)/total_sample_valid
    
    def _apply_metrics(self, y, y_hat, y_hat_prob=None):
        """
        Private function : apply provided metrics
        
        !!! Provided metric function must be callable !!!
        """
        if y_hat_prob is not None:
            return self.f_metrics(y, y_hat, y_hat_prob)
        else:
            return self.f_metrics(y, y_hat)   
    
    
class AD_Framework(BasedClassifTrainer):
    """
    Appliance Detection Framework class : child of BasedClassifTrainer
    
    Class Based on BasedClassifTrainer : 
    This class is made for training/testing and evaluating a deep Pytorch model on binary appliance classification cases
    """
    def __init__(self,
                 model, 
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=1e-2,
                 criterion=nn.CrossEntropyLoss(),
                 patience_es=None, patience_rlr=None,
                 device="cuda", all_gpu=False,
                 valid_criterion=None,
                 n_warmup_epochs=0,
                 f_metrics=getmetrics(),
                 verbose=True, plotloss=True, 
                 save_fig=False, path_fig=None,
                 save_checkpoint=False, path_checkpoint=None,
                 scale_by_subseq_in_voter=False, scale_dim=[0],
                 batch_size_voter=1):
        """
        Appliance Detection Framework Class
        """
        super().__init__(model=model, 
                         train_loader=train_loader, valid_loader=valid_loader,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         criterion=criterion,
                         patience_es=patience_es, patience_rlr=patience_rlr,
                         device=device, all_gpu=all_gpu,
                         valid_criterion=valid_criterion,
                         n_warmup_epochs=n_warmup_epochs,
                         f_metrics=f_metrics,
                         verbose=verbose, plotloss=plotloss, 
                         save_fig=save_fig, path_fig=path_fig,
                         save_checkpoint=save_checkpoint, path_checkpoint=path_checkpoint)

        self.scale_by_subseq_in_voter = scale_by_subseq_in_voter
        self.scale_dim = scale_dim
        self.batch_size_voter = batch_size_voter
    
    def ADFvoter(self, dataset_voter, win, m=1, 
                 mask='test_voter_metrics',
                 mask_time='test_voter_time', 
                 mask1='bestthreshold_valid_voter_metrics',
                 mask2='Threshold_voter',
                 n_best_pred=None, threshold=None, return_output=False):
        
        if threshold is None:
            try:
                threshold = self.log[mask1][mask2]
            except:
                warnings.warn("No Threshold provided and No optimized threshold found, set Threshold to 0.5 for the voter predictions.")
                threshold = 0.5
        
        tmp_time = time.time()
        
        if isinstance(dataset_voter, pd.core.frame.DataFrame):
            y, y_hat = self._ADFvoter_df(dataset_voter=dataset_voter, m=m, win=win, 
                                         threshold=threshold, n_best_pred=n_best_pred)
        else:
            y, y_hat = self._ADFvoter(dataset_voter=dataset_voter, m=m, win=win,
                                      threshold=threshold, n_best_pred=n_best_pred)

        metrics = self._apply_metrics(y, y_hat)
        self.voter_time = round((time.time() - tmp_time), 3)
        self.log[mask_time] = self.voter_time
        self.log[mask] = metrics

        if self.save_checkpoint:
            self.save()

        if return_output:
            return metrics, y, y_hat
        else:
            return metrics
        
        
    def ADFvoter_proba(self, dataset_voter, m, win, average_mode='quantile', q=None, threshold=None,
                       mask='test_voterproba_metrics', 
                       mask_time='test_probavoter_time', 
                       mask1='bestquantile_valid_voter_metrics',
                       mask2='quantile',
                       return_output=False): 
        tmp_time = time.time()
        
        if average_mode=='quantile':
            if q is None:
                try:
                    q = self.log[mask1][mask2]
                except:
                    warnings.warn('Average mode "quantile" but no optimize q found and no parameter q provided, set q=0.5 (median).')
                    q = 0.5
        if average_mode!='quantile' and average_mode!='mean':
            raise ValueError('Only "mean" and "quantile" average mode arguments supported for voter proba, but got = {}'
                              .format(average_mode))
    
        if isinstance(dataset_voter, pd.core.frame.DataFrame):
            y, y_hat, y_hat_prob = self._ADFvoterproba_df(dataset_voter=dataset_voter, m=m, win=win, average_mode=average_mode,
                                                          q=q, threshold=threshold)
        else:
            y, y_hat, y_hat_prob = self._ADFvoterproba(dataset_voter=dataset_voter, m=m, win=win, average_mode=average_mode,
                                                       q=q, threshold=threshold)
                
        metrics = self._apply_metrics(y, y_hat, y_hat_prob)
        self.voter_time = round((time.time() - tmp_time), 3)
        self.log[mask_time] = self.voter_time
        self.log[mask] = metrics
        
        if self.save_checkpoint:
            self.save()

        if return_output:
            return metrics, y, y_hat, y_hat_prob
        else:
            return metrics
        
    
    def _ADFvoterproba_df(self, dataset_voter, m, win, average_mode, q, threshold): 
    
        y = []
        y_hat = []
        y_hat_prob = []

        list_index = dataset_voter.index.unique()
                
        for i, id_pdl in enumerate(list_index):
            tmp_data = dataset_voter.loc[id_pdl].copy()
    
            if len(tmp_data.shape)==1:
                tmp_data = np.reshape(tmp_data.values, (1, len(tmp_data.values)))
            else:
                tmp_data = tmp_data.values
            inst_ts, inst_label = tmp_data[:, :win * m], tmp_data[:, win * m:]

            inst_ts = np.reshape(inst_ts, (inst_ts.shape[0], m, inst_ts.shape[1]//m))
            y.append(inst_label.flatten()[0])

            ts_dataset = TSDataset(inst_ts, inst_label, scaler=self.scale_by_subseq_in_voter, scale_dim=self.scale_dim)
            loader = torch.utils.data.DataLoader(ts_dataset, batch_size=self.batch_size_voter)

            with torch.no_grad():
                logits_proba = []

                for ts, labels in loader:
                    self.model.eval()
                    # ===================variables=================== #
                    ts = Variable(ts.float()).to(self.device)
                    labels = Variable(labels.float()).to(self.device)
                    # ====================forward==================== #
                    logits = self.model(ts)
                    logits = nn.Softmax(dim=1)(logits)
                    if logits_proba:
                        logits_proba = list(logits[:, 1].cpu().detach().numpy().ravel())
                    else:
                        logits_proba = logits_proba + list(logits[:, 1].cpu().detach().numpy().ravel())
                
                if average_mode=='mean':
                    proba_inst = np.mean(np.array(logits_proba))
                elif average_mode=='quantile':
                    proba_inst = np.quantile(np.array(logits_proba), q=q)

                y_hat_prob.append(proba_inst)

                if threshold is not None:
                    if proba_inst > threshold:
                        y_hat.append(1)
                else:
                    y_hat.append(np.rint(proba_inst))

        return np.array(y), np.array(y_hat), np.array(y_hat_prob)
    
    
    def _ADFvoterproba(self, dataset_voter, m, win, average_mode, q, threshold): 
    
        y = dataset_voter[:][1].flatten()
        y_hat = np.zeros(y.shape)
        y_hat_prob = np.zeros(y.shape)

        for i, inst in enumerate(dataset_voter):
            inst_ts, inst_label = inst
            inst_ts = np.reshape(inst_ts, (inst_ts.shape[0], m, inst_ts.shape[1]//m))

            if inst_ts.shape[-1] < win:
                raise ValueError('Argument win need to be smaller than the time serie length, but received length={} and win={}'
                                  .format(inst_ts.shape[-1], win))

            n_obs_per_win = inst_ts.shape[-1] // win
            inst_ts = inst_ts[:, :, :n_obs_per_win*win]

            tmp = np.empty((n_obs_per_win, m, win))
            for im in range(m):
                tmp[:, im, :] = np.reshape(inst_ts[:, im, :], (n_obs_per_win, win))
            inst_ts = tmp.astype(np.float32)
            del tmp

            ts_dataset = TSDataset(inst_ts, np.repeat(inst_label, len(inst_ts)), scaler=self.scale_by_subseq_in_voter, scale_dim=self.scale_dim)
            loader = torch.utils.data.DataLoader(ts_dataset, batch_size=self.batch_size_voter)

            with torch.no_grad():
                logits_proba = []

                for ts, labels in loader:
                    self.model.eval()
                    # ===================variables=================== #
                    ts = Variable(ts.float()).to(self.device)
                    labels = Variable(labels.float()).to(self.device)
                    # ====================forward==================== #
                    logits = self.model(ts)
                    logits = nn.Softmax(dim=1)(logits)
                    if logits_proba:
                        logits_proba = list(logits[:, 1].cpu().detach().numpy().ravel())
                    else:
                        logits_proba = logits_proba + list(logits[:, 1].cpu().detach().numpy().ravel())
                
                if average_mode=='mean':
                    proba_inst = np.mean(np.array(logits_proba))
                elif average_mode=='quantile':
                    proba_inst = np.quantile(np.array(logits_proba), q=q)

                y_hat_prob[i] = proba_inst

                if threshold is not None:
                    if proba_inst > threshold:
                        y_hat[i] = 1
                else:
                    y_hat[i] = np.rint(proba_inst)
                
        return y, y_hat, y_hat_prob
    
    
    def _ADFvoter_df(self, dataset_voter, m, win, threshold, n_best_pred=None):
    
        y = []
        y_hat = []

        list_index = dataset_voter.index.unique()
                
        for i, id_pdl in enumerate(list_index):
            tmp_data = dataset_voter.loc[id_pdl].copy()
    
            if len(tmp_data.shape)==1:
                tmp_data = np.reshape(tmp_data.values, (1, len(tmp_data.values)))
            else:
                tmp_data = tmp_data.values
            inst_ts, inst_label = tmp_data[:, :win * m], tmp_data[:, win * m:]

            inst_ts = np.reshape(inst_ts, (inst_ts.shape[0], m, inst_ts.shape[1]//m))
            y.append(inst_label.flatten()[0])

            ts_dataset = TSDataset(inst_ts, inst_label, scaler=self.scale_by_subseq_in_voter, scale_dim=self.scale_dim)
            loader = torch.utils.data.DataLoader(ts_dataset, batch_size=1)

            with torch.no_grad():
                final_pred = 0
                if n_best_pred is not None:
                    predicteds = []
                    prob_predicteds = []

                for ts, labels in loader:
                    self.model.eval()
                    # ===================variables=================== #
                    ts = Variable(ts.float()).to(self.device)
                    labels = Variable(labels.float()).to(self.device)
                    # ====================forward==================== #
                    logits = self.model(ts)
                    prob_predicted, predicted = torch.max(nn.Softmax(dim=1)(logits), 1)

                    if n_best_pred is not None:
                        predicteds.append(predicted.item())
                        prob_predicteds.append(prob_predicted.item())
                    else:
                        final_pred += predicted.item()

                if n_best_pred is not None:
                    predicteds = np.array(predicteds)
                    prob_predicteds = np.array(prob_predicteds)
                    if n_best_pred < len(ts_dataset):
                        idx = np.argsort(prob_predicteds)[-n_best_pred:]
                    else:
                        idx = np.argsort(prob_predicteds)[-len(ts_dataset):]
                    final_pred = np.mean(predicteds[idx])
                else:
                    final_pred = final_pred / len(loader)

                y_hat.append(1) if final_pred > threshold else y_hat.append(0)

        return np.array(y), np.array(y_hat)
    
    
    def _ADFvoter(self, dataset_voter, m, win, threshold, n_best_pred=None):
        y = dataset_voter[:][1].flatten()
        y_hat = np.zeros(y.shape)

        for i, inst in enumerate(dataset_voter):
            inst_ts, inst_label = inst
            inst_ts = np.reshape(inst_ts, (inst_ts.shape[0], m, inst_ts.shape[1]//m))

            if inst_ts.shape[-1] < win:
                raise ValueError('Argument win need to be smaller than the time serie length, but received length = {} and win={}'
                                  .format(inst_ts.shape[-1], win))

            n_obs_per_win = inst_ts.shape[-1] // win
            inst_ts = inst_ts[:, :, :n_obs_per_win*win]

            tmp = np.empty((n_obs_per_win, m, win))
            for im in range(m):
                tmp[:, im, :] = np.reshape(inst_ts[:, im, :], (n_obs_per_win, win))
            inst_ts = tmp.astype(np.float32)
            del tmp

            ts_dataset = TSDataset(inst_ts, np.repeat(inst_label, len(inst_ts)), scaler=self.scale_by_subseq_in_voter, scale_dim=self.scale_dim)
            loader = torch.utils.data.DataLoader(ts_dataset, batch_size=1)

            with torch.no_grad():
                final_pred = 0
                if n_best_pred is not None:
                    predicteds = []
                    prob_predicteds = []

                for ts, labels in loader:
                    self.model.eval()
                    # ===================variables=================== #
                    ts = Variable(ts.float()).to(self.device)
                    labels = Variable(labels.float()).to(self.device)
                    # ====================forward==================== #
                    logits = self.model(ts)
                    prob_predicted, predicted = torch.max(nn.Softmax(dim=1)(logits), 1)

                    if n_best_pred is not None:
                        predicteds.append(predicted.item())
                        prob_predicteds.append(prob_predicted.item())
                    else:
                        final_pred += predicted.item()

                if n_best_pred is not None:
                    predicteds = np.array(predicteds)
                    prob_predicteds = np.array(prob_predicteds)
                    idx = np.argsort(prob_predicteds)[-n_best_pred:]
                    final_pred = np.mean(predicteds[idx])
                else:
                    final_pred = final_pred / len(loader)

                if final_pred > threshold:
                    y_hat[i] = 1
        
        return np.array(y), np.array(y_hat)
        
        
    def ADFFindBestThreshold(self, dataset_voter, m, win, n_best_pred=None, 
                             mask='allthreshold_valid_voter_metrics',
                             maskbest='bestthreshold_valid_voter_metrics',
                             metric_opt='F1_SCORE_MACRO', 
                             return_output=False): 
        
        tmp_time = time.time()
        list_metrics = []
        best_metrics = None

        for threshold in np.arange(0.1, 1, 0.1):

            threshold = round(threshold, 2)
            
            if isinstance(dataset_voter, pd.core.frame.DataFrame):
                y, y_hat = self._ADFvoter_df(dataset_voter=dataset_voter, m=m, win=win, 
                                             threshold=threshold, n_best_pred=n_best_pred)
            else:
                y, y_hat = self._ADFvoter(dataset_voter=dataset_voter, m=m, win=win, 
                                          threshold=threshold, n_best_pred=n_best_pred)
            
            metrics = self._apply_metrics(y, y_hat)
            metrics['Threshold_voter'] = threshold
            list_metrics.append(metrics)
            
            if best_metrics is not None:
                if best_metrics[metric_opt] < metrics[metric_opt]:
                    best_metrics = metrics
                    self.log[maskbest] = best_metrics
            else:
                best_metrics = metrics
                self.log[maskbest] = best_metrics
        
        self.voter_time = round((time.time() - tmp_time), 3)
        self.log['valid_voter_time'] = self.voter_time
        self.log[mask] = list_metrics
        
        if self.save_checkpoint:
            self.save()

        if return_output:
            return best_metrics, y, y_hat
        else:
            return best_metrics
        
        
    def ADFFindBestQuantile(self, dataset_voter, m, win, n_best_pred=None, 
                            mask='allquantile_valid_voter_metrics',
                            maskbest='bestquantile_valid_voter_metrics',
                            maskmetric='F1_SCORE_MACRO',
                            threshold=None,
                            return_output=False): 
        
        tmp_time = time.time()
        list_metrics = []
        best_metrics = None

        for quantile in np.arange(0.1, 1, 0.1):
            
            quantile = round(quantile, 2)
            
            if isinstance(dataset_voter, pd.core.frame.DataFrame):
                y, y_hat, y_hat_prob = self._ADFvoterproba_df(dataset_voter=dataset_voter, m=m, win=win, average_mode='quantile',
                                                              q=quantile, threshold=threshold)
            else:
                y, y_hat, y_hat_prob = self._ADFvoterproba(dataset_voter=dataset_voter, m=m, win=win, average_mode='quantile',
                                                           q=quantile, threshold=threshold)
            
            metrics = self._apply_metrics(y, y_hat, y_hat_prob)
            metrics['quantile'] = quantile
            list_metrics.append(metrics)
            
            if best_metrics is not None:
                if best_metrics[maskmetric] < metrics[maskmetric]:
                    best_metrics = metrics
                    self.log[maskbest] = best_metrics
            else:
                best_metrics = metrics
                self.log[maskbest] = best_metrics
        
        self.voter_time = round((time.time() - tmp_time), 3)
        self.log['valid_proba_voter_time'] = self.voter_time
        self.log[mask] = list_metrics
        
        if self.save_checkpoint:
            self.save()

        if return_output:
            return best_metrics, y, y_hat, y_hat_prob
        else:
            return best_metrics
          
          
        
# ====================================== Sktime Framework Implementation ====================================== #
class BasedClassifTrainer_Sktime(object):
    """
    Sktime based classif trainer classs
    
    For Sktime like model classifier (Arsenal, Rocket)
    """
    def __init__(self,
                 model,
                 f_metrics=getmetrics(),
                 verbose=True, save_model=False,
                 save_checkpoint=False, path_checkpoint=None):
        """
        Trainer designed for scikit API like model and classification cases
        """
        self.model = model
        self.f_metrics = f_metrics
        self.verbose = verbose
        self.save_model = save_model
        self.save_checkpoint = save_checkpoint
        
        if path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model'
        
        self.train_time = 0
        self.test_time = 0
        self.log = {}
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Public function : fit API call
        """
        
        _t = time.time()
        
        if len(X_train.shape)==3 and X_train.shape[1]==1:
            X_train = np.squeeze(X_train, axis=1)
        
        self.model.fit(X_train, y_train.ravel())
        self.train_time = round((time.time() - _t), 3)
        self.log['training_time'] = self.train_time
        
        if self.save_model:
            self.log['model'] = self.model
            
        if X_valid is not None and y_valid is not None:
            if len(X_valid.shape)==3 and X_valid.shape[1]==1:
                X_valid = np.squeeze(X_valid, axis=1)
                
            valid_metrics = self.evaluate(X_valid, y_valid, mask='valid_metrics')
            if self.verbose:
                print('Valid metrics :', valid_metrics)
        
        if self.verbose:
            print('Training time :', self.train_time)
        
        return
    
    def evaluate(self, X_test, y_test, mask='test_metrics', predict_proba=False):
        """
        Public function : predict API call then evaluation with given metric function
        """
        
        _t = time.time()
        if len(X_test.shape)==3 and X_test.shape[1]==1:
            X_test = np.squeeze(X_test, axis=1)
            
        pred = self.model.predict(X_test)
        if predict_proba:
            metrics = self._apply_metrics(y_test.ravel(), pred, self.model.predict_proba(X_test))
        else:
            metrics = self._apply_metrics(y_test.ravel(), pred)
        self.log[mask] = metrics
        self.test_time = round((time.time() - _t), 3)
        self.log['test_time'] = self.test_time
        
        if self.save_checkpoint:
            self.save()

        return metrics
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    
    def _apply_metrics(self, y, y_hat, y_hat_prob=None):
        """
        Private function : apply provided metrics
        
        !!! Provided metric function must be callable !!!
        """
        if y_hat_prob is not None:
            return self.f_metrics(y, y_hat, y_hat_prob)
        else:
            return self.f_metrics(y, y_hat) 

          
          
class AD_Framework_Sktime(BasedClassifTrainer_Sktime):
    """
    Detection of Appliance Problem Framework for Sktime like model (ROCKET)
    
    Class Based on BasedClassifTrainer : 
    -> This class is made for training/testing and evaluating a sktime model on binary appliance detection cases.
    
    - Voter Implementation for Univariate and Multivariate TS
    - Voter Implementation for entire consumption curve TS and sliced TS
    """
    def __init__(self,
                 model,
                 f_metrics=getmetrics(),
                 verbose=True, save_model=False,
                 save_checkpoint=False, path_checkpoint=None):
        """
        Detection of Appliance Framework Class based on BasedClassifTrainer parent class
        """
        super().__init__(model=model,
                         f_metrics=f_metrics,
                         verbose=verbose,
                         save_checkpoint=save_checkpoint, path_checkpoint=path_checkpoint)
    
    def ADFvoter(self, dataset_voter, win, m=1, 
                 mask='test_voter_metrics', 
                 mask1='bestthreshold_valid_voter_metrics',
                 mask2='Threshold_voter',
                 n_best_pred=None, threshold=None, return_output=False):
        
        if threshold is None:
            try:
                threshold = self.log[mask1][mask2]
            except:
                warnings.warn("No Threshold provided and No optimized threshold found, set Threshold to 0.5 for the voter predictions.")
                threshold = 0.5
        
        tmp_time = time.time()
        
        if isinstance(dataset_voter, pd.core.frame.DataFrame):
            y, y_hat = self._ADFvoter_df(dataset_voter=dataset_voter, m=m, win=win, 
                                         threshold=threshold, n_best_pred=n_best_pred)
        else:
            y, y_hat = self._ADFvoter(dataset_voter=dataset_voter, m=m, win=win,
                                      threshold=threshold, n_best_pred=n_best_pred)

        metrics = self._apply_metrics(y, y_hat)
        self.voter_time = round((time.time() - tmp_time), 3)
        self.log['test_voter_time'] = self.voter_time
        self.log[mask] = metrics

        if self.save_checkpoint:
            self.save()

        if return_output:
            return metrics, y, y_hat
        else:
            return metrics
        
        
    def ADFvoter_proba(self, dataset_voter, m, win, average_mode='quantile', q=None, threshold=None,
                       mask='test_voterproba_metrics', 
                       mask1='bestquantile_valid_voter_metrics',
                       mask2='quantile',
                       return_output=False): 
        tmp_time = time.time()
        
        if average_mode=='quantile':
            if q is None:
                try:
                    q = self.log[mask1][mask2]
                except:
                    warnings.warn('Average mode "quantile" but no optimize q found and no parameter q provided, set q=0.5 (median).')
                    q = 0.5
        if average_mode!='quantile' and average_mode!='mean':
            raise ValueError('Only "mean" and "quantile" average mode arguments supported for voter proba, but got = {}'
                              .format(average_mode))
    
        if isinstance(dataset_voter, pd.core.frame.DataFrame):
            y, y_hat, y_hat_prob = self._ADFvoterproba_df(dataset_voter=dataset_voter, m=m, win=win, average_mode=average_mode,
                                                          q=q, threshold=threshold)
        else:
            y, y_hat, y_hat_prob = self._ADFvoterproba(dataset_voter=dataset_voter, m=m, win=win, average_mode=average_mode,
                                                       q=q, threshold=threshold)
                
        metrics = self._apply_metrics(y, y_hat, y_hat_prob)
        self.voter_time = round((time.time() - tmp_time), 3)
        self.log['test_probavoter_time'] = self.voter_time
        self.log[mask] = metrics
        
        if self.save_checkpoint:
            self.save()

        if return_output:
            return metrics, y, y_hat, y_hat_prob
        else:
            return metrics
        
    
    def _ADFvoterproba_df(self, dataset_voter, m, win, average_mode, q, threshold): 
    
        y = []
        y_hat = []
        y_hat_prob = []

        list_index = dataset_voter.index.unique()
                
        for i, id_pdl in enumerate(list_index):
            tmp_data = dataset_voter.loc[id_pdl].copy()
    
            if len(tmp_data.shape)==1:
                tmp_data = np.reshape(tmp_data.values, (1, len(tmp_data.values)))
            else:
                tmp_data = tmp_data.values
            inst_ts, inst_label = tmp_data[:, :win * m], tmp_data[:, win * m:]

            inst_ts = np.reshape(inst_ts, (inst_ts.shape[0], m, inst_ts.shape[1]//m))
            
            if len(inst_ts.shape)==3 and inst_ts.shape[1]==1:
                inst_ts = np.squeeze(inst_ts, axis=1)
                
            y.append(inst_label.flatten()[0])
            logits_proba = self.model.predict_proba(inst_ts)[:, 1]

            if average_mode=='mean':
                proba_inst = np.mean(np.array(logits_proba))
            elif average_mode=='quantile':
                proba_inst = np.quantile(np.array(logits_proba), q=q)

            y_hat_prob.append(proba_inst)

            if threshold is not None:
                if proba_inst > threshold:
                    y_hat.append(1)
            else:
                y_hat.append(np.rint(proba_inst))

        return np.array(y), np.array(y_hat), np.array(y_hat_prob)
    
    
    def _ADFvoterproba(self, dataset_voter, m, win, average_mode, q, threshold): 
    
        y = dataset_voter[:][1].flatten()
        y_hat = np.zeros(y.shape)
        y_hat_prob = np.zeros(y.shape)

        for i, inst in enumerate(dataset_voter):
            inst_ts, inst_label = inst
            inst_ts = np.reshape(inst_ts, (inst_ts.shape[0], m, inst_ts.shape[1]//m))

            if inst_ts.shape[-1] < win:
                raise ValueError('Argument win need to be smaller than the time serie length, but received length={} and win={}'
                                  .format(inst_ts.shape[-1], win))

            n_obs_per_win = inst_ts.shape[-1] // win
            inst_ts = inst_ts[:, :, :n_obs_per_win*win]

            tmp = np.empty((n_obs_per_win, m, win))
            for im in range(m):
                tmp[:, im, :] = np.reshape(inst_ts[:, im, :], (n_obs_per_win, win))
            inst_ts = tmp.astype(np.float32)
            del tmp
            
            if len(inst_ts.shape)==3 and inst_ts.shape[1]==1:
                inst_ts = np.squeeze(inst_ts, axis=1)

            logits_proba = self.model.predict_proba(inst_ts)[:, 1]
                
            if average_mode=='mean':
                proba_inst = np.mean(logits_proba)
            elif average_mode=='quantile':
                proba_inst = np.quantile(logits_proba, q=q)

            y_hat_prob[i] = proba_inst

            if threshold is not None:
                if proba_inst > threshold:
                    y_hat[i] = 1
            else:
                y_hat[i] = np.rint(proba_inst)
                
        return y, y_hat, y_hat_prob
        
        
    def ADFFindBestQuantile(self, dataset_voter, m, win, n_best_pred=None, 
                            mask='allquantile_valid_voter_metrics',
                            maskbest='bestquantile_valid_voter_metrics',
                            maskmetric='F1_SCORE_MACRO',
                            threshold=None,
                            return_output=False): 
        
        tmp_time = time.time()
        list_metrics = []
        best_metrics = None

        for quantile in np.arange(0.1, 1, 0.1):
            
            quantile = round(quantile, 2)
            
            if isinstance(dataset_voter, pd.core.frame.DataFrame):
                y, y_hat, y_hat_prob = self._ADFvoterproba_df(dataset_voter=dataset_voter, m=m, win=win, average_mode='quantile',
                                                              q=quantile, threshold=threshold)
            else:
                y, y_hat, y_hat_prob = self._ADFvoterproba(dataset_voter=dataset_voter, m=m, win=win, average_mode='quantile',
                                                           q=quantile, threshold=threshold)
            
            metrics = self._apply_metrics(y, y_hat, y_hat_prob)
            metrics['quantile'] = quantile
            list_metrics.append(metrics)
            
            if best_metrics is not None:
                if best_metrics[maskmetric] < metrics[maskmetric]:
                    best_metrics = metrics
                    self.log[maskbest] = best_metrics
            else:
                best_metrics = metrics
                self.log[maskbest] = best_metrics
        
        self.voter_time = round((time.time() - tmp_time), 3)
        self.log['valid_proba_voter_time'] = self.voter_time
        self.log[mask] = list_metrics
        
        if self.save_checkpoint:
            self.save()

        if return_output:
            return best_metrics, y, y_hat, y_hat_prob
        else:
            return best_metrics

        
        
# ====================================== EarlyStopper ====================================== #
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

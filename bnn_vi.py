import numpy as np
import torch
import time
import pyro
from pyro.infer import Predictive,SVI,Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal,AutoLowRankMultivariateNormal

from sklearn.model_selection import KFold
from joblib import Parallel,delayed

from bnn_arch import BayesianMLP
from uq_metrics import metrics_ensemble
from data_utils import Dataset

class BayesianMLPCV:
    def __init__(
        self,X:torch.Tensor,
        y:torch.Tensor,
        variational_dist='independent',
        activation='Tanh',
        n_splits=5,
        n_jobs=1
    ):
        self.X = X
        self.y = y
        self.input_dim = X.shape[1]

        if variational_dist =='independent':
            self.guideclass = AutoDiagonalNormal
        elif variational_dist == 'lowrank':
            self.guideclass = AutoLowRankMultivariateNormal
        
        self.activation=activation
        self.rkf = KFold(n_splits=n_splits,shuffle=True)
        self.n_jobs=n_jobs
    
    def fit_and_test(
        self,
        train_index,test_index,
    ):

        train_loader = torch.utils.data.DataLoader(
            Dataset(self.X[train_index,...],self.y[train_index]),
            batch_size=100,
            shuffle=True
        )

        # 5000 steps
        num_batches = np.ceil(train_loader.dataset.__len__()/train_loader.batch_size)
        num_epochs = int(np.ceil(5000/num_batches))

        # construct model
        model =  BayesianMLP(
            input_dim=self.input_dim,
            h_sizes=[50], # hardcoding 
            output_dim=1,
            activation=self.activation,
        )

        start_time = time.time()
        guide = self.guideclass(model)
        pyro.clear_param_store()
        optim = pyro.optim.Adam({"lr": 0.01})
        svi = SVI(model, guide, optim, loss=Trace_ELBO())
        loss = 0

        for _ in range(num_epochs):
            loss = 0
            for X,y in train_loader:
                # calculate the loss and take a gradient step
                loss += svi.step(X,y)
            # normalizer_train = len(train_loader.dataset)
            # total_epoch_loss_train = loss / normalizer_train
        training_time = time.time()-start_time

        with torch.no_grad():
            predictive = Predictive(
                model, guide=guide, num_samples=50,return_sites=['_RETURN']
            )
            predictions = predictive(self.X[test_index,...],None)['_RETURN']
        
        y_mean = predictions[:,:,0]
        y_std = predictions[:,:,1]
        
        metrics = metrics_ensemble(self.y[test_index],y_mean,y_std,0.05)
        metrics['training_time'] = training_time
        return metrics
    
    def cvloss(self,cv=None):
        if cv is None:
            cv = self.rkf

        scores = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_and_test)(
                train_index,test_index,
            ) for train_index,test_index in cv.split(self.X)
        )

        return scores
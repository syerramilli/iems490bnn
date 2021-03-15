import numpy as np
import torch

import time
import pyro
from pyro.infer import Predictive,SVI,Trace_ELBO
from pyro.infer.autoguide import AutoDelta,init_to_sample

from sklearn.model_selection import KFold
from joblib import Parallel,delayed

from bnn_arch import BayesianMLP
from uq_metrics import metrics_ensemble
from data_utils import Dataset

class EnsembleMLPCV:
    def __init__(
        self,X:torch.Tensor,
        y:torch.Tensor,
        n_models=10,
        activation='Tanh',
        n_splits=5,
        n_jobs=1
    ):
        self.X = X
        self.y = y
        self.input_dim = X.shape[1]
        self.n_models = n_models
        
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

        def train_map_local():
            pyro.clear_param_store()
            adam = pyro.optim.Adam({"lr": 0.01})
            guide = AutoDelta(model,init_loc_fn=init_to_sample)
            svi = SVI(model, guide, adam, loss=Trace_ELBO())
            for _ in range(num_epochs):
                loss = 0
                for X,y in train_loader:
                    # calculate the loss and take a gradient step
                    loss += svi.step(X,y)
        
            params_map = {k.replace('AutoDelta.',''):v.detach().unsqueeze(0) \
                        for k,v in pyro.get_param_store().items()}
            return params_map
        
        start_time = time.time()
        params = [train_map_local() for _ in range(self.n_models)]
        keys = params[0].keys()
        params_ensemble = {
            k:torch.cat([local[k] for local in params],axis=0)
            for k in keys
        }
        training_time = time.time()-start_time

        with torch.no_grad():
            predictive = Predictive(
                model,params_ensemble,return_sites=['_RETURN']
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
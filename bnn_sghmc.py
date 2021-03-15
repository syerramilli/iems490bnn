import numpy as np
import torch
import time

from pybnn.bohamiann import Bohamiann,predict_bnn

from sklearn.model_selection import KFold
from joblib import Parallel,delayed

from uq_metrics import metrics_ensemble
from data_utils import Dataset

class ExactBayesianMLPCV:
    def __init__(
        self,X:torch.Tensor,
        y:torch.Tensor,
        n_splits=5,
        n_jobs=1
    ):
        self.X = X
        self.y = y
        self.input_dim = X.shape[1]
        self.rkf = KFold(n_splits=n_splits,shuffle=True)
        self.n_jobs=n_jobs
    
    def fit_and_test(
        self,
        train_index,test_index,
    ):
        model_bnn = Bohamiann(
            normalize_input=False,
            normalize_output=False,
            use_double_precision=False,
            print_every_n_steps=1000
        )

        start_time = time.time()
        model_bnn.train(
            self.X[train_index,:].numpy(),self.y[train_index].numpy(),
            num_steps=12000,
            num_burn_in_steps=2000,
            keep_every=200,
            lr=0.05,
            verbose=False,
            batch_size=100
        )
        training_time = time.time()-start_time

        predictions = predict_bnn(model_bnn,self.X[test_index,:].numpy())
        y_mean = predictions[:,:,0]
        y_std = predictions[:,:,1].exp().sqrt()
        
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
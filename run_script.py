import numpy as np
import pandas as pd
import os
import argparse

from bnn_ensemble import EnsembleMLPCV
from bnn_vi import BayesianMLPCV
from bnn_sghmc import ExactBayesianMLPCV

from sklearn.model_selection import RepeatedKFold
from data_utils import load_dataset

#%%
parser = argparse.ArgumentParser('Testing UQ in simple neural networks')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--n_jobs',type=int,required=True)
args = parser.parse_args()

save_path = os.path.join(args.save_dir,args.dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%%
X,y = load_dataset(args.dataset)
cv = RepeatedKFold(n_splits=10,n_repeats=2,random_state=1)

#%%
tests = [
    BayesianMLPCV(X,y,'independent',n_jobs=args.n_jobs), # independent VI
    BayesianMLPCV(X,y,'lowrank',n_jobs=args.n_jobs),
    EnsembleMLPCV(X,y,n_models=10,n_jobs=args.n_jobs)
    ExactBayesianMLPCV(X,y,n_jobs=args.n_jobs)
]

names = ['vi_ind','vi_lowrank','ensemble','sghmc']

for i,test in enumerate(tests):
    scores = test.cvloss(cv)
    scores = pd.DataFrame(scores)
    scores.to_csv(
        os.path.join(save_path,names[i]+'.csv'),index=False
    )
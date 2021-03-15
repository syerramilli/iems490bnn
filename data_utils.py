import torch
import pandas as pd
from sklearn.datasets import load_boston

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self,idx):
        return self.X[idx,:],self.y[idx]

def load_dataset(dataset_name):
    if dataset_name == 'boston':
        X,y = load_boston(return_X_y=True)
    elif dataset_name == 'concrete':
        dat = pd.read_csv('data/concrete.csv')
        X = dat.iloc[:,:-1].values
        y = dat.iloc[:,-1].values
    elif dataset_name == 'wine':
        dat = pd.read_table('data/winequality-red.csv',sep=';')
        X = dat.iloc[:,:-1].values
        y = dat.iloc[:,-1].values
    
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float() 

    X_mean,X_std = X.mean(axis=0),X.std(axis=0)
    y_mean,y_std = y.mean(),y.std()

    X = (X-X_mean)/X_std
    y = (y-y_mean)/y_std
    return X,y
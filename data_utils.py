import torch
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.datasets import fetch_openml

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
    elif dataset_name == 'kin8nm':
        X,y = fetch_openml('kin8nm',return_X_y=True,as_frame=False,data_home='data/')
    elif dataset_name == 'abalone':
        X,y = fetch_openml('abalone',return_X_y=True,as_frame=False,data_home='data/')
        y = y.astype('float')
    elif dataset_name == 'cpu_small':
        X,y = fetch_openml('cpu_small',return_X_y=True,as_frame=False,data_home='data/')
        y = y.astype('float')
    elif dataset_name == 'autompg':
        dat = pd.read_csv(
            'data/auto-mpg.csv',na_values='?'
        ).dropna(axis=0)
        del dat['car name']
        y = dat['mpg'].values
        X = dat.drop('mpg',axis=1).values
    
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float() 

    X_mean,X_std = X.mean(axis=0),X.std(axis=0)
    y_mean,y_std = y.mean(),y.std()

    X = (X-X_mean)/X_std
    y = (y-y_mean)/y_std
    return X,y
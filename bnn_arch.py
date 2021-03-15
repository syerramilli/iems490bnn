import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from functools import partial

from pyro.nn import PyroModule
from pyro.nn import PyroSample,PyroParam
from pyro.nn.module import to_pyro_module_
from typing import List, Tuple, Optional, Dict

class MLP(nn.Module):
    def __init__(
        self,
        input_dim:int,
        h_sizes:List[int],
        output_dim:int,
        activation:str='Sigmoid',
    ):
        super(MLP,self).__init__()

        hidden_layers = []
        for hsize in h_sizes:
            hidden_layers.append(nn.Linear(input_dim,hsize))
            hidden_layers.append(getattr(nn,activation)())
            input_dim = hsize
        
        self.hidden_layers = nn.Sequential(*hidden_layers)  
        self.output = nn.Linear(h_sizes[-1],output_dim)
    
    def forward(self,x):
        x = self.hidden_layers(x)    
        return self.output(x)

class BayesianMLP(PyroModule):
    def __init__(
        self,
        input_dim,
        h_sizes,
        output_dim,
        activation='Tanh',
        noise_std:float=0.01
    ):
        super().__init__()
        self.mlp = MLP(
            input_dim = input_dim,
            h_sizes=h_sizes,
            output_dim=output_dim,
            activation = activation
        )

        to_pyro_module_(self.mlp)
        def wt_distribution(self,shape,scale_factor):
            event_idx = len(shape)
            out = (
                dist.Normal(0.,scale_factor)#/self.wt_precision.sqrt())
                .expand(shape)
                .to_event(event_idx)
            )
            return out

        for m_name,m in self.mlp.named_modules():
            if not isinstance(m,nn.Linear):
                continue
            
            for name,value in list(m.named_parameters(recurse=False)):
                scale_factor = torch.tensor(10.).sqrt() if 'bias' in name else torch.ones(1)
                setattr(m,name,PyroSample(
                    partial(wt_distribution,shape=value.shape,scale_factor=scale_factor)
                ))
        
        self.log_var = PyroSample(
            dist.Normal(
                torch.log(torch.tensor(1e-6)),0.1
            )
        )

    def forward(self,x,y=None):
        mean = self.mlp(x).squeeze(1)
        #noise_std = self.noise_std
        noise_std = (1e-16+self.log_var.exp()).sqrt()
        with pyro.plate('data',x.shape[0]):
            obs = pyro.sample('obs',dist.Normal(mean,noise_std),obs=y)
            
        return torch.stack([mean,noise_std.expand_as(mean)],dim=-1)
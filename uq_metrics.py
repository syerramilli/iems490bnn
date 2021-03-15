import torch
import math

def neg_interval_score(y_true,lq,uq,alpha):
    term1 = uq-lq
    term2 = 2/alpha*torch.nn.functional.relu(lq-y_true)
    term3 = 2/alpha*torch.nn.functional.relu(-uq+y_true)
    return (term1+term2+term3).mean()

def neg_crps_gaussian(y_true,mean,std):
    z = (y_true-mean)/std
    dist = torch.distributions.Normal(mean,std)
    term1 = 1/math.sqrt(math.pi)
    term2 = 2*dist.log_prob(y_true).exp()
    term3 = z*(2*dist.cdf(y_true)-1)
    return (std*(term2+term3-term1)).mean()

def metrics_ensemble(y_true,y_means,y_stds,alpha=0.05):
    # %% prediction metric
    rmse =((y_true-y_means.mean(axis=0)**2).mean().sqrt()
    
    means= y_means.t()
    stds = y_stds.t()
    
    # number of samples-1000
    num_samples = 1000
    I = torch.randint(high=means.shape[1],size=(means.shape[0],num_samples)).long()
    samples = torch.stack([
        means[i,I[i,0]]+stds[i,I[i,1]]*torch.randn((num_samples,)) for i in range(len(y_test))
    ],axis=0)
    
    # coverage and interval-score 
    lq = samples.quantile(alpha/2,axis=-1)
    uq = samples.quantile(1-alpha/2,axis=-1)
    cov_prob = (1.*(y_test>=lq)*(y_test<=uq)).mean()
    mean_is = neg_interval_score(y_true,lq,uq,alpha)
    
    # crps
    term1 = (samples-y_test.unsqueeze(-1)).abs().mean(axis=1)
    forecasts_diff = torch.abs(samples.unsqueeze(-1)-samples.unsqueeze(-2))
    term2 = 0.5*forecasts_diff.mean(axis=(-2,-1))
    crps = (term1-term2).mean()
    
    return {
        'rmse':rmse,
        'mean_is':mean_is,
        'coverage_prob':cov_prob,
        'crps':crps
    }
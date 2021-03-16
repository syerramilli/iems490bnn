import torch
import math

def neg_interval_score(y_true,lq,uq,alpha):
    term1 = uq-lq
    term2 = 2/alpha*torch.nn.functional.relu(lq-y_true)
    term3 = 2/alpha*torch.nn.functional.relu(-uq+y_true)
    return (term1+term2+term3).mean().item()

def neg_crps_gaussian(y_true,mean,std):
    z = (y_true-mean)/std
    dist = torch.distributions.Normal(mean,std)
    term1 = 1/math.sqrt(math.pi)
    term2 = 2*dist.log_prob(y_true).exp()
    term3 = z*(2*dist.cdf(y_true)-1)
    return (std*(term2+term3-term1)).mean()

def metrics_ensemble(y_true,y_means,y_stds,alpha=0.05):
    # %% prediction metric
    rmse =((y_true-y_means.mean(axis=0))**2).mean().sqrt().item()
    
    means= y_means.t()
    stds = y_stds.t()
    
    # number of samples-1000
    n_obs = len(y_true)
    num_samples = 1000
    lq = []
    uq = []
    interval_score = 0
    cov = 0
    crps = 0

    for i in range(n_obs):
        I = torch.randint(high=means.shape[1],size=(num_samples,))
        samples = means[i,I]+stds[i,I]*torch.randn((num_samples,))
        lq = samples.quantile(alpha/2)
        uq = samples.quantile(1-alpha/2)
        
        # coverage 
        cov += (1*(y_true[i]>=lq)*(y_true[i]<=uq)).item()

        # interval score
        term1 = uq-lq
        term2 = 2/alpha*torch.nn.functional.relu(lq-y_true[i])
        term3 = 2/alpha*torch.nn.functional.relu(-uq+y_true[i])
        interval_score += (term1+term2 + term3).item()

        # crps
        term1 = (samples-y_true[i]).abs().mean()
        forecasts_diff = torch.abs(samples.unsqueeze(-1)-samples.unsqueeze(-2))
        term2 = 0.5*forecasts_diff.mean()
        crps += (term1-term2).item()
    
    return {
        'rmse':rmse,
        'mean_is':interval_score/n_obs,
        'coverage_prob':cov/n_obs,
        'crps':crps/n_obs
    }
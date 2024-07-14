import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy.special import kl_div
import my_funcs as mf

sampling_rate = 50
recording_time = 10 #min

data_arr = np.array([])
num_fas = 0
repeat = 0
for i in range(600):
    if i % 120 == 0:
        num_fas += 1 # alarm would be false every 120 seconds
        
    if repeat >= 30: # assuming events happen every 30 seconds
        data = np.random.beta(a=10, b=5) # mode: 0.69
        data_arr = np.append(data_arr, data)
        if repeat == 40:
            repeat = 0
    else:
        if num_fas < 4:
            data = np.random.beta(a=4, b=5) # mode: 0.43
            data_arr = np.append(data_arr, data)
        if num_fas >= 4:
            data = np.random.beta(a=4, b=10) # mode: 0.25
            data_arr = np.append(data_arr, data)

    repeat += 1
data_arr.shape
    
sns.set(rc={"figure.figsize":(12, 8)}, font_scale=2)
sns.lineplot(data_arr)
plt.show()    

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0.5, sigma=1)
    #mu = pm.Uniform('mu', lower=0, upper=1)
    #sigma = pm.Uniform('sigma', lower=0, upper=3)
    variance = pm.InverseGamma('variance', alpha=2, beta=2)
    
    likelihood = pm.Normal('likelihood', mu=mu, sigma=pm.math.sqrt(variance), 
                           observed=data_arr[:30])
    
    trace = pm.sample(2000)

az.plot_trace(trace)
az.plot_posterior(trace)

data_dict = {}
repeat = 0
epoch = 0
tmp = []
for i in range(600):
    if repeat <= 30:
        tmp.append(data_arr[i])
        if repeat == 30:
            epoch += 1
            if i == 30:
                data_dict[epoch] = tmp[:30]
            else:
                data_dict[epoch] = tmp
            tmp = []
    if repeat == 40:
        repeat = 0
    repeat += 1
    

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0.5, sigma=1)
    sigma = pm.InverseGamma('sigma', alpha=2, beta=2)
    
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data_dict[1])
    
    trace = pm.sample(2000)

az.plot_trace(trace)


traces_dict = {}
for i in range(1, 16):
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0.5, sigma=1)
        sigma = pm.InverseGamma('sigma', sigma=0.4)
        
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data_dict[i])
        
        trace = pm.sample(2000)
        
    traces_dict[i] = trace

epochs_results = {}
for i in range(1, 16):
    epochs_results[i] = mf.average_chains_values('mu', traces_dict[i])
epochs_results = pd.DataFrame.from_dict(epochs_results)
#epochs_results.to_csv('epochs_results.csv')

kls = []
for i in range(1, 15):
    kl = np.sum(kl_div(epochs_results[i+1], epochs_results[i]))
    kls.append(kl)
kls = pd.Series(kls)
#kls.to_csv('kls.csv')

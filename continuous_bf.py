import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from create_df import create_df
import pymc as pm
import arviz as az
from scipy import stats
from pymc.distributions import Interpolated
import xarray
from gc import collect as gc
az.style.use('arviz-darkgrid')

num_subjects = 30

dataframes = {}
for i in range(num_subjects):
    dataframes[i] = create_df()

# 30(samling rate) x 60(seconds) x 5(mins) = 9000
# 1800 timestamps per second
# events happen every 30 seconds (900)

df = dataframes[0]
plt.plot(df['X'][:200], df['AOI_Hit'][:200])
plt.show()

plt.hist(df['AOI_Hit'][:180])
plt.show()

dataframes_hit = {}
for x in range(num_subjects):
    df_hit = pd.DataFrame()
    empty = []
    for i, hit in enumerate(dataframes[x]['AOI_Hit']):
        if i % 900 != 0 or i == 0:
            empty.append(hit)
        else:
            df_hit[f'epoch{int(i/900)}'] = empty
            empty = [hit]
            
    dataframes_hit[x] = df_hit

ratios_per_epoch = {}
for i in range(num_subjects):
    ratios = []
    for j in range(len(dataframes_hit[i].columns)):
        ratios.append(sum(dataframes_hit[i][f'epoch{j+1}']) / len(dataframes_hit[i][f'epoch{j+1}']))
    ratios_per_epoch[i] = ratios
ratios_per_epoch = pd.DataFrame.from_dict(ratios_per_epoch).transpose()

with pm.Model() as model:
    # Binomial distribution cannot be used because the probability of success is unknown
    mu = pm.Uniform('mu', lower=0, upper=1)
    sigma = pm.HalfNormal('sigma', sigma=0.4)
    
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=ratios_per_epoch[0])
    
    trace = pm.sample(2000, tune=1000)
    #trace.extend(pm.sample_prior_predictive(8000))

result = az.summary(trace).loc['mu']
df_result = pd.DataFrame()
df_result = df_result._append(result, ignore_index=True)
az.plot_trace(trace)
plt.show()

def average_chains_values(param: str, trace: pm.sample) -> np.array:
    chains = len(trace['posterior']['chain'])
    draws = len(trace['posterior']['draw'])
    all_chains_values = np.array([])
    for i in range(chains):
        all_chains_values = np.append(all_chains_values, trace['posterior'][param][i])
        # len(all_chains_values) == 16000
        
    tmp_4vals = np.array([])
    mean_of_4chains = np.array([])
    for i in range(draws):
        for j in range(i+draws, chains*draws-draws, draws):
            tmp_4vals = np.append(tmp_4vals, np.array([all_chains_values[i], all_chains_values[j]]))
        mean_of_4chains = np.append(mean_of_4chains, np.mean(tmp_4vals))
        # len(tmp_4vals) == 4
        tmp_4vals = []
    
    return mean_of_4chains # len(mean_of_4chains) == 4000
    
mean_of_4chains_mu = average_chains_values('mu', trace)
mean_of_4chains_sigma = average_chains_values('sigma', trace)

def from_posterior(param: str, 
                   samples: xarray.DataArray or np.array
                   ) -> pm.distributions.Interpolated:
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    
    return Interpolated(param, x, y)

for i in range(1, 5):
    with pm.Model() as model:
        mu = from_posterior('mu', mean_of_4chains_mu)
        #sigma = from_posterior('sigma', mean_of_4chains_sigma)
        sigma = pm.HalfNormal('sigma', sigma=0.4)
        
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=ratios_per_epoch[i])
        
        trace = pm.sample(2000, tune=1000)
    
    result = az.summary(trace).loc['mu']
    df_result = df_result._append(result, ignore_index=True)
    
az.plot_trace(trace)
plt.show()

fig, ax = plt.subplots()
ax.plot(df_result.index, df_result['mean'])
ax.fill_between(df_result.index, df_result['hdi_3%'], df_result['hdi_97%'], color='b', alpha=.1)
ax.set_xticks([i for i in range(5)])
plt.show()

with model:
    pm.compute_log_likelihood(trace)
loo = az.loo(trace)
loo


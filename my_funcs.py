import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy import stats
from pymc.distributions import Interpolated
import xarray
import gc

def create_df(sampling_rate: int,
              duration: int,
              event_interval: int,
              event_duration: int) -> pd.DataFrame:
    """ 
    Parameters
    sampling_rate: Hz
    duration: recording minutes in seconds
    event_interval: interval of event happening
    event_duration: Event duration in seconds
    """
    total_samples = duration * sampling_rate
    aoi = {'x_min': 300, 'x_max': 600, 'y_min': 200, 'y_max': 400}  # Example AOI coordinates
    
    # Initialize data
    timestamps = np.linspace(0, duration, total_samples)
    x_coords = np.random.randint(0, 800, total_samples)  # Random X coordinates
    y_coords = np.random.randint(0, 600, total_samples)  # Random Y coordinates
    fixation_durations = np.random.randint(100, 500, total_samples)  # Random fixation durations between 100ms and 500ms
    
    # Simulate events attracting attention to the AOI
    for i in range(0, duration, event_interval):
        event_start = i * sampling_rate
        event_end = event_start + event_duration * sampling_rate
        x_coords[event_start:event_end] = np.random.randint(aoi['x_min'], aoi['x_max'], event_end - event_start)
        y_coords[event_start:event_end] = np.random.randint(aoi['y_min'], aoi['y_max'], event_end - event_start)
    
    # Determine AOI hits
    aoi_hits = ((x_coords >= aoi['x_min']) & (x_coords <= aoi['x_max']) & 
                (y_coords >= aoi['y_min']) & (y_coords <= aoi['y_max'])).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({'Timestamp': timestamps,
                       'Seconds': timestamps / 60,
                       'X': x_coords,
                       'Y': y_coords,
                       'Fixation_Duration': fixation_durations,
                       'AOI_Hit': aoi_hits})
    
    return df

def calculate_posterior(observed: np.array,
                        draws: int = 2000,
                        tune: int = 1000) -> az.InferenceData:
    with pm.Model() as model:
        # Binomial distribution cannot be used because the probability of success varies
        mu = pm.Uniform('mu', lower=0, upper=1)
        sigma = pm.HalfNormal('sigma', sigma=0.4)
        
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=observed)
        
        trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
        #trace.extend(pm.sample_prior_predictive(8000))
    gc.collect()
    
    return trace

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
    
    return mean_of_4chains # len(mean_of_4chains) == draws

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

def sequential_bayes_update(df_to_append: pd.DataFrame, 
                            prior_trace: xarray.DataArray or np.array,
                            observed: pd.DataFrame,
                            epochs: iter,
                            draws: int = 2000,
                            tune: int = 1000) -> tuple[pd.DataFrame, dict]:
    df_result = df_to_append.copy()
    traces = {}
    for i in epochs:
        with pm.Model() as model:
            mu = from_posterior('mu', prior_trace)
            #sigma = from_posterior('sigma', mean_of_4chains_sigma)
            sigma = pm.HalfNormal('sigma', sigma=0.4)
            
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=observed[f'epoch{i}'])
            
            trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
        
        result = az.summary(trace).loc['mu']
        df_result = df_result._append(result, ignore_index=True)
        traces[f'epoch{i}'] = trace
        
        prior_trace = average_chains_values(param='mu', trace=trace)
        
        print(f'\nepoch{i} done\n')
        gc.collect()
    
    return df_result, traces



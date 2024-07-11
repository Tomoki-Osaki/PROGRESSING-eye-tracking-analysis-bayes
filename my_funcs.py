import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import scipy as sc
import matplotlib.pyplot as plt
import xarray
import gc

Array = xarray.DataArray or np.array

"""
make_df_gaze_data
make_df_ratios_per_epoch
make_df_subjects_data
calculate_posterior
average_chains_values
from_posterior
sequential_bayes_update
plotbeta
"""

def make_df_gaze_data(sampling_rate: int,
                      recording_duration: int,
                      event_interval: int,
                      event_duration: int) -> pd.DataFrame:
    """    
    sampling_rate: Hz
    duration: recording minutes in seconds
    event_interval: interval of event happening in seconds
    event_duration: Event duration in seconds
    
    Event interval should be longer than event_duration.
    Called in make_df_ratios_per_epoch.
    """
    total_samples = recording_duration * sampling_rate
    aoi = {'x_min': 300, 'x_max': 600, 'y_min': 200, 'y_max': 400}  # Example AOI coordinates
    
    # Initialize data
    timestamps = np.linspace(0, recording_duration, total_samples)
    x_coords = np.random.randint(0, 800, total_samples)  # Random X coordinates
    y_coords = np.random.randint(0, 600, total_samples)  # Random Y coordinates
    fixation_durations = np.random.randint(100, 500, total_samples)  # Random fixation durations between 100ms and 500ms
    
    # Simulate events attracting attention to the AOI
    for i in range(0, recording_duration, event_interval):
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


def make_df_ratios_per_epoch(num_subjects: int = 30,
                             sampling_rate: int = 30,
                             recording_duration: int = 300,
                             event_interval: int = 30,
                             event_duration: int = 2) -> pd.DataFrame:

    dataframes = {}
    for i in range(num_subjects):
        dataframes[i] = make_df_gaze_data(sampling_rate, 
                                          recording_duration, 
                                          event_interval, 
                                          event_duration)
    # 30(samling rate) x 60(seconds) x 5(mins) = 9000
    # 1800 timestamps per second
    # events happen every 30 seconds (900)
    
    dataframes_hit = {}
    epoch_duration = sampling_rate * event_interval
    for x in range(num_subjects):
        df_hit = pd.DataFrame()
        empty = []
        for i, hit in enumerate(dataframes[x]['AOI_Hit']):
            if i % epoch_duration != 0 or i == 0:
                empty.append(hit)
            else:
                df_hit[f'epoch{int(i/epoch_duration)}'] = empty
                empty = [hit]   
        dataframes_hit[x] = df_hit
    
    ratios_per_epoch = {}
    for i in range(num_subjects):
        ratios = []
        for j in range(len(dataframes_hit[i].columns)):
            ratios.append(sum(dataframes_hit[i][f'epoch{j+1}']) / len(dataframes_hit[i][f'epoch{j+1}']))
        ratios_per_epoch[i] = ratios
        
    colnames = [f'epoch{i+1}' for i in range(len(dataframes_hit[i].columns))]
    ratios_per_epoch = pd.DataFrame.from_dict(ratios_per_epoch, orient='index', columns=colnames)
    
    return ratios_per_epoch


def make_df_subjects_data(num_subjects: int) -> pd.DataFrame:
    subjects_data = {}
    alarms = [True, True, False, True, True, True, False, True, False, True]
    for i in range(num_subjects):
        subject_data = {}
        for epoch, alarm in enumerate(alarms):
            if alarm: # when an alarm is correct
                subject_data[epoch] = np.random.beta(a=4, b=10)
            else: # when an alarm is false
                subject_data[epoch] = np.random.beta(a=10, b=5)
        subjects_data[i] = subject_data
        
    subjects_data = pd.DataFrame.from_dict(subjects_data, orient='index')
    for i in range(len(alarms)):
        subjects_data.rename(columns={i: f'epoch{i}'}, inplace=True)    
    
    return subjects_data


def calculate_posterior(observed: np.array,
                        draws: int = 2000,
                        tune: int = 1000) -> az.InferenceData:
    with pm.Model() as model:
        # Bernoulli distribution cannot be used because the each trial is not independent
        # Thus, Binomial distribution is also not available
        mu = pm.Uniform('mu', lower=0, upper=1)
        sigma = pm.HalfNormal('sigma', sigma=0.4)
        
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=observed)
        
        trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True)

    gc.collect()
    
    return trace


def average_chains_values(param: str, trace: pm.sample) -> np.array:
    chains = len(trace['posterior']['chain'])
    draws = len(trace['posterior']['draw'])
    all_chains_values = np.array([])
    for i in range(chains):
        all_chains_values = np.append(all_chains_values, trace['posterior'][param][i])
        # len(all_chains_values) == 16000
        
    tmp_vals = np.array([])
    mean_of_chains = np.array([])
    for i in range(draws):
        for j in range(i+draws, chains*draws-draws, draws):
            tmp_vals = np.append(tmp_vals, np.array([all_chains_values[i], all_chains_values[j]]))
            # len(tmp_vals) == chains
        mean_of_chains = np.append(mean_of_chains, np.mean(tmp_vals))
        tmp_vals = []
    
    return mean_of_chains # len(mean_of_chains) == draws


def from_posterior(param: str, 
                   samples: Array) -> pm.distributions.Interpolated:
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = sc.stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])

    return pm.distributions.Interpolated(param, x, y)


def sequential_bayes_update(df_to_append: pd.DataFrame, 
                            prior_trace: Array,
                            observed: pd.DataFrame,
                            epochs: iter,
                            draws: int = 2000,
                            tune: int = 1000) -> tuple[pd.DataFrame, dict, list]:
    df_result = df_to_append.copy()
    traces = {}
    kl_divs = []
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
        
        posterior_trace = average_chains_values(param='mu', trace=trace)
        kl_div = sc.special.kl_div(prior_trace, posterior_trace)
        kl_divs.append(kl_div)
        prior_trace = posterior_trace

        print(f'\nepoch{i} done\n')
        gc.collect()
    
    return df_result, traces, kl_divs


def plotbeta(a, b, size=10000, bins=50):
    mode = (a-1) / (a + b -2)
    print("mode ", mode)
    data = np.random.beta(a, b, size)
    plt.hist(data, bins=bins)
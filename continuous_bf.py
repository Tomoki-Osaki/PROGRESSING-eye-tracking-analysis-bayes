# The main analysis
import os
os.chdir("C:/Users/ootmo/OneDrive/Documents/修論_AIと信頼感/py")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import my_funcs as mf
from gc import collect as gc
az.style.use('arviz-darkgrid')

ratios_per_epoch = mf.make_df_ratios_per_epoch(
    num_subjects=30,
    sampling_rate=30,
    recording_duration=300,
    event_interval=30,
    event_duration=2
)

subjects_data = {}
for i in range(50):
    subject_data = {}
    alarms = [True, True, True, False, True, True, False, True, True, False]
    for epoch, alarm in enumerate(alarms):
        if alarm:
            subject_data[epoch] = np.random.beta(a=2, b=2)
        else:
            subject_data[epoch] = np.random.beta(a=2, b=5)
    subjects_data[i] = subject_data

df = pd.DataFrame.from_dict(subjects_data, orient='index')
df[0].hist(bins=10)
for i in range(len(alarms)):
    df[i].hist(bins=10)
    plt.show()
    print(f'epoch{i}')

### inference
trace = mf.calculate_posterior(ratios_per_epoch['epoch1']) 
### 

result = az.summary(trace).loc['mu']
df_result = pd.DataFrame()
df_result = df_result._append(result, ignore_index=True)

az.plot_trace(trace)
plt.show()
    
mean_of_4chains_mu = mf.average_chains_values('mu', trace) # len(mean_of_4chains_mu)==2000
mean_of_4chains_sigma = mf.average_chains_values('sigma', trace) # len(mean_of_4chains_mu)==2000

### inference
df_result, traces = mf.sequential_bayes_update(
    df_to_append=df_result, 
    prior_trace=mean_of_4chains_mu,
    observed=ratios_per_epoch,
    epochs=range(2, 10)
) 
###

fig, ax = plt.subplots()
ax.plot(df_result.index, df_result['mean'])
ax.fill_between(df_result.index, df_result['hdi_3%'], df_result['hdi_97%'], color='b', alpha=.1)
plt.show()

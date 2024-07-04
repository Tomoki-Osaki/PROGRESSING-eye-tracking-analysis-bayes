# The main analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os
import my_funcs as mf
az.style.use('arviz-darkgrid')
os.chdir("C:/Users/ootmo/OneDrive/Documents/修論_AIと信頼感/py")

ratios_per_epoch = mf.make_df_ratios_per_epoch(num_subjects=20)

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
    epochs=range(2, 8)
) 
###

fig, ax = plt.subplots()
ax.plot(df_result.index, df_result['mean'])
ax.fill_between(df_result.index, df_result['hdi_3%'], df_result['hdi_97%'], color='b', alpha=.1)
ax.set_xticks([i for i in range(7)])
plt.show()

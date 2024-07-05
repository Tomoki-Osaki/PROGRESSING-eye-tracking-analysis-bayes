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

df = mf.make_df_subjects_data(100)
df.head()

### inference
trace = mf.calculate_posterior(df['epoch0']) 
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
    observed=df,
    epochs=range(2, 10)
) 
###

fig, ax = plt.subplots()
ax.plot(df_result.index, df_result['mean'])
ax.fill_between(df_result.index, df_result['hdi_3%'], df_result['hdi_97%'], color='b', alpha=.1)
plt.show()

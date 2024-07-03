import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import my_funcs as mf
az.style.use('arviz-darkgrid')

num_subjects = 30

dataframes = {}
for i in range(num_subjects):
    dataframes[i] = mf.create_df()

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
    
colnames = [f'epoch{i+1}' for i in range(len(dataframes_hit[i].columns))]
ratios_per_epoch = pd.DataFrame.from_dict(ratios_per_epoch, orient='index', columns=colnames)
ratios_per_epoch.head()

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




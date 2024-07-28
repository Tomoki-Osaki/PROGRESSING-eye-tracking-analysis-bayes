import stan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nest_asyncio
nest_asyncio.apply()
from plot_inferred_states import plot_inferred_states

df = pd.read_csv("fish-num591.csv")
df.head()

with open("dglm591.stan", "r") as file:
    stan_code = file.read()

data = {"T": len(df),
        "ex": list(df["temperature"]),
        "y": list(df["fish_num"])}

posterior = stan.build(stan_code, data=data)
fit = posterior.sample(num_chains=4, num_samples=1000)
df_res = fit.to_frame()  # pandas `DataFrame, requires pandas
df_res.head()

plot_inferred_states(df["fish_num"], df_res, "lambda_exp")
plot_inferred_states(df["fish_num"], df_res, "lambda_smooth")
plot_inferred_states(df["fish_num"], df_res, "lambda_smooth_fix")


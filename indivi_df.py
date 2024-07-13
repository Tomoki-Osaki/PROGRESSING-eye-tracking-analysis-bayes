import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sampling_rate = 50
recording_time = 10 #min
data_size = sampling_rate * 60 * recording_time

alarm = True
if alarm:
    ratio_of_hit = 0.2 # 50回のうちおおよそ10回
else:
    ratio_of_hit = 0.7

# np.random.poisson(10 or 35)
data_arr = np.array([])
num_fas = 0
repeat = 0
for i in range(600):
    if i % 120 == 0:
        num_fas += 1
        
    if repeat >= 30: # assuming events happen every 30 seconds
        data = np.random.beta(a=10, b=5)
        data_arr = np.append(data_arr, data)
        if repeat == 40:
            repeat = 0
    else:
        if num_fas >= 4:
            data = np.random.beta(a=8, b=6)
            data_arr = np.append(data_arr, data)
            
        data = np.random.beta(a=4, b=10)
        data_arr = np.append(data_arr, data)
            
    repeat += 1
    


def make_df_subjects_data(num_subjects: int,
                          random_seed: int = 1) -> pd.DataFrame:
    np.random.seed(random_seed)
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


df = make_df_subjects_data(30)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# ポアソン分布のパラメータ（平均発生率）
lambda_param = 3.0  # 例として λ = 3.0 とします

# 確率質量関数の計算（k = 0 から 10 まで）
k_values = np.arange(0, 11)  # k = 0, 1, 2, ..., 10
probabilities = poisson.pmf(k_values, lambda_param)

# プロット
plt.figure(figsize=(8, 4))
plt.bar(k_values, probabilities, color='skyblue', edgecolor='black', alpha=0.7)
plt.xticks(k_values)
plt.xlabel('Number of Events (k)')
plt.ylabel('Probability P(X = k)')
plt.title(f'Poisson Distribution (λ = {lambda_param})')
plt.grid(True)
plt.show()



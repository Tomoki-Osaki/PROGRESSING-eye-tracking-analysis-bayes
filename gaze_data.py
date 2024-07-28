import numpy as np
import pandas as pd

df = pd.DataFrame()
data_arr = np.array([])
recording_duration = 720

# 1(not trustable at all) - 7(completely trustable)
trustness = np.full(recording_duration, 4)
events = np.full(recording_duration, 0)

# Recording time: 12 mins
for i in range(recording_duration):
    data = np.random.beta(a=3, b=9)
    data_arr = np.append(data_arr, data)

# Events attracting gaze strongly happen every 30 seconds and last 10 seconds
event_freq = 30
event_duration = 10
for i in range(event_freq, recording_duration, event_freq):
    for j in range(event_duration):
        data_arr[i + j] = np.random.uniform(low=0.8, high=1.0)
        events[i + j] = 1

# Trustness is reported every 40 seconds
for i in range(0, recording_duration, 40):
    trustness_value = np.random.choice([5, 6, 7], p=[0.3, 0.5, 0.2])
    trustness[i:i+40] = trustness_value

df["AOI_ratio"] = data_arr
df["Trustness"] = trustness
df["Alarm"] = events

# Introduce false alarms and keep track of the cumulative count
false_alarms_count = 5
false_alarm_indices = np.random.choice(range(0, recording_duration, event_freq), false_alarms_count, replace=False)
false_alarm_cumulative = np.zeros(recording_duration)

# Update false_alarm_cumulative based on the false_alarm_indices
current_false_alarm_count = 0
for i in range(recording_duration):
    if i in false_alarm_indices:
        current_false_alarm_count += 1
    false_alarm_cumulative[i] = current_false_alarm_count

    if i in false_alarm_indices:
        false_alarm_duration = np.random.randint(90, 121)
        end_false_alarm = min(i + false_alarm_duration, recording_duration)
        trustness[i:end_false_alarm] = np.maximum(1, trustness[i:end_false_alarm] - 1)  # Decrease trustness
        data_arr[i:end_false_alarm] = np.minimum(1.0, data_arr[i:end_false_alarm] + np.random.uniform(0.1, 0.3, size=end_false_alarm - i))  # Increase AOI_ratio

# Update dataframe with the new values
df["AOI_ratio"] = data_arr
df["Trustness"] = trustness
df["Alarm"] = events
df["False_alarms"] = false_alarm_cumulative

print(df)

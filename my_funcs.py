import numpy as np
import pandas as pd

def create_df() -> pd.DataFrame:
    # Parameters
    sampling_rate = 30  # 30Hz
    duration = 300  # 5 minutes in seconds
    total_samples = duration * sampling_rate
    event_interval = 30  # Event every 30 seconds
    event_duration = 2  # Event duration in seconds (assuming strong attraction lasts 2 seconds)
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
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Seconds': timestamps / 60,
        'X': x_coords,
        'Y': y_coords,
        'Fixation_Duration': fixation_durations,
        'AOI_Hit': aoi_hits
    })
    
    return df

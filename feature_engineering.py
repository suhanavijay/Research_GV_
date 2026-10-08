import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_daily_profiles(window_data):
    """
    Collapses the high-frequency sequence into an average daily profile 
    of 48 half-hourly points, strictly segregated into weekday and weekend archetypes.
    """
    df = window_data.copy()
    
    # Extract time of day and determine if the day is a weekend (Saturday=5, Sunday=6)
    df['time_of_day'] = df['DateTime'].dt.time
    df['is_weekend'] = df['DateTime'].dt.dayofweek >= 5

    # Calculate average consumption per household, per 30-min interval, segregated by weekday/weekend
    profiles = df.groupby(['LCLid', 'is_weekend', 'time_of_day'])['KWH_hh_interpolated'].mean().reset_index()
    return profiles

def extract_load_shape_features(profiles):
    """
    Extracts behavioral load-shape summary statistics from the daily archetypes.
    """
    features = []
    
    # Split into weekday and weekend profiles
    weekdays = profiles[~profiles['is_weekend']].copy()
    weekends = profiles[profiles['is_weekend']].copy()
    
    # Convert time to a float representation (0.0 to 23.5) for numerical operations
    weekdays['hour_float'] = weekdays['time_of_day'].apply(lambda x: x.hour + x.minute/60.0)
    
    for lclid in profiles['LCLid'].unique():
        wd = weekdays[weekdays['LCLid'] == lclid]
        we = weekends[weekends['LCLid'] == lclid]
        
        # 1. Peak-to-Average Ratio (Weekday)
        wd_mean = wd['KWH_hh_interpolated'].mean()
        wd_max = wd['KWH_hh_interpolated'].max()
        peak_to_avg = wd_max / wd_mean if wd_mean > 0 else 0
        
        # 2. Timing of Diurnal Peak Demand (Weekday)
        if not wd.empty:
            peak_idx = wd['KWH_hh_interpolated'].idxmax()
            peak_timing = wd.loc[peak_idx, 'hour_float']
        else:
            peak_timing = 0
            
        # 3. Daytime vs Nighttime Ratio (Weekday)
        # Defining daytime as 08:00 to 20:00 (12 hours) and nighttime as the remainder
        daytime_mask = (wd['hour_float'] >= 8) & (wd['hour_float'] < 20)
        day_mean = wd[daytime_mask]['KWH_hh_interpolated'].mean()
        night_mean = wd[~daytime_mask]['KWH_hh_interpolated'].mean()
        day_night_ratio = day_mean / night_mean if night_mean > 0 else day_mean
        
        # 4. Weekday / Weekend Contrast
        we_mean = we['KWH_hh_interpolated'].mean() if not we.empty else 0
        wd_we_contrast = wd_mean / we_mean if we_mean > 0 else wd_mean
        
        features.append({
            'LCLid': lclid,
            'peak_to_average_ratio': peak_to_avg,
            'diurnal_peak_timing': peak_timing,
            'day_night_ratio': day_night_ratio,
            'weekday_weekend_contrast': wd_we_contrast
        })
        
    return pd.DataFrame(features)

def normalize_features(feature_df, method='zscore'):
    """
    Applies scale invariance to the extracted feature vectors using Min-Max scaling or Z-score normalization.
    """
    df = feature_df.copy()
    feature_cols = [
        'peak_to_average_ratio', 
        'diurnal_peak_timing', 
        'day_night_ratio', 
        'weekday_weekend_contrast'
    ]
    
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Normalization method must be 'zscore' or 'minmax'.")
        
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

if __name__ == "__main__":
    # Execution Flow Integration
    # Assuming 'windows' is the list of tuples (window_data, volatility) from data_preprocessing.py
    # for window_data, volatility in windows:
    #     profiles = create_daily_profiles(window_data)
    #     raw_features = extract_load_shape_features(profiles)
    #     normalized_features, fitted_scaler = normalize_features(raw_features, method='zscore')
    pass
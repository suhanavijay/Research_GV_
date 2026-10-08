import pandas as pd
import numpy as np

def load_and_filter_data(filepath):
    """Loads LCL dataset and filters exclusively for standard tariff households."""
    df = pd.read_csv(filepath)
    
    # Strip any hidden leading/trailing spaces from the column headers
    df.columns = df.columns.str.strip()
    
    # Rename the original LCL energy column to match what the pipeline expects
    if 'KWH/hh (per half hour)' in df.columns:
        df.rename(columns={'KWH/hh (per half hour)': 'KWH/hh'}, inplace=True)
        
    df = df[df['stdorToU'] == 'Std'] # Isolate standard tariff
    
    # Convert the column to numeric. 
    # 'errors='coerce'' automatically turns any "Null" text strings into standard NaNs.
    df['KWH/hh'] = pd.to_numeric(df['KWH/hh'], errors='coerce')
    
    return df

def resolve_dst_and_align(df):
    """
    Resolves the Daylight Saving Time paradox by converting localized time to UTC,
    handling duplicated (Autumn) and missing (Spring) timestamps.
    """
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # ADD THIS LINE: Drop any identical duplicate timestamps for the same household
    df = df.drop_duplicates(subset=['LCLid', 'DateTime'], keep='first')
    
    # Set index to datetime for easier resampling
    df.set_index('DateTime', inplace=True)
    
    # Localize to UK timezone, setting ambiguous/nonexistent times to NaT to handle DST
    df.index = df.index.tz_localize('Europe/London', ambiguous='NaT', nonexistent='NaT')
    
    # Convert to UTC to establish a perfectly equidistant time series
    df.index = df.index.tz_convert('UTC')
    
    # Drop the anomalous NaT rows generated during DST transitions
    df = df[df.index.notnull()]
    
    # Ensure strict half-hourly frequency per household
    # Group by household ID and resample
    df_aligned = df.groupby('LCLid').resample('30min').asfreq().reset_index()
    return df_aligned

def impute_and_clean(df):
    """
    Imputes short gaps (1-3 hours) using linear interpolation.
    Flags data for dropping if continuous gaps exceed 24 hours.
    """
    # 1 to 3 hours equals 2 to 6 half-hourly periods
    df['KWH_hh_interpolated'] = df.groupby('LCLid')['KWH/hh'].transform(
        lambda x: x.interpolate(method='linear', limit=6)
    )
    
    # Calculate rolling missing values to flag >24 hour gaps (48 half-hour periods)
    df['missing_flag'] = df['KWH_hh_interpolated'].isna().astype(int)
    df['rolling_missing'] = df.groupby('LCLid')['missing_flag'].transform(
        lambda x: x.rolling(window=48, min_periods=1).sum()
    )
    
    return df

def create_rolling_windows(df, window_weeks=4, step_weeks=1):
    """
    Segments the time series into rolling historical windows.
    Calculates intrinsic consumption volatility (standard deviation) per household per window.
    """
    window_duration = pd.Timedelta(weeks=window_weeks)
    step_duration = pd.Timedelta(weeks=step_weeks)
    
    start_time = df['DateTime'].min()
    end_time = df['DateTime'].max()
    
    windows = []
    current_start = start_time
    window_id = 1
    
    while current_start + window_duration <= end_time:
        current_end = current_start + window_duration
        
        # Extract data for current window
        mask = (df['DateTime'] >= current_start) & (df['DateTime'] < current_end)
        window_data = df.loc[mask].copy()
        
        # Filter out households that had a >24h continuous gap in this specific window
        valid_households = window_data.groupby('LCLid')['rolling_missing'].max()
        valid_lclid = valid_households[valid_households < 48].index
        window_data = window_data[window_data['LCLid'].isin(valid_lclid)]
        
        # Calculate intrinsic consumption volatility (localized standard deviation)
        volatility = window_data.groupby('LCLid')['KWH_hh_interpolated'].std().reset_index()
        volatility.rename(columns={'KWH_hh_interpolated': 'intrinsic_volatility'}, inplace=True)
        volatility['window_id'] = window_id
        
        # Store window metadata
        window_data['window_id'] = window_id
        windows.append((window_data, volatility))
        
        current_start += step_duration
        window_id += 1
        
    return windows

if __name__ == "__main__":
    # Execution Flow
    df_raw = load_and_filter_data("raw/LCL-June2015v2_0.csv")
    df_aligned = resolve_dst_and_align(df_raw)
    df_cleaned = impute_and_clean(df_aligned)
    rolling_datasets = create_rolling_windows(df_cleaned, window_weeks=4, step_weeks=1)
    
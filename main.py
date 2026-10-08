'''
import pandas as pd
import numpy as np

# Import functions from your modules
from data_preprocessing import load_and_filter_data, resolve_dst_and_align, impute_and_clean, create_rolling_windows
from feature_engineering import create_daily_profiles, extract_load_shape_features, normalize_features
from clustering_pipeline import run_longitudinal_clustering
from instability_metrics import calculate_transition_rates, calculate_longitudinal_ari, calculate_certainty_metrics
from forecasting_models import train_lightgbm_quantiles, apply_isotonization, isolate_tail_failures
from forecasting_models import run_forecasting_pipeline

def main():
    # ---------------------------------------------------------
    # Phase 1: Data Engineering (Day 1)
    # ---------------------------------------------------------
    print("Starting Data Preprocessing...")
    filepath = "raw/LCL-June2015v2_0.csv" # Update to your 1-file test set
    df_raw = load_and_filter_data(filepath)
    df_aligned = resolve_dst_and_align(df_raw)
    df_cleaned = impute_and_clean(df_aligned)
    
    # Returns a list of tuples: [(window_data, volatility_df), ...]
    windows = create_rolling_windows(df_cleaned, window_weeks=4, step_weeks=1)
    
    # ---------------------------------------------------------
    # Phase 2: Feature Engineering & Clustering (Day 2)
    # ---------------------------------------------------------
    print("Starting Feature Engineering and Clustering...")
    windows_features_list = []
    volatility_list = []
    
    for window_data, volatility in windows:
        profiles = create_daily_profiles(window_data)
        raw_features = extract_load_shape_features(profiles)
        normalized_features, scaler = normalize_features(raw_features, method='zscore')
        
        # Keep track of window_id for mapping
        normalized_features['window_id'] = volatility['window_id'].iloc[0]
        windows_features_list.append(normalized_features)
        volatility_list.append(volatility)
        
    # Run longitudinal clustering (K-Means & GMM with Hungarian alignment)
    clustered_data, k_optimal = run_longitudinal_clustering(windows_features_list)
    
    # ---------------------------------------------------------
    # Phase 3: Instability Metrics (Day 3)
    # ---------------------------------------------------------
    print("Calculating Instability Metrics...")
    transition_rates = calculate_transition_rates(clustered_data)
    ari_scores = calculate_longitudinal_ari(clustered_data)
    
    # Combine feature lists for distance calculations
    all_features_df = pd.concat(windows_features_list, ignore_index=True)
    
    # Note: For full certainty metrics, pass the exact centroids dictionary from clustering_pipeline
    # Here we calculate the Shannon entropy from the returned clustered_data posteriors
    posterior_cols = [f'posterior_cluster_{c}' for c in range(k_optimal)]
    clustered_data['shannon_entropy'] = clustered_data[posterior_cols].apply(
        lambda row: -np.sum(row * np.log(row + 1e-9)) / np.log(k_optimal), axis=1
    )
    
    # ---------------------------------------------------------
    # Phase 4: Data Export for Statistical Analysis
    # ---------------------------------------------------------
    print("Running LightGBM forecasting to identify tail event failures...")
    # Execute the forecasting model to generate the binary failure labels
    forecast_results = run_forecasting_pipeline(df_cleaned)

    print("Consolidating dataset for R analysis...")
    
    # Merge transition rates, entropy, and intrinsic volatility
    final_df = pd.merge(clustered_data[['LCLid', 'window_id', 'cluster_label', 'shannon_entropy']], 
                        transition_rates, on='LCLid', how='left')
    
    all_volatility_df = pd.concat(volatility_list, ignore_index=True)
    final_df = pd.merge(final_df, all_volatility_df, on=['LCLid', 'window_id'], how='left')
    
    # Merge the 'tail_event_failure' binary labels generated from forecasting_models.py
    # This conditional ensures that whether your forecast function outputs granular window-level 
    # labels or aggregated household-level labels, the merge will not break.
    if 'window_id' in forecast_results.columns:
        final_df = pd.merge(final_df, forecast_results[['LCLid', 'window_id', 'tail_event_failure']], 
                            on=['LCLid', 'window_id'], how='inner')
    else:
        final_df = pd.merge(final_df, forecast_results[['LCLid', 'tail_event_failure']], 
                            on='LCLid', how='inner')
    
    # Export for statistical_analysis.R
    export_path = "final_analytical_dataset.csv"
    final_df.to_csv(export_path, index=False)
    print(f"Pipeline complete. Data exported to {export_path}")

if __name__ == "__main__":
    main()
'''

import pandas as pd
import numpy as np
import glob
import os

# Import functions from your modules
from data_preprocessing import load_and_filter_data, resolve_dst_and_align, impute_and_clean, create_rolling_windows
from feature_engineering import create_daily_profiles, extract_load_shape_features, normalize_features
from clustering_pipeline import run_longitudinal_clustering
from instability_metrics import calculate_transition_rates, calculate_longitudinal_ari, calculate_certainty_metrics
from forecasting_models import train_lightgbm_quantiles, apply_isotonization, isolate_tail_failures
from forecasting_models import run_forecasting_pipeline

def process_single_file(filepath):
    """
    Executes the end-to-end data engineering and analytical pipeline for a single 
    raw data block, returning only the condensed, memory-efficient analytical metrics.
    """
    print(f"\n--- Processing {os.path.basename(filepath)} ---")
    
    # ---------------------------------------------------------
    # Phase 1: Data Engineering 
    # ---------------------------------------------------------
    df_raw = load_and_filter_data(filepath)
    df_aligned = resolve_dst_and_align(df_raw)
    df_cleaned = impute_and_clean(df_aligned)
    
    # Returns a list of tuples: [(window_data, volatility_df), ...]
    windows = create_rolling_windows(df_cleaned, window_weeks=4, step_weeks=1)
    
    # ---------------------------------------------------------
    # Phase 2: Feature Engineering & Clustering 
    # ---------------------------------------------------------
    windows_features_list = []
    volatility_list = []
    
    for window_data, volatility in windows:
        profiles = create_daily_profiles(window_data)
        raw_features = extract_load_shape_features(profiles)
        normalized_features, scaler = normalize_features(raw_features, method='zscore')
        
        # Keep track of window_id for mapping
        normalized_features['window_id'] = volatility['window_id'].iloc[0]
        windows_features_list.append(normalized_features)
        volatility_list.append(volatility)
        
    # Run longitudinal clustering (K-Means & GMM with Hungarian alignment)
    clustered_data, k_optimal = run_longitudinal_clustering(windows_features_list)
    
    # ---------------------------------------------------------
    # Phase 3: Instability Metrics 
    # ---------------------------------------------------------
    transition_rates = calculate_transition_rates(clustered_data)
    ari_scores = calculate_longitudinal_ari(clustered_data)
    
    # Combine feature lists for distance calculations
    all_features_df = pd.concat(windows_features_list, ignore_index=True)
    
    # Calculate the Shannon entropy from the returned clustered_data posteriors
    posterior_cols = [f'posterior_cluster_{c}' for c in range(k_optimal)]
    clustered_data['shannon_entropy'] = clustered_data[posterior_cols].apply(
        lambda row: -np.sum(row * np.log(row + 1e-9)) / np.log(k_optimal), axis=1
    )
    
    # ---------------------------------------------------------
    # Phase 4: Data Export for Statistical Analysis
    # ---------------------------------------------------------
    # Execute the forecasting model to generate the binary failure labels
    forecast_results, raw_forecasts = run_forecasting_pipeline(df_cleaned)

    # Merge transition rates, entropy, and intrinsic volatility
    final_df = pd.merge(clustered_data[['LCLid', 'window_id', 'cluster_label', 'shannon_entropy']], 
                        transition_rates, on='LCLid', how='left')
    
    all_volatility_df = pd.concat(volatility_list, ignore_index=True)
    final_df = pd.merge(final_df, all_volatility_df, on=['LCLid', 'window_id'], how='left')
    
    # Merge the 'tail_event_failure' binary labels generated from forecasting_models.py
    if 'window_id' in forecast_results.columns:
        final_df = pd.merge(final_df, forecast_results[['LCLid', 'window_id', 'tail_event_failure']], 
                            on=['LCLid', 'window_id'], how='inner')
    else:
        final_df = pd.merge(final_df, forecast_results[['LCLid', 'tail_event_failure']], 
                            on='LCLid', how='inner')
                            
    return final_df, raw_forecasts

def main():
    print("Starting Batch Processing Pipeline for all LCL Blocks...")
    
    # Target all CSV files in the raw data folder
    # Note: Ensure your 168 files are saved inside a folder named 'raw' in your project directory
    all_files = glob.glob("raw/*.csv") 
    
    if not all_files:
        print("No CSV files found in the 'raw' directory. Please check the file path.")
        return

    all_processed_data = []
    all_raw_forecasts = []
    
    # Process files one by one to prevent RAM overload
    for file in all_files:
        try:
            condensed_file_data, file_raw_forecasts = process_single_file(file)
            all_processed_data.append(condensed_file_data)
            all_raw_forecasts.append(file_raw_forecasts)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    print("\nConsolidating all processed batches...")
    
    if all_processed_data:
        # Export the R analytical dataset
        final_analytical_dataset = pd.concat(all_processed_data, ignore_index=True)
        final_analytical_dataset.to_csv("final_analytical_dataset.csv", index=False)
        print("Exported final_analytical_dataset.csv")
        
        # Export the raw LightGBM predictions for evaluation metrics
        final_raw_forecasts = pd.concat(all_raw_forecasts, ignore_index=True)
        final_raw_forecasts.to_csv("localized_forecasts.csv", index=False)
        print("Exported localized_forecasts.csv for intervention simulation")
        
    else:
        print("No data was successfully processed.")

if __name__ == "__main__":
    main()
    
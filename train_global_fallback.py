import pandas as pd
import numpy as np
import glob
import os
import lightgbm as lgb
from forecasting_models import apply_isotonization
from data_preprocessing import load_and_filter_data, resolve_dst_and_align, impute_and_clean

def build_global_training_set(file_list, sample_fraction=0.10):
    """
    Extracts a memory-efficient sample of the training data across all files
    to build the universal, unclustered baseline dataset.
    """
    print("Phase 1: Building Global Training Matrix...")
    global_X_train = []
    global_y_train = []
    
    for file in file_list:
        try:
            df_raw = load_and_filter_data(file)
            df_aligned = resolve_dst_and_align(df_raw)
            df = impute_and_clean(df_aligned)
            
            df = df.sort_values(by=['LCLid', 'DateTime']).copy()
            df['target'] = df.groupby('LCLid')['KWH/hh'].shift(-1)
            df['lag_1'] = df.groupby('LCLid')['KWH/hh'].shift(1)
            df = df.dropna(subset=['target', 'lag_1'])
            
            # Strict chronological split to match the localized model (first 80%)
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            
            # Randomly sample the training data to prevent RAM overload
            train_sample = train_df.sample(frac=sample_fraction, random_state=42)
            
            global_X_train.append(train_sample[['lag_1']].values)
            global_y_train.append(train_sample['target'].values)
            
        except Exception as e:
            print(f"Skipping {os.path.basename(file)} during matrix build: {e}")
            
    X_train_final = np.vstack(global_X_train)
    y_train_final = np.concatenate(global_y_train)
    
    print(f"Global training matrix complete. Total pooled observations: {len(y_train_final)}")
    return X_train_final, y_train_final

def train_global_quantiles(X_train, y_train):
    """
    Trains the universal LightGBM quantile regression model.
    """
    print("Phase 2: Training Global LightGBM Models...")
    models = {}
    quantiles = [0.025, 0.5, 0.975]
    
    for q in quantiles:
        print(f"Training quantile {q}...")
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        models[q] = model
        
    return models

def generate_global_forecasts(file_list, models, output_file="global_fallback_forecasts.csv"):
    """
    Iterates through the testing periods, applies the global model, and 
    incrementally writes the unclustered forecasts to disk.
    """
    print("Phase 3: Generating Global Forecasts...")
    
    # Create or overwrite the output file with headers
    pd.DataFrame(columns=['LCLid', 'DateTime', 'q_0.025', 'q_0.5', 'q_0.975']).to_csv(output_file, index=False)
    
    for file in file_list:
        try:
            df_raw = load_and_filter_data(file)
            df_aligned = resolve_dst_and_align(df_raw)
            df = impute_and_clean(df_aligned)
            
            df = df.sort_values(by=['LCLid', 'DateTime']).copy()
            df['target'] = df.groupby('LCLid')['KWH/hh'].shift(-1)
            df['lag_1'] = df.groupby('LCLid')['KWH/hh'].shift(1)
            df = df.dropna(subset=['target', 'lag_1'])
            
            # Isolate the exact same test set (final 20%)
            split_idx = int(len(df) * 0.8)
            test_df = df.iloc[split_idx:].copy()
            
            if test_df.empty:
                continue
                
            X_test = test_df[['lag_1']].values
            
            # Predict using the global models
            preds = pd.DataFrame()
            for q, model in models.items():
                preds[f'q_{q}'] = model.predict(X_test)
                
            # Apply your existing isotonization to prevent quantile crossing
            iso_preds = apply_isotonization(preds)
            
            # Format for export
            export_df = pd.DataFrame({
                'LCLid': test_df['LCLid'].values,
                'DateTime': test_df['DateTime'].values,
                'q_0.025': iso_preds['q_0.025'],
                'q_0.5': iso_preds['q_0.5'],
                'q_0.975': iso_preds['q_0.975']
            })
            
            # Append to CSV
            export_df.to_csv(output_file, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"Skipping {os.path.basename(file)} during prediction: {e}")
            
    print(f"Global fallback forecasts successfully exported to {output_file}")

if __name__ == "__main__":
    all_files = glob.glob("raw/*.csv")
    if not all_files:
        print("No CSV files found in the 'raw' directory.")
        exit()
        
    X_train_global, y_train_global = build_global_training_set(all_files, sample_fraction=0.10)
    global_models = train_global_quantiles(X_train_global, y_train_global)
    generate_global_forecasts(all_files, global_models)
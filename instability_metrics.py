import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from scipy.stats import entropy
from scipy.spatial.distance import euclidean

def calculate_transition_rates(clustered_data):
    """
    Calculates the transition rate for each household, defined as the number of 
    K-Means label changes between consecutive windows normalized by total possible transitions.
    """
    df = clustered_data.sort_values(by=['LCLid', 'window_id']).copy()
    
    # Identify if the cluster label changed from the previous window for the same household
    df['prev_label'] = df.groupby('LCLid')['cluster_label'].shift(1)
    df['is_transition'] = (df['cluster_label'] != df['prev_label']) & (df['prev_label'].notna())
    
    # Calculate transition rate per household
    transition_summary = df.groupby('LCLid').agg(
        total_transitions=('is_transition', 'sum'),
        valid_windows=('cluster_label', 'count')
    ).reset_index()
    
    # Normalize by (N-1) possible transitions
    transition_summary['transition_rate'] = transition_summary['total_transitions'] / (transition_summary['valid_windows'] - 1)
    # Handle edge case where a household only appears in one window
    transition_summary['transition_rate'] = transition_summary['transition_rate'].fillna(0)
    
    return transition_summary[['LCLid', 'transition_rate']]

def calculate_longitudinal_ari(clustered_data):
    """
    Computes the Adjusted Rand Index (ARI) between adjacent temporal windows 
    to measure structural stability of the population.
    """
    windows = sorted(clustered_data['window_id'].unique())
    ari_results = []
    
    for i in range(len(windows) - 1):
        w_current = windows[i]
        w_next = windows[i+1]
        
        df_curr = clustered_data[clustered_data['window_id'] == w_current][['LCLid', 'cluster_label']]
        df_next = clustered_data[clustered_data['window_id'] == w_next][['LCLid', 'cluster_label']]
        
        # Merge to find households present in both adjacent windows
        merged = pd.merge(df_curr, df_next, on='LCLid', suffixes=('_curr', '_next'))
        
        if len(merged) > 1:
            ari = adjusted_rand_score(merged['cluster_label_curr'], merged['cluster_label_next'])
            ari_results.append({'window_pair': f"{w_current}_to_{w_next}", 'ari_score': ari})
            
    return pd.DataFrame(ari_results)

def calculate_certainty_metrics(clustered_data, k_optimal, centroids_dict, features_data):
    """
    Calculates normalized Shannon entropy from the GMM posterior probability vectors 
    and computes the silhouette margin for each household per window.
    """
    df = clustered_data.copy()
    posterior_cols = [f'posterior_cluster_{c}' for c in range(k_optimal)]
    
    # Calculate Normalized Shannon Entropy
    # Normalizing by log(K) bounds the entropy between 0 (certainty) and 1 (maximum ambiguity)
    df['shannon_entropy'] = df[posterior_cols].apply(
        lambda row: entropy(row.values, base=np.e) / np.log(k_optimal), axis=1
    )
    
    # Calculate Silhouette Margin
    # Merging features to calculate Euclidean distances to centroids
    df = pd.merge(df, features_data, on=['LCLid', 'window_id'])
    feature_cols = ['peak_to_average_ratio', 'diurnal_peak_timing', 'day_night_ratio', 'weekday_weekend_contrast']
    
    margins = []
    for _, row in df.iterrows():
        w_id = row['window_id']
        if w_id in centroids_dict:
            window_centroids = centroids_dict[w_id]
            x_i = row[feature_cols].values.astype(float)
            
            # Compute distances to all centroids in the current window
            distances = [euclidean(x_i, c) for c in window_centroids]
            distances.sort()
            
            # Margin = Distance to second closest - Distance to closest
            if len(distances) >= 2:
                margin = distances[1] - distances[0]
            else:
                margin = 0
            margins.append(margin)
        else:
            margins.append(np.nan)
            
    df['silhouette_margin'] = margins
    
    return df[['LCLid', 'window_id', 'cluster_label', 'shannon_entropy', 'silhouette_margin'] + posterior_cols]

if __name__ == "__main__":
    # Execution Flow Integration
    # Assuming 'clustered_data' is the output DataFrame from clustering_pipeline.py
    # Assuming 'centroids_dict' maps window_id to its aligned centroids array
    # Assuming 'features_data' contains the normalized feature vectors per LCLid per window_id
    
    # transition_df = calculate_transition_rates(clustered_data)
    # ari_df = calculate_longitudinal_ari(clustered_data)
    # certainty_df = calculate_certainty_metrics(clustered_data, k_optimal=4, centroids_dict={}, features_data=pd.DataFrame())
    pass
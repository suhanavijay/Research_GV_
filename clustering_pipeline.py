import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def determine_optimal_clusters(feature_matrix, max_k=10):
    """
    Determines the optimal number of clusters on the initial window 
    utilizing the Silhouette score.
    """
    best_k = 2
    best_score = -1
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        score = silhouette_score(feature_matrix, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
            
    return best_k

def cluster_window(feature_matrix, k):
    """
    Deploys K-Means for hard boundary assignments and GMM for soft clustering.
    Returns the models, cluster labels, posterior probabilities, and centroids.
    """
    # K-Means for hard boundaries
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    hard_labels = kmeans.fit_predict(feature_matrix)
    centroids = kmeans.cluster_centers_
    
    # GMM for soft probabilistic assignments
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(feature_matrix)
    posteriors = gmm.predict_proba(feature_matrix)
    
    return hard_labels, posteriors, centroids, kmeans, gmm

def align_clusters(prev_centroids, curr_centroids, curr_labels, curr_posteriors):
    """
    Resolves the cluster label alignment problem across rolling windows using 
    a pairwise Euclidean distance cost matrix and the Hungarian algorithm.
    """
    # Compute pairwise Euclidean distance cost matrix between historical and current centroids
    cost_matrix = cdist(curr_centroids, prev_centroids, metric='euclidean')
    
    # Apply Hungarian algorithm (linear assignment) to find optimal one-to-one mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create a mapping dictionary: current_label -> previous_label
    mapping = {curr: prev for curr, prev in zip(row_ind, col_ind)}
    
    # Remap hard labels
    aligned_labels = np.array([mapping[label] for label in curr_labels])
    
    # Remap posterior probabilities to align columns with previous cluster IDs
    aligned_posteriors = np.zeros_like(curr_posteriors)
    for curr_col, prev_col in mapping.items():
        aligned_posteriors[:, prev_col] = curr_posteriors[:, curr_col]
        
    # Remap centroids
    aligned_centroids = np.zeros_like(curr_centroids)
    for curr_row, prev_row in mapping.items():
        aligned_centroids[prev_row] = curr_centroids[curr_row]
        
    return aligned_labels, aligned_posteriors, aligned_centroids, mapping

def run_longitudinal_clustering(windows_features_list):
    """
    Iterates through all rolling windows, determines K on the first window, 
    and applies aligned clustering across time.
    """
    results = []
    prev_centroids = None
    k_optimal = None
    
    for window_idx, feature_df in enumerate(windows_features_list):
        # Extract numerical features for clustering
        feature_cols = ['peak_to_average_ratio', 'diurnal_peak_timing', 
                        'day_night_ratio', 'weekday_weekend_contrast']
        X = feature_df[feature_cols].values
        
        # Determine K on the first window and hold it constant
        if window_idx == 0:
            k_optimal = determine_optimal_clusters(X, max_k=8)
            
        # Run independent clustering for the current window
        hard_labels, posteriors, centroids, kmeans, gmm = cluster_window(X, k_optimal)
        
        # Align labels if it's not the first window
        if prev_centroids is not None:
            aligned_labels, aligned_posteriors, aligned_centroids, _ = align_clusters(
                prev_centroids, centroids, hard_labels, posteriors
            )
        else:
            aligned_labels = hard_labels
            aligned_posteriors = posteriors
            aligned_centroids = centroids
            
        # Update previous centroids for the next iteration tracking
        prev_centroids = aligned_centroids
        
        # Store aligned outputs back into a DataFrame
        result_df = feature_df[['LCLid']].copy()
        result_df['window_id'] = window_idx + 1
        result_df['cluster_label'] = aligned_labels
        
        # Store individual posterior probabilities
        for c in range(k_optimal):
            result_df[f'posterior_cluster_{c}'] = aligned_posteriors[:, c]
            
        results.append(result_df)
        
    return pd.concat(results, ignore_index=True), k_optimal

if __name__ == "__main__":
    # Execution Flow Integration
    # Assuming 'windows_features_list' is a list of normalized feature DataFrames from feature_engineering.py
    # clustered_data, k_optimal = run_longitudinal_clustering(windows_features_list)
    pass
import pandas as pd
import numpy as np
from evaluation_metrics import evaluate_pipeline

def simulate_dynamic_routing(clustered_preds, global_preds, metrics_df, entropy_quantile=0.75):
    """
    Executes the dynamic fallback routing intervention.
    Flags households exceeding the operational instability threshold (e.g., top quartile of Shannon entropy)
    and dynamically routes them from the localized cluster model to an alternative global fallback model.
    """
    df = pd.merge(metrics_df, clustered_preds, on=['LCLid', 'window_id'], suffixes=('', '_cluster'))
    df = pd.merge(df, global_preds, on=['LCLid', 'window_id'], suffixes=('', '_global'))
    
    # Establish the operational instability threshold
    threshold = df['shannon_entropy'].quantile(entropy_quantile)
    df['is_flagged_unstable'] = df['shannon_entropy'] >= threshold
    
    # Route predictions based on the instability flag
    df['routed_q025'] = np.where(df['is_flagged_unstable'], df['q_0.025_global'], df['q_0.025'])
    df['routed_q50'] = np.where(df['is_flagged_unstable'], df['q_0.5_global'], df['q_0.5'])
    df['routed_q975'] = np.where(df['is_flagged_unstable'], df['q_0.975_global'], df['q_0.975'])
    
    print(f"Instability Threshold set at Shannon Entropy: {threshold:.4f}")
    print(f"Routed {df['is_flagged_unstable'].sum()} erratic households to the global model.")
    
    return df, threshold

def evaluate_intervention_effect(routed_df, actual_outcomes, predicted_risk_probs):
    """
    Compares the predictive performance of the baseline strategy against the intervention strategy.
    Evaluates risk discrimination and probabilistic risk calibration using PR-AUC, Brier, Winkler, and CRPS.
    """
    # Baseline: 100% Localized Cluster Models
    baseline_quantiles = {
        0.025: routed_df['q_0.025'],
        0.5: routed_df['q_0.5'],
        0.975: routed_df['q_0.975']
    }
    
    print("Evaluating Baseline (100% Localized Models)...")
    baseline_metrics = evaluate_pipeline(
        actual_outcomes=actual_outcomes,
        predicted_probs=predicted_risk_probs,
        forecast_actuals=routed_df['actual'],
        forecast_quantiles=baseline_quantiles
    )
    
    # Intervention: Dynamically Routed Models
    routed_quantiles = {
        0.025: routed_df['routed_q025'],
        0.5: routed_df['routed_q50'],
        0.975: routed_df['routed_q975']
    }
    
    print("Evaluating Intervention (Dynamically Routed Models)...")
    intervention_metrics = evaluate_pipeline(
        actual_outcomes=actual_outcomes,
        predicted_probs=predicted_risk_probs, 
        forecast_actuals=routed_df['actual'],
        forecast_quantiles=routed_quantiles
    )
    
    comparison_df = pd.DataFrame({
        'Baseline': baseline_metrics,
        'Intervention': intervention_metrics
    })
    
    # Calculate the percentage improvement for the interval metrics (lower is better for Winkler, CRPS, Brier)
    comparison_df['Improvement (%)'] = np.where(
        comparison_df.index.isin(['Winkler_Score', 'CRPS_Approx', 'Brier_Score']),
        ((comparison_df['Baseline'] - comparison_df['Intervention']) / comparison_df['Baseline']) * 100,
        ((comparison_df['Intervention'] - comparison_df['Baseline']) / comparison_df['Baseline']) * 100
    )
    
    return comparison_df

def temporal_holdout_test(df, date_col='DateTime'):
    """
    Executes a final temporal holdout test on the last six months of the dataset.
    Simulates real-world deployment on unseen, future data.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max()
    cutoff_date = max_date - pd.DateOffset(months=6)
    
    training_period = df[df[date_col] < cutoff_date]
    deployment_period = df[df[date_col] >= cutoff_date]
    
    print(f"Training Data: {len(training_period)} rows.")
    print(f"Deployment (Holdout) Data: {len(deployment_period)} rows.")
    
    return training_period, deployment_period

def run_window_ablation_study(window_lengths=[2, 4, 8]):
    """
    Runs ablation studies by varying the rolling window lengths (2, 4, and 8 weeks).
    """
    results = {}
    for weeks in window_lengths:
        print(f"Running pipeline ablation for {weeks}-week rolling windows...")
        # Placeholder for ablation execution logic linking back to the data engineering module
        results[weeks] = "Completed" 
    return results


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import os
    from evaluation_metrics import evaluate_pipeline
    
    print("Initiating Final Intervention Simulation with Descriptive Stats...")
    
    # 1. Load and Map Data
    metrics_df = pd.read_csv("final_analytical_dataset.csv", usecols=['LCLid', 'transition_rate'])
    metrics_mapping = metrics_df.drop_duplicates(subset=['LCLid']).set_index('LCLid')['transition_rate']
    df_local = pd.read_csv("localized_forecasts.csv")
    df_global = pd.read_csv("global_fallback_forecasts.csv")

    if 'target' in df_local.columns:
        df_local = df_local.rename(columns={'target': 'actual'})

    df_local['transition_rate'] = df_local['LCLid'].map(metrics_mapping)
    df_local = df_local.dropna(subset=['transition_rate', 'actual', 'tail_event_failure'])

    # 2. Align Local and Global Forecasts
    df_local = df_local.set_index(['LCLid', 'DateTime'])
    df_global = df_global.set_index(['LCLid', 'DateTime'])
    df_local['q_0.025_global_tuned'] = df_global['q_0.025']
    df_local['q_0.5_global_tuned'] = df_global['q_0.5']
    df_local['q_0.975_global_tuned'] = df_global['q_0.975']
    df_local = df_local.reset_index().dropna(subset=['q_0.025_global_tuned'])

    # --- NEW: Calculate Tail-Event Prevalence ---
    total_obs = len(df_local)
    failures = int(df_local['tail_event_failure'].sum())
    prevalence = (failures / total_obs) * 100
    print("\n--- Descriptive Statistics ---")
    print(f"Total Observations: {total_obs}")
    print(f"Tail-Event Failures: {failures} ({prevalence:.2f}%)")

    # 3. Dynamic Routing & Random Control Selection
    threshold = df_local['transition_rate'].quantile(0.75)
    df_local['is_flagged_unstable'] = df_local['transition_rate'] >= threshold
    
    unique_households = df_local['LCLid'].unique()
    flagged_households = df_local[df_local['is_flagged_unstable']]['LCLid'].unique()
    num_flagged = len(flagged_households)
    
    np.random.seed(42)
    random_control_households = np.random.choice(unique_households, size=num_flagged, replace=False)
    df_local['is_flagged_random'] = df_local['LCLid'].isin(random_control_households)

    # Swap 1: True Tuned Intervention
    df_local['routed_q025_tuned'] = np.where(df_local['is_flagged_unstable'], df_local['q_0.025_global_tuned'], df_local['q_0.025'])
    df_local['routed_q50_tuned']  = np.where(df_local['is_flagged_unstable'], df_local['q_0.5_global_tuned'], df_local['q_0.5'])
    df_local['routed_q975_tuned'] = np.where(df_local['is_flagged_unstable'], df_local['q_0.975_global_tuned'], df_local['q_0.975'])

    # Swap 2: Random Control Intervention
    df_local['routed_q025_random'] = np.where(df_local['is_flagged_random'], df_local['q_0.025_global_tuned'], df_local['q_0.025'])
    df_local['routed_q50_random']  = np.where(df_local['is_flagged_random'], df_local['q_0.5_global_tuned'], df_local['q_0.5'])
    df_local['routed_q975_random'] = np.where(df_local['is_flagged_random'], df_local['q_0.975_global_tuned'], df_local['q_0.975'])

    # 4. Evaluate Pipelines
    predicted_risk_probs = df_local['transition_rate'] / df_local['transition_rate'].max()
    
    baseline_metrics = evaluate_pipeline(df_local['tail_event_failure'], predicted_risk_probs, df_local['actual'], 
                                         {0.025: df_local['q_0.025'], 0.5: df_local['q_0.5'], 0.975: df_local['q_0.975']})
    
    tuned_metrics = evaluate_pipeline(df_local['tail_event_failure'], predicted_risk_probs, df_local['actual'], 
                                      {0.025: df_local['routed_q025_tuned'], 0.5: df_local['routed_q50_tuned'], 0.975: df_local['routed_q975_tuned']})

    random_metrics = evaluate_pipeline(df_local['tail_event_failure'], predicted_risk_probs, df_local['actual'], 
                                       {0.025: df_local['routed_q025_random'], 0.5: df_local['routed_q50_random'], 0.975: df_local['routed_q975_random']})

    comparison_df = pd.DataFrame({'Baseline': baseline_metrics, 'Random_Control': random_metrics, 'Tuned_Intervention': tuned_metrics})
    
    # --- NEW: Calculate Point Errors (MAE & RMSE) ---
    def calc_point_errors(actual, pred):
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred)**2))
        return mae, rmse

    base_mae, base_rmse = calc_point_errors(df_local['actual'], df_local['q_0.5'])
    tuned_mae, tuned_rmse = calc_point_errors(df_local['actual'], df_local['routed_q50_tuned'])
    random_mae, random_rmse = calc_point_errors(df_local['actual'], df_local['routed_q50_random'])

    point_metrics = pd.DataFrame({
        'Baseline': [base_mae, base_rmse],
        'Random_Control': [random_mae, random_rmse],
        'Tuned_Intervention': [tuned_mae, tuned_rmse]
    }, index=['MAE', 'RMSE'])
    
    comparison_df = pd.concat([comparison_df, point_metrics])
    
    print("\n--- Final Intervention Matrix (with Point Errors) ---")
    print(comparison_df)

    # =========================================================================
    # --- NEW: Logging Details and CSV Export ---
    # =========================================================================
    print("\n--- Routing Configuration Logs ---")
    print(f"Total Eligible Households (Denominator): {len(unique_households)}")
    print(f"Transition Rate Threshold (75th Percentile): {threshold:.4f}")
    print(f"Number of Flagged Households (Intervention/Control size): {num_flagged}")
    
    print("\nTargeted Household IDs (Top 25% TR):")
    print(list(flagged_households))
    
    print("\nRandom Control Household IDs (Seed=42):")
    print(list(random_control_households))

    # Save to manuscript_assets folder
    os.makedirs("manuscript_assets", exist_ok=True)
    export_path = "manuscript_assets/intervention_results.csv"
    
    # We use reset_index() so that the metric names (MAE, Winkler, etc.) are saved as a column 
    # rather than dropped entirely when index=False is passed.
    comparison_df.reset_index().rename(columns={'index': 'Metric'}).to_csv(export_path, index=False)
    print(f"\nSuccess! Intervention results safely exported to: {export_path}")
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. LightGBM Quantile Regression
# ==========================================

def train_lightgbm_quantiles(X_train, y_train, X_test, quantiles=[0.025, 0.5, 0.975]):
    """
    Trains independent LightGBM models for specified quantiles using the pinball loss objective.
    Predicts distribution-free bounds for the out-of-sample data.
    """
    predictions = {}
    
    for alpha in quantiles:
        # Initialize LightGBM with quantile objective and specific alpha
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            metric='quantile',
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        )
        
        # Chronological forward-rolling training
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[f'q_{alpha}'] = preds
        
    return pd.DataFrame(predictions)

# ==========================================
# 2. Quantile LSTM Architecture (PyTorch)
# ==========================================

class QuantileLSTM(nn.Module):
    """
    Recurrent architecture to capture complex temporal dependencies 
    and project outputs through a multi-quantile head.
    """
    def __init__(self, input_size, hidden_size, num_layers, quantiles):
        super(QuantileLSTM, self).__init__()
        self.quantiles = quantiles
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Output size equals the number of target quantiles (e.g., 3 for 2.5, 50, 97.5)
        self.fc = nn.Linear(hidden_size, len(quantiles))
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

def pinball_loss(preds, target, quantiles):
    """
    Computes the asymmetric piecewise linear penalty (pinball loss) for multiple quantiles.
    Aggressively penalizes under-predictions for high quantiles and over-predictions for low quantiles.
    """
    loss = 0
    for i, q in enumerate(quantiles):
        error = target - preds[:, i]
        loss += torch.max((q - 1) * error, q * error).mean()
    return loss

def train_evaluate_lstm(X_train, y_train, X_test, quantiles=[0.025, 0.5, 0.975], epochs=50):
    """
    Trains the Quantile LSTM using the custom pinball loss function.
    """
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False) # Strict chronological order
    
    model = QuantileLSTM(input_size=X_train.shape[2], hidden_size=64, num_layers=2, quantiles=quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = pinball_loss(preds, batch_y, quantiles)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).numpy()
        
    df_preds = pd.DataFrame(test_preds, columns=[f'q_{q}' for q in quantiles])
    return df_preds

# ==========================================
# 3. Post-Hoc Isotonization & Failure Labeling
# ==========================================

def apply_isotonization(predictions_df, quantiles=[0.025, 0.5, 0.975]):
    """
    Resolves quantile crossing by sorting the predicted quantiles per row, 
    ensuring strict monotonicity (lower bound <= median <= upper bound).
    """
    cols = [f'q_{q}' for q in quantiles]
    # Sorting operators efficiently re-order the predicted quantiles
    sorted_preds = np.sort(predictions_df[cols].values, axis=1)
    
    isotonized_df = pd.DataFrame(sorted_preds, columns=cols, index=predictions_df.index)
    return isotonized_df

def isolate_tail_failures(predictions_df, actual_y, quantile_95th_threshold):
    """
    Isolates extreme failures by tagging absolute errors exceeding the 95th percentile,
    or observations breaching the 2.5th-97.5th prediction interval.
    """
    df = predictions_df.copy()
    df['actual'] = actual_y
    df['absolute_error'] = np.abs(df['actual'] - df['q_0.5'])
    
    # Label 1: Absolute error exceeds the 95th percentile threshold
    condition_1 = df['absolute_error'] > quantile_95th_threshold
    
    # Label 2: Actual load completely breaches the 2.5th-97.5th boundaries
    condition_2 = (df['actual'] < df['q_0.025']) | (df['actual'] > df['q_0.975'])
    
    # Binary classification label for a "Tail-Event Failure"
    df['tail_event_failure'] = (condition_1 | condition_2).astype(int)
    
    return df

def run_forecasting_pipeline(df):
    """
    Orchestrates the temporal data splitting, LightGBM training, and failure extraction.
    Returns both the aggregated failure labels and the raw quantile predictions.
    """
    df = df.sort_values(by=['LCLid', 'DateTime']).copy()
    df['target'] = df.groupby('LCLid')['KWH/hh'].shift(-1)
    df['lag_1'] = df.groupby('LCLid')['KWH/hh'].shift(1)
    df = df.dropna(subset=['target', 'lag_1'])
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[['lag_1']].values
    y_train = train_df['target'].values
    X_test = test_df[['lag_1']].values
    y_test = test_df['target'].values
    
    lgb_preds = train_lightgbm_quantiles(X_train, y_train, X_test)
    lgb_iso_preds = apply_isotonization(lgb_preds)
    
    threshold_95 = np.percentile(np.abs(y_test - lgb_iso_preds['q_0.5']), 95)
    failure_labels = isolate_tail_failures(lgb_iso_preds, y_test, threshold_95)
    
    # Attach tracking IDs to the raw forecasts
    failure_labels['LCLid'] = test_df['LCLid'].values
    failure_labels['DateTime'] = test_df['DateTime'].values
    
    # Aggregate failure rate per household
    household_failures = failure_labels.groupby('LCLid', as_index=False)['tail_event_failure'].mean()
    overall_avg_rate = household_failures['tail_event_failure'].mean()
    household_failures['tail_event_failure'] = (household_failures['tail_event_failure'] > overall_avg_rate).astype(int)
    
    return household_failures, failure_labels

if __name__ == "__main__":
    # Execution Flow Integration
    # Assuming X_train, y_train, X_test, y_test are generated temporally via windowing logic
    # lgb_preds = train_lightgbm_quantiles(X_train, y_train, X_test)
    # lgb_iso_preds = apply_isotonization(lgb_preds)
    # threshold_95 = np.percentile(np.abs(y_test - lgb_iso_preds['q_0.5']), 95)
    # failure_labels = isolate_tail_failures(lgb_iso_preds, y_test, threshold_95)
    pass
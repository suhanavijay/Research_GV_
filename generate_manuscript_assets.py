import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# IEEE Formatting Configuration
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# Define output directory
OUT_DIR = "manuscript_assets"
os.makedirs(OUT_DIR, exist_ok=True)

print("Generating IEEE-formatted vector graphics and tables...")

# ==========================================
# Load Datasets
# ==========================================
try:
    df_analytical = pd.read_csv("final_analytical_dataset.csv")
    df_glm_preds = pd.read_csv("glm_predictions.csv")
    df_reg_results = pd.read_csv("regression_results.csv")
    df_intervention = pd.read_csv(f"{OUT_DIR}/intervention_results.csv") 
except FileNotFoundError as e:
    print(f"Error loading required files: {e}")
    print("Please ensure all R and Python exports have been run first.")
    exit()

# Collapse analytical dataset to household-level metrics for plotting
df_households = df_analytical.drop_duplicates(subset=['LCLid']).copy()

# ==========================================
# Figure 2: Representative Behavioral Trajectories
# ==========================================
print("Generating Figure 2: Trajectories...")
fig2, ax2 = plt.subplots(figsize=(7, 3))

# Selection Rule: 
# Find an erratic household (TR > 0.4) that also has a long history (e.g., > 80 windows)
window_counts = df_analytical.groupby('LCLid').size()
valid_households = window_counts[window_counts > 80].index
df_valid = df_households[df_households['LCLid'].isin(valid_households)]

stable_id = df_valid.loc[df_valid['transition_rate'].idxmin(), 'LCLid']
erratic_id = df_valid.loc[df_valid['transition_rate'].idxmax(), 'LCLid']

df_stable = df_analytical[df_analytical['LCLid'] == stable_id].sort_values('window_id')
df_erratic = df_analytical[df_analytical['LCLid'] == erratic_id].sort_values('window_id')

# Step plots for cluster assignments
ax2.step(df_stable['window_id'], df_stable['cluster_label'], where='mid', 
         label=f'Stable Profile (TR = {df_stable["transition_rate"].iloc[0]:.2f})', 
         linewidth=2, color='#1f77b4', marker='o')
ax2.step(df_erratic['window_id'], df_erratic['cluster_label'], where='mid', 
         label=f'Erratic Profile (TR = {df_erratic["transition_rate"].iloc[0]:.2f})', 
         linewidth=2, color='#d62728', linestyle='--', marker='x')

ax2.set_xlabel('Rolling Analysis Window')
ax2.set_ylabel('Assigned Behavioral Cluster (K)')
ax2.set_yticks(range(int(df_analytical['cluster_label'].max()) + 1))
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

fig2.savefig(f"{OUT_DIR}/Figure_2_Trajectories.pdf", format='pdf')
plt.close(fig2)


# ==========================================
# Figure 3: Transition Rate vs Intrinsic Volatility
# ==========================================
print("Generating Figure 3: Scatter Plot...")
fig3, ax3 = plt.subplots(figsize=(5, 4))

ax3.scatter(df_households['transition_rate'], df_households['intrinsic_volatility'], 
            alpha=0.4, edgecolors='w', linewidth=0.5, color='#2ca02c', s=30)

ax3.set_xlabel('Temporal Behavioral Instability (Transition Rate)')
ax3.set_ylabel('Intrinsic Consumption Volatility (CV)')
ax3.grid(True, linestyle=':', alpha=0.6)

fig3.savefig(f"{OUT_DIR}/Figure_3_Instability_vs_Volatility.pdf", format='pdf')
plt.close(fig3)


# ==========================================
# Figure 4: Adjusted GLM Curve
# ==========================================
print("Generating Figure 4: GLM Marginal Effects...")
fig4, ax4 = plt.subplots(figsize=(6, 4))

# Plotting against the standardized transition rate to match model scale
ax4.plot(df_glm_preds['standardized_transition_rate'], df_glm_preds['predicted_probability'], 
         color='black', linewidth=2, label='Predicted Failure Probability')
ax4.fill_between(df_glm_preds['standardized_transition_rate'], 
                 df_glm_preds['lower_CI'], 
                 df_glm_preds['upper_CI'], 
                 color='black', alpha=0.15, label='95% Confidence Interval')

ax4.set_xlabel('Transition Rate (Standardized)')
ax4.set_ylabel('Probability of Tail-Event Failure (Y=1)')
ax4.grid(True, linestyle=':', alpha=0.6)
ax4.legend(loc='upper left')

fig4.savefig(f"{OUT_DIR}/Figure_4_GLM_Curve.pdf", format='pdf')
plt.close(fig4)


# ==========================================
# Figure 5: Routing Performance (Winkler & CRPS)
# ==========================================
print("Generating Figure 5: Intervention Bar Charts...")
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(8, 3.5))

winkler_row = df_intervention[df_intervention['Metric'].str.contains('Winkler', case=False, na=False)].iloc[0]
crps_row = df_intervention[df_intervention['Metric'].str.contains('CRPS', case=False, na=False)].iloc[0]

labels = ['Baseline', 'Profile-Based\n(Targeted)', 'Random Control']
colors = ['#7f7f7f', '#1f77b4', '#d62728']

# Plot Winkler Score
winkler_vals = [winkler_row['Baseline'], winkler_row['Tuned_Intervention'], winkler_row['Random_Control']]
ax5a.bar(labels, winkler_vals, color=colors, edgecolor='black', width=0.6)
ax5a.set_ylabel('Winkler Score (Lower is Better)')
ax5a.set_ylim(min(winkler_vals) * 0.99, max(winkler_vals) * 1.01)
ax5a.grid(axis='y', linestyle=':', alpha=0.6)

# Plot CRPS Approx
crps_vals = [crps_row['Baseline'], crps_row['Tuned_Intervention'], crps_row['Random_Control']]
ax5b.bar(labels, crps_vals, color=colors, edgecolor='black', width=0.6)
ax5b.set_ylabel('CRPS (Lower is Better)')
ax5b.set_ylim(min(crps_vals) * 0.99, max(crps_vals) * 1.01)
ax5b.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
fig5.savefig(f"{OUT_DIR}/Figure_5_Intervention_Metrics.pdf", format='pdf')
plt.close(fig5)


# ==========================================
# Table 1: Cleaned Regression Results
# ==========================================
print("Generating Table 1: Cleaned Regression Results...")
table1 = df_reg_results.copy()

# Map raw formula predictors to clean publication names
predictor_map = {
    '(Intercept)': 'Intercept',
    'scale(transition_rate)': 'Transition Rate (Scaled)',
    'scale(shannon_entropy)': 'Shannon Entropy (Scaled)',
    'scale(intrinsic_volatility)': 'Intrinsic Volatility (Scaled)'
}
table1['Predictor'] = table1['Predictor'].replace(predictor_map)

# Format Odds Ratios and CIs into a single string: "OR [Lower, Upper]"
table1['OR (95% CI)'] = table1.apply(
    lambda row: f"{row['Odds_Ratio']:.3f} [{row['CI_Lower_OR']:.3f}, {row['CI_Upper_OR']:.3f}]", axis=1
)

# Scientifically format p-values
table1['p-value'] = table1['p_value'].apply(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}")

# Round numeric columns
table1['Log_Odds'] = table1['Log_Odds'].round(3)
table1['Std_Error'] = table1['Std_Error'].round(3)
table1['ICC'] = table1['ICC'].round(4)

# Subset to final presentation columns
table1_clean = table1[['Model', 'Predictor', 'Log_Odds', 'Std_Error', 'OR (95% CI)', 'p-value', 'ICC']]
table1_clean.to_csv(f"{OUT_DIR}/table1_regression.csv", index=False)


# ==========================================
# Table 2: Cleaned Intervention Results
# ==========================================
print("Generating Table 2: Cleaned Intervention Results...")
table2 = df_intervention[['Metric', 'Baseline', 'Random_Control', 'Tuned_Intervention']].copy()

# Rename columns to match manuscript
table2 = table2.rename(columns={
    'Random_Control': 'Random Control',
    'Tuned_Intervention': 'Targeted'
})

# Format numeric values to 6 decimal places strictly mapping to Table I in manuscript
for col in ['Baseline', 'Random Control', 'Targeted']:
    table2[col] = pd.to_numeric(table2[col], errors='coerce').apply(lambda x: f"{x:.6f}")

table2.to_csv(f"{OUT_DIR}/table2_intervention.csv", index=False)

print("\n=====================================================")
print(f"Success! All vector graphics and tables exported to: {os.path.abspath(OUT_DIR)}")
print("=====================================================")
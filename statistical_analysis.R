# statistical_analysis.R
# Required packages for mixed-effects modeling and data manipulation
#install.packages(c("lme4", "dplyr", "broom.mixed"))
library(lme4)
library(dplyr)

run_mixed_effects_model <- function(data_path) {
  # Load the aggregated longitudinal dataset containing failure labels, 
  # instability metrics, and intrinsic volatility control covariates.
  df <- read.csv(data_path)
  
  # Ensure the household identifier is treated as a categorical factor for the random intercept
  df$LCLid <- as.factor(df$LCLid)
  
  # Fit the Mixed-Effects Logistic Regression Model
  # Fixed effects: transition_rate, shannon_entropy, intrinsic_volatility
  # Random effect: (1 | LCLid) assigns a random intercept per household
  # Note: A random slope (e.g., + transition_rate | LCLid) could be added to account for 
  # variations in how specific households respond to independent variables.
# Update your glmer formula in statistical_analysis.R
  model <- glmer(
  tail_event_failure ~ scale(transition_rate) + scale(shannon_entropy) + scale(intrinsic_volatility) + (1 | LCLid),
  data = df,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e+05))
)
  
  print(summary(model))
  return(model)
}

extract_odds_ratios <- function(model) {
  # Extract the fixed-effect coefficients (log-odds)
  fixed_effects <- fixef(model)
  
  # Exponentiate the coefficients to derive the Odds Ratios
  odds_ratios <- exp(fixed_effects)
  
  # Calculate 95% Confidence Intervals using the Wald method
  conf_intervals <- exp(confint(model, parm = "beta_", method = "Wald"))
  
  # Compile the results into a structured data frame for the final manuscript tables
  results_df <- data.frame(
    Predictor = names(fixed_effects),
    Odds_Ratio = odds_ratios,
    Lower_CI = conf_intervals[, 1],
    Upper_CI = conf_intervals[, 2]
  )
  
  return(results_df)
}

calculate_icc <- function(model) {
  # Extract the variance components from the mixed-effects model
  variance_components <- as.data.frame(VarCorr(model))
  
  # Extract the random intercept variance (between-household variance)
  tau_00 <- variance_components$vcov[1]
  
  # For binary logistic regression using a logit link, the level-1 residual variance 
  # is mathematically fixed at pi^2 / 3
  pi_squared_over_3 <- (pi^2) / 3
  
  # Calculate the Intraclass Correlation Coefficient (ICC)
  # Represents the proportion of total variance attributable to persistent household differences
  icc <- tau_00 / (tau_00 + pi_squared_over_3)
  
  return(icc)
}

# ==========================================
# Execution Flow Integration
# ==========================================
if (sys.nframe() == 0) {
  dataset_filepath <- "final_analytical_dataset.csv"
  
  # 1. Run the mixed-effects logistic regression
  mixed_model <- run_mixed_effects_model(dataset_filepath)
  
  # 2. Extract and print Odds Ratios to check if predictors are significantly > 1.0 (p < 0.05)
  odds_ratio_results <- extract_odds_ratios(mixed_model)
  print("--- Odds Ratios and 95% Confidence Intervals ---")
  print(odds_ratio_results)
  
  # 3. Calculate and print the Intraclass Correlation Coefficient
  icc_value <- calculate_icc(mixed_model)
  print(paste("Intraclass Correlation Coefficient (ICC):", round(icc_value, 4)))
}

df <- read.csv("final_analytical_dataset.csv")

print("=====================================================")
print("MODEL 1: MIXED-EFFECTS LOGISTIC REGRESSION (WITHIN-HOUSEHOLD)")
print("=====================================================")
# The scaled mixed-effects model
model_mixed <- glmer(
  tail_event_failure ~ scale(transition_rate) + scale(shannon_entropy) + scale(intrinsic_volatility) + (1 | LCLid),
  data = df,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e+05))
)

print(summary(model_mixed))
print("--- Mixed-Effects Odds Ratios ---")
print(exp(fixef(model_mixed)))


print("=====================================================")
print("MODEL 2: STANDARD LOGISTIC REGRESSION (POPULATION-LEVEL)")
print("=====================================================")
# The standard GLM evaluating macro-behavior without the random household intercept
model_glm <- glm(
  tail_event_failure ~ scale(transition_rate) + scale(shannon_entropy) + scale(intrinsic_volatility),
  data = df,
  family = binomial(link = "logit")
)

print(summary(model_glm))
print("--- Population-Level Odds Ratios and 95% CI ---")
# Suppress the waiting message for confidence intervals using suppressMessages
suppressMessages(print(exp(cbind(Odds_Ratio = coef(model_glm), confint(model_glm)))))

print("=====================================================")
print("MODEL 3: VOLATILITY ABLATION (No Intrinsic Volatility)")
print("=====================================================")
# Tests how much effect volatility was masking/inflating
model_ablated <- glm(
  tail_event_failure ~ scale(transition_rate) + scale(shannon_entropy),
  data = df,
  family = binomial(link = "logit")
)
suppressMessages(print(exp(cbind(Odds_Ratio = coef(model_ablated), confint(model_ablated)))))

print("=====================================================")
print("MODEL 4: PLACEBO SHUFFLE TEST")
print("=====================================================")
# Randomly shuffles the transition_rate to ensure the signal is real, not a statistical artifact
set.seed(42)
df$shuffled_transition <- sample(df$transition_rate)

model_placebo <- glm(
  tail_event_failure ~ scale(shuffled_transition) + scale(shannon_entropy) + scale(intrinsic_volatility),
  data = df,
  family = binomial(link = "logit")
)
print(summary(model_placebo)$coefficients)
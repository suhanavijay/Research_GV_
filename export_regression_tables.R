# export_regression_tables.R
# Extracts publication-ready inferential statistics from the GridVision models

suppressPackageStartupMessages(library(lme4))

# ==========================================
# 1. Load and Prepare Data
# ==========================================
cat("Loading data...\n")
df <- read.csv("final_analytical_dataset.csv")
df$LCLid <- as.factor(df$LCLid)

# ==========================================
# 2. Population-Level GLM
# ==========================================
cat("Fitting Population GLM...\n")
model_glm <- glm(
  tail_event_failure ~ scale(transition_rate) + scale(shannon_entropy) + scale(intrinsic_volatility),
  data = df,
  family = binomial(link = "logit")
)

# Extract GLM metrics
glm_summary <- coef(summary(model_glm))
# Default confint() for glm uses Profile Likelihood
cat("Calculating Profile Likelihood CIs for GLM...\n")
glm_ci <- suppressMessages(confint(model_glm)) 

glm_results <- data.frame(
  Model = "Population GLM",
  Predictor = rownames(glm_summary),
  Log_Odds = glm_summary[, "Estimate"],
  Std_Error = glm_summary[, "Std. Error"],
  p_value = glm_summary[, "Pr(>|z|)"],
  Odds_Ratio = exp(glm_summary[, "Estimate"]),
  CI_Lower_OR = exp(glm_ci[, 1]),
  CI_Upper_OR = exp(glm_ci[, 2]),
  CI_Method = "Profile Likelihood",
  ICC = NA
)

# ==========================================
# 3. Mixed-Effects Logistic Regression (GLMM)
# ==========================================
cat("Fitting Mixed-Effects GLMM (this may take a moment)...\n")
model_glmm <- glmer(
  tail_event_failure ~ scale(transition_rate) + scale(shannon_entropy) + scale(intrinsic_volatility) + (1 | LCLid),
  data = df,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e+05))
)

# Extract GLMM metrics
glmm_summary <- coef(summary(model_glmm))
# Use Wald method for GLMM CIs to match original codebase and ensure rapid convergence
cat("Calculating Wald CIs for GLMM...\n")
glmm_ci <- confint(model_glmm, parm = "beta_", method = "Wald") 

# Calculate the Intraclass Correlation Coefficient (ICC)
# Definition: Proportion of total variance attributable to persistent between-household differences.
# In a logistic model with a logit link, the level-1 residual variance is fixed at pi^2 / 3.
var_components <- as.data.frame(VarCorr(model_glmm))
tau_00 <- var_components$vcov[1] # Random intercept variance
pi_squared_over_3 <- (pi^2) / 3
icc_value <- tau_00 / (tau_00 + pi_squared_over_3)

glmm_results <- data.frame(
  Model = "Mixed-Effects GLMM",
  Predictor = rownames(glmm_summary),
  Log_Odds = glmm_summary[, "Estimate"],
  Std_Error = glmm_summary[, "Std. Error"],
  p_value = glmm_summary[, "Pr(>|z|)"],
  Odds_Ratio = exp(glmm_summary[, "Estimate"]),
  CI_Lower_OR = exp(glmm_ci[, 1]),
  CI_Upper_OR = exp(glmm_ci[, 2]),
  CI_Method = "Wald",
  ICC = icc_value
)

# ==========================================
# 4. Consolidate and Export
# ==========================================
final_results <- rbind(glm_results, glmm_results)

# Clean up row names for export
rownames(final_results) <- NULL

export_path <- "regression_results.csv"
write.csv(final_results, export_path, row.names = FALSE)

cat("\n=====================================================\n")
cat(sprintf("Success! Results exported to: %s\n", export_path))
cat(sprintf("Calculated ICC: %.4f\n", icc_value))
cat("=====================================================\n")
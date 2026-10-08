# export_glm_predictions.R
# Generates marginal predictions and pointwise 95% confidence bands for transition_rate
# CIs are computed on the logit (link) scale and transformed to prevent bounds < 0 or > 1.

# ==========================================
# 1. Load and Prepare Data
# ==========================================
cat("Loading data...\n")
df <- read.csv("final_analytical_dataset.csv")

# Ensure household ID is a factor (matching previous behavior, though unused in GLM)
df$LCLid <- as.factor(df$LCLid)

# ==========================================
# 2. Fit Population GLM
# ==========================================
cat("Fitting Population GLM...\n")
model_glm <- glm(
  tail_event_failure ~ scale(transition_rate) + scale(shannon_entropy) + scale(intrinsic_volatility),
  data = df,
  family = binomial(link = "logit")
)

# ==========================================
# 3. Build Prediction Grid
# ==========================================
cat("Building prediction grid...\n")

# Extract exact scaling parameters to ensure identical transformations
tr_center <- mean(df$transition_rate, na.rm = TRUE)
tr_scale <- sd(df$transition_rate, na.rm = TRUE)
mean_shannon <- mean(df$shannon_entropy, na.rm = TRUE)
mean_volatility <- mean(df$intrinsic_volatility, na.rm = TRUE)

# Build a dense grid over the observed empirical range of transition_rate
tr_range <- seq(
  from = min(df$transition_rate, na.rm = TRUE),
  to = max(df$transition_rate, na.rm = TRUE),
  length.out = 300
)

# Hold other covariates strictly at their empirical means
prediction_grid <- data.frame(
  transition_rate = tr_range,
  shannon_entropy = mean_shannon,
  intrinsic_volatility = mean_volatility
)

# Explicitly calculate the standardized transition rate for the plot's X-axis
prediction_grid$standardized_transition_rate <- (prediction_grid$transition_rate - tr_center) / tr_scale

# ==========================================
# 4. Generate Predictions on the Link Scale
# ==========================================
cat("Calculating link-scale predictions and standard errors...\n")
# R's predict() will safely reapply the exact scale() attributes saved inside the model formula
pred_link <- predict(model_glm, newdata = prediction_grid, type = "link", se.fit = TRUE)

# Calculate 95% CIs on the linear link scale (eta)
z_value <- qnorm(0.975) # Approx 1.96
eta_hat <- pred_link$fit
eta_lower <- eta_hat - (z_value * pred_link$se.fit)
eta_upper <- eta_hat + (z_value * pred_link$se.fit)

# ==========================================
# 5. Transform to Probability Scale
# ==========================================
cat("Applying inverse-logit transformation...\n")
# plogis() performs the logistic transform: exp(eta) / (1 + exp(eta))
prediction_grid$predicted_probability <- plogis(eta_hat)
prediction_grid$lower_CI <- plogis(eta_lower)
prediction_grid$upper_CI <- plogis(eta_upper)

# Isolate the requested export columns
final_output <- prediction_grid[, c(
  "transition_rate",
  "standardized_transition_rate",
  "predicted_probability",
  "lower_CI",
  "upper_CI",
  "shannon_entropy",
  "intrinsic_volatility"
)]

# Rename covariates for clarity in the output CSV
names(final_output)[names(final_output) == "shannon_entropy"] <- "conditioned_shannon_entropy"
names(final_output)[names(final_output) == "intrinsic_volatility"] <- "conditioned_intrinsic_volatility"

# ==========================================
# 6. Export Results
# ==========================================
export_path <- "glm_predictions.csv"
write.csv(final_output, export_path, row.names = FALSE)

cat("\n=====================================================\n")
cat(sprintf("Success! GLM predictions exported to: %s\n", export_path))
cat(sprintf("Grid covers transition_rate range: [%.4f, %.4f]\n", min(tr_range), max(tr_range)))
cat("Conditioning covariates held at their empirical means.\n")
cat("=====================================================\n")
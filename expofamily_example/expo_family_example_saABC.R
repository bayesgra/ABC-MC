# ===============================
# Semi-Automatic ABC for Model Selection
# ===============================

library(abctools)
library(reticulate)

set.seed(1234)

# ================================================================
# Define input/output folders
# ================================================================
data_dir <- "data"
results_dir <- "results"
saABC_dir <- file.path(results_dir, "saABC")

if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)
if (!dir.exists(saABC_dir)) dir.create(saABC_dir, recursive = TRUE)

# ================================================================
# Load observed datasets from .npz file
# ================================================================
np <- import("numpy")
observed <- np$load(file.path(data_dir, "observed_datasets.npz"), allow_pickle = TRUE)
observed_datasets <- observed$f[["observed_datasets"]]  # Adjust key name if needed
dim(observed_datasets)  # Should print (100, 100)

# ================================================================
# Simulated parameter and summary statistics
# ================================================================
id <- seq(from = 5, to = 100, by = 5)

theta_exp <- matrix(rexp(333333, rate = 1), ncol = 1)
theta_lognormal <- matrix(rnorm(333333, 0, 1), ncol = 1)
theta_gamma <- matrix(rexp(333334, rate = 1), ncol = 1)
theta <- matrix(NA, ncol = 2, nrow = 10^6)
theta[, 1] <- rbind(theta_exp, theta_lognormal, theta_gamma)
theta[, 2] <- c(rep(0, 333333), rep(1, 333333), rep(2, 333334))
colnames(theta) <- c("P1", "M")

# Simulated summary statistics
sim_ss <- matrix(NA, nrow = 10^6, ncol = length(id))
colnames(sim_ss) <- paste0("S", 1:length(id))

for (i in 1:333333) {
  sim_data <- rexp(100, rate = theta[i, 1])
  sim_ss_iter <- sort(sim_data)
  sim_ss[i, ] <- sim_ss_iter[id]
}
for (i in 333334:666666) {
  x <- rnorm(100, theta[i, 1], 1)
  sim_data <- exp(x)
  sim_ss_iter <- sort(sim_data)
  sim_ss[i, ] <- sim_ss_iter[id]
}
for (i in 666667:10^6) {
  sim_data <- rgamma(100, rate = theta[i, 1], shape = 2)
  sim_ss_iter <- sort(sim_data)
  sim_ss[i, ] <- sim_ss_iter[id]
}

# ================================================================
# Semi-Automatic ABC analysis
# ================================================================
tab_models_exp <- matrix(NA, ncol = 3, nrow = 100)
param_models_exp <- matrix(NA, ncol = 3, nrow = 100)

for (j in 1:100) {
  obs_data <- observed_datasets[j, ]
  obs_ss <- sort(obs_data)
  obs_ss <- obs_ss[id]
  obs_ss <- matrix(obs_ss, nrow = 1)
  colnames(obs_ss) <- paste0("S", 1:length(obs_ss))
  
  tmp <- selectsumm(
    obs_ss, theta, sim_ss,
    ssmethod = AS.select,
    tol = 0.001,
    method = "rejection",
    allow.none = FALSE,
    inturn = TRUE,
    hcorr = TRUE,
    final.dens = TRUE
  )
  
  tab_models_exp[j, 1] <- sum(tmp$post.sample[1001:2000] == 0)
  tab_models_exp[j, 2] <- sum(tmp$post.sample[1001:2000] == 1)
  tab_models_exp[j, 3] <- sum(tmp$post.sample[1001:2000] == 2)
  param_models_exp[j, 1] <- mean(tmp$post.sample[1:1000][tmp$post.sample[1001:2000] == 0])
  param_models_exp[j, 2] <- mean(tmp$post.sample[1:1000][tmp$post.sample[1001:2000] == 1])
  param_models_exp[j, 3] <- mean(tmp$post.sample[1:1000][tmp$post.sample[1001:2000] == 2])
  
  print(paste("Dataset", j, "completed"))
}

# ================================================================
# Save results to results/ folder
# ================================================================
save(tab_models_exp, param_models_exp,
     file = file.path(saABC_dir, "expo_family_SA.RData"))

prob_expo <- matrix(tab_models_exp[, 1] / 1000, ncol = 1)
colnames(prob_expo) <- "SA"
write.csv(prob_expo, file.path(saABC_dir, "expo_family_exp_SA_probabilities.csv"), row.names = FALSE)

params_expo <- matrix(param_models_exp[, 1], ncol = 1)
colnames(params_expo) <- "SA"
write.csv(params_expo, file.path(saABC_dir, "expo_family_exp_SA_params.csv"), row.names = FALSE)


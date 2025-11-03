rm(list = ls())

library(abctools)
library(RcppCNPy)
set.seed(1234)

# ================================================================
# Define folders
# ================================================================
data_dir <- "data"
results_dir <- "results"
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

# ================================================================
# Load observed datasets from Python
# ================================================================
observed_data <- npyLoad(file.path(data_dir, "observed_datasets.npy"))
dim(observed_data)  # should be 100 x 100

# ================================================================
# Simulate from models
# ================================================================
mu_sim <- c(rep(0, 500000), rnorm(500000, 0, 10))
theta <- matrix(NA, ncol = 2, nrow = 10^6)
theta[, 1] <- mu_sim
theta[, 2] <- c(rep(0, 500000), rep(1, 500000))
colnames(theta) <- c("mu", "M")

id <- seq(from = 5, to = 100, by = 5)

# Simulated summary statistics
sim_ss <- matrix(NA, nrow = 10^6, ncol = length(id) + 1)
colnames(sim_ss) <- paste0("S", 1:(length(id) + 1))

for (i in 1:10^6) {
  sim_data <- rnorm(100, mean = theta[i, 1], sd = 1)
  sim_ss_iter <- sort(sim_data)
  sim_ss[i, ] <- c(sim_ss_iter[id], mean(sim_data))
}

# ================================================================
# ABC analysis for observed datasets
# ================================================================
tab_models_M0 <- matrix(NA, ncol = 2, nrow = 100)
param_models_M0 <- matrix(NA, ncol = 2, nrow = 100)

for (j in 1:100) {
  obs_data <- observed_data[j, ]
  
  obs_ss <- sort(obs_data)
  obs_ss <- c(obs_ss[id], mean(obs_data))
  obs_ss <- matrix(obs_ss, nrow = 1)
  colnames(obs_ss) <- paste0("S", 1:length(obs_ss))
  
  tmp <- selectsumm(
    obs_ss, theta, sim_ss,
    ssmethod = AS.select,    # semi-automatic ABC
    tol = 0.001,
    method = "rejection",
    allow.none = FALSE,
    inturn = TRUE,
    hcorr = TRUE,
    final.dens = TRUE
  )
  
  tab_models_M0[j, 1] <- sum(tmp$post.sample[1001:2000] == 0)
  tab_models_M0[j, 2] <- sum(tmp$post.sample[1001:2000] == 1)
  param_models_M0[j, 1] <- 0
  param_models_M0[j, 2] <- mean(tmp$post.sample[1:1000][tmp$post.sample[1:1000] != 0])
  
  print(paste("Completed dataset", j))
}

# ================================================================
# Save results
# ================================================================
save(tab_models_M0, param_models_M0,
     file = file.path(results_dir, "normal_saABC.RData"))

# Compute posterior probabilities for Model 0
prob_model0 <- tab_models_M0[, 1] / rowSums(tab_models_M0)
prob_model0_df <- data.frame(ObsID = 1:100, Prob_Model0 = prob_model0)

# Extract mean parameter estimates (Model 1)
param_model1_df <- data.frame(ObsID = 1:100, Mean_Mu_Model1 = param_models_M0[, 2])

# Save both as CSVs
write.csv(prob_model0_df, file.path(results_dir, "normal_model0_SA_probabilities.csv"), row.names = FALSE)
write.csv(param_model1_df, file.path(results_dir, "normal_model0_SA_params.csv"), row.names = FALSE)


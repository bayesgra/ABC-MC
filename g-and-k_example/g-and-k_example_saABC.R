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
observed_npz <- np$load("data/observed_datasets.npz")
X_observed <- observed_npz$f[["observed_datasets"]]  # shape: (num_obs, sample_size)
num_obs <- dim(X_observed)[1]
sample_size <- dim(X_observed)[2]

# ================================================================
# Simulated parameter and summary statistics
# ================================================================
simulate_gk <- function(n, A = 0, B = 1, g = 0, k = 2, c = 0.8) {
  z <- rnorm(n)  # standard normal quantiles
  Q <- A + B * (1 + c * ((1 - exp(-g * z)) / (1 + exp(-g * z)))) * (1 + z^2)^k * z
  return(Q)
}

g_sim <- c(rep(0, 500000),runif(500000,0,4))
k_sim <- runif(10^6,-0.5,5)
theta <- matrix(NA,ncol=3,nrow=10^6)
theta[,1] <- g_sim
theta[,2] <- k_sim
theta[,3] <- c(rep(0,500000),rep(1,500000))
colnames(theta) <- c("g","k","M")

id <- seq(from=5,to=100,by=5)

# Matrix of simulated summary statistics.
sim_ss <- matrix(NA,nrow=10^6,ncol=length(id))
colnames(sim_ss) <- paste0("S",1:length(id))
sim_ss[1:5,1:5]  

for(i in 1:10^6)
{
  sim_data <- simulate_gk(100, g=theta[i,1], k=theta[i,2]) ##g-and-k
  sim_ss_iter <- sort(sim_data)
  sim_ss[i,] <- sim_ss_iter[id] 
}

# ================================================================
# Semi-Automatic ABC analysis
# ================================================================
tab_models_M0 <- matrix(NA, ncol=2, nrow=num_obs)
param_models_M0 <- matrix(NA, ncol=4, nrow=num_obs)

for(j in 1:num_obs) {
  obs_data <- X_observed[j, ]
  
  # Observed summary statistics
  obs_ss <- sort(obs_data)
  obs_ss <- obs_ss[id]
  obs_ss <- matrix(obs_ss, nrow=1)
  colnames(obs_ss) <- paste0("S", 1:length(obs_ss))
  
  tmp <- selectsumm(obs_ss, theta, sim_ss, ssmethod = AS.select, tol = .001,
                    method = "rejection", allow.none = FALSE, inturn = TRUE, 
                    hcorr = TRUE, final.dens = TRUE)
  
  # Model probabilities
  tab_models_M0[j,1] <- sum(tmp$post.sample[2001:3000] == 0)
  tab_models_M0[j,2] <- sum(tmp$post.sample[2001:3000] == 1)
  
  # Parameter estimates
  param_models_M0[j,1] <- mean(tmp$post.sample[1:1000][tmp$post.sample[2001:3000]==0])
  param_models_M0[j,2] <- mean(tmp$post.sample[1:1000][tmp$post.sample[2001:3000]==1])
  param_models_M0[j,3] <- mean(tmp$post.sample[1001:2000][tmp$post.sample[2001:3000]==0])
  param_models_M0[j,4] <- mean(tmp$post.sample[1001:2000][tmp$post.sample[2001:3000]==1])
  
  print(j)
}

# ========================================
# Save Results and Export Results to CSV
# ========================================

save(tab_models_M0, param_models_M0,
     file = file.path(saABC_dir, "gk_example_g0_SA.RData"))

prob_M0 <- matrix(tab_models_M0[,1]/1000, ncol=1)
colnames(prob_M0) <- "SA"
write.csv(prob_M0, file.path(saABC_dir, "gk_example_g0_SA_probabilities.csv"), row.names = FALSE)

params_M0 <- param_models_M0
colnames(params_M0) <- rep("SA", 4)
write.csv(params_M0, file.path(saABC_dir, "gk_example_g0_SA_params.csv"), row.names = FALSE)


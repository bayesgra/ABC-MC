rm(list = ls())

library(abctools)
library(reticulate)
set.seed(1234)

# -----------------------------
# Folders
# -----------------------------
data_dir <- "data"
results_dir <- "results"
saABC_dir <- file.path(results_dir, "saABC")

if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)
if (!dir.exists(saABC_dir)) dir.create(saABC_dir, recursive = TRUE)

# -----------------------------
# Load observed datasets (.npz)
# -----------------------------
np <- import("numpy")
obs_npz <- np$load(file.path(data_dir, "observed_datasets.npz"), allow_pickle = TRUE)
obs_list <- obs_npz$f[["observed_datasets"]]   # shape: (n_obs, num_days, num_toads)
n_obs <- dim(obs_list)[1]
dim(observed_datasets)

# -----------------------------
# Load simulation summaries/params (if you use them)
#   Adjust paths/columns to your setup
# -----------------------------
sim_ss <- read.csv("data/toad_simulated_stats.csv", header = TRUE)
theta  <- read.csv("data/toad_simulated_param.csv", header = TRUE)
# If distance model lacks d0 in some rows, fill with 0 to keep a consistent matrix
if (!"d0" %in% names(theta)) theta$d0 <- NA_real_
theta$d0 <- ifelse(is.na(theta$d0), 0, theta$d0)

# drop any rows with NA across sim_ss+theta
stopifnot(nrow(sim_ss) == nrow(theta))
tot <- cbind(sim_ss, theta)
tot <- stats::na.omit(tot)
sim_ss <- tot[, seq_len(ncol(sim_ss)), drop = FALSE]
theta  <- tot[, (ncol(sim_ss) + 1):ncol(tot), drop = FALSE]

# -----------------------------
# Make summaries for observed data to match sim_ss columns
# (Example: per-toad mean and variance, then select 'id' subset)
# -----------------------------
obs_summary <- t(apply(obs_list, 1, summarize_obs))
colnames(obs_summary) <- colnames(sim_ss)

# -----------------------------
# SA-ABC per observed dataset
# -----------------------------
# Expect theta to have columns like: alpha, gamma, p0, d0, M (model id)
# Where M in {0=RANDOM, 1=NEAREST, 2=DISTANCE}
stopifnot("M" %in% names(theta))
model_names <- c("Random", "Nearest", "Distance")

prob_mat   <- matrix(NA_real_, nrow = n_obs, ncol = 3)
colnames(prob_mat) <- model_names

param_mat  <- matrix(NA_real_, nrow = n_obs, ncol = 4)  # Alpha, Gamma, P0, D0
colnames(param_mat) <- c("Alpha", "Gamma", "P0", "D0")

for (j in seq_len(n_obs)) {
  obs_ss <- matrix(obs_summary[j, ], nrow = 1)
  colnames(obs_ss) <- colnames(sim_ss)
  
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
  
  # tmp$post.sample should include sampled parameters and model labels.
  # Adjust these column names to match what abctools returns in your run:
  ps <- tmp$post.sample
  # Common names you may have:
  #   alpha, gamma, p0, d0 (some rows may have d0=0 for non-distance models)
  #   M (model id: 0,1,2)
  stopifnot("M" %in% colnames(ps))
  
  # Posterior model probabilities
  for (m in 0:2) {
    prob_mat[j, m + 1] <- mean(ps$M == m)
  }
  
  # Pick a model to report parameters (e.g., highest posterior probability)
  chosen_m <- which.max(prob_mat[j, ])
  
  # Compute posterior mean parameters **for that chosen model**
  # RANDOM/NEAREST -> 3 params (alpha,gamma,p0); DISTANCE -> 4 params (+ d0)
  if (chosen_m %in% c(1, 2)) {
    # models 0 or 1
    rows_m <- ps$M == (chosen_m - 1)
    param_mat[j, 1:3] <- colMeans(ps[rows_m, c("alpha", "gamma", "p0"), drop = FALSE])
    param_mat[j, 4]   <- NA_real_
  } else {
    # model 2 = DISTANCE
    rows_m <- ps$M == 2
    param_mat[j, ] <- colMeans(ps[rows_m, c("alpha", "gamma", "p0", "d0"), drop = FALSE])
  }
  
  if (j %% 10 == 0 || j == n_obs) cat("Processed", j, "of", n_obs, "\n")
}

# -----------------------------
# Save output
# -----------------------------
write.csv(prob_mat,  file.path(results_dir, "toad_example_random_SA_probabilities.csv"), row.names = FALSE)
write.csv(param_mat, file.path(results_dir, "toad_example_random_SA_params.csv"),        row.names = FALSE)
save.image(file.path(results_dir, "toad_SA_ABC.RData"))

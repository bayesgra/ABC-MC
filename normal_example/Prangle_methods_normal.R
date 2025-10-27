rm(list=ls())

library(abctools)
set.seed(1234)

# Matrix of simulated model parameter values.
mu_sim <- c(rep(0, 500000),rnorm(500000,0,10))
theta <- matrix(NA,ncol=2,nrow=10^6)
theta[,1] <- mu_sim
theta[,2] <- c(rep(0,500000),rep(1,500000))
colnames(theta) <- c("mu","M")

id <- seq(from=5,to=100,by=5)

# Matrix of simulated summary statistics.
sim_ss <- matrix(NA,nrow=10^6,ncol=length(id)+1)
colnames(sim_ss) <- paste0("S",1:(length(id)+1))
sim_ss[1:5,1:5]  

for(i in 1:10^6)
{
  sim_data <- rnorm(100, mean=theta[i,1], sd=1) 
  sim_ss_iter <- sort(sim_data)
  sim_ss[i,] <- c(sim_ss_iter[id],mean(sim_data)) 
}

############## Data from M0

set.seed(1234)

tab_models_M0 <- matrix(NA,ncol=2,nrow=100)
param_models_M0 <- matrix(NA,ncol=2,nrow=100)

for(j in 1:100)
{
  # Simulate from M0
  obs_data <- rnorm(100,0,1)
  
  # Observed summary statistics.
  obs_ss <- sort(obs_data)
  obs_ss <- c(obs_ss[id],mean(obs_data)) 
  obs_ss <- matrix(obs_ss,nrow=1)
  colnames(obs_ss) <- paste0("S",1:length(obs_ss))
  
  tmp<-selectsumm(obs_ss, theta, sim_ss, ssmethod =AS.select, tol =.001,
                  method = "rejection", allow.none = FALSE, inturn = TRUE, 
                  hcorr = TRUE,final.dens=TRUE)
  tab_models_M0[j,1] <- sum(tmp$post.sample[1001:2000]==0)
  tab_models_M0[j,2] <- sum(tmp$post.sample[1001:2000]==1)
  param_models_M0[j,1] <- 0
  param_models_M0[j,2] <- mean(tmp$post.sample[1:1000][tmp$post.sample[1:1000]!=0])
  
  print(j)
}

save.image("normal.RData")

#load("quantile.RData")


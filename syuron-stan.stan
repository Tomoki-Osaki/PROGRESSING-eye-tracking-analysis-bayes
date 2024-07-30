data {
  int T;
  vector[T] ex; // explanatory variable (trustness)
  vector[T] y; // observed values (AOI ratio)
}

parameters {
  vector[T] mu; // level 
  vector[T] beta; // slope
  vector[T] gamma; // seasonal 
  real<lower=0> s_level; // SD of level
  real<lower=0> s_beta; // SD of slope
  real<lower=0> s_season; // SD of seasonal effect
  real<lower=0> s_obs; // SD of observational error
}

transformed parameters {
  vector[T] alpha;
  // alpha = level + time-varying-coef * ex + seasonal effect + error
  alpha = mu + beta * ex + gamma;
}

model {
  for (t in 2:T) {
    mu[t] ~ normal(mu[t-1], s_level);
    beta[t] ~ normal(beta[t-1], s_beta);
  }
  
  for (t in 30:T) {
    gamma[t] ~ normal(-sum(gamma[(t-30):(t-1)]), s_season);
  }
  
  y ~ normal(alpha, s_obs);
}

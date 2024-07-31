// replaced the seasonality in syuron-stan.stan with the external factor
data {
  int T;
  // trustness may be replaced with: vector[T] FAs;
  vector[T] ex; // explanatory variable (trustness)
  vector[T] alm; // flags of alarms (0: No alarm, 1: Alarm)
  vector[T] y; // observed values (AOI ratio)
}

parameters {
  vector[T] mu; // level 
  vector[T] beta; // slope
  real k; // coefficient of an external factor (alarm)
  
  real<lower=0> s_level; // SD of level
  real<lower=0> s_beta; // SD of slope (time varying coefficient)
  real<lower=0> s_obs; // SD of observational error
}

transformed parameters {
  vector[T] alpha;
  alpha = mu + beta * ex + k * alm;
}

model {
  for (t in 2:T) {
    mu[t] ~ normal(mu[t-1], s_level);
    beta[t] ~ normal(beta[t-1], s_beta);
  }
  
  y ~ normal(alpha, s_obs);
}

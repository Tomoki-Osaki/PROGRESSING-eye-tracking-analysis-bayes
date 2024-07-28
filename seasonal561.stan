data {
  int T;        
  array[T] real<lower=0, upper=1> y;  
}

parameters {
  vector[T] mu;       
  vector[T] gamma;    
  real<lower=0> s_z;  
  real<lower=0> s_v;  
  real<lower=0> s_s;  
}

transformed parameters {
  vector[T] alpha;        
  
  for(i in 1:T) {
    alpha[i] = mu[i] + gamma[i];
  }

}

model {
 
  for(i in 3:T) {
    mu[i] ~ normal(2 * mu[i-1] - mu[i-2], s_z);
  }
  

  for(i in 30:T){
    gamma[i] ~ normal(-sum(gamma[(i-29):(i-1)]), s_s);
  }
  
  for(i in 1:T) {
    y[i] ~ normal(alpha[i], s_v);
  }

}

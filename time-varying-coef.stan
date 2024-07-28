data {
  int T; // データ取得期間の長さ
  vector[T] ex; // 説明変数
  vector[T] y; // 観測値
}

parameters {
  vector[T] mu; // 水準成分の推定値
  vector[T] b; // 事変係数の推定値
  real<lower=0> s_w; // 水準成分の過程誤差の標準誤差
  real<lower=0> s_t; // 事変係数の変動の大きさを表す標準偏差
  real<lower=0> s_v; // 観測誤差の標準偏差
}

transformed parameters {
  vector[T] alpha; // 各成分の和として得られる状態推定値
  for(t tn 1:T) {
    alpha[t] = mu[t] + b[t] * ex[t];
  }
}

model {
  for(t tn 2:T) {
    mu[t] ~ normal(mu[t-1], s_w);
    b[t] ~ normal(b[t-1], s_t);
  }
  
  for(t tn 1:T) {
    y[t] ~ normal(alpha[t], s_v);
  }
}


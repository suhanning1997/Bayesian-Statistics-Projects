functions{
  real myweibull_lpdf(real x, real eta, real alpha, real beta_0) {
    return log(alpha) +  (alpha-1)*log(x) + (beta_0 + eta)  -
     (x^alpha) * exp(beta_0 + eta);
  }
  
  real myweibull_lccdf(real x, real eta, real alpha, real beta_0){
    return -(x^alpha) * exp(eta + beta_0);
  }
}
  
  data {
    int<lower=1> N_censored; // Number of censored individuals
    int<lower=1> N_uncensored; // Number of uncensored individuals
    int<lower=1> N_typical; // Number of typical individuals
    int<lower=0> N_predictors;
    vector<lower=0>[N_censored] Observed_times_censored;
    vector<lower=0>[N_uncensored] Observed_times_uncensored;
    matrix[N_censored, N_predictors] Z_censored;
    matrix[N_uncensored, N_predictors] Z_uncensored;
    matrix[N_typical, N_predictors] Z_typical;
    real scale_alpha; //pior sd on alpha
    real scale_beta; // prior sds on beta
}

  parameters {
    real<lower=0> alpha; // =1/sigma sigma is the scale paramter of log weibull
    vector[N_predictors] beta;  // regression coefficients beta vector
    real beta_0; //intercept 
  }
      
  transformed parameters {
    vector[N_predictors] exp_beta;
    vector[N_predictors] gamma;
    real mu_tilda; 
    real sigma;

    exp_beta = exp(beta);
    gamma = -beta/alpha;
    mu_tilda = -beta_0 / alpha;
    sigma = 1/alpha;
    }
      
  model {
    real eta;
    row_vector[N_predictors] Z;
    real x;
    
    // priors
    alpha ~ normal(0, scale_alpha); // half normal
    beta_0 ~ cauchy(0, scale_beta);
    beta ~ normal(0, scale_beta);
    
    for (i in 1:N_censored){
      x = Observed_times_censored[i];
      Z = Z_censored[i];
      eta = Z * beta;
      target += myweibull_lccdf(x|eta, alpha, beta_0);
    }
    
    for (i in 1:N_uncensored){
      x = Observed_times_uncensored[i];
      Z = Z_uncensored[i];
      eta = Z * beta;
      target += myweibull_lpdf(x|eta, alpha, beta_0);
    }
  }
  generated quantities{
  vector[N_typical] times_typical_sampled;
    
  for(i in 1:N_typical) {
      times_typical_sampled[i] = exp(mu_tilda + Z_typical[i]*gamma - 
      sigma *gumbel_rng(0,1));
  } }

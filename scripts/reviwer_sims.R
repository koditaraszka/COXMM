library(genio)

get_censor_lambda = function(T, target = 0.20){
  f = function(log_rate){
    rate = exp(log_rate)
    mean(1 - exp(-rate * T)) - target
  }
  rate = exp(uniroot(f, interval = c(-20, 20))$root)
  return(rate)
}

# environmental exposure simulations 
EA_sims = function(L, prevalence, cumulate_E, N){
  Y = rep(0, N)
  age = rep(NA, N)
  cases = integer(0)
  
  for(age_i in 1:5){
    risk = setdiff(seq_len(N), cases)
    threshold = quantile(L[risk], 1 - prevalence / (1 - prevalence * (age_i - 1)))
    incident = which(L > threshold & seq_len(N) %in% risk)
    
    Y[incident] = 1
    age[incident] = age_i
    cases = c(cases, incident)
    
    L = L + sqrt(cumulate_E) * rnorm(N)
  }
  
  print(var(L))
  Y = as.numeric(!is.na(age))
  age[is.na(age)] = 5
  list(Y = Y, age = age)
}

# weibull aft simulations
weibull_aft = function(N, mu, shape, l_g){
  # Use Gumbel (minimum) distribution for standard Weibull
  e = log(rexp(N)) 
  
  # e has mean digamma(1), so we subtract it to center at 0
  logT = mu + l_g + (e - digamma(1)) / shape
  T = exp(logT)
  return(T)
}

weibull_aft_sims = function(shape, l_g, censorRate, mu = 0){
  onset = weibull_aft(N, mu, shape, l_g)
  censor = rexp(N, rate = get_censor_lambda(onset, censorRate))
  
  # indicator for who was censored
  y = as.numeric(onset <= censor)
  onset = pmin(onset, censor)
  
  return(list("Y" = y, "age" = round(onset, 4)))
}

gumbel = function(N, scale, l_g){
  # Use Gumbel (minimum) distribution for Exponential (Weibull shape=1)
  e = log(rexp(N))
  # e has mean digamma(1), so we subtract it to center at 0
  logT = scale + l_g + e - digamma(1)
  return(exp(logT))
}

gumbel_sims = function(scale, l_g, censorRate){
  onset = gumbel(N, scale, l_g)
  censor = rexp(N, rate=get_censor_lambda(onset, censorRate))
  
  # indicator for who was censored
  y = as.numeric(onset <= censor)
  onset = pmin(onset, censor)
  
  return(list("Y"=y, "age"=round(onset, 4)))
}


weibull = function(N, shape, scale){
  #set hazard based on liability
  base = exp(scale)
  # simulate baseline hazard
  u = log(1-runif(N))
  # return weibull distribution
  return((-1*u/base)^(1/as.numeric(shape)))
}

# this will simulate data under the weibull distribution with uniform distribution
# censoring is based on quantile i.e. last 10% censored
weibull_sims = function(shape, scale, cenRate){
  onset = weibull(N, shape, scale)
  censor = quantile(onset, probs = cenRate)
  
  # indicator for who was censored
  y = as.numeric(onset <= censor)
  onset = pmin(onset, censor)
  
  return(list("Y"=y, "age"=round(onset, 4)))
}

write_output = function(data, shape, ftype, run, bayesW=FALSE){
  tte_file = paste0(output, "tte", shape, "_", ftype, "_outcome", run, ".txt")
  tte = data.frame('sample_id' = paste0('ID',1:N), 'start' = rep(0, N), 'stop' = data$age, 'event' = data$Y)
  write.table(tte, tte_file, row.names = F, col.names = T, sep = '\t', quote = F)

  if(bayesW){  
    hydra_file = paste0(output, "tte", shape, "_", ftype, "_outcome", run, ".phen")
    hydra_tte = data.frame('FID' = paste0('ID', 1:N), 'IID' = paste0('ID', 1:N), 'time' = data$age)
    write.table(hydra_tte, hydra_file, row.names = F, col.names = F, sep = '\t', quote = F)
  
    hydra_file = paste0(output, "tte", shape, "_", ftype, "_outcome", run, ".fail")
    hydra_tte = data.frame('status' = data$Y)
    write.table(hydra_tte, hydra_file, row.names = F, col.names = F, sep = '\t', quote = F)
  }
}

write_output_ascertained = function(data, shape, ftype, run, keep, bayesW=FALSE){
  tte_file = paste0(output, "tte", shape, "_", ftype, "_outcome", run, ".txt")
  tte = data.frame('sample_id' = paste0("ID", keep), 'start' = rep(0, length(data$Y)), 
               'stop' = data$age, 'event' = data$Y)
  write.table(tte, tte_file, row.names = F, col.names = T, sep = '\t', quote = F)
  
  if(bayesW){
    hydra_file = paste0(output, "tte", shape, "_", ftype, "_outcome", run, ".phen")
    hydra_tte = data.frame('FID' = paste0("ID", keep), 'IID' = paste0("ID", keep), 'time' = data$age)
    write.table(hydra_tte, hydra_file, row.names = F, col.names = F, sep = '\t', quote = F)
  
    hydra_file = paste0(output, "tte", shape, "_", ftype, "_outcome", run, ".fail")
    hydra_tte = data.frame('status' = data$Y)
    write.table(hydra_tte, hydra_file, row.names = F, col.names = F, sep = '\t', quote = F)
 }
}

# I use her and h2 but this is most definitely sigma2g!!
her=30
h2=her/100
MAF=0.05 # min minor allele frequency
#args = commandArgs(trailingOnly=TRUE)

for(N in c(5000, 10000, 15000, 20000)){
  #M is 25% of the sample size
  M=ceiling(N/4)
  output = paste0("sims_N", N, "_")
  
  aFreq = runif(M, min=MAF, max=(1-MAF))
  G = matrix(NA,nrow=N,ncol=M)
  for(m in 1:M){
  # 0, 1, 2 based on allele freq
    G[,m] = rbinom(N, 2, prob = aFreq[m])
  }
  X = scale(G)
  GRM = (X %*% t(X))/M
  # Genotypes written for input to BayesW/Hydra
  fam_df <- data.frame('fam' = paste0('ID', 1:N), 'id' = paste0('ID', 1:N),
	'pat' = rep(0,N), 'mat' = rep(0,N), 'sex' = rep(0,N), 'pheno' = -9)
  write_plink(file = paste0(output, "genotypes"), X = t(G), fam = fam_df)
  # GRM written for input to ldak and COXMM
  # writing at the beginning to keep things clean
  write_grm(paste0(output,  "ldak"), as.matrix(GRM))
  write.table(GRM, paste0(output, "grm.txt"), row.names = F, col.names = F, sep = '\t', quote = F)
  write.table(data.frame('sample_id' = paste0('ID', 1:N)), paste0(output, "names.txt"),
        row.names = F, col.names = T, sep = '\t', quote = F)
  write.table(data.frame('FID' = paste0('ID', 1:N), 'IID' = paste0('ID', 1:N)), paste0(output, "ldak.grm.id"),
        row.names = F, col.names = F, sep = '\t', quote = F)

  # this simulated weibull aft with censoring based on exponential
  for(run in 1:25){
    beta = rnorm(M)
    l_g = sqrt(h2)*scale(X %*% beta)
    cenRate=0.2
    K=cenRate*100
    shape=1
    z=weibull_sims(shape, l_g, cenRate)
    if(N == 10000){
      write_output(z, shape, paste0('weibull_K',K), run, TRUE)
    } else {
      write_output(z, shape, paste0('weibull_K',K), run, FALSE)
    }
    if(N == 10000){
      # % removed
      for(cenRate in c(0.5, 0.8)){
      	#changing so K is the cenRate
	K=(1-cenRate)*100
      	for(shape in c(0.5, 2)){
    	  z=weibull_aft_sims(shape, -1*l_g, cenRate)
    	  write_output(z, shape, paste0('weibull_aft0_K',K), run, TRUE)
    	  z=weibull_aft_sims(shape, -1*l_g, cenRate, 4)
    	  write_output(z, shape, paste0('weibull_aft4_K', K), run, TRUE)
      	}
      	z=gumbel_sims(1, -1*l_g, cenRate)
      	write_output(z, '', paste0('gumbel_K',K), run, TRUE)
      } 
      # set to 20% observed events, this is a flip from other scenarios
      # legacy weirdness that I own :)
      cenRate=0.2
      K=cenRate*100
      # EA SIMULATION (environment accumulation)
      E_variance = 0.5
      E = sqrt(E_variance) * rnorm(N)
      noise_variance <- 1 - h2 - E_variance
      noise <- sqrt(noise_variance) * rnorm(N)
      z = EA_sims((l_g+E+noise), 0.02, 0.05, N)
      write_output(z, '', paste0("EA_K", K), run, TRUE)

      shape=3
      z=weibull_sims(shape, l_g, cenRate)
      write_output(z, shape, paste0('weibull_K',K), run, TRUE)

      # this cenRate is the observed rate
      shape=1
      # Random 50% sample (cutting full cohort in half at random)
      half = N/2
      z=weibull_sims(shape, l_g, cenRate)
      keep = sample(seq_len(N), size = half)
      z1 = list(Y = z$Y[keep], age = z$age[keep])
      write_output_ascertained(z1, shape, paste0("random_sample_K", K), run, keep, FALSE)
    
      cases = which(z$Y == 1)
      censored = which(z$Y == 0)
    
      # Only keep as many controls as there are cases
      keep = c(sample(cases, length(cases)), sample(censored, length(cases)))
      z2 = list(Y = z$Y[keep], age = z$age[keep])
      write_output_ascertained(z2, shape, paste0("ascertained_sample_K", K), run, keep, FALSE)

      # set 75% of indices to non-causal SNPs
      # 20% observed events
      beta[sample(1:M, ceiling(0.75*M))] = 0
      l_g = sqrt(h2)*scale(X %*% beta)
      z=weibull_sims(shape, l_g, cenRate)
      write_output(z, shape, paste0('causal25_K', K), run, FALSE)

      #NULL simulations
      l_g = rep(0,N)
      z=weibull_sims(shape, l_g, cenRate)
      write_output(z, shape, paste0('null_K', K), run, TRUE)
    }
  }
}

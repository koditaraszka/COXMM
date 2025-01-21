library(genio)

args = commandArgs(trailingOnly=TRUE)

N=10000
M=2500
her=as.numeric(args[1])
h2=her/100
minOnset=30
maxOnset=70
dir=args[2]
# sub-directory based on the heritability
# i.e. 20 == 0.2 heritability
output = paste0(dir, "her", her, "/")

MAF=0.05 # min minor allele frequency
aFreq = runif(M, min=MAF, max=(1-MAF))
G = matrix(NA,nrow=N,ncol=M)
for(m in 1:M){
  # 0, 1, 2 based on allele freq
  G[,m] = rbinom(N, 2, prob = aFreq[m])
}
X = scale(G)
GRM = (X %*% t(X))/M

# GRM written for input to ldak and COXMM
# writing at the beginning to keep things clean
write_grm(paste0(output,  "ldak"), as.matrix(GRM))
write.table(GRM, paste0(output, "grm.txt"), row.names = F, col.names = F, sep = '\t', quote = F)
write.table(data.frame('sample_id' = paste0('ID', 1:N)), paste0(output, "names.txt"),
        row.names = F, col.names = T, sep = '\t', quote = F)
write.table(data.frame('FID' = paste0('ID', 1:N), 'FID' = paste0('ID', 1:N)), paste0(output, "ldak.grm.id"),
        row.names = F, col.names = F, sep = '\t', quote = F)

# weibull distribution with uniform baseline hazard
weibull = function(N, shape, scale){
  #set hazard based on liability
  base = exp(scale)
  # simulate baseline hazard
  u = log(1-runif(N))
  # return weibull distribution
  return((-1*u/base)^(1/as.numeric(shape)))
}

# weibull distribution without baseline hazard
weibull_set = function(N, shape, scale){
  #set hazard based on liability
  base = exp(scale)
  # return weibull distribution
  return((-1/base)^(1/as.numeric(shape)))
}

# this will simulate data under the weibull distribution with uniform distribution
# censoring is based on quantile i.e. last 10% censored
weibull_sims = function(shape, scale, K){
  onset = weibull(N, shape, scale)
  censor = quantile(onset, probs = K)

  # indicator for who was censored
  y = as.numeric(onset <= censor)
  onset = pmin(onset, censor)

  minTTE = min(onset)
  maxTTE = max(onset)
  age = minOnset + (onset-minTTE)/(maxTTE-minTTE)*(maxOnset - minOnset)

  return(list("Y"=y, "age"=round(age, 4)))
}

# this is simulated weibull without baseline hazard
# censoring is based on quantile i.e. last 10% censored
weibull_baseline = function(shape, scale, K){
  onset = weibull_set(N, shape, scale)
  censor = quantile(onset, probs = K)

  # indicator for who was censored
  y = as.numeric(onset <= censor)
  onset = pmin(onset, censor)

  minTTE = min(onset)
  maxTTE = max(onset)
  age = minOnset + (onset-minTTE)/(maxTTE-minTTE)*(maxOnset - minOnset)

  return(list("Y"=y, "age"=round(age, 4)))
}

# this simulated weibull distribution with baseline hazard
# this simulated censoring based on exponential (special weibull distribution)
weibull_expcensor = function(shape, scale, censorRate){
  onset = weibull(N, shape, scale)
  censor = rexp(N, rate=censorRate)

  # indicator for who was censored
  y = as.numeric(onset <= censor)
  onset = pmin(onset, censor)

  minTTE = min(onset)
  maxTTE = max(onset)
  age = minOnset + (onset-minTTE)/(maxTTE-minTTE)*(maxOnset - minOnset)

  return(list("Y"=y, "age"=round(age, 4)))
}

write_output = function(data, shape, ftype, run){
  tte_file = paste0(output, "tte", shape, "_", ftype, "_outcome", run, ".txt")
  tte = data.frame('sample_id' = paste0('ID',1:N), 'start' = rep(0, N), 'stop' = data$age, 'event' = data$Y)
  write.table(tte, tte_file, row.names = F, col.names = T, sep = '\t', quote = F)

  cc_file = paste0(output, "cc", shape, "_", ftype, "_outcome", run, ".txt")
  cc = data.frame('FID' = paste0('ID',1:N), 'IID' = paste0('ID',1:N), 'cc' = data$Y)
  write.table(cc, cc_file, row.names = F, col.names = F, sep = '\t', quote = F)

  aoo_file = paste0(output, "age", shape, "_", ftype, "_outcome", run, ".txt")
  aoo = data.frame('FID' = paste0('ID',1:N), 'IID' = paste0('ID',1:N), 'age' = data$age)
  aoo$age = qnorm(rank(aoo$age, ties.method = "random")/(length(aoo$age)+1))
  write.table(aoo, aoo_file, row.names = F, col.names = F, sep = '\t', quote = F)
}

#cat("H2, Censor, Run, Cases\n")
for(run in 1:50){
  beta = rnorm(M)
  l_g = sqrt(h2)*scale(X %*% beta)
  l_e = sqrt(1-h2)*scale(rnorm(N))
  x=rnorm(N)
  x2=rnorm(N)
  covars = scale(x+x2)
  covariates = data.frame('sample_id' = paste0('ID',1:N), 'var1'=x, 'var2'=x2)
  covar_file = paste0(output, "tte_covars", run, ".txt")
  write.table(covariates, covar_file, row.names = F, col.names = T, sep = '\t', quote = F)
  shape=1
  # ideal sims: scale=l_g, alt censoring
  for(cenRate in c(20)){
    label=cenRate*100
    z=weibull_expcensor(shape, l_g, cenRate)
    write_output(z, shape, paste0('cen',label), run)
    cat(her, label, run, round(sum(z$Y)/length(z$Y),3), "\n")
  }
  for(K in c(0.01, 0.05, 0.10, 0.20, 0.40, 0.95)){
    l = l_g + l_e
    z=weibull_baseline(shape, l, K)
    write_output(z, shape, paste0('liab',(K*100)), run)
  }
  for(shape in c(1:3)){
    # ideal sims: scale=l_g
    z=weibull_sims(shape, l_g, 0.95)
    write_output(z, shape, 'base95', run)

    # env sims: scale=l_g + l_e
    l = l_g + l_e
    z=weibull_sims(shape, l, 0.95)
    write_output(z, shape, 'env95', run)

    # fixed sims: scale=l_g + fixed effect
    l = l_g + sqrt(0.01)*covars
    z=weibull_sims(shape, l, 0.95)
    write_output(z, shape, 'fixed1_95', run)

    l = l_g + sqrt(0.50)*covars
    z=weibull_sims(shape, l, 0.95)
    write_output(z, shape, 'fixed50_95', run)
  }
}  


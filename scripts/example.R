# This script is written with the assumption you are working in the COXMM directory.

# This library is needed to create binary GRM)
if (!requireNamespace("genio", quietly = TRUE)) {
  print("Package 'genio' is not installed; use install.packages('genio')")
} else {
  library(genio)
}
# seed to replicate data generation
set.seed(12345)

# Sample Size
N=10000
# Number of SNPs
M=2500
# Variance component (whole number for when writing to a directory)
sigma2g=40
s2g=sigma2g/100
# this is a rescaling of "age" to look like ages
minOnset=30
maxOnset=70
# output directory for data, relative to COXMM directory update if working elsewhere
dir="example/"
# number of replicates you want to create
reps=1

# can write to a sub-directy in example to separate different heritability levels
#output = paste0(dir, "sigma2g", sigam2g, "/")
output = dir

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
}

#cat("H2, Censor, Run, Cases\n")
for(run in 1:reps){
  # snp effect sizes 
  beta = rnorm(M)
  # genetic liability 
  l_g = sqrt(s2g)*scale(X %*% beta)
  # two standard normal covariates
  x=rnorm(N)
  x2=rnorm(N)
  covars = scale(x+x2)
  covariates = data.frame('sample_id' = paste0('ID',1:N), 'var1'=x, 'var2'=x2)
  covar_file = paste0(output, "tte_covars", run, ".txt")
  write.table(covariates, covar_file, row.names = F, col.names = T, sep = '\t', quote = F)
  shape=1
  # tte simulations where scale=l_g, and censoring follows exponential distribution
  for(cenRate in c(20)){
    z=weibull_expcensor(shape, l_g, cenRate)
    write_output(z, shape, paste0('cen'), run)
    cat(sigma2g, label, run, round(sum(z$Y)/length(z$Y),3), "\n")
  }
  for(shape in c(1)){
    for(K in c(0.4)){
      # ideal sims: scale=l_g
      z=weibull_sims(shape, l_g, K)
      write_output(z, shape, paste0('base',K*100), run)

      # fixed sims: scale=l_g + fixed effect
      #l = l_g + sqrt(0.01)*covars
      l = l_g + sqrt(0.50)*covars
      z=weibull_sims(shape, l, K)
      write_output(z, shape, paste0('fixed50_',K*100), run)
    }
  }  
}

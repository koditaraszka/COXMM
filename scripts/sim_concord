library(genio)
library(survival)
library(DescTools)
library(metafor)


N=10000
M=2500
minOnset=30
maxOnset=70
MAF=0.05 # min minor allele frequency

aFreq = runif(M, min=MAF, max=(1-MAF))
G = matrix(NA,nrow=N,ncol=M)
for(m in 1:M){
  # 0, 1, 2 based on allele freq
  G[,m] = rbinom(N, 2, prob = aFreq[m])
}
X = scale(G)
GRM = (X %*% t(X))/M

# weibull distribution with uniform baseline hazard
weibull = function(N, shape, scale){
  #set hazard based on liability
  base = exp(scale)
  # simulate baseline hazard
  u = log(1-runif(N))
  # return weibull distribution
  return((-1*u/base)^(1/as.numeric(shape)))
}


shape=1
cases=rep(1,N)
results = NULL
for (h2 in seq(0.1,3,0.1)){
  h2_log = (h2)/(h2 + (pi^2/6))
  for(run in 1:50){
    beta = rnorm(M)
    l_g = sqrt(h2)*scale(X %*% beta)
    z = weibull(N, shape, l_g)
    reg = coxph(Surv(z, cases) ~ l_g)
    coefs = summary(reg)$coefficients[c(1,3)]
    concord = summary(reg)$concordance[1]
    fit = royston(reg)
    d = fit[1]
    rko = fit[4]
    rn = fit[5]
    results = rbind(results, c(run, h2, 1, h2_log, coefs, concord, d, rko, rn))
    for(K in c(0.01, 0.05, 0.2, 0.4)){
      onset = z
      censor = quantile(onset, probs = K)
      y = as.numeric(onset <= censor)
      onset = pmin(onset, censor)
      reg = coxph(Surv(onset, y) ~ l_g)
      coefs = summary(reg)$coefficients[c(1,3)]
      concord = summary(reg)$concordance[1]
      fit = royston(reg)
      d = fit[1]
      rko = fit[4]
      rn = fit[5]
      results = rbind(results, c(run, h2, K, h2_log, coefs, concord, d, rko, rn))
    }
  }
}

results = data.frame(results)
colnames(results) = c("run", "sigma_g2", "K", "h2", "beta_tte", "se_tte", "concordance", "d", "r2_ko", "r2_n")

final = NULL
for(sigma_g2 in unique(results$sigma_g2)){
	for(K in unique(results$K)){
		temp = results[which(results$sigma_g2==sigma_g2 & results$K==K),]
		mean_concord = mean(temp$concordance)
		se_concord = sd(temp$concordance)/sqrt(50)
		mean_d= mean(temp$d)
		se_d = sd(temp$d)/sqrt(50)
		mean_r2_ko = mean(temp$r2_ko)
		se_r2_ko = sd(temp$r2_ko)/sqrt(50)
		mean_r2_n = mean(temp$r2_n)
		se_r2_n = sd(temp$r2_n)/sqrt(50)
		final = rbind(final, c(sigma_g2, unique(temp$h2), K, mean_concord, se_concord, mean_d, se_d, mean_r2_ko, se_r2_ko, mean_r2_n, se_r2_n))
	}	
}

final = data.frame(final)
colnames(final) = c('sigma_g2', 'h2', 'K', 'concord', 'concord_se', 'd', 'd_se', 'r2_ko', 'r2_ko_se', 'r2_n', 'r2_n_se')

write.table(final, "~/Desktop/Heritability/sims/concord_data.txt", col.names=T, row.names=F, sep='\t', quote=F)

library(genio)

args = commandArgs(trailingOnly=TRUE)

N=10000
M=2500
her=as.numeric(args[1])
h2=her/100
cases=as.numeric(args[2])
K=cases/100
minOnset=30
maxOnset=70
dir='/u/scratch/k/kodicoll/simulations/ltm_data/'
output = paste0(dir, "her", her, "_K", cases, "/")

MAF=0.05
aFreq = runif(M, min=MAF, max=(1-MAF))
G = matrix(NA,nrow=N,ncol=M)
for(m in 1:M){
  # 0, 1, 2 based on allele freq
  G[,m] = rbinom(N, 2, prob = aFreq[m])
}
X = scale(G)
GRM = (X %*% t(X))/M

write_grm(paste0(output,  "ldak"), as.matrix(GRM))
write.table(GRM, paste0(output, "grm.txt"), row.names = F, col.names = F, sep = '\t', quote = F)
write.table(data.frame('sample_id' = paste0('ID', 1:N)), paste0(output, "names.txt"), 
	row.names = F, col.names = T, sep = '\t', quote = F)
write.table(data.frame('FID' = paste0('ID', 1:N), 'FID' = paste0('ID', 1:N)), paste0(output, "ldak.grm.id"), 
	row.names = F, col.names = F, sep = '\t', quote = F)

# simulate data under the case-control liability threshold model
ltm_sims = function(liab){
  t = qnorm((1-K))
  Y = as.numeric(liab > t)
  
  liab = liab[which(Y==1)]
  p_liab = stats::pnorm(liab, lower.tail = F)
  onset = -1*log(1/p_liab - 1)
  
  minMove = min(onset)
  maxMove = max(onset)
  ageOnset = minOnset + ((onset - minMove)/(maxMove - minMove))*(maxOnset - minOnset)
  
  cenLiab = rnorm(length(which(Y==0)))
  p_liab = stats::pnorm(cenLiab, lower.tail = F)
  censor = -1*log(1/p_liab - 1)
  
  minCen = min(censor)
  maxCen = max(censor)
  cenAge = minOnset + ((censor - minCen)/(maxCen - minCen))*(maxOnset - minOnset)
  
  age = rep(NA, length(Y))
  age[which(Y==1)] = ageOnset
  age[which(Y==0)] = cenAge

  return(list("Y"=Y, "age"=round(age, 4)))
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

for(run in 1:50){
  beta = rnorm(M)
  l_g = sqrt(h2)*scale(X %*% beta)
  l_e = sqrt(1-h2)*scale(rnorm(N))
  # case-control sims: LTM
  l = l_g + l_e 
  z=ltm_sims(l)
  write_output(z, '', 'ltm', run)
}

library(coxmeg)
args <- commandArgs(trailingOnly = TRUE)

corr = read.table(args[1], head=F)
corr = as.matrix(corr)

outcome=read.table(args[2],head=T) 
outcome=outcome[,c("stop","event")]

# this is to match coxmm as coxmeg was failing coxmeg only does < not <=
min_case = min(outcome[which(outcome$event==1),"stop"])
drop <- which(outcome$event==0 & outcome$stop <= min_case)
if (length(drop) > 0 ){
  outcome <- outcome[-drop, ]
  corr <- corr[-drop, -drop]
}

#z=coxmeg(outcome, corr, spd=F, type='dense', detap = 'exact', opt = 'bobyqa', solver=1, verbose = F)
z=coxmeg(outcome, corr, spd=F, type='dense', detap = 'exact', opt = 'bobyqa', solver=2, verbose = F)

cat(paste(z$tau, z$tau/(z$tau + pi^2/6)))

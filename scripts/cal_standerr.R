args = commandArgs(trailingOnly=TRUE)

data = read.table(args[1], head=T)
print(data)
theta = data[which(data$type=='original'),]
jack = data[which(data$type!='original'),]

N=theta$N
    
sigma2g = theta$sigma2g
her = theta$h2

m_j = N - jack$N
h_j = N/m_j

sigma2g_notj = jack$sigma2g
her_notj = jack$h2

sigma2g_j = sum(sigma2g - sigma2g_notj) + sum(m_j*sigma2g_notj/N)
her_j = sum(her - her_notj) + sum(m_j*her_notj/N)
    
est_sigma2g_j = h_j*sigma2g - (h_j - 1)*sigma2g_notj
est_her_j = h_j*her - (h_j - 1)*her_notj
    
var_sigma2g_j = 1/dim(jack)[1]*sum((est_sigma2g_j - sigma2g_j)^2/(h_j - 1))
var_her_j = 1/dim(jack)[1]*sum((est_her_j - her_j)^2/(h_j - 1))

# colnames if written to data.frame
cat(paste('source', 'h2', 'h2_var', 'sigma2g', 'sigma2g_var', '\n'))
cat(paste(args[1], round(her,3), round(var_her_j,5), round(sigma2g,3), round(var_sigma2g_j,5), '\n'))

# this can be repeated for each subset when meta-analyzing cohorts with >30,000 samples
# we will want to use a fixed effect meta-analysis
#rma(yi=h2, vi=var, method="FE", data=subsets) 

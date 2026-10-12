args = commandArgs(trailingOnly = T)

trait = args[1]

# Process the whole gwas
data = NULL
for(chr in 1:22){
    # Add the TTE GWAS
    file=paste0("tte/", trait, "", chr, "_tte.txt")
    add_tte = read.csv(file, sep = '\t', head = T)
    colnames(add_tte)[1] = "SNP"
    add_cc = read.csv(paste0("cc/", trait, "_cc", chr, ".assoc"), sep = ' ', head = T)
    colnames(add_cc)[2] = "SNP"
    add = merge(add_tte, add_cc, by = "SNP")
    add = subset(add, select = c(SNP, A1, A2, Score_Stat, z, Score_P, p.value.spa))
    add_age = read.csv(paste0("age/", trait, "_age", chr, ".assoc"), sep = ' ', head = T)
    colnames(add_age)[2] = "SNP"
    add = merge(add, add_age, by = c("SNP", "A1", "A2"))
    add = subset(add, select = c(SNP, A1, A2, Score_Stat, z, Wald_Stat, Score_P, p.value.spa, Wald_P))
    colnames(add) = c("SNP", "A1", "A2", "CC_Z", "TTE_Z", "AGE_Z", "CC_P", "TTE_P", "AGE_P")
    data = rbind(data, add)
}

write.table(data, paste0(trait, "_gwas.txt"), col.names = T, row.names = F, sep = '\t', quote = F)

rm(list=setdiff(ls(), "trait"))
gc()

# Process the training (90% of the cohort GWAS
data = NULL
for(chr in 1:22){
    # Add the TTE GWAS
    file=paste0("tte/", trait, "_train", chr, "_tte.txt")
    add_tte = read.csv(file, sep = '\t', head = T)
    colnames(add_tte)[1] = "SNP"
    add_cc = read.csv(paste0("cc/", trait, "_cc_train", chr, ".assoc"), sep = ' ', head = T)
    colnames(add_cc)[2] = "SNP"
    add = merge(add_tte, add_cc, by = "SNP")
    add = subset(add, select = c(SNP, A1, A2, Score_Stat, z, Score_P, p.value.spa))
    add_age = read.csv(paste0("age/", trait, "_age_train", chr, ".assoc"), sep = ' ', head = T)
    colnames(add_age)[2] = "SNP" 
    add = merge(add, add_age, by = c("SNP", "A1", "A2"))
    add = subset(add, select = c(SNP, A1, A2, Score_Stat, z, Wald_Stat, Score_P, p.value.spa, Wald_P))
    colnames(add) = c("SNP", "A1", "A2", "CC_Z", "TTE_Z", "AGE_Z", "CC_P", "TTE_P", "AGE_P")
    data = rbind(data, add)
}

write.table(data, paste0(trait, "_prs.txt"), col.names = T, row.names = F, sep = '\t', quote = F)

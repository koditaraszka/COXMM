library(SPACox)
library(survival)

args = commandArgs(trailingOnly = T)
trait = args[1]
chr = args[2]
interval = args[4]
covars = read.table('covars/coxmm_noAge.txt', head = T)

# Training samples for PRS
geno_file=paste0(trait, '_train', chr, ".fam")
if(file.exists(geno_file)){
    geno = read.table(geno_file, head = F)
    geno = geno$V1

    pheno = read.table(paste0("sample/", trait, "_train_tte.txt"), head=T)
    pheno = merge(pheno, covars, by = 'sample_id')
    pheno = pheno[complete.cases(pheno),]
    pheno = pheno[which(pheno$sample_id %in% geno),]

    plink_data = paste0(trait, '_train', chr)
    gc()
    output = paste0('tte/', trait, '_train', chr, "_tte.txt")
    # Null model without Age covariate
    null_model = SPACox_Null_Model(Surv(start,stop,event)~Sex+PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10, data=pheno, pIDs=pheno$sample_id, gIDs=geno)
    SPACox.plink(obj.null = null_model, plink.file = plink_data, output.file = output)
    gc()

    # Rerun if interval trait with left truncation
    if (interval==1){
        pheno = read.table(paste0(pheno_dir, 'sample/', trait, "_train_interval.txt"), head=T)
        pheno = merge(pheno, covars, by = 'sample_id')
        pheno = pheno[complete.cases(pheno),]
        pheno = pheno[which(pheno$sample_id %in% geno),]
        output = paste0('tte/', trait, '_train', chr, "_interval.txt")
        # Interval model with noAge covariate
        null_model = SPACox_Null_Model(Surv(start,stop,event)~Sex+PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10, data=pheno, pIDs=pheno$sample_id, gIDs=geno)
        # Full interval model with noAge covariate
        SPACox.plink(obj.null = null_model, plink.file = plink_data, output.file = output)
    }

}
print(paste0("PRS Training Samples GWAS complete for ", trait, " on chromosome ", chr))                                                

# All samples for GWAS
geno_file=paste0(trait, chr, ".fam")
if(file.exists(geno_file)){
    geno = read.table(geno_file, head = F)
    geno = geno$V1
    
    pheno = read.table(paste0("sample/", trait, "_tte.txt"), head=T)
    pheno = merge(pheno, covars, by = 'sample_id')
    pheno = pheno[complete.cases(pheno),]
    pheno = pheno[which(pheno$sample_id %in% geno),]
    
    plink_data = paste0(trait, chr)
    gc()
    output = paste0('tte/', trait, chr, "_tte.txt")
    # Null model without Age covariate
    null_model = SPACox_Null_Model(Surv(start,stop,event)~Sex+PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10, data=pheno, pIDs=pheno$sample_id, gIDs=geno)
    SPACox.plink(obj.null = null_model, plink.file = plink_data, output.file = output)
    gc()
    
    # Rerun if interval trait with left truncation
    if (interval==1){
        pheno = read.table(paste0(pheno_dir, 'sample/', trait, "_interval.txt"), head=T)
        pheno = merge(pheno, covars, by = 'sample_id')
        pheno = pheno[complete.cases(pheno),]
        pheno = pheno[which(pheno$sample_id %in% geno),]
        output = paste0('tte/', trait, chr, "_interval.txt")
        # Interval model with noAge covariate
        null_model = SPACox_Null_Model(Surv(start,stop,event)~Sex+PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10, data=pheno, pIDs=pheno$sample_id, gIDs=geno)
        # Full interval model with noAge covariate
        SPACox.plink(obj.null = null_model, plink.file = plink_data, output.file = output)
    }

}
print(paste0("All Samples GWAS complete for ", trait, " on chromosome ", chr))

module load plink/1.90b624

# Prepare genotypes for all methods
plink --bfile orig/${CHR} \
    --snps-only \
    --exclude mhc.snp.txt \
    --geno 0.01 \
    --hwe 1e-7 \
    --maf 0.01 \
    --keep all_samples.txt \
    --make-bed \
    --out ${CHR}

# How SNPs are pruned
plink --bfile ${CHR} \
      --indep-pairwise 50 5 0.5
      --out pruned_snps.${CHR}

# All samples included for the trait, SPACox needs the genotype to only have samples in the trait
plink --bfile ${CHR} \
      --keep samples/${trait}_tte.txt \
      --out ${trait}${CHR} \
      --make-bed

# Included 90% of the samples
plink --bfile ${CHR} \
      --keep samples/${trait}_train.txt \
      --out ${trait}_train${CHR} \
      --make-bed

# Includes the 10% heldout samples
plink --bfile ${CHR} \
    --keep samples/${trait}_test.txt \
    --out ${trait}_prs_data \
    --extract pruned_snps.${CHR}.prune.in \
    --make-bed

# SPACox (interval is 0/1) for whether the trait is duration or left truncated
Rscript run_spacox.R $trait $CHR $interval

# LDAK for age-of-onset (no age covariate, gcta.txt also exists with age-at-enrollment as a covariate)
./ldak --bfile ${SCRATCH}/${CHR} \
    --pheno data/${trait}_age.txt \
    --covar covars/gcta_noAge.txt \
    --linear age/${trait}_age${CHR}

# LDAK for case-control (no age covariate, gcta.txt includes age-at-enrollment)
./ldak --bfile ${SCRATCH}/${CHR} \
    --pheno data/${trait}_cc.txt \
    --covar covars/gcta_noAge.txt \
    --logistic cc/${trait}_cc${CHR}

# LDAK for case-control PRS training
./ldak --bfile ${SCRATCH}/${CHR} \
    --pheno data/${trait}_train.txt \
    --covar covars/gcta_noAge.txt \
    --logistic cc/${trait}_cc_train${CHR}

# this Rscript processes the gwas results for both the whole cohort and training GWAS for the PRS
Rscript process_gwas.R ${trait}

# PRS in plink
plink --bfile ${trait}_prs_data \
    --score ${trait}_prs.txt 1 3 4 header \
    --out ${trait}_prs_cc

plink --bfile ${trait}_prs_data \
    --score ${trait}_prs.txt 1 3 5 header \
    --out ${trait}_prs_tte

# LD-Score Regression
# Process the TTE GWAS
python LDSC/munge_sumstats.py \
    --snp SNP \
    --a1 A1 \
    --a2 A2 \
    --p TTE_P \
    --signed-sumstats TTE_Z,0 \
    --sumstats gwas/${trait}_gwas.txt \
    --N ${N} \
    --out ldsc/${trait}_tte

# Process the TTE GWAS
python LDSC/munge_sumstats.py \
    --snp SNP \
    --a1 A1 \
    --a2 A2 \
    --p CC_P \
    --signed-sumstats CC_Z,0 \
    --sumstats gwas/${trait}_gwas.txt \
    --N ${N} \
    --out ldsc/${trait}_cc

# Gather the genetic correlation between the traits
# The reference ld is stored elsewhere called ${ldsc_dir}
files=(ldsc/*gz)
for ((i=0; i<${#files[@]}; i++)); do
    file1=$(basename "${files[i]}")
    file1="${file1%.sumstats.gz}"
    for ((j=i+1; j<${#files[@]}; j++)); do
        file2=$(basename "${files[j]}")
        file2="${file2%.sumstats.gz}"
        if [ ! -f "${dir}/${file1}_${file2}.log" ]; then
            python LDSC/ldsc.py --rg ${files[i]},${files[j]} --ref-ld-chr ${ldsc_dir} --w-ld-chr ${ldsc_dir} --out gencorr/${file1}_${file2}
        fi
    done
done


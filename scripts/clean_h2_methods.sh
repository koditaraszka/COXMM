# Example from scripts/reviwer_sims.R
pref="sims_N10000"
sims="${pref}_tte1_weibull_K20"
num=1

# Running COXMM
module load python/3.9.6
/usr/bin/time -f "%e %M" python3 COXMM/coxmm.py \
    -e sims/${sims}_outcome${num}.txt \
    -g sims/${pref}_grm.txt \
    -n sims/${pref}_names.txt \
    -o coxmm/${sims}_result${num}.txt 2>> coxmm/${sims}_result${num}_compute.log

# Running BayesW
module load gcc boost eigen
/usr/bin/time -f "%e %M" hydra/src/hydra_G \
  --bfile sims/${pref}_genotypes \
  --pheno sims/${sims}_outcome${num}.phen \
  --failure sims/${sims}_outcome${num}.fail \
  --number-individuals 10000 \
  --number-markers 2500 \
  --mpibayes bayesWMPI \
  --quad_points 11 \
  --chain-length 1000 \
  --thin 5 \
  --mcmc-out-dir bayesW \
  --mcmc-out-name ${sims}_result${num} \
  --shuf-mark 1 \
  --sync-rate 1 \
  --S 0.001,0.01,0.1

# NOTE these three methods were not ran on these specific simulations, but could have been
# or were ran on nearly identical simulations in an early version of the paper

# Running COXMEG
module load R/4.3.0
Rscript run_coxmeg.R sims/${pref}_grm.txt \
    sims/${sims}_outcome${num}.txt 1> coxmeg/${sims}_result${num}.txt

# when running LDAK on real data, we need to adjust the GRM for each randomly sampled grouping of cohort:
./$ldak \
   --adjust-grm ${covars}/group${num}.covar_noage \
   --grm ${covars}/group${num} \
   --covar ${covars}/gcta_noAge.txt #gcta.txt includes age-at-enrollment covariate

# Running LDAK case-control
./ldak --pheno sims/${sims}_outcome${num}_cc.txt \
   --prevalence 0.2 \
   --grm sims/${pref}_ldak
   --kinship-details NO \
   --pcgc pcgc/${sims}_result${num}

# Running LDAK case-control
./ldak --pheno sims/${sims}_outcome${num}_cc.txt \ 
   --prevalence 0.2 \
   --grm sims/${pref}_ldak
   --kinship-details NO \
   --pcgc pcgc/${sims}_result${num}

# Running LDAK age-of-onset
./ldak --pheno sims/${sims}_outcome${num}_age.txt \
   --grm sims/${pref}_ldak
   --kinship-details NO \
   --pcgc age/${sims}_result${num}

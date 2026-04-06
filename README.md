# COXMM

This is an implementation of COX proportional hazard Mixed Model (frailty model)

Much of the implemenation follows that of COXMEG: a maximum likelihood implementation of a cox proportional hazard model for genome-wide association studies (He and Kulminski Genetics, 2020). We note that our method is more scalable and efficient than COXMEG when using semidefinite relatedness matrixes (GRM). We do not implement any speedups for sparsity, etc. We expect COXMM to be able to converge for ~30,000 samples using 64G of RAM in a few hours.

While the standard error can be directly computed from the second derivative, previous work has shown that this underestimates the true standard error. We therefore only provide the option to do block jackknifing.

## Installation

The following modules will need to be installed:
`numpy, pandas, pybobyqa, scipy`

The dependencies can be installed using pip:

`pip install -r install/requirements.txt` 

They can also be installed using conda:

` conda env create -f install/coxmm_env.yaml`
 
### Versions

Our analyses were conducted using Python 3.9.6

The module versions for the results were:
	numpy 1.24.2
	pandas 2.0.0
	pybobyqa 1.3
	scipy 1.10.1

Use caution when using newer python versions though python 3.10 and 3.11 are likely compatible.
COXMM is likely forward compatible with newer versions of the modules as well, but this has not been fully vetted.

## Running COXMM

To list off the arguments and run the method run:
`python coxmm.py -h`

## Arguments

### Required Arguments
  
  1. -e/--events

This argument is the path to a tab-separated file which contains (at least) four columns and the header rows: 
	* sample_id: unique identifier for each person (or IID for binary GRM).
		The column name can be changed using -s/--sample_id; must be consistent across files
	* start: set to 0 for everyone if there is no left-censoring in the model
	* stop: time-to-event for every individual
	* event indicator for whether stop is the time-to-event (1) or time-at-censoring (0)
  
  2. -g/--grm

This argument is the path to the GRM. Assumes binary format (auto-detected by /path/prefix.grm.bin exists)
		Otherwise, assumes full/path/filename.suffix which is a tab-delimited file without header/rownames
		
### Optional Arguments
  1. -n/--grm_names

This argument is the path to the file containing a single column with the header (sample_ids: see -s/--sample_id). This file continas the unique identifier for each individual in -g/--grm only used if grm is tab-delimited file
  
  2. -s/--sample_id

This argument contains the column name for sample_id if not using sample_id (or IID if using GCTA format grm; 
	Examples: FID or EID, etc.
	
  3. -f/--fixed

This argument is the path to the tab-separated file containing fixed effects features. This file contains a header row which includes sample_id (see -s/--sample_id)

  4. -o/--output 

This argument is the path to the tab-separated output file. Default = results.txt

  5. -j/--jackknife

This argument needs two arguments: total number of splits and which split to currently run (base 1). This is used for estimating the standard error. While COXMM provides the analytical standard error, previous work indicates it may be underestimated  

  6. --joint_jackknife

This argument is the number of jackknife splits to run. This allows the user to run the full sample heritability estimation and jackknife estimations in a single run instead of running the script on the full sample followed by running -j/--jackknife for each split. It cannot be used with --gwas or --jackknife.

  7. -d/--seed

This argument should be use dwith -j/--jackknife to have consistent data splitting across jackknife runs. Default = 123  

## Sample Run/Scripts

For the following steps, I used a MacBook Pro with Apple M2 Pro Chip (16GB RAM, 512GB SSD Storage)

### Simulations

We can simulate example data which takes ~3 min for 10,000 samples (creating/writing the GRM has the largest effect on running time).

`Rscript scripts/example.R`

This script is written to be easily adaptible to different heritability values, sample sizes, and number of replications for further testing. (There is a seed set as well to explicitly recreate the data in the `example` directory. This script depends on the 'genio' package (install.packages('genio'))

There are outcomes and results in the `example/her40` directory, but the GRM could not be uploaded.

### Heritability estimate with covariates

To get a point estimate for heritability with covariates present (~10 min)

`python coxmm.py -e example/tte1_fixed50_40_outcome1.txt -g example/grm.txt -n example/names.txt -f example/tte_covars1.txt -o example/tte_covar_h2.txt`

The true variance component (sigma2g) is 0.4, and the weibull distribution had a scale=1 and shape=genetic liability + 2 fixed effects. There was cohort censoring with 40% of cases observed. The GRM is a square with the sample names in a separate file. The output for this run is written to `example/tte_covar_h2.txt`. The true heritability is 0.4/(0.4 + pi^2/6) = 0.196 and the estimate is 0.213.

### Jackknife estimate for exponentially censored data

To get a jackknife estimate to establish the std error for the heritability (~5 min)

`python coxmm.py -e example/tte1_cen20_outcome1.txt -g example/ldak -j 10 1 -d 123 -o example/tte_expcen_h2_jack1.txt`

The true variance component (sigma2g) is 0.4, and the weibull distribution had a scale=1 and shape=genetic liability. There was exponential censoring with a censoring rate = 20 resulting in 5% of cases observed. The GRM is a binary GRM (which only differs from the square GRM by floating point precision). The output is written to `example/tte_expcen_h2_jack1.txt`. The true heritability is 0.196, the jackknife estimate (using 90% of the cohort) is 0.147. This step needs to be repeated for -j 10 2, all the way to -j 10 10 using the same seed (as well as needing a point estimate).

### Joint jackknife estimate for ideal simulation

To get the point estimate and all jackknife estimates for establishing standard error (~30 min)

`python coxmm.py -e example/tte1_base40_outcome1.txt -g example/ldak --joint_jackknife 5 -d 1234 -o example/tte_idealsim.txt`

The true variance component (sigma2g) is 0.4, and the weibull distribution had a scale=1 and shape=genetic liability. There was cohort censoring with 40% of cases observed. The GRM is a binary GRM. The output is written to `example/tte_idealsim.txt`. The true heritability is 0.196. The point estimate and all 5 jackknife estimates (using 80% of the cohort each) are written to this file.

To create the final output run:

`Rscript scripts/cal_standerr.R example/tte_idealsim.txt`

which gives the following output:
  `source h2 h2_var sigma2g sigma2g_var
  example/tte_idealsim.txt 0.212 0.00023 0.441 0.00159` 

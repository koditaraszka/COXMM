# COXMM

This is an implementation of COX proportional hazard Mixed Model (frailty model)

While much of the implemenation follows that of COXMEG and its notation, it does not utilize any of the speedups.
We note that our method is more scalable and efficient than COXMEG when using semidefinite relatedness matrixes (GRM).
We expect that samples of 30,000 can be ran on 64G of RAM in a few hours.

While we provide the standard error directly computed from the second derivative, previous work has shown that this underestimates the true standard error. We therefore provide an option to do block jackknifing.

## Installation

The following packages will need to be installed:
`logging, argparse, numpy, pandas, pandas_plink, os, pybobyqa, scipy`

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

  8. -w/--gwas

This argument is used in conjunction with -p/--plink and takes in the heritability estimate (i.e. estimate heritability then provide point estimate here: -w 0.25. Note: use SPACox or GATE -- implemented/exact but SLOW.

  9. -p/--plink

This argument is the path to the plink bim/bed/fam prefix file (NOTE: this method is not especially scalable (SPACox/GATE) but is exact (COXMEG).

  10. -c/--centerScale

This argument is an idicator for whether the SNPs passed in with -p/--plink should be centered and scaled 

## Sample Run/Scripts

This will a todo task, but there are simulation scripts readily available in the scripts directory.

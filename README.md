# COXMM

This is an implementation of COX proportional hazard Mixed Model (frailty model)

While much of the implemenation follows that of COXMEG and its notation, it does not utilize any of the speedups.
We note that our method is more scalable and efficient than COXMEG when using semidefinite relatedness matrixes (GRM).
We expect that samples of 30,000 can be ran on 64G of RAM in a few hours.

While we provide the standard error directly computed from the second derivative, previous work has shown that this underestimates the true standard error. We therefore provide an option to do block jackknifing.


## Installation

The following packages will need to be installed:
`argparse, numpy, pandas, pandas_plink, os, random, math, pybobyqa, scipy`


## Arguments

### Required Arguments
  
  1. -e/--events

This argument is the path to a tab-separated file which contains four columns and the header rows: 
	* sample_id: unique identifier for each person (column name can be changed using -s/--sample_id)
	* start: set to 0 for everyone if there is no left-censoring in the model
	* stop: time-to-event for every individual
	* event indicator for whether stop is the time-to-event (1) or time-at-censoring (0)
  
  2. -g/--grm

This argument is the path to a tab-separated file containing precomputed relatedness matrices. There is no header row in this file.
  
  3. -n/--grm_names

This argument is the path to the file containing a single column with the header (sample_ids: see -s/--sample_id). This file continas the unique identifier for each individual in -g/--grm

### Optional Arguments
  
  1. -s/--sample_id

This argument contains the column name for sample_id if not using sample_id (e.g. IID or EID, etc.)

  2. -f/--fixed

This argument is the path to the tab-separated file containing fixed effects features. This file contains a header row which includes sample_id (see -s/--sample_id)

  3. -o/--output 

This argument is the path to the tab-separated output file. Default = results.txt

  4. -j/--jackknife

This argument needs two arguments: total number of splits and which split to currently run (base 1). This is used for estimating the standard error. While COXMM provides the analytical standard error, previous work indicates it may be underestimated  

  5. -d/--seed

This argument should be use dwith -j/--jackknife to have consistent data splitting across jackknife runs. Default = 123  

  6. -w/--gwas

This argument is used in conjunction with -p/--plink and takes in the heritability estimate (i.e. estimate heritability then provide point estimate here: -w 0.25

  7. -p/--plink

This argument is the path to the plink bim/bed/fam prefix file (NOTE: this method is not especially scalable (SPACox/GATE) but is exact (COXMEG)

  8. -c/--centerScale

This argument is an idicator for whether the SNPs passed in with -p/--plink should be centered and scaled 


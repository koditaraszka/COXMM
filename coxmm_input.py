"""This script handles the input/output for COXMM
		parses command-line arguments
		loads and subsets the GRM (text or GCTA format)
		loads and filters the phenotpye and (optional) covariate files 

Author: Kodi Taraszka
Email:  Kodi_Taraszka@dfci.harvard.edu
"""

import logging
import numpy as np
import argparse
import pandas as pd
import os.path as path
from pandas_plink import read_plink

logger = logging.getLogger(__name__)

# TODO: handle missing data in fixed effects/outcome, assumes data preprocessed
# TODO: allow multiple GRMs
class COXMM_IO():
	"""Handles all data input and preprocessing for COXMM
		parses CLI arguments
		loads phenotypes and (optional) covariates
		read/inverts GRM
		
	Attributes:
		N (int): Number of analysis samples (after intersection and filtering).
		M (int): Number of fixed-effect covariates (0 if none supplied).
		grm (np.ndarray): Inverted GRM, shape (N, N), float64.
		times (np.ndarray): N×2 array of [start, stop] times.
		events (np.ndarray): Length-N binary event indicator (1=case, 0=censored).
		fixed (np.ndarray or None): N×M covariate matrix, or None if not supplied.
		names (Index): Sample IDs in analysis order.
		output (str): Path for the results output file.
		gwas (float or None): Fixed tau for GWAS mode; None for estimation mode.
		plink (str or None): Plink file prefix for GWAS mode.
	"""

	def __init__(self):
		self.output = ''
		self.grmN = 0
		self.N = 0
		self.M = 0
		self.names = None
		self.grm = None
		self.fixed = None
		self.events = None
		self.plink = None
		self.gwas = None
		self.center = False
		self.joint_jackknife = None
		self.jackknife_type_label = 'original'
		self.setup()
		self.times = self.events[['start', 'stop']].to_numpy()
		self.names = self.events.index.astype('str')
		self.events = self.events.event.to_numpy()

	def setup(self):
		"""Parse arguments, load the GRM, and prepare phenotypes and covariates.

				Detects GRM format automatically: binary when <prefix>.grm.bin exists, 
						otherwise tab-delimited text needs separate names file. 
				Subsets the GRM, inverts via eigendecomposition, and one-hot encodes or scales covariates as needed.
				
				Results are stored in self.grm, self.fixed, and self.M.
				"""	
		args = self.def_parser()
		self.center = args.center
		self.output = args.output
		self.gwas = args.gwas
		self.plink = args.plink
		self.joint_jackknife = args.joint_jackknife
		# type label written to the output file; set here so fit() can pass it through
		if args.jackknife is not None:
			self.jackknife_type_label = f'jackknife_{args.jackknife[1]}'
		else:
			self.jackknife_type_label = 'original'

		# Detect GRM format: binary (GCTA .grm.bin/.grm.id) when
		# <grm>.grm.bin exists, otherwise text (tab-delimited square matrix
		# with a separate names file supplied via -n/--grm_names).
		binary_mode = path.isfile(args.grm + ".grm.bin")

		if binary_mode:
			# Read sample IDs from the GCTA .grm.id file (two columns: FID IID,
			# no header). We use the IID column (column 1) as the sample identifier.
			grm_id_file = args.grm + ".grm.id"
			grm_names = pd.read_csv(grm_id_file, sep='\t', header=None, names=['FID', 'IID'])
			grm_names = grm_names.rename(columns={'IID': 'sample_id'})
			# Drop the FID column — only IID is used for matching
			grm_names = grm_names[['sample_id']]
			self.grmN = grm_names.shape[0]
		else:
			# Text mode: -n/--grm_names required
			if args.grm_names is None:
				raise ValueError("Text GRM format requires -n/--grm_names (path to sample names file)")
			grm_names = pd.read_csv(args.grm_names, header=0)
			self.grmN = grm_names.shape[0]

		
		# Cache args to re-load data for each joint jackknife split; raw GRM is NOT cached
		# re-read from disk inside _reload_for_jackknife() for each split.
		if args.joint_jackknife is not None:
			self._joint_n_splits	   = args.joint_jackknife
			self._joint_grm_path	   = args.grm
			self._joint_grm_names_path = args.grm_names  # None for binary mode
			self._joint_events_path	= args.events
			self._joint_sample_id	  = args.sample_id
			self._joint_seed		   = args.seed
			self._joint_fixed_path	 = args.fixed
			self._joint_binary_mode	= binary_mode
			self._joint_grm_names	  = grm_names.copy()

		self.events = self.process_events(args.sample_id, args.events, grm_names, args.jackknife, args.seed, binary_mode)

		idx = self.events.rownum.to_numpy()
		grm = self._read_grm(args.grm, binary_mode, idx, self.grmN, self.N)
		self.grm = self._invert_grm(grm)

		if args.fixed is not None:
			logger.warning("Minimal checks on issues with fixed effects input. Be careful")
			fixed_df, self.M = self._load_fixed(
				args.fixed, args.sample_id, binary_mode, self.events.index
			)
			if self.plink is not None:
				fixed_df.insert(0, 'SNP', np.nan)
				self.M = fixed_df.shape[1]
			self.fixed = fixed_df.to_numpy()

	def _read_grm(self, grm_path, binary_mode, idx, grmN, N):
		"""Read and subset the GRM from disk.

		Handles both GCTA binary (.grm.bin) and tab-delimited text formats.
		Uses row streaming when more than 10% of GRM samples are excluded, to
		avoid allocating the full grmN×grmN matrix.

		Args:
			grm_path (str): GRM file prefix (binary) or file path (text).
			binary_mode (bool): True for GCTA binary format (.grm.bin/.grm.id).
			idx (np.ndarray): Integer row/column indices to extract (GRM order).
			grmN (int): Total number of samples in the GRM file.
			N (int): Number of analysis samples (== len(idx)).

		Returns:
			np.ndarray: Subset GRM, shape (N, N), float64, not yet inverted.
		"""
		if binary_mode:
			# Note: .grm.bin stored as float32 while the text GRM is float64
			# Results may differ slightly between the two formats.
			
			# Read GCTA binary GRM (.grm.bin): lower triangle
			grm_bin_file = grm_path + ".grm.bin"
			n = grmN

			# Stream only the needed rows when >10% of GRM samples are excluded.
			# 10% threshold as there is a trade-off between this task and just reading in full file
			excluded_frac = 1.0 - N / n
			if excluded_frac > 0.1:
				idx_sorted = np.sort(idx)
				# Map original index → position in the output M×M matrix
				idx_pos = np.full(n, -1, dtype=np.int64)
				for pos, orig in enumerate(idx):
					idx_pos[orig] = pos
				M_out = len(idx)
				grm = np.zeros((M_out, M_out), dtype=np.float64)
				# ptr tracks how many entries of idx_sorted are <= the current row i.
				ptr = 0
				with open(grm_bin_file, 'rb') as fh:
					for i in idx_sorted:
						# Seek to start of row i in the lower triangle and read
						# all i+1 entries: element (i,j) for j=0..i
						fh.seek(int(i * (i + 1) // 2) * 4)
						row_vals = np.frombuffer(fh.read((i + 1) * 4), dtype=np.float32).astype(np.float64)
						ri = idx_pos[i]
						# Advance pointer to include all idx_sorted entries <= i
						while ptr < M_out and idx_sorted[ptr] <= i:
							ptr += 1
						cols_in_idx = idx_sorted[:ptr]
						ci_vals = idx_pos[cols_in_idx]
						vals = row_vals[cols_in_idx]
						grm[ri, ci_vals] = vals
						grm[ci_vals, ri] = vals
			else:
				n_tri = n * (n + 1) // 2
				tri = np.fromfile(grm_bin_file, dtype=np.float32, count=n_tri).astype(np.float64)

				# Reconstruct full symmetric NxN matrix from lower triangle
				grm_full = np.zeros((n, n), dtype=np.float64)
				rows, cols = np.tril_indices(n)
				grm_full[rows, cols] = tri
				grm_full[cols, rows] = tri

				# Subset to analysis samples
				grm = grm_full[np.ix_(idx, idx)]
		else:
			# Text mode: tab-delimited square matrix, no header, no row labels.
			# Stream rows when >10% of GRM samples are excluded from the analysis
			# 10% threshold as there is a trade-off between this task and just reading in full file
			excluded_frac = 1.0 - N / grmN
			if excluded_frac > 0.1:
				idx_sorted = np.sort(idx)
				idx_set = set(idx_sorted.tolist())
				row_buf = {}
				with open(grm_path) as fh:
					for line_num, line in enumerate(fh):
						if line_num in idx_set:
							row_buf[line_num] = np.array(line.split(), dtype=np.float64)
						if line_num == idx_sorted[-1]:
							break
				grm = np.array([row_buf[i] for i in idx])
				if grm.shape[1] != grmN:
					raise ValueError("GRM: " + grm_path + " column count does not match names file")
				grm = grm[:, idx]
			else:
				grm = pd.read_csv(grm_path, sep='\t', header=None).to_numpy(dtype=np.float64)
				if grm.shape[0] != grm.shape[1]:
					raise ValueError("GRM: " + grm_path + " is not a square matrix")
				grm = grm[np.ix_(idx, idx)]
		return grm

	def _invert_grm(self, grm):
		"""Invert GRM via eigendecomposition

		Args:
			grm (np.ndarray): Raw symmetric GRM, shape (N, N), float64.

		Returns:
			np.ndarray: Inverted GRM, shape (N, N), float64.
		"""
		Lambda, U = np.linalg.eigh(grm)
		Lambda[Lambda < 1e-10] = 1e-6  # numerical stability
		return (U * (1/Lambda)) @ U.T

	def _load_fixed(self, fixed_path, sample_id, binary_mode, index):
		"""Processes fixed-effects file. Steps:
			Reads the tab-delimited fixed effects file
			renames the ID column to 'sample_id'
			one-hot encodes categorical columns, centers
			scales numerical covariates with variance outside [0.01, 100]
			reindexes to match the analysis sample order.

		Args:
			fixed_path (str): Path to the tab-delimited fixed effects file.
			sample_id (str or None): ID column name override; None uses the
				default ('IID' for binary GRM, 'sample_id' for text GRM).
			binary_mode (bool): True → default ID column is 'IID'.
			index: pandas Index to reindex the result against (events order).

		Returns:
			tuple[pd.DataFrame, int]: Reindexed fixed-effects DataFrame (not
				yet converted to numpy) and M (number of columns). The caller
				is responsible for calling .to_numpy() and inserting any
				additional columns (e.g. SNP placeholder for GWAS mode).
		"""
		fixed_df = pd.read_csv(fixed_path, sep='\t', header=0)
		if sample_id is not None:
			fixed_df = fixed_df.rename(columns={sample_id: 'sample_id'})
		elif not binary_mode:
			pass  # text mode default column name is already 'sample_id'
		else:
			fixed_df = fixed_df.rename(columns={'IID': 'sample_id'})

		fixed_df = fixed_df.set_index('sample_id')
		columns = fixed_df.columns
		for col in columns:
			if fixed_df[col].dtype == object:
				logger.info(f"Column '{col}' is a categorical variable and will be one-hot encoded")
				one_hot = pd.get_dummies(fixed_df[col])
				one_hot.pop(one_hot.columns[0])
				fixed_df = fixed_df.drop(col, axis=1)
				fixed_df = fixed_df.join(one_hot)
			elif fixed_df[col].dtype == np.float64:
				var = fixed_df[col].var()
				# 100 and 0.01 chosen as reasonable thresholds may want to adjust
				if var > 100 or var < 0.01:
					logger.warning(f"Column '{col}' has variance {var:.4g} (outside [0.01, 100]) and will be centered and scaled")
					mean = fixed_df[col].mean()
					fixed_df[col] = (fixed_df[col] - mean) / np.sqrt(var)
					logger.info(f"  '{col}' after scaling: mean={fixed_df[col].mean():.4g}  var={fixed_df[col].var():.4g}")

		fixed_df = fixed_df.reindex(index=index)
		return fixed_df, fixed_df.shape[1]

	def def_parser(self):
		"""Define and parse CLI arguments; validate input file existence.

		Returns:
			argparse.Namespace: Parsed arguments with validated file paths.

		Raises:
			ValueError: If required files are missing or argument combinations
				are invalid (e.g. --gwas without --plink).
		"""
		parser = argparse.ArgumentParser(description = "This program runs a cox proportional hazard mixed model with multiple components")

		required = parser.add_argument_group('Required Arguments')
		required.add_argument('-e', '--events', required = True,
			help = 'path to outcome file with four columns: sample_id (or IID for binary GRM), start, stop, event. The ID column name is changeable with -s/--sample_id')
		required.add_argument('-g', '--grm', dest = 'grm', required = True,
			help = 'GRM input. For binary format (GCTA): prefix for <prefix>.grm.bin and <prefix>.grm.id files (auto-detected when <prefix>.grm.bin exists). For text format: path to tab-delimited square matrix file (no header, no row labels); also requires -n/--grm_names.')
		optional = parser.add_argument_group('Optional Arguments')
		optional.add_argument('-n', '--grm_names', dest = 'grm_names',
			help = 'text GRM only: path to file with one column of sample IDs matching GRM row order (header row names the ID column). Not used for binary GRM.')
		optional.add_argument('-s', '--sample_id', dest = 'sample_id',
			help = 'column name for the sample ID in the events/fixed effects files. For binary GRM defaults to IID; for text GRM defaults to sample_id.')
		optional.add_argument('-f', '--fixed', dest = 'fixed',
			help = 'path tab delim file containing fixed effects features. First row containing column names')
		optional.add_argument('-o', '--output', dest = 'output', default = 'results.txt',
			help = 'path to output file. Default = results.txt')
		optional.add_argument('-j', '--jackknife', dest = 'jackknife', nargs = '*',
			help = 'perform one round of jackknife sampling. Needs two arguments: total number of splits and which split currently running (base 1). E.G. -j 10 1')
		optional.add_argument('--joint_jackknife', dest = 'joint_jackknife', type = int, default = None,
			help = 'run heritability estimation on the full sample and all jackknife splits in a single invocation. '
				   'Argument is the total number of splits. Output has a "type" column: original, jackknife_1, ... jackknife_N. '
				   'Cannot be used with --gwas or --jackknife.')
		optional.add_argument('-d', '--seed', dest = 'seed', default = 123, type = int,
			help = 'random seed to be used with -j/--jackknife, to set split (use same over all splits. Default = 123')
		optional.add_argument('-w', '--gwas', dest = 'gwas', type = float,
			help = 'run GWAS with the heritabilty estimate provided with this argument and used alongside -p/--plink; Note: use SPACox or GATE -- implemented but SLOW')
		optional.add_argument('-p', '--plink', dest = 'plink',
			help = 'path to prefix for plink bim/bed/fam files and used alongside -w/--gwas; Note: use SPACox or GATE -- GWAS is SLOW')
		optional.add_argument('-c', '--centerScale', dest = 'center', action = 'store_true', default = False,
			help = 'indicate if GWAS SNPs should be centered and scaled; Note: use SPACox or GATE -- GWAS is SLOW')

		args = parser.parse_args()
		# basic checks on input
		if args.fixed is not None:
			if not path.isfile(args.fixed):
				raise ValueError("The fixed effect file does not exist")

		if not path.isfile(args.events):
			raise ValueError("The outcomes/events file does not exist")

		# GRM existence checks depend on detected format
		binary_mode = path.isfile(args.grm + ".grm.bin")
		if binary_mode:
			if not path.isfile(args.grm + ".grm.id"):
				raise ValueError("Binary GRM detected but .grm.id file not found: " + args.grm + ".grm.id")
		else:
			if not path.isfile(args.grm):
				raise ValueError("The GRM file does not exist: " + args.grm)
			if args.grm_names is None:
				raise ValueError("Text GRM format requires -n/--grm_names")
			if not path.isfile(args.grm_names):
				raise ValueError("The GRM names file does not exist: " + args.grm_names)

		if args.gwas is not None and args.plink is None:
			raise ValueError("GWAS indicated but plink files not included")

		if args.gwas is None and args.plink is not None:
			raise ValueError("Plink path included but GWAS was not indicated/heritabilty estimate not provided")

		if args.plink is not None and not path.isfile(args.plink + ".bed"):
			raise ValueError("The plink file does not exist")

		if args.gwas is not None:
			if args.gwas > 1 or args.gwas < 0:
				logger.warning(f"The heritability estimate provided ({args.gwas}) is not between [0, 1], which is unexpected")

		if args.joint_jackknife is not None and args.gwas is not None:
			raise ValueError("--joint_jackknife cannot be used with --gwas")
		if args.joint_jackknife is not None and args.jackknife is not None:
			raise ValueError("--joint_jackknife and --jackknife cannot be used together")
		if args.joint_jackknife is not None and args.joint_jackknife < 2:
			raise ValueError("--joint_jackknife requires at least 2 splits")

		return(args)


	def process_events(self, sample_id, file, grm_names, jackknife, seed, binary_mode=False):
		"""Process the phenotype file. Steps:
			Reads a tab-delimited file with columns [id, start, stop, event],
			Intersects with GRM samples, (optionally subset to jackknife split sample)
			Removes individuals with start >= stop and censored before the earliest observed event time
			Sorts by stop time.

		Args:
			sample_id (str or None): Name of the ID column in the events file;
				defaults to 'IID' (binary GRM) or 'sample_id' (text GRM).
			file (str): Path to the events/outcomes file.
			grm_names (pd.DataFrame): DataFrame with a 'sample_id' column
				listing GRM samples in GRM row order.
			jackknife (list or None): [n_splits, split_index] for one round of
				block jackknife; None to use all samples.
			seed (int): Random seed for jackknife shuffling.
			binary_mode (bool): True if loading a GCTA binary GRM.

		Returns:
			pd.DataFrame: Filtered events indexed by sample_id, with additional
				columns 'rownum' (GRM row index) and 'start', 'stop', 'event'.

		Raises:
			ValueError: If the events file does not have exactly four columns
				or jackknife split index is zero (must be base-1).
		"""
		events = pd.read_csv(file, sep = '\t', header = 0)

		if events.shape[1] != 4:
			raise ValueError("There should be four columns in the events/outcome file")

		# Determine the default ID column name based on format
		default_id = 'IID' if binary_mode else 'sample_id'

		drop = None
		if jackknife is not None:
			id_col = sample_id if sample_id is not None else default_id
			event_split = events.sample(frac=1, random_state=seed).reset_index(drop=True)
			event_drop = np.array_split(event_split, int(jackknife[0]))
			if int(jackknife[1]) == 0:
				raise ValueError("Jackknife splits need to be base 1, i.e. 1-10 for 10 splits not 0-9")
			drop = event_drop[(int(jackknife[1])-1)]
			events = events[~events[id_col].isin(drop[id_col])]

		# Rename the sample ID column to 'sample_id' for uniform internal handling.
		# grm_names already has column 'sample_id'.
		if sample_id is not None:
			events = events.rename(columns = {sample_id:'sample_id'})
			if drop is not None:
				drop = drop.rename(columns = {sample_id:'sample_id'})
		else:
			events = events.rename(columns = {default_id:'sample_id'})
			if drop is not None:
				drop = drop.rename(columns = {default_id:'sample_id'})

		grm_names["rownum"] = grm_names.index
		grm_names["real_id"] = grm_names.sample_id

		grm_names = grm_names.set_index('sample_id')
		events = events.set_index('sample_id')
		events = pd.concat([grm_names, events], axis=1, join='inner')

		self.N = events.shape[0]

		events = events[(events.stop > events.start)]
		if events.shape[0] != self.N:
			logger.warning(f"{self.N - events.shape[0]} individuals dropped because start time is after end time")
			self.N = events.shape[0]

		events = events.sort_values("stop", ascending=True).reset_index(drop=True)

		first_case_stop = events.loc[events.event == 1, 'stop'].min()
		pre_case = (events.event == 0) & (events.stop <= first_case_stop)
		n_pre_case = pre_case.sum()
		if n_pre_case > 0:
			logger.warning(f"{n_pre_case} individuals were censored prior to the first observed event and were removed from the analysis")
			events = events[~pre_case].reset_index(drop=True)
			self.N = events.shape[0]

		events = events.rename(columns= {'real_id':'sample_id'})
		events = events.set_index('sample_id')
		return events

	def _reload_for_jackknife(self, split_index):
		"""Reload all data for joint jackknife std error processing. Steps:
			Re-reads the GRM from disk (not cached)
			Re-filters the phenotypes/covariates to exclude jackknife block split_index
			Re-inverts the GRM subset
		Updates: self.grm, self.events, self.times, self.names, self.N, self.fixed, and self.M

		Note:
			The caller (COXMM._fit_heritability_joint_jackknife) must call
			self.reset() after this method to re-initialise the working arrays
			(theta, risk_set, etc.) for the new sample size. reset() is defined
			on COXMM, not COXMM_IO, so it cannot be called from here.

		Args:
			split_index (int): 1-based index of the jackknife block to drop.
				Must be in range [1, self._joint_n_splits].
		"""
		original_M = self.M
		jackknife_arg = [self._joint_n_splits, split_index]

		# Re-filter events, dropping this split's jackknife block.
		# Process_events mutates its grm_names argument, so pass a fresh copy each call.
		events_df = self.process_events(
			self._joint_sample_id,
			self._joint_events_path,
			self._joint_grm_names.copy(),
			jackknife_arg,
			self._joint_seed,
			self._joint_binary_mode,
		)
		# self.N is updated as a side effect of process_events

		# Re-read and re-subset the GRM from disk, then invert.
		idx = events_df.rownum.to_numpy()
		grm = self._read_grm(
			self._joint_grm_path, self._joint_binary_mode, idx, self.grmN, self.N
		)
		self.grm = self._invert_grm(grm)

		# Derive numpy arrays from the filtered events DataFrame (mirrors __init__)
		self.times  = events_df[['start', 'stop']].to_numpy()
		self.names  = events_df.index.astype('str')
		self.events = events_df.event.to_numpy()

		# Reload and re-subset fixed effects for this split's sample set.
		# Note: GWAS is disallowed with --joint_jackknife, so we never insert
		# the SNP placeholder column that setup() adds when self.plink is set.
		if self._joint_fixed_path is not None:
			logger.warning("Minimal checks on issues with fixed effects input. Be careful")
			fixed_df, self.M = self._load_fixed(
				self._joint_fixed_path, self._joint_sample_id,
				self._joint_binary_mode, events_df.index
			)
			self.fixed = fixed_df.to_numpy()
			if self.M != original_M:
				raise ValueError(
					f"Fixed effects produced {self.M} columns for jackknife split {split_index} "
					f"but {original_M} columns for the original fit. A categorical variable may "
					f"be missing a level in this split. Consider using fewer jackknife splits."
				)
		else:
			self.fixed = None
			self.M = 0

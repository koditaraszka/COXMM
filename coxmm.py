"""
Cox Proportional Hazard Mixed Model (COXMM).

Implements a Cox mixed model to estimate heritability

Note: heritability is distinct from the variance component under this model
heritability = sigma2g/(sigma2g + pi^2/6)

Author: Kodi Taraszka
Email:  Kodi_Taraszka@dfci.harvard.edu
"""

import logging
import numpy as np
import pandas as pd
import pybobyqa
from scipy.stats import norm
# TODO: remove dead code
#from pandas_plink import read_plink
from coxmm_input import COXMM_IO

logger = logging.getLogger(__name__)

class COXMM(COXMM_IO):
	"""Cox proportional hazard mixed model.

		Inherits all I/O and data-loading from COXMM_IO. Call fit() after
		Call fit() to run optimization/estimate heritability
		Can run a GWAS but not recommended -- use SPACox or GATE		

	Attributes:
		results (pd.DataFrame or None): GWAS results, one row per SNP.
			None until _fit_gwas() is called.
		theta (np.ndarray): Parameter vector [beta (M,), u (N,)] where beta
			are fixed-effect coefficients and u are random effects.
		exp_eta (np.ndarray or None): exp(X*beta + u), shape (N,).
			None until first l_1() call; updated each Newton step thereafter.
		risk_set (np.ndarray): N×N uint8 upper-triangular matrix;
			entry [j, i] = 1 means individual i is in the risk set at event time j.
		A (np.ndarray or None): Length-N vector; nonzero only at case positions,
			equal to 1 / (sum of exp_eta over the risk set at that event time).
			None until first l_1_deriv() call.
		WB (np.ndarray or None): Length-N vector equal to W*M*A*1 (COXMEG notation). 
			None until first l_1_deriv() call.
		s (np.ndarray): Score vector, length M+N.
		V (np.ndarray): Information matrix, shape (M+N)×(M+N).
	"""

	def __init__(self):
		super().__init__()
		# From COXMM_IO, we have N, M, grm, times, events, fixed, output
		self.results = None  # populated by _fit_gwas(); None in heritability mode
		self.theta = np.zeros(self.N+self.M)
		self.update = np.zeros(self.N+self.M)
		self.exp_eta = None  # set by l_1() before first read
		self.risk_set = np.tril(np.ones((self.N, self.N), dtype=np.uint8)).T
		self.grm_u = None  # set by l_1() before first read
		self.loc = np.where(self.events==1)
		self.WB = None  # set by l_1_deriv() before first read
		self.A = None   # set by l_1_deriv() before first read
		self.s = np.zeros((self.M+self.N))
		self.V = np.zeros((self.M+self.N, self.M+self.N))
		self.R_j()
		self._precompute_cases()

	def _precompute_cases(self):
		"""Cache case indices and risk_set slices that are constant across Newton iterations."""
		self.cases = self.loc[0]						   # (n_cases,)
		self.rs_cases_T = self.risk_set.T[:, self.cases]  # (N, n_cases) uint8
		self.rs_cases   = self.risk_set[self.cases, :]	# (n_cases, N) uint8

	def fit(self):
		"""Run the model: estimate heritability (tau)
						can run GWAS -- but it really shouldn't be used (use SPACox/GATE/etc)

				Heritability -- optimizes the Laplace-approximated marginal log-likelihood over sigma2g/tau via BOBYQA
						writes results to the output file, and returns a dict with the estimates.

				GWAS -- (if you ignore all warnings) iterates over SNPs and estimates SNP effect for fixed sigma2g/tau
						writes a results CSV and returns the results DataFrame.

		Returns:
			dict: In heritability mode (no --joint_jackknife):
				{
				  "tau":   float,		# genetic variance component (sigma²_g)
				  "h2":	float,		# heritability on liability scale
				  "N":	 int,		  # number of analysis samples
				  "cases": int,		  # number of cases
				  "betas": list[float],  # fixed-effect estimates (empty if M=0)
				  "ses":   list[float],  # corresponding standard errors
				}
			list[dict]: In joint jackknife mode (--joint_jackknife) — one dict
				per run (original + all replicates) in the same format as above.
			pd.DataFrame: In GWAS mode — one row per SNP with columns
				snp, chrom, pos, a0, a1, maf, geno, n, beta, se, pval.

		Side effects:
			Writes results to self.output. 
			Heritability output always includes a 'type' column: 
				'original' for a full-sample run,
				'jackknife_k' when --jackknife or --joint_jackknife is used.
		"""
		if self.gwas is None:
			if self.joint_jackknife is not None:
				return self._fit_heritability_joint_jackknife()
			return self._fit_heritability(type_label=self.jackknife_type_label)
		else:
			return self._fit_gwas()

	def _fit_heritability(self, type_label='original', write_header=True):
		"""Estimate sigma2g/tau by maximising the Laplace marginal log-likelihood.

		Args:
			type_label (str): Value written to the 'type' column of the output
				file. Use 'original' for the full-sample fit, 'jackknife_k' for
				the k-th jackknife replicate.
			write_header (bool): If True, open the output file in write mode
				and write the header line first. If False, open in append mode
				and write only the data line (for jackknife replicates after
				the first).

		Returns:
			dict: Keys tau, h2, N, cases, betas, ses.
		"""
		# just chosen (half way point when we considered sigma2g == h2
		# we do find a local optimum, so it's possible this starting point impacts the estimate
		# may be worthwhile to exhaustively test, but it didn't seem to affect simulations
		tau = 0.5
		# 1e-4 is the lower bound because we want the order of magnitude to be reasonable between upper and lower
		# 1e-4 should be treated as 0
		# 5 would result in a heritability estimate of 0.75 which is pretty high
		# to get to 1 we essentially need pi^2/6 to approach 0 (relatively)
		soln = pybobyqa.solve(self.marg_loglike, x0=[tau], bounds=([1e-4], [5]))
		tau = soln.x[0]
		her = tau / (tau + (np.pi * np.pi) / 6)
		n_cases = self.loc[0].shape[0]

		# report the tau/sigma2g varience component estimate
		# also report heritability = tau/(tau + pi^2/6)
		mode = 'w' if write_header else 'a'
		with open(self.output, mode) as output:
			if write_header:
				output.write("sigma2g h2 N Cases type\n")
			output.write(f"{tau} {her} {self.N} {n_cases} {type_label}\n")

		logger.info(f"Heritability estimate: {tau}")
		logger.info(f"The sample size is {self.N}")
		logger.info(f"The number of cases is {n_cases}")

		V_inv = np.linalg.inv(self.V)
		betas, ses = [], []
		if self.M > 0:
			logger.info("Covariate effect sizes and std errors:")
			for var in range(0, self.M):
				b = self.theta[var]
				se = np.sqrt(V_inv[var, var])
				betas.append(b)
				ses.append(se)
				logger.info(f"Beta: {b}  SE: {se}")

		return {"tau": tau, "h2": her, "N": self.N, "cases": n_cases,
				"betas": betas, "ses": ses}

	def _fit_heritability_joint_jackknife(self):
		"""Run the original heritability fit then all jackknife replicates.

		For each split, the GRM is re-read from disk and the working arrays re-initialised

		All results are written to self.output: one header row followed by one
		data row per run (original + n_splits jackknife replicates), each with
		a 'type' column ('original', 'jackknife_1', ..., 'jackknife_n').
		TODO: compute the std error estimate internally (currently done externally/afterwards)

		Returns:
			list[dict]: One dict per run in the same format as _fit_heritability,
				in order [original, jackknife_1, ..., jackknife_n].
		"""
		results = []

		# Original fit on the full sample; writes header + first data line
		logger.info("Running original (full-sample) heritability fit")
		result = self._fit_heritability(type_label='original', write_header=True)
		results.append(result)

		n_splits = self._joint_n_splits
		for split in range(1, n_splits + 1):
			logger.info(f"Running joint jackknife split {split}/{n_splits}")
			# Re-read GRM from disk, re-filter events, update self.grm,
			# self.events, self.times, self.names, self.N, self.fixed, self.M
			self._reload_for_jackknife(split)
			# Re-initialise all working arrays (theta, risk_set, etc.) for new N
			self.reset()
			result = self._fit_heritability(
				type_label=f'jackknife_{split}',
				write_header=False,
			)
			results.append(result)

		return results

	def _fit_gwas(self):
		"""Fit the model at fixed tau for every SNP in the plink file.

		Returns:
			pd.DataFrame: One row per SNP with columns
				snp, chrom, pos, a0, a1, maf, geno, n, beta, se, pval.
		"""
		(bim, fam, bed) = read_plink(self.plink, verbose=False)
		fam = fam.rename(columns={'fid': 'sample_id'})
		fam = fam.set_index('sample_id')
		fam.index = fam.index.astype('str')
		fam = fam.reindex(index=self.names)
		missing = fam[fam['iid'].isna()]
		if missing.shape[0] > 0:
			raise ValueError("There are phenotyped individuals who are not genotyped")
		self.results = bim[['snp', 'chrom', 'pos', 'a0', 'a1']].copy()
		self.results["maf"] = np.nan
		self.results["geno"] = np.nan
		self.results["n"] = np.nan
		self.results["beta"] = np.nan
		self.results["se"] = np.nan
		self.results["pval"] = np.nan
		orig_grm = self.grm.copy()
		if self.fixed is not None:
			orig_fixed = self.fixed.copy()
		orig_events = self.events.copy()
		orig_times = self.times.copy()
		orig_n = self.N
		for index in range(0, bed.shape[0]):
			# 10 is small but if you're doing a lot of SNP... why?!
			if index % 10:
				logger.info(f"Now at SNP {index}")
			self.fixed = orig_fixed.copy()
			self.grm = orig_grm.copy()
			self.events = orig_events.copy()
			self.times = orig_times.copy()
			self.N = orig_n
			snp = np.asarray(bed[index,])
			self.fixed[:,0] = snp[fam.i.to_numpy()]
			missing = np.argwhere(np.isnan(self.fixed[:,0])).flatten()
			if missing.shape[0] > 0:
				geno = missing.shape[0]/self.N
				self.fixed = np.delete(self.fixed, missing, axis=0)
				self.grm = np.delete(orig_grm, missing, axis=0)
				self.grm = np.delete(self.grm, missing, axis=1)
				self.events = np.delete(orig_events, missing)
				self.times = np.delete(orig_times, missing, axis=0)
			else:
				geno = 0

			mean = np.mean(self.fixed[:, 0])
			if (mean / 2 > 0.5):
				self.fixed[:,0][self.fixed[:,0] == 0] = 3
				self.fixed[:,0][self.fixed[:,0] == 2] = 0
				self.fixed[:,0][self.fixed[:,0] == 3] = 2
				mean = np.mean(self.fixed[:, 0])
				a1 = self.results.loc[index, "a1"]
				self.results.loc[index, "a1"] = self.results.loc[index, "a0"]
				self.results.loc[index, "a0"] = a1

			stderr = np.std(self.fixed[:, 0])
			self.reset()
			if self.center:
				self.fixed[:,0] = (self.fixed[:,0] - mean)/stderr
			likelihood = self.marg_loglike(self.gwas)
			V = np.linalg.inv(self.V)
			self.results.loc[index, "maf"] = round(mean/2, 3)
			self.results.loc[index, "geno"] = round(geno, 3)
			self.results.loc[index, "n"] = self.N
			self.results.loc[index, "beta"] = self.theta[0]
			self.results.loc[index, "se"] = np.sqrt(V[0,0])
			self.results.loc[index, "pval"] = norm.sf(abs(self.theta[0]/np.sqrt(V[0,0])))*2
		self.results.to_csv(self.output, index=False, sep=' ')
		return self.results

	def reset(self):
		"""Re-initialise all working arrays after the GRM or event data changes.

		Called once per SNP in GWAS mode after missingness subsetting.
		"""
		self.N = self.grm.shape[0]
		self.theta = np.zeros(self.N+self.M)
		self.update = np.zeros(self.N+self.M)
		self.exp_eta = None  # set by l_1() before first read
		self.risk_set = np.tril(np.ones((self.N, self.N), dtype=np.uint8)).T
		self.grm_u = None  # set by l_1() before first read
		self.loc = np.where(self.events==1)
		self.WB = None  # set by l_1_deriv() before first read
		self.A = None   # set by l_1_deriv() before first read
		self.s = np.zeros((self.M+self.N))
		self.V = np.zeros((self.M+self.N, self.M+self.N))
		self.R_j()
		self._precompute_cases()

	def R_j(self):
		"""Adjust self.risk_set for delayed entry and tied event times.

		self.risk_set is an upper-triangular N×N matrix (ordered by stop time)
			entry [j, i] = 1 means individual i is at risk when individual j has their event. 

		Two corrections are applied:
			- Late entry: if individual i enters after individual j's stop time,
		  		i cannot be in j's risk set and those entries are zeroed.
			- Tied stop times: individuals with the same stop time are mutually
		  		in each other's risk sets regardless of row order
		"""
		if np.unique(self.times[:,0]).shape[0] > 1:
			min_start = np.min(self.times[:,0])
			late_start = np.where(self.times[:,0]!=min_start)[0]
			for who in late_start:
				start_time = self.times[who,0]
				for before in range(0,who):
					if start_time <= self.times[before,0]:
						self.risk_set[0:before,who] = 0
						break

		# fixing duplicated stopping times
		unq, count = np.unique(self.times[:,1], return_counts=True)
		dups = np.where(count>1)[0]
		if dups.shape[0] > 0:
			for stop_time in unq[dups]:
				same_time = np.where(self.times[:,1]==stop_time)[0]
				min_pos = np.min(same_time)
				max_pos = np.max(same_time)
				for who in same_time:
					self.risk_set[min_pos:(max_pos+1),who] = 1

	def l_1(self, tau):
		"""Penalised partial log-likelihood at the current theta/sigma2g

		Computes the Cox partial log-likelihood minus the quadratic random-effect
		penalty u^T * GRM^{-1} * u / (2*tau). Also updates self.exp_eta and
		self.grm_u as side effects used by l_1_deriv.

		Args:
			tau: Genetic variance component (sigma²_g). Must be > 0.

		Returns:
			Scalar penalised partial log-likelihood.
		"""
		# eta = Xb + Zu, theta = [beta, u]
		if self.M > 0:
			self.exp_eta = np.exp(np.matmul(self.fixed, self.theta[0:self.M]) + self.theta[self.M:(self.M+self.N)])
		else:
			self.exp_eta = np.exp(self.theta)

		risk_eta = np.multiply(self.risk_set[self.loc,:][0], self.exp_eta)
		result = np.sum(np.log(self.exp_eta[self.loc])) - np.sum(np.log(np.sum(risk_eta,axis=1)))

		self.grm_u = np.matmul(self.grm, self.theta[self.M:(self.M+self.N)])
		result -= 1/(2*tau)*(np.matmul(self.theta[self.M:(self.M+self.N)].T, self.grm_u))
		return result

	def l_1_deriv(self, tau):
		"""Score vector and information matrix for theta at fixed tau.

				Populates self.s (score) and self.V (information matrix) in place. 

				Args:
						tau: Genetic variance component (sigma2g). Must be > 0.

				Side effects:
						Updates self.A, self.WB, self.s, self.V.
				"""
		# A = diag(D) diag^-1(M^TW1) with only cases (events==1) have nonzero A.
				# row sums of MTW via matmul to avoid full N×N MTW matrix.	
		MTW_rowsum = self.risk_set.astype(np.float64) @ self.exp_eta  # (N,)
		A_vals = 1.0 / MTW_rowsum[self.cases]  # (n_cases,); events[cases] are all 1
		self.A = np.zeros(self.N)
		self.A[self.cases] = A_vals

		# MTW at case rows: (n_cases, N). Needed for H and (if M>0) V[two].
		MTW_cases = self.rs_cases.astype(np.float64) * self.exp_eta  # (n_cases, N)

		# WB = WMA1 — sparse: sum only over nonzero (case) columns of risk_set.T
		self.WB = self.exp_eta * (self.rs_cases_T.astype(np.float64) * A_vals).sum(axis=1)

		# setting score function s with parts [one, two]
		# s[two] -- d - WMA1 - (Sigma^-1 gamma) / tau (save sigma^-1 gamma / tau for now)
		self.s[self.M:(self.M+self.N)] = self.events - self.WB

		# H = WB - QQ^T = WB - WMA^2M^TW.
		# The left factor has nonzero columns only at case positions; use those
		# n_cases columns for an O(N²×n_cases) matmul instead of O(N³).
		# step2_cols[:,k] = exp_eta[cases[k]] * rs_cases_T[:,k] * A_vals[k]²
		step2_cols = self.rs_cases_T.astype(np.float64) * (self.exp_eta[self.cases] * np.square(A_vals))
		H = step2_cols @ MTW_cases  # (N, n_cases) @ (n_cases, N) = (N, N)

		#V[four] -- H + sigma^-1/tau: always exists since we're looking at random effect
		# Use fill_diagonal to add WB to diagonal; avoids allocating N×N np.diag(WB).
		self.V[self.M:(self.M+self.N), self.M:(self.M+self.N)] = self.grm/tau - H
		np.fill_diagonal(
			self.V[self.M:(self.M+self.N), self.M:(self.M+self.N)],
			self.V[self.M:(self.M+self.N), self.M:(self.M+self.N)].diagonal() + self.WB
		)
		# setting information matrix V with quadrants [[one, two], [three, four]]
		# one, two, three only exist if there were fixed effect/covariates
		if self.M > 0:
			#V[two] -- X^TH
			# left_full[:,k] = 0 when A[k]=0 (k not a case), so only the n_cases
			# case columns are nonzero. Compute them directly:
			#   left_cases[m,i] = (fixed.T * exp_eta)[m,:] @ rs_cases_T[:,i] * A_vals[i]²
			#   Avoids the full (N,N)@(N,M) matmul and intermediate (M,N) allocation.
			left_cases = np.multiply(self.fixed.T, self.exp_eta) @ self.rs_cases_T.astype(np.float64) * np.square(A_vals)
			self.V[0:self.M, self.M:(self.M+self.N)] = np.multiply(self.fixed.T, self.WB) - np.matmul(left_cases, MTW_cases)
			#V[one] -- X^THX = (V[two]X)
			self.V[0:self.M, 0:self.M] = np.matmul(self.V[0:self.M, self.M:(self.M+self.N)], self.fixed)
			#V[three] -- HX = (X^TH)^T = V[two]^T
			self.V[self.M:(self.M+self.N), 0:self.M] = self.V[0:self.M, self.M:(self.M+self.N)].T
			#s[one] -- X^T(d - WMA1)
			self.s[0:self.M] = np.matmul(self.fixed.T, self.s[self.M:(self.M+self.N)])
		#update s[two] after setting s[one]
		self.s[self.M:(self.M+self.N)] = self.s[self.M:(self.M+self.N)] - self.grm_u/tau

	def marg_loglike(self, tau):
		"""Laplace-approximated marginal log-likelihood for sigma2g/tau.

				Runs Newton–Raphson on theta (with step-halving if the likelihood
				decreases) until convergence, then evaluates the Laplace approximation
				to the marginal likelihood by computing the log-determinant of the
				observed information J at the mode. This scalar is minimised by BOBYQA
				to obtain the MLE for tau.

		Args:
			tau: Genetic variance component (sigma2g). Must be > 0.

		Returns:
			Scalar Laplace marginal log-likelihood (to be minimised by BOBYQA).

		Side effects:
			Converges self.theta
			Updates self.V, self.WB, self.A, self.exp_eta
		"""
		# these values could be adjusted, were chosen as reasonable choices
		eps = 1e-6
		run = 0
		max_runs = 200
		loglike = 0
		update_loglike = self.l_1(tau)
		diff_loglike = 0 #update_loglike - loglike
		eps_s = 0 #eps*(-1)*loglike
		while (run == 0 or diff_loglike > eps) and run < max_runs:
			damp = 1
			loglike = update_loglike
			self.l_1_deriv(tau)
			self.update = np.linalg.solve(self.V, self.s)
			self.theta[self.M:(self.M+self.N)] = self.theta[self.M:(self.M+self.N)] + self.update[self.M:(self.M+self.N)]
			if self.M > 0:
				self.theta[0:self.M] = self.theta[0:self.M] + self.update[0:self.M]

			update_loglike = self.l_1(tau)
			eps_s = eps*(-1)*loglike
			diff_loglike = update_loglike - loglike
			while diff_loglike < -eps_s:
				damp = damp/2
				if damp < 1e-2:
					logger.warning("The optimization of PPL may not converge.")
					diff_loglike = 0
					break
				self.update = damp*self.update
				self.theta[self.M:(self.M+self.N)] = self.theta[self.M:(self.M+self.N)] - self.update[self.M:(self.M+self.N)]
				if self.M > 0:
					self.theta[0:self.M] = self.theta[0:self.M] - self.update[0:self.M]
				update_loglike = self.l_1(tau)
				diff_loglike = update_loglike - loglike

			run += 1

		if run == max_runs:
			logger.warning(f"The PPL ran for the maximum number of iterations ({max_runs}). It probably didn't converge")

		A = np.cumsum(np.multiply(self.A[self.loc], self.A[self.loc]))
		whoStarts = np.zeros(self.N)
		whoStarts[self.loc] = 1
		whoStarts = np.cumsum(whoStarts) - 1
		ws = whoStarts.astype(int)
		starter_matrix = np.minimum.outer(ws, ws)
		J = np.outer(self.exp_eta, self.exp_eta) * A[starter_matrix]

		J = self.grm/tau - J
		np.fill_diagonal(J, J.diagonal() + self.WB)
		_, log_det = np.linalg.slogdet(J)
		return self.N*np.log(tau) + log_det - 2*update_loglike

if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(levelname)s: %(message)s",
	)
	COXMM().fit()

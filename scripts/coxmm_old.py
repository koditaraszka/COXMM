'''
Author: Kodi Taraszka
Email: Kodi_Taraszka@dfci.harvard.edu
This is prelim code for cox proportional hazard mixed model components (CoxMM)
'''

#TODO: double check that left censoring and work correctly
#TODO: classier handling of output/write to file

import numpy as np
import pandas as pd
import pybobyqa
from scipy.stats import norm
#from pandas_plink import read_plink
from coxmm_input_old import IO

class COXMM(IO):

	def __init__(self):
		super().__init__()
		#From IO, we have N, M, grm, times, events, fixed, output
		self.theta = np.zeros(self.N+self.M)
		self.update = np.zeros(self.N+self.M)
		self.exp_eta = np.zeros(self.N)
		self.risk_set = np.tril(np.ones((self.N, self.N))).T
		self.grm_u = np.zeros(self.N)
		self.loc = np.where(self.events==1)
		self.MTW = np.zeros((self.N, self.N))
		self.WB = np.zeros(self.N)
		self.A = np.identity(self.N)
		self.H = np.identity(self.N)
		self.s = np.zeros((self.M+self.N))
		self.V = np.zeros((self.M+self.N, self.M+self.N))
		self.R_j()
		if self.gwas is None:
			tau = 0.5
			soln = pybobyqa.solve(self.marg_loglike, x0 = [tau], bounds = ([1e-4], [5]))
			tau = soln.x[0]
			# dropped this definition of se due to being overly conservative
			# instead us block jackknifing
			#se = np.sqrt(1/self.second_deriv(tau))
			her = tau/(tau + (math.pi*math.pi)/6)
			output = open(self.output, 'w')
			#report the untransformed heritability estimate, but work indicates tau/(tau + pi^2/6) to be the right correction
			output.writelines(["sigma2g h2 N Cases\n", 
				str(tau) + " " + str(her) + " " + str(self.N) + " " + str(self.loc[0].shape[0]) + "\n"])

			print("Heritability estimate: " + str(tau))
			print("The sample size is " + str(self.N))
			print("The number of cases is " + str(self.loc[0].shape[0]))  	
			print("Covariate effect sizes and std errors:")
			V = np.linalg.inv(self.V)
			for var in range(0,self.M):
				print("Beta: " + str(self.theta[var]) + " SE: " + str(np.sqrt(V[var,var])))
		else:
			(bim, fam, bed) = read_plink(self.plink, verbose=False)
			fam = fam.rename(columns = {'fid':'sample_id'})
			fam = fam.set_index('sample_id')
			fam.index = fam.index.astype('str')
			fam = fam.reindex(index = self.names)
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
			origGRM = self.grm.copy()
			if self.fixed is not None:
				origFixed = self.fixed.copy()
			origEvents = self.events.copy()
			origTimes = self.times.copy()
			origN = self.N
			for index in range(0, 4): #bed.shape[0]):
				if index % 10:
					print("Now at the " + str(index) + " SNP")
				self.fixed = origFixed.copy()
				self.grm = origGRM.copy()
				self.events = origEvents.copy()
				self.times = origTimes.copy()
				self.N = origN
				snp = np.asarray(bed[index,])
				self.fixed[:,0] = snp[fam.i.to_numpy()]
				missing = np.argwhere(np.isnan(self.fixed[:,0])).flatten()
				if missing.shape[0] > 0:
					geno = missing.shape[0]/self.N
					self.fixed = np.delete(self.fixed, missing, axis=0)
					self.grm = np.delete(origGRM, missing, axis=0)
					self.grm = np.delete(self.grm, missing, axis=1)
					self.events = np.delete(origEvents, missing)
					self.times = np.delete(origTimes, missing, axis=0)
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
				self.results.loc[index, "maf"] = round(mean/2,3)
				self.results.loc[index, "geno"] = round(geno,3)
				self.results.loc[index, "n"] = self.N
				self.results.loc[index, "beta"] = self.theta[0]
				self.results.loc[index, "se"] = np.sqrt(V[0,0])
				self.results.loc[index, "pval"] = norm.sf(abs(self.theta[0]/np.sqrt(V[0,0])))*2
				#break	
			#print(self.results.loc[index,])
			self.results.to_csv(self.output, index=False, sep=' ')

	# this method doesn't work and gives an underestimate for the standard error
	#def second_deriv(self, tau):
	#	ut_grm_u = np.matmul(self.theta[self.M:(self.M+self.N)].T, np.matmul(self.grm, self.theta[self.M:(self.M+self.N)]))
	#	return (((0.5*self.N - 0.5)*tau - ut_grm_u)/(tau ** 3))

	# who is at risk when the jth individual has their event
	def reset(self):
		self.N = self.grm.shape[0]
		self.theta = np.zeros(self.N+self.M)
		self.update = np.zeros(self.N+self.M)
		self.exp_eta = np.zeros(self.N)
		self.risk_set = np.tril(np.ones((self.N, self.N))).T
		self.grm_u = np.zeros(self.N)
		self.loc = np.where(self.events==1)
		self.MTW = np.zeros((self.N, self.N))
		self.WB = np.zeros(self.N)
		self.A = np.identity(self.N)
		self.H = np.identity(self.N)
		self.s = np.zeros((self.M+self.N))
		self.V = np.zeros((self.M+self.N, self.M+self.N))
		self.R_j()


	def R_j(self):
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

	# l_1 as defined in equations 2 in COXMEG paper (mostly their notation)
	def l_1(self, tau):
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

	# l_2 as defined in Equation 3 in COXMEG paper (mostly their notation)
	def l_1_deriv(self, tau):
		# W = diag(exp(eta)) but working withexp(eta) explicitly (self.exp_eta)
		self.MTW = np.multiply(self.risk_set, self.exp_eta)
		# A = diag(D) diag^-1(M^TW1)
		self.A = np.multiply(self.events, 1/np.sum(self.MTW, axis=1))
		# NOTE: WB = WMA1
		self.WB = np.multiply(self.exp_eta, np.sum(np.multiply(self.risk_set.T, self.A), axis = 1))
		
		# setting score function s with parts [one, two]
		# s[two] -- d - WMA1 - (Sigma^-1 gamma) / tau (save sigma^-1 gamma / tau for now)
		self.s[self.M:(self.M+self.N)] = self.events - self.WB

		# H = WB - QQ^T = WB - WMA^2M^TW
		self.H = np.matmul(np.multiply(np.multiply(self.exp_eta, self.risk_set.T), np.square(self.A)), self.MTW)
	
		#V[four] -- H + sigma^-1/tau: always exists since we're looking at random effect		 
		self.V[self.M:(self.M+self.N), self.M:(self.M+self.N)] = np.add((np.diag(self.WB) - self.H), (self.grm/tau))
		# setting information matrix V with quadrants [[one, two], [three, four]]
		# one, two, three only exist if there were fixed effect/covariates
		if self.M > 0:
			#V[two] -- X^TH
			self.V[0:self.M, self.M:(self.M+self.N)] = np.multiply(self.fixed.T, self.WB) - np.matmul(np.multiply(np.dot(self.risk_set, np.multiply(self.fixed.T, self.exp_eta).T).T, np.square(self.A)), self.MTW)
			#V[one] -- X^THX = (V[two]X)
			self.V[0:self.M, 0:self.M] = np.matmul(self.V[0:self.M, self.M:(self.M+self.N)], self.fixed)
			#V[three] -- HX = (X^TH)^T = V[two]^T
			self.V[self.M:(self.M+self.N), 0:self.M] = self.V[0:self.M, self.M:(self.M+self.N)].T
			#s[one] -- X^T(d - WMA1)
			self.s[0:self.M] = np.matmul(self.fixed.T, self.s[self.M:(self.M+self.N)])
		#update s[two] after setting s[one]
		self.s[self.M:(self.M+self.N)] = self.s[self.M:(self.M+self.N)] - self.grm_u/tau
	
	# treat l_2 as a marginal log-likelihood to get estimate for tau
	def marg_loglike(self, tau):	
		# initialize	
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
			# TODO: add comments to describe
			eps_s = eps*(-1)*loglike
			diff_loglike = update_loglike - loglike
			# TODO: copy-paste, but should actually know why
			while diff_loglike < -eps_s:
				damp = damp/2
				if damp < 1e-2:
					print("The optimization of PPL may not converge.")
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
			print("The PPL ran for the maximum number of iterations (" + str(max_runs) + "). It probably didn't converge")

		A = np.cumsum(np.multiply(self.A[self.loc], self.A[self.loc]))
		whoStarts = np.zeros(self.N)
		whoStarts [self.loc] = 1	
		whoStarts = np.cumsum(whoStarts) - 1 
		J = np.zeros((self.N, self.N))
		for i in range(0,self.N):
			for j in range(i, self.N):
				starter = int(min(whoStarts[i], whoStarts[j]))
				J[i,j] = self.exp_eta[i]*self.exp_eta[j]*A[starter]
				J[j,i] = J[i,j]

		J = self.grm/tau - J + np.diag(self.WB)
		_, log_det = np.linalg.slogdet(J)
		return self.N*np.log(tau) + log_det - 2*update_loglike

if __name__=="__main__":
	COXMM()

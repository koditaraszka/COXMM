'''
Author: Kodi Taraszka
Email: Kodi_Taraszka@dfci.harvard.edu
This is prelim code for cox proportional hazard mulitple mixed model components (CoxPHMM)
'''

#TODO: double check that left censoring and right censoring work correctly
#TODO: classier handling of output/write to file
#TODO: estimate standard error of tau
#TODO: offer a correction to the estimated tau, it should be 2*tau/(1+tau)

import numpy as np
import pandas as pd
import pybobyqa
from input import IO
import os.path as path

class COXPHMM(IO):

	def __init__(self):
		super().__init__()
		#From IO, we have N, M, grm, times, events, fixed, output
		self.beta = None
		if self.M > 0:
			self.beta = np.memmap(path.join(self.temp, 'beta.dat'), dtype='float64', mode='w+', shape=(self.M))	
		self.u = np.memmap(path.join(self.temp, 'u.dat'), dtype='float64', mode='w+', shape=(self.N))
		self.exp_eta = np.memmap(path.join(self.temp, 'exp_eta.dat'), dtype='float64', mode='w+', shape=(self.N))
		self.risk_set = np.memmap(path.join(self.temp, 'risk_set.dat'), dtype='float64', mode='w+', shape=(self.N, self.N))
		self.risk_set[:,:] = np.tril(np.ones((self.N, self.N))).T
		self.risk_eta = np.memmap(path.join(self.temp, 'risk_eta.dat'), dtype='float64', mode='w+', shape=(self.N,self.N))
		self.grm_u_tau = np.memmap(path.join(self.temp, 'grm_utau.dat'), dtype='float64', mode='w+', shape=(self.N))
		self.loc = np.memmap(path.join(self.temp, 'loc.dat'), dtype='int64', mode='w+', shape=(np.where(self.events==1)[0].shape[0]))
		self.loc[:] = np.where(self.events==1)[0]
		self.ONE = np.memmap(path.join(self.temp, 'one.dat'), dtype='float64', mode='w+', shape=(self.N))
		self.MTW = np.memmap(path.join(self.temp, 'MTW.dat'), dtype='float64', mode='w+', shape=(self.N, self.N))
		self.WB = np.memmap(path.join(self.temp, 'WB.dat'), dtype='float64', mode='w+', shape=(self.N))
		self.A = np.memmap(path.join(self.temp, 'A.dat'), dtype='float64', mode='w+', shape=(self.N))
		self.B = np.memmap(path.join(self.temp, 'B.dat'), dtype='float64', mode='w+', shape=(self.N))
		self.H = np.memmap(path.join(self.temp, 'H.dat'), dtype='float64', mode='w+', shape=(self.N, self.N))
		self.s = np.memmap(path.join(self.temp, 's.dat'), dtype='float64', mode='w+', shape=(self.M+self.N))
		self.V = np.memmap(path.join(self.temp, 'V.dat'), dtype='float64', mode='w+', shape=(self.M+self.N, self.M+self.N))
		self.theta = np.memmap(path.join(self.temp, 'theta.dat'), dtype='float64', mode='w+', shape=(self.M+self.N))	
		tau = 0.5
		soln = pybobyqa.solve(self.marg_loglike, x0 = [tau], bounds = ([1e-4], [5]))
		print(soln)

	# for each non-censored person, who is at risk at that time
	# don't fix right censoring now do it inline
	def R_j(self):
		# fixing left censoring, TODO vet the approach
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
				same_time = np.where(self.times[:,1]==stop_time)
				min_pos = np.min(who)
				max_pos = np.max(who)
				# risk set is currently a
				for who in same_time:
					self.risk_set[min_pos:(max_pos+1),who] = 1

	# l_1 as defined in equations 2 in COXMEG paper (mostly their notation)
	def l_1(self, tau):
		# eta = Xb + Zu
		# works even when no fixed effects, just become 0
		if self.M > 0:
			self.exp_eta[:] = np.exp(np.matmul(self.fixed, self.beta) + self.u)
		else:
			self.exp_eta[:] = np.exp(self.u)

		self.risk_eta[:,:] = np.multiply(self.risk_set[self.loc,:][0], self.exp_eta)
		# NOTE: changing the order of tau division may impact likelihood precision
		self.grm_u_tau[:] = np.matmul(self.grm, self.u)/tau
		return (np.sum(np.log(self.exp_eta[self.loc])) - np.sum(np.log(np.sum(self.risk_eta,axis=1))) \
				 - 0.5*(np.matmul(self.u.T, self.grm_u_tau)))	

	# l_2 as defined in Equation 3 in COXMEG paper (mostly their notation)
	def l_1_deriv(self, tau):
		# W = diag(exp(eta)) but working with exp(eta) explicitly (self.exp_eta)
		self.MTW[:,:] = np.multiply(self.risk_set, self.exp_eta)
		# A = diag(D) diag^-1(M^TW1)
		self.A[:] = np.multiply(self.events, 1/np.sum(self.MTW, axis=1))
		# B = diag(MA1)
		self.B[:] = np.sum(np.multiply(self.risk_set.T, self.A), axis = 1)
		# H = WB - QQ^T = WB - WMA^2M^TW
		# NOTE: WB = WMA1
		self.WB[:] = np.multiply(self.exp_eta, self.B)
		self.H[:,:] = np.diag(self.WB) - np.matmul(np.multiply(np.multiply(self.exp_eta, self.risk_set.T), np.square(self.A)), self.MTW)
		# setting information matrix V with quadrants [[one, two], [three, four]]
		#V[four] -- H + sigma^-1/tau: always exists since we're looking at random effect		 
		self.V[self.M:(self.M+self.N), self.M:(self.M+self.N)] = np.add(self.H, (self.grm/tau))
		# setting score function s with parts [one, two]
		# s[two] -- d - WMA1 - (Sigma^-1 gamma) / tau (save sigma^-1 gamma / tau for now)
		self.s[self.M:(self.M+self.N)] = self.events - self.WB
		# one, two, three only exist if there were fixed effect/covariates
		if self.M > 0:
			#V[two] -- X^TH
			self.V[0:self.M, self.M:(self.M+self.N)] = np.matmul(self.fixed.T, self.H)
			#V[one] -- X^THX = (V[two]X)
			self.V[0:self.M, 0:self.M] = np.matmul(self.V[0:self.M, self.M:(self.M+self.N)], self.fixed)
			#V[three] -- HX = (X^TH)^T = V[two]^T
			self.V[self.M:(self.M+self.N), 0:self.M] = self.V[0:self.M, self.M:(self.M+self.N)].T
			#s[one] -- X^T(d - WMA1)
			self.s[0:self.M] = np.matmul(self.fixed.T, self.s[self.M:(self.M+self.N)])
		self.s[self.M:(self.M+self.N)] = self.s[self.M:(self.M+self.N)] - self.grm_u_tau

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
			self.theta[:] = np.linalg.solve(self.V, self.s)
			self.u[:] = self.u + self.theta[self.M:(self.M+self.N)]	
			if self.M > 0:
				self.beta[:] = self.beta + self.theta[0:self.M]

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
				self.theta[:] = damp*self.theta
				self.u[:] = self.u - self.theta[self.M:(self.M+self.N)]
				if self.M > 0:
					self.beta[:] = self.beta - self.theta[0:self.M]
				update_loglike = self.l_1(tau)
				diff_loglike = update_loglike - loglike

			run += 1
		
		if run == max_runs:
			print("The PPL ran for the maximum number of iterations (" + str(max_runs) + "). It probably didn't converge")
			
		_, log_det = np.linalg.slogdet(self.V)
		return (self.N*np.log(tau) + log_det + (-2)*update_loglike)


if __name__=="__main__":
	COXPHMM()

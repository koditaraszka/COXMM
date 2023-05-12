'''
Author: Kodi Taraszka
Email: Kodi_Taraszka@dfci.harvard.edu
This is prelim code for cox proportional hazard mulitple mixed model components (CoxPHself.Mself.M)
'''

#TODO: double check that left censoring and right censoring work correctly
#TODO: implement Breslow tie breaker, now we add noise to them :(
#TODO: classier handling of output/write to file
#TODO: estimate standard error of tau
#TODO: offer a correction to the estimated tau

import numpy as np
import pandas as pd
import pybobyqa
import os.path as path
from input import IO

class COXPHMM(IO):

	def __init__(self):
		super().__init__()
		self.R_j()
		
		tau = 0.5 #initized value, may allow user input
		soln = pybobyqa.solve(self.marg_loglike, x0 = [tau], bounds = ([1e-4], [5]))
		tau_hat = soln.x[0]
		dist = np.zeros(1000)
		for i in range(0,1000):
			soln = pybobyqa.solve(self.marg_stderror, x0 = [tau], bounds = ([1e-4], [5]))
			dist[i] = soln.x[0]

	# who is at risk when the jth individual has their event
	# TODO actually do the work for the risk set
	def R_j(self):
		print('do nothing for now')
		#risk_set = np.memmap(path.join(self.temp, self.risk_set), dtype='float64', mode='r+', shape=(self.N,self.N))
		#del risk_set

	# l_1 as defined in equations 2 in COXself.MEG paper (mostly their notation)
	def l_1(self, tau):
		# eta = Xb + Zu, theta =[beta, u]
		exp_eta = np.memmap(path.join(self.temp, self.exp_eta), dtype='float64', mode='r+', shape=(self.N))
		theta = np.memmap(path.join(self.temp, self.theta), dtype='float64', mode='r+', shape=(self.N+self.M))
		if self.M > 0:
			fixed = np.memmap(path.join(self.temp, self.fixed), dtype='float64', mode='r', shape=(self.N, self.M))
			exp_eta[:] = np.exp(np.matmul(fixed, theta[0:self.M]) + theta[self.M:(self.M+self.N)])
			del fixed
		else:
			exp_eta[:] = np.exp(theta)
		
		risk_set = np.memmap(path.join(self.temp, self.risk_set), dtype='float64', mode='r', shape=(self.N,self.N))
		loc = np.memmap(path.join(self.temp, self.loc), dtype='int64', mode='r', shape=(self.uncensored))
		risk_eta = np.multiply(risk_set[loc,:], exp_eta)
		del risk_set

		result = np.sum(np.log(exp_eta[loc])) - np.sum(np.log(np.sum(risk_eta,axis=1)))
		del exp_eta, loc, risk_eta 
		
		grm = np.memmap(path.join(self.temp, self.grm), dtype='float64', mode='r', shape=(self.N,self.N))
		grm_u = np.memmap(path.join(self.temp, self.grm_u), dtype='float64', mode='r+', shape=(self.N))
		grm_u[:] = np.matmul(grm, theta[self.M:(self.M+self.N)])
		del grm	
		
		result -= 1/(2*tau)*(np.matmul(theta[self.M:(self.M+self.N)].T, grm_u))
		del theta, grm_u
		return result

	# l_2 as defined in Equation 3 in COXself.MEG paper (mostly their notation)
	def l_1_deriv(self, tau):
		risk_set = np.memmap(path.join(self.temp, self.risk_set), dtype='float64', mode='r', shape=(self.N,self.N))
		exp_eta = np.memmap(path.join(self.temp, self.exp_eta), dtype='float64', mode='r+', shape=(self.N))

		# W = diag(exp(eta)) but working with exp(eta) explicitly (exp_eta)
		MTW = np.multiply(risk_set, exp_eta)
		# A = diag(D) diag^-1(self.M^TW1)
		events = np.memmap(path.join(self.temp, self.events), dtype='int64', mode='r', shape=(self.N))
		A = np.multiply(events, 1/np.sum(MTW, axis=1))		
		# NOTE: WB = WMA1 
		WB = np.multiply(exp_eta, np.sum(np.multiply(risk_set.T, A), axis = 1))
		# setting score function s with parts [one, two]
		s = np.memmap(path.join(self.temp, self.s), dtype='float64', mode='r+', shape=(self.N+self.M))
		# s[two] -- d - WMA1 - (Sigma^-1 gamma) / tau (save sigma^-1 gamma / tau for now)
		s[self.M:(self.M+self.N)] = events - WB
		del events

		# H = WB - QQ^T = WB - WMA^2M^TW	
		H = np.diag(WB) - np.matmul(np.multiply(np.multiply(exp_eta, risk_set.T), np.square(A)), MTW)
		del risk_set, exp_eta, WB, MTW, A

		# setting information matrix V with quadrants [[one, two], [three, four]]
		#V[four] -- H + sigma^-1/tau: always exists since we're looking at random effect		 
		V = np.memmap(path.join(self.temp, self.V), dtype='float64', mode='r+', shape=(self.N+self.M, self.N+self.M))
		grm = np.memmap(path.join(self.temp, self.grm), dtype='float64', mode='r', shape=(self.N, self.N))
		V[self.M:(self.M+self.N), self.M:(self.M+self.N)] = np.add(H, (grm/tau))
		del grm
		# one, two, three only exist if there were fixed effect/covariates
		if self.M > 0:
			fixed = np.memmap(path.join(self.temp, self.fixed), dtype='float64', mode='r', shape=(self.N, self.M))
			#V[two] -- X^TH
			V[0:self.M, self.M:(self.M+self.N)] = np.matmul(fixed.T, H)
			#V[one] -- X^THX = (V[two]X)
			V[0:self.M, 0:self.M] = np.matmul(V[0:self.M, self.M:(self.M+self.N)], fixed)
			#V[three] -- HX = (X^TH)^T = V[two]^T
			V[self.M:(self.M+self.N), 0:self.M] = V[0:self.M, self.M:(self.M+self.N)].T
			#s[one] -- X^T(d - WMA1)
			s[0:self.M] = np.matmul(fixed.T, s[self.M:(self.M+self.N)])
			del fixed
		del H
		# wait to do this in case, we need d - WMA1 if self.M > 0
		grm_u = np.memmap(path.join(self.temp, self.grm_u), dtype='float64', mode='r', shape=(self.N))
		s[self.M:(self.M+self.N)] = s[self.M:(self.M+self.N)] - grm_u/tau		
		del V, s, grm_u

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
			V = np.memmap(path.join(self.temp, self.V), dtype='float64', mode='r', shape=(self.N+self.M,self.N+self.M))
			s = np.memmap(path.join(self.temp, self.s), dtype='float64', mode='r', shape=(self.N+self.M))
			update = np.linalg.solve(V, s)
			del V,s
			theta = np.memmap(path.join(self.temp, self.theta), dtype='float64', mode='r+', shape=(self.N+self.M))

			theta[self.M:(self.M+self.N)] = theta[self.M:(self.M+self.N)] + update[self.M:(self.M+self.N)]
			if self.M > 0:
				theta[0:self.M] = theta[0:self.M] + update[0:self.M]
			del theta

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

				update = damp*update
				theta = np.memmap(path.join(self.temp, self.theta), dtype='float64', mode='r+', shape=(self.N+self.M))
				theta[self.M:(self.M+self.N)] = theta[self.M:(self.M+self.N)] - update[self.M:(self.M+self.N)] 
				if self.M > 0:
					theta[0:self.M] = theta[0:self.M] - update[0:self.M]
				del theta
				
				update_loglike = self.l_1(tau)
				diff_loglike = update_loglike - loglike

			run += 1
		
		if run == max_runs:
			print("The PPL ran for the maximum number of iterations (" + str(max_runs) + "). It probably didn't converge")
			
		V = np.memmap(path.join(self.temp, self.V), dtype='float64', mode='r', shape=(self.N+self.M,self.N+self.M))
		_, log_det = np.linalg.slogdet(V)
		return (self.N*np.log(tau) + log_det + (-2)*update_loglike)

if __name__=="__main__":
	COXPHMM()


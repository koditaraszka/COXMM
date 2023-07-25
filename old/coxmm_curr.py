'''
Auehor: Kodi Taraszka
Email: Kodi_Taraszka@dfci.harvard.edu
This is prelim code for cox proportional hazard mixed model components (CoxMM)
'''

#TODO: double check that left censoring and right censoring work correctly
#TODO: implement Breslow tie breaker, now we add noise to them :(
#TODO: classier handling of output/write to file
#TODO: estimate standard error of tau
#TODO: offer a correction to the estimated tau

import numpy as np
import pandas as pd
import pybobyqa
from scipy.optimize import minimize
from input import IO

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
		self.ONE = np.ones(self.N)
		self.MTW = np.zeros((self.N, self.N))
		self.WB = np.zeros(self.N)
		self.A = np.identity(self.N)
		self.H = np.identity(self.N)
		self.s = np.zeros((self.M+self.N))
		self.V = np.zeros((self.M+self.N, self.M+self.N))
		self.R_j()
		tau = 0.5
		soln = pybobyqa.solve(self.marg_loglike, x0 = [tau], bounds = ([1e-4], [5]))
		print("H2: " + str(soln.x[0]) + " +/- " + str(np.sqrt(1/self.second_deriv(soln.x[0]))))
		print("Betas: " + str(self.theta[0:self.M]))
		self.marg_loglike(0.2360726)
		print("Betas: " + str(self.theta[0:self.M]))


	def second_deriv(self, tau):
		ut_grm_u = np.matmul(self.theta[self.M:(self.M+self.N)].T, np.matmul(self.grm, self.theta[self.M:(self.M+self.N)]))
		return (((0.5*self.N - 0.5)*tau - ut_grm_u)/(tau ** 3))

	# who is at risk when the jth individual has their event
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
		result -= 1/(2*tau)*(np.matmul(self.theta[self.M:(self.M+self.N)].T, self.grm_u)) # + self.N*np.log(tau))
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

		#TODO: there's definitely a doubling of functions, figure this out!!
		##print('two')
		##print(two)
		#exit()



		self.H = np.matmul(np.multiply(np.multiply(self.exp_eta, self.risk_set.T), np.square(self.A)), self.MTW)
		#print('exp(eta)')
		#print(self.exp_eta[0:3])
		#print(self.exp_eta[self.M:(self.M+3)])
		#print('WB')
		#print(self.WB.shape)
		#print(self.WB[0:3])
		#print('H')
		H = np.diag(self.WB) - self.H
		#print(H.shape)
		#print(H[0:3,0:3])
		#print('Hx')
		#Hx = np.dot(self.fixed.T, H)
		#print(Hx.shape)
		#print(Hx[0:3,0:3])
	
		#V[four] -- H + sigma^-1/tau: always exists since we're looking at random effect		 
		self.V[self.M:(self.M+self.N), self.M:(self.M+self.N)] = np.add((np.diag(self.WB) - self.H), (self.grm/tau))
		#print('V[quad four]')
		#print(self.V[self.M:(self.M+3), self.M:(self.M+3)])
		# setting information matrix V with quadrants [[one, two], [three, four]]
		# one, two, three only exist if there were fixed effect/covariates
		if self.M > 0:
			one = np.multiply(self.fixed.T, self.exp_eta)
			Hx = np.matmul(np.multiply(np.dot(self.risk_set, np.multiply(self.fixed.T, self.exp_eta).T).T, np.square(self.A)), self.MTW)
			#V[two] -- X^TH
			#print('WBtX')
			#print(np.multiply(self.fixed.T, self.WB)[0:3,0:3])
			self.V[0:self.M, self.M:(self.M+self.N)] = np.multiply(self.fixed.T, self.WB) - Hx

			#print("V[quad two]")
			#print(self.V[0:3, self.M:(self.M+3)]) 
			#V[one] -- X^THX = (V[two]X)
			self.V[0:self.M, 0:self.M] = np.matmul(self.V[0:self.M, self.M:(self.M+self.N)], self.fixed)
			#print("V[quad one]")
			#print(self.V[0:3, 0:3])
			#V[three] -- HX = (X^TH)^T = V[two]^T
			self.V[self.M:(self.M+self.N), 0:self.M] = self.V[0:self.M, self.M:(self.M+self.N)].T
			#print('V[quad three]')
			#print(self.V[self.M:(self.M+3), 0:3])
			#s[one] -- X^T(d - WMA1)
			self.s[0:self.M] = np.matmul(self.fixed.T, self.s[self.M:(self.M+self.N)])
			#print('s[one]')
			#print(self.s[0:3])
		self.s[self.M:(self.M+self.N)] = self.s[self.M:(self.M+self.N)] - self.grm_u/tau
		#print('s[two]')
		#print(self.s[self.M:(self.M+3)])
	
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
					#print("The optimization of PPL may not converge.")
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

		_, log_det = np.linalg.slogdet(self.V)
		return -1*(update_loglike - self.N*np.log(tau)/2 - log_det/2)

if __name__=="__main__":
	COXMM()

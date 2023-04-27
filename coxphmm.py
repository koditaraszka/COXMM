'''
Author: Kodi Taraszka
Email: Kodi_Taraszka@dfci.harvard.edu
This is prelim code for cox proportional hazard mulitple mixed model components (CoxPHMM)
'''

#TODO: double check that left censoring and right censoring work correctly
#TODO: implement Breslow tie breaker, now we add noise to them :(
#TODO: classier handling of output/write to file
#TODO: estimate standard error of tau
#TODO: offer a correction to the estimated tau

import numpy as np
import pandas as pd
import pybobyqa
from utils import Utils

class COXPHMM(Utils):

	def __init__(self):
		super().__init__()
		self.R_j()
		tau = 0.5 #initized value, may allow user input
		soln = pybobyqa.solve(self.marg_loglike, x0 = [tau], bounds = ([1e-4], [5]))
		print(soln)

	# who is at risk when the jth individual has their event
	def R_j(self):
		# read in items
		self.io_times()
		self.io_risk_set(True)		

		# fixing left censoring, TODO vet the approach
		#if np.unique(self.times[:,0]).shape[0] > 1:
		#	min_start = np.min(self.times[:,0])
		#	late_start = np.where(self.times[:,0]!=min_start)[0]
		#	for who in late_start:
		#		start_time = self.times[who,0]
		#		for before in range(0,who):
		#			if start_time <= self.times[before,0]:
		#				self.risk_set[0:before,who] = 0
		#				break

		# fixing duplicated stopping times
		#for stop_time in self.dup_ttop:
		#	same_time = np.where(self.times[:,1]==stop_time)
		#	min_pos = np.min(who)
		#	max_pos = np.max(who)
			# risk set is currently a
		#	for who in same_time:
		#		self.risk_set[min_pos:(max_pos+1),who] = 1

		# clean up items
		self.risk_set.flush()
		self.risk_set, self.times = None, None

	# l_1 as defined in equations 2 in COXMEG paper (mostly their notation)
	def l_1(self, tau):
		# read in items
		self.io_theta()
		self.io_exp_eta()
		self.io_risk_set()
		self.io_risk_eta()
		self.io_loc()
		self.io_grm()
		self.io_grm_u()

		# eta = Xb + Zu, theta =[beta, u]
		if self.M > 0:
			# only read it if needed
			self.io_fixed()
			self.exp_eta[:] = np.exp(np.matmul(self.fixed, self.theta[0:self.M]) + self.theta[self.M:(self.M+self.N)])
			self.fixed = None
		else:
			self.exp_eta[:] = np.exp(self.theta)
		self.risk_eta[:,:] = np.multiply(self.risk_set[self.loc,:], self.exp_eta)
		self.grm_u[:] = np.matmul(self.grm, self.theta[self.M:(self.M+self.N)])
		
		result = (np.sum(np.log(self.exp_eta[self.loc])) - np.sum(np.log(np.sum(self.risk_eta,axis=1))) \
				 - 1/(2*tau)*(np.matmul(self.theta[self.M:(self.M+self.N)].T, self.grm_u))) 
		
		# clean up items
		self.exp_eta.flush()
		self.risk_eta.flush()
		self.grm_u.flush()
		self.theta, self.grm, self.exp_eta, self.risk_set, self.risk_eta, self.loc, self.grm_u = None, None, None, None, None, None, None
		return result

	# l_2 as defined in Equation 3 in COXMEG paper (mostly their notation)
	def l_1_deriv(self, tau):
		# read in items
		self.io_MTW()
		self.io_risk_set()
		self.io_exp_eta()
		self.io_events()
		self.io_A()
		self.io_B()
		self.io_WB()	
		self.io_H()
		self.io_s()
		self.io_V()
		self.io_grm()
		self.io_grm_u()

		# W = diag(exp(eta)) but working with exp(eta) explicitly (self.exp_eta)
		self.MTW[:] = np.multiply(self.risk_set, self.exp_eta)
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
			self.io_fixed()
			#V[two] -- X^TH
			self.V[0:self.M, self.M:(self.M+self.N)] = np.matmul(self.fixed.T, self.H)
			#V[one] -- X^THX = (V[two]X)
			self.V[0:self.M, 0:self.M] = np.matmul(self.V[0:self.M, self.M:(self.M+self.N)], self.fixed)
			#V[three] -- HX = (X^TH)^T = V[two]^T
			self.V[self.M:(self.M+self.N), 0:self.M] = self.V[0:self.M, self.M:(self.M+self.N)].T
			#s[one] -- X^T(d - WMA1)
			self.s[0:self.M] = np.matmul(self.fixed.T, self.s[self.M:(self.M+self.N)])
			self.fixed = None
		# wait to do this in case, we need d - WMA1 if self.M > 0
		self.s[self.M:(self.M+self.N)] = self.s[self.M:(self.M+self.N)] - self.grm_u/tau
		
		#clean up items, no changes to exp_eta, grm_u
		self.MTW.flush()
		self.A.flush()
		self.B.flush()
		self.WB.flush()
		self.H.flush()
		self.V.flush()
		self.s.flush()
		self.MTW, self.risk_set, self.exp_eta, self.events = None, None, None, None
		self.A, self.B, self.WB, self.H = None, None, None, None
		self.s, self.V, self.grm, self.grm_u = None, None, None, None

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
			
			# read in items
			self.io_theta()
			self.io_update()
			self.io_s()
			self.io_V()
			
			self.update[:] = np.linalg.solve(self.V, self.s)
			self.theta[self.M:(self.M+self.N)] = self.theta[self.M:(self.M+self.N)] + self.update[self.M:(self.M+self.N)]
			if self.M > 0:
				self.theta[0:self.M] = self.theta[0:self.M] + self.update[0:self.M]
			self.update.flush()
			self.theta.flush()
			self.update, self.theta, self.s, self.V = None, None, None, None

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

				# read in items
				self.io_update()
				self.io_theta()
				self.update[:] = damp*self.update
				self.theta[self.M:(self.M+self.N)] = self.theta[self.M:(self.M+self.N)] - self.update[self.M:(self.M+self.N)] 
				if self.M > 0:
					self.theta[0:self.M] = self.theta[0:self.M] - self.update[0:self.M]
				
				# clean up items
				self.theta.flush()
				self.update.flush()
				self.update, self.theta = None, None
				
				update_loglike = self.l_1(tau)
				diff_loglike = update_loglike - loglike

			run += 1
		
		if run == max_runs:
			print("The PPL ran for the maximum number of iterations (" + str(max_runs) + "). It probably didn't converge")
			
		# read in items
		self.io_V()
		_, log_det = np.linalg.slogdet(self.V)
		# clean up items
		self.V = None
		return (self.N*np.log(tau) + log_det + (-2)*update_loglike)


if __name__=="__main__":
	COXPHMM()


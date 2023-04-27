'''
Author: Kodi Taraszka
Email: Kodi_Taraszka@dfci.harvard.edu
This script is the preliminary io script for coxphmm.py
'''

import numpy as np
import pandas as pd
import os.path as path
from input import IO

# TODO: handle missing data, assumes data preprocessed

# This class handles all input/output functions
class Utils(IO):

	def __init__(self):
		super().__init__()
		self.risk_set = None # initalized in parent class (R_j function)
		self.theta, self.update, self.exp_eta = None, None, None
		self.io_theta(True)
		self.io_update(True)
		self.io_exp_eta(True) 
		self.risk_eta, self.grm_u, self.ONE = None, None, None
		self.io_risk_eta(True)
		self.io_grm_u(True) 
		self.io_ONE(True)
		self.MTW, self.WB, self.A = None, None, None
		self.io_MTW(True)
		self.io_WB(True)
		self.io_A(True)
		self.B, self.H, self.s, self.V = None, None, None, None
		self.io_B(True)
		self.io_H(True)
		self.io_s(True)
		self.io_V(True)

	def temp_dir(self, clean=False):
		if clean:
			print("TODO: delete temp directory")
		else:
			print("TODO: generate real temp directory")

	def io_grm(self):
		#first read will be done in IO
		self.grm = np.memmap(path.join(self.temp, "grm.dat"), dtype='float64', mode='r', shape=(self.N,self.N))

	def io_events(self):
		self.events = np.memmap(path.join(self.temp, "events.dat"), dtype='int64', mode='r', shape=(self.N))

	def io_times(self):
		# first read will be done in IO
		self.times = np.memmap(path.join(self.temp, "times.dat"), dtype='float64', mode='r', shape=(self.N,2))
	
	def io_fixed(self):
		# first read will be done in IO
		self.fixed = np.memmap(path.join(self.temp, "fixed.dat"), dtype='float64', mode='r', shape=(self.N,self.M))

	def io_loc(self):
		# first read will be done in IO
		self.loc = np.memmap(path.join(self.temp, "loc.dat"), dtype='int64', mode='r', shape=(self.uncensored))		

	def io_theta(self, first=False):
		if first:
			self.theta = np.memmap(path.join(self.temp, "theta.dat"), dtype='float64', mode='w+', shape=(self.M+self.N)) 
			#self.theta.flush()
			self.theta = None
		else:
			self.theta = np.memmap(path.join(self.temp, "theta.dat"), dtype='float64', mode='r+', shape=(self.M+self.N))

	def io_exp_eta(self, first = False):
		if first:
			self.exp_eta = np.memmap(path.join(self.temp, "exp_eta.dat"), dtype='float64', mode='w+', shape=(self.N)) 
			#self.exp_eta.flush()
			self.exp_eta = None
		else:
			self.exp_eta = np.memmap(path.join(self.temp, "exp_eta.dat"), dtype='float64', mode='r+', shape=(self.N))

	def io_update(self, first = False):
		if first:
			self.update = np.memmap(path.join(self.temp, "update.dat"), dtype='float64', mode='w+', shape=(self.M+self.N)) 
			#self.update.flush()
			self.update = None
		else:
			self.update = np.memmap(path.join(self.temp, "update.dat"), dtype='float64', mode='r+', shape=(self.M+self.N))
	
	def io_risk_set(self, first=False):
		if first:
			self.risk_set = np.memmap(path.join(self.temp, "risk_set.dat"), dtype='int64', mode='w+', shape=(self.N, self.N))
			self.risk_set[:,:] = np.tril(np.ones((self.N, self.N))).T
		else:
			# only update the one time
			self.risk_set = np.memmap(path.join(self.temp, "risk_set.dat"), dtype='int64', mode='r', shape=(self.N, self.N))

	def io_risk_eta(self, first=False):
		if first:
			self.risk_eta = np.memmap(path.join(self.temp, "risk_eta.dat"), dtype='float64', mode='w+', shape=(self.N, self.N))
			#self.risk_eta.flush()
			self.risk_eta = None
		else:
			self.risk_eta = np.memmap(path.join(self.temp, "risk_eta.dat"), dtype='float64', mode='r+', shape=(self.N, self.N))
		
	def io_grm_u(self, first=False):
		if first:
			self.grm_u = np.memmap(path.join(self.temp, "grm_u.dat"), dtype='float64', mode='w+', shape=(self.N))
			self.grm_u.flush()
			self.grm_u = None
		else:
			self.grm_u = np.memmap(path.join(self.temp, "grm_u.dat"), dtype='float64', mode='r+', shape=(self.N))

	def io_ONE(self, first=False):
		if first:
			self.ONE = np.memmap(path.join(self.temp, "ONE.dat"), dtype='int64', mode='w+', shape=(self.N))
			self.ONE[:] = np.ones(self.N)
			self.ONE.flush()
			self.ONE = None
		else:
			# only update the one time
			self.ONE = np.memmap(path.join(self.temp, "ONE.dat"), dtype='int64', mode='r', shape=(self.N))
		
	def io_MTW(self, first=False):
		if first:
			self.MTW = np.memmap(path.join(self.temp, "MTW.dat"), dtype='float64', mode='w+', shape=(self.N, self.N))
			self.MTW.flush()
			self.MTW = None
		else:
			self.MTW = np.memmap(path.join(self.temp, "MTW.dat"), dtype='float64', mode='r+', shape=(self.N, self.N))

	def io_WB(self, first=False):
		if first:
			self.WB = np.memmap(path.join(self.temp, "WB.dat"), dtype='float64', mode='w+', shape=(self.N))
			self.WB.flush()
			self.WB = None
		else:
			self.WB = np.memmap(path.join(self.temp, "WB.dat"), dtype='float64', mode='r+', shape=(self.N))
		
	def io_A(self, first=False):
		if first:
			self.A = np.memmap(path.join(self.temp, "A.dat"), dtype='float64', mode='w+', shape=(self.N))
			self.A.flush()
			self.A = None
		else:
			self.A = np.memmap(path.join(self.temp, "A.dat"), dtype='float64', mode='r+', shape=(self.N))

	def io_B(self, first=False):
		if first:
			self.B = np.memmap(path.join(self.temp, "B.dat"), dtype='float64', mode='w+', shape=(self.N))
			self.B.flush()
			self.B = None
		else:
			self.B = np.memmap(path.join(self.temp, "B.dat"), dtype='float64', mode='r+', shape=(self.N)) 

	def io_H(self, first=False):
		if first:
			self.H = np.memmap(path.join(self.temp, "H.dat"), dtype='float64', mode='w+', shape=(self.N, self.N))
			self.H.flush()
			self.H = None
		else:
			self.H = np.memmap(path.join(self.temp, "H.dat"), dtype='float64', mode='r+', shape=(self.N, self.N))

	def io_s(self, first=False):
		if first:
			self.s = np.memmap(path.join(self.temp, "s.dat"), dtype='float64', mode='w+', shape=(self.M+self.N))
			self.s.flush()
			self.s = None
		else:
			self.s = np.memmap(path.join(self.temp, "s.dat"), dtype='float64', mode='r+', shape=(self.M+self.N))

	def io_V(self, first=False):
		if first:
			self.V = np.memmap(path.join(self.temp, "V.dat"), dtype='float64', mode='w+', shape=(self.M+self.N, self.M+self.N))
			self.V.flush()
			self.V = None
		else:
			self.V = np.memmap(path.join(self.temp, "V.dat"), dtype='float64', mode='r+', shape=(self.M+self.N, self.M+self.N))

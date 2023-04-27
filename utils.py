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
		self.theta = np.memmap(path.join(self.temp, "theta.dat"), dtype='float64', mode='w+', shape=(self.M+self.N)) 
		self.exp_eta = np.memmap(path.join(self.temp, "exp_eta.dat"), dtype='float64', mode='w+', shape=(self.N)) 
		self.update = np.memmap(path.join(self.temp, "update.dat"), dtype='float64', mode='w+', shape=(self.M+self.N)) 
		self.risk_set = np.memmap(path.join(self.temp, "risk_set.dat"), dtype='int64', mode='w+', shape=(self.N, self.N))
		self.risk_eta = np.memmap(path.join(self.temp, "risk_eta.dat"), dtype='float64', mode='w+', shape=(self.N, self.N))
		self.grm_u = np.memmap(path.join(self.temp, "grm_u.dat"), dtype='float64', mode='w+', shape=(self.N))
		self.ONE = np.memmap(path.join(self.temp, "ONE.dat"), dtype='int64', mode='w+', shape=(self.N))
		self.ONE[:] = np.ones(self.N)
		self.ONE.flush()
		self.MTW = np.memmap(path.join(self.temp, "MTW.dat"), dtype='float64', mode='w+', shape=(self.N, self.N))
		self.WB = np.memmap(path.join(self.temp, "WB.dat"), dtype='float64', mode='w+', shape=(self.N))
		self.A = np.memmap(path.join(self.temp, "A.dat"), dtype='float64', mode='w+', shape=(self.N))
		self.B = np.memmap(path.join(self.temp, "B.dat"), dtype='float64', mode='w+', shape=(self.N))
		self.H = np.memmap(path.join(self.temp, "H.dat"), dtype='float64', mode='w+', shape=(self.N, self.N))
		self.s = np.memmap(path.join(self.temp, "s.dat"), dtype='float64', mode='w+', shape=(self.M+self.N))
		self.V = np.memmap(path.join(self.temp, "V.dat"), dtype='float64', mode='w+', shape=(self.M+self.N, self.M+self.N))

	def temp_dir(self, clean=False):
		if clean:
			print("TODO: delete temp directory")
		else:
			print("TODO: generate real temp directory")

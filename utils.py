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
		self.risk_set = 'risk_set.dat'
		np.memmap(path.join(self.temp, "risk_set.dat"), dtype='float64', mode='w+', shape=(self.M+self.N))
		self.theta = 'theta.dat'
		np.memmap(path.join(self.temp, "theta.dat"), dtype='float64', mode='w+', shape=(self.M+self.N)) 
		self.exp_eta = 'exp_eta.dat'
		np.memmap(path.join(self.temp, "exp_eta.dat"), dtype='float64', mode='w+', shape=(self.N)) 
		self.grm_u = 'grm_u.dat'
		np.memmap(path.join(self.temp, "grm_u.dat"), dtype='float64', mode='w+', shape=(self.N))
		self.s = 's.dat'
		np.memmap(path.join(self.temp, "s.dat"), dtype='float64', mode='w+', shape=(self.M+self.N))
		self.V = 'V.dat'
		q:np.memmap(path.join(self.temp, "V.dat"), dtype='float64', mode='w+', shape=(self.M+self.N, self.M+self.N))

	def temp_dir(self, clean=False):
		if clean:
			print("TODO: delete temp directory")
		else:
			print("TODO: generate real temp directory")

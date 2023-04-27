'''
Author: Kodi Taraszka
Email: Kodi_Taraszka@dfci.harvard.edu
This script is the preliminary io script for coxphmm.py
'''

import numpy as np
import argparse
import pandas as pd
import os.path as path


# TODO: handle missing data, assumes data preprocessed

# This class handles all input/output functions
class IO():

	def __init__(self):
		self.temp = 'temp'
		self.output = ''
		self.N = 0
		self.M = 0
		self.grm = None
		self.events = None
		self.fixed = None
		self.setup()
		self.times = np.memmap(path.join(self.temp, 'times.dat'), dtype='float64', mode='w+', shape=(self.N,2))
		self.times[:,:] = self.events[['start', 'stop']].to_numpy()
		self.times.flush()
		self.times = None
		events = self.events.event.to_numpy()
		self.uncensored = np.where(events==1)[0].shape[0]
		self.loc = np.memmap(path.join(self.temp, 'loc.dat'), dtype='int64', mode='w+', shape=(self.uncensored))
		self.loc[:] = np.where(events==1)[0]
		self.loc.flush()
		self.loc = None
		self.events = np.memmap(path.join(self.temp, 'events.dat'), dtype='int64', mode='w+', shape=(self.N))
		self.events[:] = events
		self.events.flush()
		self.events = None

	# this function calls the parser and processes the input
	def setup(self):
		args = self.def_parser()
		self.output = args.output
		self.process_events(args.sample_id, args.events)

		if len(args.grm) > 1:
			print("Warning: Only reading in/working with the first GRM")

		self.grm = np.memmap(path.join(self.temp, "grm.dat"), dtype='float64', mode='w+', shape=(self.N,self.N))
		grm = np.memmap(args.grm[0], dtype='float64', mode='r', shape=(self.N,self.N))
		if grm.shape[0] != grm.shape[1]:
			raise ValueError("GRM: " + args.grm[0] + " is not a square matrix")
		# intentionally reassign variable, not memory
		grm = grm[self.events.rownum.to_numpy(), :]
		grm = grm[:, self.events.rownum.to_numpy()]
		Lambda, U = np.linalg.eigh(grm)#, driver = 'evd')
		Lambda[Lambda < 1e-10] = 1e-6		
		self.grm[:,:] = np.matmul(np.matmul(U, np.diag(1/Lambda)), np.transpose(U))
		self.grm.flush()
		self.grm = None
		del grm, Lambda, U

		if args.fixed is not None:
			print("Warning: minimal checks on issues with fixed effects input. Be careful")
			fixed = pd.read_csv(args.fixed, sep = '\t', header = 0)
			if args.sample_id is not None:
				fixed = fixed.rename(columns = {args.sample_id:'sample_id'})

			fixed = fixed.set_index('sample_id')
			columns = fixed.columns
			for col in columns:
				if fixed[col].dtype == object:
					print("Column: " + col + " is a categorical variable and will be one hot encoded")
					one_hot = pd.get_dummies(fixed[col])
					one_hot.pop(one_hot.columns[0])
					fixed = fixed.drop(col, axis = 1)
					fixed = fixed.join(one_hot)
				elif fixed[col].dtype == np.float64:	
					var = fixed[col].var()
					if(var > 100 or var < 0.01):
						print("Column: " + col + " has a variance > 100 or < 0.01 and will be centered and scaled")
						mean = fixed[col].mean()
						fixed[col] = (fixed[col] - mean)/np.sqrt(var)
						print("mean: " + str(fixed[col].mean()) + " var: " + str(fixed[col].var()))

			fixed = fixed.reindex(index = self.events.index)
			fixed = fixed.to_numpy()
			self.M = fixed.shape[1]
			self.fixed = np.memmap(path.join(self.temp, "fixed.dat"), dtype='float64', mode='w+', shape=(self.N, self.M))
			self.fixed[:,:] = fixed
			self.fixed.flush()
			self.fixed = None

	# this function contains the parser and it's arguments
	def def_parser(self):
		parser = argparse.ArgumentParser(description = "This program runs a cox proportional hazard mixed model with multiple components")

		required = parser.add_argument_group('Required Arguments')
		required.add_argument('-e', '--events', required = True,
			help = 'path to outcome file which contains four columns: sample_id, start, stop, event with sample_id changeable by argument -s/--sample_id')
		required.add_argument('-g', '--grm', dest = 'grm', nargs='*',
			help = 'path to tab delim files containing precomputed relatedness matrices. There is no header row in any file nor is there a column for sample ids.')

		optional = parser.add_argument_group('Optional Arguments')
		optional.add_argument('-s', '--sample_id', dest = 'sample_id',
			help = 'column name across events/fixed effects files. Order in events file same order as GRM. Default is sample_id')
		optional.add_argument('-f', '--fixed', dest = 'fixed',
			help = 'path tab delim file containing fixed effects features. First row containing column names')
		optional.add_argument('-o', '--output', dest = 'output', default = 'results.txt',
			help = 'path to output file. Default = results.txt')
	
		args = parser.parse_args()
		if args.fixed is not None:
			if not path.isfile(args.fixed):
				raise ValueError("The fixed effect file does not exist")

		if not path.isfile(args.events):
			raise ValueError("The outcomes/events file does not exist")

		if not path.isfile(args.grm[0]):
			raise ValueError("The grm file does not exist")

		return(args)

	def process_events(self, sample_id, file):
		self.events = pd.read_csv(file, sep = '\t', header = 0)
		print("Warning: There are no checks for ordering between GRM and events file")
		if self.events.shape[1] != 4:
			raise ValueError("There should be four columns in the events/outcome file")
		
		self.events['rownum'] = self.events.index
		if sample_id is not None:
			self.events = self.events.rename(columns = {sample_id:'sample_id'})
		
		self.N = self.events.shape[0]
		self.events = self.events[(self.events.stop > self.events.start)]
		if self.events.shape[0] != self.N:
			print(str(orig - self.events.shape[0]) + " individuals dropped because start time is after end time")
			self.N = self.events.shape[0]
	
		self.events = self.events.sort_values("stop", ascending=True).reset_index(drop=True)
		self.events = self.events.set_index('sample_id')


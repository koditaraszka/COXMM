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
		self.uncensored = 0
		self.grm = 'grm.dat'
		self.fixed = 'fixed.dat'
		self.events = 'events.dat'
		self.times = 'times.dat'
		self.loc = 'loc.dat'
		self.risk_set = 'risk_set.dat'
		self.theta = 'theta.dat'
		self.exp_eta = 'exp_eta.dat'
		self.grm_u = 'grm_u.dat'
		self.s = 's.dat'
		self.V = 'V.dat'
		self.setup()

	# most of the memory maps are made here
	def mem_map(self):
		a = np.memmap(path.join(self.temp, 'times.dat'), dtype='float64', mode='w+', shape=(self.N,2))
		b = np.memmap(path.join(self.temp, "risk_set.dat"), dtype='float64', mode='w+', shape=(self.M+self.N))
		c = np.memmap(path.join(self.temp, "theta.dat"), dtype='float64', mode='w+', shape=(self.M+self.N))
		d = np.memmap(path.join(self.temp, "exp_eta.dat"), dtype='float64', mode='w+', shape=(self.N))
		e = np.memmap(path.join(self.temp, "grm_u.dat"), dtype='float64', mode='w+', shape=(self.N))
		f = np.memmap(path.join(self.temp, "s.dat"), dtype='float64', mode='w+', shape=(self.M+self.N))
		g = np.memmap(path.join(self.temp, "V.dat"), dtype='float64', mode='w+', shape=(self.M+self.N, self.M+self.N))
		del a,b,c,d,e,f,g

	def temp_dir(self, clean=False):
		if clean:
			print("TODO: delete temp directory")
		else:
			print("TODO: generate real temp directory")

	# this function calls the parser and processes the input
	def setup(self):
		args = self.def_parser()
		self.output = args.output
		events = self.process_events(args.sample_id, args.events)

		if len(args.grm) > 1:
			print("Warning: Only reading in/working with the first GRM")

		new_grm = np.memmap(path.join(self.temp, "grm.dat"), dtype='float64', mode='w+', shape=(self.N,self.N))
		grm = np.memmap(args.grm[0], dtype='float64', mode='r', shape=(self.N,self.N))
		if grm.shape[0] != grm.shape[1]:
			raise ValueError("GRM: " + args.grm[0] + " is not a square matrix")
		# intentionally reassign variable, not memory
		grm = grm[events.rownum.to_numpy(), :]
		grm = grm[:, events.rownum.to_numpy()]
		Lambda, U = np.linalg.eigh(grm)#, driver = 'evd')
		Lambda[Lambda < 1e-10] = 1e-6		
		new_grm[:,:] = np.matmul(np.matmul(U, np.diag(1/Lambda)), np.transpose(U))
		del new_grm

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

			fixed = fixed.reindex(index = events.index)
			fixed = fixed.to_numpy()
			self.M = fixed.shape[1]
			new_fixed = np.memmap(path.join(self.temp, "fixed.dat"), dtype='float64', mode='w+', shape=(self.N, self.M))
			new_fixed[:,:] = fixed
			del new_fixed
		
		self.mem_map()
		times = np.memmap(path.join(self.temp, "times.dat"), dtype='float64', mode='w+', shape=(self.N,2))
		times[:,:] = events[['start', 'stop']].to_numpy() 
		new_events = np.memmap(path.join(self.temp, "events.dat"), dtype='float64', mode='w+', shape=(self.N))
		new_events[:] = events.event.to_numpy()
		self.uncensored = np.where(events==1)[0].shape[0]
		loc = np.memmap(path.join(self.temp, "loc.dat"), dtype='int64', mode='w+', shape=(self.uncensored))
		loc[:] = np.where(events==1)[0]
		del loc, times, new_events

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
		events = pd.read_csv(file, sep = '\t', header = 0)
		print("Warning: There are no checks for ordering between GRM and events file")
		if events.shape[1] != 4:
			raise ValueError("There should be four columns in the events/outcome file")
		
		events['rownum'] = events.index
		if sample_id is not None:
			events = events.rename(columns = {sample_id:'sample_id'})
		
		self.N = events.shape[0]
		events = events[(events.stop > events.start)]
		if events.shape[0] != self.N:
			print(str(orig - events.shape[0]) + " individuals dropped because start time is after end time")
			self.N = events.shape[0]
	
		events = events.sort_values("stop", ascending=True).reset_index(drop=True)
		events = events.set_index('sample_id')
		return events

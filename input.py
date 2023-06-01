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
		risk_set = np.memmap(path.join(self.temp, self.risk_set), dtype='float64', mode='w+', shape=(self.N,self.N))
		risk_set[:,:] = np.tril(np.ones((self.N, self.N))).T
		del risk_set

		theta = np.memmap(path.join(self.temp, self.theta), dtype='float64', mode='w+', shape=(self.M+self.N))
		del theta
		
		exp_eta = np.memmap(path.join(self.temp, self.exp_eta), dtype='float64', mode='w+', shape=(self.N))
		del exp_eta
		
		grm_u = np.memmap(path.join(self.temp, self.grm_u), dtype='float64', mode='w+', shape=(self.N))
		del grm_u
		
		s = np.memmap(path.join(self.temp, self.s), dtype='float64', mode='w+', shape=(self.M+self.N))
		del s
		
		V = np.memmap(path.join(self.temp, self.V), dtype='float64', mode='w+', shape=(self.M+self.N, self.M+self.N))
		del V

	def temp_dir(self, clean=False):
		if clean:
			print("TODO: delete temp directory")
		else:
			print("TODO: generate real temp directory")

	# this function calls the parser and processes the input
	def setup(self):
		args = self.def_parser()
		# set output/temp
		self.output = args.output
		self.temp = args.temp
		# set events
		events = self.process_events(args.sample_id, args.events, args.grm_names)

		# set GRM/random effect
		if len(args.grm) > 1:
			print("Warning: Only reading in/working with the first GRM")
		new_grm = np.memmap(path.join(self.temp, self.grm), dtype='float64', mode='w+', shape=(self.N,self.N))
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

		# set fixed effects
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
			new_fixed = np.memmap(path.join(self.temp, self.fixed), dtype='float64', mode='w+', shape=(self.N, self.M))
			new_fixed[:,:] = fixed
			del new_fixed
		
		# initialize all other variables
		self.mem_map()
		times = np.memmap(path.join(self.temp, self.times), dtype='float64', mode='w+', shape=(self.N,2))
		times[:,:] = events[['start', 'stop']].to_numpy() 
		new_events = np.memmap(path.join(self.temp, self.events), dtype='int64', mode='w+', shape=(self.N))
		new_events[:] = events.event.to_numpy()
		self.uncensored = np.where(events.event==1)[0].shape[0]
		loc = np.memmap(path.join(self.temp, self.loc), dtype='int64', mode='w+', shape=(self.uncensored))
		loc[:] = np.where(events.event==1)[0]
		del loc, times, new_events

	# this function contains the parser and it's arguments
	def def_parser(self):
		parser = argparse.ArgumentParser(description = "This program runs a cox proportional hazard mixed model with multiple components")

		required = parser.add_argument_group('Required Arguments')
		required.add_argument('-e', '--events', required = True,
			help = 'path to outcome file which contains four columns: sample_id, start, stop, event with sample_id changeable by argument -s/--sample_id')
		required.add_argument('-g', '--grm', dest = 'grm', nargs='*',
			help = 'path to tab delim files containing precomputed relatedness matrices. There is no header row in any file nor is there a column for sample ids.')
		required.add_argument('-n', '--grm_names', dest = 'grm_names',
                        help = 'path to file containing one column of names/sample_ids with grm sample order. There is a header row which makes the sample_id in the events file.')
		optional = parser.add_argument_group('Optional Arguments')
		optional.add_argument('-s', '--sample_id', dest = 'sample_id',
			help = 'column name across events/fixed effects files. Order in events file same order as GRM. Default is sample_id')
		optional.add_argument('-f', '--fixed', dest = 'fixed',
			help = 'path tab delim file containing fixed effects features. First row containing column names')
		optional.add_argument('-o', '--output', dest = 'output', default = 'results.txt',
			help = 'path to output file. Default = results.txt')
		optional.add_argument('-t', '--temp', dest = 'temp', default = 'temp',
			help = 'path to temp directory. Default = temp')
	
		args = parser.parse_args()
		if args.fixed is not None:
			if not path.isfile(args.fixed):
				raise ValueError("The fixed effect file does not exist")

		if not path.isfile(args.events):
			raise ValueError("The outcomes/events file does not exist")

		if not path.isfile(args.grm[0]):
			raise ValueError("The grm file does not exist")

		return(args)

	def process_events(self, sample_id, file, names):
		grm_names = pd.read_csv(names, header = 0)
		events = pd.read_csv(file, sep = '\t', header = 0)
		
		if events.shape[1] != 4:
			raise ValueError("There should be four columns in the events/outcome file")	
		if sample_id is not None:
			grm_names = grm_names.rename(columns = {sample_id:'sample_id'})
			events = events.rename(columns = {sample_id:'sample_id'})	
		
		grm_names["rownum"] = grm_names.index
		grm_names["real_id"] = grm_names.sample_id
		grm_names = grm_names.set_index('sample_id')
		
		events = events.set_index('sample_id')
		events = pd.concat([grm_names, events], axis=1)
		self.N = events.shape[0]
		events = events[(events.stop > events.start)]
		if events.shape[0] != self.N:
			print(str(self.N - events.shape[0]) + " individuals dropped because start time is after end time")
			self.N = events.shape[0]
	
		events = events.sort_values("stop", ascending=True).reset_index(drop=True)
		events = events.rename(columns = {'real_id':'sample_id'})
		events = events.set_index('sample_id')
		return events

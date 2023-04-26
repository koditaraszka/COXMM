'''
Author: Kodi Taraszka
Email: Kodi_Taraszka@dfci.harvard.edu
This script is the preliminary io script for coxphmm.py
'''

import numpy as np
import argparse
import pandas as pd
import os


# TODO: handle missing data, assumes data preprocessed

# This class handles all input/output functions
class IO():

	def __init__(self):
		self.N = 0
		self.M = 0
		self.grm = []
		self.events = None
		self.fixed = None
		self.output = ''
		#self.method = ''
		#self.solver = 'exact'
		self.setup()
		self.times = self.events[['start', 'stop']].to_numpy()
		self.events = self.events.event.to_numpy()
		#TODO: allow more than one GRM for analyses
		# already allow it as input, but only have one
		self.grm = self.grm[0]

	# this function calls the parser and processes the input
	def setup(self):
		args = self.def_parser()
		self.output = args.output
		#self.method = args.method
		self.process_events(args.sample_id, args.events)
		if args.random is not None:
			for i in args.random:
				self.compute_grm(i, args.sample_id)
				first = False

		elif args.grm is not None:
			for i in range(0, len(args.grm)):
				grm = np.loadtxt(args.grm[i])
				#pd.read_csv(args.grm[i], sep = '\t', header = None)
				if grm.shape[0] != grm.shape[1]:
					raise ValueError("GRM: " + args.grm[i] + " is not a square matrix")

				grm_names = pd.read_csv(args.grm_names[i], sep = '\t', header = 0)
				if grm.shape[0] != grm_names.shape[0]:
					raise ValueError("The GRM and GRM names file are not the same size")
				
				if args.sample_id is not None:
					grm_names = grm_names.rename(columns = {args.sample_id:'sample_id'})

				grm_names = grm_names.sample_id.to_list()
				#TODO compare grm names to self.events.index
				grm = grm[self.events.rownum.to_numpy(), :]
				grm = grm[:, self.events.rownum.to_numpy()]
				#grm = grm.reindex(index = self.events.index, columns = self.events.index)
				self.grm.append(grm)

		else:
			raise ValueError("Need to provide GRM or features to create ``GRM``")

		if args.fixed is not None:
			self.fixed = pd.read_csv(args.fixed, sep = '\t', header = 0)
			if args.sample_id is not None:
				self.fixed = self.fixed.rename(columns = {args.sample_id:'sample_id'})

			self.fixed = self.fixed.set_index('sample_id')
			columns = self.fixed.columns
			for col in columns:
				if self.fixed[col].dtype == object:
					print("Column: " + col + " is a categorical variable and will be one hot encoded")
					one_hot = pd.get_dummies(self.fixed[col])
					one_hot.pop(one_hot.columns[0])
					self.fixed = self.fixed.drop(col, axis = 1)
					self.fixed = self.fixed.join(one_hot)
				elif self.fixed[col].dtype == np.float64:	
					var = self.fixed[col].var()
					if(var > 100 or var < 0.01):
						print("Column: " + col + " has a variance > 100 or < 0.01 and will be centered and scaled")
						mean = self.fixed[col].mean()
						self.fixed[col] = (self.fixed[col] - mean)/np.sqrt(var)
						print("mean: " + str(self.fixed[col].mean()) + " var: " + str(self.fixed[col].var()))

			if len(self.grm) > 0:
				compare = np.unique(self.grm[0].sort_index().index == self.fixed.sort_index().index)
				if len(compare) > 1 or compare[0] == False:
						raise ValueError("The sample ids are not consistent across file")

			self.fixed = self.fixed.reindex(index = self.events.index)
			self.fixed = self.fixed.to_numpy()
			self.M = self.fixed.shape[1]
		else:
			self.fixed = np.array([])
			self.M = 0

	# this function contains the parser and it's arguments
	def def_parser(self):
		parser = argparse.ArgumentParser(description = "This program runs a cox proportional hazard mixed model with multiple components")

		required = parser.add_argument_group('Required Arguments')
		required.add_argument('-e', '--events', required = True,
			help = 'path to outcome file which contains four columns: sample_id, start, stop, event with sample_id changeable by argument -s/--sample_id')

		optional = parser.add_argument_group('Optional Arguments (all for now)')
		optional.add_argument('-s', '--sample_id', dest = 'sample_id',
			help = 'column name across files to merge outcome, fixed effect and random effect features. Default is sample_id')
		optional.add_argument('-f', '--fixed', dest = 'fixed',
			help = 'path tab delim file containing fixed effects features. First row containing column names')
		optional.add_argument('-r', '--random', dest = 'random', nargs='*',
			help = 'path of tab delim file(s) containing random effect features. First row of each file contains column names')
		optional.add_argument('-g', '--grm', dest = 'grm', nargs='*',
			help = 'path to tab delim files containing precomputed relatedness matrices. There is no header row in any file nor is there a column for sample ids. Only use one of -g/-grm or -r/--random')
		optional.add_argument('-gn', '--grm_names', dest = 'grm_names', nargs='*',
			help = 'path to tab delim files(s) containing sample_id (in order) for each grm with a header row. Must provide one for each grm. Only use one of -g/--grm or -r/--random.')
		#optional.add_argument('-m', '--method', dest = 'method', default = "BOBYQA",
		#	help = 'solver method passed to nlopt, default is BOBYQA')
		optional.add_argument('-o', '--output', dest = 'output', default = 'results.txt',
			help = 'path to output file. Default = results.txt')
	
		args = parser.parse_args()
		if args.fixed is None and args.random is None and args.grm is None:
			raise ValueError("There is are no fixed or random effects, please include at least one")

		if args.fixed is not None:
			if not os.path.isfile(args.fixed):
				raise ValueError("The fixed effect file does not exist")

		if args.random is not None and args.grm is not None:
			 raise ValueError("Please only use -r/--random or -g/--grm not both")

		if args.random is not None:
			for i in args.random:
				if not os.path.isfile(i):
					raise ValueError("Random effect input file: " + i + " does not exist")

		if args.grm is not None:
			if args.grm_names is None:
				raise ValueError("Each GRM needs a matching file containing the sample names in order")
			if len(args.grm) != len(args.grm_names):
				raise ValueError("The number of arguments passed to -g/--grm does not equal the number of arguments passed to -gn/--grm_names")
			
			for i in range(0, len(args.grm)):
				if not os.path.isfile(args.grm[i]):
					raise ValueError("GRM input file: " + args.grm[i] + " does not exist")
				if not os.path.isfile(args.grm_names[i]):
					raise ValueError("GRM input file: " + args.grm_names[i] + " does not exist")

		return(args)

	def process_events(self, sample_id, file):
		self.events = pd.read_csv(file, sep = '\t', header = 0)
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

	# this function process the random effect files
	def compute_grm(self, file, name):
		random = pd.read_csv(file, sep = '\t', header = 0)
		if name is not None:
			random = random.rename(columns = {name:'sample_id'})

		random = random.set_index('sample_id')
		random = random.reindex(index = self.events.index)

		self.grm.append((random.dot(random.T) / random.shape[1]).to_numpy())

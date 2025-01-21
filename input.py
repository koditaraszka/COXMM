'''
Author: Kodi Taraszka
Email: Kodi_Taraszka@dfci.harvard.edu
This script is the preliminary io script for coxmm.py
'''

import numpy as np
import argparse
import pandas as pd
import os.path as path
from pandas_plink import read_plink
import random
import math

# TODO: handle missing data in fixed effects/outcome, assumes data preprocessed
# TODO: allow multiple GRMs

# This class handles all input/output functions
class IO():

	def __init__(self):
		self.output = ''
		self.grmN = 0
		self.N = 0
		self.M = 0
		self.names = None
		self.grm = None
		self.fixed = None
		self.events = None
		self.plink = None
		self.results = None
		self.gwas = None
		self.center = False
		self.names = None
		self.setup()
		self.times = self.events[['start', 'stop']].to_numpy()
		self.names = self.events.index.astype('str')
		self.events = self.events.event.to_numpy()
		
	# this function calls the parser and processes the input
	def setup(self):
		args = self.def_parser()
		self.center = args.center
		self.output = args.output
		self.gwas = args.gwas
		self.plink = args.plink
		grm_names = pd.read_csv(args.grm_names, header = 0)
		self.grmN = grm_names.shape[0]
		
		self.events = self.process_events(args.sample_id, args.events, args.grm_names, args.jackknife, args.seed)
		if len(args.grm) > 1:
			print("Warning: Only reading in/working with the first GRM")

		grm = np.loadtxt(args.grm[0])
		if grm.shape[0] != grm.shape[1]:
			raise ValueError("GRM: " + args.grm[0] + " is not a square matrix")

		grm = grm[self.events.rownum.to_numpy(), :]
		grm = grm[:, self.events.rownum.to_numpy()]
		Lambda, U = np.linalg.eigh(grm)
		Lambda[Lambda < 1e-10] = 1e-6
		self.grm = np.matmul(np.matmul(U, np.diag(1/Lambda)), U.T)
		
		if args.fixed is not None:
			print("Warning: minimal checks on issues with fixed effects input. Be careful")
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

			self.fixed = self.fixed.reindex(index = self.events.index)
			if self.plink is not None:
				self.fixed.insert(0,'SNP',np.nan)
			self.fixed = self.fixed.to_numpy()
			self.M = self.fixed.shape[1]

	# this function contains the parser and it's arguments
	def def_parser(self):
		parser = argparse.ArgumentParser(description = "This program runs a cox proportional hazard mixed model with multiple components")

		required = parser.add_argument_group('Required Arguments')
		required.add_argument('-e', '--events', required = True,
			help = 'path to outcome file which contains four columns: sample_id, start, stop, event with sample_id changeable by argument -s/--sample_id')
		required.add_argument('-g', '--grm', dest = 'grm', nargs = '*',
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
		optional.add_argument('-j', '--jackknife', dest = 'jackknife', nargs = '*',
			help = 'perform one round of jackknife sampling. Needs two arguments: total number of splits and which split currently running (base 1). E.G. -j 10 1')
		optional.add_argument('-d', '--seed', dest = 'seed', default = 123, type = int,
			help = 'random seed to be used with -j/--jackknife, to set split (use same over all splits. Default = 123')
		optional.add_argument('-w', '--gwas', dest = 'gwas', type = float,
			help = 'run GWAS with the heritabilty estimate provided with this argument and used alongside -p/--plink')
		optional.add_argument('-p', '--plink', dest = 'plink',
			help = 'path to prefix for plink bim/bed/fam files and used alongside -w/--gwas')
		optional.add_argument('-c', '--centerScale', dest = 'center', action = 'store_true', default = False,
			help = 'indicate if GWAS SNPs should be centered and scaled')
		
		args = parser.parse_args()
		# basic checks on input
		if args.fixed is not None:
			if not path.isfile(args.fixed):
				raise ValueError("The fixed effect file does not exist")

		if not path.isfile(args.events):
			raise ValueError("The outcomes/events file does not exist")

		if not path.isfile(args.grm[0]):
			raise ValueError("The grm file does not exist")

		if args.gwas is not None and args.plink is None:
			raise ValueError("GWAS indicated but plink files not included")

		if args.gwas is None and args.plink is not None:
			raise ValueError("Plink path included but GWAS was not indicated/heritabilty estimate not provided")

		if args.plink is not None and not path.isfile(args.plink + ".bed"):
			raise ValueError("The plink file does not exist")

		if args.gwas is not None:
			if args.gwas > 1 or args.gwas < 0:
				print("The heritability estimate provided was not between [0, 1] which is unexpected")
		
		return(args)


	def process_events(self, sample_id, file, names, jackknife, seed):
		grm_names = pd.read_csv(names, header = 0)
		events = pd.read_csv(file, sep = '\t', header = 0)
		
		if events.shape[1] != 4:
			raise ValueError("There should be four columns in the events/outcome file")

		if sample_id is not None:
			grm_names = grm_names.rename(columns = {sample_id:'sample_id'})
			events = events.rename(columns = {sample_id:'sample_id'})
			if drop is not None:
				drop = drop.rename(columns = {sample_id:'sample_id'})
		
		drop = None
		if jackknife is not None:
			event_split = events.sample(frac=1, random_state=seed).reset_index(drop=True)
			event_drop = np.array_split(event_split, int(jackknife[0]))
			if int(jackknife[1]) == 0:
				raise ValueError("Jackknife splits need to be base 1, i.e. 1-10 for 10 splits not 0-9")
			drop = event_drop[(int(jackknife[1])-1)]
			events = events[~events.sample_id.isin(drop.sample_id)]

		grm_names["rownum"] = grm_names.index
		grm_names["real_id"] = grm_names.sample_id

		grm_names = grm_names.set_index('sample_id')
		events = events.set_index('sample_id')
		events = pd.concat([grm_names, events], axis=1, join='inner')
		
		self.N = events.shape[0]

		events = events[(events.stop > events.start)]
		if events.shape[0] != self.N:
			print(str(self.N - events.shape[0]) + " individuals dropped because start time is after end time")
			self.N = events.shape[0]

		events = events.sort_values("stop", ascending=True).reset_index(drop=True)
		events = events.rename(columns= {'real_id':'sample_id'})
		events = events.set_index('sample_id')
		return events

#!/usr/bin/env python3
###
import sys,os,time,argparse,logging
import pyreadr,pickle
import numpy as np
import pandas as pd
import networkx as nx

from ProteinGraphML.DataAdapter import OlegDB,selectAsDF
from ProteinGraphML.GraphTools import ProteinDiseaseAssociationGraph
from ProteinGraphML.MLTools.MetapathFeatures import metapathFeatures,ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode,getMetapaths 
from ProteinGraphML.MLTools.Data import BinaryLabel
from ProteinGraphML.MLTools.Models import XGBoostModel
from ProteinGraphML.MLTools.Procedures import *
from ProteinGraphML.Analysis import Visualize

t0 = time.time()
 
PROCEDURES = ["XGBPredict", "SVMPredict"]

parser = argparse.ArgumentParser(description='Run ML Procedure', epilog='--file must be specified; available procedures: {0}'.format(str(PROCEDURES)))
parser.add_argument('procedure', choices=PROCEDURES, help='ML procedure to run')
parser.add_argument('--testfile', help='input file, pickled test data, e.g. "diabetesTestData.pkl"')
parser.add_argument('--modelfile', help='ML model file full path')
parser.add_argument('--resultdir', help='folder where results will be saved, e.g. "diabetes_no_lincs"')
parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

args = parser.parse_args()

logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if args.verbose>1 else logging.INFO))

#get test data from the file
if args.testfile is None:
	parser.error("--test data file must be specified.")
else:
	try:
		with open(args.testfile, 'rb') as f:
			testData = pickle.load(f)
	except:
		logging.error('Failed to open pickled test data file {0}'.format(args.testfile)) 
		exit()

#Get ML procedure
Procedure = args.procedure
logging.info('Procedure: {0}'.format(Procedure))

# directory and file name for the ML Model
if (args.modelfile is None):
	logging.error("--modelfile required.")
	exit()
else:
	logging.info("Model '{0}' will be used for prediction".format(args.modelfile))

#Get reult directory 
if (args.resultdir is not None):
	logging.info('Results will be saved in directory: {0}'.format('results/'+args.resultdir))
else:
	logging.error('Result directory is needed')
	exit()

#fetch the description of proteins and pathway_ids
dbAdapter = OlegDB()
idDescription = dbAdapter.fetchPathwayIdDescription() #fetch the description
idNameSymbol = dbAdapter.fetchSymbolForProteinId() #fetch name and symbol for protein


#call ML codes
d = BinaryLabel()
d.loadTestData(testData) 

#print('calling function...', locals()[Procedure])
locals()[Procedure](d, idDescription, idNameSymbol, args.modelfile, args.resultdir)

logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

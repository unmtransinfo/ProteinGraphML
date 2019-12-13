#!/usr/bin/env python3
###
import sys,os,time,argparse,logging,yaml
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

NUM_OF_ROUNDS = 10
RSEED = 1234
NTHREADS = 1
PROCEDURES = ["XGBCrossValPred", "XGBKfoldsRunPred", "XGBGridSearch"]
XGB_PARAMETERS_FILE = 'XGBparams.txt'

parser = argparse.ArgumentParser(description='Run ML Procedure', epilog='--disease or --file must be specified; available procedures: {0}'.format(str(PROCEDURES)))
parser.add_argument('procedure', choices=PROCEDURES, help='ML procedure to run')
parser.add_argument('--trainingfile', help='input file, pickled training data, e.g. "diabetesTrainData.pkl"')
parser.add_argument('--resultdir', help='folder where results will be saved, e.g. "diabetes_no_lincs"')
parser.add_argument('--rseed', type=int, default=RSEED, help='random seed for XGboost')
parser.add_argument('--nthreads', type=int, default=NTHREADS, help='Number of CPU threads for GridSearch')
parser.add_argument('--nrounds_for_avg', type=int, default=NUM_OF_ROUNDS, help='number of iterations for average AUC,ACC,MCC (default: "{0}")'.format(NUM_OF_ROUNDS))
parser.add_argument('--xgboost_param_file', default=XGB_PARAMETERS_FILE, help='text file containing parameters for XGBoost classifier (e.g. XGBparams.txt)')
parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

args = parser.parse_args()
#argData = vars(parser.parse_args())

logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if args.verbose >1 else logging.INFO))

#Get data from file 
trainingDataFile = args.trainingfile

if trainingDataFile is None:
	parser.error("--trainingfile must be specified.")
else:
	try:
		with open(trainingDataFile, 'rb') as f:
			trainData = pickle.load(f)
	except:
		logging.error('Must generate pickled training data file') 
		exit()

Procedure = args.procedure
logging.info('Procedure: {0}'.format(Procedure))

#Get reult directory and number of folds
if (args.resultdir is not None):
	resultDir = args.resultdir #folder where all results will be stored
	logging.info('Results will be saved in directory: {0}'.format(resultDir))
else:
	logging.error('Result directory is needed')
	exit()
#nfolds = args.nrounds_for_avg # applicable for average CV

#fetch the parameters for XGboost from the text file
paramVals = ""
with open(args.xgboost_param_file, 'r') as fh:
	for line in fh:
		paramVals+=line.strip().strip(' ')
xgbParams = yaml.full_load(paramVals)


#fetch the description of proteins and pathway_ids
dbAdapter = OlegDB()
idDescription = dbAdapter.fetchPathwayIdDescription() #fetch the description
idNameSymbol = dbAdapter.fetchSymbolForProteinId() #fetch name and symbol for protein
 
#call ML codes
d = BinaryLabel()
d.loadData(trainData)
if (Procedure == "XGBKfoldsRunPred"):
	locals()[Procedure](d, idDescription, idNameSymbol, resultDir, args.nrounds_for_avg, params=xgbParams)
elif (Procedure == "XGBCrossValPred"):
	locals()[Procedure](d, idDescription, idNameSymbol, resultDir, params=xgbParams)
elif (Procedure == "XGBGridSearch"):
	locals()[Procedure](d, idDescription, idNameSymbol, resultDir, args.rseed, args.nthreads)
else:
	logging.error('Wrong procedure entered !!!')
logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

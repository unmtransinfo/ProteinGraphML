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

NUM_OF_FOLDS = 2
PROCEDURES = ["XGBCrossVal", "XGBCrossValPred"]

parser = argparse.ArgumentParser(description='Run ML Procedure', epilog='--disease or --file must be specified; available procedures: {0}'.format(str(PROCEDURES)))
parser.add_argument('procedure', metavar='procedure', type=str, choices=PROCEDURES, nargs='+', help='ML procedure to run')
parser.add_argument('--trainingDataFile', type=str, nargs='?', help='input file, pickled training data, e.g. "diabetesTrainData.pkl"')
parser.add_argument('--resultdir', type=str, nargs='?', help='folder where results will be saved, e.g. "diabetes_no_lincs"')
parser.add_argument('--crossval_folds', type=int, default=NUM_OF_FOLDS, help='number of folds for average CV (default: "{0}")'.format(NUM_OF_FOLDS))
parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

argData = vars(parser.parse_args())

logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if argData['verbose']>1 else logging.INFO))

#Get data from file 
trainingDataFile = argData['trainingDataFile']

if trainingDataFile is None:
	parser.error("--trainingDataFile must be specified.")
else:
	try:
		with open(trainingDataFile, 'rb') as f:
			trainData = pickle.load(f)
	except:
		logging.error('Must generate pickled training data file') 
		exit()

Procedure = argData['procedure'][0]
logging.info('Procedure: {0}'.format(Procedure))

#Get reult directory and number of folds
if (argData['resultdir'] is not None):
	resultDir = argData['resultdir'] #folder where all results will be stored
	logging.info('Results will be saved in directory: {0}'.format(resultDir))
else:
	logging.error('Result directory is needed')
	exit()
nfolds = argData['crossval_folds'] # applicable for average CV

#fetch the description of proteins and pathway_ids
dbAdapter = OlegDB()
idDescription = dbAdapter.fetchPathwayIdDescription() #fetch the description
idNameSymbol = dbAdapter.fetchSymbolForProteinId() #fetch name and symbol for protein
 
#call ML codes
d = BinaryLabel()
d.loadData(trainData)
locals()[Procedure](d, idDescription, idNameSymbol, resultDir, nfolds)

logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

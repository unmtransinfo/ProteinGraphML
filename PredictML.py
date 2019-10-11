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

DATA_DIR = os.getcwd() + '/DataForML/'   
PROCEDURES = ["XGBPredict"] 

parser = argparse.ArgumentParser(description='Run ML Procedure', epilog='--file must be specified; available procedures: {0}'.format(str(PROCEDURES)))
parser.add_argument('procedure', metavar='procedure', type=str, choices=PROCEDURES, nargs='+', help='ML procedure to run')
parser.add_argument('--dir', default=DATA_DIR, help='input dir (default: "{0}")'.format(DATA_DIR))
parser.add_argument('--testdatafile', type=str, nargs='?', help='input file, pickled test data, e.g. "diabetesTestData.pkl"')
parser.add_argument('--model', type=str, nargs='?', help='ML model name with full path')
parser.add_argument('--resultdir', type=str, nargs='?', help='folder where results will be saved, e.g. "diabetes_no_lincs"')
#parser.add_argument('--kgfile', default=DEFAULT_GRAPH, help='input pickled KG (default: "{0}")'.format(DEFAULT_GRAPH))
#parser.add_argument('--static_data', default=DEFAULT_STATIC_FEATURES, help='(default: "{0}")'.format(DEFAULT_STATIC_FEATURES))
parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

argData = vars(parser.parse_args())

logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if argData['verbose']>1 else logging.INFO))

#get test data from the file
fileName = argData['testdatafile']
if fileName is None:
	parser.error("--test data file must be specified.")
else:
	pklFile = argData['dir'] + fileName
	try:
		with open(pklFile, 'rb') as f:
			testData = pickle.load(f)
	except:
		logging.error('Must generate pickled test data file') 
		exit()

#Get ML procedure
Procedure = argData['procedure'][0]
logging.info('Procedure: {0}'.format(Procedure))

# directory and file name for the ML Model
if (argData['model'] is None):
	logging.error("Model name not entered")
	exit()
else:
	modelName = argData['model']
	logging.info("Model '{0}' will be used for prediction".format(modelName))

#Get reult directory 
if (argData['resultdir'] is not None):
	resultDir = argData['resultdir'] #folder where all results will be stored
	logging.info('Results will be saved in directory: {0}'.format('results/'+resultDir))
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
locals()[Procedure](d, idDescription, idNameSymbol, modelName, resultDir)

logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

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

dataDir = os.getcwd() + '/DataForML/'  #IMPORTANT: change it if you have saved pkl files in a different folder

DEFAULT_GRAPH = "ProteinDisease_GRAPH.pkl"
DEFAULT_STATIC_FEATURES = "gtex,lincs,ccle,hpa"

PROCEDURES = ["XGBPredict"]

parser = argparse.ArgumentParser(description='Run ML Procedure', epilog='--disease or --file must be specified; available procedures: {0}'.format(str(PROCEDURES)))
parser.add_argument('procedure', metavar='procedure', type=str, choices=PROCEDURES, nargs='+', help='ML procedure to run')
parser.add_argument('--disease', metavar='disease', type=str, nargs='?', help='Mammalian Phenotype ID, e.g. MP_0000180')
parser.add_argument('--file', type=str, nargs='?', help='input file, pickled training set, e.g. "diabetes.pkl"')
parser.add_argument('--dir', default=dataDir, help='input dir (default: "{0}")'.format(dataDir))
parser.add_argument('--kgfile', default=DEFAULT_GRAPH, help='input pickled KG (default: "{0}")'.format(DEFAULT_GRAPH))
parser.add_argument('--static_data', default=DEFAULT_STATIC_FEATURES, help='(default: "{0}")'.format(DEFAULT_STATIC_FEATURES))
parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

argData = vars(parser.parse_args())

logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if argData['verbose']>1 else logging.INFO))

disease = argData['disease']
fileName = argData['file']
fileData = None


if disease is None and fileName is None: # NO INPUT
	parser.error("--disease or --file must be specified.")
elif disease is None and fileName is not None: # NO disease, use file
	pklFile = argData['dir'] + fileName
	diseaseName = fileName
	try:
		with open(pklFile, 'rb') as f:
			fileData = pickle.load(f)
	except:
		logging.error('Must generate pickled training set file for the given disease') 
		exit()

elif fileName is None and disease is not None:
	logging.info("running on this disease: {0}".format(disease))
	diseaseName = disease
else:
	logging.error('Wrong parameters passed')

Procedure = argData['procedure'][0]
logging.info('Procedure: {0}'.format(Procedure))

graphString = argData['kgfile']

# Current graph
currentGraph = ProteinDiseaseAssociationGraph.load(graphString)
logging.info("GRAPH {0} LOADED".format(graphString))

nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]

staticFeatures = argData['static_data'].split(',')
logging.info(staticFeatures)
#staticFeatures = []

logging.info("--- USING {0} METAPATH FEATURE SETS".format(len(nodes)))
logging.info("--- USING {0} STATIC FEATURE SETS".format(len(staticFeatures)))


#fetch the description of proteins and pathway_ids
dbAdapter = OlegDB()
idDescription = dbAdapter.fetchPathwayIdDescription() #fetch the description
idNameSymbol = dbAdapter.fetchSymbolForProteinId() #fetch name and symbol for protein

if fileData is not None:
	#logging.info("FOUND {0} POSITIVE LABELS".format(len(fileData[True])))
	#logging.info("FOUND {0} NEGATIVE LABELS".format(len(fileData[False])))
	testData = metapathFeatures(disease,currentGraph,nodes,idDescription,staticFeatures,loadedLists=fileData).fillna(0) 
else:
	loggin.error("Test data for prediction not provided")

#call ML codes
d = BinaryLabel()
d.loadData(testData)

#print('calling function...', locals()[Procedure])
locals()[Procedure](d, idDescription, idNameSymbol, diseaseName)

logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

#!/usr/bin/env python3
###
import sys,os,time,argparse,logging
import pyreadr,pickle
import numpy as np
import pandas as pd
import networkx as nx

from ProteinGraphML.DataAdapter import OlegDB,selectAsDF
from ProteinGraphML.GraphTools import ProteinDiseaseAssociationGraph
from ProteinGraphML.MLTools.MetapathFeatures import metapathFeatures,ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode,getMetapaths,getTrainingProteinIds 
from ProteinGraphML.MLTools.Data import BinaryLabel
from ProteinGraphML.MLTools.Models import XGBoostModel
from ProteinGraphML.MLTools.Procedures import *


def savePickleObject(fileName, data):
	'''
	This function saves data into a pickle file
	'''
	with open(fileName, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def saveTrainTestSet(allData):
	'''
	This function saves training and test in pickle format 
	'''
	logging.info('Number of rows and features in allData: {0}'.format(allData.shape))
	if (disease is not None):
		pklTrainFile = outputDir + '/' + disease + '_TrainingData.pkl'
		pklTestFile = outputDir + '/' + disease + '_TestData.pkl'
		
		# extract train data from the dataframe
		trainData = allData.loc[allData['Y'].isin([0,1])]
		logging.info('Number of rows and features in training data: {0}'.format(trainData.shape))
		logging.info("Writing train data to file: {0}".format(pklTrainFile))
		savePickleObject(pklTrainFile, trainData)
		#print (trainData)
		
		# extract test data from the dataframe
		testData = allData.loc[allData['Y'] == -1]
		#testData = testData.drop('Y', axis=1) #drop label from the test data
		logging.info('Number of rows and features in test data: {0}'.format(testData.shape))
		logging.info("Writing test data to file: {0}".format(pklTestFile))
		savePickleObject(pklTestFile, testData)
		#print (testData)
	
	elif (testfile is None and trainingfile is not None):
		pklTrainFile = outputDir + '/' + os.path.basename(trainingfile).split('.')[0] + '_TrainingData.pkl'
		logging.info("Writing train data to file: {0}".format(pklTrainFile))
		savePickleObject(pklTrainFile, allData)

	elif (testfile is not None and trainingfile is not None):
		pklTrainFile = outputDir + '/' + os.path.basename(trainingfile).split('.')[0] + '_TrainingData.pkl'
		pklTestFile = outputDir + '/' + os.path.basename(testfile).split('.')[0] + '_TestData.pkl'

		# extract train data from the dataframe
		trainData = allData.loc[allData['Y'].isin([0,1])]
		logging.info('Number of rows and features in training data: {0}'.format(trainData.shape))
		logging.info("Writing train data to file: {0}".format(pklTrainFile))
		savePickleObject(pklTrainFile, trainData)
		#print (trainData)
		
		# extract test data from the dataframe
		testData = allData.loc[allData['Y'] == -1]
		#testData = testData.drop('Y', axis=1) #drop label from the test data
		logging.info('Number of rows and features in test data: {0}'.format(testData.shape))
		logging.info("Writing test data to file: {0}".format(pklTestFile))
		savePickleObject(pklTestFile, testData)
		#print (testData)
	else:
		logging.error('Missing argument(s)')
	
 
############### START OF THE CODE ##############################
t0 = time.time()

DATA_DIR = os.getcwd() + '/DataForML/'
DEFAULT_GRAPH = "ProteinDisease_GRAPH.pkl"
DEFAULT_STATIC_FEATURES = "gtex,lincs,ccle,hpa"

parser = argparse.ArgumentParser(description='Generate features for training and test set', epilog='Protein Ids with True label must be provided')
parser.add_argument('--disease', metavar='disease', type=str, nargs='?', help='Mammalian Phenotype ID, e.g. MP_0000180')
parser.add_argument('--trainingfile', type=str, nargs='?', help='pickled training set, e.g. "diabetes.pkl"')
parser.add_argument('--testfile', type=str, nargs='?', help='pickled test set, e.g. "diabetes_test.pkl"')
parser.add_argument('--outputdir', default=DATA_DIR, type=str, nargs='?', help='directory where train and test data with features will be saved, e.g. "diabetes_no_lincs"')
parser.add_argument('--kgfile', default=DEFAULT_GRAPH, help='input pickled KG (default: "{0}")'.format(DEFAULT_GRAPH))
parser.add_argument('--static_data', default=DEFAULT_STATIC_FEATURES, help='(default: "{0}")'.format(DEFAULT_STATIC_FEATURES))
parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

argData = vars(parser.parse_args())

logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if argData['verbose']>1 else logging.INFO))

#Get data from file or disease
disease = argData['disease']
trainingfile = argData['trainingfile']
testfile = argData['testfile']
fileData = None

#folder where train and test data with features will be stored
outputDir = argData['outputdir'] 

#check whether file or disease was given
if (trainingfile is None and disease is None): 
	parser.error("--disease or -- training file must be specified.")

#fetch KG data
graphString = argData['kgfile']
currentGraph = ProteinDiseaseAssociationGraph.load(graphString)
logging.info("GRAPH {0} LOADED".format(graphString))

#Access the adapter
dbAdapter = OlegDB()

if (trainingfile is not None and disease is None): 
	trainingPklFile = trainingfile
	logging.info('Input training file: {0}'.format(trainingPklFile))
	try:
		with open(trainingPklFile, 'rb') as f:
			fileData = pickle.load(f)
	except:
		logging.error('Invalid pickled training set file') 
		exit()

	#Also add test data if provided
	if (testfile is not None):
		testPklFile = testfile
		logging.info('Input test file: {0}'.format(testPklFile))
		try:
			with open(testPklFile, 'rb') as f:
				fileData.update(pickle.load(f)) #fileData will now have both train and test set
		except:
			logging.error('Invalid pickled test set file') 
			exit()	    		
elif (trainingfile is None and disease is not None): 
	logging.info("running on this disease: {0}".format(disease))
	fullData = {}
	#get positive and negative training protein ids
	trainP, trainF = getTrainingProteinIds(disease,currentGraph) 
	fullData[True] = trainP
	fullData[False] = trainF
	#get all protein ids
	allProteinIds = dbAdapter.fetchAllProteinIds()
	allProteinIds = set(allProteinIds['protein_id'].tolist())
	#prepare test set
	testProteinSet = allProteinIds.difference(trainP)
	testProteinSet = testProteinSet.difference(trainF)
	fullData['unknown'] = testProteinSet
	fileData = fullData
else:
	logging.error('Wrong parameters passed')	


#Nodes
nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]

#Static features that need to be included
staticFeatures = argData['static_data'].split(',')
#staticFeatures = []
logging.info(staticFeatures)
logging.info("--- USING {0} METAPATH FEATURE SETS".format(len(nodes)))
logging.info("--- USING {0} STATIC FEATURE SETS".format(len(staticFeatures)))

#fetch the description of proteins
idDescription = dbAdapter.fetchPathwayIdDescription() #fetch the description

#generate features
if fileData is not None:
	#logging.info("FOUND {0} POSITIVE LABELS".format(len(fileData[True])))
	#logging.info("FOUND {0} NEGATIVE LABELS".format(len(fileData[False])))
	allData = metapathFeatures(disease,currentGraph,nodes,idDescription,staticFeatures,loadedLists=fileData).fillna(0) 
else:
	#allData = metapathFeatures(disease,currentGraph,nodes,idDescription,staticFeatures).fillna(0)
	logging.error('fileData should not be None')
	exit()
	

#Divide allData into training/test set and save them
saveTrainTestSet(allData)

logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))

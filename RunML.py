#!/usr/bin/env python3
###
import sys
import argparse
import pyreadr
import numpy as np
import os

from ProteinGraphML.DataAdapter import OlegDB,selectAsDF
import networkx as nx
from ProteinGraphML.GraphTools import ProteinDiseaseAssociationGraph
from ProteinGraphML.Analysis import Visualize
import pandas as pd
from ProteinGraphML.MLTools.MetapathFeatures import metapathFeatures,ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode
from ProteinGraphML.MLTools.MetapathFeatures import getMetapaths
import pickle
from ProteinGraphML.MLTools.Data import BinaryLabel
from ProteinGraphML.MLTools.Models import XGBoostModel
from ProteinGraphML.MLTools.Procedures import *
from ProteinGraphML.DataAdapter import OlegDB

#path_to_files = '/home/pkumar/ITDC/ProteinGraphML/DataForML/'  #IMPORTANT: change it if you have saved pkl files in a different folder
path_to_files = os.getcwd() + '/DataForML/'  #IMPORTANT: change it if you have saved pkl files in a different folder

#DEFAULT_GRAPH = "newCURRENT_GRAPH"
DEFAULT_GRAPH = "ProteinDisease_GRAPH.pkl"

parser = argparse.ArgumentParser(description='Run ML Procedure')

##parser.add_argument('disease', metavar='disease', type=str, nargs='+',
# #                   help='phenotype')

parser.add_argument('procedure', metavar='procedure', type=str, nargs='+', help='ML to run')
parser.add_argument('--file', type=str, nargs='?', help='input file, pickled training set')
parser.add_argument('--dir', default=path_to_files, help='input dir')
parser.add_argument('--disease', metavar='disease', type=str, nargs='?', help='Mammalian Phenotype ID, e.g. MP_0000180')
parser.add_argument('--kgfile', default=DEFAULT_GRAPH, help='input pickled KG')
#parser.add_argument('--disease', metavar='disease', type=str, nargs='?',help='phenotype')


argData = vars(parser.parse_args())

#print(argData)
#print(argData['procedure'][0])

disease = argData['disease']
fileName = argData['file']
fileData = None


if disease is None and fileName is None: # NO INPUT
	print("--disease or --file must be specified")
	exit()
elif disease is None and fileName is not None: # NO disease, use file
	pklFile = argData['dir'] + fileName
	diseaseName = fileName
	try:
		with open(pklFile, 'rb') as f:
			fileData = pickle.load(f)
	except:
		print ('ERROR: Must generate pickled training set file for the given disease')
		exit()
    		
	#def load_obj(name):
	#with open(pklFile, 'rb') as f:
	#	fileData = pickle.load(f)
	#loadList = load_obj('nextDataset')
elif fileName is None and disease is not None:
	print("running on this disease",disease)
	diseaseName = disease
else:
	print ('Wrong parameters passed')
#if fileName is not None and disease is not None:
#	print("file and disease detected, will use disease string {0}".format(disease))
#	print("running on this disease",disease)

print("")

# CANT FIND THIS DISEASE
#disease = sys.argv[1]
Procedure = argData['procedure'][0]
print('Procedure', Procedure)

graphString = argData['kgfile']


# CANT FIND THIS GRAPH
currentGraph = ProteinDiseaseAssociationGraph.load(graphString)
# SOME DISEASES CAUSE "DIVIDE BY 0 error"
print("GRAPH {0} LOADED".format(graphString))

nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]
staticFeatures = []

print("--- USING {0} METAPATH FEATURE SETS".format(len(nodes)))
print("--- USING {0} STATIC FEATURE SETS".format(len(staticFeatures)))


#fetch the description of proteins and pathway_ids
dbAdapter = OlegDB()
idDescription = dbAdapter.fetchPathwayIdDescription() #fetch the description

if fileData is not None:
	#print("FOUND {0} POSITIVE LABELS".format(len(fileData[True])))
	#print("FOUND {0} NEGATIVE LABELS".format(len(fileData[False])))
	trainData = metapathFeatures(disease,currentGraph,nodes,idDescription,staticFeatures,loadedLists=fileData).fillna(0) 
else:
	trainData = metapathFeatures(disease,currentGraph,nodes,idDescription,staticFeatures).fillna(0)

d = BinaryLabel()
d.loadData(trainData)
#XGBCrossVal(d)
#print('calling function...', locals()[Procedure])
locals()[Procedure](d,idDescription,diseaseName)


#print("FEATURES CREATED, STARTING ML")
#d = BinaryLabel()
#d.loadData(trainData)
#newModel = XGBoostModel()
#print("SHAPE",d.features.shape)
#roc,acc,CM,report = newModel.cross_val_predict(d,["roc","acc","ConfusionMatrix","report"]) #"report","roc","rocCurve","ConfusionMatrix"
#roc.printOutput()



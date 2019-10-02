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


parser = argparse.ArgumentParser(description='Run ML Procedure')

##parser.add_argument('disease', metavar='disease', type=str, nargs='+',
# #                   help='phenotype')

parser.add_argument('procedure', metavar='procedure', type=str, nargs='+',
					help='ML to run')
parser.add_argument('--file', type=str, nargs='?', help='some help')
parser.add_argument('--disease', metavar='disease', type=str, nargs='?',help='phenotype')
#parser.add_argument('--disease', metavar='disease', type=str, nargs='?',help='phenotype')


argData = vars(parser.parse_args())

#print(argData)
#print(argData['procedure'][0])

disease = argData['disease']
file = argData['file']
fileData = None
#path_to_files = '/home/pkumar/ITDC/ProteinGraphML/DataForML/'  #IMPORTANT: change it if you have saved pkl files in a different folder
path_to_files = os.getcwd() + '/DataForML/'  #IMPORTANT: change it if you have saved pkl files in a different folder


if disease is None and file is None: # NO INPUT
	print("disease or file must be specified")
	exit()
elif disease is None and file is not None: # NO disease, use file
	pklFile = path_to_files + file + '.pkl'
	diseaseName = file
	try:
		with open(pklFile, 'rb') as f:
			fileData = pickle.load(f)
	except:
		print ('Run generate_pkl_dict.py to generate the pickle dictionary file for the given disease')
		exit()
    		
	#def load_obj(name):
	#with open(pklFile, 'rb') as f:
	#	fileData = pickle.load(f)
	#loadList = load_obj('nextDataset')
elif file is None and disease is not None:
	print("running on this disease",disease)
	diseaseName = disease
else:
	print ('Wrong parameters passed')
#if file is not None and disease is not None:
#	print("file and disease detected, will use disease string {0}".format(disease))
#	print("running on this disease",disease)

print("")
DEFAULT_GRAPH = "newCURRENT_GRAPH"

# CANT FIND THIS DISEASE
#disease = sys.argv[1]
Procedure = argData['procedure'][0]
print('Procedure', Procedure)

graphString = None

graphString = DEFAULT_GRAPH


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


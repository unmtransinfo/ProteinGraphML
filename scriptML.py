import sys
import argparse

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


parser = argparse.ArgumentParser(description='Run ML Procedure')
parser.add_argument('disease', metavar='disease', type=str, nargs='+',
                    help='phenotype')
parser.add_argument('procedure', metavar='procedure', type=str, nargs='+',
                    help='ML to run')



argData = vars(parser.parse_args())

print(argData)

print(argData['procedure'][0])

disease = argData['disease'][0]

DEFAULT_GRAPH = "newCURRENT_GRAPH"

# CANT FIND THIS DISEASE
#disease = sys.argv[1]
Procedure = argData['procedure'][0]

graphString = None

graphString = DEFAULT_GRAPH


# CANT FIND THIS GRAPH
currentGraph = ProteinDiseaseAssociationGraph.load(graphString)
# SOME DISEASES CAUSE "DIVIDE BY 0 error"
print("GRAPH {0} LOADED".format(graphString))

nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]
staticFeatures = []

print("CREATING Metapath FEATURES - USING {0} METAPATH FEATURE SETS".format(len(nodes)))
print("CREATING Static FEATURES - USING {0} STATIC FEATURE SETS".format(len(staticFeatures)))
trainData = metapathFeatures(disease,currentGraph,nodes,staticFeatures).fillna(0)
d = BinaryLabel()
d.loadData(trainData)
#XGBCrossVal(d)
locals()[Procedure](d)


#print("FEATURES CREATED, STARTING ML")
#d = BinaryLabel()
#d.loadData(trainData)
#newModel = XGBoostModel()
#print("SHAPE",d.features.shape)
#roc,acc,CM,report = newModel.cross_val_predict(d,["roc","acc","ConfusionMatrix","report"]) #"report","roc","rocCurve","ConfusionMatrix"
#roc.printOutput()



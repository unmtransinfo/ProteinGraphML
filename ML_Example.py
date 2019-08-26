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


# CANT FIND THIS DISEASE
disease = "MP_0000180"

graphString = None
fileData = None

graphString = "newCURRENT_GRAPH"
# CANT FIND THIS GRAPH
currentGraph = ProteinDiseaseAssociationGraph.load(graphString)
# SOME DISEASES CAUSE "DIVIDE BY 0 error"
print("GRAPH {0} LOADED".format(graphString))

nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]
staticFeatures = [] # ALL OPTIONS HERE... ["gtex","lincs","hpa","ccle"]

print("--- USING {0} METAPATH FEATURE SETS".format(len(nodes)))
print("--- USING {0} STATIC FEATURE SETS".format(len(staticFeatures)))

if fileData is not None:
    #print("FOUND {0} POSITIVE LABELS".format(len(fileData[True])))
    #print("FOUND {0} NEGATIVE LABELS".format(len(fileData[False])))
    trainData = metapathFeatures(disease,currentGraph,nodes,staticFeatures,loadedLists=fileData).fillna(0)
else:
    trainData = metapathFeatures(disease,currentGraph,nodes,staticFeatures).fillna(0)

d = BinaryLabel()
d.loadData(trainData)
XGBCrossVal(d)






import sys

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



DEFAULT_GRAPH = "newCURRENT_GRAPH"

# CANT FIND THIS DISEASE
disease = sys.argv[1]

graphString = None
if len(sys.argv) > 2:
	graphString = sys.argv[2]
	print(graph)
	if graphString is None:
		graphString = DEFAULT_GRAPH
else:
	graphString = DEFAULT_GRAPH


# CANT FIND THIS GRAPH
currentGraph = ProteinDiseaseAssociationGraph.load(graphString)
# SOME DISEASES CAUSE "DIVIDE BY 0 error"
print("load this graph {0}".format(len(currentGraph.graph)))

nodes = [KeggNode] #,ReactomeNode,GoNode,InterproNode]

trainData = metapathFeatures(disease,currentGraph,nodes,[]).fillna(0)

d = BinaryLabel()
d.loadData(trainData)

newModel = XGBoostModel()
roc,acc,CM,report = newModel.cross_val_predict(d,["roc","acc","ConfusionMatrix","report"]) #"report","roc","rocCurve","ConfusionMatrix"
roc.printOutput()
importance = newModel.average_cross_val(d,[])

for g in importance.most_common(3):
	print("PRINTING THIS IMPORTANT FEATURES- {0}".format(Disease),g)
	Visualize(g,currentGraph.graph,Disease)



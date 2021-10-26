#!/usr/bin/env python3
###
import sys, os, logging, time
import argparse
import pandas as pd
import pickle
import networkx as nx
from pony.orm import *

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

logging.info("Python version: {0}".format(sys.version.split()[0]))
logging.info("Pandas version: {0}".format(pd.__version__))
logging.info("NetworkX version: {0}".format(nx.__version__))

from proteingraphml.DataAdapter import OlegDB,selectAsDF
from proteingraphml.GraphTools import ProteinDiseaseAssociationGraph
from proteingraphml.MLTools.MetapathFeatures import metapathFeatures,ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode
from proteingraphml.MLTools.Data import BinaryLabel
from proteingraphml.MLTools.Procedures import * #XGBCrossVal

t0 = time.time()

dbAdapter = OlegDB()

# CANT FIND THIS DISEASE(?)
# disease = "MP_0000180"
disease = "MP_0000184"
with db_session:
  dname = dbAdapter.db.get("SELECT name FROM mp_onto WHERE mp_term_id = '"+disease+"'")
  logging.info("disease: {0}: \"{1}\"".format(disease, dname))

fileData = None

pickleFile = "ProteinDisease_GRAPH.pickle"
# CANT FIND THIS GRAPH(?)
currentGraph = ProteinDiseaseAssociationGraph.load(pickleFile)

# SOME DISEASES CAUSE "DIVIDE BY 0 error"
logging.info("GRAPH LOADED: {0}".format(pickleFile))

nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]
staticFeatures = [] # ALL OPTIONS HERE... ["gtex","lincs","hpa","ccle"]

logging.info("USING {0} METAPATH FEATURE SETS".format(len(nodes)))
logging.info("USING {0} STATIC FEATURE SETS".format(len(staticFeatures)))

if fileData is not None:
    logging.info("FOUND {0} POSITIVE LABELS".format(len(fileData[True])))
    logging.info("FOUND {0} NEGATIVE LABELS".format(len(fileData[False])))
    trainData = metapathFeatures(disease, currentGraph, nodes, staticFeatures, loadedLists=fileData).fillna(0)
else:
    trainData = metapathFeatures(disease, currentGraph, nodes, staticFeatures).fillna(0)

d = BinaryLabel()
d.loadData(trainData)

# PK sorry if this is wrong -- feel free to fix!
idDescription = dbAdapter.fetchPathwayIdDescription()
XGBCrossVal(d, idDescription)

logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))


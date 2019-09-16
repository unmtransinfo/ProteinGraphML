#!/usr/bin/env python3
"""
	Create a Protein Disease graph from the DB adapter 'OlegDB'
"""
import sys, os, time
import logging
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(filename='Build_Graph_Example.log')

from ProteinGraphML.DataAdapter import OlegDB
from ProteinGraphML.GraphTools import ProteinDiseaseAssociationGraph

t0 = time.time()

## Construct base map of protein-to-disease by creating the ProteinDiseaseAssociationGraph.
## Db is PonyORM Database (https://docs.ponyorm.org/api_reference.html#api-reference).

dbAdapter = OlegDB()

pdg = ProteinDiseaseAssociationGraph(dbAdapter)

## The 'ProteinDiseaseAssociationGraph' object has helper methods, but 
## the networkx Graph methods also available.
## https://networkx.github.io/documentation/stable/reference/

logging.info('Total nodes: %d; edges: %d'%(len(pdg.graph.nodes), len(pdg.graph.edges)))

## Filter by proteins of interest; this list comes from a DB adapter, but any set will do.
proteins = dbAdapter.loadTotalProteinList().protein_id
filterByProteins = set(proteins)
logging.info('Protein list: %d'%(len(filterByProteins)))

# Using attach() will add edges from a DB as defined by the adapter.
# With this method create a graph of data, which can itself be saved, prevents the
# need for rebuilding as we work on different diseases, perform analysis.
# Also filter by proteins of interest, in this case it is our original list.

pdg.attach(dbAdapter.loadPPI(filterByProteins))
pdg.attach(dbAdapter.loadKegg(filterByProteins)) 
pdg.attach(dbAdapter.loadReactome(filterByProteins)) 
pdg.attach(dbAdapter.loadInterpro(filterByProteins))
pdg.attach(dbAdapter.loadGo(filterByProteins))

# networkx provides an api we can nodes from \n",
# here i exploit the unique features of each node to count them\n",
# we can get a count of the nodes in the current graph

keggNodes = [n for n in list(pdg.graph.nodes)
	if isinstance(n, str) and n[0:3]=="hsa"]
reactome = [n for n in list(pdg.graph.nodes)
	if isinstance(n, str) and n[0:2]=="R-"]
goNodes = [n for n in list(pdg.graph.nodes)
	if isinstance(n, str) and n[0:3]=="GO:"]
interNodes = [n for n in list(pdg.graph.nodes)
	if isinstance(n, str) and n[0:3]=="IPR"]

logging.info("KEGG nodes: %d"%(len(keggNodes)))
logging.info("REACT nodes: %d"%(len(reactome)))
logging.info("GO nodes: %d"%(len(goNodes)))
logging.info("INTERP nodes: %d"%(len(interNodes)))

# Save graph.
gfile="newCURRENT_GRAPH"
logging.info("Saving graph to {0}".format(gfile))
pdg.save(gfile)

# Fetch pathway information from db.
# (Not stored in graph?)
idDescription = dbAdapter.fetchPathwayIdDescription()

# Log node and edge info.
recCount = 0
logfile = 'graphData.log'
with open(logfile, 'w') as flog:
	allEdges = set(pdg.graph.edges)
	for edge in allEdges:
		recCount+=1
		try:
			flog.write('Edge ==>   '+'(' + str(edge[0])+':'+idDescription[edge[0]]+', '+str(edge[1])+':'+idDescription[edge[1]]+')'+'\n')
		except:
			logging.error('Name not found in the database: {0}'.format(edge))

logging.info('{0} records written to {1}'.format(recCount, logfile))

logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))


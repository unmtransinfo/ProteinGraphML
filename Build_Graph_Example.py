#!/usr/bin/env python3
"""
	Create a Protein Disease graph from the DB adapter 'OlegDB'
"""
import logging
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(filename='Build_Graph_Example.log')

from ProteinGraphML.DataAdapter import OlegDB
from ProteinGraphML.GraphTools import ProteinDiseaseAssociationGraph

## Construct base map of protein-to-disease by creating the ProteinDiseaseAssociationGraph.
## Db is PonyORM Database (https://docs.ponyorm.org/api_reference.html#api-reference).

dbAdapter = OlegDB()

proteinGraph = ProteinDiseaseAssociationGraph(dbAdapter)

## The 'ProteinDiseaseAssociationGraph' object has helper methods, but 
## the networkx Graph methods also available.
## https://networkx.github.io/documentation/stable/reference/

logging.info('Total nodes: %d; edges: %d'%(len(proteinGraph.graph.nodes), len(proteinGraph.graph.edges)))

## we will want to filter by the proteins we are interested in, this list comes from a DB adapter, but any set will do
proteins = dbAdapter.loadTotalProteinList().protein_id
filterByProteins = set(proteins)

# using .attach will add edges from a DB as defined by the adapter,
# with this method we can create a graph of data, which can itself be saved, prevents the
# need from, rebuilding as we work on different diseases, perform analysis
# We've also filter by proteins we care about, in this case it is our original list

proteinGraph.attach(dbAdapter.loadPPI(filterByProteins))
proteinGraph.attach(dbAdapter.loadKegg(filterByProteins)) 
proteinGraph.attach(dbAdapter.loadReactome(filterByProteins)) 
proteinGraph.attach(dbAdapter.loadInterpro(filterByProteins))
proteinGraph.attach(dbAdapter.loadGo(filterByProteins))

# networkx provides an api we can nodes from \n",
# here i exploit the unique features of each node to count them\n",
# we can get a count of the nodes in the current graph

keggNodes = [g for g in list(proteinGraph.graph.nodes) if isinstance(g,str) and g[0:3] == "hsa"]
reactome = [r for r in list(proteinGraph.graph.nodes) if isinstance(r,str) and r[0:2] == "R-"]
goNodes = [go for go in list(proteinGraph.graph.nodes) if isinstance(go,str) and go[0:3] == "GO:"]
interNodes = [inter for inter in list(proteinGraph.graph.nodes) if isinstance(inter,str) and inter[0:3] == "IPR"]

print("KEGG nodes", len(keggNodes))
print("REACT nodes", len(reactome))
print("GO nodes", len(goNodes))
print("INTERP nodes", len(interNodes))

# this will save our graph
proteinGraph.save("newCURRENT_GRAPH")

# create a log file to save the information about nodes and edges in the graphs
idDescription = dbAdapter.fetchPathwayIdDescription()
recCount = 0
with open('graphData.log', 'w') as fo:
	allEdges = set(proteinGraph.graph.edges)
	for edge in allEdges:
		recCount+=1
		try:
			line = 'Edge ==>   ' + '(' + str(edge[0]) + ':' + idDescription[edge[0]] + ', ' + str(edge[1]) + ':' + idDescription[edge[1]] + ')' + '\n'
			fo.write(line)
		except:
			print ('Name not found in the database', edge)
		if (recCount%1000000 ==0): #just to check the progess
			print ('{0} records written to the log file'.format(recCount))
			

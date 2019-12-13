#!/usr/bin/env python3
"""
	Create Protein Disease knowledge graph from DB adapter 'OlegDB' or 'TCRD'.
"""
import sys,os,argparse,time
import logging

from ProteinGraphML.DataAdapter import OlegDB, TCRD
from ProteinGraphML.GraphTools import ProteinDiseaseAssociationGraph

if __name__ == "__main__":

  DBS=['olegdb', 'tcrd']
  OPS=['build', 'test']
  parser = argparse.ArgumentParser(description='Create Protein-Disease knowledge graph (KG) from source RDB')
  parser.add_argument('operation', choices=OPS, help='{0}'.format(str(OPS)))
  parser.add_argument('--db', choices=DBS, default="olegdb", help='{0}'.format(str(OPS)))
  parser.add_argument('--o', dest="ofile", default="ProteinDisease_GRAPH.pkl", help='output pickled KG')
  parser.add_argument('--logfile', default="ProteinDisease_GRAPH.log", help='KG construction log')
  parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

  args = parser.parse_args()

  logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=(logging.DEBUG if args.verbose>1 else logging.INFO))

  t0 = time.time()

  ## Construct base protein-disease map from ProteinDiseaseAssociationGraph.
  ## Db is PonyORM db (https://docs.ponyorm.org/api_reference.html).

  dbAdapter = TCRD() if args.db == "tcrd" else OlegDB()

  pdg = ProteinDiseaseAssociationGraph(dbAdapter)

  ## ProteinDiseaseAssociationGraph object has helper methods, but 
  ## NetworkX methods also available.
  ## https://networkx.github.io/documentation/stable/reference/

  logging.info('Total nodes: %d; edges: %d'%(pdg.graph.order(), pdg.graph.size()))

  ## Filter by proteins of interest; this list comes from a DB adapter, but any set will do.
  proteins = dbAdapter.loadTotalProteinList().protein_id
  filterByProteins = set(proteins)
  logging.info('Protein list: %d'%(len(filterByProteins)))

  # Using attach() add edges from DB.
  # With this method create graph, which can be saved, avoiding
  # need for rebuilding for different diseases, models and analyses.
  # Also filter by proteins of interest, in this case it is our original list.

  pdg.attach(dbAdapter.loadPPI(filterByProteins))
  pdg.attach(dbAdapter.loadKegg(filterByProteins)) 
  pdg.attach(dbAdapter.loadReactome(filterByProteins)) 
  pdg.attach(dbAdapter.loadInterpro(filterByProteins))
  pdg.attach(dbAdapter.loadGo(filterByProteins))

  # Count node types based on IDs using NetworkX API.
  keggNodes = [n for n in list(pdg.graph.nodes)
	if isinstance(n, str) and n[0:3]=="hsa"]
  reactomeNodes = [n for n in list(pdg.graph.nodes)
	if isinstance(n, str) and n[0:2]=="R-"]
  goNodes = [n for n in list(pdg.graph.nodes)
	if isinstance(n, str) and n[0:3]=="GO:"]
  interNodes = [n for n in list(pdg.graph.nodes)
	if isinstance(n, str) and n[0:3]=="IPR"]

  logging.info("KEGG nodes: %d"%(len(keggNodes)))
  logging.info("REACTOME nodes: %d"%(len(reactomeNodes)))
  logging.info("GO nodes: %d"%(len(goNodes)))
  logging.info("INTERPRO nodes: %d"%(len(interNodes)))

  # Save graph in pickle format.
  if args.operation == 'build':
    logging.info("Saving pickled graph to: {0}".format(args.ofile))
    pdg.save(args.ofile)

  # Fetch pathway information from db.
  # (Not stored in graph?)
  idDescription = dbAdapter.fetchPathwayIdDescription()

  # Log node and edge info. Could be formatted for downstream use (e.g. Neo4j).
  edgeCount=0; nodeCount=0;
  with open(args.logfile, 'w') as flog:
    allNodes = set(pdg.graph.nodes)
    for node in allNodes:
      nodeCount+=1
      try:
        flog.write('NODE '+'{id:"'+str(node)+'", desc:"'+idDescription[node]+'"}'+'\n')
      except:
        logging.error('Node not found: {0}'.format(node))

    allEdges = set(pdg.graph.edges)
    for edge in allEdges:
      edgeCount+=1
      flog.write('EDGE '+'{idSource:"'+str(edge[0])+'", idTarget:"'+str(edge[1])+'"}'+'\n')

  logging.info('{0} nodes, {1} edges written to {2}'.format(nodeCount, edgeCount, args.logfile))

  logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))


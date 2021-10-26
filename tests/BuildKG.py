#!/usr/bin/env python3
"""
	Create Protein Disease knowledge graph from DB adapter 'OlegDB' or 'TCRD'.
"""
import sys, os, re, argparse, time, json
import logging
from networkx.readwrite.json_graph import cytoscape_data
from networkx.readwrite.graphml import generate_graphml

from proteingraphml.DataAdapter import OlegDB, TCRD
from proteingraphml.GraphTools import ProteinDiseaseAssociationGraph

if __name__ == "__main__":
    """
    Create a Protein-Disease knowledge graph (KG) from database and save it as a pickle file.
    Also, save KG data in log, cyjs, graphml, and tsv format.
    """
    DBS = ['olegdb', 'tcrd']
    parser = argparse.ArgumentParser(description='Create Protein-Disease knowledge graph (KG) from source RDB')
    parser.add_argument('--db', choices=DBS, default="tcrd", help='{0}'.format(str(DBS)))
    parser.add_argument('--o', dest="ofile", help='output pickled KG')
    parser.add_argument('--logfile', help='KG construction log.')
    parser.add_argument('--cyjsfile', help='Save KG as CYJS.')
    parser.add_argument('--graphmlfile', help='Save KG as GraphML.')
    parser.add_argument('--tsvfile', help='Save KG as TSV.')
    # parser.add_argument('--test', help='Build KG but do not save.')
    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                        level=(logging.DEBUG if args.verbose > 1 else logging.INFO))

    t0 = time.time()

    ## Construct base protein-disease map from ProteinDiseaseAssociationGraph.
    ## Db is PonyORM db (https://docs.ponyorm.org/api_reference.html).

    # Make TCRD as the default DB
    dbad = OlegDB() if args.db == "olegdb" else TCRD()

    pdg = ProteinDiseaseAssociationGraph(dbad)

    ## ProteinDiseaseAssociationGraph object has helper methods, but
    ## NetworkX methods also available.
    ## https://networkx.github.io/documentation/stable/reference/

    logging.info('Total nodes: %d; edges: %d' % (pdg.graph.order(), pdg.graph.size()))

    ## Filter by proteins of interest; this list comes from a DB adapter, but any set will do.
    proteins = dbad.loadTotalProteinList().protein_id
    proteinSet = set(proteins)
    logging.info('Protein set: %d' % (len(proteinSet)))

    # Using attach() add edges from DB.
    # With this method create graph, which can be saved, avoiding
    # need for rebuilding for different diseases, models and analyses.
    # Also filter by proteins of interest, in this case it is our original list.

    pdg.attach(dbad.loadPPI(proteinSet))
    pdg.attach(dbad.loadKegg(proteinSet))
    pdg.attach(dbad.loadReactome(proteinSet))
    pdg.attach(dbad.loadGo(proteinSet))
    try:
        pdg.attach(dbad.loadInterpro(proteinSet))
    except Exception as e:
        logging.error("InterPro failed to load: {0}".format(e))

    # TCRD only: (Would these add value?)
    try:
        pdg.attach(dbad.loadOMIM(proteinSet))
    except Exception as e:
        logging.error("OMIM failed to load: {0}".format(e))
    # try:
    #  pdg.attach(dbad.loadPfam(proteinSet))
    # except Exception as e:
    #  logging.error("Pfam failed to load: {0}".format(e))
    # try:
    #  pdg.attach(dbad.loadProsite(proteinSet))
    # except Exception as e:
    #  logging.error("Prosite failed to load: {0}".format(e))

    # Count node types based on IDs using NetworkX API.
    keggNodes = [n for n in list(pdg.graph.nodes) if isinstance(n, str) and n[0:3] == "hsa"]
    logging.info("KEGG nodes: %d" % (len(keggNodes)))
    reactomeNodes = [n for n in list(pdg.graph.nodes) if isinstance(n, str) and n[0:2] == "R-"]
    logging.info("REACTOME nodes: %d" % (len(reactomeNodes)))
    goNodes = [n for n in list(pdg.graph.nodes) if isinstance(n, str) and n[0:3] == "GO:"]
    logging.info("GO nodes: %d" % (len(goNodes)))
    interNodes = [n for n in list(pdg.graph.nodes) if isinstance(n, str) and n[0:3] == "IPR"]
    logging.info("INTERPRO nodes: %d" % (len(interNodes)))
    # pfamNodes = [n for n in list(pdg.graph.nodes) if isinstance(n, str) and n[0:2]=="PF"]
    # logging.info("Pfam nodes: %d"%(len(pfamNodes)))
    # prositeNodes = [n for n in list(pdg.graph.nodes) if isinstance(n, str) and n[0:2]=="PS"]
    # logging.info("PROSITE nodes: %d"%(len(prositeNodes)))

    # Fetch node/edge information from db.
    idDescription = dbad.fetchPathwayIdDescription()
    idSymbol = dbad.fetchSymbolForProteinId()
    try:
        idUniprot = dbad.fetchUniprotForProteinId()
    except Exception as e:
        logging.error("No Uniprot in OlegDB: {0}".format(e))
        idUniprot = {}

    # add name, symbol and uniprot id to graph nodes
    for n in pdg.graph.nodes:
        if n in idUniprot:
            pdg.graph.nodes[n]['UniprotId'] = idUniprot[n]
        else:
            pdg.graph.nodes[n]['UniprotId'] = ''
        if n in idSymbol:
            pdg.graph.nodes[n]['Symbol'] = idSymbol[n]
        else:
            pdg.graph.nodes[n]['Symbol'] = ''
        if n in idDescription:
            pdg.graph.nodes[n]['Description'] = idDescription[n]
        else:
            pdg.graph.nodes[n]['Description'] = ''

    # print(pdg.graph.nodes.data())

    # Save graph in pickle format.
    if args.ofile is not None:
        logging.info("Saving pickled graph to: {0}".format(args.ofile))
        pdg.save(args.ofile)

    # Save graph in CYJS format.
    if args.cyjsfile is not None:
        logging.info("Saving KG to CYJS file: {0}".format(args.cyjsfile))
        gdata = cytoscape_data(pdg.graph)
        with open(args.cyjsfile, 'w') as fcyjs:
            json.dump(gdata, fcyjs, indent=2)

    # Save graph in GraphML format.
    if args.graphmlfile is not None:
        logging.info("Saving KG to GraphML file: {0}".format(args.graphmlfile))
        with open(args.graphmlfile, 'w') as fgraphml:
            for line in generate_graphml(pdg.graph, encoding="utf-8", prettyprint=True):
                fgraphml.write(line + "\n")

    # Log node and edge info.
    if args.logfile is not None:
        edgeCount = 0
        nodeCount = 0
        with open(args.logfile, 'w') as flog:
            allNodes = set(pdg.graph.nodes)
            for node in allNodes:
                nodeCount += 1
                try:
                    flog.write('NODE ' + '{id:"' + str(node) + '", desc:"' + idDescription[node] + '"}' + '\n')
                except:
                    logging.error('Node not found: {0}'.format(node))
            allEdges = set(pdg.graph.edges)
            for edge in allEdges:
                edgeCount += 1
                try:
                    flog.write('EDGE ' + '{idSource:"' + str(edge[0]) + '", idTarget:"' + str(edge[1]) + '"}' + '\n')
                except:
                    logging.error('Edge node not found: {0}'.format(node))
        logging.info('{0} nodes, {1} edges written to {2}'.format(nodeCount, edgeCount, args.logfile))

    # TSV node and edge info, importable by Neo4j.
    if args.tsvfile is not None:
        edgeCount = 0;
        nodeCount = 0;
        with open(args.tsvfile, 'w') as fout:
            fout.write('node_or_edge\tclass\tid\tname\tsourceId\ttargetId\n')
            allNodes = set(pdg.graph.nodes)
            for node in allNodes:
                nodeCount += 1
                try:
                    name = idDescription[node]
                except:
                    logging.error('idDescription[node] not found: {0}'.format(node))
                    continue

                if re.match(r'\d+$', str(node)):
                    node_class = "PROTEIN"
                elif str(node)[0:3] == "hsa":
                    node_class = "KEGG"
                elif str(node)[0:2] == "R-":
                    node_class = "REACTOME"
                elif str(node)[0:3] == "GO:":
                    node_class = "GO"
                elif str(node)[0:3] == "IPR":
                    node_class = "INTERPRO"
                elif str(node)[0:2] == "MP":
                    node_class = "MP"
                else:
                    node_class = "Unknown"
                fout.write('node\t' + node_class + '\t' + str(node) + '\t' + idDescription[node] + '\t\t\n')
            allEdges = set(pdg.graph.edges)
            for edge in allEdges:
                # if re.match(r'\d+$', str(edge[0])) and re.match(r'\d+$', str(edge[1])):
                #    continue #protein-protein edges: STRING?
                edgeCount += 1
                fout.write('edge\t\t\t\t' + str(edge[0]) + '\t' + str(edge[1]) + '\n')
        logging.info('{0} nodes, {1} edges written to {2}'.format(nodeCount, edgeCount, args.tsvfile))

    logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]),
                                                 time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - t0))))

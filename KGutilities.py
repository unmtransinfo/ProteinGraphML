import pickle
import re
import logging
from py2neo import Graph, Node
import os
from tqdm import tqdm

# Send Cypher Queries to Neo4j
def writeToNeo4j(bolt="bolt://127.0.0.1:7687"):
    graph = Graph(bolt)

    directories = ["./cql/cypher/nodes", "./cql/cypher/relationships"]
    for directory in directories:
        files = os.listdir(directory)

        if not files:
            os.write(1, "Please run the option `--cypher` when running BuildKG.py to generate cypher queries.".encode())
            break

        for file in files:
            cqlData = open(f"{os.path.join(directory, file)}", "r").readlines()

            for query in tqdm(cqlData, desc=f"Processing {file} in directory {directory} to Neo4j Host ({bolt})"):
                graph.run(query)

# Load network X graph.
def createNeo4jCypherQueries(graphInstance):
    nodeCount = 0
    edgeCount = 0
    allNodes = list(graphInstance.nodes.data())
    allEdges = list(graphInstance.edges)

    if not os.path.exists("./cql/cypher"):
        logging.error("./cql/cypher directory does not exist. Creating now.")
        os.makedirs("./cql/cypher")

    nodeSavePath = "./cql/cypher/nodes.cql"
    relationshipSavePath = "./cql/cypher/edge_data.cql"

    with open(nodeSavePath, "w") as nodeDatadir:
        logging.info(f"There are a total of {len(allNodes)} vertices in the graph.")
        for node in allNodes:
            nodeCount += 1
            currentNode = node[0]

            if re.match(r'\d+$', str(currentNode)):
                if node[1]["Description"]:
                    node_class = "PROTEIN"
                    nodeData = str(Node(node_class,ID=nodeCount,Name=node[1]["Description"]))
                    nodeDatadir.write(f"CREATE {nodeData}\n")
                else:
                    continue
            elif "hsa" in currentNode:
                node_class = "KEGG"
                nodeData = str(Node(node_class,Name=node[1]["Description"]))
                nodeDatadir.write(f"CREATE {nodeData}\n")
            elif "R-" in currentNode:
                node_class = "REACTOME"
                nodeData = str(Node(node_class,Name=node[1]["Description"]))
                nodeDatadir.write(f"CREATE {nodeData}\n")
            elif "GO:" in currentNode:
                node_class = "GO"
                nodeData = str(Node(node_class,ID=node[0],Name=node[1]["Description"]))
                nodeDatadir.write(f"CREATE {nodeData}\n")
            elif "IPR" in currentNode:
                node_class = "INTERPRO"
                nodeData = str(Node(node_class,ID=node[0],Name=node[1]["Description"]))
                nodeDatadir.write(f"CREATE {nodeData}\n")
            elif "MP" in currentNode:
                node_class = "MP"
                nodeData = str(Node(node_class,ID=node[0],Name=node[1]["Description"]))
                nodeDatadir.write(f"CREATE {nodeData}\n")
            else:
                node_class = "Unknown"
                nodeData = str(Node(node_class,**node[1]))
                print(f"Unkown: {nodeData}")
        nodeDatadir.close()

    with open(relationshipSavePath,"w") as edgeDatadir:
        logging.info(f"There are a total of {len(allEdges)} edges in the graph.")
        for edge in allEdges:
            edgeDatadir.write(f"{edge[0]} -> {edge[1]}\n")
            # if re.match(r'\d+$', str(edge[0])) and re.match(r'\d+$', str(edge[1])):
            #    continue #protein-protein edges: STRING?
            edgeCount += 1
        edgeDatadir. close()
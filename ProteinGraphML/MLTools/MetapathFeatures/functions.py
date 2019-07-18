import networkx as nx
import numpy as np
import itertools
import pandas as pd


# THESE ARE THE KEY METAPATH FUNCTIONS, SOME MAYBE REMOVED OVERTIME 


def listCompute(graph,falseP,trueP,middleNode,edgeNode):

	# this is now the slowest operation ... followed up w/ this loop
	result = pd.DataFrame(data={"protein_id":edgeNode,"pathway_id":middleNode})
	# we need to sort this frame into....proteins, and pathways ... 

	right = result[result.protein_id.isin(trueP)]
	left = result[result.protein_id.isin(falseP)|result.protein_id.isin(trueP)]
	merged = pd.merge(left,right,on='pathway_id')
	deduplicates = merged[merged.protein_id_x != merged.protein_id_y].copy()
	UNIQUE_DISEASE = len(set(deduplicates.protein_id_y))

	deduplicates['middle'] = deduplicates.groupby(['protein_id_y'])['protein_id_y'].transform('count')
	deduplicates['edge'] = deduplicates.groupby(['protein_id_x'])['protein_id_x'].transform('count')
	deduplicates['endSet'] = deduplicates['middle']**(-0.5) * deduplicates['edge']**(-0.5) * (UNIQUE_DISEASE**(-0.5))


	# pathway ID??
	final = deduplicates.pivot_table(index=['protein_id_x'],columns=['pathway_id'], values='endSet').fillna(0)
	return final


# we need every path from A to B ... B has to be true, but thats it
def singleHop(graph,nodes,trueP,falseP):
	
	filterNodes = [k for k in nodes if len(set(graph.adj[k]).intersection(trueP)) > 0]
	# lets get all of the nodes, that connect to true nodes
	#print(len(filterNodes),len(trueP)) # you've cross with EVERY true, only some exist
	
	edgeFinal = list(itertools.product(filterNodes, list(trueP)))
	filteredEdges = []
	
	
	for e in edgeFinal:
		if graph.has_edge(e[0],e[1]): # and (e[1],e[0]) not in savedEdges and (e[0],e[1]) not in savedEdges: # make sure edge is one of a kind? #ie if both true, push once
			filteredEdges.append(e)
		
	savedEdges = {}
	finalEdges = []

	count = 0
	for e in filteredEdges: # filter down to edges we've got
		if e not in savedEdges:
			finalEdges.append(e)
			savedEdges[(e[0],e[1])] = True
			savedEdges[(e[1],e[0])] = True

	middleNodes =  []
	edgeNodes = []
	combinedScores = []
	for e in finalEdges:
		middleNodes.append(e[1])
		edgeNodes.append(e[0])
		combinedScores.append(graph.get_edge_data(e[0],e[1])['combined_score'])


	dataset = pd.DataFrame(data={"protein_id":edgeNodes,"protein_m_id":middleNodes,"scores":combinedScores})
	#result.to_csv('STUFF')

	#return listCompute({},falseP,trueP,middleNodes,edgeNodes)
	UNIQUE_DISEASE = len(set(dataset.protein_m_id))
	dataset['middle'] = dataset.groupby(['protein_id'])['protein_id'].transform('count').astype(float)
	dataset['edge'] = dataset.groupby(['protein_m_id'])['protein_m_id'].transform('count').astype(float)
	
	print(dataset.dtypes,type(UNIQUE_DISEASE))
	# .astype(float) prevents "ZeroDivisionError: 0.0 cannot be raised to a negative power
	UNIQUE_DISEASE = float(UNIQUE_DISEASE)
	dataset['pdp'] = dataset['middle']**(-0.5) * dataset['edge']**(-0.5) * UNIQUE_DISEASE**(-0.5) * dataset['scores']
	final = dataset.pivot_table(index=['protein_id'],columns=['protein_m_id'], values='pdp').fillna(0)
	return final

	# filter this list-no


def computeType(graph,nodes,trueP,falseP):
	
	filterNodes = [k for k in nodes if len(set(graph.adj[k]).intersection(trueP)) > 0]

	sub = graph.subgraph(filterNodes+list(falseP|trueP))

	# we can actually filter out edges which matter, only those connected to kegg
	#for path in nx.all_simple_paths(G, source=0, target=3)


	proteinEdges = set()

	edgeNode = set()

	middleNodeList = []
	edgeNodeList = []

	for n in filterNodes:

		edges = itertools.product([n], list(sub.adj[n]))
		middleNodes,edgeNodes = zip(*edges)
		middleNodeList = middleNodeList + list(middleNodes)
		edgeNodeList = edgeNodeList + list(edgeNodes)
		proteinEdges = proteinEdges | set(edges)	

	edges = list(proteinEdges)
	
	middleNode = filterNodes

	newG = {}

	return listCompute(newG,falseP,trueP,middleNodeList,edgeNodeList)




def metapathMatrix(adjMatrix,weight=-0.5):
	across = np.sum(adjMatrix,axis=1) # compute count for each base node
	down = np.sum(adjMatrix,axis=0)   # compute count for each connection
	uniqueCount = sum(np.where(down>0,1,0)) #scalar ... compute unique values of the connection (in total graph)

	uniqueVector = np.full_like(np.arange(len(down),dtype=float),uniqueCount) # make a vector of the unique count
	return adjMatrix * (down**weight) * (across[:,np.newaxis]**weight) * (uniqueVector**weight) # lets perform the computations



def completePPI(graph,trues,allNodes,adjGraph):

	scoredGraph = adjGraph[trues].loc[allNodes]
	resultsGraph = np.nan_to_num(np.divide(scoredGraph,scoredGraph))

	final = metapathMatrix(resultsGraph) * scoredGraph

	final = final.fillna(0)
	
	combinedScores = list(itertools.product(trues,allNodes))

	return final


def sPPICompute(graph,proteinNodes,trueP,falseP):
	#computeType(graph,nodes,trueP,falseP)
	return singleHop(graph,proteinNodes,trueP,falseP) #computeType(graph,proteinNodes,trueP,falseP)


def PPICompute(graph,proteinNodes,trueP,falseP):


	#print('PPI - starting subgraph')
	subPROTEIN = graph.subgraph(proteinNodes) # did the PPI filter out some proteins?
	#subPROTEIN = subPROTEIN.subgraph( set(subPROTEIN.nodes) - set(nx.isolates(subPROTEIN))  )
	trues = set()
	neighborSet = set()
	#print('PPI - making neighborSet')
	for protein in trueP:
		if protein in set(proteinNodes):
			trues.add(protein)
			neighborSet = neighborSet | set(nx.all_neighbors(subPROTEIN, protein))

	nodesList = trues | neighborSet

	finalGraph = subPROTEIN.subgraph(nodesList)

	#print('PPI - making adj subgraph') # this is slow
	adjIt = nx.to_pandas_adjacency(finalGraph,weight='combined_score')
	
	# this does our PPI computations with the values we want
	#print('PPI - matrix and loop')
	RR = completePPI(finalGraph,trues,nodesList,adjIt)
	#print('PPI - done with these computations')
	return RR



def getMetapaths(proteinGraph,start):

	children = getChildren(proteinGraph.graph,start)
	
	if start in proteinGraph.childParentDict.keys(): # if we've got parents, lets remove them from this search
		children = list( set(children) - set(proteinGraph.childParentDict[start]) )  


	proteinMap = {
		True:set(),
		False:set()
	}
	for c in children:
		p = filterNeighbors(proteinGraph.graph,c,True)
		n = filterNeighbors(proteinGraph.graph,c,False)
		posPaths = len(p)
		negPaths = len(n)
			
		for pid in p:
			proteinMap[True].add(pid)
		
	   
		for pid in n:
			proteinMap[False].add(pid)
		   
	return proteinMap

# new graph stuff, things below have been removed 
def filterNeighbors(graph,start,association): # hard coded ... "association"
	return [a for a in graph.adj[start] if "association" in graph.edges[(start,a)].keys() and graph.edges[(start,a)]["association"] == association]

def getChildren(graph,start): # hard coded ... "association"
	return [a for a in graph.adj[start] if "association" not in graph.edges[(start,a)].keys()]


def metapathFeatures(disease,proteinGraph,featureList,staticFeatures=None,test=False,loadedLists=None):
	# we compute a genelist.... 
	# get the proteins 
	# for each of the features, compute their metapaths, given an object, and graph+list... then they get joined 
	print(len(proteinGraph.graph.nodes))

	paths = getMetapaths(proteinGraph,disease)

	G = proteinGraph.graph # this is our networkx api 
	
	if loadedLists is not None:
		trueP = loadedLists[True] 
		falseP = loadedLists[False]
	else:
		trueP = paths[True]
		falseP = paths[False]

	proteinNodes = [pro for pro in list(G.nodes) if isinstance(pro,int)]
	
	nodeListPairs = []
	for n in featureList:
		nodeListPairs.append((n,[nval for nval in list(G.nodes) if n.isThisNode(nval)]))
		
	metapaths = []
	for pair in nodeListPairs:
		nodes = pair[1]
		nonTrueAssociations = set(proteinNodes) - trueP
		METAPATH = pair[0].computeMetapaths(G,nodes,trueP,nonTrueAssociations)
		METAPATH = (METAPATH - METAPATH.mean())/METAPATH.std()
		metapaths.append(METAPATH)
		
	
	if test:
		fullList = list(proteinNodes)
		df = pd.DataFrame(fullList, columns=['protein_id'])
		df = df.set_index('protein_id')
	else:
		fullList = list(itertools.product(trueP,[1])) + list(itertools.product(falseP,[0]))
		df = pd.DataFrame(fullList, columns=['protein_id', 'Y'])
		df = df.set_index('protein_id')


	for metapathframe in metapaths:
		#print(metapathframe.shape)
		#print(sum(metapathframe.sum(axis=1)))
		df = df.join(metapathframe,on="protein_id")

	#print(len(df))

	if staticFeatures is not None:
		df = joinStaticFeatures(df,staticFeatures)

	return df



def joinStaticFeatures(dataFrame,featureList):
	
	for feature in featureList:
		unpickled_df = pd.read_pickle("./"+feature+".csv.pkl")
		# these are needed edits right now for the joins we do, drop the unnamed column and set the index to the protein id
		unpickled_df = unpickled_df.drop(["Unnamed: 0"],axis=1)
		unpickled_df = unpickled_df.set_index('protein_id')

		if feature == "gtex" or feature == "ccle":  # we normed it all except hpa
			unpickled_df = (unpickled_df - unpickled_df.mean())/unpickled_df.std()
		
		dataFrame = dataFrame.join(unpickled_df,on="protein_id")

	return dataFrame


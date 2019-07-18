import networkx as nx
import pickle


# chain a list of compose functions together to build the graph
class GraphData:
	def test(self):
		pass

	def output(self):
		pass 

	def graphBuilder(self,edges,base=None):    
	    endGraph = base  
	    for graphEdge in edges:
	        if graphEdge.directed:
	            currentGraph = nx.from_pandas_edgelist(
	                graphEdge.data,
	                graphEdge.nodeLeft,
	                graphEdge.nodeRight,
	                edge_attr=graphEdge.edge,
	                create_using=nx.DiGraph
	            )
	        else:
	            currentGraph = nx.from_pandas_edgelist(
	                graphEdge.data,
	                graphEdge.nodeLeft,
	                graphEdge.nodeRight,
	                edge_attr=graphEdge.edge
	            )
	        
	        if endGraph is None:
	            endGraph = currentGraph
	        else:
	            endGraph = nx.compose(endGraph,currentGraph)
	    
	    return endGraph




class ProteinDiseaseAssociationGraph(GraphData): # on top of networkx?
	graph = None
	edges = None
	graphMap = {}  #we can put other graphs here...

	namesMap = {}
	

	# NOTE THIS IS A HACK, since undirected and directed graphs aren't working directly together right now, 
		# this will help us keep track of child/parent relationships... when we query a nodes children, we can use this to filter out parents

	parentChildDict = None # HACK


	def load(path):
		pickle_in = open(path,"rb")
		newGraph = pickle.load(pickle_in)
		return newGraph

	
	def __init__(self,adapter=None,graph=None):
		

		# would be great to fix this order problem, make it all more robust

		if graph is not None:
			self.graph = graph
		else:

			self.edges = [adapter.geneToDisease,adapter.phenotypeHierarchy] #self.PPI] # order matters!!
			graph = self.graphBuilder(self.edges)
			self.graph = graph
			self.childParentDict = adapter.childParentDict

		self.addNameData(adapter)



	def addNameData(self,adapter):
		for node in adapter.names:
			self.namesMap[node.name] = node


	def attach(self,edge):
		# helps build the multigraph, builds the interactions data, saves matrix also 
		# here it maybe prudent to save a dictionary of the nodes themselves

		self.edges.append(edge)
		self.graph = self.graphBuilder([edge],self.graph)


	def save(self,path):
		
		#nx.write_gpickle(self.graph,path)

		self.edges = None # clear out the edges/ these have large amounts of data
		pickle_out = open(path,"wb")
		pickle.dump(self, pickle_out)
		pickle_out.close()

	# parent dictionary? 

	def getDiseaseList(self): # this will filter out all MP deeper than 1
		# filter out diseases
		disease = [node for node in list(self.graph.nodes) if isinstance(node,str) and node[:3] == "MP_"]
		diseaseTree = nx.Graph(self.graph.subgraph(disease))
		# remove isolates 
		diseaseTree.remove_nodes_from(list(nx.isolates(diseaseTree)))

		fd = []
		for d in diseaseTree.nodes:
		    children = set(diseaseTree.adj[d]) - self.childParentDict.get(d,set())
		    if len(children) > 0:
		        fd.append(d)


		return fd


	def loadNames(self,nameType,valueList):

		thisNode = self.namesMap[nameType]
		return thisNode.dataframe[thisNode.dataframe.mp_term_id.isin(valueList)].reset_index(drop=True)


class GraphEdge:
	#nodeLeft =nodeRight,association = None
	directed = False
	def __init__(self,nodeLeft,nodeRight,edge=None,data=None):
		self.nodeLeft = nodeLeft
		self.nodeRight = nodeRight
		
		if edge is None:
			self.edge = None
		else:
			self.edge = edge
		
		self.data = data

	def setDirected(self):
		self.directed = True




# new graph stuff, things below have been removed 
def filterNeighbors(ProteinGraph,start,association): # hard coded ... "association"
	#graph = ProteinGraph
	return [a for a in graph.adj[start] if "association" in graph.edges[(start,a)].keys() and graph.edges[(start,a)]["association"] == association]

def getChildren(ProteinGraph,start): # hard coded ... "association"
	return [a for a in graph.adj[start] if "association" not in graph.edges[(start,a)].keys()]



def getMetapaths(ProteinGraph,start):
	children = getChildren(ProteinGraph,start) # really, this shouldn't hit networkx directly ... but should remain an external function
	proteinMap = {
		True:set(),
		False:set()
	}
	for c in children:
		# just edit this, to use our extended graph...
		p = filterNeighbors(ProteinGraph,c,True)
		n = filterNeighbors(ProteinGraph,c,False)
		posPaths = len(p)
		negPaths = len(n)
		#print(posPaths,negPaths)
		#if posPaths < 50 and negPaths < 50:
		#    continue
			
		for pid in p:
			proteinMap[True].add(pid)
		
	   
		for pid in n:
			proteinMap[False].add(pid)
		   
	return proteinMap


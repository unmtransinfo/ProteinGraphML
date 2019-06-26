import networkx as nx


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
	
	def __init__(self,adapter=None,graph=None):
		

		# would be great to fix this order problem, make it all more robust

		if graph is not None:
			self.graph = graph
		else:

			self.edges = [adapter.geneToDisease,adapter.phenotypeHierarchy] #self.PPI] # order matters!!
			graph = self.graphBuilder(self.edges)
			self.graph = graph

	def attach(self,edge):
		# helps build the multigraph, builds the interactions data, saves matrix also 
		self.edges.append(edge)
		self.graph = self.graphBuilder([edge],self.graph)


	def save(self,path):
		nx.write_gpickle(self.graph,path)





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
def filterNeighbors(graph,start,association): # hard coded ... "association"
	return [a for a in graph.adj[start] if "association" in graph.edges[(start,a)].keys() and graph.edges[(start,a)]["association"] == association]

def getChildren(graph,start): # hard coded ... "association"
	return [a for a in graph.adj[start] if "association" not in graph.edges[(start,a)].keys()]



def getMetapaths(graph,start):
	children = getChildren(graph,start)
	proteinMap = {
		True:set(),
		False:set()
	}
	for c in children:
		p = filterNeighbors(graph,c,True)
		n = filterNeighbors(graph,c,False)
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


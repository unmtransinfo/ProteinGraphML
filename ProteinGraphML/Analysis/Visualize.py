import networkx as nx

def Visualize(importance,graph,disease):
	#print(importance)

	#get shortest paths 
	# make the graph, dump it to JSON, save that in an HTML template with our formatting 

	#print(importance.most_common())

	firstFeature = importance.most_common()[4]

	print(firstFeature[0])

	print("hsa04144" in graph.nodes)
	print("hsa04144" in graph.nodes)
	print(firstFeature[0] in graph.nodes)
	print(disease in graph.nodes)

	# this parameter will change based on the features ... we will need the name saving ability here...
	# maybe we can test the disease list feature as well 
	
	nodesInGraph = set()
	for path in nx.all_simple_paths(graph, source=disease, target=firstFeature[0], cutoff=3):
		nodesInGraph |= set(path)


	finalGraph = graph.subgraph(list(nodesInGraph))
	#print(len(finalGraph.nodes))
	#print(nx.cytoscape_data(finalGraph))

	dataOut = str(nx.cytoscape_data(finalGraph)).replace("True","true").replace("False","false")[:-1]



	header = """
		<style type="text/css">
		.disease {
			background-color: blue;
			color: blue;
		}
	</style>

	<!--cy.getElementById("GO:0016323").addClass()-->

	<script type="text/javascript" src=https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.7.3/cytoscape.min.js></script>
	<script src="https://unpkg.com/layout-base/layout-base.js"></script>
	<script src="https://unpkg.com/avsdf-base/avsdf-base.js"></script>
	<script type="text/javascript" src="https://ivis-at-bilkent.github.io/cytoscape.js-avsdf/cytoscape-avsdf.js"></script>

	<div id="cy" style="width:900px; height:750px; border-style: solid">


	</div>
	<script type="text/javascript">
	data = 
	"""

	footer = """
	,'container':document.getElementById('cy')}

	var cy = cytoscape(data);

let options = {
  name: 'avsdf', //'breadthfirst',

  fit: false, // whether to fit the viewport to the graph
  directed: false, // whether the tree is directed downwards (or edges can point in any direction if false)
  padding: 30, // padding on fit
  circle: false, // put depths in concentric circles if true, put depths top down if false
  grid: false, // whether to create an even grid into which the DAG is placed (circle:false only)
  spacingFactor: 1.75, // positive spacing factor, larger => more space between nodes (N.B. n/a if causes overlap)
  boundingBox: undefined, // constrain layout bounds; { x1, y1, x2, y2 } or { x1, y1, w, h }
  avoidOverlap: true, // prevents node overlap, may overflow boundingBox if not enough space
  nodeDimensionsIncludeLabels: false, // Excludes the label when calculating node bounding boxes for the layout algorithm
  roots: undefined, // the roots of the trees
  maximal: false, // whether to shift nodes down their natural BFS depths in order to avoid upwards edges (DAGS only)
  animate: false, // whether to transition the node positions
  animationDuration: 500, // duration of animation in ms if enabled
  animationEasing: undefined, // easing of animation if enabled,
  animateFilter: function ( node, i ){ return true; }, // a function that determines whether the node should be animated.  All nodes animated by default on animate enabled.  Non-animated nodes are positioned immediately when the layout starts
  ready: undefined, // callback on layoutready
  stop: undefined, // callback on layoutstop
  transform: function (node, position ){ return position; } // transform a given node position. Useful for changing flow direction in discrete layouts
};

cy.layout(options).run();

//cy.$('#j, #e').addClass('foo'); ## ADD A CLASS TO THE MP nodes, and their label


// * 
// here we can build a harness that will color the nodes, and set edge weights? 
// the more this map is annotated, tbe better 

function isMPNode(input) {
	if(input[0]+input[1] == "MP") {
		return true;
	}

	return false;
}
function isGoNode(input) {
	if(input[0]+input[1] == "GO") {
		return true;
	}

	return false;
}

function isNotProteinOrMP(input) {
	if(input.length > 5) {
		return true && !isMPNode(input);
	}
	return false;
}

function edgeHasNode(inputEdge,nodeCheck) {
	if(nodeCheck(inputEdge.source) || nodeCheck(inputEdge.target)) {		
		return true
	}
	else {
		return false
	}
}


for(var node of Object.values(cy.nodes())) {
	//console.log(node)
	if(node.id) { 
		cy.getElementById(id).style('label',id)
		var id = node.id()
		//console.log(id);
		if(isMPNode(id)) {
			cy.getElementById(id).style('background-color','#0081CF')			
		} else if(id.length > 5) {
			cy.getElementById(id).style('background-color','lightgreen')
		} else {
			cy.getElementById(id).style('background-color','rgb(0, 149, 149)')
		}
	}
	//node.addClass("disease")
}

// for each edge in the graph, if its got an association, color that, if its got a combined score, color that


for(var edge of Object.values(cy.edges())) {
	if(edge.data && edge.data().association != undefined) {
		console.log(edge.data().association)
		if(edge.data().association) {
			edge.style('line-color','#00C9A7');
			edge.style('width','12px');
			edge.style('label','KNOWN POSITIVE ASSOCIATION');
			edge.style('text-rotation','autorotate')
		} else {
			//edge.style('line-color','#FF6F91');
			//edge.style('width','3px');



			var data = edge.data();
			var noderemove = null;
			if(isMPNode(data.source)) {
				noderemove = data.target;
			} else {
				noderemove = data.source;
			}


			cy.remove(edge)
			cy.remove(cy.getElementById(noderemove))
		}
	} else {

		if(edge.data) {
			if(edgeHasNode(edge.data(),isNotProteinOrMP)){
				edge.style('line-color','lightgreen')
				edge.style('width','6px');
			} else if(edgeHasNode(edge.data(),isMPNode)){
				edge.style('line-color','lightblue')
				edge.style('label','')
				edge.style('width','6px');
			} else {
				var score = parseFloat(edge.data().combined_score)/1000.0
				edge.style('line-color','#444')
				console.log(parseFloat(edge.data().combined_score)/1000.0,score)
				edge.style('opacity',score.toString())
				edge.style('width','5%');
				edge.style('line-style','dotted')
			}

		}

		
		//console.log("NO!",edge)



	}
}


//for(var edge in nod)


</script>
	"""

	text_file = open("testIT.html", "w")
	text_file.write(header+dataOut+footer)
	text_file.close()

	#print(header+dataOut+footer)




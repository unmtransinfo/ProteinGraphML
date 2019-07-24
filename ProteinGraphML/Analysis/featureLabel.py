from ProteinGraphML.MLTools.MetapathFeatures import ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode

def getValueForId(label,inputValue,extractKey,DB):
    results = DB[DB[label] == inputValue]
    if len(results) == 0:
        return None
    else:
        return results.iloc[0][extractKey]


def convertLabels(featureLabels,adapter,selectAsDF,type="graph"):

	# given a list, map them to real things, also w/ a DB adapter
	# print('beans')

	protein = selectAsDF("select name,symbol,protein_id from protein",["name","symbol","protein_id"],adapter.db)
	#MPONTO = selectAsDF("select * ")
	kegg = selectAsDF("select kegg_pathway_id,kegg_pathway_name from kegg_pathway",["kegg_pathway_id","kegg_pathway_name"],adapter.db)
	mpOnto = selectAsDF("select mp_term_id,name from mp_onto",["mp_term_id","name"],adapter.db)


	newMap = {}
	for item in featureLabels:
		newValue = item
		prefix = ""
		if ProteinInteractionNode.isThisNode(item): # is a protein node:...
			
			if type != 'graph':
				prefix = "PPI:"

			print(protein)
			newValue = "{0}{1}".format(prefix,getValueForId("protein_id",item,"symbol",protein))

		#name = getValueForId("kegg_pathway_id",value,"kegg_pathway_name",kegg)
		if KeggNode.isThisNode(item):
			newValue = getValueForId("kegg_pathway_id",item,"kegg_pathway_name",kegg)

		if isinstance(item,str) and item[:3] == "MP_":
			newValue = getValueForId("mp_term_id",item,"name",mpOnto)



		newMap[item] = newValue
		#print(newValue)

	return newMap







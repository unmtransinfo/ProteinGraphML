#!/usr/bin/env python
# coding: utf-8

### NOTE use the build_graph_example notebook to build your graph first!!! ###

# to read our graph directly

from ProteinGraphML.DataAdapter import OlegDB,selectAsDF
import networkx as nx
from ProteinGraphML.GraphTools import ProteinDiseaseAssociationGraph

from ProteinGraphML.Analysis import Visualize
import pandas as pd
#currentGraph = nx.read_gpickle("newGRAPH.pickle")
#####ProteinDiseaseAssociationGraph()

#ProteinDiseaseAssociationGraph()
currentGraph = ProteinDiseaseAssociationGraph.load("newCURRENT_GRAPH")
# time to build features from the graph! 

# the objects that we need to build our features:

from ProteinGraphML.MLTools.MetapathFeatures import metapathFeatures,ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode

diseaseList = currentGraph.getDiseaseList()

from ProteinGraphML.MLTools.MetapathFeatures import getMetapaths

finalSet = []
for d in diseaseList:
    mp = getMetapaths(currentGraph,d)
    if len(mp[True]) >= 50 and len(mp[False]) >= 50:
        finalSet.append(d)
print(len(finalSet))


print(currentGraph.loadNames("MP_ontology",finalSet).head())

#<<<<<<< Updated upstream

nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]
Disease = "MP_0000180"


#def load_dict_from_file():
import pickle
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
loadList = load_obj('nextDataset')

#print(loadList)

#res = load_dict_from_file()
trainData = metapathFeatures(Disease,currentGraph,nodes,['lincs','gtex','ccle','hpa'],loadedLists=loadList).fillna(0)
#=======
print(currentGraph.loadNames("MP_ontology",finalSet).shape)
#trainData = metapathFeatures(Disease,currentGraph,nodes,["gtex"]).fillna(0)


#trainData.shape # looks good 

from ProteinGraphML.MLTools.Data import BinaryLabel

# binary label wrapper, means we don't have to worry about anything else... default label value is 'Y' and thats
# what label was generated above

# the object is designed to manage our data, it's split, separating features/label etc...
d = BinaryLabel()
d.loadData(trainData)


#param = {'n_estimators':7,'learning_rate':0.02,'max_depth':7, 'eta':0.1,'subsample':0.9,'silent':0,'min_child_weight':5, 'objective':'binary:logistic'}


from ProteinGraphML.MLTools.Models import XGBoostModel

#note, there are built in apis which work with Scikit cross validation- this is TODO

# this version works as well, we split the data with 10 random folds: 80% of data as train... 20% as test 

newModel = XGBoostModel()


#newModel.setClassifier(clfg)
#newModel.setClassifier(rf)
#newModel.setClassifier(clfc)
Report,roc,rocCurve,CM = newModel.cross_val_predict(d,["report","roc","rocCurve","ConfusionMatrix"])

Report.printOutput()
roc.printOutput()

importance = newModel.average_cross_val(d,[])




for g in importance.most_common(10):
	print(g)
	Visualize(g,currentGraph.graph,Disease)

'''
featureImportance = []
ROC = 0.
CROSSVAL = 10 # 10 
for k in range(0,CROSSVAL):
    # get a split.... train the model

    # train is first 80% of data, test is other 20% ... note the "Data" object handles labels etc
    train,test = d.splitSet(0.8) # lets get a ROC result

    # make a new model
    newModel = XGBoostModel() 
    
    #newModel.m  -> accces the XGBOOST api directly 
    
    newModel.train(train,param)
   
    
    rocResult = newModel.predict(test,["roc"]) # current work on expanding the list of resutls here...
    rocResult.printOutput() # prints ROC, and adds for average 
    ROC+=rocResult.data
    
    

    # this can do the feature importance as well, but working on adding support here first
    importance = newModel.m.get_score(importance_type='gain')
    featureImportance.append(importance)
    #ECGimportanceSet.append(importance)
    

print("")    
print("AVG AUC-ROC",ROC/CROSSVAL)


# access saved feature importance here
featureImportance
'''








dbAdapter = OlegDB()
protein = selectAsDF("select name,symbol,protein_id from protein",["name","symbol","protein_id"],dbAdapter.db)
drugname = selectAsDF("select drug_id,drug_name from drug_name",["drug_id","drug_name"],dbAdapter.db)
kegg = selectAsDF("select kegg_pathway_id,kegg_pathway_name from kegg_pathway",["kegg_pathway_id","kegg_pathway_name"],dbAdapter.db)
ccle = selectAsDF("select distinct cell_id,tissue from ccle",["cell_id","tissue"],dbAdapter.db)
def isProtein(value):
    return value[:3] == "pp."

#def findReplacement(value,typeIs):
#drug_name <- dbGetQuery(conn, "select drug_id,drug_name from drug_name")    

import re
import matplotlib.pyplot as plt


def getValueForId(label,inputValue,extractKey,DB):
    results = DB[DB[label] == inputValue]
    if len(results) == 0:
        return None
    else:
        return results.iloc[0][extractKey]

def processFeature(value):
    
    replace = value
    if value[:3] == "pp." or value.isdigit():
        replace = "PPI:"+protein[protein.protein_id == int(value[3:])].iloc[0]['symbol']
    
    isDrug = re.compile("\\d+:[A-Z]")
    
    if isDrug.match(value):
        
        ID = value[:value.find(':')]
        CELL_ID = value[value.find(':'):]
        name = getValueForId("drug_id",ID,"drug_name",drugname)
        
        if name is None:
            name = ID
            
        replace = name + CELL_ID + " signature"
    
    if value.find('_') > 0:
        #query to make sure the tissue exists? IDK
        replace = value[value.find('_')+1:] + " " + "("+value[:value.find('_')]+")"
    elif len(ccle[ccle.cell_id == value]) > 0:
        replace = "expression in "+value
        
    if value[:3] == "hsa":
        name = getValueForId("kegg_pathway_id",value,"kegg_pathway_name",kegg)
        if name is None:
            replace = value
        else:
            replace = name
    
    return replace

def featureVisualize(features,AUC,TITLE): 
    
	plt.rcParams.update({'font.size': 15,'lines.linewidth': 1000}) #   axes.labelweight': 'bold'})
	FILETITLE = TITLE
	TITLE = TITLE + "-AUC: "+str(AUC)  

	df = pd.DataFrame(features.most_common(20), columns=['feature', 'gain'])
	plt.figure()
	df['gain'] = (df['gain']/sum(df['gain'][:20]))
	df['feature'] = df['feature'].map(processFeature)
	r = df.head(20).plot( kind='barh',title=TITLE, x='feature', y='gain',color='tomato', legend=False, figsize=(10, 12))
	r.set_xlabel('Importance')
	r.set_ylabel('Features')
	r.invert_yaxis()

	r.figure.savefig(FILETITLE+'.png',bbox_inches='tight')


featureVisualize(importance,roc.data,"DATASETFEATIMPORT")

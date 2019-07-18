#!/usr/bin/env python
# coding: utf-8

### NOTE use the build_graph_example notebook to build your graph first!!! ###

# to read our graph directly
import networkx as nx
from ProteinGraphML.GraphTools import ProteinDiseaseAssociationGraph
#currentGraph = nx.read_gpickle("newGRAPH.pickle")
#####ProteinDiseaseAssociationGraph()

#ProteinDiseaseAssociationGraph()
currentGraph = ProteinDiseaseAssociationGraph.load("CURRENT_GRAPH_H")
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


nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]
Disease = "MP_0000180"

trainData = metapathFeatures(Disease,currentGraph,nodes,[]).fillna(0)

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
Report,roc,rocCurve,CM = newModel.cross_val_predict(dataR,["report","roc","rocCurve","ConfusionMatrix"])

Report.printOutput()

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
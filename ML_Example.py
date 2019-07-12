#!/usr/bin/env python
# coding: utf-8

### NOTE use the build_graph_example notebook to build your graph first!!! ###

# to read our graph directly
import networkx as nx

currentGraph = nx.read_gpickle("CURRENT_GRAPH")

# time to build features from the graph! 

# the objects that we need to build our features:

from ProteinGraphML.MLTools.MetapathFeatures import metapathFeatures,ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode


# what did we grab:: 

# metapathFeatures - the main function that assembles our metapath features


# ProteinInteractionNode
# KeggNode
# ReactomeNode
# GoNode
# InterproNode

# each of these nodes has instructions/function for how to compute their type of metapath,
# IE: for ProteinInteractionNode the metapath is P1 <-> P2 -> Disease but for others there is a middle node like: 
# P1 -> KEGG_pathway <- P2 -> Disease

# features we want, and a disease we want to analyse  
# may change "ProteinInteractionNode" .. to "DattrainDataaType"... these aren't really "nodes"
nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]
Disease = "MP_0000184"

# empty array is static features resolving bugs and then will add to this function


'''
we are ready to make data!... this will build a pandas frame with labels, 
labeling metapaths where a protein is connected through another

# NOTE::
label is true if protein on the edge of the path to a disease has a true association to it
and label is false, if protein on the edge of path to the disease has a false association to it 


# this function will build training set by default, but can build test set as well with test=True
'''
trainData = metapathFeatures(Disease,currentGraph,nodes,[]).fillna(0)

trainData.shape # looks good 

from ProteinGraphML.MLTools.Data import BinaryLabel

# binary label wrapper, means we don't have to worry about anything else... default label value is 'Y' and thats
# what label was generated above

# the object is designed to manage our data, it's split, separating features/label etc...
d = BinaryLabel()
d.loadData(trainData)

'''
d - is now our data set, if we split it we get new BinaryLabel objects, and the ProteinGraphML.MLTools.Models
can parse out the labels / features automatically

if the label isn't Y, you can use: 

d.loadData(trainData,labelColumn='mylabel')

'''


#READY FOR XGBOOST: Here are some parameters

param = {'n_estimators':7,'learning_rate':0.02,'max_depth':7, 'eta':0.1,'subsample':0.9,'silent':0,'min_child_weight':5, 'objective':'binary:logistic'}


# NOTE!!!! THIS EXAMPLE WON'T SAVE THE MODEL(s) TRAINED !!!  THIS IS JUST TO DEMONSTRATE THE API 

# you can access the package xgboost directly to save the model using XGBoostModel.m


from ProteinGraphML.MLTools.Models import XGBoostModel

#note, there are built in apis which work with Scikit cross validation- this is TODO

# this version works as well, we split the data with 10 random folds: 80% of data as train... 20% as test 

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


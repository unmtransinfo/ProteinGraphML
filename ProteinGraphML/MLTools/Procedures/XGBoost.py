# this will run a version of XGB, with a gross validation score 


# this will compute a cross val score...

# ASSUME LOCKED HYPER PARAMS 
# ASSUME LOCKED ALGO, PASS IN THE DATA

from ProteinGraphML.MLTools.Models import XGBoostModel
import pickle
import os
CROSSVAL = 5

def TEST(dataObject):
	print('tehe')

def XGBCrossValPred(dataObject, idDescription, diseaseName):
	newModel = XGBoostModel("XGBCrossValPred")
	params = {'scale_pos_weight':dataObject.posWeight, 'n_jobs':8} 		#XGboost parameters	
	
	#roc,acc,CM,report = newModel.cross_val_predict(dataObject,["roc","acc","ConfusionMatrix","report"]) 
	
	roc,acc,mcc, CM,report,importance = newModel.cross_val_predict(dataObject, idDescription, ["roc","acc", "mcc", "ConfusionMatrix","report"], params=params,cv=CROSSVAL) #Pass parameters 
	saveImportantFeatures(importance, diseaseName)
	#print("AUCROC --- {0}".format(roc.data))
	#print("Accuracy --- {0}".format(acc.data))
	#print("MCC --- {0}".format(mcc.data))
	#print("ConfusionMatrix:\n")
	#print(CM.data)
	#print("Report:\n")
	#print(report.data)
	#roc.printOutput() #plot roc

def XGBCrossVal(dataObject, idDescription, diseaseName):
	newModel = XGBoostModel("XGBCrossVal")
	params = {'scale_pos_weight':dataObject.posWeight, 'n_jobs':8} 		#XGboost parameters	
	
	# CUSTOM PARAMS 
	#params={'max_depth':10,'gamma':0.2}
	#importance = newModel.average_cross_val(dataObject, ["roc","acc","ConfusionMatrix","report"], folds=2, params=params)

	importance = newModel.average_cross_val(dataObject, idDescription, ["roc","rocCurve","acc","mcc"], folds=2, params=params,cv=CROSSVAL)
	saveImportantFeatures(importance, diseaseName)
	#print (importance)
	##"seed"	"max_depth"	"eta"	"gamma"	"min_child_weight"	"subsample"	"colsample_bytree"	"nrounds"	"auc"
		#1001	                      10	0.2	                 0.1	0	                                   0.9	  
	
	#roc,acc,CM,report = newModel.cross_val_predict(dataObject,["roc","acc","ConfusionMatrix","report"])
	#print("AUCROC--- {0}".format(roc.data))
	#roc.printOutput() #plot roc
	

	#importance = newModel.average_cross_val(d,[])
	#print("ML COMPLETE, CREATING VISUALIZATION")

	#XGBCrossVal

	#for g in importance.most_common(2):
	#	print("PRINTING THIS IMPORTANT FEATURES- {0}".format(disease),g)
	#	Visualize(g,currentGraph.graph,disease)



def saveImportantFeatures(importance, diseaseName):
	'''
	Save important features in a pickle dictionary
	'''
	featureDir = 'results/XGBFeatures'
	if not os.path.exists(featureDir): os.mkdir(featureDir) 
	featureFile = featureDir + '/{0}'.format(diseaseName)
	with open(featureFile, 'wb') as ff:
		pickle.dump(importance, ff, pickle.HIGHEST_PROTOCOL)

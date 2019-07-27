# this will run a version of XGB, with a gross validation score 


# this will compute a cross val score...

# ASSUME LOCKED HYPER PARAMS 
# ASSUME LOCKED ALGO, PASS IN THE DATA

from ProteinGraphML.MLTools.Models import XGBoostModel

def TEST(dataObject):
	print('tehe')

def XGBCrossValPred(dataObject):
	newModel = XGBoostModel("XGBCrossValPred")
	#roc,acc,CM,report = newModel.cross_val_predict(dataObject,["roc","acc","ConfusionMatrix","report"])
	roc,acc,CM,report = newModel.cross_val_predict(dataObject,["roc","acc","ConfusionMatrix","report"]) #,"acc","ConfusionMatrix","report"
	print("AUCROC--- {0}".format(roc.data))

def XGBCrossVal(dataObject):

	newModel = XGBoostModel("XGBCrossVal")
	# CUSTOM PARAMS 
	params={'max_depth':10,'gamma':0.2}
	newModel.average_cross_val(dataObject,["roc","acc","ConfusionMatrix","report"],folds=2,params=params)
	##"seed"	"max_depth"	"eta"	"gamma"	"min_child_weight"	"subsample"	"colsample_bytree"	"nrounds"	"auc"
		#1001	                      10	0.2	                 0.1	0	                                   0.9	  
	
	#roc,acc,CM,report = newModel.cross_val_predict(dataObject,["roc","acc","ConfusionMatrix","report"])
	#print("AUCROC--- {0}".format(roc.data))
	


	#importance = newModel.average_cross_val(d,[])
	#print("ML COMPLETE, CREATING VISUALIZATION")

	#XGBCrossVal

	#for g in importance.most_common(2):
	#	print("PRINTING THIS IMPORTANT FEATURES- {0}".format(disease),g)
	#	Visualize(g,currentGraph.graph,disease)





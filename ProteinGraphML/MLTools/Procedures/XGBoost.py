# this will run a version of XGB, with a gross validation score 


# this will compute a cross val score...

# ASSUME LOCKED HYPER PARAMS 
# ASSUME LOCKED ALGO, PASS IN THE DATA

from ProteinGraphML.MLTools.Models import XGBoostModel
import pickle, logging
import os

CROSSVAL = 5


def TEST(dataObject):
    print('tehe')


def XGBCrossValPred(dataObject, idDescription, idNameSymbol, idSource, resultDir, params=None):
    newModel = XGBoostModel("XGBCrossValPred", resultDir)

    # params['scale_pos_weight'] = dataObject.posWeight
    logging.info('Parameters for XGBoost are: {0}'.format(params))

    # roc,acc,CM,report = newModel.cross_val_predict(dataObject,["roc","acc","ConfusionMatrix","report"])

    newModel.cross_val_predict(dataObject, idDescription, idNameSymbol, idSource,
                               ["roc", "rocCurve", "acc", "mcc", "ConfusionMatrix", "report"], params=params,
                               cv=CROSSVAL)  # Pass parameters


# print("AUCROC --- {0}".format(roc.data))
# print("Accuracy --- {0}".format(acc.data))
# print("MCC --- {0}".format(mcc.data))

# print("ConfusionMatrix:\n")
# print(CM.data)
# print("Report:\n")
# print(report.data)
# roc.printOutput() #plot roc

# def XGBCrossVal(dataObject, idDescription, idNameSymbol, resultDir, nfolds=1, params=None):
def XGBKfoldsRunPred(dataObject, idDescription, idNameSymbol, resultDir, nrounds, params=None):
    newModel = XGBoostModel("XGBKfoldsRunPred", resultDir)
    # params['scale_pos_weight'] = dataObject.posWeight
    logging.info('Parameters for XGBoost are: {0}'.format(params))

    # newModel.average_cross_val(dataObject, idDescription, idNameSymbol, ["roc","rocCurve","acc","mcc"], folds=nfolds, params=params,cv=CROSSVAL)
    newModel.average_cross_val(dataObject, idDescription, idNameSymbol, ["roc", "acc", "mcc"], nrounds, params=params)


# print (importance)
##"seed"	"max_depth"	"eta"	"gamma"	"min_child_weight"	"subsample"	"colsample_bytree"	"nrounds"	"auc"
# 1001	                      10	0.2	                 0.1	0	                                   0.9

# roc,acc,CM,report = newModel.cross_val_predict(dataObject,["roc","acc","ConfusionMatrix","report"])
# print("AUCROC--- {0}".format(roc.data))
# roc.printOutput() #plot roc


# importance = newModel.average_cross_val(d,[])
# print("ML COMPLETE, CREATING VISUALIZATION")

# XGBCrossVal

# for g in importance.most_common(2):
#	print("PRINTING THIS IMPORTANT FEATURES- {0}".format(disease),g)
#	Visualize(g,currentGraph.graph,disease)


def XGBPredict(dataObject, idDescription, idNameSymbol, modelName, resultDir, infoFile):
    newModel = XGBoostModel("XGBPredict", resultDir)
    newModel.predict_using_saved_model(dataObject, idDescription, idNameSymbol, modelName, infoFile)


def XGBGridSearch(dataObject, idDescription, idNameSymbol, resultDir, rseed, nthreads):
    newModel = XGBoostModel("XGBGridSearch", resultDir)

    paramGrid = {'max_depth': [5, 7, 8, 10],
                 'eta': [0.05, 0.10, 0.15, 0.20, 0.25, 0.50],
                 'gamma': [0.01, 0.1, 0.5, 1],
                 'min_child_weight': [0, 1, 2],
                 'subsample': [0.7, 0.8, 0.9, 1],
                 'colsample_bytree': [0.7, 0.8, 0.9, 1]
                 }

    '''
	##ONLY FOR TESTING
	paramGrid = {'max_depth': [7,8],
				 'eta': [0.05],
				 'learning_rate': [0.1],
				 'gamma': [0.01],
				 'min_child_weight': [0],
				 'subsample': [0.8],
				 'colsample_bytree': [0.5]
				 }
	'''

    newModel.gridSearch(dataObject, idDescription, idNameSymbol, ["roc", "acc", "mcc"], paramGrid, rseed, nthreads)

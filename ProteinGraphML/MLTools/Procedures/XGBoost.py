# this will run a version of XGB, with a gross validation score 


# this will compute a cross val score...

# ASSUME LOCKED HYPER PARAMS 
# ASSUME LOCKED ALGO, PASS IN THE DATA

from ProteinGraphML.MLTools.Models import XGBoostModel
import pickle, logging
import os

CROSSVAL = 5


def TEST(dataObject):
    print('test')


def XGBCrossValPred(dataObject, idDescription, idNameSymbol, idSource, resultDir, params=None):
    """
    Run the ML model once using 5-fold cross-validation.
    """
    newModel = XGBoostModel("XGBCrossValPred", resultDir)

    params['scale_pos_weight'] = dataObject.posWeight
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
def XGBKfoldsRunPred(dataObject, idDescription, idNameSymbol, idSource, resultDir, nrounds, params=None):
    """
    Run the ML model n-times using 5-fold cross-validation.
    """
    newModel = XGBoostModel("XGBKfoldsRunPred", resultDir)
    params['scale_pos_weight'] = dataObject.posWeight
    logging.info('Parameters for XGBoost are: {0}'.format(params))

    # newModel.average_cross_val(dataObject, idDescription, idNameSymbol, ["roc","rocCurve","acc","mcc"],
    # folds=nfolds, params=params,cv=CROSSVAL)
    newModel.average_cross_val(dataObject, idDescription, idNameSymbol, idSource, ["roc", "acc", "mcc"], nrounds,
                               params=params)

    # print (importance)
    # "seed"	"max_depth"	"eta"	"gamma"	"min_child_weight"	"subsample"	"colsample_bytree"	"nrounds"	"auc"
    # 1001	                      10	0.2	                 0.1	0	                                   0.9

    # roc,acc,CM,report = newModel.cross_val_predict(dataObject,["roc","acc","ConfusionMatrix","report"])
    # print("AUCROC--- {0}".format(roc.data))
    # roc.printOutput() #plot roc

    # importance = newModel.average_cross_val(d,[])
    # print("ML COMPLETE, CREATING VISUALIZATION")

    # XGBCrossVal
    # for g in importance.most_common(2):
    #   print("PRINTING THIS IMPORTANT FEATURES- {0}".format(disease),g)
    #   Visualize(g,currentGraph.graph,disease)


def XGBPredict(dataObject, idDescription, idNameSymbol, modelName, resultDir, infoFile):
    """
    Predict class-1 probability for predict data using the trained model.
    """
    newModel = XGBoostModel("XGBPredict", resultDir)
    newModel.predict_using_saved_model(dataObject, idDescription, idNameSymbol, modelName, infoFile)


def XGBGridSearch(dataObject, idDescription, idNameSymbol, resultDir, rseed, nthreads):
    """
    Run the GridSearch() function to find the optimal values for the XGBoost parameters.
    """
    newModel = XGBoostModel("XGBGridSearch", resultDir)
    neg_pos_ratio = dataObject.posWeight
    paramGrid = {'booster': ['gbtree', 'gblinear', 'dart'],
                 'gamma': [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                 'eta': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 1],
                 'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 15],
                 'max_delta_step': [0, 1, 3, 5, 7, 9, 10],
                 'lambda': [1, 3, 5],
                 'alpha': [0, 1, 3, 5],
                 'min_child_weight': [0, 1, 2],
                 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                 'scale_pos_weight': [neg_pos_ratio*0.5, neg_pos_ratio*0.75, neg_pos_ratio, neg_pos_ratio*1.25,
                                      neg_pos_ratio*1.5, neg_pos_ratio*2]
                 }
    """
    # ONLY FOR TESTING
    paramGrid = {'max_depth': [7,8], 
                'eta': [0.05], 
                'learning_rate': [0.1], 
                'gamma': [0.01], 
                'min_child_weight': [0],
                'subsample': [0.8],
                'colsample_bytree': [0.5]
                }
    """

    newModel.gridSearch(dataObject, idDescription, idNameSymbol, ["roc", "acc", "mcc"], paramGrid, rseed, nthreads)

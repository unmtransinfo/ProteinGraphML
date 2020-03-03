import os, sys
import time, logging, random
import pickle, random
from collections import Counter
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_predict, StratifiedShuffleSplit, \
    cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import cross_val_score,
from sklearn.metrics import accuracy_score, auc, confusion_matrix, classification_report, matthews_corrcoef
from sklearn.metrics import roc_auc_score, \
    roc_curve  # .roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None)


# from ProteinGraphML.MLTools.Data import Data
# this model system will hopefully make a simple API for dealing with large data 

# iterating on our platform across domains 

class Result:
    data = None
    predictions = None
    space = None
    predLabel = None

    def __init__(self, dataOut, predictions, space="", modelDIR=None):
        self.data = dataOut
        self.predictions = predictions
        # self.modelName = modelName
        self.space = space
        # print("HERE IS THE MODEL")
        self.resultDIR = modelDIR
        # we put the functions here which actually convert the data to a binary score
        self.predLabel = [round(p) for p in self.predictions]  # generate label using probability

    # print ('PRINT ALL VALUES....>>>')
    # print (self.predictions, len(self.predictions))
    # print (self.predLabel, len(self.predLabel))
    # print (self.data.labels, len(self.data.labels))
    def acc(self):
        return Output("ACC", accuracy_score(self.data.labels, self.predLabel))

    def mcc(self):  # Add MCC since data is imbalanced
        return Output("MCC", matthews_corrcoef(self.data.labels, self.predLabel))

    def roc(self):
        roc = Output("AUCROC", roc_auc_score(self.data.labels, self.predictions))
        # roc.fileOutput(self.modelName)
        return roc

    def ConfusionMatrix(self):
        return ConfusionMatrix(self.data.labels, self.predLabel)

    def rocCurve(self):
        # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        fpr, tpr, threshold = roc_curve(self.data.labels, self.predictions)
        rocCurve = RocCurve("rocCurve", fpr, tpr)
        logging.info("RESULT DIR: {0}".format(self.resultDIR))
        # rocCurve.fileOutput(self.resultDIR)
        return rocCurve

    def report(self):
        return Report(self.data.labels, self.predLabel)


class Output:  # base output...
    data = None
    stringType = None

    def __init__(self, type, modelOutput):
        self.data = modelOutput
        self.stringType = type

    def fileOutput(self, modelName):  # now what if its a table? or a graph?
        rootName = self.stringType
        base = modelName + "/" + rootName
        # this is ...
        # if os.path.isdir("../results"):
        #	if os.path.isdir(base):

        # if not os.path.isdir("results"):
        # 	os.mkdir("results")
        # if not os.path.isdir(base):
        # 	os.mkdir(base)

        # os.mkdir(path)

        logging.info("results/" + modelName)
        f = open(base, "w")
        f.write(str(self.textOutput()[1]))  # this needs to be some kind of representation
        f.close()

    def textOutput(self):
        return (self.stringType, self.data)

    def printOutput(self, file=None):
        if file is not None:
            print(self.data, file=file)
        # print(self.textOutput(),file=file)
        else:
            print(self.data)
        # print(self.textOutput())


# FEATURE VISUALIZER 

# class FeatureVisualizer(Output): # this requires the model....
# def __init__(self,labels,predictions):


class LabelOutput(Output):
    def __init__(self, labels, predictions):
        self.labels = labels
        self.predictions = predictions
        self.data = self.setData()

    def setData(self):
        pass


class ConfusionMatrix(LabelOutput):
    def setData(self):
        return confusion_matrix(self.labels, self.predictions)


class Report(LabelOutput):
    def setData(self):
        return classification_report(self.labels, self.predictions)


class RocCurve(Output):
    fpr = None
    tpr = None

    def __init__(self, type, fpr, tpr):
        # self.data = modelOutput
        self.stringType = type
        self.fpr = fpr
        self.tpr = tpr

    def fileOutput(self, file=None, fileString=None):
        rootName = self.stringType
        # base = modelName+"/"+rootName
        logging.info("ROOT: {0}".format(rootName))
        # root is the type...

        # print('HERE IS THE BASE',fileString)

        roc_auc = auc(self.fpr, self.tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(self.fpr, self.tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        if fileString is not None:
            pltfile = fileString + '.png'
            logging.info("INFO: AUC-ROC curve will be saved as {0}".format(pltfile))
            plt.savefig(pltfile)

    # plt ROC curves for n folds
    def fileOutputForAverage(self, savedData, fileString=None, folds=5):

        rootName = self.stringType
        logging.info("ROOT: {0}".format(rootName))
        rocValues = []
        for n in range(folds):
            labels, predictions = zip(*list(savedData[n]))  # unzip the data
            # predictions = list(zip(*savedData[n])[1]) #unzip the data
            fpr, tpr, threshold = roc_curve(labels, predictions)
            roc_auc = auc(fpr, tpr)
            rocValues.append(roc_auc)
            plt.plot(fpr, tpr, color='gainsboro')

        plt.plot(fpr, tpr, color='darkblue', label='Mean AUC = %0.3f' % np.mean(rocValues))
        plt.plot(fpr, tpr, color='darkred', label='Median AUC = %0.3f' % np.median(rocValues))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title('Receiver Operating Characteristic,' + 'Range: ' + str('%.3f' % np.min(rocValues)) + ' - ' + str(
            '%.3f' % np.max(rocValues)))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        # logging.info("RESULT DIR: {0}".format(self.resultDIR))
        if fileString is not None:
            pltfile = fileString + '.png'
            logging.info("INFO: AUC-ROC curve will be saved as {0}".format(pltfile))
            plt.savefig(pltfile)

    def printOutput(self, file=None):
        if file is not None:  # if we've got a file, we wont print it
            return

        # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = auc(self.fpr, self.tpr)
        # method I: plt

        plt.title('Receiver Operating Characteristic')
        plt.plot(self.fpr, self.tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


class BaseModel:
    MODEL_PROCEDURE = ""

    def __init__(self, MODEL_PROCEDURE, RESULT_DIR=None):
        self.MODEL_PROCEDURE = MODEL_PROCEDURE
        if RESULT_DIR is None:  # control will NEVER come here as RESULT_DIR is mandatory now
            self.MODEL_RUN_NAME = "{0}-{1}".format(self.MODEL_PROCEDURE, str(int(time.time())))
            self.MODEL_DIR = "results/{0}".format(self.MODEL_RUN_NAME)
        else:
            self.MODEL_RUN_NAME = "{0}".format(self.MODEL_PROCEDURE)
            self.MODEL_DIR = RESULT_DIR

    def getFile(self):

        self.createDirectoryIfNeed(self.MODEL_DIR)
        WRITEFILE = self.MODEL_DIR + '/metrics_' + self.MODEL_PROCEDURE + '.txt'
        # open(WRITEDIR, 'a').close()

        fileName = WRITEFILE
        writeSpace = open(fileName, 'w')
        return writeSpace

    def createDirectoryIfNeed(self, dir):
        logging.info("AYYEE: {0}".format(dir))

        if not os.path.isdir(dir):
            os.mkdir(dir)

    def setClassifier(self, classifier):
        self.m = classifier

    def createResultObjects(self, testData, outputTypes, predictions, saveData=True):

        self.createDirectoryIfNeed("results")

        if saveData:  # we can turn off saving of data...

            writeSpace = self.getFile()

            print(self.m, file=writeSpace)
            print("", file=writeSpace)

            resultList = []
            # resultObject = Result(testData,predictions,self.MODEL_RUN_NAME,modelDIR=self.MODEL_RUN_NAME)
            resultObject = Result(testData, predictions, modelDIR=self.MODEL_DIR)
            for resultType in outputTypes:

                print(resultType, file=writeSpace)
                logging.info("HERES MODEL NAME: {0}".format(self.MODEL_RUN_NAME))
                newResultObject = getattr(resultObject, resultType)()  # self.MODEL_RUN_NAME
                # print(type(newResultObject))
                resultList.append(newResultObject)
                # print(resultObject)
                # print("MODEL DIR",self.MODEL_PROCEDURE) #self.MODEL_RUN_NAME
                # if resultType == "rocCurve" and self.MODEL_PROCEDURE == "XGBCrossVal": # if it's XGB cross val we will write output (hack)
                if resultType == "rocCurve":
                    aucFileName = self.MODEL_DIR + '/auc_' + self.MODEL_PROCEDURE
                    # newResultObject.fileOutput(fileString=self.MODEL_RUN_NAME)
                    newResultObject.fileOutput(fileString=aucFileName)
                else:
                    newResultObject.printOutput(file=writeSpace)
                # resultObject.printOutput(file=writeSpace)
                print("", file=writeSpace)

        else:
            for resultType in outputTypes:
                newResultObject = getattr(resultObject, resultType)(self.MODEL_RUN_NAME)
                resultList.append(newResultObject)

        # for each of the items in the result list, write them to the shared space
        # print ('resultList...........', resultList)
        if (len(resultList) == 1):
            return resultList[0]
        else:
            return iter(resultList)


class SkModel(BaseModel):
    m = None

    def train(self, trainData, param=None):
        # clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')#.fit(X, y)
        self.m = clf.fit(trainData.features, trainData.labels)

    def predict(self, testData, outputTypes):
        # inputData = xgb.DMatrix(testData.features)
        predictions = self.m.predict(testData.features)
        return self.createResultObjects(testData, outputTypes, predictions)

    def cross_val_predict(self, testData, outputTypes):
        # clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')#.fit(X, y)
        # self.m = clf.fit(testData.features,testData.labels)
        predictions = cross_val_predict(self.m, testData.features, y=testData.labels, cv=10)
        return self.createResultObjects(testData, outputTypes, predictions)


class XGBoostModel(BaseModel):
    m = None
    param = None

    def setParam(self, ):
        self.param = param

    def train(self, trainData, param):
        # print (param)
        # np.random.seed(1234)
        # random.seed(1234)
        # dtrain = xgb.DMatrix(trainData.features,label=trainData.labels)
        # bst = xgb.train(param, dtrain, num_boost_round=47) #use the default values of parameters
        # self.m = bst
        # modelName = self.MODEL_DIR + '/' + self.MODEL_PROCEDURE + '.model'
        # bst.save_model(modelName)

        ###FOR SKLEARN WRAPPER###
        bst = xgb.XGBClassifier(**param).fit(trainData.features, trainData.labels)
        # self.m = bst
        modelName = self.MODEL_DIR + '/' + self.MODEL_PROCEDURE + '.model'
        pickle.dump(bst, open(modelName, 'wb'))
        logging.info('Trained ML Model was saved as {0}'.format(modelName))

    def predict(self, testData, outputTypes):
        inputData = xgb.DMatrix(testData.features)
        predictions = self.m.predict(inputData)  #
        # print ('predictions.................', predictions)
        # ypred_bst = np.array(bst.predict(dtest,ntree_limit=bst.best_iteration))`
        # ypred_bst  = ypred_bst > 0.5
        # ypred_bst = ypred_bst.astype(int)
        # if "report" in outputTypes: # small hack for the report feature, we can use this to make sure
        return self.createResultObjects(testData, outputTypes, predictions)

    def predict_using_saved_model(self, testData, idDescription, idNameSymbol, modelName, infoFile):
        # bst = xgb.Booster({'nthread':8})
        # bst.load_model(modelName)
        # inputData = xgb.DMatrix(testData.features)
        # predictions = bst.predict(inputData)

        ###FOR SKLEARN WRAPPER###
        bst = pickle.load(open(modelName, 'rb'))
        print(bst.get_xgb_params())
        inputData = testData.features  # for wrapper
        class01Probs = bst.predict_proba(inputData)  # for wrapper
        predictions = [i[1] for i in class01Probs]  # select class1 probability - wrapper
        proteinInfo = self.fetchProteinInformation(infoFile)
        self.savePredictedProbability(testData, predictions, idDescription, idNameSymbol, proteinInfo, "TEST")

    # def cross_val_predict(self,testData,outputTypes):
    def cross_val_predict(self, testData, idDescription, idNameSymbol, outputTypes, params={}, cv=1):
        # print (params,cv)
        # other model options
        # clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')#.fit(X, y)
        # self.m = clf.fit(testData.features,testData.labels)
        # inputData = xgb.DMatrix(testData.features)
        # these are the params... for this kind of classifier
        # "seed"	"max_depth"	"eta"	"gamma"	"min_child_weight"	"subsample"	"colsample_bytree"	"nrounds"	"auc"
        # 1001	                      10	0.2	                 0.1	0	                                   0.9	                               0.5	39	0.7980698
        # base_score=0.5, booster='gbtree', colsample_bylevel=1,
        # colsample_bynode=1, colsample_bytree=1, gamma=0,
        # learning_rate=0.02, max_delta_step=0, max_depth=4,
        '''
		min_child_weight=1, missing=None, n_estimators=5, n_jobs=1,
			  nthread=1, objective='binary:logistic', random_state=0,
			  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
			  silent=False, subsample=1, verbosity=1
		'''
        # scale_pos_weight
        # scale_pos_weight=testData.posWeight,
        # clf = xgb.XGBClassifier(scale_pos_weight=testData.posWeight,max_depth=10,gamma=0.1,min_child_weight=0,subsample=0.9,colsample_bytree=0.5, n_jobs=8)

        metrics = {"roc": 0., "mcc": 0., "acc": 0.}
        clf = xgb.XGBClassifier(**params)
        self.m = clf
        class01Probs = cross_val_predict(self.m, testData.features, y=testData.labels, cv=cv,
                                         method='predict_proba')  # calls sklearn's cross_val_predict
        predictions = [i[1] for i in class01Probs]  # select class1 probability
        roc, rc, acc, mcc, CM, report = self.createResultObjects(testData, outputTypes, predictions)
        metrics["roc"] = roc.data
        metrics["mcc"] = mcc.data
        metrics["acc"] = acc.data

        # find imporant features and save them in a text file
        importance = Counter(
            clf.fit(testData.features, testData.labels).get_booster().get_score(importance_type='gain'))
        self.saveImportantFeatures(importance, idDescription)
        self.saveImportantFeaturesAsPickle(importance)

        # save predicted class 1 probabilty in a text file
        self.savePredictedProbability(testData, predictions, idDescription, idNameSymbol, "TRAIN")

        # train the model using all train data and save it
        self.train(testData, param=params)
        # return roc,acc,mcc, CM,report,importance
        logging.info("METRICS: {0}".format(str(metrics)))

    '''
	def average_cross_val(self,testData,idDescription,idNameSymbol,outputTypes,folds=1,split=0.8,params={},cv=1):
		# this function will take the average of metrics per fold... which is a random fold
		#print (params,cv)
		#CROSSVAL = 10
		collection = []
		importance = None
		#self.DIR = 
		#self.MODEL_PROCEDURE+"-"+
		#dummyModel = XGBoostModel(self.MODEL_PROCEDURE)
		#DIR = "results/{0}-{1}".format(self.MODEL_PROCEDURE,str(int(time.time())))
		
		#logging.info("THING: {0}".format(dummyModel.MODEL_RUN_NAME))
		
		#DIR = dummyModel.MODEL_RUN_NAME # This will get us a model with a specific timestamp
		
		#logging.info("THIS IS THE DIR {0}".format(DIR))

		metrics = {"average-roc":0., "average-mcc":0., "average-acc":0.} #add mcc and accuracy too
		logging.info("=== RUNNING {0} FOLDS".format(folds))
		
		#Initialize variable to store predicted probs
		predictedProb = []
		totalData = len(testData.labels.tolist())
		logging.info('Total records...{0}'.format(totalData))
		for r in range(totalData):
			predictedProb.append([])
		#print (predictedProb)
		
		for k in range(0,folds):
			logging.info("DOING {0} FOLD".format(k+1))
			
			#newModel = XGBoostModel(self.MODEL_PROCEDURE)
			#train,test = testData.splitSet(split)
			# make a loop, so we can split it 
			
			#newModel = XGBoostModel() 

			# YOU can add in parameters to here
			#{'max_depth':0,'eta':0.1,'gamma':1,'min_child_weight':2} NO PARAMS
			
			#newModel.train(train,{})
			#newModel.train(train,params) #pass the default parameters
			#roc,rc = newModel.predict(test,["roc","rocCurve"])
			#roc,rc,acc,mcc = newModel.predict(test,outputTypes) #use the passed outputTypes
			#roc.printOutput()
		
			clf = xgb.XGBClassifier(**params)
			self.m = clf
			class01Probs = cross_val_predict(self.m,testData.features,y=testData.labels,cv=cv,method='predict_proba') #calls sklearn's cross_val_predict
			predictions = [i[1] for i in class01Probs] #select class1 probability
			roc,rc,acc,mcc = self.createResultObjects(testData,outputTypes,predictions)
			
			#append predicted class 1 probability 
			for r in range(totalData):
				predictedProb[r].append(predictions[r])
			#print (predictedProb)
			metrics["average-roc"] += roc.data
			metrics["average-mcc"] += mcc.data
			metrics["average-acc"] += acc.data
			
			#model.predict ...
			if importance:
				importance = importance + Counter(clf.fit(testData.features,testData.labels).get_booster().get_score(importance_type='gain'))
				#print(Counter(newModel.m.get_score(importance_type='gain')))
			else:
				importance = Counter(clf.fit(testData.features,testData.labels).get_booster().get_score(importance_type='gain'))				
		
		for key in importance:
			importance[key] = importance[key]/folds

		for key in metrics:
			metrics[key] = metrics[key]/folds			
		
		avgPredictedProb = []
		for r in range(totalData):
			avgPredictedProb.append(sum(predictedProb[r])/folds)
			
			
		logging.info("METRICS: {0}".format(str(metrics))) # write this metric to a file...
		
		self.saveImportantFeatures(importance, idDescription) #save important features
		self.saveImportantFeaturesAsPickle(importance)
		#print (avgPredictedProb)
		self.savePredictedProbability(testData, avgPredictedProb, idDescription, idNameSymbol, "TRAIN") #save predicted probabilities
	
		#train the model using all train data and save it
		self.train(testData, param=params)

		#with open(FINALDIR, 'wb') as f:
		#	pickle.dump(importance, f, pickle.HIGHEST_PROTOCOL)
		#return importance
	'''

    # This function divides the data into train and test sets 'n' (number of folds) times.
    # Model trained on the train data is tested on the test data. Average MCC, Accuracy and ROC
    # is reported.
    def average_cross_val(self, allData, idDescription, idNameSymbol, outputTypes, iterations, testSize=0.2, params={}):

        collection = []
        importance = None
        metrics = {"average-roc": 0., "average-mcc": 0., "average-acc": 0.}  # add mcc and accuracy too
        logging.info("=== RUNNING {0} FOLDS".format(iterations))

        # Initialize variable to store predicted probs of test data
        predictedProb_ROC = []
        predictedProbs = {}  # will be used for o/p file
        seedAUC = {}  # to store seed value and corresponding classification resutls
        for r in range(iterations):
            predictedProb_ROC.append([])
        # print (predictedProb)

        for k in range(0, iterations):
            logging.info("DOING {0} FOLD".format(k + 1))
            clf = xgb.XGBClassifier(**params)
            self.m = clf
            randomState = 1000 + k
            trainData, testData = allData.splitSet(testSize, randomState)

            # Train the model
            bst = clf.fit(trainData.features, trainData.labels)
            # test the model
            class01Probs = bst.predict_proba(testData.features)
            predictions = [i[1] for i in class01Probs]  # select class1 probability
            roc, acc, mcc = self.createResultObjects(testData, outputTypes, predictions)

            # append predicted probability and true class for ROC curve
            predictedProb_ROC[k] = zip(testData.labels.tolist(), predictions)
            proteinIds = list(testData.features.index.values)
            # print ('Selected ids are: ', proteinIds)
            for p in range(len(proteinIds)):
                try:
                    predictedProbs[proteinIds[p]].append(predictions[p])
                except:
                    predictedProbs[proteinIds[p]] = [predictions[p]]

            # print (predictedProb)
            metrics["average-roc"] += roc.data
            metrics["average-mcc"] += mcc.data
            metrics["average-acc"] += acc.data
            seedAUC[randomState] = [roc.data, acc.data, mcc.data]

            # model.predict ...
            if importance:
                importance = importance + Counter(bst.get_booster().get_score(importance_type='gain'))
            else:
                importance = Counter(bst.get_booster().get_score(importance_type='gain'))

        # compute average values
        for key in importance:
            importance[key] = importance[key] / iterations

        for key in metrics:
            metrics[key] = metrics[key] / iterations

        avgPredictedProbs = {}
        for k, v in predictedProbs.items():
            avgPredictedProbs[k] = np.mean(v)

        logging.info("METRICS: {0}".format(str(metrics)))  # write this metrics to a file...

        self.saveImportantFeatures(importance, idDescription)  # save important features
        self.saveImportantFeaturesAsPickle(importance)
        self.saveSeedPerformance(seedAUC)
        # print (avgPredictedProb)
        self.savePredictedProbability(allData, avgPredictedProbs, idDescription, idNameSymbol,
                                      "AVERAGE")  # save predicted probabilities
        # plot ROC curves
        rc = RocCurve("rocCurve", None, None)
        aucFileName = self.MODEL_DIR + '/auc_' + self.MODEL_PROCEDURE
        rc.fileOutputForAverage(predictedProb_ROC, fileString=aucFileName, folds=iterations)

    # FEATURE SEARCH, will create the dataset with different sets of features, and search over them to get resutls
    def gridSearch(self, allData, idDescription, idNameSymbol, outputTypes, paramGrid, rseed, nthreads):

        # param_comb = 200
        # clf = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
        #			silent=True, nthread=1)

        # random_search = GridSearchCV(clf,
        # 	param_distributions=params,
        # 	n_iter=param_comb,
        # 	scoring='roc_auc', n_jobs=1, cv=None,
        # 	verbose=3,
        # 	random_state=1001)

        # colsample bytree = 0.6
        # gamma = 0.5
        # max depth = 0.8
        # min child weight = 2
        # subsample = 1

        # split test and train data
        testSize = 0.20
        trainData, testData = allData.splitSet(testSize, rseed)
        # print (trainData.features.shape)
        # print (testData.features.shape)
        # print (trainData.labels)
        # print (testData.labels)

        logging.info("XGBoost parameters search started")
        # clf = xgb.XGBClassifier(n_jobs=nthreads, scale_pos_weight=allData.posWeight, random_state=rseed)
        clf = xgb.XGBClassifier(n_jobs=nthreads, random_state=rseed)
        random_search = GridSearchCV(clf,
                                     param_grid=paramGrid,
                                     scoring='roc_auc', cv=5, verbose=3)

        # save the output of each iteration of gridsearch to a file
        tempFileName = self.MODEL_DIR + '/temp.tsv'
        sys.stdout = open(tempFileName, 'w')
        # random_search.fit(trainData.features, trainData.labels)
        random_search.fit(allData.features, allData.labels)

        # model trained with best parameters
        bst = random_search.best_estimator_
        # self.m = bst
        sys.stdout.close()

        # predict the test data using the best estimator
        # metrics = {"roc":0., "mcc":0., "acc":0.}
        # class01Probs = bst.predict_proba(testData.features)
        # predictions = [i[1] for i in class01Probs] #select class1 probability

        # roc,acc,mcc = self.createResultObjects(testData,outputTypes,predictions)
        # metrics["roc"] = roc.data
        # metrics["mcc"] = mcc.data
        # metrics["acc"] = acc.data

        # find imporant features and save them in a text file
        # importance = Counter(bst.get_booster().get_score(importance_type='gain'))
        # self.saveImportantFeatures(importance, idDescription)
        # self.saveImportantFeaturesAsPickle(importance)

        # save predicted class 1 probabilty in a text file
        # self.savePredictedProbability(testData, predictions, idDescription, idNameSymbol, "TRAIN")

        # train the model using all train data and save it
        # self.train(allData, param=random_search.best_params_)

        # save the XGBoost parameters for the best estimator
        self.saveBestEstimator(str(bst))

    # return roc,acc,mcc, CM,report,importance
    # logging.info("METRICS: {0}".format(str(metrics)))

    # save the xgboost parameters selected using GirdSearchCV
    def saveBestEstimator(self, estimator):

        xgbParamFile = self.MODEL_DIR + '/XGBParameters.txt'
        logging.info("XGBoost parameters for the best estimator written to: {0}".format(xgbParamFile))

        # save the optimized parameters for XGboost
        paramVals = estimator.strip().split('(')[1].split(',')
        with open(xgbParamFile, 'w') as fo:
            fo.write('{')
            for vals in paramVals:
                keyVal = vals.strip(' ').split('=')
                if ('scale_pos_weight' in keyVal[0] or
                        'n_jobs' in keyVal[0] or
                        'nthread' in keyVal[0] or
                        'None' in keyVal[1]):
                    continue
                elif (')' in keyVal[1]):  # last parameter
                    line = "'" + keyVal[0].strip().strip(' ') + "': " + keyVal[1].strip().strip(' ').strip(')') + '\n'
                else:
                    line = "'" + keyVal[0].strip().strip(' ') + "': " + keyVal[1].strip().strip(' ').strip(')') + ',\n'
                fo.write(line)
            fo.write('}')

        # save parameters used in each iteration
        tuneFileName = self.MODEL_DIR + '/tune.tsv'
        logging.info("Parameter values in each iteration of GridSearchCV written to: {0}".format(tuneFileName))
        ft = open(tuneFileName, 'w')
        headerWritten = 'N'
        tempFileName = self.MODEL_DIR + '/temp.tsv'
        with open(tempFileName, 'r') as fin:
            for line in fin:
                header = ''
                rec = ''
                if ('score' in line):
                    if (headerWritten == 'N'):
                        vals = line.strip().strip('[CV]').split(',')
                        for val in vals:
                            k, v = val.strip(' ').split('=')
                            header = header + k + '\t'
                            rec = rec + v + '\t'
                        ft.write(header + '\n')
                        ft.write(rec + '\n')
                        headerWritten = 'Y'
                    else:
                        vals = line.strip().strip('[CV]').split(',')
                        for val in vals:
                            k, v = val.strip(' ').split('=')
                            rec = rec + v + '\t'
                        ft.write(rec + '\n')
        ft.close()
        os.remove(tempFileName)  # delete temp file

    # Save important features as pickle file. It will be used by visualization code
    def saveImportantFeaturesAsPickle(self, importance):
        '''
		Save important features in a pickle dictionary
		'''
        featureFile = self.MODEL_DIR + '/featImportance_' + self.MODEL_PROCEDURE + '.pkl'
        logging.info("IMPORTANT FEATURES WRITTEN TO PICKLE FILE {0}".format(featureFile))
        with open(featureFile, 'wb') as ff:
            pickle.dump(importance, ff, pickle.HIGHEST_PROTOCOL)

    # Save seed number and corresponding AUC, ACC and MCC
    def saveSeedPerformance(self, seedAUC):
        '''
		Save important features in a pickle dictionary
		'''
        seedFile = self.MODEL_DIR + '/seed_val_auc.tsv'
        logging.info("SEED VALUES AND THEIR CORRESPONDING AUC/ACC/MCC WRITTEN TO {0}".format(seedFile))

        with open(seedFile, 'w') as ff:
            hdr = 'Seed' + '\t' + 'AUC' + '\t' + 'Accuracy' + '\t' + 'MCC' + '\n'
            ff.write(hdr)
            for k, v in seedAUC.items():
                rec = str(k) + '\t' + str(v[0]) + '\t' + str(v[1]) + '\t' + str(v[2]) + '\n'
                ff.write(rec)

    # Save the important features in a text file.
    def saveImportantFeatures(self, importance, idDescription):
        '''
		This function saves the important features in a text file.
		'''

        dataForDataframe = {'Feature': [], 'Name': [], 'Gain Value': []}
        for feature, gain in importance.items():
            dataForDataframe['Feature'].append(feature)
            dataForDataframe['Gain Value'].append(gain)
            if (feature.lower().islower()):  # alphanumeric feature
                try:
                    dataForDataframe['Name'].append(idDescription[feature])
                except:
                    dataForDataframe['Name'].append(feature)
                    logging.debug('INFO: saveImportantFeatures - Unknown feature = {0}'.format(feature))
            else:
                try:
                    dataForDataframe['Name'].append(idDescription[int(feature)])
                except:
                    dataForDataframe['Name'].append(feature)
                    logging.debug('INFO: saveImportantFeatures - Unknown feature = {0}'.format(feature))

        df = pd.DataFrame(dataForDataframe)
        impFileTsv = self.MODEL_DIR + '/featImportance_' + self.MODEL_PROCEDURE + '.tsv'
        fout = open(impFileTsv, "w")
        df.to_csv(fout, '\t', index=False)
        fout.close()
        logging.info("IMPORTANT FEATURES WRITTEN TO {0}".format(impFileTsv))
        impFileXlsx = self.MODEL_DIR + '/featImportance_' + self.MODEL_PROCEDURE + '.xlsx'
        writer = pd.ExcelWriter(impFileXlsx, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.save()
        logging.info("IMPORTANT FEATURES WRITTEN TO {0}".format(impFileXlsx))

    def fetchProteinInformation(self, infoFile):
        """
        Read infoFile to fetch information about test proteins.
        """
        df = pd.read_excel(infoFile)
        df = df.fillna(value='')
        symbol = df["sym"]
        uniprot = df["uniprot"]
        tdl = df["tdl"]
        fam = df["fam"]
        novelty = df["novelty"]
        importance = df["importance"]
        symInfo = {symbol[i]: [uniprot[i], tdl[i], fam[i], novelty[i], importance[i]] for i in range(len(symbol))}
        return symInfo

    # save predicted probability
    def savePredictedProbability(self, testData, predictions, idDescription, idNameSymbol, proteinInfo, DataType):
        '''
		This function will save true labels and predicted class 1 probability of all protein ids.
		'''
        TrueLabels = []
        proteinIds = list(testData.features.index.values)
        if (DataType == "TEST"):
            for p in proteinIds:
                TrueLabels.append('')
        elif (DataType == "AVERAGE"):
            avgPredictions = []
            trueClass = []
            pids = []
            j = 0
            TrueLabels = testData.labels.tolist()
            for p in proteinIds:
                try:
                    avgPredictions.append(predictions[p])
                    trueClass.append(TrueLabels[j])
                    pids.append(p)
                except:
                    continue
                j += 1
            predictions = avgPredictions
            TrueLabels = trueClass
            proteinIds = pids
        else:
            TrueLabels = testData.labels.tolist()

        # Write data to the file
        dataForDataframe = {'Protein Id': [], 'Symbol': [], 'Name': [], 'Uniprot': [], 'tdl': [], 'fam': [],
                            'novelty': [], 'importance': [], 'True Label': [], 'Predicted Probability': []
                            }

        for i in range(len(proteinIds)):
            proteinId = proteinIds[i]
            dataForDataframe['Protein Id'].append(proteinId)
            dataForDataframe['True Label'].append(TrueLabels[i])
            dataForDataframe['Predicted Probability'].append(predictions[i])

            if proteinId in idNameSymbol:
                dataForDataframe['Name'].append(idDescription[proteinId])
                dataForDataframe['Symbol'].append(idNameSymbol[proteinId])
                if idNameSymbol[proteinId] in proteinInfo:
                    dataForDataframe['Uniprot'].append(proteinInfo[idNameSymbol[proteinId]][0])
                    dataForDataframe['tdl'].append(proteinInfo[idNameSymbol[proteinId]][1])
                    dataForDataframe['fam'].append(proteinInfo[idNameSymbol[proteinId]][2])
                    dataForDataframe['novelty'].append(proteinInfo[idNameSymbol[proteinId]][3])
                    dataForDataframe['importance'].append(proteinInfo[idNameSymbol[proteinId]][4])
                else:
                    dataForDataframe['Uniprot'].append("")
                    dataForDataframe['tdl'].append("")
                    dataForDataframe['fam'].append("")
                    dataForDataframe['novelty'].append("")
                    dataForDataframe['importance'].append("")
            else:
                dataForDataframe['Name'].append("")
                dataForDataframe['Symbol'].append("")
                dataForDataframe['Uniprot'].append("")
                dataForDataframe['tdl'].append("")
                dataForDataframe['fam'].append("")
                dataForDataframe['novelty'].append("")
                dataForDataframe['importance'].append("")

        df = pd.DataFrame(dataForDataframe)
        df = df.sort_values(by=['Predicted Probability'], ascending=False)

        resultsFileTsv = self.MODEL_DIR + '/classificationResults_' + self.MODEL_PROCEDURE + '.tsv'
        fout = open(resultsFileTsv, "w")
        df.to_csv(fout, '\t', index=False)
        fout.close()
        logging.info("CLASSIFICATION RESULTS WRITTEN TO {0}".format(resultsFileTsv))
        resultsFileXlsx = self.MODEL_DIR + '/classificationResults_' + self.MODEL_PROCEDURE + '.xlsx'
        writer = pd.ExcelWriter(resultsFileXlsx, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.save()
        logging.info("CLASSIFICATION RESULTS WRITTEN TO {0}".format(resultsFileXlsx))

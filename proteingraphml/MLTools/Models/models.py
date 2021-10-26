import os, sys
import time, logging, random
import pickle, random
from collections import Counter
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_predict,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)  # .roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None)


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
        self.predLabel = [
            round(p) for p in self.predictions
        ]  # generate label using probability

    def acc(self):
        return Output("ACC", accuracy_score(self.data.labels, self.predLabel))

    def mcc(self):  # Add MCC since data is imbalanced
        return Output(
            "MCC", matthews_corrcoef(self.data.labels, self.predLabel)
        )

    def roc(self):
        roc = Output(
            "AUCROC", roc_auc_score(self.data.labels, self.predictions)
        )
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

        logging.info("results/" + modelName)
        f = open(base, "w")
        f.write(
            str(self.textOutput()[1])
        )  # this needs to be some kind of representation
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

        roc_auc = auc(self.fpr, self.tpr)
        plt.title("Receiver Operating Characteristic")
        plt.plot(self.fpr, self.tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")

        if fileString is not None:
            pltfile = fileString + ".png"
            logging.info(
                "INFO: AUC-ROC curve will be saved as {0}".format(pltfile)
            )
            plt.savefig(pltfile)

    # plt ROC curves for n folds
    def fileOutputForAverage(self, savedData, fileString=None, folds=5):

        rootName = self.stringType
        logging.info("ROOT: {0}".format(rootName))
        rocValues = []
        for n in range(folds):
            labels, predictions = zip(*list(savedData[n]))  # unzip the data
            fpr, tpr, threshold = roc_curve(labels, predictions)
            roc_auc = auc(fpr, tpr)
            rocValues.append(roc_auc)
            plt.plot(fpr, tpr, color="gainsboro")

        plt.plot(
            fpr,
            tpr,
            color="darkblue",
            label="Mean AUC = %0.3f" % np.mean(rocValues),
        )
        plt.plot(
            fpr,
            tpr,
            color="darkred",
            label="Median AUC = %0.3f" % np.median(rocValues),
        )
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(
            "Receiver Operating Characteristic,"
            + "Range: "
            + str("%.3f" % np.min(rocValues))
            + " - "
            + str("%.3f" % np.max(rocValues))
        )
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")

        # logging.info("RESULT DIR: {0}".format(self.resultDIR))
        if fileString is not None:
            pltfile = fileString + ".png"
            logging.info(
                "INFO: AUC-ROC curve will be saved as {0}".format(pltfile)
            )
            plt.savefig(pltfile)

    def printOutput(self, file=None):
        if file is not None:  # if we've got a file, we wont print it
            return

        roc_auc = auc(self.fpr, self.tpr)
        plt.title("Receiver Operating Characteristic")
        plt.plot(self.fpr, self.tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()


class BaseModel:
    MODEL_PROCEDURE = ""

    def __init__(self, MODEL_PROCEDURE, RESULT_DIR=None):
        self.MODEL_PROCEDURE = MODEL_PROCEDURE
        if (
            RESULT_DIR is None
        ):  # control will NEVER come here as RESULT_DIR is mandatory now
            self.MODEL_RUN_NAME = "{0}-{1}".format(
                self.MODEL_PROCEDURE, str(int(time.time()))
            )
            self.MODEL_DIR = "results/{0}".format(self.MODEL_RUN_NAME)
        else:
            self.MODEL_RUN_NAME = "{0}".format(self.MODEL_PROCEDURE)
            self.MODEL_DIR = RESULT_DIR

    def getFile(self):

        self.createDirectoryIfNeed(self.MODEL_DIR)
        WRITEFILE = (
            self.MODEL_DIR + "/metrics_" + self.MODEL_PROCEDURE + ".txt"
        )

        fileName = WRITEFILE
        writeSpace = open(fileName, "w")
        return writeSpace

    def createDirectoryIfNeed(self, dir):
        logging.info("AYYEE: {0}".format(dir))

        if not os.path.isdir(dir):
            os.mkdir(dir)

    def setClassifier(self, classifier):
        self.m = classifier

    def createResultObjects(
        self, testData, outputTypes, predictions, saveData=True
    ):

        self.createDirectoryIfNeed("results")

        if saveData:  # we can turn off saving of data...

            writeSpace = self.getFile()

            print(self.m, file=writeSpace)
            print("", file=writeSpace)

            resultList = []
            # resultObject = Result(testData,predictions,self.MODEL_RUN_NAME,modelDIR=self.MODEL_RUN_NAME)
            resultObject = Result(
                testData, predictions, modelDIR=self.MODEL_DIR
            )
            for resultType in outputTypes:

                print(resultType, file=writeSpace)
                logging.info(
                    "HERES MODEL NAME: {0}".format(self.MODEL_RUN_NAME)
                )
                newResultObject = getattr(
                    resultObject, resultType
                )()  # self.MODEL_RUN_NAME
                # print(type(newResultObject))
                resultList.append(newResultObject)
                # (hack)
                if resultType == "rocCurve":
                    aucFileName = (
                        self.MODEL_DIR + "/auc_" + self.MODEL_PROCEDURE
                    )
                    # newResultObject.fileOutput(fileString=self.MODEL_RUN_NAME)
                    newResultObject.fileOutput(fileString=aucFileName)
                else:
                    newResultObject.printOutput(file=writeSpace)
                # resultObject.printOutput(file=writeSpace)
                print("", file=writeSpace)

        else:
            for resultType in outputTypes:
                newResultObject = getattr(resultObject, resultType)(
                    self.MODEL_RUN_NAME
                )
                resultList.append(newResultObject)

        # for each of the items in the result list, write them to the shared space
        if len(resultList) == 1:
            return resultList[0]
        else:
            return iter(resultList)


class SkModel(BaseModel):
    m = None

    def train(self, trainData, param=None):
        self.m = clf.fit(trainData.features, trainData.labels)

    def predict(self, testData, outputTypes):
        # inputData = xgb.DMatrix(testData.features)
        predictions = self.m.predict(testData.features)
        return self.createResultObjects(testData, outputTypes, predictions)

    def cross_val_predict(self, testData, outputTypes):
        predictions = cross_val_predict(
            self.m, testData.features, y=testData.labels, cv=10
        )
        return self.createResultObjects(testData, outputTypes, predictions)


class XGBoostModel(BaseModel):
    m = None
    param = None

    def setParam(
        self,
    ):
        self.param = param

    def train(self, trainData, param):

        ###FOR SKLEARN WRAPPER###
        bst = xgb.XGBClassifier(**param).fit(
            trainData.features, trainData.labels
        )
        # self.m = bst
        modelName = self.MODEL_DIR + "/" + self.MODEL_PROCEDURE + ".model"
        pickle.dump(bst, open(modelName, "wb"))
        logging.info("Trained ML Model was saved as {0}".format(modelName))

    def predict(self, testData, outputTypes):
        inputData = xgb.DMatrix(testData.features)
        predictions = self.m.predict(inputData)  #
        return self.createResultObjects(testData, outputTypes, predictions)

    def predict_using_saved_model(
        self, testData, idDescription, idNameSymbol, modelName, infoFile
    ):

        ###FOR SKLEARN WRAPPER###
        bst = pickle.load(open(modelName, "rb"))
        print(bst.get_xgb_params())
        inputData = testData.features  # for wrapper
        class01Probs = bst.predict_proba(inputData)  # for wrapper
        predictions = [
            i[1] for i in class01Probs
        ]  # select class1 probability - wrapper
        proteinInfo = self.fetchProteinInformation(infoFile)
        self.savePredictedProbability(
            testData,
            predictions,
            idDescription,
            idNameSymbol,
            proteinInfo,
            "TEST",
        )

    def cross_val_predict(
        self,
        testData,
        idDescription,
        idNameSymbol,
        idSource,
        outputTypes,
        params={},
        cv=5,
    ):
        logging.info(
            "Running XGboost 5-fold cross-validation on the train set"
        )

        metrics = {"roc": 0.0, "mcc": 0.0, "acc": 0.0}
        clf = xgb.XGBClassifier(**params)
        self.m = clf
        class01Probs = cross_val_predict(
            self.m,
            testData.features,
            y=testData.labels,
            cv=cv,
            method="predict_proba",
        )  # calls sklearn's cross_val_predict
        predictions = [i[1] for i in class01Probs]  # select class1 probability
        roc, rc, acc, mcc, CM, report = self.createResultObjects(
            testData, outputTypes, predictions
        )
        metrics["roc"] = roc.data
        metrics["mcc"] = mcc.data
        metrics["acc"] = acc.data

        # find important features and save them in a text file
        importance = Counter(
            clf.fit(testData.features, testData.labels)
            .get_booster()
            .get_score(importance_type="gain")
        )
        self.saveImportantFeatures(
            importance, idDescription, idNameSymbol, idSource=idSource
        )
        self.saveImportantFeaturesAsPickle(importance)

        # save predicted class 1 probability in a text file
        self.savePredictedProbability(
            testData, predictions, idDescription, idNameSymbol, "", "TRAIN"
        )

        # train the model using all train data and save it
        self.train(testData, param=params)
        # return roc,acc,mcc, CM,report,importance
        logging.info("METRICS: {0}".format(str(metrics)))

    def average_cross_val(
        self,
        allData,
        idDescription,
        idNameSymbol,
        idSource,
        outputTypes,
        iterations,
        testSize=0.2,
        params={},
    ):
        # This function divides the data into train and test sets 'n' (number of folds) times.
        # Model trained on the train data is tested on the test data. Average MCC, Accuracy and ROC
        # is reported.
        logging.info("Running ML models to compute average MCC/ROC/ACC")
        importance = None
        metrics = {
            "average-roc": 0.0,
            "average-mcc": 0.0,
            "average-acc": 0.0,
        }  # add mcc and accuracy too
        logging.info("=== RUNNING {0} FOLDS".format(iterations))

        # Initialize variable to store predicted probs of test data
        predictedProb_ROC = []
        predictedProbs = {}  # will be used for o/p file
        seedAUC = (
            {}
        )  # to store seed value and corresponding classification resutls
        for r in range(iterations):
            predictedProb_ROC.append([])

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
            predictions = [
                i[1] for i in class01Probs
            ]  # select class1 probability
            roc, acc, mcc = self.createResultObjects(
                testData, outputTypes, predictions
            )

            # append predicted probability and true class for ROC curve
            predictedProb_ROC[k] = zip(testData.labels.tolist(), predictions)
            proteinIds = list(testData.features.index.values)
            # print ('Selected ids are: ', proteinIds)
            for p in range(len(proteinIds)):
                try:
                    predictedProbs[proteinIds[p]].append(predictions[p])
                except:
                    predictedProbs[proteinIds[p]] = [predictions[p]]

            metrics["average-roc"] += roc.data
            metrics["average-mcc"] += mcc.data
            metrics["average-acc"] += acc.data
            seedAUC[randomState] = [roc.data, acc.data, mcc.data]

            # model.predict ...
            if importance:
                importance = importance + Counter(
                    bst.get_booster().get_score(importance_type="gain")
                )
            else:
                importance = Counter(
                    bst.get_booster().get_score(importance_type="gain")
                )

        # compute average values
        for key in importance:
            importance[key] = importance[key] / iterations

        for key in metrics:
            metrics[key] = metrics[key] / iterations

        avgPredictedProbs = {}
        for k, v in predictedProbs.items():
            avgPredictedProbs[k] = np.mean(v)

        logging.info(
            "METRICS: {0}".format(str(metrics))
        )  # write this metrics to a file...

        self.saveImportantFeatures(
            importance, idDescription, idNameSymbol, idSource=idSource
        )  # save important features
        self.saveImportantFeaturesAsPickle(importance)
        self.saveSeedPerformance(seedAUC)
        # print (avgPredictedProb)
        self.savePredictedProbability(
            allData,
            avgPredictedProbs,
            idDescription,
            idNameSymbol,
            "",
            "AVERAGE",
        )  # save predicted probabilities
        # plot ROC curves
        rc = RocCurve("rocCurve", None, None)
        aucFileName = self.MODEL_DIR + "/auc_" + self.MODEL_PROCEDURE
        rc.fileOutputForAverage(
            predictedProb_ROC, fileString=aucFileName, folds=iterations
        )

    # FEATURE SEARCH, will create the dataset with different sets of features, and search over them to get resutls
    def gridSearch(
        self,
        allData,
        idDescription,
        idNameSymbol,
        outputTypes,
        paramGrid,
        rseed,
        nthreads,
    ):

        # split test and train data

        logging.info("XGBoost parameters search started")
        clf = xgb.XGBClassifier(random_state=rseed)
        random_search = GridSearchCV(
            clf,
            n_jobs=nthreads,
            param_grid=paramGrid,
            scoring="roc_auc",
            cv=5,
            verbose=7,
        )

        # save the output of each iteration of gridsearch to a file
        tempFileName = self.MODEL_DIR + "/temp.tsv"
        sys.stdout = open(tempFileName, "w")
        random_search.fit(allData.features, allData.labels)

        # model trained with best parameters
        bst = random_search.best_estimator_
        # self.m = bst
        sys.stdout.close()
        self.saveBestEstimator(str(bst))

        # predict the test data using the best estimator

    # save the xgboost parameters selected using GirdSearchCV
    def saveBestEstimator(self, estimator):

        xgbParamFile = self.MODEL_DIR + "/XGBParameters.txt"
        logging.info(
            "XGBoost parameters for the best estimator written to: {0}".format(
                xgbParamFile
            )
        )

        # save the optimized parameters for XGboost
        paramVals = estimator.strip().split("(")[1].split(",")
        with open(xgbParamFile, "w") as fo:
            fo.write("{")
            for vals in paramVals:
                keyVal = vals.strip(" ").split("=")
                if (
                    "scale_pos_weight" in keyVal[0]
                    or "n_jobs" in keyVal[0]
                    or "nthread" in keyVal[0]
                    or "None" in keyVal[1]
                ):
                    continue
                elif ")" in keyVal[1]:  # last parameter
                    line = (
                        "'"
                        + keyVal[0].strip().strip(" ")
                        + "': "
                        + keyVal[1].strip().strip(" ").strip(")")
                        + "\n"
                    )
                else:
                    line = (
                        "'"
                        + keyVal[0].strip().strip(" ")
                        + "': "
                        + keyVal[1].strip().strip(" ").strip(")")
                        + ",\n"
                    )
                fo.write(line)
            fo.write("}")

        # save parameters used in each iteration
        tuneFileName = self.MODEL_DIR + "/tune.tsv"
        logging.info(
            "Parameter values in each iteration of GridSearchCV written to: {0}".format(
                tuneFileName
            )
        )
        ft = open(tuneFileName, "w")
        headerWritten = "N"
        tempFileName = self.MODEL_DIR + "/temp.tsv"
        with open(tempFileName, "r") as fin:
            for line in fin:
                header = ""
                rec = ""
                if "score" in line:
                    if headerWritten == "N":
                        vals = line.strip().strip("[CV]").split(",")
                        for val in vals:
                            k, v = val.strip(" ").split("=")
                            header = header + k + "\t"
                            rec = rec + v + "\t"
                        ft.write(header + "\n")
                        ft.write(rec + "\n")
                        headerWritten = "Y"
                    else:
                        vals = line.strip().strip("[CV]").split(",")
                        for val in vals:
                            k, v = val.strip(" ").split("=")
                            rec = rec + v + "\t"
                        ft.write(rec + "\n")
        ft.close()
        os.remove(tempFileName)  # delete temp file

    # Save important features as pickle file. It will be used by visualization code
    def saveImportantFeaturesAsPickle(self, importance):
        """
        Save important features in a pickle dictionary
        """
        featureFile = (
            self.MODEL_DIR + "/featImportance_" + self.MODEL_PROCEDURE + ".pkl"
        )
        logging.info(
            "IMPORTANT FEATURES WRITTEN TO PICKLE FILE {0}".format(featureFile)
        )
        with open(featureFile, "wb") as ff:
            pickle.dump(importance, ff, pickle.HIGHEST_PROTOCOL)

    # Save seed number and corresponding AUC, ACC and MCC
    def saveSeedPerformance(self, seedAUC):
        """
        Save important features in a pickle dictionary
        """
        seedFile = self.MODEL_DIR + "/seed_val_auc.tsv"
        logging.info(
            "SEED VALUES AND THEIR CORRESPONDING AUC/ACC/MCC WRITTEN TO {0}".format(
                seedFile
            )
        )

        with open(seedFile, "w") as ff:
            hdr = (
                "Seed" + "\t" + "AUC" + "\t" + "Accuracy" + "\t" + "MCC" + "\n"
            )
            ff.write(hdr)
            for k, v in seedAUC.items():
                rec = (
                    str(k)
                    + "\t"
                    + str(v[0])
                    + "\t"
                    + str(v[1])
                    + "\t"
                    + str(v[2])
                    + "\n"
                )
                ff.write(rec)

    # Save the important features in a text file.
    def saveImportantFeatures(
        self, importance, idDescription, idNameSymbol, idSource=None
    ):
        """
        This function saves the important features in a text file.
        """

        dataForDataframe = {
            "Feature": [],
            "Symbol": [],
            "Cell_id": [],
            "Drug_name": [],
            "Tissue": [],
            "Source": [],
            "Name": [],
            "Gain Value": [],
        }
        for feature, gain in importance.items():
            dataForDataframe["Feature"].append(feature)
            dataForDataframe["Gain Value"].append(gain)

            if feature.lower().islower():  # alphanumeric feature
                # source
                if idSource is not None and feature in idSource:
                    dataForDataframe["Source"].append(idSource[feature])
                else:
                    dataForDataframe["Source"].append("")

                # Name
                if feature in idDescription:
                    dataForDataframe["Name"].append(idDescription[feature])
                else:
                    dataForDataframe["Name"].append("")
                    logging.debug(
                        "INFO: saveImportantFeatures - Unknown feature = {0}".format(
                            feature
                        )
                    )

                # Symbol
                if feature in idNameSymbol:
                    dataForDataframe["Symbol"].append(idNameSymbol[feature])
                else:
                    dataForDataframe["Symbol"].append("")

            else:  # numeric feature
                # Source
                if idSource is not None:
                    dataForDataframe["Source"].append(idSource[int(feature)])
                else:
                    dataForDataframe["Source"].append("")

                # Name
                if int(feature) in idDescription:
                    dataForDataframe["Name"].append(
                        idDescription[int(feature)]
                    )
                else:
                    dataForDataframe["Name"].append("")
                    logging.debug(
                        "INFO: saveImportantFeatures - Unknown feature = {0}".format(
                            feature
                        )
                    )

                # Symbol
                if int(feature) in idNameSymbol:
                    dataForDataframe["Symbol"].append(
                        idNameSymbol[int(feature)]
                    )
                else:
                    dataForDataframe["Symbol"].append("")

            # for CCLE only
            if feature in idSource and idSource[feature] == "ccle":
                cid = feature[: feature.index("_")]
                tissue = feature[feature.index("_") + 1 :]
                dataForDataframe["Cell_id"].append(cid)
                dataForDataframe["Tissue"].append(tissue)
                dataForDataframe["Drug_name"].append("")

            # for LINCS only.
            # LINCS features contain pert_id and cell_id, separated by :. The drug_id in “olegdb” is the DrugCentral
            # ID, which is DrugCentral chemical structure (active ingredient) ID. The pert_id from LINCS features is
            # used as drug_id to fetch the drug name from the dictionary.

            elif feature in idSource and idSource[feature] == "lincs":
                drugid = feature[: feature.index(":")]
                try:
                    drugname = idSource["drug_" + drugid]
                except:
                    drugname = ""
                cid = feature[feature.index(":") + 1 :]
                dataForDataframe["Cell_id"].append(cid)
                dataForDataframe["Drug_name"].append(drugname)
                dataForDataframe["Tissue"].append("")
            else:
                dataForDataframe["Cell_id"].append("")
                dataForDataframe["Drug_name"].append("")
                dataForDataframe["Tissue"].append("")

        # for k,v in dataForDataframe.items():
        #    print(k, len(v))

        df = pd.DataFrame(dataForDataframe)
        df = df.sort_values(by=["Gain Value"], ascending=False)
        impFileTsv = (
            self.MODEL_DIR + "/featImportance_" + self.MODEL_PROCEDURE + ".tsv"
        )
        fout = open(impFileTsv, "w")
        df.to_csv(fout, "\t", index=False)
        fout.close()
        logging.info("IMPORTANT FEATURES WRITTEN TO {0}".format(impFileTsv))
        impFileXlsx = (
            self.MODEL_DIR
            + "/featImportance_"
            + self.MODEL_PROCEDURE
            + ".xlsx"
        )
        writer = pd.ExcelWriter(impFileXlsx, engine="xlsxwriter")
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        writer.save()
        logging.info("IMPORTANT FEATURES WRITTEN TO {0}".format(impFileXlsx))

    def fetchProteinInformation(self, infoFile):
        """
        Read infoFile to fetch information about test proteins.
        """
        df = pd.read_excel(infoFile)
        df = df.fillna(value="")
        symbol = df["sym"]
        uniprot = df["uniprot"]
        tdl = df["tdl"]
        fam = df["fam"]
        novelty = df["novelty"]
        importance = df["importance"]
        symInfo = {
            symbol[i]: [uniprot[i], tdl[i], fam[i], novelty[i], importance[i]]
            for i in range(len(symbol))
        }
        return symInfo

    # save predicted probability
    def savePredictedProbability(
        self,
        testData,
        predictions,
        idDescription,
        idNameSymbol,
        proteinInfo,
        DataType,
    ):
        """
        This function will save true labels and predicted class 1 probability of all protein ids.
        """
        TrueLabels = []
        proteinIds = list(testData.features.index.values)
        if DataType == "TEST":
            for p in proteinIds:
                TrueLabels.append("")
        elif DataType == "AVERAGE":
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
        dataForDataframe = {
            "Protein Id": [],
            "Symbol": [],
            "Name": [],
            "Uniprot": [],
            "tdl": [],
            "fam": [],
            "novelty": [],
            "importance": [],
            "True Label": [],
            "Predicted Probability": [],
        }

        for i in range(len(proteinIds)):
            proteinId = proteinIds[i]
            dataForDataframe["Protein Id"].append(proteinId)
            dataForDataframe["True Label"].append(TrueLabels[i])
            dataForDataframe["Predicted Probability"].append(predictions[i])

            if proteinId in idNameSymbol:
                dataForDataframe["Name"].append(idDescription[proteinId])
                dataForDataframe["Symbol"].append(idNameSymbol[proteinId])
            else:
                dataForDataframe["Name"].append(proteinId)
                dataForDataframe["Symbol"].append(proteinId)

            if DataType == "TEST" and idNameSymbol[proteinId] in proteinInfo:
                v = proteinInfo[idNameSymbol[proteinId]]
                dataForDataframe["Uniprot"].append(v[0])
                dataForDataframe["tdl"].append(v[1])
                dataForDataframe["fam"].append(v[2])
                dataForDataframe["novelty"].append(v[3])
                dataForDataframe["importance"].append(v[4])
            else:
                dataForDataframe["Uniprot"].append("")
                dataForDataframe["tdl"].append("")
                dataForDataframe["fam"].append("")
                dataForDataframe["novelty"].append("")
                dataForDataframe["importance"].append("")

        df = pd.DataFrame(dataForDataframe)
        df = df.sort_values(by=["Predicted Probability"], ascending=False)

        resultsFileTsv = (
            self.MODEL_DIR
            + "/classificationResults_"
            + self.MODEL_PROCEDURE
            + ".tsv"
        )
        fout = open(resultsFileTsv, "w")
        df.to_csv(fout, "\t", index=False)
        fout.close()
        logging.info(
            "CLASSIFICATION RESULTS WRITTEN TO {0}".format(resultsFileTsv)
        )
        resultsFileXlsx = (
            self.MODEL_DIR
            + "/classificationResults_"
            + self.MODEL_PROCEDURE
            + ".xlsx"
        )
        writer = pd.ExcelWriter(resultsFileXlsx, engine="xlsxwriter")
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        writer.save()
        logging.info(
            "CLASSIFICATION RESULTS WRITTEN TO {0}".format(resultsFileXlsx)
        )

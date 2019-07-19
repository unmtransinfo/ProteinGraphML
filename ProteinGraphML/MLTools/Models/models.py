import time

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_predict,StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import cross_val_score,

from sklearn.metrics import accuracy_score,auc,confusion_matrix,classification_report
from sklearn.metrics import roc_auc_score,roc_curve #.roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None)

# this model system will hopefully make a simple API for dealing with large data 
import matplotlib.pyplot as plt
# iterating on our platform across domains 

import os
from collections import Counter

# update all of this.... later 

class Result:
	
	data = None
	predictions = None
	
	def __init__(self,dataOut,predictions,modelName):
		self.data = dataOut
		self.predictions = predictions
		self.modelName = modelName
		# we put the functions here which actually convert the data to a binary score 
	def acc(self):
		return Output("ACC",accuracy_score(self.data.labels,self.predictions))
	def roc(self):
		roc = Output("AUCROC",roc_auc_score(self.data.labels,self.predictions))
		roc.fileOutput(self.modelName)
		return roc

	def ConfusionMatrix(self):
		return ConfusionMatrix(self.data.labels,self.predictions)

	def rocCurve(self):
		 #fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

		fpr, tpr, threshold = roc_curve(self.data.labels,self.predictions)
		rocCurve = RocCurve("rocCurve",fpr, tpr)
		rocCurve.fileOutput(self.modelName)
		return rocCurve
	
	def report(self):
		return Report(self.data.labels,self.predictions)

class Output: # base output...
	data = None
	stringType = None
	def __init__(self,type,modelOutput):
		self.data = modelOutput
		self.stringType = type

	def fileOutput(self,modelName): # now what if its a table? or a graph?
		
		rootName = self.stringType
		
		base = modelName+"/"+rootName
		
		#if os.path.isdir("../results"):
		#	if os.path.isdir(base):

		# if not os.path.isdir("results"):
		# 	os.mkdir("results")
		# if not os.path.isdir(base):
		# 	os.mkdir(base)

			#os.mkdir(path)

		print("results/"+modelName)
		f = open(base, "w")
		f.write(str(self.textOutput()[1])) # this needs to be some kind of representation
		f.close()

	def textOutput(self):
		return (self.stringType,self.data)

	def printOutput(self):
		print(self.textOutput())


# FEATURE VISUALIZER 

#class FeatureVisualizer(Output): # this requires the model.... 
#def __init__(self,labels,predictions):



class LabelOutput(Output):
	def __init__(self,labels,predictions):
		self.labels = labels
		self.predictions = predictions

class ConfusionMatrix(LabelOutput):
	
	def printOutput(self):
		print(confusion_matrix(self.labels,self.predictions))

class Report(LabelOutput):
	def printOutput(self):
		print(classification_report(self.labels,self.predictions))

class RocCurve(Output):
	fpr = None 
	tpr = None
	def __init__(self,type,fpr,tpr):
		#self.data = modelOutput
		self.stringType = type
		self.fpr = fpr
		self.tpr = tpr

	def fileOutput(self,modelName):
		rootName = self.stringType		
		base = modelName+"/"+rootName

		roc_auc = auc(self.fpr, self.tpr)
		plt.title('Receiver Operating Characteristic')
		plt.plot(self.fpr, self.tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')

		plt.savefig(base+'.png')

	def printOutput(self):

		#fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
		roc_auc = auc(self.fpr, self.tpr)
		# method I: plt
		
		plt.title('Receiver Operating Characteristic')
		plt.plot(self.fpr, self.tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()

class BaseModel:
	def setClassifier(self,classifier):
		self.m = classifier
	
	def createResultObjects(self,testData,outputTypes,predictions):
		if not os.path.isdir("results"):
			os.mkdir("results")
		
		modelName = "results/"+type(self.m).__name__+"-"+str(int(time.time()))
		if not os.path.isdir(modelName):
			os.mkdir(modelName)
		
		#if not os.path.isdir("results/"+type(self.m).__name__+"-"+str(int(time.time()))):
		#	os.mkdir("results/"+type(self.m).__name__+)


		#print('HERE IS THE MODEL',type(self.m).__name__)
		# write a file with this name, this timestamp
		# copy the model to the file as well? no, more like copy the splits...
		resultList = []
		resultObject = Result(testData,predictions,modelName)
		for resultType in outputTypes:
			resultList.append(getattr(resultObject,resultType)())

		if(len(resultList) == 1):
			return resultList[0]
		else:
			return iter(resultList)

class SkModel(BaseModel):
	m = None
	


	def train(self,trainData,param=None):
		#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')#.fit(X, y)
		self.m = clf.fit(trainData.features,trainData.labels)
	
	def predict(self,testData,outputTypes): 
		#inputData = xgb.DMatrix(testData.features)
		predictions = self.m.predict(testData.features)
		return self.createResultObjects(testData,outputTypes,predictions)
	
	def cross_val_predict(self,testData,outputTypes):
		#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')#.fit(X, y)
		#self.m = clf.fit(testData.features,testData.labels)
		predictions = cross_val_predict(self.m,testData.features,y=testData.labels,cv=10)
		return self.createResultObjects(testData,outputTypes,predictions)

class XGBoostModel(BaseModel):
	m = None
	param = None
	def setParam(self,):
		self.param = param

	def train(self,trainData,param):		
		dtrain = xgb.DMatrix(trainData.features,label=trainData.labels)				
		bst = xgb.train(param, dtrain,num_boost_round=50)
		self.m = bst
	
	def predict(self,testData,outputTypes):
		inputData = xgb.DMatrix(testData.features)
		predictions = self.m.predict(inputData)
		return self.createResultObjects(testData,outputTypes,predictions)		

	def cross_val_predict(self,testData,outputTypes):
		#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')#.fit(X, y)
		#self.m = clf.fit(testData.features,testData.labels)
		
		#inputData = xgb.DMatrix(testData.features)
		clf = xgb.XGBClassifier(learning_rate=0.02, n_estimators=5, objective='binary:logistic',
                    silent=False, nthread=1)
		self.m = clf
		predictions = cross_val_predict(self.m,testData.features,y=testData.labels,cv=10)
		return self.createResultObjects(testData,outputTypes,predictions)

	def average_cross_val(self,testData,outputTypes,folds=1,split=0.8):
		# this function will take the average of metrics per fold... which is a random fold
		CROSSVAL = 10

		collection = []
		importance = None


		for k in range(0,folds):

			newModel = XGBoostModel()

			train,test = testData.splitSet(split)
			# make a loop, so we can split it 
			print("train",train.features.shape)

			#newModel = XGBoostModel()
			newModel.train(train,{'max_depth':7,'eta':0.1,'gamma':1,'min_child_weight':2})

			#model.predict
			if importance:
				importance = importance + Counter(newModel.m.get_score(importance_type='gain'))
			else:
				importance = Counter(newModel.m.get_score(importance_type='gain'))
				print(importance)

		
		for key in importance:
			importance[key] = importance[key]/folds

		return importance


	# FEATURE SEARCH, will create the dataset with different sets of features, and search over them to get resutls

	def gridSearch(self,totalData,params):

		
		param_comb = 200
		clf = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

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

		random_search = GridSearchCV(clf, 
			param_grid=params, 
			# n_iter=param_comb, 
			scoring='roc_auc', n_jobs=1, cv=None, 
			verbose=3)

		a = random_search.fit(totalData.features, totalData.labels)

		return a


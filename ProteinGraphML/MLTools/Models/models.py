import time
import pickle

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
	space = None

	def __init__(self,dataOut,predictions,modelName,space=""):
		self.data = dataOut
		self.predictions = predictions
		self.modelName = modelName
		self.space = space
		# we put the functions here which actually convert the data to a binary score 
	def acc(self):
		return Output("ACC",accuracy_score(self.data.labels,self.predictions))
	def roc(self):
		roc = Output("AUCROC",roc_auc_score(self.data.labels,self.predictions))
		#roc.fileOutput(self.modelName)


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
		# this is ... 
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

	def printOutput(self,file=None):
		if file is not None:
			print(self.data,file=file)
			#print(self.textOutput(),file=file)
		else:
			print(self.data)
			#print(self.textOutput())


# FEATURE VISUALIZER 

#class FeatureVisualizer(Output): # this requires the model.... 
#def __init__(self,labels,predictions):



class LabelOutput(Output):
	def __init__(self,labels,predictions):
		self.labels = labels
		self.predictions = predictions
		self.data = self.setData()
	
	def setData(self):
		pass


class ConfusionMatrix(LabelOutput):
	def setData(self):
		return confusion_matrix(self.labels,self.predictions)

class Report(LabelOutput):
	def setData(self):
		return classification_report(self.labels,self.predictions)

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
	
	def createResultObjects(self,testData,outputTypes,predictions,saveData=True):

		# can we 

		if not os.path.isdir("results"):
			os.mkdir("results")

		
		if saveData:  # we can turn off saving of data... 
			

			modelName = "results/"+type(self.m).__name__+"-"+str(int(time.time()))		
	 

			fileName = '{0}.txt'.format(modelName)
			open(fileName, 'a').close()
			if not os.path.isdir(modelName):
				os.mkdir(modelName)
			
			writeSpace = open(fileName, 'w')

			print(self.m,file=writeSpace)
			print("",file=writeSpace)

			resultList = []
			resultObject = Result(testData,predictions,modelName)
			for resultType in outputTypes:
				print(resultType,file=writeSpace)
				newResultObject = getattr(resultObject,resultType)()
				resultList.append(newResultObject)
				newResultObject.printOutput(file=writeSpace)
				print("",file=writeSpace)
		else:
			for resultType in outputTypes:
				newResultObject = getattr(resultObject,resultType)()
				resultList.append(newResultObject)

		# for each of the items in the result list, write them to the shared space

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
		# these are the params... for this kind of classifier
		#"seed"	"max_depth"	"eta"	"gamma"	"min_child_weight"	"subsample"	"colsample_bytree"	"nrounds"	"auc"
		#1001	                      10	0.2	                 0.1	0	                                   0.9	                               0.5	39	0.7980698
		#base_score=0.5, booster='gbtree', colsample_bylevel=1,
		#colsample_bynode=1, colsample_bytree=1, gamma=0,
		#learning_rate=0.02, max_delta_step=0, max_depth=4,
		'''
		min_child_weight=1, missing=None, n_estimators=5, n_jobs=1,
			  nthread=1, objective='binary:logistic', random_state=0,
			  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
			  silent=False, subsample=1, verbosity=1
		'''
		#scale_pos_weight
		#scale_pos_weight=testData.posWeight,
		clf = xgb.XGBClassifier(scale_pos_weight=testData.posWeight,max_depth=10,gamma=0.1,min_child_weight=0,subsample=0.9,colsample_bytree=0.5)
		self.m = clf
		predictions = cross_val_predict(self.m,testData.features,y=testData.labels,cv=10)
		return self.createResultObjects(testData,outputTypes,predictions)

	def average_cross_val(self,testData,outputTypes,folds=1,split=0.8,params={}):
		# this function will take the average of metrics per fold... which is a random fold
		CROSSVAL = 10
		collection = []
		importance = None

		metrics = {"roc":0.}
		for k in range(0,folds):
			print("DOING 1 FOLD")
			newModel = XGBoostModel()
			train,test = testData.splitSet(split)
			# make a loop, so we can split it 
			#print("train",train.features.shape,train.posWeight)
			#print("test",test.features.shape,test.posWeight)
			#newModel = XGBoostModel() 
			#{'max_depth':0,'eta':0.1,'gamma':1,'min_child_weight':2} NO PARAMS

			#'max_depth':0
			newModel.train(train,{})

			roc = newModel.predict(test,["roc"])
			print("roc")
			roc.printOutput()
			metrics["roc"] += roc.data

			#model.predict ...
			if importance:
				importance = importance + Counter(newModel.m.get_score(importance_type='gain'))
				print(Counter(newModel.m.get_score(importance_type='gain')))
			else:
				importance = Counter(newModel.m.get_score(importance_type='gain'))				
		
		for key in importance:
			importance[key] = importance[key]/folds

		for key in metrics:
			metrics[key] = metrics[key]/folds			

		print("METRTCS",metrics) # write this metric to a file...
		#print(importance)

		with open('results/firsty.pkl', 'wb') as f:
			pickle.dump(importance, f, pickle.HIGHEST_PROTOCOL)

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




from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score #.roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None)


# this model system will hopefully make a simple API for dealing with large data 

# iterating on our platform across domains 

class Result:
	
	data = None
	predictions = None
	
	def __init__(self,dataOut,predictions):
		self.data = dataOut
		self.predictions = predictions
		# we put the functions here which actually convert the data to a binary score 
	def acc(self):
		return Output("ACC",accuracy_score(self.data.labels,self.predictions))
	def roc(self):
		return Output("AUCROC",roc_auc_score(self.data.labels,self.predictions))

class Output: # base output...
	data = None
	stringType = None
	def __init__(self,type,modelOutput):
		self.data = modelOutput
		self.stringType = type
		
	def fileOutput(): # now what if its a table? or a graph?
		f = open(self.stringType+time.time(), "x")
		f.write(self.textOutput())
		f.close()

	def textOutput(self):
		return (self.stringType,self.data)

	def printOutput(self):
		print(self.textOutput())


class BaseModel:

	def createResultObjects(self,testData,outputTypes,predictions):
		resultList = []
		resultObject = Result(testData,predictions)
		for resultType in outputTypes:
			resultList.append(getattr(resultObject,resultType)())

		if(len(resultList) == 1):
			return resultList[0]
		else:
			return iter(resultList)



class XGBoostModel(BaseModel):
	m = None
	def train(self,trainData,param):		
		dtrain = xgb.DMatrix(trainData.features,label=trainData.labels)				
		bst = xgb.train(param, dtrain,num_boost_round=50)
		self.m = bst
	
	def predict(self,testData,outputTypes):
		inputData = xgb.DMatrix(testData.features)
		predictions = self.m.predict(inputData)
		return self.createResultObjects(testData,outputTypes,predictions)		

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
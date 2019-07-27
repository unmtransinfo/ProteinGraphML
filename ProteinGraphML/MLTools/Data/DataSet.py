
from sklearn.model_selection import train_test_split


class Data:    

	def loadFromNumpy(self,features,labels):
		self.features = features
		self.labels = labels
		self.posWeight = len([l for l in self.labels if l == 0.])/len([l for l in self.labels if l == 1.])
		print("SCALE POS",self.posWeight)

	def splitSet(self,splitSize):
		#,random_state=565
		#RS = 42
		TrainFeatures,TestFeatures,TrainLabels,TestLabels = train_test_split(self.features,self.labels,test_size=(1-splitSize))
		train = Data()
		train.loadFromNumpy(TrainFeatures,TrainLabels)
		test = Data()
		test.loadFromNumpy(TestFeatures,TestLabels)	
		return train,test


class BRData(Data): 
	features = []
	labels = []
	def loadData(self):
		self.data = load_breast_cancer()
		self.features = self.data['data']
		self.labels = self.data['target']

class BinaryLabel(Data):
	features = []
	labels = []
	labelColumn = None
	posWeight = None
	# can we clean this up
	def loadNoLabel(self,dataIN):
		self.data = dataIN
		self.labels = []
		self.features = self.data#.drop(['Y'],axis=1)
	
	def loadData(self,dataIN,labelColumn='Y'):
		self.data = dataIN
		self.labelColumn = labelColumn
		self.labels = self.data[self.labelColumn]
		self.features = self.data.drop([self.labelColumn],axis=1)
		self.posWeight = len([l for l in self.labels if l == 0.])/len([l for l in self.labels if l == 1.])
		
		#print("SCALE POS",self.posWeight)

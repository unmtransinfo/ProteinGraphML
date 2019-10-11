#!/usr/bin/env python3
###
import os,argparse,pyreadr,pickle,logging
import numpy as np
import pandas as pd
from ProteinGraphML.DataAdapter import OlegDB,selectAsDF

def generateTrainTestFromExcel(negProtein=None):
	'''
	This function reads the XLS file and generates training/test set
	using the given symbols and labels.
	'''
	flpath = args.dir + fileName
	df = pd.read_excel(flpath, sheet_name='Sheet1') #change 'Sheet1' to the name in your spreadsheet
	
	if (args.symbol_or_pid=='symbol'):
		symbols = df['Symbol'].values.tolist()
		symbolLabel = df.set_index('Symbol').T.to_dict('records')[0] #DataFrame to dictionary
		# Access the adapter to get protein_id for symbols
		symbolProteinId = dbAdapter.fetchProteinIdForSymbol(symbols)
		#Protein_Ids for training set
		for symbol, proteinId in symbolProteinId.items():
			trainProteinSet.add(int(proteinId))
			if (symbolLabel[symbol] == 1):	
				posLabelProteinIds.add(int(proteinId))
			elif (symbolLabel[symbol] == 0):	
				negLabelProteinIds.add(int(proteinId))
			else:
				logging.error('Invalid label')
	elif(args.symbol_or_pid=='pid'):
		proteinIdLabel = df.set_index('Protein_id').T.to_dict('records')[0] #DataFrame to dictionary
		#Protein_Ids for training set
		for proteinId, label in proteinIdLabel.items():
			trainProteinSet.add(int(proteinId))
			if (label == 1):	
				posLabelProteinIds.add(int(proteinId))
			elif (symbolLabel[symbol] == 0):	
				negLabelProteinIds.add(int(proteinId))
			else:
				logging.error('Invalid label')	
	else:
		logging.error('Invalid value for symbol_or_pid')
		exit()
	# if negative label was not provided, use default protein ids
	if (negProtein is not None):
		negLabelProteinIds.update(negProtein)
		trainProteinSet.update(negProtein)
	
	#determine train and test set
	testProteinSet = allProteinIds.difference(trainProteinSet)
	trainData[True] = posLabelProteinIds
	trainData[False] = negLabelProteinIds
	testData['unknown'] = testProteinSet
	logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
	if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
		logging.error ('ML codes cannot be run with one class')
		exit()
	else:
		return trainData, testData

def generateTrainTestFromText (negProtein=None):
	'''
	This function reads the text file and generates training/test set
	using the given symbols and labels.
	'''
	symbolLabel = {}
	symbols = []
	proteinIdLabel = {}
	flpath = args.dir + fileName

	if (args.symbol_or_pid=='symbol'):
		with open(flpath, 'r') as recs:
			for rec in recs:
				vals = rec.strip().split(',')
				symbolLabel[vals[0]] = vals[1]
				symbols.append(vals[0])
		# Access the adapter to get protein_id for symbols
		symbolProteinId = dbAdapter.fetchProteinIdForSymbol(symbols)
		for symbol, proteinId in symbolProteinId.items():
			trainProteinSet.add(int(proteinId))
			if (symbolLabel[symbol] == '1'):	
				posLabelProteinIds.add(int(proteinId))
			elif (symbolLabel[symbol] == '0'):
				negLabelProteinIds.add(int(proteinId))
			else:
				logging.info('Invalid label')
	elif(args.symbol_or_pid=='pid'):
		with open(flpath, 'r') as recs:
			for rec in recs:
				vals = rec.strip().split(',')
				proteinIdLabel[vals[0]] = vals[1]
						
		for proteinId, label in proteinIdLabel.items():
			trainProteinSet.add(int(proteinId))
			if (label == '1'):	
				posLabelProteinIds.add(int(proteinId))
			elif (symbolLabel[symbol] == '0'):
				negLabelProteinIds.add(int(proteinId))
			else:
				logging.info('Invalid label')
	else:
		logging.error('Invalid value for symbol_or_pid')
		exit()

	# if negative label was not provided, use default protein ids
	if (negProtein is not None):
		negLabelProteinIds.update(negProtein)
		trainProteinSet.update(negProtein)
	
	#determine train and test set
	testProteinSet = allProteinIds.difference(trainProteinSet) 
	trainData[True] = posLabelProteinIds
	trainData[False] = negLabelProteinIds
	testData['unknown'] = testProteinSet
	logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
	if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
		logging.error('ML codes cannot be run with one class')
		exit()	
	else:
		return trainData, testData

def generateTrainTestFromRDS(negProtein=None):
	'''
	This function reads the rds file and generates training/test set
	using the given symbols and labels.
	'''
	filenames = next(os.walk(path_to_rds_files))[2]
	flpath = path_to_rds_files + fileName
	if (fileName not in filenames):
		logging.error('RDS file not found!!!')
		exit()
	else:
		logging.info('Loading data from RDS file to create a dictionary')
		rdsdata = pyreadr.read_r(flpath)
		trainData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
		trainData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])

		# if negative label was not provided, use default protein ids
		if (negProtein is not None):
			trainData[False].update(negProtein)
		
		#determine train and test set			
		testProteinSet = allProteinIds.difference(trainData[True])
		testProteinSet = testProteinSet.difference(trainData[False])
		testData['unknown'] = testProteinSet
		logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
		if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
			logging.error('ML codes cannot be run with one class')
			exit()
		else:
			return trainData, testData

 
def saveTrainTestSet(trainData, testData):
	'''
	This function saves training and test in pickle format 
	'''
	pklTrainFile = args.dir + fileName.split('.')[0] + '.pkl'
	pklTestFile = args.dir + fileName.split('.')[0] + '_test.pkl'
	
	#Save the training set
	with open(pklTrainFile, 'wb') as handle:
		logging.info("Writing train data to file: {0}".format(pklTrainFile))
		pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#save the test set
	with open(pklTestFile, 'wb') as handle:
		logging.info("Writing test data to file: {0}".format(pklTestFile))
		pickle.dump(testData, handle, protocol=pickle.HIGHEST_PROTOCOL)	

###########START OF THE CODE###########################	

########################################################################
##IMPORTANT: change these values according to your local machine.
path_to_rds_files = '/home/oleg/workspace/metap/data/input/' 
path_to_files = os.getcwd() + '/DataForML/' 
########################################################################

parser = argparse.ArgumentParser(description='Generate dictionary file using proteinIds')
parser.add_argument('--file', required=True, type=str, nargs='?', help='input file')
parser.add_argument('--dir', default=path_to_files, help='input dir')
parser.add_argument('--symbol_or_pid', choices=('symbol', 'pid'), default='symbol', help='symbol|pid')
parser.add_argument('--use_default_negatives', default=False, action='store_true')
parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

args = parser.parse_args()

logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if args.verbose>1 else logging.INFO))

fileName = args.file

#Access the db adaptor 
dbAdapter = OlegDB()
allProteinIds = dbAdapter.fetchAllProteinIds()
allProteinIds = set(allProteinIds['protein_id'].tolist())

# check if negative labels need to be fetched from the database
if (args.use_default_negatives):
	logging.info('INFO: Default protein ids will be selected for negative labels')	
	negProteinIds = dbAdapter.fetchNegativeClassProteinIds()
	negProteinIds = set(negProteinIds['protein_id'].tolist())

### Generate a dictionary to store the protein_ids for class 0 and class 1.
### The dictionary will be saved in pickle format.

posLabelProteinIds = set()	#protein_ids for class 1
negLabelProteinIds = set()	#protein_ids for class 0
trainProteinSet = set() #protein_ids for training
testProteinSet = set() #protein_ids for test
trainData = {}	#dictionary to store training protein_ids
testData = {}	#dictionary to store test protein_ids

if (fileName is None):
	parser.error("--file required")

#input file contains symbols
elif (fileName is not None and args.symbol_or_pid=='symbol'): 
	logging.info('File name given and file contains symbols....>>>')
	if ('.xlsx' in fileName or '.xls' in fileName):
		if (args.use_default_negatives):
			trainData,testData = generateTrainTestFromExcel(negProtein=negProteinIds)
		else:
			trainData,testData = generateTrainTestFromExcel()
		
		#save the training/test data dictionary in pickle format
		saveTrainTestSet(trainData, testData)
	elif ('.txt' in fileName):
		if (args.use_default_negatives):
			trainData,testData = generateTrainTestFromText(negProtein=negProteinIds)
		else:
			trainData,testData = generateTrainTestFromText()
		
		#save the training/test data dictionary in pickle format
		saveTrainTestSet(trainData, testData)
	elif ('.rds' in fileName): #rds file
		if (args.use_default_negatives):
			trainData,testData = generateTrainTestFromRDS(negProtein=negProteinIds)
		else:
			trainData,testData = generateTrainTestFromRDS()
		
		#save the training/test data dictionary in pickle format
		saveTrainTestSet(trainData, testData)
	else:
		logging.error('File extension unknown.')
		exit()

#input file does not contain symbols
elif (fileName is not None and args.symbol_or_pid=='pid'): 
	logging.info('File name given and file has protein ids....>>>')
	if ('.xlsx' in fileName or '.xls' in fileName):
		if (args.use_default_negatives):
			trainData,testData = generateTrainTestFromExcel(negProtein=negProteinIds)
		else:
			trainData,testData = generateTrainTestFromExcel()
		
		#save the training/test data dictionary in pickle format
		saveTrainTestSet(trainData, testData)
	elif ('.txt' in fileName):
		if (args.use_default_negatives):
			trainData,testData = generateTrainTestFromText(negProtein=negProteinIds)
		else:
			trainData,testData = generateTrainTestFromText()
		
		#save the training/test data dictionary in pickle format
		saveTrainTestSet(trainData, testData)
	elif ('.rds' in fileName):
		if (args.use_default_negatives):
			trainData,testData = generateTrainTestFromRDS(negProtein=negProteinIds)
		else:
			trainData,testData = generateTrainTestFromRDS()
		
		#save the training/test data dictionary in pickle format
		saveTrainTestSet(trainData, testData)
	else:
		logging.error('File extension unknown.')
		exit()
else:
	logging.error('Wrong command-line arguments were passed')

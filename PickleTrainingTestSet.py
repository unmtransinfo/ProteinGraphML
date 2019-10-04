#!/usr/bin/env python3
###
import os,argparse,pyreadr,pickle,logging
import numpy as np
import pandas as pd
from ProteinGraphML.DataAdapter import OlegDB,selectAsDF

########################################################################
##IMPORTANT: change these values according to your local machine.
path_to_rds_files = '/home/oleg/workspace/metap/data/input/' 
path_to_files = os.getcwd() + '/DataForML/' 
########################################################################

parser = argparse.ArgumentParser(description='Generate dictionary file using proteinIds')
parser.add_argument('--file', required=True, type=str, nargs='?', help='input file')
parser.add_argument('--dir', default=path_to_files, help='input dir')
parser.add_argument('--symbol_or_pid', choices=('symbol', 'pid'), default='symbol', help='symbol|pid')
parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

args = parser.parse_args()

logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if args.verbose>1 else logging.INFO))

fileName = args.file

#Access the db adaptor 
dbAdapter = OlegDB()
allProteinIds = dbAdapter.fetchAllProteinIds()
allProteinIds = set(allProteinIds['protein_id'].tolist())

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
		flpath = args.dir + fileName
		pklTrainFile = args.dir + fileName.split('.')[0] + '.pkl'
		pklTestFile = args.dir + fileName.split('.')[0] + '_test.pkl'
		df = pd.read_excel(flpath, sheet_name='Sheet1') #change 'Sheet1' to the name in your spreadsheet
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
				logging.info('Invalid label')
		testProteinSet = allProteinIds.difference(trainProteinSet)
		trainData[True] = posLabelProteinIds
		trainData[False] = negLabelProteinIds
		logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
		if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
			logging.error ('ML codes cannot be run with one class')
			exit()
		
		#save the training data dictionary in pickle format
		with open(pklTrainFile, 'wb') as handle:
			logging.info("Writing train data to file: {0}".format(pklTrainFile))
			pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)

		#save the test data dictionary in pickle format
		with open(pklTestFile, 'wb') as handle:
			logging.info("Writing test data to file: {0}".format(pklTestFile))
			logging.info("Number of records in test set: {0}".format(len(testProteinSet)))
			l = int(len(testProteinSet)/2)
			testData[True] = set(list(testProteinSet)[:l])
			testData[False] = set(list(testProteinSet)[l:])
			pickle.dump(testData, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
	elif ('.txt' in fileName):
		symbolLabel = {}
		symbols = []
		
		flpath = args.dir + fileName
		pklTrainFile = args.dir + fileName.split('.')[0] + '.pkl'
		pklTestFile = args.dir + fileName.split('.')[0] + '_test.pkl'
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
		
		testProteinSet = allProteinIds.difference(trainProteinSet) 
		trainData[True] = posLabelProteinIds
		trainData[False] = negLabelProteinIds
		logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
		if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
			logging.error('ML codes cannot be run with one class')
			exit()
		
		#save the dictionary in pickle format
		with open(pklTrainFile, 'wb') as handle:
			logging.info("Writing file: {0}".format(pklTrainFile))
			pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		#save the test data dictionary in pickle format
		with open(pklTestFile, 'wb') as handle:
			logging.info("Writing test data to file: {0}".format(pklTestFile))
			logging.info("Number of records in test set: {0}".format(len(testProteinSet)))
			l = int(len(testProteinSet)/2)
			testData[True] = set(list(testProteinSet)[:l])
			testData[False] = set(list(testProteinSet)[l:])
			pickle.dump(testData, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
	elif ('.rds' in fileName): #rds file
		filenames = next(os.walk(path_to_rds_files))[2]
		flpath = path_to_rds_files + fileName
		pklTrainFile = args.dir + fileName.split('.')[0] + '.pkl'
		pklTestFile = args.dir + fileName.split('.')[0] + '_test.pkl'
		if (fileName not in filenames):
			logging.error('RDS file not found!!!')
			exit()
		else:
			logging.info('Loading data from RDS file to create a dictionary')
			rdsdata = pyreadr.read_r(flpath)
			trainData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
			trainData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])
			testProteinSet = allProteinIds.difference(trainData[True])
			testProteinSet = testProteinSet.difference(trainData[False])
			logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
			if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
				logging.error('ML codes cannot be run with one class')
				exit()

			#save the dictionary in pickle format
			with open(pklTrainFile, 'wb') as handle:
				logging.info("Writing file: {0}".format(pklTrainFile))
				pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)

			#save the test data dictionary in pickle format
			with open(pklTestFile, 'wb') as handle:
				logging.info("Writing test data to file: {0}".format(pklTestFile))
				logging.info("Number of records in test set: {0}".format(len(testProteinSet)))
				l = int(len(testProteinSet)/2)
				testData[True] = set(list(testProteinSet)[:l])
				testData[False] = set(list(testProteinSet)[l:])
				pickle.dump(testData, handle, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		logging.error('File extension unknown.')
		exit()

#input file does not contain symbols
elif (fileName is not None and args.symbol_or_pid=='pid'): 
	logging.info('File name given and file has protein ids....>>>')
	if ('.xlsx' in fileName or '.xls' in fileName):
		flpath = args.dir + fileName
		pklTrainFile = args.dir + fileName.split('.')[0] + '.pkl'
		pklTestFile = args.dir + fileName.split('.')[0] + '_test.pkl'
		df = pd.read_excel(flpath, sheet_name='Sheet1')	#change 'Sheet1' to the name in your spreadsheet
		proteinIdLabel = df.set_index('Protein_id').T.to_dict('records')[0] #DataFrame to dictionary
		
		for proteinId, label in proteinIdLabel.items():
			trainProteinSet.add(int(proteinId))
			if (label == 1):	
				posLabelProteinIds.add(int(proteinId))
			elif (symbolLabel[symbol] == 0):	
				negLabelProteinIds.add(int(proteinId))
			else:
				logging.info('Invalid label')
		
		testProteinSet = allProteinIds.difference(trainProteinSet)
		trainData[True] = posLabelProteinIds
		trainData[False] = negLabelProteinIds
		logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
		if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
			logging.error('ML codes cannot be run with one class')
			exit()
		
		#save the dictionary in pickle format
		with open(pklTrainFile, 'wb') as handle:
			logging.info("Writing file: {0}".format(pklTrainFile))
			pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)

		#save the test data dictionary in pickle format
		with open(pklTestFile, 'wb') as handle:
			logging.info("Writing test data to file: {0}".format(pklTestFile))
			logging.info("Number of records in test set: {0}".format(len(testProteinSet)))
			l = int(len(testProteinSet)/2)
			testData[True] = set(list(testProteinSet)[:l])
			testData[False] = set(list(testProteinSet)[l:])
			pickle.dump(testData, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
	elif ('.txt' in fileName):
		proteinIdLabel = {}
		flpath = args.dir + fileName
		pklTrainFile = args.dir + fileName.split('.')[0] + '.pkl'
		pklTestFile = args.dir + fileName.split('.')[0] + '_test.pkl'
		
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

		testProteinSet = allProteinIds.difference(trainProteinSet)
		trainData[True] = posLabelProteinIds
		trainData[False] = negLabelProteinIds
		logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
		if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
			logging.error('ML codes cannot be run with one class')
			exit()
		
		#save the dictionary in pickle format
		with open(pklTrainFile, 'wb') as handle:
			logging.info("Writing file: {0}".format(pklTrainFile))
			pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)

		#save the test data dictionary in pickle format
		with open(pklTestFile, 'wb') as handle:
			logging.info("Writing test data to file: {0}".format(pklTestFile))
			logging.info("Number of records in test set: {0}".format(len(testProteinSet)))
			l = int(len(testProteinSet)/2)
			testData[True] = set(list(testProteinSet)[:l])
			testData[False] = set(list(testProteinSet)[l:])
			pickle.dump(testData, handle, protocol=pickle.HIGHEST_PROTOCOL)

	elif ('.rds' in fileName):
		filenames = next(os.walk(path_to_rds_files))[2]
		flpath = path_to_rds_files + fileName
		pklTrainFile = args.dir + fileName.split('.')[0] + '.pkl'
		pklTestFile = args.dir + fileName.split('.')[0] + '_test.pkl'
		if (fileName not in filenames):
			logging.error('RDS file not found!!! ')
			exit()
		else:
			logging.info('Loading data from RDS file to create a dictionary')
			rdsdata = pyreadr.read_r(flpath)
			trainData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
			trainData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])
			testProteinSet = allProteinIds.difference(trainData[True])
			testProteinSet = testProteinSet.difference(trainData[False])
			
			logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
			if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
				logging.error('ML codes cannot be run with one class')
				exit()
				
			#save the dictionary in pickle format
			with open(pklTrainFile, 'wb') as handle:
				logging.info("Writing file: {0}".format(pklTrainFile))
				pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)				

			#save the test data dictionary in pickle format
			with open(pklTestFile, 'wb') as handle:
				logging.info("Writing test data to file: {0}".format(pklTestFile))
				logging.info("Number of records in test set: {0}".format(len(testProteinSet)))
				l = int(len(testProteinSet)/2)
				testData[True] = set(list(testProteinSet)[:l])
				testData[False] = set(list(testProteinSet)[l:])
				pickle.dump(testData, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
	else:
		logging.error('File extension unknown.')
		exit()

#input file is an RDS file	
elif (fileName is not None and '.rds' in fileName): 
	logging.info('Only file name provided....>>>')
	filenames = next(os.walk(path_to_rds_files))[2]
	flpath = path_to_rds_files + fileName
	pklTrainFile = args.dir + fileName.split('.')[0] + '.pkl'
	pklTestFile = args.dir + fileName.split('.')[0] + '_test.pkl'
	if (fileName not in filenames):
		logging.error('RDS file not found!!! ')
		exit()
	else:
		logging.info('Loading data from RDS file to create a dictionary')
		rdsdata = pyreadr.read_r(flpath)
		trainData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
		trainData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])
		testProteinSet = allProteinIds.difference(trainData[True])
		testProteinSet = testProteinSet.difference(trainData[False])

		logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
		if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
			logging.error('ML codes cannot be run with one class')
			exit()
		
		#save the dictionary in pickle format
		with open(pklTrainFile, 'wb') as handle:
			logging.info("Writing file: {0}".format(pklTrainFile))
			pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)

		#save the test data dictionary in pickle format
		with open(pklTestFile, 'wb') as handle:
			logging.info("Writing test data to file: {0}".format(pklTestFile))
			logging.info("Number of records in test set: {0}".format(len(testProteinSet)))
			l = int(len(testProteinSet)/2)
			testData[True] = set(list(testProteinSet)[:l])
			testData[False] = set(list(testProteinSet)[l:])
			pickle.dump(testData, handle, protocol=pickle.HIGHEST_PROTOCOL)				
else:
	logging.error('Wrong command-line arguments were passed')

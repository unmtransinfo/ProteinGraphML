#!/usr/bin/env python3
###
import os,argparse,pyreadr,pickle,logging
import numpy as np
import pandas as pd
from ProteinGraphML.DataAdapter import OlegDB,selectAsDF

def generateTrainPredictFromExcel(inFile, idType, negProtein=None):
	'''
	This function reads the XLS file and generates training/predict set
	using the given symbols and labels.
	'''
	df = pd.read_excel(inFile, sheet_name='Sheet1') #change 'Sheet1' to the name in your spreadsheet
	
	if (idType=='symbol'):
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
	elif(idType=='pid'):
		proteinIdLabel = df.set_index('Protein_id').T.to_dict('records')[0] #DataFrame to dictionary
		#Protein_Ids for training set
		for proteinId, label in proteinIdLabel.items():
			trainProteinSet.add(int(proteinId))
			if (label == 1):	
				posLabelProteinIds.add(int(proteinId))
			elif (label == 0):	
				negLabelProteinIds.add(int(proteinId))
			else:
				logging.error('Invalid label')	
	else:
		logging.error('Invalid idType: {0}'.format(idType))
		exit()
	# if negative label was not provided, use default protein ids
	if (negProtein is not None):
		negLabelProteinIds.update(negProtein)
		trainProteinSet.update(negProtein)
	
	#determine train and predict set
	predictProteinSet = allProteinIds.difference(trainProteinSet)
	trainData[True] = posLabelProteinIds
	trainData[False] = negLabelProteinIds
	predictData['unknown'] = predictProteinSet
	logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
	logging.info('Count of predict set (unlabeled): {0}'. format(len(predictData['unknown'])))
	if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
		logging.error ('ML codes cannot be run with one class')
		exit()
	else:
		return trainData, predictData

def generateTrainPredictFromText (inFile, idType, negProtein=None):
	'''
	This function reads the text file and generates training/predict set
	using the given symbols and labels.
	'''
	symbolLabel = {}
	symbols = []
	proteinIdLabel = {}

	if (idType=='symbol'):
		with open(inFile, 'r') as recs:
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
	elif(idType=='pid'):
		with open(inFile, 'r') as recs:
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
		logging.error('Invalid idType: {0}'.format(idType))
		exit()

	# if negative label was not provided, use default protein ids
	if (negProtein is not None):
		negLabelProteinIds.update(negProtein)
		trainProteinSet.update(negProtein)
	
	#determine train and predict set
	predictProteinSet = allProteinIds.difference(trainProteinSet) 
	trainData[True] = posLabelProteinIds
	trainData[False] = negLabelProteinIds
	predictData['unknown'] = predictProteinSet
	logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
	if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
		logging.error('ML codes cannot be run with one class')
		exit()	
	else:
		return trainData, predictData

def generateTrainPredictFromRDS(inFile, negProtein=None):
	'''
	This function reads the rds file and generates training/predict set
	using the given symbols and labels.
	'''
	logging.info('Loading data from RDS file to create a dictionary')
	rdsdata = pyreadr.read_r(inFile)
	df = rdsdata[None]
	trainData[True] = set(df.loc[(df['Y']=='pos') & (df['subset'] == 'train')]["id1"])
	trainData[False] = set(df.loc[(df['Y']=='neg') & (df['subset'] == 'train')]["id1"])
	#trainData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
	#trainData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])

	# if negative label was not provided, use default protein ids
	if (negProtein is not None):
		trainData[False].update(negProtein)
		
	#determine train and predict set			
	predictProteinSet = allProteinIds.difference(trainData[True])
	predictProteinSet = predictProteinSet.difference(trainData[False])
	predictData['unknown'] = predictProteinSet
	logging.info('Count of positive labels: {0}, count of negative labels: {1}'. format(len(trainData[True]), len(trainData[False])))
	if (len(trainData[True]) == 0 or len(trainData[False]) == 0):
		logging.error('ML codes cannot be run with one class')
		exit()
	else:
		return trainData, predictData

 
def saveTrainPredictSet(trainData, predictData, outDir, outBaseName):
	'''
	This function saves training and predict in pickle format.
	'''
	pklTrainFile = outDir + '/' + outBaseName + '.pkl'
	pklPredictFile = outDir + '/' + outBaseName + '_predict.pkl'

	#Save the training set
	with open(pklTrainFile, 'wb') as handle:
		logging.info("Training dataset: {0} positive, {1} negative, {2} total".format(len(trainData[True]), len(trainData[False]), len(trainData[True])+len(trainData[False])))
		logging.info("Writing train data to file: {0}".format(pklTrainFile))
		pickle.dump(trainData, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#save the predict set
	with open(pklPredictFile, 'wb') as handle:
		logging.info("Predict dataset: {0} cases".format(len(predictData['unknown'])))
		logging.info("Writing predict data to file: {0}".format(pklPredictFile))
		pickle.dump(predictData, handle, protocol=pickle.HIGHEST_PROTOCOL)	

###########START OF MAIN PROGRAM###########################	
if __name__ == '__main__':
	#dataDirRDS = '/home/oleg/workspace/metap/data/input/' 
	parser = argparse.ArgumentParser(description='Generate training and predict set definition files.', epilog='Output files to same dir as input file, pickled dictionary files suffixed ".pkl" and "_predict.pkl".')
	parser.add_argument('--i', dest='ifile', metavar='INPUT_FILE', required=True, help='input file, with protein IDs or symbols, positive and optionally negative labels (CSV|XLSX)')
	parser.add_argument('--symbol_or_pid', choices=('symbol', 'pid'), default='symbol', help='symbol|pid')
	parser.add_argument('--use_default_negatives', default=False, action='store_true', help='required if negatives not specified by input')
	parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

	args = parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if args.verbose>1 else logging.INFO))

	fileName = os.path.basename(args.ifile)
	fileExt = fileName.split('.')[-1]
	dataDir = os.path.dirname(os.path.abspath(args.ifile))
	outBaseName = fileName.split('.')[0]

	if fileExt.lower() not in ('csv', 'tsv', 'txt', 'rds', 'xlsx', 'xls'):
		parser.error('Unsupported filetype: {0} ({1})'.format(fileName, fileExt))

	#Access the db adaptor 
	dbAdapter = OlegDB()
	allProteinIds = dbAdapter.fetchAllProteinIds()
	allProteinIds = set(allProteinIds['protein_id'].tolist())

	# check if negative labels need to be fetched from the database
	if (args.use_default_negatives):
		logging.info('INFO: Default negative protein ids will be selected for negative labels.')	
		negProteinIds = dbAdapter.fetchNegativeClassProteinIds()
		negProteinIds = set(negProteinIds['protein_id'].tolist())

	### Generate a dictionary to store the protein_ids for class 0 and class 1.
	### The dictionary will be saved in pickle format.

	posLabelProteinIds = set()	#protein_ids for class 1
	negLabelProteinIds = set()	#protein_ids for class 0
	trainProteinSet = set() #protein_ids for training
	predictProteinSet = set() #protein_ids for predict
	trainData = {}	#dictionary to store training protein_ids
	predictData = {}	#dictionary to store predict protein_ids
	
	if fileExt.lower() == 'rds':
		logging.info('Input file specified: {0}'.format(fileName))
		trainData,predictData = generateTrainPredictFromRDS(args.ifile, negProtein=(negProteinIds if args.use_default_negatives else None))
	elif fileExt.lower() in ('xlsx', 'xls'):
		logging.info('Input file with ID type "{0}" specified: {1}'.format(args.symbol_or_pid, fileName))
		trainData,predictData = generateTrainPredictFromExcel(args.ifile, args.symbol_or_pid, negProtein=(negProteinIds if args.use_default_negatives else None))
	elif fileExt.lower() in ('csv', 'tsv', 'txt'):
		logging.info('Input file with ID type "{0}" specified: {1}'.format(args.symbol_or_pid, fileName))
		trainData,predictData = generateTrainPredictFromText(args.ifile, args.symbol_or_pid, negProtein=(negProteinIds if args.use_default_negatives else None))
	else:
		pass #ERROR
	logging.info('Writing output to: {0}'.format(dataDir))
	saveTrainPredictSet(trainData, predictData, dataDir, outBaseName)

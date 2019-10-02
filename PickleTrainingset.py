#!/usr/bin/env python3
###
import pickle
import argparse
import pyreadr
import pandas as pd
from ProteinGraphML.DataAdapter import OlegDB,selectAsDF
import os
import numpy as np

########################################################################
##IMPORTANT: change these values according to your local machine.
PROTEIN_COUNT = 20237 #Change it if needed
path_to_rds_files = '/home/oleg/workspace/metap/data/input/' 
path_to_files = os.getcwd() + '/DataForML/' 
########################################################################

parser = argparse.ArgumentParser(description='Generate dictionary file using proteinIds')
parser.add_argument('--file', required=True, type=str, nargs='?', help='input file')
parser.add_argument('--dir', default=path_to_files, help='input dir')
parser.add_argument('--symbol_or_pid', choices=('symbol', 'pid'), default='symbol', help='symbol|pid')
argData = vars(parser.parse_args())
symbolPresent = argData['symbol_or_pid']=='symbol'
fileName = argData['file']

#Access the adaptor
dbAdapter = OlegDB()

### Generate a dictionary to store the protein_ids for class 0 and class 1.
### The dictionary will be saved in pickle format.

posLabelProteinIds = set()	#protein_ids for class 1
fileData = {}	#dictionary to store protein_ids
proteinIds = set([i for i in range(PROTEIN_COUNT)])

if (fileName is None):
	print ("Please provide the input filename")
	exit()

elif (fileName is not None and argData['symbol_or_pid']=='symbol'): #input file contains symbols
	print ('File name given and file has symbols....>>>')
	if ('.xlsx' in fileName or '.xls' in fileName):
		flpath = argData['dir'] + fileName
		pklFile = argData['dir'] + fileName.split('.')[0] + '.pkl'
		df = pd.read_excel(flpath, sheet_name='Sheet1') #change 'Sheet1' to the name in your spreadsheet
		symbols = df['Symbol'].values.tolist()
		symbolLabel = df.set_index('Symbol').T.to_dict('records')[0] #DataFrame to dictionary
		
		# Access the adapter to get protein_id for symbols
		symbolProteinId = dbAdapter.fetchProteinIdForSymbol(symbols)
		for symbol, proteinId in symbolProteinId.items():
			if (symbolLabel[symbol] == 1):	
				posLabelProteinIds.add(int(proteinId))
		negLabelProteinIds = proteinIds.difference(posLabelProteinIds)
		fileData[True] = posLabelProteinIds
		fileData[False] = negLabelProteinIds
		print ('Count of positive labels: {0}, count of negative labels: {1}'. format(len(fileData[True]), len(fileData[False])))
		
		#save the dictionary in pickle format
		with open(pklFile, 'wb') as handle:
			print("Writing file: {0}".format(pklFile))
			pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
	elif ('.txt' in fileName):
		symbolLabel = {}
		symbols = []
		
		flpath = argData['dir'] + fileName
		pklFile = argData['dir'] + fileName.split('.')[0] + '.pkl'
		with open(flpath, 'r') as recs:
			for rec in recs:
				vals = rec.strip().split(',')
				symbolLabel[vals[0]] = vals[1]
				symbols.append(vals[0])
		
		# Access the adapter to get protein_id for symbols
		symbolProteinId = dbAdapter.fetchProteinIdForSymbol(symbols)
		for symbol, proteinId in symbolProteinId.items():
			if (symbolLabel[symbol] == '1'):	
				posLabelProteinIds.add(int(proteinId))
		negLabelProteinIds = proteinIds.difference(posLabelProteinIds)
		fileData[True] = posLabelProteinIds
		fileData[False] = negLabelProteinIds
		print ('Count of positive labels: {0}, count of negative labels: {1}'. format(len(fileData[True]), len(fileData[False])))
		
		#save the dictionary in pickle format
		with open(pklFile, 'wb') as handle:
			print("Writing file: {0}".format(pklFile))
			pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)

	else: #rds file
		filenames = next(os.walk(path_to_rds_files))[2]
		flname = fileName + '.rds'
		pklFile = argData['dir'] + fileName + '.pkl'
		if (flname not in filenames):
			print ('RDS file not found!!!')
			exit()
		else:
			print ('Loading data from RDS file to craete a dictionary')
			rdsdata = pyreadr.read_r(path_to_rds_files+flname)
			fileData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
			fileData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])
			print ('Count of positive labels: {0}, count of negative labels: {1}'. format(len(fileData[True]), len(fileData[False])))

			#save the dictionary in pickle format
			with open(pklFile, 'wb') as handle:
				print("Writing file: {0}".format(pklFile))
				pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif (fileName is not None and argData['symbol_or_pid']=='pid'): #input file does not contain symbols
	print ('File name provided and file has protein ids....>>>')
	if ('.xlsx' in fileName or '.xls' in fileName):
		flpath = argData['dir'] + fileName
		pklFile = argData['dir'] + fileName.split('.')[0] + '.pkl'
		df = pd.read_excel(flpath, sheet_name='Sheet1')	#change 'Sheet1' to the name in your spreadsheet
		proteinIdLabel = df.set_index('Protein_id').T.to_dict('records')[0] #DataFrame to dictionary
		
		for proteinId, label in proteinIdLabel.items():
			if (label == 1):	
				posLabelProteinIds.add(int(proteinId))
		negLabelProteinIds = proteinIds.difference(posLabelProteinIds)
		#print (posLabelProteinIds) 
		fileData[True] = posLabelProteinIds
		fileData[False] = negLabelProteinIds
		print ('Count of positive labels: {0}, count of negative labels: {1}'. format(len(fileData[True]), len(fileData[False])))
		
		#save the dictionary in pickle format
		with open(pklFile, 'wb') as handle:
			print("Writing file: {0}".format(pklFile))
			pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
	elif ('.txt' in fileName):
		proteinIdLabel = {}
		flpath = argData['dir'] + fileName
		pklFile = argData['dir'] + fileName.split('.')[0] + '.pkl'
		
		with open(flpath, 'r') as recs:
			for rec in recs:
				vals = rec.strip().split(',')
				proteinIdLabel[vals[0]] = vals[1]
						
		for proteinId, label in proteinIdLabel.items():
			if (label == '1'):	
				posLabelProteinIds.add(int(proteinId))
		negLabelProteinIds = proteinIds.difference(posLabelProteinIds)
		#print (posLabelProteinIds) 
		fileData[True] = posLabelProteinIds
		fileData[False] = negLabelProteinIds
		print ('Count of positive labels: {0}, count of negative labels: {1}'. format(len(fileData[True]), len(fileData[False])))
		
		#save the dictionary in pickle format
		with open(pklFile, 'wb') as handle:
			print("Writing file: {0}".format(pklFile))
			pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)

	elif ('.rds' in fileName):
		filenames = next(os.walk(path_to_rds_files))[2]
		pklFile = argData['dir'] + fileName + '.pkl'
		if (fileName not in filenames):
			print ('RDS file not found!!! ')
			exit()
		else:
			print ('Loading data from RDS file to craete a dictionary')
			rdsdata = pyreadr.read_r(path_to_rds_files+fileName)
			fileData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
			fileData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])
			print ('Count of positive labels: {0}, count of negative labels: {1}'. format(len(fileData[True]), len(fileData[False])))
				
			#save the dictionary in pickle format
			with open(pklFile, 'wb') as handle:
				print("Writing file: {0}".format(pklFile))
				pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)				
	else:
		print ('File extension unknown.')
		exit()
	
elif (fileName is not None and '.rds' in fileName): #input file is an RDS file
	
	print ('Only file name provided....>>>')
	filenames = next(os.walk(path_to_rds_files))[2]
	pklFile = argData['dir'] + fileName + '.pkl'
	if (fileName not in filenames):
		print ('RDS file not found!!! ')
		exit()
	else:
		print ('Loading data from RDS file to craete a dictionary')
		rdsdata = pyreadr.read_r(path_to_rds_files+fileName)
		fileData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
		fileData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])
		print ('Count of positive labels: {0}, count of negative labels: {1}'. format(len(fileData[True]), len(fileData[False])))
		
		#save the dictionary in pickle format
		with open(pklFile, 'wb') as handle:
			print("Writing file: {0}".format(pklFile))
			pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)
				
else:
	print ('Wrong command-line arguments were passed')

import pickle
import argparse
import pyreadr
import pandas as pd
from ProteinGraphML.DataAdapter import OlegDB,selectAsDF
import os
import numpy as np

PROTEIN_COUNT = 20237 #Change it if needed
path_to_rds_files = '/home/oleg/workspace/metap/data/input/' #IMPORTANT: change it if you have saved rds files in a different folder
path_to_files = '/home/pkumar/ITDC/ProteinGraphML/DataForML/' 

parser = argparse.ArgumentParser(description='Dictionary file using proteinIds')
parser.add_argument('--file', required=True, type=str, nargs='?', help='input file')
parser.add_argument('--symbol', type=str, nargs='?', help='does file have symbol (Y/N)')
argData = vars(parser.parse_args())
symbolPresent = argData['symbol']
fileName = argData['file']

dbAdapter = OlegDB()


if (fileName is None):
	print ("Please provide filename")
	exit()

elif (fileName is not None and symbolPresent == 'Y'):
	print ('File name given and file has symbols....>>>')
	if ('.xlsx' in fileName or '.xls' in fileName):
		posLabelProteinIds = set()
		fileData = {}
		proteinIds = set([i for i in range(PROTEIN_COUNT)])
		flpath = path_to_files + fileName
		pklFile = path_to_files + fileName.split('.')[0] + '.pkl'
		df = pd.read_excel(flpath, sheetname='Sheet1')
		symbols = df['Symbol'].values.tolist()
		symbolLabel = df.set_index('Symbol').T.to_dict('records')[0] #DataFrame to dictionary
		
		symbolProteinId = dbAdapter.fetchProteinIdForSymbol(symbols)
		for symbol, proteinId in symbolProteinId.items():
			if (symbolLabel[symbol] == 1):	
				posLabelProteinIds.add(int(proteinId))
		negLabelProteinIds = proteinIds.difference(posLabelProteinIds)
		#print (posLabelProteinIds) 
		fileData[True] = posLabelProteinIds
		fileData[False] = negLabelProteinIds
		
		with open(pklFile, 'wb') as handle:
			pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
	elif ('.txt' in fileName or '.csv' in fileName):
		posLabelProteinIds = set()
		fileData = {}
		symbolLabel = {}
		symbols = []
		proteinIds = set([i for i in range(PROTEIN_COUNT)])
		flpath = path_to_files + fileName
		pklFile = path_to_files + fileName.split('.')[0] + '.pkl'
		
		with open(flpath, 'r') as recs:
			for rec in recs:
				vals = rec.strip().split(',')
				symbolLabel[vals[0]] = vals[1]
				symbols.appen(vals[0])
		
		symbolProteinId = dbAdapter.fetchProteinIdForSymbol(symbols)
		for symbol, proteinId in symbolProteinId.items():
			if (symbolLabel[symbol] == 1):	
				posLabelProteinIds.add(int(proteinId))
		negLabelProteinIds = proteinIds.difference(posLabelProteinIds)
		print (posLabelProteinIds) 
		fileData[True] = posLabelProteinIds
		fileData[False] = negLabelProteinIds
		with open(pklFile, 'wb') as handle:
			pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

	else:
		filenames = next(os.walk(path_to_rds_files))[2]
		flname = fileName + '.rds'
		pklFile = path_to_files + fileName + '.pkl'
		if (flname not in filenames):
			print ('RDS file not found!!! Set the variable path_to_rds_files correctly')
		else:
			print ('Loading data from RDS file to craete a dictionary')
			rdsdata = pyreadr.read_r(path_to_rds_files+flname)
			fileData = {}
			fileData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
			fileData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])
			with open(pklFile, 'wb') as handle:
				pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif (fileName is not None and symbolPresent == 'N'):
	print ('File name provided and file has protein ids....>>>')
	if ('.xlsx' in fileName or '.xls' in fileName):
		posLabelProteinIds = set()
		fileData = {}
		proteinIds = set([i for i in range(PROTEIN_COUNT)])
		flpath = path_to_files + fileName
		pklFile = path_to_files + fileName.split('.')[0] + '.pkl'
		df = pd.read_excel(flpath, sheetname='Sheet1')
		proteinIdLabel = df.set_index('Protein_id').T.to_dict('records')[0] #DataFrame to dictionary
		for proteinId, label in proteinIdLabel.items():
			if (label == 1):	
				posLabelProteinIds.add(int(proteinId))
		negLabelProteinIds = proteinIds.difference(posLabelProteinIds)
		#print (posLabelProteinIds) 
		fileData[True] = posLabelProteinIds
		fileData[False] = negLabelProteinIds
		
		with open(pklFile, 'wb') as handle:
			pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
	elif ('.txt' in fileName or '.csv' in fileName):
		posLabelProteinIds = set()
		fileData = {}
		proteinIdLabel = {}
		proteinIds = set([i for i in range(PROTEIN_COUNT)])
		flpath = path_to_files + fileName
		pklFile = path_to_files + fileName.split('.')[0] + '.pkl'
		
		with open(flpath, 'r') as recs:
			for rec in recs:
				vals = rec.strip().split(',')
				proteinIdLabel[vals[0]] = vals[1]
						
		for proteinId, label in proteinIdLabel.items():
			if (label == 1):	
				posLabelProteinIds.add(int(proteinId))
		negLabelProteinIds = proteinIds.difference(posLabelProteinIds)
		print (posLabelProteinIds) 
		fileData[True] = posLabelProteinIds
		fileData[False] = negLabelProteinIds
		with open(pklFile, 'wb') as handle:
			pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

	else:
		filenames = next(os.walk(path_to_rds_files))[2]
		flname = fileName + '.rds'
		pklFile = path_to_files + fileName + '.pkl'
		if (flname not in filenames):
			print ('RDS file not found!!! Set the variable path_to_rds_files correctly')
		else:
			print ('Loading data from RDS file to craete a dictionary')
			rdsdata = pyreadr.read_r(path_to_rds_files+flname)
			fileData = {}
			fileData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
			fileData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])
			with open(pklFile, 'wb') as handle:
				pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
elif (fileName is not None):
	print ('Onlye file name provided....>>>')
	filenames = next(os.walk(path_to_rds_files))[2]
	flname = fileName + '.rds'
	pklFile = path_to_files + fileName + '.pkl'
	if (flname not in filenames):
		print ('RDS file not found!!! Set the variable path_to_rds_files correctly')
	else:
		print ('Loading data from RDS file to craete a dictionary')
		rdsdata = pyreadr.read_r(path_to_rds_files+flname)
		fileData = {}
		fileData[True] = set(np.where(rdsdata[None]['Y']=='pos')[0])
		fileData[False] = set(np.where(rdsdata[None]['Y']=='neg')[0])
		with open(pklFile, 'wb') as handle:
			pickle.dump(fileData, handle, protocol=pickle.HIGHEST_PROTOCOL)
	


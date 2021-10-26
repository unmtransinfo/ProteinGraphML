#!/usr/bin/env python3
###
# SAVED DATA 
# WHERE TO SAVE 
# TYPE, FIG OR GRAPH 
import logging
from proteingraphml.Analysis import Visualize

from proteingraphml.DataAdapter import OlegDB,selectAsDF,TCRD
from proteingraphml.GraphTools import ProteinDiseaseAssociationGraph

import pickle
import argparse
## we construct a base map of protein to disease just by creating the ProteinDiseaseAs

''''
def featureVisualize(features,AUC,TITLE,count=20): 
    
	plt.rcParams.update({'font.size': 15,'lines.linewidth': 1000}) #   axes.labelweight': 'bold'})
	FILETITLE = TITLE
	TITLE = TITLE + "-AUC: "+str(AUC)  

	df = pd.DataFrame(features.most_common(count), columns=['feature', 'gain'])
	plt.figure()
	df['gain'] = (df['gain']/sum(df['gain'][:count]))
	#df['feature'] = df['feature'].map(processFeature)
	r = df.head(count).plot( kind='barh',title=TITLE, x='feature', y='gain',color='tomato', legend=False, figsize=(10, 12))
	r.set_xlabel('Importance')
	r.set_ylabel('Features')
	r.invert_yaxis()

	r.figure.savefig(FILETITLE+'.png',bbox_inches='tight')
'''

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

#Get the name of the disease
DEFAULT_GRAPH = "ProteinDisease_GRAPH.pkl"
DBS=['olegdb', 'tcrd']
parser = argparse.ArgumentParser(description='Run ML Procedure')
parser.add_argument('--disease', metavar='disease', required=True, type=str, nargs='?', help='pickled file with ML features')
#parser.add_argument('--dir', default=dataDir, help='input dir')
parser.add_argument('--db', choices=DBS, default="olegdb", help='{0}'.format(str(DBS)))
parser.add_argument('--featurefile', required=True, help='full path to the pickle feature file')
parser.add_argument('--num', metavar='featureCount', required=True, type=int, nargs='?',help='Number of top features')
parser.add_argument('--kgfile', default=DEFAULT_GRAPH, help='input pickled KG')

logging.info('Generate HTML files for visualization...!!!')

argData = vars(parser.parse_args())
fileName = argData['featurefile']
numOfFeatures = argData['num']
diseaseName = argData['disease']

logging.info('Running visualization using file...{0}'.format(fileName))
#filePath = argData['dir'] + fileName #IMPORTANT: update this if folder name changes
tmpPath = fileName.split('/')[:-1]
filePath = '/'.join(i for i in tmpPath)

#fetch the saved important features
importance = load_obj(fileName)
#importance = Counter({'R-HSA-927802': 0.31735258141642814, 'hsa04740': 0.2208299216149202, 'hsa05100': 0.1847905733996812, 'hsa04930': 0.10625980494746863, 'hsa04514': 0.047493659101048136, 'hsa04114': 0.03542724660274679, 'hsa04810': 0.03365848585388666, 'hsa04144': 0.030556051003490892})


#access the database to get the description of important features
#dbAdapter = OlegDB()
dbAdapter = TCRD() if argData['db'] == "tcrd" else OlegDB()
#labelMap = convertLabels(importance.keys(),dbAdapter,selectAsDF,type='plot')

if True:
	currentGraph = ProteinDiseaseAssociationGraph.load(argData['kgfile'])

	# for the graph, we need the original importance 
	for imp in importance.most_common(numOfFeatures):
		print(imp)
		Visualize(imp, currentGraph.graph, diseaseName, filePath, dbAdapter=dbAdapter) #g,currentGraph.graph,Disease)
		#break

#newSet = {}
#for key in importance.keys():
#	newSet[labelMap[key]] = importance[key]

#print('STARTING FEAT VIS')
#AUC = 0.9
#print(newSet,labelMap)
#featureVisualize(Counter(newSet),AUC,"AAA")
#Visualize
#convertLabels([343,30001],dbAdapter,selectAsDF,type="han")

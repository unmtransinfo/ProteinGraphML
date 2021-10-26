#!/usr/bin/env python3
###
import sys,os,time,argparse,logging
import pickle


from proteingraphml.DataAdapter import OlegDB,selectAsDF
from proteingraphml.GraphTools import ProteinDiseaseAssociationGraph
from proteingraphml.MLTools.MetapathFeatures import metapathFeatures,ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode,getMetapaths 
from proteingraphml.MLTools.Data import BinaryLabel
from proteingraphml.MLTools.Procedures import *

t0 = time.time()

DATA_DIR = os.getcwd() + '/DataForML/'
NUM_OF_FOLDS = 2
DEFAULT_GRAPH = "ProteinDisease_GRAPH.pkl"
DEFAULT_STATIC_FEATURES = "gtex,lincs,ccle,hpa"

PROCEDURES = ["XGBCrossVal", "XGBCrossValPred"]

parser = argparse.ArgumentParser(description='Run ML Procedure', epilog='--disease or --file must be specified; available procedures: {0}'.format(str(PROCEDURES)))
parser.add_argument('procedure', metavar='procedure', type=str, choices=PROCEDURES, nargs='+', help='ML procedure to run')
parser.add_argument('--disease', metavar='disease', type=str, nargs='?', help='Mammalian Phenotype ID, e.g. MP_0000180')
parser.add_argument('--file', type=str, nargs='?', help='input file, pickled training set, e.g. "diabetes.pkl"')
parser.add_argument('--dir', default=DATA_DIR, help='input dir (default: "{0}")'.format(DATA_DIR))
parser.add_argument('--resultdir', type=str, nargs='?', help='folder where results will be saved, e.g. "diabetes_no_lincs"')
parser.add_argument('--crossval_folds', type=int, default=NUM_OF_FOLDS, help='number of folds for average CV (default: "{0}")'.format(NUM_OF_FOLDS))
parser.add_argument('--kgfile', default=DEFAULT_GRAPH, help='input pickled KG (default: "{0}")'.format(DEFAULT_GRAPH))
parser.add_argument('--static_data', default=DEFAULT_STATIC_FEATURES, help='(default: "{0}")'.format(DEFAULT_STATIC_FEATURES))
parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity")

argData = vars(parser.parse_args())

logging.basicConfig(format='%(levelname)s:%(message)s', level=(logging.DEBUG if argData['verbose']>1 else logging.INFO))

#Get data from file or disease
disease = argData['disease']
fileName = argData['file']
fileData = None

if disease is None and fileName is None: # NO INPUT
	parser.error("--disease or --file must be specified.")
elif disease is None and fileName is not None: # NO disease, use file
	pklFile = argData['dir'] + fileName
	diseaseName = fileName.split('.')[0]
	try:
		with open(pklFile, 'rb') as f:
			fileData = pickle.load(f)
	except:
		logging.error('Must generate pickled training set file for the given disease') 
		exit()
    		
	#def load_obj(name):
	#with open(pklFile, 'rb') as f:
	#	fileData = pickle.load(f)
	#loadList = load_obj('nextDataset')
elif fileName is None and disease is not None:
	logging.info("running on this disease: {0}".format(disease))
	diseaseName = disease
else:
	logging.error('Wrong parameters passed')


# CANT FIND THIS DISEASE
#disease = sys.argv[1]
Procedure = argData['procedure'][0]
logging.info('Procedure: {0}'.format(Procedure))

graphString = argData['kgfile']

# CANT FIND THIS GRAPH
currentGraph = ProteinDiseaseAssociationGraph.load(graphString)
# SOME DISEASES CAUSE "DIVIDE BY 0 error"
logging.info("GRAPH {0} LOADED".format(graphString))

#Get reult directory and number of folds
if (argData['resultdir'] is not None):
	resultDir = argData['resultdir'] #folder where all results will be stored
else:
	logging.error('Result directory is needed')
	exit()
nfolds = argData['crossval_folds'] # applicable for average CV

#Nodes
nodes = [ProteinInteractionNode,KeggNode,ReactomeNode,GoNode,InterproNode]


#staticFeatures = []
staticFeatures = argData['static_data'].split(',')
logging.info(staticFeatures)

logging.info("--- USING {0} METAPATH FEATURE SETS".format(len(nodes)))
logging.info("--- USING {0} STATIC FEATURE SETS".format(len(staticFeatures)))


#fetch the description of proteins and pathway_ids
dbAdapter = OlegDB()
idDescription = dbAdapter.fetchPathwayIdDescription() #fetch the description
idNameSymbol = dbAdapter.fetchSymbolForProteinId() #fetch name and symbol for protein


if fileData is not None:
	#logging.info("FOUND {0} POSITIVE LABELS".format(len(fileData[True])))
	#logging.info("FOUND {0} NEGATIVE LABELS".format(len(fileData[False])))
	trainData = metapathFeatures(disease,currentGraph,nodes,idDescription,staticFeatures,loadedLists=fileData).fillna(0) 
else:
	trainData = metapathFeatures(disease,currentGraph,nodes,idDescription,staticFeatures).fillna(0)

'''
# directory and file name for the ML Model
if not os.path.isdir(argData['modeldir']):os.mkdir(argData['modeldir'])
if ('.pkl' in diseaseName):
	modelName = argData['modeldir'] + diseaseName.split('.')[0] + '.model'
else:
	modelName = argData['modeldir'] + diseaseName + '.model'
'''

#call ML codes
d = BinaryLabel()
d.loadData(trainData)
#XGBCrossVal(d)
#print('calling function...', locals()[Procedure])
locals()[Procedure](d, idDescription, idNameSymbol, resultDir, nfolds)


#print("FEATURES CREATED, STARTING ML")
#d = BinaryLabel()
#d.loadData(trainData)
#newModel = XGBoostModel()
#print("SHAPE",d.features.shape)
#roc,acc,CM,report = newModel.cross_val_predict(d,["roc","acc","ConfusionMatrix","report"]) #"report","roc","rocCurve","ConfusionMatrix"
#roc.printOutput()


logging.info('{0}: elapsed time: {1}'.format(os.path.basename(sys.argv[0]), time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-t0))))
